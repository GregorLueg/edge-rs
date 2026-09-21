//! GPU NEBULA against the CPU path, on the same genes.
//!
//! The unit of comparison is one `opt_pml` solve per gene: the CPU fans out
//! over genes with rayon on every core, the GPU runs one thread per gene. Both
//! numbers are wall clock for the whole batch, and the GPU number includes
//! staging and read-back, because that is what a caller pays.
//!
//! The first launch compiles the shader, so every cell runs once untimed first.
//!
//! Run with:
//! ```text
//! cargo bench --features gpu --bench nebula_gpu_bench
//! ```
//!
//! `NEBULA_GPU_GENES` and `NEBULA_GPU_CELLS` change the shape.

#![cfg(feature = "gpu")]

use std::env;
use std::hint::black_box;
use std::time::Instant;

use cubecl::Runtime;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use rand::prelude::*;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, Gamma, LogNormal, Poisson};
use rayon::prelude::*;

use edge_rs::gpu::nebula_gpu::{
    GpuGene, GpuSolveParams, ResidentBatch, opt_pml_batch, solve_per_gene,
};
use edge_rs::gpu::pml_kernel::F32_NOISE_SCALE;
use edge_rs::sc::pml::{PmlData, PmlParams, PmlVariance, opt_pml};

////////////
// Shapes //
////////////

/// Shapes swept when no environment override is given: genes then cells.
///
/// The first is a small batch where the GPU cannot fill the device; the last is
/// the regime a real single-cell run is in, where the per-gene work dominates.
const SWEEP: [(usize, usize); 4] = [(256, 20_000), (1024, 20_000), (4096, 20_000), (1024, 80_000)];

/// Solves run against one resident upload, to show what stage two would see.
///
/// NEBULA's stage two calls `opt_pml` once per Nelder-Mead evaluation and once
/// per polish stencil point, on the order of a hundred times per gene, with
/// only the two variance components changed. Eight is enough to separate the
/// upload from the solve without making the bench tedious.
const RESIDENT_SOLVES: usize = 8;

/// Subjects in every shape. Twenty donors is a typical NEBULA design.
const SUBJECTS: usize = 20;

/// Design columns: intercept, a subject-level group and a cell-level covariate.
const COEF: usize = 3;

/// Seed for the count generator.
const SEED: u64 = 0x6D11_0001;

/// Subject-level overdispersion the counts are drawn with.
const TRUE_SIGMA: f64 = 0.25;

/// Cell-level overdispersion the counts are drawn with.
const TRUE_PHI_INV: f64 = 0.5;

///////////////
// Test data //
///////////////

/// One generated batch.
struct Batch {
    /// Design, row-major `n_cells * COEF`.
    design: Vec<f64>,
    /// Log offset per cell.
    log_offset: Vec<f64>,
    /// Subject boundaries, length `SUBJECTS + 1`.
    subject_start: Vec<usize>,
    /// Positive counts per gene.
    counts: Vec<Vec<f64>>,
    /// Cell index of each positive count, per gene.
    cells: Vec<Vec<usize>>,
    /// Count total per subject, per gene.
    subject_totals: Vec<Vec<f64>>,
}

/// Draws a batch from the gamma-gamma-Poisson model NEBULA fits.
///
/// ### Params
///
/// * `n_genes` - Number of genes
/// * `n_cells` - Number of cells
///
/// ### Returns
///
/// The assembled [`Batch`].
fn make_batch(n_genes: usize, n_cells: usize) -> Batch {
    let mut rng = SmallRng::seed_from_u64(SEED);

    let mut subject_id = Vec::with_capacity(n_cells);
    let mut subject_start = vec![0usize];
    let base = n_cells / SUBJECTS;
    let mut assigned = 0;
    for s in 0..SUBJECTS {
        let size = if s + 1 == SUBJECTS {
            n_cells - assigned
        } else {
            base + usize::from(s % 3 == 0)
        };
        subject_id.extend(std::iter::repeat_n(s, size));
        assigned += size;
        subject_start.push(assigned);
    }

    let lib = LogNormal::new(0.0, 0.4).expect("valid lognormal");
    let offset: Vec<f64> = (0..n_cells).map(|_| lib.sample(&mut rng)).collect();
    let log_offset: Vec<f64> = offset.iter().map(|v| v.ln()).collect();

    let mut design = vec![0.0; n_cells * COEF];
    for c in 0..n_cells {
        design[c * COEF] = 1.0;
        design[c * COEF + 1] = f64::from(subject_id[c] % 2 == 1);
        for j in 2..COEF {
            design[c * COEF + j] = rng.random_range(-1.0..1.0);
        }
    }

    let frailty = Gamma::new(1.0 / TRUE_SIGMA, TRUE_SIGMA).expect("valid gamma");
    let cell_noise = Gamma::new(1.0 / TRUE_PHI_INV, TRUE_PHI_INV).expect("valid gamma");

    let mut counts = Vec::with_capacity(n_genes);
    let mut cells = Vec::with_capacity(n_genes);
    let mut subject_totals = Vec::with_capacity(n_genes);
    let mut beta = [0.0; COEF];

    for _ in 0..n_genes {
        beta[0] = rng.random_range(-1.0..1.5);
        for b in beta.iter_mut().skip(1) {
            *b = rng.random_range(-0.5..0.5);
        }
        let w: Vec<f64> = (0..SUBJECTS).map(|_| frailty.sample(&mut rng)).collect();

        let mut gene_counts = Vec::new();
        let mut gene_cells = Vec::new();
        let mut totals = vec![0.0; SUBJECTS];
        for c in 0..n_cells {
            let mut eta = 0.0f64;
            for j in 0..COEF {
                eta += design[c * COEF + j] * beta[j];
            }
            let mu = offset[c] * eta.exp() * w[subject_id[c]] * cell_noise.sample(&mut rng);
            let y = Poisson::new(mu.max(1e-12))
                .expect("positive rate")
                .sample(&mut rng);
            if y > 0.0 {
                gene_counts.push(y);
                gene_cells.push(c);
                totals[subject_id[c]] += y;
            }
        }
        counts.push(gene_counts);
        cells.push(gene_cells);
        subject_totals.push(totals);
    }

    Batch {
        design,
        log_offset,
        subject_start,
        counts,
        cells,
        subject_totals,
    }
}

//////////
// Main //
//////////

/// Reads a `usize` from the environment.
///
/// ### Params
///
/// * `key` - Environment variable name
///
/// ### Returns
///
/// The parsed value, or `None` when unset or unparseable.
fn env_usize(key: &str) -> Option<usize> {
    env::var(key).ok().and_then(|v| v.parse().ok())
}

/// Runs every gene through the CPU solver, in parallel over genes.
///
/// ### Params
///
/// * `batch` - The generated problem
/// * `n_genes` - Number of genes
///
/// ### Returns
///
/// The sum of the fitted intercepts, to keep the work from being elided.
fn run_cpu(batch: &Batch, n_genes: usize) -> f64 {
    let params = PmlParams {
        ord: 1,
        ..PmlParams::default()
    };
    let beta_init = vec![0.0; COEF];
    (0..n_genes)
        .into_par_iter()
        .map(|g| {
            let data = PmlData {
                design: &batch.design,
                offset: &batch.log_offset,
                counts: &batch.counts[g],
                cell_index: &batch.cells[g],
                subject_start: &batch.subject_start,
                subject_total: &batch.subject_totals[g],
            };
            opt_pml(
                &data,
                &beta_init,
                &PmlVariance {
                    subject: TRUE_SIGMA,
                    cell: 1.0 / TRUE_PHI_INV,
                },
                Some(params),
            )
            .expect("cpu pml fits")
            .beta[0]
        })
        .sum()
}

/// Runs every gene through the GPU solver, staging and read-back included.
///
/// ### Params
///
/// * `batch` - The generated problem
/// * `n_genes` - Number of genes
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// The sum of the fitted intercepts.
fn run_gpu(
    batch: &Batch,
    n_genes: usize,
    client: &cubecl::prelude::ComputeClient<WgpuRuntime>,
) -> f64 {
    run_gpu_capped(batch, n_genes, client, PmlParams::default().max_iter as u32)
}

/// As [`run_gpu`], with the Newton budget capped.
///
/// Running the same batch at a budget of one isolates what the caller pays
/// regardless of the solve: staging, launch and read-back. The difference
/// against the full budget is the kernel's own work.
///
/// ### Params
///
/// * `batch` - The generated problem
/// * `n_genes` - Number of genes
/// * `client` - CubeCL compute client
/// * `max_iter` - Newton budget
///
/// ### Returns
///
/// The sum of the fitted intercepts.
fn run_gpu_capped(
    batch: &Batch,
    n_genes: usize,
    client: &cubecl::prelude::ComputeClient<WgpuRuntime>,
    max_iter: u32,
) -> f64 {
    let params = PmlParams {
        ord: 1,
        ..PmlParams::default()
    };
    let beta_init = vec![0.0; COEF];
    let genes: Vec<GpuGene<'_>> = (0..n_genes)
        .map(|g| GpuGene {
            counts: &batch.counts[g],
            cell_index: &batch.cells[g],
            subject_total: &batch.subject_totals[g],
            beta_init: &beta_init,
            sigma: TRUE_SIGMA,
            gamma: 1.0 / TRUE_PHI_INV,
        })
        .collect();
    let fit = opt_pml_batch::<WgpuRuntime>(
        &batch.design,
        &batch.log_offset,
        &batch.subject_start,
        &genes,
        params.eps,
        f64::from(F32_NOISE_SCALE),
        max_iter,
        params.max_backtrack as u32,
        client,
    )
    .expect("gpu pml fits");
    (0..n_genes).map(|g| fit.beta[g * COEF]).sum()
}

/// Builds the per-gene views the GPU entry points take.
///
/// ### Params
///
/// * `batch` - The generated problem
/// * `n_genes` - Number of genes
/// * `beta_init` - Starting fixed effects, shared by every gene
///
/// ### Returns
///
/// One [`GpuGene`] per gene, borrowed against `batch`.
fn gpu_genes<'a>(batch: &'a Batch, n_genes: usize, beta_init: &'a [f64]) -> Vec<GpuGene<'a>> {
    (0..n_genes)
        .map(|g| GpuGene {
            counts: &batch.counts[g],
            cell_index: &batch.cells[g],
            subject_total: &batch.subject_totals[g],
            beta_init,
            sigma: TRUE_SIGMA,
            gamma: 1.0 / TRUE_PHI_INV,
        })
        .collect()
}

fn main() {
    let device = WgpuDevice::default();
    let client = WgpuRuntime::client(&device);

    let shapes: Vec<(usize, usize)> = match (env_usize("NEBULA_GPU_GENES"), env_usize("NEBULA_GPU_CELLS")) {
        (Some(g), Some(c)) => vec![(g, c)],
        _ => SWEEP.to_vec(),
    };

    println!(
        "\nNEBULA opt_pml, {SUBJECTS} subjects, {COEF} coefficients, {} CPU threads",
        rayon::current_num_threads()
    );
    println!("    genes    cells        cpu        gpu   speedup");

    let beta_init = vec![0.0; COEF];
    for (n_genes, n_cells) in shapes {
        let batch = make_batch(n_genes, n_cells);

        // Warm up both: the first GPU launch compiles the shader, and the first
        // CPU pass faults in the design.
        black_box(run_cpu(&batch, 8.min(n_genes)));
        black_box(run_gpu(&batch, 8.min(n_genes), &client));

        let t = Instant::now();
        let cpu = black_box(run_cpu(&batch, n_genes));
        let cpu_time = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let gpu = black_box(run_gpu(&batch, n_genes, &client));
        let gpu_time = t.elapsed().as_secs_f64();

        black_box(run_gpu_capped(&batch, n_genes, &client, 1));
        let t = Instant::now();
        black_box(run_gpu_capped(&batch, n_genes, &client, 1));
        let fixed = t.elapsed().as_secs_f64();

        println!(
            "  {n_genes:>7}  {n_cells:>7}  {cpu_time:>7.3} s  {gpu_time:>7.3} s   {:>6.2}x",
            cpu_time / gpu_time
        );
        println!(
            "           of which staging, launch and read-back {fixed:.3} s; kernel {:.3} s",
            gpu_time - fixed
        );

        // Amortised: one upload, several solves, against the same number of
        // CPU passes. This is the shape stage two would run in.
        let mut resident = ResidentBatch::upload(
            &batch.design,
            &batch.log_offset,
            &batch.subject_start,
            &gpu_genes(&batch, n_genes, &beta_init),
            &client,
        )
        .expect("upload");
        let genes = gpu_genes(&batch, n_genes, &beta_init);
        let params = PmlParams {
            ord: 1,
            ..PmlParams::default()
        };
        let solve_params = GpuSolveParams {
            eps: params.eps,
            noise_scale: f64::from(F32_NOISE_SCALE),
            max_iter: params.max_iter as u32,
            max_backtrack: params.max_backtrack as u32,
            full: false,
            information: true,
        };
        black_box(solve_per_gene(&mut resident, &genes, &solve_params, &client).expect("solve"));

        let t = Instant::now();
        for _ in 0..RESIDENT_SOLVES {
            black_box(
                solve_per_gene(&mut resident, &genes, &solve_params, &client).expect("solve"),
            );
        }
        let resident_time = t.elapsed().as_secs_f64();

        let t = Instant::now();
        for _ in 0..RESIDENT_SOLVES {
            black_box(run_cpu(&batch, n_genes));
        }
        let cpu_many = t.elapsed().as_secs_f64();
        println!(
            "           {RESIDENT_SOLVES} solves on one upload: cpu {cpu_many:.3} s, gpu {resident_time:.3} s, {:.2}x",
            cpu_many / resident_time
        );
        // The two sums are not expected to match bit for bit; printing them
        // guards against a kernel that returns zeros and looks fast.
        println!("           checksum cpu {cpu:.6}  gpu {gpu:.6}");
    }
}
