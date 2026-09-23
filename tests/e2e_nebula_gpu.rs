//! GPU NEBULA against the `f64` CPU path.
//!
//! The reference here is `edge-rs`'s own `opt_pml`, not the R package: the CPU
//! path is already gated against nebula 1.5.8 in `tests/e2e_nebula.rs`, so
//! anything this suite catches is the `f32` device arithmetic and nothing else.
//!
//! The tolerances are deliberately far looser than the `1e-6` the CPU path
//! holds, and they are measured rather than guessed. wgpu exposes no `f64`, so
//! the device evaluates one `exp` and one `ln` per cell at about `1e-7`
//! relative each, which compensated summation cannot undo. See the module doc
//! of `edge_rs::gpu::pml_kernel`.
//!
//! Past the sweep, a handful of tiny shapes check what neither the sweep nor
//! the R fixtures in `tests/e2e_nebula.rs` can: the error paths, run-to-run
//! determinism of the stage-two cohorts, odd request counts, and a launch wide
//! enough to need the second grid dimension.
//!
//! The resolution floor is swept rather than asserted at one value. A gate that
//! does not move across the sweep is insensitive rather than permissive, and
//! the sweep is also how the shipped default was chosen.
//!
//! Run with:
//! ```text
//! cargo test --release --features gpu-tests --test e2e_nebula_gpu -- --nocapture
//! ```

#![cfg(feature = "gpu-tests")]

mod common;

use cubecl::Runtime;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use rand::prelude::*;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, Gamma, LogNormal, Poisson};

use edge_rs::errors::EdgeErrors;
use edge_rs::gpu::nebula_gpu::{GpuGene, opt_pml_batch};
use edge_rs::gpu::pml_kernel::F32_NOISE_SCALE;
use edge_rs::gpu::stage_two::nebula_sparse_gpu;
use edge_rs::prelude::{CompressedSparse, SparseFormat};
use edge_rs::sc::nebula::{NebulaFit, NebulaMethod, NebulaParams};
use edge_rs::sc::pml::{PmlData, PmlParams, PmlResult, PmlVariance, opt_pml};

////////////////
// Tolerances //
////////////////

/// Worst relative disagreement allowed on a fitted coefficient.
const BETA_TOL: f64 = 5e-3;

/// Worst relative disagreement allowed on an entry of the observed information.
const INFO_TOL: f64 = 5e-3;

/// Worst relative disagreement allowed on the penalised log-likelihood.
const LL_TOL: f64 = 1e-4;

/// Worst relative disagreement allowed on `beta` after a single Newton step.
///
/// Looser than [`BETA_TOL`] because one step from a cold start lands far from
/// the optimum, where the iterate is still large and its components are not yet
/// separated; at convergence the two paths agree an order of magnitude better.
const ONE_STEP_BETA_TOL: f64 = 5e-2;

/// Resolution floors the suite sweeps, to show the gate can move.
const NOISE_SWEEP: [f64; 4] = [0.0, 1e-7, 1e-6, 1e-5];

/// Absolute stopping tolerances the suite sweeps. Zero means iterate until a
/// step stops improving the objective at all.
const EPS_SWEEP: [f64; 3] = [1e-6, 1e-9, 0.0];

/// Coefficients smaller than this in absolute value are compared absolutely.
///
/// A relative comparison against a coefficient that is legitimately near zero
/// measures nothing but the rounding of the smaller operand.
const NEAR_ZERO: f64 = 1e-3;

///////////////
// Test data //
///////////////

/// Problem shape: genes, cells, subjects, design columns.
///
/// `EDGE_RS_GPU_CELLS` overrides the cell count. The cancellation in the Schur
/// complement and in the gradient scales with the per-subject count totals, so
/// sweeping the cell count is how that is told apart from ordinary rounding.
fn shape() -> (usize, usize, usize, usize) {
    let cells = std::env::var("EDGE_RS_GPU_CELLS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4_000);
    (128, cells, 10, 3)
}

/// Seed for the count generator.
const SEED: u64 = 0xBEEF_0042;

/// Subject-level overdispersion the counts are drawn with.
const TRUE_SIGMA: f64 = 0.25;

/// Cell-level overdispersion the counts are drawn with.
const TRUE_PHI_INV: f64 = 0.5;

/// Requests per launch past which the grid needs its second dimension.
///
/// A dispatch dimension holds at most 65535 workgroups and a workgroup carries
/// two requests, so anything above `2 * 65535` spills into `y`.
const ONE_DIM_REQUESTS: usize = 2 * 65_535;

/// One generated batch, in the layout both paths read.
struct Batch {
    /// Design, row-major `n_cells * n_coef`.
    design: Vec<f64>,
    /// Design columns.
    n_coef: usize,
    /// Log offset per cell.
    log_offset: Vec<f64>,
    /// Subject boundaries, length `n_subjects + 1`.
    subject_start: Vec<usize>,
    /// Positive counts per gene.
    counts: Vec<Vec<f64>>,
    /// Cell index of each positive count, per gene.
    cells: Vec<Vec<usize>>,
    /// Count total per subject, per gene.
    subject_totals: Vec<Vec<f64>>,
}

/// Draws the sweep's batch, at [`shape`].
///
/// ### Returns
///
/// The assembled [`Batch`].
fn make_batch() -> Batch {
    let (n_genes, n_cells, n_subjects, n_coef) = shape();
    make_batch_with(n_genes, n_cells, n_subjects, n_coef)
}

/// Draws a batch from the gamma-gamma-Poisson model NEBULA fits.
///
/// Cells are laid out subject by subject, with uneven block sizes so the ragged
/// subject path is exercised. Column one is a subject-level group, the rest
/// vary by cell.
///
/// ### Params
///
/// * `n_genes` - Genes
/// * `n_cells` - Cells
/// * `n_subjects` - Subjects
/// * `n_coef` - Design columns, intercept included
///
/// ### Returns
///
/// The assembled [`Batch`].
fn make_batch_with(n_genes: usize, n_cells: usize, n_subjects: usize, n_coef: usize) -> Batch {
    let mut rng = SmallRng::seed_from_u64(SEED);

    let mut subject_id = Vec::with_capacity(n_cells);
    let mut subject_start = vec![0usize];
    let base = n_cells / n_subjects;
    let mut assigned = 0;
    for s in 0..n_subjects {
        let size = if s + 1 == n_subjects {
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

    let mut design = vec![0.0; n_cells * n_coef];
    for c in 0..n_cells {
        design[c * n_coef] = 1.0;
        if n_coef > 1 {
            design[c * n_coef + 1] = f64::from(subject_id[c] % 2 == 1);
        }
        for j in 2..n_coef {
            design[c * n_coef + j] = rng.random_range(-1.0..1.0);
        }
    }

    let frailty = Gamma::new(1.0 / TRUE_SIGMA, TRUE_SIGMA).expect("valid gamma");
    let cell_noise = Gamma::new(1.0 / TRUE_PHI_INV, TRUE_PHI_INV).expect("valid gamma");

    let mut counts = Vec::with_capacity(n_genes);
    let mut cells = Vec::with_capacity(n_genes);
    let mut subject_totals = Vec::with_capacity(n_genes);
    let mut beta = vec![0.0; n_coef];

    for _ in 0..n_genes {
        beta[0] = rng.random_range(-1.0..1.5);
        for b in beta.iter_mut().skip(1) {
            *b = rng.random_range(-0.5..0.5);
        }
        let w: Vec<f64> = (0..n_subjects).map(|_| frailty.sample(&mut rng)).collect();

        let mut gene_counts = Vec::new();
        let mut gene_cells = Vec::new();
        let mut totals = vec![0.0; n_subjects];
        for c in 0..n_cells {
            let mut eta = 0.0f64;
            for j in 0..n_coef {
                eta += design[c * n_coef + j] * beta[j];
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
        n_coef,
        log_offset,
        subject_start,
        counts,
        cells,
        subject_totals,
    }
}

/// Relative disagreement, falling back to absolute near zero.
///
/// ### Params
///
/// * `got` - Value under test
/// * `want` - Reference value
///
/// ### Returns
///
/// `|got - want| / |want|`, or the absolute difference when `want` is smaller
/// than [`NEAR_ZERO`].
fn rel(got: f64, want: f64) -> f64 {
    if want.abs() < NEAR_ZERO {
        (got - want).abs()
    } else {
        (got - want).abs() / want.abs()
    }
}

///////////
// Tests //
///////////

/// Fits every gene on the CPU, which is the reference for this suite.
///
/// ### Params
///
/// * `batch` - The generated problem
///
/// ### Returns
///
/// One [`PmlResult`] per gene.
fn cpu_reference(batch: &Batch) -> Vec<PmlResult> {
    let (n_genes, n_coef) = (batch.counts.len(), batch.n_coef);
    let params = PmlParams {
        ord: 1,
        ..PmlParams::default()
    };
    let beta_init = vec![0.0; n_coef];
    (0..n_genes)
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
        })
        .collect()
}

/// One GPU run against the CPU reference, at a given resolution floor.
///
/// ### Params
///
/// * `batch` - The generated problem
/// * `cpu` - The CPU fits, one per gene
/// * `client` - CubeCL compute client
/// * `noise_scale` - Resolution floor on the objective, relative to its
///   magnitude
/// * `eps` - Absolute stopping tolerance
///
/// ### Returns
///
/// The worst relative disagreement on the coefficients, on the observed
/// information and on the log-likelihood, then the number of genes whose
/// backtracking search stalled.
fn compare(
    batch: &Batch,
    cpu: &[PmlResult],
    client: &cubecl::prelude::ComputeClient<WgpuRuntime>,
    noise_scale: f64,
    eps: f64,
) -> (f64, f64, f64, usize) {
    let (n_genes, n_coef) = (batch.counts.len(), batch.n_coef);
    let params = PmlParams {
        ord: 1,
        ..PmlParams::default()
    };
    let beta_init = vec![0.0; n_coef];
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

    let gpu = opt_pml_batch::<WgpuRuntime>(
        &batch.design,
        &batch.log_offset,
        &batch.subject_start,
        &genes,
        eps,
        noise_scale,
        params.max_iter as u32,
        params.max_backtrack as u32,
        client,
    )
    .expect("gpu pml fits");

    let mut worst_beta = 0.0f64;
    let mut worst_info = 0.0f64;
    let mut worst_ll = 0.0f64;
    let mut stalled = 0usize;
    for (g, fit) in cpu.iter().enumerate().take(n_genes) {
        for j in 0..n_coef {
            worst_beta = worst_beta.max(rel(gpu.beta[g * n_coef + j], fit.beta[j]));
        }
        for i in 0..n_coef * n_coef {
            worst_info = worst_info.max(rel(
                gpu.information[g * n_coef * n_coef + i],
                fit.information[i],
            ));
        }
        worst_ll = worst_ll.max(rel(gpu.log_likelihood[g], fit.log_likelihood));
        if gpu.backtracks[g] > params.max_backtrack as u32 {
            stalled += 1;
        }
    }
    (worst_beta, worst_info, worst_ll, stalled)
}

/// Dumps the worst genes at one resolution floor, split by what is wrong.
///
/// A relative error on a coefficient that is statistically zero says nothing,
/// so the absolute difference and the error restricted to coefficients of
/// substance are printed alongside it.
///
/// ### Params
///
/// * `batch` - The generated problem
/// * `cpu` - The CPU fits
/// * `client` - CubeCL compute client
/// * `noise_scale` - Resolution floor to run at
/// * `eps` - Absolute stopping tolerance
fn diagnose(
    batch: &Batch,
    cpu: &[PmlResult],
    client: &cubecl::prelude::ComputeClient<WgpuRuntime>,
    noise_scale: f64,
    eps: f64,
) {
    let (n_genes, _, _, n_coef) = shape();
    let params = PmlParams {
        ord: 1,
        ..PmlParams::default()
    };
    let beta_init = vec![0.0; n_coef];
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
    let gpu = opt_pml_batch::<WgpuRuntime>(
        &batch.design,
        &batch.log_offset,
        &batch.subject_start,
        &genes,
        eps,
        noise_scale,
        params.max_iter as u32,
        params.max_backtrack as u32,
        client,
    )
    .expect("gpu pml fits");

    let mut worst_abs = 0.0f64;
    let mut worst_substantive = 0.0f64;
    let mut rows: Vec<(f64, usize)> = Vec::with_capacity(n_genes);
    for (g, fit) in cpu.iter().enumerate().take(n_genes) {
        let mut e = 0.0f64;
        for j in 0..n_coef {
            let want = fit.beta[j];
            let got = gpu.beta[g * n_coef + j];
            e = e.max(rel(got, want));
            worst_abs = worst_abs.max((got - want).abs());
            if want.abs() > 0.05 {
                worst_substantive = worst_substantive.max((got - want).abs() / want.abs());
            }
        }
        rows.push((e, g));
    }
    rows.sort_by(|a, b| b.0.partial_cmp(&a.0).expect("finite"));

    println!("  worst absolute beta difference:              {worst_abs:.3e}");
    println!("  worst relative beta where |beta| > 0.05:     {worst_substantive:.3e}");
    for &(e, g) in rows.iter().take(4) {
        println!(
            "  gene {g:4} rel {e:.3e} nnz {:5} cpu it/bt {}/{} gpu it/bt {}/{}",
            batch.counts[g].len(),
            cpu[g].iterations,
            cpu[g].backtracks,
            gpu.iterations[g],
            gpu.backtracks[g],
        );
        let cb: Vec<String> = (0..n_coef)
            .map(|j| format!("{:+.6}", cpu[g].beta[j]))
            .collect();
        let gb: Vec<String> = (0..n_coef)
            .map(|j| format!("{:+.6}", gpu.beta[g * n_coef + j]))
            .collect();
        println!("        cpu [{}]  gpu [{}]", cb.join(", "), gb.join(", "));
    }
}

/// One Newton step on both paths, from the same starting point.
///
/// Isolates the step assembly from the search: after exactly one step the
/// gradient, the curvature, the Schur complement and the `LDL'` solve have all
/// run once at `(beta_init, 0)`, so the step itself must agree closely.
///
/// The observed information is deliberately not compared here. The two paths
/// assemble it at different points by design — nebula and the CPU port report
/// the penultimate iterate's, the kernel the final one's — and at a budget of a
/// single Newton step those two points are as far apart as they ever get.
/// `gpu_opt_pml_matches_cpu` is where the information is checked, at
/// convergence, which is the only place the comparison means anything.
#[test]
fn gpu_first_newton_step_matches_cpu() {
    let batch = make_batch();
    let (n_genes, _, _, n_coef) = shape();
    let params = PmlParams {
        ord: 1,
        max_iter: 1,
        ..PmlParams::default()
    };
    let beta_init = vec![0.0; n_coef];

    let cpu: Vec<PmlResult> = (0..n_genes)
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
        })
        .collect();

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
    let Some(client) = common::gpu_client() else {
        return;
    };
    let gpu = opt_pml_batch::<WgpuRuntime>(
        &batch.design,
        &batch.log_offset,
        &batch.subject_start,
        &genes,
        params.eps,
        f64::from(F32_NOISE_SCALE),
        1,
        params.max_backtrack as u32,
        &client,
    )
    .expect("gpu pml fits");

    let mut worst_beta = 0.0f64;
    let mut worst_ll = 0.0f64;
    let mut bt_mismatch = 0usize;
    for (g, fit) in cpu.iter().enumerate().take(n_genes) {
        for j in 0..n_coef {
            worst_beta = worst_beta.max(rel(gpu.beta[g * n_coef + j], fit.beta[j]));
        }
        worst_ll = worst_ll.max(rel(gpu.log_likelihood[g], fit.log_likelihood));
        if gpu.backtracks[g] != fit.backtracks as u32 {
            bt_mismatch += 1;
        }
    }

    println!("\none Newton step, {n_genes} genes:");
    println!("  worst relative beta   {worst_beta:.3e} (needs {ONE_STEP_BETA_TOL:.0e})");
    println!("  worst relative loglik {worst_ll:.3e} (needs {LL_TOL:.0e})");
    println!("  genes with a different backtrack count: {bt_mismatch}");

    assert_eq!(
        bt_mismatch, 0,
        "the two paths disagreed on the backtrack count for {bt_mismatch} genes"
    );
    assert!(
        worst_beta < ONE_STEP_BETA_TOL,
        "one Newton step disagrees by {worst_beta:.3e}, above {ONE_STEP_BETA_TOL:.0e}"
    );
    assert!(
        worst_ll < LL_TOL,
        "log-likelihood disagrees by {worst_ll:.3e}, above {LL_TOL:.0e}"
    );
}

#[test]
fn gpu_opt_pml_matches_cpu() {
    let batch = make_batch();
    let cpu = cpu_reference(&batch);
    let Some(client) = common::gpu_client() else {
        return;
    };

    println!(
        "\n{} genes, {} cells, {} subjects, {} coefficients",
        shape().0,
        shape().1,
        shape().2,
        shape().3
    );
    println!("  noise_scale       eps     beta        info      loglik   stalled");
    for &eps in EPS_SWEEP.iter() {
        for &scale in NOISE_SWEEP.iter() {
            let (b, i, l, stalled) = compare(&batch, &cpu, &client, scale, eps);
            println!("  {scale:>11.0e} {eps:>9.0e}  {b:.3e}  {i:.3e}  {l:.3e}   {stalled:>3}");
        }
    }

    let default = f64::from(F32_NOISE_SCALE);
    let (worst_beta, worst_info, worst_ll, _) =
        compare(&batch, &cpu, &client, default, PmlParams::default().eps);
    diagnose(&batch, &cpu, &client, default, PmlParams::default().eps);
    println!(
        "\nat the shipped default {default:.0e}: beta {worst_beta:.3e} (needs {BETA_TOL:.0e}), \
info {worst_info:.3e} (needs {INFO_TOL:.0e}), loglik {worst_ll:.3e} (needs {LL_TOL:.0e})"
    );

    assert!(
        worst_beta < BETA_TOL,
        "coefficients disagree by {worst_beta:.3e}, above {BETA_TOL:.0e}"
    );
    assert!(
        worst_info < INFO_TOL,
        "information disagrees by {worst_info:.3e}, above {INFO_TOL:.0e}"
    );
    assert!(
        worst_ll < LL_TOL,
        "log-likelihood disagrees by {worst_ll:.3e}, above {LL_TOL:.0e}"
    );
}

/// The inner fit across the range of subject-level variance stage two visits.
///
/// Stage two's search runs `sigma^2` down to its lower bound of `1e-4`, where
/// the gamma prior's `alpha` and `lambda` both grow like `1 / sigma^2`. The fit
/// has to hold there too, not only at the variance the counts were drawn with.
#[test]
fn gpu_opt_pml_holds_across_the_variance_range() {
    let batch = make_batch();
    let (n_genes, _, _, n_coef) = shape();
    let Some(client) = common::gpu_client() else {
        return;
    };
    let params = PmlParams {
        ord: 1,
        ..PmlParams::default()
    };
    let beta_init = vec![0.0; n_coef];

    println!(
        "\n  sigma^2    subject     worst |dll|   worst rel ll   worst |dlogdet|   mean dll    iter cpu/gpu"
    );
    for sigma2 in [1e-4, 1e-3, 1e-2, 1e-1, 1.0] {
        let subject = f64::ln_1p(sigma2);
        let cpu: Vec<PmlResult> = (0..n_genes)
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
                        subject,
                        cell: 1.0 / TRUE_PHI_INV,
                    },
                    Some(params),
                )
                .expect("cpu pml fits")
            })
            .collect();
        let genes: Vec<GpuGene<'_>> = (0..n_genes)
            .map(|g| GpuGene {
                counts: &batch.counts[g],
                cell_index: &batch.cells[g],
                subject_total: &batch.subject_totals[g],
                beta_init: &beta_init,
                sigma: subject,
                gamma: 1.0 / TRUE_PHI_INV,
            })
            .collect();
        let gpu = opt_pml_batch::<WgpuRuntime>(
            &batch.design,
            &batch.log_offset,
            &batch.subject_start,
            &genes,
            params.eps,
            f64::from(F32_NOISE_SCALE),
            params.max_iter as u32,
            params.max_backtrack as u32,
            &client,
        )
        .expect("gpu pml fits");

        let mut abs_ll = 0.0f64;
        let mut rel_ll = 0.0f64;
        let mut abs_det = 0.0f64;
        let mut signed = 0.0f64;
        let mut it_cpu = 0.0f64;
        let mut it_gpu = 0.0f64;
        for (g, fit) in cpu.iter().enumerate() {
            let signed_d = gpu.log_likelihood[g] - fit.log_likelihood;
            let d = signed_d.abs();
            signed += signed_d;
            it_cpu += fit.iterations as f64;
            it_gpu += f64::from(gpu.iterations[g]);
            abs_ll = abs_ll.max(d);
            rel_ll = rel_ll.max(d / fit.log_likelihood.abs());
            abs_det = abs_det.max((gpu.log_det[g] - fit.log_det).abs());
        }
        let n = n_genes as f64;
        println!(
            "  {sigma2:>7.0e}  {subject:>9.3e}   {abs_ll:>10.3e}   {rel_ll:>10.3e}   {abs_det:>10.3e}   {:>+9.2e}   {:.1}/{:.1}",
            signed / n,
            it_cpu / n,
            it_gpu / n
        );
    }
}

//////////////////
// Tiny shapes  //
//////////////////

/// A fixture from `tests/data/e2e`, as [`nebula_sparse_gpu`] takes it.
struct Fixture {
    /// Counts, CSR over `(n_genes, n_cells)`.
    counts: CompressedSparse<f64>,
    /// Subject index per cell, zero-based.
    subject: Vec<usize>,
    /// Design, row-major `n_cells * n_coef`.
    design: Vec<f64>,
    /// Design columns.
    n_coef: usize,
    /// Per-cell offset, on the linear scale.
    offset: Vec<f64>,
}

/// Loads one single-cell fixture written by `tests/r/generate_fixtures.R`.
///
/// ### Params
///
/// * `tag` - Fixture prefix
/// * `design_file` - Whether the design is `{tag}_design.csv`, rather than
///   `[1, grp, cov2]` from the meta file
///
/// ### Returns
///
/// The fixture.
fn fixture(tag: &str, design_file: bool) -> Fixture {
    let t = common::table(&format!("{tag}_counts.csv"));
    let (n_genes, n_cells) = (t.n_rows(), t.n_cols());
    let dense = t.row_major_counts();
    let meta = common::table(&format!("{tag}_meta.csv"));
    let subject = meta.column_usize("subject").iter().map(|s| s - 1).collect();
    let (design, n_coef) = if design_file {
        let (design, _, cols) = common::matrix(&format!("{tag}_design.csv"));
        (design, cols)
    } else {
        let (grp, cov2) = (meta.column("grp"), meta.column("cov2"));
        (
            (0..n_cells).flat_map(|c| [1.0, grp[c], cov2[c]]).collect(),
            3,
        )
    };

    let mut data = Vec::new();
    let mut indices = Vec::new();
    let mut indptr = vec![0u32];
    for row in dense.chunks_exact(n_cells) {
        for (c, &v) in row.iter().enumerate() {
            if v > 0.0 {
                data.push(v);
                indices.push(c as u32);
            }
        }
        indptr.push(data.len() as u32);
    }
    let counts =
        CompressedSparse::from_parts(data, indices, indptr, SparseFormat::Csr, (n_genes, n_cells))
            .expect("well-formed CSR");

    Fixture {
        counts,
        subject,
        design,
        n_coef,
        offset: meta.column("offset").to_vec(),
    }
}

/// Keeps only the genes with at least `min_cells` positive counts.
///
/// A gene with no counts at all has its intercept running to minus infinity,
/// which neither path is meant to fit and NEBULA's own filter never passes.
///
/// ### Params
///
/// * `batch` - The batch, filtered in place
/// * `min_cells` - Fewest expressed cells a gene keeps
fn drop_sparse_genes(batch: &mut Batch, min_cells: usize) {
    let keep: Vec<bool> = batch.counts.iter().map(|c| c.len() >= min_cells).collect();
    let mut it = keep.iter();
    batch
        .counts
        .retain(|_| *it.next().expect("one flag per gene"));
    let mut it = keep.iter();
    batch
        .cells
        .retain(|_| *it.next().expect("one flag per gene"));
    let mut it = keep.iter();
    batch
        .subject_totals
        .retain(|_| *it.next().expect("one flag per gene"));
}

/// Asserts two NEBULA fits are bit-identical, NaNs included.
///
/// ### Params
///
/// * `a` - First fit
/// * `b` - Second fit
fn assert_same_bits(a: &NebulaFit, b: &NebulaFit) {
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<u64>>();
    assert_eq!(a.gene_index, b.gene_index, "gene_index");
    assert_eq!(bits(&a.coefficients), bits(&b.coefficients), "coefficients");
    assert_eq!(bits(&a.se), bits(&b.se), "se");
    assert_eq!(bits(&a.covariance), bits(&b.covariance), "covariance");
    assert_eq!(
        bits(&a.subject_overdispersion),
        bits(&b.subject_overdispersion),
        "subject_overdispersion"
    );
    assert_eq!(
        bits(&a.cell_overdispersion),
        bits(&b.cell_overdispersion),
        "cell_overdispersion"
    );
    assert_eq!(a.convergence, b.convergence, "convergence");
}

/// Asserts one GPU run of `batch` against its CPU reference at the default
/// resolution floor.
///
/// ### Params
///
/// * `batch` - The problem
/// * `client` - CubeCL compute client
/// * `label` - What the batch is, for the report
fn assert_batch_matches_cpu(
    batch: &Batch,
    client: &cubecl::prelude::ComputeClient<WgpuRuntime>,
    label: &str,
) {
    let cpu = cpu_reference(batch);
    let (beta, info, ll, stalled) = compare(
        batch,
        &cpu,
        client,
        f64::from(F32_NOISE_SCALE),
        PmlParams::default().eps,
    );
    println!(
        "{label}: {} genes, beta {beta:.3e}, info {info:.3e}, loglik {ll:.3e}, stalled {stalled}",
        batch.counts.len()
    );
    assert!(
        beta < BETA_TOL,
        "{label}: coefficients disagree by {beta:.3e}"
    );
    assert!(
        info < INFO_TOL,
        "{label}: information disagrees by {info:.3e}"
    );
    assert!(ll < LL_TOL, "{label}: log-likelihood disagrees by {ll:.3e}");
}

/// The two things the device fit cannot do come back as errors, not as
/// answers: `reml`, and a design wider than [`MAX_BETA_CAP`](edge_rs::gpu::pml_kernel::MAX_BETA_CAP).
///
/// HL, so every gene reaches stage two and the width check actually runs; under
/// LN a gene that needs no refit never touches the device.
#[test]
fn gpu_rejects_what_it_cannot_fit() {
    let f = fixture("sc_k2", true);
    let client = WgpuRuntime::client(&WgpuDevice::default());
    let hl = NebulaParams {
        method: NebulaMethod::Hl,
        ..NebulaParams::default()
    };

    let reml = nebula_sparse_gpu(
        &f.counts,
        &f.subject,
        &f.design,
        f.n_coef,
        Some(&f.offset),
        Some(NebulaParams { reml: true, ..hl }),
        &client,
    );
    assert!(
        matches!(reml, Err(EdgeErrors::InvalidArgument(_))),
        "reml should be refused, got {:?}",
        reml.map(|_| ())
    );

    // Nine columns: the fixture's three plus six cell-level ones.
    let mut rng = SmallRng::seed_from_u64(SEED);
    let wide: Vec<f64> = f
        .design
        .chunks_exact(f.n_coef)
        .flat_map(|row| {
            let mut row = row.to_vec();
            row.extend((0..6).map(|_| rng.random_range(-1.0..1.0)));
            row
        })
        .collect();
    let nine = nebula_sparse_gpu(
        &f.counts,
        &f.subject,
        &wide,
        f.n_coef + 6,
        Some(&f.offset),
        Some(hl),
        &client,
    );
    assert!(
        matches!(nine, Err(EdgeErrors::InvalidArgument(_))),
        "nine columns should be refused, got {:?}",
        nine.map(|_| ())
    );
}

/// Two runs of the same input give the same bits.
///
/// Whether stage two splits its searches into two cohorts is decided by timing
/// the first rounds, so it can differ between runs. The cohorts are meant to
/// change only when a search is told its values, never what, and `sc_small`
/// runs 118 HL searches, above the count at which the cohorts merge. A machine
/// that never splits only shows plain run-to-run determinism.
#[test]
fn gpu_nebula_is_deterministic() {
    let f = fixture("sc_small", false);
    let Some(client) = common::gpu_client() else {
        return;
    };
    let run = || {
        nebula_sparse_gpu(
            &f.counts,
            &f.subject,
            &f.design,
            f.n_coef,
            Some(&f.offset),
            None,
            &client,
        )
        .expect("gpu nebula fits")
    };
    assert_same_bits(&run(), &run());
}

/// A launch wide enough to need the grid's second dimension.
///
/// A request the grid does not cover fails silently: its output is whatever
/// the buffer held. Tiny genes keep the launch cheap; the ones past
/// [`ONE_DIM_REQUESTS`] are the point.
#[test]
fn gpu_grid_past_one_dimension() {
    let mut batch = make_batch_with(ONE_DIM_REQUESTS + 4096, 40, 2, 1);
    drop_sparse_genes(&mut batch, 5);
    assert!(
        batch.counts.len() > ONE_DIM_REQUESTS,
        "only {} genes survived, which fits in one grid dimension",
        batch.counts.len()
    );
    let Some(client) = common::gpu_client() else {
        return;
    };
    assert_batch_matches_cpu(&batch, &client, "two-dimensional grid");
}

/// One request, and three: a workgroup carries two, so both leave a plane
/// idle.
#[test]
fn gpu_odd_request_counts() {
    let Some(client) = common::gpu_client() else {
        return;
    };
    for n in [1, 3] {
        let batch = make_batch_with(n, 400, 5, 3);
        assert_batch_matches_cpu(&batch, &client, &format!("{n} requests"));
    }
}
