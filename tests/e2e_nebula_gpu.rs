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
//! The resolution floor is swept rather than asserted at one value. A gate that
//! does not move across the sweep is insensitive rather than permissive, and
//! the sweep is also how the shipped default was chosen.
//!
//! Run with:
//! ```text
//! cargo test --release --features gpu --test e2e_nebula_gpu -- --nocapture
//! ```

#![cfg(feature = "gpu")]

use cubecl::Runtime;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use rand::prelude::*;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, Gamma, LogNormal, Poisson};

use edge_rs::gpu::nebula_gpu::{GpuGene, opt_pml_batch};
use edge_rs::gpu::pml_kernel::F32_NOISE_SCALE;
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

/// One generated batch, in the layout both paths read.
struct Batch {
    /// Design, row-major `n_cells * n_coef`.
    design: Vec<f64>,
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

/// Draws a batch from the gamma-gamma-Poisson model NEBULA fits.
///
/// Cells are laid out subject by subject, with uneven block sizes so the ragged
/// subject path is exercised.
///
/// ### Returns
///
/// The assembled [`Batch`].
fn make_batch() -> Batch {
    let (n_genes, n_cells, n_subjects, n_coef) = shape();
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
        design[c * n_coef + 1] = f64::from(subject_id[c] % 2 == 1);
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
    let (n_genes, _, _, n_coef) = shape();
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
        let cb: Vec<String> = (0..n_coef).map(|j| format!("{:+.6}", cpu[g].beta[j])).collect();
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
    let device = WgpuDevice::default();
    let client = WgpuRuntime::client(&device);
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
    let device = WgpuDevice::default();
    let client = WgpuRuntime::client(&device);

    println!(
        "\n{} genes, {} cells, {} subjects, {} coefficients",
        shape().0, shape().1, shape().2, shape().3
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
    let device = WgpuDevice::default();
    let client = WgpuRuntime::client(&device);
    let params = PmlParams {
        ord: 1,
        ..PmlParams::default()
    };
    let beta_init = vec![0.0; n_coef];

    println!("\n  sigma^2    subject     worst |dll|   worst rel ll   worst |dlogdet|   mean dll    iter cpu/gpu");
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
