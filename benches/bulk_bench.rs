//! Where the bulk edgeR and limma stack spends its wall clock.
//!
//! Nothing outside `src/sc` had a timing harness. This one exists to decide
//! whether any of the shared-design products (`lm_fit` without per-gene
//! weights, `remove_batch_effect`) are worth routing through a GEMM, so it times
//! each public entry point on its own rather than one pipeline end to end.
//!
//! The counts are negative binomial: a log-normal baseline per gene, a two-group
//! fold change on a tenth of the genes, a per-sample library size and a gamma
//! frailty per count. The design is an intercept, the group and one continuous
//! covariate.
//!
//! The measured cells are:
//!
//! * `lm_plain` - [`lm_fit`] on log-CPM with no weights: the shared-design case
//!   a single QR and two GEMMs could replace.
//! * `lm_weighted` - [`lm_fit`] with voom's per-gene weights, which is what
//!   `voom_lmfit`'s final fit pays and which no GEMM helps.
//! * `voom_lmfit` - [`voom_lmfit`] end to end.
//! * `rbe` - [`remove_batch_effect`] with a three-level batch.
//! * `disp` - [`estimate_disp`], trended and tagwise.
//! * `glm` - [`glm_fit`] at the trended dispersion, the Levenberg path.
//! * `ql` - [`glm_ql_fit`] with its own dispersion.
//!
//! Run with:
//! ```text
//! cargo bench --bench bulk_bench
//! ```
//!
//! `BULK_BENCH_GENES` and `BULK_BENCH_SAMPLES` change the shape.
//! `BULK_BENCH_ONLY=lm_plain,rbe` runs a comma-separated subset of the cells.

use std::env;
use std::hint::black_box;
use std::time::{Duration, Instant};

use rand::prelude::*;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, Gamma, LogNormal, Normal, Poisson};

use edge_rs::core::expression::ave_log_cpm;
use edge_rs::dispersion::estimate::estimate_disp;
use edge_rs::glm::fit::glm_fit;
use edge_rs::glm::ql_fit::glm_ql_fit;
use edge_rs::limma::lm_fit::lm_fit;
use edge_rs::limma::remove_batch_effect::remove_batch_effect;
use edge_rs::limma::voom::voom_lmfit;
use edge_rs::prelude::*;

////////////
// Shapes //
////////////

/// Genes by default: a typical filtered bulk RNA-seq experiment.
const DEFAULT_GENES: usize = 20_000;

/// Samples by default.
const DEFAULT_SAMPLES: usize = 24;

/// Coefficients in the design: intercept, group, covariate.
const N_COEF: usize = 3;

/// Batch levels for `remove_batch_effect`.
const N_BATCHES: usize = 3;

/// NB dispersion of the simulated counts.
const TRUE_DISPERSION: f64 = 0.1;

/// Timed repeats per cell; the median is reported.
const REPEATS: usize = 5;

/// RNG seed, so every run sees the same counts.
const SEED: u64 = 0x5EED_B01C;

/////////////
// Problem //
/////////////

/// One simulated bulk experiment.
struct Problem {
    /// Counts, row-major `n_genes * n_samples`.
    counts: Vec<f64>,
    /// Genes.
    n_genes: usize,
    /// Samples.
    n_samples: usize,
    /// Design, row-major `n_samples * N_COEF`.
    design: Vec<f64>,
    /// Library size per sample.
    lib_size: Vec<f64>,
    /// Batch label per sample.
    batch: Vec<usize>,
}

/// Draws the counts and the design.
///
/// ### Params
///
/// * `n_genes` - Genes
/// * `n_samples` - Samples
///
/// ### Returns
///
/// The problem.
fn make_problem(n_genes: usize, n_samples: usize) -> Problem {
    let mut rng = SmallRng::seed_from_u64(SEED);
    let baseline = LogNormal::new(3.0, 1.5).expect("valid log-normal");
    let covariate = Normal::new(0.0, 1.0).expect("valid normal");
    let frailty = Gamma::new(1.0 / TRUE_DISPERSION, TRUE_DISPERSION).expect("valid gamma");

    let mut design = Vec::with_capacity(n_samples * N_COEF);
    let mut lib_scale = Vec::with_capacity(n_samples);
    for s in 0..n_samples {
        design.extend_from_slice(&[1.0, (s % 2) as f64, covariate.sample(&mut rng)]);
        lib_scale.push(rng.random_range(0.6..1.4));
    }
    let batch = (0..n_samples).map(|s| (s / 2) % N_BATCHES).collect();

    let mut counts = Vec::with_capacity(n_genes * n_samples);
    for g in 0..n_genes {
        let mu = baseline.sample(&mut rng);
        let lfc = if g % 10 == 0 { rng.random_range(-2.0..2.0) } else { 0.0 };
        for s in 0..n_samples {
            let m = mu * lib_scale[s] * (lfc * design[s * N_COEF + 1]).exp() * frailty.sample(&mut rng);
            let y = if m > 0.0 {
                Poisson::new(m).expect("positive mean").sample(&mut rng)
            } else {
                0.0
            };
            counts.push(y);
        }
    }

    let lib_size = (0..n_samples)
        .map(|s| (0..n_genes).map(|g| counts[g * n_samples + s]).sum())
        .collect();

    Problem {
        counts,
        n_genes,
        n_samples,
        design,
        lib_size,
        batch,
    }
}

/////////////
// Harness //
/////////////

/// Reads a `usize` from the environment.
///
/// ### Params
///
/// * `key` - Variable name
/// * `fallback` - Value when unset or unparsable
///
/// ### Returns
///
/// The value.
fn env_usize(key: &str, fallback: usize) -> usize {
    env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(fallback)
}

/// Whether a cell is selected by `BULK_BENCH_ONLY`.
///
/// ### Params
///
/// * `name` - Cell name
///
/// ### Returns
///
/// `true` when the variable is unset or lists `name`.
fn selected(name: &str) -> bool {
    match env::var("BULK_BENCH_ONLY") {
        Ok(list) => list.split(',').any(|c| c.trim() == name),
        Err(_) => true,
    }
}

/// Runs `f` once untimed, then [`REPEATS`] times, and prints the median.
///
/// ### Params
///
/// * `name` - Cell name
/// * `f` - The work
fn time<F: FnMut()>(name: &str, mut f: F) {
    if !selected(name) {
        return;
    }
    f();
    let mut runs: Vec<Duration> = (0..REPEATS)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed()
        })
        .collect();
    runs.sort();
    let median = runs[REPEATS / 2];
    println!(
        "  {name:<12} {:>9.1} ms   (min {:.1}, max {:.1})",
        median.as_secs_f64() * 1e3,
        runs[0].as_secs_f64() * 1e3,
        runs[REPEATS - 1].as_secs_f64() * 1e3
    );
}

fn main() {
    let n_genes = env_usize("BULK_BENCH_GENES", DEFAULT_GENES);
    let n_samples = env_usize("BULK_BENCH_SAMPLES", DEFAULT_SAMPLES);
    let p = make_problem(n_genes, n_samples);
    let (g, n, d) = (p.n_genes, p.n_samples, &p.design);

    println!(
        "bulk stack, {g} genes, {n} samples, {N_COEF} coefficients, {} threads",
        rayon::current_num_threads()
    );

    let offset = Recycled::by_sample(p.lib_size.iter().map(|l| l.ln()).collect());
    let alc = ave_log_cpm(&p.counts, g, n, Some(&p.lib_size), None, 2.0, None)
        .expect("ave_log_cpm failed");
    let (voom, _) = voom_lmfit(&p.counts, g, n, d, N_COEF, None, None, None)
        .expect("voom_lmfit failed");
    let voom_weights =
        Recycled::full(voom.weights.clone(), g, n).expect("voom weights have the counts' shape");

    time("lm_plain", || {
        black_box(lm_fit(&voom.e, g, n, d, N_COEF, None, None, None).expect("lm_fit failed"));
    });
    time("lm_weighted", || {
        black_box(
            lm_fit(&voom.e, g, n, d, N_COEF, Some(&voom_weights), None, None)
                .expect("lm_fit failed"),
        );
    });
    time("voom_lmfit", || {
        black_box(voom_lmfit(&p.counts, g, n, d, N_COEF, None, None, None).expect("voom_lmfit failed"));
    });
    time("rbe", || {
        black_box(
            remove_batch_effect(&voom.e, g, n, Some(&p.batch), None, None, Some((d, N_COEF)), None)
                .expect("remove_batch_effect failed"),
        );
    });

    let disp = estimate_disp(&p.counts, g, n, d, N_COEF, &offset, None, Some(&alc), None)
        .expect("estimate_disp failed");
    time("disp", || {
        black_box(
            estimate_disp(&p.counts, g, n, d, N_COEF, &offset, None, Some(&alc), None)
                .expect("estimate_disp failed"),
        );
    });

    let trended = Recycled::by_gene(
        disp.trended
            .clone()
            .unwrap_or_else(|| vec![disp.common; g]),
    );
    time("glm", || {
        black_box(
            glm_fit(&p.counts, g, n, d, N_COEF, &trended, &offset, None, 0.0)
                .expect("glm_fit failed"),
        );
    });
    time("ql", || {
        black_box(
            glm_ql_fit(&p.counts, g, n, d, N_COEF, None, &offset, None, &alc, None)
                .expect("glm_ql_fit failed"),
        );
    });
}
