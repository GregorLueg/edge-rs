//! Where NEBULA's wall clock goes on realistic single-cell counts.
//!
//! There was no timing harness for `src/sc` before this, so nothing about the
//! cost of a NEBULA run was measured. This bench establishes the CPU baseline
//! that any optimisation, on the host or on a GPU, has to beat.
//!
//! The counts come from the model NEBULA fits rather than from a uniform draw:
//! a per-subject gamma frailty shared by that subject's cells, a per-cell gamma
//! overdispersion on top, and a Poisson draw from the product. A uniform
//! generator would put every gene on the same NEBULA-LN branch and hide the
//! stage-two cost that dominates real data.
//!
//! The measured cells are:
//!
//! * `ptmg` - one [`ptmg_value_and_gradient`] call on one gene. The stage-one
//!   inner kernel; L-BFGS-B calls it tens to hundreds of times per gene.
//! * `pml` - one [`opt_pml`] call on one gene. The stage-two and stage-three
//!   inner kernel; stage two calls it once per Nelder-Mead and per polish
//!   stencil point.
//! * `ln` - `nebula_sparse` end to end on NEBULA's own defaults.
//! * `hl` - the same, forced onto NEBULA-HL, so every gene pays the full
//!   stage-two search. This is the upper bound and the shape a GPU would target.
//!
//! `ln` minus `hl` on the same data is the stage-two cost; `pml` against the
//! `hl` total says how many `opt_pml` calls a gene really takes.
//!
//! Run with:
//! ```text
//! cargo bench --bench nebula_bench
//! ```
//!
//! `NEBULA_BENCH_GENES`, `NEBULA_BENCH_CELLS`, `NEBULA_BENCH_SUBJECTS` and
//! `NEBULA_BENCH_COEF` change the shape. `NEBULA_BENCH_ONLY=ptmg,pml` runs a
//! comma-separated subset of the cells. `NEBULA_BENCH_SWEEP=1` adds a cell-count
//! sweep of the two inner kernels, which is the scaling a GPU port cares about.

use std::env;
use std::hint::black_box;
use std::time::{Duration, Instant};

use rand::prelude::*;
use rand::rngs::SmallRng;
use rand_distr::{Distribution, Gamma, LogNormal, Poisson};

use edge_rs::prelude::*;
use edge_rs::sc::nebula::{NebulaMethod, NebulaParams, nebula_sparse};
use edge_rs::sc::pml::{PmlData, PmlParams, PmlVariance, opt_pml};
use edge_rs::sc::ptmg::{GeneData, ptmg_value_and_gradient};

////////////
// Shapes //
////////////

/// Default gene count. Small enough that the forced-HL cell finishes in a
/// minute or two; a filtered feature set is two orders of magnitude bigger and
/// scales linearly, since genes are the parallel axis.
const DEFAULT_GENES: usize = 200;

/// Default cell count. Large enough that the `O(n_cells)` inner loop dominates
/// the per-gene setup, which is the regime every real single-cell run is in.
const DEFAULT_CELLS: usize = 20_000;

/// Default subject count. Twenty donors at a thousand cells each, which is well
/// above [`MIN_CELLS_PER_SUBJECT_LN`]-equivalent so the LN branch is live.
const DEFAULT_SUBJECTS: usize = 20;

/// Default design width: intercept, a subject-level group and a cell-level
/// covariate. Three is what `tests/e2e_nebula.rs` uses and what a typical
/// NEBULA call has.
const DEFAULT_COEF: usize = 3;

/// Cell counts for the inner-kernel sweep, to expose the `O(n_cells)` scaling.
const SWEEP_CELLS: [usize; 4] = [5_000, 20_000, 80_000, 320_000];

/// Seed for the count generator. Fixed, so two runs measure the same problem.
const SEED: u64 = 0x5EED_1234;

/// Subject-level overdispersion the counts are drawn with, nebula's `sigma^2`.
const TRUE_SIGMA: f64 = 0.25;

/// Cell-level overdispersion the counts are drawn with, nebula's `phi^-1`.
const TRUE_PHI_INV: f64 = 0.5;

/// Repeats for the inner-kernel cells, which are sub-millisecond individually.
const INNER_REPEATS: usize = 20;

///////////////
// Test data //
///////////////

/// One generated problem: counts in CSR, the design, the offsets and the labels.
struct Problem {
    /// Counts, gene-major CSR over `(n_genes, n_cells)`, positives only.
    counts: CompressedSparse<f64>,
    /// Design, row-major `n_cells * n_coef`, first column the intercept.
    design: Vec<f64>,
    /// Strictly positive scaling factor per cell.
    offset: Vec<f64>,
    /// Subject of each cell, contiguous by construction.
    subject_id: Vec<usize>,
    /// Number of genes.
    n_genes: usize,
    /// Number of subjects.
    n_subjects: usize,
    /// Number of design columns.
    n_coef: usize,
}

/// Draws a NEBULA-shaped problem from the gamma-gamma-Poisson model.
///
/// Cells are laid out subject by subject, which is what `nebula_sparse`
/// requires. Each gene gets its own baseline expression and its own draw of the
/// per-subject frailties, so genes differ in how hard they are to fit.
///
/// ### Params
///
/// * `n_genes` - Number of genes
/// * `n_cells` - Number of cells, split as evenly as possible across subjects
/// * `n_subjects` - Number of subjects
/// * `n_coef` - Design width, at least two: intercept plus a group column
///
/// ### Returns
///
/// The assembled [`Problem`].
fn make_problem(n_genes: usize, n_cells: usize, n_subjects: usize, n_coef: usize) -> Problem {
    assert!(n_coef >= 2, "the design needs an intercept and a group");
    let mut rng = SmallRng::seed_from_u64(SEED);

    // Cells blocked by subject, sizes uneven so the ragged-block path is live.
    let mut subject_id = Vec::with_capacity(n_cells);
    let base = n_cells / n_subjects;
    let mut assigned = 0;
    for s in 0..n_subjects {
        let extra = if s < n_cells % n_subjects { 1 } else { 0 };
        let size = if s + 1 == n_subjects {
            n_cells - assigned
        } else {
            base + extra
        };
        subject_id.extend(std::iter::repeat_n(s, size));
        assigned += size;
    }

    // Library sizes, lognormal as they are in practice.
    let lib = LogNormal::new(0.0, 0.4).expect("valid lognormal");
    let offset: Vec<f64> = (0..n_cells).map(|_| lib.sample(&mut rng)).collect();

    // Design: intercept, subject-level group, then cell-level covariates.
    let mut design = vec![0.0; n_cells * n_coef];
    for c in 0..n_cells {
        design[c * n_coef] = 1.0;
        design[c * n_coef + 1] = if subject_id[c] % 2 == 0 { 0.0 } else { 1.0 };
        for j in 2..n_coef {
            design[c * n_coef + j] = rng.random_range(-1.0..1.0);
        }
    }

    // Counts. `beta` varies by gene so the filter keeps a realistic spread.
    let frailty = Gamma::new(1.0 / TRUE_SIGMA, TRUE_SIGMA).expect("valid gamma");
    let cell_noise = Gamma::new(1.0 / TRUE_PHI_INV, TRUE_PHI_INV).expect("valid gamma");

    let mut data = Vec::new();
    let mut indices = Vec::new();
    let mut indptr = Vec::with_capacity(n_genes + 1);
    indptr.push(0u32);

    let mut beta = vec![0.0; n_coef];
    for _ in 0..n_genes {
        beta[0] = rng.random_range(-2.0..1.5);
        for b in beta.iter_mut().skip(1) {
            *b = rng.random_range(-0.5..0.5);
        }
        let w: Vec<f64> = (0..n_subjects).map(|_| frailty.sample(&mut rng)).collect();

        for c in 0..n_cells {
            let mut eta = 0.0f64;
            for j in 0..n_coef {
                eta += design[c * n_coef + j] * beta[j];
            }
            let mu = offset[c] * eta.exp() * w[subject_id[c]] * cell_noise.sample(&mut rng);
            let y = Poisson::new(mu.max(1e-12)).expect("positive rate").sample(&mut rng);
            if y > 0.0 {
                data.push(y);
                indices.push(c as u32);
            }
        }
        indptr.push(data.len() as u32);
    }

    let counts = CompressedSparse::from_parts(
        data,
        indices,
        indptr,
        SparseFormat::Csr,
        (n_genes, n_cells),
    )
    .expect("well-formed CSR");

    Problem {
        counts,
        design,
        offset,
        subject_id,
        n_genes,
        n_subjects,
        n_coef,
    }
}

/// The pieces one gene's inner kernels need, extracted from a [`Problem`].
///
/// Mirrors what `fit_gene` assembles: the centred design is skipped, since the
/// cost of a kernel call does not depend on whether the design was centred.
struct OneGene {
    /// Log offset per cell.
    log_offset: Vec<f64>,
    /// This gene's positive counts.
    counts: Vec<f64>,
    /// Cell index of each positive count.
    cells: Vec<usize>,
    /// Count total per subject.
    subject_totals: Vec<f64>,
    /// Subject boundaries, length `n_subjects + 1`.
    fid: Vec<usize>,
}

/// Pulls gene `g` out of a problem in the form the kernels want.
///
/// ### Params
///
/// * `problem` - The generated problem
/// * `g` - Gene index
///
/// ### Returns
///
/// The per-gene view, borrowed against the problem's design.
fn one_gene(problem: &Problem, g: usize) -> OneGene {
    let lo = problem.counts.indptr[g] as usize;
    let hi = problem.counts.indptr[g + 1] as usize;
    let cells: Vec<usize> = problem.counts.indices[lo..hi]
        .iter()
        .map(|&c| c as usize)
        .collect();
    let counts = problem.counts.data[lo..hi].to_vec();

    let mut fid = vec![0usize; problem.n_subjects + 1];
    for &s in &problem.subject_id {
        fid[s + 1] += 1;
    }
    for s in 0..problem.n_subjects {
        fid[s + 1] += fid[s];
    }

    let mut subject_totals = vec![0.0; problem.n_subjects];
    for (&c, &y) in cells.iter().zip(counts.iter()) {
        subject_totals[problem.subject_id[c]] += y;
    }

    OneGene {
        log_offset: problem.offset.iter().map(|v| v.ln()).collect(),
        counts,
        cells,
        subject_totals,
        fid,
    }
}

/// Picks the gene with the median number of positive counts.
///
/// The cheapest gene and the most expressed gene are not representative of the
/// per-call cost; the median one is.
///
/// ### Params
///
/// * `problem` - The generated problem
///
/// ### Returns
///
/// The gene index.
fn median_gene(problem: &Problem) -> usize {
    let mut nnz: Vec<(usize, usize)> = (0..problem.n_genes)
        .map(|g| {
            (
                (problem.counts.indptr[g + 1] - problem.counts.indptr[g]) as usize,
                g,
            )
        })
        .collect();
    nnz.sort_unstable();
    nnz[nnz.len() / 2].1
}

///////////////
// Reporting //
///////////////

/// Prints one timed cell.
///
/// ### Params
///
/// * `label` - Cell name
/// * `elapsed` - Total time for `repeats` runs
/// * `repeats` - Number of runs folded into `elapsed`
/// * `note` - Free-text suffix, usually the shape or a derived rate
fn report(label: &str, elapsed: Duration, repeats: usize, note: &str) {
    let per = elapsed.as_secs_f64() / repeats as f64;
    let unit = if per < 1e-3 {
        format!("{:>9.1} us", per * 1e6)
    } else if per < 1.0 {
        format!("{:>9.2} ms", per * 1e3)
    } else {
        format!("{per:>9.3} s ")
    };
    println!("{label:<24} {unit}   {note}");
}

/// Reads a `usize` from the environment, falling back to a default.
///
/// ### Params
///
/// * `key` - Environment variable name
/// * `fallback` - Value to use when unset or unparseable
///
/// ### Returns
///
/// The resolved value.
fn env_usize(key: &str, fallback: usize) -> usize {
    env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(fallback)
}

/// Whether a named cell should run, given `NEBULA_BENCH_ONLY`.
///
/// ### Params
///
/// * `name` - Cell name
///
/// ### Returns
///
/// `true` when the filter is unset or lists the cell.
fn selected(name: &str) -> bool {
    match env::var("NEBULA_BENCH_ONLY") {
        Ok(list) => list.split(',').any(|s| s.trim() == name),
        Err(_) => true,
    }
}

//////////
// Main //
//////////

fn main() {
    let n_genes = env_usize("NEBULA_BENCH_GENES", DEFAULT_GENES);
    let n_cells = env_usize("NEBULA_BENCH_CELLS", DEFAULT_CELLS);
    let n_subjects = env_usize("NEBULA_BENCH_SUBJECTS", DEFAULT_SUBJECTS);
    let n_coef = env_usize("NEBULA_BENCH_COEF", DEFAULT_COEF);

    println!(
        "\nNEBULA bench: {n_genes} genes, {n_cells} cells, {n_subjects} subjects, {n_coef} coefficients"
    );

    let build = Instant::now();
    let problem = make_problem(n_genes, n_cells, n_subjects, n_coef);
    let nnz = problem.counts.data.len();
    println!(
        "generated in {:.1} s, {} nnz, {:.1}% dense\n",
        build.elapsed().as_secs_f64(),
        nnz,
        100.0 * nnz as f64 / (n_genes * n_cells) as f64
    );

    if selected("ptmg") || selected("pml") {
        let g = median_gene(&problem);
        let gene = one_gene(&problem, g);
        inner_kernels(&problem, &gene, "");
    }

    if selected("ln") {
        let t = Instant::now();
        let fit = nebula_sparse(
            &problem.counts,
            &problem.subject_id,
            &problem.design,
            problem.n_coef,
            Some(&problem.offset),
            None,
        )
        .expect("nebula-ln fits");
        report(
            "ln (end to end)",
            t.elapsed(),
            1,
            &format!("{} genes kept", fit.gene_index.len()),
        );
    }

    if selected("hl") {
        let params = NebulaParams {
            method: NebulaMethod::Hl,
            ..NebulaParams::default()
        };
        let t = Instant::now();
        let fit = nebula_sparse(
            &problem.counts,
            &problem.subject_id,
            &problem.design,
            problem.n_coef,
            Some(&problem.offset),
            Some(params),
        )
        .expect("nebula-hl fits");
        report(
            "hl (end to end)",
            t.elapsed(),
            1,
            &format!("{} genes kept", fit.gene_index.len()),
        );
    }

    if env::var("NEBULA_BENCH_SWEEP").is_ok() {
        println!("\ncell-count sweep of the inner kernels:");
        for &cells in SWEEP_CELLS.iter() {
            let p = make_problem(8, cells, n_subjects, n_coef);
            let g = median_gene(&p);
            let gene = one_gene(&p, g);
            inner_kernels(&p, &gene, &format!("n_cells = {cells}"));
        }
    }
}

/// Times one `ptmg_value_and_gradient` call and one `opt_pml` call.
///
/// ### Params
///
/// * `problem` - The problem the gene came from, for the design and shape
/// * `gene` - The extracted gene
/// * `note` - Suffix for the report lines
fn inner_kernels(problem: &Problem, gene: &OneGene, note: &str) {
    let data = GeneData::new(
        &problem.design,
        &gene.log_offset,
        &gene.counts,
        &gene.cells,
        &gene.subject_totals,
        &gene.fid,
        problem.n_coef,
    )
    .expect("well-formed gene");

    let mut params = vec![0.0; problem.n_coef + 2];
    params[0] = -1.0;
    params[problem.n_coef] = TRUE_SIGMA;
    params[problem.n_coef + 1] = 1.0 / TRUE_PHI_INV;

    if selected("ptmg") {
        // One untimed call first: the first touch of the design pulls it in.
        black_box(ptmg_value_and_gradient(&data, &params));
        let t = Instant::now();
        for _ in 0..INNER_REPEATS {
            black_box(ptmg_value_and_gradient(black_box(&data), black_box(&params)));
        }
        report("ptmg (one eval)", t.elapsed(), INNER_REPEATS, note);
    }

    if selected("pml") {
        let pml = PmlData {
            design: &problem.design,
            offset: &gene.log_offset,
            counts: &gene.counts,
            cell_index: &gene.cells,
            subject_start: &gene.fid,
            subject_total: &gene.subject_totals,
        };
        let variance = PmlVariance {
            subject: TRUE_SIGMA,
            cell: 1.0 / TRUE_PHI_INV,
        };
        let beta = vec![0.0; problem.n_coef];
        let pml_params = PmlParams {
            ord: 1,
            ..PmlParams::default()
        };

        black_box(opt_pml(&pml, &beta, &variance, Some(pml_params)).expect("pml fits"));
        let t = Instant::now();
        for _ in 0..INNER_REPEATS {
            black_box(
                opt_pml(
                    black_box(&pml),
                    black_box(&beta),
                    black_box(&variance),
                    Some(pml_params),
                )
                .expect("pml fits"),
            );
        }
        report("pml (one solve)", t.elapsed(), INNER_REPEATS, note);
    }
}
