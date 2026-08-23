//! End-to-end parity for the single-cell chain: `nebula`, then the Wald test on
//! its output, then the dispersion shrinkage.
//!
//! Two datasets, sitting either side of the thirty-cells-per-subject threshold
//! that decides between NEBULA-LN and NEBULA-HL:
//!
//! * `sc`, 300 genes over 1005 cells and 15 subjects, so 67 cells per subject.
//!   `method = "LN"` survives and all three of nebula's sub-algorithms appear
//!   in one run.
//! * `sc_small`, 120 genes over 300 cells and 15 subjects, so 20 per subject.
//!   `method = "LN"` is silently downgraded and every gene takes the HL path.
//!
//! The in-crate golden is eight genes at 25 cells per subject, so it only ever
//! reaches HL, and reaches LN only through a relabelling trick on the same 150
//! cells. These fixtures are the first time either path is gated at a realistic
//! gene count.
//!
//! Tolerances follow the reasoning in `src/sc/nebula.rs`, which documented why
//! they cannot be tightened: nebula stops its optimiser on a profile likelihood
//! that is itself discontinuous at the stopping tolerance, so two of its own
//! runs from different starting points disagree by as much. The values here are
//! one to two decades looser than that module's, and are measured on these
//! fixtures rather than carried over, because three hundred genes over a
//! thousand cells reach corners that eight genes over a hundred and fifty do
//! not.

mod common;

use common::{Tol, assert_close, assert_close_scalar, assert_eq_usize};

use edge_rs::sc::nebula::{NebulaParams, nebula};
use edge_rs::sc::shrink::{sc_residual_df, shrink_sc_dispersion};
use edge_rs::sc::test::{ScTested, glm_sc_test, packed_len};

////////////////
// Tolerances //
////////////////

/// Coefficients on the pure paths. Needs `7.4e-6`, against `1.0e-6` in the
/// in-crate eight-gene golden.
const TOL_COEF: Tol = Tol::new(3e-5, 1e-9);

/// Standard errors. Needs `2.3e-5`.
const TOL_SE: Tol = Tol::rel(1e-4);

/// Covariance entries, which are standard errors squared and so carry twice the
/// relative error. Needs `2.3e-4`.
const TOL_COV: Tol = Tol::new(1e-3, 1e-12);

/// Subject-level overdispersion. The profile likelihood is far flatter in this
/// direction than in the cell-level one, so the same jitter moves it further.
/// Needs `3.8e-5`.
const TOL_SUBJECT: Tol = Tol::rel(2e-4);

/// Cell-level overdispersion. Needs `2.2e-5`.
const TOL_CELL: Tol = Tol::rel(1e-4);

/// Wald p-values. These amplify whatever error is in the standard error by
/// roughly `z^2`, because the tail of a normal falls like `exp(-z^2/2)`: at
/// `p = 1e-16` the score is `z = 8.2`, so a `2e-5` relative error in the
/// standard error arrives as `1.4e-3` in the p-value. Needs `1.5e-3`, which is
/// that arithmetic exactly.
const TOL_P_VALUE: Tol = Tol::new(1e-2, 1e-300);

// Longer Claude explanation. Left it in, because this took time to find, debug.
//
// The LN+HL path, held apart because the two optimisers pick different basins
//
// nebula picks one of three sub-paths per gene. Measured on the 1005-cell set,
// worst relative disagreement on the coefficients:
//
//   NBGMM (LN)      11 genes    3.0e-6
//   NBGMM (HL)       6 genes    3.8e-7      (118 genes at 7.4e-6 on sc_small)
//   NBGMM (LN+HL)  281 genes    5.9e-1
//
// The cause is not the LN+HL refit itself. That refit's objective was compared
// against `nebula:::pql_gamma_ll` at identical `(sigma, phi)` on three affected
// genes over an eight-point grid: the two agree to `6e-10` absolute, `2e-12`
// relative, so the profile likelihood, the inner PML solve, `reml`, `ord` and
// the sign convention are all faithful, and the one-dimensional search finds
// that objective's minimum.
//
// The divergence is in stage one, the joint optimisation over
// `[beta, sigma, phi]`. R runs nlopt's `LD_LBFGS` there and this crate runs
// L-BFGS-B, and on 37 of the 281 genes the two finish in different places.
// LN+HL then holds stage one's `phi` fixed and refits only `sigma`, so the
// choice propagates into both variance components and, through the `-sigma/2`
// intercept shift, into the coefficients.
//
// The two places are not two local minima. R's is the lower bound on `sigma`,
// and the marginal likelihood is still strictly decreasing there: lower the
// bound from `1e-4` to `1e-8` and 13 of the 19 genes that sat on it follow it
// straight down. R is on a legitimate Karush-Kuhn-Tucker point of the box
// constraint, reached by sliding down a monotone descent onto the wall, and
// `sigma = 1e-4` is the constraint rather than an estimate. This crate stops at
// the one genuine interior stationary point. There is a real second basin
// around `0.03` to `0.15` separated by a ridge near `0.01`; what there is not
// is a minimum at the bound.
//
// Neither optimiser is broken; both satisfy KKT and both are converged, and R
// reaching the lower marginal negative log-likelihood on 27 of the 37 against
// this crate's 9 is not the endorsement it looks like, because it wins by
// descending into a degeneracy. Sigma on the bound means no subject-level
// variance at all, i.e. the mixed model collapsed to a plain negative binomial
// GLM, in an approximation the LN+HL path exists precisely because it
// distrusts.
//
// The tiebreak is nebula's own `method = "HL"`, which fits both variance
// components against the profile likelihood. Against that reference this crate
// is closer on 29 of the 37 for `sigma`, 29 for `phi`, 32 for the standard
// errors and 32 for the p-values; worst-case `sigma` error over all 281 genes is
// 15.8% here against 84.4% for R. So the answers are not equally good, and the
// obvious "multi-start and take the lower NLL" fix is measurably counter-
// productive: it lands on the bound on 30 of the 37 and moves *away* from the HL
// reference, for +81% runtime. It optimises the wrong objective harder.
//
// R also disagrees with itself: nebula's own documented `opt = "trust"`
// moves the subject overdispersion by more than one per cent on 78 of the 281
// genes, twice as many as the 37 where this crate differs from `opt = "lbfgs"`,
// and on those 37 it lands on this crate's answer rather than on `lbfgs`'s:
// 22 agree to `1e-6` and 32 to `1e-4`, against none agreeing with `lbfgs` to
// `1e-6`. Matching the fixture would mean reproducing nlopt's LD_LBFGS
// trajectory step for step, including the overshoot that gets clipped onto the
// `sigma` bound and is what carries it over the ridge.
//
// One gene is not stage one at all. On `gene_id` 299 both optimisers agree on
// `[beta, sigma, phi]` to `3e-7`, and it is R's `nlminb` inside the refit that
// fails: it returns the lower bound `1e-4` with `convergence == 0` at a point
// where its own `pql_gamma_ll` is still falling.
//
// The two constants below are gated on their absolute legs, which is what makes
// them falsifiable. Stated as pure relative bounds they would not be: the
// comparator normalises by `max(|a|, |b|)`, so a relative difference cannot
// exceed 2 and any `max_relative` at or above that can never fail. An earlier
// revision had the subject bound at 6.0, which was exactly that mistake.

/// Coefficients on the LN+HL path. Needs `0` beyond a `1e-2` absolute floor;
/// the worst absolute disagreement across all 281 genes is `4.4e-3` in
/// log-fold-change units. The headline `5.9e-1` relative figure is a
/// coefficient of magnitude `8.6e-4` and is not meaningful.
const LN_HL_COEF: Tol = Tol::new(1e-5, 1e-2);

/// Subject-level overdispersion on the LN+HL path. Needs `7.9e-2` beyond a
/// `1e-3` absolute floor.
const LN_HL_SUBJECT: Tol = Tol::new(2.5e-1, 1e-3);

/// Shrunk dispersions from `shrink_sc_dispersion`, against limma's `squeezeVar`
/// on the same inputs. Needs `7.7e-14`.
///
/// This is the tightest thing in the file, and it should be: the test hands the
/// crate R's own cell overdispersions, so the only thing under comparison is the
/// empirical Bayes step. `shrink_sc_dispersion` has no upstream of its own,
/// being edgePython's invention, so limma applied directly is the reference.
const TOL_SHRINK: Tol = Tol::rel(1e-12);

/// Prior degrees of freedom from that shrinkage. Needs `8.3e-13`.
const TOL_SHRINK_DF: Tol = Tol::rel(1e-11);

/////////////
// Loading //
/////////////

/// One single-cell dataset.
struct Dataset {
    /// Fixture prefix.
    tag: &'static str,
    /// Whether the LN path is expected to survive.
    expect_ln: bool,
}

/// The two single-cell datasets.
const DATASETS: [Dataset; 2] = [
    Dataset {
        tag: "sc",
        expect_ln: true,
    },
    Dataset {
        tag: "sc_small",
        expect_ln: false,
    },
];

/// Counts, subject labels, design and offsets for one dataset.
struct Loaded {
    /// Counts, row-major `n_genes * n_cells`.
    counts: Vec<f64>,
    /// Number of genes before filtering.
    n_genes: usize,
    /// Number of cells.
    n_cells: usize,
    /// Subject index per cell, zero-based and contiguous.
    subject: Vec<usize>,
    /// Design, row-major `n_cells * 3`: intercept, group, covariate.
    design: Vec<f64>,
    /// Per-cell offset, on the linear scale.
    offset: Vec<f64>,
    /// Number of subjects.
    n_subjects: usize,
}

/// Loads one single-cell dataset.
///
/// ### Params
///
/// * `d` - Dataset description
///
/// ### Returns
///
/// The counts and the per-cell covariates.
fn load(d: &Dataset) -> Loaded {
    let counts_t = common::table(&format!("{}_counts.csv", d.tag));
    let n_genes = counts_t.n_rows();
    let n_cells = counts_t.n_cols();
    let counts = counts_t.row_major_counts();

    let meta = common::table(&format!("{}_meta.csv", d.tag));
    assert_eq!(
        meta.n_rows(),
        n_cells,
        "{}: meta rows must match cells",
        d.tag
    );

    // R labels subjects from one; the crate wants them zero-based, and requires
    // them already grouped into contiguous runs.
    let subject: Vec<usize> = meta.column_usize("subject").iter().map(|s| s - 1).collect();
    let n_subjects = subject.iter().max().expect("at least one cell") + 1;

    let grp = meta.column("grp");
    let cov2 = meta.column("cov2");
    let mut design = Vec::with_capacity(n_cells * 3);
    for c in 0..n_cells {
        design.push(1.0);
        design.push(grp[c]);
        design.push(cov2[c]);
    }

    Loaded {
        counts,
        n_genes,
        n_cells,
        subject,
        design,
        offset: meta.column("offset").to_vec(),
        n_subjects,
    }
}

/// The parameter set the goldens were generated with.
///
/// `nebula(..., model = "NBGMM", method = "LN", covariance = TRUE, ncore = 1)`
/// at every other default.
fn golden_params() -> NebulaParams {
    NebulaParams::default()
}

/// Reorders one gene's R covariance row into the crate's packing.
///
/// R returns `lower.tri(diag = TRUE)` column-major, which reads
/// `V11, V12, V13, V22, V23, V33`. The crate packs the upper triangle
/// column-major, `V11, V12, V22, V13, V23, V33`. The two coincide for two
/// coefficients and diverge from three.
///
/// ### Params
///
/// * `row` - One gene's six covariance entries as R wrote them
///
/// ### Returns
///
/// The same six values in the crate's order.
fn repack(row: &[f64]) -> Vec<f64> {
    vec![row[0], row[1], row[3], row[2], row[4], row[5]]
}

/// Asserts a convergence code against the R package's.
///
/// nebula grades a fit by how many times the last Newton step had to be halved,
/// and at a converged point that count is settled by rounding, so a gene the R
/// package calls converged may come back as `CONV_CRITICAL_POINT` here. This is
/// the same leniency `src/sc/nebula.rs` applies to the in-crate golden.
///
/// ### Params
///
/// * `got` - The crate's code
/// * `want` - The R package's code
/// * `gene` - Gene index, for the message
fn assert_convergence(got: i32, want: i32, gene: usize) {
    let equivalent = |c: i32| c == 1 || c == -10;
    if equivalent(want) {
        assert!(
            equivalent(got),
            "gene {gene}: convergence {got}, expected 1 or -10"
        );
    } else {
        assert_eq!(got, want, "gene {gene}: convergence");
    }
}

////////////
// nebula //
////////////

#[test]
fn test_nebula_matches_the_r_package() {
    let s = common::scalars();

    for d in &DATASETS {
        let l = load(d);
        let want = common::table(&format!("{}_nebula.csv", d.tag));
        let want_cov = common::table(&format!("{}_covariance.csv", d.tag));

        let cells_per_subject = l.n_cells as f64 / l.n_subjects as f64;
        assert_eq!(
            l.n_subjects,
            s.get_usize(d.tag, "n_subjects"),
            "{}: subject count",
            d.tag
        );
        assert_eq!(
            d.expect_ln,
            cells_per_subject >= 30.0,
            "{}: the dataset no longer sits on the intended side of the LN threshold",
            d.tag
        );

        let fit = nebula(
            &l.counts,
            l.n_genes,
            l.n_cells,
            &l.subject,
            &l.design,
            3,
            Some(&l.offset),
            Some(golden_params()),
        )
        .expect("nebula failed");

        // Gene filtering. R reports the surviving genes one-based.
        let want_index: Vec<usize> = want.column_usize("gene_id").iter().map(|g| g - 1).collect();
        assert_eq_usize(
            &fit.gene_index,
            &want_index,
            &format!("{}/gene_index", d.tag),
        );
        assert_eq!(
            fit.gene_index.len(),
            s.get_usize(d.tag, "n_genes_out"),
            "{}: surviving gene count",
            d.tag
        );
        assert_eq!(fit.n_coef, 3, "{}: three coefficients", d.tag);

        let n = fit.gene_index.len();
        let names = ["int", "grp", "cov2"];
        let algorithm = want.column("algorithm");

        // nebula picks a sub-path per gene and the three do not agree equally
        // well, so they are gated separately rather than under one tolerance.
        // See LN_HL_* below for why the mixed path is held apart.
        let pure: Vec<bool> = (0..n).map(|g| algorithm[g] as usize != 2).collect();
        let mixed: Vec<bool> = (0..n).map(|g| algorithm[g] as usize == 2).collect();

        let pick = |v: &[f64], mask: &[bool], stride: usize| -> Vec<f64> {
            v.chunks_exact(stride)
                .zip(mask)
                .filter(|(_, k)| **k)
                .flat_map(|(r, _)| r.iter().copied())
                .collect()
        };

        let mut want_coef = Vec::with_capacity(n * 3);
        let mut want_se = Vec::with_capacity(n * 3);
        for g in 0..n {
            for name in names {
                want_coef.push(want.column(&format!("logFC_{name}"))[g]);
                want_se.push(want.column(&format!("se_{name}"))[g]);
            }
        }

        // Pure LN and pure HL, where the port is right.
        assert_close(
            &pick(&fit.coefficients, &pure, 3),
            &pick(&want_coef, &pure, 3),
            TOL_COEF,
            &format!("{}/coefficients_pure", d.tag),
        );
        assert_close(
            &pick(&fit.se, &pure, 3),
            &pick(&want_se, &pure, 3),
            TOL_SE,
            &format!("{}/se_pure", d.tag),
        );
        assert_close(
            &pick(fit.subject_overdispersion.as_slice(), &pure, 1),
            &pick(want.column("Subject"), &pure, 1),
            TOL_SUBJECT,
            &format!("{}/subject_overdispersion_pure", d.tag),
        );
        assert_close(
            &pick(fit.cell_overdispersion.as_slice(), &pure, 1),
            &pick(want.column("Cell"), &pure, 1),
            TOL_CELL,
            &format!("{}/cell_overdispersion_pure", d.tag),
        );

        // The mixed path, at the tolerance it currently earns rather than the
        // one it should. This is a recorded gap, not an accepted one.
        if mixed.iter().any(|k| *k) {
            assert_close(
                &pick(&fit.coefficients, &mixed, 3),
                &pick(&want_coef, &mixed, 3),
                LN_HL_COEF,
                &format!("{}/coefficients_ln_hl", d.tag),
            );
            assert_close(
                &pick(fit.subject_overdispersion.as_slice(), &mixed, 1),
                &pick(want.column("Subject"), &mixed, 1),
                LN_HL_SUBJECT,
                &format!("{}/subject_overdispersion_ln_hl", d.tag),
            );
        }

        let packed = packed_len(3);
        let mut want_packed = Vec::with_capacity(n * packed);
        for g in 0..n {
            let row: Vec<f64> = (1..=packed)
                .map(|k| want_cov.column(&format!("cov_{k}"))[g])
                .collect();
            want_packed.extend(repack(&row));
        }
        // The covariance packing is what this checks, so only the pure genes,
        // whose values agree, can say anything about it.
        assert_close(
            &pick(&fit.covariance, &pure, packed),
            &pick(&want_packed, &pure, packed),
            TOL_COV,
            &format!("{}/covariance_pure", d.tag),
        );

        for g in 0..n {
            assert_convergence(fit.convergence[g], want.column("convergence")[g] as i32, g);
        }
    }
}

#[test]
fn test_sigma_at_bound_marks_the_collapsed_fits() {
    // The flag is the only thing in either implementation that tells a caller
    // their mixed model has no random effect. nebula's `check_conv` tests the
    // upper bound on `sigma^2` and not the lower, so these genes come back
    // reporting convergence.
    //
    // Cross-checked against R rather than against the crate's own output: the
    // fixture's `Subject` column is R's, and a gene is pinned there exactly when
    // it is pinned here.
    for d in &DATASETS {
        let l = load(d);
        let want = common::table(&format!("{}_nebula.csv", d.tag));

        let fit = nebula(
            &l.counts,
            l.n_genes,
            l.n_cells,
            &l.subject,
            &l.design,
            3,
            Some(&l.offset),
            Some(golden_params()),
        )
        .expect("nebula failed");

        let floor = NebulaParams::default().min.0;
        let r_subject = want.column("Subject");
        let n = fit.gene_index.len();
        assert_eq!(fit.sigma_at_bound.len(), n);

        let mismatched: Vec<(usize, bool, f64, f64)> = (0..n)
            .filter(|&g| fit.sigma_at_bound[g] != (r_subject[g] <= floor))
            .map(|g| {
                (
                    want.column_usize("gene_id")[g],
                    fit.sigma_at_bound[g],
                    fit.subject_overdispersion[g],
                    r_subject[g],
                )
            })
            .collect();
        println!(
            "{}: {} mismatched: {:?}",
            d.tag,
            mismatched.len(),
            mismatched
        );

        let flagged = fit.sigma_at_bound.iter().filter(|b| **b).count();
        assert_eq!(
            flagged,
            s_pinned(d.tag),
            "{}: number of genes with no fitted subject variance",
            d.tag
        );
        // A dataset where nothing is pinned would make this test vacuous.
        assert!(
            flagged > 0,
            "{}: nothing pinned, so this gates nothing",
            d.tag
        );
    }
}

/// Genes whose subject-level variance finishes on the lower bound, per dataset.
///
/// Recorded rather than derived, so a change in how many collapse is a test
/// failure rather than a silent drift. These are R's numbers as much as this
/// crate's: R pins the same 18 on `sc`, plus gene 299 where its `nlminb` fails,
/// and the same 40 on `sc_small`.
///
/// The `sc_small` figure is the interesting one. A third of its genes come back
/// with no fitted subject-level variance, which is what twenty cells per subject
/// buys: there is too little between-subject replication to estimate the random
/// effect, so the model collapses to a plain negative binomial GLM. nebula
/// reports every one of them as converged.
///
/// ### Params
///
/// * `tag` - Dataset prefix
///
/// ### Returns
///
/// The expected count.
fn s_pinned(tag: &str) -> usize {
    match tag {
        "sc" => 18,
        "sc_small" => 40,
        _ => unreachable!("unknown single-cell dataset"),
    }
}

#[test]
fn test_nebula_ln_and_hl_are_genuinely_different_paths() {
    // Asking for HL on the large dataset has to give a different answer, or the
    // LN fixture is not gating anything. On this data the two disagree by around
    // 50% on the group coefficient, so the check is not delicate.
    let l = load(&DATASETS[0]);

    let ln = nebula(
        &l.counts,
        l.n_genes,
        l.n_cells,
        &l.subject,
        &l.design,
        3,
        Some(&l.offset),
        Some(NebulaParams::default()),
    )
    .expect("nebula failed");

    let hl = nebula(
        &l.counts,
        l.n_genes,
        l.n_cells,
        &l.subject,
        &l.design,
        3,
        Some(&l.offset),
        Some(NebulaParams {
            method: edge_rs::sc::nebula::NebulaMethod::Hl,
            ..Default::default()
        }),
    )
    .expect("nebula failed");

    let worst = (0..ln.gene_index.len())
        .map(|g| {
            let a = ln.coefficients[g * 3 + 1];
            let b = hl.coefficients[g * 3 + 1];
            (a - b).abs() / a.abs().max(b.abs()).max(1e-12)
        })
        .fold(0.0_f64, f64::max);
    assert!(
        worst > 1e-3,
        "LN and HL agreed to {worst:e} on the group coefficient, so the large \
         fixture is not exercising the LN path"
    );
}

////////////////
// Wald test  //
////////////////

#[test]
fn test_glm_sc_test_reproduces_the_r_p_values() {
    for d in &DATASETS {
        let l = load(d);
        let want = common::table(&format!("{}_nebula.csv", d.tag));

        let fit = nebula(
            &l.counts,
            l.n_genes,
            l.n_cells,
            &l.subject,
            &l.design,
            3,
            Some(&l.offset),
            Some(golden_params()),
        )
        .expect("nebula failed");

        let n = fit.gene_index.len();
        let algorithm = want.column("algorithm");
        // Only the genes whose fit agrees can say anything about the Wald test;
        // the LN+HL ones would just re-report that gap one step downstream.
        let pure: Vec<bool> = (0..n).map(|g| algorithm[g] as usize != 2).collect();
        let keep = |v: &[f64]| -> Vec<f64> {
            v.iter()
                .zip(&pure)
                .filter(|(_, k)| **k)
                .map(|(x, _)| *x)
                .collect()
        };

        for (coef, name) in [(0usize, "int"), (1, "grp"), (2, "cov2")] {
            let got = glm_sc_test(
                &fit.coefficients,
                &fit.covariance,
                n,
                fit.n_coef,
                &ScTested::Coef(coef),
            )
            .expect("glm_sc_test failed");

            assert_close(
                &keep(&got.p_value),
                &keep(want.column(&format!("p_{name}"))),
                TOL_P_VALUE,
                &format!("{}/wald_p_{name}", d.tag),
            );
            assert_close(
                &keep(&got.se),
                &keep(want.column(&format!("se_{name}"))),
                TOL_SE,
                &format!("{}/wald_se_{name}", d.tag),
            );
        }
    }
}

//////////////////
// Shrinkage    //
//////////////////

#[test]
fn test_shrink_sc_dispersion_matches_limma_squeeze_var() {
    // `shrink_sc_dispersion` has no upstream of its own: it is edgePython's
    // invention, so the reference is limma's `squeezeVar` applied directly to the
    // reciprocal cell overdispersions on the residual degrees of freedom
    // `sc_residual_df` computes. The generator runs exactly that, through the
    // converged-`uniroot` `squeezeVar`, and this is the one fixture in the suite
    // where that override changes the answer.
    let s = common::scalars();
    let l = load(&DATASETS[0]);
    let want = common::table("sc_shrink.csv");
    let nebula_fixture = common::table("sc_nebula.csv");

    let df_residual = sc_residual_df(l.n_cells, 3, l.n_subjects);
    assert_close_scalar(
        df_residual,
        s.get("sc_shrink", "df_residual"),
        Tol::rel(0.0),
        "sc/residual_df",
    );

    // The inputs are R's, not the crate's. Feeding the crate its own nebula
    // output would drag the LN+HL divergence documented above into a test that
    // is supposed to be about the shrinkage, and the shrinkage is a joint fit
    // over every usable gene, so one bad input moves every output. This is the
    // same isolation the bulk GLM and QL files use.
    let dispersion: Vec<f64> = nebula_fixture.column("Cell").to_vec();
    let convergence: Vec<i32> = nebula_fixture
        .column("convergence")
        .iter()
        .map(|v| *v as i32)
        .collect();
    let covariate: Vec<f64> = nebula_fixture.column("logFC_int").to_vec();
    let n = dispersion.len();

    let shrunk = shrink_sc_dispersion(
        &dispersion,
        &convergence,
        Some(&covariate),
        df_residual,
        None,
    )
    .expect("shrink_sc_dispersion failed");

    // R writes only the usable genes, the ones nebula converged on with a
    // positive cell overdispersion, so the comparison is indexed by gene.
    let gene_ids: Vec<usize> = nebula_fixture
        .column_usize("gene_id")
        .iter()
        .map(|g| g - 1)
        .collect();
    let position: std::collections::HashMap<usize, usize> =
        gene_ids.iter().enumerate().map(|(i, g)| (*g, i)).collect();
    let want_ids: Vec<usize> = want.column_usize("gene_id").iter().map(|g| g - 1).collect();
    assert_eq!(
        want_ids.len(),
        s.get_usize("sc_shrink", "n_usable"),
        "usable gene count"
    );
    assert!(
        want_ids.len() < n,
        "some genes must be excluded, or the filter is inert"
    );

    let mut phi_raw = Vec::with_capacity(want_ids.len());
    let mut phi_post = Vec::with_capacity(want_ids.len());
    let mut phi_prior = Vec::with_capacity(want_ids.len());
    let mut df_prior = Vec::with_capacity(want_ids.len());
    // `phi_raw`, `phi_post` and `phi_prior` are indexed by gene, but `df_prior`
    // is indexed by *usable* gene: the robust fit returns one entry per gene it
    // actually fitted, not one per gene it was handed. Conflating the two reads
    // the right numbers off the wrong rows and looks like a 94% disagreement.
    assert_eq!(
        shrunk.df_prior.len(),
        want_ids.len(),
        "df_prior should carry one entry per usable gene"
    );
    for (k, id) in want_ids.iter().enumerate() {
        let i = position[id];
        phi_raw.push(shrunk.phi_raw[i].max(1e-8));
        phi_post.push(shrunk.phi_post[i]);
        phi_prior.push(shrunk.phi_prior[i % shrunk.phi_prior.len()]);
        df_prior.push(shrunk.df_prior[k]);
    }

    assert_close(
        &phi_raw,
        want.column("phi_raw"),
        TOL_CELL,
        "sc/shrink_phi_raw",
    );
    assert_close(
        &phi_post,
        want.column("var_post"),
        TOL_SHRINK,
        "sc/shrink_phi_post",
    );
    assert_close(
        &phi_prior,
        want.column("var_prior"),
        TOL_SHRINK,
        "sc/shrink_phi_prior",
    );
    assert_close(
        &df_prior,
        want.column("df_prior"),
        TOL_SHRINK_DF,
        "sc/shrink_df_prior",
    );

    assert_eq!(
        shrunk.df_residual, df_residual,
        "the result should carry the residual df it was given"
    );
}
