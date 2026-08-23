//! End-to-end parity for `voomLmFit` and `squeezeVar`.
//!
//! The crate has no `eBayes` and no limma `topTable`, so the chain stops at the
//! moderated variances rather than at a moderated t table. What is gated is the
//! log-CPM matrix, the precision weights, the fitted linear model and the
//! empirical Bayes squeeze on its residual variances.
//!
//! Both datasets here have non-constant residual degrees of freedom, {4, 9} on
//! the factorial set and {4, 5} on the unbalanced one, because the planted
//! genes with an empty group lose observations to `voomLmFit`'s structural-zero
//! masking. That is the branch where the trend switches from `stats::lowess` to
//! `weightedLowess`, and it would never fire on plain random counts.
//!
//! Tolerances are measured, not guessed:
//! `EDGE_RS_TOL_REPORT=1 cargo test --release --test e2e_voom -- --nocapture`.

mod common;

use common::{Tol, assert_close, assert_close_scalar};

use edge_rs::limma::squeeze_var::{SqueezeVarParams, squeeze_var};
use edge_rs::limma::voom::voom_lmfit;

////////////////
// Tolerances //
////////////////

// Figures are `NEEDS rel` from:
//   EDGE_RS_TOL_REPORT=1 cargo test --release --test e2e_voom -- --nocapture

/// The log-CPM matrix. Arithmetic only: `log2((count + 0.5) / (lib + 1) * 1e6)`.
/// Needs `0` beyond a `1e-13` absolute floor, which covers the entries whose
/// log lands near zero.
const TOL_E: Tol = Tol::new(1e-12, 1e-13);

/// Precision weights, the reciprocal fourth power of a lowess fit read back at
/// the fitted values. Needs `7.8e-15`, which is far better than the smoother's
/// sensitivity would suggest and worth knowing: a one-ULP wobble in a residual
/// can move which points sit inside a neighbourhood, and on this data it does
/// not happen.
const TOL_WEIGHTS: Tol = Tol::rel(1e-13);

/// Coefficients and unscaled standard errors from the weighted least squares.
/// Needs `0` beyond a `1e-12` absolute floor.
const TOL_COEF: Tol = Tol::new(1e-10, 1e-12);

/// Residual standard deviations. Needs `1.6e-14`.
const TOL_SIGMA: Tol = Tol::rel(1e-13);

/// The fitted mean-variance trend. Needs `0` beyond a `1e-12` absolute floor.
const TOL_TREND: Tol = Tol::new(1e-10, 1e-12);

/// Squeezed variances from `squeezeVar`. With non-constant residual degrees of
/// freedom this runs through `fitFDistUnequalDF1`, whose `optimize` call sits at
/// R's default `tol` of `1.2e-4` on a very flat likelihood, and through
/// statmod's `logmdigamma`, which is off in the fourteenth digit and gets
/// exponentiated. `src/limma/squeeze_var.rs` documents both. Needs `5.6e-9`.
const TOL_SQUEEZE: Tol = Tol::rel(2e-8);

/// The prior degrees of freedom `squeezeVar` fits, which is the quantity
/// `optimize` actually searches over and so the loosest thing it produces.
/// Needs `1.4e-8`.
const TOL_SQUEEZE_DF: Tol = Tol::rel(5e-8);

/////////////
// Loading //
/////////////

/// A dataset with a voom fixture.
struct Dataset {
    /// Fixture prefix.
    tag: &'static str,
    /// Number of design columns.
    n_coef: usize,
}

/// The two datasets voom runs on. The near-Poisson set is left out: its counts
/// are too low for a mean-variance trend to mean anything.
const DATASETS: [Dataset; 2] = [
    Dataset {
        tag: "fac",
        n_coef: 3,
    },
    Dataset {
        tag: "unbal",
        n_coef: 4,
    },
];

/// Counts, design and shape for one dataset.
struct Loaded {
    /// Counts of retained genes, row-major.
    counts: Vec<f64>,
    /// Retained gene count.
    n_genes: usize,
    /// Sample count.
    n_samples: usize,
    /// Design, row-major.
    design: Vec<f64>,
    /// Number of design columns.
    n_coef: usize,
}

/// Loads one dataset.
///
/// ### Params
///
/// * `d` - Dataset description
///
/// ### Returns
///
/// The retained counts and the design.
fn load(d: &Dataset) -> Loaded {
    let counts_t = common::table(&format!("{}_kept_counts.csv", d.tag));
    let n_genes = counts_t.n_rows();
    let n_samples = counts_t.n_cols();
    let counts = counts_t.row_major_counts();
    let (design, _, n_coef) = common::matrix(&format!("{}_design.csv", d.tag));
    assert_eq!(n_coef, d.n_coef, "{}: unexpected design width", d.tag);
    Loaded {
        counts,
        n_genes,
        n_samples,
        design,
        n_coef,
    }
}

///////////////
// voomLmFit //
///////////////

#[test]
fn test_voom_lmfit_matches_edger() {
    for d in &DATASETS {
        let l = load(d);
        let want_e = common::matrix(&format!("{}_voom_e.csv", d.tag)).0;
        let want_w = common::matrix(&format!("{}_voom_weights.csv", d.tag)).0;
        let want_fit = common::table(&format!("{}_voom_lmfit.csv", d.tag));
        let want_fitted = common::matrix(&format!("{}_voom_fitted.csv", d.tag)).0;
        let want_trend = common::table(&format!("{}_voom_trend.csv", d.tag));

        let (voom, fit) = voom_lmfit(
            &l.counts,
            l.n_genes,
            l.n_samples,
            &l.design,
            l.n_coef,
            None,
            None,
            None,
        )
        .expect("voom_lmfit failed");

        assert_close(&voom.e, &want_e, TOL_E, &format!("{}/voom_E", d.tag));
        assert_close(
            &voom.weights,
            &want_w,
            TOL_WEIGHTS,
            &format!("{}/voom_weights", d.tag),
        );

        assert_close(
            &voom.trend_x,
            want_trend.column("x"),
            TOL_TREND,
            &format!("{}/voom_trend_x", d.tag),
        );
        assert_close(
            &voom.trend_y,
            want_trend.column("y"),
            TOL_TREND,
            &format!("{}/voom_trend_y", d.tag),
        );

        let mut want_coef = Vec::with_capacity(l.n_genes * l.n_coef);
        let mut want_stdev = Vec::with_capacity(l.n_genes * l.n_coef);
        for g in 0..l.n_genes {
            for c in 0..l.n_coef {
                want_coef.push(want_fit.column(&format!("coef{}", c + 1))[g]);
                want_stdev.push(want_fit.column(&format!("stdev{}", c + 1))[g]);
            }
        }
        assert_close(
            &fit.coefficients,
            &want_coef,
            TOL_COEF,
            &format!("{}/lmfit_coefficients", d.tag),
        );
        assert_close(
            &fit.stdev_unscaled,
            &want_stdev,
            TOL_COEF,
            &format!("{}/lmfit_stdev_unscaled", d.tag),
        );
        assert_close(
            &fit.sigma,
            want_fit.column("sigma"),
            TOL_SIGMA,
            &format!("{}/lmfit_sigma", d.tag),
        );
        assert_close(
            &fit.df_residual,
            want_fit.column("df_residual"),
            Tol::rel(0.0),
            &format!("{}/lmfit_df_residual", d.tag),
        );
        assert_close(
            &fit.fitted,
            &want_fitted,
            TOL_COEF,
            &format!("{}/lmfit_fitted", d.tag),
        );
    }
}

#[test]
fn test_voom_structural_zeros_reduce_the_residual_df() {
    // The planted genes with an entirely empty group lose those observations to
    // voomLmFit's masking, so the residual degrees of freedom are not constant.
    // That is the branch where the trend switches to weightedLowess, and if the
    // dataset ever stopped triggering it this file would quietly become a much
    // weaker test.
    let s = common::scalars();
    for d in &DATASETS {
        let l = load(d);
        let (_, fit) = voom_lmfit(
            &l.counts,
            l.n_genes,
            l.n_samples,
            &l.design,
            l.n_coef,
            None,
            None,
            None,
        )
        .expect("voom_lmfit failed");

        let mut distinct: Vec<f64> = fit.df_residual.clone();
        distinct.sort_by(f64::total_cmp);
        distinct.dedup();
        assert_eq!(
            distinct.len(),
            s.get_usize(&format!("{}_voom", d.tag), "n_distinct_df"),
            "{}: number of distinct residual df",
            d.tag
        );
        assert!(
            distinct.len() > 1,
            "{}: the structural-zero branch is no longer exercised",
            d.tag
        );
    }
}

///////////////
// squeezeVar //
///////////////

#[test]
fn test_squeeze_var_on_voom_variances_matches_limma() {
    let s = common::scalars();

    for d in &DATASETS {
        let l = load(d);
        let want = common::table(&format!("{}_voom_squeeze.csv", d.tag));
        let key = format!("{}_voom", d.tag);

        let (_, fit) = voom_lmfit(
            &l.counts,
            l.n_genes,
            l.n_samples,
            &l.design,
            l.n_coef,
            None,
            None,
            None,
        )
        .expect("voom_lmfit failed");

        let var: Vec<f64> = fit.sigma.iter().map(|s| s * s).collect();

        // Untrended. The residual df vary, so limma auto-resolves legacy to
        // FALSE and this goes through fitFDistUnequalDF1.
        let got = squeeze_var(&var, &fit.df_residual, None, None).expect("squeeze_var failed");
        assert_close(
            &got.var_post,
            want.column("var_post"),
            TOL_SQUEEZE,
            &format!("{}/squeeze_var_post", d.tag),
        );
        assert_close_scalar(
            got.df_prior[0],
            s.get(&key, "df_prior"),
            TOL_SQUEEZE_DF,
            &format!("{}/squeeze_df_prior", d.tag),
        );
        assert_close_scalar(
            got.var_prior[0],
            s.get(&key, "var_prior"),
            TOL_SQUEEZE,
            &format!("{}/squeeze_var_prior", d.tag),
        );

        // Trended against abundance.
        let amean = common::table(&format!("{}_voom_lmfit.csv", d.tag));
        let got = squeeze_var(&var, &fit.df_residual, Some(amean.column("amean")), None)
            .expect("squeeze_var failed");
        assert_close(
            &got.var_post,
            want.column("trend_var_post"),
            TOL_SQUEEZE,
            &format!("{}/squeeze_trend_var_post", d.tag),
        );
        assert_close_scalar(
            got.df_prior[0],
            s.get(&key, "trend_df_prior"),
            TOL_SQUEEZE_DF,
            &format!("{}/squeeze_trend_df_prior", d.tag),
        );

        // Robust. The reference is limma with its uniroot converged past its own
        // default, per UPSTREAM_DEVIATIONS.md entry 12; on this data the branch
        // that uses uniroot is not reached, which the generator records.
        let params = SqueezeVarParams {
            robust: true,
            ..Default::default()
        };
        let got =
            squeeze_var(&var, &fit.df_residual, None, Some(params)).expect("squeeze_var failed");
        assert_close(
            &got.var_post,
            want.column("robust_var_post"),
            TOL_SQUEEZE,
            &format!("{}/squeeze_robust_var_post", d.tag),
        );
        assert_eq!(
            got.df_prior.len(),
            s.get_usize(&key, "robust_df_prior_len"),
            "{}: robust df_prior length",
            d.tag
        );

        // A scalar df forces limma's legacy branch, which is the only route to
        // fitFDistRobustly and the one the converged-uniroot override exists for.
        let df_one = s.get(&key, "legacy_df_one");
        let got = squeeze_var(&var, &[df_one], None, None).expect("squeeze_var failed");
        assert_close(
            &got.var_post,
            want.column("legacy_var_post"),
            TOL_SQUEEZE,
            &format!("{}/squeeze_legacy_var_post", d.tag),
        );
        assert_close_scalar(
            got.df_prior[0],
            s.get(&key, "legacy_df_prior"),
            TOL_SQUEEZE_DF,
            &format!("{}/squeeze_legacy_df_prior", d.tag),
        );
    }
}
