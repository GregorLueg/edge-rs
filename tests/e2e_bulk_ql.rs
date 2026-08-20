//! End-to-end parity for the quasi-likelihood chain: `glmQLFit` then
//! `glmQLFTest`.
//!
//! This is the gate the original port plan named and never built: match edgeR's
//! `s2.post`, `F` and `PValue` columns on a real dataset.
//!
//! The abundances the fit keys off come from the fixture rather than from a
//! fresh `estimate_disp` run, so a failure here is the quasi-likelihood stage
//! and not the dispersion estimation feeding it. `estimate_disp` has its own
//! gate in `e2e_bulk_classic.rs`.
//!
//! Both pipelines are covered. The current one carries `df.residual.adj`,
//! `deviance.adj` and `average.ql.dispersion`; the legacy one carries
//! `df.residual.zeros` instead, and is the only one where the Poisson bound does
//! anything. That last point is why the near-Poisson dataset exists: on it the
//! bound moves 83 of 226 p-values under the legacy pipeline and exactly none
//! under the current one, matching edgeR, which quietly forces the flag off when
//! `df.residual.zeros` is absent.
//!
//! Tolerances are measured, not guessed:
//! `EDGE_RS_TOL_REPORT=1 cargo test --release --test e2e_bulk_ql -- --nocapture`.

mod common;

use common::{Tol, assert_close, assert_close_scalar};

use edge_rs::glm::fit::GlmFit;
use edge_rs::glm::ql_fit::{QlFit, QlFitParams, glm_ql_fit};
use edge_rs::glm::test::{GlmTestInput, QlSummary, Tested, glm_lrt, glm_ql_ftest};
use edge_rs::prelude::Recycled;

////////////////
// Tolerances //
////////////////

/// Coefficients from the quasi-likelihood fit. Same story as the plain GLM: the
/// stopping rule is on the deviance, so the coefficients are only settled to
/// around `1e-4`. Measured worst case `6.0e-4` absolute.
const TOL_COEF: Tol = Tol::new(1e-2, 2e-3);

/// Largest coefficient magnitude, on the natural log scale, that counts as
/// identified. A gene with an entirely empty group sits above this and its
/// estimate is arbitrary in both implementations.
const IDENTIFIED_COEF_LIMIT: f64 = 20.0;

/// Residual deviance. Measured worst case `4.0e-7`.
const TOL_DEVIANCE: Tol = Tol::new(1e-5, 1e-9);

/// Posterior and prior quasi-likelihood dispersions, and the adjusted residual
/// degrees of freedom. These run the deviance through the Chebyshev
/// approximation in `src/ql/chebyshev.rs` and then through `squeezeVar`.
/// Measured worst case `1.5e-5`.
const TOL_S2: Tol = Tol::new(1e-4, 1e-9);

/// Prior degrees of freedom from the empirical Bayes fit. Measured worst case
/// `6.1e-5`.
const TOL_DF_PRIOR: Tol = Tol::new(1e-3, 1e-9);

/// F statistics. Measured worst case `4.2e-5` relative, `4.5e-6` absolute.
///
/// The absolute floor covers genes whose F is around `1e-4`, which is a p-value
/// of one whichever implementation computed it.
const TOL_F: Tol = Tol::new(1e-3, 1e-4);

/// The top-abundance proportion used when the negative binomial dispersion is
/// estimated rather than supplied. It comes off its own lowess span, so it is
/// coarser than the quantities it feeds. Measured worst case `2.4e-4`.
const TOL_TOP_PROPORTION: Tol = Tol::rel(1e-3);

/// P-values on the natural scale. Measured worst case `1.1e-4`.
const TOL_P_VALUE: Tol = Tol::new(1e-3, 1e-300);

/// P-values as `log(p)`, which is where the small ones live. The absolute floor
/// covers the near-one end, where `log p` approaches zero and the relative test
/// stops meaning anything. Measured worst case `2.6e-4` absolute.
const TOL_LOG_P: Tol = Tol::new(1e-3, 1e-3);

/////////////
// Loading //
/////////////

/// One dataset's fixture prefix and shape.
struct Dataset {
    /// Fixture prefix.
    tag: &'static str,
    /// Number of design columns.
    n_coef: usize,
}

/// The three bulk datasets.
const DATASETS: [Dataset; 3] = [
    Dataset { tag: "fac", n_coef: 3 },
    Dataset { tag: "unbal", n_coef: 4 },
    Dataset { tag: "pois", n_coef: 2 },
];

/// The retained counts, design and offsets for one dataset.
struct Loaded {
    /// Counts of retained genes, row-major.
    counts: Vec<f64>,
    /// Retained gene count.
    n_genes: usize,
    /// Sample count.
    n_samples: usize,
    /// Design, row-major `n_samples * n_coef`.
    design: Vec<f64>,
    /// Number of design columns.
    n_coef: usize,
    /// Log offsets, one per sample.
    offset: Recycled<f64>,
    /// Average log-CPM per gene, as `estimateDisp` reported it.
    ave_log_cpm: Vec<f64>,
    /// Trended dispersion, which the legacy pipeline has to be handed.
    trended: Vec<f64>,
}

/// Borrows the GLM half of a quasi-likelihood fit.
///
/// `glm_ql_ftest` wants a [`GlmFit`], and the right one is the fit
/// `glm_ql_fit` already performed, not a fresh one. Refitting at
/// `QlFit::dispersion` would be wrong: that field holds the *undivided*
/// dispersion, while the fit itself ran at `dispersion / average_ql_dispersion`.
/// edgeR has no equivalent trap because its `DGEGLM` is one object carrying
/// both halves.
///
/// ### Params
///
/// * `fit` - The quasi-likelihood fit
///
/// ### Returns
///
/// The same coefficients, fitted means and deviances, as a [`GlmFit`].
fn as_glm_fit(fit: &QlFit) -> GlmFit {
    GlmFit {
        coefficients: fit.coefficients.clone(),
        unshrunk_coefficients: fit.unshrunk_coefficients.clone(),
        fitted: fit.fitted.clone(),
        deviance: fit.deviance.clone(),
        df_residual: fit.df_residual,
        method: fit.method,
    }
}

/// Loads one dataset.
///
/// ### Params
///
/// * `d` - Dataset description
///
/// ### Returns
///
/// Everything the quasi-likelihood fit needs, taken from the fixtures.
fn load(d: &Dataset) -> Loaded {
    let counts_t = common::table(&format!("{}_kept_counts.csv", d.tag));
    let n_genes = counts_t.n_rows();
    let n_samples = counts_t.n_cols();
    let counts = counts_t.row_major_counts();

    let (design, _, n_coef) = common::matrix(&format!("{}_design.csv", d.tag));
    assert_eq!(n_coef, d.n_coef, "{}: unexpected design width", d.tag);

    let ks = common::table(&format!("{}_kept_samples.csv", d.tag));
    let disp = common::table(&format!("{}_disp.csv", d.tag));

    Loaded {
        counts,
        n_genes,
        n_samples,
        design,
        n_coef,
        offset: Recycled::by_sample(ks.column("offset").to_vec()),
        ave_log_cpm: disp.column("ave_log_cpm").to_vec(),
        trended: disp.column("trended").to_vec(),
    }
}

////////////////////////
// The current pipeline //
////////////////////////

#[test]
fn test_glm_ql_fit_matches_edger() {
    let s = common::scalars();

    for d in &DATASETS {
        let l = load(d);
        let want = common::table(&format!("{}_ql.csv", d.tag));

        // `dispersion = None` is the path that estimates the negative binomial
        // dispersion from the most abundant genes and reports `top_proportion`.
        // The R fixture uses glmQLFit.default for the same reason: the DGEList
        // method precomputes its own scalar from a hardcoded top decile instead.
        let fit = glm_ql_fit(
            &l.counts,
            l.n_genes,
            l.n_samples,
            &l.design,
            l.n_coef,
            None,
            &l.offset,
            None,
            &l.ave_log_cpm,
            None,
        )
        .expect("glm_ql_fit failed");

        assert_close_scalar(
            fit.average_ql_dispersion
                .expect("the current pipeline reports an average quasi-dispersion"),
            s.get(d.tag, "ql_average_ql_dispersion"),
            TOL_S2,
            &format!("{}/average_ql_dispersion", d.tag),
        );
        assert_close_scalar(
            fit.top_proportion.expect("a dispersion was estimated, so this is reported"),
            s.get(d.tag, "ql_top_proportion"),
            TOL_TOP_PROPORTION,
            &format!("{}/top_proportion", d.tag),
        );
        assert_close_scalar(
            fit.dispersion.get(0, 0, l.n_samples),
            s.get(d.tag, "ql_dispersion"),
            TOL_S2,
            &format!("{}/resolved_dispersion", d.tag),
        );

        // Coefficients, over the genes where they are identified. A gene with an
        // entirely empty group has an infinite estimate and both implementations
        // stop somewhere arbitrary; see the note in e2e_bulk_glm.rs.
        let mut got_coef = Vec::new();
        let mut want_coef = Vec::new();
        for g in 0..l.n_genes {
            let cols: Vec<f64> = (0..l.n_coef)
                .map(|c| want.column(&format!("coef{}", c + 1))[g])
                .collect();
            if cols.iter().all(|v| v.is_finite() && v.abs() < IDENTIFIED_COEF_LIMIT) {
                for (c, w) in cols.iter().enumerate() {
                    got_coef.push(fit.coefficients[g * l.n_coef + c]);
                    want_coef.push(*w);
                }
            }
        }
        assert_close(&got_coef, &want_coef, TOL_COEF, &format!("{}/ql_coefficients", d.tag));

        assert_close(
            &fit.s2_post,
            want.column("s2_post"),
            TOL_S2,
            &format!("{}/s2_post", d.tag),
        );
        assert_close(
            &fit.deviance,
            want.column("deviance"),
            TOL_DEVIANCE,
            &format!("{}/ql_deviance", d.tag),
        );

        // s2.prior and df.prior are length one or length n_genes depending on
        // whether the fit was trended or robust, so both are compared recycled.
        let n = l.n_genes;
        let s2_prior: Vec<f64> = (0..n).map(|g| fit.s2_prior[g % fit.s2_prior.len()]).collect();
        assert_close(
            &s2_prior,
            want.column("s2_prior"),
            TOL_S2,
            &format!("{}/s2_prior", d.tag),
        );
        assert_eq!(
            fit.s2_prior.len(),
            s.get_usize(d.tag, "ql_s2_prior_len"),
            "{}: s2_prior length",
            d.tag
        );

        let df_prior: Vec<f64> = (0..n).map(|g| fit.df_prior[g % fit.df_prior.len()]).collect();
        assert_close(
            &df_prior,
            want.column("df_prior"),
            TOL_DF_PRIOR,
            &format!("{}/df_prior", d.tag),
        );
        assert_eq!(
            fit.df_prior.len(),
            s.get_usize(d.tag, "ql_df_prior_len"),
            "{}: df_prior length",
            d.tag
        );

        let adj = fit
            .df_residual_adj
            .as_ref()
            .expect("the current pipeline reports adjusted residual degrees of freedom");
        assert_close(
            adj,
            want.column("df_residual_adj"),
            TOL_S2,
            &format!("{}/df_residual_adj", d.tag),
        );
        let dev_adj = fit
            .deviance_adj
            .as_ref()
            .expect("the current pipeline reports an adjusted deviance");
        assert_close(
            dev_adj,
            want.column("deviance_adj"),
            TOL_S2,
            &format!("{}/deviance_adj", d.tag),
        );

        assert!(
            fit.df_residual_zeros.is_none(),
            "{}: df_residual_zeros marks the legacy pipeline and must be absent here",
            d.tag
        );
    }
}

#[test]
fn test_glm_ql_ftest_matches_edger() {
    let s = common::scalars();

    for d in &DATASETS {
        let l = load(d);
        let want = common::table(&format!("{}_ql.csv", d.tag));

        let fit = glm_ql_fit(
            &l.counts,
            l.n_genes,
            l.n_samples,
            &l.design,
            l.n_coef,
            None,
            &l.offset,
            None,
            &l.ave_log_cpm,
            None,
        )
        .expect("glm_ql_fit failed");

        // The summary has to be assembled by hand; there is no conversion, and
        // `df_residual_zeros: None` is what selects the current pipeline and
        // switches the Poisson bound off, exactly as edgeR does.
        let base = as_glm_fit(&fit);

        let ql = QlSummary {
            s2_post: &fit.s2_post,
            df_prior: &fit.df_prior,
            df_residual_adj: fit.df_residual_adj.as_ref().expect("current pipeline"),
            df_residual_zeros: None,
            fitted: &fit.fitted,
            average_ql_dispersion: fit.average_ql_dispersion,
        };

        let input = GlmTestInput {
            counts: &l.counts,
            n_genes: l.n_genes,
            n_samples: l.n_samples,
            design: &l.design,
            n_coef: l.n_coef,
            dispersion: &fit.dispersion,
            offset: &l.offset,
            weights: None,
            log_cpm: Some(&l.ave_log_cpm),
        };

        let got = glm_ql_ftest(&input, &base, &ql, &Tested::Coef(vec![1]), false)
            .expect("glm_ql_ftest failed");

        assert_close(&got.statistic, want.column("F"), TOL_F, &format!("{}/qlf_F", d.tag));
        assert_close(
            &got.p_value,
            want.column("PValue"),
            TOL_P_VALUE,
            &format!("{}/qlf_PValue", d.tag),
        );

        let got_log: Vec<f64> = got.p_value.iter().map(|p| p.ln()).collect();
        assert_close(
            &got_log,
            want.column("logPValue"),
            TOL_LOG_P,
            &format!("{}/qlf_logPValue", d.tag),
        );

        let df_total = got.df_total.as_ref().expect("the F test reports denominator df");
        assert_close(
            df_total,
            want.column("df_total"),
            TOL_DF_PRIOR,
            &format!("{}/qlf_df_total", d.tag),
        );
        assert_close_scalar(
            got.df_test,
            s.get(d.tag, "qlf_df_test"),
            Tol::rel(0.0),
            &format!("{}/qlf_df_test", d.tag),
        );
    }
}

/////////////////////////
// The legacy pipeline //
/////////////////////////

#[test]
fn test_glm_ql_fit_legacy_matches_edger() {
    // The legacy pipeline is the only one that reports `df.residual.zeros`, and
    // it errors in R if no dispersion is supplied, so the trended dispersion is
    // handed to it explicitly.
    let l = load(&DATASETS[2]);
    let want = common::table("pois_ql_legacy.csv");
    let s = common::scalars();

    let dispersion = Recycled::by_gene(l.trended.clone());
    let params = QlFitParams { legacy: true, ..Default::default() };
    let fit = glm_ql_fit(
        &l.counts,
        l.n_genes,
        l.n_samples,
        &l.design,
        l.n_coef,
        Some(&dispersion),
        &l.offset,
        None,
        &l.ave_log_cpm,
        Some(params),
    )
    .expect("glm_ql_fit failed");

    assert!(
        fit.average_ql_dispersion.is_none(),
        "the legacy pipeline has no average quasi-dispersion"
    );
    assert!(fit.df_residual_adj.is_none(), "the legacy pipeline has no adjusted df");

    let zeros = fit
        .df_residual_zeros
        .as_ref()
        .expect("the legacy pipeline reports residual df after dropping structural zeros");
    assert_close(
        zeros,
        want.column("df_residual_zeros"),
        TOL_S2,
        "pois/legacy_df_residual_zeros",
    );
    assert_close(&fit.s2_post, want.column("s2_post"), TOL_S2, "pois/legacy_s2_post");
    assert_close(
        &fit.deviance,
        want.column("deviance"),
        TOL_DEVIANCE,
        "pois/legacy_deviance",
    );

    let n = l.n_genes;
    let df_prior: Vec<f64> = (0..n).map(|g| fit.df_prior[g % fit.df_prior.len()]).collect();
    assert_close(
        &df_prior,
        want.column("df_prior"),
        TOL_DF_PRIOR,
        "pois/legacy_df_prior",
    );
    assert_eq!(
        fit.df_prior.len(),
        s.get_usize("pois_legacy", "df_prior_len"),
        "legacy df_prior length"
    );
}

#[test]
fn test_poisson_bound_moves_p_values_only_on_the_legacy_pipeline() {
    // edgeR forces `poisson.bound` off when `df.residual.zeros` is absent, so
    // the flag is live on the legacy fit and dead on the current one. The
    // near-Poisson dataset is the one where the bound actually bites: it moves
    // 83 of 226 p-values there.
    let l = load(&DATASETS[2]);
    let want = common::table("pois_ql_legacy.csv");
    let s = common::scalars();

    let dispersion = Recycled::by_gene(l.trended.clone());
    let params = QlFitParams { legacy: true, ..Default::default() };
    let fit = glm_ql_fit(
        &l.counts,
        l.n_genes,
        l.n_samples,
        &l.design,
        l.n_coef,
        Some(&dispersion),
        &l.offset,
        None,
        &l.ave_log_cpm,
        Some(params),
    )
    .expect("glm_ql_fit failed");

    let base = as_glm_fit(&fit);

    let zeros = fit.df_residual_zeros.clone().expect("legacy pipeline");
    let ql = QlSummary {
        s2_post: &fit.s2_post,
        df_prior: &fit.df_prior,
        df_residual_adj: &zeros,
        df_residual_zeros: Some(&zeros),
        fitted: &fit.fitted,
        average_ql_dispersion: None,
    };
    let input = GlmTestInput {
        counts: &l.counts,
        n_genes: l.n_genes,
        n_samples: l.n_samples,
        design: &l.design,
        n_coef: l.n_coef,
        dispersion: &fit.dispersion,
        offset: &l.offset,
        weights: None,
        log_cpm: Some(&l.ave_log_cpm),
    };

    let bounded = glm_ql_ftest(&input, &base, &ql, &Tested::Coef(vec![1]), true)
        .expect("glm_ql_ftest failed");
    let unbounded = glm_ql_ftest(&input, &base, &ql, &Tested::Coef(vec![1]), false)
        .expect("glm_ql_ftest failed");

    assert_close(
        &bounded.p_value,
        want.column("PValue_bound"),
        TOL_P_VALUE,
        "pois/legacy_PValue_bound",
    );
    assert_close(
        &unbounded.p_value,
        want.column("PValue_nobound"),
        TOL_P_VALUE,
        "pois/legacy_PValue_nobound",
    );
    assert_close(
        &bounded.statistic,
        want.column("F_bound"),
        TOL_F,
        "pois/legacy_F_bound",
    );

    // The bound has to actually do something, or this test is vacuous.
    let moved = (0..l.n_genes)
        .filter(|&g| bounded.p_value[g] != unbounded.p_value[g])
        .count();
    let expected = s.get_usize("pois_legacy", "n_bound_differs");
    assert!(
        moved > 0,
        "the Poisson bound changed nothing, so this fixture no longer tests it"
    );
    assert_eq!(
        moved, expected,
        "the bound moved {moved} p-values where edgeR moved {expected}"
    );
}

/////////////////////////////
// The seam the plan named //
/////////////////////////////

#[test]
fn test_lrt_and_ftest_disagree_as_they_should() {
    // A quasi-likelihood fit stores the undivided dispersion but fits with
    // `dispersion / average.ql.dispersion`, so a likelihood ratio test on that
    // fit has to divide again before refitting the null. Getting that wrong is
    // silent: the p-values stay plausible and are simply on the wrong scale.
    // Passing the summary through is what makes the two agree with edgeR.
    let l = load(&DATASETS[0]);

    let fit = glm_ql_fit(
        &l.counts,
        l.n_genes,
        l.n_samples,
        &l.design,
        l.n_coef,
        None,
        &l.offset,
        None,
        &l.ave_log_cpm,
        None,
    )
    .expect("glm_ql_fit failed");

    let base = as_glm_fit(&fit);

    let input = GlmTestInput {
        counts: &l.counts,
        n_genes: l.n_genes,
        n_samples: l.n_samples,
        design: &l.design,
        n_coef: l.n_coef,
        dispersion: &fit.dispersion,
        offset: &l.offset,
        weights: None,
        log_cpm: Some(&l.ave_log_cpm),
    };
    let ql = QlSummary {
        s2_post: &fit.s2_post,
        df_prior: &fit.df_prior,
        df_residual_adj: fit.df_residual_adj.as_ref().expect("current pipeline"),
        df_residual_zeros: None,
        fitted: &fit.fitted,
        average_ql_dispersion: fit.average_ql_dispersion,
    };

    let with = glm_lrt(&input, &base, &Tested::Coef(vec![1]), Some(&ql))
        .expect("glm_lrt failed");
    let without = glm_lrt(&input, &base, &Tested::Coef(vec![1]), None)
        .expect("glm_lrt failed");

    let differing = (0..l.n_genes)
        .filter(|&g| with.statistic[g] != without.statistic[g])
        .count();
    assert!(
        differing > l.n_genes / 2,
        "passing the quasi-likelihood summary should rescale the null refit, but only \
         {differing} of {} genes moved",
        l.n_genes
    );
}
