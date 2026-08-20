//! End-to-end parity for the GLM chain: `glmFit`, `glmLRT`, `glmTreat`.
//!
//! Runs over the same three datasets as `e2e_bulk_classic.rs`, so the
//! likelihood ratio test is exercised on a balanced factorial, on an unbalanced
//! design with a continuous covariate and five residual degrees of freedom, and
//! on near-Poisson counts where the fit dispatches one-way instead of Levenberg.
//!
//! Tolerances are measured, not guessed:
//! `EDGE_RS_TOL_REPORT=1 cargo test --release --test e2e_bulk_glm -- --nocapture`.

mod common;

use common::{Tol, assert_close};

use edge_rs::glm::fit::glm_fit;
use edge_rs::glm::test::{GlmTestInput, Tested, TreatNull, glm_lrt, glm_treat};
use edge_rs::prelude::Recycled;

////////////////
// Tolerances //
////////////////

// The coefficient tolerances below are set by edgeR's own stopping rule, not by
// anything this crate does. `mglmLevenberg` stops when the deviance improves by
// less than `tol * (|deviance| + 0.1)`, and near the optimum the deviance is
// quadratic in the coefficients, so a deviance settled to 1e-6 leaves the
// coefficients loose at around 1e-4.
//
// Measured on the factorial set, over the 869 genes with no zero count, running
// edgeR against itself at `tol = 1e-6` and `tol = 1e-14`:
//
//   coefficient movement   1.2e-4 absolute, 5.1e-3 relative
//   deviance movement      2.9e-7 absolute
//
// So the deviance is well determined and the coefficients are not. This crate
// sits 5.6e-4 from edgeR on the coefficients, the same order as edgeR's distance
// from itself. Gating coefficients any tighter would be pinning a test to the
// convergence tolerance of the reference rather than to its answer.

/// Fitted coefficients, shrunk and unshrunk. Measured worst case `5.6e-4`
/// absolute against edgeR, on coefficients of order 1.
const TOL_COEF: Tol = Tol::new(1e-2, 2e-3);

/// Fitted means, which inherit the coefficient slack through `exp(X beta)`.
/// Measured worst case `2.3e-3` relative, on the unbalanced set.
///
/// The absolute floor covers the empty-group genes, whose fitted counts for the
/// empty side sit around `1e-7` and differ between implementations only because
/// the runaway coefficient stopped in a different place. No real fitted count is
/// anywhere near that small.
const TOL_FITTED: Tol = Tol::new(1e-2, 1e-4);

/// Residual deviance, which unlike the coefficients is well determined.
/// Measured worst case `4.0e-7`, against edgeR's own `2.9e-7` self-movement.
const TOL_DEVIANCE: Tol = Tol::new(2e-6, 1e-9);

/// Log fold changes, which are just a coefficient over `ln 2` and inherit
/// [`TOL_COEF`]. Measured worst case `8.1e-4` absolute.
const TOL_LOG_FC: Tol = Tol::new(1e-2, 3e-3);

/// Likelihood ratio and F statistics. These are deviance differences, so they
/// are far better determined than the coefficients. Measured worst case
/// `3.7e-5` relative, `4.5e-6` absolute. The absolute floor covers genes whose statistic is around
/// `1e-8`, where the relative test says nothing and any chi-squared that small
/// is a p-value of one either way.
const TOL_STATISTIC: Tol = Tol::new(1e-5, 1e-5);

/// P-values on the natural scale. Measured worst case `1.4e-6`.
///
/// The absolute floor stops a p-value that underflowed to zero in one
/// implementation and not the other from failing a relative test. The log-scale
/// check below is the one with teeth down there.
const TOL_P_VALUE: Tol = Tol::new(1e-4, 1e-300);

/// P-values compared as `log(p)`, which is where the very small ones live.
/// Measured worst case `2.5e-3` relative, `3.1e-5` absolute, both at the
/// near-one end.
///
/// The absolute floor matters at the *other* end from the one this check is for:
/// a gene with `p` near one has `log p` near zero, where the relative test is as
/// meaningless as the natural-scale test is for a `p` near zero.
const TOL_LOG_P: Tol = Tol::new(1e-5, 1e-4);

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
    /// Tagwise dispersion the R fit used, one per gene.
    dispersion: Recycled<f64>,
    /// Average log-CPM per gene, as `estimateDisp` reported it.
    ave_log_cpm: Vec<f64>,
}

/// Loads one dataset.
///
/// ### Params
///
/// * `d` - Dataset description
///
/// ### Returns
///
/// Everything `glm_fit` needs, taken from the fixtures rather than recomputed,
/// so that this file tests the GLM rather than the dispersion estimation.
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
        dispersion: Recycled::by_gene(disp.column("tagwise").to_vec()),
        ave_log_cpm: disp.column("ave_log_cpm").to_vec(),
    }
}

/// Largest coefficient magnitude, on the natural log scale, that counts as
/// identified.
///
/// A gene with an entirely empty group has a coefficient whose maximum
/// likelihood estimate is infinite. Both implementations stop somewhere
/// arbitrary on a flat ridge, and edgeR moves by fourteen log units itself when
/// its tolerance changes, so those genes carry no information about parity.
/// Nothing real sits above this: a coefficient of 20 is a fold change of 5e8.
const IDENTIFIED_COEF_LIMIT: f64 = 20.0;

/// Genes whose coefficients are identified, judged from edgeR's own unshrunk fit.
///
/// The deviance stays comparable on every gene and is checked everywhere; only
/// the coefficients and the fold changes are restricted to this mask.
///
/// ### Params
///
/// * `want` - The `_glmfit.csv` fixture for this dataset
/// * `n_genes` - Number of genes
/// * `n_coef` - Number of coefficients
///
/// ### Returns
///
/// A mask, true where every coefficient of the gene is finite and moderate.
fn identified(want: &common::Table, n_genes: usize, n_coef: usize) -> Vec<bool> {
    (0..n_genes)
        .map(|g| {
            (0..n_coef).all(|c| {
                let v = want.column(&format!("unshrunk{}", c + 1))[g];
                v.is_finite() && v.abs() < IDENTIFIED_COEF_LIMIT
            })
        })
        .collect()
}

/// Keeps the rows of `v` where `mask` is true.
///
/// The mask is per gene, not per element, so `stride` says how many values each
/// gene occupies: one for a per-gene vector, `n_coef` for a coefficient matrix.
/// Passing the wrong stride silently compares the wrong genes, so it is always
/// given explicitly rather than inferred.
///
/// ### Params
///
/// * `v` - Values, row-major, `mask.len() * stride` long
/// * `mask` - One flag per gene
/// * `stride` - Values per gene
///
/// ### Returns
///
/// The retained rows, flattened, in the original order.
fn masked(v: &[f64], mask: &[bool], stride: usize) -> Vec<f64> {
    v.chunks_exact(stride)
        .zip(mask)
        .filter(|(_, keep)| **keep)
        .flat_map(|(row, _)| row.iter().copied())
        .collect()
}

//////////////
// glmFit   //
//////////////

#[test]
fn test_glm_fit_matches_edger() {
    for d in &DATASETS {
        let l = load(d);
        let want = common::table(&format!("{}_glmfit.csv", d.tag));
        let want_fitted = common::matrix(&format!("{}_glmfitted.csv", d.tag)).0;

        let fit = glm_fit(
            &l.counts,
            l.n_genes,
            l.n_samples,
            &l.design,
            l.n_coef,
            &l.dispersion,
            &l.offset,
            None,
            0.125,
        )
        .expect("glm_fit failed");

        // The deviance is identified for every gene, empty group or not.
        assert_close(
            &fit.deviance,
            want.column("deviance"),
            TOL_DEVIANCE,
            &format!("{}/deviance", d.tag),
        );
        assert_close(&fit.fitted, &want_fitted, TOL_FITTED, &format!("{}/fitted", d.tag));

        let mask = identified(&common::table(&format!("{}_glmfit.csv", d.tag)), l.n_genes, l.n_coef);
        let mut want_coef = Vec::with_capacity(l.n_genes * l.n_coef);
        let mut want_unshrunk = Vec::with_capacity(l.n_genes * l.n_coef);
        for g in 0..l.n_genes {
            for c in 0..l.n_coef {
                want_coef.push(want.column(&format!("coef{}", c + 1))[g]);
                want_unshrunk.push(want.column(&format!("unshrunk{}", c + 1))[g]);
            }
        }

        assert_close(
            &masked(&fit.coefficients, &mask, l.n_coef),
            &masked(&want_coef, &mask, l.n_coef),
            TOL_COEF,
            &format!("{}/coefficients", d.tag),
        );

        let unshrunk = fit
            .unshrunk_coefficients
            .as_ref()
            .expect("prior_count > 0 should leave the unshrunk coefficients behind");
        assert_close(
            &masked(unshrunk, &mask, l.n_coef),
            &masked(&want_unshrunk, &mask, l.n_coef),
            TOL_COEF,
            &format!("{}/unshrunk_coefficients", d.tag),
        );
    }
}

#[test]
fn test_glm_fit_without_a_prior_count_drops_the_unshrunk_copy() {
    // edgeR returns unshrunk.coefficients only when it actually shrank
    // something, and the crate mirrors that with an Option.
    let l = load(&DATASETS[0]);
    let fit = glm_fit(
        &l.counts,
        l.n_genes,
        l.n_samples,
        &l.design,
        l.n_coef,
        &l.dispersion,
        &l.offset,
        None,
        0.0,
    )
    .expect("glm_fit failed");
    assert!(
        fit.unshrunk_coefficients.is_none(),
        "no shrinkage was applied, so there is nothing to keep a second copy of"
    );
}

//////////////
// glmLRT   //
//////////////

/// Builds the test input for one dataset.
///
/// The dispersion and offsets are the ones the R fit used, taken from the
/// fixtures, so that a failure here is the test rather than the dispersion
/// estimation upstream of it.
///
/// ### Params
///
/// * `l` - The loaded dataset
/// * `log_cpm` - Average log-CPM per gene, passed through to the result table
///
/// ### Returns
///
/// An input borrowing from `l`, so it cannot outlive it.
fn input<'a>(l: &'a Loaded, log_cpm: &'a [f64]) -> GlmTestInput<'a, f64> {
    GlmTestInput {
        counts: &l.counts,
        n_genes: l.n_genes,
        n_samples: l.n_samples,
        design: &l.design,
        n_coef: l.n_coef,
        dispersion: &l.dispersion,
        offset: &l.offset,
        weights: None,
        log_cpm: Some(log_cpm),
    }
}

#[test]
fn test_glm_lrt_matches_edger() {
    for d in &DATASETS {
        let l = load(d);
        let want = common::table(&format!("{}_lrt.csv", d.tag));

        let fit = glm_fit(
            &l.counts,
            l.n_genes,
            l.n_samples,
            &l.design,
            l.n_coef,
            &l.dispersion,
            &l.offset,
            None,
            0.125,
        )
        .expect("glm_fit failed");

        let inp = input(&l, &l.ave_log_cpm);
        let got = glm_lrt(&inp, &fit, &Tested::Coef(vec![1]), None).expect("glm_lrt failed");

        let mask = identified(&common::table(&format!("{}_glmfit.csv", d.tag)), l.n_genes, l.n_coef);
        assert_close(
            &masked(&got.log_fc, &mask, 1),
            &masked(want.column("logFC"), &mask, 1),
            TOL_LOG_FC,
            &format!("{}/lrt_logFC", d.tag),
        );
        assert_close(
            &got.statistic,
            want.column("LR"),
            TOL_STATISTIC,
            &format!("{}/lrt_LR", d.tag),
        );
        assert_close(
            &got.p_value,
            want.column("PValue"),
            TOL_P_VALUE,
            &format!("{}/lrt_PValue", d.tag),
        );

        // The log scale is where the small p-values are actually checked. R
        // writes these through pchisq(log.p = TRUE), which does not underflow.
        let got_log: Vec<f64> = got.p_value.iter().map(|p| p.ln()).collect();
        assert_close(
            &got_log,
            want.column("logPValue"),
            TOL_LOG_P,
            &format!("{}/lrt_logPValue", d.tag),
        );

        assert_eq!(got.df_test, 1.0, "{}: one coefficient dropped", d.tag);
        assert!(got.df_total.is_none(), "{}: glmLRT reports no df.total", d.tag);
    }
}

#[test]
fn test_glm_lrt_on_several_coefficients_matches_edger() {
    // Dropping two columns at once. edgeR reports the first tested column's
    // fold change and a chi-squared on two degrees of freedom.
    let l = load(&DATASETS[0]);
    let want = common::table("fac_lrt_contrast.csv");
    let s = common::scalars();

    let fit = glm_fit(
        &l.counts,
        l.n_genes,
        l.n_samples,
        &l.design,
        l.n_coef,
        &l.dispersion,
        &l.offset,
        None,
        0.125,
    )
    .expect("glm_fit failed");

    let inp = input(&l, &l.ave_log_cpm);
    let got = glm_lrt(&inp, &fit, &Tested::Coef(vec![1, 2]), None).expect("glm_lrt failed");

    assert_eq!(got.df_test, s.get("fac_glm", "lrt_multi_df_test"));
    let mask = identified(&common::table("fac_glmfit.csv"), l.n_genes, l.n_coef);
    assert_close(
        &masked(&got.log_fc, &mask, 1),
        &masked(want.column("multi_logFC"), &mask, 1),
        TOL_LOG_FC,
        "fac/lrt_multi_logFC",
    );
    assert_close(&got.statistic, want.column("multi_LR"), TOL_STATISTIC, "fac/lrt_multi_LR");
    assert_close(&got.p_value, want.column("multi_PValue"), TOL_P_VALUE, "fac/lrt_multi_PValue");
}

#[test]
fn test_glm_lrt_with_a_contrast_matches_edger() {
    // Contrasts arrive column-major, `n_coef * n_contrasts`. The fixture holds
    // the transpose of R's makeContrasts matrix, one contrast per row, so
    // reading a row row-major gives exactly that layout for a single contrast.
    let l = load(&DATASETS[0]);
    let con = common::table("fac_contrasts.csv");
    let want = common::table("fac_lrt_contrast.csv");

    let fit = glm_fit(
        &l.counts,
        l.n_genes,
        l.n_samples,
        &l.design,
        l.n_coef,
        &l.dispersion,
        &l.offset,
        None,
        0.125,
    )
    .expect("glm_fit failed");
    let inp = input(&l, &l.ave_log_cpm);
    let mask = identified(&common::table("fac_glmfit.csv"), l.n_genes, l.n_coef);

    for (row, prefix) in [(0usize, "con1"), (1, "con2")] {
        let values: Vec<f64> = (0..l.n_coef)
            .map(|c| con.column(con_name(c))[row])
            .collect();
        let got = glm_lrt(
            &inp,
            &fit,
            &Tested::Contrast { values, n_contrasts: 1 },
            None,
        )
        .expect("glm_lrt failed");

        assert_close(
            &masked(&got.log_fc, &mask, 1),
            &masked(want.column(&format!("{prefix}_logFC")), &mask, 1),
            TOL_LOG_FC,
            &format!("fac/lrt_{prefix}_logFC"),
        );
        assert_close(
            &got.statistic,
            want.column(&format!("{prefix}_LR")),
            TOL_STATISTIC,
            &format!("fac/lrt_{prefix}_LR"),
        );
        assert_close(
            &got.p_value,
            want.column(&format!("{prefix}_PValue")),
            TOL_P_VALUE,
            &format!("fac/lrt_{prefix}_PValue"),
        );
    }
}

/// Column name of the factorial design's `c`-th coefficient.
///
/// The contrast fixture is headed with the design's column names, so reading a
/// contrast back needs the mapping from position to name.
///
/// ### Params
///
/// * `c` - Zero-based coefficient index
///
/// ### Returns
///
/// The header the generator wrote for that column.
///
/// ### Panics
///
/// If `c` is past the third column, since only the factorial design is read here.
fn con_name(c: usize) -> &'static str {
    match c {
        0 => "Int",
        1 => "grpB",
        2 => "batb2",
        _ => panic!("the factorial design has three columns"),
    }
}

//////////////
// glmTreat //
//////////////

#[test]
fn test_glm_treat_matches_edger() {
    // edgeR's glmTreat reports no statistic column at all, so only the fold
    // change and the p-value are gated.
    let l = load(&DATASETS[0]);
    let want = common::table("fac_treat.csv");

    let fit = glm_fit(
        &l.counts,
        l.n_genes,
        l.n_samples,
        &l.design,
        l.n_coef,
        &l.dispersion,
        &l.offset,
        None,
        0.125,
    )
    .expect("glm_fit failed");
    let inp = input(&l, &l.ave_log_cpm);
    let mask = identified(&common::table("fac_glmfit.csv"), l.n_genes, l.n_coef);

    for (null, column) in [
        (TreatNull::Interval, "interval_PValue"),
        (TreatNull::WorstCase, "worstcase_PValue"),
    ] {
        let got = glm_treat(&inp, &fit, None, &Tested::Coef(vec![1]), 1.0, null)
            .expect("glm_treat failed");
        assert_close(
            &got.p_value,
            want.column(column),
            TOL_P_VALUE,
            &format!("fac/treat_{column}"),
        );
    }

    let got = glm_treat(
        &inp,
        &fit,
        None,
        &Tested::Coef(vec![1]),
        1.0,
        TreatNull::Interval,
    )
    .expect("glm_treat failed");
    assert_close(
        &masked(&got.log_fc, &mask, 1),
        &masked(want.column("interval_logFC"), &mask, 1),
        TOL_LOG_FC,
        "fac/treat_logFC",
    );
}
