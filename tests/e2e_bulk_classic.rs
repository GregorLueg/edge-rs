//! End-to-end parity for the classic edgeR chain: filter, normalise, transform,
//! estimate dispersions, exact test, rank.
//!
//! Every reference comes from `tests/r/generate_fixtures.R` against edgeR 4.8.2.
//! Three datasets run the same assertions, chosen to differ in structure rather
//! than in seed:
//!
//! * `fac`, 1000 genes by 12 samples, balanced two-by-two factorial, abundances
//!   over sixteen log2 units, ten planted edge-case rows.
//! * `unbal`, 600 by 9, three groups of two, three and four plus a continuous
//!   covariate, five residual degrees of freedom, high dispersion.
//! * `pois`, 300 by 8, near-Poisson counts between roughly 0.5 and 40.
//!
//! Tolerances are measured rather than guessed. Each constant carries the worst
//! case actually observed across all three datasets, from
//! `EDGE_RS_TOL_REPORT=1 cargo test --release --test e2e_bulk_classic -- --nocapture`.

mod common;

use common::{Tol, assert_close, assert_close_scalar};

use edge_rs::core::dgelist::DgeList;
use edge_rs::core::expression::{ave_log_cpm, cpm};
use edge_rs::core::filtering::filter_by_expr;
use edge_rs::core::normalisation::{NormMethod, calc_norm_factors};
use edge_rs::dispersion::estimate::{EstimateDispParams, estimate_disp};
use edge_rs::exact::{ExactTestParams, RejectionRegion, exact_test};
use edge_rs::prelude::Recycled;
use edge_rs::results::{SortBy, top_tags};

///////////////////
// Tolerances    //
///////////////////

// Every constant below is set about three times the figure the calibration
// report gives as `NEEDS rel`, which is the worst relative difference among the
// entries the absolute leg does not already cover. That is the only number a
// relative tolerance can honestly be read off: the unconditional worst is
// routinely a near-zero value the epsilon handles, and quoting that instead is
// how an earlier revision of this file ended up citing figures three orders of
// magnitude away from what it gated.
//
//   EDGE_RS_TOL_REPORT=1 cargo test --release --test e2e_bulk_classic -- --nocapture

/// Normalisation factors. Trimmed means over logs, no iteration anywhere, so
/// the only cost is summation order. Needs `2.0e-15`.
const TOL_NORM: Tol = Tol::rel(1e-13);

/// CPM. Pure arithmetic on the counts and library sizes. Needs `2.2e-16`.
const TOL_CPM: Tol = Tol::rel(1e-13);

/// log-CPM. The same arithmetic wrapped in a logarithm, so a gene whose CPM
/// lands near one gives a log near zero and the relative test stops meaning
/// anything. The epsilon covers those; beyond it nothing disagrees at all.
/// Needs `0` at `1e-13` absolute.
const TOL_LOG_CPM: Tol = Tol::new(1e-12, 1e-13);

/// `aveLogCPM`, which fits an intercept-only negative binomial per gene with
/// edgeR's own Newton stopping rule, so it is converged but not to machine
/// precision. Needs `4.9e-10`.
const TOL_AVE_LOG_CPM: Tol = Tol::rel(2e-9);

/// Common dispersion, maximised over the 21-point grid and then interpolated.
/// Needs `7.8e-6` on the unbalanced set, `2.9e-6` on the factorial one and
/// `4.4e-10` on the near-Poisson one.
///
/// The floor is not the interpolation. It is the two genes per dataset with an
/// entirely empty group, whose adjusted profile likelihood is arbitrary; see
/// [`TOL_DISPERSION`]. The common dispersion sums the APL over every gene, so
/// their disagreement lands on it directly, which is also why the near-Poisson
/// set, which has no such gene, is four orders of magnitude better.
const TOL_COMMON: Tol = Tol::rel(3e-5);

/// Prior degrees of freedom from the empirical Bayes fit. Needs `1.9e-5` on the
/// factorial set, `1.8e-10` on the near-Poisson one. Same cause as
/// [`TOL_DISPERSION`], one step further on.
const TOL_PRIOR_DF: Tol = Tol::rel(1e-4);

/// Trended and tagwise dispersions. Needs `8.8e-4` tagwise and `5.9e-5`
/// trended on the default fit, and `9.1e-3` on the fixed-`prior_df` variant,
/// where less shrinkage leaves an empty-group gene closer to its own runaway
/// estimate. All three are driven by those genes; the near-Poisson set, which
/// has none, needs `2.7e-7` and `4.2e-8`.
///
/// The gap is one specific, deliberately planted situation: a gene whose counts
/// are entirely zero in one group. Its coefficient MLE is negative infinity, the
/// deviance is flat along that ridge, and the Cox-Reid term is a log determinant
/// of weights that keep shrinking as the intercept runs away, so the adjusted
/// profile likelihood depends on exactly where the damped iteration stops.
///
/// edgeR is no more settled than the crate about where that is. On the same
/// gene it returns an intercept of -34.90 from a null start, -33.30 when started
/// from the crate's own answer, and -45.39 at `tol = 1e-14`, with the deviance
/// agreeing to nine figures throughout. Neither implementation is wrong; the
/// quantity is not identified. Two such genes in 890 move the smoothed trend by
/// the amounts above.
const TOL_DISPERSION: Tol = Tol::rel(3e-2);

/// Exact test fold changes and abundances. Needs `3.9e-9` beyond a `1e-12`
/// absolute floor, which covers the fold changes that are legitimately zero.
const TOL_EXACT_FC: Tol = Tol::new(2e-8, 1e-12);

/// Exact test p-values on the natural scale. Needs `2.8e-10` beyond the floor.
///
/// The floor has to swallow the empty-group genes, where the crate underflows to
/// exactly zero and edgeR reports `1.4e-28`. That makes the natural-scale check
/// blind below `1e-25`, which is why [`TOL_EXACT_LOG_P`] exists: the log scale
/// is what actually gates the significant end.
const TOL_EXACT_P: Tol = Tol::new(1e-8, 1e-25);

/// Exact test p-values as `log(p)`, which is the check with resolution where it
/// matters. Needs `2.6e-10`.
const TOL_EXACT_LOG_P: Tol = Tol::new(1e-8, 1e-9);

/// Benjamini-Hochberg adjusted p-values, a cumulative minimum over a sort.
/// Needs `0`: these agree exactly.
const TOL_FDR: Tol = Tol::rel(1e-12);

////////////
// Datasets   //
////////////////

/// One dataset's inputs, as the fixtures hold them.
struct Dataset {
    /// Fixture prefix, for example `"fac"`.
    tag: &'static str,
    /// Number of design columns in the full design.
    n_coef: usize,
    /// Number of design columns in the group-only design.
    n_coef_grp: usize,
    /// Group labels of the pair the exact test compares, zero-based.
    pair: (usize, usize),
}

/// The three bulk datasets.
const DATASETS: [Dataset; 3] = [
    Dataset { tag: "fac", n_coef: 3, n_coef_grp: 2, pair: (0, 1) },
    Dataset { tag: "unbal", n_coef: 4, n_coef_grp: 3, pair: (0, 2) },
    Dataset { tag: "pois", n_coef: 2, n_coef_grp: 2, pair: (0, 1) },
];

/// Everything a test needs about one dataset, loaded from the fixtures.
struct Loaded {
    /// Raw counts, row-major, before filtering.
    counts: Vec<f64>,
    /// Gene count before filtering.
    n_genes: usize,
    /// Sample count.
    n_samples: usize,
    /// Group label per sample, zero-based.
    group: Vec<usize>,
    /// Full design, row-major `n_samples * n_coef`.
    design: Vec<f64>,
    /// Number of columns in `design`.
    n_coef: usize,
    /// Counts of the retained genes, row-major.
    kept_counts: Vec<f64>,
    /// Retained gene count.
    n_kept: usize,
    /// Library size per sample, after filtering.
    lib_size: Vec<f64>,
    /// Normalisation factor per sample, after filtering.
    norm_factors: Vec<f64>,
    /// Log offsets per sample, after filtering.
    offset: Vec<f64>,
}

/// Loads one dataset's inputs.
///
/// ### Params
///
/// * `d` - Dataset description
///
/// ### Returns
///
/// The counts, design and post-filter sample quantities.
fn load(d: &Dataset) -> Loaded {
    let counts_t = common::table(&format!("{}_counts.csv", d.tag));
    let n_genes = counts_t.n_rows();
    let n_samples = counts_t.n_cols();
    let counts = counts_t.row_major_counts();

    let samples = common::table(&format!("{}_samples.csv", d.tag));
    let group = samples.column_usize("grp");

    let (design, _, n_coef) = common::matrix(&format!("{}_design.csv", d.tag));
    assert_eq!(n_coef, d.n_coef, "{}: unexpected design width", d.tag);
    // Loaded only as a shape check: the exact test takes its dispersion from the
    // fixture rather than refitting against the group-only design here.
    let (_, _, n_coef_grp) = common::matrix(&format!("{}_design_grp.csv", d.tag));
    assert_eq!(n_coef_grp, d.n_coef_grp, "{}: unexpected group design width", d.tag);

    let kept_t = common::table(&format!("{}_kept_counts.csv", d.tag));
    let n_kept = kept_t.n_rows();
    let kept_counts = kept_t.row_major_counts();

    let ks = common::table(&format!("{}_kept_samples.csv", d.tag));

    Loaded {
        counts,
        n_genes,
        n_samples,
        group,
        design,
        n_coef,
        kept_counts,
        n_kept,
        lib_size: ks.column("lib_size").to_vec(),
        norm_factors: ks.column("norm_factors").to_vec(),
        offset: ks.column("offset").to_vec(),
    }
}

/// Pairs up the log p-values the crate can actually report.
///
/// The crate's exact test underflows to exactly zero on a gene whose counts are
/// entirely zero in one group, where edgeR still carries `1.4e-28`. Both are
/// "certainly significant" and the difference matters to nobody, but taking the
/// log turns it into `-inf` against `-64.1`, which no tolerance can absorb and
/// none should try to. Those genes are dropped here and counted by the caller
/// instead, so that the comparison keeps its resolution everywhere else and a
/// change in how much underflows still fails the suite.
///
/// ### Params
///
/// * `p` - The crate's p-values, natural scale
/// * `want_log` - R's log p-values
///
/// ### Returns
///
/// The crate's log p-values and R's, restricted to the genes where the crate's
/// p-value is strictly positive.
fn finite_log_pairs(p: &[f64], want_log: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut got = Vec::with_capacity(p.len());
    let mut want = Vec::with_capacity(p.len());
    for (i, &v) in p.iter().enumerate() {
        if v > 0.0 {
            got.push(v.ln());
            want.push(want_log[i]);
        }
    }
    (got, want)
}

/// How many exact-test p-values underflow to zero on each dataset.
///
/// One gene on the factorial set, none elsewhere. Recorded rather than tolerated:
/// if the crate starts losing more of the tail this fails.
///
/// ### Params
///
/// * `tag` - Dataset prefix
///
/// ### Returns
///
/// The expected count.
fn underflowed(tag: &str) -> usize {
    match tag {
        "fac" => 1,
        _ => 0,
    }
}

/// Effective library sizes, the product edgeR normalises by.
fn effective_lib(l: &Loaded) -> Vec<f64> {
    l.lib_size.iter().zip(&l.norm_factors).map(|(a, b)| a * b).collect()
}

//////////////////
// filterByExpr //
//////////////////

#[test]
fn test_filter_by_expr_matches_edger() {
    for d in &DATASETS {
        let l = load(d);
        let want = common::table(&format!("{}_filter.csv", d.tag));

        // The group path. edgeR takes the group off the DGEList when neither a
        // group nor a design is given, and the generator built that object with
        // `DGEList(counts, group = grp)`, so its `default` and `group` columns
        // are the same computation. They are checked as one rather than looped
        // over, which would have read as covering two API paths while issuing
        // one call.
        let want_default = want.column_bool("default");
        let want_group = want.column_bool("group");
        assert_eq!(
            want_default, want_group,
            "{}: the fixture's default and group columns should be identical, \
             since the DGEList carried the group",
            d.tag
        );
        let got = filter_by_expr(
            &l.counts, l.n_genes, l.n_samples, None, Some(&l.group), None, None,
        )
        .expect("filter_by_expr failed");
        let diffs: Vec<usize> = (0..l.n_genes).filter(|&g| got[g] != want_group[g]).collect();
        assert!(
            diffs.is_empty(),
            "{}/group: {} genes differ, first at {}",
            d.tag,
            diffs.len(),
            diffs[0]
        );

        // The design path. A disagreement here is a different gene set rather
        // than a numeric drift, which is why it is compared exactly.
        let got = filter_by_expr(
            &l.counts,
            l.n_genes,
            l.n_samples,
            None,
            None,
            Some((&l.design, l.n_coef)),
            None,
        )
        .expect("filter_by_expr failed");
        let expected = want.column_bool("design");
        let diffs: Vec<usize> = (0..l.n_genes).filter(|&g| got[g] != expected[g]).collect();
        assert!(
            diffs.is_empty(),
            "{}/design: {} genes differ, first at {}",
            d.tag,
            diffs.len(),
            diffs[0]
        );
    }
}

#[test]
fn test_filter_by_expr_after_normalisation_uses_effective_libraries() {
    // edgeR's DGEList method filters on lib.size * norm.factors. That is a no-op
    // in the usual filter-then-normalise order, and not a no-op the other way
    // round, so both orderings are pinned. The crate takes the library sizes
    // explicitly and has no opinion, which is exactly why this is worth a test.
    for d in &DATASETS {
        let l = load(d);
        let want = common::table(&format!("{}_filter.csv", d.tag));

        let nf = calc_norm_factors(
            &l.counts, l.n_genes, l.n_samples, None, NormMethod::Tmm, None, None,
        )
        .expect("calc_norm_factors failed");
        let raw_lib: Vec<f64> = (0..l.n_samples)
            .map(|j| (0..l.n_genes).map(|g| l.counts[g * l.n_samples + j]).sum())
            .collect();
        let eff: Vec<f64> = raw_lib.iter().zip(&nf).map(|(a, b)| a * b).collect();

        let got = filter_by_expr(
            &l.counts,
            l.n_genes,
            l.n_samples,
            Some(&eff),
            None,
            Some((&l.design, l.n_coef)),
            None,
        )
        .expect("filter_by_expr failed");
        let expected = want.column_bool("after_norm");
        let diffs: Vec<usize> = (0..l.n_genes).filter(|&g| got[g] != expected[g]).collect();
        assert!(
            diffs.is_empty(),
            "{}/after_norm: {} genes differ, first at {}",
            d.tag,
            diffs.len(),
            diffs[0]
        );
    }
}

//////////////////////
// calcNormFactors  //
//////////////////////

#[test]
fn test_calc_norm_factors_matches_edger() {
    let methods = [
        ("tmm", NormMethod::Tmm),
        ("tmmwsp", NormMethod::TmmwSp),
        ("rle", NormMethod::Rle),
        ("uq", NormMethod::UpperQuartile),
        ("none", NormMethod::None),
    ];

    for d in &DATASETS {
        let l = load(d);
        let want = common::table(&format!("{}_norm.csv", d.tag));
        let lib = want.column("lib_size");

        for (column, method) in methods {
            let got = calc_norm_factors(
                &l.counts, l.n_genes, l.n_samples, Some(lib), method, None, None,
            )
            .expect("calc_norm_factors failed");
            assert_close(&got, want.column(column), TOL_NORM, &format!("{}/{column}", d.tag));
        }

        // refColumn is one-based in R, so R's 3 is index 2 here.
        let got = calc_norm_factors(
            &l.counts,
            l.n_genes,
            l.n_samples,
            Some(lib),
            NormMethod::Tmm,
            Some(2),
            None,
        )
        .expect("calc_norm_factors failed");
        assert_close(
            &got,
            want.column("tmm_ref3"),
            TOL_NORM,
            &format!("{}/tmm_ref3", d.tag),
        );
    }
}

///////////////////////
// cpm and aveLogCPM //
///////////////////////

#[test]
fn test_cpm_matches_edger() {
    for d in &DATASETS {
        let l = load(d);
        let eff = effective_lib(&l);

        let got = cpm(&l.kept_counts, l.n_kept, l.n_samples, Some(&eff), None, false, 0.0)
            .expect("cpm failed");
        let (want, _, _) = common::matrix(&format!("{}_cpm.csv", d.tag));
        assert_close(&got, &want, TOL_CPM, &format!("{}/cpm", d.tag));

        let got = cpm(&l.kept_counts, l.n_kept, l.n_samples, Some(&eff), None, true, 2.0)
            .expect("cpm failed");
        let (want, _, _) = common::matrix(&format!("{}_logcpm.csv", d.tag));
        assert_close(&got, &want, TOL_LOG_CPM, &format!("{}/logcpm", d.tag));
    }
}

#[test]
fn test_ave_log_cpm_matches_edger() {
    for d in &DATASETS {
        let l = load(d);
        let eff = effective_lib(&l);
        let want = common::table(&format!("{}_avelogcpm.csv", d.tag));

        // edgeR's default dispersion here is 0.05, not zero.
        let got = ave_log_cpm(
            &l.kept_counts,
            l.n_kept,
            l.n_samples,
            Some(&eff),
            None,
            2.0,
            Some(&Recycled::Scalar(0.05)),
        )
        .expect("ave_log_cpm failed");
        assert_close(&got, want.column("default"), TOL_AVE_LOG_CPM, &format!("{}/avelogcpm", d.tag));

        let got = ave_log_cpm(
            &l.kept_counts,
            l.n_kept,
            l.n_samples,
            Some(&eff),
            None,
            2.0,
            Some(&Recycled::Scalar(0.2)),
        )
        .expect("ave_log_cpm failed");
        assert_close(
            &got,
            want.column("disp02"),
            TOL_AVE_LOG_CPM,
            &format!("{}/avelogcpm_disp02", d.tag),
        );

        // The offset form. Vector offsets only: a matrix offset would hit
        // UPSTREAM_DEVIATIONS.md entry 11, where edgeR corrupts the first gene.
        let offset = Recycled::by_sample(l.offset.clone());
        let got = ave_log_cpm(
            &l.kept_counts,
            l.n_kept,
            l.n_samples,
            None,
            Some(&offset),
            2.0,
            Some(&Recycled::Scalar(0.05)),
        )
        .expect("ave_log_cpm failed");
        assert_close(
            &got,
            want.column("with_offset"),
            TOL_AVE_LOG_CPM,
            &format!("{}/avelogcpm_offset", d.tag),
        );
    }
}

//////////////////
// estimateDisp //
//////////////////

#[test]
fn test_estimate_disp_matches_edger() {
    let s = common::scalars();

    for d in &DATASETS {
        let l = load(d);
        let want = common::table(&format!("{}_disp.csv", d.tag));
        let offset = Recycled::by_sample(l.offset.clone());

        // edgeR computes its own AveLogCPM inside estimateDisp, at the first-pass
        // common dispersion rather than the 0.05 default, so the two differ by
        // up to a hundredth of a log2 unit. Passing R's value in isolates the
        // dispersion estimation from that difference.
        let got = estimate_disp(
            &l.kept_counts,
            l.n_kept,
            l.n_samples,
            &l.design,
            l.n_coef,
            &offset,
            None,
            Some(want.column("ave_log_cpm")),
            None,
        )
        .expect("estimate_disp failed");

        assert_close_scalar(
            got.common,
            s.get(d.tag, "common_dispersion"),
            TOL_COMMON,
            &format!("{}/common", d.tag),
        );
        assert_close_scalar(got.span, s.get(d.tag, "span"), TOL_COMMON, &format!("{}/span", d.tag));

        let trended = got.trended.as_ref().expect("trended dispersion missing");
        assert_close(trended, want.column("trended"), TOL_DISPERSION, &format!("{}/trended", d.tag));

        let tagwise = got.tagwise.as_ref().expect("tagwise dispersion missing");
        assert_close(tagwise, want.column("tagwise"), TOL_DISPERSION, &format!("{}/tagwise", d.tag));

        let prior_df = got.prior_df.as_ref().expect("prior df missing");
        assert_eq!(
            prior_df.len(),
            s.get_usize(d.tag, "prior_df_len"),
            "{}: prior_df length",
            d.tag
        );
        assert_close_scalar(
            prior_df[0],
            s.get(d.tag, "prior_df"),
            TOL_PRIOR_DF,
            &format!("{}/prior_df", d.tag),
        );
    }
}

#[test]
fn test_estimate_disp_without_tagwise_matches_edger() {
    // The factorial set is the one with enough abundance range for the trend to
    // be interesting, so the parameter sweeps run there only.
    let s = common::scalars();
    let l = load(&DATASETS[0]);
    let disp = common::table("fac_disp.csv");
    let extra = common::table("fac_trend_extra.csv");
    let offset = Recycled::by_sample(l.offset.clone());

    let params = EstimateDispParams { tagwise: false, ..Default::default() };
    let got = estimate_disp(
        &l.kept_counts,
        l.n_kept,
        l.n_samples,
        &l.design,
        l.n_coef,
        &offset,
        None,
        Some(disp.column("ave_log_cpm")),
        Some(params),
    )
    .expect("estimate_disp failed");

    assert!(got.tagwise.is_none(), "tagwise should be absent when it was not asked for");
    assert_close_scalar(
        got.common,
        s.get("fac_trend", "common_no_tagwise"),
        TOL_COMMON,
        "fac/common_no_tagwise",
    );
    assert_close(
        got.trended.as_ref().expect("trended missing"),
        extra.column("trended_no_tagwise"),
        TOL_DISPERSION,
        "fac/trended_no_tagwise",
    );
}

#[test]
fn test_estimate_disp_with_a_fixed_prior_df_matches_edger() {
    let l = load(&DATASETS[0]);
    let disp = common::table("fac_disp.csv");
    let extra = common::table("fac_trend_extra.csv");
    let offset = Recycled::by_sample(l.offset.clone());

    let params = EstimateDispParams { prior_df: Some(10.0), ..Default::default() };
    let got = estimate_disp(
        &l.kept_counts,
        l.n_kept,
        l.n_samples,
        &l.design,
        l.n_coef,
        &offset,
        None,
        Some(disp.column("ave_log_cpm")),
        Some(params),
    )
    .expect("estimate_disp failed");

    assert_close(
        got.tagwise.as_ref().expect("tagwise missing"),
        extra.column("tagwise_pdf10"),
        TOL_DISPERSION,
        "fac/tagwise_pdf10",
    );
}

////////////////
// exactTest  //
////////////////

#[test]
fn test_exact_test_matches_edger() {
    for d in &DATASETS {
        let l = load(d);
        let want = common::table(&format!("{}_exact.csv", d.tag));

        let mut dge = DgeList::new(
            l.kept_counts.clone(),
            l.n_kept,
            l.n_samples,
            Some(l.group.clone()),
        )
        .expect("DgeList::new failed");
        dge.lib_size = l.lib_size.clone();
        dge.norm_factors = l.norm_factors.clone();
        // edgeR's exactTest reuses whatever AveLogCPM the DGEList already
        // carries rather than recomputing one, and estimateDisp put a value
        // there computed at its own common dispersion rather than aveLogCPM's
        // 0.05 default. The crate honours the stored field the same way, so it
        // has to be populated or the two are computing different things.
        dge.ave_log_cpm = Some(want.column("ave_log_cpm").to_vec());

        let dispersion = Recycled::by_gene(want.column("tagwise").to_vec());

        let got = exact_test(&dge, d.pair, Some(&dispersion), None).expect("exact_test failed");
        assert_close(&got.log_fc, want.column("logFC"), TOL_EXACT_FC, &format!("{}/exact_logFC", d.tag));
        assert_close(
            &got.log_cpm,
            want.column("logCPM"),
            TOL_EXACT_FC,
            &format!("{}/exact_logCPM", d.tag),
        );
        assert_close(
            &got.p_value,
            want.column("PValue"),
            TOL_EXACT_P,
            &format!("{}/exact_PValue", d.tag),
        );
        // The natural-scale check above is blind below its absolute floor, and
        // the floor has to be there because a p-value that underflowed to zero
        // in one implementation and not the other cannot be compared relatively.
        // So the significant end is gated on the log instead, over the genes
        // where the crate did not underflow, with the underflow itself counted
        // rather than swallowed.
        let (log_got, log_want) = finite_log_pairs(&got.p_value, want.column("logPValue"));
        assert_close(
            &log_got,
            &log_want,
            TOL_EXACT_LOG_P,
            &format!("{}/exact_logPValue", d.tag),
        );
        assert_eq!(
            got.p_value.len() - log_got.len(),
            underflowed(d.tag),
            "{}: number of p-values underflowing to zero has changed",
            d.tag
        );

        // big_count = 5 forces almost every gene onto the beta approximation
        // instead of the exact negative binomial sum.
        let params = ExactTestParams {
            rejection_region: RejectionRegion::DoubleTail,
            big_count: 5.0,
            prior_count: 0.125,
        };
        let got = exact_test(&dge, d.pair, Some(&dispersion), Some(params))
            .expect("exact_test failed");
        assert_close(
            &got.p_value,
            want.column("PValue_big5"),
            TOL_EXACT_P,
            &format!("{}/exact_PValue_big5", d.tag),
        );
        let (log_got, log_want) =
            finite_log_pairs(&got.p_value, want.column("logPValue_big5"));
        assert_close(
            &log_got,
            &log_want,
            TOL_EXACT_LOG_P,
            &format!("{}/exact_logPValue_big5", d.tag),
        );
        assert_close(
            &got.log_fc,
            want.column("logFC_big5"),
            TOL_EXACT_FC,
            &format!("{}/exact_logFC_big5", d.tag),
        );
    }
}

//////////////
// topTags  //
//////////////

#[test]
fn test_top_tags_matches_edger() {
    for d in &DATASETS {
        let lrt = common::table(&format!("{}_lrt.csv", d.tag));

        let log_fc = lrt.column("logFC");
        let log_cpm = lrt.column("logCPM");
        let statistic = lrt.column("LR");
        let p_value = lrt.column("PValue");

        for (variant, sort_by) in [
            ("PValue", SortBy::PValue),
            ("logFC", SortBy::LogFc),
            ("none", SortBy::None),
        ] {
            let want = common::table(&format!("{}_toptags_{variant}.csv", d.tag));
            let got = top_tags(log_fc, log_cpm, Some(statistic), p_value, 50, sort_by, 1.0)
                .expect("top_tags failed");

            assert_eq!(got.index.len(), want.n_rows(), "{}/{variant}: row count", d.tag);

            // R's index column is one-based. edgeR orders with a stable radix
            // sort and the crate documents that it matches, so this is compared
            // exactly rather than as a set.
            let want_index: Vec<usize> =
                want.column_usize("index").iter().map(|i| i - 1).collect();
            common::assert_eq_usize(&got.index, &want_index, &format!("{}/{variant}_index", d.tag));

            assert_close(
                &got.p_value,
                want.column("PValue"),
                TOL_EXACT_P,
                &format!("{}/{variant}_PValue", d.tag),
            );
            assert_close(
                &got.fdr,
                want.column("FDR"),
                TOL_FDR,
                &format!("{}/{variant}_FDR", d.tag),
            );
        }
    }
}

#[test]
fn test_top_tags_fdr_is_computed_over_the_whole_table() {
    // edgeR adjusts before it truncates, so the FDR column does not move with
    // `n`. Asking for ten rows and for everything has to give the same numbers.
    let d = &DATASETS[0];
    let lrt = common::table(&format!("{}_lrt.csv", d.tag));
    let n = lrt.n_rows();

    let small = top_tags(
        lrt.column("logFC"),
        lrt.column("logCPM"),
        Some(lrt.column("LR")),
        lrt.column("PValue"),
        10,
        SortBy::PValue,
        1.0,
    )
    .expect("top_tags failed");
    let full = top_tags(
        lrt.column("logFC"),
        lrt.column("logCPM"),
        Some(lrt.column("LR")),
        lrt.column("PValue"),
        n,
        SortBy::PValue,
        1.0,
    )
    .expect("top_tags failed");

    assert_close(&small.fdr, &full.fdr[..10], Tol::rel(0.0), "fac/fdr_independent_of_n");
}
