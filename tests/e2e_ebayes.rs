//! End-to-end parity for `contrasts.fit` and `eBayes`.
//!
//! Chain: `voomLmFit -> [contrasts.fit] -> eBayes`, against limma 3.66.0. The voom
//! half is gated in `e2e_voom.rs`; this file covers the moderated t and F, the
//! B-statistic, the contrast rotation and `topTable`.
//!
//! Tolerances come from
//! `EDGE_RS_TOL_REPORT=1 cargo test --release --test e2e_ebayes -- --nocapture`.

mod common;

use common::{Tol, assert_close, assert_close_scalar};
use edge_rs::limma::contrasts::contrasts_fit;
use edge_rs::limma::ebayes::{EBayesParams, EBayesTrend, ebayes};
use edge_rs::limma::marray::MArrayLm;
use edge_rs::limma::voom::voom_lmfit;

////////////////
// Tolerances //
////////////////

// Figures are the worst observed over both datasets and all four eBayes
// variants (NEEDS column).
//
// `unbal` is a decade or two worse than `fac` throughout: its prior df comes out
// of `fitFDistUnequalDF1`, which maximises a flat likelihood with R's `optimize`
// at its default `tol` of 1.2e-4. That puts `df.prior` at 1.4e-8 relative, and
// everything downstream inherits it, amplified by how steeply it enters.

/// Moderated t: a coefficient over `stdev.unscaled * sqrt(s2.post)`. Tracks
/// `s2_post` at half the exponent. Needs `2.8e-9`.
const TOL_T: Tol = Tol::rel(1e-8);

/// Moderated t and moderated F p-values. A relative wobble in the statistic is
/// multiplied by roughly `t^2` in the tail; the worst case is the most
/// significant gene, with `t` around 40. Needs `9.6e-7`. The absolute floor
/// admits p-values that underflowed to zero on both sides.
const TOL_P: Tol = Tol::new(3e-6, 1e-300);

/// Squeezed posterior variance (gated through `squeezeVar` in `e2e_voom.rs`).
/// Needs `5.6e-9`.
const TOL_S2: Tol = Tol::rel(2e-8);

/// Prior and total degrees of freedom: the `optimize` tolerance itself,
/// unamplified, and the source of every other tolerance here. Needs `1.4e-8`.
const TOL_DF: Tol = Tol::rel(5e-8);

/// The B-statistic, the loosest quantity. Its kernel carries a factor of
/// `(1 + df_total) / 2`, so the prior's `1.4e-8` is multiplied by about eighteen
/// on `unbal`, and the robust variant adds a per-gene prior on top. Needs
/// `1.7e-6`. The absolute floor covers the log-odds crossing zero (range about
/// -6 to +114).
const TOL_LODS: Tol = Tol::new(5e-6, 2e-6);

/// Moderated F: a sum of squared rotated t values, so twice the t error before
/// the eigendecomposition adds anything. Needs `4.8e-9`.
const TOL_F: Tol = Tol::rel(2e-8);

/// Contrast coefficients, unscaled standard errors and the rotated covariance.
/// Pure linear algebra with no prior, hence four decades tighter than the rest.
/// Needs `1.4e-11`, and `7.5e-15` absolute.
const TOL_CONTRAST: Tol = Tol::new(1e-10, 1e-13);

/// The prior coefficient variance `tmixture` estimates, averaged over the top one
/// per cent of genes by `|t|`. It would move sharply if a gene crossed the
/// `ntarget` boundary; none does here. Needs `8.1e-10`.
const TOL_VAR_PRIOR: Tol = Tol::rel(3e-9);

//////////////
// Datasets //
//////////////

/// A dataset with an eBayes fixture.
struct Dataset {
    /// Fixture prefix.
    tag: &'static str,
    /// Design columns.
    n_coef: usize,
    /// Contrasts in the matching `<tag>_contrasts.csv`.
    n_contrasts: usize,
}

/// The two voom datasets. The near-Poisson set has no voom or eBayes fixture.
const DATASETS: [Dataset; 2] = [
    Dataset {
        tag: "fac",
        n_coef: 3,
        n_contrasts: 2,
    },
    Dataset {
        tag: "unbal",
        n_coef: 4,
        n_contrasts: 1,
    },
];

/// Counts, design and contrasts for one dataset.
struct Loaded {
    /// Retained counts, row-major.
    counts: Vec<f64>,
    /// Number of genes.
    n_genes: usize,
    /// Number of samples.
    n_samples: usize,
    /// Design, row-major `n_samples * n_coef`.
    design: Vec<f64>,
    /// Number of design columns.
    n_coef: usize,
    /// Contrast matrix, column-major `n_coef * n_contrasts`.
    contrasts: Vec<f64>,
    /// Number of contrasts.
    n_contrasts: usize,
}

/// Loads one dataset.
///
/// ### Params
///
/// * `d` - Dataset description
///
/// ### Returns
///
/// The retained counts, the design and the contrast matrix.
fn load(d: &Dataset) -> Loaded {
    let counts_t = common::table(&format!("{}_kept_counts.csv", d.tag));
    let n_genes = counts_t.n_rows();
    let n_samples = counts_t.n_cols();
    let counts = counts_t.row_major_counts();
    let (design, _, n_coef) = common::matrix(&format!("{}_design.csv", d.tag));
    assert_eq!(n_coef, d.n_coef, "{}: unexpected design width", d.tag);
    // The generator writes the transpose, so row-major reads give the
    // column-major layout `contrasts_fit` wants.
    let (contrasts, n_rows, n_cols) = common::matrix(&format!("{}_contrasts.csv", d.tag));
    assert_eq!(
        n_rows, d.n_contrasts,
        "{}: unexpected contrast count",
        d.tag
    );
    assert_eq!(n_cols, n_coef, "{}: contrast width", d.tag);
    Loaded {
        counts,
        n_genes,
        n_samples,
        design,
        n_coef,
        contrasts,
        n_contrasts: d.n_contrasts,
    }
}

/// Runs `voomLmFit` and wraps the result.
///
/// ### Params
///
/// * `l` - Loaded dataset
///
/// ### Returns
///
/// The fit, with `amean` attached so the trended prior is reachable.
fn fit(l: &Loaded) -> MArrayLm {
    let (voom, lm) = voom_lmfit(
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
    MArrayLm::from_lm_fit(lm, &l.design, l.n_coef, l.n_samples, Some(voom.amean))
        .expect("MArrayLm::from_lm_fit failed")
}

/// Unflattens a per-coefficient block of a fixture into row-major order.
///
/// R writes `t1..tP` as separate columns; the crate holds them row-major.
///
/// ### Params
///
/// * `table` - Loaded fixture
/// * `prefix` - Column name prefix, `t`, `p` or `lods`
/// * `n_genes` - Number of genes
/// * `n_coef` - Number of coefficients
///
/// ### Returns
///
/// Row-major `n_genes * n_coef`.
fn unflatten(table: &common::Table, prefix: &str, n_genes: usize, n_coef: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(n_genes * n_coef);
    let cols: Vec<&[f64]> = (1..=n_coef)
        .map(|c| table.column(&format!("{prefix}{c}")))
        .collect();
    for g in 0..n_genes {
        for col in &cols {
            out.push(col[g]);
        }
    }
    out
}

/// Compares a moderated fit against one of the eBayes fixtures.
///
/// ### Params
///
/// * `out` - The moderated fit
/// * `want` - Loaded fixture
/// * `label` - Prefix for the tolerance report
fn assert_ebayes(out: &MArrayLm, want: &common::Table, label: &str) {
    let n_genes = out.n_genes;
    let n_coef = out.n_coef;
    assert_close(
        out.t.as_ref().unwrap(),
        &unflatten(want, "t", n_genes, n_coef),
        TOL_T,
        &format!("{label}/t"),
    );
    assert_close(
        out.p_value.as_ref().unwrap(),
        &unflatten(want, "p", n_genes, n_coef),
        TOL_P,
        &format!("{label}/p_value"),
    );
    assert_close(
        out.lods.as_ref().unwrap(),
        &unflatten(want, "lods", n_genes, n_coef),
        TOL_LODS,
        &format!("{label}/lods"),
    );
    assert_close(
        out.s2_post.as_ref().unwrap(),
        want.column("s2_post"),
        TOL_S2,
        &format!("{label}/s2_post"),
    );
    assert_close(
        out.df_total.as_ref().unwrap(),
        want.column("df_total"),
        TOL_DF,
        &format!("{label}/df_total"),
    );
}

////////////
// eBayes //
////////////

#[test]
fn test_ebayes_matches_limma() {
    for d in &DATASETS {
        let l = load(d);
        let out = ebayes(fit(&l), None).expect("ebayes failed");
        let want = common::table(&format!("{}_ebayes.csv", d.tag));
        assert_ebayes(&out, &want, d.tag);

        let scalars = common::scalars();
        let scenario = format!("{}_ebayes", d.tag);
        assert_close_scalar(
            out.df_prior.as_ref().unwrap()[0],
            scalars.get(&scenario, "df_prior"),
            TOL_DF,
            &format!("{}/df_prior", d.tag),
        );
        assert_close_scalar(
            out.s2_prior.as_ref().unwrap()[0],
            scalars.get(&scenario, "s2_prior"),
            TOL_S2,
            &format!("{}/s2_prior", d.tag),
        );
        let var_prior = out.var_prior.as_ref().unwrap();
        assert_eq!(var_prior.len(), scalars.get_usize(&scenario, "n_var_prior"));
        for (j, &v) in var_prior.iter().enumerate() {
            assert_close_scalar(
                v,
                scalars.get(&scenario, &format!("var_prior{}", j + 1)),
                TOL_VAR_PRIOR,
                &format!("{}/var_prior{}", d.tag, j + 1),
            );
        }
    }
}

#[test]
fn test_moderated_f_matches_limma() {
    for d in &DATASETS {
        let l = load(d);
        let out = ebayes(fit(&l), None).expect("ebayes failed");
        let want = common::table(&format!("{}_ebayes_f.csv", d.tag));
        assert_close(
            out.f_stat.as_ref().unwrap(),
            want.column("F"),
            TOL_F,
            &format!("{}/F", d.tag),
        );
        assert_close(
            out.f_p_value.as_ref().unwrap(),
            want.column("F_p"),
            TOL_P,
            &format!("{}/F_p_value", d.tag),
        );
    }
}

#[test]
fn test_ebayes_with_an_abundance_trend_matches_limma() {
    for d in &DATASETS {
        let l = load(d);
        let params = EBayesParams {
            trend: EBayesTrend::Amean,
            ..Default::default()
        };
        let out = ebayes(fit(&l), Some(params)).expect("trended ebayes failed");
        let want = common::table(&format!("{}_ebayes_trend.csv", d.tag));
        assert_ebayes(&out, &want, &format!("{}/trend", d.tag));

        assert_close_scalar(
            out.df_prior.as_ref().unwrap()[0],
            common::scalars().get(&format!("{}_ebayes", d.tag), "trend_df_prior"),
            TOL_DF,
            &format!("{}/trend_df_prior", d.tag),
        );
    }
}

#[test]
fn test_robust_ebayes_matches_limma() {
    for d in &DATASETS {
        let l = load(d);
        let params = EBayesParams {
            robust: true,
            ..Default::default()
        };
        let out = ebayes(fit(&l), Some(params)).expect("robust ebayes failed");
        let want = common::table(&format!("{}_ebayes_robust.csv", d.tag));
        assert_ebayes(&out, &want, &format!("{}/robust", d.tag));

        // The robust fit yields a per-gene prior only when it finds an outlier,
        // so the length is part of the parity.
        assert_eq!(
            out.df_prior.as_ref().unwrap().len(),
            common::scalars().get_usize(&format!("{}_ebayes", d.tag), "robust_n_df_prior"),
            "{}: robust df_prior length",
            d.tag
        );
    }
}

///////////////////
// contrasts.fit //
///////////////////

#[test]
fn test_contrasts_fit_matches_limma() {
    for d in &DATASETS {
        let l = load(d);
        let out =
            contrasts_fit(fit(&l), &l.contrasts, l.n_contrasts).expect("contrasts_fit failed");
        assert_eq!(out.n_coef, l.n_contrasts);

        let want = common::table(&format!("{}_contrasts_fit.csv", d.tag));
        let mut want_coef = Vec::with_capacity(l.n_genes * l.n_contrasts);
        let mut want_stdev = Vec::with_capacity(l.n_genes * l.n_contrasts);
        for g in 0..l.n_genes {
            for c in 1..=l.n_contrasts {
                want_coef.push(want.column(&format!("coef{c}"))[g]);
                want_stdev.push(want.column(&format!("stdev{c}"))[g]);
            }
        }
        assert_close(
            &out.coefficients,
            &want_coef,
            TOL_CONTRAST,
            &format!("{}/contrast_coefficients", d.tag),
        );
        assert_close(
            &out.stdev_unscaled,
            &want_stdev,
            TOL_CONTRAST,
            &format!("{}/contrast_stdev_unscaled", d.tag),
        );

        let (want_cov, _, _) = common::matrix(&format!("{}_contrasts_cov.csv", d.tag));
        assert_close(
            &out.cov_coefficients,
            &want_cov,
            TOL_CONTRAST,
            &format!("{}/contrast_cov", d.tag),
        );
    }
}

#[test]
fn test_ebayes_on_contrasts_matches_limma() {
    for d in &DATASETS {
        let l = load(d);
        let rotated =
            contrasts_fit(fit(&l), &l.contrasts, l.n_contrasts).expect("contrasts_fit failed");
        let out = ebayes(rotated, None).expect("ebayes failed");

        let want = common::table(&format!("{}_ebayes_contrasts.csv", d.tag));
        assert_ebayes(&out, &want, &format!("{}/contrasts", d.tag));

        let want_f = common::table(&format!("{}_ebayes_contrasts_f.csv", d.tag));
        assert_close(
            out.f_stat.as_ref().unwrap(),
            want_f.column("F"),
            TOL_F,
            &format!("{}/contrasts_F", d.tag),
        );
        assert_close(
            out.f_p_value.as_ref().unwrap(),
            want_f.column("F_p"),
            TOL_P,
            &format!("{}/contrasts_F_p_value", d.tag),
        );

        let scenario = format!("{}_ebayes_con", d.tag);
        let scalars = common::scalars();
        for (j, &v) in out.var_prior.as_ref().unwrap().iter().enumerate() {
            assert_close_scalar(
                v,
                scalars.get(&scenario, &format!("var_prior{}", j + 1)),
                TOL_VAR_PRIOR,
                &format!("{}/contrast_var_prior{}", d.tag, j + 1),
            );
        }
    }
}

////////////////////
// make_contrasts //
////////////////////

#[test]
fn test_make_contrasts_rebuilds_the_fixture_matrices() {
    // The expressions the R generator passes to `makeContrasts`.
    let (fac, n) = edge_rs::limma::contrasts::make_contrasts(
        &["Int", "grpB", "batb2"],
        &["grpB", "grpB - 0.5 * batb2"],
    )
    .expect("make_contrasts failed");
    assert_eq!(n, 2);
    let (want, _, _) = common::matrix("fac_contrasts.csv");
    assert_close(&fac, &want, Tol::rel(0.0), "fac/make_contrasts");

    let (unbal, n) =
        edge_rs::limma::contrasts::make_contrasts(&["Int", "g2", "g3", "score"], &["g3 - g2"])
            .expect("make_contrasts failed");
    assert_eq!(n, 1);
    let (want, _, _) = common::matrix("unbal_contrasts.csv");
    assert_close(&unbal, &want, Tol::rel(0.0), "unbal/make_contrasts");
}

//////////////
// topTable //
//////////////

/// Tolerance for the table's own columns (the eBayes ones, reordered). Selection
/// and ordering are exact, so only the values carry error.
const TOL_TABLE: Tol = Tol::new(5e-6, 2e-6);

/// Confidence interval bounds: `logFC -/+ margin`, with the margin inheriting
/// `s2_post` and `df_total`. A fold change close to its margin cancels most
/// digits; the worst relative error is `3.2e-7`, on a bound of 4.6e-3 against a
/// fold change of order one. Absolutely it is never worse than `8.6e-9`, so the
/// epsilon carries the real bound.
const TOL_CI: Tol = Tol::new(1e-6, 5e-8);

/// Runs the standard pipeline and returns the moderated fit.
///
/// ### Params
///
/// * `d` - Dataset description
///
/// ### Returns
///
/// The fit after `voomLmFit` and `eBayes` at their defaults.
fn moderated(d: &Dataset) -> (Loaded, MArrayLm) {
    let l = load(d);
    let out = ebayes(fit(&l), None).expect("ebayes failed");
    (l, out)
}

/// Compares a table against a fixture, column by column.
///
/// ### Params
///
/// * `tt` - The table
/// * `want` - Loaded fixture
/// * `label` - Prefix for the tolerance report
fn assert_table(tt: &edge_rs::limma::toptable::TopTable, want: &common::Table, label: &str) {
    // R writes one-based gene indices as row names.
    let want_index: Vec<usize> = want
        .column("index")
        .iter()
        .map(|v| *v as usize - 1)
        .collect();
    assert_eq!(tt.index, want_index, "{label}: row order");
    assert_close(
        &tt.log_fc,
        want.column("logFC"),
        TOL_CONTRAST,
        &format!("{label}/logFC"),
    );
    assert_close(
        tt.ave_expr.as_ref().unwrap(),
        want.column("AveExpr"),
        TOL_CONTRAST,
        &format!("{label}/AveExpr"),
    );
    assert_close(&tt.t, want.column("t"), TOL_T, &format!("{label}/t"));
    assert_close(
        &tt.p_value,
        want.column("P.Value"),
        TOL_P,
        &format!("{label}/P.Value"),
    );
    assert_close(
        &tt.adj_p_value,
        want.column("adj.P.Val"),
        TOL_P,
        &format!("{label}/adj.P.Val"),
    );
    assert_close(&tt.b, want.column("B"), TOL_TABLE, &format!("{label}/B"));
}

#[test]
fn test_top_table_matches_limma_under_every_sort() {
    use edge_rs::limma::toptable::{TopTableParams, TopTableSort, top_table};

    // The fixture suffix R wrote each variant under.
    const SORTS: [(&str, TopTableSort); 4] = [
        ("B", TopTableSort::B),
        ("P", TopTableSort::PValue),
        ("logFC", TopTableSort::LogFc),
        ("none", TopTableSort::None),
    ];

    for d in &DATASETS {
        let (_, fit) = moderated(d);
        for (suffix, sort_by) in SORTS {
            let params = TopTableParams {
                sort_by,
                ..Default::default()
            };
            let tt = top_table(&fit, 1, Some(params)).expect("top_table failed");
            let want = common::table(&format!("{}_toptable_{}.csv", d.tag, suffix));
            assert_eq!(
                tt.index.len(),
                want.n_rows(),
                "{}/{}: row count",
                d.tag,
                suffix
            );
            assert_table(&tt, &want, &format!("{}/{}", d.tag, suffix));
        }
    }
}

#[test]
fn test_top_table_filtering_matches_limma() {
    use edge_rs::limma::toptable::{TopTableParams, TopTableSort, top_table};

    for d in &DATASETS {
        let (_, fit) = moderated(d);
        let params = TopTableParams {
            sort_by: TopTableSort::PValue,
            p_value: 0.05,
            lfc: 1.0,
            ..Default::default()
        };
        let tt = top_table(&fit, 1, Some(params)).expect("top_table failed");
        let want = common::table(&format!("{}_toptable_filtered.csv", d.tag));
        assert_eq!(
            tt.index.len(),
            common::scalars().get_usize(&format!("{}_toptable", d.tag), "n_filtered"),
            "{}: filtered row count",
            d.tag
        );
        assert_table(&tt, &want, &format!("{}/filtered", d.tag));
    }
}

#[test]
fn test_top_table_confidence_intervals_match_limma() {
    use edge_rs::limma::toptable::{TopTableParams, top_table};

    for d in &DATASETS {
        let (_, fit) = moderated(d);
        let params = TopTableParams {
            confint: Some(0.95),
            ..Default::default()
        };
        let tt = top_table(&fit, 1, Some(params)).expect("top_table failed");
        let want = common::table(&format!("{}_toptable_confint.csv", d.tag));
        assert_table(&tt, &want, &format!("{}/confint", d.tag));
        assert_close(
            tt.ci_lower.as_ref().unwrap(),
            want.column("CI.L"),
            TOL_CI,
            &format!("{}/CI.L", d.tag),
        );
        assert_close(
            tt.ci_upper.as_ref().unwrap(),
            want.column("CI.R"),
            TOL_CI,
            &format!("{}/CI.R", d.tag),
        );
    }
}

#[test]
fn test_top_table_confidence_intervals_survive_filtering() {
    use edge_rs::limma::toptable::{TopTableParams, top_table};

    // The interval must stay attached to its gene once a threshold removes rows.
    // limma had a misindexing bug here until mid-2025; fixed in 3.66.0, so the
    // fixture comes straight from `topTable`.
    for d in &DATASETS {
        let (_, fit) = moderated(d);
        let params = TopTableParams {
            confint: Some(0.95),
            lfc: 1.0,
            ..Default::default()
        };
        let tt = top_table(&fit, 1, Some(params)).expect("top_table failed");
        let want = common::table(&format!("{}_toptable_confint_filtered.csv", d.tag));
        assert_eq!(
            tt.index.len(),
            common::scalars().get_usize(&format!("{}_toptable", d.tag), "n_confint_filtered"),
            "{}: filtered row count",
            d.tag
        );
        assert_close(
            tt.ci_lower.as_ref().unwrap(),
            want.column("CI.L"),
            TOL_CI,
            &format!("{}/filtered_CI.L", d.tag),
        );
        assert_close(
            tt.ci_upper.as_ref().unwrap(),
            want.column("CI.R"),
            TOL_CI,
            &format!("{}/filtered_CI.R", d.tag),
        );
    }
}

#[test]
fn test_top_table_f_matches_limma() {
    use edge_rs::limma::toptable::top_table_f;

    for d in &DATASETS {
        let (l, fit) = moderated(d);
        let coefs: Vec<usize> = (0..l.n_coef).collect();
        let tt = top_table_f(&fit, &coefs, None).expect("top_table_f failed");
        let want = common::table(&format!("{}_toptable_f.csv", d.tag));

        let want_index: Vec<usize> = want
            .column("index")
            .iter()
            .map(|v| *v as usize - 1)
            .collect();
        assert_eq!(tt.index, want_index, "{}: F table row order", d.tag);

        let mut want_coef = Vec::with_capacity(tt.index.len() * l.n_coef);
        for g in 0..tt.index.len() {
            for c in 1..=l.n_coef {
                want_coef.push(want.column(&format!("Coef{c}"))[g]);
            }
        }
        assert_close(
            &tt.coefficients,
            &want_coef,
            TOL_CONTRAST,
            &format!("{}/F_table_coefficients", d.tag),
        );
        assert_close(
            &tt.f_stat,
            want.column("F"),
            TOL_F,
            &format!("{}/F_table_F", d.tag),
        );
        assert_close(
            &tt.p_value,
            want.column("P.Value"),
            TOL_P,
            &format!("{}/F_table_P", d.tag),
        );
        assert_close(
            &tt.adj_p_value,
            want.column("adj.P.Val"),
            TOL_P,
            &format!("{}/F_table_adj_P", d.tag),
        );
    }
}

/////////////////
// limma-trend //
/////////////////

// The other bulk route: log-CPM straight into `lm_fit`, with the mean-variance
// relationship absorbed by a trended prior instead of voom weights.
//
// `robust = TRUE` with `trend` reaches `fitFDistUnequalDF1`'s robust branch,
// which nothing else in the suite does.

/// Residual standard deviations from the unweighted fit on log-CPM. Needs
/// `1.5e-14`.
///
/// The limma-trend path runs two to four decades tighter than voom: worst
/// moderated t is `1.4e-10` against voom's `2.8e-9`, and most quantities land at
/// `1e-13` or better. Residual df are constant, so `squeezeVar` takes the legacy
/// moment-matching branch, not the `optimize` profile likelihood. The shared
/// tolerances above are inherited and loose for this path.
const TOL_TREND_SIGMA: Tol = Tol::rel(5e-14);

/// Builds the effective library sizes, `lib.size * norm.factors`.
///
/// Read from `<tag>_kept_samples.csv`; TMM is not re-run.
///
/// ### Params
///
/// * `tag` - Fixture prefix
///
/// ### Returns
///
/// One effective library size per sample.
fn effective_lib(tag: &str) -> Vec<f64> {
    let s = common::table(&format!("{tag}_kept_samples.csv"));
    s.column("lib_size")
        .iter()
        .zip(s.column("norm_factors"))
        .map(|(a, b)| a * b)
        .collect()
}

/// Runs the limma-trend pipeline up to the moderated fit.
///
/// ### Params
///
/// * `d` - Dataset description
/// * `robust` - Whether to Winsorise the prior's moments
///
/// ### Returns
///
/// The loaded dataset and the moderated fit.
fn limma_trend(d: &Dataset, robust: bool) -> (Loaded, MArrayLm) {
    use edge_rs::core::expression::cpm;
    use edge_rs::limma::lm_fit::lm_fit;

    let l = load(d);
    let eff = effective_lib(d.tag);
    let y = cpm(
        &l.counts,
        l.n_genes,
        l.n_samples,
        Some(&eff),
        None,
        true,
        2.0,
    )
    .expect("cpm failed");

    // limma takes `Amean` from `getEAWP`: the row means of the expression matrix.
    let amean: Vec<f64> = y
        .chunks_exact(l.n_samples)
        .map(|r| r.iter().sum::<f64>() / l.n_samples as f64)
        .collect();

    let lm = lm_fit(
        &y,
        l.n_genes,
        l.n_samples,
        &l.design,
        l.n_coef,
        None,
        None,
        None,
    )
    .expect("lm_fit failed");
    let fit = MArrayLm::from_lm_fit(lm, &l.design, l.n_coef, l.n_samples, Some(amean))
        .expect("from_lm_fit failed");

    let out = ebayes(
        fit,
        Some(EBayesParams {
            trend: EBayesTrend::Amean,
            robust,
            ..Default::default()
        }),
    )
    .expect("ebayes failed");
    (l, out)
}

#[test]
fn test_limma_trend_lm_fit_matches_limma() {
    for d in &DATASETS {
        let (_, fit) = limma_trend(d, false);
        let want = common::table(&format!("{}_limma_trend_lmfit.csv", d.tag));
        assert_close(
            &fit.sigma,
            want.column("sigma"),
            TOL_TREND_SIGMA,
            &format!("{}/trend_sigma", d.tag),
        );
        assert_close(
            &fit.df_residual,
            want.column("df_residual"),
            Tol::rel(0.0),
            &format!("{}/trend_df_residual", d.tag),
        );
        assert_close(
            fit.amean.as_ref().unwrap(),
            want.column("amean"),
            TOL_CONTRAST,
            &format!("{}/trend_amean", d.tag),
        );
    }
}

#[test]
fn test_limma_trend_matches_limma() {
    for d in &DATASETS {
        let (_, fit) = limma_trend(d, false);
        let want = common::table(&format!("{}_limma_trend_ebayes.csv", d.tag));
        assert_ebayes(&fit, &want, &format!("{}/limma_trend", d.tag));

        let scenario = format!("{}_limma_trend", d.tag);
        let scalars = common::scalars();
        // A trended prior varies by gene; its df do not, until the robust fit.
        assert_eq!(
            fit.df_prior.as_ref().unwrap().len(),
            scalars.get_usize(&scenario, "n_df_prior"),
            "{}: df_prior length",
            d.tag
        );
        assert_eq!(
            fit.s2_prior.as_ref().unwrap().len(),
            scalars.get_usize(&scenario, "n_s2_prior"),
            "{}: s2_prior length",
            d.tag
        );
        for (j, &v) in fit.var_prior.as_ref().unwrap().iter().enumerate() {
            assert_close_scalar(
                v,
                scalars.get(&scenario, &format!("var_prior{}", j + 1)),
                TOL_VAR_PRIOR,
                &format!("{}/trend_var_prior{}", d.tag, j + 1),
            );
        }
    }
}

#[test]
fn test_robust_limma_trend_matches_limma() {
    for d in &DATASETS {
        let (_, fit) = limma_trend(d, true);
        let want = common::table(&format!("{}_limma_trend_ebayes_robust.csv", d.tag));
        assert_ebayes(&fit, &want, &format!("{}/limma_trend_robust", d.tag));

        // Winsorising makes the prior df a per-gene vector.
        assert_eq!(
            fit.df_prior.as_ref().unwrap().len(),
            common::scalars().get_usize(&format!("{}_limma_trend", d.tag), "robust_n_df_prior"),
            "{}: robust df_prior length",
            d.tag
        );
    }
}

#[test]
fn test_limma_trend_moderated_f_matches_limma() {
    for d in &DATASETS {
        let (_, fit) = limma_trend(d, false);
        let want = common::table(&format!("{}_limma_trend_ebayes_f.csv", d.tag));
        assert_close(
            fit.f_stat.as_ref().unwrap(),
            want.column("F"),
            TOL_F,
            &format!("{}/trend_F", d.tag),
        );
        assert_close(
            fit.f_p_value.as_ref().unwrap(),
            want.column("F_p"),
            TOL_P,
            &format!("{}/trend_F_p_value", d.tag),
        );
    }
}
