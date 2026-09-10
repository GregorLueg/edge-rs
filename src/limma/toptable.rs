//! limma's `topTable`: the moderated fit as a ranked table.
//!
//! Deliberately separate from [`crate::results::top_tags`], which is edgeR's.
//! The two look alike and are not: `topTags` breaks p-value ties on descending
//! `|logFC|` where `topTable` does not break them at all, `topTable` offers
//! `AveExpr`, `t` and `B` as sort keys and a second `resort_by` pass, and the
//! fold-change threshold is `>=` here, `>` in the F variant and `<` in
//! `decide_tests`. All three asymmetries are upstream's.
//!
//! Sequential throughout. This is one sort and one scan over the genes, with
//! nothing to fan out over.

use crate::limma::marray::MArrayLm;
use crate::numeric::dist::t_ppf;
use crate::numeric::stats::p_adjust_bh;
use crate::prelude::*;

////////////
// Consts //
////////////

/// Confidence level limma uses when `confint = TRUE` rather than a number.
pub const DEFAULT_CONF_LEVEL: f64 = 0.95;

//////////////////
// Public types //
//////////////////

/// Which column the table is ranked on.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum TopTableSort {
    /// Descending log-odds. limma's default, and only available once `eBayes`
    /// has produced them.
    #[default]
    B,
    /// Descending absolute log fold change.
    LogFc,
    /// Descending average expression.
    AveExpr,
    /// Ascending p-value.
    PValue,
    /// Descending absolute moderated t.
    T,
    /// Input order.
    None,
}

/// A second ordering applied to the rows that survived selection.
///
/// Distinct from [`TopTableSort`] in more than timing: `LogFc` and `T` order on
/// the **signed** value here, not the absolute one, so a resort by `LogFc`
/// puts the most up-regulated first rather than the most changed
/// (`R/toptable.R:280-286`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TopTableResort {
    /// Descending signed log fold change.
    LogFc,
    /// Descending average expression.
    AveExpr,
    /// Ascending p-value.
    PValue,
    /// Descending signed moderated t.
    T,
    /// Descending log-odds.
    B,
}

/// Tuning knobs for [`top_table`] and [`top_table_f`], at limma's defaults
/// except for `number`.
#[derive(Clone, Debug)]
pub struct TopTableParams {
    /// Maximum rows to return. limma defaults to ten; the default here is every
    /// row, since a caller that wants ten can say so and one that wants all of
    /// them should not have to.
    pub number: usize,
    /// Ranking column. Ignored by [`top_table_f`], which ranks on the F
    /// p-value or not at all.
    pub sort_by: TopTableSort,
    /// Optional second ordering, applied after selection. Ignored by
    /// [`top_table_f`].
    pub resort_by: Option<TopTableResort>,
    /// Keep only rows whose adjusted p-value is at or below this.
    pub p_value: f64,
    /// Keep only rows whose absolute log fold change is at least this.
    pub lfc: f64,
    /// Confidence level for an interval on the log fold change, or `None` for
    /// no interval. [`DEFAULT_CONF_LEVEL`] is limma's `confint = TRUE`.
    pub confint: Option<f64>,
}

impl Default for TopTableParams {
    fn default() -> Self {
        Self {
            number: usize::MAX,
            sort_by: TopTableSort::B,
            resort_by: None,
            p_value: 1.0,
            lfc: 0.0,
            confint: None,
        }
    }
}

/// A ranked table for one coefficient.
///
/// Every vector is the same length and in the table's own order;
/// [`TopTable::index`] maps back to the original gene.
#[derive(Clone, Debug)]
pub struct TopTable {
    /// Original gene index of each row.
    pub index: Vec<usize>,
    /// Log fold change, the coefficient itself.
    pub log_fc: Vec<f64>,
    /// Lower end of the confidence interval, when one was asked for.
    pub ci_lower: Option<Vec<f64>>,
    /// Upper end of the confidence interval.
    pub ci_upper: Option<Vec<f64>>,
    /// Average expression, from [`MArrayLm::amean`]. `None` when the fit has
    /// none.
    pub ave_expr: Option<Vec<f64>>,
    /// Moderated t.
    pub t: Vec<f64>,
    /// Two-sided p-value.
    pub p_value: Vec<f64>,
    /// Benjamini-Hochberg adjusted p-value, computed over **all** genes before
    /// any thinning, so it does not depend on `number` or `p_value`.
    pub adj_p_value: Vec<f64>,
    /// Log-odds of differential expression.
    pub b: Vec<f64>,
}

/// A ranked table across several coefficients, ordered by the moderated F.
#[derive(Clone, Debug)]
pub struct TopTableF {
    /// Original gene index of each row.
    pub index: Vec<usize>,
    /// The coefficients tested, row-major `index.len() * n_coef`.
    pub coefficients: Vec<f64>,
    /// Number of coefficients per row.
    pub n_coef: usize,
    /// Average expression. `None` when the fit has none.
    pub ave_expr: Option<Vec<f64>>,
    /// Moderated F.
    pub f_stat: Vec<f64>,
    /// P-value of the moderated F.
    pub p_value: Vec<f64>,
    /// Benjamini-Hochberg adjusted p-value, over all genes.
    pub adj_p_value: Vec<f64>,
}

/////////////
// Helpers //
/////////////

/// Checks the two cutoffs.
///
/// ### Params
///
/// * `params` - Tuning knobs
///
/// ### Returns
///
/// `Ok(())`, or [`EdgeErrors::InvalidArgument`].
fn check_cutoffs(params: &TopTableParams) -> Result<(), EdgeErrors> {
    if !(params.p_value >= 0.0 && params.p_value <= 1.0) {
        return Err(EdgeErrors::InvalidArgument(format!(
            "p_value must lie in [0, 1]; got {}",
            params.p_value
        )));
    }
    if !(params.lfc >= 0.0 && params.lfc.is_finite()) {
        return Err(EdgeErrors::InvalidArgument(format!(
            "lfc must be finite and non-negative; got {}",
            params.lfc
        )));
    }
    if let Some(level) = params.confint
        && !(level > 0.0 && level < 1.0)
    {
        return Err(EdgeErrors::InvalidArgument(format!(
            "confint must lie strictly inside (0, 1); got {level}"
        )));
    }
    Ok(())
}

/// The three per-coefficient matrices `eBayes` leaves on a fit.
struct Moderated<'a> {
    /// Moderated t, row-major `n_genes * n_coef`.
    t: &'a [f64],
    /// Two-sided p-values, same shape.
    p: &'a [f64],
    /// Log-odds, same shape.
    lods: &'a [f64],
}

/// Pulls the moderated statistics off a fit, or explains that they are missing.
///
/// ### Params
///
/// * `fit` - The fit
///
/// ### Returns
///
/// The three matrices, or [`EdgeErrors::InvalidArgument`].
fn moderated(fit: &MArrayLm) -> Result<Moderated<'_>, EdgeErrors> {
    match (
        fit.t.as_deref(),
        fit.p_value.as_deref(),
        fit.lods.as_deref(),
    ) {
        (Some(t), Some(p), Some(lods)) => Ok(Moderated { t, p, lods }),
        _ => Err(EdgeErrors::InvalidArgument(
            "the fit has no moderated statistics; run `ebayes` first".to_string(),
        )),
    }
}

/// Stable ordering of a key vector, missing values last.
///
/// R's `order` is a radix sort on doubles, so ties keep their input order in
/// both directions, and `na.last = TRUE` sends `NA` to the end regardless of
/// the direction. Both matter: a table where half the genes tie on a p-value of
/// one would otherwise come back in an arbitrary order.
///
/// ### Params
///
/// * `keys` - Sort keys
/// * `ascending` - Whether smaller comes first
///
/// ### Returns
///
/// Indices into `keys`, in sorted order.
fn order_by(keys: &[f64], ascending: bool) -> Vec<usize> {
    let mut order: Vec<usize> = (0..keys.len()).collect();
    order.sort_by(|&a, &b| {
        let (x, y) = (keys[a], keys[b]);
        match (x.is_nan(), y.is_nan()) {
            (true, true) => std::cmp::Ordering::Equal,
            (true, false) => std::cmp::Ordering::Greater,
            (false, true) => std::cmp::Ordering::Less,
            (false, false) => {
                if ascending {
                    x.total_cmp(&y)
                } else {
                    y.total_cmp(&x)
                }
            }
        }
    });
    order
}

//////////////
// Frontend //
//////////////

/// Ranks the genes for one coefficient.
///
/// Port of limma's `topTable` on a single coefficient, which upstream reaches
/// through `.topTableT` (`R/toptable.R:152-290`). The fit must have been
/// through [`crate::limma::ebayes::ebayes`].
///
/// Order of operations, which differs from `top_tags`: adjust over every gene,
/// thin on the adjusted p-value and the fold change, sort, truncate, then
/// optionally resort.
///
/// ### Params
///
/// * `fit` - A moderated fit
/// * `coef` - Which coefficient, or contrast, to tabulate
/// * `params` - Tuning knobs, or `None` for [`TopTableParams::default`]
///
/// ### Returns
///
/// The table, possibly empty. [`EdgeErrors::CoefOutOfRange`] for a coefficient
/// outside the fit, or [`EdgeErrors::InvalidArgument`] if `eBayes` has not run,
/// if a cutoff is out of range, or if a sort key was asked for that the fit
/// cannot supply.
pub fn top_table(
    fit: &MArrayLm,
    coef: usize,
    params: Option<TopTableParams>,
) -> Result<TopTable, EdgeErrors> {
    let params = params.unwrap_or_default();
    check_cutoffs(&params)?;
    if coef >= fit.n_coef {
        return Err(EdgeErrors::CoefOutOfRange {
            index: coef,
            n_coef: fit.n_coef,
        });
    }
    let stats = moderated(fit)?;
    if params.sort_by == TopTableSort::AveExpr && fit.amean.is_none() {
        return Err(EdgeErrors::InvalidArgument(
            "cannot sort by average expression: the fit carries no `amean`".to_string(),
        ));
    }

    let n_genes = fit.n_genes;
    let n_coef = fit.n_coef;
    let column =
        |src: &[f64]| -> Vec<f64> { (0..n_genes).map(|g| src[g * n_coef + coef]).collect() };
    let log_fc = column(&fit.coefficients);
    let tstat = column(stats.t);
    let p_value = column(stats.p);
    let b = column(stats.lods);
    let adj = p_adjust_bh(&p_value);

    let margin = match params.confint {
        None => None,
        Some(level) => {
            let alpha = (1.0 + level) / 2.0;
            let s2_post = fit.s2_post.as_ref().unwrap();
            let df_total = fit.df_total.as_ref().unwrap();
            let stdev = column(&fit.stdev_unscaled);
            let mut m = Vec::with_capacity(n_genes);
            for g in 0..n_genes {
                m.push(s2_post[g].sqrt() * stdev[g] * t_ppf(alpha, df_total[g])?);
            }
            Some(m)
        }
    };

    // -- thin --
    let mut kept: Vec<usize> = (0..n_genes).collect();
    if params.p_value < 1.0 || params.lfc > 0.0 {
        kept.retain(|&g| {
            let significant = adj[g] <= params.p_value;
            let large = log_fc[g].abs() >= params.lfc;
            // R's `sig[is.na(sig)] <- FALSE`.
            significant && large
        });
    }

    // -- sort, then truncate --
    let keys: Vec<f64> = match params.sort_by {
        TopTableSort::B => kept.iter().map(|&g| b[g]).collect(),
        TopTableSort::LogFc => kept.iter().map(|&g| log_fc[g].abs()).collect(),
        TopTableSort::T => kept.iter().map(|&g| tstat[g].abs()).collect(),
        TopTableSort::PValue => kept.iter().map(|&g| p_value[g]).collect(),
        TopTableSort::AveExpr => {
            let a = fit.amean.as_ref().unwrap();
            kept.iter().map(|&g| a[g]).collect()
        }
        TopTableSort::None => Vec::new(),
    };
    if params.sort_by != TopTableSort::None {
        let ascending = params.sort_by == TopTableSort::PValue;
        let order = order_by(&keys, ascending);
        kept = order.into_iter().map(|i| kept[i]).collect();
    }
    kept.truncate(params.number);

    // -- resort --
    if let Some(by) = params.resort_by {
        let keys: Vec<f64> = match by {
            TopTableResort::LogFc => kept.iter().map(|&g| log_fc[g]).collect(),
            TopTableResort::T => kept.iter().map(|&g| tstat[g]).collect(),
            TopTableResort::B => kept.iter().map(|&g| b[g]).collect(),
            TopTableResort::PValue => kept.iter().map(|&g| p_value[g]).collect(),
            TopTableResort::AveExpr => {
                let a = fit.amean.as_ref().ok_or_else(|| {
                    EdgeErrors::InvalidArgument(
                        "cannot resort by average expression: the fit carries no `amean`"
                            .to_string(),
                    )
                })?;
                kept.iter().map(|&g| a[g]).collect()
            }
        };
        let order = order_by(&keys, by == TopTableResort::PValue);
        kept = order.into_iter().map(|i| kept[i]).collect();
    }

    let pick = |src: &[f64]| -> Vec<f64> { kept.iter().map(|&g| src[g]).collect() };
    Ok(TopTable {
        log_fc: pick(&log_fc),
        ci_lower: margin
            .as_ref()
            .map(|m| kept.iter().map(|&g| log_fc[g] - m[g]).collect()),
        ci_upper: margin
            .as_ref()
            .map(|m| kept.iter().map(|&g| log_fc[g] + m[g]).collect()),
        ave_expr: fit.amean.as_ref().map(|a| pick(a)),
        t: pick(&tstat),
        p_value: pick(&p_value),
        adj_p_value: pick(&adj),
        b: pick(&b),
        index: kept,
    })
}

/// Ranks the genes across several coefficients on the moderated F.
///
/// Port of limma's `topTable` with more than one coefficient, which upstream
/// reaches through `.topTableF` (`R/toptable.R:68-150`).
///
/// The F statistic is the one already on the fit, over **every** coefficient,
/// even when `coefs` names a subset. That is upstream's behaviour and it is
/// easy to misread: `topTable` subsets the fit at `R/toptable.R:45`, but `F`
/// belongs to the per-gene group in `[.MArrayLM` (`R/subsetting.R:117-118`), so
/// subsetting columns leaves it untouched. `coefs` selects which coefficients
/// are tabulated, not which hypothesis is tested. To test a subset, rotate onto
/// it with `contrasts_fit` and moderate again.
///
/// ### Params
///
/// * `fit` - A moderated fit whose `f_stat` is set
/// * `coefs` - Which coefficients to tabulate, at least two
/// * `params` - Tuning knobs. Only `number`, `p_value`, `lfc` and whether
///   `sort_by` is [`TopTableSort::None`] are read.
///
/// ### Returns
///
/// The table, possibly empty. [`EdgeErrors::CoefOutOfRange`] for an index
/// outside the fit, or [`EdgeErrors::InvalidArgument`] if the fit has no F
/// statistic, if fewer than two coefficients were asked for, or if a cutoff is
/// out of range.
pub fn top_table_f(
    fit: &MArrayLm,
    coefs: &[usize],
    params: Option<TopTableParams>,
) -> Result<TopTableF, EdgeErrors> {
    let params = params.unwrap_or_default();
    check_cutoffs(&params)?;
    if coefs.len() < 2 {
        return Err(EdgeErrors::InvalidArgument(
            "top_table_f needs at least two coefficients; use top_table for one".to_string(),
        ));
    }
    if let Some(&c) = coefs.iter().find(|&&c| c >= fit.n_coef) {
        return Err(EdgeErrors::CoefOutOfRange {
            index: c,
            n_coef: fit.n_coef,
        });
    }
    let f_stat = fit.f_stat.as_ref().ok_or_else(|| {
        EdgeErrors::InvalidArgument(
            "the fit has no moderated F; run `ebayes` on a full rank design first".to_string(),
        )
    })?;
    let p_value = fit.f_p_value.as_ref().unwrap();

    let n_genes = fit.n_genes;
    let n_coef = fit.n_coef;
    let adj = p_adjust_bh(p_value);

    // -- thin: any coefficient over the threshold, and significant --
    let mut kept: Vec<usize> = (0..n_genes).collect();
    if params.p_value < 1.0 || params.lfc > 0.0 {
        kept.retain(|&g| {
            let large = params.lfc <= 0.0
                || coefs
                    .iter()
                    .any(|&c| fit.coefficients[g * n_coef + c].abs() > params.lfc);
            let significant = params.p_value >= 1.0 || adj[g] <= params.p_value;
            large && significant
        });
    }

    if params.sort_by != TopTableSort::None {
        let keys: Vec<f64> = kept.iter().map(|&g| p_value[g]).collect();
        let order = order_by(&keys, true);
        kept = order.into_iter().map(|i| kept[i]).collect();
    }
    kept.truncate(params.number);

    let mut coefficients = Vec::with_capacity(kept.len() * coefs.len());
    for &g in &kept {
        coefficients.extend(coefs.iter().map(|&c| fit.coefficients[g * n_coef + c]));
    }
    let pick = |src: &[f64]| -> Vec<f64> { kept.iter().map(|&g| src[g]).collect() };
    Ok(TopTableF {
        coefficients,
        n_coef: coefs.len(),
        ave_expr: fit.amean.as_ref().map(|a| pick(a)),
        f_stat: pick(f_stat),
        p_value: pick(p_value),
        adj_p_value: pick(&adj),
        index: kept,
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::limma::ebayes::ebayes;
    use crate::limma::lm_fit::lm_fit;
    use approx::assert_relative_eq;

    /// A moderated fit over eight genes, two coefficients, six samples.
    fn moderated_fit() -> MArrayLm {
        let design = vec![
            1.0, 0.0, //
            1.0, 0.0, //
            1.0, 0.0, //
            1.0, 1.0, //
            1.0, 1.0, //
            1.0, 1.0,
        ];
        let y: Vec<f64> = (0..48).map(|i| ((i * 17) % 41) as f64 / 8.0).collect();
        let amean: Vec<f64> = y
            .chunks_exact(6)
            .map(|r| r.iter().sum::<f64>() / 6.0)
            .collect();
        let fit = lm_fit(&y, 8, 6, &design, 2, None, None, None).unwrap();
        let m = MArrayLm::from_lm_fit(fit, &design, 2, 6, Some(amean)).unwrap();
        ebayes(m, None).unwrap()
    }

    #[test]
    fn test_top_table_returns_every_gene_by_default() {
        let fit = moderated_fit();
        let tt = top_table(&fit, 1, None).unwrap();
        assert_eq!(tt.index.len(), 8);
        assert_eq!(tt.log_fc.len(), 8);
        assert!(tt.ave_expr.is_some());
        assert!(tt.ci_lower.is_none());
    }

    #[test]
    fn test_top_table_sorts_by_log_odds_descending() {
        let fit = moderated_fit();
        let tt = top_table(&fit, 1, None).unwrap();
        for w in tt.b.windows(2) {
            assert!(w[0] >= w[1], "log-odds not descending: {:?}", w);
        }
    }

    #[test]
    fn test_top_table_sorts_by_p_value_ascending() {
        let fit = moderated_fit();
        let params = TopTableParams {
            sort_by: TopTableSort::PValue,
            ..Default::default()
        };
        let tt = top_table(&fit, 1, Some(params)).unwrap();
        for w in tt.p_value.windows(2) {
            assert!(w[0] <= w[1]);
        }
    }

    #[test]
    fn test_top_table_sorts_by_absolute_fold_change() {
        let fit = moderated_fit();
        let params = TopTableParams {
            sort_by: TopTableSort::LogFc,
            ..Default::default()
        };
        let tt = top_table(&fit, 1, Some(params)).unwrap();
        for w in tt.log_fc.windows(2) {
            assert!(w[0].abs() >= w[1].abs());
        }
    }

    #[test]
    fn test_top_table_leaves_the_order_alone_when_asked() {
        let fit = moderated_fit();
        let params = TopTableParams {
            sort_by: TopTableSort::None,
            ..Default::default()
        };
        let tt = top_table(&fit, 1, Some(params)).unwrap();
        assert_eq!(tt.index, (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn test_top_table_truncates_to_number() {
        let fit = moderated_fit();
        let params = TopTableParams {
            number: 3,
            ..Default::default()
        };
        let tt = top_table(&fit, 1, Some(params)).unwrap();
        assert_eq!(tt.index.len(), 3);
    }

    #[test]
    fn test_top_table_number_larger_than_the_table_is_harmless() {
        let fit = moderated_fit();
        let params = TopTableParams {
            number: 1000,
            ..Default::default()
        };
        assert_eq!(top_table(&fit, 1, Some(params)).unwrap().index.len(), 8);
    }

    #[test]
    fn test_top_table_filter_can_keep_nothing() {
        let fit = moderated_fit();
        let params = TopTableParams {
            p_value: 0.0,
            lfc: 1e6,
            ..Default::default()
        };
        let tt = top_table(&fit, 1, Some(params)).unwrap();
        assert!(tt.index.is_empty());
        assert!(tt.log_fc.is_empty());
    }

    #[test]
    fn test_top_table_adjusted_p_is_computed_before_thinning() {
        // The adjustment runs over every gene, so truncating must not change
        // the adjusted value a surviving gene carries.
        let fit = moderated_fit();
        let all = top_table(&fit, 1, None).unwrap();
        let params = TopTableParams {
            number: 2,
            ..Default::default()
        };
        let few = top_table(&fit, 1, Some(params)).unwrap();
        for (k, &g) in few.index.iter().enumerate() {
            let j = all.index.iter().position(|&x| x == g).unwrap();
            assert_relative_eq!(few.adj_p_value[k], all.adj_p_value[j], epsilon = 1e-15);
        }
    }

    #[test]
    fn test_top_table_confidence_interval_brackets_the_estimate() {
        let fit = moderated_fit();
        let params = TopTableParams {
            confint: Some(DEFAULT_CONF_LEVEL),
            ..Default::default()
        };
        let tt = top_table(&fit, 1, Some(params)).unwrap();
        let lo = tt.ci_lower.as_ref().unwrap();
        let hi = tt.ci_upper.as_ref().unwrap();
        for k in 0..tt.index.len() {
            assert!(lo[k] <= tt.log_fc[k] && tt.log_fc[k] <= hi[k]);
            // Symmetric about the estimate.
            assert_relative_eq!(tt.log_fc[k] - lo[k], hi[k] - tt.log_fc[k], epsilon = 1e-12);
        }
    }

    #[test]
    fn test_top_table_confidence_interval_survives_thinning() {
        // This is the 3.66.0 bug: the interval must belong to the gene it is
        // printed next to, whether or not a threshold removed anything.
        let fit = moderated_fit();
        let base = TopTableParams {
            confint: Some(DEFAULT_CONF_LEVEL),
            ..Default::default()
        };
        let all = top_table(&fit, 1, Some(base.clone())).unwrap();
        let filtered = top_table(&fit, 1, Some(TopTableParams { lfc: 0.05, ..base })).unwrap();
        for (k, &g) in filtered.index.iter().enumerate() {
            let j = all.index.iter().position(|&x| x == g).unwrap();
            assert_relative_eq!(
                filtered.ci_lower.as_ref().unwrap()[k],
                all.ci_lower.as_ref().unwrap()[j],
                epsilon = 1e-12
            );
        }
    }

    #[test]
    fn test_resort_by_uses_the_signed_value() {
        let fit = moderated_fit();
        let params = TopTableParams {
            sort_by: TopTableSort::PValue,
            resort_by: Some(TopTableResort::LogFc),
            ..Default::default()
        };
        let tt = top_table(&fit, 1, Some(params)).unwrap();
        // Signed, descending, not absolute.
        for w in tt.log_fc.windows(2) {
            assert!(w[0] >= w[1], "signed fold changes not descending: {:?}", w);
        }
    }

    #[test]
    fn test_order_by_is_stable_on_ties() {
        let keys = vec![1.0, 1.0, 1.0, 0.0];
        assert_eq!(order_by(&keys, false), vec![0, 1, 2, 3]);
        assert_eq!(order_by(&keys, true), vec![3, 0, 1, 2]);
    }

    #[test]
    fn test_order_by_sends_missing_values_last() {
        let keys = vec![1.0, f64::NAN, 2.0];
        assert_eq!(order_by(&keys, false), vec![2, 0, 1]);
        assert_eq!(order_by(&keys, true), vec![0, 2, 1]);
    }

    #[test]
    fn test_top_table_f_ranks_on_the_moderated_f() {
        let fit = moderated_fit();
        let tt = top_table_f(&fit, &[0, 1], None).unwrap();
        assert_eq!(tt.n_coef, 2);
        assert_eq!(tt.coefficients.len(), tt.index.len() * 2);
        for w in tt.p_value.windows(2) {
            assert!(w[0] <= w[1]);
        }
    }

    #[test]
    fn test_top_table_f_rejects_a_single_coefficient() {
        let fit = moderated_fit();
        assert!(top_table_f(&fit, &[0], None).is_err());
    }

    #[test]
    fn test_top_table_rejects_a_coefficient_out_of_range() {
        let fit = moderated_fit();
        assert!(top_table(&fit, 5, None).is_err());
    }

    #[test]
    fn test_top_table_rejects_an_unmoderated_fit() {
        let design = vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let y: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let fit = lm_fit(&y, 2, 4, &design, 2, None, None, None).unwrap();
        let m = MArrayLm::from_lm_fit(fit, &design, 2, 4, None).unwrap();
        assert!(top_table(&m, 0, None).is_err());
    }

    #[test]
    fn test_top_table_rejects_bad_cutoffs() {
        let fit = moderated_fit();
        let bad_p = TopTableParams {
            p_value: 2.0,
            ..Default::default()
        };
        assert!(top_table(&fit, 1, Some(bad_p)).is_err());
        let bad_lfc = TopTableParams {
            lfc: -1.0,
            ..Default::default()
        };
        assert!(top_table(&fit, 1, Some(bad_lfc)).is_err());
        let bad_ci = TopTableParams {
            confint: Some(1.5),
            ..Default::default()
        };
        assert!(top_table(&fit, 1, Some(bad_ci)).is_err());
    }
}
