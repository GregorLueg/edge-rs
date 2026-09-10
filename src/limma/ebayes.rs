//! limma's `eBayes`: moderated t, moderated F and the B-statistic.
//!
//! Almost all of the work is [`crate::limma::squeeze_var::squeeze_var`], which
//! fits the scaled F prior to the genewise variances. What is left is arithmetic
//! on top of it, and two things that are not:
//!
//! * the B-statistic needs a prior variance for the *coefficients*, estimated by
//!   `tmixture` from the order statistics of the largest moderated t values;
//! * the moderated F rotates the t statistics into a basis where the
//!   coefficients are uncorrelated, which is an eigendecomposition of the
//!   coefficient correlation matrix.
//!
//! Everything genewise is elementwise, so it is a rayon fan-out over genes.
//! `tmixture` is not: it ranks all genes against each other and so runs
//! sequentially.
//!
//! ### References
//!
//! Smyth, Statistical Applications in Genetics and Molecular Biology 3(1), 2004
//! Loennstedt and Speed, Statistica Sinica 12, 2002 (the B-statistic)

use rayon::prelude::*;

use crate::limma::marray::MArrayLm;
use crate::limma::squeeze_var::{SqueezeVarParams, squeeze_var};
use crate::numeric::dist::{chisq_sf, f_sf, t_cdf, t_isf_log, t_sf, t_sf_log};
use crate::numeric::stats::median;
use crate::prelude::*;
use crate::utils::design::is_full_rank;
use crate::utils::linalg::{cov2cor, self_adjoint_eigen_desc};

////////////
// Consts //
////////////

/// Prior degrees of freedom above which the posterior variance is treated as
/// the prior exactly.
///
/// limma's `Infdf <- df.prior > 10^6` (`R/ebayes.R:78`). Past it the
/// log-ratio kernel of the B-statistic is replaced by its limit, which is what
/// keeps a `df.prior` of infinity from producing a `NaN` rather than the
/// perfectly well-defined answer it has.
const INFINITE_DF_PRIOR: f64 = 1e6;

/// Relative size below which an eigenvalue of the correlation matrix counts as
/// zero.
///
/// `sum(E$values / E$values[1] > 1e-8)` in `classifyTestsF`
/// (`R/decidetests.R:213`). It sets the numerator degrees of freedom of the
/// moderated F, so it has to match upstream exactly rather than merely be
/// reasonable.
const EIGEN_RANK_TOL: f64 = 1e-8;

//////////////////
// Public types //
//////////////////

/// What the prior variance is allowed to depend on.
///
/// limma's `trend` argument is overloaded: `FALSE` for no covariate, `TRUE` for
/// the average log-expression, or a numeric vector to supply one directly
/// (`R/ebayes.R:43-54`).
#[derive(Clone, Debug, Default)]
pub enum EBayesTrend {
    /// One prior for every gene. limma's `trend = FALSE`.
    #[default]
    None,
    /// Trend the prior against [`MArrayLm::amean`]. limma's `trend = TRUE`.
    Amean,
    /// Trend the prior against a covariate of the caller's choosing, one value
    /// per gene.
    Covariate(Vec<f64>),
}

/// Tuning knobs for [`ebayes`], at limma's defaults.
#[derive(Clone, Debug)]
pub struct EBayesParams {
    /// Assumed proportion of differentially expressed genes. Only the
    /// B-statistic reads it. Must lie strictly inside `(0, 1)`.
    pub proportion: f64,
    /// Bounds on the ratio of the coefficient standard deviation to the
    /// residual one, which clamp the prior coefficient variance `tmixture`
    /// estimates. limma's `stdev.coef.lim`.
    pub stdev_coef_lim: (f64, f64),
    /// What the prior variance is trended against.
    pub trend: EBayesTrend,
    /// Lowess span for the trended prior. `None` lets `squeezeVar` choose.
    pub span: Option<f64>,
    /// Whether to Winsorise the moments so outlier genes cannot drag the prior
    /// degrees of freedom down.
    pub robust: bool,
    /// Proportions Winsorised off each tail. Only read when `robust` is set.
    pub winsor_tail_p: (f64, f64),
    /// Which family of F fit to use. `None` is limma's own rule.
    pub legacy: Option<bool>,
}

impl Default for EBayesParams {
    fn default() -> Self {
        Self {
            proportion: 0.01,
            stdev_coef_lim: (0.1, 4.0),
            trend: EBayesTrend::None,
            span: None,
            robust: false,
            winsor_tail_p: (0.05, 0.1),
            legacy: None,
        }
    }
}

/////////////
// Kernels //
/////////////

/// Reads a vector that is either length one or one value per gene.
///
/// `squeezeVar` returns a scalar prior for an untrended, non-robust fit and a
/// per-gene one otherwise, and limma relies on R's recycling to paper over the
/// difference.
///
/// ### Params
///
/// * `v` - The vector, length one or at least `i + 1`
/// * `i` - Index wanted
///
/// ### Returns
///
/// `v[0]` for a length-one vector, `v[i]` otherwise.
#[inline]
fn recycled(v: &[f64], i: usize) -> f64 {
    if v.len() == 1 { v[0] } else { v[i] }
}

/// Log-odds of differential expression, gene by gene.
///
/// `lods = log(p / (1 - p)) - log(r) / 2 + kernel` with
/// `r = (u^2 + v0) / u^2` the variance inflation a differentially expressed
/// gene would show. The kernel has two forms: the log-ratio one, and its limit
/// as the prior degrees of freedom go to infinity (`R/ebayes.R:78-88`). limma
/// switches per gene, not globally.
///
/// ### Params
///
/// * `t` - Moderated t, row-major `n_genes * n_coef`
/// * `stdev_unscaled` - Same shape
/// * `df_total` - Total degrees of freedom per gene
/// * `df_prior` - Prior degrees of freedom, length one or per gene
/// * `var_prior` - Prior coefficient variance, one per coefficient
/// * `n_genes` - Number of genes
/// * `n_coef` - Number of coefficients
/// * `proportion` - Assumed proportion of differentially expressed genes
///
/// ### Returns
///
/// The log-odds, row-major `n_genes * n_coef`.
#[allow(clippy::too_many_arguments)]
fn log_odds(
    t: &[f64],
    stdev_unscaled: &[f64],
    df_total: &[f64],
    df_prior: &[f64],
    var_prior: &[f64],
    n_genes: usize,
    n_coef: usize,
    proportion: f64,
) -> Vec<f64> {
    let offset = (proportion / (1.0 - proportion)).ln();
    let mut out = vec![0.0; n_genes * n_coef];
    out.par_chunks_mut(n_coef).enumerate().for_each(|(g, row)| {
        let dft = df_total[g];
        let infinite = recycled(df_prior, g) > INFINITE_DF_PRIOR;
        for (j, slot) in row.iter_mut().enumerate() {
            let k = g * n_coef + j;
            let u2 = stdev_unscaled[k] * stdev_unscaled[k];
            let r = (u2 + var_prior[j]) / u2;
            let t2 = t[k] * t[k];
            let kernel = if infinite {
                t2 * (1.0 - 1.0 / r) / 2.0
            } else {
                (1.0 + dft) / 2.0 * ((t2 + dft) / (t2 / r + dft)).ln()
            };
            *slot = offset - r.ln() / 2.0 + kernel;
        }
    });
    out
}

/// Prior coefficient variance for every coefficient.
///
/// A column loop over [`tmixture_vector`]. Kept sequential: each column already
/// sorts every gene, and there are only ever a handful of columns.
///
/// ### Params
///
/// * `t` - Moderated t, row-major `n_genes * n_coef`
/// * `stdev_unscaled` - Same shape
/// * `df_total` - Total degrees of freedom per gene
/// * `n_genes` - Number of genes
/// * `n_coef` - Number of coefficients
/// * `proportion` - Assumed proportion of differentially expressed genes
/// * `limits` - Lower and upper clamp on the estimate
///
/// ### Returns
///
/// One prior variance per coefficient, `NaN` where the estimate failed.
#[allow(clippy::too_many_arguments)]
fn tmixture_matrix(
    t: &[f64],
    stdev_unscaled: &[f64],
    df_total: &[f64],
    n_genes: usize,
    n_coef: usize,
    proportion: f64,
    limits: (f64, f64),
) -> Result<Vec<f64>, EdgeErrors> {
    (0..n_coef)
        .map(|j| {
            let tstat: Vec<f64> = (0..n_genes).map(|g| t[g * n_coef + j]).collect();
            let u: Vec<f64> = (0..n_genes)
                .map(|g| stdev_unscaled[g * n_coef + j])
                .collect();
            tmixture_vector(&tstat, &u, df_total, proportion, limits)
        })
        .collect()
}

/// Scale factor of a two-component mixture of t distributions.
///
/// The model is that a proportion `p` of genes have a t statistic distributed
/// as `sqrt(1 + v0 / v1) t(df)` and the rest as `t(df)`, with `v1` the squared
/// unscaled standard deviation. This estimates `v0` by comparing the largest
/// observed statistics against the order statistics the null would produce, and
/// averaging the per-gene solutions.
///
/// Genes with unequal degrees of freedom are first converted to the largest
/// `df` present, by matching tail probabilities. That conversion runs in logs,
/// because the statistics it applies to are by construction the most extreme in
/// the experiment and their tail probabilities routinely underflow
/// (`R/ebayes.R:137-138`).
///
/// ### Params
///
/// * `tstat` - Moderated t for one coefficient, one per gene
/// * `stdev_unscaled` - Matching unscaled standard deviations
/// * `df` - Total degrees of freedom per gene
/// * `proportion` - Assumed proportion of differentially expressed genes
/// * `limits` - Lower and upper clamp on each per-gene solution
///
/// ### Returns
///
/// The estimated prior variance, or `NaN` when there are too few genes for the
/// target count to reach one.
fn tmixture_vector(
    tstat: &[f64],
    stdev_unscaled: &[f64],
    df: &[f64],
    proportion: f64,
    limits: (f64, f64),
) -> Result<f64, EdgeErrors> {
    // Missing statistics drop out first, and the gene count that drives
    // everything below is the count of what survives (`R/ebayes.R:116-124`).
    let mut abs_t = Vec::with_capacity(tstat.len());
    let mut v1 = Vec::with_capacity(tstat.len());
    let mut dfs = Vec::with_capacity(tstat.len());
    for i in 0..tstat.len() {
        if !tstat[i].is_nan() {
            abs_t.push(tstat[i].abs());
            v1.push(stdev_unscaled[i] * stdev_unscaled[i]);
            dfs.push(df[i]);
        }
    }

    let n_genes = abs_t.len();
    let ntarget = (proportion / 2.0 * n_genes as f64).ceil() as usize;
    if ntarget < 1 {
        return Ok(f64::NAN);
    }
    // Keeps `ptarget` below one when the target count rounded up past the
    // proportion it came from.
    let p = (ntarget as f64 / n_genes as f64).max(proportion);

    let max_df = dfs.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    for i in 0..n_genes {
        if dfs[i] < max_df {
            let tail = t_sf_log(abs_t[i], dfs[i])?;
            abs_t[i] = t_isf_log(tail, max_df)?;
        }
    }

    // R's `order` is a stable radix sort, so ties keep their input order.
    let mut order: Vec<usize> = (0..n_genes).collect();
    order.sort_by(|&a, &b| abs_t[b].total_cmp(&abs_t[a]));

    let mut v0 = Vec::with_capacity(ntarget);
    for (rank, &i) in order.iter().take(ntarget).enumerate() {
        let stat = abs_t[i];
        let p0 = 2.0 * t_sf(stat, max_df)?;
        let ptarget = ((rank as f64 + 1.0 - 0.5) / n_genes as f64 - (1.0 - p) * p0) / p;
        let mut value = 0.0;
        if ptarget > p0 {
            let qtarget = t_isf_log((0.5 * ptarget).ln(), max_df)?;
            value = v1[i] * ((stat / qtarget).powi(2) - 1.0);
        }
        v0.push(value.clamp(limits.0, limits.1));
    }

    Ok(v0.iter().sum::<f64>() / ntarget as f64)
}

/// Moderated F across the coefficients.
///
/// limma's `classifyTestsF(fstat.only = TRUE)` (`R/decidetests.R:172-223`).
/// The t statistics for one gene are correlated with each other through the
/// design; rotating by `Q = V diag(1 / sqrt(lambda)) / sqrt(r)` decorrelates
/// and rescales them, so their sum of squares is an F on `r` numerator degrees
/// of freedom.
///
/// ### Params
///
/// * `t` - Moderated t, row-major `n_genes * n_coef`
/// * `cov_coefficients` - Unscaled coefficient covariance, row-major
///   `n_coef * n_coef`
/// * `n_genes` - Number of genes
/// * `n_coef` - Number of coefficients
///
/// ### Returns
///
/// The statistic per gene and its numerator degrees of freedom.
fn moderated_f(
    t: &[f64],
    cov_coefficients: &[f64],
    n_genes: usize,
    n_coef: usize,
) -> Result<(Vec<f64>, f64), EdgeErrors> {
    if n_coef == 1 {
        return Ok((t.iter().map(|v| v * v).collect(), 1.0));
    }

    // An all-zero contrast leaves a zero variance, which `cov2cor` would divide
    // by. limma nudges it to one first (`R/decidetests.R:181-189`).
    let mut cov = cov_coefficients.to_vec();
    let smallest = (0..n_coef).fold(f64::INFINITY, |a, i| a.min(cov[i * n_coef + i]));
    if smallest == 0.0 {
        for i in 0..n_coef {
            if cov[i * n_coef + i] == 0.0 {
                cov[i * n_coef + i] = 1.0;
            }
        }
    }

    let cormatrix = cov2cor(&cov, n_coef);
    let (values, vectors) = self_adjoint_eigen_desc(&cormatrix, n_coef)?;
    let r = values
        .iter()
        .filter(|&&v| v / values[0] > EIGEN_RANK_TOL)
        .count()
        .max(1);

    // Q, column-major r columns of length n_coef.
    let scale = (r as f64).sqrt();
    let mut q = vec![0.0; n_coef * r];
    for c in 0..r {
        let w = 1.0 / (values[c].sqrt() * scale);
        for i in 0..n_coef {
            q[c * n_coef + i] = vectors[i * n_coef + c] * w;
        }
    }

    let mut out = vec![0.0; n_genes];
    out.par_iter_mut().enumerate().for_each(|(g, slot)| {
        let row = &t[g * n_coef..(g + 1) * n_coef];
        let mut acc = 0.0;
        for c in 0..r {
            let col = &q[c * n_coef..(c + 1) * n_coef];
            let dot: f64 = row.iter().zip(col).map(|(x, y)| x * y).sum();
            acc += dot * dot;
        }
        *slot = acc;
    });

    Ok((out, r as f64))
}

//////////////
// Frontend //
//////////////

/// Empirical Bayes moderation of a linear model fit.
///
/// Port of limma's `eBayes`. Shrinks each gene's residual variance towards a
/// fitted prior, divides the coefficients by the moderated standard errors to
/// get a t statistic with more degrees of freedom than the gene itself has, and
/// adds the log-odds of differential expression. When the design is full rank
/// it also computes the moderated F across coefficients.
///
/// The fit must have been through [`crate::limma::lm_fit::lm_fit`], and through
/// [`crate::limma::contrasts::contrasts_fit`] first if the hypotheses of
/// interest are contrasts rather than bare coefficients.
///
/// ### Params
///
/// * `fit` - The fit to moderate, consumed
/// * `params` - Tuning knobs, or `None` for [`EBayesParams::default`]
///
/// ### Returns
///
/// The fit with `s2_post`, `t`, `df_total`, `p_value`, `lods` and the prior
/// quantities filled in, plus `f_stat` and `f_p_value` when the design is full
/// rank. [`EdgeErrors`] if no gene has residual degrees of freedom, no residual
/// standard deviation is finite, `proportion` is outside `(0, 1)`, or
/// `EBayesTrend::Amean` was asked for on a fit carrying no `amean`.
///
/// ### References
///
/// Smyth, Statistical Applications in Genetics and Molecular Biology 3(1), 2004
pub fn ebayes(mut fit: MArrayLm, params: Option<EBayesParams>) -> Result<MArrayLm, EdgeErrors> {
    let params = params.unwrap_or_default();
    // -- checks --
    if !(params.proportion > 0.0 && params.proportion < 1.0) {
        return Err(EdgeErrors::InvalidArgument(format!(
            "proportion must lie strictly inside (0, 1); got {}",
            params.proportion
        )));
    }
    if fit.df_residual.iter().fold(0.0_f64, |a, &b| a.max(b)) == 0.0 {
        return Err(EdgeErrors::InvalidArgument(
            "no residual degrees of freedom in the linear model fits".to_string(),
        ));
    }
    if !fit.sigma.iter().any(|v| v.is_finite()) {
        return Err(EdgeErrors::InvalidArgument(
            "no finite residual standard deviations".to_string(),
        ));
    }

    let n_genes = fit.n_genes;
    let n_coef = fit.n_coef;

    // -- the covariate the prior is trended against --
    let covariate: Option<Vec<f64>> = match &params.trend {
        EBayesTrend::None => None,
        EBayesTrend::Amean => Some(fit.amean.clone().ok_or_else(|| {
            EdgeErrors::InvalidArgument(
                "trending the prior against the average expression needs `amean` on the fit; \
                 `voom` returns it as `VoomResult::amean`"
                    .to_string(),
            )
        })?),
        EBayesTrend::Covariate(v) => {
            if v.len() != n_genes {
                return Err(EdgeErrors::LengthMismatch {
                    name: "trend covariate",
                    expected: n_genes,
                    got: v.len(),
                });
            }
            Some(v.clone())
        }
    };

    // -- the variance prior --
    let var: Vec<f64> = fit.sigma.iter().map(|s| s * s).collect();
    let squeezed = squeeze_var(
        &var,
        &fit.df_residual,
        covariate.as_deref(),
        Some(SqueezeVarParams {
            robust: params.robust,
            winsor_tail_p: params.winsor_tail_p,
            span: params.span,
            legacy: params.legacy,
        }),
    )?;
    let s2_post = squeezed.var_post;
    let s2_prior = squeezed.var_prior;
    let df_prior = squeezed.df_prior;

    // -- moderated t, and the degrees of freedom it is read against --
    //
    // The pooled cap matters on small designs: without it a gene can be handed
    // more degrees of freedom than the whole experiment has, see
    // `R/ebayes.R:62-64`.
    let df_pooled: f64 = fit.df_residual.iter().filter(|v| v.is_finite()).sum();
    let df_total: Vec<f64> = (0..n_genes)
        .map(|g| (fit.df_residual[g] + recycled(&df_prior, g)).min(df_pooled))
        .collect();

    let mut t = vec![0.0; n_genes * n_coef];
    let mut p_value = vec![0.0; n_genes * n_coef];
    t.par_chunks_mut(n_coef)
        .zip(p_value.par_chunks_mut(n_coef))
        .enumerate()
        .try_for_each(|(g, (t_row, p_row))| -> Result<(), EdgeErrors> {
            let scale = s2_post[g].sqrt();
            let df = df_total[g];
            for j in 0..n_coef {
                let k = g * n_coef + j;
                let stat = fit.coefficients[k] / fit.stdev_unscaled[k] / scale;
                t_row[j] = stat;
                p_row[j] = if stat.is_finite() {
                    2.0 * t_cdf(-stat.abs(), df)?
                } else {
                    f64::NAN
                };
            }
            Ok(())
        })?;

    // -- prior coefficient variance for the B-statistic --
    let median_prior = median(&s2_prior);
    let limits = (
        params.stdev_coef_lim.0.powi(2) / median_prior,
        params.stdev_coef_lim.1.powi(2) / median_prior,
    );
    let mut var_prior = tmixture_matrix(
        &t,
        &fit.stdev_unscaled,
        &df_total,
        n_genes,
        n_coef,
        params.proportion,
        limits,
    )?;

    let mut failures = 0usize;
    for v in var_prior.iter_mut() {
        if v.is_nan() {
            *v = 1.0 / recycled(&s2_prior, failures.min(s2_prior.len() - 1));
            failures += 1;
        }
    }

    let lods = log_odds(
        &t,
        &fit.stdev_unscaled,
        &df_total,
        &df_prior,
        &var_prior,
        n_genes,
        n_coef,
        params.proportion,
    );

    // -- moderated F, when every coefficient is estimable --
    let (f_stat, f_p_value) = if is_full_rank(&fit.design, fit.n_samples, fit.pivot.len())? {
        let df2: Vec<f64> = (0..n_genes)
            .map(|g| fit.df_residual[g] + recycled(&df_prior, g))
            .collect();
        let (stat, df1) = moderated_f(&t, &fit.cov_coefficients, n_genes, n_coef)?;
        let p = stat
            .iter()
            .zip(&df2)
            .map(|(&s, &d)| {
                if !s.is_finite() {
                    Ok(f64::NAN)
                } else if d.is_finite() {
                    f_sf(s, df1, d)
                } else {
                    chisq_sf(s * df1, df1)
                }
            })
            .collect::<Result<Vec<f64>, EdgeErrors>>()?;
        (Some(stat), Some(p))
    } else {
        (None, None)
    };

    fit.df_prior = Some(df_prior);
    fit.s2_prior = Some(s2_prior);
    fit.var_prior = Some(var_prior);
    fit.proportion = Some(params.proportion);
    fit.s2_post = Some(s2_post);
    fit.t = Some(t);
    fit.df_total = Some(df_total);
    fit.p_value = Some(p_value);
    fit.lods = Some(lods);
    fit.f_stat = f_stat;
    fit.f_p_value = f_p_value;

    Ok(fit)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::limma::lm_fit::lm_fit;
    use approx::assert_relative_eq;

    /// A four-sample, two-coefficient fit over six genes.
    fn small_fit() -> MArrayLm {
        let design = vec![
            1.0, 0.0, //
            1.0, 0.0, //
            1.0, 1.0, //
            1.0, 1.0,
        ];
        let y: Vec<f64> = (0..24).map(|i| ((i * 13) % 29) as f64 / 8.0).collect();
        let amean: Vec<f64> = y
            .chunks_exact(4)
            .map(|r| r.iter().sum::<f64>() / 4.0)
            .collect();
        let fit = lm_fit(&y, 6, 4, &design, 2, None, None, None).unwrap();
        MArrayLm::from_lm_fit(fit, &design, 2, 4, Some(amean)).unwrap()
    }

    #[test]
    fn test_ebayes_fills_every_field() {
        let out = ebayes(small_fit(), None).unwrap();
        assert_eq!(out.t.as_ref().unwrap().len(), 12);
        assert_eq!(out.p_value.as_ref().unwrap().len(), 12);
        assert_eq!(out.lods.as_ref().unwrap().len(), 12);
        assert_eq!(out.df_total.as_ref().unwrap().len(), 6);
        assert_eq!(out.s2_post.as_ref().unwrap().len(), 6);
        assert_eq!(out.var_prior.as_ref().unwrap().len(), 2);
        assert!(out.f_stat.is_some());
        assert!(out.f_p_value.is_some());
    }

    #[test]
    fn test_moderated_t_is_the_coefficient_over_its_moderated_error() {
        let out = ebayes(small_fit(), None).unwrap();
        let t = out.t.as_ref().unwrap();
        let s2 = out.s2_post.as_ref().unwrap();
        for (g, &v) in s2.iter().enumerate() {
            let scale = v.sqrt();
            for j in 0..2 {
                let k = g * 2 + j;
                let want = out.coefficients[k] / out.stdev_unscaled[k] / scale;
                assert_relative_eq!(t[k], want, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_df_total_is_capped_at_the_pooled_degrees_of_freedom() {
        let out = ebayes(small_fit(), None).unwrap();
        let pooled: f64 = out.df_residual.iter().sum();
        for &d in out.df_total.as_ref().unwrap() {
            assert!(d <= pooled + 1e-12, "{d} exceeds the pooled {pooled}");
        }
    }

    #[test]
    fn test_p_value_is_the_two_sided_t_tail() {
        let out = ebayes(small_fit(), None).unwrap();
        let t = out.t.as_ref().unwrap();
        let p = out.p_value.as_ref().unwrap();
        let df = out.df_total.as_ref().unwrap();
        for (g, &d) in df.iter().enumerate() {
            for j in 0..2 {
                let k = g * 2 + j;
                let want = 2.0 * t_cdf(-t[k].abs(), d).unwrap();
                assert_relative_eq!(p[k], want, epsilon = 1e-14);
            }
        }
    }

    #[test]
    fn test_moderated_f_on_one_coefficient_is_t_squared() {
        let t = vec![2.0, -3.0, 0.5];
        let (f, df1) = moderated_f(&t, &[1.0], 3, 1).unwrap();
        assert_eq!(df1, 1.0);
        assert_relative_eq!(f[0], 4.0, epsilon = 1e-12);
        assert_relative_eq!(f[1], 9.0, epsilon = 1e-12);
    }

    #[test]
    fn test_moderated_f_on_uncorrelated_coefficients_is_the_mean_square() {
        // An identity correlation matrix gives Q = I / sqrt(2), so the
        // statistic is the mean of the squared t values.
        let t = vec![2.0, 4.0];
        let (f, df1) = moderated_f(&t, &[1.0, 0.0, 0.0, 1.0], 1, 2).unwrap();
        assert_eq!(df1, 2.0);
        assert_relative_eq!(f[0], (4.0 + 16.0) / 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_moderated_f_nudges_a_zero_variance() {
        // A zero on the diagonal would make cov2cor divide by zero.
        let t = vec![2.0, 4.0];
        let f = moderated_f(&t, &[1.0, 0.0, 0.0, 0.0], 1, 2);
        assert!(f.is_ok());
    }

    #[test]
    fn test_infinite_prior_takes_the_limiting_kernel() {
        // The log-ratio kernel tends to the limiting one as the degrees of
        // freedom grow, so both branches must give nearly the same log-odds on
        // a large `df_total`. They agree to about 1e-5 and no better, because
        // the log-ratio form is evaluating `ln(1 + tiny)` by then. That
        // degradation is why limma splits the branch at all.
        let t = vec![3.0];
        let u = vec![0.5];
        let dft = vec![1e12];
        let var_prior = vec![2.0];
        let finite = log_odds(&t, &u, &dft, &[1e5], &var_prior, 1, 1, 0.01);
        let infinite = log_odds(&t, &u, &dft, &[1e7], &var_prior, 1, 1, 0.01);
        assert_relative_eq!(finite[0], infinite[0], max_relative = 1e-4);
    }

    #[test]
    fn test_tmixture_returns_nan_when_no_gene_is_targeted() {
        // `ceiling` of any positive number is at least one, so the target count
        // only reaches zero once every statistic has been dropped as missing.
        let out = tmixture_vector(&[f64::NAN], &[1.0], &[3.0], 0.01, (0.0, 1e6)).unwrap();
        assert!(out.is_nan());
    }

    #[test]
    fn test_tmixture_skips_missing_statistics() {
        // The gene count driving the order statistics is the count of what
        // survived, not the input length.
        let t: Vec<f64> = (0..50)
            .map(|i| {
                if i % 5 == 0 {
                    f64::NAN
                } else {
                    i as f64 / 10.0
                }
            })
            .collect();
        let u = vec![1.0; 50];
        let df = vec![8.0; 50];
        let out = tmixture_vector(&t, &u, &df, 0.2, (0.0, 1e6)).unwrap();
        assert!(out.is_finite());
    }

    #[test]
    fn test_tmixture_clamps_to_its_limits() {
        let t: Vec<f64> = (0..100).map(|i| (i as f64) / 10.0).collect();
        let u = vec![1.0; 100];
        let df = vec![10.0; 100];
        let out = tmixture_vector(&t, &u, &df, 0.1, (0.25, 0.25)).unwrap();
        assert_relative_eq!(out, 0.25, epsilon = 1e-12);
    }

    #[test]
    fn test_ebayes_rejects_a_bad_proportion() {
        let p = EBayesParams {
            proportion: 1.0,
            ..Default::default()
        };
        assert!(ebayes(small_fit(), Some(p)).is_err());
    }

    #[test]
    fn test_ebayes_rejects_amean_trend_without_amean() {
        let mut fit = small_fit();
        fit.amean = None;
        let p = EBayesParams {
            trend: EBayesTrend::Amean,
            ..Default::default()
        };
        assert!(ebayes(fit, Some(p)).is_err());
    }

    #[test]
    fn test_ebayes_rejects_a_mismatched_covariate() {
        let p = EBayesParams {
            trend: EBayesTrend::Covariate(vec![0.0; 3]),
            ..Default::default()
        };
        assert!(ebayes(small_fit(), Some(p)).is_err());
    }
}
