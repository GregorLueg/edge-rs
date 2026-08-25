//! limma's `squeezeVar` and the `fitFDist` family underneath it.
//!
//! The empirical Bayes step that shrinks a genewise variance towards a fitted
//! prior. `estimateDisp` uses it to choose the prior degrees of freedom and
//! `glmQLFit` uses it for the quasi-likelihood dispersions, so it sits on the
//! path of essentially every analysis.
//!
//! The model is `s2_g ~ s0^2 * F(df_g, df0)`. All three fits estimate `s0^2`
//! and `df0` by matching moments of `z = log(s2)` rather than of `s2` itself,
//! because `log` of a scaled F has finite moments for any `df0` while the F
//! itself does not:
//!
//! ```text
//! E[z_g] = log(s0^2) - [log(df0/2) - psi(df0/2)] + [log(df_g/2) - psi(df_g/2)]
//! Var[z_g] = psi'(df0/2) + psi'(df_g/2)
//! ```
//!
//! So `e_g = z_g + logmdigamma(df_g/2)` has mean `log(s0^2) - logmdigamma(df0/2)`
//! and variance `trigamma(df0/2)` plus the known genewise term. Subtracting the
//! mean of `trigamma(df_g/2)` from the sample variance of `e` and inverting
//! `trigamma` closes the fit. [`fit_f_dist`] takes the mean of `e` as a single
//! number, [`fit_f_dist_trend`] as a natural cubic spline in a covariate, and
//! [`fit_f_dist_robustly`] replaces the moments with Winsorised ones so a
//! handful of wild genes cannot drag `df0` down for everyone.
//!
//! ### Which limma this is
//!
//! limma 3.66's `squeezeVar` dispatches on a `legacy` flag: `TRUE` gives
//! `fitFDist`/`fitFDistRobustly`, `FALSE` gives `fitFDistUnequalDF1`, and the
//! default picks `FALSE` when the residual degrees of freedom differ between
//! genes. Both branches are here, with the same dispatch.
//!
//! [`fit_f_dist_unequal_df1`] is the non-legacy branch and is a different
//! estimator, not a variant of the moment fits: it maximises the marginal
//! likelihood over `df0` with the scale profiled out, so each gene may carry
//! its own `df_g`. It is also the branch most real data takes, because
//! `glmQLFit` produces unequal residual degrees of freedom as soon as a gene
//! has structural zeros.
//!
//! ### Numeric policy
//!
//! `f64` throughout, per the crate policy for likelihood-adjacent code: the fit
//! is a difference of logs closed by a Newton iteration on `trigamma`, and there
//! is nothing here whose memory cost would justify `f32`.
//!
//! ### References
//!
//! Smyth, Statistical Applications in Genetics and Molecular Biology, 2004
//!
//! Phipson, Lee, Majewski, Alexander and Smyth, Annals of Applied Statistics,
//! 2016 (the robust fit)

use faer::MatRef;
use faer::linalg::solvers::SolveLstsq;
use rayon::prelude::*;

use crate::errors::EdgeErrors;
use crate::limma::lowess::lowess;
use crate::limma::lowess::{LowessParams, weighted_lowess};
use crate::numeric::dist::{beta_cdf, beta_ppf, beta_sf, chisq_sf, f_ppf, f_sf, gamma_ppf};
use crate::numeric::gamma::{ln_gamma, logmdigamma, trigamma, trigamma_inverse};
use crate::numeric::interpolate::{interp_linear_extrap, natural_spline_basis};
use crate::numeric::optimise::{OPTIMIZE_TOL, brent_fmin, brentq};
use crate::numeric::stats::{median, p_adjust_bh, quantile_type7, rank_average, trimmed_mean};
use crate::utils::design::{LIMMA_LOWESS_DEFAULTS, choose_lowess_span};

/// Fewer genes than this and limma refuses to fit anything, returning the input
/// variances unmoderated with a prior of zero degrees of freedom.
const MIN_GENES: usize = 3;

/// Smallest residual degrees of freedom `fitFDist` will treat as informative.
const DF1_TOL: f64 = 1e-15;

/// Variances below this are treated as negative and dropped by `fitFDist`.
///
/// limma tests `x > -1e-15` rather than `x >= 0` so that a variance that came
/// back as a tiny negative number from a floating point cancellation is kept
/// and then clamped, rather than silently costing a gene.
const VAR_TOL: f64 = -1e-15;

/// Fraction of the median variance that exact zeros are offset to.
///
/// `log(0)` is the problem; limma lifts every variance to `1e-5 * median`
/// before taking logs and warns that the eBayes step is unreliable when more
/// than half of them needed it.
const ZERO_VAR_OFFSET: f64 = 1e-5;

/// Smallest residual degrees of freedom `fitFDistRobustly` will keep.
///
/// Looser than [`DF1_TOL`] because the robust fit needs enough genes with real
/// residual degrees of freedom to estimate a tail from.
const ROBUST_DF1_TOL: f64 = 1e-6;

/// Fraction of the median variance that the robust fit floors variances at.
///
/// Tighter than [`ZERO_VAR_OFFSET`] because the robust fit only needs to keep
/// `log(x)` finite, not to keep the moment estimate sane.
const ROBUST_VAR_FLOOR: f64 = 1e-12;

/// Absolute slack below the largest `df1` within which degrees of freedom count
/// as equal, so the unequal-`df1` unification is skipped.
const DF1_TIE_TOL: f64 = 1e-14;

/// Number of Gauss-Legendre nodes in the Winsorised moment integrals.
///
/// limma calls `statmod::gauss.quad.prob(128, "uniform")`. The integrand is a
/// smooth F density on a bounded, link-transformed interval, so 128 nodes is far
/// past convergence; the count is here to match limma bit for bit, not because
/// the integral needs it.
const QUAD_NODES: usize = 128;

/// Newton iteration budget when solving for the Legendre nodes.
const QUAD_NEWTON_MAX_ITER: usize = 100;

/// Span of the lowess trend the robust fit runs on `log(var)`, hardcoded in
/// limma's `fitFDistRobustly` as `loessFit(z, covariate, span = 0.4)`.
const ROBUST_LOWESS_SPAN: f64 = 0.4;

/// Robustness iterations for that lowess.
///
/// `loessFit` defaults to `iterations = 4` and passes `iter = iterations - 1`
/// to `stats::lowess`, which counts the robustness passes after the first fit.
const ROBUST_LOWESS_STEPS: usize = 3;

/// Absolute tolerance for the `df2` root search, in the `d/(1+d)` link scale.
///
/// limma passes `tol = 1e-8` to `uniroot`. This is tighter deliberately: the
/// link squashes large `df2` into the top of `(0, 1)`, so a `1e-8` bracket in
/// the link is worth `1e-4` on a `df2` of 100. Converging properly and letting
/// limma be the one carrying truncation error is the better of the two errors.
const ROOT_XTOL: f64 = 1e-15;

/// Relative tolerance for the `df2` root search.
const ROOT_RTOL: f64 = 8.0 * f64::EPSILON;

/// Iteration budget for the `df2` root search.
const ROOT_MAX_ITER: usize = 200;

/// Relative tolerance on the QR diagonal used to rank the spline basis, matching
/// the default `tol` of R's `lm.fit`.
const RANK_TOL: f64 = 1e-7;

/// Gene count above which the per-gene tail probabilities fan out over rayon.
///
/// Each one is a regularised incomplete beta, a few hundred flops, so the fork
/// only pays for itself once there are thousands of genes. Everything else in
/// this module is either a reduction or a spline solve on a handful of columns
/// and stays sequential.
const PARALLEL_THRESHOLD: usize = 4096;

/// `log(0.5)`, the target tail probability that defines `df2_outlier`.
const LN_HALF: f64 = -std::f64::consts::LN_2;

/// Smallest residual degrees of freedom `fitFDistUnequalDF1` will use.
///
/// Below this a gene is given a zero prior weight and its `df1` is reset to one,
/// rather than dropped, so it still gets a fitted trend value at the end.
const UNEQUAL_DF1_TOL: f64 = 0.01;

/// Fraction of the median informative variance that variances are floored at
/// before the logs are taken, `pmax(x, 1e-12 * m)` in limma.
///
/// Seven orders of magnitude tighter than [`ZERO_VAR_OFFSET`]: the maximum
/// likelihood fit only needs `log(x)` finite, whereas the moment fit needs the
/// floor not to distort the sample variance.
const UNEQUAL_ZERO_VAR_OFFSET: f64 = 1e-12;

/// `small.n` in the `chooseLowessSpan(n, small.n = 500)` that
/// `fitFDistUnequalDF1` uses, ten times limma's own default.
///
/// The span is therefore 1 for every gene set below 500 genes, and only starts
/// tapering towards `min.span` above that.
const UNEQUAL_LOWESS_SMALL_N: usize = 500;

/// Lower clamp on the lowess prior weights, `min.weight` in the `loessFit` call.
const LOESS_MIN_WEIGHT: f64 = 1e-8;

/// Upper clamp on the lowess prior weights, `max.weight` in the same call.
const LOESS_MAX_WEIGHT: f64 = 100.0;

/// Spread below which `loessFit` calls the clamped weights equal and discards
/// them, falling back to unweighted `stats::lowess`.
///
/// This is `equal.weights.as.null = TRUE`, and it is not a corner case: equal
/// `df1` with no prior weights makes every weight identical, so a trended
/// non-legacy fit on a balanced design takes the unweighted branch.
const EQUAL_WEIGHT_TOL: f64 = 1e-15;

/// Lower end of the `d2 / (1 + d2)` link interval the likelihood is maximised
/// over, so `df2 >= 2`.
const UNEQUAL_PAR_LOWER: f64 = 0.5;

/// Upper end of that interval, so `df2 <= 9998`.
const UNEQUAL_PAR_UPPER: f64 = 0.9998;

/// False discovery rate above which a gene is taken to be no evidence of an
/// outlier at all, so it re-enters the fit at full weight.
const ROBUST_FDR_CUTOFF: f64 = 0.3;

/// Left tail probability below which limma recomputes it from the lower tail of
/// the F rather than as `1 - RightP`, where the subtraction has cancelled.
const LEFT_TAIL_SWITCH: f64 = 0.001;

/// Relative tolerance on the weighted normal equations of the two-column fit
/// `loessFit` falls back to when there are too few points to smooth.
///
/// Stands in for the pivoted QR rank test in R's `lm.wfit`: below this the
/// covariate carries no weighted spread and the slope is dropped, leaving the
/// weighted mean.
const WLS_RANK_TOL: f64 = 1e-14;

/////////////////
// Public API  //
/////////////////

/// Tuning knobs for [`squeeze_var`].
///
/// The defaults are limma's: a non-robust fit with the Winsorising tail
/// proportions it would use if asked for a robust one.
#[derive(Clone, Copy, Debug)]
pub struct SqueezeVarParams {
    /// Whether to Winsorise the moments so outlier genes cannot pull the prior
    /// degrees of freedom down. Costs a quadrature and a root search, and turns
    /// `df_prior` into a per-gene vector.
    pub robust: bool,
    /// Proportions Winsorised off the lower and upper tails of the log
    /// variances. Only consulted when `robust` is set. Both must lie in
    /// `[0, 0.5)`.
    pub winsor_tail_p: (f64, f64),
    /// Span of the lowess trend fitted against the covariate. `None` lets each
    /// fit pick its own default: [`ROBUST_LOWESS_SPAN`] on the legacy robust
    /// path, `chooseLowessSpan(n, small.n = 500)` on the unequal-`df1` path.
    ///
    /// Setting it forces `legacy` off, which is what limma does and which is
    /// not only a dispatch detail: limma's `squeezeVar` never passes `span` to
    /// `fitFDistRobustly` at all, so the legacy robust span is only reachable
    /// with `legacy` set to `Some(true)` explicitly.
    pub span: Option<f64>,
    /// Which family of fits to use. `Some(true)` is `fitFDist` and
    /// `fitFDistRobustly`, `Some(false)` is `fitFDistUnequalDF1`, and `None` is
    /// limma's own rule: legacy exactly when every positive degrees of freedom
    /// is the same, which is where the two families agree best.
    pub legacy: Option<bool>,
}

impl Default for SqueezeVarParams {
    fn default() -> Self {
        Self {
            robust: false,
            winsor_tail_p: (0.05, 0.1),
            span: None,
            legacy: None,
        }
    }
}

/// What [`squeeze_var`] produces.
///
/// The two prior vectors are length one whenever the prior is a single number,
/// and length `n` when it varies by gene. `var_post` is always length `n`.
#[derive(Clone, Debug)]
pub struct SqueezeVarResult {
    /// Posterior variance per gene, the shrunken estimate callers actually use.
    pub var_post: Vec<f64>,
    /// Prior variance. Length one when the prior is a single number, which is
    /// any untrended fit except the legacy robust one, and one value per gene
    /// otherwise.
    pub var_prior: Vec<f64>,
    /// Prior degrees of freedom. Length one unless the fit produced a per-gene
    /// vector, which the legacy robust fit always does and the unequal-`df1`
    /// robust fit does whenever it finds an outlier.
    pub df_prior: Vec<f64>,
}

/// Empirical Bayes moderation of genewise variances.
///
/// Port of limma's `squeezeVar`. Fits a scaled F prior to the variances and
/// returns the posterior
/// `(df * var + df_prior * var_prior) / (df + df_prior)`, which is the
/// precision-weighted blend of each gene's own variance with the fitted prior.
/// An infinite `df_prior`, which the fit produces when the variances are close
/// enough to constant that there is no excess dispersion to explain, collapses
/// that blend onto the prior exactly.
///
/// Two families of fit sit behind it. `params.legacy` picks between them, and
/// its `None` default reproduces limma's rule: legacy when every positive `df`
/// is the same, non-legacy otherwise, with a supplied `span` forcing non-legacy
/// either way. Within the legacy family the arguments pick one of three:
///
/// * no covariate, `robust` clear - [`fit_f_dist`], a single prior for all genes
/// * covariate, `robust` clear - [`fit_f_dist_trend`], a spline in the covariate
/// * `robust` set - [`fit_f_dist_robustly`], with or without the covariate
///
/// The non-legacy family is one function, [`fit_f_dist_unequal_df1`], which
/// maximises the marginal likelihood directly instead of matching moments and
/// so does not need the residual degrees of freedom to be shared. `glmQLFit`
/// produces unequal degrees of freedom as soon as any gene has structural
/// zeros, which is why this is the path most real data takes.
///
/// Genes with `df == 0` carry no information. limma zeroes their variance
/// first, drops them from the fit, and hands them back the prior; this does the
/// same, but only when `df` is a per-gene vector, exactly as limma does.
///
/// ### Params
///
/// * `var` - Genewise variances, non-negative and finite
/// * `df` - Residual degrees of freedom, either one value shared by every gene
///   or one per gene. Non-negative and finite; zero is allowed and means the
///   gene is uninformative.
/// * `covariate` - Covariate for a trended prior, one per gene, or `None`
/// * `params` - Tuning knobs, or `None` for [`SqueezeVarParams::default`]
///
/// ### Returns
///
/// The posterior variances and the prior that produced them, or [`EdgeErrors`]
/// if `var` is empty, `df` or `covariate` is the wrong length, a variance is
/// negative or not finite, a degrees of freedom is negative or not finite, a
/// covariate is `NaN`, or a Winsorising proportion is outside `[0, 0.5)`.
///
/// ### References
///
/// Smyth, Statistical Applications in Genetics and Molecular Biology, 2004
pub fn squeeze_var(
    var: &[f64],
    df: &[f64],
    covariate: Option<&[f64]>,
    params: Option<SqueezeVarParams>,
) -> Result<SqueezeVarResult, EdgeErrors> {
    let params = params.unwrap_or_default();
    let n = var.len();

    if n == 0 {
        return Err(EdgeErrors::InvalidArgument(
            "squeeze_var was given an empty variance vector.".to_string(),
        ));
    }
    if df.len() != 1 && df.len() != n {
        return Err(EdgeErrors::LengthMismatch {
            name: "df",
            expected: n,
            got: df.len(),
        });
    }
    if let Some(cov) = covariate
        && cov.len() != n
    {
        return Err(EdgeErrors::LengthMismatch {
            name: "covariate",
            expected: n,
            got: cov.len(),
        });
    }
    check_winsor_tail_p(params.winsor_tail_p)?;
    if let Some(s) = params.span
        && !(s > 0.0 && s <= 1.0)
    {
        return Err(EdgeErrors::InvalidArgument(format!(
            "squeeze_var span must lie in (0, 1]; got {s}."
        )));
    }
    for (i, &d) in df.iter().enumerate() {
        if !d.is_finite() || d < 0.0 {
            return Err(EdgeErrors::InvalidArgument(format!(
                "squeeze_var needs finite non-negative degrees of freedom; df[{i}] is {d}."
            )));
        }
    }
    if let Some(cov) = covariate
        && let Some(i) = cov.iter().position(|v| v.is_nan())
    {
        return Err(EdgeErrors::InvalidArgument(format!(
            "squeeze_var covariate must not contain NaN; covariate[{i}] is NaN."
        )));
    }

    // limma zeroes the variance of every uninformative gene before doing
    // anything else, and only when df is genewise. That is what makes an
    // infinite or missing variance at df == 0 harmless, so it has to happen
    // before the finiteness check below rather than after it.
    let mut var = var.to_vec();
    if df.len() > 1 {
        for (v, &d) in var.iter_mut().zip(df) {
            if d == 0.0 {
                *v = 0.0;
            }
        }
    }
    for (i, &v) in var.iter().enumerate() {
        if !v.is_finite() || v < 0.0 {
            return Err(EdgeErrors::InvalidArgument(format!(
                "squeeze_var needs finite non-negative variances; var[{i}] is {v}."
            )));
        }
    }

    if n < MIN_GENES {
        return Ok(SqueezeVarResult {
            var_post: var.clone(),
            var_prior: var,
            df_prior: vec![0.0],
        });
    }

    // limma's dispatch, in its order: a supplied span overrides an explicit
    // legacy = TRUE, and only then does the automatic rule get a look in.
    let legacy = if params.span.is_some() {
        false
    } else {
        params.legacy.unwrap_or_else(|| all_positive_df_equal(df))
    };

    let (var_prior, df_prior) = if !legacy {
        let fit = fit_f_dist_unequal_df1(&var, df, covariate, params.span, params.robust, None)?;
        let df_prior = fit.df2_shrunk.unwrap_or_else(|| vec![fit.df2]);
        if df_prior.iter().any(|v| v.is_nan()) {
            return Err(EdgeErrors::InvalidArgument(
                "squeeze_var could not estimate the prior degrees of freedom: fewer than two \
                 genes carry a positive variance and a positive weight."
                    .to_string(),
            ));
        }
        (fit.scale, df_prior)
    } else if params.robust {
        let (scale, df2_shrunk) =
            robust_fit(&var, df, covariate, params.winsor_tail_p, params.span)?;
        (recycle(&scale, n), df2_shrunk)
    } else if let Some(cov) = covariate {
        let (scale, df2) = fit_f_dist_trend(&var, df, cov, params.span)?;
        (recycle(&scale, n), vec![df2])
    } else {
        let (scale, df2) = fit_f_dist(&var, df)?;
        (vec![scale], vec![df2])
    };

    let var_post = posterior_var(&var, df, &var_prior, &df_prior);
    Ok(SqueezeVarResult {
        var_post,
        var_prior,
        df_prior,
    })
}

/// Fits a scaled F distribution to a set of variances by moment matching.
///
/// Port of limma's `fitFDist` without a covariate. The sample mean and variance
/// of `e_g = log(var_g) + logmdigamma(df_g / 2)` identify the prior: the
/// variance of `e` in excess of `mean(trigamma(df_g / 2))` is `trigamma(df0/2)`,
/// so `df0 = 2 * trigammaInverse(excess)`, and the scale follows from the mean.
/// A non-positive excess means the variances are less dispersed than the
/// residual degrees of freedom alone would imply, which is reported as an
/// infinite `df2` and a scale equal to the mean variance.
///
/// Genes with a non-finite or negligible `df1`, or a non-finite or negative
/// variance, are dropped. Exact zeros among the survivors are lifted to
/// [`ZERO_VAR_OFFSET`] times the median so their logs stay finite.
///
/// ### Params
///
/// * `x` - Genewise variances
/// * `df1` - Residual degrees of freedom, either one value shared by every gene
///   or one per gene
///
/// ### Returns
///
/// The prior scale `s0^2` and the prior degrees of freedom `df0`, or
/// [`EdgeErrors`] if `x` is empty, `df1` is the wrong length, or nothing
/// survives the filter.
pub fn fit_f_dist(x: &[f64], df1: &[f64]) -> Result<(f64, f64), EdgeErrors> {
    let n = check_fit_lengths("fit_f_dist", x, df1)?;
    if n == 1 {
        return Ok((x[0], 0.0));
    }

    let ok = informative_mask(x, df1, DF1_TOL);
    let nok = ok.iter().filter(|&&k| k).count();
    if nok == 1 {
        let i = ok.iter().position(|&k| k).unwrap_or(0);
        return Ok((x[i], 0.0));
    }
    if nok == 0 {
        return Err(EdgeErrors::InvalidArgument(
            "fit_f_dist found no gene with a finite variance and positive degrees of freedom."
                .to_string(),
        ));
    }

    let x_ok: Vec<f64> = subset(x, &ok);
    let df_ok: Vec<f64> = subset_recycled(df1, &ok);
    let x_ok = offset_from_zero(&x_ok);

    let e: Vec<f64> = x_ok
        .iter()
        .zip(&df_ok)
        .map(|(&v, &d)| v.ln() + logmdigamma(0.5 * d))
        .collect();
    let emean = mean(&e);
    let evar = e.iter().map(|&v| (v - emean) * (v - emean)).sum::<f64>() / (nok - 1) as f64
        - mean_trigamma(&df_ok);

    if evar > 0.0 {
        let df2 = 2.0 * trigamma_inverse(evar);
        Ok(((emean - logmdigamma(0.5 * df2)).exp(), df2))
    } else {
        Ok((mean(&x_ok), f64::INFINITY))
    }
}

/// Fits a scaled F distribution whose scale follows a trend in a covariate.
///
/// Port of limma's `fitFDist` with a covariate. Identical to [`fit_f_dist`]
/// except that the mean of `e` is a natural cubic spline in the covariate
/// rather than a single number, so the prior variance varies by gene while the
/// prior degrees of freedom stay shared. The spline gets 2, 3 or 4 basis columns
/// depending on how many genes survive the filter, capped at the number of
/// distinct covariate values; below two it degenerates and the fit falls back to
/// [`fit_f_dist`].
///
/// Infinite covariate values are pushed one unit past the finite range, as
/// limma does, so a gene with an undefined average abundance still lands at the
/// correct end of the trend.
///
/// ### Deviation from limma
///
/// The spline basis is built over the surviving genes' covariates, as limma
/// does, so the fit itself and the prior of every surviving gene match limma to
/// rounding. limma then evaluates that spline at the dropped genes' covariates
/// with `predict.ns`; [`natural_spline_basis`] does not expose its knots, so the
/// trend is instead read off the fitted values with [`extend_trend`]. A dropped
/// gene therefore carries an interpolation error of order the local covariate
/// spacing squared, around a part in `1e3` on forty genes and far less on a real
/// gene set. Nothing is affected unless some `df1` is zero or some variance is
/// not finite, which is the only way a gene is dropped.
///
/// ### Params
///
/// * `var` - Genewise variances
/// * `df1` - Residual degrees of freedom, either one value shared by every gene
///   or one per gene
/// * `covariate` - Covariate, one per gene, typically the average log CPM
/// * `span` - Accepted for signature parity with the robust fit and otherwise
///   unused: limma's trended `fitFDist` is a spline whose complexity is fixed by
///   the gene count, and has no span to set. Validated to lie in `(0, 1]`.
///
/// ### Returns
///
/// The per-gene prior scale and the shared prior degrees of freedom. The scale
/// holds one value per gene, except when a single gene survives the filter and
/// there is no trend to fit, where it collapses to one value for all of them.
/// Errors as [`fit_f_dist`], plus [`EdgeErrors::LengthMismatch`] if `covariate`
/// is the wrong length.
pub fn fit_f_dist_trend(
    var: &[f64],
    df1: &[f64],
    covariate: &[f64],
    span: Option<f64>,
) -> Result<(Vec<f64>, f64), EdgeErrors> {
    let n = check_fit_lengths("fit_f_dist_trend", var, df1)?;
    if covariate.len() != n {
        return Err(EdgeErrors::LengthMismatch {
            name: "covariate",
            expected: n,
            got: covariate.len(),
        });
    }
    if let Some(s) = span
        && !(s > 0.0 && s <= 1.0)
    {
        return Err(EdgeErrors::InvalidArgument(format!(
            "fit_f_dist_trend span must lie in (0, 1]; got {s}."
        )));
    }
    if n == 1 {
        return Ok((vec![var[0]], 0.0));
    }

    let covariate = finitise_covariate(covariate)?;
    let ok = informative_mask(var, df1, DF1_TOL);
    let nok = ok.iter().filter(|&&k| k).count();
    if nok == 1 {
        let i = ok.iter().position(|&k| k).unwrap_or(0);
        return Ok((vec![var[i]], 0.0));
    }
    if nok == 0 {
        return Err(EdgeErrors::InvalidArgument(
            "fit_f_dist_trend found no gene with a finite variance and positive degrees of freedom."
                .to_string(),
        ));
    }

    let cov_ok: Vec<f64> = subset(&covariate, &ok);
    let spline_df = spline_columns(nok, &cov_ok);
    if spline_df < MIN_SPLINE_DF {
        let x_ok: Vec<f64> = subset(var, &ok);
        let df_ok: Vec<f64> = subset_recycled(df1, &ok);
        let (scale, df2) = fit_f_dist(&x_ok, &df_ok)?;
        return Ok((vec![scale; n], df2));
    }

    let x_ok = offset_from_zero(&subset(var, &ok));
    let df_ok: Vec<f64> = subset_recycled(df1, &ok);
    let e: Vec<f64> = x_ok
        .iter()
        .zip(&df_ok)
        .map(|(&v, &d)| v.ln() + logmdigamma(0.5 * d))
        .collect();

    let (basis, ncol) = natural_spline_basis(&cov_ok, spline_df)?;
    let (fitted_ok, rss, rank) = spline_fit(&basis, ncol, &e)?;
    let emean = extend_trend(&covariate, &ok, &cov_ok, &fitted_ok)?;

    let evar = if nok > rank {
        rss / (nok - rank) as f64
    } else {
        0.0
    } - mean_trigamma(&df_ok);

    if evar > 0.0 {
        let df2 = 2.0 * trigamma_inverse(evar);
        let shift = logmdigamma(0.5 * df2);
        Ok((emean.iter().map(|&m| (m - shift).exp()).collect(), df2))
    } else {
        Ok((emean.iter().map(|&m| m.exp()).collect(), f64::INFINITY))
    }
}

/// Fits a scaled F distribution with Winsorised moments.
///
/// Port of limma's `fitFDistRobustly`. The moment estimate in [`fit_f_dist`] is
/// not robust: one gene with a wildly inflated variance inflates the sample
/// variance of `e`, which drives `df0` down and weakens the moderation for every
/// gene. This replaces the sample mean and variance of the log residuals with
/// Winsorised ones, clipped at the `winsor_tail_p` quantiles, and solves for the
/// `df2` whose Winsorised moments match. The theoretical Winsorised moments have
/// no closed form and come from a 128-node Gauss-Legendre rule over the F
/// density, mapped onto `(0, 1)` by `d -> d / (1 + d)` so the tail is finite.
///
/// Each gene then gets its own `df2`. A gene whose F statistic is further into
/// the tail than its rank among the genes says it should be is downweighted
/// towards `df2_outlier`, the degrees of freedom at which that observation would
/// be unremarkable, and the resulting vector is made monotone in the tail
/// probability so a more extreme gene never ends up with more prior support than
/// a less extreme one.
///
/// ### Params
///
/// * `x` - Genewise variances
/// * `df1` - Residual degrees of freedom, either one value shared by every gene
///   or one per gene
/// * `covariate` - Covariate for a trended prior, one per gene, or `None`. Must
///   be finite when supplied, as limma requires.
/// * `winsor_tail_p` - Proportions Winsorised off the lower and upper tails,
///   each in `[0, 0.5)`. When both are below `1 / n` there is nothing to clip
///   and the non-robust fit is returned unchanged.
///
/// ### Returns
///
/// The prior scale, length one when untrended and one value per gene when
/// trended, and the per-gene prior degrees of freedom. Errors as [`fit_f_dist`],
/// plus [`EdgeErrors::InvalidArgument`] if fewer than two genes are supplied,
/// the covariate is not finite, a Winsorising proportion is outside `[0, 0.5)`,
/// or more than half the variances are non-positive.
///
/// ### References
///
/// Phipson, Lee, Majewski, Alexander and Smyth, Annals of Applied Statistics,
/// 2016
pub fn fit_f_dist_robustly(
    x: &[f64],
    df1: &[f64],
    covariate: Option<&[f64]>,
    winsor_tail_p: (f64, f64),
) -> Result<(Vec<f64>, Vec<f64>), EdgeErrors> {
    robust_fit(x, df1, covariate, winsor_tail_p, None)
}

/// What [`fit_f_dist_unequal_df1`] produces.
///
/// Mirrors the list limma returns, including the fact that the two robust
/// fields are only present when the robust branch actually found an outlier.
#[derive(Clone, Debug)]
pub struct UnequalDf1Fit {
    /// Prior scale `s0^2`. Length one without a covariate, one per gene with
    /// one. A single `NaN` means the fit gave up for want of informative genes.
    pub scale: Vec<f64>,
    /// Shared prior degrees of freedom, `NaN` when the fit gave up.
    pub df2: f64,
    /// Degrees of freedom the most extreme gene is shrunk towards. `None`
    /// unless the robust branch ran to completion.
    pub df2_outlier: Option<f64>,
    /// Per-gene prior degrees of freedom. `None` unless the robust branch ran
    /// to completion, in which case callers should prefer it over `df2`.
    pub df2_shrunk: Option<Vec<f64>>,
}

/// Fits a scaled F prior by maximum likelihood, with per-gene `df1`.
///
/// Port of limma 3.66's `fitFDistUnequalDF1`. [`fit_f_dist`] matches the first
/// two moments of `e_g = log(s2_g) + logmdigamma(df_g / 2)`, which needs one
/// shared `df_g` to invert `trigamma` against; this instead maximises the
/// marginal log likelihood of the scaled F directly, so every gene may carry
/// its own residual degrees of freedom. `glmQLFit` hands out unequal degrees of
/// freedom as soon as a gene has structural zeros, so this is the usual path.
///
/// The likelihood is maximised over `par = d2 / (1 + d2)` on `[0.5, 0.9998]`,
/// which bounds `df2` to `[2, 9998]` and turns an unbounded search into a
/// bounded one. `s0^2` is profiled out rather than searched: at any `d2` the
/// maximising scale is `exp(emean - logmdigamma(d2))`, where `emean` is the
/// weight-weighted mean of `e` or, with a covariate, a lowess trend in it.
/// The weights are `1 / trigamma(df_g / 2)`, the inverse variance of `e_g`.
///
/// Genes are handled rather than dropped. A `NaN` variance or a `df1` below
/// [`UNEQUAL_DF1_TOL`] gets a zero prior weight and a placeholder `df1` of one,
/// so it contributes nothing to the fit yet still comes back with a trend
/// value. Variances are floored at [`UNEQUAL_ZERO_VAR_OFFSET`] times the median
/// informative variance so their logs stay finite.
///
/// With `robust` set the whole fit runs twice: once to get a working prior, and
/// again with the Benjamini-Hochberg adjusted two-sided F p-values as prior
/// weights, so genes that look like outliers are held out of the second fit.
/// Each gene then gets its own `df2`, interpolated between the fitted `df2` and
/// the much smaller `df2_outlier` at which the single most extreme gene would
/// be unremarkable, and the result is made monotone in the tail probability.
///
/// ### Precision limit
///
/// limma computes the outlier's tail probability with `pf(..., log.p = TRUE)`.
/// [`crate::numeric::dist`] has no log-scale F tail, so this takes
/// `f_sf(..).ln()` instead. The two agree to rounding until the tail underflows
/// below `1e-308`, which needs an F statistic tens of orders of magnitude past
/// anything a variance ratio produces; past that point `df2_outlier` here is
/// wrong and limma's is right.
///
/// ### Params
///
/// * `x` - Genewise variances, non-negative. `NaN` is read as missing.
/// * `df1` - Residual degrees of freedom, either one value shared by every gene
///   or one per gene
/// * `covariate` - Covariate for a trended prior, one per gene, or `None`
/// * `span` - Lowess span, or `None` for `chooseLowessSpan(n, small.n = 500)`.
///   Only consulted when a covariate is supplied.
/// * `robust` - Whether to run the outlier-downweighted second pass
/// * `prior_weights` - Non-negative weight per gene, or `None`. Used by the
///   robust recursion and exposed because limma exposes it.
///
/// ### Returns
///
/// The fit, or [`EdgeErrors`] if `x` is empty, `df1`, `covariate` or
/// `prior_weights` is the wrong length, a covariate or prior weight is `NaN`,
/// or a prior weight is negative. Fewer than two informative genes is not an
/// error: it comes back as a `NaN` scale and `df2`, exactly as limma does.
///
/// ### References
///
/// Smyth, Statistical Applications in Genetics and Molecular Biology, 2004
///
/// Phipson, Lee, Majewski, Alexander and Smyth, Annals of Applied Statistics,
/// 2016
pub fn fit_f_dist_unequal_df1(
    x: &[f64],
    df1: &[f64],
    covariate: Option<&[f64]>,
    span: Option<f64>,
    robust: bool,
    prior_weights: Option<&[f64]>,
) -> Result<UnequalDf1Fit, EdgeErrors> {
    let n = check_fit_lengths("fit_f_dist_unequal_df1", x, df1)?;
    if let Some(cov) = covariate {
        if cov.len() != n {
            return Err(EdgeErrors::LengthMismatch {
                name: "covariate",
                expected: n,
                got: cov.len(),
            });
        }
        if let Some(i) = cov.iter().position(|v| v.is_nan()) {
            return Err(EdgeErrors::InvalidArgument(format!(
                "fit_f_dist_unequal_df1 covariate must not contain NaN; covariate[{i}] is NaN."
            )));
        }
    }
    if let Some(pw) = prior_weights {
        if pw.len() != n {
            return Err(EdgeErrors::LengthMismatch {
                name: "prior_weights",
                expected: n,
                got: pw.len(),
            });
        }
        if let Some(i) = pw.iter().position(|v| v.is_nan() || *v < 0.0) {
            return Err(EdgeErrors::InvalidArgument(format!(
                "fit_f_dist_unequal_df1 prior weights must be non-negative and not NaN; \
                 prior_weights[{i}] is {}.",
                pw[i]
            )));
        }
    }
    if let Some(d) = df1.iter().find(|v| v.is_nan()) {
        return Err(EdgeErrors::InvalidArgument(format!(
            "fit_f_dist_unequal_df1 degrees of freedom must not be NaN; found {d}."
        )));
    }

    unequal_df1_inner(
        x,
        &recycle(df1, n),
        covariate,
        span,
        robust,
        prior_weights.map(|p| p.to_vec()),
    )
}

///////////////////////
// Posterior blend   //
///////////////////////

/// Blends each gene's variance with the prior, limma's internal `.squeezeVar`.
///
/// `(df * var + df_prior * var_prior) / (df + df_prior)` wherever every
/// `df_prior` is finite. An infinite prior sends a gene to `var_prior` exactly,
/// which is why the infinite case is a separate branch rather than a division
/// that happens to converge: the finite genes still need the blend, and
/// `inf * var_prior / inf` would be `NaN`.
///
/// ### Params
///
/// * `var` - Genewise variances
/// * `df` - Residual degrees of freedom, length one or one per gene
/// * `var_prior` - Prior variance, length one or one per gene
/// * `df_prior` - Prior degrees of freedom, length one or one per gene
///
/// ### Returns
///
/// One posterior variance per gene.
fn posterior_var(var: &[f64], df: &[f64], var_prior: &[f64], df_prior: &[f64]) -> Vec<f64> {
    let n = var.len();
    let largest = df_prior.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    if largest.is_finite() {
        return (0..n)
            .map(|i| {
                let d = at(df, i);
                let dp = at(df_prior, i);
                (d * var[i] + dp * at(var_prior, i)) / (d + dp)
            })
            .collect();
    }

    let mut out: Vec<f64> = (0..n).map(|i| at(var_prior, i)).collect();
    // limma treats anything past 1e100 as infinite here, so a prior that is
    // merely enormous still collapses the blend rather than producing a
    // posterior that differs from the prior in the last bits.
    let smallest = df_prior.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    if smallest > 1e100 {
        return out;
    }
    for i in 0..n {
        let dp = at(df_prior, i);
        if dp.is_finite() {
            let d = at(df, i);
            out[i] = (d * var[i] + dp * out[i]) / (d + dp);
        }
    }
    out
}

///////////////////
// Robust fit    //
///////////////////

/// Everything `fitFDistRobustly` returns that the recursion needs.
#[derive(Clone, Debug)]
struct RobustFit {
    /// Prior scale, length one when untrended and one per gene when trended.
    scale: Vec<f64>,
    /// The shared prior degrees of freedom before the per-gene shrinkage.
    df2: f64,
    /// Per-gene prior degrees of freedom.
    df2_shrunk: Vec<f64>,
}

/// [`fit_f_dist_robustly`] with the lowess span exposed.
///
/// ### Params
///
/// * `x` - Genewise variances
/// * `df1` - Residual degrees of freedom, length one or one per gene
/// * `covariate` - Covariate, one per gene, or `None`
/// * `winsor_tail_p` - Lower and upper Winsorising proportions
/// * `span` - Lowess span, or `None` for [`ROBUST_LOWESS_SPAN`]
///
/// ### Returns
///
/// The prior scale and the per-gene prior degrees of freedom.
fn robust_fit(
    x: &[f64],
    df1: &[f64],
    covariate: Option<&[f64]>,
    winsor_tail_p: (f64, f64),
    span: Option<f64>,
) -> Result<(Vec<f64>, Vec<f64>), EdgeErrors> {
    check_winsor_tail_p(winsor_tail_p)?;
    let fit = robust_fit_inner(x, df1, covariate, winsor_tail_p, span)?;
    Ok((fit.scale, fit.df2_shrunk))
}

/// The body of `fitFDistRobustly`, recursing once when some genes are dropped.
///
/// ### Params
///
/// * `x` - Genewise variances
/// * `df1` - Residual degrees of freedom, length one or one per gene
/// * `covariate` - Covariate, one per gene, or `None`
/// * `winsor_tail_p` - Lower and upper Winsorising proportions
/// * `span` - Lowess span, or `None` for [`ROBUST_LOWESS_SPAN`]
///
/// ### Returns
///
/// The scale, the shared `df2` and the per-gene `df2`.
fn robust_fit_inner(
    x: &[f64],
    df1: &[f64],
    covariate: Option<&[f64]>,
    winsor_tail_p: (f64, f64),
    span: Option<f64>,
) -> Result<RobustFit, EdgeErrors> {
    let n = check_fit_lengths("fit_f_dist_robustly", x, df1)?;
    if n < 2 {
        return Err(EdgeErrors::InvalidArgument(
            "fit_f_dist_robustly needs at least two genes.".to_string(),
        ));
    }
    if let Some(cov) = covariate {
        if cov.len() != n {
            return Err(EdgeErrors::LengthMismatch {
                name: "covariate",
                expected: n,
                got: cov.len(),
            });
        }
        if let Some(i) = cov.iter().position(|v| !v.is_finite()) {
            return Err(EdgeErrors::InvalidArgument(format!(
                "fit_f_dist_robustly needs a finite covariate; covariate[{i}] is {}.",
                cov[i]
            )));
        }
    }

    // Two genes cannot support a tail estimate, so limma hands the job to the
    // non-robust fit. limma then omits df2.shrunk entirely, which makes the
    // result unusable by its own caller; this fills it with the shared df2.
    if n == 2 {
        return Ok(match covariate {
            Some(cov) => {
                let (scale, df2) = fit_f_dist_trend(x, df1, cov, span)?;
                RobustFit {
                    scale,
                    df2,
                    df2_shrunk: vec![df2; n],
                }
            }
            None => {
                let (scale, df2) = fit_f_dist(x, df1)?;
                RobustFit {
                    scale: vec![scale],
                    df2,
                    df2_shrunk: vec![df2; n],
                }
            }
        });
    }

    let ok: Vec<bool> = (0..n)
        .map(|i| {
            let d = at(df1, i);
            !x[i].is_nan() && d.is_finite() && d > ROBUST_DF1_TOL
        })
        .collect();

    if ok.iter().any(|&k| !k) {
        return robust_fit_dropping(x, df1, covariate, winsor_tail_p, span, &ok);
    }

    let mut x = x.to_vec();
    let m = median(&x);
    if m <= 0.0 {
        return Err(EdgeErrors::InvalidArgument(
            "fit_f_dist_robustly needs a positive median variance; more than half are <= 0."
                .to_string(),
        ));
    }
    let floor = m * ROBUST_VAR_FLOOR;
    for v in &mut x {
        if *v < floor {
            *v = floor;
        }
    }

    let non_robust = match covariate {
        Some(cov) => {
            let (scale, df2) = fit_f_dist_trend(&x, df1, cov, span)?;
            (recycle(&scale, n), df2)
        }
        None => {
            let (scale, df2) = fit_f_dist(&x, df1)?;
            (vec![scale], df2)
        }
    };
    let (nr_scale, nr_df2) = non_robust;

    // Nothing would be clipped, so the Winsorised fit is the plain one.
    if winsor_tail_p.0 < 1.0 / n as f64 && winsor_tail_p.1 < 1.0 / n as f64 {
        return Ok(RobustFit {
            scale: nr_scale,
            df2: nr_df2,
            df2_shrunk: vec![nr_df2; n],
        });
    }

    let df1_val = unify_df1(&mut x, df1, &nr_scale, nr_df2)?;
    let z: Vec<f64> = x.iter().map(|v| v.ln()).collect();

    // Winsorised centre and spread of the log variances about their trend.
    let ztrend: Vec<f64> = match covariate {
        Some(cov) => lowess(
            cov,
            &z,
            span.unwrap_or(ROBUST_LOWESS_SPAN),
            ROBUST_LOWESS_STEPS,
        )?,
        None => vec![trimmed_mean(&z, winsor_tail_p.1)?],
    };
    let zresid: Vec<f64> = z
        .iter()
        .enumerate()
        .map(|(i, &v)| v - at(&ztrend, i))
        .collect();
    let lo = quantile_type7(&zresid, winsor_tail_p.0)?;
    let hi = quantile_type7(&zresid, 1.0 - winsor_tail_p.1)?;
    let zwins: Vec<f64> = zresid.iter().map(|&v| v.clamp(lo, hi)).collect();
    let zwmean = mean(&zwins);
    let zwvar = zwins
        .iter()
        .map(|&v| (v - zwmean) * (v - zwmean))
        .sum::<f64>()
        / (n - 1) as f64;

    let quad = gauss_legendre_unit(QUAD_NODES);
    let mom_inf = winsorized_moments(df1_val, f64::INFINITY, winsor_tail_p, &quad)?;
    let funval_inf = (zwvar / mom_inf.1).ln();

    if funval_inf <= 0.0 {
        return robust_infinite_df2(&z, &ztrend, zwmean, mom_inf.0, df1_val);
    }
    if nr_df2.is_infinite() {
        return Ok(RobustFit {
            scale: nr_scale,
            df2: nr_df2,
            df2_shrunk: vec![nr_df2; n],
        });
    }

    let mut objective = |par: f64| -> f64 {
        let d2 = link_inv(par);
        match winsorized_moments(df1_val, d2, winsor_tail_p, &quad) {
            Ok((_, v)) => (zwvar / v).ln(),
            Err(_) => funval_inf,
        }
    };
    let rbx = link_fun(nr_df2);
    let funval_low = objective(rbx);
    let df2 = if funval_low >= 0.0 {
        nr_df2
    } else {
        link_inv(brentq(
            &mut objective,
            rbx,
            1.0,
            ROOT_XTOL,
            ROOT_RTOL,
            ROOT_MAX_ITER,
        )?)
    };

    let (mom_mean, _) = winsorized_moments(df1_val, df2, winsor_tail_p, &quad)?;
    let corrected: Vec<f64> = (0..n).map(|i| at(&ztrend, i) + zwmean - mom_mean).collect();
    let scale: Vec<f64> = if covariate.is_some() {
        corrected.iter().map(|&v| v.exp()).collect()
    } else {
        vec![corrected[0].exp()]
    };
    let fstat: Vec<f64> = (0..n).map(|i| (z[i] - corrected[i]).exp()).collect();

    let log_tail_p = map_maybe_par(&fstat, |f| Ok(f_sf(f, df1_val, df2)?.ln()))?;
    let df2_shrunk = shrink_df2(&log_tail_p, &fstat, df1_val, df2)?;

    Ok(RobustFit {
        scale,
        df2,
        df2_shrunk,
    })
}

/// Handles the genes `fitFDistRobustly` cannot use, then recurses on the rest.
///
/// Dropped genes take the shared `df2` rather than a shrunken one, and, when the
/// fit is trended, a scale interpolated from the fitted trend on the log scale
/// and held constant beyond its ends, which is R's `approx(..., rule = 2)`.
///
/// ### Params
///
/// * `x` - Genewise variances
/// * `df1` - Residual degrees of freedom, length one or one per gene
/// * `covariate` - Covariate, one per gene, or `None`
/// * `winsor_tail_p` - Lower and upper Winsorising proportions
/// * `span` - Lowess span, or `None`
/// * `ok` - Which genes survive the filter
///
/// ### Returns
///
/// The fit, expanded back to the full gene set.
fn robust_fit_dropping(
    x: &[f64],
    df1: &[f64],
    covariate: Option<&[f64]>,
    winsor_tail_p: (f64, f64),
    span: Option<f64>,
    ok: &[bool],
) -> Result<RobustFit, EdgeErrors> {
    let n = x.len();
    let x_ok = subset(x, ok);
    let df_ok = subset_recycled(df1, ok);
    let cov_ok = covariate.map(|c| subset(c, ok));

    let fit = robust_fit_inner(&x_ok, &df_ok, cov_ok.as_deref(), winsor_tail_p, span)?;

    let mut df2_shrunk = vec![fit.df2; n];
    let mut k = 0;
    for i in 0..n {
        if ok[i] {
            df2_shrunk[i] = fit.df2_shrunk[k];
            k += 1;
        }
    }

    let scale = match (covariate, cov_ok) {
        (Some(cov), Some(cov_ok)) => {
            let log_scale: Vec<f64> = fit.scale.iter().map(|v| v.ln()).collect();
            let (knots, values) = collapse_ties(&cov_ok, &recycle(&log_scale, cov_ok.len()));
            let missing: Vec<f64> = (0..n).filter(|&i| !ok[i]).map(|i| cov[i]).collect();
            let filled = interp_linear_extrap(&missing, &knots, &values)?;
            let mut scale = vec![0.0; n];
            let (mut k, mut j) = (0, 0);
            for i in 0..n {
                if ok[i] {
                    scale[i] = at(&fit.scale, k);
                    k += 1;
                } else {
                    scale[i] = filled[j].exp();
                    j += 1;
                }
            }
            scale
        }
        _ => fit.scale,
    };

    Ok(RobustFit {
        scale,
        df2: fit.df2,
        df2_shrunk,
    })
}

/// The branch where the Winsorised spread is already at or below its value at
/// `df2 = Inf`, so no finite prior can explain it.
///
/// The prior degrees of freedom are infinite for every gene that is not an
/// outlier. Outliers are given `ProbNotOutlier * n * df1`, the pooled degrees of
/// freedom scaled by how unsurprising the gene is, and the result is made
/// monotone in the tail probability.
///
/// ### Params
///
/// * `z` - Log variances
/// * `ztrend` - Fitted trend in `z`, length one or one per gene
/// * `zwmean` - Winsorised mean of the residuals
/// * `mom_mean` - Theoretical Winsorised mean at `df2 = Inf`
/// * `df1_val` - The unified residual degrees of freedom
///
/// ### Returns
///
/// The fit, with `df2 = Inf`.
fn robust_infinite_df2(
    z: &[f64],
    ztrend: &[f64],
    zwmean: f64,
    mom_mean: f64,
    df1_val: f64,
) -> Result<RobustFit, EdgeErrors> {
    let n = z.len();
    let corrected: Vec<f64> = (0..n).map(|i| at(ztrend, i) + zwmean - mom_mean).collect();
    let scale: Vec<f64> = if ztrend.len() > 1 {
        corrected.iter().map(|&v| v.exp()).collect()
    } else {
        vec![corrected[0].exp()]
    };
    let fstat: Vec<f64> = (0..n).map(|i| (z[i] - corrected[i]).exp()).collect();

    // At df2 = Inf the F statistic is a chi-square on df1 degrees of freedom.
    let tail_p = map_maybe_par(&fstat, |f| chisq_sf(f * df1_val, df1_val))?;
    let ranks = rank_average(&fstat);
    let pooled = n as f64 * df1_val;

    let mut df2_shrunk = vec![f64::INFINITY; n];
    let mut any = false;
    for i in 0..n {
        let empirical = (n as f64 - ranks[i] + 0.5) / n as f64;
        let prob = (tail_p[i] / empirical).min(1.0);
        if prob < 1.0 {
            df2_shrunk[i] = prob * pooled;
            any = true;
        }
    }
    if any {
        let order = stable_order(&tail_p);
        cummax_in_order(&mut df2_shrunk, &order);
    }

    Ok(RobustFit {
        scale,
        df2: f64::INFINITY,
        df2_shrunk,
    })
}

/// Turns the shared `df2` into a per-gene one by discounting outliers.
///
/// A gene whose F statistic sits further into the tail than its rank predicts is
/// evidence against the fitted prior for that gene alone. `ProbNotOutlier` is
/// the ratio of its tail probability to the uniform one its rank implies, capped
/// at one, and each gene's prior is the mixture of `df2` and `df2_outlier`
/// weighted by it. `df2_outlier` is the degrees of freedom at which the most
/// extreme gene would sit at the median, found by one rescaling step.
///
/// The final `cummax` in tail-probability order, preceded by flattening the
/// leading run to its running mean, is limma's: without it the per-gene prior is
/// not monotone in the evidence, and a gene could end up better supported than a
/// less extreme neighbour.
///
/// ### Params
///
/// * `log_tail_p` - Log upper-tail probability of each gene's F statistic
/// * `fstat` - The F statistics themselves
/// * `df1_val` - The unified residual degrees of freedom
/// * `df2` - The shared prior degrees of freedom
///
/// ### Returns
///
/// One prior degrees of freedom per gene.
fn shrink_df2(
    log_tail_p: &[f64],
    fstat: &[f64],
    df1_val: f64,
    df2: f64,
) -> Result<Vec<f64>, EdgeErrors> {
    let n = fstat.len();
    let ranks = rank_average(fstat);
    let log_prob_not: Vec<f64> = (0..n)
        .map(|i| (log_tail_p[i] - ((n as f64 - ranks[i] + 0.5).ln() - (n as f64).ln())).min(0.0))
        .collect();

    if !log_prob_not.iter().any(|&v| v < 0.0) {
        return Ok(vec![df2; n]);
    }

    let min_log_tail = log_tail_p.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let mut df2_shrunk: Vec<f64> = if min_log_tail == f64::NEG_INFINITY {
        log_prob_not.iter().map(|&v| v.exp() * df2).collect()
    } else {
        let mut df2_outlier = LN_HALF / min_log_tail * df2;
        let fmax = fstat.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let new_log_tail = f_sf(fmax, df1_val, df2_outlier)?.ln();
        df2_outlier *= LN_HALF / new_log_tail;
        log_prob_not
            .iter()
            .map(|&v| v.exp() * df2 - v.exp_m1() * df2_outlier)
            .collect()
    };

    let order = stable_order(log_tail_p);
    flatten_then_cummax(&mut df2_shrunk, &order);

    Ok(df2_shrunk)
}

/// Rescales the variances of genes with fewer degrees of freedom than the rest.
///
/// The Winsorised moments are derived for a single `df1`, so a genewise `df1`
/// has to be collapsed first. limma maps each low-`df1` variance through its own
/// F distribution and back through the one at the largest `df1`, matching on
/// whichever tail is smaller so the transformation never runs through a
/// cancelled probability.
///
/// ### Params
///
/// * `x` - Variances, rescaled in place
/// * `df1` - Residual degrees of freedom, length one or one per gene
/// * `scale` - Non-robust prior scale, length one or one per gene
/// * `df2` - Non-robust prior degrees of freedom
///
/// ### Returns
///
/// The single `df1` the rescaled variances now correspond to.
fn unify_df1(x: &mut [f64], df1: &[f64], scale: &[f64], df2: f64) -> Result<f64, EdgeErrors> {
    if df1.len() == 1 {
        return Ok(df1[0]);
    }
    let df1max = df1.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let low: Vec<usize> = (0..x.len())
        .filter(|&i| df1[i] < df1max - DF1_TIE_TOL)
        .collect();
    if low.is_empty() {
        return Ok(df1[0]);
    }
    if !df2.is_finite() {
        // An infinite df2 leaves nothing to map through; limma would produce
        // NaN quantiles here, so keep the variances as they are.
        return Ok(df1max);
    }
    for &i in &low {
        let s = at(scale, i);
        let f = x[i] / s;
        let upper = f_sf(f, df1[i], df2)?;
        let lower = beta_cdf(df1[i] * f / (df1[i] * f + df2), 0.5 * df1[i], 0.5 * df2)?;
        let mapped = if upper < lower {
            f_isf(upper, df1max, df2)?
        } else {
            f_ppf(lower, df1max, df2)?
        };
        x[i] = mapped * s;
    }
    Ok(df1max)
}

//////////////////////////
// Unequal df1 fit      //
//////////////////////////

/// The body of `fitFDistUnequalDF1`, recursing once for the robust pass.
///
/// Split from [`fit_f_dist_unequal_df1`] so the recursion skips revalidating
/// inputs it has just built itself, and so `df1` arrives already recycled to
/// full length. Everything here is limma's, in limma's order.
///
/// ### Params
///
/// * `x` - Genewise variances, `NaN` read as missing
/// * `df1` - Residual degrees of freedom, one per gene
/// * `covariate` - Covariate, one per gene, or `None`
/// * `span` - Lowess span, or `None` for `chooseLowessSpan(n, small.n = 500)`
/// * `robust` - Whether to run the outlier-downweighted second pass
/// * `prior_weights` - Non-negative weight per gene, or `None`
///
/// ### Returns
///
/// The fit, or [`EdgeErrors`] out of the tail probabilities or the lowess.
fn unequal_df1_inner(
    x: &[f64],
    df1: &[f64],
    covariate: Option<&[f64]>,
    span: Option<f64>,
    robust: bool,
    prior_weights: Option<Vec<f64>>,
) -> Result<UnequalDf1Fit, EdgeErrors> {
    let n = x.len();
    let mut x = x.to_vec();
    let mut df1 = df1.to_vec();
    let mut prior_weights = prior_weights;
    let mut covariate = covariate;
    let mut robust = robust;

    // A missing variance is not dropped. It is zeroed and given no weight, so
    // the gene contributes nothing to the fit yet still comes back with a
    // trend value.
    if x.iter().any(|v| v.is_nan()) {
        let mask: Vec<bool> = x.iter().map(|v| v.is_nan()).collect();
        zero_out_weights(&mut prior_weights, &mask);
        for (v, &m) in x.iter_mut().zip(&mask) {
            if m {
                *v = 0.0;
            }
        }
    }
    // Same treatment for a gene with no usable residual degrees of freedom,
    // except that df1 is reset to one so trigamma and lgamma stay finite.
    if df1.iter().any(|&d| d < UNEQUAL_DF1_TOL) {
        let mask: Vec<bool> = df1.iter().map(|&d| d < UNEQUAL_DF1_TOL).collect();
        zero_out_weights(&mut prior_weights, &mask);
        for (d, &m) in df1.iter_mut().zip(&mask) {
            if m {
                *d = 1.0;
            }
        }
    }

    // limma latches `PriorWeights` here, after the two blocks above may have
    // created the weights, and never clears it again. That matters below.
    let weighted = prior_weights.is_some();

    let informative: Vec<bool> = (0..n)
        .map(|i| x[i] > 0.0 && prior_weights.as_ref().is_none_or(|p| p[i] != 0.0))
        .collect();
    let n_informative = informative.iter().filter(|&&k| k).count();
    if n_informative < 2 {
        return Ok(UnequalDf1Fit {
            scale: vec![f64::NAN],
            df2: f64::NAN,
            df2_outlier: None,
            df2_shrunk: None,
        });
    }
    if n_informative == 2 {
        covariate = None;
        robust = false;
        prior_weights = None;
    }

    let floor = UNEQUAL_ZERO_VAR_OFFSET * median(&subset(&x, &informative));
    let xpos: Vec<f64> = x.iter().map(|&v| v.max(floor)).collect();
    let d1: Vec<f64> = df1.iter().map(|&d| 0.5 * d).collect();
    let e: Vec<f64> = xpos
        .iter()
        .zip(&d1)
        .map(|(&v, &h)| v.ln() + logmdigamma(h))
        .collect();
    let w: Vec<f64> = match (weighted, prior_weights.as_ref()) {
        // `w * prior.weights` with the weights cleared by the n.informative == 2
        // branch but the flag still set. R's zero-length recycling empties the
        // vector rather than erroring, which sends every reduction below to
        // 0 / 0. Reproduced, not repaired.
        (true, None) => Vec::new(),
        (true, Some(p)) => d1
            .iter()
            .zip(p)
            .map(|(&h, &q)| 1.0 / trigamma(h) * q)
            .collect(),
        (false, _) => d1.iter().map(|&h| 1.0 / trigamma(h)).collect(),
    };

    let emean: Vec<f64> = match covariate {
        None => {
            let num: f64 = w.iter().zip(&e).map(|(&a, &b)| a * b).sum();
            let den: f64 = w.iter().sum();
            vec![num / den]
        }
        Some(cov) => {
            let span = span.unwrap_or_else(|| {
                let (_, min_span, power) = LIMMA_LOWESS_DEFAULTS;
                choose_lowess_span(n, UNEQUAL_LOWESS_SMALL_N, min_span, power)
            });
            loess_fit_weighted(&e, cov, &w, span)?
        }
    };

    let d1x: Vec<f64> = d1.iter().zip(&xpos).map(|(&h, &v)| h * v).collect();
    let objective = |par: f64| -> f64 {
        let d2 = par / (1.0 - par);
        let lmd2 = logmdigamma(d2);
        let lg_d2 = ln_gamma(d2);
        let term = |g: usize| -> f64 {
            let d2s20 = d2 * (at(&emean, g) - lmd2).exp();
            -(d1[g] + d2) * (d1x[g] / d2s20).ln_1p() - d1[g] * d2s20.ln() + ln_gamma(d1[g] + d2)
                - lg_d2
        };
        let total: f64 = match (weighted, prior_weights.as_ref()) {
            (true, None) => 0.0,
            (true, Some(p)) => (0..n).map(|g| p[g] * term(g)).sum(),
            (false, _) => (0..n).map(term).sum(),
        };
        -2.0 * total
    };

    let par = brent_fmin(
        UNEQUAL_PAR_LOWER,
        UNEQUAL_PAR_UPPER,
        objective,
        OPTIMIZE_TOL,
    );
    let d2 = par / (1.0 - par);
    let shift = logmdigamma(d2);
    let s20: Vec<f64> = emean.iter().map(|&m| (m - shift).exp()).collect();
    let mut df2 = 2.0 * d2;

    if !robust {
        return Ok(UnequalDf1Fit {
            scale: s20,
            df2,
            df2_outlier: None,
            df2_shrunk: None,
        });
    }

    // Two-sided F p-values against the prior just fitted. A gene whose variance
    // is far into either tail is evidence that the prior is being dragged, so
    // the second pass holds it out in proportion to its FDR.
    let f_stat: Vec<f64> = (0..n).map(|g| x[g] / at(&s20, g)).collect();
    // Elementwise, so the fork is exactly reproducible. The likelihood sum
    // above deliberately is not forked: a tree reduction would reassociate it
    // and move `df2` in the last few digits, which on a likelihood this flat is
    // visible in the answer.
    let right_p: Vec<f64> = if n >= PARALLEL_THRESHOLD {
        (0..n)
            .into_par_iter()
            .map(|g| f_sf(f_stat[g], df1[g], df2))
            .collect::<Result<_, _>>()?
    } else {
        (0..n)
            .map(|g| f_sf(f_stat[g], df1[g], df2))
            .collect::<Result<_, _>>()?
    };
    let mut left_p: Vec<f64> = right_p.iter().map(|&p| 1.0 - p).collect();
    if left_p.iter().copied().fold(f64::INFINITY, f64::min) < LEFT_TAIL_SWITCH {
        for g in 0..n {
            if left_p[g] < LEFT_TAIL_SWITCH {
                left_p[g] = f_cdf(f_stat[g], df1[g], df2)?;
            }
        }
    }
    let two_sided: Vec<f64> = (0..n).map(|g| 2.0 * left_p[g].min(right_p[g])).collect();
    let mut fdr = p_adjust_bh(&two_sided);
    for v in &mut fdr {
        if *v > ROBUST_FDR_CUTOFF {
            *v = 1.0;
        }
    }
    if fdr.iter().copied().fold(f64::INFINITY, f64::min) == 1.0 {
        return Ok(UnequalDf1Fit {
            scale: s20,
            df2,
            df2_outlier: None,
            df2_shrunk: None,
        });
    }

    // limma leaves `span` out of this call, so a caller-supplied span applies to
    // the first pass only and the second always uses chooseLowessSpan.
    let outpw = unequal_df1_inner(&x, &df1, covariate, None, false, Some(fdr))?;
    let scale = outpw.scale;
    df2 = outpw.df2;

    // How surprising each gene's tail probability is given its rank. A gene
    // sitting exactly where the uniform order statistics say it should gives 1
    // and keeps the full prior.
    let ranks = rank_average(&f_stat);
    let n_f = n as f64;
    let prob_not_outlier: Vec<f64> = (0..n)
        .map(|g| (right_p[g] / ((n_f - ranks[g] + 0.5) / n_f)).min(1.0))
        .collect();
    if prob_not_outlier
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min)
        == 1.0
    {
        return Ok(UnequalDf1Fit {
            scale,
            df2,
            df2_outlier: None,
            df2_shrunk: None,
        });
    }

    let imin = which_min(&right_p);
    let min_right_p = right_p[imin];
    let (df2_outlier, mut df2_shrunk) = if min_right_p == 0.0 {
        (
            0.0,
            prob_not_outlier
                .iter()
                .map(|&p| p * df2)
                .collect::<Vec<_>>(),
        )
    } else {
        // Two passes of the same idea: find the df2 at which the most extreme
        // gene would sit at the median of its own null, starting from a
        // log-linear guess and correcting once.
        let first = LN_HALF / min_right_p.ln() * df2;
        let new_log_right_p = f_sf(f_stat[imin], df1[imin], first)?.ln();
        let outlier = LN_HALF / new_log_right_p * first;
        (
            outlier,
            prob_not_outlier
                .iter()
                .map(|&p| p * df2 + (1.0 - p) * outlier)
                .collect(),
        )
    };

    // Make the per-gene prior monotone in the tail probability: level the
    // leading block off at its smallest running mean, then run a maximum
    // through the rest, so a more extreme gene never keeps more prior support
    // than a less extreme one.
    let order = stable_order(&right_p);
    flatten_then_cummax(&mut df2_shrunk, &order);

    Ok(UnequalDf1Fit {
        scale,
        df2,
        df2_outlier: Some(df2_outlier),
        df2_shrunk: Some(df2_shrunk),
    })
}

/// Zeroes the flagged prior weights, creating the vector if there was none.
///
/// limma's idiom for retiring a gene without dropping it: `prior.weights[i] <- 0`
/// when weights exist, `as.numeric(!i)` when they do not.
///
/// ### Params
///
/// * `prior_weights` - Weights, created in place when absent
/// * `mask` - Flags, one per gene, true meaning retire this gene
fn zero_out_weights(prior_weights: &mut Option<Vec<f64>>, mask: &[bool]) {
    match prior_weights {
        Some(p) => {
            for (v, &m) in p.iter_mut().zip(mask) {
                if m {
                    *v = 0.0;
                }
            }
        }
        None => {
            *prior_weights = Some(
                mask.iter()
                    .map(|&m| if m { 0.0 } else { 1.0 })
                    .collect::<Vec<f64>>(),
            );
        }
    }
}

/// Index of the smallest element, the first one when they tie.
///
/// ### Params
///
/// * `v` - Values, non-empty
///
/// ### Returns
///
/// The index, as R's `which.min` picks it. Zero for an empty slice.
fn which_min(v: &[f64]) -> usize {
    let mut best = 0;
    for i in 1..v.len() {
        if v[i] < v[best] {
            best = i;
        }
    }
    best
}

/// Whether every strictly positive degrees of freedom is the same value.
///
/// limma's automatic `legacy` rule. An empty set of positive degrees of freedom
/// gives `false`, because R's `min` and `max` of an empty vector are `Inf` and
/// `-Inf` and it compares them for identity.
///
/// ### Params
///
/// * `df` - Residual degrees of freedom, length one or one per gene
///
/// ### Returns
///
/// Whether the legacy fits apply.
fn all_positive_df_equal(df: &[f64]) -> bool {
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &d in df.iter().filter(|&&d| d > 0.0) {
        lo = lo.min(d);
        hi = hi.max(d);
    }
    lo == hi
}

/// Lower tail of the F distribution, R's `pf(q, df1, df2, lower.tail = TRUE)`.
///
/// Formed from whichever side of the incomplete beta has not been squeezed
/// against one, which is the switch R makes and the reason this is not simply
/// `1 - f_sf`.
///
/// ### Params
///
/// * `x` - Test statistic
/// * `df1` - Numerator degrees of freedom, finite and strictly positive
/// * `df2` - Denominator degrees of freedom, finite and strictly positive
///
/// ### Returns
///
/// `P(F <= x)`, or [`EdgeErrors::InvalidArgument`] for a non-positive `df`.
fn f_cdf(x: f64, df1: f64, df2: f64) -> Result<f64, EdgeErrors> {
    if x <= 0.0 {
        return Ok(0.0);
    }
    if x.is_infinite() {
        return Ok(1.0);
    }
    if df1 * x > df2 {
        beta_sf(df2 / (df2 + df1 * x), 0.5 * df2, 0.5 * df1)
    } else {
        beta_cdf(df1 * x / (df2 + df1 * x), 0.5 * df1, 0.5 * df2)
    }
}

//////////////////////////
// Winsorised moments   //
//////////////////////////

/// Mean and variance of the Winsorised log F distribution.
///
/// The Winsorised log statistic is `log(F)` clipped at its `p1` and `1 - p2`
/// quantiles, so its moments are an integral of `log(F)` over the central part
/// of the density plus point masses `p1` and `p2` at the two clipped values. The
/// integral runs in the link scale `q = f / (1 + f)`, which maps `[0, inf)` onto
/// `[0, 1)` and turns the upper tail into a finite interval; the Jacobian is the
/// `1 / (1 - q)^2` factor on the density.
///
/// ### Params
///
/// * `df1` - Numerator degrees of freedom, finite and positive
/// * `df2` - Denominator degrees of freedom, positive; `inf` is allowed and is
///   evaluated as the chi-square limit
/// * `winsor_tail_p` - Lower and upper Winsorising proportions
/// * `quad` - Gauss-Legendre nodes and weights on `(0, 1)`
///
/// ### Returns
///
/// The Winsorised mean and variance of `log(F)`.
fn winsorized_moments(
    df1: f64,
    df2: f64,
    winsor_tail_p: (f64, f64),
    quad: &(Vec<f64>, Vec<f64>),
) -> Result<(f64, f64), EdgeErrors> {
    let fq = (
        f_quantile(winsor_tail_p.0, df1, df2)?,
        f_quantile(1.0 - winsor_tail_p.1, df1, df2)?,
    );
    let zq = (fq.0.ln(), fq.1.ln());
    let q = (link_fun(fq.0), link_fun(fq.1));
    let width = q.1 - q.0;

    let (nodes, weights) = quad;
    let mut z_nodes = Vec::with_capacity(nodes.len());
    let mut density = Vec::with_capacity(nodes.len());
    for &node in nodes {
        let u = q.0 + width * node;
        let f = link_inv(u);
        z_nodes.push(f.ln());
        density.push(f_density(f, df1, df2) / ((1.0 - u) * (1.0 - u)));
    }

    let mut m = 0.0;
    for i in 0..nodes.len() {
        m += weights[i] * density[i] * z_nodes[i];
    }
    m = width * m + zq.0 * winsor_tail_p.0 + zq.1 * winsor_tail_p.1;

    let mut v = 0.0;
    for i in 0..nodes.len() {
        let d = z_nodes[i] - m;
        v += weights[i] * density[i] * d * d;
    }
    v = width * v
        + (zq.0 - m) * (zq.0 - m) * winsor_tail_p.0
        + (zq.1 - m) * (zq.1 - m) * winsor_tail_p.1;

    Ok((m, v))
}

/// Gauss-Legendre nodes and weights on `(0, 1)`, weights summing to one.
///
/// `statmod::gauss.quad.prob(n, "uniform")`. The nodes are the roots of the
/// `n`-th Legendre polynomial found by Newton from the Chebyshev-like starting
/// guess `cos(pi (i - 1/4) / (n + 1/2))`, which is accurate enough that three or
/// four steps reach machine precision, then mapped from `[-1, 1]` onto `[0, 1]`.
///
/// ### Params
///
/// * `n` - Number of nodes, at least one
///
/// ### Returns
///
/// The nodes in increasing order and their weights.
///
/// ### References
///
/// Press, Teukolsky, Vetterling and Flannery, Numerical Recipes, 3rd edition,
/// section 4.6
fn gauss_legendre_unit(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut nodes = vec![0.0; n];
    let mut weights = vec![0.0; n];
    let half = n.div_ceil(2);

    for i in 0..half {
        let mut z = (std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let mut pp = 0.0;
        for _ in 0..QUAD_NEWTON_MAX_ITER {
            // Legendre recurrence: (j + 1) P_{j+1} = (2j + 1) z P_j - j P_{j-1}.
            let (mut p1, mut p2) = (1.0_f64, 0.0_f64);
            for j in 0..n {
                let p3 = p2;
                p2 = p1;
                p1 = ((2 * j + 1) as f64 * z * p2 - j as f64 * p3) / (j + 1) as f64;
            }
            pp = n as f64 * (z * p1 - p2) / (z * z - 1.0);
            let z1 = z;
            z = z1 - p1 / pp;
            if (z - z1).abs() <= f64::EPSILON {
                break;
            }
        }
        // Ascending order on [-1, 1]: the i-th root from the top is -z.
        nodes[i] = 0.5 * (1.0 - z);
        nodes[n - 1 - i] = 0.5 * (1.0 + z);
        weights[i] = 1.0 / ((1.0 - z * z) * pp * pp);
        weights[n - 1 - i] = weights[i];
    }
    (nodes, weights)
}

/// The link `d -> d / (1 + d)` that maps `[0, inf]` onto `[0, 1]`.
///
/// ### Params
///
/// * `d` - Degrees of freedom, non-negative
///
/// ### Returns
///
/// `d / (1 + d)`, and 1 for an infinite argument.
#[inline]
fn link_fun(d: f64) -> f64 {
    if d.is_infinite() { 1.0 } else { d / (1.0 + d) }
}

/// Inverse of [`link_fun`].
///
/// ### Params
///
/// * `p` - Value in `[0, 1]`
///
/// ### Returns
///
/// `p / (1 - p)`, and `inf` at 1.
#[inline]
fn link_inv(p: f64) -> f64 {
    p / (1.0 - p)
}

/////////////////////////////
// F distribution helpers  //
/////////////////////////////

/// F quantile that also accepts an infinite denominator.
///
/// `F(df1, Inf)` is `chi^2_df1 / df1`, so the infinite case is a gamma quantile.
///
/// ### Params
///
/// * `p` - Probability in `[0, 1]`
/// * `df1` - Numerator degrees of freedom, positive
/// * `df2` - Denominator degrees of freedom, positive or infinite
///
/// ### Returns
///
/// The quantile, or [`EdgeErrors`] from the underlying inversion.
fn f_quantile(p: f64, df1: f64, df2: f64) -> Result<f64, EdgeErrors> {
    if df2.is_infinite() {
        Ok(gamma_ppf(p, 0.5 * df1, 2.0)? / df1)
    } else {
        f_ppf(p, df1, df2)
    }
}

/// F density that also accepts an infinite denominator.
///
/// Evaluated through logs so the two large powers never overflow separately.
///
/// ### Params
///
/// * `x` - Quantile, positive
/// * `df1` - Numerator degrees of freedom, positive
/// * `df2` - Denominator degrees of freedom, positive or infinite
///
/// ### Returns
///
/// The density at `x`.
fn f_density(x: f64, df1: f64, df2: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if df2.is_infinite() {
        // df1 * dchisq(df1 x, df1)
        let y = df1 * x;
        let k = 0.5 * df1;
        return df1
            * ((k - 1.0) * y.ln() - 0.5 * y - ln_gamma(k) - k * std::f64::consts::LN_2).exp();
    }
    let (a, b) = (0.5 * df1, 0.5 * df2);
    let ln_beta = ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b);
    (a * (df1 * x).ln() + b * df2.ln() - (a + b) * (df1 * x + df2).ln() - x.ln() - ln_beta).exp()
}

/// Inverse of the F survival function.
///
/// Built from the beta quantile rather than as `f_ppf(1 - p)`, which would round
/// a small upper-tail probability straight to one and return `inf`. Since
/// `sf(x) = I(df2 / (df1 x + df2); df2/2, df1/2)`, the quantile is recovered
/// from the beta quantile of `p` at the swapped shapes.
///
/// ### Params
///
/// * `p` - Upper-tail probability in `[0, 1]`
/// * `df1` - Numerator degrees of freedom, positive
/// * `df2` - Denominator degrees of freedom, positive
///
/// ### Returns
///
/// The `x` with `P(F > x) = p`.
fn f_isf(p: f64, df1: f64, df2: f64) -> Result<f64, EdgeErrors> {
    let w = beta_ppf(p, 0.5 * df2, 0.5 * df1)?;
    if w <= 0.0 {
        return Ok(f64::INFINITY);
    }
    Ok(df2 * (1.0 - w) / (df1 * w))
}

//////////////////
// Spline fit   //
//////////////////

/// Smallest usable spline basis; below this the trended fit degenerates.
const MIN_SPLINE_DF: usize = 2;

/// Number of spline basis columns limma gives the trended fit.
///
/// `1 + (n >= 3) + (n >= 6) + (n >= 30)`, capped at the number of distinct
/// covariate values because a basis cannot have more columns than the covariate
/// has support points.
///
/// ### Params
///
/// * `nok` - Number of genes surviving the filter
/// * `covariate` - Covariates of those genes
///
/// ### Returns
///
/// The column count, between 1 and 4.
fn spline_columns(nok: usize, covariate: &[f64]) -> usize {
    let mut sorted = covariate.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted.dedup();
    let requested = 1 + usize::from(nok >= 3) + usize::from(nok >= 6) + usize::from(nok >= 30);
    requested.min(sorted.len())
}

/// Least squares fit of `e` on a spline basis.
///
/// The rank comes from the QR diagonal at [`RANK_TOL`], matching R's `lm.fit`,
/// because it is the divisor of the residual variance and an over-counted rank
/// would inflate the estimated prior degrees of freedom.
///
/// ### Params
///
/// * `basis` - Row-major basis, one row per response
/// * `ncol` - Number of basis columns
/// * `e` - Response
///
/// ### Returns
///
/// The fitted values, the residual sum of squares, and the numerical rank of
/// the basis.
fn spline_fit(basis: &[f64], ncol: usize, e: &[f64]) -> Result<(Vec<f64>, f64, usize), EdgeErrors> {
    let nok = e.len();

    let a = MatRef::from_row_major_slice(basis, nok, ncol);
    let qr = a.qr();
    let r = qr.thin_R();
    let pivot = r[(0, 0)].abs();
    let rank = (0..ncol.min(nok))
        .filter(|&i| r[(i, i)].abs() > RANK_TOL * pivot)
        .count();

    let rhs = MatRef::from_column_major_slice(e, nok, 1);
    let coef = qr.solve_lstsq(rhs);
    if (0..ncol).any(|j| !coef[(j, 0)].is_finite()) {
        return Err(EdgeErrors::SolveFailed(
            "the natural spline basis for the variance trend is rank deficient.".to_string(),
        ));
    }

    let fitted: Vec<f64> = basis
        .chunks_exact(ncol)
        .map(|row| row.iter().enumerate().map(|(j, &b)| b * coef[(j, 0)]).sum())
        .collect();
    let rss = e
        .iter()
        .zip(&fitted)
        .map(|(&y, &f)| (y - f) * (y - f))
        .sum();
    Ok((fitted, rss, rank))
}

/// Spreads a trend fitted on the surviving genes over all of them.
///
/// limma evaluates the fitted natural spline at the dropped genes' covariates
/// with `predict.ns`. The basis in [`crate::numeric::interpolate`] does not
/// expose its knots, so the curve is instead read off the fitted values
/// themselves: a natural cubic spline is exactly linear beyond its boundary
/// knots, so a dropped gene outside the surviving covariate range gets the same
/// value limma would give it, and one inside is interpolated between its two
/// neighbours with an error of order the local covariate spacing squared.
///
/// ### Params
///
/// * `covariate` - Covariate of every gene
/// * `ok` - Which genes were fitted
/// * `cov_ok` - Covariates of the fitted genes, in gene order
/// * `fitted_ok` - Fitted trend at those genes
///
/// ### Returns
///
/// The trend at every gene.
fn extend_trend(
    covariate: &[f64],
    ok: &[bool],
    cov_ok: &[f64],
    fitted_ok: &[f64],
) -> Result<Vec<f64>, EdgeErrors> {
    let n = covariate.len();
    if ok.iter().all(|&k| k) {
        return Ok(fitted_ok.to_vec());
    }

    let (knots, values) = collapse_ties(cov_ok, fitted_ok);
    if knots.len() < 2 {
        return Ok(vec![values[0]; n]);
    }

    let mut out = vec![0.0; n];
    let mut k = 0;
    for i in 0..n {
        if ok[i] {
            out[i] = fitted_ok[k];
            k += 1;
        } else {
            out[i] = interp_linear_extend(covariate[i], &knots, &values);
        }
    }
    Ok(out)
}

/// Piecewise-linear interpolation that keeps the end slopes outside the range.
///
/// [`interp_linear_extrap`] holds the end values constant, which is R's
/// `approx(rule = 2)` and the wrong shape for a natural spline: those are linear
/// past their boundary knots, not flat.
///
/// ### Params
///
/// * `x` - Point to evaluate at
/// * `xp` - Knot abscissae, strictly increasing, at least two of them
/// * `fp` - Knot ordinates, same length
///
/// ### Returns
///
/// The interpolated or extrapolated value.
fn interp_linear_extend(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let last = xp.len() - 1;
    let (lo, hi) = if x <= xp[0] {
        (0, 1)
    } else if x >= xp[last] {
        (last - 1, last)
    } else {
        let hi = xp.partition_point(|&knot| knot <= x);
        (hi - 1, hi)
    };
    let t = (x - xp[lo]) / (xp[hi] - xp[lo]);
    fp[lo] + t * (fp[hi] - fp[lo])
}

///////////////
// Lowess    //
///////////////

/// limma's `loessFit` with prior weights, as `fitFDistUnequalDF1` calls it.
///
/// `loessFit`'s `method` argument is a red herring here. It defaults to
/// `weightedLowess`, but there is an early return before the switch: with no
/// weights it is `stats::lowess` and nothing else. `fitFDistUnequalDF1` does
/// supply weights, so the `weightedLowess` branch is normally the live one, but
/// `equal.weights.as.null = TRUE` discards weights that are all equal after
/// clamping and drops it back into `stats::lowess`. Equal `df1` with no prior
/// weights does exactly that, so both branches matter.
///
/// The weights arrive unscaled and are divided by their upper quartile here,
/// then clamped to `[LOESS_MIN_WEIGHT, LOESS_MAX_WEIGHT]`, matching the
/// `min.weight` and `max.weight` limma passes.
///
/// ### Deviation from limma
///
/// The too-few-points fallback is a weighted straight line fitted from the
/// normal equations, where limma uses `lm.wfit`'s pivoted QR. The fitted values
/// agree to rounding whenever the covariate has any weighted spread, and the
/// rank test is [`WLS_RANK_TOL`] rather than the QR's own. It only runs when
/// there are fewer than `4 + 1 / span` points, which needs three or four genes.
///
/// ### Params
///
/// * `e` - Response, one per gene
/// * `covariate` - Covariate, one per gene
/// * `w` - Prior weights before quantile scaling, one per gene
/// * `span` - Fraction of the total weight in each local window, in `(0, 1]`
///
/// ### Returns
///
/// The fitted trend, `NaN` at any gene whose response or covariate is not
/// finite, or [`EdgeErrors`] out of the quantile or the lowess.
fn loess_fit_weighted(
    e: &[f64],
    covariate: &[f64],
    w: &[f64],
    span: f64,
) -> Result<Vec<f64>, EdgeErrors> {
    let n = e.len();
    let q75 = quantile_type7(w, 0.75)?;

    let obs: Vec<usize> = (0..n)
        .filter(|&i| e[i].is_finite() && covariate[i].is_finite())
        .collect();
    let nobs = obs.len();
    let mut fitted = vec![f64::NAN; n];
    if nobs == 0 {
        return Ok(fitted);
    }

    let xobs: Vec<f64> = obs.iter().map(|&i| covariate[i]).collect();
    let yobs: Vec<f64> = obs.iter().map(|&i| e[i]).collect();
    let scatter = |fitted: &mut [f64], values: &[f64]| {
        for (&i, &v) in obs.iter().zip(values) {
            fitted[i] = v;
        }
    };

    // A window narrower than one point cannot smooth anything.
    if span < 1.0 / nobs as f64 || nobs < 2 {
        scatter(&mut fitted, &yobs);
        return Ok(fitted);
    }

    let wobs: Vec<f64> = obs
        .iter()
        .map(|&i| {
            let v = w[i] / q75;
            let v = if v.is_nan() { 0.0 } else { v };
            v.clamp(LOESS_MIN_WEIGHT, LOESS_MAX_WEIGHT)
        })
        .collect();

    let lo = wobs.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = wobs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if hi - lo < EQUAL_WEIGHT_TOL {
        let ys = lowess(&xobs, &yobs, span, 0)?;
        scatter(&mut fitted, &ys);
        return Ok(fitted);
    }

    // `min.weight` is positive, so every observation counts towards the point
    // budget whatever its weight.
    if (nobs as f64) < 4.0 + 1.0 / span {
        let ys = weighted_line_fitted(&xobs, &yobs, &wobs);
        scatter(&mut fitted, &ys);
        return Ok(fitted);
    }

    let fit = weighted_lowess(
        &xobs,
        &yobs,
        Some(&wobs),
        Some(LowessParams::new(span, 1, 200, None)),
    )?;
    scatter(&mut fitted, &fit.fitted);
    Ok(fitted)
}

/// Fitted values of a weighted straight line, `lm.wfit(cbind(1, x), y, w)`.
///
/// Solved from the two-by-two weighted normal equations. When the covariate has
/// no weighted spread the slope is dropped and every fitted value is the
/// weighted mean, which is what the pivoted QR in `lm.wfit` does with a rank
/// deficient design.
///
/// ### Params
///
/// * `x` - Covariate
/// * `y` - Response
/// * `w` - Weights, non-negative
///
/// ### Returns
///
/// One fitted value per point.
fn weighted_line_fitted(x: &[f64], y: &[f64], w: &[f64]) -> Vec<f64> {
    let sw: f64 = w.iter().sum();
    let swx: f64 = w.iter().zip(x).map(|(&a, &b)| a * b).sum();
    let swy: f64 = w.iter().zip(y).map(|(&a, &b)| a * b).sum();
    let swxx: f64 = w.iter().zip(x).map(|(&a, &b)| a * b * b).sum();
    let swxy: f64 = (0..x.len()).map(|i| w[i] * x[i] * y[i]).sum();

    let det = sw * swxx - swx * swx;
    if det.abs() <= WLS_RANK_TOL * (sw * swxx).abs() {
        return vec![swy / sw; x.len()];
    }
    let slope = (sw * swxy - swx * swy) / det;
    let intercept = (swy - slope * swx) / sw;
    x.iter().map(|&v| intercept + slope * v).collect()
}

//////////////////////
// Small utilities  //
//////////////////////

/// Checks that `x` is non-empty and `df1` is length one or matches it.
///
/// ### Params
///
/// * `name` - Caller name, for the error message
/// * `x` - Variances
/// * `df1` - Residual degrees of freedom
///
/// ### Returns
///
/// The gene count, or [`EdgeErrors`] if the shapes do not work.
fn check_fit_lengths(name: &'static str, x: &[f64], df1: &[f64]) -> Result<usize, EdgeErrors> {
    let n = x.len();
    if n == 0 {
        return Err(EdgeErrors::InvalidArgument(format!(
            "{name} was given an empty variance vector."
        )));
    }
    if df1.len() != 1 && df1.len() != n {
        return Err(EdgeErrors::LengthMismatch {
            name: "df1",
            expected: n,
            got: df1.len(),
        });
    }
    Ok(n)
}

/// Checks that both Winsorising proportions lie in `[0, 0.5)`.
///
/// ### Params
///
/// * `winsor_tail_p` - Lower and upper proportions
///
/// ### Returns
///
/// Nothing, or [`EdgeErrors::InvalidArgument`].
fn check_winsor_tail_p(winsor_tail_p: (f64, f64)) -> Result<(), EdgeErrors> {
    for p in [winsor_tail_p.0, winsor_tail_p.1] {
        if !(0.0..0.5).contains(&p) {
            return Err(EdgeErrors::InvalidArgument(format!(
                "winsor_tail_p entries must lie in [0, 0.5); got {p}."
            )));
        }
    }
    Ok(())
}

/// Which genes carry usable information, by limma's `fitFDist` rule.
///
/// ### Params
///
/// * `x` - Variances
/// * `df1` - Residual degrees of freedom, length one or one per gene
/// * `df_tol` - Smallest degrees of freedom to accept
///
/// ### Returns
///
/// One flag per gene.
fn informative_mask(x: &[f64], df1: &[f64], df_tol: f64) -> Vec<bool> {
    (0..x.len())
        .map(|i| {
            let d = at(df1, i);
            d.is_finite() && d > df_tol && x[i].is_finite() && x[i] > VAR_TOL
        })
        .collect()
}

/// Clamps variances to zero and lifts exact zeros off the floor.
///
/// ### Params
///
/// * `x` - Variances
///
/// ### Returns
///
/// The clamped variances, none of them zero unless every one of them was.
fn offset_from_zero(x: &[f64]) -> Vec<f64> {
    let mut out: Vec<f64> = x.iter().map(|&v| v.max(0.0)).collect();
    let m = median(&out);
    // More than half the variances are exactly zero: limma warns that eBayes is
    // unreliable and carries on with a unit scale.
    let m = if m == 0.0 { 1.0 } else { m };
    let floor = ZERO_VAR_OFFSET * m;
    for v in &mut out {
        *v = v.max(floor);
    }
    out
}

/// Maps infinite covariate values one unit past the finite range, as limma does.
///
/// ### Params
///
/// * `covariate` - Covariate values
///
/// ### Returns
///
/// A finite covariate, or [`EdgeErrors::InvalidArgument`] if any value is `NaN`.
fn finitise_covariate(covariate: &[f64]) -> Result<Vec<f64>, EdgeErrors> {
    if let Some(i) = covariate.iter().position(|v| v.is_nan()) {
        return Err(EdgeErrors::InvalidArgument(format!(
            "fit_f_dist_trend covariate must not contain NaN; covariate[{i}] is NaN."
        )));
    }
    if covariate.iter().all(|v| v.is_finite()) {
        return Ok(covariate.to_vec());
    }
    let finite: Vec<f64> = covariate
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    if finite.is_empty() {
        return Ok(covariate.iter().map(|v| v.signum()).collect());
    }
    let lo = finite.iter().fold(f64::INFINITY, |a, &b| a.min(b)) - 1.0;
    let hi = finite.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)) + 1.0;
    Ok(covariate
        .iter()
        .map(|&v| {
            if v == f64::NEG_INFINITY {
                lo
            } else if v == f64::INFINITY {
                hi
            } else {
                v
            }
        })
        .collect())
}

/// Reads a length-one or full-length vector at a gene index.
///
/// ### Params
///
/// * `v` - Vector of length one or at least `i + 1`
/// * `i` - Gene index
///
/// ### Returns
///
/// The recycled value.
#[inline]
fn at(v: &[f64], i: usize) -> f64 {
    if v.len() == 1 { v[0] } else { v[i] }
}

/// Expands a length-one vector to `n`, leaving longer ones alone.
///
/// ### Params
///
/// * `v` - Vector of length one or `n`
/// * `n` - Target length
///
/// ### Returns
///
/// A vector of length `n`.
fn recycle(v: &[f64], n: usize) -> Vec<f64> {
    if v.len() == 1 {
        vec![v[0]; n]
    } else {
        v.to_vec()
    }
}

/// Keeps the flagged entries.
///
/// ### Params
///
/// * `v` - Values
/// * `ok` - Flags, same length
///
/// ### Returns
///
/// The flagged values in order.
fn subset(v: &[f64], ok: &[bool]) -> Vec<f64> {
    v.iter()
        .zip(ok)
        .filter(|(_, k)| **k)
        .map(|(&x, _)| x)
        .collect()
}

/// Keeps the flagged entries of a vector that may be length one.
///
/// A length-one vector is expanded to the number of flagged entries rather than
/// left alone, so callers can zip it against the subset without silently
/// truncating to one element.
///
/// ### Params
///
/// * `v` - Values, length one or matching `ok`
/// * `ok` - Flags
///
/// ### Returns
///
/// The flagged values, or the single value repeated.
fn subset_recycled(v: &[f64], ok: &[bool]) -> Vec<f64> {
    if v.len() == 1 {
        vec![v[0]; ok.iter().filter(|&&k| k).count()]
    } else {
        subset(v, ok)
    }
}

/// Arithmetic mean.
///
/// ### Params
///
/// * `x` - Values, non-empty
///
/// ### Returns
///
/// The mean.
fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

/// Mean of `trigamma(df / 2)` over the given degrees of freedom.
///
/// This is the part of the observed variance of `e` that the genewise residual
/// degrees of freedom already explain, and subtracting it is what leaves the
/// prior's own contribution behind.
///
/// ### Params
///
/// * `df` - Residual degrees of freedom, non-empty
///
/// ### Returns
///
/// The mean.
fn mean_trigamma(df: &[f64]) -> f64 {
    df.iter().map(|&d| trigamma(0.5 * d)).sum::<f64>() / df.len() as f64
}

/// Indices that sort `x` ascending, ties keeping their input order.
///
/// ### Params
///
/// * `x` - Values
///
/// ### Returns
///
/// The permutation, as R's `order` produces it.
fn stable_order(x: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..x.len()).collect();
    order.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
    order
}

/// Running maximum applied along a permutation.
///
/// ### Params
///
/// * `v` - Values, modified in place
/// * `order` - Permutation to walk
fn cummax_in_order(v: &mut [f64], order: &[usize]) {
    let mut running = f64::NEG_INFINITY;
    for &i in order {
        running = running.max(v[i]);
        v[i] = running;
    }
}

/// Flattens the leading run to its smallest running mean, then takes a
/// [`cummax_in_order`] through the rest.
///
/// limma's per-gene prior monotonicity trick: without the flatten, the most
/// extreme gene could keep more prior support than a less extreme one, since a
/// bare cummax only enforces monotonicity from that point onward.
///
/// ### Params
///
/// * `v` - Values, modified in place
/// * `order` - Permutation to walk, e.g. by ascending tail probability
fn flatten_then_cummax(v: &mut [f64], order: &[usize]) {
    let mut running = 0.0;
    let (mut best, mut imin) = (f64::INFINITY, 0);
    for (k, &idx) in order.iter().enumerate() {
        running += v[idx];
        let avg = running / (k + 1) as f64;
        if avg < best {
            best = avg;
            imin = k;
        }
    }
    for &idx in &order[..=imin] {
        v[idx] = best;
    }
    cummax_in_order(v, order);
}

/// Sorts knots ascending and averages the ordinates at duplicate abscissae.
///
/// R's `approx(..., ties = mean)`; [`interp_linear_extrap`] needs strictly
/// increasing knots and a covariate can easily repeat a value.
///
/// ### Params
///
/// * `x` - Knot abscissae, in any order
/// * `y` - Knot ordinates, same length
///
/// ### Returns
///
/// The distinct abscissae in increasing order and the mean ordinate at each.
fn collapse_ties(x: &[f64], y: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let order = stable_order(x);
    let mut knots = Vec::with_capacity(x.len());
    let mut values = Vec::with_capacity(x.len());
    let mut k = 0;
    while k < order.len() {
        let xi = x[order[k]];
        let mut sum = 0.0;
        let mut count = 0;
        while k < order.len() && x[order[k]] == xi {
            sum += y[order[k]];
            count += 1;
            k += 1;
        }
        knots.push(xi);
        values.push(sum / count as f64);
    }
    (knots, values)
}

/// Maps a fallible scalar function over a slice, in parallel above
/// [`PARALLEL_THRESHOLD`].
///
/// ### Params
///
/// * `x` - Inputs
/// * `f` - Function to apply
///
/// ### Returns
///
/// The mapped values, or the first error encountered.
fn map_maybe_par<F>(x: &[f64], f: F) -> Result<Vec<f64>, EdgeErrors>
where
    F: Fn(f64) -> Result<f64, EdgeErrors> + Sync + Send,
{
    if x.len() >= PARALLEL_THRESHOLD {
        x.par_iter().map(|&v| f(v)).collect()
    } else {
        x.iter().map(|&v| f(v)).collect()
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Variances for fixture A, `k / 64` with `k = ((i * 37) mod 97) + 1` for
    /// `i` in `1..=24`. Every value is an exact `f64`, so R and Rust see the
    /// same bits and only the outputs need embedding.
    fn fixture_a() -> Vec<f64> {
        (1..=24)
            .map(|i| (((i * 37) % 97) + 1) as f64 / 64.0)
            .collect()
    }

    /// Covariate for fixture A, `((i * 53) mod 101) / 8`.
    fn covariate_a() -> Vec<f64> {
        (1..=24).map(|i| ((i * 53) % 101) as f64 / 8.0).collect()
    }

    /// Variances for fixture B, `k / 64` with `k = ((i * 37) mod 251) + 1` for
    /// `i` in `1..=40`, with three genes replaced by gross outliers.
    fn fixture_b() -> Vec<f64> {
        let mut x: Vec<f64> = (1..=40)
            .map(|i| (((i * 37) % 251) + 1) as f64 / 64.0)
            .collect();
        x[6] = 40.0;
        x[22] = 60.0;
        x[30] = 25.0;
        x
    }

    /// Covariate for fixture B, `((i * 29) mod 97) / 8`.
    fn covariate_b() -> Vec<f64> {
        (1..=40).map(|i| ((i * 29) % 97) as f64 / 8.0).collect()
    }

    /// Nearly constant variances, `1 + (((j * 17) mod 7) - 3) / 4096` for `j` in
    /// `1..=60`. Dispersed far less than a chi-square on six degrees of freedom
    /// would be, which is what drives the robust fit to `df2 = Inf`.
    fn fixture_d() -> Vec<f64> {
        (1..=60)
            .map(|j| 1.0 + ((((j * 17) % 7) as f64) - 3.0) / 4096.0)
            .collect()
    }

    /// Asserts two slices agree elementwise, with infinities matching exactly.
    fn assert_slices_close(got: &[f64], want: &[f64], tol: f64) {
        assert_eq!(got.len(), want.len(), "length mismatch");
        for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
            if w.is_infinite() {
                assert_eq!(g, w, "element {i}: got {g}, want {w}");
            } else {
                assert_relative_eq!(g, w, max_relative = tol, epsilon = 1e-300);
            }
        }
    }

    // -- fitFDist, untrended --

    /// `Rscript -e 'suppressMessages(library(limma)); i <- 1:24; x <- (((i*37) %% 97)+1)/64;
    ///  r <- limma:::fitFDist(x, df1=rep(4,24)); cat(sprintf("%.17g %.17g", r$scale, r$df2))'`
    const A_FIT_SCALE: f64 = 0.640_837_559_492_241_4;
    const A_FIT_DF2: f64 = 8.773_643_845_753_776;

    #[test]
    fn test_fit_f_dist_matches_limma() {
        let (scale, df2) = fit_f_dist(&fixture_a(), &[4.0]).unwrap();
        assert_relative_eq!(scale, A_FIT_SCALE, max_relative = 1e-12);
        assert_relative_eq!(df2, A_FIT_DF2, max_relative = 1e-12);

        // A genewise df vector of identical values must give the same answer.
        let (scale2, df2b) = fit_f_dist(&fixture_a(), &[4.0; 24]).unwrap();
        assert_relative_eq!(scale2, A_FIT_SCALE, max_relative = 1e-12);
        assert_relative_eq!(df2b, A_FIT_DF2, max_relative = 1e-12);
    }

    /// `Rscript -e 'suppressMessages(library(limma)); i <- 1:24; x <- (((i*37) %% 97)+1)/64;
    ///  r <- limma:::fitFDist(x, df1=rep(c(2,4,7,11),6)); cat(sprintf("%.17g %.17g", r$scale, r$df2))'`
    const A_FIT_UNEQUAL_SCALE: f64 = 0.642_853_466_657_464_3;
    const A_FIT_UNEQUAL_DF2: f64 = 8.830_849_273_819_364;

    #[test]
    fn test_fit_f_dist_matches_limma_with_unequal_df1() {
        let df: Vec<f64> = (0..24).map(|i| [2.0, 4.0, 7.0, 11.0][i % 4]).collect();
        let (scale, df2) = fit_f_dist(&fixture_a(), &df).unwrap();
        assert_relative_eq!(scale, A_FIT_UNEQUAL_SCALE, max_relative = 1e-12);
        assert_relative_eq!(df2, A_FIT_UNEQUAL_DF2, max_relative = 1e-12);
    }

    /// `Rscript -e 'suppressMessages(library(limma)); i <- 1:24; x <- (((i*37) %% 97)+1)/64;
    ///  s <- squeezeVar(x, df=rep(4,24), legacy=TRUE); cat(s$var.post, sep=",")'`
    const A_VAR_POST: [f64; 24] = [
        0.626_092_335_635_731_7,
        0.807_129_166_466_796_7,
        0.513_555_927_281_286_1,
        0.694_592_758_112_351,
        0.875_629_588_943_415_9,
        0.582_056_349_757_905_2,
        0.763_093_180_588_970_1,
        0.46951994140345943,
        0.650_556_772_234_524_3,
        0.831_593_603_065_589_3,
        0.538_020_363_880_078_6,
        0.719_057_194_711_143_5,
        0.900_094_025_542_208_4,
        0.606_520_786_356_697_7,
        0.787_557_617_187_762_6,
        0.493_984_378_002_252,
        0.675_021_208_833_317,
        0.856_058_039_664_381_9,
        0.562_484_800_478_871_2,
        0.743_521_631_309_936_1,
        0.449_948_392_124_425_4,
        0.630_985_222_955_490_3,
        0.812_022_053_786_555_2,
        0.518_448_814_601_044_5,
    ];

    #[test]
    fn test_squeeze_var_untrended_matches_limma() {
        let out = squeeze_var(&fixture_a(), &[4.0; 24], None, None).unwrap();
        assert_eq!(out.var_prior.len(), 1);
        assert_eq!(out.df_prior.len(), 1);
        assert_relative_eq!(out.var_prior[0], A_FIT_SCALE, max_relative = 1e-12);
        assert_relative_eq!(out.df_prior[0], A_FIT_DF2, max_relative = 1e-12);
        assert_slices_close(&out.var_post, &A_VAR_POST, 1e-12);
    }

    /// `Rscript -e 'suppressMessages(library(limma)); i <- 1:24; x <- (((i*37) %% 97)+1)/64;
    ///  s <- squeezeVar(x, df=rep(c(2,4,7,11),6), legacy=TRUE); cat(s$var.post, sep=",")'`
    const A_VAR_POST_UNEQUAL: [f64; 24] = [
        0.633_786_132_154_682_8,
        0.807_775_217_993_745_6,
        0.462_234_649_742_128_9,
        0.736_954_926_509_288_6,
        0.780_935_257_741_026,
        0.583_705_872_415_329_6,
        0.814_592_562_038_404_4,
        0.338_270_538_824_601_3,
        0.648_212_517_016_089,
        0.832_130_581_643_573_5,
        0.496_779_543_104_508_9,
        0.780_290_186_040_232_9,
        0.795_361_642_602_432_3,
        0.608_061_236_065_157_4,
        0.849_137_455_400_784_3,
        0.381_605_798_355_545_6,
        0.662_638_901_877_495_2,
        0.856_485_945_293_401_3,
        0.531_324_436_466_888_9,
        0.823_625_445_571_177_2,
        0.529_916_161_152_558_1,
        0.632_416_599_714_985_2,
        0.883_682_348_763_164_3,
        0.42494105788648984,
    ];

    #[test]
    fn test_squeeze_var_with_unequal_df_matches_limma_legacy() {
        // Unequal df would send the default straight to fitFDistUnequalDF1, so
        // the legacy fit these numbers came from has to be asked for.
        let df: Vec<f64> = (0..24).map(|i| [2.0, 4.0, 7.0, 11.0][i % 4]).collect();
        let out = squeeze_var(
            &fixture_a(),
            &df,
            None,
            Some(SqueezeVarParams {
                legacy: Some(true),
                ..Default::default()
            }),
        )
        .unwrap();
        assert_relative_eq!(out.var_prior[0], A_FIT_UNEQUAL_SCALE, max_relative = 1e-12);
        assert_relative_eq!(out.df_prior[0], A_FIT_UNEQUAL_DF2, max_relative = 1e-12);
        assert_slices_close(&out.var_post, &A_VAR_POST_UNEQUAL, 1e-12);
    }

    // -- fitFDist, trended --

    /// `Rscript -e 'suppressMessages(library(limma)); i <- 1:24; x <- (((i*37) %% 97)+1)/64;
    ///  a <- ((i*53) %% 101)/8; s <- squeezeVar(x, df=rep(4,24), covariate=a, legacy=TRUE);
    ///  cat(s$df.prior, "\n"); cat(s$var.prior, sep=",")'`
    const A_TREND_DF_PRIOR: f64 = 8.048_268_025_548_438;
    const A_TREND_VAR_PRIOR: [f64; 24] = [
        0.723_110_005_839_652_5,
        0.4143418913007032,
        0.742_711_177_713_071_7,
        0.445_533_797_163_167_8,
        0.757_534_747_932_870_6,
        0.47844567134565685,
        0.767_947_074_917_522_7,
        0.512_694_459_358_742_5,
        0.774_435_547_851_372_9,
        0.547_775_357_103_530_8,
        0.777_577_435_996_847_5,
        0.583_053_065_165_493,
        0.778_010_618_468_01,
        0.617_759_245_685_393_9,
        0.776_408_046_642_088_2,
        0.650_998_304_607_132_7,
        0.773_457_194_450_546_4,
        0.681_763_500_708_267_1,
        0.769_845_216_183_354_9,
        0.708_964_967_024_256_9,
        0.396_570_490_459_764_1,
        0.731_541_265_734_613_1,
        0.42659244020765413,
        0.749_196_478_552_777_4,
    ];

    /// Same command, `cat(s$var.post, sep=",")`.
    const A_TREND_VAR_POST: [f64; 24] = [
        0.680_162_752_154_610_3,
        0.665_841_312_493_174_1,
        0.573_944_620_848_585_9,
        0.567_365_815_527_526_2,
        0.967_719_398_780_496_7,
        0.470_039_260_966_351_2,
        0.855_363_100_033_876_8,
        0.37360576761637165,
        0.740_385_658_644_428_8,
        0.780_912_482_342_576_5,
        0.623_172_691_676_536,
        0.685_166_309_718_917_1,
        1.0073346607510243,
        0.589_038_355_511_984_6,
        0.886_952_384_683_656_7,
        0.491_930_361_035_111_1,
        0.765_669_454_536_127_1,
        0.896_354_178_113_888_8,
        0.643_944_890_799_087,
        0.795_213_059_256_225,
        0.275_284_845_191_880_1,
        0.690_982_318_846_792,
        0.679_212_172_165_421_3,
        0.583_464_282_856_523_7,
    ];

    #[test]
    fn test_squeeze_var_trended_matches_limma() {
        let cov = covariate_a();
        let out = squeeze_var(&fixture_a(), &[4.0; 24], Some(&cov), None).unwrap();
        assert_eq!(out.var_prior.len(), 24);
        assert_eq!(out.df_prior.len(), 1);
        assert_relative_eq!(out.df_prior[0], A_TREND_DF_PRIOR, max_relative = 1e-12);
        assert_slices_close(&out.var_prior, &A_TREND_VAR_PRIOR, 1e-12);
        assert_slices_close(&out.var_post, &A_TREND_VAR_POST, 1e-12);
    }

    #[test]
    fn test_fit_f_dist_trend_agrees_with_squeeze_var() {
        let cov = covariate_a();
        let (scale, df2) = fit_f_dist_trend(&fixture_a(), &[4.0], &cov, None).unwrap();
        assert_relative_eq!(df2, A_TREND_DF_PRIOR, max_relative = 1e-12);
        assert_slices_close(&scale, &A_TREND_VAR_PRIOR, 1e-12);
    }

    // -- fitFDistRobustly --

    /// Every robust reference below comes from limma with its `uniroot` call
    /// converged past its own default.
    ///
    /// `fitFDistRobustly` solves for `df2` with `uniroot(..., tol = 1e-8)`, and
    /// `1e-8` is a bracket width in the `d / (1 + d)` link, worth roughly `1e-4`
    /// on a `df2` of 100. limma's stock answer therefore sits a few parts in
    /// `1e8` from the root of its own objective. Converging properly and
    /// comparing against limma's own converged value is the sharper test; the
    /// stock output is checked at the tolerance it deserves by
    /// `test_robust_fit_matches_stock_limma_to_its_own_root_tolerance`.
    ///
    /// Every command below assumes this preamble:
    ///
    /// ```text
    /// suppressMessages(library(limma))
    /// i <- 1:40; x <- (((i*37) %% 251)+1)/64; x[7] <- 40; x[23] <- 60; x[31] <- 25
    /// a <- ((i*29) %% 97)/8
    /// mk <- function(t) function(f, interval, tol=1e-8, ...) stats::uniroot(f, interval, tol=t, ...)
    /// tightR <- local({ f <- limma:::fitFDistRobustly; e <- new.env(parent=environment(f))
    ///   assign("uniroot", mk(1e-14), envir=e); environment(f) <- e; f })
    /// tightS <- local({ f <- limma::squeezeVar; e <- new.env(parent=environment(f))
    ///   assign("fitFDistRobustly", tightR, envir=e); environment(f) <- e; f })
    /// ```
    ///
    /// `r <- tightR(x, df1=rep(4,40)); cat(sprintf("%.17g %.17g", r$scale, r$df2))`
    const B_ROBUST_SCALE: f64 = 2.249456506635014;
    const B_ROBUST_DF2: f64 = 40.76894452916315;

    /// Same command, `cat(r$df2.shrunk[c(7,23,31)], sep=",")`. Every other gene
    /// takes [`B_ROBUST_DF2`].
    const B_ROBUST_SHRUNK_OUTLIERS: [f64; 3] =
        [0.37277999082103946, 0.37276287480907766, 0.3749930112639601];

    /// `s <- tightS(x, df=rep(4,40), robust=TRUE, legacy=TRUE); cat(s$var.post, sep=",")`
    const B_ROBUST_VAR_POST: [f64; 40] = [
        2.1015230206841475,
        2.15317713101265,
        2.204831241341153,
        2.2564853516696552,
        2.3081394619981577,
        2.35979357232666,
        36.781761879974304,
        2.1126914769713916,
        2.164345587299894,
        2.2159996976283964,
        2.267653807956899,
        2.3193079182854013,
        2.3709620286139037,
        2.072205822930133,
        2.123859933258635,
        2.1755140435871376,
        2.22716815391564,
        2.2788222642441425,
        2.330476374572645,
        2.3821304849011478,
        2.0833742792173764,
        2.1350283895458793,
        55.07696638699776,
        2.238336610202884,
        2.2899907205313865,
        2.341644830859889,
        2.3932989411883914,
        2.0945427355046204,
        2.146196845833123,
        2.1978509561616253,
        23.049986642149197,
        2.30115917681863,
        2.3528132871471326,
        2.0540570814633616,
        2.105711191791864,
        2.1573653021203665,
        2.209019412448869,
        2.2606735227773713,
        2.312327633105874,
        2.3639817434343766,
    ];

    /// Stock limma, `uniroot` left at `tol = 1e-8`:
    /// `r <- limma:::fitFDistRobustly(x, df1=rep(4,40));
    ///  cat(sprintf("%.17g %.17g", r$scale, r$df2))`
    const B_ROBUST_SCALE_STOCK: f64 = 2.2494565078913564;
    const B_ROBUST_DF2_STOCK: f64 = 40.768945564031306;

    #[test]
    fn test_robust_fit_matches_stock_limma_to_its_own_root_tolerance() {
        let (scale, shrunk) = fit_f_dist_robustly(&fixture_b(), &[4.0], None, (0.05, 0.1)).unwrap();
        assert_relative_eq!(scale[0], B_ROBUST_SCALE_STOCK, max_relative = 1e-7);
        assert_relative_eq!(shrunk[0], B_ROBUST_DF2_STOCK, max_relative = 1e-7);
        // And closer to limma's converged root than limma's own default
        // tolerance gets, which is the point of the exercise.
        let ours = (shrunk[0] - B_ROBUST_DF2).abs();
        let stock = (B_ROBUST_DF2_STOCK - B_ROBUST_DF2).abs();
        assert!(ours < stock, "ours {ours:e} should beat limma's {stock:e}");
    }

    #[test]
    fn test_fit_f_dist_robustly_matches_limma() {
        let (scale, shrunk) = fit_f_dist_robustly(&fixture_b(), &[4.0], None, (0.05, 0.1)).unwrap();
        assert_eq!(scale.len(), 1);
        assert_relative_eq!(scale[0], B_ROBUST_SCALE, max_relative = 1e-12);
        for (k, &i) in [6_usize, 22, 30].iter().enumerate() {
            assert_relative_eq!(shrunk[i], B_ROBUST_SHRUNK_OUTLIERS[k], max_relative = 1e-12);
        }
        for (i, &v) in shrunk.iter().enumerate() {
            if ![6, 22, 30].contains(&i) {
                assert_relative_eq!(v, B_ROBUST_DF2, max_relative = 1e-12);
            }
        }
    }

    #[test]
    fn test_squeeze_var_robust_matches_limma() {
        let out = squeeze_var(
            &fixture_b(),
            &vec![4.0; 40],
            None,
            Some(SqueezeVarParams {
                robust: true,
                ..Default::default()
            }),
        )
        .unwrap();
        assert_eq!(out.var_prior.len(), 40);
        assert_eq!(out.df_prior.len(), 40);
        for &v in &out.var_prior {
            assert_relative_eq!(v, B_ROBUST_SCALE, max_relative = 1e-12);
        }
        assert_slices_close(&out.var_post, &B_ROBUST_VAR_POST, 1e-12);
    }

    /// `r <- tightR(x, df1=rep(c(3,5,8,12),10));
    ///  cat(sprintf("%.17g %.17g", r$scale, r$df2)); cat(r$df2.shrunk[c(7,23,31)], sep=",")`
    const B_ROBUST_UNEQUAL_SCALE: f64 = 1.8234463961467966;
    const B_ROBUST_UNEQUAL_DF2: f64 = 6.128948796846419;
    const B_ROBUST_UNEQUAL_SHRUNK: [f64; 3] =
        [0.38502316453516383, 0.378098698877172, 0.4906283453912008];

    #[test]
    fn test_fit_f_dist_robustly_with_unequal_df1_matches_limma() {
        let df: Vec<f64> = (0..40).map(|i| [3.0, 5.0, 8.0, 12.0][i % 4]).collect();
        let (scale, shrunk) = fit_f_dist_robustly(&fixture_b(), &df, None, (0.05, 0.1)).unwrap();
        assert_relative_eq!(scale[0], B_ROBUST_UNEQUAL_SCALE, max_relative = 1e-12);
        for (k, &i) in [6_usize, 22, 30].iter().enumerate() {
            assert_relative_eq!(shrunk[i], B_ROBUST_UNEQUAL_SHRUNK[k], max_relative = 1e-12);
        }
        for (i, &v) in shrunk.iter().enumerate() {
            if ![6, 22, 30].contains(&i) {
                assert_relative_eq!(v, B_ROBUST_UNEQUAL_DF2, max_relative = 1e-12);
            }
        }
    }

    /// `s <- tightS(x, df=rep(c(3,5,8,12),10), robust=TRUE, legacy=TRUE);
    ///  cat(s$var.post, sep=",")`
    const B_ROBUST_UNEQUAL_VAR_POST: [f64; 40] = [
        1.4193375255050005,
        1.5307092257091761,
        1.7818600631773178,
        2.1575056576132927,
        2.179282635767482,
        2.56966629264046,
        38.24701051014726,
        1.0922205042149082,
        1.650401917138863,
        1.8466083338977421,
        2.179978853250044,
        2.622921501331034,
        2.4103470274013445,
        1.123550375155024,
        1.2687291781946928,
        1.5576363479326492,
        1.881466308772726,
        2.162507442086308,
        2.5780976433227707,
        3.0883373450487746,
        1.3525855901441068,
        1.4394494833435902,
        57.37452612897447,
        2.02305219165039,
        2.1125307004065883,
        2.4784065502748747,
        2.9762164333954972,
        0.9577670382520053,
        1.5836499817779695,
        1.7553485915321565,
        23.66074998410438,
        2.4884680353681308,
        2.343595092040451,
        1.0322906327894383,
        1.1537170832847938,
        1.423182881969746,
        1.8147143734118323,
        2.0712476997207228,
        2.463085548412872,
        2.953883879085872,
    ];

    #[test]
    fn test_squeeze_var_robust_unequal_df_matches_limma() {
        // As above: these are `legacy = TRUE` numbers and unequal df no longer
        // reaches that fit by default.
        let df: Vec<f64> = (0..40).map(|i| [3.0, 5.0, 8.0, 12.0][i % 4]).collect();
        let out = squeeze_var(
            &fixture_b(),
            &df,
            None,
            Some(SqueezeVarParams {
                robust: true,
                legacy: Some(true),
                ..Default::default()
            }),
        )
        .unwrap();
        assert_slices_close(&out.var_post, &B_ROBUST_UNEQUAL_VAR_POST, 1e-12);
    }

    /// `Rscript -e '... s <- squeezeVar(x, df=rep(4,40), covariate=a, robust=TRUE, legacy=TRUE)'`
    /// with `a <- ((i*29) %% 97)/8`. The Winsorised spread of the residuals about
    /// the lowess trend is already below its value at `df2 = Inf`, so the fit
    /// takes the infinite branch and only the three outliers get a finite prior.
    const B_ROBUST_TREND_VAR_PRIOR: [f64; 40] = [
        1.465_770_201_514_433,
        2.358_668_761_366_851,
        3.339_447_111_256_858,
        1.8174232291408439,
        1.9734580517452869,
        3.0165766069245397,
        2.8946582725605468,
        1.5611743355219518,
        2.5311378090478764,
        3.4492900227288428,
        1.4817076508386355,
        2.3319515209800286,
        3.3190827874202276,
        1.903968165960005,
        1.9370106534545692,
        2.9673210210542877,
        3.0348075460931447,
        1.5176429150692603,
        2.503943464329236,
        3.4441373069445405,
        1.4977238662889754,
        2.300227928362065,
        3.298_061_598_485_803,
        1.9936741757277605,
        1.9002124768758293,
        2.914_072_802_964_981,
        3.182_043_611_244_255,
        1.4812391846190578,
        2.4831810620408024,
        3.4371897720428874,
        1.5133318524086241,
        2.2621453665329994,
        3.2773255485123842,
        2.087_594_717_837_246,
        1.8645313934402432,
        2.860_942_861_174_906,
        3.3366770288668715,
        1.4533273348912759,
        2.4663072510486352,
        3.4283856470227216,
    ];

    /// Same command, `s$df.prior[c(7,23,31)]`; every other gene is `Inf`.
    const B_ROBUST_TREND_SHRUNK: [f64; 3] = [
        7.286_571_062_696_712e-8,
        7.552_642_835_491_027e-11,
        6.50314614594886e-10,
    ];

    #[test]
    fn test_squeeze_var_robust_trended_matches_limma() {
        let cov = covariate_b();
        let out = squeeze_var(
            &fixture_b(),
            &vec![4.0; 40],
            Some(&cov),
            Some(SqueezeVarParams {
                robust: true,
                ..Default::default()
            }),
        )
        .unwrap();
        assert_slices_close(&out.var_prior, &B_ROBUST_TREND_VAR_PRIOR, 1e-12);
        for (k, &i) in [6_usize, 22, 30].iter().enumerate() {
            assert_relative_eq!(
                out.df_prior[i],
                B_ROBUST_TREND_SHRUNK[k],
                max_relative = 1e-12
            );
        }
        for (i, &v) in out.df_prior.iter().enumerate() {
            if ![6, 22, 30].contains(&i) {
                assert!(v.is_infinite(), "gene {i} should have an infinite prior");
            }
        }
        // An infinite prior must send the posterior onto the prior exactly.
        for (i, (&post, &prior)) in out.var_post.iter().zip(&out.var_prior).enumerate() {
            if ![6, 22, 30].contains(&i) {
                assert_eq!(post, prior, "gene {i}");
            }
        }
    }

    /// `s <- tightS(x, df=rep(4,40), robust=TRUE, winsor.tail.p=c(0.2,0.25), legacy=TRUE);
    ///  cat(sprintf("%.17g %.17g %.17g", s$var.prior, s$df.prior[1], s$df.prior[7]))`
    const B_ROBUST_WIDE_SCALE: f64 = 2.3500199318927755;
    const B_ROBUST_WIDE_DF2: f64 = 25.050701533836126;
    const B_ROBUST_WIDE_OUTLIER: f64 = 0.3596862065996353;

    #[test]
    fn test_squeeze_var_robust_honours_winsor_tail_p() {
        let out = squeeze_var(
            &fixture_b(),
            &vec![4.0; 40],
            None,
            Some(SqueezeVarParams {
                robust: true,
                winsor_tail_p: (0.2, 0.25),
                span: None,
                legacy: None,
            }),
        )
        .unwrap();
        assert_relative_eq!(out.var_prior[0], B_ROBUST_WIDE_SCALE, max_relative = 1e-12);
        assert_relative_eq!(out.df_prior[0], B_ROBUST_WIDE_DF2, max_relative = 1e-12);
        assert_relative_eq!(out.df_prior[6], B_ROBUST_WIDE_OUTLIER, max_relative = 1e-12);
    }

    /// `Rscript -e '... s <- squeezeVar(x, df=rep(4,40), robust=TRUE,
    ///  winsor.tail.p=c(0.005,0.005), legacy=TRUE); cat(sprintf("%.17g %.17g",
    ///  s$var.prior, s$df.prior[1]))'`. Both proportions are below `1/40`, so
    /// nothing would be clipped and limma returns the non-robust fit.
    const B_TINY_WINSOR_SCALE: f64 = 1.8682476400861205;
    const B_TINY_WINSOR_DF2: f64 = 3.305496787977293;

    #[test]
    fn test_squeeze_var_robust_falls_back_when_nothing_would_be_winsorised() {
        let out = squeeze_var(
            &fixture_b(),
            &vec![4.0; 40],
            None,
            Some(SqueezeVarParams {
                robust: true,
                winsor_tail_p: (0.005, 0.005),
                span: None,
                legacy: None,
            }),
        )
        .unwrap();
        assert_relative_eq!(out.var_prior[0], B_TINY_WINSOR_SCALE, max_relative = 1e-12);
        for &v in &out.df_prior {
            assert_relative_eq!(v, B_TINY_WINSOR_DF2, max_relative = 1e-12);
        }
    }

    // -- degrees of freedom of zero --

    /// `Rscript -e '... d <- rep(4,40); d[c(3,17,29)] <- 0;
    ///  s <- squeezeVar(x, df=d, legacy=TRUE); cat(sprintf("%.17g %.17g", s$df.prior, s$var.prior))'`
    const B_ZERODF_DF_PRIOR: f64 = 3.001_027_369_685_683;
    const B_ZERODF_VAR_PRIOR: f64 = 1.8360659952892042;

    /// Same command, `cat(s$var.post, sep=",")`.
    const B_ZERODF_VAR_POST: [f64; 40] = [
        1.1262753147565674,
        1.456_583_979_169_605,
        1.8360659952892042,
        2.117_201_307_995_68,
        2.4475099724087177,
        2.7778186368217557,
        23.640_828_062_045_24,
        1.1976934043593863,
        1.5280020687724238,
        1.8583107331854616,
        2.188_619_397_598_499,
        2.5189280620115366,
        2.8492367264245746,
        0.938_802_829_549_167_7,
        1.2691114939622052,
        1.5994201583752428,
        1.8360659952892042,
        2.260037487201318,
        2.5903461516143556,
        2.9206548160273935,
        1.0102209191519866,
        1.3405295835650242,
        35.067_722_398_496_27,
        2.0011469123910994,
        2.331455576804137,
        2.6617642412171745,
        2.9920729056302124,
        1.0816390087548056,
        1.8360659952892042,
        1.7422563375808808,
        15.070657309706968,
        2.402_873_666_406_956,
        2.733_182_330_819_994,
        0.822_748_433_944_587,
        1.1530570983576245,
        1.483365762770662,
        1.8136744271836998,
        2.1439830915967373,
        2.474_291_756_009_775,
        2.8046004204228128,
    ];

    #[test]
    fn test_squeeze_var_with_zero_df_matches_limma() {
        let mut df = vec![4.0; 40];
        for i in [2, 16, 28] {
            df[i] = 0.0;
        }
        let out = squeeze_var(&fixture_b(), &df, None, None).unwrap();
        assert_relative_eq!(out.df_prior[0], B_ZERODF_DF_PRIOR, max_relative = 1e-12);
        assert_relative_eq!(out.var_prior[0], B_ZERODF_VAR_PRIOR, max_relative = 1e-12);
        assert_slices_close(&out.var_post, &B_ZERODF_VAR_POST, 1e-12);
        // A gene with no residual degrees of freedom gets the prior outright.
        for i in [2, 16, 28] {
            assert_relative_eq!(out.var_post[i], B_ZERODF_VAR_PRIOR, max_relative = 1e-12);
        }
    }

    /// `d <- rep(4,40); d[c(3,17,29)] <- 0;
    ///  s <- tightS(x, df=d, robust=TRUE, legacy=TRUE);
    ///  cat(sprintf("%.17g %.17g", s$var.prior, s$df.prior[1]));
    ///  cat(s$df.prior[c(7,23,31)], sep=",")`
    const B_ROBUST_ZERODF_SCALE: f64 = 2.2031641721288566;
    const B_ROBUST_ZERODF_DF2: f64 = 17.940854604905585;
    const B_ROBUST_ZERODF_OUTLIERS: [f64; 3] =
        [0.34184209142922345, 0.34036007756885583, 0.3636949709480071];

    #[test]
    fn test_squeeze_var_robust_with_zero_df_matches_limma() {
        let mut df = vec![4.0; 40];
        for i in [2, 16, 28] {
            df[i] = 0.0;
        }
        let out = squeeze_var(
            &fixture_b(),
            &df,
            None,
            Some(SqueezeVarParams {
                robust: true,
                ..Default::default()
            }),
        )
        .unwrap();
        assert_relative_eq!(
            out.var_prior[0],
            B_ROBUST_ZERODF_SCALE,
            max_relative = 1e-12
        );
        for (k, &i) in [6_usize, 22, 30].iter().enumerate() {
            assert_relative_eq!(
                out.df_prior[i],
                B_ROBUST_ZERODF_OUTLIERS[k],
                max_relative = 1e-12
            );
        }
        // The dropped genes take the shared df2, not a shrunken one.
        for i in [2, 16, 28] {
            assert_relative_eq!(out.df_prior[i], B_ROBUST_ZERODF_DF2, max_relative = 1e-12);
        }
    }

    // -- zero variance --

    /// `Rscript -e '... x[5] <- 0; x[19] <- 0;
    ///  s <- suppressWarnings(squeezeVar(x, df=rep(4,40), legacy=TRUE));
    ///  cat(sprintf("%.17g %.17g", s$df.prior, s$var.prior)); cat(s$var.post[c(5,19)], sep=",")'`
    const B_ZEROVAR_DF_PRIOR: f64 = 0.800_079_544_363_124_5;
    const B_ZEROVAR_VAR_PRIOR: f64 = 0.276_423_301_654_354_7;
    const B_ZEROVAR_VAR_POST_AT_ZERO: f64 = 0.04607436755890474;

    #[test]
    fn test_squeeze_var_with_zero_variance_matches_limma() {
        let mut x = fixture_b();
        x[4] = 0.0;
        x[18] = 0.0;
        let out = squeeze_var(&x, &vec![4.0; 40], None, None).unwrap();
        assert_relative_eq!(out.df_prior[0], B_ZEROVAR_DF_PRIOR, max_relative = 1e-12);
        assert_relative_eq!(out.var_prior[0], B_ZEROVAR_VAR_PRIOR, max_relative = 1e-12);
        for i in [4, 18] {
            assert_relative_eq!(
                out.var_post[i],
                B_ZEROVAR_VAR_POST_AT_ZERO,
                max_relative = 1e-12
            );
        }
    }

    /// `x[5] <- 0; x[19] <- 0;
    ///  s <- suppressWarnings(tightS(x, df=rep(4,40), robust=TRUE, legacy=TRUE));
    ///  cat(sprintf("%.17g %.17g %.17g", s$var.prior, s$df.prior[1], s$df.prior[5]))`
    const B_ZEROVAR_ROBUST_SCALE: f64 = 0.9409970501549559;
    const B_ZEROVAR_ROBUST_DF2: f64 = 1.517197722874929;
    const B_ZEROVAR_ROBUST_AT_ZERO: f64 = 1.7402266327836557;

    #[test]
    fn test_squeeze_var_robust_with_zero_variance_matches_limma() {
        let mut x = fixture_b();
        x[4] = 0.0;
        x[18] = 0.0;
        let out = squeeze_var(
            &x,
            &vec![4.0; 40],
            None,
            Some(SqueezeVarParams {
                robust: true,
                ..Default::default()
            }),
        )
        .unwrap();
        assert_relative_eq!(
            out.var_prior[0],
            B_ZEROVAR_ROBUST_SCALE,
            max_relative = 1e-12
        );
        assert_relative_eq!(out.df_prior[0], B_ZEROVAR_ROBUST_DF2, max_relative = 1e-12);
        assert_relative_eq!(
            out.df_prior[4],
            B_ZEROVAR_ROBUST_AT_ZERO,
            max_relative = 1e-12
        );
    }

    // -- infinite prior degrees of freedom --

    /// `Rscript -e 'suppressMessages(library(limma)); s <- squeezeVar(rep(0.5,20), df=rep(4,20),
    ///  legacy=TRUE); cat(s$df.prior, s$var.prior, s$var.post[1])'`
    #[test]
    fn test_constant_variances_give_an_infinite_prior() {
        let out = squeeze_var(&[0.5; 20], &[4.0; 20], None, None).unwrap();
        assert!(out.df_prior[0].is_infinite());
        assert_eq!(out.var_prior[0], 0.5);
        // The posterior must be the prior bit for bit, not merely close to it.
        for &v in &out.var_post {
            assert_eq!(v, 0.5);
        }
    }

    /// `Rscript -e 'suppressMessages(library(limma)); j <- 1:60;
    ///  x <- 1 + (((j*17) %% 7) - 3)/4096; r <- limma:::fitFDistRobustly(x, df1=rep(6,60));
    ///  cat(sprintf("%.17g %.17g", r$scale, r$df2))'`
    const D_ROBUST_SCALE: f64 = 1.1942105542067107;

    #[test]
    fn test_robust_fit_takes_the_infinite_branch_on_flat_variances() {
        let x = fixture_d();
        let (scale, shrunk) = fit_f_dist_robustly(&x, &[6.0], None, (0.05, 0.1)).unwrap();
        assert_relative_eq!(scale[0], D_ROBUST_SCALE, max_relative = 1e-12);
        for &v in &shrunk {
            assert!(v.is_infinite());
        }

        let out = squeeze_var(
            &x,
            &vec![6.0; 60],
            None,
            Some(SqueezeVarParams {
                robust: true,
                ..Default::default()
            }),
        )
        .unwrap();
        for &v in &out.var_post {
            assert_eq!(v, out.var_prior[0]);
        }
    }

    // -- small inputs --

    #[test]
    fn test_fewer_than_three_genes_returns_the_input_unmoderated() {
        let out = squeeze_var(&[0.5, 2.0], &[4.0], None, None).unwrap();
        assert_eq!(out.var_post, vec![0.5, 2.0]);
        assert_eq!(out.var_prior, vec![0.5, 2.0]);
        assert_eq!(out.df_prior, vec![0.0]);
    }

    #[test]
    fn test_fit_f_dist_with_a_single_usable_gene_gives_zero_prior_df() {
        let (scale, df2) = fit_f_dist(&[0.25], &[4.0]).unwrap();
        assert_eq!(scale, 0.25);
        assert_eq!(df2, 0.0);

        // Only one gene has positive degrees of freedom.
        let (scale, df2) = fit_f_dist(&[0.25, 1.0, 2.0], &[4.0, 0.0, 0.0]).unwrap();
        assert_eq!(scale, 0.25);
        assert_eq!(df2, 0.0);
    }

    // -- error branches --

    #[test]
    fn test_empty_input_is_an_error() {
        assert!(matches!(
            squeeze_var(&[], &[4.0], None, None),
            Err(EdgeErrors::InvalidArgument(_))
        ));
        assert!(matches!(
            fit_f_dist(&[], &[4.0]),
            Err(EdgeErrors::InvalidArgument(_))
        ));
        assert!(matches!(
            fit_f_dist_trend(&[], &[4.0], &[], None),
            Err(EdgeErrors::InvalidArgument(_))
        ));
        assert!(matches!(
            fit_f_dist_robustly(&[], &[4.0], None, (0.05, 0.1)),
            Err(EdgeErrors::InvalidArgument(_))
        ));
    }

    #[test]
    fn test_mismatched_df_length_is_an_error() {
        assert!(matches!(
            squeeze_var(&[1.0, 2.0, 3.0], &[4.0, 4.0], None, None),
            Err(EdgeErrors::LengthMismatch { name: "df", .. })
        ));
        assert!(matches!(
            fit_f_dist(&[1.0, 2.0, 3.0], &[4.0, 4.0]),
            Err(EdgeErrors::LengthMismatch { name: "df1", .. })
        ));
    }

    #[test]
    fn test_mismatched_covariate_length_is_an_error() {
        assert!(matches!(
            squeeze_var(&[1.0, 2.0, 3.0], &[4.0], Some(&[1.0, 2.0]), None),
            Err(EdgeErrors::LengthMismatch {
                name: "covariate",
                ..
            })
        ));
        assert!(matches!(
            fit_f_dist_trend(&[1.0, 2.0, 3.0], &[4.0], &[1.0, 2.0], None),
            Err(EdgeErrors::LengthMismatch {
                name: "covariate",
                ..
            })
        ));
        assert!(matches!(
            fit_f_dist_robustly(&[1.0, 2.0, 3.0], &[4.0], Some(&[1.0, 2.0]), (0.05, 0.1)),
            Err(EdgeErrors::LengthMismatch {
                name: "covariate",
                ..
            })
        ));
    }

    #[test]
    fn test_negative_or_non_finite_variance_is_an_error() {
        for bad in [-1.0, f64::NAN, f64::INFINITY] {
            let x = [1.0, bad, 3.0, 4.0];
            assert!(
                matches!(
                    squeeze_var(&x, &[4.0], None, None),
                    Err(EdgeErrors::InvalidArgument(_))
                ),
                "variance {bad} should have been rejected"
            );
        }
    }

    #[test]
    fn test_negative_or_non_finite_df_is_an_error() {
        for bad in [-1.0, f64::NAN, f64::INFINITY] {
            let df = [4.0, bad, 4.0, 4.0];
            assert!(
                matches!(
                    squeeze_var(&[1.0, 2.0, 3.0, 4.0], &df, None, None),
                    Err(EdgeErrors::InvalidArgument(_))
                ),
                "df {bad} should have been rejected"
            );
        }
    }

    #[test]
    fn test_nan_covariate_is_an_error() {
        let cov = [1.0, f64::NAN, 3.0];
        assert!(matches!(
            squeeze_var(&[1.0, 2.0, 3.0], &[4.0], Some(&cov), None),
            Err(EdgeErrors::InvalidArgument(_))
        ));
    }

    #[test]
    fn test_non_finite_covariate_is_an_error_for_the_robust_fit() {
        let cov: Vec<f64> = (0..40)
            .map(|i| if i == 3 { f64::INFINITY } else { i as f64 })
            .collect();
        assert!(matches!(
            squeeze_var(
                &fixture_b(),
                &vec![4.0; 40],
                Some(&cov),
                Some(SqueezeVarParams {
                    robust: true,
                    ..Default::default()
                })
            ),
            Err(EdgeErrors::InvalidArgument(_))
        ));
    }

    #[test]
    fn test_bad_winsor_tail_p_is_an_error() {
        for bad in [(-0.1, 0.1), (0.05, 0.5), (0.6, 0.1), (f64::NAN, 0.1)] {
            assert!(
                matches!(
                    squeeze_var(
                        &fixture_b(),
                        &vec![4.0; 40],
                        None,
                        Some(SqueezeVarParams {
                            robust: true,
                            winsor_tail_p: bad,
                            span: None,
                            legacy: None
                        })
                    ),
                    Err(EdgeErrors::InvalidArgument(_))
                ),
                "winsor_tail_p {bad:?} should have been rejected"
            );
        }
    }

    #[test]
    fn test_bad_span_is_an_error() {
        for bad in [0.0, -0.5, 1.5, f64::NAN] {
            assert!(
                matches!(
                    squeeze_var(
                        &fixture_b(),
                        &vec![4.0; 40],
                        None,
                        Some(SqueezeVarParams {
                            robust: false,
                            winsor_tail_p: (0.05, 0.1),
                            span: Some(bad),
                            legacy: None
                        })
                    ),
                    Err(EdgeErrors::InvalidArgument(_))
                ),
                "span {bad} should have been rejected"
            );
        }
    }

    #[test]
    fn test_all_zero_degrees_of_freedom_is_an_error() {
        // Every gene is uninformative, so there is nothing to fit a prior to.
        assert!(fit_f_dist(&[1.0, 2.0, 3.0], &[0.0, 0.0, 0.0]).is_err());
    }

    // -- trended fits with some genes dropped --

    /// `Rscript -e 'suppressMessages(library(limma)); i <- 1:40;
    ///  x <- (((i*37) %% 251)+1)/64; x[7] <- 40; x[23] <- 60; x[31] <- 25;
    ///  a <- ((i*29) %% 97)/8; d <- rep(4,40); d[c(3,17,29)] <- 0;
    ///  s <- squeezeVar(x, df=d, covariate=a, legacy=TRUE);
    ///  cat(sprintf("%.17g", s$df.prior), "\n"); cat(s$var.prior, sep=",")'`
    const B_TREND_ZERODF_DF_PRIOR: f64 = 3.1428781403090924;
    const B_TREND_ZERODF_VAR_PRIOR: [f64; 40] = [
        1.093287070145291,
        1.809109212366991,
        3.1695976440542672,
        1.4961864064084491,
        1.2740702863969318,
        2.9264700283297334,
        2.3815894804427904,
        1.032235139796427,
        2.4219204766484834,
        3.252655312361141,
        1.116778949797686,
        1.7444188196732309,
        3.15474204792806,
        1.5597445620654493,
        1.2361400035569214,
        2.888530494276931,
        2.504499838754266,
        1.0253943465690296,
        2.357027456392529,
        3.2458117931091035,
        1.1436967402780736,
        1.681767030539927,
        3.1381481763422743,
        1.6282724259646797,
        1.2012295261536194,
        2.8478006100933224,
        2.634365811751067,
        1.0219796133102823,
        2.2900015399385154,
        3.2387961322379866,
        1.1741260384330265,
        1.621382550891335,
        3.1196645521474418,
        1.701986005663366,
        1.169348321864014,
        2.804251325390985,
        2.7712878609845677,
        1.0220851562051536,
        2.221426882044691,
        3.2314233934405734,
    ];

    /// Same command, `cat(s$var.post, sep=",")`.
    const B_TREND_ZERODF_VAR_POST: [f64; 40] = [
        0.813547133199552,
        1.4522591024535012,
        3.1695976440542672,
        1.962070649286237,
        2.1880882391279237,
        3.238895894044488,
        23.447837444140717,
        0.8566839775682209,
        1.7918968617700142,
        2.481170605328731,
        1.86513053802427,
        2.4650421622806604,
        3.4093357526736754,
        0.8350397376824595,
        1.0164022475156618,
        2.0672058318833684,
        2.504499838754266,
        1.894920956392324,
        2.804590765136243,
        3.519406384697505,
        0.7219778054255823,
        1.2824786672029191,
        34.98069159187206,
        1.906439049713815,
        2.042288519094182,
        3.090531554898673,
        3.3203689405659866,
        0.7384218634151563,
        2.2900015399385154,
        2.3613228776430644,
        14.516576234866568,
        2.2971563358583267,
        3.28015193116923,
        0.7838765414061916,
        0.8732641320020631,
        1.916373193217012,
        2.2256182629071732,
        1.779715241568084,
        2.631176623581593,
        3.399325799540674,
    ];

    #[test]
    fn test_squeeze_var_trended_with_zero_df_tracks_limma() {
        // The fit itself is exact: the basis is built over the surviving
        // covariates, so limma's knots are reproduced. Only the three dropped
        // genes are read off the fitted curve by interpolation instead of by
        // limma's `predict.ns`, and those sit inside the covariate range where
        // the spline is a cubic rather than a line.
        let cov = covariate_b();
        let mut df = vec![4.0; 40];
        for i in [2, 16, 28] {
            df[i] = 0.0;
        }
        let out = squeeze_var(&fixture_b(), &df, Some(&cov), None).unwrap();
        assert_relative_eq!(
            out.df_prior[0],
            B_TREND_ZERODF_DF_PRIOR,
            max_relative = 1e-12
        );
        for i in 0..40 {
            let tol = if [2, 16, 28].contains(&i) {
                2e-3
            } else {
                1e-12
            };
            assert_relative_eq!(
                out.var_prior[i],
                B_TREND_ZERODF_VAR_PRIOR[i],
                max_relative = tol
            );
            assert_relative_eq!(
                out.var_post[i],
                B_TREND_ZERODF_VAR_POST[i],
                max_relative = tol
            );
        }
    }

    /// Same fixture, `s <- squeezeVar(x, df=d, covariate=a, robust=TRUE, legacy=TRUE);
    ///  cat(s$var.prior, sep=",")`. The prior degrees of freedom come back
    /// infinite for every gene but the three outliers.
    const B_ROBUST_TREND_ZERODF_VAR_PRIOR: [f64; 40] = [
        1.4333622450698638,
        2.5136312844707507,
        3.4635126858697296,
        1.866767433369885,
        1.9544625952714154,
        3.2323044560151404,
        3.13685328484227,
        1.5428088570392688,
        2.6849287924221663,
        3.55523767986531,
        1.4658742706464782,
        2.454592962182901,
        3.449378536691651,
        1.9674417532494404,
        1.9169967731877045,
        3.1708692309376967,
        3.3075073569912745,
        1.4997577733903524,
        2.6568219418972934,
        3.556394698841825,
        1.483800368786478,
        2.40465716256889,
        3.4323358503679278,
        2.0710253843357904,
        1.8819153740480852,
        3.104511743149066,
        3.4874455140803566,
        1.4602023584528363,
        2.6356497747742536,
        3.5545140440586525,
        1.5103304449849393,
        2.3464067797790813,
        3.4189103519815345,
        2.179707101052931,
        1.8463856775751422,
        3.0413936864461837,
        3.6774195208558744,
        1.4233480539718921,
        2.614646328277017,
        3.549542156913126,
    ];

    #[test]
    fn test_squeeze_var_robust_trended_with_zero_df_tracks_limma() {
        let cov = covariate_b();
        let mut df = vec![4.0; 40];
        for i in [2, 16, 28] {
            df[i] = 0.0;
        }
        let out = squeeze_var(
            &fixture_b(),
            &df,
            Some(&cov),
            Some(SqueezeVarParams {
                robust: true,
                ..Default::default()
            }),
        )
        .unwrap();
        for (i, &want) in B_ROBUST_TREND_ZERODF_VAR_PRIOR.iter().enumerate() {
            let tol = if [2, 16, 28].contains(&i) {
                2e-3
            } else {
                1e-12
            };
            assert_relative_eq!(out.var_prior[i], want, max_relative = tol);
        }
        for (i, &v) in out.df_prior.iter().enumerate() {
            if [6, 22, 30].contains(&i) {
                assert!(v < 1e-5, "gene {i} should be an outlier, got {v:e}");
            } else {
                assert!(v.is_infinite(), "gene {i} should have an infinite prior");
            }
        }
    }

    /// The same fixture with the dropped genes moved to the ends of the
    /// covariate range, which is where the knot-placement deviation
    /// documented on [`fit_f_dist_trend`] actually bites: limma's boundary
    /// knots then sit at the surviving covariates rather than at all of them.
    ///
    /// `Rscript -e 'suppressMessages(library(limma)); i <- 1:40;
    ///  x <- (((i*37) %% 251)+1)/64; x[7] <- 40; x[23] <- 60; x[31] <- 25;
    ///  a <- ((i*29) %% 97)/8; d <- rep(4,40); d[c(37,10,27)] <- 0;
    ///  s <- squeezeVar(x, df=d, covariate=a, legacy=TRUE);
    ///  cat(sprintf("%.17g", s$df.prior), "\n"); cat(s$var.prior, sep=",")'`
    const B_TREND_EDGE_DROP_DF_PRIOR: f64 = 3.1133565191151713;
    const B_TREND_EDGE_DROP_VAR_PRIOR: [f64; 40] = [
        1.1351392212183793,
        1.6259983592918905,
        3.2030895998182918,
        1.5024948284759898,
        1.2174047384019633,
        2.694290075307271,
        2.217266528757394,
        1.0503433271810356,
        2.1359568377083917,
        3.6559084086236546,
        1.1584457342962922,
        1.5754758393831132,
        3.1536027723671953,
        1.5570273656314104,
        1.1893481618062514,
        2.6409046242394476,
        2.31027252755222,
        1.0483627938149591,
        2.0775047082558817,
        3.6037124755635026,
        1.1845712505241406,
        1.5268769976093624,
        3.1039609894905236,
        1.615135524840441,
        1.1637403076684496,
        2.5869588372036225,
        2.407219452267836,
        1.0493507247734664,
        2.01877998407745,
        3.552242266978073,
        1.213563940539991,
        1.4803154007539074,
        3.0540953483565847,
        1.676904040679061,
        1.14060019322546,
        2.5324475693162083,
        2.508234601013182,
        1.0532518576907344,
        1.9601021721177017,
        3.50139194976571,
    ];

    #[test]
    fn test_squeeze_var_trended_with_dropped_edge_genes_tracks_limma() {
        // The fit is exact for every surviving gene however the dropped ones
        // are placed. The three dropped here are the smallest, second smallest
        // and largest covariate, so they are the worst case for reading the
        // trend off the fitted values: two of them are extrapolated.
        let cov = covariate_b();
        let mut df = vec![4.0; 40];
        for i in [36, 9, 26] {
            df[i] = 0.0;
        }
        let out = squeeze_var(&fixture_b(), &df, Some(&cov), None).unwrap();
        assert_relative_eq!(
            out.df_prior[0],
            B_TREND_EDGE_DROP_DF_PRIOR,
            max_relative = 1e-12
        );
        for (i, &want) in B_TREND_EDGE_DROP_VAR_PRIOR.iter().enumerate() {
            let tol = if [36, 9, 26].contains(&i) {
                1e-2
            } else {
                1e-12
            };
            assert_relative_eq!(out.var_prior[i], want, max_relative = tol);
        }
    }

    // -- scale --

    #[test]
    fn test_robust_fit_over_the_parallel_threshold_stays_consistent() {
        // Past PARALLEL_THRESHOLD the tail probabilities fan out over rayon.
        // There is no limma fixture at this size worth embedding, so the check
        // is the invariant every posterior has to satisfy: it is a weighted
        // average of the gene's own variance and the prior, so it lies between
        // them.
        let n = 5_000;
        assert!(n > PARALLEL_THRESHOLD);
        let var: Vec<f64> = (1..=n)
            .map(|i| (((i * 37) % 4093) + 1) as f64 / 1024.0)
            .collect();
        let cov: Vec<f64> = (1..=n).map(|i| ((i * 29) % 997) as f64 / 64.0).collect();

        for covariate in [None, Some(cov.as_slice())] {
            let out = squeeze_var(
                &var,
                &vec![5.0; n],
                covariate,
                Some(SqueezeVarParams {
                    robust: true,
                    ..Default::default()
                }),
            )
            .unwrap();
            assert_eq!(out.var_post.len(), n);
            assert_eq!(out.df_prior.len(), n);
            for (i, &v) in var.iter().enumerate() {
                let (lo, hi) = if v < out.var_prior[i] {
                    (v, out.var_prior[i])
                } else {
                    (out.var_prior[i], v)
                };
                assert!(
                    out.var_post[i] >= lo - 1e-12 && out.var_post[i] <= hi + 1e-12,
                    "gene {i}: posterior {} outside [{lo}, {hi}]",
                    out.var_post[i]
                );
                assert!(out.df_prior[i] > 0.0);
            }
        }
    }

    // -- component checks --

    /// `Rscript -e 'g <- statmod::gauss.quad.prob(128, dist="uniform");
    ///  cat(sprintf("%.17g", c(g$nodes[1], g$nodes[64], g$nodes[128], sum(g$weights),
    ///  sum(g$weights*g$nodes), sum(g$weights*g$nodes^2))), sep=",")'`
    #[test]
    fn test_gauss_legendre_integrates_polynomials_exactly() {
        let (nodes, weights) = gauss_legendre_unit(QUAD_NODES);
        assert_eq!(nodes.len(), QUAD_NODES);
        assert_relative_eq!(weights.iter().sum::<f64>(), 1.0, max_relative = 1e-14);
        // Moments of the uniform distribution on (0, 1): 1/(k + 1).
        for k in 1..=6_u32 {
            let m: f64 = nodes
                .iter()
                .zip(&weights)
                .map(|(&x, &w)| w * x.powi(k as i32))
                .sum();
            assert_relative_eq!(m, 1.0 / (k as f64 + 1.0), max_relative = 1e-12);
        }
        // Symmetric about 1/2 and in increasing order.
        for i in 0..QUAD_NODES / 2 {
            assert_relative_eq!(
                nodes[i] + nodes[QUAD_NODES - 1 - i],
                1.0,
                max_relative = 1e-15
            );
            assert!(nodes[i] < nodes[i + 1]);
        }
    }

    #[test]
    fn test_f_density_and_quantile_are_consistent() {
        // The density integrates to the quantile spacing under the trapezium
        // rule closely enough to catch a wrong normalising constant.
        for (df1, df2) in [(4.0, 10.0), (1.0, 3.0), (8.0, f64::INFINITY)] {
            let lo = f_quantile(0.25, df1, df2).unwrap();
            let hi = f_quantile(0.75, df1, df2).unwrap();
            let steps = 20_000;
            let h = (hi - lo) / steps as f64;
            let mut area = 0.0;
            for k in 0..=steps {
                let x = lo + k as f64 * h;
                let w = if k == 0 || k == steps { 0.5 } else { 1.0 };
                area += w * f_density(x, df1, df2) * h;
            }
            assert_relative_eq!(area, 0.5, max_relative = 1e-6);
        }
    }

    #[test]
    fn test_f_isf_inverts_the_survival_function() {
        for p in [0.5, 0.1, 1e-3, 1e-8, 1e-30] {
            let x = f_isf(p, 4.0, 9.0).unwrap();
            assert_relative_eq!(f_sf(x, 4.0, 9.0).unwrap(), p, max_relative = 1e-12);
        }
    }

    #[test]
    fn test_posterior_var_handles_a_mixed_finite_and_infinite_prior() {
        let var = [1.0, 2.0, 3.0];
        let df = [4.0];
        let prior = [0.5, 0.5, 0.5];
        let df_prior = [f64::INFINITY, 4.0, 0.0];
        let out = posterior_var(&var, &df, &prior, &df_prior);
        assert_eq!(out[0], 0.5);
        assert_relative_eq!(out[1], (4.0 * 2.0 + 4.0 * 0.5) / 8.0, max_relative = 1e-15);
        assert_relative_eq!(out[2], 3.0, max_relative = 1e-15);
    }

    // -- fitFDistUnequalDF1 --
    //
    // Every reference below comes from limma 3.66 under R 4.5, generated with
    // the command in the doc comment above it. The two fixtures are set up once
    // for the whole block:
    //
    //   i <- 1:24; xa <- (((i*37) %% 97)+1)/64;  ca <- ((i*53) %% 101)/8
    //   dfa <- rep(c(2,4,7,11),6)
    //   j <- 1:40; xb <- (((j*37) %% 251)+1)/64
    //   xb[7] <- 40; xb[23] <- 60; xb[31] <- 25;  cb <- ((j*29) %% 97)/8
    //   dfb <- rep(c(3,5,8,12),10)

    /// Relative tolerance for the unequal-`df1` references.
    ///
    /// Two orders of magnitude looser than the legacy fits get, and the reason
    /// is not this module. `crate::numeric::gamma::logmdigamma` is a more
    /// accurate function than statmod's: `logmdigamma(1)` should be the
    /// Euler-Mascheroni constant `0.5772156649015329`, this crate returns
    /// `0.5772156649015328` and statmod returns `0.5772156649015084`, wrong in
    /// the fourteenth digit. `ln_gamma` differs from R's by a similar margin.
    /// Those feed `emean`, and `emean` is exponentiated into the scale, so a
    /// relative error of `1e-13` in `emean` is a relative error of `1e-13` in
    /// the scale before anything else happens. Matching limma any closer would
    /// mean reproducing statmod's error, not fixing anything.
    ///
    /// Worst observed disagreement on any scale, prior or posterior variance in
    /// this block is `3e-11`.
    const UNEQUAL_TOL: f64 = 1e-10;

    /// Relative tolerance where the likelihood is flat enough to amplify that.
    ///
    /// The log likelihood is quadratic in `par` near its maximum with a small
    /// curvature, so an error of `eps` in the objective moves the maximiser by
    /// far more than `eps`. Substituting R's own `emean` into the objective
    /// here takes the `df2` disagreement on fixture A from `1.3e-11` to
    /// `7.4e-12`, which is what leaves `logmdigamma` and `lgamma` as the
    /// binding constraint rather than `optimize`'s `tol` or the search.
    ///
    /// Worst observed disagreement on a `df2` below ten is `1.6e-10`. Larger
    /// `df2` is flatter still: the six hundred gene robust fixture lands on
    /// `123.71`, where the disagreement is `4.1e-09`.
    const UNEQUAL_FLAT_TOL: f64 = 1e-8;

    /// Unequal residual degrees of freedom for fixture A.
    fn df_a() -> Vec<f64> {
        (0..24).map(|i| [2.0, 4.0, 7.0, 11.0][i % 4]).collect()
    }

    /// Unequal residual degrees of freedom for fixture B.
    fn df_b() -> Vec<f64> {
        (0..40).map(|i| [3.0, 5.0, 8.0, 12.0][i % 4]).collect()
    }

    /// Prior weights for fixture A, `((i * 13) mod 7 + 1) / 8`.
    fn weights_a() -> Vec<f64> {
        (1..=24)
            .map(|i| (((i * 13) % 7) + 1) as f64 / 8.0)
            .collect()
    }

    /// Six hundred variances, `(((k * 37) mod 1009) + 1) / 64`, the only fixture
    /// large enough for `chooseLowessSpan(n, small.n = 500)` to drop below one.
    fn fixture_l() -> Vec<f64> {
        (1..=600)
            .map(|k| (((k * 37) % 1009) + 1) as f64 / 64.0)
            .collect()
    }

    /// Covariate for fixture L, `((k * 53) mod 601) / 16`.
    fn covariate_l() -> Vec<f64> {
        (1..=600).map(|k| ((k * 53) % 601) as f64 / 16.0).collect()
    }

    /// Residual degrees of freedom for fixture L.
    fn df_l() -> Vec<f64> {
        (0..600)
            .map(|i| [2.0, 4.0, 7.0, 11.0, 5.0, 9.0][i % 6])
            .collect()
    }

    /// The `par` R's `optimize` lands on when the objective is flat or pushes
    /// against the upper bound, and the `df2` it implies.
    ///
    /// `Rscript -e 'p <- optimize(function(p) 0, c(0.5, 0.9998))$minimum;
    ///  cat(sprintf("%.17g", 2*p/(1-p)))'`
    const FLAT_OBJECTIVE_DF2: f64 = 8134.844780382662;

    /// `Rscript -e 'suppressMessages(library(limma)); i <- 1:24;
    ///  xa <- (((i*37) %% 97)+1)/64; dfa <- rep(c(2,4,7,11),6);
    ///  r <- limma:::fitFDistUnequalDF1(xa, df1=dfa);
    ///  cat(sprintf("%.17g %.17g", r$scale, r$df2))'`
    const A_UNEQUAL_SCALE: f64 = 0.5184343518289642;
    const A_UNEQUAL_DF2: f64 = 9.397745433510156;

    #[test]
    fn test_fit_f_dist_unequal_df1_matches_limma() {
        let fit = fit_f_dist_unequal_df1(&fixture_a(), &df_a(), None, None, false, None).unwrap();
        assert_eq!(fit.scale.len(), 1);
        assert_relative_eq!(fit.scale[0], A_UNEQUAL_SCALE, max_relative = UNEQUAL_TOL);
        assert_relative_eq!(fit.df2, A_UNEQUAL_DF2, max_relative = UNEQUAL_FLAT_TOL);
        assert!(fit.df2_shrunk.is_none());
        assert!(fit.df2_outlier.is_none());
    }

    /// `Rscript -e '... r <- limma:::fitFDistUnequalDF1(xa, df1=dfa, covariate=ca);
    ///  cat(sprintf("%.17g", r$df2), "\n"); cat(sprintf("%.17g", r$scale), sep=",")'`
    const A_UNEQUAL_TREND_DF2: f64 = 9.001725728292067;
    const A_UNEQUAL_TREND_SCALE: [f64; 24] = [
        0.4984351761472433,
        0.5259190759879766,
        0.5220579477174045,
        0.5172782059081166,
        0.5407967205854338,
        0.5101683139607902,
        0.5579627148258335,
        0.5037522439866956,
        0.5753054276830244,
        0.4972336242762296,
        0.5933503701444417,
        0.49073140218864636,
        0.6123167707662234,
        0.48464653984385686,
        0.6325650467286809,
        0.4793839190198376,
        0.6541719142254177,
        0.4761910799071218,
        0.6767040827302486,
        0.4820446203847311,
        0.5319677784294495,
        0.5086669035819186,
        0.5222449750000389,
        0.5299374322002258,
    ];

    #[test]
    fn test_fit_f_dist_unequal_df1_with_covariate_matches_limma() {
        // Unequal df1 makes the lowess weights genuinely unequal, so this is
        // the weightedLowess branch of loessFit.
        let fit = fit_f_dist_unequal_df1(
            &fixture_a(),
            &df_a(),
            Some(&covariate_a()),
            None,
            false,
            None,
        )
        .unwrap();
        assert_relative_eq!(
            fit.df2,
            A_UNEQUAL_TREND_DF2,
            max_relative = UNEQUAL_FLAT_TOL
        );
        assert_slices_close(&fit.scale, &A_UNEQUAL_TREND_SCALE, UNEQUAL_TOL);
    }

    /// `Rscript -e '... j <- 1:40; xb <- (((j*37) %% 251)+1)/64;
    ///  xb[7] <- 40; xb[23] <- 60; xb[31] <- 25; cb <- ((j*29) %% 97)/8;
    ///  r <- limma:::fitFDistUnequalDF1(xb, df1=rep(4,40), covariate=cb);
    ///  cat(sprintf("%.17g", r$df2), "\n"); cat(sprintf("%.17g", r$scale), sep=",")'`
    const B_EQUAL_DF1_TREND_DF2: f64 = 3.527946129589157;
    const B_EQUAL_DF1_TREND_SCALE: [f64; 40] = [
        1.5628785365841626,
        1.8410717969706698,
        3.0064784474370954,
        1.6047464897309185,
        1.550962277050608,
        2.530664196930327,
        1.6641285766162621,
        1.538087660141601,
        2.1446475377847625,
        3.5300284478905115,
        1.5663924718712003,
        1.8077937363480212,
        2.953894247581084,
        1.6097397188234954,
        1.5432004715623675,
        2.488620106869826,
        1.671851655024166,
        1.5401244842767179,
        2.109571135042458,
        3.467530326022418,
        1.5700489929120354,
        1.7741615853753157,
        2.902473821111272,
        1.6148765515354977,
        1.5381810706324393,
        2.447437340451914,
        1.6800178539961415,
        1.542377223892185,
        2.074969835233503,
        3.4060893876131297,
        1.5738540922468784,
        1.7400131855210033,
        2.85222400349544,
        1.6201678713903858,
        1.5351342022111514,
        2.407078925442979,
        1.688630274244667,
        1.5448212279155087,
        2.040793487554918,
        3.3457097973886314,
    ];

    #[test]
    fn test_fit_f_dist_unequal_df1_falls_back_to_plain_lowess_on_equal_weights() {
        // Equal df1 and no prior weights makes every lowess weight identical,
        // so `equal.weights.as.null` throws them away and loessFit becomes a
        // bare `stats::lowess`. Different smoother, different answer.
        let fit = fit_f_dist_unequal_df1(
            &fixture_b(),
            &[4.0; 40],
            Some(&covariate_b()),
            None,
            false,
            None,
        )
        .unwrap();
        assert_relative_eq!(
            fit.df2,
            B_EQUAL_DF1_TREND_DF2,
            max_relative = UNEQUAL_FLAT_TOL
        );
        assert_slices_close(&fit.scale, &B_EQUAL_DF1_TREND_SCALE, UNEQUAL_TOL);
    }

    /// `Rscript -e '... pw <- ((i*13) %% 7 + 1)/8;
    ///  r <- limma:::fitFDistUnequalDF1(xa, df1=dfa, prior.weights=pw);
    ///  cat(sprintf("%.17g %.17g", r$scale, r$df2))'`
    const A_WEIGHTED_SCALE: f64 = 0.4245678072068865;
    const A_WEIGHTED_DF2: f64 = 6.027707924548528;

    /// Same fixtures, `r <- limma:::fitFDistUnequalDF1(xa, df1=dfa, covariate=ca,
    /// prior.weights=pw)`.
    const A_WEIGHTED_TREND_DF2: f64 = 6.475915483944781;
    const A_WEIGHTED_TREND_SCALE: [f64; 24] = [
        0.38573627933620297,
        0.4835687382808096,
        0.42103063978713945,
        0.46075826846740037,
        0.4547533830567043,
        0.4413284036654768,
        0.4888752712215187,
        0.42391913837253875,
        0.5248949744299783,
        0.40740842545075884,
        0.5636874738132431,
        0.3918717174675754,
        0.6059157342338731,
        0.37786373396983325,
        0.652431899788031,
        0.36627090352801195,
        0.7039146802298344,
        0.3591970718950207,
        0.7603684158920332,
        0.3646918210679019,
        0.4992697757771913,
        0.4001001268268095,
        0.47396419697150305,
        0.43461067033555517,
    ];

    #[test]
    fn test_fit_f_dist_unequal_df1_with_prior_weights_matches_limma() {
        let w = weights_a();
        let fit =
            fit_f_dist_unequal_df1(&fixture_a(), &df_a(), None, None, false, Some(&w)).unwrap();
        assert_relative_eq!(fit.scale[0], A_WEIGHTED_SCALE, max_relative = UNEQUAL_TOL);
        assert_relative_eq!(fit.df2, A_WEIGHTED_DF2, max_relative = UNEQUAL_FLAT_TOL);

        let fit = fit_f_dist_unequal_df1(
            &fixture_a(),
            &df_a(),
            Some(&covariate_a()),
            None,
            false,
            Some(&w),
        )
        .unwrap();
        assert_relative_eq!(
            fit.df2,
            A_WEIGHTED_TREND_DF2,
            max_relative = UNEQUAL_FLAT_TOL
        );
        assert_slices_close(&fit.scale, &A_WEIGHTED_TREND_SCALE, UNEQUAL_TOL);
    }

    /// `Rscript -e '... r <- limma:::fitFDistUnequalDF1(xb, df1=dfb, robust=TRUE);
    ///  cat(sprintf("%.17g %.17g %.17g", r$scale, r$df2, r$df2.outlier), "\n");
    ///  cat(sprintf("%.17g", unique(r$df2.shrunk)), sep=",")'`
    const B_ROBUST_UNEQ_SCALE: f64 = 1.8836705716043158;
    const B_ROBUST_UNEQ_DF2: f64 = 5.139905765416566;
    const B_ROBUST_UNEQ_OUTLIER: f64 = 0.31992093620075585;
    const B_ROBUST_UNEQ_SHRUNK_OUTLIER: f64 = 2.1424687001362597;

    #[test]
    fn test_fit_f_dist_unequal_df1_robust_matches_limma() {
        let fit = fit_f_dist_unequal_df1(&fixture_b(), &df_b(), None, None, true, None).unwrap();
        assert_relative_eq!(
            fit.scale[0],
            B_ROBUST_UNEQ_SCALE,
            max_relative = UNEQUAL_TOL
        );
        assert_relative_eq!(fit.df2, B_ROBUST_UNEQ_DF2, max_relative = UNEQUAL_FLAT_TOL);
        assert_relative_eq!(
            fit.df2_outlier.unwrap(),
            B_ROBUST_UNEQ_OUTLIER,
            max_relative = UNEQUAL_FLAT_TOL
        );
        // The three planted outliers are the only genes shrunk off the shared
        // prior, and the monotonicity pass levels them at a common value.
        let shrunk = fit.df2_shrunk.unwrap();
        for (i, &v) in shrunk.iter().enumerate() {
            let want = if [6, 22, 30].contains(&i) {
                B_ROBUST_UNEQ_SHRUNK_OUTLIER
            } else {
                B_ROBUST_UNEQ_DF2
            };
            assert_relative_eq!(v, want, max_relative = UNEQUAL_FLAT_TOL);
        }
    }

    /// Same command with `covariate=cb`.
    const B_ROBUST_UNEQ_TREND_DF2: f64 = 5.203801107375211;
    const B_ROBUST_UNEQ_TREND_OUTLIER: f64 = 0.35604121424293306;
    const B_ROBUST_UNEQ_TREND_SHRUNK_OUTLIER: f64 = 2.4117706409509885;
    const B_ROBUST_UNEQ_TREND_SCALE_HEAD: [f64; 4] = [
        1.7738825995062002,
        1.7039177845915112,
        2.5263541324098675,
        1.9214959973859211,
    ];

    #[test]
    fn test_fit_f_dist_unequal_df1_robust_with_covariate_matches_limma() {
        let fit = fit_f_dist_unequal_df1(
            &fixture_b(),
            &df_b(),
            Some(&covariate_b()),
            None,
            true,
            None,
        )
        .unwrap();
        assert_relative_eq!(
            fit.df2,
            B_ROBUST_UNEQ_TREND_DF2,
            max_relative = UNEQUAL_FLAT_TOL
        );
        assert_relative_eq!(
            fit.df2_outlier.unwrap(),
            B_ROBUST_UNEQ_TREND_OUTLIER,
            max_relative = UNEQUAL_FLAT_TOL
        );
        assert_slices_close(
            &fit.scale[..4],
            &B_ROBUST_UNEQ_TREND_SCALE_HEAD,
            UNEQUAL_TOL,
        );
        let shrunk = fit.df2_shrunk.unwrap();
        for (i, &v) in shrunk.iter().enumerate() {
            let want = if [6, 22, 30].contains(&i) {
                B_ROBUST_UNEQ_TREND_SHRUNK_OUTLIER
            } else {
                B_ROBUST_UNEQ_TREND_DF2
            };
            assert_relative_eq!(v, want, max_relative = UNEQUAL_FLAT_TOL);
        }
    }

    /// `Rscript -e '... x2 <- rep(0,24); x2[3] <- 0.5; x2[9] <- 2;
    ///  r <- limma:::fitFDistUnequalDF1(x2, df1=dfa, covariate=ca);
    ///  cat(sprintf("%.17g %.17g", r$scale, r$df2))'`
    const A_TWO_INFORMATIVE_SCALE: f64 = 4.104697185054297e-12;
    const A_TWO_INFORMATIVE_DF2: f64 = 2.000332267390954;

    #[test]
    fn test_fit_f_dist_unequal_df1_with_two_informative_genes_matches_limma() {
        // Two informative genes drops the covariate, so the scale is a single
        // number even though a covariate was supplied.
        let mut x = vec![0.0; 24];
        x[2] = 0.5;
        x[8] = 2.0;
        let fit =
            fit_f_dist_unequal_df1(&x, &df_a(), Some(&covariate_a()), None, false, None).unwrap();
        assert_eq!(fit.scale.len(), 1);
        assert_relative_eq!(
            fit.scale[0],
            A_TWO_INFORMATIVE_SCALE,
            max_relative = UNEQUAL_TOL
        );
        assert_relative_eq!(
            fit.df2,
            A_TWO_INFORMATIVE_DF2,
            max_relative = UNEQUAL_FLAT_TOL
        );
    }

    #[test]
    fn test_fit_f_dist_unequal_df1_with_two_informative_genes_and_prior_weights() {
        // limma clears `prior.weights` here but leaves `PriorWeights` set, so
        // `w * prior.weights` recycles a zero-length vector and every reduction
        // that follows is 0 / 0. The scale comes back NaN and the flat
        // likelihood sends `optimize` to its usual near-boundary point. This
        // reproduces limma rather than repairing it.
        //
        // `Rscript -e '... pw <- rep(0,24); pw[c(3,9)] <- 1;
        //  r <- limma:::fitFDistUnequalDF1(xa, df1=dfa, prior.weights=pw);
        //  cat(sprintf("%.17g", r$scale), sprintf("%.17g", r$df2))'`
        let mut w = vec![0.0; 24];
        w[2] = 1.0;
        w[8] = 1.0;
        let fit =
            fit_f_dist_unequal_df1(&fixture_a(), &df_a(), None, None, false, Some(&w)).unwrap();
        assert!(fit.scale[0].is_nan());
        assert_relative_eq!(fit.df2, FLAT_OBJECTIVE_DF2, max_relative = UNEQUAL_FLAT_TOL);
    }

    /// `Rscript -e '... x3 <- xa; x3[5] <- 0;
    ///  r <- limma:::fitFDistUnequalDF1(x3, df1=dfa);
    ///  cat(sprintf("%.17g %.17g", r$scale, r$df2), "\n");
    ///  r <- limma:::fitFDistUnequalDF1(x3, df1=dfa, covariate=ca);
    ///  cat(sprintf("%.17g", r$df2), "\n"); cat(sprintf("%.17g", r$scale[1:4]), sep=",")'`
    const A_ZERO_VAR_SCALE: f64 = 0.3718178967944987;
    const A_ZERO_VAR_DF2: f64 = 6.528194830553999;
    const A_ZERO_VAR_TREND_DF2: f64 = 6.99622580953699;
    const A_ZERO_VAR_TREND_SCALE_HEAD: [f64; 4] = [
        0.29131645136046747,
        0.6312057539864651,
        0.30766450270877077,
        0.5792677420376308,
    ];

    #[test]
    fn test_fit_f_dist_unequal_df1_with_a_zero_variance_matches_limma() {
        // A zero variance is not informative but is not dropped either: it is
        // floored at 1e-12 times the median and still carries full weight into
        // the likelihood.
        let mut x = fixture_a();
        x[4] = 0.0;
        let fit = fit_f_dist_unequal_df1(&x, &df_a(), None, None, false, None).unwrap();
        assert_relative_eq!(fit.scale[0], A_ZERO_VAR_SCALE, max_relative = UNEQUAL_TOL);
        assert_relative_eq!(fit.df2, A_ZERO_VAR_DF2, max_relative = UNEQUAL_FLAT_TOL);

        let fit =
            fit_f_dist_unequal_df1(&x, &df_a(), Some(&covariate_a()), None, false, None).unwrap();
        assert_relative_eq!(
            fit.df2,
            A_ZERO_VAR_TREND_DF2,
            max_relative = UNEQUAL_FLAT_TOL
        );
        assert_slices_close(&fit.scale[..4], &A_ZERO_VAR_TREND_SCALE_HEAD, UNEQUAL_TOL);
    }

    /// `Rscript -e '... d4 <- dfa; d4[c(2,11)] <- 0;
    ///  r <- limma:::fitFDistUnequalDF1(xa, df1=d4);
    ///  cat(sprintf("%.17g %.17g", r$scale, r$df2), "\n");
    ///  r <- limma:::fitFDistUnequalDF1(xa, df1=d4, covariate=ca);
    ///  cat(sprintf("%.17g", r$df2), "\n"); cat(sprintf("%.17g", r$scale[1:4]), sep=",")'`
    const A_SMALL_DF1_SCALE: f64 = 0.5133612398309259;
    const A_SMALL_DF1_DF2: f64 = 8.777398446239493;
    const A_SMALL_DF1_TREND_DF2: f64 = 7.86264899990717;
    const A_SMALL_DF1_TREND_SCALE_HEAD: [f64; 4] = [
        0.5178088802810284,
        0.4553705927989164,
        0.5500370079233148,
        0.4554740575669429,
    ];

    #[test]
    fn test_fit_f_dist_unequal_df1_masks_tiny_df1_matches_limma() {
        // df1 below 0.01 is retired with a zero prior weight and a placeholder
        // df1 of one, which also latches `PriorWeights` on for the whole fit.
        let mut df = df_a();
        df[1] = 0.0;
        df[10] = 0.0;
        let fit = fit_f_dist_unequal_df1(&fixture_a(), &df, None, None, false, None).unwrap();
        assert_relative_eq!(fit.scale[0], A_SMALL_DF1_SCALE, max_relative = UNEQUAL_TOL);
        assert_relative_eq!(fit.df2, A_SMALL_DF1_DF2, max_relative = UNEQUAL_FLAT_TOL);

        let fit =
            fit_f_dist_unequal_df1(&fixture_a(), &df, Some(&covariate_a()), None, false, None)
                .unwrap();
        assert_relative_eq!(
            fit.df2,
            A_SMALL_DF1_TREND_DF2,
            max_relative = UNEQUAL_FLAT_TOL
        );
        assert_slices_close(&fit.scale[..4], &A_SMALL_DF1_TREND_SCALE_HEAD, UNEQUAL_TOL);
    }

    /// `Rscript -e '... r <- limma:::fitFDistUnequalDF1(xa, df1=dfa, covariate=ca, span=0.6);
    ///  cat(sprintf("%.17g", r$df2), "\n"); cat(sprintf("%.17g", r$scale[1:4]), sep=",")'`
    const A_SPAN_06_DF2: f64 = 8.458142188315215;
    const A_SPAN_06_SCALE_HEAD: [f64; 4] = [
        0.47373315982609476,
        0.636071387038315,
        0.49890191505523185,
        0.5689163443682537,
    ];

    #[test]
    fn test_fit_f_dist_unequal_df1_honours_a_supplied_span() {
        let fit = fit_f_dist_unequal_df1(
            &fixture_a(),
            &df_a(),
            Some(&covariate_a()),
            Some(0.6),
            false,
            None,
        )
        .unwrap();
        assert_relative_eq!(fit.df2, A_SPAN_06_DF2, max_relative = UNEQUAL_FLAT_TOL);
        assert_slices_close(&fit.scale[..4], &A_SPAN_06_SCALE_HEAD, UNEQUAL_TOL);
    }

    /// `Rscript -e '... r <- limma:::fitFDistUnequalDF1(xb, df1=dfb, covariate=cb,
    ///  span=0.5, robust=TRUE); cat(sprintf("%.17g %.17g", r$df2, r$df2.outlier), "\n");
    ///  cat(sprintf("%.17g", sort(unique(r$df2.shrunk))), sep=",")'`
    const B_ROBUST_SPAN_05_DF2: f64 = 5.169786259019881;
    const B_ROBUST_SPAN_05_OUTLIER: f64 = 0.35610213377147193;
    const B_ROBUST_SPAN_05_SHRUNK_OUTLIER: f64 = 2.207052068686688;

    #[test]
    fn test_fit_f_dist_unequal_df1_robust_drops_the_span_in_the_recursion() {
        // limma leaves `span` out of the `Recall`, so the second pass smooths at
        // chooseLowessSpan while the first used 0.5. Passing the span through
        // instead, as edgePython does, gives a different `df2`.
        let fit = fit_f_dist_unequal_df1(
            &fixture_b(),
            &df_b(),
            Some(&covariate_b()),
            Some(0.5),
            true,
            None,
        )
        .unwrap();
        assert_relative_eq!(
            fit.df2,
            B_ROBUST_SPAN_05_DF2,
            max_relative = UNEQUAL_FLAT_TOL
        );
        assert_relative_eq!(
            fit.df2_outlier.unwrap(),
            B_ROBUST_SPAN_05_OUTLIER,
            max_relative = UNEQUAL_FLAT_TOL
        );
        let shrunk = fit.df2_shrunk.unwrap();
        for (i, &v) in shrunk.iter().enumerate() {
            let want = if [6, 22, 30].contains(&i) {
                B_ROBUST_SPAN_05_SHRUNK_OUTLIER
            } else {
                B_ROBUST_SPAN_05_DF2
            };
            assert_relative_eq!(v, want, max_relative = UNEQUAL_FLAT_TOL);
        }
    }

    /// `Rscript -e '... k <- 1:60; xd <- 1 + (((k*17) %% 7) - 3)/4096;
    ///  dd <- rep(c(5,6),30); r <- limma:::fitFDistUnequalDF1(xd, df1=dd, robust=TRUE);
    ///  cat(sprintf("%.17g %.17g", r$scale, r$df2), is.null(r$df2.shrunk))'`
    const D_ROBUST_UNEQ_SCALE: f64 = 1.2121144382645828;

    #[test]
    fn test_fit_f_dist_unequal_df1_robust_returns_early_when_nothing_is_an_outlier() {
        // No gene clears the FDR cutoff, so the second pass never runs and the
        // per-gene degrees of freedom are never built.
        let df: Vec<f64> = (0..60).map(|i| [5.0, 6.0][i % 2]).collect();
        let fit = fit_f_dist_unequal_df1(&fixture_d(), &df, None, None, true, None).unwrap();
        assert_relative_eq!(
            fit.scale[0],
            D_ROBUST_UNEQ_SCALE,
            max_relative = UNEQUAL_TOL
        );
        assert_relative_eq!(fit.df2, FLAT_OBJECTIVE_DF2, max_relative = UNEQUAL_FLAT_TOL);
        assert!(fit.df2_shrunk.is_none());
        assert!(fit.df2_outlier.is_none());
    }

    /// `Rscript -e '... xs <- c(0.4,1.1,2.7,0.9); cs <- c(1,2.5,3.5,5); ds <- c(2,5,3,9);
    ///  r <- limma:::fitFDistUnequalDF1(xs, df1=ds, covariate=cs);
    ///  cat(sprintf("%.17g", r$df2), "\n"); cat(sprintf("%.17g", r$scale), sep=",")'`
    const TINY_TREND_SCALE: [f64; 4] = [
        1.464482684259621,
        1.3640607292931846,
        1.3009682044455255,
        1.2117587027943375,
    ];

    #[test]
    fn test_fit_f_dist_unequal_df1_takes_the_weighted_line_fallback() {
        // Four points and a span of one is below loessFit's `4 + 1/span`, so it
        // drops to `lm.wfit` on an intercept and the covariate.
        let fit = fit_f_dist_unequal_df1(
            &[0.4, 1.1, 2.7, 0.9],
            &[2.0, 5.0, 3.0, 9.0],
            Some(&[1.0, 2.5, 3.5, 5.0]),
            None,
            false,
            None,
        )
        .unwrap();
        assert_relative_eq!(fit.df2, FLAT_OBJECTIVE_DF2, max_relative = UNEQUAL_FLAT_TOL);
        assert_slices_close(&fit.scale, &TINY_TREND_SCALE, UNEQUAL_TOL);
    }

    /// `Rscript -e '... k <- 1:600; xl <- (((k*37) %% 1009)+1)/64;
    ///  cl <- ((k*53) %% 601)/16; dl <- rep(c(2,4,7,11,5,9),100);
    ///  cat(sprintf("%.17g", chooseLowessSpan(600, small.n=500)), "\n");
    ///  r <- limma:::fitFDistUnequalDF1(xl, df1=dl, covariate=cl);
    ///  cat(sprintf("%.17g", r$df2), "\n"); cat(sprintf("%.17g", r$scale[1:6]), sep=",")'`
    const L_SPAN: f64 = 0.95872522021672;
    const L_TREND_DF2: f64 = 12.1626698953769;
    const L_TREND_SCALE_HEAD: [f64; 6] = [
        6.266793017150486,
        6.237055818513943,
        6.2106185628474035,
        6.189155437282101,
        6.17402435035205,
        6.164737281752587,
    ];
    const L_TREND_SCALE_TAIL: [f64; 3] = [6.187788511158097, 6.221097044299279, 6.2552025692849655];

    #[test]
    fn test_fit_f_dist_unequal_df1_on_six_hundred_genes_matches_limma() {
        // The only fixture past `small.n = 500`, so the only one whose span is
        // not pinned at 1.
        let (_, min_span, power) = LIMMA_LOWESS_DEFAULTS;
        assert_relative_eq!(
            choose_lowess_span(600, UNEQUAL_LOWESS_SMALL_N, min_span, power),
            L_SPAN,
            max_relative = 1e-14
        );

        let x = fixture_l();
        let cov = covariate_l();
        let fit = fit_f_dist_unequal_df1(&x, &df_l(), Some(&cov), None, false, None).unwrap();
        assert_relative_eq!(fit.df2, L_TREND_DF2, max_relative = UNEQUAL_FLAT_TOL);
        assert_slices_close(&fit.scale[..6], &L_TREND_SCALE_HEAD, UNEQUAL_TOL);
        assert_slices_close(&fit.scale[597..], &L_TREND_SCALE_TAIL, UNEQUAL_TOL);
    }

    /// `Rscript -e '... r <- limma:::fitFDistUnequalDF1(xl, df1=dl, covariate=cl, robust=TRUE);
    ///  cat(sprintf("%.17g %.17g %.17g %.17g %.17g", r$df2, r$df2.outlier,
    ///  min(r$df2.shrunk), sum(r$df2.shrunk), sum(r$scale)))'`
    const L_ROBUST_DF2: f64 = 123.71294709048462;
    const L_ROBUST_OUTLIER: f64 = 5.540757379149584;
    const L_ROBUST_SHRUNK_MIN: f64 = 92.75014611625062;
    const L_ROBUST_SHRUNK_SUM: f64 = 61510.817250394095;
    const L_ROBUST_SCALE_SUM: f64 = 4771.896665776024;

    #[test]
    fn test_fit_f_dist_unequal_df1_robust_on_six_hundred_genes_matches_limma() {
        let x = fixture_l();
        let cov = covariate_l();
        let fit = fit_f_dist_unequal_df1(&x, &df_l(), Some(&cov), None, true, None).unwrap();
        assert_relative_eq!(fit.df2, L_ROBUST_DF2, max_relative = UNEQUAL_FLAT_TOL);
        assert_relative_eq!(
            fit.df2_outlier.unwrap(),
            L_ROBUST_OUTLIER,
            max_relative = UNEQUAL_FLAT_TOL
        );
        let shrunk = fit.df2_shrunk.unwrap();
        let lo = shrunk.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = shrunk.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert_relative_eq!(lo, L_ROBUST_SHRUNK_MIN, max_relative = UNEQUAL_FLAT_TOL);
        assert_relative_eq!(hi, L_ROBUST_DF2, max_relative = UNEQUAL_FLAT_TOL);
        assert_relative_eq!(
            shrunk.iter().sum::<f64>(),
            L_ROBUST_SHRUNK_SUM,
            max_relative = UNEQUAL_FLAT_TOL
        );
        assert_relative_eq!(
            fit.scale.iter().sum::<f64>(),
            L_ROBUST_SCALE_SUM,
            max_relative = UNEQUAL_FLAT_TOL
        );
    }

    // -- squeezeVar dispatch --

    /// `Rscript -e '... s <- squeezeVar(xa, df=dfa, legacy=FALSE);
    ///  cat(sprintf("%.17g %.17g", s$var.prior, s$df.prior), "\n");
    ///  cat(sprintf("%.17g", s$var.post), sep=",")'`
    const A_NONLEGACY_VAR_POST: [f64; 24] = [
        0.5316502371302103,
        0.7135240858185843,
        0.39717283628309796,
        0.6770166883144111,
        0.671480522803574,
        0.49893574225974113,
        0.7373491747082962,
        0.28941257658688035,
        0.5453590886668146,
        0.7368489057706324,
        0.43052345769733313,
        0.7191475700239253,
        0.6851893743401781,
        0.5222605622117893,
        0.7706997961225314,
        0.33154345829639453,
        0.5590679402034189,
        0.7601737257226807,
        0.46387407911156825,
        0.7612784517334394,
        0.43294650606665963,
        0.5455853821638375,
        0.8040504175367664,
        0.37367434000590877,
    ];

    #[test]
    fn test_squeeze_var_dispatches_to_the_unequal_df1_fit() {
        // Unequal df with no legacy flag: limma's automatic rule sends this to
        // fitFDistUnequalDF1, and so does this.
        let out = squeeze_var(&fixture_a(), &df_a(), None, None).unwrap();
        assert_eq!(out.var_prior.len(), 1);
        assert_relative_eq!(
            out.var_prior[0],
            A_UNEQUAL_SCALE,
            max_relative = UNEQUAL_TOL
        );
        assert_relative_eq!(
            out.df_prior[0],
            A_UNEQUAL_DF2,
            max_relative = UNEQUAL_FLAT_TOL
        );
        assert_slices_close(&out.var_post, &A_NONLEGACY_VAR_POST, UNEQUAL_TOL);

        // Asking for it explicitly must give the same thing.
        let same = squeeze_var(
            &fixture_a(),
            &df_a(),
            None,
            Some(SqueezeVarParams {
                legacy: Some(false),
                ..Default::default()
            }),
        )
        .unwrap();
        assert_slices_close(&same.var_post, &A_NONLEGACY_VAR_POST, UNEQUAL_TOL);
    }

    /// `Rscript -e '... s <- squeezeVar(xa, df=dfa, covariate=ca, legacy=FALSE);
    ///  cat(sprintf("%.17g", s$df.prior), "\n"); cat(sprintf("%.17g", s$var.post), sep=",")'`
    const A_NONLEGACY_TREND_VAR_POST: [f64; 24] = [
        0.5157624257454849,
        0.7246483639336246,
        0.39621023177628334,
        0.6796361833708949,
        0.6952867161375217,
        0.4926188547111884,
        0.7650035710730789,
        0.27827046580821047,
        0.5928607775802931,
        0.7288233044325285,
        0.470491584511392,
        0.710653904660662,
        0.7680074779125342,
        0.49898416275024776,
        0.8411469040591076,
        0.31026860591294947,
        0.6715924695348725,
        0.7382898006143429,
        0.5515579820461769,
        0.74970948333179,
        0.44094246280250154,
        0.5396498972314272,
        0.8132626598467574,
        0.3759851284824094,
    ];

    #[test]
    fn test_squeeze_var_trended_dispatches_to_the_unequal_df1_fit() {
        let out = squeeze_var(&fixture_a(), &df_a(), Some(&covariate_a()), None).unwrap();
        assert_eq!(out.var_prior.len(), 24);
        assert_relative_eq!(
            out.df_prior[0],
            A_UNEQUAL_TREND_DF2,
            max_relative = UNEQUAL_FLAT_TOL
        );
        assert_slices_close(&out.var_prior, &A_UNEQUAL_TREND_SCALE, UNEQUAL_TOL);
        assert_slices_close(&out.var_post, &A_NONLEGACY_TREND_VAR_POST, UNEQUAL_TOL);
    }

    /// `Rscript -e '... s <- squeezeVar(xb, df=dfb, robust=TRUE, legacy=FALSE);
    ///  cat(sprintf("%.17g", s$var.prior), "\n"); cat(sprintf("%.17g", s$var.post), sep=",")'`
    const B_NONLEGACY_ROBUST_VAR_POST: [f64; 40] = [
        1.4082643658894873,
        1.5326833000894342,
        1.802287600377153,
        2.1948422439427717,
        2.260546959807817,
        2.6729798933216284,
        31.948405740377268,
        1.0680857573950386,
        1.6674043437700605,
        1.8793951020857094,
        2.2303728622064014,
        2.687114495347121,
        2.51968693768839,
        1.0858103108497903,
        1.250533262908344,
        1.560358008799388,
        1.926544321650634,
        2.226106904081985,
        2.6584581240356497,
        3.1793867467514705,
        1.3334017056128773,
        1.4325221128460657,
        47.72365777520489,
        2.0526302602037374,
        2.185684299531207,
        2.57281870607826,
        3.086543385864898,
        0.9258737736560045,
        1.5925416834934505,
        1.7792339148423408,
        20.116966714256552,
        2.5449025116080866,
        2.4448242774117803,
        0.9856491236064219,
        1.1268641872687837,
        1.4181460250603537,
        1.851681661374024,
        2.125945716838616,
        2.534789048396089,
        3.037174763012436,
    ];

    #[test]
    fn test_squeeze_var_robust_dispatches_to_the_unequal_df1_fit() {
        let out = squeeze_var(
            &fixture_b(),
            &df_b(),
            None,
            Some(SqueezeVarParams {
                robust: true,
                ..Default::default()
            }),
        )
        .unwrap();
        assert_relative_eq!(
            out.var_prior[0],
            B_ROBUST_UNEQ_SCALE,
            max_relative = UNEQUAL_TOL
        );
        assert_eq!(out.df_prior.len(), 40);
        assert_slices_close(&out.var_post, &B_NONLEGACY_ROBUST_VAR_POST, UNEQUAL_TOL);
    }

    /// `Rscript -e '... d <- dfb; d[c(3,17,29)] <- 0; s <- squeezeVar(xb, df=d, legacy=FALSE);
    ///  cat(sprintf("%.17g %.17g", s$var.prior, s$df.prior), "\n");
    ///  cat(sprintf("%.17g", s$var.post), sep=",")'`
    const B_NONLEGACY_ZERODF_VAR_PRIOR: f64 = 1.7559995759381142;
    const B_NONLEGACY_ZERODF_DF_PRIOR: f64 = 2.976084781693831;

    #[test]
    fn test_squeeze_var_with_zero_df_dispatches_to_the_unequal_df1_fit() {
        let mut df = df_b();
        for i in [2, 16, 28] {
            df[i] = 0.0;
        }
        let out = squeeze_var(
            &fixture_b(),
            &df,
            None,
            Some(SqueezeVarParams {
                legacy: Some(false),
                ..Default::default()
            }),
        )
        .unwrap();
        assert_relative_eq!(
            out.var_prior[0],
            B_NONLEGACY_ZERODF_VAR_PRIOR,
            max_relative = UNEQUAL_TOL
        );
        assert_relative_eq!(
            out.df_prior[0],
            B_NONLEGACY_ZERODF_DF_PRIOR,
            max_relative = UNEQUAL_FLAT_TOL
        );
        // A gene with no residual degrees of freedom gets the prior outright.
        for i in [2, 16, 28] {
            assert_relative_eq!(
                out.var_post[i],
                B_NONLEGACY_ZERODF_VAR_PRIOR,
                max_relative = UNEQUAL_TOL
            );
        }
    }

    /// `Rscript -e '... s <- squeezeVar(xa, df=rep(4,24), covariate=ca, span=0.7);
    ///  cat(sprintf("%.17g", s$df.prior), "\n"); cat(sprintf("%.17g", s$var.prior[1:3]), sep=",")'`
    const A_SPAN_FORCES_NONLEGACY_VAR_PRIOR_HEAD: [f64; 3] =
        [0.755640423479099, 0.460011626412849, 0.7753712958504761];

    #[test]
    fn test_a_supplied_span_forces_the_unequal_df1_fit() {
        // Equal df would normally pick the legacy spline fit. limma sets
        // `legacy <- FALSE` the moment a span is supplied, before the automatic
        // rule gets a look in, so this goes to fitFDistUnequalDF1 instead.
        let out = squeeze_var(
            &fixture_a(),
            &[4.0; 24],
            Some(&covariate_a()),
            Some(SqueezeVarParams {
                span: Some(0.7),
                ..Default::default()
            }),
        )
        .unwrap();
        assert_relative_eq!(
            out.df_prior[0],
            FLAT_OBJECTIVE_DF2,
            max_relative = UNEQUAL_FLAT_TOL
        );
        assert_slices_close(
            &out.var_prior[..3],
            &A_SPAN_FORCES_NONLEGACY_VAR_PRIOR_HEAD,
            1e-12,
        );

        // Without the span the same call takes the legacy spline.
        let legacy = squeeze_var(&fixture_a(), &[4.0; 24], Some(&covariate_a()), None).unwrap();
        assert_relative_eq!(
            legacy.df_prior[0],
            A_TREND_DF_PRIOR,
            max_relative = UNEQUAL_TOL
        );
    }

    #[test]
    fn test_unequal_df1_input_validation() {
        let x = fixture_a();
        let df = df_a();
        assert!(matches!(
            fit_f_dist_unequal_df1(&x, &[1.0, 2.0], None, None, false, None),
            Err(EdgeErrors::LengthMismatch { .. })
        ));
        assert!(matches!(
            fit_f_dist_unequal_df1(&x, &df, Some(&[1.0, 2.0]), None, false, None),
            Err(EdgeErrors::LengthMismatch { .. })
        ));
        assert!(matches!(
            fit_f_dist_unequal_df1(&x, &df, None, None, false, Some(&[1.0, 2.0])),
            Err(EdgeErrors::LengthMismatch { .. })
        ));
        let mut bad = vec![1.0; 24];
        bad[3] = -1.0;
        assert!(matches!(
            fit_f_dist_unequal_df1(&x, &df, None, None, false, Some(&bad)),
            Err(EdgeErrors::InvalidArgument(_))
        ));
        let mut nan_cov = covariate_a();
        nan_cov[5] = f64::NAN;
        assert!(matches!(
            fit_f_dist_unequal_df1(&x, &df, Some(&nan_cov), None, false, None),
            Err(EdgeErrors::InvalidArgument(_))
        ));
    }

    #[test]
    fn test_unequal_df1_with_nothing_informative_is_reported_as_nan() {
        // limma returns NA rather than erroring, and squeeze_var turns that into
        // the "could not estimate prior df" error on the way out.
        let fit =
            fit_f_dist_unequal_df1(&[0.0, 0.0, 1.5], &[4.0, 4.0, 4.0], None, None, false, None)
                .unwrap();
        assert!(fit.scale[0].is_nan());
        assert!(fit.df2.is_nan());

        let err = squeeze_var(
            &[0.0, 0.0, 1.5],
            &[4.0, 0.5, 4.0],
            None,
            Some(SqueezeVarParams {
                legacy: Some(false),
                ..Default::default()
            }),
        )
        .unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));
    }

    #[test]
    fn test_brent_fmin_reproduces_r_optimize_on_a_flat_objective() {
        // R's optimize on a constant function walks to a fixed point just inside
        // the upper bound. Reproducing it is the cheapest check that the
        // convergence test is R's and not scipy's.
        let par = brent_fmin(UNEQUAL_PAR_LOWER, UNEQUAL_PAR_UPPER, |_| 0.0, OPTIMIZE_TOL);
        assert_relative_eq!(
            2.0 * par / (1.0 - par),
            FLAT_OBJECTIVE_DF2,
            max_relative = 1e-13
        );

        // And it still finds an ordinary interior minimum.
        let par = brent_fmin(0.0, 4.0, |p: f64| (p - 1.75).powi(2), OPTIMIZE_TOL);
        assert_relative_eq!(par, 1.75, max_relative = 1e-6);
    }

    /// `Rscript -e '... xn <- xa; xn[c(4,15)] <- NA;
    ///  r <- limma:::fitFDistUnequalDF1(xn, df1=dfa);
    ///  cat(sprintf("%.17g %.17g", r$scale, r$df2), "\n");
    ///  r <- limma:::fitFDistUnequalDF1(xn, df1=dfa, covariate=ca);
    ///  cat(sprintf("%.17g", r$df2), "\n"); cat(sprintf("%.17g", r$scale[1:4]), sep=",")'`
    const A_MISSING_SCALE: f64 = 0.4642260179706808;
    const A_MISSING_DF2: f64 = 7.768255432326142;
    const A_MISSING_TREND_DF2: f64 = 7.21925270452184;
    const A_MISSING_TREND_SCALE_HEAD: [f64; 4] = [
        0.47205935858500275,
        0.4287917134093374,
        0.483877239045185,
        0.4304121690245412,
    ];

    #[test]
    fn test_fit_f_dist_unequal_df1_treats_nan_variances_as_missing() {
        // limma zeroes a missing variance and takes its prior weight away
        // rather than dropping the gene, so it still gets a trend value.
        let mut x = fixture_a();
        x[3] = f64::NAN;
        x[14] = f64::NAN;
        let fit = fit_f_dist_unequal_df1(&x, &df_a(), None, None, false, None).unwrap();
        assert_relative_eq!(fit.scale[0], A_MISSING_SCALE, max_relative = UNEQUAL_TOL);
        assert_relative_eq!(fit.df2, A_MISSING_DF2, max_relative = UNEQUAL_FLAT_TOL);

        let fit =
            fit_f_dist_unequal_df1(&x, &df_a(), Some(&covariate_a()), None, false, None).unwrap();
        assert_relative_eq!(
            fit.df2,
            A_MISSING_TREND_DF2,
            max_relative = UNEQUAL_FLAT_TOL
        );
        assert_slices_close(&fit.scale[..4], &A_MISSING_TREND_SCALE_HEAD, UNEQUAL_TOL);
    }

    /// `Rscript -e '... r <- limma:::fitFDistUnequalDF1(xa, df1=6);
    ///  cat(sprintf("%.17g %.17g", r$scale, r$df2))'`
    const A_SCALAR_DF1_SCALE: f64 = 0.6313881564196626;
    const A_SCALAR_DF1_DF2: f64 = 26.23142957377924;

    #[test]
    fn test_fit_f_dist_unequal_df1_accepts_a_scalar_df1() {
        let fit = fit_f_dist_unequal_df1(&fixture_a(), &[6.0], None, None, false, None).unwrap();
        assert_relative_eq!(fit.scale[0], A_SCALAR_DF1_SCALE, max_relative = UNEQUAL_TOL);
        assert_relative_eq!(fit.df2, A_SCALAR_DF1_DF2, max_relative = UNEQUAL_FLAT_TOL);

        // A scalar df1 below the cutoff retires every gene at once, which
        // leaves nothing informative. limma returns NA; so does this.
        let fit = fit_f_dist_unequal_df1(&fixture_a(), &[0.005], None, None, false, None).unwrap();
        assert!(fit.scale[0].is_nan());
        assert!(fit.df2.is_nan());
    }

    #[test]
    fn test_unequal_df1_robust_over_the_parallel_threshold_stays_consistent() {
        // Past PARALLEL_THRESHOLD the tail probabilities fan out over rayon.
        // They are an elementwise map, so the answer must not depend on it.
        let n = PARALLEL_THRESHOLD + 64;
        let x: Vec<f64> = (1..=n)
            .map(|k| (((k * 37) % 1009) + 1) as f64 / 64.0)
            .collect();
        let df: Vec<f64> = (0..n).map(|i| [2.0, 4.0, 7.0, 11.0][i % 4]).collect();
        let a = fit_f_dist_unequal_df1(&x, &df, None, None, true, None).unwrap();
        let b = fit_f_dist_unequal_df1(&x, &df, None, None, true, None).unwrap();
        assert_eq!(a.df2, b.df2);
        assert_eq!(a.scale, b.scale);
        assert!(a.df2.is_finite() && a.df2 > 0.0);
    }

    #[test]
    fn test_all_positive_df_equal_matches_limma_dispatch() {
        assert!(all_positive_df_equal(&[4.0, 4.0, 0.0, 4.0]));
        assert!(all_positive_df_equal(&[4.0]));
        assert!(!all_positive_df_equal(&[4.0, 5.0]));
        // R's min and max of an empty vector are Inf and -Inf, which are not
        // identical, so an all-zero df vector is not legacy.
        assert!(!all_positive_df_equal(&[0.0, 0.0]));
    }
}
