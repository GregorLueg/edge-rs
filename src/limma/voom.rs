//! voom: the mean-variance trend that lets counts go through a linear model.
//!
//! The idea is one paragraph long. Take log2-CPM, fit the linear model to it,
//! and the residual standard deviation is a decreasing function of the count a
//! gene was observed at. Smooth `sqrt(sigma)` against average log-count with a
//! lowess, read that trend back at every *fitted* value rather than every
//! observed one, and the reciprocal fourth power of what comes out is a
//! precision weight per observation. `lmFit` with those weights then behaves as
//! if the counts had been continuous all along.
//!
//! Three entry points, mirroring three upstream functions:
//!
//! * [`voom`] is limma's `voom`.
//! * [`voom_lmfit`] is edgeR's `voomLmFit` with `block = NULL` and
//!   `sample.weights = FALSE`: the structural-zero machinery and the final
//!   weighted refit, without the block correlation or array weight loops.
//! * [`voom_basic`] is the thin wrapper edgePython keeps for its older API:
//!   explicit normalisation factors and a fixed span.
//!
//! ### Why neither `cpm` nor `ave_log_cpm` appears here
//!
//! They look like they should. They do not fit. [`crate::core::expression::cpm`]
//! scales its prior count by the library size, as edgeR's `cpm` does, whereas
//! voom adds a flat `0.5` to every count and a flat `1` to every library size.
//! The two agree only when all libraries are the same size.
//! [`crate::core::expression::ave_log_cpm`] fits an intercept-only negative
//! binomial GLM; voom's `Amean` is a plain row mean of the log2-CPM. Reusing
//! either would be wrong rather than merely different.
//!
//! ### Which lowess
//!
//! limma's `voom` and edgeR's `voomLmFit` both smooth with `stats::lowess`,
//! that is [`crate::limma::lowess::lowess`], not with limma's own
//! `weightedLowess`. The single exception is `voomLmFit` once it has found
//! structural zeros, at which point it switches to
//! [`crate::limma::lowess::weighted_lowess`] so that the residual degrees of
//! freedom can act as prior weights. Both are reproduced here exactly as
//! dispatched upstream. edgePython uses `weightedLowess` unconditionally, which
//! is entry 1 of the deviations listed in the [`voom`] documentation.
//!
//! Genes are the parallel axis: the log-CPM transform and the weight lookup are
//! rayon fan-outs over contiguous gene rows.

use faer::MatRef;
use rayon::prelude::*;

use crate::core::expression::PER_MILLION;
use crate::core::normalisation::{NormMethod, calc_norm_factors};
use crate::errors::EdgeErrors;
use crate::glm::fit::glm_fit;
use crate::limma::lm_fit::{LmFitResult, lm_fit};
use crate::limma::lowess::{LowessParams, lowess, weighted_lowess};
use crate::numeric::interpolate::interp_linear_extrap;
use crate::utils::design::{LIMMA_LOWESS_DEFAULTS, choose_lowess_span, hat_diagonal, is_full_rank};
use crate::utils::recycled::Recycled;
use crate::utils::traits::EdgeFloat;

///////////////
// Constants //
///////////////

/// Reciprocal of [`PER_MILLION`], written as a literal.
///
/// `1e-6` and `1.0 / 1e6` are not the same double. R's voom writes the literal,
/// so this does too; the difference is one ulp on the fitted counts and it
/// propagates into the fourth power of the trend.
const INV_PER_MILLION: f64 = 1e-6;

/// Robustness iterations R's `lowess` performs by default, its `iter = 3`.
const LOWESS_ITERATIONS: usize = 3;

/// Tolerance edgeR's `voomLmFit` uses to call a count or a fitted value zero.
///
/// Doubles as the slack on the row-count comparison, which is why the test is
/// `> max(2, MinGroupSize) - eps` rather than `>=`.
const STRUCTURAL_ZERO_EPS: f64 = 1e-4;

/// Smallest number of zeros in a row that makes it a structural-zero candidate,
/// before the leverage-derived group size is taken into account.
const MIN_ZERO_RUN: f64 = 2.0;

/// Work below which the per-gene weight lookup stays sequential.
///
/// One gene costs `n_samples` binary searches over the trend knots. Below a few
/// tens of thousands of those, rayon's fork costs more than the scan.
const PARALLEL_WORK_THRESHOLD: usize = 32_768;

//////////////////
// Public types //
//////////////////

/// Tuning knobs for [`voom`] and [`voom_lmfit`].
#[derive(Clone, Copy, Debug)]
pub struct VoomParams {
    /// Scaling rule folded into the library sizes before the log-CPM transform.
    ///
    /// This is edgeR's `calcNormFactors`, not limma's `normalizeBetweenArrays`:
    /// the factors multiply the library sizes rather than shifting the log
    /// ratios. [`NormMethod::None`] leaves the library sizes alone and is what
    /// upstream `voom(counts, design)` does.
    pub normalize_method: NormMethod,
    /// Lowess span, in `(0, 1]`. Ignored when `adaptive_span` is set.
    pub span: f64,
    /// Whether to derive the span from the gene count with
    /// [`choose_lowess_span`] at limma's defaults. `true` upstream since limma
    /// 3.56, which is why it is `true` here.
    pub adaptive_span: bool,
    /// Count added to every observation before the log, and half the constant
    /// added to every library size. Upstream hardcodes `0.5`.
    pub prior_count: f64,
    /// Whether to return the fitted trend in [`VoomResult::trend_x`] and
    /// [`VoomResult::trend_y`]. When `false` both come back empty.
    pub save_trend: bool,
}

impl VoomParams {
    /// Builds a parameter set.
    ///
    /// ### Params
    ///
    /// * `normalize_method` - Scaling rule for the library sizes
    /// * `span` - Lowess span, used only when `adaptive_span` is `false`
    /// * `adaptive_span` - Whether to derive the span from the gene count
    /// * `prior_count` - Count added before the log; upstream uses `0.5`
    /// * `save_trend` - Whether to return the trend knots
    ///
    /// ### Returns
    ///
    /// The parameter set. Values are validated by [`voom`], not here.
    pub fn new(
        normalize_method: NormMethod,
        span: f64,
        adaptive_span: bool,
        prior_count: f64,
        save_trend: bool,
    ) -> Self {
        Self {
            normalize_method,
            span,
            adaptive_span,
            prior_count,
            save_trend,
        }
    }
}

impl Default for VoomParams {
    fn default() -> Self {
        Self {
            normalize_method: NormMethod::None,
            span: 0.5,
            adaptive_span: true,
            prior_count: 0.5,
            save_trend: true,
        }
    }
}

/// What voom produces: an expression matrix and a weight for every entry of it.
#[derive(Clone, Debug)]
pub struct VoomResult {
    /// log2-CPM, row-major `n_genes * n_samples`. limma's `E`.
    pub e: Vec<f64>,
    /// Precision weights, same shape and layout as [`VoomResult::e`].
    pub weights: Vec<f64>,
    /// Abscissae of the fitted mean-variance trend, strictly increasing. Empty
    /// when `save_trend` is off or the design had no replication.
    pub trend_x: Vec<f64>,
    /// Ordinates of the fitted mean-variance trend, one per
    /// [`VoomResult::trend_x`].
    pub trend_y: Vec<f64>,
}

/////////////////
// Public API  //
/////////////////

/// Mean-variance trend and precision weights, as limma's `voom`.
///
/// Fits the linear model to log2-CPM once, smooths `sqrt(sigma)` against
/// average log-count with [`crate::limma::lowess::lowess`], and reads the trend
/// back at each gene's fitted log-count. Genes that are zero in every sample are
/// dropped from the trend, since their zero residual scale would drag the low
/// end of it down, but they still receive weights.
///
/// The returned weights do **not** include `weights`. limma multiplies prior
/// weights into the fit but returns the voom weights on their own; only
/// [`voom_lmfit`] combines the two.
///
/// ### Params
///
/// * `counts` - Row-major counts, `n_genes * n_samples`. Non-negative and
///   finite; at least two genes.
/// * `n_genes` - Number of genes, that is, rows
/// * `n_samples` - Number of samples, that is, columns
/// * `design` - Design matrix, row-major `n_samples * n_coef`. Must be full
///   rank.
/// * `n_coef` - Number of coefficients, that is, design columns
/// * `lib_size` - Library size per sample. `None` uses the column sums. Each
///   entry must be positive and finite.
/// * `weights` - Prior observation weights handed to the linear model, or
///   `None`
/// * `params` - Tuning knobs, or `None` for [`VoomParams::default`]
///
/// ### Returns
///
/// The log2-CPM matrix, the precision weights and the trend, or [`EdgeErrors`]
/// if a shape disagrees, a count is negative, a library size is not positive, a
/// parameter is outside its domain, or the design is rank deficient.
///
/// When fewer than two genes have residual degrees of freedom there is nothing
/// to fit a trend to, and every weight comes back as one, which is what limma
/// does after warning.
///
/// ### Where edgePython disagrees with limma
///
/// All of these are reproduced as limma has them, not as edgePython has them.
/// The numbering is local to this module and does not index
/// `UPSTREAM_DEVIATIONS.md`.
///
/// * edgePython always smooths with `weightedLowess`; limma's `voom` always
///   uses `stats::lowess`. This is the same class of mistake as entry 16 of
///   `UPSTREAM_DEVIATIONS.md`, which cost 7e-4 on the quasi-likelihood prior.
/// * edgePython uses `npts = 120` and three iterations for the unweighted
///   smooth, neither of which is a limma default.
/// * edgePython never drops the all-zero rows from the trend.
/// * edgePython restricts the trend to genes with residual degrees of freedom;
///   limma's `voom` does not, only `voomLmFit` does.
/// * edgePython clamps `sigma`, the trend and the fitted counts away from zero
///   with a `1e-8` floor; limma clamps nothing.
/// * edgePython collapses tied trend abscissae by taking the first ordinate;
///   `approxfun(ties = list("ordered", mean))` takes the mean.
/// * edgePython multiplies prior weights into the returned weights; limma's
///   `voom` returns the voom weights alone.
/// * edgePython's `normalize_method` is limma's `normalizeBetweenArrays`; this
///   port takes edgeR's `calcNormFactors` instead, because that is the type the
///   crate carries. Neither is the other, and quantile normalisation of the
///   log-CPM is not available here.
///
/// Two guards are added on top of limma, both no-ops whenever limma itself
/// produces a usable answer. Non-finite `(sx, sy)` pairs are dropped before the
/// smooth, and so are genes with no residual degrees of freedom. limma passes
/// both straight to `lowess`, and a single such gene turns the entire weight
/// matrix into `NaN`; the crate's `lm_fit` reports `sigma = 0` rather than `NA`
/// there, which would corrupt the trend silently instead of loudly.
///
/// ### References
///
/// Law et al., Genome Biology, 2014
#[allow(clippy::too_many_arguments)]
pub fn voom<T: EdgeFloat>(
    counts: &[T],
    n_genes: usize,
    n_samples: usize,
    design: &[f64],
    n_coef: usize,
    lib_size: Option<&[f64]>,
    weights: Option<&Recycled<f64>>,
    params: Option<VoomParams>,
) -> Result<VoomResult, EdgeErrors> {
    let params = params.unwrap_or_default();
    let prep = prepare(
        counts, n_genes, n_samples, design, n_coef, lib_size, weights, &params,
    )?;

    let fit = lm_fit(
        &prep.e, n_genes, n_samples, design, n_coef, weights, None, None,
    )?;

    if fit.df_residual.iter().filter(|d| **d > 0.0).count() < 2 {
        return Ok(no_trend_result(prep.e, n_genes, n_samples));
    }

    // limma keeps every gene in the trend but the ones that are zero throughout,
    // whose residual scale is exactly zero and carries no information.
    let shift = corrected_mean(&prep.log_lib_adj);
    let log2_million = PER_MILLION.log2();
    let keep: Vec<usize> = (0..n_genes)
        .filter(|&g| {
            let row = &counts[g * n_samples..(g + 1) * n_samples];
            fit.df_residual[g] > 0.0 && row.iter().any(|v| *v != T::zero())
        })
        .collect();
    let sx: Vec<f64> = keep
        .iter()
        .map(|&g| row_mean(&prep.e[g * n_samples..(g + 1) * n_samples]) + shift - log2_million)
        .collect();
    let sy: Vec<f64> = keep.iter().map(|&g| fit.sigma[g].sqrt()).collect();

    let trend = plain_trend(&sx, &sy, prep.span)?;
    let w = voom_weights(&fit.fitted, n_genes, n_samples, &prep.lib_adj, &trend);

    Ok(finish(prep.e, w, trend, params.save_trend))
}

/// voom followed by the weighted linear model fit, as edgeR's `voomLmFit`.
///
/// Two things separate this from [`voom`]. Genes with a whole group of exact
/// zeros get those observations masked out and their residual scale and degrees
/// of freedom recomputed on what is left, so that a group of structural zeros
/// does not masquerade as a low-variance gene; when any such gene exists the
/// trend switches to [`crate::limma::lowess::weighted_lowess`] with the residual
/// degrees of freedom as prior weights. And the model is then refitted with the
/// voom weights in place, which is the fit the caller actually wants.
///
/// The block correlation and array weight loops of `voomLmFit` are not here.
/// This is `voomLmFit(counts, design, block = NULL, sample.weights = FALSE)`.
///
/// ### Params
///
/// * `counts` - Row-major counts, `n_genes * n_samples`
/// * `n_genes` - Number of genes
/// * `n_samples` - Number of samples
/// * `design` - Design matrix, row-major `n_samples * n_coef`, full rank
/// * `n_coef` - Number of coefficients
/// * `lib_size` - Library size per sample. `None` uses the column sums.
/// * `weights` - Prior observation weights, or `None`. Unlike [`voom`], these
///   are multiplied into the returned weights.
/// * `params` - Tuning knobs, or `None` for [`VoomParams::default`]
///
/// ### Returns
///
/// The voom result and the weighted fit, or [`EdgeErrors`] under the same
/// conditions as [`voom`].
///
/// ### Where edgePython disagrees with edgeR
///
/// In addition to the list on [`voom`]:
///
/// * edgePython passes `block` and `correlation` to the *first* fit; edgeR fits
///   that one with prior weights only and introduces the correlation later.
///   Moot here, since neither is supported.
/// * edgePython clamps the trend's prior weights to at least one; edgeR passes
///   `df.residual` through unchanged.
/// * edgePython runs the structural-zero detection from `voom` itself behind a
///   flag; in edgeR it exists only in `voomLmFit`.
#[allow(clippy::too_many_arguments)]
pub fn voom_lmfit<T: EdgeFloat>(
    counts: &[T],
    n_genes: usize,
    n_samples: usize,
    design: &[f64],
    n_coef: usize,
    lib_size: Option<&[f64]>,
    weights: Option<&Recycled<f64>>,
    params: Option<VoomParams>,
) -> Result<(VoomResult, LmFitResult), EdgeErrors> {
    let params = params.unwrap_or_default();
    let prep = prepare(
        counts, n_genes, n_samples, design, n_coef, lib_size, weights, &params,
    )?;

    let mut fit = lm_fit(
        &prep.e, n_genes, n_samples, design, n_coef, weights, None, None,
    )?;

    let zeros =
        detect_structural_zeros(counts, n_genes, n_samples, design, n_coef, &prep.lib_size)?;
    for (k, &g) in zeros.rows.iter().enumerate() {
        let mask = &zeros.observed[k * n_samples..(k + 1) * n_samples];
        let row_w: Option<Vec<f64>> =
            weights.map(|w| w.row(g, n_samples).iter(n_samples).collect());
        let (sigma, df) = masked_fit(
            &prep.e[g * n_samples..(g + 1) * n_samples],
            mask,
            design,
            n_samples,
            n_coef,
            row_w.as_deref(),
        )?;
        fit.sigma[g] = sigma;
        fit.df_residual[g] = df;
    }

    let has_rep: Vec<usize> = (0..n_genes).filter(|&g| fit.df_residual[g] > 0.0).collect();
    if has_rep.len() < 2 {
        return Ok((no_trend_result(prep.e, n_genes, n_samples), fit));
    }

    // Genes whose zeros were masked take their abundance from the samples that
    // survived, so that the masked-out zeros do not pull `sx` down as well.
    let shift = corrected_mean(&prep.log_lib_adj);
    let log2_million = PER_MILLION.log2();
    let mut amean: Vec<f64> = (0..n_genes)
        .map(|g| row_mean(&prep.e[g * n_samples..(g + 1) * n_samples]))
        .collect();
    for (k, &g) in zeros.rows.iter().enumerate() {
        let mask = &zeros.observed[k * n_samples..(k + 1) * n_samples];
        amean[g] = masked_row_mean(&prep.e[g * n_samples..(g + 1) * n_samples], mask);
    }

    let sx: Vec<f64> = has_rep
        .iter()
        .map(|&g| amean[g] + shift - log2_million)
        .collect();
    let sy: Vec<f64> = has_rep.iter().map(|&g| fit.sigma[g].sqrt()).collect();

    let trend = if zeros.rows.is_empty() {
        plain_trend(&sx, &sy, prep.span)?
    } else {
        let df: Vec<f64> = has_rep.iter().map(|&g| fit.df_residual[g]).collect();
        weighted_trend(&sx, &sy, &df, prep.span)?
    };

    let mut w = voom_weights(&fit.fitted, n_genes, n_samples, &prep.lib_adj, &trend);
    if let Some(prior) = weights {
        w.par_chunks_mut(n_samples)
            .enumerate()
            .for_each(|(g, row)| {
                let p = prior.row(g, n_samples);
                for (j, v) in row.iter_mut().enumerate() {
                    *v *= p.get(j);
                }
            });
    }

    let full = Recycled::full(w.clone(), n_genes, n_samples)?;
    let mut final_fit = lm_fit(
        &prep.e,
        n_genes,
        n_samples,
        design,
        n_coef,
        Some(&full),
        None,
        None,
    )?;
    for (k, &g) in zeros.rows.iter().enumerate() {
        let mask = &zeros.observed[k * n_samples..(k + 1) * n_samples];
        let (sigma, df) = masked_fit(
            &prep.e[g * n_samples..(g + 1) * n_samples],
            mask,
            design,
            n_samples,
            n_coef,
            Some(&w[g * n_samples..(g + 1) * n_samples]),
        )?;
        final_fit.sigma[g] = sigma;
        final_fit.df_residual[g] = df;
    }

    Ok((finish(prep.e, w, trend, params.save_trend), final_fit))
}

/// voom with explicit normalisation factors and a fixed span.
///
/// edgePython's older entry point, kept because it is the shape most callers
/// want: hand it library sizes and normalisation factors separately rather than
/// pre-multiplying them, and get the canonical `span = 0.5` without the
/// adaptive rule. Equivalent to
/// `voom(counts, design, lib.size = lib.size * norm.factors, span = 0.5, adaptive.span = FALSE)`.
///
/// Unlike edgePython's version, `prior_count` is honoured rather than accepted
/// and discarded. At the default `0.5` the two agree.
///
/// ### Params
///
/// * `counts` - Row-major counts, `n_genes * n_samples`
/// * `n_genes` - Number of genes
/// * `n_samples` - Number of samples
/// * `design` - Design matrix, row-major `n_samples * n_coef`, full rank
/// * `n_coef` - Number of coefficients
/// * `lib_size` - Library size per sample. `None` uses the column sums.
/// * `norm_factors` - Normalisation factor per sample, multiplied into the
///   library sizes. `None` leaves them alone.
/// * `prior_count` - Count added before the log. Must be positive; upstream
///   uses `0.5`.
///
/// ### Returns
///
/// The voom result, or [`EdgeErrors`] under the same conditions as [`voom`],
/// plus a length or positivity failure on `norm_factors`.
#[allow(clippy::too_many_arguments)]
pub fn voom_basic<T: EdgeFloat>(
    counts: &[T],
    n_genes: usize,
    n_samples: usize,
    design: &[f64],
    n_coef: usize,
    lib_size: Option<&[f64]>,
    norm_factors: Option<&[f64]>,
    prior_count: f64,
) -> Result<VoomResult, EdgeErrors> {
    let mut effective = match lib_size {
        Some(l) => {
            check_lib_size(l, n_samples)?;
            l.to_vec()
        }
        None => column_sums(counts, n_genes, n_samples),
    };
    if let Some(nf) = norm_factors {
        if nf.len() != n_samples {
            return Err(EdgeErrors::LengthMismatch {
                name: "norm_factors",
                expected: n_samples,
                got: nf.len(),
            });
        }
        if let Some(bad) = nf.iter().find(|v| !v.is_finite() || **v <= 0.0) {
            return Err(EdgeErrors::InvalidArgument(format!(
                "normalisation factors must be positive and finite, found {bad}"
            )));
        }
        for (l, f) in effective.iter_mut().zip(nf.iter()) {
            *l *= f;
        }
    }

    let params = VoomParams {
        normalize_method: NormMethod::None,
        span: 0.5,
        adaptive_span: false,
        prior_count,
        save_trend: true,
    };
    voom(
        counts,
        n_genes,
        n_samples,
        design,
        n_coef,
        Some(&effective),
        None,
        Some(params),
    )
}

///////////////
// Internals //
///////////////

/// Everything the two entry points compute before they touch the linear model.
struct Prepared {
    /// log2-CPM, row-major `n_genes * n_samples`.
    e: Vec<f64>,
    /// Library size per sample after normalisation factors.
    lib_size: Vec<f64>,
    /// `lib_size + 2 * prior_count`, the denominator of the log-CPM transform.
    lib_adj: Vec<f64>,
    /// `log2(lib_adj)`, whose mean shifts log2-CPM back onto a log-count scale.
    log_lib_adj: Vec<f64>,
    /// Span the trend will actually be fitted with.
    span: f64,
}

/// Validates the inputs and builds the log2-CPM matrix.
///
/// Shared by [`voom`] and [`voom_lmfit`], which agree exactly up to and
/// including the first `lm_fit` call.
///
/// ### Params
///
/// * `counts` - Row-major counts
/// * `n_genes` - Number of genes
/// * `n_samples` - Number of samples
/// * `design` - Row-major design matrix
/// * `n_coef` - Number of coefficients
/// * `lib_size` - Library size per sample, or `None` for the column sums
/// * `weights` - Prior weights, checked against the shape only
/// * `params` - Tuning knobs, already resolved
///
/// ### Returns
///
/// The prepared quantities, or [`EdgeErrors`] on any input failure.
#[allow(clippy::too_many_arguments)]
fn prepare<T: EdgeFloat>(
    counts: &[T],
    n_genes: usize,
    n_samples: usize,
    design: &[f64],
    n_coef: usize,
    lib_size: Option<&[f64]>,
    weights: Option<&Recycled<f64>>,
    params: &VoomParams,
) -> Result<Prepared, EdgeErrors> {
    if n_genes == 0 || n_samples == 0 {
        return Err(EdgeErrors::EmptyCounts { n_genes, n_samples });
    }
    if n_genes < 2 {
        return Err(EdgeErrors::InvalidArgument(
            "need at least two genes to fit a mean-variance trend".to_string(),
        ));
    }
    if counts.len() != n_genes * n_samples {
        return Err(EdgeErrors::LengthMismatch {
            name: "counts",
            expected: n_genes * n_samples,
            got: counts.len(),
        });
    }
    if let Some(bad) = counts.iter().find(|v| !v.is_finite() || **v < T::zero()) {
        return Err(EdgeErrors::InvalidArgument(format!(
            "counts must be non-negative and finite, found {bad}"
        )));
    }
    if design.len() != n_samples * n_coef {
        return Err(EdgeErrors::LengthMismatch {
            name: "design",
            expected: n_samples * n_coef,
            got: design.len(),
        });
    }
    if !is_full_rank(design, n_samples, n_coef)? {
        return Err(EdgeErrors::DesignNotFullRank {
            n_cols: n_coef,
            rank: crate::utils::design::matrix_rank(design, n_samples, n_coef)?,
        });
    }
    // Explicit rather than a negated comparison, so a NaN prior count is
    // rejected instead of slipping through.
    if params.prior_count <= 0.0 || !params.prior_count.is_finite() {
        return Err(EdgeErrors::InvalidArgument(format!(
            "prior_count must be positive and finite, got {}",
            params.prior_count
        )));
    }
    if let Some(w) = weights {
        w.validate(n_genes, n_samples)?;
    }

    let mut lib = match lib_size {
        Some(l) => {
            check_lib_size(l, n_samples)?;
            l.to_vec()
        }
        None => column_sums(counts, n_genes, n_samples),
    };
    if params.normalize_method != NormMethod::None {
        let factors = calc_norm_factors(
            counts,
            n_genes,
            n_samples,
            Some(&lib),
            params.normalize_method,
            None,
            None,
        )?;
        for (l, f) in lib.iter_mut().zip(factors.iter()) {
            *l *= f;
        }
    }
    check_lib_size(&lib, n_samples)?;

    let span = if params.adaptive_span {
        let (small_n, min_span, power) = LIMMA_LOWESS_DEFAULTS;
        choose_lowess_span(n_genes, small_n, min_span, power)
    } else {
        params.span
    };
    if !(span > 0.0 && span <= 1.0) {
        return Err(EdgeErrors::InvalidArgument(format!(
            "span must lie in (0, 1]; got {span}"
        )));
    }

    let pc = params.prior_count;
    let lib_adj: Vec<f64> = lib.iter().map(|l| l + 2.0 * pc).collect();
    let log_lib_adj: Vec<f64> = lib_adj.iter().map(|l| l.log2()).collect();

    let mut e = vec![0.0_f64; n_genes * n_samples];
    e.par_chunks_mut(n_samples)
        .enumerate()
        .for_each(|(gene, row)| {
            let y = &counts[gene * n_samples..(gene + 1) * n_samples];
            for (j, v) in row.iter_mut().enumerate() {
                let count = y[j].to_f64().unwrap_or(f64::NAN);
                *v = ((count + pc) / lib_adj[j] * PER_MILLION).log2();
            }
        });

    Ok(Prepared {
        e,
        lib_size: lib,
        lib_adj,
        log_lib_adj,
        span,
    })
}

/// Column sums of a count matrix, which is voom's default library size.
///
/// ### Params
///
/// * `counts` - Row-major counts
/// * `n_genes` - Number of genes
/// * `n_samples` - Number of samples
///
/// ### Returns
///
/// One total per sample.
fn column_sums<T: EdgeFloat>(counts: &[T], n_genes: usize, n_samples: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; n_samples];
    for gene in 0..n_genes {
        let row = &counts[gene * n_samples..(gene + 1) * n_samples];
        for (acc, v) in out.iter_mut().zip(row.iter()) {
            *acc += v.to_f64().unwrap_or(f64::NAN);
        }
    }
    out
}

/// Checks that a library size vector is the right length and strictly positive.
///
/// ### Params
///
/// * `lib_size` - Library size per sample
/// * `n_samples` - Number of samples expected
///
/// ### Returns
///
/// `Ok(())`, or [`EdgeErrors`] on a length or positivity failure.
fn check_lib_size(lib_size: &[f64], n_samples: usize) -> Result<(), EdgeErrors> {
    if lib_size.len() != n_samples {
        return Err(EdgeErrors::LengthMismatch {
            name: "lib_size",
            expected: n_samples,
            got: lib_size.len(),
        });
    }
    if let Some(bad) = lib_size.iter().find(|v| !v.is_finite() || **v <= 0.0) {
        return Err(EdgeErrors::InvalidArgument(format!(
            "library sizes must be positive and finite, found {bad}"
        )));
    }
    Ok(())
}

/// Arithmetic mean of a gene's row.
///
/// Plain sum then divide, which is what R's `rowMeans` reduces to on a platform
/// where `long double` is `double`.
///
/// ### Params
///
/// * `row` - The values
///
/// ### Returns
///
/// Their mean.
fn row_mean(row: &[f64]) -> f64 {
    row.iter().sum::<f64>() / row.len() as f64
}

/// Mean over the observed entries of a row.
///
/// ### Params
///
/// * `row` - The values
/// * `observed` - One flag per value; `false` entries are skipped
///
/// ### Returns
///
/// The mean of the observed entries, or `NaN` if there are none.
fn masked_row_mean(row: &[f64], observed: &[bool]) -> f64 {
    let mut sum = 0.0;
    let mut n = 0_usize;
    for (v, keep) in row.iter().zip(observed.iter()) {
        if *keep {
            sum += v;
            n += 1;
        }
    }
    if n == 0 { f64::NAN } else { sum / n as f64 }
}

/// R's corrected mean: a naive pass followed by a mean-of-residuals correction.
///
/// `mean.default` on doubles does exactly this, and the shift it produces lands
/// in every `sx` value, so the correction has to be here for the trend to agree
/// to the last digits.
///
/// ### Params
///
/// * `x` - The values
///
/// ### Returns
///
/// Their mean.
fn corrected_mean(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let s = x.iter().sum::<f64>() / n;
    if !s.is_finite() {
        return s;
    }
    s + x.iter().map(|v| v - s).sum::<f64>() / n
}

/// Knots of a fitted mean-variance trend.
#[derive(Debug)]
struct Trend {
    /// Strictly increasing abscissae.
    x: Vec<f64>,
    /// Ordinates, one per abscissa.
    y: Vec<f64>,
}

/// Trend from Cleveland's lowess, the path limma takes when it has no weights.
///
/// ### Params
///
/// * `sx` - Average log-count per gene
/// * `sy` - Square root of the residual standard deviation per gene
/// * `span` - Lowess span
///
/// ### Returns
///
/// The knots, or [`EdgeErrors`] if fewer than two usable points remain.
fn plain_trend(sx: &[f64], sy: &[f64], span: f64) -> Result<Trend, EdgeErrors> {
    let (x, y) = finite_pairs(sx, sy);
    if x.len() < 2 {
        return Err(EdgeErrors::InvalidArgument(
            "fewer than two genes carry a usable residual scale; no mean-variance trend exists"
                .to_string(),
        ));
    }
    let fitted = lowess(&x, &y, span, LOWESS_ITERATIONS)?;
    collapse(&x, &fitted)
}

/// Trend from limma's `weightedLowess`, the path `voomLmFit` takes once it has
/// masked structural zeros and the residual degrees of freedom vary by gene.
///
/// ### Params
///
/// * `sx` - Average log-count per gene
/// * `sy` - Square root of the residual standard deviation per gene
/// * `df` - Residual degrees of freedom per gene, used as prior weights
/// * `span` - Lowess span
///
/// ### Returns
///
/// The knots, or [`EdgeErrors`] if fewer than two usable points remain.
fn weighted_trend(sx: &[f64], sy: &[f64], df: &[f64], span: f64) -> Result<Trend, EdgeErrors> {
    let mut x = Vec::with_capacity(sx.len());
    let mut y = Vec::with_capacity(sx.len());
    let mut w = Vec::with_capacity(sx.len());
    for ((a, b), d) in sx.iter().zip(sy.iter()).zip(df.iter()) {
        if a.is_finite() && b.is_finite() {
            x.push(*a);
            y.push(*b);
            w.push(*d);
        }
    }
    if x.len() < 2 {
        return Err(EdgeErrors::InvalidArgument(
            "fewer than two genes carry a usable residual scale; no mean-variance trend exists"
                .to_string(),
        ));
    }
    let params = LowessParams {
        span,
        ..LowessParams::default()
    };
    let fit = weighted_lowess(&x, &y, Some(&w), Some(params))?;
    collapse(&x, &fit.fitted)
}

/// Drops the pairs either half of which is not finite.
///
/// ### Params
///
/// * `sx` - Abscissae
/// * `sy` - Ordinates
///
/// ### Returns
///
/// The surviving pairs, in input order.
fn finite_pairs(sx: &[f64], sy: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut x = Vec::with_capacity(sx.len());
    let mut y = Vec::with_capacity(sx.len());
    for (a, b) in sx.iter().zip(sy.iter()) {
        if a.is_finite() && b.is_finite() {
            x.push(*a);
            y.push(*b);
        }
    }
    (x, y)
}

/// Sorts a smooth by abscissa and averages the ordinates of tied abscissae.
///
/// This is `approxfun(l, ties = list("ordered", mean))` applied to a lowess
/// output. The averaging is a formality when the smoother has already given
/// tied points the same fitted value, but the deduplication is not: the
/// interpolator needs strictly increasing knots.
///
/// ### Params
///
/// * `x` - Abscissae, in any order
/// * `fitted` - Smoothed ordinates, one per abscissa, in the same order
///
/// ### Returns
///
/// Strictly increasing knots, or [`EdgeErrors::InvalidArgument`] if fewer than
/// two distinct abscissae survive, which is where R's `approxfun` also gives up.
fn collapse(x: &[f64], fitted: &[f64]) -> Result<Trend, EdgeErrors> {
    let mut order: Vec<usize> = (0..x.len()).collect();
    order.sort_by(|&a, &b| x[a].total_cmp(&x[b]));

    let mut kx: Vec<f64> = Vec::with_capacity(x.len());
    let mut ky: Vec<f64> = Vec::with_capacity(x.len());
    let mut i = 0;
    while i < order.len() {
        let xi = x[order[i]];
        let mut j = i;
        let mut sum = 0.0;
        while j < order.len() && x[order[j]] == xi {
            sum += fitted[order[j]];
            j += 1;
        }
        kx.push(xi);
        ky.push(sum / (j - i) as f64);
        i = j;
    }

    if kx.len() < 2 {
        return Err(EdgeErrors::InvalidArgument(
            "the mean-variance trend collapsed to a single abscissa; nothing to interpolate"
                .to_string(),
        ));
    }
    Ok(Trend { x: kx, y: ky })
}

/// Reads the trend back at every fitted value and turns it into a weight.
///
/// The fitted log2-CPM is converted to a fitted log2-count with the same
/// library sizes the transform used, the trend is evaluated there with constant
/// extension beyond the end knots (`approxfun(rule = 2)`), and the weight is the
/// reciprocal fourth power. Genes are the parallel axis; each gene interpolates
/// its own row.
///
/// ### Params
///
/// * `fitted` - Fitted log2-CPM, row-major `n_genes * n_samples`
/// * `n_genes` - Number of genes
/// * `n_samples` - Number of samples
/// * `lib_adj` - `lib_size + 2 * prior_count` per sample
/// * `trend` - The fitted trend, with strictly increasing abscissae
///
/// ### Returns
///
/// Row-major `n_genes * n_samples` precision weights.
fn voom_weights(
    fitted: &[f64],
    n_genes: usize,
    n_samples: usize,
    lib_adj: &[f64],
    trend: &Trend,
) -> Vec<f64> {
    let mut out = vec![0.0_f64; n_genes * n_samples];
    let fill = |(row, fit_row): (&mut [f64], &[f64])| {
        for ((v, f), l) in row.iter_mut().zip(fit_row.iter()).zip(lib_adj.iter()) {
            // 2^x rather than exp2(x): R's `^` dispatches to pow(2, x) and the
            // two libm entry points disagree in the last bit.
            *v = (INV_PER_MILLION * (2.0_f64.powf(*f) * l)).log2();
        }
        // Cannot fail: `collapse` guarantees at least two strictly increasing
        // knots, which is all `interp_linear_extrap` validates.
        let t = interp_linear_extrap(row, &trend.x, &trend.y)
            .expect("trend knots were validated when the trend was built");
        for (v, ti) in row.iter_mut().zip(t.iter()) {
            let square = ti * ti;
            *v = 1.0 / (square * square);
        }
    };

    if n_genes * n_samples >= PARALLEL_WORK_THRESHOLD {
        out.par_chunks_mut(n_samples)
            .zip(fitted.par_chunks(n_samples))
            .for_each(fill);
    } else {
        out.chunks_mut(n_samples)
            .zip(fitted.chunks(n_samples))
            .for_each(fill);
    }
    out
}

/// Assembles the result, honouring `save_trend`.
///
/// ### Params
///
/// * `e` - log2-CPM matrix
/// * `weights` - Precision weights
/// * `trend` - The fitted trend
/// * `save_trend` - Whether to keep it
///
/// ### Returns
///
/// The result.
fn finish(e: Vec<f64>, weights: Vec<f64>, trend: Trend, save_trend: bool) -> VoomResult {
    let (trend_x, trend_y) = if save_trend {
        (trend.x, trend.y)
    } else {
        (Vec::new(), Vec::new())
    };
    VoomResult {
        e,
        weights,
        trend_x,
        trend_y,
    }
}

/// The result upstream returns when the design has no replication to speak of.
///
/// ### Params
///
/// * `e` - log2-CPM matrix
/// * `n_genes` - Number of genes
/// * `n_samples` - Number of samples
///
/// ### Returns
///
/// The expression matrix with every weight set to one and no trend.
fn no_trend_result(e: Vec<f64>, n_genes: usize, n_samples: usize) -> VoomResult {
    VoomResult {
        e,
        weights: vec![1.0; n_genes * n_samples],
        trend_x: Vec::new(),
        trend_y: Vec::new(),
    }
}

//////////////////////
// Structural zeros //
//////////////////////

/// Which genes carry structural zeros, and where.
struct StructuralZeros {
    /// Gene indices, in increasing order.
    rows: Vec<usize>,
    /// One flag per sample of each listed gene, row-major `rows.len() *
    /// n_samples`. `false` marks an observation to mask out.
    observed: Vec<bool>,
}

/// Finds observations that are zero because the design makes them zero.
///
/// A gene with a whole group of exact zeros has a residual standard deviation
/// that says nothing about its mean-variance behaviour, because the zeros are
/// determined rather than sampled. edgeR's test is two-stage: shortlist rows
/// with more zeros than the smallest group the design can resolve, then fit a
/// Poisson GLM to those rows and keep the entries where both the count and the
/// fitted value are zero. The smallest resolvable group is `1 / max(h)`, the
/// reciprocal of the largest leverage.
///
/// ### Params
///
/// * `counts` - Row-major counts
/// * `n_genes` - Number of genes
/// * `n_samples` - Number of samples
/// * `design` - Row-major design matrix
/// * `n_coef` - Number of coefficients
/// * `lib_size` - Library size per sample, which becomes the Poisson offset
///
/// ### Returns
///
/// The affected genes and their observation masks, or [`EdgeErrors`] if the
/// Poisson fit fails.
fn detect_structural_zeros<T: EdgeFloat>(
    counts: &[T],
    n_genes: usize,
    n_samples: usize,
    design: &[f64],
    n_coef: usize,
    lib_size: &[f64],
) -> Result<StructuralZeros, EdgeErrors> {
    let empty = StructuralZeros {
        rows: Vec::new(),
        observed: Vec::new(),
    };

    let h = hat_diagonal(design, n_samples, n_coef)?;
    let max_h = h.iter().fold(0.0_f64, |acc, v| acc.max(*v));
    if max_h <= 0.0 {
        return Ok(empty);
    }
    let threshold = MIN_ZERO_RUN.max(1.0 / max_h) - STRUCTURAL_ZERO_EPS;

    let eps = T::from(STRUCTURAL_ZERO_EPS).unwrap_or_else(T::zero);
    let candidates: Vec<usize> = (0..n_genes)
        .filter(|&g| {
            let row = &counts[g * n_samples..(g + 1) * n_samples];
            row.iter().filter(|v| **v < eps).count() as f64 > threshold
        })
        .collect();
    if candidates.is_empty() {
        return Ok(empty);
    }

    let mut sub = Vec::with_capacity(candidates.len() * n_samples);
    for &g in &candidates {
        sub.extend_from_slice(&counts[g * n_samples..(g + 1) * n_samples]);
    }
    let offset = Recycled::by_sample(lib_size.iter().map(|l| l.ln()).collect());
    let poisson = glm_fit(
        &sub,
        candidates.len(),
        n_samples,
        design,
        n_coef,
        &Recycled::scalar(0.0),
        &offset,
        None,
        0.0,
    )?;

    let mut rows = Vec::new();
    let mut observed = Vec::new();
    for (k, &g) in candidates.iter().enumerate() {
        let count_row = &sub[k * n_samples..(k + 1) * n_samples];
        let fit_row = &poisson.fitted[k * n_samples..(k + 1) * n_samples];
        let mask: Vec<bool> = count_row
            .iter()
            .zip(fit_row.iter())
            .map(|(c, f)| !(*f < STRUCTURAL_ZERO_EPS && *c < eps))
            .collect();
        if mask.iter().any(|keep| !keep) {
            rows.push(g);
            observed.extend_from_slice(&mask);
        }
    }

    Ok(StructuralZeros { rows, observed })
}

/// Residual scale and degrees of freedom of one gene over a subset of samples.
///
/// limma's `lm.series` drops the missing observations, refits on the reduced
/// design, and reports `sqrt(RSS / df)` with `df = n_observed - rank`. This does
/// the same, projecting onto the column space with a thin SVD so that a reduced
/// design that has lost its rank is handled rather than silently mis-scaled.
///
/// ### Params
///
/// * `y` - The gene's log2-CPM row, `n_samples` values
/// * `observed` - One flag per sample; `false` entries are dropped
/// * `design` - Row-major design matrix, `n_samples * n_coef`
/// * `n_samples` - Number of samples
/// * `n_coef` - Number of coefficients
/// * `weights` - Per-sample weights for this gene, or `None` for all ones
///
/// ### Returns
///
/// `(sigma, df_residual)`, with `sigma` set to `NaN` when there are no residual
/// degrees of freedom, or [`EdgeErrors`] if the decomposition fails.
fn masked_fit(
    y: &[f64],
    observed: &[bool],
    design: &[f64],
    n_samples: usize,
    n_coef: usize,
    weights: Option<&[f64]>,
) -> Result<(f64, f64), EdgeErrors> {
    let keep: Vec<usize> = (0..n_samples).filter(|&j| observed[j]).collect();
    let n_obs = keep.len();
    if n_obs == 0 {
        return Ok((f64::NAN, 0.0));
    }

    // Row scaling by sqrt(w) turns weighted least squares into ordinary least
    // squares, which is what `lm.wfit` does internally.
    let mut a = Vec::with_capacity(n_obs * n_coef);
    let mut z = Vec::with_capacity(n_obs);
    for &j in &keep {
        let scale = match weights {
            Some(w) => w[j].max(0.0).sqrt(),
            None => 1.0,
        };
        for c in 0..n_coef {
            a.push(design[j * n_coef + c] * scale);
        }
        z.push(y[j] * scale);
    }

    let mat = MatRef::from_row_major_slice(&a, n_obs, n_coef);
    let svd = mat
        .thin_svd()
        .map_err(|e| EdgeErrors::SolveFailed(format!("SVD of the reduced design failed: {e:?}")))?;
    let s = svd.S().column_vector();
    let largest = (0..s.nrows()).fold(0.0_f64, |acc, i| acc.max(s[i]));
    let tol = largest * n_obs.max(n_coef) as f64 * f64::EPSILON;
    let rank = (0..s.nrows()).filter(|&i| s[i] > tol).count();

    let df = n_obs as f64 - rank as f64;
    if df <= 0.0 {
        return Ok((f64::NAN, 0.0));
    }

    // RSS = ||z||^2 - ||U_r' z||^2, with U_r the left singular vectors that
    // actually span the column space.
    let u = svd.U();
    let total: f64 = z.iter().map(|v| v * v).sum();
    let explained: f64 = (0..rank)
        .map(|c| {
            let dot: f64 = (0..n_obs).map(|i| u[(i, c)] * z[i]).sum();
            dot * dot
        })
        .sum();
    let rss = (total - explained).max(0.0);

    Ok(((rss / df).sqrt(), df))
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    #![allow(clippy::excessive_precision)]

    use super::*;
    use approx::assert_relative_eq;

    /// Tolerance every reference comparison uses.
    ///
    /// The worst observed disagreement with R across these fixtures is 1.1e-13
    /// relative, on a voom weight, which is the fourth power of a trend value
    /// and so amplifies the lowess by roughly four ulps. This leaves two orders
    /// of magnitude of headroom over that.
    const TOL: f64 = 1e-11;

    /// R prints a matrix column-major; the crate stores it row-major.
    ///
    /// ### Params
    ///
    /// * `values` - Column-major values, exactly as `cat()` emitted them
    /// * `n_genes` - Number of rows
    /// * `n_samples` - Number of columns
    ///
    /// ### Returns
    ///
    /// The same matrix, row-major.
    fn from_r(values: &[f64], n_genes: usize, n_samples: usize) -> Vec<f64> {
        let mut out = vec![0.0; n_genes * n_samples];
        for j in 0..n_samples {
            for i in 0..n_genes {
                out[i * n_samples + j] = values[j * n_genes + i];
            }
        }
        out
    }

    /// Compares two row-major matrices entrywise.
    ///
    /// ### Params
    ///
    /// * `got` - Values produced here
    /// * `want` - Reference values
    fn assert_matrix(got: &[f64], want: &[f64]) {
        assert_eq!(got.len(), want.len());
        for (i, (a, b)) in got.iter().zip(want.iter()).enumerate() {
            assert_relative_eq!(a, b, max_relative = TOL, epsilon = 1e-12);
            let _ = i;
        }
    }

    /// The four-gene matrix used throughout, row-major.
    fn counts4() -> Vec<f64> {
        vec![
            8.0, 16.0, 32.0, 64.0, 128.0, 256.0, //
            4.0, 4.0, 8.0, 8.0, 16.0, 16.0, //
            64.0, 64.0, 64.0, 128.0, 128.0, 128.0, //
            12.0, 20.0, 28.0, 40.0, 56.0, 72.0,
        ]
    }

    /// Two-group design with an intercept, row-major `6 * 2`.
    fn design6() -> Vec<f64> {
        vec![
            1.0, 0.0, //
            1.0, 0.0, //
            1.0, 0.0, //
            1.0, 1.0, //
            1.0, 1.0, //
            1.0, 1.0,
        ]
    }

    /// Six-gene matrix whose fifth gene is zero throughout the first group.
    fn counts6() -> Vec<f64> {
        vec![
            8.0, 16.0, 32.0, 64.0, 128.0, 256.0, //
            4.0, 4.0, 8.0, 8.0, 16.0, 16.0, //
            64.0, 64.0, 64.0, 128.0, 128.0, 128.0, //
            12.0, 20.0, 28.0, 40.0, 56.0, 72.0, //
            0.0, 0.0, 0.0, 64.0, 128.0, 256.0, //
            32.0, 64.0, 32.0, 64.0, 32.0, 64.0,
        ]
    }

    // -- voom against limma --

    /// `Rscript -e 'suppressMessages(library(limma)); y <- matrix(c(8,16,32,64,128,256,
    /// 4,4,8,8,16,16, 64,64,64,128,128,128, 12,20,28,40,56,72), nrow=4, byrow=TRUE);
    /// X <- cbind(1, c(0,0,0,1,1,1)); v <- voom(y, X); cat(v$E, "\n"); cat(v$weights, "\n")'`
    #[test]
    fn test_voom_matches_limma_at_the_defaults() {
        let e_ref = from_r(
            &[
                16.543297979608116,
                15.625760139800089,
                19.467062393781031,
                17.099691328132501,
                17.261717171016503,
                15.387248053100365,
                19.228550307081306,
                17.574875056276134,
                17.898653946851439,
                15.963748975073324,
                18.887513389246237,
                17.709176147987726,
                18.029906488517465,
                15.106142074344552,
                19.02430378228809,
                17.358529235978839,
                18.575249344782812,
                15.614018914947385,
                18.575249344782812,
                17.38980375800412,
                19.048687211591833,
                15.090266315343232,
                18.051496745178657,
                17.225781285999712,
            ],
            4,
            6,
        );
        let w_ref = from_r(
            &[
                12.989516763018962,
                11.788172674942306,
                9.3400041055101237,
                13.424969710279226,
                13.448173520443326,
                11.788172674942306,
                8.0405860435236782,
                13.903012579712057,
                14.139735668616346,
                11.788172674942306,
                6.5509069836352065,
                14.624125584031328,
                5.868810395011649,
                12.043013330985403,
                5.868810395011649,
                12.619269054888733,
                5.868810395011649,
                12.843401260887665,
                5.868810395011649,
                9.2417883578712434,
                5.868810395011649,
                13.862823955356282,
                5.868810395011649,
                6.7037143001373245,
            ],
            4,
            6,
        );

        let v = voom(&counts4(), 4, 6, &design6(), 2, None, None, None).unwrap();
        assert_matrix(&v.e, &e_ref);
        assert_matrix(&v.weights, &w_ref);
        // Four genes, all with a distinct abundance, so four trend knots.
        assert_eq!(v.trend_x.len(), 4);
        assert!(v.trend_x.windows(2).all(|w| w[0] < w[1]));
    }

    /// `Rscript -e '...; v2 <- voom(y, X, span=0.5, adaptive.span=FALSE); cat(v2$weights, "\n")'`
    #[test]
    fn test_voom_with_a_fixed_span() {
        let w_ref = from_r(
            &[
                13.4903049167572,
                11.583173793916272,
                3.2580125050046709,
                14.214268703039076,
                14.253336810331142,
                11.583173793916272,
                3.9361590954998409,
                15.029198753333942,
                15.44058688155137,
                11.583173793916272,
                5.2441514872140278,
                16.298637918552512,
                6.2131371711955588,
                11.976716996930278,
                6.2131371711955588,
                5.7725491799065809,
                6.2131371711955588,
                13.251289157596918,
                6.2131371711955588,
                3.3000750221847794,
                6.2131371711955588,
                14.959873649851554,
                6.2131371711955588,
                5.0683737001853411,
            ],
            4,
            6,
        );
        let params = VoomParams {
            adaptive_span: false,
            ..VoomParams::default()
        };
        let v = voom(&counts4(), 4, 6, &design6(), 2, None, None, Some(params)).unwrap();
        assert_matrix(&v.weights, &w_ref);
    }

    /// `Rscript -e '...; ls <- c(1024,2048,4096,8192,16384,32768); v3 <- voom(y, X, lib.size=ls);
    /// cat(v3$E, "\n"); cat(v3$weights, "\n")'`
    #[test]
    fn test_voom_with_explicit_library_sizes() {
        let e_ref = from_r(
            &[
                13.017623216181706,
                12.100085376373679,
                15.941387630354621,
                13.57401656470609,
                12.97525841967138,
                11.10078930175524,
                14.942091555736182,
                13.288416304931012,
                12.953584204872328,
                11.018679233094213,
                13.942443647267128,
                12.764106406008615,
                12.942619725260986,
                10.018855311088071,
                13.937017019031609,
                12.271242472722356,
                12.937105066087931,
                9.9758746362525059,
                12.937105066087931,
                11.751659479309239,
                12.934339558044401,
                8.9759186617958004,
                11.937149091631225,
                11.111433632452281,
            ],
            4,
            6,
        );
        let w_ref = from_r(
            &[
                2.7536849673907171,
                2.7536849673907171,
                5.2741383203758962,
                2.9131289915881435,
                3.7547718985781091,
                2.7536849673907171,
                1.787312986192773,
                4.0578651158269201,
                5.1699701092509223,
                3.102797273267663,
                1.1065455050978712,
                3.9756837393164166,
                1.7955399673136154,
                2.7536849673907171,
                1.7972552974054368,
                4.8459755006154195,
                1.1065455050978712,
                3.3669376329697984,
                1.1065455050978712,
                2.2813486326451544,
                1.1065455050978712,
                4.7513384094313045,
                1.1065455050978712,
                1.1065455050978712,
            ],
            4,
            6,
        );
        let lib = [1024.0, 2048.0, 4096.0, 8192.0, 16384.0, 32768.0];
        let v = voom(&counts4(), 4, 6, &design6(), 2, Some(&lib), None, None).unwrap();
        assert_matrix(&v.e, &e_ref);
        assert_matrix(&v.weights, &w_ref);
    }

    /// `Rscript -e '...; W <- matrix(c(1,1,1,2,2,2, 1,1,1,1,1,1, 2,2,2,1,1,1, 1,2,1,2,1,2),
    /// nrow=4, byrow=TRUE); v4 <- voom(y, X, weights=W); cat(v4$weights, "\n")'`
    ///
    /// limma feeds the prior weights to `lmFit` but returns the voom weights on
    /// their own, so the reference is not the elementwise product.
    #[test]
    fn test_voom_with_prior_weights() {
        let w_ref = from_r(
            &[
                12.545829169909583,
                11.821385659106866,
                7.7865434695555136,
                12.83266231656685,
                12.813765454846195,
                11.821385659106866,
                6.5731002482206105,
                13.108298328154543,
                13.209424386966319,
                11.821385659106866,
                5.2204087038675269,
                13.515402260955836,
                4.6159929524537251,
                11.977939455603519,
                4.6159929524537251,
                11.094090814823417,
                4.6159929524537251,
                12.459505266540475,
                4.6159929524537251,
                7.7666894383291645,
                4.6159929524537251,
                13.052165669301132,
                4.6159929524537251,
                5.4034876846567874,
            ],
            4,
            6,
        );
        let prior = Recycled::full(
            vec![
                1.0, 1.0, 1.0, 2.0, 2.0, 2.0, //
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, //
                2.0, 2.0, 2.0, 1.0, 1.0, 1.0, //
                1.0, 2.0, 1.0, 2.0, 1.0, 2.0,
            ],
            4,
            6,
        )
        .unwrap();
        let v = voom(&counts4(), 4, 6, &design6(), 2, None, Some(&prior), None).unwrap();
        assert_matrix(&v.weights, &w_ref);
    }

    /// `Rscript -e 'suppressMessages(library(edgeR)); ...; nf <- calcNormFactors(y, method="TMM");
    /// v5 <- voom(y, X, lib.size=colSums(y)*nf); cat(v5$E, "\n"); cat(v5$weights, "\n")'`
    #[test]
    fn test_voom_with_tmm_normalisation() {
        let e_ref = from_r(
            &[
                16.861265746711329,
                15.943727906903302,
                19.785030160884244,
                17.417659095235713,
                17.089336967658667,
                15.214867849742527,
                19.056170103723471,
                17.402494852918299,
                17.585371625078697,
                15.650466653300583,
                18.574231067473498,
                17.395893826214984,
                18.062279282451577,
                15.138514868278664,
                19.056676576222202,
                17.390902029912947,
                18.572999990825448,
                15.611769560990021,
                18.572999990825448,
                17.387554404046757,
                19.185470125772795,
                15.227049229524194,
                18.188279659359619,
                17.362564200180675,
            ],
            4,
            6,
        );
        let w_ref = from_r(
            &[
                17.264659803237596,
                10.507917099333257,
                156.3323671159518,
                26.136042500464217,
                79.154159262930762,
                10.507917099333257,
                9.0863357877747521,
                147.490684118144,
                341.8801646693878,
                10.507917099333257,
                3.305171391993599,
                867.70087742282089,
                3.305171391993599,
                13.855965744683104,
                3.305171391993599,
                193.21189552796963,
                3.305171391993599,
                33.636099355143109,
                3.305171391993599,
                21.586347629305376,
                3.305171391993599,
                81.076074427711745,
                3.305171391993599,
                6.8749514771531359,
            ],
            4,
            6,
        );
        let params = VoomParams {
            normalize_method: NormMethod::Tmm,
            ..VoomParams::default()
        };
        let v = voom(&counts4(), 4, 6, &design6(), 2, None, None, Some(params)).unwrap();
        assert_matrix(&v.e, &e_ref);
        assert_matrix(&v.weights, &w_ref);
    }

    /// `Rscript -e '...; ya <- yz; ya[5,] <- 0; va <- voom(ya, X); cat(va$E, "\n");
    /// cat(va$weights, "\n")'` with `yz` the six-gene matrix from [`counts6`].
    ///
    /// Gene five is zero in every sample, so limma drops it from the trend but
    /// still weights it.
    #[test]
    fn test_voom_drops_all_zero_genes_from_the_trend() {
        let mut counts = counts6();
        for j in 0..6 {
            counts[4 * 6 + j] = 0.0;
        }
        let w_ref = from_r(
            &[
                6.0850468609983226,
                5.8746685507205978,
                4.9855632007775563,
                6.1616972871648121,
                5.8746685507205978,
                6.4834658553198903,
                6.2494805425900557,
                5.8746685507205978,
                5.3148308887908016,
                6.3287329314830183,
                5.8746685507205978,
                4.873188492141316,
                6.2375083696851856,
                5.8746685507205978,
                5.2903367057336022,
                6.3165707242276845,
                5.8746685507205978,
                4.949636893577944,
                5.5522780635723539,
                5.9627219351602356,
                5.5522780635723539,
                5.1350268805368637,
                5.8746685507205978,
                5.4435184700247508,
                5.5522780635723539,
                6.0429322786662558,
                5.5522780635723539,
                4.9650536255914721,
                5.8746685507205978,
                4.9013463087604547,
                5.5522780635723539,
                6.2372962172784252,
                5.5522780635723539,
                5.357390040024689,
                5.8746685507205978,
                5.2873399170816571,
            ],
            6,
            6,
        );
        let v = voom(&counts, 6, 6, &design6(), 2, None, None, None).unwrap();
        assert_matrix(&v.weights, &w_ref);
        // Five genes contribute to the trend; the all-zero one does not.
        assert_eq!(v.trend_x.len(), 5);
    }

    /// `Rscript -e '...; vz <- voom(yz, X); cat(vz$weights, "\n")'`
    ///
    /// The same matrix `voom_lmfit` treats as having structural zeros. Plain
    /// `voom` has no such machinery, so the weights differ; this pins the
    /// difference down rather than leaving it implicit.
    #[test]
    fn test_voom_ignores_structural_zeros() {
        let w_ref = from_r(
            &[
                6.3036181560952809,
                6.6281248871960514,
                4.2106003725631087,
                6.2220948255866508,
                6.6281248871960514,
                5.807719481896001,
                6.1318458318826385,
                6.6281248871960514,
                3.9716140662581934,
                6.0530854492544925,
                6.6281248871960514,
                4.3046020876481803,
                6.1439644820277231,
                6.6281248871960514,
                3.9881741062630556,
                6.065009793075042,
                6.6281248871960514,
                4.3826380242821577,
                3.8223542300269586,
                6.4955267502860323,
                3.8227784629411659,
                4.9986460622971682,
                3.8223542300269564,
                5.3196811821704806,
                3.8199973957099402,
                6.3448481701412751,
                3.8199973957099402,
                4.2201332208554003,
                3.8199973957099402,
                4.2712404960030694,
                3.8199973957099402,
                6.0961758376170474,
                3.8199973957099402,
                3.8788546942794055,
                3.8199973957099402,
                3.9248417971148495,
            ],
            6,
            6,
        );
        let v = voom(&counts6(), 6, 6, &design6(), 2, None, None, None).unwrap();
        assert_matrix(&v.weights, &w_ref);
    }

    /// `Rscript -e '...; vn <- voom(matrix(c(8,16,4,8,64,128,12,20), nrow=4, byrow=TRUE),
    /// cbind(c(1,0), c(0,1))); cat(vn$E, "\n"); cat(vn$weights, "\n")'`
    #[test]
    fn test_voom_without_replication_returns_unit_weights() {
        let counts = vec![8.0, 16.0, 4.0, 8.0, 64.0, 128.0, 12.0, 20.0];
        let design = vec![1.0, 0.0, 0.0, 1.0];
        let e_ref = from_r(
            &[
                16.543297979608116,
                15.625760139800089,
                19.467062393781031,
                17.099691328132501,
                16.541334461045903,
                15.584403182937789,
                19.502564890881327,
                16.854492346305534,
            ],
            4,
            2,
        );
        let v = voom(&counts, 4, 2, &design, 2, None, None, None).unwrap();
        assert_matrix(&v.e, &e_ref);
        assert_eq!(v.weights, vec![1.0; 8]);
        assert!(v.trend_x.is_empty());
    }

    // -- voom_lmfit against edgeR --

    /// `Rscript -e 'suppressMessages(library(edgeR)); ...; f <- voomLmFit(y, X);
    /// cat(f$coefficients, "\n"); cat(f$sigma, "\n"); cat(f$stdev.unscaled, "\n");
    /// cat(f$df.residual, "\n"); cat(f$EList$weights, "\n")'`
    #[test]
    fn test_voom_lmfit_matches_edger() {
        let coef_ref = from_r(
            &[
                17.253688064337041,
                15.658919055991259,
                19.228282991681084,
                17.469629066273878,
                1.2975929506269903,
                -0.39012131564957925,
                -0.67793303426456375,
                -0.13213524493363815,
            ],
            4,
            2,
        );
        let sdu_ref = from_r(
            &[
                0.15698485386706904,
                0.1681574559185659,
                0.20441608469903982,
                0.1543914004557172,
                0.28537987300047385,
                0.23255943049211469,
                0.31397987714786207,
                0.24257959742479149,
            ],
            4,
            2,
        );
        let sigma_ref = [
            1.9677816645250998,
            1.0331365101630736,
            1.0094164261014424,
            0.85804511835137942,
        ];
        // voomLmFit with no prior weights, no block and no sample weights leaves
        // the voom weights untouched, so they match limma's `voom` exactly.
        let w_ref = from_r(
            &[
                12.989516763018962,
                11.788172674942306,
                9.3400041055101237,
                13.424969710279226,
                13.448173520443326,
                11.788172674942306,
                8.0405860435236782,
                13.903012579712057,
                14.139735668616346,
                11.788172674942306,
                6.5509069836352065,
                14.624125584031328,
                5.868810395011649,
                12.043013330985403,
                5.868810395011649,
                12.619269054888733,
                5.868810395011649,
                12.843401260887665,
                5.868810395011649,
                9.2417883578712434,
                5.868810395011649,
                13.862823955356282,
                5.868810395011649,
                6.7037143001373245,
            ],
            4,
            6,
        );

        let (v, fit) = voom_lmfit(&counts4(), 4, 6, &design6(), 2, None, None, None).unwrap();
        assert_matrix(&v.weights, &w_ref);
        assert_matrix(&fit.coefficients, &coef_ref);
        assert_matrix(&fit.stdev_unscaled, &sdu_ref);
        assert_matrix(&fit.sigma, &sigma_ref);
        assert_eq!(fit.df_residual, vec![4.0; 4]);
    }

    /// `Rscript -e 'suppressMessages(library(edgeR)); yz <- matrix(c(8,16,32,64,128,256,
    /// 4,4,8,8,16,16, 64,64,64,128,128,128, 12,20,28,40,56,72, 0,0,0,64,128,256,
    /// 32,64,32,64,32,64), nrow=6, byrow=TRUE); X <- cbind(1, c(0,0,0,1,1,1));
    /// fz <- voomLmFit(yz, X); cat(fz$sigma, "\n"); cat(fz$df.residual, "\n");
    /// cat(fz$coefficients, "\n"); cat(fz$EList$weights, "\n")'`
    ///
    /// Gene five is zero throughout the first group, which is a structural zero
    /// rather than a low count: its three zeros are masked out, its degrees of
    /// freedom drop from four to two, and the trend switches to
    /// `weightedLowess`.
    #[test]
    fn test_voom_lmfit_handles_structural_zeros() {
        let coef_ref = from_r(
            &[
                16.753484246892054,
                15.178651221434963,
                18.719852973417467,
                16.980581346187428,
                11.702880273389976,
                18.050494281497048,
                1.1538183851209396,
                -0.55205177082949897,
                -0.81339318167408614,
                -0.29198995978922893,
                6.2044223586230114,
                -1.4000487474400647,
            ],
            6,
            2,
        );
        let sigma_ref = [
            1.4594821094354082,
            1.0416715905373415,
            0.88148685337439425,
            0.7029086399970913,
            0.90424348699908519,
            1.3583674345722949,
        ];
        let sdu_ref = from_r(
            &[
                0.23643807278872803,
                0.23549352491466158,
                0.28343765582211972,
                0.23666360929282665,
                0.23549352491466158,
                0.26036977579072357,
                0.37293618339558077,
                0.33347652801626071,
                0.40436879624326538,
                0.36123249547197228,
                0.37233806533625696,
                0.37441894800497211,
            ],
            6,
            2,
        );
        let w_ref = from_r(
            &[
                5.9782694333283963,
                6.0106412092567689,
                4.3028323106194417,
                5.9668730729180517,
                6.0106412092567689,
                5.8657264290895395,
                5.9540697090692882,
                6.0106412092567689,
                4.0690464144677367,
                5.9427309705966955,
                6.0106412092567689,
                4.4045079426983227,
                5.9558005822322064,
                6.0106412092567689,
                4.0757006689912068,
                5.9444577243175951,
                6.0106412092567689,
                4.4806772766902538,
                4.0081001112951933,
                6.0044873803847505,
                4.0082758683797346,
                5.0819774089678535,
                4.0081001112951933,
                5.3937399916896087,
                4.0071234206812321,
                5.9839733576330847,
                4.0071234206812321,
                4.3132015349234951,
                4.0071234206812321,
                4.3688248362346691,
                4.0071234206812321,
                5.948953823474957,
                4.0071234206812321,
                4.0313791316109793,
                4.0071234206812321,
                4.0501377059681651,
            ],
            6,
            6,
        );

        let (v, fit) = voom_lmfit(&counts6(), 6, 6, &design6(), 2, None, None, None).unwrap();
        assert_matrix(&v.weights, &w_ref);
        assert_matrix(&fit.coefficients, &coef_ref);
        assert_matrix(&fit.stdev_unscaled, &sdu_ref);
        assert_matrix(&fit.sigma, &sigma_ref);
        assert_eq!(fit.df_residual, vec![4.0, 4.0, 4.0, 4.0, 2.0, 4.0]);
    }

    /// The structural-zero detection is what separates `voom_lmfit` from
    /// `voom` on this matrix; without it the weights would be the ones
    /// [`test_voom_ignores_structural_zeros`] pins.
    #[test]
    fn test_voom_lmfit_and_voom_disagree_on_structural_zeros() {
        let v = voom(&counts6(), 6, 6, &design6(), 2, None, None, None).unwrap();
        let (vl, _) = voom_lmfit(&counts6(), 6, 6, &design6(), 2, None, None, None).unwrap();
        assert!(
            v.weights
                .iter()
                .zip(vl.weights.iter())
                .any(|(a, b)| (a - b).abs() > 1e-6)
        );
        assert_matrix(&v.e, &vl.e);
    }

    // -- voom_basic --

    /// `Rscript -e 'suppressMessages(library(limma)); ...; nf <- c(0.8,1.25,1,1,1.25,0.8);
    /// vb <- voom(y, X, lib.size=colSums(y)*nf, span=0.5, adaptive.span=FALSE);
    /// cat(vb$E, "\n"); cat(vb$weights, "\n")'`
    #[test]
    fn test_voom_basic_matches_limma() {
        let e_ref = from_r(
            &[
                16.861179241432776,
                15.943641401624749,
                19.784943655605691,
                17.41757258995716,
                16.942539687145178,
                15.068070569229036,
                18.909372823209978,
                17.255697572404809,
                17.898653946851439,
                15.963748975073324,
                18.887513389246237,
                17.709176147987726,
                18.029906488517465,
                15.106142074344552,
                19.02430378228809,
                17.358529235978839,
                18.25419853483637,
                15.292968105000945,
                18.25419853483637,
                17.068752948057679,
                19.369852984114885,
                15.411432087866284,
                18.372662517701709,
                17.546947058522765,
            ],
            4,
            6,
        );
        let w_ref = from_r(
            &[
                8.2304052146780471,
                7.0117701095209126,
                3.3906406912974139,
                9.13572408412257,
                12.514602283760407,
                7.0117701095209126,
                3.9454716483144021,
                14.054580240819252,
                12.653594344317138,
                7.0117701095209126,
                4.0078292963001818,
                14.215303796935652,
                4.6079138658233783,
                7.4933637530713577,
                4.6079138658233783,
                5.1608543770166948,
                4.6079138658233783,
                10.711666654383157,
                4.6079138658233783,
                3.3815506858146174,
                4.6079138658233783,
                10.117309391034542,
                4.6079138658233783,
                3.1191105441426035,
            ],
            4,
            6,
        );
        let nf = [0.8, 1.25, 1.0, 1.0, 1.25, 0.8];
        let v = voom_basic(&counts4(), 4, 6, &design6(), 2, None, Some(&nf), 0.5).unwrap();
        assert_matrix(&v.e, &e_ref);
        assert_matrix(&v.weights, &w_ref);
    }

    /// `voom_basic` with unit factors is `voom` at a fixed span of 0.5.
    #[test]
    fn test_voom_basic_without_factors_is_voom_at_span_half() {
        let params = VoomParams {
            adaptive_span: false,
            ..VoomParams::default()
        };
        let a = voom(&counts4(), 4, 6, &design6(), 2, None, None, Some(params)).unwrap();
        let b = voom_basic(&counts4(), 4, 6, &design6(), 2, None, None, 0.5).unwrap();
        assert_matrix(&a.weights, &b.weights);
        assert_matrix(&a.e, &b.e);
    }

    // -- options --

    /// `save_trend = false` drops the knots and nothing else.
    #[test]
    fn test_save_trend_off_drops_the_knots() {
        let params = VoomParams {
            save_trend: false,
            ..VoomParams::default()
        };
        let with = voom(&counts4(), 4, 6, &design6(), 2, None, None, None).unwrap();
        let without = voom(&counts4(), 4, 6, &design6(), 2, None, None, Some(params)).unwrap();
        assert!(without.trend_x.is_empty() && without.trend_y.is_empty());
        assert_matrix(&with.weights, &without.weights);
    }

    /// Counts held as `f32` reach the same weights: the transform upcasts.
    #[test]
    fn test_f32_counts_agree_with_f64() {
        let wide: Vec<f32> = counts4().iter().map(|v| *v as f32).collect();
        let a = voom(&counts4(), 4, 6, &design6(), 2, None, None, None).unwrap();
        let b = voom(&wide, 4, 6, &design6(), 2, None, None, None).unwrap();
        assert_matrix(&a.weights, &b.weights);
    }

    // -- error branches --

    #[test]
    fn test_rejects_a_single_gene() {
        let err = voom(&[1.0, 2.0], 1, 2, &[1.0, 1.0], 1, None, None, None).unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));
    }

    #[test]
    fn test_rejects_an_empty_matrix() {
        let err = voom::<f64>(&[], 0, 0, &[1.0], 1, None, None, None).unwrap_err();
        assert!(matches!(err, EdgeErrors::EmptyCounts { .. }));
    }

    #[test]
    fn test_rejects_a_count_length_mismatch() {
        let err = voom(&[1.0, 2.0, 3.0], 4, 6, &design6(), 2, None, None, None).unwrap_err();
        assert!(matches!(
            err,
            EdgeErrors::LengthMismatch { name: "counts", .. }
        ));
    }

    #[test]
    fn test_rejects_negative_counts() {
        let mut counts = counts4();
        counts[3] = -1.0;
        let err = voom(&counts, 4, 6, &design6(), 2, None, None, None).unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));
    }

    #[test]
    fn test_rejects_a_design_shape_mismatch() {
        let err = voom(&counts4(), 4, 6, &[1.0, 1.0], 2, None, None, None).unwrap_err();
        assert!(matches!(
            err,
            EdgeErrors::LengthMismatch { name: "design", .. }
        ));
    }

    #[test]
    fn test_rejects_a_rank_deficient_design() {
        // Third column is the sum of the first two.
        let design = vec![
            1.0, 0.0, 1.0, //
            1.0, 0.0, 1.0, //
            1.0, 0.0, 1.0, //
            1.0, 1.0, 2.0, //
            1.0, 1.0, 2.0, //
            1.0, 1.0, 2.0,
        ];
        let err = voom(&counts4(), 4, 6, &design, 3, None, None, None).unwrap_err();
        assert!(matches!(err, EdgeErrors::DesignNotFullRank { .. }));
    }

    #[test]
    fn test_rejects_a_library_size_length_mismatch() {
        let lib = [1.0, 2.0];
        let err = voom(&counts4(), 4, 6, &design6(), 2, Some(&lib), None, None).unwrap_err();
        assert!(matches!(
            err,
            EdgeErrors::LengthMismatch {
                name: "lib_size",
                ..
            }
        ));
    }

    #[test]
    fn test_rejects_a_non_positive_library_size() {
        let lib = [100.0, 100.0, 100.0, 100.0, 100.0, 0.0];
        let err = voom(&counts4(), 4, 6, &design6(), 2, Some(&lib), None, None).unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));
    }

    #[test]
    fn test_rejects_a_non_positive_prior_count() {
        let params = VoomParams {
            prior_count: 0.0,
            ..VoomParams::default()
        };
        let err = voom(&counts4(), 4, 6, &design6(), 2, None, None, Some(params)).unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));
    }

    #[test]
    fn test_rejects_a_span_outside_the_unit_interval() {
        let params = VoomParams {
            span: 1.5,
            adaptive_span: false,
            ..VoomParams::default()
        };
        let err = voom(&counts4(), 4, 6, &design6(), 2, None, None, Some(params)).unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));
    }

    #[test]
    fn test_rejects_prior_weights_of_the_wrong_shape() {
        let bad = Recycled::by_sample(vec![1.0, 1.0]);
        let err = voom(&counts4(), 4, 6, &design6(), 2, None, Some(&bad), None).unwrap_err();
        assert!(matches!(err, EdgeErrors::LengthMismatch { .. }));
    }

    #[test]
    fn test_voom_basic_rejects_bad_norm_factors() {
        let nf = [1.0, 1.0];
        let err = voom_basic(&counts4(), 4, 6, &design6(), 2, None, Some(&nf), 0.5).unwrap_err();
        assert!(matches!(
            err,
            EdgeErrors::LengthMismatch {
                name: "norm_factors",
                ..
            }
        ));

        let nf = [1.0, 1.0, 1.0, 1.0, 1.0, -1.0];
        let err = voom_basic(&counts4(), 4, 6, &design6(), 2, None, Some(&nf), 0.5).unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));
    }

    #[test]
    fn test_voom_lmfit_rejects_the_same_inputs_as_voom() {
        let err = voom_lmfit(&counts4(), 4, 6, &[1.0, 1.0], 2, None, None, None).unwrap_err();
        assert!(matches!(
            err,
            EdgeErrors::LengthMismatch { name: "design", .. }
        ));
    }

    // -- internals --

    #[test]
    fn test_collapse_averages_tied_abscissae() {
        let t = collapse(&[1.0, 2.0, 2.0, 3.0], &[1.0, 2.0, 4.0, 5.0]).unwrap();
        assert_eq!(t.x, vec![1.0, 2.0, 3.0]);
        assert_eq!(t.y, vec![1.0, 3.0, 5.0]);
    }

    #[test]
    fn test_collapse_rejects_a_single_abscissa() {
        let err = collapse(&[1.0, 1.0], &[2.0, 4.0]).unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));
    }

    /// `Rscript -e 'cat(sprintf("%.17g", mean(c(0.1, 0.2, 0.3))), "\n")'` gives
    /// 0.20000000000000001, which the naive sum does not.
    #[test]
    fn test_corrected_mean_matches_r() {
        assert_eq!(corrected_mean(&[0.1, 0.2, 0.3]), 0.20000000000000001);
    }
}
