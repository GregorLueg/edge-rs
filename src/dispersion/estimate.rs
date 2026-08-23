//! `estimateDisp`: common, trended and tagwise dispersions.
//!
//! Everything runs off one object: the adjusted profile likelihood of every gene
//! evaluated on a shared grid of dispersions, from [`crate::dispersion::apl`].
//! The three estimates are three different ways of reading that grid.
//!
//! * The **common** dispersion maximises the likelihood summed over all genes.
//! * The **trended** dispersion smooths the per-gene curves against abundance,
//!   then maximises each smoothed curve.
//! * The **tagwise** dispersion adds `prior_n` times the smoothed curve to each
//!   gene's own curve before maximising, which is the weighted likelihood
//!   empirical Bayes step. A large `prior_n` pulls a gene onto the trend, a
//!   small one lets it speak for itself.
//!
//! The grid is uniform in `log2(dispersion / 0.1)`, and the maximisation happens
//! in that coordinate, not in the dispersion itself. That is what makes a
//! twenty-one point grid enough to cover four orders of magnitude.
//!
//! ### References
//!
//! Chen, Lun and Smyth, F1000Research 5:1438, 2016
//! McCarthy, Chen and Smyth, Nucleic Acids Research 40(10), 2012

use crate::dispersion::apl::apl_grid;
use crate::glm::fit::glm_fit;
use crate::limma::smoothing::{locfit_by_col, loess_by_col};
use crate::limma::squeeze_var::{SqueezeVarParams, squeeze_var};
use crate::numeric::dist::beta_cdf;
use crate::numeric::interpolate::{maximize_interpolant, maximize_interpolant_many};
use crate::numeric::stats::moving_average_by_col;
use crate::prelude::*;
use crate::utils::design::{LIMMA_LOWESS_DEFAULTS, choose_lowess_span, matrix_rank};

////////////
// Consts //
////////////

/// Reference dispersion the grid is centred on.
///
/// The grid runs over `0.1 * 2^t`, so `t = 0` is a dispersion of 0.1, a typical
/// value for a well-behaved bulk RNA-seq experiment. edgeR's choice.
const GRID_CENTRE: f64 = 0.1;

/// Prior sample size above which shrinkage is treated as total.
///
/// Past this the tagwise estimate is indistinguishable from the trend, and the
/// weighted likelihood is dominated by the prior to the point where the
/// arithmetic stops being meaningful. edgeR clamps here.
const MAX_PRIOR_N: f64 = 1e6;

/// Count below which an observation is treated as a structural zero when
/// adjusting the residual degrees of freedom.
const ZERO_TOLERANCE: f64 = 1e-4;

/// Prior count used when the trend covariate is recomputed internally.
///
/// `aveLogCPM`'s own default, which is what `estimateDisp.default` calls it
/// with.
const AVE_LOG_CPM_PRIOR_COUNT: f64 = 2.0;

/////////////////
// TrendMethod //
/////////////////

/// How the abundance trend is smoothed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrendMethod {
    /// No trend: every gene shares the same prior curve.
    None,
    /// Moving average over genes ordered by abundance.
    MovingAverage,
    /// Local linear regression.
    Loess,
    /// Local constant regression, edgeR's default.
    Locfit,
    /// A beta-weighted blend of local constant and local linear, which behaves
    /// better at the ends of the abundance range.
    LocfitMixed,
}

/// Parses a trend method name.
///
/// ### Params
///
/// * `s` - Name, matching edgeR's spelling
///
/// ### Returns
///
/// The method, or `None` if unrecognised.
pub fn parse_trend_method(s: &str) -> Option<TrendMethod> {
    match s {
        "none" => Some(TrendMethod::None),
        "movingave" => Some(TrendMethod::MovingAverage),
        "loess" => Some(TrendMethod::Loess),
        "locfit" => Some(TrendMethod::Locfit),
        "locfit.mixed" => Some(TrendMethod::LocfitMixed),
        _ => None,
    }
}

////////////////
// WlebParams //
////////////////

/// Which parts of the weighted likelihood fit are wanted.
#[derive(Clone, Debug)]
pub struct WlebParams {
    /// Prior sample size, one value or one per gene.
    pub prior_n: Vec<f64>,
    /// Smoothing span. `None` derives it from the gene count.
    pub span: Option<f64>,
    /// How to smooth the trend.
    pub trend_method: TrendMethod,
    /// Compute the single overall maximiser.
    pub overall: bool,
    /// Compute the per-gene trended maximiser.
    pub trend: bool,
    /// Compute the per-gene shrunk maximiser.
    pub individual: bool,
}

impl Default for WlebParams {
    /// edgeR's defaults: a prior sample size of 5, a locfit trend, everything on.
    fn default() -> Self {
        Self {
            prior_n: vec![5.0],
            span: None,
            trend_method: TrendMethod::Locfit,
            overall: true,
            trend: true,
            individual: true,
        }
    }
}

////////////////
// WlebResult //
////////////////

/// Output of [`wleb`].
#[derive(Clone, Debug)]
pub struct WlebResult {
    /// Maximiser of the summed likelihood, in grid coordinates.
    pub overall: Option<f64>,
    /// Per-gene maximiser of the smoothed likelihood, in grid coordinates.
    pub trend: Option<Vec<f64>>,
    /// Per-gene maximiser of the shrunk likelihood, in grid coordinates.
    pub individual: Option<Vec<f64>>,
    /// Span actually used.
    pub span: f64,
    /// The smoothed likelihood surface, row-major `n_genes * n_grid`.
    pub shared_loglik: Vec<f64>,
}

/////////////
// Helpers //
/////////////

/// Default smoothing span for a given number of genes.
///
/// Wider for small experiments, since there is less to average over. This is
/// `chooseLowessSpan(ntags)` at limma's defaults, which is what edgeR's `WLEB`
/// reaches for when `span` is `NULL`.
///
/// edgeR keeps a second rule behind `legacy.span = TRUE`, namely
/// `chooseLowessSpan(ntags, small.n = 50, min.span = 0.25, power = 0.5)`. That
/// was the default before edgeR 4.0 and is not the default now. The two agree
/// only below fifty genes, where both return 1, so a small fixture cannot tell
/// them apart: at 890 genes they are 0.568 and 0.428, and the gap lands on every
/// trended and tagwise dispersion. Pass an explicit `span` to get the old rule.
///
/// ### Params
///
/// * `n_genes` - Number of genes entering the fit
///
/// ### Returns
///
/// A span in `(0, 1]`.
fn default_span(n_genes: usize) -> f64 {
    let (small_n, min_span, power) = LIMMA_LOWESS_DEFAULTS;
    choose_lowess_span(n_genes, small_n, min_span, power)
}

/// Smooths the likelihood surface down the gene axis.
///
/// Each grid column is smoothed independently against the abundance covariate.
///
/// ### Params
///
/// * `loglik` - Row-major `n_genes * n_grid` surface
/// * `n_genes` - Number of genes
/// * `n_grid` - Number of grid points
/// * `covariate` - Abundance per gene
/// * `method` - How to smooth
/// * `span` - Smoothing span
///
/// ### Returns
///
/// The smoothed surface, same shape as the input.
fn smooth_surface(
    loglik: &[f64],
    n_genes: usize,
    n_grid: usize,
    covariate: Option<&[f64]>,
    method: TrendMethod,
    span: f64,
) -> Result<Vec<f64>, EdgeErrors> {
    match method {
        TrendMethod::None => {
            // Every gene shares the column means.
            let mut mean = vec![0.0; n_grid];
            for row in loglik.chunks_exact(n_grid) {
                for (acc, v) in mean.iter_mut().zip(row.iter()) {
                    *acc += v;
                }
            }
            for v in mean.iter_mut() {
                *v /= n_genes as f64;
            }
            Ok(mean.repeat(n_genes))
        }
        TrendMethod::MovingAverage => {
            let covariate = covariate.expect("a trend method implies a covariate");
            // Order genes by abundance, smooth, then put them back.
            let mut order: Vec<usize> = (0..n_genes).collect();
            order.sort_by(|&a, &b| covariate[a].total_cmp(&covariate[b]));

            let mut sorted = vec![0.0; n_genes * n_grid];
            for (position, &gene) in order.iter().enumerate() {
                sorted[position * n_grid..(position + 1) * n_grid]
                    .copy_from_slice(&loglik[gene * n_grid..(gene + 1) * n_grid]);
            }

            let width = ((span * n_genes as f64).floor() as usize).max(1);
            let smoothed = moving_average_by_col(&sorted, n_genes, n_grid, width, true)?;

            let mut out = vec![0.0; n_genes * n_grid];
            for (position, &gene) in order.iter().enumerate() {
                out[gene * n_grid..(gene + 1) * n_grid]
                    .copy_from_slice(&smoothed[position * n_grid..(position + 1) * n_grid]);
            }
            Ok(out)
        }
        TrendMethod::Loess => {
            let covariate = covariate.expect("a trend method implies a covariate");
            Ok(loess_by_col(loglik, n_genes, n_grid, Some(covariate), span)?.fitted)
        }
        TrendMethod::Locfit => {
            let covariate = covariate.expect("a trend method implies a covariate");
            locfit_by_col(loglik, n_genes, n_grid, Some(covariate), None, span, 0)
        }
        TrendMethod::LocfitMixed => {
            let covariate = covariate.expect("a trend method implies a covariate");
            let degree0 = locfit_by_col(loglik, n_genes, n_grid, Some(covariate), None, span, 0)?;
            let degree1 = locfit_by_col(loglik, n_genes, n_grid, Some(covariate), None, span, 1)?;

            // Blend by a beta(2, 2) weight on the abundance range, so the local
            // constant fit dominates at the ends where the linear one is least
            // stable.
            let lo = covariate.iter().cloned().fold(f64::INFINITY, f64::min);
            let hi = covariate.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            let mut out = vec![0.0; n_genes * n_grid];
            for (gene, &abundance) in covariate.iter().enumerate().take(n_genes) {
                let w = if hi > lo {
                    beta_cdf((abundance - lo) / (hi - lo), 2.0, 2.0)?
                } else {
                    0.5
                };
                let base = gene * n_grid;
                for k in 0..n_grid {
                    out[base + k] = w * degree0[base + k] + (1.0 - w) * degree1[base + k];
                }
            }
            Ok(out)
        }
    }
}

//////////
// WLEP //
//////////

/// Weighted likelihood empirical Bayes over a likelihood grid.
///
/// ### Params
///
/// * `theta` - Grid coordinates, one per column of `loglik`
/// * `loglik` - Log-likelihood surface, row-major `n_genes * theta.len()`
/// * `n_genes` - Number of genes
/// * `covariate` - Abundance per gene, required for any trend method but
///   [`TrendMethod::None`]
/// * `m0` - Precomputed smoothed surface. Supplying it skips the smoothing,
///   which is how `estimate_disp` avoids smoothing twice.
/// * `params` - What to compute and how, or [`WlebParams::default`]
///
/// ### Returns
///
/// The requested maximisers in grid coordinates, together with the smoothed
/// surface, or [`EdgeErrors`] if the shapes disagree.
pub fn wleb(
    theta: &[f64],
    loglik: &[f64],
    n_genes: usize,
    covariate: Option<&[f64]>,
    m0: Option<&[f64]>,
    params: Option<WlebParams>,
) -> Result<WlebResult, EdgeErrors> {
    let params = params.unwrap_or_default();
    let n_grid = theta.len();

    if n_genes == 0 || n_grid == 0 {
        return Err(EdgeErrors::MustBePositive("loglik dimensions".to_string()));
    }
    if loglik.len() != n_genes * n_grid {
        return Err(EdgeErrors::LengthMismatch {
            name: "loglik",
            expected: n_genes * n_grid,
            got: loglik.len(),
        });
    }
    if let Some(c) = covariate
        && c.len() != n_genes
    {
        return Err(EdgeErrors::LengthMismatch {
            name: "covariate",
            expected: n_genes,
            got: c.len(),
        });
    }
    if params.prior_n.len() != 1 && params.prior_n.len() != n_genes {
        return Err(EdgeErrors::LengthMismatch {
            name: "prior_n",
            expected: n_genes,
            got: params.prior_n.len(),
        });
    }

    // A trend needs something to trend against.
    let trend_method = match covariate {
        None => TrendMethod::None,
        Some(_) => params.trend_method,
    };
    let span = params.span.unwrap_or_else(|| default_span(n_genes));

    let overall = if params.overall {
        let mut summed = vec![0.0; n_grid];
        for row in loglik.chunks_exact(n_grid) {
            for (acc, v) in summed.iter_mut().zip(row.iter()) {
                *acc += v;
            }
        }
        Some(maximize_interpolant(theta, &summed)?)
    } else {
        None
    };

    let shared = match m0 {
        Some(precomputed) => {
            if precomputed.len() != n_genes * n_grid {
                return Err(EdgeErrors::LengthMismatch {
                    name: "m0",
                    expected: n_genes * n_grid,
                    got: precomputed.len(),
                });
            }
            precomputed.to_vec()
        }
        None => smooth_surface(loglik, n_genes, n_grid, covariate, trend_method, span)?,
    };

    let trend = if params.trend {
        Some(maximize_interpolant_many(theta, &shared, n_genes)?)
    } else {
        None
    };

    let individual = if params.individual {
        let mut shrunk = vec![0.0; n_genes * n_grid];
        for gene in 0..n_genes {
            let weight = if params.prior_n.len() == 1 {
                params.prior_n[0]
            } else {
                params.prior_n[gene]
            };
            let base = gene * n_grid;
            for k in 0..n_grid {
                shrunk[base + k] = loglik[base + k] + weight * shared[base + k];
            }
        }
        Some(maximize_interpolant_many(theta, &shrunk, n_genes)?)
    } else {
        None
    };

    Ok(WlebResult {
        overall,
        trend,
        individual,
        span,
        shared_loglik: shared,
    })
}

////////////////////////
// EstimateDispParams //
////////////////////////

/// Tuning knobs for [`estimate_disp`].
#[derive(Clone, Debug)]
pub struct EstimateDispParams {
    /// How to smooth the abundance trend.
    pub trend_method: TrendMethod,
    /// Whether to compute per-gene dispersions.
    pub tagwise: bool,
    /// Smoothing span, or `None` to derive it.
    pub span: Option<f64>,
    /// Genes whose total count falls below this are excluded from the fit.
    pub min_row_sum: f64,
    /// Number of grid points.
    pub grid_length: usize,
    /// Grid range in `log2(dispersion / 0.1)`.
    pub grid_range: (f64, f64),
    /// Prior degrees of freedom. `None` derives it from the residual variances
    /// with [`crate::limma::squeeze_var`], which is what edgeR does.
    pub prior_df: Option<f64>,
    /// Use the robust empirical Bayes fit when deriving the prior degrees of
    /// freedom, which gives outlier genes their own, smaller prior.
    pub robust: bool,
    /// Winsorising tail proportions for the robust fit.
    pub winsor_tail_p: (f64, f64),
}

impl Default for EstimateDispParams {
    /// edgeR's defaults.
    fn default() -> Self {
        Self {
            trend_method: TrendMethod::Locfit,
            tagwise: true,
            span: None,
            min_row_sum: 5.0,
            grid_length: 21,
            grid_range: (-10.0, 10.0),
            prior_df: None,
            robust: false,
            winsor_tail_p: (0.05, 0.1),
        }
    }
}

/////////////////////////
// DispersionEstimates //
/////////////////////////

/// The three dispersion estimates.
#[derive(Clone, Debug)]
pub struct DispersionEstimates {
    /// One dispersion shared by every gene.
    pub common: f64,
    /// Abundance-trended dispersion, one per gene, when a trend was fitted.
    pub trended: Option<Vec<f64>>,
    /// Shrunk per-gene dispersion, when requested.
    pub tagwise: Option<Vec<f64>>,
    /// Span used for the trend.
    pub span: f64,
    /// Prior degrees of freedom used for the shrinkage. One value, or one per
    /// gene when the robust fit was used.
    pub prior_df: Option<Vec<f64>>,
}

/////////////
// Helpers //
/////////////

/// Derives the prior degrees of freedom from the residual variances.
///
/// Fits the GLM at the working dispersion, turns each gene's residual deviance
/// into a variance on its residual degrees of freedom, then hands those to
/// limma's empirical Bayes fit. This is what edgeR's `estimateDisp` does when
/// the caller does not supply `prior.df`.
///
/// ### Params
///
/// * `counts` - Counts of the retained genes, row-major
/// * `n_genes` - Number of retained genes
/// * `n_samples` - Number of samples
/// * `design` - Row-major design
/// * `n_coef` - Number of coefficients
/// * `dispersion` - Working dispersion, the trend where there is one
/// * `offset` - Log-scale offsets
/// * `weights` - Optional observation weights
/// * `covariate` - Abundance per retained gene, for the trended prior
/// * `robust` - Use the robust empirical Bayes fit
/// * `winsor_tail_p` - Winsorising tails for the robust fit
///
/// ### Returns
///
/// The prior degrees of freedom, one value or one per gene when robust.
#[allow(clippy::too_many_arguments)]
fn derive_prior_df<T: EdgeFloat>(
    counts: &[T],
    n_genes: usize,
    n_samples: usize,
    design: &[f64],
    n_coef: usize,
    dispersion: &Recycled<f64>,
    offset: &Recycled<f64>,
    weights: Option<&Recycled<f64>>,
    covariate: Option<&[f64]>,
    robust: bool,
    winsor_tail_p: (f64, f64),
) -> Result<Vec<f64>, EdgeErrors> {
    let fit = glm_fit(
        counts, n_genes, n_samples, design, n_coef, dispersion, offset, weights, 0.0,
    )?;

    let df = residual_df(counts, &fit.fitted, n_genes, n_samples, design, n_coef)?;

    // A gene with no residual degrees of freedom carries no variance
    // information, and limma drops it from the fit rather than dividing by zero.
    let variance: Vec<f64> = fit
        .deviance
        .iter()
        .zip(df.iter())
        .map(|(dev, d)| if *d > 0.0 { (dev / d).max(0.0) } else { 0.0 })
        .collect();

    let squeezed = squeeze_var(
        &variance,
        &df,
        covariate,
        Some(SqueezeVarParams {
            robust,
            winsor_tail_p,
            span: None,
            legacy: None,
        }),
    )?;

    Ok(squeezed.df_prior)
}

/// Scatters per-gene values from the fitted subset back over all genes.
///
/// Genes that were filtered out take the value belonging to the least abundant
/// retained gene, as edgeR does, rather than the common dispersion: a gene too
/// sparse to fit is more like the low-abundance end of the trend than like the
/// average gene.
///
/// ### Params
///
/// * `values` - One value per retained gene
/// * `kept` - Indices of the retained genes
/// * `n_genes` - Total number of genes
/// * `fallback` - Value used when there is no covariate to order by
/// * `covariate` - Abundance of the retained genes
///
/// ### Returns
///
/// One value per gene.
fn expand_to_all_genes(
    values: &[f64],
    kept: &[usize],
    n_genes: usize,
    fallback: f64,
    covariate: Option<&[f64]>,
) -> Vec<f64> {
    let filler = match covariate {
        Some(c) => {
            let least = c
                .iter()
                .enumerate()
                .fold((0usize, f64::INFINITY), |(bi, bv), (i, &v)| {
                    if v < bv { (i, v) } else { (bi, bv) }
                })
                .0;
            values[least]
        }
        None => fallback,
    };

    let mut out = vec![filler; n_genes];
    for (slot, &gene) in kept.iter().enumerate() {
        out[gene] = values[slot];
    }
    out
}


/// Residual degrees of freedom per gene, reduced where the fit hit exact zeros.
///
/// A gene with structural zeros has fewer effective observations than samples,
/// and using the nominal residual degrees of freedom would inflate the
/// quasi-likelihood dispersion. edgeR recomputes the rank of the design
/// restricted to the non-zero samples.
///
/// ### Params
///
/// * `counts` - Counts, row-major `n_genes * n_samples`
/// * `fitted` - Fitted means, row-major `n_genes * n_samples`
/// * `n_genes` - Number of genes
/// * `n_samples` - Number of samples
/// * `design` - Row-major design
/// * `n_coef` - Number of coefficients
///
/// ### Returns
///
/// One residual degrees of freedom per gene.
pub fn residual_df<T: EdgeFloat>(
    counts: &[T],
    fitted: &[f64],
    n_genes: usize,
    n_samples: usize,
    design: &[f64],
    n_coef: usize,
) -> Result<Vec<f64>, EdgeErrors> {
    let base = (n_samples - n_coef) as f64;
    let mut out = vec![base; n_genes];

    for gene in 0..n_genes {
        let y = &counts[gene * n_samples..(gene + 1) * n_samples];
        let mu = &fitted[gene * n_samples..(gene + 1) * n_samples];

        let zero: Vec<bool> = y
            .iter()
            .zip(mu.iter())
            .map(|(v, m)| v.to_f64().unwrap_or(0.0) < ZERO_TOLERANCE && *m < ZERO_TOLERANCE)
            .collect();
        let n_zero = zero.iter().filter(|z| **z).count();

        if n_zero == 0 {
            continue;
        }
        if n_zero >= n_samples - 1 {
            out[gene] = 0.0;
            continue;
        }

        let mut reduced = Vec::with_capacity((n_samples - n_zero) * n_coef);
        for (sample, is_zero) in zero.iter().enumerate() {
            if !is_zero {
                reduced.extend_from_slice(&design[sample * n_coef..(sample + 1) * n_coef]);
            }
        }
        let rows = n_samples - n_zero;
        let rank = matrix_rank(&reduced, rows, n_coef)?;
        out[gene] = (rows - rank) as f64;
    }

    Ok(out)
}

///////////////
// Front end //
///////////////

/// Estimates common, trended and tagwise dispersions.
///
/// ### Params
///
/// * `counts` - Counts, row-major `n_genes * n_samples`
/// * `n_genes` - Number of genes
/// * `n_samples` - Number of samples
/// * `design` - Design matrix, row-major `n_samples * n_coef`
/// * `n_coef` - Number of coefficients
/// * `offset` - Log-scale offsets
/// * `weights` - Optional observation weights
/// * `ave_log_cpm` - Average log2 counts per million per gene, the trend
///   covariate. `None` recomputes it internally once the common dispersion is
///   known, which is what `estimateDisp.default` does; anything supplied here
///   overrides that. The recomputed covariate ignores `weights`, which
///   [`crate::core::expression::ave_log_cpm`] does not take.
/// * `prior_df` - Prior degrees of freedom for the tagwise shrinkage. Supply
///   `None` in `params` to let the caller pass it here instead.
/// * `params` - Tuning knobs, or [`EstimateDispParams::default`]
///
/// ### Returns
///
/// The estimates, or [`EdgeErrors`] if the shapes disagree or there are no
/// residual degrees of freedom.
#[allow(clippy::too_many_arguments)]
pub fn estimate_disp<T: EdgeFloat>(
    counts: &[T],
    n_genes: usize,
    n_samples: usize,
    design: &[f64],
    n_coef: usize,
    offset: &Recycled<f64>,
    weights: Option<&Recycled<f64>>,
    ave_log_cpm: Option<&[f64]>,
    params: Option<EstimateDispParams>,
) -> Result<DispersionEstimates, EdgeErrors> {
    let params = params.unwrap_or_default();

    if n_genes == 0 || n_samples == 0 {
        return Err(EdgeErrors::EmptyCounts { n_genes, n_samples });
    }
    if counts.len() != n_genes * n_samples {
        return Err(EdgeErrors::LengthMismatch {
            name: "counts",
            expected: n_genes * n_samples,
            got: counts.len(),
        });
    }
    if n_coef >= n_samples {
        return Err(EdgeErrors::InvalidArgument(format!(
            "no residual degrees of freedom: {n_coef} coefficients for {n_samples} samples"
        )));
    }
    if params.grid_length < 2 {
        return Err(EdgeErrors::MustBePositive("grid_length".to_string()));
    }
    if let Some(a) = ave_log_cpm
        && a.len() != n_genes
    {
        return Err(EdgeErrors::LengthMismatch {
            name: "ave_log_cpm",
            expected: n_genes,
            got: a.len(),
        });
    }

    // Genes with too few counts carry no information about the dispersion and
    // would only add noise to the trend.
    let kept = crate::dispersion::filter_by_min_row_sum(
        counts,
        n_genes,
        n_samples,
        params.min_row_sum,
    )?;

    let mut selected = Vec::with_capacity(kept.len() * n_samples);
    for &gene in &kept {
        selected.extend_from_slice(&counts[gene * n_samples..(gene + 1) * n_samples]);
    }
    let selected_offset = offset.subset(&kept, n_samples);
    let selected_weights = weights.map(|w| w.subset(&kept, n_samples));

    // The grid is uniform in log2(dispersion / 0.1).
    let n_grid = params.grid_length;
    let (lo, hi) = params.grid_range;
    let spline_pts: Vec<f64> = (0..n_grid)
        .map(|i| lo + (hi - lo) * i as f64 / (n_grid - 1) as f64)
        .collect();
    let spline_disp: Vec<f64> = spline_pts
        .iter()
        .map(|t| GRID_CENTRE * 2.0_f64.powf(*t))
        .collect();

    let loglik = apl_grid(
        &selected,
        kept.len(),
        n_samples,
        design,
        n_coef,
        &spline_disp,
        &selected_offset,
        selected_weights.as_ref(),
    )?;

    // Common dispersion: maximise the summed curve.
    let mut summed = vec![0.0; n_grid];
    for row in loglik.chunks_exact(n_grid) {
        for (acc, v) in summed.iter_mut().zip(row.iter()) {
            *acc += v;
        }
    }
    let common = GRID_CENTRE * 2.0_f64.powf(maximize_interpolant(&spline_pts, &summed)?);

    // edgeR recomputes the covariate here, at the dispersion it has just
    // estimated, and over the full matrix before subsetting to the kept genes.
    let recomputed: Option<Vec<f64>> = match (ave_log_cpm, params.trend_method) {
        (None, TrendMethod::None) => None,
        (None, _) => Some(crate::core::expression::ave_log_cpm(
            counts,
            n_genes,
            n_samples,
            None,
            Some(offset),
            AVE_LOG_CPM_PRIOR_COUNT,
            Some(&Recycled::scalar(common)),
        )?),
        (Some(_), _) => None,
    };
    let covariate: Option<Vec<f64>> = ave_log_cpm
        .or(recomputed.as_deref())
        .map(|a| kept.iter().map(|&g| a[g]).collect());

    // Trended dispersion.
    let trend_fit = wleb(
        &spline_pts,
        &loglik,
        kept.len(),
        covariate.as_deref(),
        None,
        Some(WlebParams {
            prior_n: vec![0.0],
            span: params.span,
            trend_method: params.trend_method,
            overall: false,
            trend: params.trend_method != TrendMethod::None,
            individual: false,
        }),
    )?;
    let span = trend_fit.span;

    let trended = trend_fit.trend.as_ref().map(|t| {
        let fitted: Vec<f64> = t.iter().map(|v| GRID_CENTRE * 2.0_f64.powf(*v)).collect();
        expand_to_all_genes(&fitted, &kept, n_genes, common, covariate.as_deref())
    });

    if !params.tagwise {
        return Ok(DispersionEstimates {
            common,
            trended,
            tagwise: None,
            span,
            prior_df: None,
        });
    }

    // The dispersion the residual variances are measured against: the trend
    // where there is one, the common dispersion otherwise.
    let fit_dispersion = match trended.as_ref() {
        Some(t) => Recycled::by_gene(kept.iter().map(|&g| t[g]).collect()),
        None => Recycled::scalar(common),
    };

    let prior_df: Vec<f64> = match params.prior_df {
        Some(v) => vec![v],
        None => derive_prior_df(
            &selected,
            kept.len(),
            n_samples,
            design,
            n_coef,
            &fit_dispersion,
            &selected_offset,
            selected_weights.as_ref(),
            covariate.as_deref(),
            params.robust,
            params.winsor_tail_p,
        )?,
    };

    let residual = (n_samples - n_coef) as f64;
    let prior_n: Vec<f64> = prior_df
        .iter()
        .map(|d| (d / residual).min(MAX_PRIOR_N))
        .collect();

    let shrunk = wleb(
        &spline_pts,
        &loglik,
        kept.len(),
        covariate.as_deref(),
        Some(&trend_fit.shared_loglik),
        Some(WlebParams {
            prior_n,
            span: Some(span),
            trend_method: params.trend_method,
            overall: false,
            trend: false,
            individual: true,
        }),
    )?;

    let per_gene: Vec<f64> = shrunk
        .individual
        .as_ref()
        .expect("individual was requested")
        .iter()
        .map(|v| GRID_CENTRE * 2.0_f64.powf(*v))
        .collect();
    let tagwise = Some(expand_to_all_genes(
        &per_gene,
        &kept,
        n_genes,
        common,
        covariate.as_deref(),
    ));

    Ok(DispersionEstimates {
        common,
        trended,
        tagwise,
        span,
        prior_df: Some(prior_df),
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// The 30 gene by 6 sample experiment used for the edgeR parity tests.
    ///
    /// ```r
    /// set.seed(11); n <- 30
    /// y <- matrix(rnbinom(n*6, mu = rep(c(20,200,2000), length.out = n),
    ///                     size = 1/0.15), nrow = n)
    /// X <- cbind(1, c(0,0,0,1,1,1))
    /// d <- DGEList(counts = y); d$samples$lib.size <- colSums(y)
    /// d <- estimateDisp(d, design = X, robust = FALSE)
    /// ```
    ///
    /// Returns the counts, the design, edgeR's own `AveLogCPM` (passed in rather
    /// than recomputed, so these tests isolate `estimate_disp` from
    /// `ave_log_cpm`), the library sizes, and edgeR's trended dispersions.
    #[allow(clippy::type_complexity)]
    fn edger_fixture() -> (Vec<f64>, Vec<f64>, Vec<f64>, [f64; 6], Vec<f64>) {
        #[rustfmt::skip]
        let counts: Vec<f64> = vec![
            6.0, 17.0, 25.0, 31.0, 32.0, 29.0,
            85.0, 171.0, 224.0, 165.0, 203.0, 224.0,
            3001.0, 1204.0, 4629.0, 743.0, 3125.0, 3197.0,
            17.0, 11.0, 10.0, 29.0, 14.0, 19.0,
            215.0, 248.0, 90.0, 180.0, 159.0, 166.0,
            1653.0, 1144.0, 1616.0, 2442.0, 1368.0, 851.0,
            18.0, 20.0, 24.0, 35.0, 22.0, 12.0,
            218.0, 430.0, 266.0, 109.0, 211.0, 361.0,
            572.0, 2028.0, 1169.0, 1644.0, 1239.0, 3118.0,
            16.0, 17.0, 20.0, 20.0, 10.0, 32.0,
            261.0, 182.0, 280.0, 307.0, 317.0, 286.0,
            2402.0, 3116.0, 1738.0, 1485.0, 1914.0, 1334.0,
            9.0, 41.0, 37.0, 25.0, 19.0, 23.0,
            191.0, 290.0, 212.0, 180.0, 200.0, 90.0,
            1393.0, 2356.0, 870.0, 1128.0, 4431.0, 1424.0,
            22.0, 15.0, 19.0, 31.0, 23.0, 9.0,
            117.0, 199.0, 192.0, 147.0, 182.0, 118.0,
            562.0, 2537.0, 3276.0, 1402.0, 1707.0, 1739.0,
            23.0, 15.0, 23.0, 18.0, 25.0, 36.0,
            160.0, 249.0, 305.0, 185.0, 166.0, 112.0,
            1994.0, 2966.0, 1322.0, 1794.0, 1723.0, 493.0,
            15.0, 24.0, 12.0, 16.0, 33.0, 16.0,
            177.0, 188.0, 110.0, 187.0, 58.0, 244.0,
            1140.0, 1112.0, 1565.0, 2895.0, 994.0, 2716.0,
            23.0, 22.0, 45.0, 32.0, 11.0, 24.0,
            189.0, 162.0, 133.0, 267.0, 187.0, 82.0,
            2401.0, 2235.0, 1277.0, 3047.0, 814.0, 1078.0,
            21.0, 24.0, 19.0, 24.0, 20.0, 16.0,
            184.0, 174.0, 114.0, 193.0, 114.0, 321.0,
            1937.0, 1573.0, 2213.0, 2776.0, 2221.0, 2023.0,
        ];
        #[rustfmt::skip]
        let ave_log_cpm: Vec<f64> = vec![
            10.217_844_608_314_6, 13.05275618345479, 16.95244230819978,
            9.79438372378658, 13.04824894232326, 16.13124517429634,
            10.13290346589442, 13.62525863196988, 16.23074973173857,
            9.97485753416446, 13.67017155953262, 16.52813179718733,
            10.33323866151903, 13.16620578297977, 16.47253747739089,
            10.01353802321997, 12.88459140535754, 16.40860240277475,
            10.23898261485237, 13.181_178_624_459_1, 16.29912570790799,
            9.97329965658649, 12.92303330440534, 16.33493178423032,
            10.37858458796799, 12.99071066727291, 16.391_968_461_271_3,
            10.06545265527433, 13.11475433624568, 16.62134884972848,
        ];
        #[rustfmt::skip]
        let trended: Vec<f64> = vec![
            0.117863286574298, 0.131881281350169, 0.197225247650793,
            0.117163355951516, 0.131851147030253, 0.192427732557431,
            0.117724250619365, 0.141800030636821, 0.193008210455128,
            0.117463746552762, 0.143178219824444, 0.194744286736151,
            0.118051096089876, 0.132629902478078, 0.194419580856211,
            0.117527721682232, 0.130735839190030, 0.194046250715124,
            0.117897781868701, 0.132727310885643, 0.193407208206644,
            0.117461166931415, 0.131001581671346, 0.193616188259492,
            0.118124563420063, 0.131463760157284, 0.193949137162521,
            0.117613361805908, 0.132292721492398, 0.195288906118998,
        ];
        let design = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let libraries: [f64; 6] = [19022.0, 22770.0, 21835.0, 21537.0, 21542.0, 20193.0];
        (counts, design, ave_log_cpm, libraries, trended)
    }

    /// End-to-end parity with edgeR 4.8.2 on a 30 gene by 6 sample experiment.
    ///
    /// ```r
    /// set.seed(11); n <- 30
    /// y <- matrix(rnbinom(n*6, mu = rep(c(20,200,2000), length.out = n),
    ///                     size = 1/0.15), nrow = n)
    /// X <- cbind(1, c(0,0,0,1,1,1))
    /// d <- DGEList(counts = y); d$samples$lib.size <- colSums(y)
    /// d <- estimateDisp(d, design = X, robust = FALSE)
    /// ```
    ///
    /// edgeR returns `prior.df = Inf` here, so its tagwise dispersions are
    /// exactly its trended ones. That is a real case rather than a degenerate
    /// one: it happens whenever the residual variances are near constant, and it
    /// exercises the whole trend path while pinning the shrinkage limit.
    ///
    /// The covariate is edgeR's own `AveLogCPM`, passed in rather than
    /// recomputed, so this test isolates `estimate_disp` from `ave_log_cpm`.
    #[test]
    fn test_matches_edger_estimate_disp() {
        let (counts, design, ave_log_cpm, libraries, expected_trended) = edger_fixture();
        let offset = Recycled::by_sample(libraries.iter().map(|v| v.ln()).collect());

        let out = estimate_disp(
            &counts,
            30,
            6,
            &design,
            2,
            &offset,
            None,
            Some(&ave_log_cpm),
            Some(EstimateDispParams {
                // edgeR reports prior.df = Inf for this data, which the
                // implementation clamps to MAX_PRIOR_N.
                prior_df: Some(f64::INFINITY),
                ..Default::default()
            }),
        )
        .unwrap();

        assert_relative_eq!(out.common, 0.165_469_228_575_145, max_relative = 1e-6);
        assert_relative_eq!(out.span, 1.0, max_relative = 1e-12);

        let trended = out.trended.as_ref().unwrap();
        for (got, want) in trended.iter().zip(expected_trended.iter()) {
            assert_relative_eq!(got, want, max_relative = 1e-5);
        }

        // prior.df = Inf means total shrinkage onto the trend.
        let tagwise = out.tagwise.as_ref().unwrap();
        for (got, want) in tagwise.iter().zip(expected_trended.iter()) {
            assert_relative_eq!(got, want, max_relative = 1e-4);
        }
    }

    /// The same experiment again, but letting `estimate_disp` derive the prior
    /// degrees of freedom through `squeeze_var` rather than being handed it.
    ///
    /// edgeR reports `prior.df = Inf` for this data, so the derivation must land
    /// somewhere that produces total shrinkage, and the tagwise dispersions must
    /// come back on the trend.
    #[test]
    fn test_derives_prior_df_without_being_told() {
        let (counts, design, ave_log_cpm, libraries, expected_trended) = edger_fixture();
        let offset = Recycled::by_sample(libraries.iter().map(|v| v.ln()).collect());

        let out = estimate_disp(
            &counts,
            30,
            6,
            &design,
            2,
            &offset,
            None,
            Some(&ave_log_cpm),
            None,
        )
        .unwrap();

        let prior_df = out.prior_df.as_ref().unwrap();
        assert!(
            prior_df[0] > 1e6,
            "expected near-total shrinkage, got prior_df {}",
            prior_df[0]
        );

        let tagwise = out.tagwise.as_ref().unwrap();
        for (got, want) in tagwise.iter().zip(expected_trended.iter()) {
            assert_relative_eq!(got, want, max_relative = 1e-4);
        }
    }

    /// A 40 gene by 6 sample run from raw counts, with no covariate supplied.
    ///
    /// ```r
    /// set.seed(23); n <- 40
    /// y <- matrix(rnbinom(n*6, mu = rep(c(20,200,2000,50), length.out = n),
    ///                     size = 1/0.2), nrow = n)
    /// X <- cbind(1, c(0,0,0,1,1,1))
    /// d <- estimateDisp(DGEList(counts = y), design = X, robust = FALSE)
    /// ```
    ///
    /// This is the one test that does *not* hand in edgeR's own `AveLogCPM`, so
    /// it is what pins the two-pass structure: edgeR recomputes the covariate at
    /// the common dispersion before smoothing, and a covariate taken at any
    /// other dispersion moves the trend by around 2e-3 relative and the prior
    /// degrees of freedom by 7e-3.
    #[test]
    fn test_matches_edger_estimate_disp_from_raw_counts() {
        #[rustfmt::skip]
        let counts: Vec<f64> = vec![
            17.0, 25.0, 2.0, 26.0, 27.0, 13.0,
            176.0, 236.0, 33.0, 206.0, 318.0, 103.0,
            2847.0, 2053.0, 917.0, 1262.0, 917.0, 2824.0,
            32.0, 68.0, 34.0, 74.0, 55.0, 46.0,
            15.0, 31.0, 7.0, 20.0, 17.0, 26.0,
            285.0, 244.0, 142.0, 102.0, 195.0, 176.0,
            2480.0, 1518.0, 4296.0, 1742.0, 1211.0, 1925.0,
            45.0, 48.0, 65.0, 27.0, 24.0, 37.0,
            9.0, 9.0, 17.0, 40.0, 15.0, 11.0,
            234.0, 178.0, 133.0, 317.0, 171.0, 265.0,
            2470.0, 964.0, 1813.0, 3782.0, 3897.0, 2419.0,
            50.0, 54.0, 71.0, 30.0, 96.0, 66.0,
            17.0, 19.0, 20.0, 16.0, 24.0, 15.0,
            136.0, 255.0, 133.0, 290.0, 212.0, 73.0,
            4500.0, 1122.0, 3483.0, 1744.0, 3759.0, 311.0,
            40.0, 39.0, 77.0, 54.0, 75.0, 76.0,
            24.0, 67.0, 22.0, 31.0, 13.0, 25.0,
            327.0, 101.0, 209.0, 325.0, 99.0, 196.0,
            1040.0, 1448.0, 5378.0, 2157.0, 1217.0, 2719.0,
            46.0, 55.0, 78.0, 41.0, 32.0, 41.0,
            21.0, 11.0, 25.0, 23.0, 20.0, 35.0,
            298.0, 244.0, 87.0, 255.0, 127.0, 81.0,
            2509.0, 1644.0, 1600.0, 2614.0, 956.0, 1586.0,
            32.0, 27.0, 38.0, 92.0, 40.0, 46.0,
            20.0, 22.0, 11.0, 19.0, 31.0, 10.0,
            141.0, 175.0, 266.0, 221.0, 160.0, 260.0,
            1912.0, 1477.0, 1289.0, 3091.0, 2318.0, 1848.0,
            26.0, 48.0, 97.0, 21.0, 25.0, 35.0,
            5.0, 17.0, 9.0, 22.0, 20.0, 31.0,
            126.0, 92.0, 90.0, 80.0, 191.0, 139.0,
            3536.0, 2158.0, 2899.0, 2438.0, 1857.0, 685.0,
            88.0, 41.0, 71.0, 26.0, 26.0, 22.0,
            26.0, 10.0, 13.0, 28.0, 16.0, 9.0,
            154.0, 180.0, 103.0, 132.0, 160.0, 256.0,
            2176.0, 1381.0, 2288.0, 1559.0, 2375.0, 1604.0,
            44.0, 64.0, 41.0, 58.0, 106.0, 80.0,
            22.0, 6.0, 18.0, 7.0, 24.0, 29.0,
            362.0, 131.0, 172.0, 87.0, 199.0, 122.0,
            1037.0, 1399.0, 1746.0, 2543.0, 1242.0, 1027.0,
            32.0, 70.0, 31.0, 62.0, 17.0, 56.0,
        ];
        #[rustfmt::skip]
        let expected_trended: [f64; 40] = [
            0.21837504923102202, 0.22114546362792034, 0.2207356614118576,
            0.21954588311037646, 0.21846250557172514, 0.22120500474232507,
            0.2207104807147686, 0.21924797561694073, 0.2182110764964012,
            0.22129296391927716, 0.22067465471360104, 0.2197338099466015,
            0.21835579687872922, 0.2211499594361139, 0.22069147729507904,
            0.21969264304075797, 0.21897599759625927, 0.2212356026936647,
            0.22069610935881623, 0.2194446572454325, 0.21856729661997779,
            0.22112342066785085, 0.2207444962081235, 0.21936070003926927,
            0.21838331118383023, 0.22126462166732655, 0.22072318510741426,
            0.21926611932667234, 0.21833074164156294, 0.22063812831499874,
            0.22070442150405675, 0.21932213440294568, 0.21821457465100513,
            0.2210574343681132, 0.22073532634315018, 0.2198600681493942,
            0.21829957882093082, 0.22106976140409162, 0.22078396671589307,
            0.21939963897624193,
        ];
        #[rustfmt::skip]
        let expected_tagwise: [f64; 40] = [
            0.26264323176063703, 0.28243591028679577, 0.24977981840001498,
            0.21885006624029352, 0.2426969638226232, 0.2184480506608656,
            0.20336812379373626, 0.19749111797121818, 0.21122405343224196,
            0.2048097847793373, 0.19578217804982176, 0.21847120120941918,
            0.19613526417311763, 0.2374059564656287, 0.27989535766298096,
            0.19936502301217643, 0.2501474523439529, 0.21709189939212875,
            0.24636531239932158, 0.19513279743284845, 0.19958497564795658,
            0.23741175183867438, 0.20676320027566777, 0.19542968523532167,
            0.21792494189834125, 0.20473215339372977, 0.1961486645415931,
            0.22381094320449546, 0.22567430214578674, 0.20952784298268654,
            0.20736503372276133, 0.18844561958013636, 0.2051890784540556,
            0.22208529362992852, 0.19468134790483918, 0.21335200722389336,
            0.23142983650712093, 0.2153323241218121, 0.20699543541717647,
            0.2504739292780125,
        ];
        let design = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let libraries: [f64; 6] = [27357.0, 17731.0, 27824.0, 25664.0, 22284.0, 19328.0];
        let offset = Recycled::by_sample(libraries.iter().map(|v| v.ln()).collect());

        let out = estimate_disp(&counts, 40, 6, &design, 2, &offset, None, None, None).unwrap();

        assert_relative_eq!(out.common, 0.21796291038648624, max_relative = 1e-12);
        assert_relative_eq!(out.span, 1.0, max_relative = 1e-12);
        assert_relative_eq!(
            out.prior_df.as_ref().unwrap()[0],
            24.50373429920392,
            max_relative = 1e-11
        );

        let trended = out.trended.as_ref().unwrap();
        for (got, want) in trended.iter().zip(expected_trended.iter()) {
            assert_relative_eq!(got, want, max_relative = 1e-12);
        }
        let tagwise = out.tagwise.as_ref().unwrap();
        for (got, want) in tagwise.iter().zip(expected_tagwise.iter()) {
            assert_relative_eq!(got, want, max_relative = 1e-11);
        }
    }

    /// The span rule, against edgeR's current default rather than its legacy one.
    ///
    /// `WLEB` takes `chooseLowessSpan(ntags)` when `legacy.span` is `FALSE`, which
    /// it is by default, and only falls back to `min.span = 0.25, power = 0.5`
    /// when asked. Both rules return 1 below fifty genes, so the sizes below are
    /// chosen to separate them.
    ///
    /// ```r
    /// # Rscript, limma 3.66
    /// limma::chooseLowessSpan(200)   # 0.740972367463206
    /// limma::chooseLowessSpan(890)   # 0.568096640916288
    /// limma::chooseLowessSpan(10000) # 0.419698316267369
    /// ```
    #[test]
    fn test_default_span_matches_edger_choose_lowess_span() {
        assert_relative_eq!(default_span(10), 1.0, max_relative = 1e-12);
        assert_relative_eq!(default_span(50), 1.0, max_relative = 1e-12);
        assert_relative_eq!(default_span(200), 0.740972367463206, max_relative = 1e-12);
        assert_relative_eq!(default_span(890), 0.568096640916288, max_relative = 1e-12);
        assert_relative_eq!(default_span(10000), 0.419698316267369, max_relative = 1e-12);
    }

    #[test]
    fn test_parse_trend_method_matches_edger_spelling() {
        assert_eq!(parse_trend_method("locfit"), Some(TrendMethod::Locfit));
        assert_eq!(
            parse_trend_method("locfit.mixed"),
            Some(TrendMethod::LocfitMixed)
        );
        assert_eq!(
            parse_trend_method("movingave"),
            Some(TrendMethod::MovingAverage)
        );
        assert_eq!(parse_trend_method("none"), Some(TrendMethod::None));
        assert!(parse_trend_method("lowess").is_none());
    }

    /// Without a covariate the smoothed surface is the column mean repeated,
    /// so every gene shares one trend and the individual estimates are pulled
    /// towards it by `prior_n`.
    #[test]
    fn test_wleb_without_a_covariate_shares_one_curve() {
        let theta = [-2.0, -1.0, 0.0, 1.0, 2.0];
        // Two genes with peaks either side of zero.
        let loglik = vec![
            -4.0, -1.0, 0.0, -1.0, -4.0, //
            -9.0, -4.0, -1.0, 0.0, -1.0, //
        ];
        let out = wleb(
            &theta,
            &loglik,
            2,
            None,
            None,
            Some(WlebParams {
                prior_n: vec![0.0],
                trend_method: TrendMethod::None,
                ..Default::default()
            }),
        )
        .unwrap();

        // With no prior weight the individual estimates are the raw maximisers.
        let individual = out.individual.unwrap();
        assert!(individual[0] < individual[1]);
        // The shared curve is the mean of the two, so both genes see the same one.
        let shared = out.shared_loglik;
        assert_relative_eq!(shared[0], -6.5, max_relative = 1e-12);
        assert_relative_eq!(shared[5], -6.5, max_relative = 1e-12);
    }

    /// Increasing the prior sample size must pull both genes towards the shared
    /// curve, which is the whole point of the weighting.
    #[test]
    fn test_wleb_prior_weight_shrinks_towards_the_trend() {
        let theta = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let loglik = vec![
            -4.0, -1.0, 0.0, -1.0, -4.0, //
            -9.0, -4.0, -1.0, 0.0, -1.0, //
        ];
        let weak = wleb(
            &theta,
            &loglik,
            2,
            None,
            None,
            Some(WlebParams {
                prior_n: vec![0.0],
                trend_method: TrendMethod::None,
                overall: false,
                trend: false,
                individual: true,
                span: None,
            }),
        )
        .unwrap()
        .individual
        .unwrap();

        let strong = wleb(
            &theta,
            &loglik,
            2,
            None,
            None,
            Some(WlebParams {
                prior_n: vec![100.0],
                trend_method: TrendMethod::None,
                overall: false,
                trend: false,
                individual: true,
                span: None,
            }),
        )
        .unwrap()
        .individual
        .unwrap();

        let gap_weak = (weak[1] - weak[0]).abs();
        let gap_strong = (strong[1] - strong[0]).abs();
        assert!(
            gap_strong < gap_weak,
            "strong prior gap {gap_strong} should be under weak {gap_weak}"
        );
    }

    #[test]
    fn test_wleb_rejects_a_shape_mismatch() {
        let err = wleb(&[0.0, 1.0], &[1.0, 2.0, 3.0], 2, None, None, None).unwrap_err();
        assert!(matches!(err, EdgeErrors::LengthMismatch { .. }));
    }

    #[test]
    fn test_residual_df_drops_structural_zeros() {
        // 6 samples, 2 coefficients, so 4 residual df when nothing is zero.
        let design = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let counts = vec![
            5.0, 5.0, 5.0, 5.0, 5.0, 5.0, //
            0.0, 0.0, 0.0, 5.0, 5.0, 5.0, //
        ];
        let fitted = vec![
            5.0, 5.0, 5.0, 5.0, 5.0, 5.0, //
            0.0, 0.0, 0.0, 5.0, 5.0, 5.0, //
        ];
        let df = residual_df(&counts, &fitted, 2, 6, &design, 2).unwrap();
        assert_relative_eq!(df[0], 4.0, max_relative = 1e-12);
        // Three zeros leave three samples and a rank-1 design, so 2 df.
        assert_relative_eq!(df[1], 2.0, max_relative = 1e-12);
    }

    #[test]
    fn test_expand_to_all_genes_fills_with_the_least_abundant() {
        let values = [0.2, 0.5];
        let kept = [1usize, 3];
        let covariate = [4.0, 9.0];
        let out = expand_to_all_genes(&values, &kept, 4, 0.9, Some(&covariate));
        // Genes 0 and 2 were filtered, so they take the least abundant fitted value.
        assert_eq!(out, vec![0.2, 0.2, 0.2, 0.5]);
    }
}
