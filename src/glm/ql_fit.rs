//! `glmQLFit`: quasi-likelihood negative binomial fits.
//!
//! The quasi-likelihood pipeline is edgeR's default for bulk RNA-seq. It sits on
//! top of an ordinary negative binomial fit and adds a second layer of
//! dispersion: the negative binomial dispersion describes variation shared
//! across genes, and the quasi-likelihood dispersion `s2` describes what is left
//! over for each gene individually. Shrinking `s2` towards a fitted prior is
//! what buys the method its power on small experiments.
//!
//! ### Two pipelines
//!
//! edgeR carries an old and a new one, and they differ in more than bookkeeping.
//!
//! The **legacy** path takes the residual deviance at face value, divides by the
//! residual degrees of freedom after dropping structural zeros, and shrinks
//! that. It is what every edgeR result before 4.0 used.
//!
//! The **current** path, the default here as in edgeR, replaces both numerator
//! and denominator. Each observation contributes a fraction of a degree of
//! freedom rather than one, because a low-count observation carries less
//! information about the dispersion than a high-count one, and the deviance is
//! rescaled to match. Those adjustments come from [`crate::ql::weights`]. It
//! also rescales the negative binomial dispersion by an average
//! quasi-dispersion first, so the two layers do not describe the same variation
//! twice.
//!
//! ### References
//!
//! Lun, Chen and Smyth, Methods in Molecular Biology 1418, 2016
//! Chen, Lun and Smyth, F1000Research 5:1438, 2016

use crate::dispersion::cox_reid::{CoxReidParams, common_dispersion_cox_reid};
use crate::dispersion::estimate::residual_df;
use crate::errors::EdgeErrors;
use crate::glm::fit::{DEFAULT_PRIOR_COUNT, FitMethod, glm_fit};
use crate::limma::squeeze_var::{SqueezeVarParams, squeeze_var};
use crate::ql::weights::{compute_adjust_vec, update_prior};
use crate::utils::design::choose_lowess_span;
use crate::utils::recycled::Recycled;
use crate::utils::traits::EdgeFloat;

/// Largest negative binomial dispersion the current pipeline will accept.
///
/// edgeR clamps here rather than erroring. Above 4 the quasi-likelihood weights
/// are being asked about a regime the Chebyshev fits do not cover.
const MAX_DISPERSION: f64 = 4.0;

/// Floor on the residual degrees of freedom before dividing a deviance by it.
const MIN_DF: f64 = 1e-8;

/// `small_n` for the automatic top-abundance proportion.
const TOP_PROPORTION_SMALL_N: usize = 20;

/// `min_span` for the automatic top-abundance proportion.
const TOP_PROPORTION_MIN_SPAN: f64 = 0.02;

/// Exponent for the automatic top-abundance proportion.
const TOP_PROPORTION_POWER: f64 = 1.0 / 3.0;

/// Tuning knobs for [`glm_ql_fit`].
#[derive(Clone, Copy, Debug)]
pub struct QlFitParams {
    /// Fit the prior against abundance rather than sharing one prior across all
    /// genes.
    pub abundance_trend: bool,
    /// Use the robust empirical Bayes fit, giving outlier genes their own
    /// smaller prior degrees of freedom.
    pub robust: bool,
    /// Winsorising tail proportions for the robust fit.
    pub winsor_tail_p: (f64, f64),
    /// Take edgeR's pre-4.0 pipeline instead of the current one.
    pub legacy: bool,
    /// Proportion of the most abundant genes used to estimate the negative
    /// binomial dispersion when none is supplied. `None` derives it.
    pub top_proportion: Option<f64>,
    /// Prior count for the log-fold-change shrinkage inside the underlying fit.
    pub prior_count: f64,
}

impl Default for QlFitParams {
    /// edgeR's defaults.
    fn default() -> Self {
        Self {
            abundance_trend: true,
            robust: false,
            winsor_tail_p: (0.05, 0.1),
            legacy: false,
            top_proportion: None,
            prior_count: DEFAULT_PRIOR_COUNT,
        }
    }
}

/// Output of [`glm_ql_fit`].
#[derive(Clone, Debug)]
pub struct QlFit {
    /// Coefficients, row-major `n_genes * n_coef`.
    pub coefficients: Vec<f64>,
    /// Coefficients before the prior-count shrinkage.
    pub unshrunk_coefficients: Option<Vec<f64>>,
    /// Fitted means, row-major `n_genes * n_samples`.
    pub fitted: Vec<f64>,
    /// Residual deviance per gene.
    pub deviance: Vec<f64>,
    /// Nominal residual degrees of freedom, `n_samples - n_coef`.
    pub df_residual: usize,
    /// Which fitter ran.
    pub method: FitMethod,
    /// The negative binomial dispersion actually used, after any clamping or
    /// estimation.
    pub dispersion: Recycled<f64>,
    /// Posterior quasi-likelihood dispersion per gene.
    pub s2_post: Vec<f64>,
    /// Prior quasi-likelihood dispersion. One value, or one per gene when the
    /// fit was trended or robust.
    pub s2_prior: Vec<f64>,
    /// Prior degrees of freedom. One value, or one per gene when robust.
    pub df_prior: Vec<f64>,
    /// Adjusted residual degrees of freedom, current pipeline only.
    pub df_residual_adj: Option<Vec<f64>>,
    /// Adjusted residual deviance, current pipeline only.
    pub deviance_adj: Option<Vec<f64>>,
    /// Residual degrees of freedom after dropping structural zeros, legacy
    /// pipeline only.
    pub df_residual_zeros: Option<Vec<f64>>,
    /// Average quasi-dispersion the negative binomial dispersion was divided
    /// by, current pipeline only.
    pub average_ql_dispersion: Option<f64>,
    /// Top-abundance proportion used, when a dispersion was estimated.
    pub top_proportion: Option<f64>,
}

/// Applies a map to every stored value of a recycled matrix.
///
/// The compressed forms stay compressed, so clamping a scalar dispersion still
/// costs one value rather than `n_genes * n_samples`.
///
/// ### Params
///
/// * `source` - The recycled matrix
/// * `f` - Map applied to each stored value
///
/// ### Returns
///
/// A recycled matrix of the same shape.
fn map_recycled(source: &Recycled<f64>, f: impl Fn(f64) -> f64) -> Recycled<f64> {
    match source {
        Recycled::Scalar(v) => Recycled::Scalar(f(*v)),
        Recycled::ByGene(v) => Recycled::ByGene(v.iter().map(|x| f(*x)).collect()),
        Recycled::BySample(v) => Recycled::BySample(v.iter().map(|x| f(*x)).collect()),
        Recycled::Full(v) => Recycled::Full(v.iter().map(|x| f(*x)).collect()),
    }
}

/// Restricts a recycled matrix to a subset of genes.
///
/// ### Params
///
/// * `source` - The recycled matrix
/// * `kept` - Gene indices to retain
/// * `n_samples` - Number of samples
///
/// ### Returns
///
/// A recycled matrix over the retained genes.
fn subset_recycled(source: &Recycled<f64>, kept: &[usize], n_samples: usize) -> Recycled<f64> {
    match source {
        Recycled::Scalar(v) => Recycled::Scalar(*v),
        Recycled::BySample(v) => Recycled::BySample(v.clone()),
        Recycled::ByGene(v) => Recycled::ByGene(kept.iter().map(|&g| v[g]).collect()),
        Recycled::Full(v) => {
            let mut out = Vec::with_capacity(kept.len() * n_samples);
            for &gene in kept {
                out.extend_from_slice(&v[gene * n_samples..(gene + 1) * n_samples]);
            }
            Recycled::Full(out)
        }
    }
}

/// Estimates a negative binomial dispersion from the most abundant genes.
///
/// edgeR does not use every gene. The quasi-likelihood weights need a dispersion
/// describing the well-measured genes, and the low-count tail that dominates a
/// typical experiment would drag it around.
///
/// ### Params
///
/// * `counts` - Counts, row-major `n_genes * n_samples`
/// * `n_genes` - Number of genes
/// * `n_samples` - Number of samples
/// * `design` - Row-major design
/// * `n_coef` - Number of coefficients
/// * `offset` - Log-scale offsets
/// * `weights` - Optional observation weights
/// * `ave_log_cpm` - Abundance per gene
/// * `top_proportion` - Proportion of genes to keep, most abundant first
///
/// ### Returns
///
/// The common dispersion of the retained genes.
#[allow(clippy::too_many_arguments)]
fn dispersion_from_top_genes<T: EdgeFloat>(
    counts: &[T],
    n_genes: usize,
    n_samples: usize,
    design: &[f64],
    n_coef: usize,
    offset: &Recycled<f64>,
    weights: Option<&Recycled<f64>>,
    ave_log_cpm: &[f64],
    top_proportion: f64,
) -> Result<f64, EdgeErrors> {
    let n_top = ((top_proportion * n_genes as f64).ceil() as usize).clamp(1, n_genes);

    let mut order: Vec<usize> = (0..n_genes).collect();
    order.sort_by(|&a, &b| ave_log_cpm[b].total_cmp(&ave_log_cpm[a]));
    let kept = &order[..n_top];

    let mut selected = Vec::with_capacity(n_top * n_samples);
    for &gene in kept {
        selected.extend_from_slice(&counts[gene * n_samples..(gene + 1) * n_samples]);
    }
    let selected_offset = subset_recycled(offset, kept, n_samples);
    let selected_weights = weights.map(|w| subset_recycled(w, kept, n_samples));

    // Subsetting is off here: this already is a subset, chosen by abundance
    // rather than stratified across it, and a second one would compound them.
    let params = CoxReidParams {
        subset: None,
        ..Default::default()
    };
    common_dispersion_cox_reid(
        &selected,
        n_top,
        n_samples,
        design,
        n_coef,
        &selected_offset,
        selected_weights.as_ref(),
        None,
        Some(params),
    )
}

/// What the chosen pipeline left behind beyond the residual variance.
enum Extras {
    /// Residual degrees of freedom after dropping structural zeros.
    Legacy(Vec<f64>),
    /// Adjusted deviance and degrees of freedom, plus the average
    /// quasi-dispersion the fit was rescaled by.
    Current {
        /// Adjusted residual deviance per gene.
        deviance_adj: Vec<f64>,
        /// Adjusted residual degrees of freedom per gene.
        df_residual_adj: Vec<f64>,
        /// Average quasi-dispersion.
        average: f64,
    },
}

/// Settles which negative binomial dispersion the fit will use.
///
/// ### Params
///
/// * `counts` - Counts, row-major
/// * `n_genes` - Number of genes
/// * `n_samples` - Number of samples
/// * `design` - Row-major design
/// * `n_coef` - Number of coefficients
/// * `dispersion` - Supplied dispersion, if any
/// * `offset` - Log-scale offsets
/// * `weights` - Optional observation weights
/// * `ave_log_cpm` - Abundance per gene
/// * `params` - Tuning knobs
///
/// ### Returns
///
/// The dispersion to fit with, and the top-abundance proportion if one was used
/// to estimate it.
#[allow(clippy::too_many_arguments)]
fn resolve_dispersion<T: EdgeFloat>(
    counts: &[T],
    n_genes: usize,
    n_samples: usize,
    design: &[f64],
    n_coef: usize,
    dispersion: Option<&Recycled<f64>>,
    offset: &Recycled<f64>,
    weights: Option<&Recycled<f64>>,
    ave_log_cpm: &[f64],
    params: &QlFitParams,
) -> Result<(Recycled<f64>, Option<f64>), EdgeErrors> {
    if let Some(d) = dispersion {
        d.validate(n_genes, n_samples)?;
        // The current pipeline clamps; the legacy one takes what it is given.
        let resolved = if params.legacy {
            d.clone()
        } else {
            map_recycled(d, |v| v.min(MAX_DISPERSION))
        };
        return Ok((resolved, None));
    }

    if params.legacy {
        return Err(EdgeErrors::InvalidArgument(
            "the legacy quasi-likelihood pipeline needs an explicit dispersion".to_string(),
        ));
    }

    let proportion = match params.top_proportion {
        Some(p) => {
            if !(0.0..=1.0).contains(&p) {
                return Err(EdgeErrors::InvalidArgument(format!(
                    "top_proportion must lie in [0, 1], got {p}"
                )));
            }
            p
        }
        None => {
            let df_residual = (n_samples - n_coef) as f64;
            let scale = (n_genes as f64 * df_residual.sqrt()).round() as usize;
            choose_lowess_span(
                scale,
                TOP_PROPORTION_SMALL_N,
                TOP_PROPORTION_MIN_SPAN,
                TOP_PROPORTION_POWER,
            )
        }
    };

    let estimated = dispersion_from_top_genes(
        counts,
        n_genes,
        n_samples,
        design,
        n_coef,
        offset,
        weights,
        ave_log_cpm,
        proportion,
    )?;
    Ok((Recycled::scalar(estimated), Some(proportion)))
}

/// Fits quasi-likelihood negative binomial GLMs.
///
/// ### Params
///
/// * `counts` - Counts, row-major `n_genes * n_samples`
/// * `n_genes` - Number of genes
/// * `n_samples` - Number of samples
/// * `design` - Design matrix, row-major `n_samples * n_coef`
/// * `n_coef` - Number of coefficients
/// * `dispersion` - Negative binomial dispersion. `None` estimates one from the
///   most abundant genes, which the legacy pipeline does not support.
/// * `offset` - Log-scale offsets
/// * `weights` - Optional observation weights
/// * `ave_log_cpm` - Average log2 counts per million per gene
/// * `params` - Tuning knobs, or [`QlFitParams::default`]
///
/// ### Returns
///
/// The fit with its squeezed quasi-likelihood dispersions, or [`EdgeErrors`] if
/// the shapes disagree or the legacy pipeline is asked to estimate a dispersion.
#[allow(clippy::too_many_arguments)]
pub fn glm_ql_fit<T: EdgeFloat>(
    counts: &[T],
    n_genes: usize,
    n_samples: usize,
    design: &[f64],
    n_coef: usize,
    dispersion: Option<&Recycled<f64>>,
    offset: &Recycled<f64>,
    weights: Option<&Recycled<f64>>,
    ave_log_cpm: &[f64],
    params: Option<QlFitParams>,
) -> Result<QlFit, EdgeErrors> {
    let params = params.unwrap_or_default();

    if ave_log_cpm.len() != n_genes {
        return Err(EdgeErrors::LengthMismatch {
            name: "ave_log_cpm",
            expected: n_genes,
            got: ave_log_cpm.len(),
        });
    }
    if n_coef >= n_samples {
        return Err(EdgeErrors::InvalidArgument(format!(
            "no residual degrees of freedom: {n_coef} coefficients for {n_samples} samples"
        )));
    }

    let (dispersion, top_proportion) = resolve_dispersion(
        counts,
        n_genes,
        n_samples,
        design,
        n_coef,
        dispersion,
        offset,
        weights,
        ave_log_cpm,
        &params,
    )?;

    let fit = glm_fit(
        counts,
        n_genes,
        n_samples,
        design,
        n_coef,
        &dispersion,
        offset,
        weights,
        params.prior_count,
    )?;

    // Which residual variance gets squeezed, and on which degrees of freedom,
    // is the whole difference between the two pipelines.
    let (fit, s2, df, extras) = if params.legacy {
        let zeros = residual_df(counts, &fit.fitted, n_genes, n_samples, design, n_coef)?;
        let s2: Vec<f64> = fit
            .deviance
            .iter()
            .zip(zeros.iter())
            .map(|(dev, d)| if *d > 0.0 { dev / d.max(MIN_DF) } else { 0.0 })
            .collect();
        let df = zeros.clone();
        (fit, s2, df, Extras::Legacy(zeros))
    } else {
        let average = update_prior(
            counts,
            n_genes,
            n_samples,
            &fit.fitted,
            design,
            n_coef,
            &dispersion,
            weights,
            ave_log_cpm,
        )?;

        // Refit with the negative binomial dispersion rescaled, so the two
        // dispersion layers do not describe the same variation twice.
        let scaled = map_recycled(&dispersion, |v| v / average[0]);
        let refit = glm_fit(
            counts,
            n_genes,
            n_samples,
            design,
            n_coef,
            &scaled,
            offset,
            weights,
            params.prior_count,
        )?;

        let adjusted = compute_adjust_vec(
            counts,
            n_genes,
            n_samples,
            &refit.fitted,
            design,
            n_coef,
            &dispersion,
            &average,
            weights,
        )?;

        let s2 = adjusted.s2;
        let df = adjusted.df.clone();
        (
            refit,
            s2,
            df,
            Extras::Current {
                deviance_adj: adjusted.deviance,
                df_residual_adj: adjusted.df,
                average: average[0],
            },
        )
    };

    let s2: Vec<f64> = s2.iter().map(|v| v.max(0.0)).collect();
    let covariate = params.abundance_trend.then_some(ave_log_cpm);

    let squeezed = squeeze_var(
        &s2,
        &df,
        covariate,
        Some(SqueezeVarParams {
            robust: params.robust,
            winsor_tail_p: params.winsor_tail_p,
            span: None,
            legacy: None,
        }),
    )?;

    let (df_residual_adj, deviance_adj, df_residual_zeros, average_ql_dispersion) = match extras {
        Extras::Legacy(zeros) => (None, None, Some(zeros), None),
        Extras::Current {
            deviance_adj,
            df_residual_adj,
            average,
        } => (
            Some(df_residual_adj),
            Some(deviance_adj),
            None,
            Some(average),
        ),
    };

    Ok(QlFit {
        coefficients: fit.coefficients,
        unshrunk_coefficients: fit.unshrunk_coefficients,
        fitted: fit.fitted,
        deviance: fit.deviance,
        df_residual: fit.df_residual,
        method: fit.method,
        dispersion,
        s2_post: squeezed.var_post,
        s2_prior: squeezed.var_prior,
        df_prior: squeezed.df_prior,
        df_residual_adj,
        deviance_adj,
        df_residual_zeros,
        average_ql_dispersion,
        top_proportion,
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Eight genes by six samples on dyadic counts, so R and Rust see identical
    /// bits.
    ///
    /// ```r
    /// y <- matrix(c(8,16,32,64,128,256, 4,4,8,8,16,16, 64,64,64,128,128,128,
    ///               12,20,28,40,56,72, 100,110,90,220,200,240, 6,6,6,12,12,12,
    ///               33,29,31,60,66,58, 5,7,9,11,13,15), nrow = 8, byrow = TRUE)
    /// X <- cbind(1, c(0,0,0,1,1,1))
    /// off <- matrix(log(1024), 8, 6)
    /// glmQLFit(y, design = X, dispersion = 0.1, offset = off, legacy = FALSE)
    /// ```
    #[allow(clippy::type_complexity)]
    fn fixture() -> (Vec<f64>, Vec<f64>, Vec<f64>, Recycled<f64>) {
        #[rustfmt::skip]
        let counts: Vec<f64> = vec![
            8.0, 16.0, 32.0, 64.0, 128.0, 256.0,
            4.0, 4.0, 8.0, 8.0, 16.0, 16.0,
            64.0, 64.0, 64.0, 128.0, 128.0, 128.0,
            12.0, 20.0, 28.0, 40.0, 56.0, 72.0,
            100.0, 110.0, 90.0, 220.0, 200.0, 240.0,
            6.0, 6.0, 6.0, 12.0, 12.0, 12.0,
            33.0, 29.0, 31.0, 60.0, 66.0, 58.0,
            5.0, 7.0, 9.0, 11.0, 13.0, 15.0,
        ];
        let design = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        #[rustfmt::skip]
        let ave_log_cpm: Vec<f64> = vec![
            16.352208774832395, 13.428_444_360_659_48, 16.540653864245503,
            15.247872115017659, 17.265_794_023_014_92, 13.385375638767593,
            15.515907201909817, 13.510906520851451,
        ];
        let offset = Recycled::scalar(1024.0_f64.ln());
        (counts, design, ave_log_cpm, offset)
    }

    /// Full parity with `glmQLFit(legacy = FALSE)`: the rescaling, the adjusted
    /// deviances and degrees of freedom, and the squeezed dispersions.
    #[test]
    fn test_matches_edger_glm_ql_fit() {
        let (counts, design, ave_log_cpm, offset) = fixture();
        let fit = glm_ql_fit(
            &counts,
            8,
            6,
            &design,
            2,
            Some(&Recycled::scalar(0.1)),
            &offset,
            None,
            &ave_log_cpm,
            None,
        )
        .unwrap();

        assert_relative_eq!(fit.average_ql_dispersion.unwrap(), 1.0, max_relative = 1e-9);

        #[rustfmt::skip]
        let expected_deviance_adj = [
            14.056_702_642_505_744, 2.637_071_899_051_510_3, 0.0,
            3.622_743_572_123_198, 0.335_888_879_589_633_1, 0.0,
            0.13845513353745997, 0.901_705_566_985_588,
        ];
        for (got, want) in fit
            .deviance_adj
            .as_ref()
            .unwrap()
            .iter()
            .zip(expected_deviance_adj.iter())
        {
            assert_relative_eq!(got, want, epsilon = 1e-12, max_relative = 1e-9);
        }

        #[rustfmt::skip]
        let expected_df_adj = [
            3.9930661446445948, 3.9165793318728928, 4.000_432_236_818_225,
            3.993_449_436_198_199, 4.000_786_650_272_691, 3.9135554751192987,
            3.997_871_470_245_632, 3.9254667939092465,
        ];
        for (got, want) in fit
            .df_residual_adj
            .as_ref()
            .unwrap()
            .iter()
            .zip(expected_df_adj.iter())
        {
            assert_relative_eq!(got, want, max_relative = 1e-9);
        }

        // The reference here is `glmQLFit` given a *vector* offset, not the
        // matrix form used for the deviances above. The two describe the same
        // model and edgeR agrees with itself on everything up to this point,
        // but its internal `aveLogCPM` corrupts the first gene when the offset
        // arrives as a matrix (`UPSTREAM_DEVIATIONS.md` entry 11), which throws
        // the trended prior for every gene. Feeding limma's own `squeezeVar`
        // the matrix-offset covariate reproduces edgeR's wrong answer exactly,
        // and the vector-offset covariate reproduces this one, which is what
        // pins the cause.
        #[rustfmt::skip]
        let expected_s2_post = [
            2.345_460_198_842_870_3, 0.445_691_045_010_048_86, 0.00006973623748113801,
            0.648_121_251_900_738_4, 0.055_977_102_693_401_455, 0.00001201585381793704,
            0.038_430_468_762_857_07, 0.152_181_733_621_699_12,
        ];
        for (got, want) in fit.s2_post.iter().zip(expected_s2_post.iter()) {
            assert_relative_eq!(got, want, epsilon = 1e-12, max_relative = 1e-9);
        }

        #[rustfmt::skip]
        let expected_s2_prior = [
            3.900_756_118_528_421e-4, 4.082_620_246_643_251e-5, 2.0919450483691712e-4,
            1.3097776273064543e-1, 2.063_339_839_045_184e-5, 3.552_327_374_654_404e-5,
            4.6021345637373576e-2, 5.306_012_383_376_751e-5,
        ];
        for (got, want) in fit.s2_prior.iter().zip(expected_s2_prior.iter()) {
            assert_relative_eq!(got, want, epsilon = 1e-12, max_relative = 1e-9);
        }

        assert_relative_eq!(fit.df_prior[0], 2.000_419_894_664_604, max_relative = 1e-9);
    }

    /// The coefficients come from the rescaled refit, not the first fit, so they
    /// pin that the rescaling actually happened.
    #[test]
    fn test_coefficients_match_edger() {
        let (counts, design, ave_log_cpm, offset) = fixture();
        let fit = glm_ql_fit(
            &counts,
            8,
            6,
            &design,
            2,
            Some(&Recycled::scalar(0.1)),
            &offset,
            None,
            &ave_log_cpm,
            None,
        )
        .unwrap();

        #[rustfmt::skip]
        let expected = [
            -3.998_302_407_272_247_3, 2.073_604_138_029_604,
            -5.234_572_423_573_774, 0.902_455_000_021_505_2,
            -2.770_881_612_936_046, 0.692_172_046_401_738_7,
            -3.929_753_093_122_353, 1.025_618_522_757_791_4,
            -2.325_296_511_038_456_7, 0.787_776_161_427_847_1,
            -5.119_337_159_996_189, 0.682_890_680_392_755_9,
            -3.493_704_561_642_108, 0.680_348_083_641_102_9,
            -4.968_106_190_272_266, 0.610_909_082_322_973_7,
        ];
        for (got, want) in fit.coefficients.iter().zip(expected.iter()) {
            assert_relative_eq!(got, want, epsilon = 1e-7, max_relative = 1e-6);
        }
    }

    /// The legacy pipeline squeezes the raw deviance on the zero-adjusted
    /// degrees of freedom, so it must produce different numbers and fill the
    /// other set of optional fields.
    #[test]
    fn test_legacy_pipeline_takes_the_other_route() {
        let (counts, design, ave_log_cpm, offset) = fixture();
        let params = QlFitParams {
            legacy: true,
            ..Default::default()
        };
        let fit = glm_ql_fit(
            &counts,
            8,
            6,
            &design,
            2,
            Some(&Recycled::scalar(0.1)),
            &offset,
            None,
            &ave_log_cpm,
            Some(params),
        )
        .unwrap();

        assert!(fit.df_residual_zeros.is_some());
        assert!(fit.deviance_adj.is_none());
        assert!(fit.df_residual_adj.is_none());
        assert!(fit.average_ql_dispersion.is_none());
        // Every gene here is fully observed, so the zero adjustment leaves the
        // nominal residual degrees of freedom alone.
        for d in fit.df_residual_zeros.as_ref().unwrap() {
            assert_relative_eq!(d, &4.0, max_relative = 1e-12);
        }
    }

    #[test]
    fn test_dispersion_is_clamped_on_the_current_pipeline() {
        let (counts, design, ave_log_cpm, offset) = fixture();
        let fit = glm_ql_fit(
            &counts,
            8,
            6,
            &design,
            2,
            Some(&Recycled::scalar(9.0)),
            &offset,
            None,
            &ave_log_cpm,
            None,
        )
        .unwrap();
        assert_eq!(fit.dispersion, Recycled::Scalar(MAX_DISPERSION));
    }

    #[test]
    fn test_legacy_pipeline_does_not_clamp() {
        let (counts, design, ave_log_cpm, offset) = fixture();
        let params = QlFitParams {
            legacy: true,
            ..Default::default()
        };
        let fit = glm_ql_fit(
            &counts,
            8,
            6,
            &design,
            2,
            Some(&Recycled::scalar(9.0)),
            &offset,
            None,
            &ave_log_cpm,
            Some(params),
        )
        .unwrap();
        assert_eq!(fit.dispersion, Recycled::Scalar(9.0));
    }

    /// With no dispersion supplied the current pipeline estimates one from the
    /// most abundant genes and records the proportion it used.
    #[test]
    fn test_estimates_a_dispersion_when_none_is_given() {
        let (counts, design, ave_log_cpm, offset) = fixture();
        let fit = glm_ql_fit(
            &counts,
            8,
            6,
            &design,
            2,
            None,
            &offset,
            None,
            &ave_log_cpm,
            None,
        )
        .unwrap();

        let proportion = fit.top_proportion.unwrap();
        assert!(
            (0.0..=1.0).contains(&proportion),
            "proportion {proportion} outside [0, 1]"
        );
        match fit.dispersion {
            Recycled::Scalar(d) => {
                assert!(d.is_finite() && d >= 0.0, "estimated dispersion {d}");
            }
            _ => panic!("an estimated dispersion should be a scalar"),
        }
    }

    #[test]
    fn test_legacy_pipeline_refuses_to_estimate_a_dispersion() {
        let (counts, design, ave_log_cpm, offset) = fixture();
        let params = QlFitParams {
            legacy: true,
            ..Default::default()
        };
        let err = glm_ql_fit(
            &counts,
            8,
            6,
            &design,
            2,
            None,
            &offset,
            None,
            &ave_log_cpm,
            Some(params),
        )
        .unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));
    }

    #[test]
    fn test_rejects_a_bad_top_proportion() {
        let (counts, design, ave_log_cpm, offset) = fixture();
        let params = QlFitParams {
            top_proportion: Some(1.5),
            ..Default::default()
        };
        let err = glm_ql_fit(
            &counts,
            8,
            6,
            &design,
            2,
            None,
            &offset,
            None,
            &ave_log_cpm,
            Some(params),
        )
        .unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));
    }

    #[test]
    fn test_rejects_a_covariate_of_the_wrong_length() {
        let (counts, design, _, offset) = fixture();
        let err = glm_ql_fit(
            &counts,
            8,
            6,
            &design,
            2,
            Some(&Recycled::scalar(0.1)),
            &offset,
            None,
            &[0.0; 3],
            None,
        )
        .unwrap_err();
        assert!(matches!(err, EdgeErrors::LengthMismatch { .. }));
    }

    /// Turning the abundance trend off shares one prior across all genes, so
    /// `s2_prior` collapses to a single value.
    #[test]
    fn test_untrended_prior_is_shared() {
        let (counts, design, ave_log_cpm, offset) = fixture();
        let params = QlFitParams {
            abundance_trend: false,
            ..Default::default()
        };
        let fit = glm_ql_fit(
            &counts,
            8,
            6,
            &design,
            2,
            Some(&Recycled::scalar(0.1)),
            &offset,
            None,
            &ave_log_cpm,
            Some(params),
        )
        .unwrap();
        assert_eq!(fit.s2_prior.len(), 1);
    }
}
