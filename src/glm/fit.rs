//! The `glmFit` front door.
//!
//! Resolves offsets and dispersions, picks between the closed-form one-way fit
//! and the general Levenberg fit, and optionally shrinks the coefficients
//! towards zero with a prior count.
//!
//! The dispatch is edgeR's: if the number of distinct design rows equals the
//! number of coefficients, the design is a one-way layout and
//! [`crate::glm::one_way`] can fit it in closed form. Otherwise
//! [`crate::glm::levenberg`] runs, with a larger iteration budget than its own
//! default because it is now the only thing standing between the caller and a
//! failed fit.

use crate::core::dgelist::DgeList;
use crate::core::expression::{augment_counts, resolve_lib_sizes};
use crate::glm::deviance::unit_nb_deviance;
use crate::glm::levenberg::{LevenbergParams, mglm_levenberg};
use crate::glm::one_way::mglm_one_way;
use crate::prelude::*;
use crate::utils::design::{design_as_factor, non_estimable};

////////////
// Consts //
////////////

/// Iteration budget for the general path.
///
/// `mglmLevenberg` defaults to 200, but `glmFit` raises it: at this point there
/// is no fallback left, so it is worth spending more iterations on the awkward
/// genes rather than reporting them unconverged.
const GLM_FIT_MAX_ITER: usize = 250;

/// Default prior count used to shrink log-fold-changes.
///
/// edgeR's default. Small enough not to move a well-powered gene, large enough
/// to stop a gene with a zero in one group reporting an infinite fold change.
pub const DEFAULT_PRIOR_COUNT: f64 = 0.125;

///////////////
// FitMethod //
///////////////

/// Which fitter produced a result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FitMethod {
    /// The closed-form one-way fit.
    OneWay,
    /// The general Levenberg-damped fit.
    Levenberg,
}

/// Output of [`glm_fit`].
#[derive(Clone, Debug)]
pub struct GlmFit {
    /// Coefficients, row-major `n_genes * n_coef`. Shrunk when `prior_count`
    /// is positive.
    pub coefficients: Vec<f64>,
    /// Coefficients before shrinkage, present only when shrinkage was applied.
    pub unshrunk_coefficients: Option<Vec<f64>>,
    /// Fitted means, row-major `n_genes * n_samples`.
    pub fitted: Vec<f64>,
    /// Residual deviance per gene.
    pub deviance: Vec<f64>,
    /// Residual degrees of freedom, the same for every gene.
    pub df_residual: usize,
    /// Which fitter ran.
    pub method: FitMethod,
}

/// Adds library-size-scaled prior counts to the data.
///
/// edgeR's `addPriorCount`. The prior is scaled by each library's size relative
/// to the average, so it perturbs every sample by the same relative amount
/// rather than penalising small libraries.
///
/// Note that edgePython's version takes a different branch when the offsets
/// arrive as a matrix, which is exactly how `glmFit` passes them, and ends up
/// with `log(lib + 2 * prior)` instead of `log(lib + 2 * prior * lib / mean)`.
/// See `UPSTREAM_DEVIATIONS.md` entry 6. This follows edgeR.
///
/// ### Params
///
/// * `counts` - Counts, row-major `n_genes * n_samples`
/// * `n_genes` - Number of genes
/// * `n_samples` - Number of samples
/// * `library_sizes` - Effective library size per sample
/// * `prior_count` - Prior count before scaling
///
/// ### Returns
///
/// The augmented counts and the matching per-sample log offsets.
pub fn add_prior_count<T: EdgeFloat>(
    counts: &[T],
    n_genes: usize,
    n_samples: usize,
    library_sizes: &[f64],
    prior_count: f64,
) -> Result<(Vec<f64>, Vec<f64>), EdgeErrors> {
    if library_sizes.len() != n_samples {
        return Err(EdgeErrors::LengthMismatch {
            name: "library_sizes",
            expected: n_samples,
            got: library_sizes.len(),
        });
    }

    let average: f64 = library_sizes.iter().sum::<f64>() / n_samples as f64;
    // Explicit rather than a negated comparison, so a NaN library size is
    // rejected instead of slipping through.
    if average <= 0.0 || average.is_nan() {
        return Err(EdgeErrors::InvalidArgument(
            "average library size must be positive".to_string(),
        ));
    }

    let scaled: Vec<f64> = library_sizes
        .iter()
        .map(|lib| prior_count * lib / average)
        .collect();
    let offsets: Vec<f64> = library_sizes
        .iter()
        .zip(scaled.iter())
        .map(|(lib, prior)| (lib + 2.0 * prior).ln())
        .collect();

    let mut augmented = Vec::with_capacity(n_genes * n_samples);
    for row in counts.chunks_exact(n_samples) {
        for (value, prior) in row.iter().zip(scaled.iter()) {
            augmented.push(value.to_f64().unwrap_or(0.0) + prior);
        }
    }

    Ok((augmented, offsets))
}

/// Fits genewise negative binomial GLMs.
///
/// ### Params
///
/// * `counts` - Counts, row-major `n_genes * n_samples`
/// * `n_genes` - Number of genes
/// * `n_samples` - Number of samples
/// * `design` - Design matrix, row-major `n_samples * n_coef`
/// * `n_coef` - Number of coefficients
/// * `dispersion` - Dispersion, recycled over genes and samples
/// * `offset` - Log-scale offsets, recycled over genes and samples
/// * `weights` - Optional observation weights
/// * `prior_count` - Shrinkage prior. Zero disables shrinkage.
///
/// ### Returns
///
/// The fit, or [`EdgeErrors`] if the shapes disagree, the design is rank
/// deficient, or a dispersion is negative.
#[allow(clippy::too_many_arguments)]
pub fn glm_fit<T: EdgeFloat>(
    counts: &[T],
    n_genes: usize,
    n_samples: usize,
    design: &[f64],
    n_coef: usize,
    dispersion: &Recycled<f64>,
    offset: &Recycled<f64>,
    weights: Option<&Recycled<f64>>,
    prior_count: f64,
) -> Result<GlmFit, EdgeErrors> {
    if n_coef > n_samples {
        return Err(EdgeErrors::DesignNotFullRank {
            n_cols: n_coef,
            rank: n_samples,
        });
    }
    if let Some(bad) = non_estimable(design, n_samples, n_coef)? {
        return Err(EdgeErrors::DesignNotFullRank {
            n_cols: n_coef,
            rank: n_coef - bad.len(),
        });
    }
    if prior_count < 0.0 {
        return Err(EdgeErrors::InvalidArgument(format!(
            "prior_count must be non-negative, got {prior_count}"
        )));
    }

    let (coefficients, fitted, method) = fit_dispatch(
        counts, n_genes, n_samples, design, n_coef, dispersion, offset, weights,
    )?;

    let deviance = residual_deviance(counts, n_genes, n_samples, &fitted, dispersion, weights);

    // Shrinkage refits the augmented data and replaces the coefficients, but
    // keeps the fitted values and deviance from the unshrunk fit, as edgeR does.
    let (coefficients, unshrunk) = if prior_count > 0.0 {
        // Per gene, off the compressed offsets: a gene-varying offset gives a
        // gene-varying library size, and collapsing it to one row would scale
        // every gene's prior by the wrong libraries.
        let library_sizes = resolve_lib_sizes(counts, n_genes, n_samples, None, Some(offset))?;
        let (augmented, augmented_offsets) =
            augment_counts(counts, n_genes, n_samples, &library_sizes, prior_count)?;
        let (shrunk, _, _) = fit_dispatch(
            &augmented,
            n_genes,
            n_samples,
            design,
            n_coef,
            dispersion,
            &augmented_offsets,
            weights,
        )?;
        (shrunk, Some(coefficients))
    } else {
        (coefficients, None)
    };

    Ok(GlmFit {
        coefficients,
        unshrunk_coefficients: unshrunk,
        fitted,
        deviance,
        df_residual: n_samples - n_coef,
        method,
    })
}

/// Chooses a fitter and runs it.
///
/// ### Params
///
/// * `counts` - Counts, row-major
/// * `n_genes` - Number of genes
/// * `n_samples` - Number of samples
/// * `design` - Row-major design
/// * `n_coef` - Number of coefficients
/// * `dispersion` - Dispersion
/// * `offset` - Offsets
/// * `weights` - Optional weights
///
/// ### Returns
///
/// Coefficients, fitted means and which fitter ran.
#[allow(clippy::too_many_arguments)]
fn fit_dispatch<T: EdgeFloat>(
    counts: &[T],
    n_genes: usize,
    n_samples: usize,
    design: &[f64],
    n_coef: usize,
    dispersion: &Recycled<f64>,
    offset: &Recycled<f64>,
    weights: Option<&Recycled<f64>>,
) -> Result<(Vec<f64>, Vec<f64>, FitMethod), EdgeErrors> {
    let (labels, n_groups) = design_as_factor(design, n_samples, n_coef)?;

    if n_groups == n_coef {
        let fit = mglm_one_way(
            counts,
            n_genes,
            n_samples,
            design,
            n_coef,
            Some(&labels),
            dispersion,
            offset,
            weights,
            None,
        )?;
        Ok((fit.coefficients, fit.fitted, FitMethod::OneWay))
    } else {
        let params = LevenbergParams {
            max_iter: GLM_FIT_MAX_ITER,
            ..Default::default()
        };
        let fit = mglm_levenberg(
            counts,
            n_genes,
            n_samples,
            design,
            n_coef,
            dispersion,
            offset,
            weights,
            None,
            Some(params),
        )?;
        Ok((fit.coefficients, fit.fitted, FitMethod::Levenberg))
    }
}

/// Residual deviance per gene from a set of fitted means.
///
/// ### Params
///
/// * `counts` - Counts, row-major
/// * `n_genes` - Number of genes
/// * `n_samples` - Number of samples
/// * `fitted` - Fitted means, row-major
/// * `dispersion` - Dispersion
/// * `weights` - Optional weights
///
/// ### Returns
///
/// One deviance per gene.
fn residual_deviance<T: EdgeFloat>(
    counts: &[T],
    n_genes: usize,
    n_samples: usize,
    fitted: &[f64],
    dispersion: &Recycled<f64>,
    weights: Option<&Recycled<f64>>,
) -> Vec<f64> {
    (0..n_genes)
        .map(|gene| {
            let disp = dispersion.row(gene, n_samples);
            let weight = weights.map(|w| w.row(gene, n_samples));
            let y = &counts[gene * n_samples..(gene + 1) * n_samples];
            let mu = &fitted[gene * n_samples..(gene + 1) * n_samples];
            y.iter()
                .zip(mu.iter())
                .enumerate()
                .map(|(j, (y_j, &mu_j))| {
                    let d = unit_nb_deviance(y_j.to_f64().unwrap_or(0.0), mu_j, disp.get(j));
                    weight.map_or(d, |w| w.get(j) * d)
                })
                .sum()
        })
        .collect()
}

/// Fits a [`DgeList`] using its own offsets and dispersion.
///
/// ### Params
///
/// * `dge` - The container, which must already carry a dispersion estimate
/// * `design` - Design matrix, row-major `n_samples * n_coef`
/// * `n_coef` - Number of coefficients
/// * `prior_count` - Shrinkage prior, or [`DEFAULT_PRIOR_COUNT`]
///
/// ### Returns
///
/// The fit, or [`EdgeErrors::InvalidArgument`] if no dispersion has been
/// estimated yet.
pub fn glm_fit_dge<T: EdgeFloat>(
    dge: &DgeList<T>,
    design: &[f64],
    n_coef: usize,
    prior_count: Option<f64>,
) -> Result<GlmFit, EdgeErrors> {
    let (dispersion, _) = dge.dispersion().ok_or_else(|| {
        EdgeErrors::InvalidArgument("no dispersion has been estimated for this DgeList".to_string())
    })?;
    let offset = dge.offset()?;

    glm_fit(
        &dge.counts,
        dge.n_genes,
        dge.n_samples,
        design,
        n_coef,
        &dispersion,
        &offset,
        dge.weights.as_ref(),
        prior_count.unwrap_or(DEFAULT_PRIOR_COUNT),
    )
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn fixture() -> (Vec<f64>, Vec<f64>, Recycled<f64>) {
        let counts = vec![
            10.0, 12.0, 11.0, 40.0, 44.0, 38.0, //
            50.0, 48.0, 52.0, 49.0, 51.0, 50.0, //
            2.0, 0.0, 5.0, 1.0, 3.0, 0.0, //
        ];
        let design = vec![
            1.0, 0.0, //
            1.0, 0.0, //
            1.0, 0.0, //
            1.0, 1.0, //
            1.0, 1.0, //
            1.0, 1.0, //
        ];
        let libraries: [f64; 6] = [1e6, 1.2e6, 0.9e6, 1.1e6, 1e6, 1.3e6];
        let offset = Recycled::by_sample(libraries.iter().map(|v| v.ln()).collect());
        (counts, design, offset)
    }

    /// Parity with edgeR 4.8.2, including the prior-count shrinkage:
    /// ```r
    /// y <- matrix(c(10,12,11,40,44,38, 50,48,52,49,51,50, 2,0,5,1,3,0),
    ///             nrow = 3, byrow = TRUE)
    /// X <- cbind(1, c(0,0,0,1,1,1))
    /// lib <- c(1e6,1.2e6,0.9e6,1.1e6,1e6,1.3e6)
    /// off <- matrix(rep(log(lib), each = 3), nrow = 3)
    /// glmFit(y, design = X, dispersion = 0.1, offset = off, prior.count = 0.125)
    /// ```
    #[test]
    fn test_matches_edger_glmfit_with_shrinkage() {
        let (counts, design, offset) = fixture();
        let fit = glm_fit(
            &counts,
            3,
            6,
            &design,
            2,
            &Recycled::scalar(0.1),
            &offset,
            None,
            DEFAULT_PRIOR_COUNT,
        )
        .unwrap();

        let expected = [
            -11.435_024_757_662_948,
            1.217_421_382_877_951,
            -9.918_936_523_346_485,
            -0.096_726_798_172_613,
            -12.929_342_857_228_917,
            -0.617_513_775_750_691,
        ];
        for (got, want) in fit.coefficients.iter().zip(expected.iter()) {
            assert_relative_eq!(got, want, epsilon = 1e-7, max_relative = 1e-6);
        }
        assert_eq!(fit.method, FitMethod::OneWay);
        assert_eq!(fit.df_residual, 4);
        assert!(fit.unshrunk_coefficients.is_some());
    }

    /// Prior-count shrinkage under a gene-varying offset, against edgeR 4.8.2:
    /// ```r
    /// y <- matrix(c(10,12,11,40,44,38, 50,48,52,49,51,50, 2,0,5,1,3,0,
    ///               100,90,110,300,280,320, 1,1,0,8,9,7,
    ///               500,520,480,505,495,510), nrow = 6, byrow = TRUE)
    /// X <- cbind(1, c(0,0,0,1,1,1))
    /// off <- outer(0:5, 0:5, function(g, j) log(1e3 * (1 + 0.25*j*(1 + 0.6*g))))
    /// glmFit(y, X, dispersion = 0.1, offset = off, prior.count = 0.125)
    /// ```
    ///
    /// The library sizes the prior is scaled by vary by gene here, so a fit that
    /// took them from one offset row would still get the unshrunk coefficients
    /// right and the shrunk ones wrong by up to 0.6 on the natural log scale.
    #[test]
    fn test_matches_edger_glmfit_with_shrinkage_and_a_gene_varying_offset() {
        let counts = vec![
            10.0, 12.0, 11.0, 40.0, 44.0, 38.0, //
            50.0, 48.0, 52.0, 49.0, 51.0, 50.0, //
            2.0, 0.0, 5.0, 1.0, 3.0, 0.0, //
            100.0, 90.0, 110.0, 300.0, 280.0, 320.0, //
            1.0, 1.0, 0.0, 8.0, 9.0, 7.0, //
            500.0, 520.0, 480.0, 505.0, 495.0, 510.0, //
        ];
        let design = vec![
            1.0, 0.0, //
            1.0, 0.0, //
            1.0, 0.0, //
            1.0, 1.0, //
            1.0, 1.0, //
            1.0, 1.0, //
        ];
        let offsets: Vec<f64> = (0..6)
            .flat_map(|g| {
                (0..6).map(move |j| (1e3 * (1.0 + 0.25 * j as f64 * (1.0 + 0.6 * g as f64))).ln())
            })
            .collect();

        let fit = glm_fit(
            &counts,
            6,
            6,
            &design,
            2,
            &Recycled::scalar(0.1),
            &Recycled::full(offsets, 6, 6).unwrap(),
            None,
            DEFAULT_PRIOR_COUNT,
        )
        .unwrap();

        let expected = [
            -4.71385797554227,
            0.8317728696432809,
            -3.2854254113008703,
            -0.6503664257070851,
            -6.475901113939795,
            -1.1813613300490617,
            -2.7255694787676576,
            0.206162145980485,
            -7.799915823609166,
            1.5260709158222285,
            -1.186163762454638,
            -1.0830875826258086,
        ];
        for (got, want) in fit.coefficients.iter().zip(expected.iter()) {
            assert_relative_eq!(got, want, epsilon = 1e-11, max_relative = 1e-11);
        }
    }

    /// Shrinkage must pull a fold change towards zero, and must leave the
    /// unshrunk coefficients available.
    #[test]
    fn test_shrinkage_pulls_towards_zero() {
        let (counts, design, offset) = fixture();
        let fit = glm_fit(
            &counts,
            3,
            6,
            &design,
            2,
            &Recycled::scalar(0.1),
            &offset,
            None,
            DEFAULT_PRIOR_COUNT,
        )
        .unwrap();
        let unshrunk = fit.unshrunk_coefficients.as_ref().unwrap();

        // Gene 2 has zeros in one group, so its slope is the one that moves.
        assert!(
            fit.coefficients[5].abs() < unshrunk[5].abs(),
            "shrunk {} should be smaller than unshrunk {}",
            fit.coefficients[5],
            unshrunk[5]
        );
    }

    #[test]
    fn test_zero_prior_count_leaves_the_coefficients_alone() {
        let (counts, design, offset) = fixture();
        let fit = glm_fit(
            &counts,
            3,
            6,
            &design,
            2,
            &Recycled::scalar(0.1),
            &offset,
            None,
            0.0,
        )
        .unwrap();
        assert!(fit.unshrunk_coefficients.is_none());
    }

    /// A continuous covariate is not a one-way layout, so the general fitter
    /// must be selected.
    #[test]
    fn test_dispatches_to_levenberg_for_a_continuous_design() {
        let (counts, _, offset) = fixture();
        let design = vec![1.0, 0.1, 1.0, 0.4, 1.0, 0.2, 1.0, 0.9, 1.0, 0.7, 1.0, 0.8];
        let fit = glm_fit(
            &counts,
            3,
            6,
            &design,
            2,
            &Recycled::scalar(0.1),
            &offset,
            None,
            0.0,
        )
        .unwrap();
        assert_eq!(fit.method, FitMethod::Levenberg);
    }

    #[test]
    fn test_rejects_a_rank_deficient_design() {
        let (counts, _, offset) = fixture();
        // Third column is the sum of the first two.
        let design = vec![
            1.0, 0.0, 1.0, //
            1.0, 0.0, 1.0, //
            1.0, 0.0, 1.0, //
            1.0, 1.0, 2.0, //
            1.0, 1.0, 2.0, //
            1.0, 1.0, 2.0, //
        ];
        let err = glm_fit(
            &counts,
            3,
            6,
            &design,
            3,
            &Recycled::scalar(0.1),
            &offset,
            None,
            0.0,
        )
        .unwrap_err();
        assert!(matches!(err, EdgeErrors::DesignNotFullRank { .. }));
    }

    #[test]
    fn test_rejects_a_negative_prior_count() {
        let (counts, design, offset) = fixture();
        let err = glm_fit(
            &counts,
            3,
            6,
            &design,
            2,
            &Recycled::scalar(0.1),
            &offset,
            None,
            -1.0,
        )
        .unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));
    }

    /// The prior is scaled by library size, so a sample with twice the library
    /// receives twice the prior. That is the property edgePython's matrix branch
    /// loses.
    #[test]
    fn test_prior_count_scales_with_library_size() {
        let counts = vec![0.0, 0.0, 0.0, 0.0];
        let libraries = [1e6, 3e6];
        let (augmented, offsets) = add_prior_count(&counts, 2, 2, &libraries, 1.0).unwrap();

        let average = 2e6;
        assert_relative_eq!(augmented[0], 1e6 / average, max_relative = 1e-12);
        assert_relative_eq!(augmented[1], 3e6 / average, max_relative = 1e-12);
        assert_relative_eq!(
            offsets[0],
            (1e6 + 2.0 * 1e6 / average).ln(),
            max_relative = 1e-12
        );
        assert_relative_eq!(
            offsets[1],
            (3e6 + 2.0 * 3e6 / average).ln(),
            max_relative = 1e-12
        );
    }

    #[test]
    fn test_add_prior_count_rejects_a_bad_library_length() {
        let err = add_prior_count(&[0.0_f64; 4], 2, 2, &[1e6], 1.0).unwrap_err();
        assert!(matches!(err, EdgeErrors::LengthMismatch { .. }));
    }
}
