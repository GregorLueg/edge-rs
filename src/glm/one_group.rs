//! Single-group negative binomial fits.
//!
//! edgeR's `mglmOneGroup`: one coefficient per gene, fitted by Fisher scoring.
//! With a single group the score and information both collapse to sums over
//! samples, so there is no linear system to solve and each Newton step is a
//! division.
//!
//! This is not just a special case kept for tidiness. `aveLogCPM` calls it for
//! every gene on every analysis, and `mglmOneWay` calls it once per group, so it
//! sits underneath both `estimateDisp` and the fast path of `glmFit`.

use rayon::prelude::*;

use crate::core::expression::check_dispersion;
use crate::errors::EdgeErrors;
use crate::utils::recycled::{Recycled, RecycledRow};
use crate::utils::traits::EdgeFloat;

/// Bound on the linear predictor before exponentiating, as in the Levenberg fit.
const ETA_CLAMP: f64 = 500.0;

/// Floor applied to fitted means, which appear in denominators.
const MIN_POSITIVE: f64 = 1e-300;

/// Coefficient assigned to a gene with no counts or no library.
///
/// `log(0)` is negative infinity, which would poison every downstream sum.
/// edgeR substitutes a large negative number, giving a fitted mean of about
/// `2e-9` counts, indistinguishable from zero in practice and finite in
/// arithmetic.
const EMPTY_GENE_COEF: f64 = -20.0;

/// Information below which a step is treated as undefined.
const MIN_INFORMATION: f64 = 1e-300;

/// Tuning knobs for [`mglm_one_group`].
#[derive(Clone, Copy, Debug)]
pub struct OneGroupParams {
    /// Iteration budget per gene.
    pub max_iter: usize,
    /// Convergence tolerance on the coefficient step.
    ///
    /// A gene stops when the step is smaller than `tol * (|coef| + 0.1)`. Much
    /// tighter than the Levenberg fit's, because a single Newton step here costs
    /// two passes over the samples rather than a factorisation.
    pub tol: f64,
}

impl Default for OneGroupParams {
    /// edgeR's defaults.
    fn default() -> Self {
        Self {
            max_iter: 50,
            tol: 1e-10,
        }
    }
}

/// Fits an intercept-only negative binomial GLM to every gene.
///
/// ### Params
///
/// * `counts` - Counts, row-major `n_genes * n_samples`
/// * `n_genes` - Number of genes
/// * `n_samples` - Number of samples
/// * `dispersion` - Dispersion, recycled over genes and samples
/// * `offset` - Log-scale offsets, recycled over genes and samples
/// * `weights` - Optional observation weights
/// * `coef_start` - Optional starting coefficient per gene. A non-finite entry
///   asks for that gene to be initialised from the data, matching edgeR's use of
///   `NA` as a per-gene opt-out.
/// * `params` - Tuning knobs, or [`OneGroupParams::default`]
///
/// ### Returns
///
/// One coefficient per gene on the log scale, or [`EdgeErrors`] if the shapes
/// disagree.
#[allow(clippy::too_many_arguments)]
pub fn mglm_one_group<T: EdgeFloat>(
    counts: &[T],
    n_genes: usize,
    n_samples: usize,
    dispersion: &Recycled<f64>,
    offset: &Recycled<f64>,
    weights: Option<&Recycled<f64>>,
    coef_start: Option<&[f64]>,
    params: Option<OneGroupParams>,
) -> Result<Vec<f64>, EdgeErrors> {
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
    dispersion.validate(n_genes, n_samples)?;
    check_dispersion(dispersion)?;
    offset.validate(n_genes, n_samples)?;
    if let Some(w) = weights {
        w.validate(n_genes, n_samples)?;
    }
    if let Some(start) = coef_start
        && start.len() != n_genes
    {
        return Err(EdgeErrors::LengthMismatch {
            name: "coef_start",
            expected: n_genes,
            got: start.len(),
        });
    }

    let mut coefficients = vec![0.0; n_genes];
    // Built once and shared, rather than per gene.
    let all_samples: Vec<usize> = (0..n_samples).collect();

    coefficients
        .par_iter_mut()
        .enumerate()
        .for_each(|(gene, coef)| {
            let disp_row = dispersion.row(gene, n_samples);
            let offset_row = offset.row(gene, n_samples);
            let weight_row = weights.map(|w| w.row(gene, n_samples));
            let y = &counts[gene * n_samples..(gene + 1) * n_samples];

            let start = coef_start
                .map(|s| s[gene])
                .filter(|v| v.is_finite())
                .unwrap_or_else(|| initial_coefficient(y, &all_samples, offset_row, weight_row));

            *coef = fit_one_gene(
                y,
                &all_samples,
                disp_row,
                offset_row,
                weight_row,
                start,
                &params,
            );
        });

    Ok(coefficients)
}

/// Starting coefficient from the weighted total count over the weighted total
/// library size.
///
/// ### Params
///
/// * `y` - Counts for this gene, indexed by the full sample range
/// * `samples` - Which samples take part. [`mglm_one_way`] passes one group's
///   columns; [`mglm_one_group`] passes all of them.
/// * `offset` - Offset row
/// * `weights` - Optional weight row
///
/// ### Returns
///
/// The log of the crude rate, or [`EMPTY_GENE_COEF`] if the gene or the library
/// is empty.
///
/// [`mglm_one_way`]: crate::glm::one_way::mglm_one_way
pub(crate) fn initial_coefficient<T: EdgeFloat>(
    y: &[T],
    samples: &[usize],
    offset: RecycledRow<'_, f64>,
    weights: Option<RecycledRow<'_, f64>>,
) -> f64 {
    let mut total_counts = 0.0;
    let mut total_library = 0.0;
    for &j in samples {
        let w = weights.map_or(1.0, |w| w.get(j));
        total_counts += w * y[j].to_f64().unwrap_or(0.0);
        total_library += w * offset.get(j).exp();
    }
    if total_counts > 0.0 && total_library > 0.0 {
        (total_counts / total_library).ln()
    } else {
        EMPTY_GENE_COEF
    }
}

/// Runs Fisher scoring for one gene.
///
/// The score is `sum_j w_j (y_j - mu_j) / (1 + phi_j mu_j)` and the information
/// is `sum_j w_j mu_j / (1 + phi_j mu_j)`, both accumulated in one pass since
/// they share the denominator.
///
/// ### Params
///
/// * `y` - Counts for this gene, indexed by the full sample range
/// * `samples` - Which samples take part
/// * `dispersion` - Dispersion row
/// * `offset` - Offset row
/// * `weights` - Optional weight row
/// * `start` - Starting coefficient
/// * `params` - Tuning knobs
///
/// ### Returns
///
/// The fitted coefficient.
pub(crate) fn fit_one_gene<T: EdgeFloat>(
    y: &[T],
    samples: &[usize],
    dispersion: RecycledRow<'_, f64>,
    offset: RecycledRow<'_, f64>,
    weights: Option<RecycledRow<'_, f64>>,
    start: f64,
    params: &OneGroupParams,
) -> f64 {
    let mut beta = start;

    for _ in 0..params.max_iter {
        let mut score = 0.0;
        let mut information = 0.0;

        for &j in samples {
            let eta = (beta + offset.get(j)).clamp(-ETA_CLAMP, ETA_CLAMP);
            let mu = eta.exp().max(MIN_POSITIVE);
            let denom = 1.0 + dispersion.get(j) * mu;
            let w = weights.map_or(1.0, |w| w.get(j));
            score += w * (y[j].to_f64().unwrap_or(0.0) - mu) / denom;
            information += w * mu / denom;
        }

        if information <= MIN_INFORMATION {
            break;
        }

        let step = score / information;
        beta += step;
        if step.abs() < params.tol * (beta.abs() + 0.1) {
            break;
        }
    }

    beta
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// With zero dispersion and no offset the fit is the Poisson maximum
    /// likelihood estimate, which is the log of the sample mean.
    #[test]
    fn test_poisson_fit_is_the_log_sample_mean() {
        let counts = vec![10.0, 12.0, 14.0, 5.0, 5.0, 5.0];
        let coef = mglm_one_group(
            &counts,
            2,
            3,
            &Recycled::scalar(0.0),
            &Recycled::scalar(0.0),
            None,
            None,
            None,
        )
        .unwrap();
        assert_relative_eq!(coef[0], 12.0_f64.ln(), max_relative = 1e-10);
        assert_relative_eq!(coef[1], 5.0_f64.ln(), max_relative = 1e-10);
    }

    /// The dispersion cancels out of the score when every sample shares an
    /// offset, so the estimate is the log sample mean whatever the dispersion.
    #[test]
    fn test_dispersion_does_not_move_a_balanced_fit() {
        let counts = vec![10.0, 12.0, 14.0];
        for phi in [0.0, 0.01, 0.5, 10.0] {
            let coef = mglm_one_group(
                &counts,
                1,
                3,
                &Recycled::scalar(phi),
                &Recycled::scalar(0.0),
                None,
                None,
                None,
            )
            .unwrap();
            assert_relative_eq!(coef[0], 12.0_f64.ln(), max_relative = 1e-8);
        }
    }

    /// Parity with edgeR 4.8.2:
    /// ```r
    /// y <- matrix(c(10,12,11,40,44,38, 50,48,52,49,51,50, 2,0,5,1,3,0),
    ///             nrow = 3, byrow = TRUE)
    /// mglmOneGroup(y, dispersion = 0.1, offset = log(c(1e6,1.2e6,0.9e6,1.1e6,1e6,1.3e6)))
    /// ```
    #[test]
    fn test_matches_edger_mglm_one_group() {
        let counts = vec![
            10.0, 12.0, 11.0, 40.0, 44.0, 38.0, //
            50.0, 48.0, 52.0, 49.0, 51.0, 50.0, //
            2.0, 0.0, 5.0, 1.0, 3.0, 0.0, //
        ];
        let libraries: [f64; 6] = [1e6, 1.2e6, 0.9e6, 1.1e6, 1e6, 1.3e6];
        let offset = Recycled::by_sample(libraries.iter().map(|v| v.ln()).collect());

        let coef = mglm_one_group(
            &counts,
            3,
            6,
            &Recycled::scalar(0.1),
            &offset,
            None,
            None,
            None,
        )
        .unwrap();

        let expected = [
            -10.649_844_006_464_7,
            -9.968_996_644_248_97,
            -13.269_736_410_245_08,
        ];
        for (got, want) in coef.iter().zip(expected.iter()) {
            assert_relative_eq!(got, want, epsilon = 1e-9, max_relative = 1e-8);
        }
    }

    #[test]
    fn test_offsets_shift_the_coefficient() {
        let counts = vec![10.0, 12.0, 14.0];
        let plain = mglm_one_group(
            &counts,
            1,
            3,
            &Recycled::scalar(0.1),
            &Recycled::scalar(0.0),
            None,
            None,
            None,
        )
        .unwrap();
        let shifted = mglm_one_group(
            &counts,
            1,
            3,
            &Recycled::scalar(0.1),
            &Recycled::scalar(3.0),
            None,
            None,
            None,
        )
        .unwrap();
        assert_relative_eq!(shifted[0], plain[0] - 3.0, max_relative = 1e-8);
    }

    #[test]
    fn test_zero_weight_removes_a_sample() {
        let counts = vec![10.0, 12.0, 1000.0];
        let weights = Recycled::by_sample(vec![1.0, 1.0, 0.0]);
        let coef = mglm_one_group(
            &counts,
            1,
            3,
            &Recycled::scalar(0.0),
            &Recycled::scalar(0.0),
            Some(&weights),
            None,
            None,
        )
        .unwrap();
        assert_relative_eq!(coef[0], 11.0_f64.ln(), max_relative = 1e-10);
    }

    #[test]
    fn test_empty_gene_gets_the_finite_fallback() {
        let counts = vec![0.0, 0.0, 0.0];
        let coef = mglm_one_group(
            &counts,
            1,
            3,
            &Recycled::scalar(0.1),
            &Recycled::scalar(0.0),
            None,
            None,
            None,
        )
        .unwrap();
        assert!(coef[0].is_finite());
        assert!(coef[0] <= EMPTY_GENE_COEF);
    }

    /// A non-finite starting value asks for that gene to be initialised from the
    /// data, which is how edgeR uses `NA` per gene.
    #[test]
    fn test_non_finite_start_falls_back_to_the_data() {
        let counts = vec![10.0, 12.0, 14.0, 5.0, 5.0, 5.0];
        let start = vec![f64::NAN, 5.0_f64.ln()];
        let coef = mglm_one_group(
            &counts,
            2,
            3,
            &Recycled::scalar(0.0),
            &Recycled::scalar(0.0),
            None,
            Some(&start),
            None,
        )
        .unwrap();
        assert_relative_eq!(coef[0], 12.0_f64.ln(), max_relative = 1e-10);
        assert_relative_eq!(coef[1], 5.0_f64.ln(), max_relative = 1e-10);
    }

    #[test]
    fn test_rejects_a_shape_mismatch() {
        let err = mglm_one_group(
            &[1.0_f64; 5],
            2,
            3,
            &Recycled::scalar(0.0),
            &Recycled::scalar(0.0),
            None,
            None,
            None,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            EdgeErrors::LengthMismatch { name: "counts", .. }
        ));
    }
}
