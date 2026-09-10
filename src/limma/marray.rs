//! The fit object the limma pipeline passes along.
//!
//! limma's `MArrayLM` is an untyped tagged list whose field set depends on
//! which functions have run over it: `lmFit` sets one group, `contrasts.fit`
//! replaces three of them, `eBayes` adds eleven more. The map from field to
//! dimension lives in `R/subsetting.R:115-121`.
//!
//! Here it is one struct whose later fields are `Option`, populated in that
//! same order. Each stage consumes the object and returns it, so nothing is
//! cloned and a fit that has not been through `eBayes` cannot be handed to
//! something that needs a moderated t: the `Option` is the check.

use crate::limma::lm_fit::LmFitResult;
use crate::prelude::*;

//////////////
// MArrayLm //
//////////////

/// A linear model fit, growing as the pipeline runs over it.
///
/// Everything before [`MArrayLm::contrasts`] comes from `lm_fit`. After
/// `contrasts_fit` the coefficient axis is the contrast axis, and
/// [`MArrayLm::n_coef`] counts contrasts rather than design columns.
#[derive(Clone, Debug)]
pub struct MArrayLm {
    /// Number of genes.
    pub n_genes: usize,
    /// Number of samples in the design.
    pub n_samples: usize,
    /// Width of the coefficient axis: design columns, or contrasts once
    /// `contrasts_fit` has run.
    pub n_coef: usize,
    /// Coefficients, row-major `n_genes * n_coef`. `NaN` where not estimable.
    pub coefficients: Vec<f64>,
    /// Unscaled standard deviations, row-major `n_genes * n_coef`.
    pub stdev_unscaled: Vec<f64>,
    /// Residual standard deviation per gene.
    pub sigma: Vec<f64>,
    /// Residual degrees of freedom per gene.
    pub df_residual: Vec<f64>,
    /// Unscaled covariance of the estimable coefficients, row-major
    /// `rank * rank`, or `n_coef * n_coef` after `contrasts_fit`.
    pub cov_coefficients: Vec<f64>,
    /// Design column indices, accepted ones first. Length is the original
    /// `n_coef`, which after `contrasts_fit` is no longer this struct's.
    pub pivot: Vec<usize>,
    /// Rank of the design.
    pub rank: usize,
    /// Design matrix, row-major `n_samples * pivot.len()`.
    pub design: Vec<f64>,
    /// Row means of the response, one per gene. limma's `Amean`.
    ///
    /// `None` when the caller had none to give, which makes
    /// `EBayesTrend::Amean` an error rather than a silent no-trend fit.
    pub amean: Option<Vec<f64>>,
    /// Contrast matrix, column-major `original n_coef * n_coef`, once
    /// `contrasts_fit` has run.
    pub contrasts: Option<Vec<f64>>,
    /// Prior degrees of freedom. Length one, or `n_genes` for a robust fit.
    pub df_prior: Option<Vec<f64>>,
    /// Prior variance. Length one, or `n_genes` for a trended fit.
    pub s2_prior: Option<Vec<f64>>,
    /// Prior variance of the coefficients, one per coefficient. The
    /// B-statistic's `v0`.
    pub var_prior: Option<Vec<f64>>,
    /// Assumed proportion of differentially expressed genes.
    pub proportion: Option<f64>,
    /// Posterior variance per gene.
    pub s2_post: Option<Vec<f64>>,
    /// Moderated t, row-major `n_genes * n_coef`.
    pub t: Option<Vec<f64>>,
    /// Total degrees of freedom per gene, residual plus prior, capped.
    pub df_total: Option<Vec<f64>>,
    /// Two-sided p-values, row-major `n_genes * n_coef`.
    pub p_value: Option<Vec<f64>>,
    /// Log-odds of differential expression, row-major `n_genes * n_coef`.
    pub lods: Option<Vec<f64>>,
    /// Moderated F per gene. `None` when the design is not full rank.
    pub f_stat: Option<Vec<f64>>,
    /// P-value of [`MArrayLm::f_stat`].
    pub f_p_value: Option<Vec<f64>>,
}

impl MArrayLm {
    /// Wraps an `lm_fit` result, with the response's row means alongside.
    ///
    /// ### Params
    ///
    /// * `fit` - What `lm_fit` returned
    /// * `design` - Row-major design, `n_samples * n_coef`
    /// * `n_coef` - Number of design columns
    /// * `n_samples` - Number of samples
    /// * `amean` - Row means of the response, one per gene, or `None`.
    ///   `voom` returns them as [`crate::limma::voom::VoomResult::amean`].
    ///
    /// ### Returns
    ///
    /// The fit object with every empirical Bayes field unset, or
    /// [`EdgeErrors::LengthMismatch`] if `design` or `amean` disagrees with the
    /// fit's shape.
    pub fn from_lm_fit(
        fit: LmFitResult,
        design: &[f64],
        n_coef: usize,
        n_samples: usize,
        amean: Option<Vec<f64>>,
    ) -> Result<Self, EdgeErrors> {
        if n_coef == 0 {
            return Err(EdgeErrors::MustBePositive("n_coef".to_string()));
        }
        let n_genes = fit.sigma.len();
        if fit.coefficients.len() != n_genes * n_coef {
            return Err(EdgeErrors::LengthMismatch {
                name: "coefficients",
                expected: n_genes * n_coef,
                got: fit.coefficients.len(),
            });
        }
        if design.len() != n_samples * n_coef {
            return Err(EdgeErrors::LengthMismatch {
                name: "design",
                expected: n_samples * n_coef,
                got: design.len(),
            });
        }
        if let Some(a) = amean.as_ref()
            && a.len() != n_genes
        {
            return Err(EdgeErrors::LengthMismatch {
                name: "amean",
                expected: n_genes,
                got: a.len(),
            });
        }

        Ok(Self {
            n_genes,
            n_samples,
            n_coef,
            coefficients: fit.coefficients,
            stdev_unscaled: fit.stdev_unscaled,
            sigma: fit.sigma,
            df_residual: fit.df_residual,
            cov_coefficients: fit.cov_coefficients,
            pivot: fit.pivot,
            rank: fit.rank,
            design: design.to_vec(),
            amean,
            contrasts: None,
            df_prior: None,
            s2_prior: None,
            var_prior: None,
            proportion: None,
            s2_post: None,
            t: None,
            df_total: None,
            p_value: None,
            lods: None,
            f_stat: None,
            f_p_value: None,
        })
    }

    /// Clears everything `eBayes` and `treat` set.
    ///
    /// `contrasts.fit` drops the test statistics because they belong to the old
    /// coefficient axis (`R/contrasts.R:21-25`); rotating them would be
    /// meaningless and leaving them would be worse.
    pub(crate) fn clear_tests(&mut self) {
        self.df_prior = None;
        self.s2_prior = None;
        self.var_prior = None;
        self.proportion = None;
        self.s2_post = None;
        self.t = None;
        self.df_total = None;
        self.p_value = None;
        self.lods = None;
        self.f_stat = None;
        self.f_p_value = None;
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    fn fit(n_genes: usize, n_coef: usize) -> LmFitResult {
        LmFitResult {
            coefficients: vec![0.0; n_genes * n_coef],
            stdev_unscaled: vec![1.0; n_genes * n_coef],
            sigma: vec![1.0; n_genes],
            df_residual: vec![3.0; n_genes],
            fitted: vec![0.0; n_genes * 5],
            cov_coefficients: vec![1.0; n_coef * n_coef],
            pivot: (0..n_coef).collect(),
            rank: n_coef,
        }
    }

    #[test]
    fn test_from_lm_fit_carries_the_shape() {
        let m = MArrayLm::from_lm_fit(fit(4, 2), &[1.0; 10], 2, 5, None).unwrap();
        assert_eq!(m.n_genes, 4);
        assert_eq!(m.n_coef, 2);
        assert!(m.t.is_none());
    }

    #[test]
    fn test_from_lm_fit_rejects_a_bad_amean() {
        let e = MArrayLm::from_lm_fit(fit(4, 2), &[1.0; 10], 2, 5, Some(vec![0.0; 3]));
        assert!(e.is_err());
    }

    #[test]
    fn test_from_lm_fit_rejects_a_bad_design() {
        assert!(MArrayLm::from_lm_fit(fit(4, 2), &[1.0; 8], 2, 5, None).is_err());
    }

    #[test]
    fn test_clear_tests_empties_every_ebayes_field() {
        let mut m = MArrayLm::from_lm_fit(fit(4, 2), &[1.0; 10], 2, 5, None).unwrap();
        m.t = Some(vec![1.0; 8]);
        m.f_stat = Some(vec![1.0; 4]);
        m.clear_tests();
        assert!(m.t.is_none());
        assert!(m.f_stat.is_none());
    }
}
