//! Empirical Bayes shrinkage of the per-gene cell-level overdispersion.
//!
//! This has no upstream in edgeR, limma or `nebula`. It is edgePython's own
//! addition (`sc_fit.py`, `shrink_sc_disp`).
//!
//! The idea is to treat `phi = 1 / dispersion` as if it were a sample variance
//! on `df_residual` degrees of freedom and push it through limma's
//! [`squeeze_var`]. That assumption is the weak part: `phi` is a maximum
//! likelihood overdispersion, not a scaled chi-squared variance, so the prior
//! degrees of freedom that come back are a shrinkage knob rather than a
//! calibrated quantity. It works in the sense that it pulls extreme genes
//! towards the middle. It does not have edgeR's `squeezeVar` guarantees.
//!
//! The fallback ladder is edgePython's: trended, then untrended, then untrended
//! and non-robust, then no shrinkage at all. Every step is a genuine numerical
//! failure of the fit below it, not a preference.

use crate::errors::EdgeErrors;
use crate::limma::squeeze_var::{SqueezeVarParams, squeeze_var};

////////////
// Consts //
////////////

/// Smallest `phi` fed to [`squeeze_var`].
///
/// A gene whose dispersion overflows gives `phi = 0`, which `squeeze_var`
/// accepts but which drags the moment estimate of the prior to zero. edgePython
/// floors at the same value.
const PHI_FLOOR: f64 = 1e-8;

/// Fewest usable genes worth fitting a prior from.
///
/// Below this the moment estimator has nothing to work with, so the result is
/// returned unshrunk rather than shrunk towards noise.
const MIN_GENES_FOR_PRIOR: usize = 3;

/////////////////
// ShrinkParam //
/////////////////

/// Tuning knobs for [`shrink_sc_dispersion`].
#[derive(Clone, Copy, Debug)]
pub struct ShrinkParams {
    /// Whether to Winsorise the moments, protecting the prior from genes with
    /// extreme dispersion. edgePython defaults this on, so this does too, and
    /// it is the reason the fallback ladder exists: the robust fit is the one
    /// that can fail.
    pub robust: bool,
}

impl Default for ShrinkParams {
    fn default() -> Self {
        Self { robust: true }
    }
}

/// What [`shrink_sc_dispersion`] produces. Every vector is one entry per gene.
#[derive(Clone, Debug)]
pub struct ShrinkResult {
    /// `1 / dispersion` as fitted, before shrinkage. Infinite where the
    /// dispersion is zero.
    pub phi_raw: Vec<f64>,
    /// Posterior `phi`. Genes that were excluded from the prior fit carry the
    /// median prior rather than their own estimate.
    pub phi_post: Vec<f64>,
    /// The prior `phi` each gene was shrunk towards. Constant unless a
    /// covariate produced a trended prior.
    pub phi_prior: Vec<f64>,
    /// Residual degrees of freedom used for every gene.
    pub df_residual: f64,
    /// Prior degrees of freedom from the empirical Bayes fit. Length one unless
    /// the robust fit produced a per-gene vector, and all zero when the ladder
    /// fell through to no shrinkage.
    pub df_prior: Vec<f64>,
    /// `1 / phi_post`, the quantity callers actually substitute back into the
    /// model. Infinite where `phi_post` is zero.
    pub dispersion_shrunk: Vec<f64>,
}

//////////////
// Residual //
//////////////

/// Residual degrees of freedom for a NEBULA fit.
///
/// `n_cells - n_coef - (n_subjects - 1)`, floored at one. The subject term is
/// there because the random effect absorbs one degree of freedom per subject
/// beyond the first.
///
/// ### Params
///
/// * `n_cells` - Number of cells in the fit
/// * `n_coef` - Number of design columns
/// * `n_subjects` - Number of subjects
///
/// ### Returns
///
/// The residual degrees of freedom, at least one.
pub fn sc_residual_df(n_cells: usize, n_coef: usize, n_subjects: usize) -> f64 {
    let used = n_coef + n_subjects.saturating_sub(1);
    (n_cells.saturating_sub(used)).max(1) as f64
}

////////////
// Shrink //
////////////

/// Shrinks per-gene cell-level dispersions towards an empirical Bayes prior.
///
/// Only genes that converged and have a finite, positive `phi` contribute to
/// the prior. Everything else is handed the median prior, which is the most
/// honest thing to give a gene whose own estimate is not trustworthy.
///
/// See the module documentation: this procedure has no upstream reference and
/// the chi-squared assumption behind it is not satisfied.
///
/// ### Params
///
/// * `dispersion` - Per-gene cell-level dispersion from the NEBULA fit
/// * `convergence` - Per-gene convergence code; only `1` counts as converged
/// * `covariate` - Per-gene covariate for a trended prior, or `None` for a
///   constant one. edgePython uses `log2(mean count per cell + 0.5)`.
/// * `df_residual` - Residual degrees of freedom, see [`sc_residual_df`]
/// * `params` - Tuning knobs, or `None` for [`ShrinkParams::default`]
///
/// ### Returns
///
/// The shrunken dispersions and the prior that produced them, or
/// [`EdgeErrors`] if the inputs disagree on length, are empty, or
/// `df_residual` is not positive and finite.
pub fn shrink_sc_dispersion(
    dispersion: &[f64],
    convergence: &[i32],
    covariate: Option<&[f64]>,
    df_residual: f64,
    params: Option<ShrinkParams>,
) -> Result<ShrinkResult, EdgeErrors> {
    let params = params.unwrap_or_default();
    let n_genes = dispersion.len();

    if n_genes == 0 {
        return Err(EdgeErrors::InvalidArgument(
            "shrink_sc_dispersion was given an empty dispersion vector.".to_string(),
        ));
    }
    if convergence.len() != n_genes {
        return Err(EdgeErrors::LengthMismatch {
            name: "convergence",
            expected: n_genes,
            got: convergence.len(),
        });
    }
    if let Some(cov) = covariate
        && cov.len() != n_genes
    {
        return Err(EdgeErrors::LengthMismatch {
            name: "covariate",
            expected: n_genes,
            got: cov.len(),
        });
    }
    if !(df_residual.is_finite() && df_residual > 0.0) {
        return Err(EdgeErrors::InvalidArgument(format!(
            "shrink_sc_dispersion needs a positive finite df_residual; got {df_residual}."
        )));
    }

    let phi_raw: Vec<f64> = dispersion
        .iter()
        .map(|&d| if d > 0.0 { 1.0 / d } else { f64::INFINITY })
        .collect();

    // A gene earns a say in the prior by converging and by having a finite phi.
    let usable: Vec<usize> = (0..n_genes)
        .filter(|&g| convergence[g] == 1 && phi_raw[g].is_finite())
        .collect();

    if usable.len() < MIN_GENES_FOR_PRIOR {
        return Ok(unshrunk(phi_raw, df_residual, dispersion));
    }

    let phi_usable: Vec<f64> = usable.iter().map(|&g| phi_raw[g].max(PHI_FLOOR)).collect();
    let cov_usable: Option<Vec<f64>> = covariate.map(|c| usable.iter().map(|&g| c[g]).collect());

    let df = [df_residual];

    // edgePython's ladder: trended, then untrended, then untrended and
    // non-robust. Each rung is only reached when the one above it fails.
    let mut attempts: Vec<(Option<&[f64]>, bool)> = Vec::with_capacity(3);
    if let Some(cov) = cov_usable.as_deref() {
        attempts.push((Some(cov), params.robust));
    }
    attempts.push((None, params.robust));
    if params.robust {
        attempts.push((None, false));
    }

    let mut fit = None;
    for (cov, robust) in attempts {
        let sv_params = SqueezeVarParams {
            robust,
            ..SqueezeVarParams::default()
        };
        if let Ok(result) = squeeze_var(&phi_usable, &df, cov, Some(sv_params)) {
            fit = Some(result);
            break;
        }
    }

    let Some(sv) = fit else {
        return Ok(unshrunk(phi_raw, df_residual, dispersion));
    };

    // The prior a gene outside the fit gets. Trended priors give one value per
    // usable gene, so the median is the only defensible single number.
    let median_prior = median(&sv.var_prior);

    let mut phi_post = vec![median_prior; n_genes];
    let mut phi_prior = vec![median_prior; n_genes];
    for (slot, &gene) in usable.iter().enumerate() {
        phi_post[gene] = sv.var_post[slot];
        phi_prior[gene] = if sv.var_prior.len() == 1 {
            sv.var_prior[0]
        } else {
            sv.var_prior[slot]
        };
    }

    let dispersion_shrunk = phi_post
        .iter()
        .map(|&p| if p > 0.0 { 1.0 / p } else { f64::INFINITY })
        .collect();

    Ok(ShrinkResult {
        phi_raw,
        phi_post,
        phi_prior,
        df_residual,
        df_prior: sv.df_prior,
        dispersion_shrunk,
    })
}

/// Builds the result for the cases where no prior could be fitted.
///
/// ### Params
///
/// * `phi_raw` - Unshrunken `phi`, moved into the result
/// * `df_residual` - Residual degrees of freedom to record
/// * `dispersion` - The dispersions as fitted
///
/// ### Returns
///
/// A result in which nothing was shrunk and `df_prior` is zero.
fn unshrunk(phi_raw: Vec<f64>, df_residual: f64, dispersion: &[f64]) -> ShrinkResult {
    let finite: Vec<f64> = phi_raw.iter().copied().filter(|p| p.is_finite()).collect();
    let prior = median(&finite);
    ShrinkResult {
        phi_post: phi_raw.clone(),
        phi_prior: vec![prior; phi_raw.len()],
        phi_raw,
        df_residual,
        df_prior: vec![0.0],
        dispersion_shrunk: dispersion.to_vec(),
    }
}

/// Median of a slice, ignoring non-finite entries.
///
/// ### Params
///
/// * `values` - Values to reduce
///
/// ### Returns
///
/// The median, or `NaN` when nothing finite is present.
fn median(values: &[f64]) -> f64 {
    let mut finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return f64::NAN;
    }
    finite.sort_by(|a, b| a.total_cmp(b));
    let mid = finite.len() / 2;
    if finite.len().is_multiple_of(2) {
        0.5 * (finite[mid - 1] + finite[mid])
    } else {
        finite[mid]
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Dispersions with one clear outlier, all converged.
    fn fixture() -> (Vec<f64>, Vec<i32>) {
        let dispersion = vec![0.5, 0.4, 0.6, 0.55, 0.45, 0.52, 0.48, 5.0, 0.51, 0.49];
        let convergence = vec![1; 10];
        (dispersion, convergence)
    }

    #[test]
    fn test_residual_df_subtracts_coefficients_and_subjects() {
        assert_relative_eq!(sc_residual_df(100, 2, 6), 93.0);
        assert_relative_eq!(sc_residual_df(10, 2, 20), 1.0);
    }

    #[test]
    fn test_shrinkage_pulls_the_outlier_towards_the_others() {
        let (dispersion, convergence) = fixture();
        let out = shrink_sc_dispersion(&dispersion, &convergence, None, 93.0, None).unwrap();

        let raw_outlier = out.phi_raw[7];
        let post_outlier = out.phi_post[7];
        let prior = out.phi_prior[7];
        // phi = 1/0.5 = 2 for the bulk, 0.2 for the outlier gene.
        assert!(raw_outlier < prior);
        assert!(post_outlier > raw_outlier);
        assert!(post_outlier < prior);
    }

    #[test]
    fn test_non_converged_genes_get_the_prior() {
        let (dispersion, mut convergence) = fixture();
        convergence[3] = -20;
        let out = shrink_sc_dispersion(&dispersion, &convergence, None, 93.0, None).unwrap();
        assert_relative_eq!(out.phi_post[3], out.phi_prior[3]);
    }

    #[test]
    fn test_zero_dispersion_gives_infinite_phi_and_is_excluded() {
        let (mut dispersion, convergence) = fixture();
        dispersion[0] = 0.0;
        let out = shrink_sc_dispersion(&dispersion, &convergence, None, 93.0, None).unwrap();
        assert!(out.phi_raw[0].is_infinite());
        assert!(out.phi_post[0].is_finite());
        assert_relative_eq!(out.phi_post[0], out.phi_prior[0]);
    }

    #[test]
    fn test_too_few_converged_genes_returns_unshrunk() {
        let (dispersion, mut convergence) = fixture();
        for c in convergence.iter_mut().skip(2) {
            *c = -20;
        }
        let out = shrink_sc_dispersion(&dispersion, &convergence, None, 93.0, None).unwrap();
        assert_eq!(out.dispersion_shrunk, dispersion);
        assert_relative_eq!(out.df_prior[0], 0.0);
    }

    #[test]
    fn test_a_covariate_gives_a_trended_prior() {
        let (dispersion, convergence) = fixture();
        let covariate: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let out =
            shrink_sc_dispersion(&dispersion, &convergence, Some(&covariate), 93.0, None).unwrap();
        let first = out.phi_prior[0];
        assert!(out.phi_prior.iter().any(|p| (p - first).abs() > 1e-12));
    }

    #[test]
    fn test_shrunk_dispersion_is_the_reciprocal_of_the_posterior() {
        let (dispersion, convergence) = fixture();
        let out = shrink_sc_dispersion(&dispersion, &convergence, None, 93.0, None).unwrap();
        for (&d, &p) in out.dispersion_shrunk.iter().zip(out.phi_post.iter()) {
            assert_relative_eq!(d, 1.0 / p, max_relative = 1e-14);
        }
    }

    #[test]
    fn test_empty_input_is_rejected() {
        assert!(shrink_sc_dispersion(&[], &[], None, 93.0, None).is_err());
    }

    #[test]
    fn test_length_mismatch_is_rejected() {
        let (dispersion, _) = fixture();
        assert!(shrink_sc_dispersion(&dispersion, &[1, 1], None, 93.0, None).is_err());
    }

    #[test]
    fn test_non_positive_df_is_rejected() {
        let (dispersion, convergence) = fixture();
        assert!(shrink_sc_dispersion(&dispersion, &convergence, None, 0.0, None).is_err());
        assert!(shrink_sc_dispersion(&dispersion, &convergence, None, f64::NAN, None).is_err());
    }
}
