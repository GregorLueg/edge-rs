//! Closed-form fits for one-way layouts.
//!
//! edgeR's `mglmOneWay`. When the design has exactly one column per distinct
//! design row, every group's mean is an independent intercept-only fit, so
//! there is no linear system and no Levenberg damping: fit each group with
//! Fisher scoring, then rotate the group means back into coefficient space with
//! a single small solve shared by all genes.
//!
//! This is the path `glmFit` takes for the overwhelming majority of bulk
//! experiments, since a design built from a group factor always qualifies.
//! [`crate::glm::levenberg`] is the fallback for anything else.

use faer::MatRef;
use faer::linalg::solvers::Solve;
use rayon::prelude::*;

use crate::core::expression::check_dispersion;
use crate::errors::EdgeErrors;
use crate::glm::one_group::{OneGroupParams, fit_one_gene, initial_coefficient};
use crate::utils::design::design_as_factor;
use crate::utils::recycled::Recycled;
use crate::utils::traits::EdgeFloat;

/// Bound on the linear predictor before exponentiating.
const ETA_CLAMP: f64 = 500.0;

/// Floor applied to group coefficients before they are used.
///
/// A group whose counts are all zero fits to [`crate::glm::one_group`]'s empty
/// gene fallback, and edgeR clamps here so the subsequent solve cannot produce
/// an infinite coefficient from a finite design.
const MIN_COEF: f64 = -1e8;

/// Result of a one-way fit.
#[derive(Clone, Debug)]
pub struct OneWayFit {
    /// Coefficients, row-major `n_genes * n_coef`.
    ///
    /// Expressed in the supplied design's parametrisation, not as group means,
    /// unless the design was already a group indicator.
    pub coefficients: Vec<f64>,
    /// Fitted means, row-major `n_genes * n_samples`.
    pub fitted: Vec<f64>,
}

/// Whether a design is a plain group indicator.
///
/// edgeR skips the rotation entirely in that case, since the group means already
/// are the coefficients. The test counts entries rather than checking structure,
/// matching edgeR: exactly one 1 per row and zeros everywhere else.
///
/// ### Params
///
/// * `unique_rows` - One representative design row per group, row-major
/// * `n_groups` - Number of groups
///
/// ### Returns
///
/// `true` when no rotation is needed.
fn is_group_indicator(unique_rows: &[f64], n_groups: usize) -> bool {
    let ones = unique_rows.iter().filter(|v| **v == 1.0).count();
    let zeros = unique_rows.iter().filter(|v| **v == 0.0).count();
    ones == n_groups && zeros == (n_groups - 1) * n_groups
}

/// Fits genewise negative binomial GLMs for a one-way layout.
///
/// ### Params
///
/// * `counts` - Counts, row-major `n_genes * n_samples`
/// * `n_genes` - Number of genes
/// * `n_samples` - Number of samples
/// * `design` - Design matrix, row-major `n_samples * n_coef`
/// * `n_coef` - Number of coefficients, which must equal the number of groups
/// * `group` - Optional explicit group per sample. When absent the groups are
///   derived from the design with [`design_as_factor`].
/// * `dispersion` - Dispersion, recycled over genes and samples
/// * `offset` - Log-scale offsets, recycled over genes and samples
/// * `weights` - Optional observation weights
/// * `params` - Tuning knobs for the per-group fits
///
/// ### Returns
///
/// Coefficients and fitted means, or [`EdgeErrors`] if the shapes disagree or
/// the design is not equivalent to a one-way layout.
#[allow(clippy::too_many_arguments)]
pub fn mglm_one_way<T: EdgeFloat>(
    counts: &[T],
    n_genes: usize,
    n_samples: usize,
    design: &[f64],
    n_coef: usize,
    group: Option<&[usize]>,
    dispersion: &Recycled<f64>,
    offset: &Recycled<f64>,
    weights: Option<&Recycled<f64>>,
    params: Option<OneGroupParams>,
) -> Result<OneWayFit, EdgeErrors> {
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
    if design.len() != n_samples * n_coef {
        return Err(EdgeErrors::LengthMismatch {
            name: "design",
            expected: n_samples * n_coef,
            got: design.len(),
        });
    }
    dispersion.validate(n_genes, n_samples)?;
    check_dispersion(dispersion)?;
    offset.validate(n_genes, n_samples)?;
    if let Some(w) = weights {
        w.validate(n_genes, n_samples)?;
    }

    let labels: Vec<usize> = match group {
        Some(g) => {
            if g.len() != n_samples {
                return Err(EdgeErrors::LengthMismatch {
                    name: "group",
                    expected: n_samples,
                    got: g.len(),
                });
            }
            g.to_vec()
        }
        None => design_as_factor(design, n_samples, n_coef)?.0,
    };

    let n_groups = labels.iter().copied().max().map_or(0, |m| m + 1);
    if n_groups != n_coef {
        return Err(EdgeErrors::InvalidArgument(format!(
            "design is not equivalent to a one-way layout: {n_coef} coefficients but {n_groups} distinct groups"
        )));
    }

    // Sample indices belonging to each group.
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); n_groups];
    for (sample, &label) in labels.iter().enumerate() {
        members[label].push(sample);
    }
    if let Some(empty) = members.iter().position(|m| m.is_empty()) {
        return Err(EdgeErrors::InvalidArgument(format!(
            "group {empty} has no samples"
        )));
    }

    // One representative design row per group.
    let mut unique_rows = Vec::with_capacity(n_groups * n_coef);
    for group_members in &members {
        let first = group_members[0];
        unique_rows.extend_from_slice(&design[first * n_coef..(first + 1) * n_coef]);
    }
    let needs_rotation = !is_group_indicator(&unique_rows, n_groups);

    // Factorise the rotation once, outside the per-gene loop. It is the same
    // `n_groups` by `n_groups` system for every gene.
    let rotation = needs_rotation
        .then(|| MatRef::from_row_major_slice(&unique_rows, n_groups, n_coef).partial_piv_lu());

    let mut coefficients = vec![0.0; n_genes * n_coef];
    let mut fitted = vec![0.0; n_genes * n_samples];

    coefficients
        .par_chunks_mut(n_coef)
        .zip(fitted.par_chunks_mut(n_samples))
        .enumerate()
        .for_each(|(gene, (beta, fit))| {
            let disp_row = dispersion.row(gene, n_samples);
            let offset_row = offset.row(gene, n_samples);
            let weight_row = weights.map(|w| w.row(gene, n_samples));
            let y = &counts[gene * n_samples..(gene + 1) * n_samples];

            // One intercept per group.
            let mut group_means = vec![0.0; n_groups];
            for (g, group_members) in members.iter().enumerate() {
                let start = initial_coefficient(y, group_members, offset_row, weight_row);
                group_means[g] = fit_one_gene(
                    y,
                    group_members,
                    disp_row,
                    offset_row,
                    weight_row,
                    start,
                    &params,
                )
                .max(MIN_COEF);
            }

            // Fitted values follow from the group means directly, before any
            // rotation: rotating and re-expanding would only lose precision.
            for (g, group_members) in members.iter().enumerate() {
                for &sample in group_members {
                    let eta =
                        (group_means[g] + offset_row.get(sample)).clamp(-ETA_CLAMP, ETA_CLAMP);
                    fit[sample] = eta.exp();
                }
            }

            match &rotation {
                Some(lu) => {
                    let rhs = MatRef::from_column_major_slice(&group_means, n_groups, 1);
                    let solved = lu.solve(rhs);
                    for (k, slot) in beta.iter_mut().enumerate() {
                        *slot = solved[(k, 0)];
                    }
                }
                None => beta.copy_from_slice(&group_means),
            }
        });

    Ok(OneWayFit {
        coefficients,
        fitted,
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Two groups of three, one gene up and one flat.
    fn fixture() -> (Vec<f64>, usize, usize) {
        let counts = vec![
            10.0, 12.0, 11.0, 40.0, 44.0, 38.0, //
            50.0, 48.0, 52.0, 49.0, 51.0, 50.0, //
        ];
        (counts, 2, 6)
    }

    /// Treatment contrast design: intercept plus a group effect. Not an
    /// indicator, so the rotation runs.
    fn treatment_design() -> Vec<f64> {
        vec![
            1.0, 0.0, //
            1.0, 0.0, //
            1.0, 0.0, //
            1.0, 1.0, //
            1.0, 1.0, //
            1.0, 1.0, //
        ]
    }

    /// Group indicator design: one column per group, no intercept.
    fn indicator_design() -> Vec<f64> {
        vec![
            1.0, 0.0, //
            1.0, 0.0, //
            1.0, 0.0, //
            0.0, 1.0, //
            0.0, 1.0, //
            0.0, 1.0, //
        ]
    }

    #[test]
    fn test_indicator_design_gives_group_means_directly() {
        let (counts, n_genes, n_samples) = fixture();
        let fit = mglm_one_way(
            &counts,
            n_genes,
            n_samples,
            &indicator_design(),
            2,
            None,
            &Recycled::scalar(0.1),
            &Recycled::scalar(0.0),
            None,
            None,
        )
        .unwrap();

        let mean_a: f64 = (10.0 + 12.0 + 11.0) / 3.0;
        let mean_b: f64 = (40.0 + 44.0 + 38.0) / 3.0;
        assert_relative_eq!(fit.coefficients[0], mean_a.ln(), max_relative = 1e-9);
        assert_relative_eq!(fit.coefficients[1], mean_b.ln(), max_relative = 1e-9);
    }

    #[test]
    fn test_treatment_design_is_rotated_into_contrast_form() {
        let (counts, n_genes, n_samples) = fixture();
        let fit = mglm_one_way(
            &counts,
            n_genes,
            n_samples,
            &treatment_design(),
            2,
            None,
            &Recycled::scalar(0.1),
            &Recycled::scalar(0.0),
            None,
            None,
        )
        .unwrap();

        let mean_a: f64 = (10.0 + 12.0 + 11.0) / 3.0;
        let mean_b: f64 = (40.0 + 44.0 + 38.0) / 3.0;
        // Intercept is group A, the second coefficient is the log ratio.
        assert_relative_eq!(fit.coefficients[0], mean_a.ln(), max_relative = 1e-9);
        assert_relative_eq!(
            fit.coefficients[1],
            mean_b.ln() - mean_a.ln(),
            max_relative = 1e-9
        );
    }

    /// The fitted values do not depend on the parametrisation, only the
    /// coefficients do.
    #[test]
    fn test_fitted_values_are_parametrisation_independent() {
        let (counts, n_genes, n_samples) = fixture();
        let indicator = mglm_one_way(
            &counts,
            n_genes,
            n_samples,
            &indicator_design(),
            2,
            None,
            &Recycled::scalar(0.1),
            &Recycled::scalar(0.0),
            None,
            None,
        )
        .unwrap();
        let treatment = mglm_one_way(
            &counts,
            n_genes,
            n_samples,
            &treatment_design(),
            2,
            None,
            &Recycled::scalar(0.1),
            &Recycled::scalar(0.0),
            None,
            None,
        )
        .unwrap();
        for (a, b) in indicator.fitted.iter().zip(treatment.fitted.iter()) {
            assert_relative_eq!(a, b, max_relative = 1e-12);
        }
    }

    /// The one-way fit and the general Levenberg fit describe the same model, so
    /// they must land on the same answer. This is the check that matters: it
    /// pins the fast path against the general one.
    #[test]
    fn test_agrees_with_the_levenberg_fit() {
        use crate::glm::levenberg::mglm_levenberg;

        let (counts, n_genes, n_samples) = fixture();
        let design = treatment_design();
        let offset = Recycled::by_sample(vec![
            1e6_f64.ln(),
            1.2e6_f64.ln(),
            0.9e6_f64.ln(),
            1.1e6_f64.ln(),
            1e6_f64.ln(),
            1.3e6_f64.ln(),
        ]);

        let one_way = mglm_one_way(
            &counts,
            n_genes,
            n_samples,
            &design,
            2,
            None,
            &Recycled::scalar(0.15),
            &offset,
            None,
            None,
        )
        .unwrap();

        let general = mglm_levenberg(
            &counts,
            n_genes,
            n_samples,
            &design,
            2,
            &Recycled::scalar(0.15),
            &offset,
            None,
            None,
            None,
        )
        .unwrap();

        for (a, b) in one_way.coefficients.iter().zip(general.coefficients.iter()) {
            assert_relative_eq!(a, b, epsilon = 1e-6, max_relative = 1e-5);
        }
        for (a, b) in one_way.fitted.iter().zip(general.fitted.iter()) {
            assert_relative_eq!(a, b, max_relative = 1e-5);
        }
    }

    #[test]
    fn test_explicit_group_overrides_the_design() {
        let (counts, n_genes, n_samples) = fixture();
        let groups = vec![0, 0, 0, 1, 1, 1];
        let fit = mglm_one_way(
            &counts,
            n_genes,
            n_samples,
            &indicator_design(),
            2,
            Some(&groups),
            &Recycled::scalar(0.1),
            &Recycled::scalar(0.0),
            None,
            None,
        )
        .unwrap();
        let mean_a: f64 = (10.0 + 12.0 + 11.0) / 3.0;
        assert_relative_eq!(fit.coefficients[0], mean_a.ln(), max_relative = 1e-9);
    }

    #[test]
    fn test_rejects_a_design_that_is_not_one_way() {
        let (counts, n_genes, n_samples) = fixture();
        // A continuous covariate gives six groups for two coefficients.
        let design = vec![1.0, 0.1, 1.0, 0.4, 1.0, 0.2, 1.0, 0.9, 1.0, 0.7, 1.0, 0.8];
        let err = mglm_one_way(
            &counts,
            n_genes,
            n_samples,
            &design,
            2,
            None,
            &Recycled::scalar(0.1),
            &Recycled::scalar(0.0),
            None,
            None,
        )
        .unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));
    }

    #[test]
    fn test_is_group_indicator_recognises_both_forms() {
        assert!(is_group_indicator(&[1.0, 0.0, 0.0, 1.0], 2));
        assert!(!is_group_indicator(&[1.0, 0.0, 1.0, 1.0], 2));
    }

    #[test]
    fn test_rejects_a_negative_dispersion() {
        let (counts, n_genes, n_samples) = fixture();
        let err = mglm_one_way(
            &counts,
            n_genes,
            n_samples,
            &treatment_design(),
            2,
            None,
            &Recycled::scalar(-0.1),
            &Recycled::scalar(0.0),
            None,
            None,
        )
        .unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidDispersion(_)));
    }

    #[test]
    fn test_rejects_a_shape_mismatch() {
        let err = mglm_one_way(
            &[1.0_f64; 5],
            2,
            3,
            &[1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            2,
            None,
            &Recycled::scalar(0.1),
            &Recycled::scalar(0.0),
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
