//! `filterByExpr`: dropping genes with too little expression to model.
//!
//! A gene is kept when it clears two cutoffs at once: it reaches a CPM of
//! `min_count` counts in the median-sized library, in at least as many samples
//! as the smallest group has, and its total across all samples reaches
//! `min_total_count`.

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::core::expression::{PER_MILLION, check_counts, column_sums, cpm};
use crate::numeric::stats::quantile_type7;
use crate::prelude::*;
use crate::utils::design::{hat_diagonal, matrix_rank};

////////////
// Consts //
////////////

/// Slack allowed on both cutoffs.
///
/// edgeR compares against `cutoff - tol` so that a gene sitting exactly on a
/// threshold, which for the total-count rule is the common case since counts are
/// integers, is not dropped by a rounding error in the CPM.
const CUTOFF_TOLERANCE: f64 = 1e-14;

//////////////////
// FilterParams //
//////////////////

/// Tuning knobs for [`filter_by_expr`].
#[derive(Clone, Copy, Debug)]
pub struct FilterParams {
    /// Counts a gene must reach in the median-sized library, expressed on the
    /// raw count scale and converted to a CPM cutoff internally.
    pub min_count: f64,
    /// Total count across all samples a gene must reach.
    pub min_total_count: f64,
    /// Group size past which the sample requirement stops growing one for one.
    pub large_n: f64,
    /// Fraction of the excess over [`FilterParams::large_n`] that still counts.
    ///
    /// A very large experiment does not need a gene expressed in *every*
    /// replicate of the smallest group, so the requirement grows at this rate
    /// beyond `large_n` rather than linearly.
    pub min_prop: f64,
}

impl FilterParams {
    /// Builds a parameter set.
    ///
    /// No validation happens here; [`filter_by_expr`] checks the domains so that
    /// a bad value surfaces at the call that uses it rather than at
    /// construction.
    ///
    /// ### Params
    ///
    /// * `min_count` - Count a gene must reach in the median-sized library
    /// * `min_total_count` - Total count across all samples
    /// * `large_n` - Group size past which the requirement is damped
    /// * `min_prop` - Damping fraction beyond `large_n`, in `[0, 1]`
    ///
    /// ### Returns
    ///
    /// The parameter set.
    pub fn new(min_count: f64, min_total_count: f64, large_n: f64, min_prop: f64) -> Self {
        Self {
            min_count,
            min_total_count,
            large_n,
            min_prop,
        }
    }
}

impl Default for FilterParams {
    /// edgeR's defaults.
    fn default() -> Self {
        Self {
            min_count: 10.0,
            min_total_count: 15.0,
            large_n: 10.0,
            min_prop: 0.7,
        }
    }
}

/// Rejects parameters outside their domain.
///
/// edgeR does no checking here, but a negative `min_prop` or a non-finite
/// `min_count` silently turns the filter into something other than a filter, so
/// it is worth catching.
///
/// ### Params
///
/// * `params` - The resolved parameter set
///
/// ### Returns
///
/// `Ok(())`, or [`EdgeErrors::InvalidArgument`] naming the offending knob.
fn check_params(params: &FilterParams) -> Result<(), EdgeErrors> {
    for (name, value) in [
        ("min_count", params.min_count),
        ("min_total_count", params.min_total_count),
        ("large_n", params.large_n),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(EdgeErrors::InvalidArgument(format!(
                "{name} must be non-negative and finite, got {value}"
            )));
        }
    }
    if !(0.0..=1.0).contains(&params.min_prop) {
        return Err(EdgeErrors::InvalidArgument(format!(
            "min_prop must lie in [0, 1], got {}",
            params.min_prop
        )));
    }
    Ok(())
}

////////////////////////
// Minimum group size //
////////////////////////

/// How many samples a gene must be expressed in, before the `large_n` damping.
///
/// edgeR's precedence in `filterByExpr.default`, in order:
///
/// 1. `group`, when supplied: the size of the smallest non-empty group. A
///    supplied `group` wins even if a `design` is also given, because edgeR
///    tests `is.null(group)` first.
/// 2. `design`, otherwise: `1 / max(hat(design))`, the reciprocal of the
///    largest leverage. For a one-way layout that is exactly the smallest group
///    size, and for anything else it is the natural continuous generalisation.
/// 3. Neither: every sample is treated as one group, so `n_samples`.
///
/// ### Params
///
/// * `n_samples` - Number of samples
/// * `group` - Group label per sample. Labels need not be contiguous.
/// * `design` - Row-major design matrix and its column count
///
/// ### Returns
///
/// The minimum effective group size, or [`EdgeErrors`] if `group` is the wrong
/// length or `design` is the wrong shape or rank deficient.
fn resolve_min_sample_size(
    n_samples: usize,
    group: Option<&[usize]>,
    design: Option<(&[f64], usize)>,
) -> Result<f64, EdgeErrors> {
    if let Some(labels) = group {
        if labels.len() != n_samples {
            return Err(EdgeErrors::LengthMismatch {
                name: "group",
                expected: n_samples,
                got: labels.len(),
            });
        }
        let mut sizes: FxHashMap<usize, usize> = FxHashMap::default();
        for label in labels {
            *sizes.entry(*label).or_insert(0) += 1;
        }
        // Every observed label has at least one member, so the "n > 0" filter
        // edgeR applies to `tabulate` output is already satisfied.
        let smallest = sizes.values().min().copied().unwrap_or(n_samples);
        return Ok(smallest as f64);
    }

    match design {
        Some((values, n_coef)) => {
            let leverage = design_leverage(values, n_samples, n_coef)?;
            let largest = leverage.iter().fold(0.0_f64, |acc, v| acc.max(*v));
            if !largest.is_finite() || largest <= 0.0 {
                return Err(EdgeErrors::InvalidArgument(
                    "design has no non-zero leverage, so the minimum group size is undefined."
                        .to_string(),
                ));
            }
            Ok(1.0 / largest)
        }
        None => Ok(n_samples as f64),
    }
}

/// Leverages of a design, matching R's `hat(design)`.
///
/// `hat` prepends a column of ones, then takes the row sums of squares of the
/// first `rank` columns of `Q`. Prepending only matters when the intercept is
/// not already in the column space, so this checks which case it is and hands
/// the full rank matrix to [`hat_diagonal`]: a rank deficient matrix would give
/// a thin `Q` with arbitrary extra columns and inflate every leverage.
///
/// ### Params
///
/// * `design` - Row-major design matrix, `n_samples * n_coef`
/// * `n_samples` - Number of samples, that is, rows
/// * `n_coef` - Number of coefficients, that is, columns
///
/// ### Returns
///
/// One leverage per sample, or [`EdgeErrors::DesignNotFullRank`] if the design
/// itself is rank deficient.
fn design_leverage(
    design: &[f64],
    n_samples: usize,
    n_coef: usize,
) -> Result<Vec<f64>, EdgeErrors> {
    if design.len() != n_samples * n_coef {
        return Err(EdgeErrors::LengthMismatch {
            name: "design",
            expected: n_samples * n_coef,
            got: design.len(),
        });
    }
    if n_coef == 0 {
        return Err(EdgeErrors::MustBePositive("design dimensions".to_string()));
    }

    let mut augmented = Vec::with_capacity(n_samples * (n_coef + 1));
    for row in design.chunks_exact(n_coef) {
        augmented.push(1.0);
        augmented.extend_from_slice(row);
    }

    if matrix_rank(&augmented, n_samples, n_coef + 1)? == n_coef + 1 {
        return hat_diagonal(&augmented, n_samples, n_coef + 1);
    }

    let rank = matrix_rank(design, n_samples, n_coef)?;
    if rank != n_coef {
        return Err(EdgeErrors::DesignNotFullRank {
            n_cols: n_coef,
            rank,
        });
    }
    hat_diagonal(design, n_samples, n_coef)
}

///////////////
// Front end //
///////////////

/// Flags the genes worth keeping, as edgeR's `filterByExpr`.
///
/// ### Params
///
/// * `counts` - Row-major counts, `n_genes * n_samples`
/// * `n_genes` - Number of genes
/// * `n_samples` - Number of samples
/// * `lib_size` - Library size per sample, normally
///   [`DgeList::norm_lib_sizes`](crate::core::dgelist::DgeList::norm_lib_sizes).
///   `None` uses the column sums.
/// * `group` - Group label per sample. Takes precedence over `design`.
/// * `design` - Row-major design matrix and its column count, used only when
///   `group` is absent.
/// * `params` - Tuning knobs, or [`FilterParams::default`]
///
/// ### Returns
///
/// One keep flag per gene, or [`EdgeErrors`] if a shape disagrees, a library
/// size is not positive, the design is rank deficient, or a parameter is
/// outside its domain.
pub fn filter_by_expr<T: EdgeFloat>(
    counts: &[T],
    n_genes: usize,
    n_samples: usize,
    lib_size: Option<&[f64]>,
    group: Option<&[usize]>,
    design: Option<(&[f64], usize)>,
    params: Option<FilterParams>,
) -> Result<Vec<bool>, EdgeErrors> {
    let params = params.unwrap_or_default();
    check_counts(counts, n_genes, n_samples)?;
    check_params(&params)?;

    let lib_size = match lib_size {
        Some(l) => {
            if l.len() != n_samples {
                return Err(EdgeErrors::LengthMismatch {
                    name: "lib_size",
                    expected: n_samples,
                    got: l.len(),
                });
            }
            l.to_vec()
        }
        None => column_sums(counts, n_samples),
    };

    let mut min_sample_size = resolve_min_sample_size(n_samples, group, design)?;
    if min_sample_size > params.large_n {
        min_sample_size = params.large_n + (min_sample_size - params.large_n) * params.min_prop;
    }

    // The median library defines the cutoff, so one enormous sample does not
    // drag the threshold up for everything else.
    let median_lib_size = quantile_type7(&lib_size, 0.5)?;
    if !median_lib_size.is_finite() || median_lib_size <= 0.0 {
        return Err(EdgeErrors::InvalidArgument(format!(
            "median library size must be positive, got {median_lib_size}"
        )));
    }
    let cpm_cutoff = params.min_count / median_lib_size * PER_MILLION;

    let cpm_values = cpm(
        counts,
        n_genes,
        n_samples,
        Some(&lib_size),
        None,
        false,
        0.0,
    )?;

    Ok(cpm_values
        .par_chunks_exact(n_samples)
        .zip(counts.par_chunks_exact(n_samples))
        .map(|(row, raw)| {
            let above = row.iter().filter(|v| **v >= cpm_cutoff).count() as f64;
            let total: f64 = raw.iter().map(|v| v.to_f64().unwrap_or(f64::NAN)).sum();
            above >= min_sample_size - CUTOFF_TOLERANCE
                && total >= params.min_total_count - CUTOFF_TOLERANCE
        })
        .collect())
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// 8 genes by 6 samples, the fixture every reference below was generated
    /// from:
    /// ```r
    /// y <- matrix(c(
    ///   0,0,0,0,0,0, 1,0,2,1,0,1, 10,12,11,40,44,38, 50,48,52,49,51,50,
    ///   2,0,5,1,3,0, 200,180,220,5,3,4, 6,7,5,6,8,7, 1,1,1,1,200,200),
    ///   nrow = 8, byrow = TRUE)
    /// ```
    /// Column sums are `270 248 296 103 309 300`, so the median library is 283
    /// and the default CPM cutoff is `10 / 283 * 1e6 = 35335.7`.
    const COUNTS: [f64; 48] = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
        1.0, 0.0, 2.0, 1.0, 0.0, 1.0, //
        10.0, 12.0, 11.0, 40.0, 44.0, 38.0, //
        50.0, 48.0, 52.0, 49.0, 51.0, 50.0, //
        2.0, 0.0, 5.0, 1.0, 3.0, 0.0, //
        200.0, 180.0, 220.0, 5.0, 3.0, 4.0, //
        6.0, 7.0, 5.0, 6.0, 8.0, 7.0, //
        1.0, 1.0, 1.0, 1.0, 200.0, 200.0, //
    ];

    /// `g <- c(1,1,1,2,2,2)`
    const GROUP: [usize; 6] = [0, 0, 0, 1, 1, 1];

    /// `model.matrix(~ factor(g))`, row-major.
    const DESIGN: [f64; 12] = [
        1.0, 0.0, //
        1.0, 0.0, //
        1.0, 0.0, //
        1.0, 1.0, //
        1.0, 1.0, //
        1.0, 1.0, //
    ];

    /// `cat(filterByExpr(y, group = g))`
    #[test]
    fn test_filter_by_group_matches_edger() {
        let keep = filter_by_expr(&COUNTS, 8, 6, None, Some(&GROUP), None, None).unwrap();
        assert_eq!(
            keep,
            vec![false, false, true, true, false, true, false, false]
        );
    }

    /// `cat(filterByExpr(y, design = model.matrix(~ factor(g))))`
    #[test]
    fn test_filter_by_design_matches_edger() {
        let keep = filter_by_expr(&COUNTS, 8, 6, None, None, Some((&DESIGN, 2)), None).unwrap();
        assert_eq!(
            keep,
            vec![false, false, true, true, false, true, false, false]
        );
    }

    /// `cat(filterByExpr(y))`. Every sample is one group, so a gene has to clear
    /// the CPM cutoff in all six.
    #[test]
    fn test_filter_without_group_or_design_matches_edger() {
        let keep = filter_by_expr(&COUNTS, 8, 6, None, None, None, None).unwrap();
        assert_eq!(
            keep,
            vec![false, false, true, true, false, false, false, false]
        );
    }

    /// A continuous design with no intercept column of its own:
    /// `d2 <- cbind(c(0.1,0.2,0.3,0.4,0.5,0.9))`, for which
    /// `1 / max(hat(d2))` is 1.263158 and `cat(filterByExpr(y, design = d2))`
    /// keeps gene 8, which the three-per-group `group` call drops.
    #[test]
    fn test_filter_by_a_design_without_an_intercept_matches_edger() {
        let design = [0.1, 0.2, 0.3, 0.4, 0.5, 0.9];
        let min_size = resolve_min_sample_size(6, None, Some((&design, 1))).unwrap();
        assert_relative_eq!(min_size, 24.0 / 19.0, max_relative = 1e-12);

        let keep = filter_by_expr(&COUNTS, 8, 6, None, None, Some((&design, 1)), None).unwrap();
        assert_eq!(
            keep,
            vec![false, false, true, true, false, true, false, true]
        );
    }

    /// `cat(filterByExpr(y, group = g, lib.size = ls))` with
    /// `ls <- c(1e6, 1.2e6, 0.9e6, 1.1e6, 1e6, 1.3e6)`.
    #[test]
    fn test_filter_with_explicit_library_sizes_matches_edger() {
        let lib = [1e6, 1.2e6, 0.9e6, 1.1e6, 1e6, 1.3e6];
        let keep = filter_by_expr(&COUNTS, 8, 6, Some(&lib), Some(&GROUP), None, None).unwrap();
        assert_eq!(
            keep,
            vec![false, false, true, true, false, true, false, false]
        );
    }

    /// `cat(filterByExpr(y, group = g, min.count = 1, min.total.count = 5))`
    #[test]
    fn test_filter_honours_the_parameters() {
        let params = FilterParams::new(1.0, 5.0, 10.0, 0.7);
        let keep = filter_by_expr(&COUNTS, 8, 6, None, Some(&GROUP), None, Some(params)).unwrap();
        assert_eq!(keep, vec![false, true, true, true, true, true, true, true]);
    }

    /// `cat(filterByExpr(y, group = c(1,1,1,1,2,2)))`. The smallest group has
    /// two members, not three.
    #[test]
    fn test_filter_uses_the_smallest_group() {
        let group = [0, 0, 0, 0, 1, 1];
        let keep = filter_by_expr(&COUNTS, 8, 6, None, Some(&group), None, None).unwrap();
        assert_eq!(
            keep,
            vec![false, false, true, true, false, true, false, true]
        );
    }

    /// `group` is tested before `design` in edgeR, so it wins when both are
    /// given.
    #[test]
    fn test_group_takes_precedence_over_design() {
        let design = [0.1, 0.2, 0.3, 0.4, 0.5, 0.9];
        let with_both =
            filter_by_expr(&COUNTS, 8, 6, None, Some(&GROUP), Some((&design, 1)), None).unwrap();
        let with_group = filter_by_expr(&COUNTS, 8, 6, None, Some(&GROUP), None, None).unwrap();
        assert_eq!(with_both, with_group);
    }

    /// Group labels need not be contiguous or start at zero; only the counts
    /// matter.
    #[test]
    fn test_group_labels_need_not_be_contiguous() {
        let sparse = [7, 7, 7, 42, 42, 42];
        let keep = filter_by_expr(&COUNTS, 8, 6, None, Some(&sparse), None, None).unwrap();
        let want = filter_by_expr(&COUNTS, 8, 6, None, Some(&GROUP), None, None).unwrap();
        assert_eq!(keep, want);
    }

    /// Past `large_n` the requirement grows at `min_prop` per extra replicate:
    /// 16 samples in one group become `10 + (16 - 10) * 0.7 = 14.2`, so a gene
    /// that misses one sample is still kept.
    ///
    /// ```r
    /// y <- rbind(c(0, rep(1, 15)), rep(999, 16))
    /// cat(filterByExpr(y, min.count = 0.5, min.total.count = 5))
    /// ```
    #[test]
    fn test_large_group_sizes_are_damped() {
        // Undamped the requirement is the sample count itself.
        assert_relative_eq!(
            resolve_min_sample_size(16, None, None).unwrap(),
            16.0,
            max_relative = 1e-12
        );

        let mut counts = vec![1.0_f64; 32];
        counts[0] = 0.0;
        for v in counts[16..].iter_mut() {
            *v = 999.0;
        }

        let params = FilterParams::new(0.5, 5.0, 10.0, 0.7);
        let keep = filter_by_expr(&counts, 2, 16, None, None, None, Some(params)).unwrap();
        assert_eq!(keep, vec![true, true]);

        // Without the damping the first gene would need all 16 samples.
        let undamped = FilterParams::new(0.5, 5.0, 16.0, 0.7);
        let keep = filter_by_expr(&counts, 2, 16, None, None, None, Some(undamped)).unwrap();
        assert_eq!(keep, vec![false, true]);
    }

    /// The three-per-group design gives a leverage of 1/3 everywhere, so the
    /// design branch recovers the group size exactly.
    #[test]
    fn test_design_branch_recovers_the_group_size() {
        let from_design = resolve_min_sample_size(6, None, Some((&DESIGN, 2))).unwrap();
        assert_relative_eq!(from_design, 3.0, max_relative = 1e-12);
    }

    /// `f32` counts must give the same flags.
    #[test]
    fn test_is_generic_over_the_count_type() {
        let counts32: Vec<f32> = COUNTS.iter().map(|v| *v as f32).collect();
        let keep = filter_by_expr(&counts32, 8, 6, None, Some(&GROUP), None, None).unwrap();
        let want = filter_by_expr(&COUNTS, 8, 6, None, Some(&GROUP), None, None).unwrap();
        assert_eq!(keep, want);
    }

    #[test]
    fn test_rejects_empty_counts() {
        let err = filter_by_expr::<f64>(&[], 0, 6, None, None, None, None).unwrap_err();
        assert!(matches!(err, EdgeErrors::EmptyCounts { .. }));
    }

    #[test]
    fn test_rejects_a_shape_mismatch() {
        let err = filter_by_expr(&COUNTS[..24], 8, 6, None, None, None, None).unwrap_err();
        assert!(matches!(
            err,
            EdgeErrors::LengthMismatch { name: "counts", .. }
        ));
    }

    #[test]
    fn test_rejects_a_group_of_the_wrong_length() {
        let err = filter_by_expr(&COUNTS, 8, 6, None, Some(&[0, 0, 1]), None, None).unwrap_err();
        assert!(matches!(
            err,
            EdgeErrors::LengthMismatch { name: "group", .. }
        ));
    }

    #[test]
    fn test_rejects_library_sizes_of_the_wrong_length() {
        let err = filter_by_expr(&COUNTS, 8, 6, Some(&[1e6; 3]), None, None, None).unwrap_err();
        assert!(matches!(
            err,
            EdgeErrors::LengthMismatch {
                name: "lib_size",
                ..
            }
        ));
    }

    #[test]
    fn test_rejects_a_design_of_the_wrong_shape() {
        let err = filter_by_expr(&COUNTS, 8, 6, None, None, Some((&DESIGN, 3)), None).unwrap_err();
        assert!(matches!(
            err,
            EdgeErrors::LengthMismatch { name: "design", .. }
        ));
    }

    /// A design whose columns are collinear even after the intercept is added
    /// has no well-defined leverage.
    #[test]
    fn test_rejects_a_rank_deficient_design() {
        let design = [
            1.0, 2.0, //
            1.0, 2.0, //
            2.0, 4.0, //
            2.0, 4.0, //
            3.0, 6.0, //
            3.0, 6.0, //
        ];
        let err = filter_by_expr(&COUNTS, 8, 6, None, None, Some((&design, 2)), None).unwrap_err();
        assert!(matches!(err, EdgeErrors::DesignNotFullRank { .. }));
    }

    #[test]
    fn test_rejects_a_non_positive_library_size() {
        let err = filter_by_expr(
            &COUNTS,
            8,
            6,
            Some(&[1e6, 1e6, 1e6, 1e6, 1e6, 0.0]),
            Some(&GROUP),
            None,
            None,
        )
        .unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));
    }

    #[test]
    fn test_rejects_bad_parameters() {
        let bad = FilterParams::new(-1.0, 15.0, 10.0, 0.7);
        let err = filter_by_expr(&COUNTS, 8, 6, None, None, None, Some(bad)).unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));

        let bad = FilterParams::new(10.0, 15.0, 10.0, 1.5);
        let err = filter_by_expr(&COUNTS, 8, 6, None, None, None, Some(bad)).unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));
    }
}
