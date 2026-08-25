//! Design matrix inspection.
//!
//! Rank, estimability, leverage, and the factor construction that decides
//! whether a GLM fit can take the cheap one-way path.
//!
//! Design matrices are `f64` throughout the crate, not generic over
//! [`crate::prelude::EdgeFloat`]. They are `n_samples` by `n_coef`, so tens of
//! values against count matrices of hundreds of millions, and making them
//! generic would buy no memory while putting a rank decision at the mercy of
//! `f32` rounding. edgeR is `f64` here and so is this.
//!
//! Matrices are row-major: row `i` is `design[i * n_coef..(i + 1) * n_coef]`.

use faer::MatRef;
use rustc_hash::FxHashMap;

use crate::prelude::*;

////////////
// Consts //
////////////

/// Multiplier edgeR uses to hash a design row down to one number.
///
/// `designAsFactor` forms `sum_j design[i, j] * z^j` with `z = (e + pi) / 5`
/// and treats rows sharing a value as the same group. The constant is arbitrary
/// but must match edgeR exactly: it decides whether `glm_fit` takes the one-way
/// path or the Levenberg path, and the two do not agree to the last digit.
const FACTOR_HASH_BASE: f64 = (std::f64::consts::E + std::f64::consts::PI) / 5.0;

/// limma's defaults for [`choose_lowess_span`]: `small_n`, `min_span`, `power`.
pub const LIMMA_LOWESS_DEFAULTS: (usize, f64, f64) = (50, 0.3, 1.0 / 3.0);

/////////////
// Helpers //
/////////////

/// Wraps a row-major slice as a faer matrix, checking the shape first.
///
/// ### Params
///
/// * `design` - Row-major values
/// * `n_rows` - Number of rows
/// * `n_cols` - Number of columns
///
/// ### Returns
///
/// The borrowed matrix, or [`EdgeErrors::LengthMismatch`] if the slice does not
/// match the stated shape.
fn as_matrix(design: &[f64], n_rows: usize, n_cols: usize) -> Result<MatRef<'_, f64>, EdgeErrors> {
    validate_shape(design.len(), n_rows, n_cols)?;
    Ok(MatRef::from_row_major_slice(design, n_rows, n_cols))
}

/// Checks that a row-major design's length matches its stated shape, and that
/// the shape itself is non-degenerate.
///
/// ### Params
///
/// * `len` - Length of the row-major slice
/// * `n_rows` - Number of rows
/// * `n_cols` - Number of columns
///
/// ### Returns
///
/// `Ok(())` when `len == n_rows * n_cols` and both dimensions are positive,
/// otherwise [`EdgeErrors::LengthMismatch`] or [`EdgeErrors::MustBePositive`].
fn validate_shape(len: usize, n_rows: usize, n_cols: usize) -> Result<(), EdgeErrors> {
    if len != n_rows * n_cols {
        return Err(EdgeErrors::LengthMismatch {
            name: "design",
            expected: n_rows * n_cols,
            got: len,
        });
    }
    if n_rows == 0 || n_cols == 0 {
        return Err(EdgeErrors::MustBePositive("design dimensions".to_string()));
    }
    Ok(())
}

///////////////
// Front end //
///////////////

/// Numerical rank of a design matrix.
///
/// Counts singular values above `max(S) * max(n_rows, n_cols) * eps`, which is
/// numpy's `matrix_rank` rule and therefore what edgePython compares against.
///
/// ### Params
///
/// * `design` - Row-major design matrix
/// * `n_rows` - Number of samples
/// * `n_cols` - Number of coefficients
///
/// ### Returns
///
/// The rank, or [`EdgeErrors`] if the shape is wrong or the SVD fails.
pub fn matrix_rank(design: &[f64], n_rows: usize, n_cols: usize) -> Result<usize, EdgeErrors> {
    let mat = as_matrix(design, n_rows, n_cols)?;
    let svd = mat
        .thin_svd()
        .map_err(|e| EdgeErrors::SolveFailed(format!("SVD of the design matrix failed: {e:?}")))?;

    let s = svd.S();
    let values = s.column_vector();
    let largest = (0..values.nrows()).fold(0.0_f64, |acc, i| acc.max(values[i]));
    let tol = largest * n_rows.max(n_cols) as f64 * f64::EPSILON;

    Ok((0..values.nrows()).filter(|&i| values[i] > tol).count())
}

/// Whether every coefficient in the design is identifiable.
///
/// ### Params
///
/// * `design` - Row-major design matrix
/// * `n_rows` - Number of samples
/// * `n_cols` - Number of coefficients
///
/// ### Returns
///
/// `true` when the rank equals the column count. Port of limma's `is.fullrank`.
pub fn is_full_rank(design: &[f64], n_rows: usize, n_cols: usize) -> Result<bool, EdgeErrors> {
    Ok(matrix_rank(design, n_rows, n_cols)? == n_cols)
}

/// Coefficients that cannot be estimated from this design.
///
/// Factorises the design and flags the columns whose diagonal entry of `R` is
/// negligible. Port of limma's `nonEstimable`.
///
/// ### Params
///
/// * `design` - Row-major design matrix
/// * `n_rows` - Number of samples
/// * `n_cols` - Number of coefficients
///
/// ### Returns
///
/// `None` when every coefficient is estimable, otherwise the offending column
/// indices in increasing order.
pub fn non_estimable(
    design: &[f64],
    n_rows: usize,
    n_cols: usize,
) -> Result<Option<Vec<usize>>, EdgeErrors> {
    let mat = as_matrix(design, n_rows, n_cols)?;
    let qr = mat.qr();
    let r = qr.thin_R();

    let diag_len = r.nrows().min(r.ncols());
    let diag: Vec<f64> = (0..diag_len).map(|i| r[(i, i)].abs()).collect();
    if diag.is_empty() {
        return Ok(Some((0..n_cols).collect()));
    }

    let largest = diag.iter().fold(0.0_f64, |acc, &v| acc.max(v));
    let tol = largest * n_rows.max(n_cols) as f64 * f64::EPSILON;

    let flagged: Vec<usize> = diag
        .iter()
        .enumerate()
        .filter(|&(_, &v)| v < tol)
        .map(|(i, _)| i)
        .collect();

    if flagged.is_empty() {
        Ok(None)
    } else {
        Ok(Some(flagged))
    }
}

/// Leverage of each sample, that is, the diagonal of the hat matrix.
///
/// Computed as the row sums of squares of the thin `Q` factor, which avoids
/// forming the `n_rows` by `n_rows` hat matrix.
///
/// ### Params
///
/// * `design` - Row-major design matrix
/// * `n_rows` - Number of samples
/// * `n_cols` - Number of coefficients
///
/// ### Returns
///
/// One leverage per sample, each in `[0, 1]` for a full rank design.
pub fn hat_diagonal(design: &[f64], n_rows: usize, n_cols: usize) -> Result<Vec<f64>, EdgeErrors> {
    let mat = as_matrix(design, n_rows, n_cols)?;
    let q = mat.qr().compute_thin_Q();

    Ok((0..n_rows)
        .map(|i| (0..q.ncols()).map(|j| q[(i, j)] * q[(i, j)]).sum())
        .collect())
}

/// Groups samples by their design row.
///
/// Reproduces edgeR's `designAsFactor`: hash each row to a single number with
/// [`FACTOR_HASH_BASE`], then label the distinct values in ascending order.
/// When the number of groups equals the number of coefficients, `glm_fit` can
/// use the closed-form one-way fit instead of the iterative Levenberg one, so
/// this is on the hot path for every bulk analysis.
///
/// Rows are matched on the exact `f64` hash, as in edgeR. Two rows that differ
/// only by rounding therefore land in different groups, which is the intended
/// conservative behaviour: the fallback path is correct, just slower.
///
/// ### Params
///
/// * `design` - Row-major design matrix
/// * `n_rows` - Number of samples
/// * `n_cols` - Number of coefficients
///
/// ### Returns
///
/// A group index per sample and the number of distinct groups.
pub fn design_as_factor(
    design: &[f64],
    n_rows: usize,
    n_cols: usize,
) -> Result<(Vec<usize>, usize), EdgeErrors> {
    validate_shape(design.len(), n_rows, n_cols)?;

    let mut powers = Vec::with_capacity(n_cols);
    let mut power = 1.0_f64;
    for _ in 0..n_cols {
        powers.push(power);
        power *= FACTOR_HASH_BASE;
    }

    let hashes: Vec<f64> = design
        .chunks_exact(n_cols)
        .map(|row| row.iter().zip(powers.iter()).map(|(a, b)| a * b).sum())
        .collect();

    // Distinct hashes in ascending order, matching numpy's `unique`.
    let mut sorted: Vec<f64> = hashes.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    sorted.dedup_by(|a, b| a.to_bits() == b.to_bits());

    let level: FxHashMap<u64, usize> = sorted
        .iter()
        .enumerate()
        .map(|(i, v)| (v.to_bits(), i))
        .collect();

    let groups: Vec<usize> = hashes.iter().map(|v| level[&v.to_bits()]).collect();

    Ok((groups, sorted.len()))
}

////////////////////////////
// Contrast reformulation //
////////////////////////////

/// A design rewritten so that a contrast becomes a coefficient.
#[derive(Clone, Debug)]
pub struct ContrastDesign {
    /// The reformed design, row-major `n_rows * n_cols`. Same shape as the
    /// input and the same column span, just a different basis.
    pub design: Vec<f64>,
    /// Column indices holding the contrasts, in ascending order.
    pub coef: Vec<usize>,
}

/// Rewrites a design matrix so that a contrast becomes one of its coefficients.
///
/// A likelihood ratio test drops the tested columns and refits, so testing an
/// arbitrary contrast means first rotating the design until that contrast *is*
/// a column. Port of limma's `contrastAsCoef`, which edgeR's `glmLRT` and
/// `glmQLFTest` both go through whenever a contrast rather than a coefficient
/// index is supplied.
///
/// The rotation is `Q` from the QR factorisation of the contrast, followed by a
/// triangular solve against `R` so the new coefficient reads as the contrast
/// itself rather than a scaled version of it. Note that the sign of `Q` is not
/// pinned down by the factorisation, but the contrast columns come out
/// sign-invariant because the `R` solve carries the same sign; only the
/// nuisance columns can flip, and their span, hence the fit, is unchanged.
///
/// ### Params
///
/// * `design` - Row-major design, `n_rows * n_cols`
/// * `n_rows` - Number of samples
/// * `n_cols` - Number of coefficients
/// * `contrast` - Column-major contrast, `n_cols * n_contrasts`
/// * `n_contrasts` - Number of contrasts, usually one
/// * `first` - Place the contrasts in the leading columns rather than the
///   trailing ones. limma defaults to `true`; edgeR calls it with `false`.
///
/// ### Returns
///
/// The reformed design and the indices of its contrast columns, or
/// [`EdgeErrors`] if the shapes disagree, either design dimension is zero, or
/// the contrast is entirely zero.
pub fn contrast_as_coef(
    design: &[f64],
    n_rows: usize,
    n_cols: usize,
    contrast: &[f64],
    n_contrasts: usize,
    first: bool,
) -> Result<ContrastDesign, EdgeErrors> {
    validate_shape(design.len(), n_rows, n_cols)?;
    if contrast.len() != n_cols * n_contrasts {
        return Err(EdgeErrors::LengthMismatch {
            name: "contrast",
            expected: n_cols * n_contrasts,
            got: contrast.len(),
        });
    }
    if n_contrasts == 0 || n_contrasts > n_cols {
        return Err(EdgeErrors::InvalidArgument(format!(
            "expected between 1 and {n_cols} contrasts, got {n_contrasts}"
        )));
    }
    if contrast.iter().all(|v| *v == 0.0) {
        return Err(EdgeErrors::InvalidArgument(
            "contrast is entirely zero".to_string(),
        ));
    }

    let contrast_mat = MatRef::from_column_major_slice(contrast, n_cols, n_contrasts);
    let qr = contrast_mat.qr();
    let q = qr.compute_Q();
    let r = qr.thin_R();

    // designT = Q' X', of shape n_cols by n_rows.
    let mut designt = vec![0.0; n_cols * n_rows];
    for a in 0..n_cols {
        for sample in 0..n_rows {
            let mut acc = 0.0;
            for k in 0..n_cols {
                acc += q[(k, a)] * design[sample * n_cols + k];
            }
            designt[a * n_rows + sample] = acc;
        }
    }

    // Solve R z = designT[..n_contrasts, ..] so the contrast rows read as the
    // contrast rather than a multiple of it. R is upper triangular and tiny.
    for sample in 0..n_rows {
        for row in (0..n_contrasts).rev() {
            let mut acc = designt[row * n_rows + sample];
            for k in (row + 1)..n_contrasts {
                acc -= r[(row, k)] * designt[k * n_rows + sample];
            }
            let pivot = r[(row, row)];
            if pivot == 0.0 {
                return Err(EdgeErrors::InvalidArgument(
                    "contrast is rank deficient".to_string(),
                ));
            }
            designt[row * n_rows + sample] = acc / pivot;
        }
    }

    // Back to row-major, moving the contrast columns to the end unless asked
    // for them first.
    let order: Vec<usize> = if first {
        (0..n_cols).collect()
    } else {
        (n_contrasts..n_cols).chain(0..n_contrasts).collect()
    };
    let coef: Vec<usize> = if first {
        (0..n_contrasts).collect()
    } else {
        (n_cols - n_contrasts..n_cols).collect()
    };

    let mut out = vec![0.0; n_rows * n_cols];
    for sample in 0..n_rows {
        for (slot, &source) in order.iter().enumerate() {
            out[sample * n_cols + slot] = designt[source * n_rows + sample];
        }
    }

    Ok(ContrastDesign { design: out, coef })
}

/// Lowess span for a given number of observations.
///
/// Port of limma's `chooseLowessSpan`: wider windows for small experiments,
/// tapering towards `min_span` as the count grows.
///
/// limma's defaults are `small_n = 50`, `min_span = 0.3`, `power = 1/3`, and
/// [`LIMMA_LOWESS_DEFAULTS`] carries them. edgePython passes `25` and `0.2`
/// instead, which is a materially different span. See `UPSTREAM_DEVIATIONS.md`
/// entry 13.
///
/// ### Params
///
/// * `n` - Number of observations
/// * `small_n` - Count below which the span is 1
/// * `min_span` - Asymptotic span for large `n`
/// * `power` - Exponent controlling how fast the span tapers
///
/// ### Returns
///
/// A span in `(0, 1]`.
pub fn choose_lowess_span(n: usize, small_n: usize, min_span: f64, power: f64) -> f64 {
    if n <= small_n {
        return 1.0;
    }
    (min_span + (1.0 - min_span) * (small_n as f64 / n as f64).powf(power)).min(1.0)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Two-group design with an intercept: 6 samples, 2 coefficients.
    fn two_group() -> (Vec<f64>, usize, usize) {
        let design = vec![
            1.0, 0.0, //
            1.0, 0.0, //
            1.0, 0.0, //
            1.0, 1.0, //
            1.0, 1.0, //
            1.0, 1.0, //
        ];
        (design, 6, 2)
    }

    /// Rank deficient: the third column is the sum of the first two.
    fn rank_deficient() -> (Vec<f64>, usize, usize) {
        let design = vec![
            1.0, 0.0, 1.0, //
            1.0, 0.0, 1.0, //
            1.0, 1.0, 2.0, //
            1.0, 1.0, 2.0, //
        ];
        (design, 4, 3)
    }

    #[test]
    fn test_matrix_rank_of_a_full_rank_design() {
        let (d, r, c) = two_group();
        assert_eq!(matrix_rank(&d, r, c).unwrap(), 2);
        assert!(is_full_rank(&d, r, c).unwrap());
    }

    #[test]
    fn test_matrix_rank_detects_a_collinear_column() {
        let (d, r, c) = rank_deficient();
        assert_eq!(matrix_rank(&d, r, c).unwrap(), 2);
        assert!(!is_full_rank(&d, r, c).unwrap());
    }

    #[test]
    fn test_non_estimable_is_none_for_a_full_rank_design() {
        let (d, r, c) = two_group();
        assert!(non_estimable(&d, r, c).unwrap().is_none());
    }

    #[test]
    fn test_non_estimable_flags_the_dependent_column() {
        let (d, r, c) = rank_deficient();
        let flagged = non_estimable(&d, r, c).unwrap().unwrap();
        assert_eq!(flagged, vec![2]);
    }

    /// R: `diag(X %*% solve(t(X) %*% X) %*% t(X))` on the two-group design is
    /// 1/3 for every sample, since each group has three replicates.
    #[test]
    fn test_hat_diagonal_matches_the_analytic_leverage() {
        let (d, r, c) = two_group();
        let hat = hat_diagonal(&d, r, c).unwrap();
        for h in &hat {
            assert_relative_eq!(h, &(1.0 / 3.0), max_relative = 1e-12);
        }
        // The leverages of a full rank design sum to its rank.
        let total: f64 = hat.iter().sum();
        assert_relative_eq!(total, 2.0, max_relative = 1e-12);
    }

    #[test]
    fn test_design_as_factor_recovers_two_groups() {
        let (d, r, c) = two_group();
        let (groups, n) = design_as_factor(&d, r, c).unwrap();
        assert_eq!(n, 2);
        assert_eq!(groups, vec![0, 0, 0, 1, 1, 1]);
    }

    /// A design where every row is distinct gives one group per sample, which is
    /// what pushes `glm_fit` off the one-way path.
    #[test]
    fn test_design_as_factor_with_a_continuous_covariate() {
        let design = vec![1.0, 0.1, 1.0, 0.2, 1.0, 0.3, 1.0, 0.4];
        let (groups, n) = design_as_factor(&design, 4, 2).unwrap();
        assert_eq!(n, 4);
        assert_eq!(groups, vec![0, 1, 2, 3]);
    }

    /// Group labels follow the ascending order of the row hash, not order of
    /// first appearance. That is numpy's `unique` and therefore edgeR's.
    #[test]
    fn test_design_as_factor_labels_in_ascending_hash_order() {
        let design = vec![
            1.0, 1.0, //
            1.0, 0.0, //
            1.0, 1.0, //
        ];
        let (groups, n) = design_as_factor(&design, 3, 2).unwrap();
        assert_eq!(n, 2);
        // The (1, 0) row hashes lower, so it takes label 0 despite appearing second.
        assert_eq!(groups, vec![1, 0, 1]);
    }

    #[test]
    fn test_rejects_a_shape_mismatch() {
        let err = matrix_rank(&[1.0, 2.0, 3.0], 2, 2).unwrap_err();
        assert!(matches!(err, EdgeErrors::LengthMismatch { .. }));
        let err = design_as_factor(&[1.0, 2.0, 3.0], 2, 2).unwrap_err();
        assert!(matches!(err, EdgeErrors::LengthMismatch { .. }));
    }

    /// Parity with limma 3.66:
    /// ```r
    /// X <- cbind(1, c(0,0,0,1,1,1), c(0.1,0.4,0.2,0.9,0.7,0.8))
    /// contrastAsCoef(X, c(0, 1, -1), first = FALSE)
    /// ```
    ///
    /// The contrast lands in the last column, which is where edgeR's `glmLRT`
    /// expects it. The two nuisance columns may carry the opposite sign to
    /// limma's, since the QR does not pin `Q` down, so the assertion is on the
    /// contrast column exactly and on the column span for the rest.
    #[test]
    fn test_contrast_as_coef_matches_limma() {
        let design = vec![
            1.0, 0.0, 0.1, //
            1.0, 0.0, 0.4, //
            1.0, 0.0, 0.2, //
            1.0, 1.0, 0.9, //
            1.0, 1.0, 0.7, //
            1.0, 1.0, 0.8, //
        ];
        let contrast = vec![0.0, 1.0, -1.0];
        let out = contrast_as_coef(&design, 6, 3, &contrast, 1, false).unwrap();

        assert_eq!(out.coef, vec![2]);

        // limma's third column, the contrast itself.
        let expected_contrast = [-0.05, -0.2, -0.1, 0.05, 0.15, 0.1];
        for (sample, want) in expected_contrast.iter().enumerate() {
            assert_relative_eq!(
                out.design[sample * 3 + 2],
                want,
                epsilon = 1e-12,
                max_relative = 1e-9
            );
        }

        // The reformed design must span the same space as the original, which
        // is what makes the refit a genuine nested model.
        assert_eq!(matrix_rank(&out.design, 6, 3).unwrap(), 3);
    }

    /// The contrast column must reproduce the contrast applied to the original
    /// coefficients. That is the property the test statistic depends on, and it
    /// holds regardless of how the QR signed itself.
    #[test]
    fn test_contrast_column_carries_the_contrast() {
        let design = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let contrast = vec![0.0, 1.0];
        let out = contrast_as_coef(&design, 6, 2, &contrast, 1, false).unwrap();
        assert_eq!(out.coef, vec![1]);
        // With this contrast the reformed design is the original one, so the
        // group indicator column survives unchanged up to sign.
        let column: Vec<f64> = (0..6).map(|i| out.design[i * 2 + 1].abs()).collect();
        assert_eq!(column, vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_contrast_as_coef_can_put_the_contrast_first() {
        let design = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let out = contrast_as_coef(&design, 6, 2, &[0.0, 1.0], 1, true).unwrap();
        assert_eq!(out.coef, vec![0]);
    }

    #[test]
    fn test_contrast_as_coef_rejects_a_zero_contrast() {
        let design = vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let err = contrast_as_coef(&design, 4, 2, &[0.0, 0.0], 1, false).unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));
    }

    #[test]
    fn test_contrast_as_coef_rejects_a_shape_mismatch() {
        let design = vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let err = contrast_as_coef(&design, 4, 2, &[0.0, 1.0, 2.0], 1, false).unwrap_err();
        assert!(matches!(err, EdgeErrors::LengthMismatch { .. }));
    }

    /// limma: `chooseLowessSpan(10)`, `(100)`, `(1000)` give 1, 0.8555904,
    /// 0.5578822 at limma's own defaults of `small.n = 50, min.span = 0.3`.
    /// edgePython passes 25 and 0.2 instead, which is entry 13.
    #[test]
    fn test_choose_lowess_span_matches_limma() {
        let (small_n, min_span, power) = LIMMA_LOWESS_DEFAULTS;
        assert_relative_eq!(
            choose_lowess_span(10, small_n, min_span, power),
            1.0,
            max_relative = 1e-12
        );
        assert_relative_eq!(
            choose_lowess_span(100, small_n, min_span, power),
            0.855_590_4,
            max_relative = 1e-7
        );
        assert_relative_eq!(
            choose_lowess_span(1000, small_n, min_span, power),
            0.557_882_2,
            max_relative = 1e-7
        );
    }

    #[test]
    fn test_rejects_empty_dimensions() {
        let err = matrix_rank(&[], 0, 0).unwrap_err();
        assert!(matches!(err, EdgeErrors::MustBePositive(_)));
    }
}
