//! Small dense factorisations shared by the limma modules.
//!
//! Everything here operates on matrices of order `n_coef` or `n_samples`, that
//! is, tens of values. They are hand-rolled rather than routed through faer
//! because at this size the call overhead dominates, and because two of them
//! need to report failure rather than panic. The one exception is the
//! eigendecomposition, which is faer's.
//!
//! Matrices are row-major with an explicit stride, so a caller can factor a
//! leading submatrix of a larger buffer without copying it out first.

use faer::linalg::solvers::SelfAdjointEigen;
use faer::{MatRef, Side};

use crate::prelude::*;

///////////////
// Cholesky  //
///////////////

/// Lower Cholesky factor of a symmetric positive definite matrix, in place.
///
/// Only the lower triangle of `a` is written; the upper triangle is left as it
/// was and must not be read afterwards.
///
/// ### Params
///
/// * `a` - The matrix, row-major with row stride `stride`, overwritten by its
///   lower Cholesky factor
/// * `n` - Order of the matrix
/// * `stride` - Row stride of `a`
///
/// ### Returns
///
/// `false` if a pivot is not strictly positive, meaning the matrix is not
/// positive definite.
pub(crate) fn cholesky_lower(a: &mut [f64], n: usize, stride: usize) -> bool {
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * stride + j];
            for k in 0..j {
                sum -= a[i * stride + k] * a[j * stride + k];
            }
            if i == j {
                // Rejects NaN as well as a non-positive pivot.
                if sum <= 0.0 || sum.is_nan() {
                    return false;
                }
                a[i * stride + j] = sum.sqrt();
            } else {
                a[i * stride + j] = sum / a[j * stride + j];
            }
        }
    }
    true
}

/// Solves `L x = b` in place for lower triangular `L`.
///
/// ### Params
///
/// * `l` - Lower triangular factor, row-major with row stride `stride`
/// * `stride` - Row stride of `l`
/// * `n` - Order of the system
/// * `b` - Right-hand side, overwritten by the solution
pub(crate) fn forward_substitute(l: &[f64], stride: usize, n: usize, b: &mut [f64]) {
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i * stride + k] * b[k];
        }
        b[i] = sum / l[i * stride + i];
    }
}

//////////////////
// Triangular   //
//////////////////

/// Inverts the leading `n` by `n` block of an upper triangular matrix.
///
/// Back substitution column by column. Entries of `r_inv` outside the leading
/// block are left untouched, so a scratch buffer can be reused across calls of
/// differing order without clearing it.
///
/// ### Params
///
/// * `r` - Upper triangular factor, row-major with row stride `stride`
/// * `r_inv` - Destination, row-major with the same stride
/// * `n` - Order of the block to invert
/// * `stride` - Row stride of both matrices
pub(crate) fn invert_upper_triangular(r: &[f64], r_inv: &mut [f64], n: usize, stride: usize) {
    for i in (0..n).rev() {
        let inv_diag = 1.0 / r[i * stride + i];
        r_inv[i * stride + i] = inv_diag;
        for j in (i + 1)..n {
            let mut sum = 0.0;
            for t in (i + 1)..=j {
                sum += r[i * stride + t] * r_inv[t * stride + j];
            }
            r_inv[i * stride + j] = -sum * inv_diag;
        }
    }
}

/// Forms `R^-1 R^-T` from the inverse of an upper triangular factor.
///
/// That is `(R'R)^-1`, R's `chol2inv`, which for a least squares QR is the
/// unscaled covariance of the estimable coefficients.
///
/// ### Params
///
/// * `r_inv` - Inverse of the upper triangular factor, row-major with row
///   stride `stride`, only its leading `n` by `n` block read
/// * `n` - Order of the block
/// * `stride` - Row stride of `r_inv`
///
/// ### Returns
///
/// The symmetric product, row-major `n * n`, densely packed.
pub(crate) fn cross_inverse(r_inv: &[f64], n: usize, stride: usize) -> Vec<f64> {
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        for j in i..n {
            // Row i of an upper triangular inverse is zero before column i, so
            // the shared support starts at the later of the two rows.
            let mut sum = 0.0;
            for s in j..n {
                sum += r_inv[i * stride + s] * r_inv[j * stride + s];
            }
            out[i * n + j] = sum;
            out[j * n + i] = sum;
        }
    }
    out
}

/////////////////
// Correlation //
/////////////////

/// Rescales a covariance matrix to a correlation matrix.
///
/// R's `cov2cor`. A zero or negative diagonal entry would divide by zero;
/// callers that can produce one are expected to nudge it first, which is what
/// limma's `classifyTestsF` does for an all-zero contrast.
///
/// ### Params
///
/// * `v` - Covariance, row-major `n * n`
/// * `n` - Order of the matrix
///
/// ### Returns
///
/// The correlation matrix, row-major `n * n`.
pub(crate) fn cov2cor(v: &[f64], n: usize) -> Vec<f64> {
    let scale: Vec<f64> = (0..n).map(|i| v[i * n + i].sqrt()).collect();
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = v[i * n + j] / (scale[i] * scale[j]);
        }
    }
    out
}

////////////////////
// Eigenvalues    //
////////////////////

/// Eigendecomposition of a symmetric matrix, eigenvalues descending.
///
/// R's `eigen(symmetric = TRUE)` returns eigenvalues in decreasing order and
/// limma's `classifyTestsF` relies on that, indexing the leading components and
/// comparing every eigenvalue against the first. faer follows the LAPACK
/// convention instead, so the pairs are reordered here.
///
/// ### Params
///
/// * `a` - Symmetric matrix, row-major `n * n`. Only the lower triangle is
///   read.
/// * `n` - Order of the matrix
///
/// ### Returns
///
/// The eigenvalues in decreasing order, and the matching eigenvectors as a
/// row-major `n * n` matrix whose column `j` is the eigenvector for eigenvalue
/// `j`. [`EdgeErrors::EigenFailed`] if the decomposition does not converge.
pub(crate) fn self_adjoint_eigen_desc(
    a: &[f64],
    n: usize,
) -> Result<(Vec<f64>, Vec<f64>), EdgeErrors> {
    if a.len() != n * n {
        return Err(EdgeErrors::LengthMismatch {
            name: "a",
            expected: n * n,
            got: a.len(),
        });
    }

    let mat = MatRef::from_row_major_slice(a, n, n);
    let eigen = SelfAdjointEigen::new(mat, Side::Lower).map_err(|_| {
        EdgeErrors::EigenFailed(format!("symmetric eigendecomposition of order {n}"))
    })?;

    let s = eigen.S();
    let u = eigen.U();

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| s[j].total_cmp(&s[i]));

    let values: Vec<f64> = order.iter().map(|&j| s[j]).collect();
    let mut vectors = vec![0.0; n * n];
    for (slot, &j) in order.iter().enumerate() {
        for i in 0..n {
            vectors[i * n + slot] = u[(i, j)];
        }
    }

    Ok((values, vectors))
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_cholesky_lower_reproduces_the_matrix() {
        // [[4, 2], [2, 5]] = L L' with L = [[2, 0], [1, 2]].
        let mut a = vec![4.0, 2.0, 2.0, 5.0];
        assert!(cholesky_lower(&mut a, 2, 2));
        assert_relative_eq!(a[0], 2.0, epsilon = 1e-12);
        assert_relative_eq!(a[2], 1.0, epsilon = 1e-12);
        assert_relative_eq!(a[3], 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_cholesky_lower_rejects_an_indefinite_matrix() {
        let mut a = vec![1.0, 2.0, 2.0, 1.0];
        assert!(!cholesky_lower(&mut a, 2, 2));
    }

    #[test]
    fn test_forward_substitute_solves_a_lower_system() {
        // L = [[2, 0], [1, 2]], b = [2, 5] gives x = [1, 2].
        let l = vec![2.0, 0.0, 1.0, 2.0];
        let mut b = vec![2.0, 5.0];
        forward_substitute(&l, 2, 2, &mut b);
        assert_relative_eq!(b[0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(b[1], 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_invert_upper_triangular_round_trips() {
        let r = vec![2.0, 1.0, 0.0, 4.0];
        let mut inv = vec![0.0; 4];
        invert_upper_triangular(&r, &mut inv, 2, 2);
        // R R^-1 must be the identity.
        for i in 0..2 {
            for j in 0..2 {
                let mut acc = 0.0;
                for k in 0..2 {
                    acc += r[i * 2 + k] * inv[k * 2 + j];
                }
                let want = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(acc, want, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_cross_inverse_matches_the_explicit_inverse() {
        // R = [[2, 1], [0, 4]], so R'R = [[4, 2], [2, 17]] and its inverse is
        // [[17, -2], [-2, 4]] / 64.
        let r = vec![2.0, 1.0, 0.0, 4.0];
        let mut inv = vec![0.0; 4];
        invert_upper_triangular(&r, &mut inv, 2, 2);
        let cov = cross_inverse(&inv, 2, 2);
        assert_relative_eq!(cov[0], 17.0 / 64.0, epsilon = 1e-12);
        assert_relative_eq!(cov[1], -2.0 / 64.0, epsilon = 1e-12);
        assert_relative_eq!(cov[2], -2.0 / 64.0, epsilon = 1e-12);
        assert_relative_eq!(cov[3], 4.0 / 64.0, epsilon = 1e-12);
    }

    #[test]
    fn test_cov2cor_puts_ones_on_the_diagonal() {
        let v = vec![4.0, 2.0, 2.0, 9.0];
        let c = cov2cor(&v, 2);
        assert_relative_eq!(c[0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(c[3], 1.0, epsilon = 1e-12);
        assert_relative_eq!(c[1], 2.0 / 6.0, epsilon = 1e-12);
    }

    #[test]
    fn test_self_adjoint_eigen_desc_orders_descending() {
        // Eigenvalues 3 and 1, eigenvectors (1, 1) and (1, -1) up to scale.
        let a = vec![2.0, 1.0, 1.0, 2.0];
        let (values, vectors) = self_adjoint_eigen_desc(&a, 2).expect("eigen failed");
        assert_relative_eq!(values[0], 3.0, epsilon = 1e-12);
        assert_relative_eq!(values[1], 1.0, epsilon = 1e-12);
        // The leading eigenvector has equal components; sign is arbitrary.
        assert_relative_eq!(vectors[0].abs(), vectors[2].abs(), epsilon = 1e-12);
    }

    #[test]
    fn test_self_adjoint_eigen_desc_rejects_a_bad_shape() {
        let a = vec![1.0, 0.0, 0.0];
        assert!(self_adjoint_eigen_desc(&a, 2).is_err());
    }
}
