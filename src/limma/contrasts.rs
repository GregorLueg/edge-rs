//! limma's `contrasts.fit`, and enough of `makeContrasts` to build its input.
//!
//! Rotating a fit onto contrasts is a change of basis on the coefficient axis:
//! `beta -> beta C` and `V -> C' V C`. What makes it more than two matrix
//! products is the bookkeeping around it: aliased coefficients that a zero
//! contrast entry should be allowed to ignore, coefficients no contrast touches
//! at all, and the orthogonal case where the standard errors collapse to a
//! second matrix product instead of a per-gene solve.
//!
//! The per-gene branch is the one that costs anything, and it is the axis this
//! crate parallelises everywhere else, so it is a rayon fan-out over genes.
//! limma runs it as an R loop.
//!
//! ### Approximate under probe weights
//!
//! `cov.coefficients` describes the design, not each gene's own weighted fit,
//! so under voom weights these standard errors are an approximation. That is
//! upstream's position, stated at `R/contrasts.R:5-6`, and
//! [`crate::limma::lm_fit::lm_fit`] documents where the covariance comes from.
//! Exact contrast standard errors need the model refitted per gene in the
//! contrast basis, which is what limma 3.99's `lmFit(contrasts = )` added and
//! this does not have.

use rayon::prelude::*;

use crate::limma::marray::MArrayLm;
use crate::prelude::*;
use crate::utils::linalg::{cholesky_lower, cov2cor};

////////////
// Consts //
////////////

/// Below this, the off-diagonal correlations count as zero.
///
/// limma computes an orthogonality flag twice, at `1e-12` (`R/contrasts.R:60`)
/// and again at `1e-14` (`R/contrasts.R:103`). The first is overwritten before
/// it is ever read, so `1e-14` is the one that decides which branch runs.
const ORTHOGONAL_TOL: f64 = 1e-14;

/// Standard deviation standing in for a non-estimable coefficient.
///
/// limma sets aliased coefficients to zero and their unscaled standard
/// deviation to this, so a contrast that gives the coefficient zero weight
/// comes out finite while one that does not comes out enormous
/// (`R/contrasts.R:89-94`). The result is then read back through
/// [`NA_DETECT`].
const NA_SENTINEL: f64 = 1e30;

/// Above this, a rotated standard deviation means the contrast touched an
/// aliased coefficient, and both it and its coefficient go back to `NaN`.
///
/// One tenth of the square root of [`NA_SENTINEL`] squared, which is to say
/// limma's own `1e20` (`R/contrasts.R:128`): far above anything a real fit
/// produces, far below the sentinel itself.
const NA_DETECT: f64 = 1e20;

/////////////////////
// contrasts_fit   //
/////////////////////

/// Rotates a fit onto a set of contrasts.
///
/// Port of limma's `contrasts.fit`. The coefficient axis of the returned fit is
/// the contrast axis: `n_coef` counts contrasts, and `cov_coefficients` is
/// `n_contrasts` square. Any moderated statistic already on the fit is dropped,
/// because it belonged to the old basis.
///
/// ### Params
///
/// * `fit` - The fit to rotate, consumed
/// * `contrasts` - Contrast matrix, **column-major** `n_coef * n_contrasts`:
///   contrast `k` occupies `contrasts[k * n_coef..(k + 1) * n_coef]`. This is
///   the layout [`crate::utils::design::contrast_as_coef`] uses and what
///   [`make_contrasts`] produces.
/// * `n_contrasts` - Number of contrasts
///
/// ### Returns
///
/// The rotated fit, or [`EdgeErrors`] if the contrast matrix is the wrong
/// shape, holds a non-finite entry, asks for a non-estimable coefficient, or
/// the covariance is not positive definite.
///
/// ### References
///
/// Smyth, Statistical Applications in Genetics and Molecular Biology 3(1), 2004
pub fn contrasts_fit(
    mut fit: MArrayLm,
    contrasts: &[f64],
    n_contrasts: usize,
) -> Result<MArrayLm, EdgeErrors> {
    let n_coef = fit.n_coef;
    if n_contrasts == 0 {
        return Err(EdgeErrors::MustBePositive("n_contrasts".to_string()));
    }
    if contrasts.len() != n_coef * n_contrasts {
        return Err(EdgeErrors::LengthMismatch {
            name: "contrasts",
            expected: n_coef * n_contrasts,
            got: contrasts.len(),
        });
    }
    if let Some(i) = contrasts.iter().position(|v| !v.is_finite()) {
        return Err(EdgeErrors::InvalidArgument(format!(
            "contrasts must be finite; entry {i} is {}",
            contrasts[i]
        )));
    }

    fit.clear_tests();
    fit.contrasts = Some(contrasts.to_vec());

    // -- reduce to the estimable coefficients --
    //
    // `cov_coefficients` only ever describes the accepted columns, so a rank
    // deficient design needs the contrast rows lining up with those. A contrast
    // that puts weight on a rejected column has no answer (`R/contrasts.R:64`).
    let rank = fit.rank;
    let (contrasts, n_coef, coefficients, stdev_unscaled) = if rank < n_coef {
        let est = &fit.pivot[..rank];
        for (k, col) in contrasts.chunks_exact(n_coef).enumerate() {
            for (j, &v) in col.iter().enumerate() {
                if v != 0.0 && !est.contains(&j) {
                    return Err(EdgeErrors::NotEstimableContrast {
                        contrast: k,
                        coef: j,
                    });
                }
            }
        }
        let mut c = Vec::with_capacity(rank * n_contrasts);
        for col in contrasts.chunks_exact(n_coef) {
            c.extend(est.iter().map(|&j| col[j]));
        }
        (
            c,
            rank,
            subset_columns(&fit.coefficients, fit.n_genes, n_coef, est),
            subset_columns(&fit.stdev_unscaled, fit.n_genes, n_coef, est),
        )
    } else {
        (
            contrasts.to_vec(),
            n_coef,
            std::mem::take(&mut fit.coefficients),
            std::mem::take(&mut fit.stdev_unscaled),
        )
    };

    // -- drop coefficients no contrast touches --
    //
    // Not required for correctness; limma does it because it shrinks the
    // per-gene loop below (`R/contrasts.R:77`).
    let keep: Vec<usize> = (0..n_coef)
        .filter(|&j| contrasts.chunks_exact(n_coef).any(|col| col[j] != 0.0))
        .collect();
    let (contrasts, n_coef, mut coefficients, mut stdev_unscaled, cov) = if keep.len() < n_coef {
        let mut c = Vec::with_capacity(keep.len() * n_contrasts);
        for col in contrasts.chunks_exact(n_coef) {
            c.extend(keep.iter().map(|&j| col[j]));
        }
        let mut v = Vec::with_capacity(keep.len() * keep.len());
        for &i in &keep {
            v.extend(keep.iter().map(|&j| fit.cov_coefficients[i * n_coef + j]));
        }
        (
            c,
            keep.len(),
            subset_columns(&coefficients, fit.n_genes, n_coef, &keep),
            subset_columns(&stdev_unscaled, fit.n_genes, n_coef, &keep),
            v,
        )
    } else {
        (
            contrasts,
            n_coef,
            coefficients,
            stdev_unscaled,
            std::mem::take(&mut fit.cov_coefficients),
        )
    };

    // -- let a zero contrast entry clobber an aliased coefficient --
    let any_na = coefficients.iter().any(|v| v.is_nan());
    if any_na {
        for (c, s) in coefficients.iter_mut().zip(stdev_unscaled.iter_mut()) {
            if c.is_nan() {
                *c = 0.0;
                *s = NA_SENTINEL;
            }
        }
    }

    let cormatrix = cov2cor(&cov, n_coef);
    let orthogonal = n_coef < 2
        || (0..n_coef).all(|i| (0..i).all(|j| cormatrix[i * n_coef + j].abs() < ORTHOGONAL_TOL));

    // -- new coefficients: beta C --
    let new_coef = gemm(&coefficients, fit.n_genes, n_coef, &contrasts, n_contrasts);

    // -- new covariance: (L' C)' (L' C), with V = L L' --
    let mut chol = cov.clone();
    if !cholesky_lower(&mut chol, n_coef, n_coef) {
        return Err(EdgeErrors::CholeskyFailed(
            "the coefficient covariance is not positive definite, so the contrasts \
             cannot be rotated"
                .to_string(),
        ));
    }
    let new_cov = crossprod_whitened(&chol, &contrasts, n_coef, n_contrasts);

    // -- new unscaled standard deviations --
    let mut new_stdev = if orthogonal {
        // sqrt(u^2 C^2), elementwise squares on both sides.
        let u2: Vec<f64> = stdev_unscaled.iter().map(|v| v * v).collect();
        let c2: Vec<f64> = contrasts.iter().map(|v| v * v).collect();
        let mut out = gemm(&u2, fit.n_genes, n_coef, &c2, n_contrasts);
        out.iter_mut().for_each(|v| *v = v.sqrt());
        out
    } else {
        let mut rchol = cormatrix.clone();
        if !cholesky_lower(&mut rchol, n_coef, n_coef) {
            return Err(EdgeErrors::CholeskyFailed(
                "the coefficient correlation matrix is not positive definite, so the \
                 contrast standard errors cannot be formed"
                    .to_string(),
            ));
        }
        correlated_stdev(
            &stdev_unscaled,
            fit.n_genes,
            n_coef,
            &contrasts,
            n_contrasts,
            &rchol,
        )
    };

    // -- put the aliased entries back --
    let mut new_coef = new_coef;
    if any_na {
        for (c, s) in new_coef.iter_mut().zip(new_stdev.iter_mut()) {
            if *s > NA_DETECT {
                *c = f64::NAN;
                *s = f64::NAN;
            }
        }
    }

    fit.coefficients = new_coef;
    fit.stdev_unscaled = new_stdev;
    fit.cov_coefficients = new_cov;
    fit.n_coef = n_contrasts;
    Ok(fit)
}

/////////////
// Kernels //
/////////////

/// Picks a subset of columns out of a row-major matrix.
///
/// ### Params
///
/// * `x` - Row-major `n_rows * n_cols`
/// * `n_rows` - Number of rows
/// * `n_cols` - Number of columns
/// * `cols` - Column indices to keep, in the order wanted
///
/// ### Returns
///
/// Row-major `n_rows * cols.len()`.
fn subset_columns(x: &[f64], n_rows: usize, n_cols: usize, cols: &[usize]) -> Vec<f64> {
    let mut out = Vec::with_capacity(n_rows * cols.len());
    for row in x.chunks_exact(n_cols).take(n_rows) {
        out.extend(cols.iter().map(|&j| row[j]));
    }
    out
}

/// Multiplies a row-major matrix by a column-major one.
///
/// `A B` with `A` row-major `n_rows * k` and `B` column-major `k * n_cols`,
/// which is the layout the contrast matrix arrives in. Rayon over rows, since
/// that is the gene axis.
///
/// ### Params
///
/// * `a` - Left operand, row-major `n_rows * k`
/// * `n_rows` - Rows of `a`
/// * `k` - Shared dimension
/// * `b` - Right operand, column-major `k * n_cols`
/// * `n_cols` - Columns of `b`
///
/// ### Returns
///
/// The product, row-major `n_rows * n_cols`.
fn gemm(a: &[f64], n_rows: usize, k: usize, b: &[f64], n_cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; n_rows * n_cols];
    out.par_chunks_mut(n_cols)
        .zip(a.par_chunks_exact(k))
        .for_each(|(dst, row)| {
            for (c, slot) in dst.iter_mut().enumerate() {
                let col = &b[c * k..(c + 1) * k];
                *slot = row.iter().zip(col).map(|(x, y)| x * y).sum();
            }
        });
    out
}

/// Forms `(L' C)' (L' C)` for a lower triangular `L`.
///
/// With `V = L L'` this is `C' V C`, the covariance in the contrast basis.
/// limma writes it as `crossprod(chol(V) %*% C)` with an upper factor
/// (`R/contrasts.R:107-108`); the transpose of that factor is `L`, so the two
/// are the same product.
///
/// ### Params
///
/// * `chol` - Lower Cholesky factor of the covariance, row-major `n * n`
/// * `contrasts` - Column-major `n * n_contrasts`
/// * `n` - Order of the covariance
/// * `n_contrasts` - Number of contrasts
///
/// ### Returns
///
/// The rotated covariance, row-major `n_contrasts * n_contrasts`.
fn crossprod_whitened(chol: &[f64], contrasts: &[f64], n: usize, n_contrasts: usize) -> Vec<f64> {
    // W[.,k] = L' C[.,k], so W[i,k] = sum_{r >= i} L[r,i] C[r,k].
    let mut w = vec![0.0; n * n_contrasts];
    for k in 0..n_contrasts {
        let col = &contrasts[k * n..(k + 1) * n];
        for i in 0..n {
            let mut acc = 0.0;
            for (r, &c) in col.iter().enumerate().skip(i) {
                acc += chol[r * n + i] * c;
            }
            w[k * n + i] = acc;
        }
    }

    let mut out = vec![0.0; n_contrasts * n_contrasts];
    for a in 0..n_contrasts {
        for b in a..n_contrasts {
            let acc: f64 = (0..n).map(|i| w[a * n + i] * w[b * n + i]).sum();
            out[a * n_contrasts + b] = acc;
            out[b * n_contrasts + a] = acc;
        }
    }
    out
}

/// Contrast standard errors when the coefficients are correlated.
///
/// For gene `i` the answer is the column norms of `R diag(u_i) C`, with `R` the
/// Cholesky factor of the coefficient correlation matrix. limma runs this as an
/// R loop over genes (`R/contrasts.R:118-123`); here it is a rayon fan-out with
/// a per-thread scratch buffer, which is the shape every other genewise loop in
/// the crate takes.
///
/// ### Params
///
/// * `stdev` - Unscaled standard deviations, row-major `n_genes * n`
/// * `n_genes` - Number of genes
/// * `n` - Number of coefficients
/// * `contrasts` - Column-major `n * n_contrasts`
/// * `n_contrasts` - Number of contrasts
/// * `rchol` - Lower Cholesky factor of the correlation matrix, row-major `n`
///
/// ### Returns
///
/// The rotated standard deviations, row-major `n_genes * n_contrasts`.
fn correlated_stdev(
    stdev: &[f64],
    n_genes: usize,
    n: usize,
    contrasts: &[f64],
    n_contrasts: usize,
    rchol: &[f64],
) -> Vec<f64> {
    let mut out = vec![0.0; n_genes * n_contrasts];
    out.par_chunks_mut(n_contrasts)
        .zip(stdev.par_chunks_exact(n))
        .for_each_init(
            || vec![0.0; n],
            |scratch, (dst, u)| {
                for (k, slot) in dst.iter_mut().enumerate() {
                    let col = &contrasts[k * n..(k + 1) * n];
                    // scratch = diag(u) C[., k], then take the norm of R times it.
                    for (r, s) in scratch.iter_mut().enumerate() {
                        *s = u[r] * col[r];
                    }
                    // R is upper triangular in limma's orientation, so row i of
                    // `R x` is sum_{r >= i} rchol[r][i] * x[r].
                    let mut acc = 0.0;
                    for i in 0..n {
                        let mut v = 0.0;
                        for (r, &x) in scratch.iter().enumerate().skip(i) {
                            v += rchol[r * n + i] * x;
                        }
                        acc += v * v;
                    }
                    *slot = acc.sqrt();
                }
            },
        );
    out
}

///////////////////////
// make_contrasts    //
///////////////////////

/// Builds a contrast matrix from linear expressions over the design columns.
///
/// The numerical half of limma's `makeContrasts`. Upstream is R
/// metaprogramming: `substitute`, then an environment mapping every level name
/// to an indicator vector, then `eval(parse(text = ...))`
/// (`R/modelmatrix.R:46-114`). None of that ports, and none of it is
/// arithmetic. What ports is the grammar people actually write in:
/// `+ - * / ( )`, decimal literals, and the column names.
///
/// So `"grpB"`, `"grpB - grpA"` and `"grpB - 0.5 * batb2"` all work, and
/// anything requiring an R evaluator does not. A level named `(Intercept)` is
/// matched as `Intercept` too, which is the rename limma does at
/// `R/modelmatrix.R:54`.
///
/// ### Params
///
/// * `levels` - Design column names, in design order
/// * `contrasts` - One expression per contrast
///
/// ### Returns
///
/// The contrast matrix, column-major `levels.len() * contrasts.len()`, and the
/// number of contrasts. [`EdgeErrors::InvalidArgument`] for an empty input, an
/// unparseable expression, or a name that is not a level.
pub fn make_contrasts(
    levels: &[&str],
    contrasts: &[&str],
) -> Result<(Vec<f64>, usize), EdgeErrors> {
    if levels.is_empty() {
        return Err(EdgeErrors::InvalidArgument(
            "make_contrasts needs at least one level".to_string(),
        ));
    }
    if contrasts.is_empty() {
        return Err(EdgeErrors::InvalidArgument(
            "make_contrasts needs at least one contrast".to_string(),
        ));
    }

    let n = levels.len();
    let mut out = Vec::with_capacity(n * contrasts.len());
    for expr in contrasts {
        let mut column = vec![0.0; n];
        let mut parser = Parser::new(expr, levels);
        parser.expression(&mut column, 1.0)?;
        parser.finish()?;
        out.extend(column);
    }
    Ok((out, contrasts.len()))
}

/// Recursive descent over the contrast grammar.
///
/// The value of an expression is a vector over the levels, so the parser
/// accumulates into one rather than returning numbers. Multiplication and
/// division are only defined when one side is a bare number, which is exactly
/// the restriction a linear contrast is under anyway.
struct Parser<'a> {
    /// Remaining characters, whitespace included.
    src: &'a [u8],
    /// Read position in `src`.
    pos: usize,
    /// Level names, in design order.
    levels: &'a [&'a str],
    /// The expression being parsed, for error messages.
    expr: &'a str,
}

/// One parsed operand: either a plain number or a combination of levels.
enum Operand {
    /// A literal, usable as a multiplier.
    Number(f64),
    /// Weights over the levels.
    Vector(Vec<f64>),
}

impl<'a> Parser<'a> {
    /// Starts a parse.
    ///
    /// ### Params
    ///
    /// * `expr` - The expression text
    /// * `levels` - Level names, in design order
    ///
    /// ### Returns
    ///
    /// A parser positioned at the start of `expr`.
    fn new(expr: &'a str, levels: &'a [&'a str]) -> Self {
        Self {
            src: expr.as_bytes(),
            pos: 0,
            levels,
            expr,
        }
    }

    /// Builds a syntax error naming the expression and position.
    ///
    /// ### Params
    ///
    /// * `what` - What went wrong
    ///
    /// ### Returns
    ///
    /// The error.
    fn error(&self, what: &str) -> EdgeErrors {
        EdgeErrors::InvalidArgument(format!(
            "contrast '{}': {what} at position {}",
            self.expr, self.pos
        ))
    }

    /// Skips whitespace.
    fn skip_space(&mut self) {
        while self.pos < self.src.len() && self.src[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    /// Returns the next non-space byte without consuming it.
    ///
    /// ### Returns
    ///
    /// The byte, or `None` at the end of the input.
    fn peek(&mut self) -> Option<u8> {
        self.skip_space();
        self.src.get(self.pos).copied()
    }

    /// Checks that the whole expression was consumed.
    ///
    /// ### Returns
    ///
    /// `Ok(())`, or [`EdgeErrors::InvalidArgument`] naming the leftover.
    fn finish(&mut self) -> Result<(), EdgeErrors> {
        match self.peek() {
            None => Ok(()),
            Some(c) => Err(self.error(&format!("unexpected '{}'", c as char))),
        }
    }

    /// Parses a sum of terms, accumulating into `out`.
    ///
    /// ### Params
    ///
    /// * `out` - Weight vector to add into
    /// * `sign` - Multiplier applied to every term, so a nested `-(...)` flips
    ///   the whole group
    ///
    /// ### Returns
    ///
    /// `Ok(())`, or a syntax error.
    fn expression(&mut self, out: &mut [f64], sign: f64) -> Result<(), EdgeErrors> {
        let mut s = sign;
        if let Some(c) = self.peek()
            && (c == b'+' || c == b'-')
        {
            self.pos += 1;
            if c == b'-' {
                s = -s;
            }
        }
        self.term(out, s)?;

        while let Some(c) = self.peek() {
            if c != b'+' && c != b'-' {
                break;
            }
            self.pos += 1;
            let s = if c == b'-' { -sign } else { sign };
            self.term(out, s)?;
        }
        Ok(())
    }

    /// Parses a product of factors and adds it to `out`.
    ///
    /// ### Params
    ///
    /// * `out` - Weight vector to add into
    /// * `sign` - Multiplier for this term
    ///
    /// ### Returns
    ///
    /// `Ok(())`, or a syntax error.
    fn term(&mut self, out: &mut [f64], sign: f64) -> Result<(), EdgeErrors> {
        let mut acc = self.factor()?;

        while let Some(c) = self.peek() {
            if c != b'*' && c != b'/' {
                break;
            }
            self.pos += 1;
            let rhs = self.factor()?;
            acc = match (acc, rhs, c) {
                (Operand::Number(a), Operand::Number(b), b'*') => Operand::Number(a * b),
                (Operand::Number(a), Operand::Number(b), _) => {
                    if b == 0.0 {
                        return Err(self.error("division by zero"));
                    }
                    Operand::Number(a / b)
                }
                (Operand::Vector(mut v), Operand::Number(a), b'*') => {
                    v.iter_mut().for_each(|x| *x *= a);
                    Operand::Vector(v)
                }
                (Operand::Vector(mut v), Operand::Number(a), _) => {
                    if a == 0.0 {
                        return Err(self.error("division by zero"));
                    }
                    v.iter_mut().for_each(|x| *x /= a);
                    Operand::Vector(v)
                }
                (Operand::Number(a), Operand::Vector(mut v), b'*') => {
                    v.iter_mut().for_each(|x| *x *= a);
                    Operand::Vector(v)
                }
                _ => {
                    return Err(self
                        .error("a contrast is linear in the levels, so it cannot divide by one"));
                }
            };
        }

        match acc {
            Operand::Vector(v) => {
                for (slot, x) in out.iter_mut().zip(v) {
                    *slot += sign * x;
                }
            }
            Operand::Number(a) => {
                if a != 0.0 {
                    return Err(self.error("a bare number is not a contrast"));
                }
            }
        }
        Ok(())
    }

    /// Parses one factor: a literal, a level name, or a bracketed expression.
    ///
    /// ### Returns
    ///
    /// The operand, or a syntax error.
    fn factor(&mut self) -> Result<Operand, EdgeErrors> {
        match self.peek() {
            None => Err(self.error("expression ended early")),
            Some(b'(') => {
                self.pos += 1;
                let mut inner = vec![0.0; self.levels.len()];
                self.expression(&mut inner, 1.0)?;
                match self.peek() {
                    Some(b')') => {
                        self.pos += 1;
                        Ok(Operand::Vector(inner))
                    }
                    _ => Err(self.error("expected ')'")),
                }
            }
            Some(b'-') => {
                self.pos += 1;
                Ok(match self.factor()? {
                    Operand::Number(a) => Operand::Number(-a),
                    Operand::Vector(mut v) => {
                        v.iter_mut().for_each(|x| *x = -*x);
                        Operand::Vector(v)
                    }
                })
            }
            Some(c) if c.is_ascii_digit() || c == b'.' => self.number(),
            Some(_) => self.name(),
        }
    }

    /// Parses a decimal literal.
    ///
    /// ### Returns
    ///
    /// The number, or a syntax error.
    fn number(&mut self) -> Result<Operand, EdgeErrors> {
        let start = self.pos;
        while self.pos < self.src.len() {
            let c = self.src[self.pos];
            let exponent = (c == b'+' || c == b'-')
                && self.pos > start
                && matches!(self.src[self.pos - 1], b'e' | b'E');
            if c.is_ascii_digit() || c == b'.' || c == b'e' || c == b'E' || exponent {
                self.pos += 1;
            } else {
                break;
            }
        }
        let text = &self.expr[start..self.pos];
        text.parse::<f64>()
            .map(Operand::Number)
            .map_err(|_| self.error(&format!("'{text}' is not a number")))
    }

    /// Parses a level name and returns its indicator vector.
    ///
    /// ### Returns
    ///
    /// The indicator, or [`EdgeErrors::InvalidArgument`] if the name is not a
    /// level.
    fn name(&mut self) -> Result<Operand, EdgeErrors> {
        let start = self.pos;
        while self.pos < self.src.len() {
            let c = self.src[self.pos];
            if c.is_ascii_alphanumeric() || c == b'.' || c == b'_' {
                self.pos += 1;
            } else {
                break;
            }
        }
        if self.pos == start {
            return Err(self.error(&format!("unexpected '{}'", self.src[self.pos] as char)));
        }
        let text = &self.expr[start..self.pos];
        let idx = self
            .levels
            .iter()
            .position(|l| *l == text || (*l == "(Intercept)" && text == "Intercept"))
            .ok_or_else(|| {
                EdgeErrors::InvalidArgument(format!(
                    "contrast '{}': '{text}' is not one of the design columns",
                    self.expr
                ))
            })?;
        let mut v = vec![0.0; self.levels.len()];
        v[idx] = 1.0;
        Ok(Operand::Vector(v))
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::limma::lm_fit::lm_fit;
    use approx::assert_relative_eq;

    const LEVELS: [&str; 3] = ["Int", "grpB", "batb2"];

    #[test]
    fn test_make_contrasts_reads_a_single_level() {
        let (c, n) = make_contrasts(&LEVELS, &["grpB"]).unwrap();
        assert_eq!(n, 1);
        assert_eq!(c, vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_make_contrasts_reads_a_scaled_difference() {
        // The expression `fac_contrasts.csv` is built from.
        let (c, _) = make_contrasts(&LEVELS, &["grpB - 0.5 * batb2"]).unwrap();
        assert_eq!(c, vec![0.0, 1.0, -0.5]);
    }

    #[test]
    fn test_make_contrasts_handles_brackets_and_division() {
        let (c, _) = make_contrasts(&LEVELS, &["(grpB + batb2) / 2"]).unwrap();
        assert_eq!(c, vec![0.0, 0.5, 0.5]);
    }

    #[test]
    fn test_make_contrasts_handles_a_leading_minus() {
        let (c, _) = make_contrasts(&LEVELS, &["-grpB + batb2"]).unwrap();
        assert_eq!(c, vec![0.0, -1.0, 1.0]);
    }

    #[test]
    fn test_make_contrasts_negates_a_whole_group() {
        let (c, _) = make_contrasts(&LEVELS, &["-(grpB - batb2)"]).unwrap();
        assert_eq!(c, vec![0.0, -1.0, 1.0]);
    }

    #[test]
    fn test_make_contrasts_renames_the_intercept() {
        let levels = ["(Intercept)", "grpB"];
        let (c, _) = make_contrasts(&levels, &["Intercept"]).unwrap();
        assert_eq!(c, vec![1.0, 0.0]);
    }

    #[test]
    fn test_make_contrasts_builds_several_columns() {
        let (c, n) = make_contrasts(&LEVELS, &["grpB", "batb2 - grpB"]).unwrap();
        assert_eq!(n, 2);
        // Column-major: contrast 1 then contrast 2.
        assert_eq!(c, vec![0.0, 1.0, 0.0, 0.0, -1.0, 1.0]);
    }

    #[test]
    fn test_make_contrasts_rejects_an_unknown_level() {
        assert!(make_contrasts(&LEVELS, &["grpC"]).is_err());
    }

    #[test]
    fn test_make_contrasts_rejects_a_malformed_expression() {
        assert!(make_contrasts(&LEVELS, &["grpB +"]).is_err());
        assert!(make_contrasts(&LEVELS, &["(grpB"]).is_err());
        assert!(make_contrasts(&LEVELS, &["grpB grpB"]).is_err());
        assert!(make_contrasts(&LEVELS, &["2 / grpB"]).is_err());
        assert!(make_contrasts(&LEVELS, &[""]).is_err());
    }

    #[test]
    fn test_make_contrasts_rejects_a_bare_number() {
        assert!(make_contrasts(&LEVELS, &["1.5"]).is_err());
    }

    /// Two genes on a design whose two columns are orthogonal.
    fn orthogonal_fit() -> MArrayLm {
        // Columns (1,1,-1,-1) and (1,-1,1,-1): orthogonal by construction, so
        // the fast branch runs and the answer is checkable by hand.
        let design = vec![
            1.0, 1.0, //
            1.0, -1.0, //
            -1.0, 1.0, //
            -1.0, -1.0,
        ];
        let y = vec![1.0, 2.0, 3.0, 4.5, 2.0, 1.0, 0.0, -1.5];
        let fit = lm_fit(&y, 2, 4, &design, 2, None, None, None).unwrap();
        MArrayLm::from_lm_fit(fit, &design, 2, 4, None).unwrap()
    }

    #[test]
    fn test_contrasts_fit_on_an_orthogonal_design() {
        let m = orthogonal_fit();
        let coef = m.coefficients.clone();
        let stdev = m.stdev_unscaled.clone();
        // The difference of the two coefficients.
        let out = contrasts_fit(m, &[1.0, -1.0], 1).unwrap();
        assert_eq!(out.n_coef, 1);
        for g in 0..2 {
            assert_relative_eq!(
                out.coefficients[g],
                coef[g * 2] - coef[g * 2 + 1],
                epsilon = 1e-12
            );
            // Orthogonal, so the variances simply add.
            let want = (stdev[g * 2].powi(2) + stdev[g * 2 + 1].powi(2)).sqrt();
            assert_relative_eq!(out.stdev_unscaled[g], want, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_contrasts_fit_identity_leaves_the_fit_alone() {
        let m = orthogonal_fit();
        let coef = m.coefficients.clone();
        let stdev = m.stdev_unscaled.clone();
        let cov = m.cov_coefficients.clone();
        let out = contrasts_fit(m, &[1.0, 0.0, 0.0, 1.0], 2).unwrap();
        for (a, b) in out.coefficients.iter().zip(&coef) {
            assert_relative_eq!(a, b, epsilon = 1e-12);
        }
        for (a, b) in out.stdev_unscaled.iter().zip(&stdev) {
            assert_relative_eq!(a, b, epsilon = 1e-12);
        }
        for (a, b) in out.cov_coefficients.iter().zip(&cov) {
            assert_relative_eq!(a, b, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_contrasts_fit_clears_the_test_statistics() {
        let mut m = orthogonal_fit();
        m.t = Some(vec![1.0; 4]);
        let out = contrasts_fit(m, &[1.0, -1.0], 1).unwrap();
        assert!(out.t.is_none());
        assert!(out.contrasts.is_some());
    }

    #[test]
    fn test_contrasts_fit_rejects_a_bad_shape() {
        let m = orthogonal_fit();
        assert!(contrasts_fit(m, &[1.0, -1.0, 0.5], 1).is_err());
    }

    #[test]
    fn test_contrasts_fit_rejects_a_non_finite_contrast() {
        let m = orthogonal_fit();
        assert!(contrasts_fit(m, &[1.0, f64::NAN], 1).is_err());
    }
}
