//! limma's `lmFit`: one weighted least squares fit per gene.
//!
//! This is the linear-model half of voom. Genes are fitted independently, so the
//! shape is a rayon fan-out over genes with a per-thread scratch buffer, the
//! same pattern as [`crate::glm::levenberg`].
//!
//! ### One code path, not four
//!
//! limma splits `lmFit` across `lm.series` and `gls.series`, each with a fast
//! all-observed branch and a slow per-gene branch, four bodies in total. They
//! are all the same computation. Write the working covariance as
//! `V = D^-1/2 R D^-1/2`, with `D` the weights and `R` the compound-symmetry
//! correlation induced by `block`; whitening by the Cholesky factor of `V` turns
//! every case into ordinary least squares. Plain weights are `R = I`, where the
//! factor is diagonal and whitening is a row scaling by `sqrt(w)`. So there is
//! one body here, and the special cases are branches inside it rather than
//! separate routines.
//!
//! The factorisation is not repeated per gene either. `V = D^-1/2 R D^-1/2`
//! factors as `L = D^-1/2 L_R`, since scaling a lower triangular matrix by a
//! positive diagonal leaves it lower triangular with a positive diagonal, and
//! Cholesky factors are unique. `L_R` therefore depends only on `block` and the
//! correlation, so it is computed once for the full sample set and reused by
//! every gene that keeps all its observations. Only genes with dropped
//! observations refactorise, on their own submatrix.
//!
//! ### Missing observations
//!
//! A gene whose response is not finite at some samples, or whose weight there is
//! zero, is refitted on what survives, with its own residual degrees of freedom
//! and its own rank. That is the whole reason limma has a per-gene loop at all.
//!
//! ### Rank deficiency
//!
//! The per-gene solve is a Householder QR that accepts design columns left to
//! right and rejects any whose residual norm has collapsed, which is what
//! LINPACK's `dqrdc2` does for R's `lm.fit` and hence for limma. Rejected
//! columns report `NaN` for their coefficient and their unscaled standard
//! deviation, as R reports `NA`.
//!
//! ### References
//!
//! Smyth, Statistical Applications in Genetics and Molecular Biology 3(1), 2004
//! Law, Chen, Shi and Smyth, Genome Biology 15, R29, 2014

use rayon::prelude::*;

use crate::errors::EdgeErrors;
use crate::utils::design::non_estimable;
use crate::utils::recycled::{Recycled, RecycledRow};

/// Relative tolerance for declaring a design column linearly dependent.
///
/// R's `lm.fit` factorises with LINPACK's `dqrdc2`, which cycles a column to the
/// back when the norm of what is left of it, after projecting out the columns
/// already accepted, falls below `1e-7` times its original norm. limma inherits
/// that threshold through `lm.fit`, so matching it is what makes the rank
/// reported here the rank R reports.
const RANK_TOL: f64 = 1e-7;

/// Weights at or below this drop the observation rather than scaling it.
///
/// limma's `gls.series` sets `M[weights < 1e-15] <- NA`; `lm.series` sets
/// `weights[weights <= 0] <- NA` and then drops the same observations. One
/// threshold covers both, and the only inputs the two rules disagree on are
/// weights inside `(0, 1e-15]`, which carry no information either way.
const MIN_WEIGHT: f64 = 1e-15;

//////////////////
// LmFitResult  //
//////////////////

/// Everything `lmFit` produces, per gene.
#[derive(Clone, Debug)]
pub struct LmFitResult {
    /// Coefficients, row-major `n_genes * n_coef`.
    ///
    /// `NaN` for a coefficient that is not estimable, either because the design
    /// column is aliased or because the gene lost too many observations. R
    /// reports `NA` in the same places.
    pub coefficients: Vec<f64>,
    /// Unscaled standard deviations, row-major `n_genes * n_coef`.
    ///
    /// The square root of the diagonal of `(X' V^-1 X)^-1`, not of the full
    /// coefficient covariance: the residual scale is carried separately in
    /// [`LmFitResult::sigma`] so that `eBayes` can moderate it.
    pub stdev_unscaled: Vec<f64>,
    /// Residual standard deviation per gene.
    ///
    /// `sqrt(rss / df_residual)` with `rss` the whitened residual sum of
    /// squares, and zero where `df_residual` is zero. limma reports `NA` in that
    /// last case; zero is the one deliberate departure in this module, and it
    /// costs nothing downstream because `squeezeVar` drops zero-df genes before
    /// it looks at their variances.
    pub sigma: Vec<f64>,
    /// Residual degrees of freedom per gene.
    ///
    /// Observations kept minus the rank achieved on them, so a gene with a
    /// dropped sample reports one fewer than its neighbours.
    pub df_residual: Vec<f64>,
    /// Fitted values, row-major `n_genes * n_samples`.
    ///
    /// `X * beta` on the original scale, aliased coefficients contributing
    /// nothing, evaluated at every sample including the dropped ones. A gene
    /// with no usable observation at all is `NaN` throughout.
    pub fitted: Vec<f64>,
}

////////////////////////
// Block correlation  //
////////////////////////

/// The compound-symmetry correlation shared by every gene.
struct BlockCorrelation<'a> {
    /// Block label per sample. Samples sharing a label are correlated.
    block: &'a [usize],
    /// Within-block correlation.
    correlation: f64,
    /// Lower Cholesky factor of the full correlation matrix, row-major
    /// `n_samples * n_samples`. Reused by every fully observed gene.
    chol: Vec<f64>,
}

/// Writes the compound-symmetry correlation matrix for a subset of samples.
///
/// ### Params
///
/// * `out` - Destination, row-major with row stride `stride`
/// * `stride` - Row stride of `out`
/// * `obs` - Sample indices to include, in increasing order
/// * `block` - Block label per sample
/// * `correlation` - Within-block correlation
fn fill_correlation(
    out: &mut [f64],
    stride: usize,
    obs: &[usize],
    block: &[usize],
    correlation: f64,
) {
    for (i, &si) in obs.iter().enumerate() {
        for (j, &sj) in obs.iter().enumerate() {
            out[i * stride + j] = if i == j {
                1.0
            } else if block[si] == block[sj] {
                correlation
            } else {
                0.0
            };
        }
    }
}

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
fn cholesky_lower(a: &mut [f64], n: usize, stride: usize) -> bool {
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
                a[i * stride + i] = sum.sqrt();
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
fn forward_substitute(l: &[f64], stride: usize, n: usize, b: &mut [f64]) {
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i * stride + k] * b[k];
        }
        b[i] = sum / l[i * stride + i];
    }
}

/////////////
// Scratch //
/////////////

/// Per-thread buffers, allocated once per rayon worker rather than per gene.
struct Scratch {
    /// Sample indices this gene actually contributes, in increasing order.
    obs: Vec<usize>,
    /// Whitened design, column-major: column `j` starts at `j * n_samples`.
    ///
    /// Column major because the Householder sweep reads one column at a time
    /// and writes every column to its right.
    a: Vec<f64>,
    /// Whitened response, overwritten by the Householder-transformed one. The
    /// tail below the rank is the residual, exactly R's `effects`.
    z: Vec<f64>,
    /// Householder reflectors, column-major with the same stride as `a`.
    hh: Vec<f64>,
    /// Reciprocal of `u'u / 2` for each stored reflector.
    tau: Vec<f64>,
    /// Original norm of each whitened design column, for the rank test.
    col_norm: Vec<f64>,
    /// Upper triangular factor of the accepted columns, row-major `n_coef`.
    r: Vec<f64>,
    /// Inverse of `r`, row-major `n_coef`.
    r_inv: Vec<f64>,
    /// Original indices of the accepted design columns, increasing.
    est: Vec<usize>,
    /// Solution in the accepted-column basis.
    beta: Vec<f64>,
    /// Correlation matrix for this gene's observed subset, then its Cholesky
    /// factor. Empty unless a block correlation was supplied.
    chol: Vec<f64>,
}

impl Scratch {
    /// Allocates scratch for one worker.
    ///
    /// ### Params
    ///
    /// * `n_samples` - Number of samples
    /// * `n_coef` - Number of coefficients
    /// * `blocked` - Whether a block correlation is in play, which is the only
    ///   case needing an `n_samples` by `n_samples` buffer
    ///
    /// ### Returns
    ///
    /// Buffers sized for a single gene's fit.
    fn new(n_samples: usize, n_coef: usize, blocked: bool) -> Self {
        Self {
            obs: Vec::with_capacity(n_samples),
            a: vec![0.0; n_samples * n_coef],
            z: vec![0.0; n_samples],
            hh: vec![0.0; n_samples * n_coef],
            tau: vec![0.0; n_coef],
            col_norm: vec![0.0; n_coef],
            r: vec![0.0; n_coef * n_coef],
            r_inv: vec![0.0; n_coef * n_coef],
            est: Vec::with_capacity(n_coef),
            beta: vec![0.0; n_coef],
            chol: if blocked {
                vec![0.0; n_samples * n_samples]
            } else {
                Vec::new()
            },
        }
    }
}

///////////////////
// Per-gene fit  //
///////////////////

/// Outputs of one gene's fit, borrowed from the result buffers.
struct GeneOut<'a> {
    /// Coefficients for this gene, length `n_coef`.
    coef: &'a mut [f64],
    /// Unscaled standard deviations, length `n_coef`.
    stdev: &'a mut [f64],
    /// Fitted values, length `n_samples`.
    fitted: &'a mut [f64],
    /// Residual standard deviation.
    sigma: &'a mut f64,
    /// Residual degrees of freedom.
    df: &'a mut f64,
}

/// Fits one gene by whitening, then a rank-revealing Householder QR.
///
/// The whitening turns generalised least squares against `V = D^-1/2 R D^-1/2`
/// into ordinary least squares, so everything after it is the same code whether
/// the caller asked for weights, a block correlation, both, or neither.
///
/// ### Params
///
/// * `scratch` - Per-thread buffers
/// * `y_row` - This gene's response, length `n_samples`
/// * `design` - Row-major design, `n_samples * n_coef`
/// * `n_samples` - Number of samples
/// * `n_coef` - Number of coefficients
/// * `weights` - Optional weight row for this gene
/// * `gls` - Optional block correlation
/// * `design_full_rank` - Whether the design is full rank on all samples, which
///   lets a fully observed gene skip the rank test
/// * `out` - Destination slices for this gene
#[allow(clippy::too_many_arguments)]
fn fit_gene(
    scratch: &mut Scratch,
    y_row: &[f64],
    design: &[f64],
    n_samples: usize,
    n_coef: usize,
    weights: Option<RecycledRow<'_, f64>>,
    gls: Option<&BlockCorrelation<'_>>,
    design_full_rank: bool,
    out: GeneOut<'_>,
) {
    let Scratch {
        obs,
        a,
        z,
        hh,
        tau,
        col_norm,
        r,
        r_inv,
        est,
        beta,
        chol,
    } = scratch;

    // -- which observations survive --
    obs.clear();
    for (sample, &value) in y_row.iter().enumerate() {
        let usable = value.is_finite()
            && match weights {
                Some(w) => {
                    let wj = w.get(sample);
                    wj.is_finite() && wj > MIN_WEIGHT
                }
                None => true,
            };
        if usable {
            obs.push(sample);
        }
    }
    let m = obs.len();

    if m == 0 {
        out.coef.fill(f64::NAN);
        out.stdev.fill(f64::NAN);
        out.fitted.fill(f64::NAN);
        *out.sigma = 0.0;
        *out.df = 0.0;
        return;
    }

    // -- whiten: scale rows by sqrt(w), then apply the inverse Cholesky factor --
    for (i, &sample) in obs.iter().enumerate() {
        let scale = match weights {
            Some(w) => w.get(sample).sqrt(),
            None => 1.0,
        };
        let row = &design[sample * n_coef..(sample + 1) * n_coef];
        for (j, &x) in row.iter().enumerate() {
            a[j * n_samples + i] = scale * x;
        }
        z[i] = scale * y_row[sample];
    }

    if let Some(g) = gls {
        // A principal submatrix of a positive definite matrix is positive
        // definite, so the refactorisation a gene with dropped observations
        // needs cannot fail once the full matrix has been accepted at entry.
        let factor: &[f64] = if m < n_samples {
            fill_correlation(chol, n_samples, obs, g.block, g.correlation);
            let factored = cholesky_lower(chol, m, n_samples);
            debug_assert!(factored, "principal submatrix must be positive definite");
            &chol[..]
        } else {
            &g.chol
        };
        for j in 0..n_coef {
            forward_substitute(
                factor,
                n_samples,
                m,
                &mut a[j * n_samples..j * n_samples + m],
            );
        }
        forward_substitute(factor, n_samples, m, &mut z[..m]);
    }

    // A full rank design stays full rank after whitening, because whitening is
    // an invertible linear map on the sample space. Only a gene that lost
    // observations can lose rank, so only it needs the reference norms.
    let pivot = !design_full_rank || m < n_samples;
    if pivot {
        for j in 0..n_coef {
            let col = &a[j * n_samples..j * n_samples + m];
            col_norm[j] = col.iter().map(|v| v * v).sum::<f64>().sqrt();
        }
    }

    // -- rank-revealing Householder QR, columns accepted left to right --
    est.clear();
    let mut rank = 0usize;
    for j in 0..n_coef {
        if rank == m {
            break;
        }
        let col = &a[j * n_samples..j * n_samples + m];
        let norm = col[rank..].iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm == 0.0 || (pivot && norm <= RANK_TOL * col_norm[j]) {
            continue;
        }

        // Record the column of R above and on the diagonal before the column is
        // consumed by its own reflector.
        for i in 0..rank {
            r[i * n_coef + rank] = a[j * n_samples + i];
        }
        let head = a[j * n_samples + rank];
        let alpha = if head >= 0.0 { -norm } else { norm };
        r[rank * n_coef + rank] = alpha;

        // u = x - alpha e_1, so u'u = 2 * norm * (norm + |head|) > 0.
        let base = rank * n_samples;
        for i in rank..m {
            hh[base + i] = a[j * n_samples + i];
        }
        hh[base + rank] = head - alpha;
        tau[rank] = 1.0 / (norm * (norm + head.abs()));

        // Apply the reflector to every column still to come, and to the
        // response. Columns already passed over are never revisited.
        for jj in (j + 1)..n_coef {
            let cbase = jj * n_samples;
            let mut dot = 0.0;
            for i in rank..m {
                dot += hh[base + i] * a[cbase + i];
            }
            let factor = dot * tau[rank];
            for i in rank..m {
                a[cbase + i] -= factor * hh[base + i];
            }
        }
        let mut dot = 0.0;
        for i in rank..m {
            dot += hh[base + i] * z[i];
        }
        let factor = dot * tau[rank];
        for i in rank..m {
            z[i] -= factor * hh[base + i];
        }

        est.push(j);
        rank += 1;
    }

    // -- residual scale --
    let df = m - rank;
    let rss: f64 = z[rank..m].iter().map(|v| v * v).sum();
    *out.df = df as f64;
    *out.sigma = if df > 0 {
        (rss / df as f64).sqrt()
    } else {
        0.0
    };

    // -- coefficients: back substitution against the leading rank block of R --
    for i in (0..rank).rev() {
        let mut sum = z[i];
        for t in (i + 1)..rank {
            sum -= r[i * n_coef + t] * beta[t];
        }
        beta[i] = sum / r[i * n_coef + i];
    }

    out.coef.fill(f64::NAN);
    for (t, &j) in est.iter().enumerate() {
        out.coef[j] = beta[t];
    }

    // -- unscaled standard deviations: row norms of R^-1 --
    //
    // diag((R'R)^-1) = diag(R^-1 R^-T), and R^-1 is upper triangular, so entry
    // i is the sum of squares of row i of R^-1. Forming the inverse of a tiny
    // triangular matrix is cheaper than a solve per coefficient.
    for i in (0..rank).rev() {
        let inv_diag = 1.0 / r[i * n_coef + i];
        r_inv[i * n_coef + i] = inv_diag;
        for j in (i + 1)..rank {
            let mut sum = 0.0;
            for t in (i + 1)..=j {
                sum += r[i * n_coef + t] * r_inv[t * n_coef + j];
            }
            r_inv[i * n_coef + j] = -sum * inv_diag;
        }
    }

    out.stdev.fill(f64::NAN);
    for (t, &j) in est.iter().enumerate() {
        let row: f64 = (t..rank).map(|s| r_inv[t * n_coef + s].powi(2)).sum();
        out.stdev[j] = row.sqrt();
    }

    // -- fitted values on the original scale, at every sample --
    for (sample, slot) in out.fitted.iter_mut().enumerate() {
        let row = &design[sample * n_coef..(sample + 1) * n_coef];
        let mut acc = 0.0;
        for (t, &j) in est.iter().enumerate() {
            acc += row[j] * beta[t];
        }
        *slot = acc;
    }
}

/////////////
// lm_fit  //
/////////////

/// Fits a linear model to every gene by weighted or generalised least squares.
///
/// Port of limma's `lmFit`, covering the `lm.series` and `gls.series` paths it
/// dispatches to. Each gene is fitted on its own finite, positively weighted
/// observations, so genes with missing values get their own residual degrees of
/// freedom rather than being dropped or imputed.
///
/// Supplying `block` and `correlation` gives the mixed-model form: samples
/// sharing a block label are assumed to have the stated correlation, and the fit
/// is generalised least squares against that compound-symmetry covariance.
/// `correlation` on its own, without `block`, is ignored, which is what `lmFit`
/// does.
///
/// ### Params
///
/// * `y` - Response, row-major `n_genes * n_samples`. Log-CPM for voom, so `f64`
///   throughout rather than generic: there are no counts here to hold in `f32`.
///   Non-finite entries are treated as missing.
/// * `n_genes` - Number of genes
/// * `n_samples` - Number of samples
/// * `design` - Design matrix, row-major `n_samples * n_coef`
/// * `n_coef` - Number of coefficients
/// * `weights` - Optional observation weights, recycled over genes and samples.
///   A weight at or below `1e-15`, or a non-finite one, drops the observation.
/// * `block` - Optional block label per sample
/// * `correlation` - Within-block correlation, required when `block` is given
///
/// ### Returns
///
/// Coefficients, unscaled standard deviations, residual standard deviations,
/// residual degrees of freedom and fitted values, or [`EdgeErrors`] if the
/// shapes disagree, the correlation is out of range, or the implied correlation
/// matrix is not positive definite.
///
/// ### References
///
/// Smyth, Michaud and Scott, Bioinformatics 21(9), 2005 (the blocked form)
#[allow(clippy::too_many_arguments)]
pub fn lm_fit(
    y: &[f64],
    n_genes: usize,
    n_samples: usize,
    design: &[f64],
    n_coef: usize,
    weights: Option<&Recycled<f64>>,
    block: Option<&[usize]>,
    correlation: Option<f64>,
) -> Result<LmFitResult, EdgeErrors> {
    if n_genes == 0 || n_samples == 0 {
        return Err(EdgeErrors::EmptyCounts { n_genes, n_samples });
    }
    if n_coef == 0 {
        return Err(EdgeErrors::MustBePositive("n_coef".to_string()));
    }
    if y.len() != n_genes * n_samples {
        return Err(EdgeErrors::LengthMismatch {
            name: "y",
            expected: n_genes * n_samples,
            got: y.len(),
        });
    }
    if design.len() != n_samples * n_coef {
        return Err(EdgeErrors::LengthMismatch {
            name: "design",
            expected: n_samples * n_coef,
            got: design.len(),
        });
    }
    if let Some(w) = weights {
        w.validate(n_genes, n_samples)?;
    }

    // -- block correlation set-up, done once for all genes --
    let gls = match block {
        None => None,
        Some(b) => {
            if b.len() != n_samples {
                return Err(EdgeErrors::LengthMismatch {
                    name: "block",
                    expected: n_samples,
                    got: b.len(),
                });
            }
            let rho = correlation.ok_or_else(|| {
                EdgeErrors::InvalidArgument(
                    "block requires a correlation; limma would estimate one with duplicateCorrelation"
                        .to_string(),
                )
            })?;
            if !rho.is_finite() || rho.abs() >= 1.0 {
                return Err(EdgeErrors::InvalidArgument(format!(
                    "correlation must be finite and strictly inside (-1, 1); got {rho}"
                )));
            }
            if rho == 0.0 {
                // Uncorrelated blocks give the identity, whose Cholesky factor
                // is the identity, so skip the whitening entirely.
                None
            } else {
                let obs: Vec<usize> = (0..n_samples).collect();
                let mut chol = vec![0.0; n_samples * n_samples];
                fill_correlation(&mut chol, n_samples, &obs, b, rho);
                if !cholesky_lower(&mut chol, n_samples, n_samples) {
                    return Err(EdgeErrors::CholeskyFailed(format!(
                        "correlation {rho} makes the block covariance indefinite; \
                         a block of size k needs correlation above -1/(k - 1)"
                    )));
                }
                Some(BlockCorrelation {
                    block: b,
                    correlation: rho,
                    chol,
                })
            }
        }
    };

    // Columns no gene can estimate. Whitening is an invertible map on the
    // sample space, so a design that is full rank here is full rank for every
    // gene that keeps all its observations, and those genes skip the rank test.
    let design_full_rank = non_estimable(design, n_samples, n_coef)?.is_none();

    let mut coefficients = vec![0.0; n_genes * n_coef];
    let mut stdev_unscaled = vec![0.0; n_genes * n_coef];
    let mut sigma = vec![0.0; n_genes];
    let mut df_residual = vec![0.0; n_genes];
    let mut fitted = vec![0.0; n_genes * n_samples];

    let blocked = gls.is_some();
    coefficients
        .par_chunks_mut(n_coef)
        .zip(stdev_unscaled.par_chunks_mut(n_coef))
        .zip(fitted.par_chunks_mut(n_samples))
        .zip(sigma.par_iter_mut())
        .zip(df_residual.par_iter_mut())
        .enumerate()
        .for_each_init(
            || Scratch::new(n_samples, n_coef, blocked),
            |scratch, (gene, ((((coef, stdev), fit), sig), df))| {
                let start = gene * n_samples;
                fit_gene(
                    scratch,
                    &y[start..start + n_samples],
                    design,
                    n_samples,
                    n_coef,
                    weights.map(|w| w.row(gene, n_samples)),
                    gls.as_ref(),
                    design_full_rank,
                    GeneOut {
                        coef,
                        stdev,
                        fitted: fit,
                        sigma: sig,
                        df,
                    },
                );
            },
        );

    Ok(LmFitResult {
        coefficients,
        stdev_unscaled,
        sigma,
        df_residual,
        fitted,
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    // Every reference below is pasted verbatim from R's 17-digit output. The
    // last digit or two is past what an f64 can hold, and one of them is
    // 1/sqrt(2), which clippy would rather were written as a constant. Keeping
    // them exactly as printed is what makes them checkable against the
    // `Rscript` line quoted above them.
    #![allow(clippy::excessive_precision, clippy::approx_constant)]

    use super::*;
    use approx::assert_relative_eq;

    const N_GENES: usize = 8;
    const N_SAMPLES: usize = 5;

    /// Stand-in for R's `NA` in the embedded reference values.
    const NA: f64 = f64::NAN;

    /// Response fixture, `y[i] = ((i * 7) mod 23) / 64`, row-major 8 by 5.
    ///
    /// Every value is a dyadic rational, so R and Rust see bit-identical inputs
    /// and only the outputs need embedding.
    fn fixture_y() -> Vec<f64> {
        (0..N_GENES * N_SAMPLES)
            .map(|i| ((i * 7) % 23) as f64 / 64.0)
            .collect()
    }

    /// Full weight fixture, `w[i] = (((i * 5) mod 7) + 1) / 8`, row-major 8 by 5.
    fn fixture_weights() -> Vec<f64> {
        (0..N_GENES * N_SAMPLES)
            .map(|i| (((i * 5) % 7) + 1) as f64 / 8.0)
            .collect()
    }

    /// Two-column design: intercept and a two-versus-three group indicator.
    fn design_two() -> Vec<f64> {
        vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    }

    /// Three-column design whose first column is the sum of the other two.
    fn design_aliased() -> Vec<f64> {
        vec![
            1.0, 0.0, 1.0, //
            1.0, 0.0, 1.0, //
            1.0, 1.0, 0.0, //
            1.0, 1.0, 0.0, //
            1.0, 1.0, 0.0,
        ]
    }

    /// Block labels used by every correlated fixture.
    fn blocks() -> Vec<usize> {
        vec![1, 1, 2, 2, 3]
    }

    /// Compares against R, treating `NaN` and `NA` as equal.
    fn assert_close(got: &[f64], want: &[f64]) {
        assert_eq!(got.len(), want.len(), "length");
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            if w.is_nan() {
                assert!(g.is_nan(), "index {i}: expected NA, got {g}");
            } else {
                assert!(g.is_finite(), "index {i}: expected {w}, got {g}");
                assert_relative_eq!(*g, *w, max_relative = 1e-13, epsilon = 0.0);
            }
        }
    }

    // -- reference values --------------------------------------------------
    //
    // Every literal below comes from limma 3.66 under R 4.5. The fixtures are
    // built in R by the same closed forms as `fixture_y` and `fixture_weights`:
    //
    //   library(limma)
    //   Y <- matrix(((0:39)*7) %% 23 / 64, 8, 5, byrow=TRUE)
    //   Wfull <- matrix((((0:39)*5) %% 7 + 1)/8, 8, 5, byrow=TRUE)
    //   X2 <- cbind(1, c(0,0,1,1,1))
    //   X3 <- cbind(1, c(0,0,1,1,1), c(1,1,0,0,0))
    //   wsamp <- c(1, 2, 0.5, 4, 0.25); b <- c(1,1,2,2,3)
    //   Ym <- Y; Ym[4,3] <- NA; Ym[6,] <- NA
    //   f <- lmFit(...)
    //   cat(sprintf("%.17g", as.vector(t(f$coefficients))), sep=", ")
    //   cat(sprintf("%.17g", as.vector(t(f$stdev.unscaled))), sep=", ")
    //   cat(sprintf("%.17g", f$sigma), sep=", ")
    //   cat(sprintf("%.17g", f$df.residual), sep=", ")
    //   cat(sprintf("%.17g", as.vector(t(f$coefficients %*% t(X)))), sep=", ")
    //
    // Where limma reports `sigma = NA` because `df.residual` is zero, the
    // literal below is `0.0` instead: this crate reports zero there, which is
    // the one deliberate departure from limma in this module.

    /// `lmFit(Y, X2)`.
    const A_COEF: [f64; 16] = [
        0.054687499999999972,
        0.15364583333333334,
        0.24218749999999997,
        -0.085937499999999986,
        0.070312499999999972,
        0.15364583333333334,
        0.25781250000000011,
        -0.085937500000000042,
        0.085937499999999972,
        0.033854166666666678,
        0.27343749999999994,
        -0.085937499999999972,
        0.10156249999999997,
        0.033854166666666678,
        0.2890625,
        -0.085937499999999972,
    ];
    const A_SDU: [f64; 16] = [
        0.70710678118654746,
        0.9128709291752769,
        0.70710678118654746,
        0.9128709291752769,
        0.70710678118654746,
        0.9128709291752769,
        0.70710678118654746,
        0.9128709291752769,
        0.70710678118654746,
        0.9128709291752769,
        0.70710678118654746,
        0.9128709291752769,
        0.70710678118654746,
        0.9128709291752769,
        0.70710678118654746,
        0.9128709291752769,
    ];
    const A_SIGMA: [f64; 8] = [
        0.11164557869909383,
        0.099845257878545909,
        0.11164557869909383,
        0.099845257878545896,
        0.11164557869909383,
        0.099845257878545896,
        0.11164557869909383,
        0.099845257878545896,
    ];
    const A_FITTED: [f64; 40] = [
        0.054687499999999972,
        0.054687499999999972,
        0.20833333333333331,
        0.20833333333333331,
        0.20833333333333331,
        0.24218749999999997,
        0.24218749999999997,
        0.15625,
        0.15625,
        0.15625,
        0.070312499999999972,
        0.070312499999999972,
        0.22395833333333331,
        0.22395833333333331,
        0.22395833333333331,
        0.25781250000000011,
        0.25781250000000011,
        0.17187500000000006,
        0.17187500000000006,
        0.17187500000000006,
        0.085937499999999972,
        0.085937499999999972,
        0.11979166666666666,
        0.11979166666666666,
        0.11979166666666666,
        0.27343749999999994,
        0.27343749999999994,
        0.18749999999999997,
        0.18749999999999997,
        0.18749999999999997,
        0.10156249999999997,
        0.10156249999999997,
        0.13541666666666666,
        0.13541666666666666,
        0.13541666666666666,
        0.2890625,
        0.2890625,
        0.20312500000000003,
        0.20312500000000003,
        0.20312500000000003,
    ];

    /// `lmFit(Y, X2, weights = c(1, 2, 0.5, 4, 0.25))`.
    const B_COEF: [f64; 16] = [
        0.072916666666666671,
        0.23053728070175439,
        0.26041666666666663,
        -0.10992324561403506,
        0.088541666666666741,
        0.23053728070175433,
        0.27604166666666669,
        -0.10992324561403508,
        0.10416666666666669,
        -0.07209429824561403,
        0.29166666666666669,
        -0.10992324561403508,
        0.11979166666666664,
        -0.072094298245614016,
        0.30729166666666669,
        -0.10992324561403506,
    ];
    const B_SDU: [f64; 2] = [0.57735026918962584, 0.73746840550820003];
    const B_SIGMA: [f64; 8] = [
        0.094323144471374784,
        0.074810921506209174,
        0.094323144471374742,
        0.074810921506209146,
        0.11154962751509892,
        0.074810921506209146,
        0.11154962751509892,
        0.07481092150620916,
    ];

    /// `lmFit(Y, X2, weights = Wfull)`.
    const C_COEF: [f64; 16] = [
        0.093749999999999972,
        0.066105769230769246,
        0.22851562500000003,
        -0.042436079545454544,
        0.10069444444444443,
        0.15451388888888887,
        0.24687500000000015,
        -0.05156250000000008,
        0.058593750000000035,
        0.084635416666666616,
        0.26432291666666669,
        -0.044010416666666691,
        0.083333333333333343,
        0.070833333333333331,
        0.32812500000000006,
        -0.099759615384615405,
    ];
    const C_SDU: [f64; 16] = [
        1.0690449676496978,
        1.325987088263592,
        1.0,
        1.3142574813455419,
        0.94280904158206336,
        1.3333333333333333,
        0.89442719099991608,
        1.1710800875382399,
        1.4142135623730956,
        1.6329931618554525,
        0.81649658092772603,
        1.2110601416389968,
        1.1547005383792515,
        1.3662601021279464,
        1.0690449676496978,
        1.325987088263592,
    ];
    const C_SIGMA: [f64; 8] = [
        0.07289162230162545,
        0.054945826898183045,
        0.054147465881078799,
        0.066068380670931875,
        0.082088067294221856,
        0.07410330019231505,
        0.091613384135604214,
        0.074599961439633961,
    ];

    /// `lmFit(Ym, X2)`, with `Ym[4, 3]` and all of `Ym[6, ]` set to `NA`.
    const D_COEF: [f64; 16] = [
        0.054687499999999972,
        0.15364583333333334,
        0.24218749999999997,
        -0.085937499999999986,
        0.070312499999999972,
        0.15364583333333334,
        0.2578125,
        -0.031249999999999965,
        0.085937499999999972,
        0.033854166666666678,
        NA,
        NA,
        0.10156249999999997,
        0.033854166666666678,
        0.2890625,
        -0.085937499999999972,
    ];
    const D_SDU: [f64; 16] = [
        0.70710678118654746,
        0.9128709291752769,
        0.70710678118654746,
        0.9128709291752769,
        0.70710678118654746,
        0.9128709291752769,
        0.70710678118654757,
        1.0,
        0.70710678118654746,
        0.9128709291752769,
        NA,
        NA,
        0.70710678118654746,
        0.9128709291752769,
        0.70710678118654746,
        0.9128709291752769,
    ];
    const D_SIGMA: [f64; 8] = [
        0.11164557869909383,
        0.099845257878545909,
        0.11164557869909383,
        0.077339804192278636,
        0.11164557869909383,
        0.0,
        0.11164557869909383,
        0.099845257878545896,
    ];
    const D_DF: [f64; 8] = [3.0, 3.0, 3.0, 2.0, 3.0, 0.0, 3.0, 3.0];
    const D_FITTED: [f64; 40] = [
        0.054687499999999972,
        0.054687499999999972,
        0.20833333333333331,
        0.20833333333333331,
        0.20833333333333331,
        0.24218749999999997,
        0.24218749999999997,
        0.15625,
        0.15625,
        0.15625,
        0.070312499999999972,
        0.070312499999999972,
        0.22395833333333331,
        0.22395833333333331,
        0.22395833333333331,
        0.2578125,
        0.2578125,
        0.22656250000000003,
        0.22656250000000003,
        0.22656250000000003,
        0.085937499999999972,
        0.085937499999999972,
        0.11979166666666666,
        0.11979166666666666,
        0.11979166666666666,
        NA,
        NA,
        NA,
        NA,
        NA,
        0.10156249999999997,
        0.10156249999999997,
        0.13541666666666666,
        0.13541666666666666,
        0.13541666666666666,
        0.2890625,
        0.2890625,
        0.20312500000000003,
        0.20312500000000003,
        0.20312500000000003,
    ];

    /// `lmFit(Ym, X2, weights = Wfull)`.
    const E_COEF: [f64; 16] = [
        0.093749999999999972,
        0.066105769230769246,
        0.22851562500000003,
        -0.042436079545454544,
        0.10069444444444443,
        0.15451388888888887,
        0.24687499999999996,
        -0.029427083333333336,
        0.058593750000000035,
        0.084635416666666616,
        NA,
        NA,
        0.083333333333333343,
        0.070833333333333331,
        0.32812500000000006,
        -0.099759615384615405,
    ];
    const E_SDU: [f64; 16] = [
        1.0690449676496978,
        1.325987088263592,
        1.0,
        1.3142574813455419,
        0.94280904158206336,
        1.3333333333333333,
        0.89442719099991586,
        1.2110601416389966,
        1.4142135623730956,
        1.6329931618554525,
        NA,
        NA,
        1.1547005383792515,
        1.3662601021279464,
        1.0690449676496978,
        1.325987088263592,
    ];
    const E_SIGMA: [f64; 8] = [
        0.07289162230162545,
        0.054945826898183045,
        0.054147465881078799,
        0.063048940228462927,
        0.082088067294221856,
        0.0,
        0.091613384135604214,
        0.074599961439633961,
    ];

    /// `lmFit(Y, X2, block = b, correlation = 0.25)`.
    const F_COEF: [f64; 16] = [
        0.0546875,
        0.14362980769230768,
        0.24218749999999997,
        -0.077524038461538436,
        0.070312500000000042,
        0.14362980769230765,
        0.25781249999999989,
        -0.077524038461538367,
        0.085937499999999958,
        0.033052884615384658,
        0.27343749999999994,
        -0.077524038461538394,
        0.10156249999999996,
        0.033052884615384651,
        0.28906249999999983,
        -0.077524038461538325,
    ];
    const F_SDU: [f64; 2] = [0.79056941504209466, 1.0047961905856253];
    const F_SIGMA: [f64; 8] = [
        0.1146379531823081,
        0.10410655316738308,
        0.11463795318230807,
        0.10410655316738308,
        0.12883085482904671,
        0.10410655316738308,
        0.12883085482904669,
        0.10410655316738311,
    ];

    /// `lmFit(Y, X2, block = b, correlation = 0.25, weights = Wfull)`.
    const G_COEF: [f64; 16] = [
        0.1020338931741292,
        0.046620906898569577,
        0.22414926304156485,
        -0.029290377485346103,
        0.10866723833352257,
        0.1386183553847985,
        0.24332672403599037,
        -0.037918062449936007,
        0.051037727636791902,
        0.097377781866368834,
        0.2613411183871478,
        -0.035655125769476104,
        0.077711629165160057,
        0.080295409798648854,
        0.3364088931741292,
        -0.10120602688561266,
    ];
    const G_SDU: [f64; 16] = [
        1.139580592613717,
        1.4040630530325249,
        1.112163511876666,
        1.4227748106584122,
        1.0256781561509392,
        1.4571513092032287,
        0.99664956319551301,
        1.2759337202972107,
        1.5469735769439759,
        1.7857323811461194,
        0.91075033227467828,
        1.2983569545962401,
        1.2788621058650516,
        1.5060416834761272,
        1.139580592613717,
        1.4040630530325249,
    ];
    const G_SIGMA: [f64; 8] = [
        0.070104095566521502,
        0.055924276262054526,
        0.057844680310825951,
        0.067449568692666786,
        0.094410200626087654,
        0.077120443875363126,
        0.10549134470829188,
        0.07499280377972585,
    ];
    const G_FITTED_GENE0: [f64; 5] = [
        0.1020338931741292,
        0.1020338931741292,
        0.14865480007269877,
        0.14865480007269877,
        0.14865480007269877,
    ];

    /// `lmFit(Ym, X2, block = b, correlation = 0.25, weights = Wfull)`.
    const H_COEF: [f64; 16] = [
        0.1020338931741292,
        0.046620906898569577,
        0.22414926304156485,
        -0.029290377485346103,
        0.10866723833352257,
        0.1386183553847985,
        0.24332672403599037,
        -0.025878807369323673,
        0.051037727636791902,
        0.097377781866368834,
        NA,
        NA,
        0.077711629165160057,
        0.080295409798648854,
        0.3364088931741292,
        -0.10120602688561266,
    ];
    const H_SDU: [f64; 16] = [
        1.139580592613717,
        1.4040630530325249,
        1.112163511876666,
        1.4227748106584122,
        1.0256781561509392,
        1.4571513092032287,
        0.99664956319551323,
        1.288400954083966,
        1.5469735769439759,
        1.7857323811461194,
        NA,
        NA,
        1.2788621058650516,
        1.5060416834761272,
        1.139580592613717,
        1.4040630530325249,
    ];
    const H_SIGMA: [f64; 8] = [
        0.070104095566521502,
        0.055924276262054526,
        0.057844680310825951,
        0.067507835762062693,
        0.094410200626087654,
        0.0,
        0.10549134470829188,
        0.07499280377972585,
    ];

    /// `lmFit(Y, X2, weights = wsamp, block = b, correlation = 0.25)`.
    const K_COEF: [f64; 16] = [
        0.07853837083483986,
        0.23106008862949068,
        0.26603837083483994,
        -0.107400358117879,
        0.094163370834839874,
        0.23106008862949068,
        0.28166337083484,
        -0.10740035811787897,
        0.10978837083483996,
        -0.094332132363834628,
        0.29728837083484005,
        -0.107400358117879,
        0.12541337083483997,
        -0.094332132363834642,
        0.31291337083483994,
        -0.10740035811787904,
    ];
    const K_SDU: [f64; 2] = [0.63943105293252589, 0.80103691702704827];
    const K_SIGMA: [f64; 8] = [
        0.10159645906688854,
        0.081201785694141715,
        0.10159645906688854,
        0.081201785694141715,
        0.12346737328937207,
        0.081201785694141715,
        0.12346737328937207,
        0.081201785694141729,
    ];

    // -- fits --------------------------------------------------------------

    #[test]
    fn test_unweighted_matches_limma() {
        let fit = lm_fit(
            &fixture_y(),
            N_GENES,
            N_SAMPLES,
            &design_two(),
            2,
            None,
            None,
            None,
        )
        .unwrap();
        assert_close(&fit.coefficients, &A_COEF);
        assert_close(&fit.stdev_unscaled, &A_SDU);
        assert_close(&fit.sigma, &A_SIGMA);
        assert_close(&fit.df_residual, &[3.0; 8]);
        assert_close(&fit.fitted, &A_FITTED);
    }

    #[test]
    fn test_sample_weights_match_limma() {
        let w = Recycled::by_sample(vec![1.0, 2.0, 0.5, 4.0, 0.25]);
        let fit = lm_fit(
            &fixture_y(),
            N_GENES,
            N_SAMPLES,
            &design_two(),
            2,
            Some(&w),
            None,
            None,
        )
        .unwrap();
        assert_close(&fit.coefficients, &B_COEF);
        assert_close(&fit.sigma, &B_SIGMA);
        // Per-sample weights are shared by every gene, so is the design metric.
        for row in fit.stdev_unscaled.chunks_exact(2) {
            assert_close(row, &B_SDU);
        }
    }

    #[test]
    fn test_full_weights_match_limma() {
        let w = Recycled::full(fixture_weights(), N_GENES, N_SAMPLES).unwrap();
        let fit = lm_fit(
            &fixture_y(),
            N_GENES,
            N_SAMPLES,
            &design_two(),
            2,
            Some(&w),
            None,
            None,
        )
        .unwrap();
        assert_close(&fit.coefficients, &C_COEF);
        assert_close(&fit.stdev_unscaled, &C_SDU);
        assert_close(&fit.sigma, &C_SIGMA);
        assert_close(&fit.df_residual, &[3.0; 8]);
    }

    #[test]
    fn test_missing_value_refits_row() {
        let mut y = fixture_y();
        y[3 * N_SAMPLES + 2] = f64::NAN;
        for s in 0..N_SAMPLES {
            y[5 * N_SAMPLES + s] = f64::NAN;
        }
        let fit = lm_fit(&y, N_GENES, N_SAMPLES, &design_two(), 2, None, None, None).unwrap();
        assert_close(&fit.coefficients, &D_COEF);
        assert_close(&fit.stdev_unscaled, &D_SDU);
        assert_close(&fit.sigma, &D_SIGMA);
        assert_close(&fit.df_residual, &D_DF);
        assert_close(&fit.fitted, &D_FITTED);
    }

    #[test]
    fn test_missing_value_with_full_weights() {
        let mut y = fixture_y();
        y[3 * N_SAMPLES + 2] = f64::NEG_INFINITY;
        for s in 0..N_SAMPLES {
            y[5 * N_SAMPLES + s] = f64::NAN;
        }
        let w = Recycled::full(fixture_weights(), N_GENES, N_SAMPLES).unwrap();
        let fit = lm_fit(
            &y,
            N_GENES,
            N_SAMPLES,
            &design_two(),
            2,
            Some(&w),
            None,
            None,
        )
        .unwrap();
        assert_close(&fit.coefficients, &E_COEF);
        assert_close(&fit.stdev_unscaled, &E_SDU);
        assert_close(&fit.sigma, &E_SIGMA);
        assert_close(&fit.df_residual, &D_DF);
    }

    #[test]
    fn test_zero_weight_drops_the_observation() {
        // limma sets the response to NA wherever the weight vanishes, so a zero
        // weight must reproduce the missing-value fit exactly.
        let mut weights = fixture_weights();
        weights[3 * N_SAMPLES + 2] = 0.0;
        for s in 0..N_SAMPLES {
            weights[5 * N_SAMPLES + s] = 0.0;
        }
        let w = Recycled::full(weights, N_GENES, N_SAMPLES).unwrap();
        let fit = lm_fit(
            &fixture_y(),
            N_GENES,
            N_SAMPLES,
            &design_two(),
            2,
            Some(&w),
            None,
            None,
        )
        .unwrap();
        assert_close(&fit.coefficients, &E_COEF);
        assert_close(&fit.stdev_unscaled, &E_SDU);
        assert_close(&fit.sigma, &E_SIGMA);
        assert_close(&fit.df_residual, &D_DF);
    }

    #[test]
    fn test_block_correlation_matches_limma() {
        let fit = lm_fit(
            &fixture_y(),
            N_GENES,
            N_SAMPLES,
            &design_two(),
            2,
            None,
            Some(&blocks()),
            Some(0.25),
        )
        .unwrap();
        assert_close(&fit.coefficients, &F_COEF);
        assert_close(&fit.sigma, &F_SIGMA);
        assert_close(&fit.df_residual, &[3.0; 8]);
        for row in fit.stdev_unscaled.chunks_exact(2) {
            assert_close(row, &F_SDU);
        }
    }

    #[test]
    fn test_block_correlation_with_full_weights() {
        let w = Recycled::full(fixture_weights(), N_GENES, N_SAMPLES).unwrap();
        let fit = lm_fit(
            &fixture_y(),
            N_GENES,
            N_SAMPLES,
            &design_two(),
            2,
            Some(&w),
            Some(&blocks()),
            Some(0.25),
        )
        .unwrap();
        assert_close(&fit.coefficients, &G_COEF);
        assert_close(&fit.stdev_unscaled, &G_SDU);
        assert_close(&fit.sigma, &G_SIGMA);
        assert_close(&fit.fitted[..N_SAMPLES], &G_FITTED_GENE0);
    }

    #[test]
    fn test_block_correlation_with_sample_weights() {
        let w = Recycled::by_sample(vec![1.0, 2.0, 0.5, 4.0, 0.25]);
        let fit = lm_fit(
            &fixture_y(),
            N_GENES,
            N_SAMPLES,
            &design_two(),
            2,
            Some(&w),
            Some(&blocks()),
            Some(0.25),
        )
        .unwrap();
        assert_close(&fit.coefficients, &K_COEF);
        assert_close(&fit.sigma, &K_SIGMA);
        for row in fit.stdev_unscaled.chunks_exact(2) {
            assert_close(row, &K_SDU);
        }
    }

    #[test]
    fn test_block_correlation_with_missing_refits_submatrix() {
        let mut y = fixture_y();
        y[3 * N_SAMPLES + 2] = f64::NAN;
        for s in 0..N_SAMPLES {
            y[5 * N_SAMPLES + s] = f64::NAN;
        }
        let w = Recycled::full(fixture_weights(), N_GENES, N_SAMPLES).unwrap();
        let fit = lm_fit(
            &y,
            N_GENES,
            N_SAMPLES,
            &design_two(),
            2,
            Some(&w),
            Some(&blocks()),
            Some(0.25),
        )
        .unwrap();
        assert_close(&fit.coefficients, &H_COEF);
        assert_close(&fit.stdev_unscaled, &H_SDU);
        assert_close(&fit.sigma, &H_SIGMA);
        assert_close(&fit.df_residual, &D_DF);
    }

    #[test]
    fn test_zero_correlation_reduces_to_least_squares() {
        let fit = lm_fit(
            &fixture_y(),
            N_GENES,
            N_SAMPLES,
            &design_two(),
            2,
            None,
            Some(&blocks()),
            Some(0.0),
        )
        .unwrap();
        assert_close(&fit.coefficients, &A_COEF);
        assert_close(&fit.sigma, &A_SIGMA);
    }

    #[test]
    fn test_correlation_without_block_is_ignored() {
        // `lmFit(Y, X2, correlation = 0.25)` dispatches to `lm.series`, which
        // never sees the correlation.
        let fit = lm_fit(
            &fixture_y(),
            N_GENES,
            N_SAMPLES,
            &design_two(),
            2,
            None,
            None,
            Some(0.25),
        )
        .unwrap();
        assert_close(&fit.coefficients, &A_COEF);
        assert_close(&fit.sigma, &A_SIGMA);
    }

    #[test]
    fn test_rank_deficient_design_aliases_the_last_column() {
        // R's `lm.fit` accepts columns left to right, so the third column is the
        // one dropped: `lmFit` prints "Coefficients not estimable: 3".
        let fit = lm_fit(
            &fixture_y(),
            N_GENES,
            N_SAMPLES,
            &design_aliased(),
            3,
            None,
            None,
            None,
        )
        .unwrap();
        for gene in 0..N_GENES {
            let coef = &fit.coefficients[gene * 3..(gene + 1) * 3];
            let sdu = &fit.stdev_unscaled[gene * 3..(gene + 1) * 3];
            assert_close(&coef[..2], &A_COEF[gene * 2..(gene + 1) * 2]);
            assert_close(&sdu[..2], &A_SDU[gene * 2..(gene + 1) * 2]);
            assert!(coef[2].is_nan());
            assert!(sdu[2].is_nan());
        }
        assert_close(&fit.sigma, &A_SIGMA);
        assert_close(&fit.df_residual, &[3.0; 8]);
        // limma's `fitted()` would propagate the NA coefficient across the whole
        // row; here the aliased column contributes nothing, which recovers the
        // projection the estimable columns actually give.
        assert_close(&fit.fitted, &A_FITTED);
    }

    #[test]
    fn test_single_gene() {
        let y: Vec<f64> = fixture_y()[..N_SAMPLES].to_vec();
        let fit = lm_fit(&y, 1, N_SAMPLES, &design_two(), 2, None, None, None).unwrap();
        assert_close(&fit.coefficients, &A_COEF[..2]);
        assert_close(&fit.stdev_unscaled, &A_SDU[..2]);
        assert_close(&fit.sigma, &A_SIGMA[..1]);
        assert_close(&fit.df_residual, &[3.0]);
        assert_close(&fit.fitted, &A_FITTED[..N_SAMPLES]);
    }

    #[test]
    fn test_saturated_design_gives_zero_sigma() {
        // Five coefficients on five samples: nothing left over, so limma would
        // report NA and this crate reports zero.
        let mut design = vec![0.0; N_SAMPLES * N_SAMPLES];
        for s in 0..N_SAMPLES {
            design[s * N_SAMPLES + s] = 1.0;
        }
        let y = fixture_y();
        let fit = lm_fit(&y, N_GENES, N_SAMPLES, &design, N_SAMPLES, None, None, None).unwrap();
        assert_close(&fit.df_residual, &[0.0; 8]);
        assert_close(&fit.sigma, &[0.0; 8]);
        assert_close(&fit.coefficients, &y);
        assert_close(&fit.fitted, &y);
    }

    // -- error branches ----------------------------------------------------

    #[test]
    fn test_rejects_empty_input() {
        assert!(matches!(
            lm_fit(&[], 0, N_SAMPLES, &design_two(), 2, None, None, None),
            Err(EdgeErrors::EmptyCounts { n_genes: 0, .. })
        ));
        assert!(matches!(
            lm_fit(&[], N_GENES, 0, &design_two(), 2, None, None, None),
            Err(EdgeErrors::EmptyCounts { n_samples: 0, .. })
        ));
    }

    #[test]
    fn test_rejects_zero_coefficients() {
        assert!(matches!(
            lm_fit(&fixture_y(), N_GENES, N_SAMPLES, &[], 0, None, None, None),
            Err(EdgeErrors::MustBePositive(_))
        ));
    }

    #[test]
    fn test_rejects_wrong_response_length() {
        assert!(matches!(
            lm_fit(
                &[0.0; 7],
                N_GENES,
                N_SAMPLES,
                &design_two(),
                2,
                None,
                None,
                None
            ),
            Err(EdgeErrors::LengthMismatch {
                name: "y",
                expected: 40,
                got: 7
            })
        ));
    }

    #[test]
    fn test_rejects_wrong_design_length() {
        assert!(matches!(
            lm_fit(
                &fixture_y(),
                N_GENES,
                N_SAMPLES,
                &[1.0; 8],
                2,
                None,
                None,
                None
            ),
            Err(EdgeErrors::LengthMismatch {
                name: "design",
                expected: 10,
                got: 8
            })
        ));
    }

    #[test]
    fn test_rejects_mismatched_weights() {
        let w = Recycled::by_sample(vec![1.0, 2.0]);
        assert!(matches!(
            lm_fit(
                &fixture_y(),
                N_GENES,
                N_SAMPLES,
                &design_two(),
                2,
                Some(&w),
                None,
                None
            ),
            Err(EdgeErrors::LengthMismatch {
                name: "by_sample",
                ..
            })
        ));
    }

    #[test]
    fn test_rejects_wrong_block_length() {
        assert!(matches!(
            lm_fit(
                &fixture_y(),
                N_GENES,
                N_SAMPLES,
                &design_two(),
                2,
                None,
                Some(&[1, 1, 2]),
                Some(0.25)
            ),
            Err(EdgeErrors::LengthMismatch {
                name: "block",
                expected: 5,
                got: 3
            })
        ));
    }

    #[test]
    fn test_rejects_block_without_correlation() {
        assert!(matches!(
            lm_fit(
                &fixture_y(),
                N_GENES,
                N_SAMPLES,
                &design_two(),
                2,
                None,
                Some(&blocks()),
                None
            ),
            Err(EdgeErrors::InvalidArgument(_))
        ));
    }

    #[test]
    fn test_rejects_out_of_range_correlation() {
        for rho in [1.0, -1.0, 2.5, f64::NAN] {
            assert!(matches!(
                lm_fit(
                    &fixture_y(),
                    N_GENES,
                    N_SAMPLES,
                    &design_two(),
                    2,
                    None,
                    Some(&blocks()),
                    Some(rho)
                ),
                Err(EdgeErrors::InvalidArgument(_))
            ));
        }
    }

    #[test]
    fn test_rejects_indefinite_block_covariance() {
        // One block of five needs a correlation above -1/4; limma's `chol` fails
        // at -0.5 too.
        assert!(matches!(
            lm_fit(
                &fixture_y(),
                N_GENES,
                N_SAMPLES,
                &design_two(),
                2,
                None,
                Some(&[1, 1, 1, 1, 1]),
                Some(-0.5)
            ),
            Err(EdgeErrors::CholeskyFailed(_))
        ));
    }
}
