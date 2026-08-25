//! Levenberg-damped iteratively reweighted least squares for negative binomial
//! GLMs.
//!
//! This is edgeR's `mglmLevenberg` and the hottest path in the crate: one fit
//! per gene, tens of thousands of genes, a design of a handful of columns.
//!
//! ### References
//!
//! McCarthy, Chen and Smyth, Nucleic Acids Research 40(10), 2012

use rayon::prelude::*;

use crate::core::expression::check_dispersion;
use crate::errors::EdgeErrors;
use crate::glm::deviance::unit_nb_deviance;
use crate::glm::{EMPTY_GENE_COEF, ETA_CLAMP, MIN_POSITIVE};
use crate::utils::recycled::{Recycled, RecycledRow};
use crate::utils::simd::EdgeSimd;
use crate::utils::traits::EdgeFloat;

////////////
// Consts //
////////////

/// Starting Levenberg damping factor.
const INITIAL_DAMPING: f64 = 1e-3;

/// Smallest damping factor, reached after repeated accepted steps.
const MIN_DAMPING: f64 = 1e-10;

/// Largest damping factor, reached after repeated rejections.
///
/// At this point the step is effectively along the gradient with a tiny length,
/// and the fit is not going to improve.
const MAX_DAMPING: f64 = 1e10;

/// Added to the diagonal before damping so an all-zero column cannot make the
/// system singular.
const DIAGONAL_FLOOR: f64 = 1e-10;

/////////////////
// StartMethod //
/////////////////

/// How the coefficients are initialised.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StartMethod {
    /// Fit an intercept-only model from the total counts and total library size.
    ///
    /// Cheap and robust, and what edgeR uses by default.
    Null,
    /// Regress the log of the normalised counts on the design.
    ///
    /// Closer to the answer when the design explains a lot, but undefined for
    /// zero counts, which have to be floored first.
    LogCounts,
}

/////////////////////
// LevenbergParams //
/////////////////////

/// Tuning knobs for [`mglm_levenberg`].
#[derive(Clone, Copy, Debug)]
pub struct LevenbergParams {
    /// Iteration budget per gene.
    pub max_iter: usize,
    /// Relative tolerance on the deviance.
    ///
    /// A gene stops when an accepted step changes the deviance by less than
    /// `tol * (|deviance| + 0.1)`. The additive term is what lets a gene whose
    /// deviance is legitimately near zero converge at all.
    pub tol: f64,
    /// How to initialise the coefficients.
    pub start_method: StartMethod,
}

impl Default for LevenbergParams {
    /// edgeR's defaults. `glm_fit` raises `max_iter` to 250 for the general
    /// path; the 200 here is `mglmLevenberg`'s own default.
    fn default() -> Self {
        Self {
            max_iter: 200,
            tol: 1e-6,
            start_method: StartMethod::Null,
        }
    }
}

//////////////////
// LevenbergFit //
//////////////////

/// Result of fitting every gene.
#[derive(Clone, Debug)]
pub struct LevenbergFit {
    /// Coefficients, row-major `n_genes * n_coef`.
    pub coefficients: Vec<f64>,
    /// Fitted means, row-major `n_genes * n_samples`.
    pub fitted: Vec<f64>,
    /// Residual deviance per gene.
    pub deviance: Vec<f64>,
    /// Iterations used per gene. A gene that exhausted the budget reports it.
    pub iterations: Vec<usize>,
    /// Genes whose normal equations went singular and fell back to the last
    /// good coefficients.
    pub failed: Vec<bool>,
}

/////////////
// Scratch //
/////////////

/// Per-thread scratch, allocated once per rayon worker rather than per gene.
pub struct Scratch {
    /// Counts for the current gene, widened to `f64`.
    pub y: Vec<f64>,
    /// Linear predictor.
    eta: Vec<f64>,
    /// Fitted mean.
    pub mu: Vec<f64>,
    /// Trial fitted mean for the proposed step.
    mu_trial: Vec<f64>,
    /// IRLS working weights.
    working: Vec<f64>,
    /// Normal equations, row-major `n_coef * n_coef`.
    xtwx: Vec<f64>,
    /// Damped copy of the normal equations, overwritten by the factorisation.
    damped: Vec<f64>,
    /// Right-hand side of the normal equations.
    xtwz: Vec<f64>,
    /// Proposed coefficient update.
    delta: Vec<f64>,
    /// Proposed coefficients.
    beta_trial: Vec<f64>,
}

impl Scratch {
    /// Allocates scratch for one worker.
    ///
    /// ### Params
    ///
    /// * `n_samples` - Number of samples
    /// * `n_coef` - Number of coefficients
    ///
    /// ### Returns
    ///
    /// Buffers sized for a single gene's fit.
    pub fn new(n_samples: usize, n_coef: usize) -> Self {
        Self {
            y: vec![0.0; n_samples],
            eta: vec![0.0; n_samples],
            mu: vec![0.0; n_samples],
            mu_trial: vec![0.0; n_samples],
            working: vec![0.0; n_samples],
            xtwx: vec![0.0; n_coef * n_coef],
            damped: vec![0.0; n_coef * n_coef],
            xtwz: vec![0.0; n_coef],
            delta: vec![0.0; n_coef],
            beta_trial: vec![0.0; n_coef],
        }
    }
}

/////////////
// Helpers //
/////////////

/// Solves a small symmetric positive definite system in place by Cholesky.
///
/// The systems here are `n_coef` by `n_coef` with `n_coef` in the low single
/// digits for essentially every RNA-seq design, so a factorisation call into
/// faer would be dominated by its own dispatch overhead. Row-major throughout.
///
/// ### Params
///
/// * `a` - The matrix, overwritten by its lower Cholesky factor
/// * `b` - Right-hand side, overwritten by the solution
/// * `n` - System size
///
/// ### Returns
///
/// `false` if a pivot is not positive, meaning the matrix is not positive
/// definite and the caller should treat the step as failed.
fn solve_spd_in_place(a: &mut [f64], b: &mut [f64], n: usize) -> bool {
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= a[i * n + k] * a[j * n + k];
            }
            if i == j {
                // Rejects NaN as well as a non-positive pivot, which a bare
                // `sum > 0.0` negation would hide.
                if sum <= 0.0 || sum.is_nan() {
                    return false;
                }
                a[i * n + i] = sum.sqrt();
            } else {
                a[i * n + j] = sum / a[j * n + j];
            }
        }
    }

    // Forward substitution.
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum -= a[i * n + k] * b[k];
        }
        b[i] = sum / a[i * n + i];
    }
    // Back substitution against the transpose.
    for i in (0..n).rev() {
        let mut sum = b[i];
        for k in (i + 1)..n {
            sum -= a[k * n + i] * b[k];
        }
        b[i] = sum / a[i * n + i];
    }
    true
}

/// Fills `eta` with `design * beta + offset`, then exponentiates into `mu`.
///
/// ### Params
///
/// * `design` - Row-major design, `n_samples * n_coef`
/// * `beta` - Coefficients, length `n_coef`
/// * `offset` - Offsets for this gene
/// * `n_samples` - Number of samples
/// * `n_coef` - Number of coefficients
/// * `eta` - Scratch for the linear predictor
/// * `mu` - Output fitted means
fn linear_predictor(
    design: &[f64],
    beta: &[f64],
    offset: RecycledRow<'_, f64>,
    n_samples: usize,
    n_coef: usize,
    eta: &mut [f64],
    mu: &mut [f64],
) {
    for (j, eta_j) in eta.iter_mut().enumerate().take(n_samples) {
        let row = &design[j * n_coef..(j + 1) * n_coef];
        *eta_j = (f64::dot_simd(row, beta) + offset.get(j)).clamp(-ETA_CLAMP, ETA_CLAMP);
    }
    mu[..n_samples].copy_from_slice(&eta[..n_samples]);
    f64::exp_in_place_simd(&mut mu[..n_samples]);
    for m in mu.iter_mut().take(n_samples) {
        *m = m.max(MIN_POSITIVE);
    }
}

/// Weighted residual deviance of one gene.
///
/// ### Params
///
/// * `y` - Counts
/// * `mu` - Fitted means
/// * `dispersion` - Dispersion row for this gene
/// * `weights` - Optional weight row for this gene
///
/// ### Returns
///
/// The weighted sum of unit deviances.
fn deviance_of(
    y: &[f64],
    mu: &[f64],
    dispersion: RecycledRow<'_, f64>,
    weights: Option<RecycledRow<'_, f64>>,
) -> f64 {
    y.iter()
        .zip(mu.iter())
        .enumerate()
        .map(|(j, (&y_j, &mu_j))| {
            let d = unit_nb_deviance(y_j, mu_j, dispersion.get(j));
            match weights {
                Some(w) => w.get(j) * d,
                None => d,
            }
        })
        .sum()
}

/// Least-squares projection of a per-sample target onto the design.
///
/// Solves `design * beta = z` in the least-squares sense through the normal
/// equations, which is what edgeR's start does with a QR. The system is
/// `n_coef` square and `n_coef` is a handful of columns for any real design, so
/// the normal equations cost nothing and lose no accuracy that survives the
/// damped iteration afterwards.
///
/// ### Params
///
/// * `design` - Row-major design, `n_samples * n_coef`
/// * `z` - Target, one value per sample
/// * `n_samples` - Number of samples
/// * `n_coef` - Number of coefficients
/// * `beta` - Output, length `n_coef`
///
/// ### Returns
///
/// `false` if the design is rank deficient, leaving `beta` untouched.
fn project_onto_design(
    design: &[f64],
    z: &[f64],
    n_samples: usize,
    n_coef: usize,
    beta: &mut [f64],
) -> bool {
    let mut xtx = vec![0.0; n_coef * n_coef];
    let mut xtz = vec![0.0; n_coef];

    for j in 0..n_samples {
        let row = &design[j * n_coef..(j + 1) * n_coef];
        for (a, &x_a) in row.iter().enumerate() {
            xtz[a] += x_a * z[j];
            for (b, &x_b) in row[..=a].iter().enumerate() {
                xtx[a * n_coef + b] += x_a * x_b;
            }
        }
    }
    for a in 0..n_coef {
        for b in 0..a {
            xtx[b * n_coef + a] = xtx[a * n_coef + b];
        }
    }

    if !solve_spd_in_place(&mut xtx, &mut xtz, n_coef) {
        return false;
    }
    beta.copy_from_slice(&xtz);
    true
}

/// Chooses starting coefficients for one gene.
///
/// Port of edgeR's `.cxx_get_levenberg_start`. Both methods build a per-sample
/// target on the log scale and project it onto the design by least squares:
///
/// * [`StartMethod::Null`] uses a single working-weighted log rate shared by
///   every sample, `log(sum(cw * y/N) / sum(cw))` with `cw = w N / (1 + phi N)`.
///   That weighting is why the dispersion and the weights are needed; the
///   unweighted ratio of totals agrees only when every library is the same size.
/// * [`StartMethod::LogCounts`] uses each sample's own log rate.
///
/// The projection is the part that is easy to skip and must not be. Writing the
/// scalar into `beta[0]` and zeroing the rest is only equivalent when the first
/// design column is an intercept. On `~0 + group + covariate` it leaves every
/// sample outside the first column starting at `mu` equal to its library size,
/// which for a dispersion below
/// [`crate::glm::deviance::POISSON_REGIME`] sits inside the region where the
/// unit deviance underflows to zero. That is the global minimum of the fitter's
/// objective, so the iteration never leaves it.
///
/// ### Params
///
/// * `y` - Counts for this gene
/// * `design` - Row-major design, `n_samples * n_coef`
/// * `offset` - Log-scale offset row
/// * `dispersion` - Dispersion row, used for the working weights
/// * `weights` - Optional observation weight row
/// * `n_samples` - Number of samples
/// * `n_coef` - Number of coefficients
/// * `method` - Which initialisation to use
/// * `beta` - Output, length `n_coef`
#[allow(clippy::too_many_arguments)]
pub(crate) fn initial_coefficients(
    y: &[f64],
    design: &[f64],
    offset: RecycledRow<'_, f64>,
    dispersion: RecycledRow<'_, f64>,
    weights: Option<RecycledRow<'_, f64>>,
    n_samples: usize,
    n_coef: usize,
    method: StartMethod,
    beta: &mut [f64],
) {
    beta.fill(0.0);

    let target: Vec<f64> = match method {
        StartMethod::Null => {
            let mut num = 0.0;
            let mut den = 0.0;
            for (j, &y_j) in y.iter().enumerate() {
                let library = offset.get(j).exp();
                let w = weights.map_or(1.0, |w| w.get(j));
                let cw = w * library / (1.0 + dispersion.get(j) * library);
                if cw.is_finite() && library > 0.0 {
                    num += cw * y_j / library;
                    den += cw;
                }
            }
            let rate = if num > 0.0 && den > 0.0 {
                (num / den).max(MIN_POSITIVE).ln()
            } else {
                EMPTY_GENE_COEF
            };
            vec![rate; n_samples]
        }
        StartMethod::LogCounts => (0..n_samples)
            .map(|j| {
                let library = offset.get(j).exp().max(MIN_POSITIVE);
                (y[j] / library).max(MIN_POSITIVE).ln()
            })
            .collect(),
    };

    if !project_onto_design(design, &target, n_samples, n_coef, beta) {
        // Rank deficient. edgeR's QR pivots through this; the normal equations
        // cannot, so fall back to putting the whole rate on the first column.
        // The damped iteration recovers from a poor start, just not from one
        // inside the zero-deviance basin, and a rank-deficient design has no
        // such basin to fall into.
        beta.fill(0.0);
        beta[0] = target[0];
    }
}

/// Runs the damped iteration for a single gene.
///
/// ### Params
///
/// * `scratch` - Per-thread buffers
/// * `beta` - Coefficients, updated in place
/// * `design` - Row-major design
/// * `n_samples` - Number of samples
/// * `n_coef` - Number of coefficients
/// * `dispersion` - Dispersion row
/// * `offset` - Offset row
/// * `weights` - Optional weight row
/// * `params` - Tuning knobs
///
/// ### Returns
///
/// The number of iterations used and whether the normal equations ever went
/// singular.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fit_one_gene(
    scratch: &mut Scratch,
    beta: &mut [f64],
    design: &[f64],
    n_samples: usize,
    n_coef: usize,
    dispersion: RecycledRow<'_, f64>,
    offset: RecycledRow<'_, f64>,
    weights: Option<RecycledRow<'_, f64>>,
    params: &LevenbergParams,
) -> (usize, bool) {
    let mut damping = INITIAL_DAMPING;
    let mut singular = false;

    linear_predictor(
        design,
        beta,
        offset,
        n_samples,
        n_coef,
        &mut scratch.eta,
        &mut scratch.mu,
    );
    let mut dev_current = deviance_of(&scratch.y, &scratch.mu, dispersion, weights);

    for iteration in 0..params.max_iter {
        // Working weights and residuals.
        for j in 0..n_samples {
            let mu_j = scratch.mu[j];
            let w_j = weights.map_or(1.0, |w| w.get(j));
            let denom = 1.0 + dispersion.get(j) * mu_j;
            scratch.working[j] = (w_j * mu_j / denom).max(MIN_POSITIVE);
        }

        // Normal equations. The design row is reused for both accumulations, so
        // this is one pass over the samples rather than two.
        scratch.xtwx.fill(0.0);
        scratch.xtwz.fill(0.0);
        for j in 0..n_samples {
            let row = &design[j * n_coef..(j + 1) * n_coef];
            let w_j = scratch.working[j];
            let z_j = (scratch.y[j] - scratch.mu[j]) / scratch.mu[j];
            for (a, &x_a) in row.iter().enumerate() {
                let wx = w_j * x_a;
                scratch.xtwz[a] += wx * z_j;
                // Lower triangle only; the upper is mirrored afterwards.
                for (b, &x_b) in row[..=a].iter().enumerate() {
                    scratch.xtwx[a * n_coef + b] += wx * x_b;
                }
            }
        }
        // Mirror the lower triangle into the upper.
        for a in 0..n_coef {
            for b in 0..a {
                scratch.xtwx[b * n_coef + a] = scratch.xtwx[a * n_coef + b];
            }
        }

        scratch.damped.copy_from_slice(&scratch.xtwx);
        for a in 0..n_coef {
            scratch.damped[a * n_coef + a] +=
                damping * (scratch.xtwx[a * n_coef + a] + DIAGONAL_FLOOR);
        }
        scratch.delta.copy_from_slice(&scratch.xtwz);

        if !solve_spd_in_place(&mut scratch.damped, &mut scratch.delta, n_coef) {
            // Not positive definite: damp harder and try again.
            singular = true;
            damping = (damping * 10.0).min(MAX_DAMPING);
            if damping >= MAX_DAMPING {
                return (iteration + 1, singular);
            }
            continue;
        }

        for ((trial, &current), &step) in scratch
            .beta_trial
            .iter_mut()
            .zip(beta.iter())
            .zip(scratch.delta.iter())
        {
            *trial = current + step;
        }
        linear_predictor(
            design,
            &scratch.beta_trial,
            offset,
            n_samples,
            n_coef,
            &mut scratch.eta,
            &mut scratch.mu_trial,
        );
        let dev_trial = deviance_of(&scratch.y, &scratch.mu_trial, dispersion, weights);

        if dev_trial <= dev_current {
            beta.copy_from_slice(&scratch.beta_trial);
            scratch.mu.copy_from_slice(&scratch.mu_trial);
            damping = (damping / 10.0).max(MIN_DAMPING);

            let change = (dev_current - dev_trial).abs();
            let threshold = params.tol * (dev_current.abs() + 0.1);
            dev_current = dev_trial;
            if change < threshold {
                return (iteration + 1, singular);
            }
        } else {
            damping = (damping * 10.0).min(MAX_DAMPING);
        }
    }

    (params.max_iter, singular)
}

///////////////
// Front end //
///////////////

/// Fits genewise negative binomial GLMs with Levenberg damping.
///
/// Each gene is fitted independently, so the work is a plain rayon fan-out over
/// genes with no shared mutable state. Damping starts at
/// [`INITIAL_DAMPING`], divides by ten on an accepted step and multiplies by ten
/// on a rejected one, exactly as edgeR does. A step is accepted when it does not
/// increase the deviance, which is what makes the method monotone and removes
/// the need for a line search.
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
/// * `coef_start` - Optional starting coefficients, row-major `n_genes * n_coef`
/// * `params` - Tuning knobs, or [`LevenbergParams::default`]
///
/// ### Returns
///
/// Coefficients, fitted means, residual deviances and per-gene iteration counts,
/// or [`EdgeErrors`] if the shapes disagree or a dispersion is negative.
#[allow(clippy::too_many_arguments)]
pub fn mglm_levenberg<T: EdgeFloat>(
    counts: &[T],
    n_genes: usize,
    n_samples: usize,
    design: &[f64],
    n_coef: usize,
    dispersion: &Recycled<f64>,
    offset: &Recycled<f64>,
    weights: Option<&Recycled<f64>>,
    coef_start: Option<&[f64]>,
    params: Option<LevenbergParams>,
) -> Result<LevenbergFit, EdgeErrors> {
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
    if n_coef == 0 {
        return Err(EdgeErrors::MustBePositive("n_coef".to_string()));
    }
    dispersion.validate(n_genes, n_samples)?;
    check_dispersion(dispersion)?;
    offset.validate(n_genes, n_samples)?;
    if let Some(w) = weights {
        w.validate(n_genes, n_samples)?;
    }
    if let Some(start) = coef_start
        && start.len() != n_genes * n_coef
    {
        return Err(EdgeErrors::LengthMismatch {
            name: "coef_start",
            expected: n_genes * n_coef,
            got: start.len(),
        });
    }

    let mut coefficients = vec![0.0; n_genes * n_coef];
    let mut fitted = vec![0.0; n_genes * n_samples];
    let mut deviance = vec![0.0; n_genes];
    let mut iterations = vec![0usize; n_genes];
    let mut failed = vec![false; n_genes];

    coefficients
        .par_chunks_mut(n_coef)
        .zip(fitted.par_chunks_mut(n_samples))
        .zip(deviance.par_iter_mut())
        .zip(iterations.par_iter_mut())
        .zip(failed.par_iter_mut())
        .enumerate()
        .for_each_init(
            || Scratch::new(n_samples, n_coef),
            |scratch, (gene, ((((beta, fit), dev), iters), fail))| {
                let disp_row = dispersion.row(gene, n_samples);
                let offset_row = offset.row(gene, n_samples);
                let weight_row = weights.map(|w| w.row(gene, n_samples));

                let start = gene * n_samples;
                for (slot, value) in scratch.y.iter_mut().zip(&counts[start..start + n_samples]) {
                    *slot = value.to_f64().unwrap_or(0.0);
                }

                match coef_start {
                    Some(s) => beta.copy_from_slice(&s[gene * n_coef..(gene + 1) * n_coef]),
                    None => initial_coefficients(
                        &scratch.y,
                        design,
                        offset_row,
                        disp_row,
                        weight_row,
                        n_samples,
                        n_coef,
                        params.start_method,
                        beta,
                    ),
                }

                let (used, singular) = fit_one_gene(
                    scratch, beta, design, n_samples, n_coef, disp_row, offset_row, weight_row,
                    &params,
                );

                linear_predictor(
                    design,
                    beta,
                    offset_row,
                    n_samples,
                    n_coef,
                    &mut scratch.eta,
                    &mut scratch.mu,
                );
                fit.copy_from_slice(&scratch.mu[..n_samples]);
                *dev = deviance_of(&scratch.y, &scratch.mu, disp_row, weight_row);
                *iters = used;
                *fail = singular;
            },
        );

    Ok(LevenbergFit {
        coefficients,
        fitted,
        deviance,
        iterations,
        failed,
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
    fn fixture() -> (Vec<f64>, Vec<f64>, usize, usize, usize) {
        let counts = vec![
            10.0, 12.0, 11.0, 40.0, 44.0, 38.0, //
            50.0, 48.0, 52.0, 49.0, 51.0, 50.0, //
        ];
        let design = vec![
            1.0, 0.0, //
            1.0, 0.0, //
            1.0, 0.0, //
            1.0, 1.0, //
            1.0, 1.0, //
            1.0, 1.0, //
        ];
        (counts, design, 2, 6, 2)
    }

    /// Parity against edgeR 4.8.2 itself.
    ///
    /// A continuous covariate is used deliberately: `designAsFactor` gives six
    /// distinct groups for six samples, so edgeR cannot take its one-way
    /// shortcut and runs the Levenberg fitter, which is what this module is.
    ///
    /// ```r
    /// y <- matrix(c(10,12,11,40,44,38, 50,48,52,49,51,50, 2,0,5,1,3,0),
    ///             nrow = 3, byrow = TRUE)
    /// X <- cbind(1, c(0.1, 0.4, 0.2, 0.9, 0.7, 0.8))
    /// glmFit(y, design = X, dispersion = 0.1,
    ///        offset = matrix(0, 3, 6), prior.count = 0)
    /// ```
    ///
    /// Agreement is asserted to 1e-7 absolute rather than tighter, because
    /// neither side is at its own maximum at the shared nominal `tol = 1e-6`.
    /// On the flat gene edgeR reports a slope of -0.021851123519974 while the
    /// true maximum likelihood estimate is -0.021851159884, confirmed
    /// independently with
    /// `glm(y ~ x, family = negative.binomial(theta = 10), epsilon = 1e-14)`.
    ///
    /// The algorithm is edgeR's, but the damping schedule and the break
    /// condition are not identical, and this one stops the earlier of the two:
    /// on a 40 by 6 fixture at `dispersion = 0.5` it sits 5.7e-4 from its own
    /// `tol = 1e-14` answer where `mglmLevenberg` at its own `tol = 1e-6` sits
    /// 2.6e-4 from the same point. Tightened, the two land on each other to
    /// 1.2e-7, so it is the stopping rule that differs and not the fit. The
    /// practical consequence is that coefficients agree with edgeR to a few
    /// parts in 1e3 on overdispersed data, and the tolerance here reflects that
    /// as much as edgeR's own residual convergence error.
    #[test]
    fn test_matches_edger_glmfit_on_a_continuous_design() {
        let counts = vec![
            10.0, 12.0, 11.0, 40.0, 44.0, 38.0, //
            50.0, 48.0, 52.0, 49.0, 51.0, 50.0, //
            2.0, 0.0, 5.0, 1.0, 3.0, 0.0, //
        ];
        let design = vec![
            1.0, 0.1, //
            1.0, 0.4, //
            1.0, 0.2, //
            1.0, 0.9, //
            1.0, 0.7, //
            1.0, 0.8, //
        ];

        let fit = mglm_levenberg(
            &counts,
            3,
            6,
            &design,
            2,
            &Recycled::scalar(0.1),
            &Recycled::scalar(0.0),
            None,
            None,
            None,
        )
        .unwrap();

        let expected_coef = [
            1.996_184_633_343_157,
            2.081_967_719_801_685,
            3.923_290_904_630_033,
            -0.021_851_123_519_974,
            1.286_187_229_994_503,
            -1.518_095_661_643_573,
        ];
        for (got, want) in fit.coefficients.iter().zip(expected_coef.iter()) {
            assert_relative_eq!(got, want, epsilon = 1e-7, max_relative = 1e-6);
        }

        let expected_dev = [
            1.895_750_938_359_449,
            0.031_167_530_245_651_8,
            8.702_866_614_403_902,
        ];
        for (got, want) in fit.deviance.iter().zip(expected_dev.iter()) {
            assert_relative_eq!(got, want, max_relative = 1e-6);
        }

        let expected_fitted = [
            9.064_642_493_571_05,
            16.928_045_123_132_83,
            11.162_703_506_518_61,
            47.940_256_583_162_03,
            31.612_797_955_800_05,
            38.929_752_700_379_63,
        ];
        for (got, want) in fit.fitted[..6].iter().zip(expected_fitted.iter()) {
            assert_relative_eq!(got, want, max_relative = 1e-6);
        }
    }

    #[test]
    fn test_solve_spd_matches_a_known_solution() {
        // [[4, 2], [2, 3]] x = [10, 8] has solution [1.75, 1.5].
        let mut a = vec![4.0, 2.0, 2.0, 3.0];
        let mut b = vec![10.0, 8.0];
        assert!(solve_spd_in_place(&mut a, &mut b, 2));
        assert_relative_eq!(b[0], 1.75, max_relative = 1e-12);
        assert_relative_eq!(b[1], 1.5, max_relative = 1e-12);
    }

    #[test]
    fn test_solve_spd_rejects_an_indefinite_matrix() {
        let mut a = vec![1.0, 2.0, 2.0, 1.0];
        let mut b = vec![1.0, 1.0];
        assert!(!solve_spd_in_place(&mut a, &mut b, 2));
    }

    /// A two-group fit with no offset must recover the group log-means exactly,
    /// since the design saturates the data.
    #[test]
    fn test_recovers_group_means_on_a_saturated_design() {
        let (counts, design, n_genes, n_samples, n_coef) = fixture();
        let fit = mglm_levenberg(
            &counts,
            n_genes,
            n_samples,
            &design,
            n_coef,
            &Recycled::scalar(0.1),
            &Recycled::scalar(0.0),
            None,
            None,
            None,
        )
        .unwrap();

        // Gene 0: group means 11 and 40.6667.
        let mean_a: f64 = (10.0 + 12.0 + 11.0) / 3.0;
        let mean_b: f64 = (40.0 + 44.0 + 38.0) / 3.0;
        assert_relative_eq!(fit.coefficients[0], mean_a.ln(), max_relative = 1e-6);
        assert_relative_eq!(
            fit.coefficients[0] + fit.coefficients[1],
            mean_b.ln(),
            max_relative = 1e-6
        );

        // The fitted values are the group means themselves.
        assert_relative_eq!(fit.fitted[0], mean_a, max_relative = 1e-6);
        assert_relative_eq!(fit.fitted[3], mean_b, max_relative = 1e-6);
    }

    #[test]
    fn test_flat_gene_has_a_near_zero_group_effect() {
        let (counts, design, n_genes, n_samples, n_coef) = fixture();
        let fit = mglm_levenberg(
            &counts,
            n_genes,
            n_samples,
            &design,
            n_coef,
            &Recycled::scalar(0.1),
            &Recycled::scalar(0.0),
            None,
            None,
            None,
        )
        .unwrap();
        // Gene 1 is flat: the group coefficient should be tiny.
        assert!(
            fit.coefficients[3].abs() < 0.02,
            "got {}",
            fit.coefficients[3]
        );
    }

    /// Offsets shift the fit on the log scale, so doubling every library size
    /// must halve the fitted rate while leaving the group contrast alone.
    #[test]
    fn test_offsets_shift_the_intercept_not_the_contrast() {
        let (counts, design, n_genes, n_samples, n_coef) = fixture();
        let plain = mglm_levenberg(
            &counts,
            n_genes,
            n_samples,
            &design,
            n_coef,
            &Recycled::scalar(0.1),
            &Recycled::scalar(0.0),
            None,
            None,
            None,
        )
        .unwrap();
        let shifted = mglm_levenberg(
            &counts,
            n_genes,
            n_samples,
            &design,
            n_coef,
            &Recycled::scalar(0.1),
            &Recycled::scalar(2.0_f64.ln()),
            None,
            None,
            None,
        )
        .unwrap();

        assert_relative_eq!(
            shifted.coefficients[0],
            plain.coefficients[0] - 2.0_f64.ln(),
            max_relative = 1e-6
        );
        assert_relative_eq!(
            shifted.coefficients[1],
            plain.coefficients[1],
            max_relative = 1e-6
        );
    }

    /// A saturated design fits the data exactly, so the residual deviance is
    /// not zero only because of the dispersion. What must hold is that a higher
    /// dispersion gives a smaller deviance for the same residuals.
    #[test]
    fn test_higher_dispersion_shrinks_the_deviance() {
        let (counts, design, n_genes, n_samples, n_coef) = fixture();
        let low = mglm_levenberg(
            &counts,
            n_genes,
            n_samples,
            &design,
            n_coef,
            &Recycled::scalar(0.01),
            &Recycled::scalar(0.0),
            None,
            None,
            None,
        )
        .unwrap();
        let high = mglm_levenberg(
            &counts,
            n_genes,
            n_samples,
            &design,
            n_coef,
            &Recycled::scalar(1.0),
            &Recycled::scalar(0.0),
            None,
            None,
            None,
        )
        .unwrap();
        assert!(high.deviance[0] < low.deviance[0]);
    }

    #[test]
    fn test_all_zero_gene_fits_without_diverging() {
        let counts = vec![0.0; 6];
        let design = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let fit = mglm_levenberg(
            &counts,
            1,
            6,
            &design,
            2,
            &Recycled::scalar(0.1),
            &Recycled::scalar(0.0),
            None,
            None,
            None,
        )
        .unwrap();
        assert!(fit.coefficients.iter().all(|c| c.is_finite()));
        assert!(fit.fitted.iter().all(|m| m.is_finite() && *m >= 0.0));
        assert!(fit.deviance[0].is_finite());
    }

    #[test]
    fn test_weights_of_zero_remove_a_sample() {
        let (counts, design, n_genes, n_samples, n_coef) = fixture();
        // Drop the last sample of each group by weighting it out.
        let weights = Recycled::by_sample(vec![1.0, 1.0, 0.0, 1.0, 1.0, 0.0]);
        let fit = mglm_levenberg(
            &counts,
            n_genes,
            n_samples,
            &design,
            n_coef,
            &Recycled::scalar(0.1),
            &Recycled::scalar(0.0),
            Some(&weights),
            None,
            None,
        )
        .unwrap();
        // Gene 0 group A is now the mean of 10 and 12.
        assert_relative_eq!(fit.fitted[0], 11.0, max_relative = 1e-5);
        // Group B is the mean of 40 and 44.
        assert_relative_eq!(fit.fitted[3], 42.0, max_relative = 1e-5);
    }

    #[test]
    fn test_supplied_starting_values_reach_the_same_optimum() {
        let (counts, design, n_genes, n_samples, n_coef) = fixture();
        let cold = mglm_levenberg(
            &counts,
            n_genes,
            n_samples,
            &design,
            n_coef,
            &Recycled::scalar(0.1),
            &Recycled::scalar(0.0),
            None,
            None,
            None,
        )
        .unwrap();
        let warm = mglm_levenberg(
            &counts,
            n_genes,
            n_samples,
            &design,
            n_coef,
            &Recycled::scalar(0.1),
            &Recycled::scalar(0.0),
            None,
            Some(&cold.coefficients),
            None,
        )
        .unwrap();
        for (a, b) in cold.coefficients.iter().zip(warm.coefficients.iter()) {
            assert_relative_eq!(a, b, max_relative = 1e-6);
        }
        assert_relative_eq!(warm.deviance[0], cold.deviance[0], max_relative = 1e-10);

        // Note that warm starting does not reduce the iteration count, and can
        // increase it. Convergence is only tested after an accepted step, so a
        // fit that begins at the optimum has nothing left to improve: its steps
        // are rejected, damping climbs, and it only stops once `delta` has
        // collapsed far enough for the deviance to come back exactly equal.
        // edgePython has the same structure. The value of a warm start here is
        // reaching the same optimum reliably, not reaching it in fewer steps.
        assert!(warm.iterations[0] <= 20);
    }

    #[test]
    fn test_rejects_shape_mismatches() {
        let (counts, design, _, n_samples, n_coef) = fixture();
        let err = mglm_levenberg(
            &counts,
            3,
            n_samples,
            &design,
            n_coef,
            &Recycled::scalar(0.1),
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

    #[test]
    fn test_rejects_zero_coefficients() {
        let err = mglm_levenberg(
            &[1.0_f64; 6],
            1,
            6,
            &[],
            0,
            &Recycled::scalar(0.1),
            &Recycled::scalar(0.0),
            None,
            None,
            None,
        )
        .unwrap_err();
        assert!(matches!(err, EdgeErrors::MustBePositive(_)));
    }

    /// Fitting one gene alone must give the same answer as fitting it inside a
    /// batch, which pins the parallel path against the serial one.
    #[test]
    fn test_batched_and_single_fits_agree() {
        let (counts, design, n_genes, n_samples, n_coef) = fixture();
        let batch = mglm_levenberg(
            &counts,
            n_genes,
            n_samples,
            &design,
            n_coef,
            &Recycled::by_gene(vec![0.05, 0.2]),
            &Recycled::scalar(0.0),
            None,
            None,
            None,
        )
        .unwrap();

        let single = mglm_levenberg(
            &counts[n_samples..],
            1,
            n_samples,
            &design,
            n_coef,
            &Recycled::scalar(0.2),
            &Recycled::scalar(0.0),
            None,
            None,
            None,
        )
        .unwrap();

        for k in 0..n_coef {
            assert_relative_eq!(
                batch.coefficients[n_coef + k],
                single.coefficients[k],
                max_relative = 1e-10
            );
        }
    }

    /// The starting coefficients, against edgeR's `.cxx_get_levenberg_start`.
    ///
    /// `maxit = 0` returns the start unchanged, which is how these were read out
    /// of edgeR. The design deliberately has no intercept column: with one, any
    /// implementation that drops the scalar into `beta[0]` and zeroes the rest
    /// happens to be right, so an intercept-bearing fixture cannot tell a
    /// correct start from a broken one. The weighted case is here for the same
    /// reason, since the working weights only bite when the libraries differ.
    ///
    /// ```r
    /// # Rscript, edgeR 4.8.2
    /// y <- matrix(c(493,205,215,683,465,538, 100,120,90,110,105,95), 2, 6, byrow = TRUE)
    /// lib <- c(1e6, 1.1e6, 0.9e6, 1.2e6, 1e6, 1.05e6)
    /// off <- matrix(rep(log(lib), each = 2), 2, 6)
    /// score <- c(0.1, 0.4, 0.2, 0.9, 0.3, 0.7)
    /// X <- cbind(c(1,1,1,0,0,0), c(0,0,0,1,1,1), score)
    /// w <- matrix(rep(c(.1,.1,.1,10,10,10), each = 2), 2, 6)
    /// mglmLevenberg(y, X, dispersion = 0.1, offset = off, maxit = 0)$coefficients
    /// mglmLevenberg(y, X, dispersion = 0.1, offset = off, weights = w, maxit = 0)$coefficients
    /// ```
    #[test]
    fn test_start_matches_edger_on_a_design_without_an_intercept() {
        let y: Vec<f64> = vec![
            493.0, 205.0, 215.0, 683.0, 465.0, 538.0, //
            100.0, 120.0, 90.0, 110.0, 105.0, 95.0,
        ];
        let lib: [f64; 6] = [1e6, 1.1e6, 0.9e6, 1.2e6, 1e6, 1.05e6];
        let offset = Recycled::by_sample(lib.iter().map(|v| v.ln()).collect());
        let score = [0.1, 0.4, 0.2, 0.9, 0.3, 0.7];
        let mut design = Vec::with_capacity(18);
        for (j, s) in score.iter().enumerate() {
            design.push(if j < 3 { 1.0 } else { 0.0 });
            design.push(if j < 3 { 0.0 } else { 1.0 });
            design.push(*s);
        }
        let stopped = LevenbergParams {
            max_iter: 0,
            ..Default::default()
        };

        let fit = mglm_levenberg(
            &y,
            2,
            6,
            &design,
            3,
            &Recycled::scalar(0.1),
            &offset,
            None,
            None,
            Some(stopped),
        )
        .unwrap();
        // Both group columns carry the rate; edgeR projects, it does not just
        // fill the first column.
        assert_relative_eq!(
            fit.coefficients[0],
            -7.797_403_899_938_82,
            max_relative = 1e-12
        );
        assert_relative_eq!(
            fit.coefficients[1],
            -7.797_403_899_938_82,
            max_relative = 1e-12
        );
        assert!(fit.coefficients[2].abs() < 1e-12);
        assert_relative_eq!(
            fit.coefficients[3],
            -9.216_637_226_798_91,
            max_relative = 1e-12
        );
        assert_relative_eq!(
            fit.coefficients[4],
            -9.216_637_226_798_91,
            max_relative = 1e-12
        );
        assert!(fit.coefficients[5].abs() < 1e-12);

        let weights = Recycled::by_sample(vec![0.1, 0.1, 0.1, 10.0, 10.0, 10.0]);
        let fit = mglm_levenberg(
            &y,
            2,
            6,
            &design,
            3,
            &Recycled::scalar(0.1),
            &offset,
            Some(&weights),
            None,
            Some(stopped),
        )
        .unwrap();
        assert_relative_eq!(
            fit.coefficients[0],
            -7.574_372_857_478_63,
            max_relative = 1e-12
        );
        assert_relative_eq!(
            fit.coefficients[3],
            -9.253_386_520_876_94,
            max_relative = 1e-12
        );
    }

    #[test]
    fn test_rejects_a_negative_dispersion() {
        let (counts, design, n_genes, n_samples, n_coef) = fixture();
        let err = mglm_levenberg(
            &counts,
            n_genes,
            n_samples,
            &design,
            n_coef,
            &Recycled::scalar(-0.1),
            &Recycled::scalar(0.0),
            None,
            None,
            None,
        )
        .unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidDispersion(_)));
    }
}
