//! NEBULA's penalised maximum likelihood inner solver.
//!
//! Given fixed variance components, this profiles out the subject random
//! effects and the fixed effects jointly. The parameter vector is
//! `(beta, log_w)` with `beta` the `n_beta` fixed effects and `log_w` one
//! random effect per subject, so the Hessian is `(n_beta + k)` square. It is
//! never formed: the random-effect block is diagonal, because subjects are
//! independent and the cells of one subject are contiguous, so a Schur
//! complement reduces every Newton step to an `n_beta` system plus `k`
//! divisions.
//!
//! Two random-effect distributions share the loop:
//!
//! * [`opt_pml`] puts a gamma prior on `w`, which is NEBULA's NBGMM. The
//!   penalty is `alpha log w - lambda w` with `alpha` and `lambda` derived from
//!   `sigma[0]`, and the Laplace approximation to the marginal likelihood can
//!   be pushed past its leading term via `ord`.
//! * [`opt_pml_nbm`] puts a normal prior on `log w`, which is NEBULA's NBLMM.
//!   The penalty is `-log_w'log_w / (2 alpha) - (k/2) log alpha`, and no
//!   higher-order correction is defined.
//!
//! ### Layout
//!
//! Row-major throughout, `f64` throughout. The design is `n_cells * n_beta`,
//! counts are stored as the positive entries only with their cell indices
//! alongside, and subjects are half-open `[start, end)` blocks of cells.
//!
//! ### Parallelism
//!
//! Sequential, deliberately. The parallel axis for NEBULA is genes, and one
//! call here is one gene; fanning out inside would oversubscribe against that.
//! The second reason is reproducibility: the outer loop stops on an absolute
//! improvement in the log-likelihood, so a rayon reduction that reassociates
//! the sums can flip the last iteration on or off and move `beta` by far more
//! than the reassociation itself.
//!
//! ### Deviations from edgePython
//!
//! This is a port of `nebula`'s own `src/optimization.cpp`, not of
//! edgePython's `_opt_pml_nb`. edgePython clamps the linear predictor at 500,
//! floors `vw` at `1e-15`, adds `1e-8` to any diagonal of the information
//! matrix below `1e-10`, and drops the REML term from the log-determinant.
//! None of those appear in the C++, and each moves the returned information
//! matrix, which is exactly the standard error. They are not reproduced here.
//!
//! ### References
//!
//! He et al., Communications Biology 4, 629, 2021

#![allow(clippy::needless_range_loop)]

use crate::errors::EdgeErrors;

////////////////////////
// Tuning and codes   //
////////////////////////

/// Newton iteration budget, matching nebula's hard-coded `maxstep`.
///
/// nebula flags a fit that reaches this as `-20`, so the number is part of the
/// contract rather than a free tuning knob.
const DEFAULT_MAX_ITER: usize = 50;

/// Backtracking budget inside one Newton step, nebula's `maxstd`.
///
/// Exceeding it is what produces the `-10` and `-40` convergence codes.
const DEFAULT_MAX_BACKTRACK: usize = 10;

/// Default stopping tolerance on the improvement in the penalised
/// log-likelihood, nebula's `eps`.
const DEFAULT_EPS: f64 = 1e-6;

/// Largest gradient component nebula will still call a critical point.
///
/// Reached only when backtracking has been exhausted: a stalled step with every
/// gradient below this is `-10`, with any gradient above it `-40`.
const GRADIENT_TOLERANCE: f64 = 0.01;

/// Magnitude past which a Newton component is treated as unbounded rather than
/// damped.
///
/// A step of `exp(40)` on the log scale is not a step, it is an overflow, so
/// nebula replaces it with a fixed displacement instead of halving it forty
/// times. The same constant seeds that displacement, which is then halved once
/// per backtrack.
const STEP_CUTOFF: f64 = 40.0;

/// Smallest eigenvalue of the information matrix nebula will invert.
///
/// Below it the fixed effects are not identifiable for this gene and the fit is
/// reported as `-25` with no standard errors.
const EIGENVALUE_CUTOFF: f64 = 1e-8;

/// Jacobi sweep budget for the smallest-eigenvalue check.
///
/// Cyclic Jacobi converges cubically once the off-diagonal is small; on the
/// handful-of-columns matrices seen here it is done in three or four sweeps and
/// thirty is pure slack.
const JACOBI_SWEEPS: usize = 30;

/// Converged, by a sufficiently small improvement in the objective.
pub const CONV_SUCCESS: i32 = 1;

/// Converged at a critical point: no step improved the objective, but every
/// gradient component is below nebula's `convd`, 0.01.
pub const CONV_CRITICAL_POINT: i32 = -10;

/// The Newton iteration budget was exhausted.
pub const CONV_MAX_ITER: i32 = -20;

/// The information matrix is numerically singular, so no standard errors.
pub const CONV_SINGULAR: i32 = -25;

/// The penalised log-likelihood was not a number at the reported optimum.
pub const CONV_NOT_FINITE: i32 = -30;

/// Backtracking stalled with a gradient still above nebula's `convd`, 0.01.
pub const CONV_STALLED: i32 = -40;

/// A variance component sat on its box constraint.
///
/// Not detectable from inside this module; pass `variance_at_bound` to
/// [`check_convergence`] to raise it.
pub const CONV_AT_BOUND: i32 = -60;

/////////////
// PmlData //
/////////////

/// One gene's data, in the layout NEBULA's C++ expects.
///
/// Cells must already be grouped by subject, which is what makes
/// `subject_start` a valid block partition. Only the positive counts are
/// stored: `counts[i]` is the count in cell `cell_index[i]`.
#[derive(Clone, Copy, Debug)]
pub struct PmlData<'a> {
    /// Design matrix, row-major `n_cells * n_beta`.
    pub design: &'a [f64],
    /// Log offset per cell, length `n_cells`.
    pub offset: &'a [f64],
    /// The positive counts, in increasing cell order.
    pub counts: &'a [f64],
    /// Cell index of each entry of `counts`, zero-based and increasing.
    pub cell_index: &'a [usize],
    /// Start of each subject's block plus a trailing `n_cells`, length `k + 1`.
    pub subject_start: &'a [usize],
    /// Total count per subject, length `k`. nebula's `cumsumy`.
    pub subject_total: &'a [f64],
}

impl PmlData<'_> {
    /// Number of cells.
    ///
    /// ### Returns
    ///
    /// The length of `offset`, which defines the cell axis.
    pub fn n_cells(&self) -> usize {
        self.offset.len()
    }

    /// Number of subjects.
    ///
    /// ### Returns
    ///
    /// One fewer than the length of `subject_start`, or zero if that is empty.
    pub fn n_subjects(&self) -> usize {
        self.subject_start.len().saturating_sub(1)
    }

    /// Number of fixed effects.
    ///
    /// ### Returns
    ///
    /// The design width implied by `design.len()` and `n_cells`, or zero when
    /// there are no cells.
    pub fn n_beta(&self) -> usize {
        let n = self.n_cells();
        self.design.len().checked_div(n).unwrap_or(0)
    }

    /// Checks every invariant the inner loops assume.
    ///
    /// ### Returns
    ///
    /// `Ok(())`, or the first violation found.
    fn validate(&self) -> Result<(), EdgeErrors> {
        let n_cells = self.n_cells();
        if n_cells == 0 {
            return Err(EdgeErrors::MustBePositive("n_cells".to_string()));
        }
        let k = self.n_subjects();
        if k == 0 {
            return Err(EdgeErrors::MustBePositive("n_subjects".to_string()));
        }
        let n_beta = self.n_beta();
        if n_beta == 0 || n_beta * n_cells != self.design.len() {
            return Err(EdgeErrors::LengthMismatch {
                name: "design",
                expected: n_cells * n_beta.max(1),
                got: self.design.len(),
            });
        }
        if self.counts.len() != self.cell_index.len() {
            return Err(EdgeErrors::LengthMismatch {
                name: "cell_index",
                expected: self.counts.len(),
                got: self.cell_index.len(),
            });
        }
        if self.subject_total.len() != k {
            return Err(EdgeErrors::LengthMismatch {
                name: "subject_total",
                expected: k,
                got: self.subject_total.len(),
            });
        }
        if self.subject_start[0] != 0 || self.subject_start[k] != n_cells {
            return Err(EdgeErrors::InvalidArgument(format!(
                "subject_start must run from 0 to {n_cells}; got {} to {}.",
                self.subject_start[0], self.subject_start[k]
            )));
        }
        for s in 0..k {
            if self.subject_start[s] >= self.subject_start[s + 1] {
                return Err(EdgeErrors::SubjectsNotContiguous { subject: s });
            }
        }
        let mut previous: Option<usize> = None;
        for &c in self.cell_index {
            if c >= n_cells {
                return Err(EdgeErrors::InvalidArgument(format!(
                    "cell_index {c} is out of range for {n_cells} cells."
                )));
            }
            if previous.is_some_and(|p| c <= p) {
                return Err(EdgeErrors::InvalidArgument(
                    "cell_index must be strictly increasing.".to_string(),
                ));
            }
            previous = Some(c);
        }
        Ok(())
    }
}

/////////////////
// PmlVariance //
/////////////////

/// The two variance components, held on nebula's own scale.
///
/// The meaning of `subject` differs between the two solvers, which is nebula's
/// convention, not a slip: the C++ reads the same `sigma[0]` slot differently
/// in `opt_pml` and `opt_pml_nbm`.
#[derive(Clone, Copy, Debug)]
pub struct PmlVariance {
    /// nebula's `sigma[0]`.
    ///
    /// For [`opt_pml`] this is `log(1 + sigma^2)`, the log of one plus the
    /// squared coefficient of variation of the subject-level gamma random
    /// effect; the solver derives `alpha = 1 / (e^s - 1)` and
    /// `lambda = 1 / (sqrt(e^s) (e^s - 1))` from it. For [`opt_pml_nbm`] it is
    /// the variance of the normal random effect on the log scale, used as is.
    pub subject: f64,
    /// nebula's `sigma[1]`, its `gamma`: the cell-level negative binomial size,
    /// the reciprocal of the cell-level overdispersion.
    pub cell: f64,
}

///////////////
// PmlParams //
///////////////

/// Tuning knobs shared by both solvers.
#[derive(Clone, Copy, Debug)]
pub struct PmlParams {
    /// Add `log|det(information)|` to the log-determinant.
    ///
    /// nebula's `reml`, which switches the outer overdispersion estimate from
    /// maximum likelihood to restricted maximum likelihood. Off by default,
    /// matching `nebula(..., reml = 0)`.
    pub reml: bool,
    /// Stop once a Newton step improves the penalised log-likelihood by less
    /// than this. Absolute, not relative, exactly as nebula.
    pub eps: f64,
    /// Order of the Laplace expansion of the marginal likelihood.
    ///
    /// `1` uses the leading term only and leaves
    /// [`PmlResult::second_order`] at zero. `2` adds the third-derivative term,
    /// `3` the fourth-derivative terms. nebula raises this to `3` for genes
    /// whose expected count per subject is below three, where the leading term
    /// alone is biased. Ignored by [`opt_pml_nbm`].
    pub ord: u32,
    /// Newton iteration budget.
    pub max_iter: usize,
    /// Backtracking halvings allowed inside one Newton step.
    pub max_backtrack: usize,
}

impl Default for PmlParams {
    /// nebula's defaults as called from `nebula()`.
    fn default() -> Self {
        Self {
            reml: false,
            eps: DEFAULT_EPS,
            ord: 1,
            max_iter: DEFAULT_MAX_ITER,
            max_backtrack: DEFAULT_MAX_BACKTRACK,
        }
    }
}

///////////////
// PmlResult //
///////////////

/// The fitted penalised maximum likelihood solution for one gene.
#[derive(Clone, Debug)]
pub struct PmlResult {
    /// Fixed effect coefficients.
    pub beta: Vec<f64>,
    /// Subject random effects on the log scale.
    pub log_w: Vec<f64>,
    /// Observed information for the fixed effects, row-major `nb * nb`.
    ///
    /// nebula's `var`: the Schur complement of the joint observed information
    /// after the random-effect block has been eliminated. Its inverse is the
    /// covariance matrix of `beta`.
    pub information: Vec<f64>,
    /// Value of the penalised log-likelihood at the optimum.
    pub log_likelihood: f64,
    /// Value at the previous Newton iterate, nebula's `loglikp`.
    ///
    /// nebula falls back to this when `log_likelihood` is not a number.
    pub log_likelihood_prev: f64,
    /// Log-determinant of the observed information.
    ///
    /// `sum log|vw|` over subjects, plus `log|det(information)|` when
    /// [`PmlParams::reml`] is set. The outer overdispersion objective subtracts
    /// half of this.
    pub log_det: f64,
    /// Higher-order Laplace correction, nebula's `second`.
    ///
    /// Zero unless [`PmlParams::ord`] exceeds one. The outer objective adds
    /// `log(1 + second)`.
    pub second_order: f64,
    /// nebula's convergence code, one of [`CONV_SUCCESS`],
    /// [`CONV_CRITICAL_POINT`], [`CONV_MAX_ITER`], [`CONV_SINGULAR`],
    /// [`CONV_NOT_FINITE`] or [`CONV_STALLED`].
    ///
    /// [`CONV_AT_BOUND`] is not reachable from here; see [`check_convergence`].
    pub convergence: i32,
    /// Newton steps taken.
    pub iterations: usize,
    /// Backtracking halvings used inside the final Newton step, nebula's
    /// `damp`. Eleven means the step stalled at a critical point, twelve means
    /// it stalled with a live gradient.
    pub backtracks: usize,
}

/// The penalised log-likelihood and its gradient at a given point.
///
/// Both are negated, matching nebula's `pml_ll_der_eigen`, because the R side
/// hands them straight to a minimiser.
#[derive(Clone, Debug)]
pub struct PmlObjective {
    /// Negative penalised log-likelihood.
    pub objective: f64,
    /// Negative gradient: `n_beta` fixed-effect entries followed by
    /// `n_subjects` random-effect entries.
    pub gradient: Vec<f64>,
}

////////////////////
// Random effects //
////////////////////

/// The prior on the subject random effects, with its scalars already resolved.
#[derive(Clone, Copy, Debug)]
enum Penalty {
    /// Gamma prior on `w`, NEBULA's NBGMM.
    Gamma {
        /// Shape, `1 / (e^s - 1)`.
        alpha: f64,
        /// Rate, `1 / (sqrt(e^s) (e^s - 1))`.
        lambda: f64,
    },
    /// Normal prior on `log w`, NEBULA's NBLMM.
    LogNormal {
        /// Variance of the normal, used directly.
        alpha: f64,
    },
}

impl Penalty {
    /// Adds the prior's contribution to the log-likelihood.
    ///
    /// The association order is the C++'s, per variant, because the outer loop
    /// compares improvements against an absolute tolerance and a reassociated
    /// sum can flip the last iteration.
    ///
    /// ### Params
    ///
    /// * `log_likelihood` - Data contribution accumulated so far
    /// * `log_w` - Random effects on the log scale
    /// * `w` - `exp(log_w)`, unused by the log-normal variant
    ///
    /// ### Returns
    ///
    /// The penalised log-likelihood.
    fn accumulate(&self, log_likelihood: f64, log_w: &[f64], w: &[f64]) -> f64 {
        match *self {
            Penalty::Gamma { alpha, lambda } => {
                let mut sum_log_w = 0.0;
                for &l in log_w {
                    sum_log_w += l;
                }
                let mut sum_w = 0.0;
                for &x in w {
                    sum_w += x;
                }
                log_likelihood + (alpha * sum_log_w - lambda * sum_w)
            }
            Penalty::LogNormal { alpha } => {
                let mut quad = 0.0;
                for &l in log_w {
                    quad += l * l;
                }
                log_likelihood - quad / alpha / 2.0 - (log_w.len() as f64) / 2.0 * alpha.ln()
            }
        }
    }

    /// The prior's contribution to the random-effect gradient.
    ///
    /// ### Params
    ///
    /// * `log_w_s` - Random effect for this subject
    /// * `w_s` - `exp(log_w_s)`
    ///
    /// ### Returns
    ///
    /// The term added to `cumsumy - sum(gstar_extb_phi)` for this subject.
    fn gradient(&self, log_w_s: f64, w_s: f64) -> f64 {
        match *self {
            Penalty::Gamma { alpha, lambda } => alpha - lambda * w_s,
            Penalty::LogNormal { alpha } => -log_w_s / alpha,
        }
    }

    /// The prior's contribution to the random-effect curvature.
    ///
    /// ### Params
    ///
    /// * `w_s` - `exp(log_w_s)`
    ///
    /// ### Returns
    ///
    /// The term added to the data curvature for this subject.
    fn curvature(&self, w_s: f64) -> f64 {
        match *self {
            Penalty::Gamma { lambda, .. } => lambda * w_s,
            Penalty::LogNormal { alpha } => 1.0 / alpha,
        }
    }
}

///////////////
// Front end //
///////////////

/// Penalised maximum likelihood fit with a gamma random effect, NEBULA's NBGMM.
///
/// Joint Newton iteration over `beta` and `log_w` with the random-effect block
/// eliminated by a Schur complement, per-coordinate backtracking on a rejected
/// step, and the higher-order Laplace corrections selected by
/// [`PmlParams::ord`].
///
/// The random effect `w` carries a `Gamma(alpha, lambda)` prior with
/// `alpha = 1 / (e^s - 1)` and `lambda = 1 / (sqrt(e^s) (e^s - 1))`, where
/// `s` is [`PmlVariance::subject`]. That parametrisation gives `w` mean
/// `sqrt(e^s)` and squared coefficient of variation `e^s - 1`.
///
/// ### Params
///
/// * `data` - One gene's design, offsets, positive counts and subject blocks
/// * `beta_init` - Starting fixed effects, length `n_beta`. `log_w` always
///   starts at zero, as in the C++
/// * `variance` - The two variance components, held fixed throughout
/// * `params` - Tuning knobs, or `None` for nebula's defaults
///
/// ### Returns
///
/// The fitted coefficients, random effects, observed information and
/// convergence diagnostics.
///
/// ### References
///
/// He et al., Communications Biology 4, 629, 2021
pub fn opt_pml(
    data: &PmlData<'_>,
    beta_init: &[f64],
    variance: &PmlVariance,
    params: Option<PmlParams>,
) -> Result<PmlResult, EdgeErrors> {
    let exps = variance.subject.exp();
    if !(exps.is_finite() && exps > 1.0) {
        return Err(EdgeErrors::InvalidArgument(format!(
            "PmlVariance::subject must be finite and strictly positive for opt_pml; got {}.",
            variance.subject
        )));
    }
    let penalty = Penalty::Gamma {
        alpha: 1.0 / (exps - 1.0),
        lambda: 1.0 / (exps.sqrt() * (exps - 1.0)),
    };
    optimise(
        data,
        beta_init,
        penalty,
        variance.cell,
        params.unwrap_or_default(),
    )
}

/// Penalised maximum likelihood fit with a log-normal random effect, NEBULA's
/// NBLMM.
///
/// The same Newton loop as [`opt_pml`] with the gamma penalty replaced by
/// `-log_w'log_w / (2 alpha) - (k/2) log alpha`, where `alpha` is
/// [`PmlVariance::subject`] read as a variance. No higher-order Laplace
/// correction is defined for this variant, so [`PmlResult::second_order`] is
/// always zero and [`PmlParams::ord`] is ignored.
///
/// ### Params
///
/// * `data` - One gene's design, offsets, positive counts and subject blocks
/// * `beta_init` - Starting fixed effects, length `n_beta`
/// * `variance` - The two variance components, held fixed throughout
/// * `params` - Tuning knobs, or `None` for nebula's defaults
///
/// ### Returns
///
/// The fitted coefficients, random effects, observed information and
/// convergence diagnostics.
pub fn opt_pml_nbm(
    data: &PmlData<'_>,
    beta_init: &[f64],
    variance: &PmlVariance,
    params: Option<PmlParams>,
) -> Result<PmlResult, EdgeErrors> {
    if !(variance.subject.is_finite() && variance.subject > 0.0) {
        return Err(EdgeErrors::InvalidArgument(format!(
            "PmlVariance::subject must be finite and strictly positive for opt_pml_nbm; got {}.",
            variance.subject
        )));
    }
    let penalty = Penalty::LogNormal {
        alpha: variance.subject,
    };
    optimise(
        data,
        beta_init,
        penalty,
        variance.cell,
        params.unwrap_or_default(),
    )
}

/// Penalised log-likelihood and gradient at a point, nebula's
/// `pml_ll_der_eigen`.
///
/// This is the gradient-only entry the R side hands to `lbfgs` and `trust`, and
/// it reads [`PmlVariance::subject`] differently again: as the shape of a
/// `Gamma(alpha, alpha)` prior with mean one, so `alpha` plays the part of both
/// the shape and the rate. That is nebula's own inconsistency with [`opt_pml`],
/// reproduced rather than corrected.
///
/// ### Params
///
/// * `data` - One gene's design, offsets, positive counts and subject blocks
/// * `beta` - Fixed effects, length `n_beta`
/// * `log_w` - Random effects on the log scale, length `n_subjects`
/// * `variance` - The two variance components
///
/// ### Returns
///
/// The negated objective and the negated gradient, laid out as `n_beta` fixed
/// effect entries followed by `n_subjects` random effect entries.
pub fn pml_log_likelihood_gradient(
    data: &PmlData<'_>,
    beta: &[f64],
    log_w: &[f64],
    variance: &PmlVariance,
) -> Result<PmlObjective, EdgeErrors> {
    data.validate()?;
    let n_beta = data.n_beta();
    let k = data.n_subjects();
    if beta.len() != n_beta {
        return Err(EdgeErrors::LengthMismatch {
            name: "beta",
            expected: n_beta,
            got: beta.len(),
        });
    }
    if log_w.len() != k {
        return Err(EdgeErrors::LengthMismatch {
            name: "log_w",
            expected: k,
            got: log_w.len(),
        });
    }

    let alpha = variance.subject;
    let gamma = variance.cell;
    // Both the shape and the rate: `alpha*sum(log w) - alpha*sum(w)`.
    let penalty = Penalty::Gamma {
        alpha,
        lambda: alpha,
    };

    let mut work = Workspace::new(data);
    work.seed(data, gamma);
    let log_likelihood = work.evaluate(data, beta, log_w, gamma, penalty);
    work.gradient_only(data, gamma, penalty, log_w);

    let mut gradient = Vec::with_capacity(n_beta + k);
    gradient.extend(work.db.iter().map(|v| -v));
    gradient.extend(work.dw.iter().map(|v| -v));
    Ok(PmlObjective {
        objective: -log_likelihood,
        gradient,
    })
}

/// nebula's `check_conv`, applied to a finished fit.
///
/// [`opt_pml`] already runs this with `variance_at_bound` false and an initial
/// code of [`CONV_SUCCESS`]. Re-run it from the outer overdispersion loop,
/// which is the only place that knows whether a variance component landed on
/// its box constraint or whether an earlier optimiser already failed.
///
/// The singular-information test overrides everything else, including
/// [`CONV_AT_BOUND`], which is nebula's ordering.
///
/// ### Params
///
/// * `result` - The fit to grade
/// * `params` - The knobs it was run with; only `max_iter` is consulted
/// * `variance_at_bound` - Whether a variance component sat on its bound
/// * `initial` - Code carried in from the outer optimiser, [`CONV_SUCCESS`]
///   when that succeeded and something negative when it did not
///
/// ### Returns
///
/// The convergence code.
pub fn check_convergence(
    result: &PmlResult,
    params: &PmlParams,
    variance_at_bound: bool,
    initial: i32,
) -> i32 {
    let mut code = initial;
    if code == CONV_SUCCESS {
        code = if variance_at_bound {
            CONV_AT_BOUND
        } else if result.log_likelihood.is_nan() {
            CONV_NOT_FINITE
        } else if result.iterations == params.max_iter {
            CONV_MAX_ITER
        } else if result.backtracks == params.max_backtrack + 1 {
            CONV_CRITICAL_POINT
        } else if result.backtracks == params.max_backtrack + 2 {
            CONV_STALLED
        } else {
            CONV_SUCCESS
        };
    }

    let n_beta = result.beta.len();
    if n_beta > 1 && min_eigenvalue(&result.information, n_beta) < EIGENVALUE_CUTOFF {
        code = CONV_SINGULAR;
    }
    code
}

///////////////
// Workspace //
///////////////

/// Every buffer the Newton loop needs, allocated once per call.
///
/// `x_phi` is the only one that scales with the data: it holds the design
/// scaled row-wise by the curvature weights, `n_cells * n_beta` doubles. The
/// C++ materialises the same matrix, and both the cross block and the
/// fixed-effect block are read off it, so fusing it away would change the
/// summation order for no memory that matters at NEBULA's design widths.
struct Workspace {
    /// `exp(offset + design*beta + log_w)`, one per cell.
    extb: Vec<f64>,
    /// `log(extb + gamma)`, one per cell.
    extbphil: Vec<f64>,
    /// Working weight, reused down the derivative chain, one per cell.
    phi: Vec<f64>,
    /// `gamma` plus the cell's count, one per cell. nebula's `gstar`.
    gstar: Vec<f64>,
    /// Scratch for the higher-order derivative summands, one per cell.
    deriv: Vec<f64>,
    /// Design scaled row-wise by `phi`, row-major `n_cells * n_beta`.
    x_phi: Vec<f64>,
    /// `exp(log_w)`, one per subject.
    w: Vec<f64>,
    /// `X' y` over the positive counts, length `n_beta`.
    yx: Vec<f64>,
    /// Fixed-effect gradient, length `n_beta`.
    db: Vec<f64>,
    /// Random-effect gradient, length `n_subjects`.
    dw: Vec<f64>,
    /// Random-effect curvature, length `n_subjects`.
    vw: Vec<f64>,
    /// Cross block of the information, row-major `n_subjects * n_beta`.
    vwb: Vec<f64>,
    /// Fixed-effect block of the information, row-major `n_beta * n_beta`.
    vb: Vec<f64>,
    /// Schur complement of the information, row-major `n_beta * n_beta`.
    vb2: Vec<f64>,
    /// Copy of `vb2`, destroyed by each factorisation.
    factor: Vec<f64>,
    /// Transposition record from the pivoted factorisation, length `n_beta`.
    perm: Vec<usize>,
    /// Scratch row for the factorisation, length `n_beta`.
    tmp: Vec<f64>,
    /// Newton step in `beta`, length `n_beta`.
    step_beta: Vec<f64>,
    /// Newton step in `log_w`, length `n_subjects`.
    step_log_w: Vec<f64>,
    /// `dw / vw`, length `n_subjects`.
    dwvw: Vec<f64>,
    /// Per-coordinate damping on `beta`, length `n_beta`.
    damp_beta: Vec<f64>,
    /// Per-coordinate damping on `log_w`, length `n_subjects`.
    damp_log_w: Vec<f64>,
    /// Trial fixed effects, length `n_beta`.
    new_beta: Vec<f64>,
    /// Trial random effects, length `n_subjects`.
    new_log_w: Vec<f64>,
    /// Third-derivative sums, length `n_subjects`.
    third: Vec<f64>,
    /// Fourth-derivative sums, length `n_subjects`.
    fourth: Vec<f64>,
}

impl Workspace {
    /// Allocates every buffer to the shape implied by `data`.
    ///
    /// ### Params
    ///
    /// * `data` - The gene whose shape sets the buffer sizes
    ///
    /// ### Returns
    ///
    /// A zeroed workspace.
    fn new(data: &PmlData<'_>) -> Self {
        let n_cells = data.n_cells();
        let k = data.n_subjects();
        let nb = data.n_beta();
        Self {
            extb: vec![0.0; n_cells],
            extbphil: vec![0.0; n_cells],
            phi: vec![0.0; n_cells],
            gstar: vec![0.0; n_cells],
            deriv: vec![0.0; n_cells],
            x_phi: vec![0.0; n_cells * nb],
            w: vec![0.0; k],
            yx: vec![0.0; nb],
            db: vec![0.0; nb],
            dw: vec![0.0; k],
            vw: vec![0.0; k],
            vwb: vec![0.0; k * nb],
            vb: vec![0.0; nb * nb],
            vb2: vec![0.0; nb * nb],
            factor: vec![0.0; nb * nb],
            perm: vec![0; nb],
            tmp: vec![0.0; nb],
            step_beta: vec![0.0; nb],
            step_log_w: vec![0.0; k],
            dwvw: vec![0.0; k],
            damp_beta: vec![0.0; nb],
            damp_log_w: vec![0.0; k],
            new_beta: vec![0.0; nb],
            new_log_w: vec![0.0; k],
            third: vec![0.0; k],
            fourth: vec![0.0; k],
        }
    }

    /// Fills the two quantities that do not change across iterations.
    ///
    /// ### Params
    ///
    /// * `data` - The gene
    /// * `gamma` - Cell-level negative binomial size
    fn seed(&mut self, data: &PmlData<'_>, gamma: f64) {
        let nb = data.n_beta();
        self.gstar.fill(gamma);
        for (i, &c) in data.cell_index.iter().enumerate() {
            self.gstar[c] += data.counts[i];
        }
        self.yx.fill(0.0);
        for (i, &c) in data.cell_index.iter().enumerate() {
            let row = &data.design[c * nb..(c + 1) * nb];
            let y = data.counts[i];
            for j in 0..nb {
                self.yx[j] += row[j] * y;
            }
        }
    }

    /// Evaluates the penalised log-likelihood, leaving `extb`, `extbphil` and
    /// `w` at the given point.
    ///
    /// ### Params
    ///
    /// * `data` - The gene
    /// * `beta` - Fixed effects
    /// * `log_w` - Random effects on the log scale
    /// * `gamma` - Cell-level negative binomial size
    /// * `penalty` - Prior on the random effects
    ///
    /// ### Returns
    ///
    /// The penalised log-likelihood.
    fn evaluate(
        &mut self,
        data: &PmlData<'_>,
        beta: &[f64],
        log_w: &[f64],
        gamma: f64,
        penalty: Penalty,
    ) -> f64 {
        let nb = data.n_beta();
        let n_cells = data.n_cells();
        let k = data.n_subjects();

        // The C++ evaluates `X*beta` into a temporary and adds the offset
        // afterwards; keeping that order keeps the last-iteration comparison
        // identical.
        for r in 0..n_cells {
            let row = &data.design[r * nb..(r + 1) * nb];
            let mut acc = 0.0;
            for j in 0..nb {
                acc += row[j] * beta[j];
            }
            self.extb[r] = data.offset[r] + acc;
        }

        let mut log_likelihood = 0.0;
        for (i, &c) in data.cell_index.iter().enumerate() {
            log_likelihood += self.extb[c] * data.counts[i];
        }
        for s in 0..k {
            log_likelihood += log_w[s] * data.subject_total[s];
        }

        for s in 0..k {
            for e in &mut self.extb[data.subject_start[s]..data.subject_start[s + 1]] {
                *e += log_w[s];
            }
        }
        for e in &mut self.extb[..n_cells] {
            *e = e.exp();
        }

        let mut sum_phil = 0.0;
        for r in 0..n_cells {
            self.extbphil[r] = (self.extb[r] + gamma).ln();
            sum_phil += self.extbphil[r];
        }
        log_likelihood -= gamma * sum_phil;
        for (i, &c) in data.cell_index.iter().enumerate() {
            log_likelihood -= data.counts[i] * self.extbphil[c];
        }

        for s in 0..k {
            self.w[s] = log_w[s].exp();
        }
        penalty.accumulate(log_likelihood, log_w, &self.w)
    }

    /// Fills `db` and `dw` from the current `extb` and `w`.
    ///
    /// Leaves `phi` holding `gstar extb / (extb + gamma)`, the weight the
    /// curvature assembly divides down again.
    ///
    /// ### Params
    ///
    /// * `data` - The gene
    /// * `gamma` - Cell-level negative binomial size
    /// * `penalty` - Prior on the random effects
    /// * `log_w` - Current random effects, needed by the log-normal penalty
    fn gradient_only(&mut self, data: &PmlData<'_>, gamma: f64, penalty: Penalty, log_w: &[f64]) {
        let nb = data.n_beta();
        let n_cells = data.n_cells();
        let k = data.n_subjects();

        for r in 0..n_cells {
            self.phi[r] = self.gstar[r] / (1.0 + gamma / self.extb[r]);
        }

        // `db = yx - X' phi`, accumulated in cell order so each component sums
        // in the same order as the Eigen gemv.
        self.db.copy_from_slice(&self.yx);
        for r in 0..n_cells {
            let row = &data.design[r * nb..(r + 1) * nb];
            let p = self.phi[r];
            for j in 0..nb {
                self.db[j] -= row[j] * p;
            }
        }

        for s in 0..k {
            let mut acc = 0.0;
            for r in data.subject_start[s]..data.subject_start[s + 1] {
                acc += self.phi[r];
            }
            self.dw[s] = (data.subject_total[s] - acc) + penalty.gradient(log_w[s], self.w[s]);
        }
    }

    /// Fills `vw`, `vwb`, `vb` and `vb2` from the weight left in `phi`.
    ///
    /// ### Params
    ///
    /// * `data` - The gene
    /// * `gamma` - Cell-level negative binomial size
    /// * `penalty` - Prior on the random effects
    fn curvature(&mut self, data: &PmlData<'_>, gamma: f64, penalty: Penalty) {
        let nb = data.n_beta();
        let n_cells = data.n_cells();
        let k = data.n_subjects();

        for r in 0..n_cells {
            self.phi[r] /= self.extb[r] + gamma;
        }

        for s in 0..k {
            let mut acc = 0.0;
            for r in data.subject_start[s]..data.subject_start[s + 1] {
                acc += self.phi[r];
            }
            self.vw[s] = gamma * acc + penalty.curvature(self.w[s]);
        }

        for r in 0..n_cells {
            let row = &data.design[r * nb..(r + 1) * nb];
            let p = self.phi[r];
            let out = &mut self.x_phi[r * nb..(r + 1) * nb];
            for j in 0..nb {
                out[j] = row[j] * p;
            }
        }

        for s in 0..k {
            let dst = &mut self.vwb[s * nb..(s + 1) * nb];
            dst.fill(0.0);
            for r in data.subject_start[s]..data.subject_start[s + 1] {
                let src = &self.x_phi[r * nb..(r + 1) * nb];
                for j in 0..nb {
                    dst[j] += src[j];
                }
            }
            for v in dst.iter_mut() {
                *v *= gamma;
            }
        }

        self.vb.fill(0.0);
        for r in 0..n_cells {
            let row = &data.design[r * nb..(r + 1) * nb];
            let scaled = &self.x_phi[r * nb..(r + 1) * nb];
            for i in 0..nb {
                for j in i..nb {
                    self.vb[i * nb + j] += row[i] * scaled[j];
                }
            }
        }
        for i in 0..nb {
            for j in i..nb {
                let v = gamma * self.vb[i * nb + j];
                self.vb[i * nb + j] = v;
                self.vb[j * nb + i] = v;
            }
        }

        // Schur complement in the square-rooted form the C++ uses: scale the
        // cross block by `1/sqrt(vw)` and subtract its own Gram matrix, which
        // keeps the result symmetric by construction.
        for s in 0..k {
            let scale = 1.0 / self.vw[s].sqrt();
            for v in &mut self.vwb[s * nb..(s + 1) * nb] {
                *v *= scale;
            }
        }
        for i in 0..nb {
            for j in 0..nb {
                let mut acc = 0.0;
                for s in 0..k {
                    acc += self.vwb[s * nb + i] * self.vwb[s * nb + j];
                }
                self.vb2[i * nb + j] = self.vb[i * nb + j] - acc;
            }
        }
        // Undo the scaling: the step needs the unscaled cross block.
        for s in 0..k {
            let scale = self.vw[s].sqrt();
            for v in &mut self.vwb[s * nb..(s + 1) * nb] {
                *v *= scale;
            }
        }
    }
}

////////////
// Solver //
////////////

/// The shared Newton loop behind [`opt_pml`] and [`opt_pml_nbm`].
///
/// ### Params
///
/// * `data` - One gene's design, offsets, positive counts and subject blocks
/// * `beta_init` - Starting fixed effects
/// * `penalty` - Prior on the random effects, scalars already resolved
/// * `gamma` - Cell-level negative binomial size
/// * `params` - Tuning knobs
///
/// ### Returns
///
/// The fitted solution and its diagnostics.
fn optimise(
    data: &PmlData<'_>,
    beta_init: &[f64],
    penalty: Penalty,
    gamma: f64,
    params: PmlParams,
) -> Result<PmlResult, EdgeErrors> {
    data.validate()?;
    let nb = data.n_beta();
    let k = data.n_subjects();
    if beta_init.len() != nb {
        return Err(EdgeErrors::LengthMismatch {
            name: "beta_init",
            expected: nb,
            got: beta_init.len(),
        });
    }
    if !(gamma.is_finite() && gamma > 0.0) {
        return Err(EdgeErrors::InvalidDispersion(gamma));
    }

    let mut beta = beta_init.to_vec();
    let mut log_w = vec![0.0; k];
    let mut work = Workspace::new(data);
    work.seed(data, gamma);

    let mut log_likelihood = work.evaluate(data, &beta, &log_w, gamma, penalty);
    let mut log_likelihood_prev = 0.0;
    let mut likdif = 0.0;
    let mut step = 0usize;
    let mut backtracks = 0usize;

    while step == 0 || (likdif > params.eps && step < params.max_iter) {
        step += 1;
        work.damp_beta.fill(1.0);
        work.damp_log_w.fill(1.0);

        work.gradient_only(data, gamma, penalty, &log_w);
        work.curvature(data, gamma, penalty);

        for s in 0..k {
            work.dwvw[s] = work.dw[s] / work.vw[s];
        }
        // `rhs = db - vwb' (dw/vw)`, then one `nb` solve gives the whole step.
        for j in 0..nb {
            let mut acc = 0.0;
            for s in 0..k {
                acc += work.vwb[s * nb + j] * work.dwvw[s];
            }
            work.step_beta[j] = work.db[j] - acc;
        }
        work.factor.copy_from_slice(&work.vb2);
        ldlt_solve(
            &mut work.factor,
            &mut work.step_beta,
            nb,
            &mut work.perm,
            &mut work.tmp,
        );
        for s in 0..k {
            let mut acc = 0.0;
            for j in 0..nb {
                acc += work.vwb[s * nb + j] * work.step_beta[j];
            }
            work.step_log_w[s] = work.dwvw[s] - acc / work.vw[s];
        }

        for j in 0..nb {
            work.new_beta[j] = beta[j] + work.step_beta[j];
        }
        for s in 0..k {
            work.new_log_w[s] = log_w[s] + work.step_log_w[s];
        }

        log_likelihood_prev = log_likelihood;
        log_likelihood = evaluate_trial(&mut work, data, gamma, penalty);
        likdif = log_likelihood - log_likelihood_prev;

        backtracks = 0;
        let mut min_step = STEP_CUTOFF;
        while likdif < 0.0 || log_likelihood.is_infinite() {
            backtracks += 1;
            min_step /= 2.0;

            if backtracks > params.max_backtrack {
                likdif = 0.0;
                log_likelihood = log_likelihood_prev;
                let grad_beta = work.db.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                let grad_log_w = work.dw.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                if grad_beta > GRADIENT_TOLERANCE || grad_log_w > GRADIENT_TOLERANCE {
                    backtracks += 1;
                }
                break;
            }

            // A step past the cutoff is not halved but replaced outright: at
            // that magnitude the halvings would need forty rounds to reach a
            // usable displacement, and the budget is ten.
            for j in 0..nb {
                let d = work.step_beta[j];
                if d < STEP_CUTOFF && d > -STEP_CUTOFF {
                    work.damp_beta[j] /= 2.0;
                    work.new_beta[j] = beta[j] + d * work.damp_beta[j];
                } else if d > 0.0 {
                    work.new_beta[j] = beta[j] + min_step;
                } else {
                    work.new_beta[j] = beta[j] - min_step;
                }
            }
            for s in 0..k {
                let d = work.step_log_w[s];
                if d < STEP_CUTOFF && d > -STEP_CUTOFF {
                    work.damp_log_w[s] /= 2.0;
                    work.new_log_w[s] = log_w[s] + d * work.damp_log_w[s];
                } else if d > 0.0 {
                    work.new_log_w[s] = log_w[s] + min_step;
                } else {
                    work.new_log_w[s] = log_w[s] - min_step;
                }
            }

            log_likelihood = evaluate_trial(&mut work, data, gamma, penalty);
            likdif = log_likelihood - log_likelihood_prev;
        }

        beta.copy_from_slice(&work.new_beta);
        log_w.copy_from_slice(&work.new_log_w);
    }

    // `vw` and `vb2` are the ones assembled at the previous iterate, which is
    // what nebula reports. Recomputing them at the final point would be a
    // different quantity, and the outer overdispersion objective is calibrated
    // against this one.
    let mut log_det = 0.0;
    for s in 0..k {
        log_det += work.vw[s].abs().ln();
    }
    if params.reml {
        work.factor.copy_from_slice(&work.vb2);
        let det = determinant(&mut work.factor, nb, &mut work.perm);
        log_det += det.abs().ln();
    }

    let second_order = match penalty {
        Penalty::Gamma { lambda, .. } if params.ord > 1 => {
            laplace_correction(&mut work, data, gamma, lambda, params.ord)
        }
        _ => 0.0,
    };

    let mut result = PmlResult {
        beta,
        log_w,
        information: work.vb2.clone(),
        log_likelihood,
        log_likelihood_prev,
        log_det,
        second_order,
        convergence: CONV_SUCCESS,
        iterations: step,
        backtracks,
    };
    result.convergence = check_convergence(&result, &params, false, CONV_SUCCESS);
    Ok(result)
}

/// Evaluates the penalised log-likelihood at the trial point in the workspace.
///
/// ### Params
///
/// * `work` - Workspace holding `new_beta` and `new_log_w`
/// * `data` - The gene
/// * `gamma` - Cell-level negative binomial size
/// * `penalty` - Prior on the random effects
///
/// ### Returns
///
/// The penalised log-likelihood, with `extb` and `w` left at the trial point.
fn evaluate_trial(work: &mut Workspace, data: &PmlData<'_>, gamma: f64, penalty: Penalty) -> f64 {
    let beta = std::mem::take(&mut work.new_beta);
    let log_w = std::mem::take(&mut work.new_log_w);
    let value = work.evaluate(data, &beta, &log_w, gamma, penalty);
    work.new_beta = beta;
    work.new_log_w = log_w;
    value
}

/// Higher-order terms of the Laplace expansion of the marginal likelihood.
///
/// The leading Laplace term is `-0.5 log|V|`; this returns the next correction,
/// which the outer objective adds as `log(1 + second)`. `ord == 2` keeps only
/// the third-derivative term `5 m3^2 / (24 V^3)`; `ord == 3` adds the
/// fourth-derivative term `-m4 / (8 V^2)` and the mixed term
/// `7 m5 m3 / (48 V^4)`. Every derivative is diagonal in the subjects, so this
/// is a scan over cells and a sum over blocks.
///
/// Recomputes the curvature at the final iterate rather than reusing the one
/// the log-determinant was taken from, which is what nebula does.
///
/// ### Params
///
/// * `work` - Workspace whose `extb` and `w` sit at the final iterate
/// * `data` - The gene
/// * `gamma` - Cell-level negative binomial size
/// * `lambda` - Rate of the gamma prior
/// * `ord` - Expansion order, greater than one
///
/// ### Returns
///
/// The correction, nebula's `second`.
fn laplace_correction(
    work: &mut Workspace,
    data: &PmlData<'_>,
    gamma: f64,
    lambda: f64,
    ord: u32,
) -> f64 {
    let n_cells = data.n_cells();
    let k = data.n_subjects();
    let mut second = 0.0;

    for r in 0..n_cells {
        work.phi[r] = work.gstar[r] / (1.0 + gamma / work.extb[r]);
        work.extbphil[r] = work.extb[r] + gamma;
        work.phi[r] /= work.extbphil[r];
    }
    for s in 0..k {
        let mut acc = 0.0;
        for r in data.subject_start[s]..data.subject_start[s + 1] {
            acc += work.phi[r];
        }
        work.vw[s] = gamma * acc + lambda * work.w[s];
    }

    for r in 0..n_cells {
        work.phi[r] /= work.extbphil[r];
        work.deriv[r] = work.phi[r] * (gamma - work.extb[r]);
    }
    for s in 0..k {
        let mut acc = 0.0;
        for r in data.subject_start[s]..data.subject_start[s + 1] {
            acc += work.deriv[r];
        }
        work.third[s] = gamma * acc + lambda * work.w[s];
    }
    let mut acc = 0.0;
    for s in 0..k {
        let v = work.vw[s];
        acc += work.third[s] * work.third[s] / (v * v * v);
    }
    second += 5.0 * acc / 24.0;

    if ord > 2 {
        for r in 0..n_cells {
            work.phi[r] /= work.extbphil[r];
            let e = work.extb[r];
            work.deriv[r] = work.phi[r] * (gamma * gamma + e * e - (4.0 * gamma) * e);
        }
        for s in 0..k {
            let mut acc = 0.0;
            for r in data.subject_start[s]..data.subject_start[s + 1] {
                acc += work.deriv[r];
            }
            work.fourth[s] = gamma * acc + lambda * work.w[s];
        }
        let mut acc = 0.0;
        for s in 0..k {
            let v = work.vw[s];
            acc += work.fourth[s] / (v * v);
        }
        second -= acc / 8.0;

        for r in 0..n_cells {
            work.phi[r] /= work.extbphil[r];
            let e = work.extb[r];
            let e2 = e * e;
            work.deriv[r] = work.phi[r]
                * (gamma * gamma * gamma - (11.0 * gamma * gamma) * e + (11.0 * gamma) * e2
                    - e2 * e);
        }
        for s in 0..k {
            let mut acc = 0.0;
            for r in data.subject_start[s]..data.subject_start[s + 1] {
                acc += work.deriv[r];
            }
            work.fourth[s] = gamma * acc + lambda * work.w[s];
        }
        let mut acc = 0.0;
        for s in 0..k {
            let vs = work.vw[s] * work.vw[s];
            acc += work.fourth[s] * work.third[s] / (vs * vs);
        }
        second += 7.0 * acc / 48.0;
    }

    second
}

////////////////////
// Linear algebra //
////////////////////

/// Solves a small symmetric system in place by pivoted `LDL'`.
///
/// The Schur complement is positive definite at any sensible optimum, but the
/// backtracking path visits points where it is not, and a plain Cholesky would
/// simply fail there. This mirrors Eigen's `LDLT`, which is what the C++ calls:
/// symmetric pivoting on the largest remaining diagonal, and a solve that zeroes
/// rather than divides by a vanishing pivot. That keeps a rank-deficient design
/// producing a finite step instead of a NaN.
///
/// See [`crate::glm::levenberg`] for the unpivoted Cholesky used where positive
/// definiteness is guaranteed; the extra pivoting here is not free and is only
/// worth it because indefinite iterates are expected.
///
/// ### Params
///
/// * `a` - Row-major `n * n` symmetric matrix, overwritten by the factor
/// * `b` - Right-hand side, overwritten by the solution
/// * `n` - System size
/// * `perm` - Scratch for the transposition record, length `n`
/// * `tmp` - Scratch row, length `n`
fn ldlt_solve(a: &mut [f64], b: &mut [f64], n: usize, perm: &mut [usize], tmp: &mut [f64]) {
    for step in 0..n {
        let mut pivot = step;
        let mut best = a[step * n + step].abs();
        for i in (step + 1)..n {
            let v = a[i * n + i].abs();
            if v > best {
                best = v;
                pivot = i;
            }
        }
        perm[step] = pivot;
        if pivot != step {
            for j in 0..step {
                a.swap(step * n + j, pivot * n + j);
            }
            for i in (pivot + 1)..n {
                a.swap(i * n + step, i * n + pivot);
            }
            a.swap(step * n + step, pivot * n + pivot);
            for i in (step + 1)..pivot {
                a.swap(i * n + step, pivot * n + i);
            }
        }

        if step > 0 {
            for j in 0..step {
                tmp[j] = a[j * n + j] * a[step * n + j];
            }
            let mut acc = 0.0;
            for j in 0..step {
                acc += a[step * n + j] * tmp[j];
            }
            a[step * n + step] -= acc;
            for i in (step + 1)..n {
                let mut acc = 0.0;
                for j in 0..step {
                    acc += a[i * n + j] * tmp[j];
                }
                a[i * n + step] -= acc;
            }
        }

        let d = a[step * n + step];
        if d != 0.0 {
            for i in (step + 1)..n {
                a[i * n + step] /= d;
            }
        }
    }

    for step in 0..n {
        b.swap(step, perm[step]);
    }
    for i in 0..n {
        let mut acc = b[i];
        for j in 0..i {
            acc -= a[i * n + j] * b[j];
        }
        b[i] = acc;
    }
    for i in 0..n {
        let d = a[i * n + i];
        // Eigen's own tolerance: a pivot below the smallest normal double is
        // treated as an exact zero and the component is dropped rather than
        // blown up.
        b[i] = if d.abs() > f64::MIN_POSITIVE {
            b[i] / d
        } else {
            0.0
        };
    }
    for i in (0..n).rev() {
        let mut acc = b[i];
        for j in (i + 1)..n {
            acc -= a[j * n + i] * b[j];
        }
        b[i] = acc;
    }
    for step in (0..n).rev() {
        b.swap(step, perm[step]);
    }
}

/// Determinant of a small square matrix by partial-pivot `LU`.
///
/// Returned as a product rather than a sum of logs, matching the C++, so an
/// under- or overflowing determinant behaves the same way here as there.
///
/// ### Params
///
/// * `a` - Row-major `n * n` matrix, overwritten by the factor
/// * `n` - Matrix order
/// * `perm` - Scratch, length `n`
///
/// ### Returns
///
/// The signed determinant.
fn determinant(a: &mut [f64], n: usize, perm: &mut [usize]) -> f64 {
    let mut det = 1.0;
    perm.fill(0);
    for step in 0..n {
        let mut pivot = step;
        let mut best = a[step * n + step].abs();
        for i in (step + 1)..n {
            let v = a[i * n + step].abs();
            if v > best {
                best = v;
                pivot = i;
            }
        }
        if pivot != step {
            for j in 0..n {
                a.swap(step * n + j, pivot * n + j);
            }
            det = -det;
        }
        let d = a[step * n + step];
        det *= d;
        if d == 0.0 {
            return 0.0;
        }
        for i in (step + 1)..n {
            let factor = a[i * n + step] / d;
            a[i * n + step] = factor;
            for j in (step + 1)..n {
                a[i * n + j] -= factor * a[step * n + j];
            }
        }
    }
    det
}

/// Smallest eigenvalue of a small symmetric matrix, by cyclic Jacobi.
///
/// Only used for the identifiability test, where the answer is compared against
/// [`EIGENVALUE_CUTOFF`] and nothing finer is needed. Jacobi is chosen over a
/// faer decomposition because the matrices are a handful of columns wide and the
/// dispatch would cost more than the sweeps.
///
/// ### Params
///
/// * `matrix` - Row-major `n * n` symmetric matrix, not modified
/// * `n` - Matrix order
///
/// ### Returns
///
/// The smallest eigenvalue, or the single entry when `n` is one.
fn min_eigenvalue(matrix: &[f64], n: usize) -> f64 {
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return matrix[0];
    }
    let mut a = matrix.to_vec();
    let mut frobenius = 0.0;
    for &v in &a {
        frobenius += v * v;
    }
    let tolerance = (frobenius * f64::EPSILON * f64::EPSILON).max(f64::MIN_POSITIVE);

    for _ in 0..JACOBI_SWEEPS {
        let mut off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off += a[p * n + q] * a[p * n + q];
            }
        }
        if off <= tolerance {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq == 0.0 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for r in 0..n {
                    let arp = a[r * n + p];
                    let arq = a[r * n + q];
                    a[r * n + p] = c * arp - s * arq;
                    a[r * n + q] = s * arp + c * arq;
                }
                for r in 0..n {
                    let apr = a[p * n + r];
                    let aqr = a[q * n + r];
                    a[p * n + r] = c * apr - s * aqr;
                    a[q * n + r] = s * apr + c * aqr;
                }
            }
        }
    }
    (0..n).map(|i| a[i * n + i]).fold(f64::INFINITY, f64::min)
}

///////////
// Tests //
///////////

#[cfg(test)]
// Reference values are pasted verbatim from R's 17-digit print so they can be
// diffed against the generating script. Truncating them to the shortest
// round-trip would change nothing numerically and lose that.
#[allow(clippy::excessive_precision)]
// The fixture builders return the five parallel arrays PmlData borrows. Naming
// the tuple would be one alias used once.
#[allow(clippy::type_complexity)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Design of fixture A, row-major `12 * 2`: intercept plus a cell-level
    /// binary covariate.
    const DESIGN: [f64; 24] = [
        1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0,
        1.0, 0.0, 1.0, 1.0, 1.0, 1.0,
    ];

    /// Offsets of fixture A, all dyadic so they round-trip exactly through R.
    const OFFSET: [f64; 12] = [
        0.25, -0.5, 0.75, 0.0, 0.5, -0.25, 0.125, 0.375, -0.125, 0.25, 0.0, -0.375,
    ];

    /// Subject block boundaries: three subjects of four, five and three cells.
    const SUBJECT_START: [usize; 4] = [0, 4, 9, 12];

    /// Positive counts of fixture A.
    const COUNTS: [f64; 8] = [3.0, 5.0, 2.0, 4.0, 1.0, 7.0, 2.0, 6.0];

    /// Cells holding those counts.
    const CELL_INDEX: [usize; 8] = [0, 2, 3, 5, 6, 7, 9, 10];

    /// Per-subject totals of fixture A.
    const SUBJECT_TOTAL: [f64; 3] = [10.0, 12.0, 8.0];

    /// Positive counts of fixture B, where subject two is entirely zero.
    const COUNTS_B: [f64; 5] = [3.0, 5.0, 2.0, 2.0, 6.0];

    /// Cells holding those counts.
    const CELL_INDEX_B: [usize; 5] = [0, 2, 3, 9, 10];

    /// Per-subject totals of fixture B.
    const SUBJECT_TOTAL_B: [f64; 3] = [10.0, 0.0, 8.0];

    /// Relative tolerance against the R package.
    ///
    /// Three orders tighter than the 1e-8 this port was asked for, and reached
    /// on every fixture below, converged and stalled alike. The binding
    /// constraint is not this arithmetic but nebula's own `eps`: it stops on an
    /// absolute improvement in the log-likelihood, so where a Newton step is a
    /// near-tie, a summation order that shifts the comparison by one ulp adds or
    /// drops a whole damped step and moves the answer by 1e-7. Fixtures here are
    /// chosen away from those ties.
    const TOL: f64 = 1e-11;

    /// Fixture A: a well-behaved gene.
    fn fixture_a() -> PmlData<'static> {
        PmlData {
            design: &DESIGN,
            offset: &OFFSET,
            counts: &COUNTS,
            cell_index: &CELL_INDEX,
            subject_start: &SUBJECT_START,
            subject_total: &SUBJECT_TOTAL,
        }
    }

    /// Fixture B: the same gene with every count in subject two set to zero.
    fn fixture_b() -> PmlData<'static> {
        PmlData {
            design: &DESIGN,
            offset: &OFFSET,
            counts: &COUNTS_B,
            cell_index: &CELL_INDEX_B,
            subject_start: &SUBJECT_START,
            subject_total: &SUBJECT_TOTAL_B,
        }
    }

    /// Compares two slices elementwise at [`TOL`].
    ///
    /// ### Params
    ///
    /// * `got` - Values produced here
    /// * `want` - Values produced by nebula 1.5.8
    fn assert_slice(got: &[f64], want: &[f64]) {
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want) {
            assert_relative_eq!(*g, *w, max_relative = TOL, epsilon = 1e-12);
        }
    }

    // Reference values from nebula 1.5.8:
    //   Rscript -e 'library(nebula); nebula:::opt_pml(X, offset, Y, fid, cumsumy,
    //     posind, posindy, nb, nind, k, beta, sigma, reml, eps, ord)'
    // with X, offset, Y, fid, cumsumy, posindy as the constants above.

    #[test]
    fn test_opt_pml_well_behaved_matches_nebula() {
        let data = fixture_a();
        let variance = PmlVariance {
            subject: 0.25,
            cell: 2.0,
        };
        let fit = opt_pml(&data, &[0.5, -0.25], &variance, None).expect("fit");

        assert_slice(&fit.beta, &[1.1055943907637618, -0.98348970166026206]);
        assert_slice(
            &fit.log_w,
            &[
                -0.025429869309964156,
                0.022419629678247836,
                0.33781545877640862,
            ],
        );
        assert_slice(
            &fit.information,
            &[
                5.4708370512237092,
                2.6193993778249318,
                2.6193993778249318,
                4.0223986651682804,
            ],
        );
        assert_relative_eq!(fit.log_likelihood, -60.378642466146786, max_relative = TOL);
        assert_relative_eq!(fit.log_det, 6.0421923783970968, max_relative = TOL);
        assert_eq!(fit.second_order, 0.0);
        assert_eq!(fit.iterations, 4);
        assert_eq!(fit.backtracks, 0);
        assert_eq!(fit.convergence, CONV_SUCCESS);
    }

    #[test]
    fn test_opt_pml_third_order_laplace_correction() {
        let data = fixture_a();
        let variance = PmlVariance {
            subject: 0.25,
            cell: 2.0,
        };
        let params = PmlParams {
            ord: 3,
            ..Default::default()
        };
        let fit = opt_pml(&data, &[0.5, -0.25], &variance, Some(params)).expect("fit");

        // Same optimum, only the correction differs from ord = 1.
        assert_slice(&fit.beta, &[1.1055943907637618, -0.98348970166026206]);
        assert_relative_eq!(fit.second_order, 0.0028301625845031939, max_relative = TOL);
    }

    #[test]
    fn test_opt_pml_reml_log_determinant() {
        let data = fixture_a();
        let variance = PmlVariance {
            subject: 0.25,
            cell: 2.0,
        };
        let params = PmlParams {
            reml: true,
            ..Default::default()
        };
        let fit = opt_pml(&data, &[0.5, -0.25], &variance, Some(params)).expect("fit");

        assert_relative_eq!(fit.log_det, 8.759838692617901, max_relative = TOL);
        assert_slice(&fit.beta, &[1.1055943907637618, -0.98348970166026206]);
    }

    #[test]
    fn test_opt_pml_subject_with_all_zero_counts() {
        let data = fixture_b();
        let variance = PmlVariance {
            subject: 0.25,
            cell: 2.0,
        };
        let fit = opt_pml(&data, &[0.5, -0.25], &variance, None).expect("fit");

        assert_slice(&fit.beta, &[0.15781792039265752, -0.098178037869394941]);
        assert_slice(
            &fit.log_w,
            &[
                0.31781101972092152,
                -0.83477809302857631,
                0.46458776408134972,
            ],
        );
        assert_slice(
            &fit.information,
            &[
                4.9151627959404793,
                2.59325292186585,
                2.59325292186585,
                3.5781777249175555,
            ],
        );
        assert_relative_eq!(fit.log_likelihood, -50.769290856943599, max_relative = TOL);
        assert_relative_eq!(fit.log_det, 5.3802869340559738, max_relative = TOL);
        assert_eq!(fit.convergence, CONV_SUCCESS);
    }

    #[test]
    fn test_opt_pml_zero_subject_third_order() {
        let data = fixture_b();
        let variance = PmlVariance {
            subject: 0.25,
            cell: 2.0,
        };
        let params = PmlParams {
            ord: 3,
            ..Default::default()
        };
        let fit = opt_pml(&data, &[0.5, -0.25], &variance, Some(params)).expect("fit");
        assert_relative_eq!(fit.second_order, 0.029026901671004347, max_relative = TOL);
    }

    #[test]
    fn test_opt_pml_start_near_bound_backtracks() {
        // beta[0] = 90 sits just inside nebula's box of plus or minus 100, so
        // the first Newton step overshoots and the per-coordinate damping runs.
        let data = fixture_a();
        let variance = PmlVariance {
            subject: 0.25,
            cell: 2.0,
        };
        let fit = opt_pml(&data, &[90.0, 0.0], &variance, None).expect("fit");

        assert_slice(&fit.beta, &[1.1055943759446161, -0.98348968952057669]);
        assert_slice(
            &fit.log_w,
            &[
                -0.025429869876909816,
                0.022419639320108459,
                0.3378154594265026,
            ],
        );
        assert_slice(
            &fit.information,
            &[
                5.4711020927179739,
                2.6193969725187061,
                2.6193969725187061,
                4.0224446400576319,
            ],
        );
        assert_eq!(fit.iterations, 9);
        assert_eq!(fit.convergence, CONV_SUCCESS);
    }

    #[test]
    fn test_opt_pml_far_outside_bounds_uses_fixed_displacement() {
        // Both coordinates start well outside the box, so the Newton step
        // exceeds the cutoff and the fixed-displacement branch takes over.
        let data = fixture_a();
        let variance = PmlVariance {
            subject: 0.25,
            cell: 2.0,
        };
        let fit = opt_pml(&data, &[-90.0, 60.0], &variance, None).expect("fit");

        assert_slice(&fit.beta, &[1.1055943725487707, -0.98348968650982715]);
        assert_relative_eq!(fit.log_det, 6.0423215051590002, max_relative = TOL);
        assert_eq!(fit.iterations, 12);
    }

    #[test]
    fn test_opt_pml_critical_point_reports_minus_ten() {
        // Backtracking is exhausted while every gradient stays under 0.01, so
        // nebula calls it a critical point rather than a failure. The stall here
        // is decisive: the rejected step is orders of magnitude from improving,
        // not a near-tie. Near-tie stalls exist and are not testable to this
        // tolerance, because whether the backtracking runs at all is then
        // decided by the last ulp of a log-likelihood in the thousands.
        let data = fixture_a();
        let variance = PmlVariance {
            subject: 8.0,
            cell: 0.001,
        };
        let fit = opt_pml(&data, &[15.0, 5.0], &variance, None).expect("fit");

        assert_slice(&fit.beta, &[4.9609375, -4.9609375]);
        assert_slice(&fit.log_w, &[0.0390625, 0.0390625, 4.5234136840078936]);
        assert_slice(
            &fit.information,
            &[
                7.0733722800021132e-05,
                7.0105348391562988e-05,
                7.0105348391562988e-05,
                0.00016310178559274507,
            ],
        );
        assert_relative_eq!(fit.log_det, -20.515549220945573, max_relative = TOL);
        assert_eq!(fit.iterations, 3);
        assert_eq!(fit.backtracks, 11);
        assert_eq!(fit.convergence, CONV_CRITICAL_POINT);
    }

    #[test]
    fn test_opt_pml_stalled_reports_minus_forty() {
        let data = fixture_a();
        let variance = PmlVariance {
            subject: 8.0,
            cell: 0.001,
        };
        let fit = opt_pml(&data, &[20.0, 0.0], &variance, None).expect("fit");

        assert_slice(&fit.beta, &[14.9609375, -4.9609375]);
        assert_slice(&fit.log_w, &[-4.9609375, -4.9609375, 4.9962571608576489]);
        assert_eq!(fit.iterations, 2);
        assert_eq!(fit.backtracks, 12);
        assert_eq!(fit.convergence, CONV_STALLED);
    }

    #[test]
    fn test_opt_pml_iteration_budget_reports_minus_twenty() {
        // A negative eps makes the stopping test unsatisfiable, so the loop runs
        // out its budget at an otherwise converged point.
        let data = fixture_a();
        let variance = PmlVariance {
            subject: 0.25,
            cell: 2.0,
        };
        let params = PmlParams {
            eps: -1e-300,
            ..Default::default()
        };
        let fit = opt_pml(&data, &[0.5, -0.25], &variance, Some(params)).expect("fit");

        assert_eq!(fit.iterations, 50);
        assert_eq!(fit.convergence, CONV_MAX_ITER);
        assert_slice(&fit.beta, &[1.1055943907637629, -0.98348970166026151]);
        assert_slice(
            &fit.information,
            &[
                5.4708368635191373,
                2.619399280826618,
                2.619399280826618,
                4.0223986123865263,
            ],
        );
    }

    #[test]
    fn test_opt_pml_collinear_design_reports_minus_twentyfive() {
        // The covariate is duplicated, so the information matrix is singular.
        let design: Vec<f64> = DESIGN
            .chunks_exact(2)
            .flat_map(|r| [r[0], r[1], r[1]])
            .collect();
        let data = PmlData {
            design: &design,
            offset: &OFFSET,
            counts: &COUNTS,
            cell_index: &CELL_INDEX,
            subject_start: &SUBJECT_START,
            subject_total: &SUBJECT_TOTAL,
        };
        let variance = PmlVariance {
            subject: 0.25,
            cell: 2.0,
        };
        let fit = opt_pml(&data, &[0.5, -0.25, 0.1], &variance, None).expect("fit");

        assert_eq!(fit.convergence, CONV_SINGULAR);
        assert_slice(
            &fit.beta,
            &[1.1055943330380458, -1.0834897078722583, 0.10000000000000001],
        );
        assert_relative_eq!(fit.log_det, 6.042689573294485, max_relative = TOL);
    }

    #[test]
    fn test_opt_pml_single_coefficient() {
        let design = [1.0; 12];
        let data = PmlData {
            design: &design,
            offset: &OFFSET,
            counts: &COUNTS,
            cell_index: &CELL_INDEX,
            subject_start: &SUBJECT_START,
            subject_total: &SUBJECT_TOTAL,
        };
        let variance = PmlVariance {
            subject: 0.25,
            cell: 2.0,
        };
        let params = PmlParams {
            reml: true,
            ord: 3,
            ..Default::default()
        };
        let fit = opt_pml(&data, &[0.0], &variance, Some(params)).expect("fit");

        assert_slice(&fit.beta, &[0.64526354365336647]);
        assert_slice(
            &fit.log_w,
            &[
                0.040002950937536336,
                0.10805651074009694,
                0.21875064209166895,
            ],
        );
        assert_slice(&fit.information, &[5.7533150000267472]);
        assert_relative_eq!(fit.log_det, 7.9235375511695514, max_relative = TOL);
        assert_relative_eq!(fit.second_order, 0.0044134970888637676, max_relative = TOL);
    }

    #[test]
    fn test_opt_pml_nbm_matches_nebula() {
        let data = fixture_a();
        let variance = PmlVariance {
            subject: 0.3,
            cell: 2.0,
        };
        let params = PmlParams {
            reml: true,
            ..Default::default()
        };
        let fit = opt_pml_nbm(&data, &[0.5, -0.25], &variance, Some(params)).expect("fit");

        assert_slice(&fit.beta, &[1.2213403122060453, -0.98985988766009869]);
        assert_slice(
            &fit.log_w,
            &[
                -0.14301073703498665,
                -0.095894342874540239,
                0.23890507990952689,
            ],
        );
        assert_slice(
            &fit.information,
            &[
                5.4025646031315233,
                2.5076353884976541,
                2.5076353884976541,
                3.9141513741254204,
            ],
        );
        assert_relative_eq!(fit.log_likelihood, -49.318101708775217, max_relative = TOL);
        assert_relative_eq!(fit.log_det, 8.6579114426780528, max_relative = TOL);
        assert_eq!(fit.second_order, 0.0);
        assert_eq!(fit.iterations, 3);
        assert_eq!(fit.convergence, CONV_SUCCESS);
    }

    #[test]
    fn test_pml_log_likelihood_gradient_matches_nebula() {
        // Rscript -e 'library(nebula); nebula:::pml_ll_der_eigen(X, offset, Y,
        //   fid, cumsumy, posind, posindy, 2, 12, 3, c(0.5,-0.25),
        //   c(0.1,-0.2,0.05), c(0.25,2))'
        let data = fixture_a();
        let variance = PmlVariance {
            subject: 0.25,
            cell: 2.0,
        };
        let out = pml_log_likelihood_gradient(&data, &[0.5, -0.25], &[0.1, -0.2, 0.05], &variance)
            .expect("gradient");

        assert_relative_eq!(out.objective, 53.926203132942852, max_relative = TOL);
        assert_slice(
            &out.gradient,
            &[
                -5.2597994101747254,
                -0.19846989257393055,
                -0.40624529144311583,
                -2.8367693486273806,
                -2.0229915782218155,
            ],
        );
    }

    /// Builds the wider fixture: 64 cells, 4 subjects, 3 coefficients.
    ///
    /// Generated by an integer recurrence rather than embedded as literals, so
    /// the R side reproduces it exactly with the same three expressions. Every
    /// value is dyadic, so nothing is lost on the way in.
    ///
    /// ### Returns
    ///
    /// Design, offsets, counts, cell indices and subject totals.
    fn fixture_large() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<usize>, Vec<f64>) {
        let n_cells = 64usize;
        let mut design = Vec::with_capacity(n_cells * 3);
        let mut offset = Vec::with_capacity(n_cells);
        let mut counts = Vec::new();
        let mut cell_index = Vec::new();
        let mut subject_total = vec![0.0; 4];
        let starts = [0usize, 12, 32, 48, 64];
        for i in 0..n_cells {
            design.push(1.0);
            design.push(if i % 3 == 0 { 1.0 } else { 0.0 });
            design.push((i % 7) as f64 / 8.0);
            offset.push(((i % 9) as f64 - 4.0) / 8.0);
            let y = ((i * 7 + 3) % 11) as f64;
            if y > 0.0 {
                counts.push(y);
                cell_index.push(i);
            }
            let s = starts.iter().rposition(|&b| b <= i).unwrap_or(0).min(3);
            subject_total[s] += y;
        }
        (design, offset, counts, cell_index, subject_total)
    }

    #[test]
    fn test_opt_pml_wider_design_and_more_cells() {
        let (design, offset, counts, cell_index, subject_total) = fixture_large();
        assert_eq!(subject_total, vec![58.0, 100.0, 83.0, 82.0]);
        let starts = [0usize, 12, 32, 48, 64];
        let data = PmlData {
            design: &design,
            offset: &offset,
            counts: &counts,
            cell_index: &cell_index,
            subject_start: &starts,
            subject_total: &subject_total,
        };
        let variance = PmlVariance {
            subject: 0.35,
            cell: 1.5,
        };
        let params = PmlParams {
            reml: true,
            ord: 3,
            ..Default::default()
        };
        let fit = opt_pml(&data, &[0.5, -0.25, 0.75], &variance, Some(params)).expect("fit");

        assert_slice(
            &fit.beta,
            &[
                1.4272053191863423,
                0.20743588590645803,
                -0.017982095723719935,
            ],
        );
        assert_slice(
            &fit.log_w,
            &[
                0.21944093016930771,
                0.12415265795703513,
                0.20913336580790309,
                0.14393280971073011,
            ],
        );
        assert_slice(
            &fit.information,
            &[
                8.398261243665857,
                2.7062153540842324,
                3.0637064599168475,
                2.7062153540842324,
                16.226178650135836,
                1.2821611476475319,
                3.0637064599168475,
                1.2821611476475319,
                5.5864643706193053,
            ],
        );
        assert_relative_eq!(fit.log_likelihood, -276.5244116459354, max_relative = TOL);
        assert_relative_eq!(fit.log_det, 18.407453390744379, max_relative = TOL);
        assert_relative_eq!(fit.second_order, 0.0029993806316514779, max_relative = TOL);
        assert_eq!(fit.iterations, 4);
        assert_eq!(fit.convergence, CONV_SUCCESS);

        // Only the REML term of the log-determinant should move.
        let plain = opt_pml(&data, &[0.5, -0.25, 0.75], &variance, None).expect("fit");
        assert_relative_eq!(plain.log_det, 12.052188343414912, max_relative = TOL);
        assert_eq!(plain.second_order, 0.0);
        assert_slice(&plain.beta, &fit.beta);
    }

    #[test]
    fn test_check_convergence_bound_and_nan_branches() {
        let base = PmlResult {
            beta: vec![0.0, 0.0],
            log_w: vec![0.0],
            information: vec![4.0, 0.0, 0.0, 4.0],
            log_likelihood: -1.0,
            log_likelihood_prev: -1.0,
            log_det: 0.0,
            second_order: 0.0,
            convergence: CONV_SUCCESS,
            iterations: 3,
            backtracks: 0,
        };
        let params = PmlParams::default();

        assert_eq!(
            check_convergence(&base, &params, false, CONV_SUCCESS),
            CONV_SUCCESS
        );
        assert_eq!(
            check_convergence(&base, &params, true, CONV_SUCCESS),
            CONV_AT_BOUND
        );

        let nan = PmlResult {
            log_likelihood: f64::NAN,
            ..base.clone()
        };
        assert_eq!(
            check_convergence(&nan, &params, false, CONV_SUCCESS),
            CONV_NOT_FINITE
        );

        // An incoming failure from the outer optimiser is carried through.
        assert_eq!(check_convergence(&base, &params, false, -50), -50);

        // The singular test overrides even that.
        let singular = PmlResult {
            information: vec![1.0, 1.0, 1.0, 1.0],
            ..base.clone()
        };
        assert_eq!(
            check_convergence(&singular, &params, true, CONV_SUCCESS),
            CONV_SINGULAR
        );
    }

    #[test]
    fn test_min_eigenvalue_known_matrix() {
        // Eigenvalues 1 and 3.
        let a = [2.0, 1.0, 1.0, 2.0];
        assert_relative_eq!(min_eigenvalue(&a, 2), 1.0, max_relative = 1e-13);
        assert_relative_eq!(min_eigenvalue(&[7.0], 1), 7.0, max_relative = 1e-15);
    }

    #[test]
    fn test_ldlt_solve_indefinite_system() {
        // Eigenvalues 3 and -1, so a plain Cholesky would fail here.
        let mut a = [1.0, 2.0, 2.0, 1.0];
        let mut b = [3.0, 3.0];
        let mut perm = [0usize; 2];
        let mut tmp = [0.0; 2];
        ldlt_solve(&mut a, &mut b, 2, &mut perm, &mut tmp);
        assert_relative_eq!(b[0], 1.0, max_relative = 1e-13);
        assert_relative_eq!(b[1], 1.0, max_relative = 1e-13);
    }

    #[test]
    fn test_determinant_matches_closed_form() {
        let mut a = [4.0, 1.0, 2.0, 1.0, 5.0, 3.0, 2.0, 3.0, 6.0];
        let mut perm = [0usize; 3];
        // 4(30-9) - 1(6-6) + 2(3-10) = 84 - 0 - 14 = 70.
        assert_relative_eq!(
            determinant(&mut a, 3, &mut perm),
            70.0,
            max_relative = 1e-13
        );
    }

    #[test]
    fn test_rejects_malformed_input() {
        let data = fixture_a();
        let variance = PmlVariance {
            subject: 0.25,
            cell: 2.0,
        };
        assert!(opt_pml(&data, &[0.5], &variance, None).is_err());

        let bad_gamma = PmlVariance {
            subject: 0.25,
            cell: 0.0,
        };
        assert!(opt_pml(&data, &[0.5, -0.25], &bad_gamma, None).is_err());

        let bad_subject = PmlVariance {
            subject: 0.0,
            cell: 2.0,
        };
        assert!(opt_pml(&data, &[0.5, -0.25], &bad_subject, None).is_err());

        let short = PmlData {
            subject_total: &SUBJECT_TOTAL[..2],
            ..data
        };
        assert!(opt_pml(&short, &[0.5, -0.25], &variance, None).is_err());
    }
}
