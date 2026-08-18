//! The marginal Poisson-gamma mixed model NEBULA starts from.
//!
//! `ptmg` is the model with the subject random effect integrated out
//! analytically: each subject contributes one gamma-distributed frailty shared
//! by all of its cells, and each cell carries its own gamma overdispersion. The
//! result is a closed-form marginal likelihood in `(beta, sigma, phi)` that the
//! R package minimises with L-BFGS-B to get the starting values every later
//! stage of NEBULA refines.
//!
//! ### What is ported
//!
//! Straight from `nebula/src/optimization.cpp`: `ptmg_ll_eigen` (193),
//! `ptmg_der_eigen` (260), `ptmg_ll_der_eigen` (554) and `ptmg_ll_der_hes_eigen`
//! (676), plus the helpers `call_cumsumy`, `call_posindy`, `center_m`,
//! `cv_offset`, `get_cv` and `get_cell`.
//!
//! Those C++ kernels are only half the objective. The gamma-function terms that
//! do not involve `beta`, `sum lgamma(cumsumy + alpha)` over subjects and
//! `sum lgamma(y + phi)` over cells, live in the R wrappers in `R/ptmg.R` and
//! are folded back in here, so [`ptmg_value_and_gradient`] equals
//! `nebula:::ptmg_ll_der` and not the bare kernel. edgePython drops the
//! marginal-likelihood Hessian entirely, which is why its standard errors
//! disagree with the R package; [`ptmg_value_gradient_hessian`] is the piece it
//! lacks.
//!
//! ### Layout and conventions
//!
//! Cells are sorted by subject and [`GeneData::fid`] holds the boundaries, so
//! subject `s` owns cells `fid[s]..fid[s + 1]`. Every inner loop walks that
//! structure. The design is row-major `n_cells * n_coef`, one contiguous row per
//! cell, which is the orientation all the per-cell accumulations want. The
//! parameter vector is `[beta (n_coef), sigma, phi]`.
//!
//! One gene is the unit of work here and everything is sequential; genes are the
//! parallel axis and belong to the caller. The only exception is [`cumsum_y`],
//! which sees the whole matrix and fans out over genes itself.
//!
//! Scratch buffers are allocated per evaluation rather than threaded through the
//! API, because the required signatures take none. They are `O(n_cells * n_coef)`
//! and the optimiser calls this a few dozen times per gene.
//!
//! ### Deliberate fidelity to the reference
//!
//! `alpha = 1 / (exp(sigma) - 1)` is computed exactly that way rather than with
//! `expm1`, even though `expm1` is more accurate at the lower bound
//! `sigma = 1e-4`. Matching the reference matters more than the last few digits:
//! the fitted `sigma` is a variance component read off this surface, and shifting
//! the surface shifts the answer.
//!
//! ### References
//!
//! He et al., Communications Biology 4, 629, 2021

use rayon::prelude::*;

use crate::errors::EdgeErrors;
use crate::numeric::gamma::{digamma, ln_gamma, trigamma};
use crate::utils::sparse::{CompressedSparse, SparseFormat};

///////////////
// Gene data //
///////////////

/// Per-gene data laid out the way the kernels want it.
///
/// Counts are held sparsely: only the non-zero entries, with the cell each one
/// belongs to. That is how NEBULA stores single-cell counts and how
/// [`positive_indices`] hands them over.
///
/// Build with [`GeneData::new`], which is the only thing that checks the
/// invariants the kernels rely on.
#[derive(Clone, Copy, Debug)]
pub struct GeneData<'a> {
    /// Design matrix, row-major `n_cells * n_coef`, one row per cell.
    pub design: &'a [f64],
    /// Log offset per cell, length `n_cells`.
    pub log_offset: &'a [f64],
    /// Non-zero counts for this gene, in increasing cell order.
    pub counts: &'a [f64],
    /// Cell index of each entry of `counts`, strictly increasing.
    pub cells: &'a [usize],
    /// Count total per subject, length `n_subjects`. `cumsumy` in the reference.
    pub subject_totals: &'a [f64],
    /// Subject boundaries, length `n_subjects + 1`. Subject `s` owns cells
    /// `fid[s]..fid[s + 1]`, so `fid[0] == 0` and `fid[n_subjects] == n_cells`.
    pub fid: &'a [usize],
    /// Number of design columns.
    pub n_coef: usize,
}

impl<'a> GeneData<'a> {
    /// Assembles and validates the per-gene inputs.
    ///
    /// ### Params
    ///
    /// * `design` - Row-major `n_cells * n_coef` design matrix
    /// * `log_offset` - Log offset per cell, its length fixes `n_cells`
    /// * `counts` - Non-zero counts for this gene
    /// * `cells` - Cell index of each count, strictly increasing
    /// * `subject_totals` - Count total per subject
    /// * `fid` - Subject boundaries, length `n_subjects + 1`
    /// * `n_coef` - Number of design columns
    ///
    /// ### Returns
    ///
    /// The validated view, or [`EdgeErrors`] if the pieces disagree in length,
    /// `fid` is not a partition of `0..n_cells`, or a cell index is out of range
    /// or out of order.
    pub fn new(
        design: &'a [f64],
        log_offset: &'a [f64],
        counts: &'a [f64],
        cells: &'a [usize],
        subject_totals: &'a [f64],
        fid: &'a [usize],
        n_coef: usize,
    ) -> Result<Self, EdgeErrors> {
        if n_coef == 0 {
            return Err(EdgeErrors::MustBePositive("n_coef".to_string()));
        }
        let n_cells = log_offset.len();
        if n_cells == 0 {
            return Err(EdgeErrors::MustBePositive("n_cells".to_string()));
        }
        if design.len() != n_cells * n_coef {
            return Err(EdgeErrors::LengthMismatch {
                name: "design",
                expected: n_cells * n_coef,
                got: design.len(),
            });
        }
        if fid.len() < 2 {
            return Err(EdgeErrors::TooFewSubjects {
                required: 1,
                got: fid.len().saturating_sub(1),
            });
        }
        let n_subjects = fid.len() - 1;
        if subject_totals.len() != n_subjects {
            return Err(EdgeErrors::LengthMismatch {
                name: "subject_totals",
                expected: n_subjects,
                got: subject_totals.len(),
            });
        }
        if fid[0] != 0 || fid[n_subjects] != n_cells {
            return Err(EdgeErrors::InvalidArgument(format!(
                "fid must run from 0 to {n_cells}; got {} to {}.",
                fid[0], fid[n_subjects]
            )));
        }
        for s in 0..n_subjects {
            if fid[s + 1] <= fid[s] {
                return Err(EdgeErrors::InvalidArgument(format!(
                    "Subject {s} spans no cells: fid[{s}] = {}, fid[{}] = {}.",
                    fid[s],
                    s + 1,
                    fid[s + 1]
                )));
            }
        }
        if counts.len() != cells.len() {
            return Err(EdgeErrors::LengthMismatch {
                name: "cells",
                expected: counts.len(),
                got: cells.len(),
            });
        }
        let mut previous: Option<usize> = None;
        for &c in cells {
            if c >= n_cells {
                return Err(EdgeErrors::InvalidArgument(format!(
                    "Cell index {c} is out of range for {n_cells} cells."
                )));
            }
            if previous.is_some_and(|p| c <= p) {
                return Err(EdgeErrors::InvalidArgument(
                    "Cell indices must be strictly increasing.".to_string(),
                ));
            }
            previous = Some(c);
        }

        Ok(Self {
            design,
            log_offset,
            counts,
            cells,
            subject_totals,
            fid,
            n_coef,
        })
    }

    /// Number of cells for this gene.
    ///
    /// ### Returns
    ///
    /// `log_offset.len()`.
    #[inline]
    pub fn n_cells(&self) -> usize {
        self.log_offset.len()
    }

    /// Number of subjects for this gene.
    ///
    /// ### Returns
    ///
    /// `fid.len() - 1`.
    #[inline]
    pub fn n_subjects(&self) -> usize {
        self.fid.len() - 1
    }

    /// Length the parameter vector must have.
    ///
    /// ### Returns
    ///
    /// `n_coef + 2`, for `[beta, sigma, phi]`.
    #[inline]
    pub fn n_params(&self) -> usize {
        self.n_coef + 2
    }
}

//////////////////////////
// Reparametrised terms //
//////////////////////////

/// The scalars every kernel derives from `(sigma, phi)`, and their derivatives.
///
/// `sigma` is the log of one plus the subject-level variance, so the gamma
/// frailty has shape `alpha = 1 / (exp(sigma) - 1)` and rate
/// `lambda = alpha / sqrt(exp(sigma))`. `phi` is the cell-level gamma shape and
/// enters the likelihood directly, so it needs no reparametrisation.
#[derive(Clone, Copy, Debug)]
struct Terms {
    /// `sqrt(exp(sigma))`, which is also `alpha / lambda`.
    exps_s: f64,
    /// Subject frailty shape `1 / (exp(sigma) - 1)`.
    alpha: f64,
    /// Subject frailty rate `alpha / sqrt(exp(sigma))`.
    lambda: f64,
    /// Cell-level gamma shape, the `phi` of the parameter vector.
    gamma: f64,
    /// `ln(lambda)`.
    log_lambda: f64,
    /// `ln(gamma)`.
    log_gamma: f64,
    /// `d alpha / d sigma`.
    alpha_pr: f64,
    /// `d lambda / d sigma`.
    lambda_pr: f64,
    /// `d^2 alpha / d sigma^2`.
    alpha_dpr: f64,
    /// `d^2 lambda / d sigma^2`.
    lambda_dpr: f64,
}

impl Terms {
    /// Derives the reparametrised scalars at a point.
    ///
    /// Reproduces the reference arithmetic verbatim, including
    /// `exp(sigma) - 1` in place of `expm1(sigma)`.
    ///
    /// ### Params
    ///
    /// * `sigma` - Subject-level variance parameter
    /// * `phi` - Cell-level gamma shape
    ///
    /// ### Returns
    ///
    /// The scalars and their first two derivatives in `sigma`.
    fn new(sigma: f64, phi: f64) -> Self {
        let exps = sigma.exp();
        let exps_m1 = exps - 1.0;
        let exps_s = exps.sqrt();
        let alpha = 1.0 / exps_m1;
        let lambda = alpha / exps_s;
        let exps_sq = exps_m1 * exps_m1;

        Self {
            exps_s,
            alpha,
            lambda,
            gamma: phi,
            log_lambda: lambda.ln(),
            log_gamma: phi.ln(),
            alpha_pr: -exps / exps_sq,
            lambda_pr: (1.0 - 3.0 * exps) / (2.0 * exps_s * exps_sq),
            alpha_dpr: 2.0 * exps * exps / (exps_sq * exps_m1) - exps / exps_sq,
            lambda_dpr: -3.0 * exps / (2.0 * exps_s * exps_sq)
                + (3.0 * exps - 1.0) / (4.0 * exps_s * exps_sq)
                + (3.0 * exps - 1.0) * exps_s / (exps_sq * exps_m1),
        }
    }
}

///////////////////////////
// Gamma-function pieces //
///////////////////////////

/// How far to differentiate the gamma-function part of the likelihood.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GammaOrder {
    /// Value only.
    Value,
    /// Value and first derivative.
    Gradient,
    /// Value, first and second derivative.
    Hessian,
}

/// The gamma-function terms the C++ kernels leave to the R wrappers.
///
/// Two blocks, one per random effect. The subject block is
/// `sum lgamma(cumsumy + alpha) - lgamma(alpha)` over subjects with a non-zero
/// total, differentiated through `alpha`. The cell block is
/// `sum lgamma(y + phi) - lgamma(phi)` over cells with a non-zero count,
/// differentiated in `phi` directly.
#[derive(Clone, Copy, Debug, Default)]
struct GammaTerms {
    /// Subject block, evaluated.
    subject_value: f64,
    /// `d subject_value / d alpha`.
    subject_digamma: f64,
    /// `d^2 subject_value / d alpha^2`.
    subject_trigamma: f64,
    /// Cell block, evaluated.
    cell_value: f64,
    /// `d cell_value / d phi`.
    cell_digamma: f64,
    /// `d^2 cell_value / d phi^2`.
    cell_trigamma: f64,
}

/// Evaluates the gamma-function terms and, on request, their derivatives.
///
/// Counts of one and two are handled in closed form, as the reference does:
/// `lgamma(1 + g) - lgamma(g) = log(g)` and
/// `lgamma(2 + g) - lgamma(g) = log(g) + log(g + 1)`. Only counts of three and
/// above pay for a `lgamma` call, which is most of the saving on single-cell
/// data where nearly every non-zero count is a one or a two.
///
/// Subjects whose total is zero contribute exactly nothing and are skipped,
/// which is what the reference's `posind` index does.
///
/// ### Params
///
/// * `data` - The gene
/// * `alpha` - Subject frailty shape
/// * `gamma` - Cell-level gamma shape, the `phi` of the parameter vector
/// * `order` - How far to differentiate
///
/// ### Returns
///
/// The two blocks and whichever derivatives `order` asked for; the rest stay at
/// zero.
fn gamma_terms(data: &GeneData<'_>, alpha: f64, gamma: f64, order: GammaOrder) -> GammaTerms {
    let mut out = GammaTerms::default();

    // -- subject block --
    let mut n_positive = 0.0;
    let mut ln_sum = 0.0;
    let mut di_sum = 0.0;
    let mut tri_sum = 0.0;
    for &total in data.subject_totals {
        if total > 0.0 {
            n_positive += 1.0;
            ln_sum += ln_gamma(total + alpha);
            if order != GammaOrder::Value {
                di_sum += digamma(total + alpha);
            }
            if order == GammaOrder::Hessian {
                tri_sum += trigamma(total + alpha);
            }
        }
    }
    out.subject_value = ln_sum - n_positive * ln_gamma(alpha);
    if order != GammaOrder::Value {
        out.subject_digamma = di_sum - n_positive * digamma(alpha);
    }
    if order == GammaOrder::Hessian {
        out.subject_trigamma = tri_sum - n_positive * trigamma(alpha);
    }

    // -- cell block --
    let mut n_one = 0.0;
    let mut n_two = 0.0;
    let mut n_big = 0.0;
    let mut ln_big = 0.0;
    let mut di_big = 0.0;
    let mut tri_big = 0.0;
    for &y in data.counts {
        if y == 1.0 {
            n_one += 1.0;
        } else if y == 2.0 {
            n_two += 1.0;
        } else {
            n_big += 1.0;
            ln_big += ln_gamma(y + gamma);
            if order != GammaOrder::Value {
                di_big += digamma(y + gamma);
            }
            if order == GammaOrder::Hessian {
                tri_big += trigamma(y + gamma);
            }
        }
    }
    let n_small = n_one + n_two;
    out.cell_value =
        ln_big - n_big * ln_gamma(gamma) + n_small * gamma.ln() + n_two * (gamma + 1.0).ln();
    if order != GammaOrder::Value {
        out.cell_digamma =
            di_big - n_big * digamma(gamma) + n_small / gamma + n_two / (gamma + 1.0);
    }
    if order == GammaOrder::Hessian {
        out.cell_trigamma = tri_big
            - n_big * trigamma(gamma)
            - n_small / (gamma * gamma)
            - n_two / ((gamma + 1.0) * (gamma + 1.0));
    }

    out
}

/////////////
// Kernels //
/////////////

/// Marginal negative log-likelihood of one gene.
///
/// Port of `ptmg_ll_eigen` plus the gamma-function terms `R/ptmg.R` adds, so
/// this equals `nebula:::ptmg_ll`. The value-only path skips every buffer the
/// gradient needs.
///
/// ### Params
///
/// * `data` - The gene, as validated by [`GeneData::new`]
/// * `params` - `[beta (n_coef), sigma, phi]`, length [`GeneData::n_params`]
///
/// ### Returns
///
/// The negative log-likelihood. Lower is better, matching what the optimiser
/// minimises.
pub fn ptmg_neg_log_likelihood(data: &GeneData<'_>, params: &[f64]) -> f64 {
    debug_assert_eq!(params.len(), data.n_params());

    let nb = data.n_coef;
    let n_cells = data.n_cells();
    let n_subjects = data.n_subjects();
    let beta = &params[..nb];
    let terms = Terms::new(params[nb], params[nb + 1]);
    let gamma = terms.gamma;

    let mut y_cell = vec![0.0; n_cells];
    for (&c, &y) in data.cells.iter().zip(data.counts.iter()) {
        y_cell[c] = y;
    }

    let mut extb = vec![0.0; n_cells];
    let mut total = linear_predictor(data, beta, &y_cell, &mut extb);

    for s in 0..n_subjects {
        let span = data.fid[s]..data.fid[s + 1];
        let cumsumxtb: f64 = extb[span.clone()].iter().sum();
        let ystar = data.subject_totals[s] + terms.alpha;
        let mustar = cumsumxtb + terms.lambda;
        total -= ystar * mustar.ln();

        let ymustar = ystar / mustar;
        total += ymustar * cumsumxtb;
        for i in span {
            let w = ymustar * extb[i] + gamma;
            total -= (gamma + y_cell[i]) * w.ln();
        }
    }

    total += (n_subjects as f64) * terms.alpha * terms.log_lambda;
    total += (n_cells as f64) * gamma * terms.log_gamma;

    let extra = gamma_terms(data, terms.alpha, gamma, GammaOrder::Value);

    -total - extra.subject_value - extra.cell_value
}

/// Gradient of the marginal negative log-likelihood.
///
/// Port of `ptmg_der_eigen` plus the R-level corrections, so this equals
/// `nebula:::ptmg_der`. It runs the same code as
/// [`ptmg_value_and_gradient`] and throws the value away; prefer that when both
/// are wanted, which for an optimiser is always.
///
/// ### Params
///
/// * `data` - The gene, as validated by [`GeneData::new`]
/// * `params` - `[beta (n_coef), sigma, phi]`, length [`GeneData::n_params`]
///
/// ### Returns
///
/// The gradient, length `n_coef + 2`.
pub fn ptmg_gradient(data: &GeneData<'_>, params: &[f64]) -> Vec<f64> {
    evaluate(data, params, false).1
}

/// Value and gradient of the marginal negative log-likelihood together.
///
/// Port of `ptmg_ll_der_eigen` plus the R-level corrections, so this equals
/// `nebula:::ptmg_ll_der`. This is the objective the bounded optimiser drives.
///
/// ### Params
///
/// * `data` - The gene, as validated by [`GeneData::new`]
/// * `params` - `[beta (n_coef), sigma, phi]`, length [`GeneData::n_params`]
///
/// ### Returns
///
/// The negative log-likelihood and its gradient, length `n_coef + 2`.
pub fn ptmg_value_and_gradient(data: &GeneData<'_>, params: &[f64]) -> (f64, Vec<f64>) {
    let (value, gradient, _) = evaluate(data, params, false);
    (value, gradient)
}

/// Value, gradient and Hessian of the marginal negative log-likelihood.
///
/// Port of `ptmg_ll_der_hes_eigen` plus the R-level corrections. This is the
/// function edgePython does not have, and the reason its standard errors drift
/// from the R package: the observed information of the marginal likelihood is
/// what the covariance of `beta` is read off.
///
/// The Hessian is in the direct parametrisation `[beta, sigma, phi]`, matching
/// the gradient. The R package's `ptmg_ll_der_hes2` and `ptmg_ll_der_hes3`
/// additionally push it through `sigma -> exp(sigma)` for the `trust` optimiser;
/// that chain rule belongs to whichever driver wants the log scale, not here.
///
/// ### Params
///
/// * `data` - The gene, as validated by [`GeneData::new`]
/// * `params` - `[beta (n_coef), sigma, phi]`, length [`GeneData::n_params`]
///
/// ### Returns
///
/// The negative log-likelihood, its gradient of length `n_coef + 2`, and its
/// Hessian as a row-major `(n_coef + 2) * (n_coef + 2)` matrix.
pub fn ptmg_value_gradient_hessian(
    data: &GeneData<'_>,
    params: &[f64],
) -> (f64, Vec<f64>, Vec<f64>) {
    evaluate(data, params, true)
}

/// Fills the exponentiated linear predictor and accumulates `sum y * eta`.
///
/// ### Params
///
/// * `data` - The gene
/// * `beta` - Coefficients, length `n_coef`
/// * `y_cell` - Count per cell, zero where the gene is not expressed
/// * `extb` - Output buffer of length `n_cells`, overwritten with `exp(eta)`
///
/// ### Returns
///
/// `sum_i y_i * eta_i`, the only part of the likelihood linear in `eta`.
fn linear_predictor(data: &GeneData<'_>, beta: &[f64], y_cell: &[f64], extb: &mut [f64]) -> f64 {
    let nb = data.n_coef;
    let mut total = 0.0;

    for (i, (row, extb_i)) in data
        .design
        .chunks_exact(nb)
        .zip(extb.iter_mut())
        .enumerate()
    {
        let mut eta = data.log_offset[i];
        for (&x, &b) in row.iter().zip(beta.iter()) {
            eta += x * b;
        }
        // Guarded so a huge eta paired with a zero count cannot make `0 * inf`.
        if y_cell[i] != 0.0 {
            total += eta * y_cell[i];
        }
        *extb_i = eta.exp();
    }

    total
}

/// Value, gradient and optionally Hessian, in one pass over the gene.
///
/// The shared body of [`ptmg_value_and_gradient`] and
/// [`ptmg_value_gradient_hessian`], mirroring how `ptmg_ll_der_eigen` and
/// `ptmg_ll_der_hes_eigen` share theirs. Everything is accumulated in
/// log-likelihood sign, exactly as the reference does, and negated once at the
/// end; the gamma-function corrections are then subtracted from the negated
/// quantities, which is the order `R/ptmg.R` uses.
///
/// ### Params
///
/// * `data` - The gene, as validated by [`GeneData::new`]
/// * `params` - `[beta (n_coef), sigma, phi]`, length [`GeneData::n_params`]
/// * `with_hessian` - Whether to build the Hessian
///
/// ### Returns
///
/// The negative log-likelihood, its gradient of length `n_coef + 2`, and its
/// row-major Hessian, which is empty when `with_hessian` is false.
fn evaluate(data: &GeneData<'_>, params: &[f64], with_hessian: bool) -> (f64, Vec<f64>, Vec<f64>) {
    debug_assert_eq!(params.len(), data.n_params());

    let nb = data.n_coef;
    let n_cells = data.n_cells();
    let k = data.n_subjects();
    let n_params = nb + 2;
    let fid = data.fid;
    let beta = &params[..nb];
    let terms = Terms::new(params[nb], params[nb + 1]);
    let gamma = terms.gamma;

    // -- linear predictor --
    let mut y_cell = vec![0.0; n_cells];
    for (&c, &y) in data.cells.iter().zip(data.counts.iter()) {
        y_cell[c] = y;
    }
    let mut extb = vec![0.0; n_cells];
    let mut total = linear_predictor(data, beta, &y_cell, &mut extb);

    // -- per-subject collapse --
    let mut cumsumxtb = vec![0.0; k];
    let mut mustar = vec![0.0; k];
    let mut mustar_log = vec![0.0; k];
    let mut ymustar = vec![0.0; k];
    let mut ymumustar = vec![0.0; k];
    let mut imustar = vec![0.0; k];
    for s in 0..k {
        let csx: f64 = extb[fid[s]..fid[s + 1]].iter().sum();
        let ystar = data.subject_totals[s] + terms.alpha;
        let mu = csx + terms.lambda;
        let log_mu = mu.ln();
        cumsumxtb[s] = csx;
        mustar[s] = mu;
        mustar_log[s] = log_mu;
        ymustar[s] = ystar / mu;
        ymumustar[s] = ystar / mu / mu;
        imustar[s] = 1.0 / mu;
        total -= ystar * log_mu;
    }
    total += (k as f64) * terms.alpha * terms.log_lambda;
    total += (n_cells as f64) * gamma * terms.log_gamma;

    // -- per-cell weights against the collapsed subject mean --
    let mut tempa = vec![0.0; n_cells];
    let mut gstar = vec![0.0; n_cells];
    let mut slpey = 0.0;
    for s in 0..k {
        let ym = ymustar[s];
        for i in fid[s]..fid[s + 1] {
            let we = ym * extb[i];
            let w = we + gamma;
            let log_w = w.ln();
            let inv_w = 1.0 / w;
            total += we;
            total -= (gamma + y_cell[i]) * log_w;
            slpey += log_w;
            tempa[i] = inv_w;
            gstar[i] = (gamma + y_cell[i]) * inv_w;
        }
    }

    // -- gradient blocks --
    let mut xexb = vec![0.0; n_cells * nb];
    let mut xexb_f = vec![0.0; k * nb];
    let mut dbeta_41 = vec![0.0; k * nb];
    let mut d42c = vec![0.0; k];
    for s in 0..k {
        let mut acc = 0.0;
        for i in fid[s]..fid[s + 1] {
            let e = extb[i];
            let g = gstar[i];
            let row = &data.design[i * nb..(i + 1) * nb];
            let out = &mut xexb[i * nb..(i + 1) * nb];
            for j in 0..nb {
                let v = row[j] * e;
                out[j] = v;
                xexb_f[s * nb + j] += v;
                dbeta_41[s * nb + j] += g * v;
            }
            acc += g * e;
        }
        d42c[s] = acc - cumsumxtb[s];
    }

    let ymm_d: Vec<f64> = (0..k).map(|s| ymumustar[s] * d42c[s]).collect();

    let mut gradient = vec![0.0; n_params];
    for j in 0..nb {
        let mut acc = 0.0;
        for s in 0..k {
            acc += xexb_f[s * nb + j] * ymm_d[s] - dbeta_41[s * nb + j] * ymustar[s];
        }
        gradient[j] = acc;
    }
    for (&c, &y) in data.cells.iter().zip(data.counts.iter()) {
        let row = &data.design[c * nb..(c + 1) * nb];
        for j in 0..nb {
            gradient[j] += row[j] * y;
        }
    }

    let ldm = terms.log_lambda * (k as f64) - mustar_log.iter().sum::<f64>();
    let adlmy = terms.exps_s * (k as f64) - ymustar.iter().sum::<f64>();
    let hbl0: Vec<f64> = (0..k).map(|s| d42c[s] * imustar[s]).collect();
    let dbim: f64 = hbl0.iter().sum();
    let sum_ymm_d: f64 = ymm_d.iter().sum();

    gradient[nb] = -terms.alpha_pr * dbim
        + terms.lambda_pr * sum_ymm_d
        + terms.alpha_pr * ldm
        + terms.lambda_pr * adlmy;
    gradient[nb + 1] =
        terms.log_gamma * (n_cells as f64) + (n_cells as f64) - slpey - gstar.iter().sum::<f64>();

    let value = -total;
    for g in gradient.iter_mut() {
        *g = -*g;
    }

    if !with_hessian {
        let extra = gamma_terms(data, terms.alpha, gamma, GammaOrder::Gradient);
        let value = value - extra.subject_value - extra.cell_value;
        gradient[nb] -= terms.alpha_pr * extra.subject_digamma;
        gradient[nb + 1] -= extra.cell_digamma;
        return (value, gradient, Vec::new());
    }

    // -- Hessian, still in log-likelihood sign --
    let mut hessian = vec![0.0; n_params * n_params];
    let lam_ratio = terms.lambda_pr / terms.lambda;
    let mut hes_sigma = (terms.alpha_dpr * terms.log_lambda + 2.0 * terms.alpha_pr * lam_ratio
        - terms.alpha * lam_ratio * lam_ratio
        + terms.alpha / terms.lambda * terms.lambda_dpr)
        * (k as f64);

    let sum_hbl0_imu: f64 = (0..k).map(|s| hbl0[s] * imustar[s]).sum();
    let sum_ymmd_imu: f64 = (0..k).map(|s| ymm_d[s] * imustar[s]).sum();
    hes_sigma -= dbim * terms.alpha_dpr
        - 2.0 * terms.alpha_pr * terms.lambda_pr * sum_hbl0_imu
        - terms.lambda_dpr * sum_ymm_d
        + 2.0 * terms.lambda_pr * terms.lambda_pr * sum_ymmd_imu;
    hes_sigma -= 2.0 * terms.alpha_pr * terms.lambda_pr * imustar.iter().sum::<f64>()
        + terms.alpha_dpr * mustar_log.iter().sum::<f64>()
        + terms.lambda_dpr * ymustar.iter().sum::<f64>()
        - terms.lambda_pr * terms.lambda_pr * ymumustar.iter().sum::<f64>();

    let mut hes_sigma_beta = vec![0.0; nb];
    for j in 0..nb {
        let mut d41_imu = 0.0;
        let mut d41_ymm = 0.0;
        let mut xf_hbl_imu = 0.0;
        let mut xf_hbl_ymm = 0.0;
        for s in 0..k {
            d41_imu += dbeta_41[s * nb + j] * imustar[s];
            d41_ymm += dbeta_41[s * nb + j] * ymumustar[s];
            xf_hbl_imu += xexb_f[s * nb + j] * hbl0[s] * imustar[s];
            xf_hbl_ymm += xexb_f[s * nb + j] * hbl0[s] * ymumustar[s];
        }
        hes_sigma_beta[j] = -terms.alpha_pr * d41_imu + terms.lambda_pr * d41_ymm
            - (-terms.alpha_pr * xf_hbl_imu + 2.0 * terms.lambda_pr * xf_hbl_ymm);
    }

    // `alpha_pr / mustar - lambda_pr * ymustar / mustar`, the sensitivity of the
    // collapsed subject weight to sigma. Appears squared in the sigma block.
    let apm: Vec<f64> = (0..k)
        .map(|s| terms.alpha_pr * imustar[s] - terms.lambda_pr * ymumustar[s])
        .collect();

    let mut hes_gamma = (n_cells as f64) / gamma - 2.0 * tempa.iter().sum::<f64>();
    let mut gt = vec![0.0; n_cells];
    let mut gte = vec![0.0; n_cells];
    for i in 0..n_cells {
        gt[i] = gstar[i] * tempa[i];
        gte[i] = gt[i] * extb[i];
    }
    hes_gamma += gt.iter().sum::<f64>();

    let mut tmp1_f = vec![0.0; k * nb];
    let mut tmp2_f = vec![0.0; k * nb];
    let mut hb2_f = vec![0.0; k];
    let mut b2f = vec![0.0; k];
    for s in 0..k {
        for i in fid[s]..fid[s + 1] {
            let f1 = gt[i] - tempa[i];
            let row = &xexb[i * nb..(i + 1) * nb];
            for j in 0..nb {
                tmp1_f[s * nb + j] += row[j] * f1;
                tmp2_f[s * nb + j] += row[j] * gte[i];
            }
            hb2_f[s] += gte[i] - extb[i] * tempa[i];
            b2f[s] += gte[i] * extb[i];
        }
    }

    let mut hes_gamma_beta = vec![0.0; nb];
    for j in 0..nb {
        let mut a = 0.0;
        let mut b = 0.0;
        for s in 0..k {
            a += tmp1_f[s * nb + j] * ymustar[s];
            b += xexb_f[s * nb + j] * hb2_f[s] * ymumustar[s];
        }
        hes_gamma_beta[j] = a - b;
    }
    let hes_sigma_gamma: f64 = (0..k).map(|s| apm[s] * hb2_f[s]).sum();

    for j in 0..nb {
        let mut a = 0.0;
        let mut b = 0.0;
        for s in 0..k {
            a += tmp2_f[s * nb + j] * apm[s] * ymustar[s];
            b += xexb_f[s * nb + j] * apm[s] * b2f[s] * ymumustar[s];
        }
        hes_sigma_beta[j] += a - b;
    }
    hes_sigma += (0..k).map(|s| b2f[s] * apm[s] * apm[s]).sum::<f64>();

    // -- beta block --
    let ymu2: Vec<f64> = ymustar.iter().map(|v| v * v).collect();
    let ymuymuimu: Vec<f64> = (0..k).map(|s| ymu2[s] * imustar[s]).collect();
    let b2f_scaled: Vec<f64> = (0..k).map(|s| b2f[s] * ymuymuimu[s] * imustar[s]).collect();
    let ymmd_imu: Vec<f64> = (0..k).map(|s| ymm_d[s] * imustar[s]).collect();

    for p in 0..nb {
        for q in p..nb {
            let mut acc = 0.0;
            for s in 0..k {
                let mut s_gstar = 0.0;
                let mut s_gte = 0.0;
                let mut s_plain = 0.0;
                for i in fid[s]..fid[s + 1] {
                    let cell = xexb[i * nb + p] * data.design[i * nb + q];
                    s_gstar += cell * gstar[i];
                    s_gte += cell * gte[i];
                    s_plain += cell;
                }
                acc -= s_gstar * ymustar[s];
                acc += s_gte * ymu2[s];
                acc += ymm_d[s] * s_plain;
            }
            for s in 0..k {
                let xf_p = xexb_f[s * nb + p];
                let xf_q = xexb_f[s * nb + q];
                acc += (dbeta_41[s * nb + p] * xf_q + dbeta_41[s * nb + q] * xf_p) * ymumustar[s];
                acc -= ymuymuimu[s] * (tmp2_f[s * nb + p] * xf_q + tmp2_f[s * nb + q] * xf_p);
                let cross = xf_p * xf_q;
                acc -= cross * ymumustar[s];
                acc += b2f_scaled[s] * cross;
                acc -= 2.0 * cross * ymmd_imu[s];
            }
            hessian[p * n_params + q] = acc;
            hessian[q * n_params + p] = acc;
        }
    }

    hessian[nb * n_params + nb] = hes_sigma;
    hessian[(nb + 1) * n_params + nb + 1] = hes_gamma;
    hessian[nb * n_params + nb + 1] = hes_sigma_gamma;
    hessian[(nb + 1) * n_params + nb] = hes_sigma_gamma;
    for j in 0..nb {
        hessian[nb * n_params + j] = hes_sigma_beta[j];
        hessian[j * n_params + nb] = hes_sigma_beta[j];
        hessian[(nb + 1) * n_params + j] = hes_gamma_beta[j];
        hessian[j * n_params + nb + 1] = hes_gamma_beta[j];
    }
    for h in hessian.iter_mut() {
        *h = -*h;
    }

    let extra = gamma_terms(data, terms.alpha, gamma, GammaOrder::Hessian);
    let value = value - extra.subject_value - extra.cell_value;
    gradient[nb] -= terms.alpha_pr * extra.subject_digamma;
    gradient[nb + 1] -= extra.cell_digamma;
    hessian[nb * n_params + nb] -= terms.alpha_dpr * extra.subject_digamma
        + terms.alpha_pr * terms.alpha_pr * extra.subject_trigamma;
    hessian[(nb + 1) * n_params + nb + 1] -= extra.cell_trigamma;

    (value, gradient, hessian)
}

/////////////
// Helpers //
/////////////

/// Per-subject count totals for every gene.
///
/// Port of `call_cumsumy`. The reference walks the sparse matrix cell by cell
/// and dumps an accumulator at each subject boundary, because its counts are
/// cell-major. Ours are gene-major, so each gene's non-zeros are contiguous and
/// the subject is looked up per entry instead. Same result, one pass, and genes
/// fan out over rayon.
///
/// ### Params
///
/// * `counts` - Gene-major counts, [`SparseFormat::Csr`] over `(n_genes, n_cells)`
/// * `fid` - Subject boundaries, length `n_subjects + 1`, running to `n_cells`
///
/// ### Returns
///
/// Row-major `n_genes * n_subjects` totals, or [`EdgeErrors`] if the matrix is
/// not gene-major or `fid` does not partition the cells.
pub fn cumsum_y(counts: &CompressedSparse<f64>, fid: &[usize]) -> Result<Vec<f64>, EdgeErrors> {
    if counts.format != SparseFormat::Csr {
        return Err(EdgeErrors::MalformedSparse(
            "cumsum_y needs gene-major counts in CSR form.".to_string(),
        ));
    }
    let n_genes = counts.nrows();
    let n_cells = counts.ncols();
    if fid.len() < 2 {
        return Err(EdgeErrors::TooFewSubjects {
            required: 1,
            got: fid.len().saturating_sub(1),
        });
    }
    let n_subjects = fid.len() - 1;
    if fid[0] != 0 || fid[n_subjects] != n_cells {
        return Err(EdgeErrors::InvalidArgument(format!(
            "fid must run from 0 to {n_cells}; got {} to {}.",
            fid[0], fid[n_subjects]
        )));
    }

    // Cell to subject, so the inner loop is a lookup rather than a search.
    let mut subject_of = vec![0usize; n_cells];
    for s in 0..n_subjects {
        if fid[s + 1] < fid[s] || fid[s + 1] > n_cells {
            return Err(EdgeErrors::InvalidArgument(
                "fid must be non-decreasing and bounded by the cell count.".to_string(),
            ));
        }
        for slot in subject_of.iter_mut().take(fid[s + 1]).skip(fid[s]) {
            *slot = s;
        }
    }

    let mut out = vec![0.0; n_genes * n_subjects];
    out.par_chunks_mut(n_subjects)
        .enumerate()
        .for_each(|(gene, row)| {
            let (indices, values) = counts.outer(gene);
            for (&cell, &value) in indices.iter().zip(values.iter()) {
                if value > 0.0 {
                    row[subject_of[cell as usize]] += value;
                }
            }
        });

    Ok(out)
}

/// The non-zero counts of one gene, and the summaries NEBULA derives from them.
#[derive(Clone, Debug)]
pub struct GeneCounts {
    /// Cell index of each non-zero count, increasing.
    pub cells: Vec<usize>,
    /// The non-zero counts themselves.
    pub counts: Vec<f64>,
    /// Mean count across all cells, zeros included. `mct` in the reference, and
    /// what the intercept is initialised from.
    pub mean_count: f64,
    /// How many counts equal one.
    pub n_one: usize,
    /// How many counts equal two.
    pub n_two: usize,
}

/// Extracts one gene's non-zero counts.
///
/// Port of `call_posindy`. The counts of one and two are tallied separately
/// because the gamma-function part of the likelihood has a closed form for them,
/// which is where most of the entries in a single-cell matrix land.
///
/// ### Params
///
/// * `counts` - Gene-major counts, [`SparseFormat::Csr`] over `(n_genes, n_cells)`
/// * `gene` - Row index of the gene
///
/// ### Returns
///
/// The gene's non-zero counts with their cells and summaries, or [`EdgeErrors`]
/// if the matrix is not gene-major or the gene index is out of range.
pub fn positive_indices(
    counts: &CompressedSparse<f64>,
    gene: usize,
) -> Result<GeneCounts, EdgeErrors> {
    if counts.format != SparseFormat::Csr {
        return Err(EdgeErrors::MalformedSparse(
            "positive_indices needs gene-major counts in CSR form.".to_string(),
        ));
    }
    let n_genes = counts.nrows();
    if gene >= n_genes {
        return Err(EdgeErrors::InvalidArgument(format!(
            "Gene index {gene} is out of range for {n_genes} genes."
        )));
    }
    let n_cells = counts.ncols();
    if n_cells == 0 {
        return Err(EdgeErrors::MustBePositive("n_cells".to_string()));
    }

    let (indices, values) = counts.outer(gene);
    let mut cells = Vec::with_capacity(indices.len());
    let mut kept = Vec::with_capacity(indices.len());
    let mut sum = 0.0;
    let mut n_one = 0;
    let mut n_two = 0;

    for (&cell, &value) in indices.iter().zip(values.iter()) {
        if value > 0.0 {
            cells.push(cell as usize);
            kept.push(value);
            sum += value;
            if value == 1.0 {
                n_one += 1;
            } else if value == 2.0 {
                n_two += 1;
            }
        }
    }

    Ok(GeneCounts {
        cells,
        counts: kept,
        mean_count: sum / (n_cells as f64),
        n_one,
        n_two,
    })
}

/// Centres and scales a design matrix.
///
/// Port of `center_m`. Each column is centred, then divided by its population
/// standard deviation. A column with no spread is not scaled: if its first entry
/// is non-zero it becomes a column of ones, which is how the intercept survives
/// centring, and otherwise it is left at zero and flagged with a standard
/// deviation of `-1` so the caller can drop it.
///
/// ### Params
///
/// * `design` - Row-major `n_rows * n_cols` design matrix
/// * `n_rows` - Number of cells
/// * `n_cols` - Number of columns
///
/// ### Returns
///
/// The centred and scaled design, row-major and the same shape, together with
/// the per-column standard deviations, or [`EdgeErrors::ShapeMismatch`] if the
/// slice length disagrees with the shape.
pub fn centre_design(
    design: &[f64],
    n_rows: usize,
    n_cols: usize,
) -> Result<(Vec<f64>, Vec<f64>), EdgeErrors> {
    if design.len() != n_rows * n_cols {
        return Err(EdgeErrors::ShapeMismatch {
            expected: (n_rows, n_cols),
            got: (design.len(), 1),
        });
    }
    if n_rows == 0 || n_cols == 0 {
        return Err(EdgeErrors::MustBePositive("design shape".to_string()));
    }

    let mut centred = vec![0.0; n_rows * n_cols];
    let mut sds = vec![0.0; n_cols];

    for j in 0..n_cols {
        let mut mean = 0.0;
        for i in 0..n_rows {
            mean += design[i * n_cols + j];
        }
        mean /= n_rows as f64;

        let mut variance = 0.0;
        for i in 0..n_rows {
            let d = design[i * n_cols + j] - mean;
            centred[i * n_cols + j] = d;
            variance += d * d;
        }
        let sd = (variance / n_rows as f64).sqrt();

        if sd > 0.0 {
            for i in 0..n_rows {
                centred[i * n_cols + j] /= sd;
            }
            sds[j] = sd;
        } else if design[j] != 0.0 {
            for i in 0..n_rows {
                centred[i * n_cols + j] = 1.0;
            }
            sds[j] = sd;
        } else {
            sds[j] = -1.0;
        }
    }

    Ok((centred, sds))
}

/// Log offsets and the summaries the NEBULA driver keys off.
#[derive(Clone, Debug)]
pub struct OffsetSummary {
    /// Log of the supplied offsets, one per cell.
    pub log_offset: Vec<f64>,
    /// Mean of `log_offset`, or zero when no offset was supplied.
    pub mean_log_offset: f64,
    /// Mean of the offsets on their original scale.
    pub mean_offset: f64,
    /// Coefficient of variation of the offsets on their original scale.
    pub cv: f64,
}

/// Prepares the offsets and measures their spread.
///
/// Port of `cv_offset`. With no offset every cell gets one, which logs to zero
/// and has no spread. The coefficient of variation is what decides later whether
/// NEBULA-LN's approximation is safe for this dataset.
///
/// ### Params
///
/// * `offset` - Offsets on their original scale, or `None` for all ones
/// * `n_cells` - Number of cells
///
/// ### Returns
///
/// The log offsets and their summaries, or [`EdgeErrors`] if the offset length
/// disagrees with `n_cells`.
pub fn offset_summary(offset: Option<&[f64]>, n_cells: usize) -> Result<OffsetSummary, EdgeErrors> {
    if n_cells == 0 {
        return Err(EdgeErrors::MustBePositive("n_cells".to_string()));
    }
    if let Some(o) = offset
        && o.len() != n_cells
    {
        return Err(EdgeErrors::LengthMismatch {
            name: "offset",
            expected: n_cells,
            got: o.len(),
        });
    }

    let values: Vec<f64> = match offset {
        Some(o) => o.to_vec(),
        None => vec![1.0; n_cells],
    };
    let mean_offset = match offset {
        Some(_) => values.iter().sum::<f64>() / (n_cells as f64),
        None => 1.0,
    };

    let mut cv = 0.0;
    if mean_offset > 0.0 {
        let ss: f64 = values.iter().map(|v| (v - mean_offset).powi(2)).sum();
        cv = (ss / n_cells as f64).sqrt() / mean_offset;
    }

    let log_offset: Vec<f64> = values.iter().map(|v| v.ln()).collect();
    let mean_log_offset = match offset {
        Some(_) => log_offset.iter().sum::<f64>() / (n_cells as f64),
        None => 0.0,
    };

    Ok(OffsetSummary {
        log_offset,
        mean_log_offset,
        mean_offset,
        cv,
    })
}

/// Squared coefficient of variation of the fitted cell-level means.
///
/// Port of `get_cv`. Only the cell-level columns of the design contribute, since
/// subject-level columns are constant within a subject and cannot make the means
/// vary across cells within one. Note this is the squared CV, unlike
/// [`offset_summary`]'s `cv`, which is the reference's own asymmetry.
///
/// ### Params
///
/// * `log_offset` - Log offset per cell, length `n_rows`
/// * `design` - Row-major `n_rows * n_cols` design matrix
/// * `n_rows` - Number of cells
/// * `n_cols` - Number of design columns
/// * `beta` - Coefficients, length `n_cols`
/// * `cell_columns` - Indices of the cell-level columns to include
///
/// ### Returns
///
/// `var(mu) / mean(mu)^2` over cells, or [`EdgeErrors`] if the shapes disagree
/// or a column index is out of range.
pub fn design_cv(
    log_offset: &[f64],
    design: &[f64],
    n_rows: usize,
    n_cols: usize,
    beta: &[f64],
    cell_columns: &[usize],
) -> Result<f64, EdgeErrors> {
    if n_rows == 0 || n_cols == 0 {
        return Err(EdgeErrors::MustBePositive("design shape".to_string()));
    }
    if design.len() != n_rows * n_cols {
        return Err(EdgeErrors::ShapeMismatch {
            expected: (n_rows, n_cols),
            got: (design.len(), 1),
        });
    }
    if log_offset.len() != n_rows {
        return Err(EdgeErrors::LengthMismatch {
            name: "log_offset",
            expected: n_rows,
            got: log_offset.len(),
        });
    }
    if beta.len() != n_cols {
        return Err(EdgeErrors::LengthMismatch {
            name: "beta",
            expected: n_cols,
            got: beta.len(),
        });
    }
    for &c in cell_columns {
        if c >= n_cols {
            return Err(EdgeErrors::CoefOutOfRange {
                index: c,
                n_coef: n_cols,
            });
        }
    }

    let mut mu = vec![0.0; n_rows];
    for i in 0..n_rows {
        let mut eta = log_offset[i];
        for &c in cell_columns {
            eta += design[i * n_cols + c] * beta[c];
        }
        mu[i] = eta.exp();
    }

    let mean = mu.iter().sum::<f64>() / (n_rows as f64);
    if mean <= 0.0 {
        return Ok(0.0);
    }
    let ss: f64 = mu.iter().map(|v| (v - mean).powi(2)).sum();

    Ok(ss / (n_rows as f64) / (mean * mean))
}

/// Flags the design columns that vary within a subject.
///
/// Port of `get_cell`. A column constant across every subject's cells is a
/// subject-level covariate and is absorbed by the random effect; only the
/// varying ones are cell-level.
///
/// ### Params
///
/// * `design` - Row-major `n_rows * n_cols` design matrix
/// * `n_rows` - Number of cells
/// * `n_cols` - Number of design columns
/// * `fid` - Subject boundaries, length `n_subjects + 1`
///
/// ### Returns
///
/// One flag per column, true where the column varies within at least one
/// subject, or [`EdgeErrors`] if the shapes disagree.
pub fn cell_level_columns(
    design: &[f64],
    n_rows: usize,
    n_cols: usize,
    fid: &[usize],
) -> Result<Vec<bool>, EdgeErrors> {
    if n_rows == 0 || n_cols == 0 {
        return Err(EdgeErrors::MustBePositive("design shape".to_string()));
    }
    if design.len() != n_rows * n_cols {
        return Err(EdgeErrors::ShapeMismatch {
            expected: (n_rows, n_cols),
            got: (design.len(), 1),
        });
    }
    if fid.len() < 2 {
        return Err(EdgeErrors::TooFewSubjects {
            required: 1,
            got: fid.len().saturating_sub(1),
        });
    }
    if fid[0] != 0 || fid[fid.len() - 1] != n_rows {
        return Err(EdgeErrors::InvalidArgument(format!(
            "fid must run from 0 to {n_rows}; got {} to {}.",
            fid[0],
            fid[fid.len() - 1]
        )));
    }

    let mut flags = vec![false; n_cols];
    for (j, flag) in flags.iter_mut().enumerate() {
        'subjects: for s in 0..fid.len() - 1 {
            let first = design[fid[s] * n_cols + j];
            for i in fid[s]..fid[s + 1] {
                if design[i * n_cols + j] != first {
                    *flag = true;
                    break 'subjects;
                }
            }
        }
    }

    Ok(flags)
}

///////////
// Tests //
///////////

// The reference literals are R's `%.17e` output pasted verbatim. Trimming them
// to the shortest round-tripping form would hide where they came from.
#[cfg(test)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Fixture A: six cells over three subjects of sizes two, three and one, so
    /// the last subject is a single cell. Two design columns, and two cells with
    /// a zero count.
    ///
    /// ### Returns
    ///
    /// Design, log offsets, non-zero counts, their cells, subject totals, `fid`.
    #[allow(clippy::type_complexity)]
    fn fixture_a() -> (
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<usize>,
        Vec<f64>,
        Vec<usize>,
    ) {
        let x = [0.0, 1.0, 0.5, -0.5, 0.25, -1.0];
        let design: Vec<f64> = x.iter().flat_map(|&v| [1.0, v]).collect();
        let log_offset = vec![0.0, 0.25, -0.5, 0.125, 0.75, -0.25];
        // Full counts 3, 0, 1, 2, 0, 5.
        let counts = vec![3.0, 1.0, 2.0, 5.0];
        let cells = vec![0, 2, 3, 5];
        let subject_totals = vec![3.0, 3.0, 5.0];
        let fid = vec![0, 2, 5, 6];
        (design, log_offset, counts, cells, subject_totals, fid)
    }

    /// Fixture B: eight cells over three subjects of sizes three, one and four.
    /// Three design columns, and the middle subject has a total count of zero.
    ///
    /// ### Returns
    ///
    /// Design, log offsets, non-zero counts, their cells, subject totals, `fid`.
    #[allow(clippy::type_complexity)]
    fn fixture_b() -> (
        Vec<f64>,
        Vec<f64>,
        Vec<f64>,
        Vec<usize>,
        Vec<f64>,
        Vec<usize>,
    ) {
        let x1 = [0.0, 1.0, 0.5, -0.5, 0.25, -1.0, 0.75, -0.25];
        let x2 = [0.5, -0.25, 0.0, 0.75, -1.0, 0.25, 1.0, -0.5];
        let design: Vec<f64> = x1
            .iter()
            .zip(x2.iter())
            .flat_map(|(&a, &b)| [1.0, a, b])
            .collect();
        let log_offset = vec![0.5, -0.25, 0.125, 0.0, 0.375, -0.75, 0.25, 1.0];
        // Full counts 0, 2, 1, 0, 4, 0, 1, 7.
        let counts = vec![2.0, 1.0, 4.0, 1.0, 7.0];
        let cells = vec![1, 2, 4, 6, 7];
        let subject_totals = vec![3.0, 0.0, 12.0];
        let fid = vec![0, 3, 4, 8];
        (design, log_offset, counts, cells, subject_totals, fid)
    }

    /// Checks all four entry points against one reference point.
    ///
    /// ### Params
    ///
    /// * `fixture` - Which fixture builder to use
    /// * `params` - `[beta, sigma, phi]`
    /// * `value` - Reference negative log-likelihood
    /// * `gradient` - Reference gradient
    /// * `hessian` - Reference Hessian, row-major
    /// * `gradient_tol` - Relative tolerance on the gradient
    /// * `hessian_tol` - Relative tolerance on the Hessian
    #[allow(clippy::type_complexity)]
    fn check(
        fixture: fn() -> (
            Vec<f64>,
            Vec<f64>,
            Vec<f64>,
            Vec<usize>,
            Vec<f64>,
            Vec<usize>,
        ),
        params: &[f64],
        value: f64,
        gradient: &[f64],
        hessian: &[f64],
        gradient_tol: f64,
        hessian_tol: f64,
    ) {
        let (design, log_offset, counts, cells, subject_totals, fid) = fixture();
        let n_coef = params.len() - 2;
        let data = GeneData::new(
            &design,
            &log_offset,
            &counts,
            &cells,
            &subject_totals,
            &fid,
            n_coef,
        )
        .unwrap();

        assert_relative_eq!(
            ptmg_neg_log_likelihood(&data, params),
            value,
            max_relative = 1e-10
        );

        let (v, g) = ptmg_value_and_gradient(&data, params);
        assert_relative_eq!(v, value, max_relative = 1e-10);
        for (got, want) in g.iter().zip(gradient.iter()) {
            assert_relative_eq!(got, want, max_relative = gradient_tol, epsilon = 1e-12);
        }

        let only_gradient = ptmg_gradient(&data, params);
        assert_eq!(only_gradient, g);

        let (v, g, h) = ptmg_value_gradient_hessian(&data, params);
        assert_relative_eq!(v, value, max_relative = 1e-10);
        for (got, want) in g.iter().zip(gradient.iter()) {
            assert_relative_eq!(got, want, max_relative = gradient_tol, epsilon = 1e-12);
        }
        for (got, want) in h.iter().zip(hessian.iter()) {
            assert_relative_eq!(got, want, max_relative = hessian_tol, epsilon = 1e-12);
        }
    }

    // Reference values from nebula 1.5.8, via
    //   Rscript -e 'library(nebula); nebula:::ptmg_ll_der_hes_eigen(X, offset, Y,
    //     fid-1, cumsumy, posind-1, posindy, nb, nind, k, beta, sigma)'
    // with the gamma-function corrections of R/ptmg.R applied, i.e. the same
    // arithmetic as nebula:::ptmg_ll_der plus the Hessian lines of
    // nebula:::ptmg_ll_der_hes2 without its log-scale chain rule. Cross-checked
    // against nebula:::ptmg_ll and nebula:::ptmg_ll_der.
    //
    // One deliberate departure: the corrections here call base R's lgamma,
    // digamma and trigamma, where R/ptmg.R calls Rfast's. Rfast::Digamma is
    // about 5e-11 absolute off base R and Rfast::Trigamma about 1.4e-9, and
    // `alpha_pr` multiplies those by up to 1e8 at the lower bound on sigma. Our
    // gamma.rs tracks base R, so comparing against Rfast would be measuring
    // Rfast's error. Against the package's own Rfast path these references move
    // the gradient by at most 1e-8 relative.
    //
    // Tolerances are 1e-10 on value and gradient and 1e-8 on the Hessian, except
    // at `sigma = 1e-4`, its lower bound, where they drop to 1e-8 and 1e-6. That
    // is not slack in the port: there `alpha_pr` is of order 1e8 and `alpha_pr^2`
    // of order 1e16, and the sigma derivatives are differences of quantities that
    // large collapsing to order 1e2 and 1e3. Six digits go in the cancellation,
    // so the last few are set by summation order and by the last bit of
    // `digamma`, not by the maths. What is actually achieved is 8e-13 on the
    // value, 1.4e-14 on the gradient and 9e-13 on the Hessian away from that
    // bound, and 4e-9 and 2e-7 at it.

    #[test]
    fn test_matches_nebula_at_an_interior_point() {
        check(
            fixture_a,
            &[0.5, -0.25, 0.5, 2.0],
            5.68919227265053351e+00,
            &[
                1.23230096727342087e-01,
                2.14374578345666933e+00,
                1.96862959139235372e+00,
                -1.46930062054576371e-01,
            ],
            &[
                2.46157581206725151e+00,
                -7.12808069390344246e-01,
                2.39173560205083291e-01,
                -2.69050975196676243e-02,
                -7.12808069390344246e-01,
                2.05333290435219373e+00,
                -1.34438513739700594e+00,
                3.31302601132097918e-01,
                2.39173560205083291e-01,
                -1.34438513739700594e+00,
                3.91220510021810242e-02,
                -1.42705917336173937e-01,
                -2.69050975196676243e-02,
                3.31302601132097918e-01,
                -1.42705917336173937e-01,
                1.68849516491021756e-01,
            ],
            1e-10,
            1e-8,
        );
    }

    /// Both variance parameters at their lower bound, where `alpha` is of order
    /// `1e4` and the cell-level shape is of order `1e-4`.
    #[test]
    fn test_matches_nebula_at_the_lower_bounds() {
        check(
            fixture_a,
            &[1.25, 0.75, 1e-4, 1e-4],
            3.29896692291768900e+01,
            &[
                2.96406695818021149e-02,
                1.31693015791212176e-02,
                1.30305458759699832e+02,
                -3.99414489355896367e+04,
            ],
            &[
                6.86679893234001110e-02,
                2.79446317794878468e-02,
                2.95430767492699658e+02,
                1.82090109226938218e-01,
                2.79446317794878468e-02,
                2.14302885561563627e-02,
                1.27035501836412706e+02,
                4.34573972169806044e+00,
                2.95430767492699658e+02,
                1.27035501836412706e+02,
                -4.52776105356216431e+03,
                -4.46684522479116310e+01,
                1.82090109226938218e-01,
                4.34573972169806044e+00,
                -4.46684522479116310e+01,
                3.99940003972201765e+08,
            ],
            1e-8,
            1e-6,
        );
    }

    /// Near the upper bounds, `sigma = 8` and `phi = 512`.
    #[test]
    fn test_matches_nebula_near_the_upper_bounds() {
        check(
            fixture_a,
            &[-0.5, 0.125, 8.0, 512.0],
            2.65308778580696725e+01,
            &[
                9.12793250876120510e-04,
                2.53045547739533250e+00,
                2.99186987816233607e+00,
                -3.60045252223439860e-07,
            ],
            &[
                9.39311823971149956e-05,
                -6.57328696371346430e-05,
                -8.66127875657092474e-04,
                3.13400144628472963e-11,
                -6.57328696371346430e-05,
                1.13575813292771000e+00,
                9.98036672113045331e-06,
                1.32876795203876600e-05,
                -8.66127875657092474e-04,
                9.98036672113045331e-06,
                6.69606190840976723e-03,
                -1.05310011543468632e-09,
                3.13400144628472963e-11,
                1.32876795203876600e-05,
                -1.05310011543468632e-09,
                1.36270078923174387e-09,
            ],
            1e-10,
            1e-8,
        );
    }

    /// A zero coefficient vector, which is where the driver starts every gene.
    #[test]
    fn test_matches_nebula_at_zero_coefficients() {
        check(
            fixture_a,
            &[0.0, 0.0, 2.0, 0.25],
            1.12063514415980094e+01,
            &[
                -3.81456775084743072e-03,
                5.74173792151410289e-01,
                2.38393039589584355e+00,
                -7.26967190944776576e+00,
            ],
            &[
                4.23398202512259736e-01,
                -2.57145238578597779e-01,
                1.94645018901316025e-01,
                -1.15565351220650037e-02,
                -2.57145238578597779e-01,
                7.01861921458433624e-01,
                -3.59445802666915337e-01,
                1.18270172064454471e+00,
                1.94645018901316025e-01,
                -3.59445802666915337e-01,
                4.36597493935281022e-01,
                1.30165671415671189e-02,
                -1.15565351220650037e-02,
                1.18270172064454471e+00,
                1.30165671415671189e-02,
                4.50117573398768016e+01,
            ],
            1e-10,
            1e-8,
        );
    }

    /// Three design columns, and a subject whose counts are all zero.
    #[test]
    fn test_matches_nebula_with_three_columns() {
        check(
            fixture_b,
            &[0.25, -0.5, 0.75, 1.5, 3.0],
            7.46199076417621399e+00,
            &[
                5.01350696391867245e-01,
                -2.04257563318932167e+00,
                5.34153324797603979e+00,
                1.07011331098338225e+00,
                2.53578958023772838e-01,
            ],
            &[
                3.48740746684502056e-01,
                2.54876819316399161e-02,
                1.45533761085350766e-02,
                -4.41548536268330516e-01,
                3.91566084164468586e-03,
                2.54876819316399161e-02,
                2.09570270083503551e+00,
                2.64460193688231204e-01,
                1.02670366878276728e-01,
                -5.76636821621331688e-02,
                1.45533761085350766e-02,
                2.64460193688231204e-01,
                3.84459569945194923e+00,
                -2.78923375808989293e-01,
                6.39405403297957253e-01,
                -4.41548536268330516e-01,
                1.02670366878276728e-01,
                -2.78923375808989293e-01,
                3.66772893546357448e-01,
                -1.49218225719068356e-02,
                3.91566084164468586e-03,
                -5.76636821621331688e-02,
                6.39405403297957253e-01,
                -1.49218225719068356e-02,
                -7.22442940506229370e-02,
            ],
            1e-10,
            1e-8,
        );
    }

    /// Three columns with `sigma` at its lower bound and `phi` at its upper one.
    #[test]
    fn test_matches_nebula_with_three_columns_at_mixed_bounds() {
        check(
            fixture_b,
            &[-1.0, 0.5, 0.25, 1e-4, 1000.0],
            1.13584681060607693e+01,
            &[
                -1.07326878153205776e+01,
                -1.48756816066168707e+00,
                7.53628836395186852e+00,
                -4.62797267317655496e+01,
                1.95947859886434761e-05,
            ],
            &[
                4.26381285975427904e+00,
                1.01145653996075602e+00,
                5.35701406304531225e-01,
                2.61662160722283907e+01,
                -6.94406121181113076e-06,
                1.01145653996075602e+00,
                1.35186752241846908e+00,
                3.72773749626097717e-01,
                6.11253074542543828e+00,
                -1.69452414787726765e-07,
                5.35701406304531225e-01,
                3.72773749626097717e-01,
                1.95363290656862110e+00,
                9.21736461732613921e-01,
                4.48552067926583857e-06,
                2.61662160722283907e+01,
                6.11253074542543828e+00,
                9.21736461732613921e-01,
                4.35795623779296875e+02,
                -6.84104019316582210e-05,
                -6.94406121181113076e-06,
                -1.69452414787726765e-07,
                4.48552067926583857e-06,
                -6.84104019316582210e-05,
                -3.90898716648611179e-08,
            ],
            1e-8,
            1e-6,
        );
    }

    /// Three columns with `sigma` at its upper bound and `phi` at its lower one.
    #[test]
    fn test_matches_nebula_with_three_columns_at_the_other_bounds() {
        check(
            fixture_b,
            &[0.75, -0.25, 0.5, 10.0, 1e-4],
            5.76091855400661359e+01,
            &[
                1.35773392448967911e-04,
                -3.59794963305848370e-04,
                4.96443874810736929e-04,
                1.99820940875096320e+00,
                -4.99346186500725235e+04,
            ],
            &[
                4.32580556442951729e-07,
                4.91633557431160259e-08,
                2.83994566060920446e-08,
                -1.35563254538641632e-04,
                -9.85888077842389521e-08,
                4.91633557431160259e-08,
                3.29393259974663849e-04,
                -2.02106061975382121e-04,
                7.65773286491442306e-06,
                -3.52089986509287156e+00,
                2.83994566060920446e-08,
                -2.02106061975382121e-04,
                4.54276735397987554e-04,
                -4.52632990189595600e-05,
                4.51109987283559022e+00,
                -1.35563254538641632e-04,
                7.65773286491442306e-06,
                -4.52632990189595600e-05,
                1.57701505319085555e-03,
                -9.74640513908427847e-02,
                -9.85888077842389521e-08,
                -3.52089986509287156e+00,
                4.51109987283559022e+00,
                -9.74640513908427847e-02,
                4.99929026939662099e+08,
            ],
            1e-10,
            1e-8,
        );
    }

    /// The Hessian must be symmetric by construction, at every point.
    #[test]
    fn test_hessian_is_symmetric() {
        let (design, log_offset, counts, cells, subject_totals, fid) = fixture_b();
        let data = GeneData::new(
            &design,
            &log_offset,
            &counts,
            &cells,
            &subject_totals,
            &fid,
            3,
        )
        .unwrap();
        let params = [0.25, -0.5, 0.75, 1.5, 3.0];
        let (_, _, h) = ptmg_value_gradient_hessian(&data, &params);
        let n = 5;
        for i in 0..n {
            for j in 0..n {
                assert_relative_eq!(h[i * n + j], h[j * n + i], max_relative = 1e-14);
            }
        }
    }

    /// The analytic gradient must agree with a central difference of the value.
    #[test]
    fn test_gradient_agrees_with_finite_differences() {
        let (design, log_offset, counts, cells, subject_totals, fid) = fixture_a();
        let data = GeneData::new(
            &design,
            &log_offset,
            &counts,
            &cells,
            &subject_totals,
            &fid,
            2,
        )
        .unwrap();
        let params = [0.5, -0.25, 0.5, 2.0];
        let (_, gradient) = ptmg_value_and_gradient(&data, &params);

        for j in 0..params.len() {
            let step = 1e-6 * params[j].abs().max(1.0);
            let mut up = params;
            let mut down = params;
            up[j] += step;
            down[j] -= step;
            let fd = (ptmg_neg_log_likelihood(&data, &up) - ptmg_neg_log_likelihood(&data, &down))
                / (2.0 * step);
            assert_relative_eq!(gradient[j], fd, max_relative = 1e-6, epsilon = 1e-8);
        }
    }

    /// And the Hessian with a central difference of the gradient.
    #[test]
    fn test_hessian_agrees_with_finite_differences() {
        let (design, log_offset, counts, cells, subject_totals, fid) = fixture_a();
        let data = GeneData::new(
            &design,
            &log_offset,
            &counts,
            &cells,
            &subject_totals,
            &fid,
            2,
        )
        .unwrap();
        let params = [0.5, -0.25, 0.5, 2.0];
        let (_, _, hessian) = ptmg_value_gradient_hessian(&data, &params);
        let n = params.len();

        for j in 0..n {
            let step = 1e-6 * params[j].abs().max(1.0);
            let mut up = params;
            let mut down = params;
            up[j] += step;
            down[j] -= step;
            let g_up = ptmg_gradient(&data, &up);
            let g_down = ptmg_gradient(&data, &down);
            for i in 0..n {
                let fd = (g_up[i] - g_down[i]) / (2.0 * step);
                assert_relative_eq!(hessian[i * n + j], fd, max_relative = 1e-5, epsilon = 1e-7);
            }
        }
    }

    #[test]
    fn test_rejects_a_design_of_the_wrong_length() {
        let (_, log_offset, counts, cells, subject_totals, fid) = fixture_a();
        let design = vec![1.0; 5];
        let err = GeneData::new(
            &design,
            &log_offset,
            &counts,
            &cells,
            &subject_totals,
            &fid,
            2,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            EdgeErrors::LengthMismatch { name: "design", .. }
        ));
    }

    #[test]
    fn test_rejects_a_fid_that_does_not_cover_the_cells() {
        let (design, log_offset, counts, cells, subject_totals, _) = fixture_a();
        let fid = vec![0, 2, 5, 7];
        let err = GeneData::new(
            &design,
            &log_offset,
            &counts,
            &cells,
            &subject_totals,
            &fid,
            2,
        )
        .unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));
    }

    #[test]
    fn test_rejects_an_empty_subject() {
        let (design, log_offset, counts, cells, _, _) = fixture_a();
        let fid = vec![0, 2, 2, 6];
        let subject_totals = vec![3.0, 0.0, 8.0];
        let err = GeneData::new(
            &design,
            &log_offset,
            &counts,
            &cells,
            &subject_totals,
            &fid,
            2,
        )
        .unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));
    }

    #[test]
    fn test_rejects_unsorted_cell_indices() {
        let (design, log_offset, counts, _, subject_totals, fid) = fixture_a();
        let cells = vec![0, 3, 2, 5];
        let err = GeneData::new(
            &design,
            &log_offset,
            &counts,
            &cells,
            &subject_totals,
            &fid,
            2,
        )
        .unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));
    }

    // -- helpers --

    /// A three-gene, six-cell count matrix in gene-major CSR form.
    ///
    /// ### Returns
    ///
    /// The matrix, with dense rows `3 0 1 2 0 5`, `0 0 0 0 0 0` and
    /// `1 1 2 0 4 0`.
    fn count_fixture() -> CompressedSparse<f64> {
        let dense = [
            3.0, 0.0, 1.0, 2.0, 0.0, 5.0, //
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            1.0, 1.0, 2.0, 0.0, 4.0, 0.0,
        ];
        CompressedSparse::from_dense(&dense, 3, 6, SparseFormat::Csr, |v: &f64| *v == 0.0).unwrap()
    }

    /// Reference from
    ///   Rscript -e 'library(nebula); library(Matrix);
    ///     m <- Matrix(c(3,0,1, 0,0,1, 1,0,2, 2,0,0, 0,0,4, 5,0,0), 3, 6, sparse=TRUE);
    ///     nebula:::call_cumsumy(m, c(0,2,5,6), 3, 3)'
    #[test]
    fn test_cumsum_y_matches_nebula() {
        let counts = count_fixture();
        let fid = vec![0, 2, 5, 6];
        let got = cumsum_y(&counts, &fid).unwrap();
        assert_eq!(got, vec![3.0, 3.0, 5.0, 0.0, 0.0, 0.0, 2.0, 6.0, 0.0]);
    }

    #[test]
    fn test_cumsum_y_rejects_cell_major_counts() {
        let counts = count_fixture().transpose();
        let fid = vec![0, 2, 5, 6];
        assert!(matches!(
            cumsum_y(&counts, &fid).unwrap_err(),
            EdgeErrors::MalformedSparse(_)
        ));
    }

    /// Reference from
    ///   Rscript -e 'library(nebula); library(Matrix);
    ///     m <- Matrix(c(3,0,1, 0,0,1, 1,0,2, 2,0,0, 0,0,4, 5,0,0), 3, 6, sparse=TRUE);
    ///     nebula:::call_posindy(t(m), 2, 6)'
    /// which is gene three, zero-based, over six cells.
    #[test]
    fn test_positive_indices_matches_nebula() {
        let counts = count_fixture();
        let got = positive_indices(&counts, 2).unwrap();
        assert_eq!(got.cells, vec![0, 1, 2, 4]);
        assert_eq!(got.counts, vec![1.0, 1.0, 2.0, 4.0]);
        assert_relative_eq!(got.mean_count, 8.0 / 6.0, max_relative = 1e-15);
        assert_eq!(got.n_one, 2);
        assert_eq!(got.n_two, 1);
    }

    #[test]
    fn test_positive_indices_on_an_all_zero_gene() {
        let counts = count_fixture();
        let got = positive_indices(&counts, 1).unwrap();
        assert!(got.cells.is_empty());
        assert_eq!(got.mean_count, 0.0);
        assert_eq!(got.n_one, 0);
        assert_eq!(got.n_two, 0);
    }

    #[test]
    fn test_positive_indices_rejects_a_gene_out_of_range() {
        let counts = count_fixture();
        assert!(matches!(
            positive_indices(&counts, 3).unwrap_err(),
            EdgeErrors::InvalidArgument(_)
        ));
    }

    /// Reference from
    ///   Rscript -e 'library(nebula);
    ///     X <- cbind(1, c(0,1,0.5,-0.5,0.25,-1), 0);
    ///     nebula:::center_m(X)'
    /// The intercept has no spread but a non-zero first entry, so it comes back
    /// as ones; the all-zero column is flagged with a standard deviation of -1.
    #[test]
    fn test_centre_design_matches_nebula() {
        let x = [0.0, 1.0, 0.5, -0.5, 0.25, -1.0];
        let design: Vec<f64> = x.iter().flat_map(|&v| [1.0, v, 0.0]).collect();
        let (centred, sds) = centre_design(&design, 6, 3).unwrap();

        assert_relative_eq!(sds[0], 0.0, epsilon = 1e-15);
        assert_relative_eq!(sds[1], 6.52186493437438730e-01, max_relative = 1e-14);
        assert_relative_eq!(sds[2], -1.0, max_relative = 1e-15);

        let mean = x.iter().sum::<f64>() / 6.0;
        for (i, &v) in x.iter().enumerate() {
            assert_relative_eq!(centred[i * 3], 1.0, max_relative = 1e-15);
            assert_relative_eq!(
                centred[i * 3 + 1],
                (v - mean) / sds[1],
                max_relative = 1e-13
            );
            assert_relative_eq!(centred[i * 3 + 2], 0.0, epsilon = 1e-15);
        }
    }

    #[test]
    fn test_centre_design_rejects_a_shape_mismatch() {
        assert!(matches!(
            centre_design(&[1.0, 2.0, 3.0], 2, 2).unwrap_err(),
            EdgeErrors::ShapeMismatch { .. }
        ));
    }

    /// Reference from
    ///   Rscript -e 'library(nebula);
    ///     nebula:::cv_offset(c(1,2,4,8), 1L, 4L)'
    #[test]
    fn test_offset_summary_matches_nebula() {
        let got = offset_summary(Some(&[1.0, 2.0, 4.0, 8.0]), 4).unwrap();
        assert_relative_eq!(got.mean_offset, 3.75, max_relative = 1e-15);
        assert_relative_eq!(got.cv, 7.14920352984240504e-01, max_relative = 1e-13);
        assert_relative_eq!(
            got.mean_log_offset,
            1.03972077083991787e+00,
            max_relative = 1e-13
        );
        for (got, want) in got
            .log_offset
            .iter()
            .zip([0.0, 2.0_f64.ln(), 4.0_f64.ln(), 8.0_f64.ln()].iter())
        {
            assert_relative_eq!(got, want, max_relative = 1e-15);
        }
    }

    /// With no offset every cell gets one, so the logs are zero and there is no
    /// spread. This is `input == 0` in the reference.
    #[test]
    fn test_offset_summary_without_an_offset() {
        let got = offset_summary(None, 4).unwrap();
        assert_eq!(got.log_offset, vec![0.0; 4]);
        assert_eq!(got.mean_log_offset, 0.0);
        assert_eq!(got.mean_offset, 1.0);
        assert_eq!(got.cv, 0.0);
    }

    #[test]
    fn test_offset_summary_rejects_a_length_mismatch() {
        assert!(matches!(
            offset_summary(Some(&[1.0, 2.0]), 3).unwrap_err(),
            EdgeErrors::LengthMismatch { .. }
        ));
    }

    /// Reference from
    ///   Rscript -e 'library(nebula);
    ///     X <- cbind(1, c(0,1,0.5,-0.5,0.25,-1));
    ///     nebula:::get_cv(rep(0,6), X, c(0.25,0.5), 2L, 1L, 6L)'
    /// which includes only column two, one-based.
    #[test]
    fn test_design_cv_matches_nebula() {
        let x = [0.0, 1.0, 0.5, -0.5, 0.25, -1.0];
        let design: Vec<f64> = x.iter().flat_map(|&v| [1.0, v]).collect();
        let got = design_cv(&[0.0; 6], &design, 6, 2, &[0.25, 0.5], &[1]).unwrap();
        assert_relative_eq!(got, 9.93386285703017347e-02, max_relative = 1e-12);
    }

    /// With no cell-level columns every fitted mean is the offset, so a constant
    /// offset has no spread at all.
    #[test]
    fn test_design_cv_with_no_cell_columns() {
        let design = vec![1.0; 6];
        let got = design_cv(&[0.0; 6], &design, 6, 1, &[0.5], &[]).unwrap();
        assert_relative_eq!(got, 0.0, epsilon = 1e-15);
    }

    #[test]
    fn test_design_cv_rejects_a_column_out_of_range() {
        let design = vec![1.0; 6];
        assert!(matches!(
            design_cv(&[0.0; 6], &design, 6, 1, &[0.5], &[1]).unwrap_err(),
            EdgeErrors::CoefOutOfRange { .. }
        ));
    }

    /// Reference from
    ///   Rscript -e 'library(nebula);
    ///     X <- cbind(1, c(0,1,0.5,-0.5,0.25,-1), c(1,1,2,2,2,3));
    ///     nebula:::get_cell(X, c(0,2,5,6), 3L, 3L)'
    /// The third column is constant within each subject, so it is subject-level.
    #[test]
    fn test_cell_level_columns_matches_nebula() {
        let x1 = [0.0, 1.0, 0.5, -0.5, 0.25, -1.0];
        let x2 = [1.0, 1.0, 2.0, 2.0, 2.0, 3.0];
        let design: Vec<f64> = x1
            .iter()
            .zip(x2.iter())
            .flat_map(|(&a, &b)| [1.0, a, b])
            .collect();
        let got = cell_level_columns(&design, 6, 3, &[0, 2, 5, 6]).unwrap();
        assert_eq!(got, vec![false, true, false]);
    }

    #[test]
    fn test_cell_level_columns_rejects_a_bad_fid() {
        let design = vec![1.0; 6];
        assert!(matches!(
            cell_level_columns(&design, 6, 1, &[0, 2, 5]).unwrap_err(),
            EdgeErrors::InvalidArgument(_)
        ));
    }
}
