//! NEBULA's driver: one negative binomial gamma mixed model per gene.
//!
//! Port of `R/nebula.R` from the `nebula` package. The numerical kernels live in
//! [`crate::sc::ptmg`] and [`crate::sc::pml`]; this module is the orchestration
//! around them: build the offsets and the centred design, filter the genes, fan
//! out over the survivors, and reassemble the coefficients, covariances and
//! overdispersions on the user's own scale.
//!
//! ### The three stages of one gene
//!
//! 1. Bounded L-BFGS-B on the marginal likelihood
//!    ([`ptmg_value_and_gradient`]) over `[beta, sigma, phi]`. Only the two
//!    variance components survive this stage; nebula throws the fixed effects
//!    away and restarts them from `log(mean count) - mean log offset`.
//! 2. A bounded search over the two variance components alone, with the fixed
//!    effects profiled out by [`opt_pml`] inside every evaluation. NEBULA-HL
//!    always runs this; NEBULA-LN runs it, or a one-dimensional restriction of
//!    it, or neither, depending on how well the large-sample approximation
//!    behind stage one is expected to hold for the gene.
//! 3. A final [`opt_pml`] at the chosen variances, whose observed information
//!    inverts to the covariance of `beta`.
//!
//! ### Deviations from the R package
//!
//! nebula drives stages one and two with `nloptr`: `NLOPT_LD_LBFGS` with
//! `ftol_abs = 1e-6`, and `NLOPT_LN_BOBYQA` with the nloptr default
//! `xtol_rel = 1e-6`. Neither is in this crate and neither is worth adding: the
//! stage-two objective is discontinuous at the `1e-6` level anyway, because
//! [`opt_pml`] stops on an *absolute* improvement of `eps = 1e-6` in the
//! penalised log-likelihood, so its Newton count flips as the variances move and
//! the profile likelihood jumps by about that much. BOBYQA cannot resolve the
//! minimum past that floor either, and two of its own runs from different
//! starting points disagree by up to `1e-6` in the resulting standard errors.
//!
//! Stage two therefore uses a bounded Nelder-Mead followed by a local quadratic
//! least-squares polish. The polish is
//! what buys back the accuracy: fitting a quadratic over a stencil that is wide
//! compared with the noise averages the jitter out, where a simplex chasing
//! individual function values does not. Measured against nebula 1.5.8 across
//! two- and three-coefficient designs and both the LN and HL paths, the worst
//! relative disagreement is `1.0e-6` on the coefficients, `2.1e-6` on the
//! standard errors and `4.4e-6` on the overdispersions. That is the
//! reproducibility floor of the reference itself rather than an error in this
//! port: re-running nebula's own optimisers to a tolerance of `1e-14` moves its
//! answers by as much.
//!
//! Stage one is a different problem: it has an exact gradient and no jitter, so
//! it is one call to [`minimise`] and lands on the reference optimum.
//!
//! Only `model = "NBGMM"` is implemented. `PMM` needs the Poisson-gamma kernels,
//! which this crate does not have, and `NBLMM` needs the log-normal branch of
//! the outer objective, which has no golden to validate against.
//!
//! ### References
//!
//! He et al., Communications Biology 4, 629, 2021

use std::cell::Cell;

use rayon::prelude::*;

use crate::numeric::gamma::ln_gamma;
use crate::numeric::lbfgsb::{LbfgsbParams, minimise};
use crate::numeric::optimise::{NelderMeadParams, nelder_mead};
use crate::prelude::*;
use crate::sc::pml::{
    CONV_SINGULAR, CONV_SUCCESS, PmlData, PmlParams, PmlVariance, check_convergence, opt_pml,
};
use crate::sc::ptmg::{
    GeneData, cell_level_columns, centre_design, cumsum_y, design_cv, offset_summary,
    positive_indices, ptmg_value_and_gradient,
};
use crate::sc::test::packed_len;

////////////
// Consts //
////////////

/// Box constraint on every fixed effect in stage one, nebula's `rep(100, nb)`.
///
/// The design is centred and scaled first, so a coefficient of a hundred is
/// already far outside anything a count model can produce.
const BETA_BOUND: f64 = 100.0;

/// Cells per subject below which NEBULA-LN is abandoned for NEBULA-HL.
///
/// The large-sample approximation stage one relies on is an approximation in
/// the number of cells per subject, so nebula silently overrides
/// `method = "LN"` below thirty.
const MIN_CELLS_PER_SUBJECT_LN: f64 = 30.0;

/// Expected count per subject below which the Laplace expansion is pushed to
/// third order.
///
/// nebula's `(mct * mfs) < 3`. Below it the leading term of the expansion is
/// visibly biased, so [`PmlParams::ord`] goes to three.
const HIGH_ORDER_COUNT_CUTOFF: f64 = 3.0;

/// Laplace order used when the expected count per subject is low.
const HIGH_ORDER: u32 = 3;

/// Relative slack allowed when deciding a fitted value sits on its lower bound.
///
/// The optimiser does not always return the constraint exactly; it can stop a
/// few parts in `1e6` above it, which is inside its own stopping tolerance and
/// means the same thing. Strict equality misses those, and on the small
/// single-cell fixture it missed two genes of 118 that R reports as pinned.
///
/// The margin cannot reach a real estimate: the smallest genuinely fitted
/// subject variance seen on any fixture is four times the bound, and this admits
/// values within one part in `1e4` of it.
const BOUND_SLACK: f64 = 1.0 + 1e-4;

/// Value of `kappa_obs` below which NEBULA-LN always refits the subject-level
/// overdispersion, whatever `kappa` was asked for.
const KAPPA_FLOOR: f64 = 20.0;

/// Numerator of nebula's second NEBULA-LN refit trigger, `sigma < 8 / kappa`.
const KAPPA_SIGMA_NUMERATOR: f64 = 8.0;

/// Relative widths of the quadratic polish stencils, applied in order.
///
/// The first two are wide enough to walk a poorly stopped simplex back to the
/// basin; the last sets the accuracy. A tenth of a per cent of the incumbent is
/// three orders above the `1e-6` jitter in the objective and still small enough
/// that the cubic term does not bias the fitted minimum.
const POLISH_WIDTHS: [f64; 3] = [1e-2, 3e-3, 1e-3];

/// Absolute floors on the polish stencil width, for `sigma` and for `phi`.
///
/// A relative width collapses when a component sits on its lower bound of
/// `1e-4`, which is exactly where the fit needs the stencil to have some reach.
const POLISH_FLOOR: [f64; 2] = [1e-5, 1e-4];

/// Largest polish step accepted, in units of the stencil width.
///
/// A quadratic model fitted over `[-1, 1]` says nothing useful two widths out,
/// so a longer step means the model is wrong and the incumbent is kept.
const POLISH_MAX_STEP: f64 = 2.0;

/// Relative objective tolerance for the stage-one quasi-Newton.
///
/// nebula stops nlopt's L-BFGS on an *absolute* change of `eps = 1e-6` in the
/// marginal log-likelihood, which on the few-thousand values seen here is about
/// `1e-10` relative. This crate's [`minimise`] tests a relative change, so
/// passing `eps` straight through would stop it four orders too early, and in
/// NEBULA-LN the cell-level overdispersion is carried out of this stage
/// untouched. A fixed tight value converges past nebula in every case instead.
const STAGE_ONE_FTOL: f64 = 1e-13;

/// Projected gradient tolerance for the stage-one quasi-Newton.
const STAGE_ONE_PGTOL: f64 = 1e-7;

/// Iteration budget for the stage-one quasi-Newton.
const STAGE_ONE_MAX_ITER: usize = 300;

/// Line search budget inside one stage-one iteration.
const STAGE_ONE_MAX_LINE_SEARCH: usize = 40;

/// Simplex tolerances for the stage-two search.
///
/// Both are absolute, and both sit below the jitter in the objective on purpose:
/// the simplex is meant to localise the basin, and the polish is meant to find
/// the minimum inside it.
const VARIANCE_XATOL: f64 = 1e-7;

/// Objective tolerance for the stage-two simplex. See [`VARIANCE_XATOL`].
const VARIANCE_FATOL: f64 = 1e-7;

/// Simplex iteration budget for the stage-two search.
const VARIANCE_MAX_ITER: usize = 500;

/// Convergence code for a stage-two search that never found a finite objective.
///
/// nebula reports `-50` when its outer optimiser returns a negative nlopt code.
/// Those codes are not reproducible without nlopt, so this port raises `-50`
/// for the one failure that is unambiguous: no evaluation of the profile
/// likelihood succeeded.
pub const CONV_OUTER_FAILED: i32 = -50;

////////////////
// Parameters //
////////////////

/// Which NEBULA variant to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NebulaMethod {
    /// NEBULA-LN. Estimates the overdispersions from the marginal likelihood
    /// and only refits them when the large-sample approximation looks unsafe.
    Ln,
    /// NEBULA-HL. Always refits both overdispersions against the profile
    /// likelihood. Slower and what nebula falls back to below
    /// [`MIN_CELLS_PER_SUBJECT_LN`] cells per subject.
    Hl,
}

/// Tuning knobs for [`nebula`].
#[derive(Clone, Copy, Debug)]
pub struct NebulaParams {
    /// Lower bounds on `(sigma, phi)`, nebula's `min`.
    pub min: (f64, f64),
    /// Upper bounds on `(sigma, phi)`, nebula's `max`.
    pub max: (f64, f64),
    /// Which variant to run. Overridden to [`NebulaMethod::Hl`] below
    /// [`MIN_CELLS_PER_SUBJECT_LN`] cells per subject, as in the R package.
    pub method: NebulaMethod,
    /// Refit both overdispersions when the product of the cells per subject and
    /// the estimated `phi` falls below this.
    pub cutoff_cell: f64,
    /// Threshold on nebula's `kappa_obs` above which the subject-level
    /// overdispersion from stage one is trusted as is.
    pub kappa: f64,
    /// Drop a gene whose mean count per cell is at most this.
    pub cpc: f64,
    /// Drop a gene expressed in fewer than this many cells.
    pub mincp: usize,
    /// Estimate the overdispersions by restricted maximum likelihood.
    ///
    /// R's `nebula` package only honours this for `NBLMM`, which this port does
    /// not implement. Here it is threaded through unconditionally and does
    /// change the fit on NBGMM, the only model implemented: [`opt_pml`] adds a
    /// log-determinant term to the outer overdispersion objective whenever it is
    /// set. That behaviour has not been validated against an R reference, since
    /// R never exercises `reml` on this model. Defaults off.
    pub reml: bool,
    /// Absolute stopping tolerance handed to [`opt_pml`], nebula's `eps`.
    pub eps: f64,
}

impl Default for NebulaParams {
    /// The R package's own defaults.
    fn default() -> Self {
        Self {
            min: (1e-4, 1e-4),
            max: (10.0, 1000.0),
            method: NebulaMethod::Ln,
            cutoff_cell: 20.0,
            kappa: 800.0,
            cpc: 0.005,
            mincp: 5,
            reml: false,
            eps: 1e-6,
        }
    }
}

impl NebulaParams {
    /// Checks the knobs the optimisers cannot recover from.
    ///
    /// ### Returns
    ///
    /// `Ok(())`, or the first violation found.
    fn validate(&self) -> Result<(), EdgeErrors> {
        for (index, (lo, hi)) in [(self.min.0, self.max.0), (self.min.1, self.max.1)]
            .into_iter()
            .enumerate()
        {
            if !(lo.is_finite() && lo > 0.0) {
                return Err(EdgeErrors::MustBePositive(format!("min.{index}")));
            }
            if !(hi.is_finite() && hi > lo) {
                return Err(EdgeErrors::InvalidBounds {
                    index,
                    lower: lo,
                    upper: hi,
                });
            }
        }
        if !(self.eps.is_finite() && self.eps > 0.0) {
            return Err(EdgeErrors::MustBePositive("eps".to_string()));
        }
        if self.mincp == 0 {
            return Err(EdgeErrors::MustBePositive("mincp".to_string()));
        }
        Ok(())
    }
}

///////////////
// NebulaFit //
///////////////

/// The fitted model for every gene that survived the expression filter.
///
/// Everything is on the user's own design scale: the centring and scaling
/// nebula applies internally is undone before these are returned.
#[derive(Clone, Debug)]
pub struct NebulaFit {
    /// Fixed effects, row-major `n_genes_kept * n_coef`.
    pub coefficients: Vec<f64>,
    /// Covariance of the fixed effects, packed upper triangular and row-major
    /// over genes, `n_genes_kept * packed_len(n_coef)`.
    ///
    /// The packing is [`crate::sc::test`]'s: entry `(i, j)` with `i <= j` lives
    /// at `j * (j + 1) / 2 + i`, so three coefficients store
    /// `V11, V12, V22, V13, V23, V33`.
    pub covariance: Vec<f64>,
    /// Standard errors, row-major `n_genes_kept * n_coef`.
    pub se: Vec<f64>,
    /// Subject-level overdispersion, nebula's `sigma^2`, one per kept gene.
    pub subject_overdispersion: Vec<f64>,
    /// Cell-level overdispersion, nebula's `phi^-1`, one per kept gene.
    pub cell_overdispersion: Vec<f64>,
    /// nebula's convergence code, one per kept gene. Anything at or below `-20`
    /// is a likely failure.
    pub convergence: Vec<i32>,
    /// Whether [`NebulaFit::subject_overdispersion`] finished on its lower bound.
    ///
    /// The bound is [`NebulaParams::min`]`.0`, `1e-4` by default. A gene sitting
    /// on it has no fitted subject-level variance at all: the mixed model has
    /// collapsed to a plain negative binomial GLM, and the reported `sigma^2` is
    /// the constraint rather than an estimate.
    ///
    /// This is not a convergence failure and [`NebulaFit::convergence`] will
    /// usually say the gene converged, because it did. The marginal likelihood
    /// really is still decreasing towards zero there, so the optimiser is
    /// sitting on a legitimate Karush-Kuhn-Tucker point of the box constraint.
    /// Lowering `min.0` simply moves the answer down with it.
    ///
    /// The R package has no equivalent flag. Its `check_conv` tests whether
    /// `sigma^2` hit its *upper* bound and never the lower one, so these genes
    /// come back from `nebula()` reporting success with no indication that the
    /// random effect was never estimated.
    ///
    /// Not raised for the related case where *stage one* hits the bound but the
    /// NEBULA-LN refit then moves `sigma^2` off it. That happens on roughly a
    /// third of genes and the final `sigma^2` is fine; what it leaves behind is
    /// a `phi` fitted jointly with the discarded boundary value. Flagging it
    /// would bury the case above in noise for no action the caller could take.
    pub sigma_at_bound: Vec<bool>,
    /// Zero-based indices of the input genes that survived the filter.
    pub gene_index: Vec<usize>,
    /// Number of coefficients, the stride of `coefficients` and `se`.
    pub n_coef: usize,
}

///////////////
// Front end //
///////////////

/// Fits NEBULA's negative binomial gamma mixed model to every gene.
///
/// Cells must already be grouped by subject: `subject_id` is read as a run
/// encoding, so a subject whose cells are split into two blocks is rejected
/// rather than silently merged. Genes are the parallel axis, one rayon task per
/// gene, with no shared mutable state.
///
/// ### Params
///
/// * `counts` - Raw counts, gene-major and row-major, `n_genes * n_cells`
/// * `n_genes` - Number of genes
/// * `n_cells` - Number of cells
/// * `subject_id` - Subject of each cell, with each subject's cells contiguous
/// * `design` - Predictors, row-major `n_cells * n_coef`, including an intercept
/// * `n_coef` - Number of design columns
/// * `offset` - Strictly positive scaling factor per cell, or `None` for ones
/// * `params` - Tuning knobs, or [`NebulaParams::default`]
///
/// ### Returns
///
/// The per-gene fits for the genes that passed the expression filter, or
/// [`EdgeErrors`] if the inputs disagree in shape, the design has no intercept
/// or more than one constant column, the cells are not grouped by subject, or no
/// gene survived the filter.
///
/// ### References
///
/// He et al., Communications Biology 4, 629, 2021
// The counts, their shape, the subject labels, the design, its width, the
// offsets and the knobs. Bundling any of it into a struct would only move the
// same arguments to a constructor.
#[allow(clippy::too_many_arguments)]
pub fn nebula<T: EdgeFloat>(
    counts: &[T],
    n_genes: usize,
    n_cells: usize,
    subject_id: &[usize],
    design: &[T],
    n_coef: usize,
    offset: Option<&[T]>,
    params: Option<NebulaParams>,
) -> Result<NebulaFit, EdgeErrors> {
    let params = params.unwrap_or_default();
    params.validate()?;

    if n_genes == 0 || n_cells == 0 {
        return Err(EdgeErrors::EmptyCounts {
            n_genes,
            n_samples: n_cells,
        });
    }
    if n_cells < 2 {
        return Err(EdgeErrors::InvalidArgument(
            "NEBULA needs more than one cell in the count matrix.".to_string(),
        ));
    }
    if n_coef == 0 {
        return Err(EdgeErrors::MustBePositive("n_coef".to_string()));
    }
    if counts.len() != n_genes * n_cells {
        return Err(EdgeErrors::LengthMismatch {
            name: "counts",
            expected: n_genes * n_cells,
            got: counts.len(),
        });
    }
    if design.len() != n_cells * n_coef {
        return Err(EdgeErrors::ShapeMismatch {
            expected: (n_cells, n_coef),
            got: (design.len(), 1),
        });
    }
    if subject_id.len() != n_cells {
        return Err(EdgeErrors::LengthMismatch {
            name: "subject_id",
            expected: n_cells,
            got: subject_id.len(),
        });
    }

    let fid = subject_boundaries(subject_id)?;
    let n_subjects = fid.len() - 1;
    if n_subjects < 2 {
        return Err(EdgeErrors::TooFewSubjects {
            required: 2,
            got: n_subjects,
        });
    }

    // Offsets. nebula logs them and keeps both the arithmetic mean, which seeds
    // the intercept, and the coefficient of variation, which the NEBULA-LN
    // triggers key off.
    let offset_f64: Option<Vec<f64>> = match offset {
        Some(o) => Some(to_f64("offset", o)?),
        None => None,
    };
    if let Some(o) = offset_f64.as_deref()
        && o.iter().any(|v| !(v.is_finite() && *v > 0.0))
    {
        return Err(EdgeErrors::MustBePositive("offset".to_string()));
    }
    let offsets = offset_summary(offset_f64.as_deref(), n_cells)?;
    let log_mean_offset = offsets.mean_offset.ln();
    let cv2 = offsets.cv * offsets.cv;

    // Design. nebula centres and scales every column, which turns the intercept
    // into a column of ones with a standard deviation of zero.
    let design_f64 = to_f64("design", design)?;
    let (centred, sds) = centre_design(&design_f64, n_cells, n_coef)?;
    if sds.iter().any(|s| *s < 0.0) {
        return Err(EdgeErrors::InvalidArgument(
            "Some predictors are a zero vector.".to_string(),
        ));
    }
    let constant: Vec<usize> = (0..n_coef).filter(|&j| sds[j] == 0.0).collect();
    let intercept = match constant.as_slice() {
        [] => return Err(EdgeErrors::MissingIntercept),
        [j] => *j,
        _ => {
            return Err(EdgeErrors::InvalidArgument(
                "More than one predictor has zero variation.".to_string(),
            ));
        }
    };

    // Columns that vary inside a subject. Only these can make the fitted cell
    // means differ within a subject, which is what nebula's `get_cv` measures.
    let cell_columns: Vec<usize> = cell_level_columns(&centred, n_cells, n_coef, &fid)?
        .iter()
        .enumerate()
        .filter(|(_, flag)| **flag)
        .map(|(j, _)| j)
        .collect();

    let sparse = build_csr(counts, n_genes, n_cells)?;
    let totals = cumsum_y(&sparse, &fid)?;

    let cells_per_subject = n_cells as f64 / n_subjects as f64;
    let method = if cells_per_subject < MIN_CELLS_PER_SUBJECT_LN {
        NebulaMethod::Hl
    } else {
        params.method
    };

    // Expression filter: mean count per cell, and number of expressed cells.
    let kept: Vec<usize> = (0..n_genes)
        .filter(|&g| {
            let total: f64 = totals[g * n_subjects..(g + 1) * n_subjects].iter().sum();
            let (indices, _) = sparse.outer(g);
            total / (n_cells as f64) > params.cpc && indices.len() >= params.mincp
        })
        .collect();
    if kept.is_empty() {
        return Err(EdgeErrors::NoGenesAfterFiltering { n_genes });
    }

    let shared = Shared {
        design: &centred,
        log_offset: &offsets.log_offset,
        fid: &fid,
        n_cells,
        n_coef,
        n_subjects,
        intercept,
        log_mean_offset,
        cells_per_subject,
        cv2,
        cell_columns: &cell_columns,
        method,
        params,
    };

    let outcomes: Vec<GeneOutcome> = kept
        .par_iter()
        .map(|&g| {
            let counts = positive_indices(&sparse, g)?;
            let subject_totals = &totals[g * n_subjects..(g + 1) * n_subjects];
            fit_gene(&shared, &counts, subject_totals)
        })
        .collect::<Result<Vec<_>, EdgeErrors>>()?;

    Ok(assemble(
        outcomes,
        &kept,
        &sds,
        intercept,
        n_coef,
        params.min.0,
    ))
}

/////////////////////
// Shared per-gene //
/////////////////////

/// Everything the per-gene fit reads but never writes.
struct Shared<'a> {
    /// Centred and scaled design, row-major `n_cells * n_coef`.
    design: &'a [f64],
    /// Log offset per cell.
    log_offset: &'a [f64],
    /// Subject boundaries, length `n_subjects + 1`.
    fid: &'a [usize],
    /// Number of cells.
    n_cells: usize,
    /// Number of design columns.
    n_coef: usize,
    /// Number of subjects.
    n_subjects: usize,
    /// Index of the intercept column.
    intercept: usize,
    /// Log of the mean offset, nebula's `moffset`.
    ///
    /// The log of the arithmetic mean, not the mean of the logs: nebula computes
    /// `log(mexpoffset)` and leaves `cv_offset`'s own `moffset`, which is the
    /// mean of the logs, unused.
    log_mean_offset: f64,
    /// Cells per subject, nebula's `mfs`.
    cells_per_subject: f64,
    /// Squared coefficient of variation of the offsets, the fallback when no
    /// design column varies within a subject.
    cv2: f64,
    /// Design columns that vary within at least one subject.
    cell_columns: &'a [usize],
    /// The variant actually being run, after the cells-per-subject override.
    method: NebulaMethod,
    /// The user's knobs.
    params: NebulaParams,
}

/// One gene's fit on the centred design scale.
struct GeneOutcome {
    /// Fixed effects, length `n_coef`.
    beta: Vec<f64>,
    /// Packed upper-triangular covariance, length `packed_len(n_coef)`.
    covariance: Vec<f64>,
    /// Standard errors, length `n_coef`.
    se: Vec<f64>,
    /// Subject-level overdispersion, nebula's `sigma^2`.
    subject: f64,
    /// Cell-level overdispersion, nebula's `phi^-1`.
    cell: f64,
    /// nebula's convergence code.
    convergence: i32,
}

//////////////////
// Per-gene fit //
//////////////////

/// Runs the three stages for one gene.
///
/// ### Params
///
/// * `shared` - The inputs common to every gene
/// * `counts` - This gene's positive counts and their summaries
/// * `subject_totals` - This gene's count total per subject, nebula's `cumsumy`
///
/// ### Returns
///
/// The fit on the centred design scale, or [`EdgeErrors`] if the kernels reject
/// the assembled per-gene view.
fn fit_gene(
    shared: &Shared<'_>,
    counts: &crate::sc::ptmg::GeneCounts,
    subject_totals: &[f64],
) -> Result<GeneOutcome, EdgeErrors> {
    let n_coef = shared.n_coef;
    let params = &shared.params;

    // nebula raises the Laplace order for genes whose expected count per
    // subject is low; the leading term alone is biased there.
    let ord = if counts.mean_count * shared.cells_per_subject < HIGH_ORDER_COUNT_CUTOFF {
        HIGH_ORDER
    } else {
        1
    };

    // Stage one: the marginal likelihood over `[beta, sigma, phi]`.
    let gene = GeneData::new(
        shared.design,
        shared.log_offset,
        &counts.counts,
        &counts.cells,
        subject_totals,
        shared.fid,
        n_coef,
    )?;
    let log_mean_count = counts.mean_count.ln();
    let mut start = vec![0.0; n_coef + 2];
    start[shared.intercept] = log_mean_count - shared.log_mean_offset;
    start[n_coef] = 1.0;
    start[n_coef + 1] = 1.0;

    let mut lower = vec![-BETA_BOUND; n_coef + 2];
    let mut upper = vec![BETA_BOUND; n_coef + 2];
    lower[n_coef] = params.min.0;
    lower[n_coef + 1] = params.min.1;
    upper[n_coef] = params.max.0;
    upper[n_coef + 1] = params.max.1;

    let (stage_one, stage_one_failed) = minimise_marginal(&gene, &start, &lower, &upper);
    let mut convergence = if stage_one_failed { 0 } else { CONV_SUCCESS };
    let mut sigma = stage_one[n_coef];
    let mut gamma = stage_one[n_coef + 1];

    // nebula discards the stage-one fixed effects and restarts them from the
    // gene's mean count, keeping only the two variance components.
    let mut beta_start = vec![0.0; n_coef];
    beta_start[shared.intercept] = log_mean_count - shared.log_mean_offset;

    let pml = PmlData {
        design: shared.design,
        offset: shared.log_offset,
        counts: &counts.counts,
        cell_index: &counts.cells,
        subject_start: shared.fid,
        subject_total: subject_totals,
    };

    // Stage two: the profile likelihood over the variance components alone.
    let refit_both = |sigma: f64, gamma: f64| -> (f64, f64, bool) {
        let x = refine_variance(
            shared,
            &pml,
            &beta_start,
            counts,
            ord,
            &[sigma, gamma],
            None,
        );
        match x {
            Some(v) => (v[0], v[1], true),
            None => (sigma, gamma, false),
        }
    };

    match shared.method {
        NebulaMethod::Hl => {
            let (s, g, ok) = refit_both(sigma, gamma);
            sigma = s;
            gamma = g;
            convergence = if ok { CONV_SUCCESS } else { CONV_OUTER_FAILED };
        }
        NebulaMethod::Ln => {
            // nebula measures how far the fitted cell means spread within a
            // subject; without a cell-level column that is just the offsets.
            let cv2p = if shared.cell_columns.is_empty() {
                shared.cv2
            } else {
                design_cv(
                    shared.log_offset,
                    shared.design,
                    shared.n_cells,
                    n_coef,
                    &stage_one[..n_coef],
                    shared.cell_columns,
                )?
            };
            let gni = shared.cells_per_subject * gamma;
            if gni < params.cutoff_cell || convergence == 0 || cv2p.is_nan() {
                let (s, g, ok) = refit_both(sigma, gamma);
                sigma = s;
                gamma = g;
                convergence = if ok { CONV_SUCCESS } else { CONV_OUTER_FAILED };
            } else {
                let kappa_obs = gni / (1.0 + cv2p);
                let weak = kappa_obs < KAPPA_FLOOR
                    || (kappa_obs < params.kappa && sigma < KAPPA_SIGMA_NUMERATOR / kappa_obs);
                if weak {
                    let x = refine_variance(
                        shared,
                        &pml,
                        &beta_start,
                        counts,
                        ord,
                        &[sigma],
                        Some(gamma),
                    );
                    match x {
                        Some(v) => {
                            sigma = v[0];
                            convergence = CONV_SUCCESS;
                        }
                        None => convergence = CONV_OUTER_FAILED,
                    }
                }
            }
        }
    }

    // Stage three: the final penalised fit, always at the leading Laplace order.
    let final_params = PmlParams {
        reml: params.reml,
        eps: params.eps,
        ord: 1,
        ..PmlParams::default()
    };
    beta_start[shared.intercept] -= sigma / 2.0;
    let fit = opt_pml(
        &pml,
        &beta_start,
        &PmlVariance {
            subject: sigma,
            cell: gamma,
        },
        Some(final_params),
    )?;

    let at_bound = sigma == params.max.0 || gamma == params.min.1;
    let mut code = check_convergence(&fit, &final_params, at_bound, convergence);

    let packed = packed_len(n_coef);
    let (covariance, se) = match cholesky_inverse(&fit.information, n_coef) {
        Some(inverse) if code != CONV_SINGULAR => {
            let mut packed_cov = vec![0.0; packed];
            for j in 0..n_coef {
                for i in 0..=j {
                    packed_cov[j * (j + 1) / 2 + i] = inverse[i * n_coef + j];
                }
            }
            let se = (0..n_coef)
                .map(|j| inverse[j * n_coef + j].sqrt())
                .collect();
            (packed_cov, se)
        }
        _ => {
            code = CONV_SINGULAR;
            (vec![f64::NAN; packed], vec![f64::NAN; n_coef])
        }
    };

    Ok(GeneOutcome {
        beta: fit.beta,
        covariance,
        se,
        subject: sigma,
        cell: 1.0 / gamma,
        convergence: code,
    })
}

/////////////////////////
// Marginal likelihood //
/////////////////////////

/// Minimises the marginal negative log-likelihood over `[beta, sigma, phi]`.
///
/// This is nebula's stage one: one call to [`minimise`] with an exact gradient.
/// `sigma` runs to its lower bound on most genes, which is the case that used to
/// defeat the line search here; see the `max_feasible_step` note in
/// `numeric::lbfgsb`.
///
/// No quadratic polish here, unlike [`refine_variance`]: the marginal likelihood
/// is smooth, so there is no jitter to average out, and the stencil costs
/// `3^(n_coef + 2)` evaluations, which is exponential in something the caller
/// chooses. What is left over is well inside nebula's own stage-one tolerance,
/// an absolute `1e-6` on the objective, which moves its `phi` by up to `4e-6`
/// relative on the same fixture.
///
/// ### Params
///
/// * `gene` - The gene, as validated by [`GeneData::new`]
/// * `start` - Starting point, `[beta, sigma, phi]`
/// * `lower` - Lower bounds
/// * `upper` - Upper bounds
///
/// ### Returns
///
/// The best point found, and whether every attempt failed outright, which is
/// what nebula's `is_conv` records.
fn minimise_marginal(
    gene: &GeneData<'_>,
    start: &[f64],
    lower: &[f64],
    upper: &[f64],
) -> (Vec<f64>, bool) {
    let n = start.len();
    let mut best: Vec<f64> = (0..n).map(|j| start[j].clamp(lower[j], upper[j])).collect();
    let best_f = ptmg_value_and_gradient(gene, &best).0;
    if !best_f.is_finite() {
        return (best, true);
    }

    let params = LbfgsbParams {
        ftol: STAGE_ONE_FTOL,
        pgtol: STAGE_ONE_PGTOL,
        max_iter: STAGE_ONE_MAX_ITER,
        max_line_search: STAGE_ONE_MAX_LINE_SEARCH,
        ..LbfgsbParams::default()
    };

    if let Ok(step) = minimise(
        |x, g| {
            let (value, gradient) = ptmg_value_and_gradient(gene, x);
            g.copy_from_slice(&gradient);
            value
        },
        &best,
        lower,
        upper,
        Some(params),
    ) && step.f < best_f
    {
        best = step.x;
    }

    (best, false)
}

////////////////////////
// Variance objective //
////////////////////////

/// nebula's `pql_ll`: the marginal likelihood with the fixed effects and the
/// random effects profiled out by [`opt_pml`].
///
/// Every evaluation runs a full penalised fit, so this is the expensive part of
/// a NEBULA run. The `invalid` flag records whether any evaluation was rejected,
/// which is how the caller learns to retry at the leading Laplace order, as the
/// R package does through its `tryCatch`.
struct VarianceObjective<'a> {
    /// The gene, in the layout [`opt_pml`] wants.
    data: &'a PmlData<'a>,
    /// Fixed effects to start the inner fit from, before the intercept shift.
    beta_start: &'a [f64],
    /// Index of the intercept column.
    intercept: usize,
    /// Knobs for the inner fit, including the Laplace order under test.
    params: PmlParams,
    /// Number of cells.
    n_cells: f64,
    /// Number of subjects.
    n_subjects: f64,
    /// This gene's positive counts.
    counts: &'a [f64],
    /// How many positive counts there are.
    n_positive: f64,
    /// How many equal one.
    n_one: f64,
    /// How many equal two.
    n_two: f64,
    /// Cell-level overdispersion held fixed, for the one-dimensional restriction
    /// NEBULA-LN uses.
    fixed_cell: Option<f64>,
    /// Whether any evaluation had to be rejected.
    invalid: Cell<bool>,
}

impl VarianceObjective<'_> {
    /// Evaluates the negated profile log-likelihood.
    ///
    /// ### Params
    ///
    /// * `x` - `[sigma]` when [`Self::fixed_cell`] is set, else `[sigma, phi]`
    ///
    /// ### Returns
    ///
    /// The objective, or positive infinity where the inner fit failed or the
    /// higher-order Laplace correction left `log(1 + second)` undefined.
    fn value(&self, x: &[f64]) -> f64 {
        let subject = x[0];
        let cell = match self.fixed_cell {
            Some(c) => c,
            None => x[1],
        };
        if !(subject.is_finite() && subject > 0.0 && cell.is_finite() && cell > 0.0) {
            self.invalid.set(true);
            return f64::INFINITY;
        }

        let exps = subject.exp();
        let alpha = 1.0 / (exps - 1.0);
        let lambda = 1.0 / (exps.sqrt() * (exps - 1.0));

        let mut beta = self.beta_start.to_vec();
        beta[self.intercept] -= subject / 2.0;

        let fit = match opt_pml(
            self.data,
            &beta,
            &PmlVariance { subject, cell },
            Some(self.params),
        ) {
            Ok(f) => f,
            Err(_) => {
                self.invalid.set(true);
                return f64::INFINITY;
            }
        };
        // nebula raises an error here and restarts the whole search at `ord = 1`.
        if fit.second_order < -1.0 {
            self.invalid.set(true);
            return f64::INFINITY;
        }

        let base = if fit.log_likelihood.is_nan() {
            fit.log_likelihood_prev
        } else {
            fit.log_likelihood
        };
        let mut log_likelihood =
            base + self.n_cells * cell * cell.ln() + self.n_subjects * alpha * lambda.ln()
                - self.n_subjects * ln_gamma(alpha);

        // Counts of one and two have closed forms for the gamma ratio, which is
        // most of a single-cell matrix; only the rest needs `lgamma`.
        let mut tail = 0.0;
        for &y in self.counts {
            if y != 1.0 && y != 2.0 {
                tail += ln_gamma(y + cell);
            }
        }
        log_likelihood += tail - (self.n_positive - self.n_one - self.n_two) * ln_gamma(cell)
            + (self.n_one + self.n_two) * cell.ln()
            + self.n_two * (cell + 1.0).ln();
        log_likelihood += -0.5 * fit.log_det + (1.0 + fit.second_order).ln();

        if log_likelihood.is_finite() {
            -log_likelihood
        } else {
            self.invalid.set(true);
            f64::INFINITY
        }
    }
}

/// Minimises the profile likelihood over the variance components.
///
/// Bounded Nelder-Mead, then the quadratic polish cascade. If the search at a
/// raised Laplace order hit an evaluation the expansion could not support, the
/// whole thing is repeated at the leading order, which is what the R package's
/// `tryCatch` around `bobyqa` does.
///
/// ### Params
///
/// * `shared` - The inputs common to every gene
/// * `pml` - The gene, in the layout [`opt_pml`] wants
/// * `beta_start` - Fixed effects to start each inner fit from
/// * `counts` - This gene's positive counts and their summaries
/// * `ord` - Laplace order to attempt first
/// * `start` - Starting variance components
/// * `fixed_cell` - Cell-level overdispersion to hold fixed, for the
///   one-dimensional restriction NEBULA-LN uses
///
/// ### Returns
///
/// The minimiser, or `None` if no evaluation of the objective was finite.
fn refine_variance(
    shared: &Shared<'_>,
    pml: &PmlData<'_>,
    beta_start: &[f64],
    counts: &crate::sc::ptmg::GeneCounts,
    ord: u32,
    start: &[f64],
    fixed_cell: Option<f64>,
) -> Option<Vec<f64>> {
    debug_assert_eq!(start.len(), if fixed_cell.is_some() { 1 } else { 2 });
    let params = &shared.params;
    let (lower, upper) = if fixed_cell.is_some() {
        (vec![params.min.0], vec![params.max.0])
    } else {
        (
            vec![params.min.0, params.min.1],
            vec![params.max.0, params.max.1],
        )
    };
    for order in [ord, 1] {
        let objective = VarianceObjective {
            data: pml,
            beta_start,
            intercept: shared.intercept,
            params: PmlParams {
                reml: params.reml,
                eps: params.eps,
                ord: order,
                ..PmlParams::default()
            },
            n_cells: shared.n_cells as f64,
            n_subjects: shared.n_subjects as f64,
            counts: &counts.counts,
            n_positive: counts.counts.len() as f64,
            n_one: counts.n_one as f64,
            n_two: counts.n_two as f64,
            fixed_cell,
            invalid: Cell::new(false),
        };
        let evaluate = |x: &[f64]| objective.value(x);
        let found = minimise_box(&evaluate, start, &lower, &upper);
        if found.is_some() && !objective.invalid.get() {
            return found;
        }
        if order == 1 {
            return found;
        }
    }
    None
}

/// Bounded derivative-free minimisation with a quadratic polish.
///
/// Nelder-Mead localises the basin, evaluating a point clamped into the box so
/// that a component can settle exactly on its bound, then a least-squares
/// quadratic over a shrinking sequence of stencils finds the minimum inside it.
/// The polish is the part that matters: the objective jitters at the `1e-6`
/// level, and a quadratic fitted over a stencil wider than the jitter averages
/// it out where a simplex does not.
///
/// ### Params
///
/// * `objective` - The profile likelihood, already clamped-safe
/// * `start` - Starting point, one or two variance components
/// * `lower` - Lower bounds
/// * `upper` - Upper bounds
///
/// ### Returns
///
/// The minimiser, clamped into the box, or `None` if nothing finite was found.
fn minimise_box(
    objective: &dyn Fn(&[f64]) -> f64,
    start: &[f64],
    lower: &[f64],
    upper: &[f64],
) -> Option<Vec<f64>> {
    let n = start.len();
    let clamped = |x: &[f64]| -> Vec<f64> {
        (0..n)
            .map(|j| x[j].clamp(lower[j], upper[j]))
            .collect::<Vec<f64>>()
    };
    let evaluate = |x: &[f64]| objective(&clamped(x));

    let x0 = clamped(start);
    if !evaluate(&x0).is_finite() {
        return None;
    }

    let simplex = nelder_mead(
        evaluate,
        &x0,
        Some(NelderMeadParams {
            xatol: VARIANCE_XATOL,
            fatol: VARIANCE_FATOL,
            max_iter: VARIANCE_MAX_ITER,
        }),
    )
    .ok()?;
    let mut best = clamped(&simplex.x);
    if !objective(&best).is_finite() {
        return None;
    }

    for width in POLISH_WIDTHS {
        let step: Vec<f64> = (0..n)
            .map(|j| (width * best[j].abs()).max(POLISH_FLOOR[j]))
            .collect();
        if let Some(next) = quadratic_polish(objective, &best, lower, upper, &step) {
            best = next;
        }
    }
    Some(best)
}

/// One step of Newton on a quadratic fitted to a tensor-product stencil.
///
/// The stencil is `3^n` points at offsets `-1, 0, 1` times `step`, shifted to
/// `0, 1, 2` or `-2, -1, 0` in any coordinate sitting on a bound so that the
/// points stay distinct and inside the box. The quadratic is fitted by least
/// squares through the normal equations, which are well conditioned at these
/// sizes, and the step is taken only if the fitted curvature is positive
/// definite and the step stays inside the stencil.
///
/// ### Params
///
/// * `objective` - The function being minimised
/// * `x` - Incumbent
/// * `lower` - Lower bounds
/// * `upper` - Upper bounds
/// * `step` - Stencil width per coordinate
///
/// ### Returns
///
/// The polished point, or `None` if the model was unusable.
fn quadratic_polish(
    objective: &dyn Fn(&[f64]) -> f64,
    x: &[f64],
    lower: &[f64],
    upper: &[f64],
    step: &[f64],
) -> Option<Vec<f64>> {
    let n = x.len();
    let offsets: Vec<[f64; 3]> = (0..n)
        .map(|j| {
            if x[j] - step[j] < lower[j] {
                [0.0, 1.0, 2.0]
            } else if x[j] + step[j] > upper[j] {
                [-2.0, -1.0, 0.0]
            } else {
                [-1.0, 0.0, 1.0]
            }
        })
        .collect();

    // 1 constant, n linear, n square and n(n-1)/2 cross terms.
    let n_terms = 1 + 2 * n + n * (n - 1) / 2;
    let n_points = 3_usize.pow(n as u32);
    let mut rows = Vec::with_capacity(n_points * n_terms);
    let mut values = Vec::with_capacity(n_points);

    for point in 0..n_points {
        let mut u = vec![0.0; n];
        let mut rest = point;
        for j in 0..n {
            let node = offsets[j][rest % 3];
            rest /= 3;
            let p = (x[j] + node * step[j]).clamp(lower[j], upper[j]);
            u[j] = (p - x[j]) / step[j];
        }
        let candidate: Vec<f64> = (0..n).map(|j| x[j] + u[j] * step[j]).collect();
        let f = objective(&candidate);
        if !f.is_finite() {
            return None;
        }
        values.push(f);

        rows.push(1.0);
        rows.extend_from_slice(&u);
        rows.extend(u.iter().map(|v| 0.5 * v * v));
        for i in 0..n {
            for j in (i + 1)..n {
                rows.push(u[i] * u[j]);
            }
        }
    }

    // Normal equations. Cheaper than a QR and perfectly conditioned here: the
    // stencil spans the quadratic space exactly.
    let mut normal = vec![0.0; n_terms * n_terms];
    let mut rhs = vec![0.0; n_terms];
    for (point, &f) in values.iter().enumerate() {
        let row = &rows[point * n_terms..(point + 1) * n_terms];
        for a in 0..n_terms {
            rhs[a] += row[a] * f;
            for b in 0..n_terms {
                normal[a * n_terms + b] += row[a] * row[b];
            }
        }
    }
    let coefficients = cholesky_solve(&normal, n_terms, &rhs)?;

    let gradient = &coefficients[1..=n];
    let mut hessian = vec![0.0; n * n];
    for j in 0..n {
        hessian[j * n + j] = coefficients[1 + n + j];
    }
    let mut cross = 1 + 2 * n;
    for i in 0..n {
        for j in (i + 1)..n {
            hessian[i * n + j] = coefficients[cross];
            hessian[j * n + i] = coefficients[cross];
            cross += 1;
        }
    }

    let negated: Vec<f64> = gradient.iter().map(|g| -g).collect();
    let delta = cholesky_solve(&hessian, n, &negated)?;
    if delta
        .iter()
        .any(|d| !d.is_finite() || d.abs() > POLISH_MAX_STEP)
    {
        return None;
    }

    Some(
        (0..n)
            .map(|j| (x[j] + delta[j] * step[j]).clamp(lower[j], upper[j]))
            .collect(),
    )
}

////////////////////
// Linear algebra //
////////////////////

/// Cholesky factor of a symmetric positive definite matrix.
///
/// ### Params
///
/// * `a` - Row-major `n * n` matrix, only the lower triangle is read
/// * `n` - Side length
///
/// ### Returns
///
/// The lower triangular factor, row-major, or `None` if `a` is not positive
/// definite.
fn cholesky_factor(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if !(sum.is_finite() && sum > 0.0) {
                    return None;
                }
                l[i * n + j] = sum.sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }
    Some(l)
}

/// Solves `A x = b` for symmetric positive definite `A`.
///
/// ### Params
///
/// * `a` - Row-major `n * n` matrix
/// * `n` - Side length
/// * `b` - Right-hand side, length `n`
///
/// ### Returns
///
/// The solution, or `None` if `a` is not positive definite.
fn cholesky_solve(a: &[f64], n: usize, b: &[f64]) -> Option<Vec<f64>> {
    let l = cholesky_factor(a, n)?;
    let mut x = b.to_vec();
    for i in 0..n {
        let mut sum = x[i];
        for k in 0..i {
            sum -= l[i * n + k] * x[k];
        }
        x[i] = sum / l[i * n + i];
    }
    for i in (0..n).rev() {
        let mut sum = x[i];
        for k in (i + 1)..n {
            sum -= l[k * n + i] * x[k];
        }
        x[i] = sum / l[i * n + i];
    }
    if x.iter().all(|v| v.is_finite()) {
        Some(x)
    } else {
        None
    }
}

/// Inverse of a symmetric positive definite matrix, nebula's `Rfast::spdinv`.
///
/// ### Params
///
/// * `a` - Row-major `n * n` matrix
/// * `n` - Side length
///
/// ### Returns
///
/// The inverse, row-major, or `None` if `a` is not positive definite.
fn cholesky_inverse(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut inverse = vec![0.0; n * n];
    let mut unit = vec![0.0; n];
    for j in 0..n {
        unit.iter_mut().for_each(|v| *v = 0.0);
        unit[j] = 1.0;
        let column = cholesky_solve(a, n, &unit)?;
        for i in 0..n {
            inverse[i * n + j] = column[i];
        }
    }
    Some(inverse)
}

/////////////
// Helpers //
/////////////

/// Turns a run encoding of subject labels into half-open cell blocks.
///
/// ### Params
///
/// * `subject_id` - Subject of each cell
///
/// ### Returns
///
/// Boundaries of length `n_subjects + 1`, or
/// [`EdgeErrors::SubjectsNotContiguous`] if a subject's cells are not one run.
fn subject_boundaries(subject_id: &[usize]) -> Result<Vec<usize>, EdgeErrors> {
    let mut fid = vec![0usize];
    let mut seen: Vec<usize> = vec![subject_id[0]];
    for i in 1..subject_id.len() {
        if subject_id[i] != subject_id[i - 1] {
            if seen.contains(&subject_id[i]) {
                return Err(EdgeErrors::SubjectsNotContiguous {
                    subject: subject_id[i],
                });
            }
            seen.push(subject_id[i]);
            fid.push(i);
        }
    }
    fid.push(subject_id.len());
    Ok(fid)
}

/// Converts a generic slice to `f64`, rejecting anything that will not fit.
///
/// ### Params
///
/// * `name` - Name used in the error message
/// * `values` - The slice to convert
///
/// ### Returns
///
/// The converted values, or [`EdgeErrors::InvalidArgument`] on a value that does
/// not convert.
fn to_f64<T: EdgeFloat>(name: &str, values: &[T]) -> Result<Vec<f64>, EdgeErrors> {
    values
        .iter()
        .map(|v| {
            v.to_f64().ok_or_else(|| {
                EdgeErrors::InvalidArgument(format!("{name} holds a non-finite value."))
            })
        })
        .collect()
}

/// Builds the gene-major sparse view the kernels read counts from.
///
/// Only strictly positive counts are stored, which is what nebula's C++ does
/// with its `dgCMatrix`.
///
/// ### Params
///
/// * `counts` - Dense counts, gene-major and row-major
/// * `n_genes` - Number of genes
/// * `n_cells` - Number of cells
///
/// ### Returns
///
/// The compressed matrix, or [`EdgeErrors::InvalidArgument`] on a negative or
/// non-finite count.
fn build_csr<T: EdgeFloat>(
    counts: &[T],
    n_genes: usize,
    n_cells: usize,
) -> Result<CompressedSparse<f64>, EdgeErrors> {
    let mut data = Vec::new();
    let mut indices = Vec::new();
    let mut indptr = Vec::with_capacity(n_genes + 1);
    indptr.push(0u32);

    for gene in 0..n_genes {
        for (cell, value) in counts[gene * n_cells..(gene + 1) * n_cells]
            .iter()
            .enumerate()
        {
            let v = value.to_f64().unwrap_or(f64::NAN);
            if !v.is_finite() || v < 0.0 {
                return Err(EdgeErrors::InvalidArgument(format!(
                    "Count for gene {gene}, cell {cell} is not a non-negative finite number."
                )));
            }
            if v > 0.0 {
                data.push(v);
                indices.push(cell as u32);
            }
        }
        indptr.push(data.len() as u32);
    }

    CompressedSparse::from_parts(data, indices, indptr, SparseFormat::Csr, (n_genes, n_cells))
}

/// Undoes the centring and scaling and packs the per-gene fits.
///
/// nebula scales every non-constant design column to unit population standard
/// deviation before fitting, so the coefficients, standard errors and
/// covariances all come back on that scale and have to be divided out again.
///
/// ### Params
///
/// * `outcomes` - Per-gene fits on the centred scale, in `kept` order
/// * `kept` - Zero-based indices of the genes that survived the filter
/// * `sds` - Per-column standard deviations from [`centre_design`]
/// * `intercept` - Index of the intercept column, which is not rescaled
/// * `n_coef` - Number of design columns
/// * `min_subject` - Lower bound the subject-level overdispersion was fitted
///   under, used to raise [`NebulaFit::sigma_at_bound`]
///
/// ### Returns
///
/// The assembled fit.
fn assemble(
    outcomes: Vec<GeneOutcome>,
    kept: &[usize],
    sds: &[f64],
    intercept: usize,
    n_coef: usize,
    min_subject: f64,
) -> NebulaFit {
    let mut scale = sds.to_vec();
    scale[intercept] = 1.0;

    let packed = packed_len(n_coef);
    let n = outcomes.len();
    let mut fit = NebulaFit {
        coefficients: Vec::with_capacity(n * n_coef),
        covariance: Vec::with_capacity(n * packed),
        se: Vec::with_capacity(n * n_coef),
        subject_overdispersion: Vec::with_capacity(n),
        cell_overdispersion: Vec::with_capacity(n),
        convergence: Vec::with_capacity(n),
        sigma_at_bound: Vec::with_capacity(n),
        gene_index: kept.to_vec(),
        n_coef,
    };

    for outcome in outcomes {
        for (j, s) in scale.iter().enumerate() {
            fit.coefficients.push(outcome.beta[j] / s);
            fit.se.push(outcome.se[j] / s);
        }
        for j in 0..n_coef {
            for i in 0..=j {
                fit.covariance
                    .push(outcome.covariance[j * (j + 1) / 2 + i] / (scale[i] * scale[j]));
            }
        }
        fit.subject_overdispersion.push(outcome.subject);
        fit.cell_overdispersion.push(outcome.cell);
        fit.convergence.push(outcome.convergence);
        fit.sigma_at_bound
            .push(outcome.subject <= min_subject * BOUND_SLACK);
    }

    fit
}

///////////
// Tests //
///////////

#[cfg(test)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::path::PathBuf;

    // Every tolerance below is set a factor of three or so above the worst case
    // actually measured against nebula 1.5.8, quoted with it. None of them is
    // this port's error: nebula stops BOBYQA at `xtol_rel = 1e-6` on a profile
    // likelihood that is itself discontinuous at the `1e-6` level, and two of
    // its own runs from different starting points disagree by as much.

    /// Tolerance on the coefficients. Measured worst case `1.0e-6`.
    ///
    /// Almost all of that is the intercept, which nebula shifts by `-sigma / 2`
    /// before the final fit, so an error in the subject-level overdispersion
    /// lands on it halved and absolute.
    const TOL_COEF: f64 = 5e-6;

    /// Tolerance on the standard errors. Measured worst case `2.1e-6`.
    const TOL_SE: f64 = 1e-5;

    /// Tolerance on the covariance entries, which are standard errors squared,
    /// so twice the relative error. Measured worst case `3.1e-6`.
    const TOL_COV: f64 = 1e-5;

    /// Tolerance on the subject-level overdispersion. Measured worst case
    /// `3.6e-6`.
    ///
    /// The profile likelihood is far flatter in this direction than in the
    /// cell-level one, so the same jitter in the objective moves it further.
    const TOL_SUBJECT: f64 = 1e-5;

    /// Tolerance on the cell-level overdispersion. Measured worst case `4.4e-6`,
    /// on the NEBULA-LN path.
    ///
    /// There it is carried straight out of stage one, where nebula's own
    /// absolute `1e-6` stopping rule leaves it uncertain at that level: running
    /// the reference's own optimiser to `1e-14` moves it by up to `3.6e-6`.
    const TOL_CELL: f64 = 1e-5;

    /// Reads a CSV from `tests/data` into a row-major matrix.
    ///
    /// ### Params
    ///
    /// * `name` - File name inside `tests/data`
    /// * `header` - Whether the first line names the columns
    ///
    /// ### Returns
    ///
    /// The values row-major, the row count and the column count.
    fn read_csv(name: &str, header: bool) -> (Vec<f64>, usize, usize) {
        let path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "tests", "data", name]
            .iter()
            .collect();
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));

        let mut lines = text.lines().filter(|l| !l.trim().is_empty());
        let mut n_cols = 0;
        if header {
            n_cols = lines
                .next()
                .expect("fixture has no header")
                .split(',')
                .count();
        }
        let mut values = Vec::new();
        let mut n_rows = 0_usize;
        for line in lines {
            let fields: Vec<&str> = line.split(',').collect();
            if n_cols == 0 {
                n_cols = fields.len();
            }
            for field in fields {
                values.push(
                    field
                        .trim()
                        .trim_matches('"')
                        .parse::<f64>()
                        .unwrap_or_else(|e| panic!("bad field {field:?} in {name}: {e}")),
                );
            }
            n_rows += 1;
        }
        assert_eq!(values.len(), n_rows * n_cols, "ragged fixture {name}");
        (values, n_rows, n_cols)
    }

    /// The 8 by 150 fixture: counts, subject labels, design and offsets.
    ///
    /// ### Returns
    ///
    /// Counts gene-major, the subject of each cell, the three-column design
    /// `[1, grp, cov2]` row-major, and the offsets.
    fn fixture() -> (Vec<f64>, Vec<usize>, Vec<f64>, Vec<f64>) {
        let (counts, n_genes, n_cells) = read_csv("nebula_counts.csv", false);
        assert_eq!((n_genes, n_cells), (8, 150));
        let (design_raw, rows, cols) = read_csv("nebula_design.csv", true);
        assert_eq!((rows, cols), (150, 4));

        let subject: Vec<usize> = (0..rows).map(|i| design_raw[i * cols] as usize).collect();
        let mut design = Vec::with_capacity(rows * 3);
        let mut offset = Vec::with_capacity(rows);
        for i in 0..rows {
            design.push(1.0);
            design.push(design_raw[i * cols + 1]);
            design.push(design_raw[i * cols + 2]);
            offset.push(design_raw[i * cols + 3]);
        }
        (counts, subject, design, offset)
    }

    /// Default knobs plus the fixture's `cutoff_cell = 0`.
    ///
    /// ### Returns
    ///
    /// The parameter set the goldens were generated with.
    fn golden_params() -> NebulaParams {
        NebulaParams {
            cutoff_cell: 0.0,
            ..NebulaParams::default()
        }
    }

    /// Reorders one gene's R covariance row into this crate's packing.
    ///
    /// R returns `lower.tri(diag = TRUE)` in column-major order, which for a
    /// symmetric matrix reads `V11, V12, V13, V22, V23, V33`. This crate packs
    /// the upper triangle column-major, `V11, V12, V22, V13, V23, V33`. The two
    /// coincide for two coefficients and diverge from three.
    ///
    /// ### Params
    ///
    /// * `row` - One gene's covariance as R prints it, three coefficients
    ///
    /// ### Returns
    ///
    /// The same six values in this crate's order.
    fn repack(row: &[f64]) -> Vec<f64> {
        vec![row[0], row[1], row[3], row[2], row[4], row[5]]
    }

    /// Asserts a convergence code against what the R package reported.
    ///
    /// nebula grades a fit by how many times the last Newton step had to be
    /// halved, and at a converged point that count is settled by rounding: R
    /// itself returns 7, 3 and 0 for three values of `phi` a part in `1e8`
    /// apart. So a gene the R package calls converged is allowed to come back as
    /// `CONV_CRITICAL_POINT` here, and nothing else is.
    ///
    /// ### Params
    ///
    /// * `got` - This crate's code
    /// * `want` - The R package's code
    fn assert_convergence(got: i32, want: i32) {
        if want == CONV_SUCCESS {
            assert!(
                got == CONV_SUCCESS || got == crate::sc::pml::CONV_CRITICAL_POINT,
                "convergence {got}, expected 1 or -10"
            );
        } else {
            assert_eq!(got, want);
        }
    }

    /// The full three-coefficient fit against nebula 1.5.8.
    ///
    /// Produced by:
    ///
    /// ```r
    /// nebula(cnt, id, pred = cbind(1, grp, cov2), offset = rep(1000, n_cell),
    ///        model = "NBGMM", method = "LN", cutoff_cell = 0,
    ///        covariance = TRUE, verbose = FALSE, ncore = 1)
    /// ```
    #[test]
    fn test_nebula_matches_r_package() {
        let (counts, subject, design, offset) = fixture();
        let fit = nebula(
            &counts,
            8,
            150,
            &subject,
            &design,
            3,
            Some(&offset),
            Some(golden_params()),
        )
        .expect("nebula failed");

        let (expected, n_rows, n_cols) = read_csv("nebula_expected.csv", true);
        let (cov, cov_rows, cov_cols) = read_csv("nebula_covariance.csv", true);
        assert_eq!((n_rows, n_cols), (8, 13));
        assert_eq!((cov_rows, cov_cols), (8, 6));
        assert_eq!(fit.gene_index, (0..8).collect::<Vec<usize>>());
        assert_eq!(fit.n_coef, 3);

        for g in 0..8 {
            let row = &expected[g * n_cols..(g + 1) * n_cols];
            for j in 0..3 {
                assert_relative_eq!(fit.coefficients[g * 3 + j], row[j], max_relative = TOL_COEF);
                assert_relative_eq!(fit.se[g * 3 + j], row[3 + j], max_relative = TOL_SE);
            }
            assert_relative_eq!(
                fit.subject_overdispersion[g],
                row[10],
                max_relative = TOL_SUBJECT
            );
            assert_relative_eq!(fit.cell_overdispersion[g], row[11], max_relative = TOL_CELL);
            assert_convergence(fit.convergence[g], row[12] as i32);

            for (k, w) in repack(&cov[g * 6..(g + 1) * 6]).iter().enumerate() {
                assert_relative_eq!(fit.covariance[g * 6 + k], w, max_relative = TOL_COV);
            }
        }
    }

    /// The fit feeds [`glm_sc_test`] and reproduces nebula's p-values.
    ///
    /// This is the layout check that matters in practice: `glm_sc_test` reads
    /// the packed covariance, so a p-value that lands on nebula's confirms both
    /// the packing and the standard errors end to end.
    #[test]
    fn test_wald_test_reproduces_the_r_p_values() {
        use crate::sc::test::{ScTested, glm_sc_test};

        let (counts, subject, design, offset) = fixture();
        let fit = nebula(
            &counts,
            8,
            150,
            &subject,
            &design,
            3,
            Some(&offset),
            Some(golden_params()),
        )
        .expect("nebula failed");
        let (expected, _, n_cols) = read_csv("nebula_expected.csv", true);

        for coef in 1..3 {
            let tested = glm_sc_test(
                &fit.coefficients,
                &fit.covariance,
                8,
                3,
                &ScTested::Coef(coef),
            )
            .expect("test failed");
            for g in 0..8 {
                assert_relative_eq!(
                    tested.p_value[g],
                    expected[g * n_cols + 6 + coef],
                    max_relative = 1e-4
                );
            }
        }
    }

    /// The same counts with a two-column design.
    ///
    /// Produced by:
    ///
    /// ```r
    /// nebula(cnt, id, pred = cbind(1, grp), offset = rep(1000, n_cell),
    ///        model = "NBGMM", method = "LN", cutoff_cell = 0,
    ///        covariance = TRUE, verbose = FALSE, ncore = 1)
    /// ```
    #[test]
    fn test_nebula_two_column_design() {
        let (counts, subject, design_three, offset) = fixture();
        let design: Vec<f64> = (0..150)
            .flat_map(|i| [design_three[i * 3], design_three[i * 3 + 1]])
            .collect();

        // logFC intercept, logFC grp, se intercept, se grp, Subject, Cell.
        const EXPECTED: [[f64; 6]; 8] = [
            [
                -3.83069111381507010e+00,
                4.67477914516942039e-01,
                1.52851759868728232e-01,
                3.05703519737456464e-01,
                1.19930138981593701e-01,
                2.50509063784787156e-01,
            ],
            [
                -5.22622507034590900e+00,
                2.86854056278882219e-01,
                1.44924959049115953e-01,
                2.89849918098231907e-01,
                1.02875112505358357e-01,
                2.46112537450009855e-01,
            ],
            [
                -2.79850766191427391e+00,
                3.82576141900636657e-01,
                1.46513262535639927e-01,
                2.93026525071279853e-01,
                1.09839244633263694e-01,
                2.79001107949189975e-01,
            ],
            [
                -5.78859782201794282e+00,
                1.39871216066227488e-01,
                1.13748802726646378e-01,
                2.27497605453292728e-01,
                5.63655105828342259e-02,
                1.68916319758526501e-01,
            ],
            [
                -3.18904397195006339e+00,
                4.63470376690180819e-01,
                1.71383246709902126e-01,
                3.42766493419804308e-01,
                1.51708497662862124e-01,
                2.53029183723021411e-01,
            ],
            [
                -4.30946341634605812e+00,
                2.84795644398078140e-01,
                1.12103980708383288e-01,
                2.24207961416766549e-01,
                6.00916997253425900e-02,
                2.56315004910657396e-01,
            ],
            [
                -2.27100390853884093e+00,
                3.57947777093715835e-01,
                1.21567297349690767e-01,
                2.43134594699381507e-01,
                7.29528753821947207e-02,
                3.03612425865116442e-01,
            ],
            [
                -4.71507442768750007e+00,
                5.85766182363788412e-01,
                1.48580330507162378e-01,
                2.97160661014324701e-01,
                1.10500262941949104e-01,
                2.60169207789380963e-01,
            ],
        ];

        // V11, V12, V22. With two coefficients R's packing and this crate's
        // agree, so no reordering is needed.
        const EXPECTED_COV: [[f64; 3]; 8] = [
            [
                2.3363660494967357e-02,
                -6.4743666988604335e-05,
                9.3454641979869441e-02,
            ],
            [
                2.1003243755387938e-02,
                -3.4610402755941365e-04,
                8.4012975021551750e-02,
            ],
            [
                2.1466136098837348e-02,
                -1.6108385963441172e-05,
                8.5864544395349407e-02,
            ],
            [
                1.2938790121745513e-02,
                -3.2251258251132168e-04,
                5.1755160486982046e-02,
            ],
            [
                2.9372217252827182e-02,
                9.4745052152154328e-05,
                1.1748886901130876e-01,
            ],
            [
                1.2567302490665573e-02,
                -1.2031208910192646e-04,
                5.0269209962662278e-02,
            ],
            [
                1.4778607784908132e-02,
                -6.2578541012701255e-05,
                5.9114431139632516e-02,
            ],
            [
                2.2076114613617605e-02,
                -3.2334840390101392e-04,
                8.8304458454470394e-02,
            ],
        ];

        let fit = nebula(
            &counts,
            8,
            150,
            &subject,
            &design,
            2,
            Some(&offset),
            Some(golden_params()),
        )
        .expect("nebula failed");
        assert_eq!(fit.n_coef, 2);
        assert_eq!(fit.covariance.len(), 8 * packed_len(2));

        for (g, want) in EXPECTED.iter().enumerate() {
            for j in 0..2 {
                assert_relative_eq!(
                    fit.coefficients[g * 2 + j],
                    want[j],
                    max_relative = TOL_COEF
                );
                assert_relative_eq!(fit.se[g * 2 + j], want[2 + j], max_relative = TOL_SE);
            }
            assert_relative_eq!(
                fit.subject_overdispersion[g],
                want[4],
                max_relative = TOL_SUBJECT
            );
            assert_relative_eq!(fit.cell_overdispersion[g], want[5], max_relative = TOL_CELL);
            for (k, w) in EXPECTED_COV[g].iter().enumerate() {
                assert_relative_eq!(fit.covariance[g * 3 + k], w, max_relative = TOL_COV);
            }
        }
    }

    /// `sigma_at_bound` marks a gene whose subject-level variance was never
    /// estimated.
    ///
    /// Three of the eight genes in this golden are pinned, and R agrees: the
    /// `Subject` column of `tests/data/nebula_expected.csv` is exactly `1e-4`
    /// for genes 2, 4 and 7. nebula reports all three as converged, because they
    /// are: the marginal likelihood is still descending at the bound, so the
    /// optimiser is sitting on a legitimate constrained optimum. What it does
    /// not report is that the mixed model has collapsed to a plain negative
    /// binomial GLM on those genes, which is what this flag is for.
    ///
    /// Verified against nebula 1.5.8 by lowering the bound and watching the
    /// estimate follow it, which is what tells a constraint from an estimate:
    ///
    /// ```r
    /// a <- nebula(cnt, id, pred = pred, offset = sf, method = "LN")
    /// b <- nebula(cnt, id, pred = pred, offset = sf, method = "LN",
    ///             min = c(1e-8, 1e-4))
    /// # On a 300-gene by 1005-cell dataset, 19 of 298 genes sit on 1e-4 in `a`,
    /// # and 13 of those follow the bound down to 1e-8 in `b`.
    /// ```
    #[test]
    fn test_sigma_at_bound_flags_a_pinned_subject_variance() {
        let (counts, subject, design, offset) = fixture();
        let fit = nebula(&counts, 8, 150, &subject, &design, 3, Some(&offset), None)
            .expect("nebula failed");

        assert_eq!(fit.sigma_at_bound.len(), fit.subject_overdispersion.len());

        let (expected, _, _) = read_csv("nebula_expected.csv", true);
        let floor = NebulaParams::default().min.0;
        let mut flagged = Vec::new();
        for g in 0..8 {
            // Column 10 of the golden is R's `Subject`.
            let r_subject = expected[g * 13 + 10];
            let r_pinned = r_subject <= floor;
            assert_eq!(
                fit.sigma_at_bound[g], r_pinned,
                "gene {g}: flag says {}, R reports sigma^2 = {r_subject:e}",
                fit.sigma_at_bound[g]
            );
            if fit.sigma_at_bound[g] {
                flagged.push(g);
            }
        }
        assert_eq!(
            flagged,
            vec![1, 3, 6],
            "genes 2, 4 and 7 of the golden are pinned, one-based"
        );

        // The flag has to track the bound rather than a hardcoded constant, so
        // raising the floor past every fitted value must flag everything.
        let highest = fit
            .subject_overdispersion
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let params = NebulaParams {
            min: (highest * 2.0, NebulaParams::default().min.1),
            ..Default::default()
        };
        let pinned = nebula(
            &counts,
            8,
            150,
            &subject,
            &design,
            3,
            Some(&offset),
            Some(params),
        )
        .expect("nebula failed");
        assert!(
            pinned.sigma_at_bound.iter().all(|b| *b),
            "every gene should be pinned once the floor is raised past the data"
        );
    }

    /// The NEBULA-LN path, which the main golden never reaches.
    ///
    /// Relabelling the same cells as three subjects of fifty puts the fixture
    /// above thirty cells per subject, so `method = "LN"` survives. Eight of the
    /// nine genes then take nebula's `LN+HL` branch, which refits only the
    /// subject-level overdispersion against the profile likelihood, and the
    /// ninth takes pure `LN`, keeping stage one's estimates untouched.
    ///
    /// Produced by:
    ///
    /// ```r
    /// extra <- rep(0, 150); extra[c(2, 5, 8, 11, 14, 17)] <- 1
    /// nebula(rbind(cnt, extra), rep(1:3, each = 50),
    ///        pred = cbind(1, grp, cov2), offset = rep(1000, 150),
    ///        model = "NBGMM", method = "LN", verbose = TRUE, ncore = 1)
    /// ```
    #[test]
    fn test_nebula_ln_path() {
        let (counts, _, design, offset) = fixture();
        let subject: Vec<usize> = (0..150).map(|i| i / 50).collect();

        // The ninth gene is expressed in six cells only, which puts its expected
        // count per subject at two and so raises the Laplace order to three.
        let mut padded = counts.clone();
        let mut sparse = vec![0.0; 150];
        for cell in [2, 5, 8, 11, 14, 17] {
            sparse[cell] = 1.0;
        }
        padded.extend_from_slice(&sparse);

        // logFC intercept, grp, cov2, se intercept, grp, cov2, Subject, Cell.
        const EXPECTED: [[f64; 8]; 8] = [
            [
                -3.81659573486463533e+00,
                5.38222100442079743e-01,
                3.83483759002136715e-01,
                4.69063795540145337e-02,
                9.36246418891787113e-02,
                5.35717183358771221e-02,
                1.00000000000000005e-04,
                2.75050598322235385e-01,
            ],
            [
                -5.22829697013958850e+00,
                3.89022934226685158e-01,
                4.16534653064868698e-01,
                5.44653730954031065e-02,
                1.08630486140098639e-01,
                6.46373400145602711e-02,
                1.00000000000000005e-04,
                2.37867608416352794e-01,
            ],
            [
                -2.79147738513702715e+00,
                4.60678103206666578e-01,
                3.98587913222329615e-01,
                4.51226141664423236e-02,
                9.01236604810476266e-02,
                5.17711920842120824e-02,
                1.00000000000000005e-04,
                2.82598066974932505e-01,
            ],
            [
                -5.78931113441487000e+00,
                2.22185044166334139e-01,
                3.07728570357372222e-01,
                5.80432215750652861e-02,
                1.16296244291096931e-01,
                6.90662800246763103e-02,
                1.00000000000000005e-04,
                1.60696729601599086e-01,
            ],
            [
                -3.16815060082354449e+00,
                5.19641589456372732e-01,
                4.12199424890375421e-01,
                4.65591163589440318e-02,
                9.28337943204101818e-02,
                5.24815231974455482e-02,
                1.00000000000000005e-04,
                2.93727040738100298e-01,
            ],
            [
                -4.29958492315647156e+00,
                3.36809984235162951e-01,
                2.86375115431901306e-01,
                4.79413972185084511e-02,
                9.58745337839272821e-02,
                5.72547237128498490e-02,
                1.00000000000000005e-04,
                2.62887984227208105e-01,
            ],
            [
                -2.26664039047022214e+00,
                4.41108571602001120e-01,
                3.46229061253199466e-01,
                4.55915524962416524e-02,
                9.13938987320008084e-02,
                5.34981551013472972e-02,
                1.00000000000000005e-04,
                2.96486842301256492e-01,
            ],
            [
                -4.70367062080705978e+00,
                6.60875227770174489e-01,
                3.84900243156389155e-01,
                5.22429472723280330e-02,
                1.04269433298777997e-01,
                6.19743862314844629e-02,
                1.00000000000000005e-04,
                2.81557437161172819e-01,
            ],
        ];

        let fit = nebula(&padded, 9, 150, &subject, &design, 3, Some(&offset), None)
            .expect("nebula failed");
        assert_eq!(fit.gene_index.len(), 9);

        for (g, want) in EXPECTED.iter().enumerate() {
            for j in 0..3 {
                assert_relative_eq!(
                    fit.coefficients[g * 3 + j],
                    want[j],
                    max_relative = TOL_COEF
                );
                assert_relative_eq!(fit.se[g * 3 + j], want[3 + j], max_relative = TOL_SE);
            }
            assert_relative_eq!(
                fit.subject_overdispersion[g],
                want[6],
                max_relative = TOL_SUBJECT
            );
            assert_relative_eq!(fit.cell_overdispersion[g], want[7], max_relative = TOL_CELL);
        }

        // The ninth gene never leaves stage one, so its fit is whatever the
        // bounded quasi-Newton returned. nebula drives that stage with nlopt's
        // L-BFGS and this crate with the Fortran L-BFGS-B, and on a gene with
        // six counts spread over 150 cells the two are not going to agree on a
        // point where both overdispersions have run to their box constraints.
        // What has to agree is which constraints: the subject-level component
        // at its floor and the cell-level one at its ceiling.
        assert_relative_eq!(fit.subject_overdispersion[8], 1e-4, max_relative = 1e-12);
        assert_relative_eq!(fit.cell_overdispersion[8], 1e-3, max_relative = 1e-12);
        assert!(fit.coefficients[24..27].iter().all(|v| v.is_finite()));
        assert!(fit.se[24..27].iter().all(|v| v.is_finite()));
    }

    /// A gene with no counts at all in one subject.
    ///
    /// The first subject's twenty-five cells are zeroed for gene one, which
    /// leaves that subject's random effect identified only by the prior. Its
    /// subject-level overdispersion jumps from 0.024 to 0.72, so this is a
    /// genuinely different fit rather than a perturbation.
    ///
    /// Produced by the golden call with `cnt[1, 1:25] <- 0`.
    #[test]
    fn test_nebula_subject_with_all_zero_counts() {
        let (mut counts, subject, design, offset) = fixture();
        for slot in counts.iter_mut().take(25) {
            *slot = 0.0;
        }
        assert_eq!(subject[24], subject[0]);
        assert_ne!(subject[25], subject[0]);

        const LOG_FC: [f64; 3] = [
            -4.64175020355696155e+00,
            2.11998305680204924e+00,
            1.63852281831067770e+00,
        ];
        const SE: [f64; 3] = [
            4.24684864629024383e-01,
            1.17140823143936190e+00,
            8.69775089819186276e-01,
        ];
        const SUBJECT: f64 = 7.23281548565090415e-01;
        const CELL: f64 = 2.57684232626537890e-01;

        let fit = nebula(
            &counts,
            8,
            150,
            &subject,
            &design,
            3,
            Some(&offset),
            Some(golden_params()),
        )
        .expect("nebula failed");
        assert_eq!(fit.gene_index.len(), 8);

        for j in 0..3 {
            assert_relative_eq!(fit.coefficients[j], LOG_FC[j], max_relative = TOL_COEF);
            assert_relative_eq!(fit.se[j], SE[j], max_relative = TOL_SE);
        }
        assert_relative_eq!(
            fit.subject_overdispersion[0],
            SUBJECT,
            max_relative = TOL_SUBJECT
        );
        assert_relative_eq!(fit.cell_overdispersion[0], CELL, max_relative = TOL_CELL);
    }

    /// The expression filter, against the genes nebula itself keeps.
    ///
    /// Three genes are appended: one expressed in three cells, and two expressed
    /// in six. `mincp = 5` drops the first and keeps the other two, which is
    /// what `nebula(..., verbose = TRUE)` reports as "Remove 1 genes having low
    /// expression" with `gene_id` 1 to 8, 10 and 11.
    #[test]
    fn test_gene_filter_drops_low_count_gene() {
        let (counts, subject, design, offset) = fixture();
        let mut padded = counts.clone();

        let mut three = vec![0.0; 150];
        for cell in [3, 40, 90] {
            three[cell] = 1.0;
        }
        let mut six_a = vec![0.0; 150];
        for cell in [2, 5, 8, 11, 14, 17] {
            six_a[cell] = 1.0;
        }
        let mut six_b = vec![0.0; 150];
        for slot in six_b.iter_mut().take(6) {
            *slot = 1.0;
        }
        padded.extend_from_slice(&three);
        padded.extend_from_slice(&six_a);
        padded.extend_from_slice(&six_b);

        let fit = nebula(
            &padded,
            11,
            150,
            &subject,
            &design,
            3,
            Some(&offset),
            Some(golden_params()),
        )
        .expect("nebula failed");
        assert_eq!(fit.gene_index, vec![0, 1, 2, 3, 4, 5, 6, 7, 9, 10]);
        assert_eq!(fit.coefficients.len(), 10 * 3);

        // The other half of the filter: `cpc = 10` keeps only the genes whose
        // mean count per cell exceeds ten, `gene_id` 1, 3, 5, 6 and 7 in R.
        let fit = nebula(
            &counts,
            8,
            150,
            &subject,
            &design,
            3,
            Some(&offset),
            Some(NebulaParams {
                cpc: 10.0,
                ..golden_params()
            }),
        )
        .expect("nebula failed");
        assert_eq!(fit.gene_index, vec![0, 2, 4, 5, 6]);
    }

    /// The fit is unchanged by holding the inputs as `f32`.
    ///
    /// Counts, design and offsets all convert exactly here, so the two runs
    /// agree to the last bit rather than merely closely.
    #[test]
    fn test_nebula_is_generic_over_the_float() {
        let (counts, subject, design, offset) = fixture();
        let wide = nebula(
            &counts,
            8,
            150,
            &subject,
            &design,
            3,
            Some(&offset),
            Some(golden_params()),
        )
        .expect("nebula failed");

        let counts32: Vec<f32> = counts.iter().map(|v| *v as f32).collect();
        let design32: Vec<f32> = design.iter().map(|v| *v as f32).collect();
        let offset32: Vec<f32> = offset.iter().map(|v| *v as f32).collect();
        let narrow = nebula(
            &counts32,
            8,
            150,
            &subject,
            &design32,
            3,
            Some(&offset32),
            Some(golden_params()),
        )
        .expect("nebula failed");

        assert_eq!(wide.coefficients, narrow.coefficients);
        assert_eq!(wide.se, narrow.se);
    }

    /// No offset is the same as an offset of ones.
    #[test]
    fn test_nebula_without_an_offset() {
        let (counts, subject, design, _) = fixture();
        let ones = vec![1.0_f64; 150];
        let with = nebula(
            &counts,
            8,
            150,
            &subject,
            &design,
            3,
            Some(&ones),
            Some(golden_params()),
        )
        .expect("nebula failed");
        let without = nebula(
            &counts,
            8,
            150,
            &subject,
            &design,
            3,
            None,
            Some(golden_params()),
        )
        .expect("nebula failed");
        assert_eq!(with.coefficients, without.coefficients);
        assert_eq!(with.subject_overdispersion, without.subject_overdispersion);
    }

    /// NEBULA-LN below thirty cells per subject silently becomes NEBULA-HL.
    ///
    /// The fixture has twenty-five, so asking for either variant has to give the
    /// same answer. This is the override that makes the golden, generated with
    /// `method = "LN"`, actually exercise the HL path.
    #[test]
    fn test_ln_falls_back_to_hl_below_thirty_cells_per_subject() {
        let (counts, subject, design, offset) = fixture();
        let ln = nebula(
            &counts,
            8,
            150,
            &subject,
            &design,
            3,
            Some(&offset),
            Some(golden_params()),
        )
        .expect("nebula failed");
        let hl = nebula(
            &counts,
            8,
            150,
            &subject,
            &design,
            3,
            Some(&offset),
            Some(NebulaParams {
                method: NebulaMethod::Hl,
                ..golden_params()
            }),
        )
        .expect("nebula failed");
        assert_eq!(ln.coefficients, hl.coefficients);
        assert_eq!(ln.se, hl.se);
    }

    /// The Cholesky helpers invert what they should and refuse what they cannot.
    #[test]
    fn test_cholesky_solves_and_inverts() {
        let a = [4.0, 1.0, 0.5, 1.0, 3.0, 0.25, 0.5, 0.25, 2.0];
        let inverse = cholesky_inverse(&a, 3).expect("not positive definite");
        for i in 0..3 {
            for j in 0..3 {
                let mut entry = 0.0;
                for k in 0..3 {
                    entry += a[i * 3 + k] * inverse[k * 3 + j];
                }
                let want = if i == j { 1.0 } else { 0.0 };
                assert_relative_eq!(entry, want, epsilon = 1e-12);
            }
        }
        assert!(cholesky_inverse(&[1.0, 2.0, 2.0, 1.0], 2).is_none());
        assert!(cholesky_solve(&[0.0], 1, &[1.0]).is_none());
    }

    /// Subject boundaries are read off runs, not off sorted order.
    #[test]
    fn test_subject_boundaries() {
        assert_eq!(
            subject_boundaries(&[7, 7, 3, 3, 3, 9]).expect("valid"),
            vec![0, 2, 5, 6]
        );
        let err = subject_boundaries(&[1, 2, 1]).expect_err("split subject");
        assert!(matches!(
            err,
            EdgeErrors::SubjectsNotContiguous { subject: 1 }
        ));
    }

    ////////////////////
    // Error branches //
    ////////////////////

    /// Runs [`nebula`] on the fixture with one input replaced.
    ///
    /// ### Params
    ///
    /// * `counts` - Counts, gene-major
    /// * `n_genes` - Number of genes
    /// * `n_cells` - Number of cells
    /// * `subject` - Subject of each cell
    /// * `design` - Design, row-major
    /// * `n_coef` - Number of design columns
    /// * `offset` - Offsets, or `None`
    /// * `params` - Knobs
    ///
    /// ### Returns
    ///
    /// The error the call produced.
    #[allow(clippy::too_many_arguments)]
    fn expect_error(
        counts: &[f64],
        n_genes: usize,
        n_cells: usize,
        subject: &[usize],
        design: &[f64],
        n_coef: usize,
        offset: Option<&[f64]>,
        params: NebulaParams,
    ) -> EdgeErrors {
        nebula(
            counts,
            n_genes,
            n_cells,
            subject,
            design,
            n_coef,
            offset,
            Some(params),
        )
        .expect_err("expected a rejection")
    }

    /// Every shape and content check on the inputs.
    #[test]
    fn test_rejects_malformed_inputs() {
        let (counts, subject, design, offset) = fixture();
        let p = golden_params();

        assert!(matches!(
            expect_error(&counts, 0, 150, &subject, &design, 3, Some(&offset), p),
            EdgeErrors::EmptyCounts { n_genes: 0, .. }
        ));
        assert!(matches!(
            expect_error(&counts[..8], 8, 1, &subject[..1], &design[..3], 3, None, p),
            EdgeErrors::InvalidArgument(_)
        ));
        assert!(matches!(
            expect_error(&counts, 8, 150, &subject, &design, 0, Some(&offset), p),
            EdgeErrors::MustBePositive(_)
        ));
        assert!(matches!(
            expect_error(
                &counts[..100],
                8,
                150,
                &subject,
                &design,
                3,
                Some(&offset),
                p
            ),
            EdgeErrors::LengthMismatch { name: "counts", .. }
        ));
        assert!(matches!(
            expect_error(
                &counts,
                8,
                150,
                &subject,
                &design[..30],
                3,
                Some(&offset),
                p
            ),
            EdgeErrors::ShapeMismatch { .. }
        ));
        assert!(matches!(
            expect_error(
                &counts,
                8,
                150,
                &subject[..10],
                &design,
                3,
                Some(&offset),
                p
            ),
            EdgeErrors::LengthMismatch {
                name: "subject_id",
                ..
            }
        ));
        assert!(matches!(
            expect_error(
                &counts,
                8,
                150,
                &subject,
                &design,
                3,
                Some(&offset[..10]),
                p
            ),
            EdgeErrors::LengthMismatch { name: "offset", .. }
        ));

        let mut negative = counts.clone();
        negative[17] = -1.0;
        assert!(matches!(
            expect_error(&negative, 8, 150, &subject, &design, 3, Some(&offset), p),
            EdgeErrors::InvalidArgument(_)
        ));

        let mut bad_offset = offset.clone();
        bad_offset[3] = 0.0;
        assert!(matches!(
            expect_error(&counts, 8, 150, &subject, &design, 3, Some(&bad_offset), p),
            EdgeErrors::MustBePositive(_)
        ));
    }

    /// The subject structure the mixed model needs.
    #[test]
    fn test_rejects_bad_subject_structure() {
        let (counts, subject, design, offset) = fixture();
        let p = golden_params();

        let mut split = subject.clone();
        split[149] = subject[0];
        assert!(matches!(
            expect_error(&counts, 8, 150, &split, &design, 3, Some(&offset), p),
            EdgeErrors::SubjectsNotContiguous { .. }
        ));

        let one = vec![1_usize; 150];
        assert!(matches!(
            expect_error(&counts, 8, 150, &one, &design, 3, Some(&offset), p),
            EdgeErrors::TooFewSubjects {
                required: 2,
                got: 1
            }
        ));
    }

    /// The design has to have exactly one constant, non-zero column.
    #[test]
    fn test_rejects_bad_design() {
        let (counts, subject, design, offset) = fixture();
        let p = golden_params();

        // No intercept: drop the constant column.
        let no_intercept: Vec<f64> = (0..150)
            .flat_map(|i| [design[i * 3 + 1], design[i * 3 + 2]])
            .collect();
        assert!(matches!(
            expect_error(
                &counts,
                8,
                150,
                &subject,
                &no_intercept,
                2,
                Some(&offset),
                p
            ),
            EdgeErrors::MissingIntercept
        ));

        // Two constant columns.
        let twice: Vec<f64> = (0..150)
            .flat_map(|i| [1.0, 2.0, design[i * 3 + 1]])
            .collect();
        assert!(matches!(
            expect_error(&counts, 8, 150, &subject, &twice, 3, Some(&offset), p),
            EdgeErrors::InvalidArgument(_)
        ));

        // A column that is identically zero.
        let zeroed: Vec<f64> = (0..150)
            .flat_map(|i| [1.0, 0.0, design[i * 3 + 1]])
            .collect();
        assert!(matches!(
            expect_error(&counts, 8, 150, &subject, &zeroed, 3, Some(&offset), p),
            EdgeErrors::InvalidArgument(_)
        ));
    }

    /// Knobs the optimisers cannot recover from.
    #[test]
    fn test_rejects_bad_parameters() {
        let (counts, subject, design, offset) = fixture();
        let base = golden_params();

        let cases: [(NebulaParams, &str); 4] = [
            (
                NebulaParams {
                    min: (0.0, 1e-4),
                    ..base
                },
                "min",
            ),
            (
                NebulaParams {
                    max: (1e-5, 1000.0),
                    ..base
                },
                "max",
            ),
            (NebulaParams { eps: 0.0, ..base }, "eps"),
            (NebulaParams { mincp: 0, ..base }, "mincp"),
        ];
        for (params, name) in cases {
            let err = expect_error(&counts, 8, 150, &subject, &design, 3, Some(&offset), params);
            assert!(
                matches!(
                    err,
                    EdgeErrors::MustBePositive(_) | EdgeErrors::InvalidBounds { .. }
                ),
                "{name} gave {err}"
            );
        }
    }

    /// Nothing surviving the filter is an error, not an empty fit.
    #[test]
    fn test_rejects_when_no_gene_passes_the_filter() {
        let (counts, subject, design, offset) = fixture();
        let err = expect_error(
            &counts,
            8,
            150,
            &subject,
            &design,
            3,
            Some(&offset),
            NebulaParams {
                cpc: 1e6,
                ..golden_params()
            },
        );
        assert!(matches!(
            err,
            EdgeErrors::NoGenesAfterFiltering { n_genes: 8 }
        ));
    }
}
