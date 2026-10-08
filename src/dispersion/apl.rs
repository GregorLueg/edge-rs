//! Cox-Reid adjusted profile likelihood.
//!
//! Profiling out the coefficients biases the dispersion downwards, badly when
//! the design has many columns relative to the samples. Cox and Reid's
//! adjustment subtracts half the log determinant of the observed information,
//! `-0.5 log|X'WX|`.
//!
//! ### References
//!
//! Cox and Reid, Journal of the Royal Statistical Society B 49(1), 1987
//! McCarthy, Chen and Smyth, Nucleic Acids Research 40(10), 2012

use rayon::prelude::*;

use crate::glm::fit::GLM_FIT_MAX_ITER;
use crate::glm::levenberg::{LevenbergParams, Scratch, initial_coefficients};
use crate::glm::one_group::{OneGroupParams, fit_one_gene as fit_one_group_gene};
use crate::numeric::gamma::ln_gamma;
use crate::prelude::*;
use crate::utils::design::design_as_factor;

////////////
// Consts //
////////////

/// Floor applied to fitted means and working weights.
const MIN_POSITIVE: f64 = 1e-300;

/// Bound on the linear predictor before exponentiating.
const ETA_CLAMP: f64 = 500.0;

/// Floor applied to the information matrix pivots before taking logarithms.
///
/// edgePython's `eps`. A gene whose design is singular (a group with no counts)
/// would otherwise give an infinite adjustment. Matches edgeR's LDL path, which
/// floors the same quantity.
const PIVOT_FLOOR: f64 = 1e-10;

/////////////////
// GeneScratch //
/////////////////

/// Per-thread buffers for the grid sweep.
struct GeneScratch {
    /// Counts for the current gene, widened to `f64`.
    y: Vec<f64>,
    /// Fitted means at the current grid point.
    mu: Vec<f64>,
    /// Information matrix, row-major.
    information: Vec<f64>,
    /// Coefficients, a warm start across dispersions, reset by
    /// [`AplWorkspace::begin_gene`] for each gene.
    beta: Vec<f64>,
    /// Scratch for the general Levenberg path.
    levenberg: Scratch,
}

impl GeneScratch {
    /// Allocates scratch for one worker.
    ///
    /// ### Params
    ///
    /// * `n_samples` - Number of samples
    /// * `n_coef` - Number of coefficients
    ///
    /// ### Returns
    ///
    /// Buffers sized for a single gene across the whole grid.
    fn new(n_samples: usize, n_coef: usize) -> Self {
        Self {
            y: vec![0.0; n_samples],
            mu: vec![0.0; n_samples],
            information: vec![0.0; n_coef * n_coef],
            beta: vec![0.0; n_coef],
            levenberg: Scratch::new(n_samples, n_coef),
        }
    }
}

/////////////
// Helpers //
/////////////

/// Cox-Reid adjustment `-0.5 log|X'WX|` from the assembled information matrix.
///
/// Factorises in place by Cholesky (`X'WX` is positive semi-definite) and reads
/// the determinant off the diagonal. A non-positive pivot means the design is
/// singular for this gene; pivots are floored rather than the gene discarded.
///
/// ### Params
///
/// * `xtwx` - Row-major `n_coef * n_coef` information matrix, overwritten
/// * `n_coef` - Number of coefficients
///
/// ### Returns
///
/// `-0.5 log|X'WX|`.
fn cox_reid_adjustment(xtwx: &mut [f64], n_coef: usize) -> f64 {
    let mut log_det = 0.0;

    for i in 0..n_coef {
        for j in 0..=i {
            let mut sum = xtwx[i * n_coef + j];
            for k in 0..j {
                sum -= xtwx[i * n_coef + k] * xtwx[j * n_coef + k];
            }
            if i == j {
                // The Cholesky pivot squared is the LDL pivot edgeR floors.
                let pivot = if sum > PIVOT_FLOOR { sum } else { PIVOT_FLOOR };
                log_det += pivot.ln();
                xtwx[i * n_coef + i] = pivot.sqrt();
            } else {
                xtwx[i * n_coef + j] = sum / xtwx[j * n_coef + j];
            }
        }
    }

    -0.5 * log_det
}

/// Negative binomial log-likelihood of one gene at a given dispersion.
///
/// Falls back to the Poisson likelihood at zero dispersion, where `1/phi` is
/// not defined.
///
/// ### Params
///
/// * `y` - Counts for this gene
/// * `mu` - Fitted means
/// * `dispersion` - Dispersion for this gene
/// * `weights` - Optional weight row
///
/// ### Returns
///
/// The weighted log-likelihood summed across samples.
fn log_likelihood(
    y: &[f64],
    mu: &[f64],
    dispersion: f64,
    weights: Option<RecycledRow<'_, f64>>,
) -> f64 {
    let mut total = 0.0;

    if dispersion > 0.0 {
        let r = 1.0 / dispersion;
        let ln_gamma_r = ln_gamma(r);
        let r_ln_r = r * r.ln();
        for (j, (&y_j, &mu_j)) in y.iter().zip(mu.iter()).enumerate() {
            let mu_j = mu_j.max(MIN_POSITIVE);
            let term =
                ln_gamma(y_j + r) - ln_gamma_r - ln_gamma(y_j + 1.0) + r_ln_r + y_j * mu_j.ln()
                    - (r + y_j) * (r + mu_j).ln();
            total += weights.map_or(term, |w| w.get(j) * term);
        }
    } else {
        for (j, (&y_j, &mu_j)) in y.iter().zip(mu.iter()).enumerate() {
            let mu_j = mu_j.max(MIN_POSITIVE);
            let term = y_j * mu_j.ln() - mu_j - ln_gamma(y_j + 1.0);
            total += weights.map_or(term, |w| w.get(j) * term);
        }
    }

    total
}

/// Assembles `X'WX` for one gene from its fitted means.
///
/// The working weight is `w mu / (1 + phi mu)`; `phi = 0` gives the Poisson `w mu`.
///
/// ### Params
///
/// * `mu` - Fitted means
/// * `design` - Row-major design
/// * `n_coef` - Number of coefficients
/// * `dispersion` - Dispersion for this gene
/// * `weights` - Optional weight row
/// * `out` - Row-major `n_coef * n_coef` destination, overwritten
fn assemble_information(
    mu: &[f64],
    design: &[f64],
    n_coef: usize,
    dispersion: f64,
    weights: Option<RecycledRow<'_, f64>>,
    out: &mut [f64],
) {
    out.fill(0.0);

    for (j, &mu_j) in mu.iter().enumerate() {
        let mu_j = mu_j.max(MIN_POSITIVE);
        let w = weights.map_or(1.0, |w| w.get(j));
        let working = (w * mu_j / (1.0 + dispersion * mu_j)).max(MIN_POSITIVE);
        let row = &design[j * n_coef..(j + 1) * n_coef];

        for (a, &x_a) in row.iter().enumerate() {
            let wx = working * x_a;
            for (b, &x_b) in row[..=a].iter().enumerate() {
                out[a * n_coef + b] += wx * x_b;
            }
        }
    }

    // The Cholesky reads the lower triangle only, so the upper is left alone.
}

/// Fits one gene on the one-way path and writes its fitted means.
///
/// ### Params
///
/// * `scratch` - Per-thread buffers
/// * `members` - Sample indices per group
/// * `dispersion` - Dispersion row
/// * `offset` - Offset row
/// * `weights` - Optional weight row
/// * `n_samples` - Number of samples
fn fit_one_way_gene(
    scratch: &mut GeneScratch,
    members: &[Vec<usize>],
    dispersion: RecycledRow<'_, f64>,
    offset: RecycledRow<'_, f64>,
    weights: Option<RecycledRow<'_, f64>>,
    n_samples: usize,
) {
    let params = OneGroupParams::default();

    for group_members in members.iter() {
        let start =
            crate::glm::one_group::initial_coefficient(&scratch.y, group_members, offset, weights);
        let coef = fit_one_group_gene(
            &scratch.y,
            group_members,
            dispersion,
            offset,
            weights,
            start,
            &params,
        );
        for &sample in group_members {
            let eta = (coef + offset.get(sample)).clamp(-ETA_CLAMP, ETA_CLAMP);
            scratch.mu[sample] = eta.exp().max(MIN_POSITIVE);
        }
    }
    let _ = n_samples;
}

/// Fits one gene on the general path and writes its fitted means.
///
/// `scratch.beta` is the starting point and is overwritten with the answer, so
/// consecutive grid points warm-start each other. The caller must reset it for
/// each new gene; see [`apl_grid`].
///
/// ### Params
///
/// * `scratch` - Per-thread buffers. `beta` is in and out, `mu` is written
/// * `design` - Row-major design
/// * `n_samples` - Number of samples
/// * `n_coef` - Number of coefficients
/// * `dispersion` - Dispersion row
/// * `offset` - Offset row
/// * `weights` - Optional weight row
fn fit_general_gene(
    scratch: &mut GeneScratch,
    design: &[f64],
    n_samples: usize,
    n_coef: usize,
    dispersion: RecycledRow<'_, f64>,
    offset: RecycledRow<'_, f64>,
    weights: Option<RecycledRow<'_, f64>>,
) {
    scratch.levenberg.y.copy_from_slice(&scratch.y);

    // Starts from the previous grid point's answer. The budget is `glmFit`'s, the
    // route `adjustedProfileLik` takes to the fitter.
    let params = LevenbergParams {
        max_iter: GLM_FIT_MAX_ITER,
        ..Default::default()
    };
    crate::glm::levenberg::fit_one_gene(
        &mut scratch.levenberg,
        &mut scratch.beta,
        design,
        n_samples,
        n_coef,
        dispersion,
        offset,
        weights,
        &params,
    );
    scratch.mu.copy_from_slice(&scratch.levenberg.mu);
}

//////////////////
// AplWorkspace //
//////////////////

/// Reusable state for evaluating the adjusted profile likelihood of one gene at
/// many dispersions.
///
/// [`apl_grid`] warm-starts each grid point from the previous one. A caller
/// running its own search over the dispersion cannot get that through
/// [`apl_at`], which allocates, re-derives the factor structure and cold-starts
/// on every call.
///
/// Build once per design, call [`AplWorkspace::begin_gene`] when the gene
/// changes, then [`AplWorkspace::eval`] per dispersion. Buffers are never
/// reallocated: use one workspace per worker thread.
pub struct AplWorkspace<'a> {
    /// Row-major design, `n_samples * n_coef`.
    design: &'a [f64],
    /// Number of samples.
    n_samples: usize,
    /// Number of coefficients.
    n_coef: usize,
    /// Sample indices per group, used only on the one-way path.
    members: Vec<Vec<usize>>,
    /// Whether the design is a one-way layout, which has the closed-form fit.
    one_way: bool,
    /// Counts, fitted means, information matrix and coefficients.
    scratch: GeneScratch,
    /// Offset row of the gene currently loaded.
    offset: RecycledRow<'a, f64>,
    /// Weight row of the gene currently loaded.
    weights: Option<RecycledRow<'a, f64>>,
    /// Whether a gene has been loaded since construction.
    started: bool,
}

impl<'a> AplWorkspace<'a> {
    /// Allocates a workspace for one design.
    ///
    /// The design's factor structure is derived once here.
    ///
    /// ### Params
    ///
    /// * `design` - Design matrix, row-major `n_samples * n_coef`
    /// * `n_samples` - Number of samples
    /// * `n_coef` - Number of coefficients
    ///
    /// ### Returns
    ///
    /// The workspace, or [`EdgeErrors`] if the shapes disagree.
    pub fn new(design: &'a [f64], n_samples: usize, n_coef: usize) -> Result<Self, EdgeErrors> {
        if n_samples == 0 {
            return Err(EdgeErrors::EmptyCounts {
                n_genes: 1,
                n_samples,
            });
        }
        if n_coef == 0 {
            return Err(EdgeErrors::MustBePositive("n_coef".to_string()));
        }
        if design.len() != n_samples * n_coef {
            return Err(EdgeErrors::LengthMismatch {
                name: "design",
                expected: n_samples * n_coef,
                got: design.len(),
            });
        }

        let (labels, n_groups) = design_as_factor(design, n_samples, n_coef)?;

        let mut members: Vec<Vec<usize>> = vec![Vec::new(); n_groups];
        for (sample, &label) in labels.iter().enumerate() {
            members[label].push(sample);
        }

        Ok(Self::from_parts(
            design,
            n_samples,
            n_coef,
            members,
            n_groups == n_coef,
        ))
    }

    /// Builds a workspace from a design whose factor structure is already known.
    ///
    /// [`apl_grid`] derives the structure once and hands each worker a workspace.
    ///
    /// ### Params
    ///
    /// * `design` - Design matrix, row-major `n_samples * n_coef`
    /// * `n_samples` - Number of samples
    /// * `n_coef` - Number of coefficients
    /// * `members` - Sample indices per group of the design read as a factor
    /// * `one_way` - Whether that factor has exactly `n_coef` groups, the case
    ///   the closed-form one-way fit applies to
    ///
    /// ### Returns
    ///
    /// The workspace, with no gene loaded.
    fn from_parts(
        design: &'a [f64],
        n_samples: usize,
        n_coef: usize,
        members: Vec<Vec<usize>>,
        one_way: bool,
    ) -> Self {
        Self {
            design,
            n_samples,
            n_coef,
            members,
            one_way,
            scratch: GeneScratch::new(n_samples, n_coef),
            offset: RecycledRow::Constant(0.0),
            weights: None,
            started: false,
        }
    }

    /// Loads a gene and cold-starts its coefficients.
    ///
    /// Coefficients carry over between [`AplWorkspace::eval`] calls, so a new gene
    /// must clear them: the previous gene's answer can be far enough out to land
    /// on the wrong optimum. Matches `estimateDisp`, which clears its warm start
    /// per gene.
    ///
    /// ### Params
    ///
    /// * `counts` - This gene's counts, one per sample
    /// * `offset` - Log-scale offset row for this gene
    /// * `weights` - Optional observation weight row for this gene
    /// * `dispersion` - Dispersion the search will evaluate first, used only to
    ///   build the starting coefficients
    ///
    /// ### Returns
    ///
    /// Nothing, or [`EdgeErrors`] if `counts` is the wrong length or the
    /// dispersion is negative.
    pub fn begin_gene<T: EdgeFloat>(
        &mut self,
        counts: &[T],
        offset: RecycledRow<'a, f64>,
        weights: Option<RecycledRow<'a, f64>>,
        dispersion: f64,
    ) -> Result<(), EdgeErrors> {
        if counts.len() != self.n_samples {
            return Err(EdgeErrors::LengthMismatch {
                name: "counts",
                expected: self.n_samples,
                got: counts.len(),
            });
        }
        if dispersion < 0.0 || !dispersion.is_finite() {
            return Err(EdgeErrors::InvalidDispersion(dispersion));
        }

        for (slot, value) in self.scratch.y.iter_mut().zip(counts) {
            *slot = value.to_f64().unwrap_or(0.0);
        }
        self.offset = offset;
        self.weights = weights;
        self.started = true;

        if !self.one_way {
            initial_coefficients(
                &self.scratch.y,
                self.design,
                self.offset,
                RecycledRow::Constant(dispersion),
                self.weights,
                self.n_samples,
                self.n_coef,
                LevenbergParams::default().start_method,
                &mut self.scratch.beta,
            );
        }

        Ok(())
    }

    /// Adjusted profile likelihood of the loaded gene at one dispersion.
    ///
    /// ### Params
    ///
    /// * `dispersion` - Dispersion to evaluate at
    ///
    /// ### Returns
    ///
    /// The adjusted profile log-likelihood, or [`EdgeErrors`] if no gene has
    /// been loaded or the dispersion is negative.
    pub fn eval(&mut self, dispersion: f64) -> Result<f64, EdgeErrors> {
        if !self.started {
            return Err(EdgeErrors::AplWorkspaceNotStarted);
        }
        if dispersion < 0.0 || !dispersion.is_finite() {
            return Err(EdgeErrors::InvalidDispersion(dispersion));
        }
        Ok(self.eval_unchecked(dispersion))
    }

    /// Coefficients of the loaded gene at the dispersion last evaluated.
    ///
    /// ### Returns
    ///
    /// The coefficients on the natural-log scale, one per design column. All
    /// zeros before the first [`AplWorkspace::eval`].
    pub fn coefficients(&self) -> &[f64] {
        &self.scratch.beta
    }

    /// The evaluation itself, with the checks already done.
    ///
    /// ### Params
    ///
    /// * `dispersion` - Dispersion to evaluate at
    ///
    /// ### Returns
    ///
    /// The adjusted profile log-likelihood.
    fn eval_unchecked(&mut self, dispersion: f64) -> f64 {
        let disp_row = RecycledRow::Constant(dispersion);

        if self.one_way {
            fit_one_way_gene(
                &mut self.scratch,
                &self.members,
                disp_row,
                self.offset,
                self.weights,
                self.n_samples,
            );
        } else {
            fit_general_gene(
                &mut self.scratch,
                self.design,
                self.n_samples,
                self.n_coef,
                disp_row,
                self.offset,
                self.weights,
            );
        }

        let ll = log_likelihood(&self.scratch.y, &self.scratch.mu, dispersion, self.weights);
        assemble_information(
            &self.scratch.mu,
            self.design,
            self.n_coef,
            dispersion,
            self.weights,
            &mut self.scratch.information,
        );

        ll + cox_reid_adjustment(&mut self.scratch.information, self.n_coef)
    }
}

///////////////
// Front-end //
///////////////

/// Adjusted profile likelihood across a grid of dispersions.
///
/// Coefficients are refitted at every grid point. The result feeds
/// [`crate::numeric::interpolate::maximize_interpolant`], which splines through
/// the grid.
///
/// Each gene is cold-started, so a gene's row does not depend on the batch or on
/// how rayon splits the work.
///
/// ### Params
///
/// * `counts` - Counts, row-major `n_genes * n_samples`
/// * `n_genes` - Number of genes
/// * `n_samples` - Number of samples
/// * `design` - Design matrix, row-major `n_samples * n_coef`
/// * `n_coef` - Number of coefficients
/// * `grid` - Dispersions to evaluate, ascending
/// * `offset` - Log-scale offsets, recycled over genes and samples
/// * `weights` - Optional observation weights
///
/// ### Returns
///
/// A row-major `n_genes * grid.len()` matrix of adjusted profile
/// log-likelihoods, or [`EdgeErrors`] if the shapes disagree or a grid
/// dispersion is negative.
#[allow(clippy::too_many_arguments)]
pub fn apl_grid<T: EdgeFloat>(
    counts: &[T],
    n_genes: usize,
    n_samples: usize,
    design: &[f64],
    n_coef: usize,
    grid: &[f64],
    offset: &Recycled<f64>,
    weights: Option<&Recycled<f64>>,
) -> Result<Vec<f64>, EdgeErrors> {
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
    if grid.is_empty() {
        return Err(EdgeErrors::MustBePositive("grid.len()".to_string()));
    }
    if let Some(&bad) = grid.iter().find(|d| **d < 0.0 || !d.is_finite()) {
        return Err(EdgeErrors::InvalidDispersion(bad));
    }
    offset.validate(n_genes, n_samples)?;
    if let Some(w) = weights {
        w.validate(n_genes, n_samples)?;
    }

    let n_grid = grid.len();
    let (labels, n_groups) = design_as_factor(design, n_samples, n_coef)?;

    // Sample indices per group, one-way path only.
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); n_groups];
    for (sample, &label) in labels.iter().enumerate() {
        members[label].push(sample);
    }

    let mut out = vec![0.0; n_genes * n_grid];

    out.par_chunks_mut(n_grid).enumerate().for_each_init(
        || {
            AplWorkspace::from_parts(
                design,
                n_samples,
                n_coef,
                members.clone(),
                n_groups == n_coef,
            )
        },
        |workspace, (gene, row)| {
            let start = gene * n_samples;
            // Shapes and grid were validated above.
            workspace
                .begin_gene(
                    &counts[start..start + n_samples],
                    offset.row(gene, n_samples),
                    weights.map(|w| w.row(gene, n_samples)),
                    grid[0],
                )
                .expect("apl_grid validated its shapes and its grid before dispatching");

            for (g, &dispersion) in grid.iter().enumerate() {
                row[g] = workspace.eval_unchecked(dispersion);
            }
        },
    );

    Ok(out)
}

/// Adjusted profile likelihood at one shared dispersion.
///
/// The shape the common-dispersion estimators need. Per-gene dispersions belong
/// in [`apl_grid`].
///
/// ### Params
///
/// * `counts` - Counts, row-major `n_genes * n_samples`
/// * `n_genes` - Number of genes
/// * `n_samples` - Number of samples
/// * `design` - Design matrix, row-major `n_samples * n_coef`
/// * `n_coef` - Number of coefficients
/// * `dispersion` - The shared dispersion
/// * `offset` - Log-scale offsets
/// * `weights` - Optional observation weights
///
/// ### Returns
///
/// One adjusted profile log-likelihood per gene.
#[allow(clippy::too_many_arguments)]
pub fn apl_at<T: EdgeFloat>(
    counts: &[T],
    n_genes: usize,
    n_samples: usize,
    design: &[f64],
    n_coef: usize,
    dispersion: f64,
    offset: &Recycled<f64>,
    weights: Option<&Recycled<f64>>,
) -> Result<Vec<f64>, EdgeErrors> {
    apl_grid(
        counts,
        n_genes,
        n_samples,
        design,
        n_coef,
        &[dispersion],
        offset,
        weights,
    )
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Three genes, two groups of three.
    fn fixture() -> (Vec<f64>, Vec<f64>, Recycled<f64>) {
        let counts = vec![
            10.0, 12.0, 11.0, 40.0, 44.0, 38.0, //
            50.0, 48.0, 52.0, 49.0, 51.0, 50.0, //
            2.0, 0.0, 5.0, 1.0, 3.0, 0.0, //
        ];
        let design = vec![
            1.0, 0.0, //
            1.0, 0.0, //
            1.0, 0.0, //
            1.0, 1.0, //
            1.0, 1.0, //
            1.0, 1.0, //
        ];
        let libraries: [f64; 6] = [1e6, 1.2e6, 0.9e6, 1.1e6, 1e6, 1.3e6];
        let offset = Recycled::by_sample(libraries.iter().map(|v| v.ln()).collect());
        (counts, design, offset)
    }

    /// Parity with edgeR 4.8.2:
    /// ```r
    /// y <- matrix(c(10,12,11,40,44,38, 50,48,52,49,51,50, 2,0,5,1,3,0),
    ///             nrow = 3, byrow = TRUE)
    /// X <- cbind(1, c(0,0,0,1,1,1))
    /// lib <- c(1e6,1.2e6,0.9e6,1.1e6,1e6,1.3e6)
    /// off <- matrix(rep(log(lib), each = 3), nrow = 3)
    /// adjustedProfileLik(0.1, y, X, off)
    /// ```
    #[test]
    fn test_matches_edger_adjusted_profile_lik() {
        let (counts, design, offset) = fixture();
        let apl = apl_grid(&counts, 3, 6, &design, 2, &[0.1], &offset, None).unwrap();

        let expected = [
            -21.640_543_417_082_1,
            -26.342_699_379_916_2,
            -13.436_451_337_328_7,
        ];
        for (got, want) in apl.iter().zip(expected.iter()) {
            assert_relative_eq!(got, want, max_relative = 1e-8);
        }
    }

    /// The grid must agree with evaluating each dispersion on its own, which
    /// pins the warm-start reuse.
    #[test]
    fn test_grid_agrees_with_pointwise_evaluation() {
        let (counts, design, offset) = fixture();
        let grid = [0.01, 0.05, 0.1, 0.3, 1.0];
        let batched = apl_grid(&counts, 3, 6, &design, 2, &grid, &offset, None).unwrap();

        for (g, &d) in grid.iter().enumerate() {
            let single = apl_grid(&counts, 3, 6, &design, 2, &[d], &offset, None).unwrap();
            for gene in 0..3 {
                assert_relative_eq!(
                    batched[gene * grid.len() + g],
                    single[gene],
                    max_relative = 1e-8
                );
            }
        }
    }

    /// The maximising grid index per gene must match edgeR.
    ///
    /// ```r
    /// grid <- 1e-4 * 2^(0:24)
    /// apl <- sapply(grid, function(d) adjustedProfileLik(d, y, X, off))
    /// apply(apl, 1, which.max) - 1   # 6, 6, 14
    /// ```
    ///
    /// Genes 0 and 1 are underdispersed relative to Poisson and tie low. Gene 2
    /// (counts 2, 0, 5 against 1, 3, 0) peaks eight grid points higher. A wrong
    /// Cox-Reid adjustment moves these.
    #[test]
    fn test_grid_maximisers_match_edger() {
        let (counts, design, offset) = fixture();
        let grid: Vec<f64> = (0..25).map(|i| 1e-4 * 2.0_f64.powi(i)).collect();
        let apl = apl_grid(&counts, 3, 6, &design, 2, &grid, &offset, None).unwrap();

        let argmax = |gene: usize| -> usize {
            apl[gene * grid.len()..(gene + 1) * grid.len()]
                .iter()
                .enumerate()
                .fold((0, f64::NEG_INFINITY), |(bi, bv), (i, &v)| {
                    if v > bv { (i, v) } else { (bi, bv) }
                })
                .0
        };

        assert_eq!([argmax(0), argmax(1), argmax(2)], [6, 6, 14]);
    }

    /// Two codings of the same one-way design must agree.
    #[test]
    fn test_general_path_agrees_with_the_one_way_path() {
        let (counts, _, offset) = fixture();
        let indicator = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let one_way = apl_grid(&counts, 3, 6, &indicator, 2, &[0.1], &offset, None).unwrap();

        // Treatment coding of the same model: still one-way, different rotation.
        let treatment = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let rotated = apl_grid(&counts, 3, 6, &treatment, 2, &[0.1], &offset, None).unwrap();

        // Invariant to reparametrisation.
        for (a, b) in one_way.iter().zip(rotated.iter()) {
            assert_relative_eq!(a, b, max_relative = 1e-9);
        }
    }

    #[test]
    fn test_cox_reid_adjustment_matches_a_known_determinant() {
        // [[4, 2], [2, 3]] has determinant 8, so the adjustment is -0.5 ln 8.
        let mut m = vec![4.0, 2.0, 2.0, 3.0];
        let got = cox_reid_adjustment(&mut m, 2);
        assert_relative_eq!(got, -0.5 * 8.0_f64.ln(), max_relative = 1e-12);
    }

    #[test]
    fn test_cox_reid_adjustment_floors_a_singular_matrix() {
        let mut m = vec![0.0, 0.0, 0.0, 0.0];
        let got = cox_reid_adjustment(&mut m, 2);
        assert!(got.is_finite());
        assert_relative_eq!(got, -0.5 * 2.0 * PIVOT_FLOOR.ln(), max_relative = 1e-12);
    }

    #[test]
    fn test_rejects_a_negative_dispersion() {
        let (counts, design, offset) = fixture();
        let err = apl_grid(&counts, 3, 6, &design, 2, &[-0.1], &offset, None).unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidDispersion(_)));
    }

    #[test]
    fn test_rejects_an_empty_grid() {
        let (counts, design, offset) = fixture();
        let err = apl_grid(&counts, 3, 6, &design, 2, &[], &offset, None).unwrap_err();
        assert!(matches!(err, EdgeErrors::MustBePositive(_)));
    }

    #[test]
    fn test_rejects_a_shape_mismatch() {
        let (counts, design, offset) = fixture();
        let err = apl_grid(&counts, 4, 6, &design, 2, &[0.1], &offset, None).unwrap_err();
        assert!(matches!(err, EdgeErrors::LengthMismatch { .. }));
    }

    /// The workspace must walk a grid to the values `apl_grid` reports.
    #[test]
    fn test_workspace_reproduces_the_grid() {
        let (counts, design, offset) = fixture();
        let grid = [0.01, 0.05, 0.1, 0.3, 1.0];
        let batched = apl_grid(&counts, 3, 6, &design, 2, &grid, &offset, None).unwrap();

        let mut workspace = AplWorkspace::new(&design, 6, 2).unwrap();
        for gene in 0..3 {
            workspace
                .begin_gene(
                    &counts[gene * 6..(gene + 1) * 6],
                    offset.row(gene, 6),
                    None,
                    grid[0],
                )
                .unwrap();
            for (g, &d) in grid.iter().enumerate() {
                let got = workspace.eval(d).unwrap();
                assert_relative_eq!(got, batched[gene * grid.len() + g], max_relative = 1e-12);
            }
        }
    }

    /// The one-way path has no coefficients to carry; the workspace must agree there too.
    #[test]
    fn test_workspace_reproduces_the_grid_on_the_one_way_path() {
        let (counts, design, offset) = fixture();
        let grid = [0.05, 0.2];
        let batched = apl_grid(&counts, 3, 6, &design, 2, &grid, &offset, None).unwrap();

        let mut workspace = AplWorkspace::new(&design, 6, 2).unwrap();
        assert!(workspace.one_way);
        for gene in 0..3 {
            workspace
                .begin_gene(
                    &counts[gene * 6..(gene + 1) * 6],
                    offset.row(gene, 6),
                    None,
                    grid[0],
                )
                .unwrap();
            for (g, &d) in grid.iter().enumerate() {
                assert_relative_eq!(
                    workspace.eval(d).unwrap(),
                    batched[gene * grid.len() + g],
                    max_relative = 1e-12
                );
            }
        }
    }

    /// Sums one gene's adjusted profile likelihood across a grid, driving the
    /// workspace as an optimiser would.
    fn walk<'a>(
        workspace: &mut AplWorkspace<'a>,
        counts: &'a [f64],
        offset: &'a Recycled<f64>,
        grid: &[f64],
        gene: usize,
    ) -> f64 {
        workspace
            .begin_gene(
                &counts[gene * 6..(gene + 1) * 6],
                offset.row(gene, 6),
                None,
                grid[0],
            )
            .unwrap();
        grid.iter().map(|&d| workspace.eval(d).unwrap()).sum()
    }

    /// A gene's fit must not depend on what the workspace saw before it.
    #[test]
    fn test_workspace_cold_starts_each_gene() {
        let (counts, _, offset) = fixture();
        // A continuous column forces the general path, which has the warm start.
        let continuous: Vec<f64> = (0..6).flat_map(|s| [1.0, (s as f64) * 0.37]).collect();
        let grid = [0.02, 0.15];

        let mut forwards = AplWorkspace::new(&continuous, 6, 2).unwrap();
        let mut backwards = AplWorkspace::new(&continuous, 6, 2).unwrap();

        // Gene 2 reached through the other two, and on its own.
        let _ = walk(&mut forwards, &counts, &offset, &grid, 0);
        let _ = walk(&mut forwards, &counts, &offset, &grid, 1);
        let through = walk(&mut forwards, &counts, &offset, &grid, 2);
        let alone = walk(&mut backwards, &counts, &offset, &grid, 2);

        assert_relative_eq!(through, alone, max_relative = 1e-12);
    }

    #[test]
    fn test_workspace_rejects_an_evaluation_before_a_gene() {
        let (_, design, _) = fixture();
        let mut workspace = AplWorkspace::new(&design, 6, 2).unwrap();
        assert!(matches!(
            workspace.eval(0.1).unwrap_err(),
            EdgeErrors::AplWorkspaceNotStarted
        ));
    }
}
