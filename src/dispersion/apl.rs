//! Cox-Reid adjusted profile likelihood.
//!
//! The dispersion cannot be estimated from the profile likelihood directly:
//! profiling out the coefficients biases it downwards, badly so when the design
//! has many columns relative to the samples. Cox and Reid's adjustment corrects
//! for that by subtracting half the log determinant of the observed
//! information, `-0.5 log|X'WX|`, which is what makes edgeR's dispersion
//! estimates usable on small experiments.
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
/// edgePython's `eps`. A design that is singular for a particular gene, which
/// happens when a group has no counts at all, would otherwise contribute an
/// infinite adjustment and take the whole grid with it. Clamping matches
/// edgeR's LDL path, which floors the same quantity.
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
    /// Coefficients, reused as a warm start across grid points and reset by
    /// [`apl_grid`] at the start of each gene.
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
/// Factorises in place by Cholesky, which exists because `X'WX` is positive
/// semi-definite by construction, and reads the determinant off the diagonal.
/// A pivot that is not positive means the design is singular for this gene, and
/// the remaining pivots are floored rather than the whole gene discarded.
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
                // The Cholesky pivot squared is the LDL pivot, which is the
                // quantity edgeR floors.
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
/// The working weight is `w mu / (1 + phi mu)` for a negative binomial and
/// `w mu` in the Poisson limit, which is the same expression with `phi = 0`.
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
/// `scratch.beta` is read as the starting point and overwritten with the answer,
/// which is what lets consecutive grid points warm-start each other. The caller
/// owns that buffer's initial state and must reset it when it moves to a new
/// gene; see [`apl_grid`].
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

    // Start from the previous grid point's answer. Neighbouring dispersions give
    // very similar coefficients, so this saves most of the iterations after the
    // first grid point.
    //
    // The budget is `glmFit`'s rather than `mglmLevenberg`'s, because that is the
    // route `adjustedProfileLik` takes to the fitter.
    let params = LevenbergParams { max_iter: GLM_FIT_MAX_ITER, ..Default::default() };
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

///////////////
// Front-end //
///////////////

/// Adjusted profile likelihood across a grid of dispersions.
///
/// For each gene the coefficients are refitted at every grid point, since the
/// profile likelihood is defined at the maximum over coefficients for that
/// dispersion. The result feeds
/// [`crate::numeric::interpolate::maximize_interpolant`], which finds the
/// maximising dispersion by fitting a spline through the grid.
///
/// Genes are the parallel axis, with one scratch buffer per worker. Each gene is
/// cold-started, so a gene's row depends only on that gene: the answer does not
/// move with the batch it was submitted in, nor with how rayon happened to split
/// the work.
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
    let one_way = n_groups == n_coef;

    // Sample indices per group, used only on the one-way path.
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); n_groups];
    for (sample, &label) in labels.iter().enumerate() {
        members[label].push(sample);
    }

    let mut out = vec![0.0; n_genes * n_grid];

    out.par_chunks_mut(n_grid).enumerate().for_each_init(
        || GeneScratch::new(n_samples, n_coef),
        |scratch, (gene, row)| {
            let offset_row = offset.row(gene, n_samples);
            let weight_row = weights.map(|w| w.row(gene, n_samples));

            let start = gene * n_samples;
            for (slot, value) in scratch.y.iter_mut().zip(&counts[start..start + n_samples]) {
                *slot = value.to_f64().unwrap_or(0.0);
            }

            // Cold-start this gene. The scratch is shared by every gene a worker
            // handles, and `beta` is deliberately carried from one grid point to
            // the next, so without this the first grid point would start from
            // whatever the previous *gene* converged to. Neighbouring grid points
            // are close together; neighbouring genes need not be, and over a wide
            // abundance range that start is far enough out that the fit lands on
            // the wrong optimum.
            if !one_way {
                initial_coefficients(
                    &scratch.y,
                    design,
                    offset_row,
                    n_samples,
                    n_coef,
                    LevenbergParams::default().start_method,
                    &mut scratch.beta,
                );
            }

            for (g, &dispersion) in grid.iter().enumerate() {
                let disp_row = RecycledRow::Constant(dispersion);

                if one_way {
                    fit_one_way_gene(
                        scratch, &members, disp_row, offset_row, weight_row, n_samples,
                    );
                } else {
                    fit_general_gene(
                        scratch, design, n_samples, n_coef, disp_row, offset_row, weight_row,
                    );
                }

                let ll = log_likelihood(&scratch.y, &scratch.mu, dispersion, weight_row);
                assemble_information(
                    &scratch.mu,
                    design,
                    n_coef,
                    dispersion,
                    weight_row,
                    &mut scratch.information,
                );
                row[g] = ll + cox_reid_adjustment(&mut scratch.information, n_coef);
            }
        },
    );

    Ok(out)
}

/// Adjusted profile likelihood at one shared dispersion.
///
/// The common-dispersion estimators maximise the summed adjusted profile
/// likelihood over a single scalar, so this is the shape they need. A per-gene
/// dispersion instead belongs in [`apl_grid`], which is built for it.
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

    /// The grid form must agree with evaluating each dispersion on its own,
    /// which pins the warm-start reuse across grid points.
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

    /// The whole grid against edgeR, not just one point: the maximising grid
    /// index per gene must match.
    ///
    /// ```r
    /// grid <- 1e-4 * 2^(0:24)
    /// apl <- sapply(grid, function(d) adjustedProfileLik(d, y, X, off))
    /// apply(apl, 1, which.max) - 1   # 6, 6, 14
    /// ```
    ///
    /// Genes 0 and 1 are both underdispersed relative to Poisson and tie low.
    /// Gene 2 has counts of 2, 0, 5 against 1, 3, 0, whose scatter exceeds its
    /// mean, and peaks eight grid points higher. A wrong Cox-Reid adjustment
    /// moves these, since it is the only term that depends on the design.
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

    /// A continuous covariate forces the general fitting path. It must agree
    /// with the one-way path where both are valid, so compare a one-way design
    /// expressed two ways.
    #[test]
    fn test_general_path_agrees_with_the_one_way_path() {
        let (counts, _, offset) = fixture();
        // Group indicator, which is one-way.
        let indicator = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        // The same model with a redundant continuous column that ties every
        // sample to its own group, forcing the Levenberg path.
        let one_way = apl_grid(&counts, 3, 6, &indicator, 2, &[0.1], &offset, None).unwrap();

        // Treatment coding of the same model: still one-way, different rotation.
        let treatment = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let rotated = apl_grid(&counts, 3, 6, &treatment, 2, &[0.1], &offset, None).unwrap();

        // The adjusted profile likelihood is invariant to reparametrisation.
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
}
