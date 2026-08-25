//! Adjusted deviances, degrees of freedom and the average quasi-dispersion.
//!
//! The second half of edgeR's quasi-likelihood machinery, `ql_glm.c`. Where
//! [`crate::ql::chebyshev`] supplies the two moments of a single unit deviance,
//! this module spends them: for each gene it rescales every observation's unit
//! deviance by `alpha`, weights the complementary leverage by `kappa`, and sums
//! both across samples. The ratio of the two is `s2`, the gene's
//! quasi-likelihood dispersion, which `glmQLFit` then hands to `squeezeVar`.
//!
//! ### Layout
//!
//! Genes are the parallel axis. Each gene needs its own weighted design matrix
//! and its own QR, so the fan-out carries a per-thread scratch buffer for both
//! and nothing larger than one gene's design is allocated inside the loop.
//!
//! ### Deviations from edgeR
//!
//! Two of its own, both documented at their use sites: the smoother behind
//! [`compute_prior`] is limma's `weightedLowess` rather than R's `lowess`, and
//! the leverages come from an unpivoted QR rather than LINPACK's rank-revealing
//! `dqrdc2`. See the constants and tests below for what each costs.
//!
//! One inherited. edgeR's `compute_weight` has no Poisson branch: a dispersion
//! of exactly zero falls into the case-1 negative binomial fit, which already
//! carries the Poisson moments as a factor. [`compute_weight`] here diverts
//! `phi <= 0` to the bare Poisson fit, and the two disagree by 4.7e-4 relative
//! on the fixture in `test_near_poisson_dispersion_matches_edger`. Nothing in
//! this module can correct that; it is noted here because the error surfaces in
//! these outputs.
//!
//! ### References
//!
//! Lund, Nettleton, McCarthy and Smyth, SAGMB 11(5), 2012
//! Chen, Lun and Smyth, F1000Research 5:1438, 2016

use rayon::prelude::*;

use crate::limma::lowess::lowess;
use crate::numeric::stats::quantile_type7;
use crate::prelude::*;
use crate::ql::chebyshev::{compute_weight, unit_nb_deviance};
use crate::utils::design::hat_diagonal;

////////////
// Consts //
////////////

/// Complementary leverage below which an observation is dropped outright.
///
/// edgeR's `thresholdzero`. A sample whose leverage is one is fitted exactly,
/// so its residual carries no information; counting its unit deviance would
/// inflate the numerator of `s2` without adding anything to the denominator.
const THRESHOLD_ZERO: f64 = 1e-4;

/// Degrees of freedom below which a gene is excluded from the prior trend.
///
/// edgeR's `t`. Genes at zero residual df have `s2 = 0` by the rule above, and
/// leaving them in would drag the lowess fit towards zero across whatever
/// abundance range they occupy.
const MIN_DF: f64 = 1e-8;

/// Span of the lowess fit behind the prior.
///
/// Half the genes per window. edgeR's `f`.
const PRIOR_SPAN: f64 = 0.5;

/// Robustness iterations behind the prior.
///
/// edgeR asks R's `lowess` for `iter = 3`, meaning three robustness passes after
/// the initial fit, and [`crate::limma::lowess::lowess`] uses the same
/// convention. Note limma's `weightedLowess` counts *total* passes instead, so
/// the equivalent there would be four; mixing the two conventions up costs
/// 3e-3 on the prior.
const PRIOR_ITERATIONS: usize = 3;

/// Quantile of the fitted trend taken as the prior.
///
/// edgeR reads the 90th percentile rather than the mean or the median: the
/// prior is meant to sit above the bulk of the genes, so that dividing by it
/// pulls the typical quasi-dispersion below one.
const PRIOR_QUANTILE: f64 = 0.9;

/// Lower bound on the prior before it is raised to the fourth power.
///
/// A prior below one would sharpen the dispersions rather than shrink them,
/// which is not what the adjustment is for. edgeR clamps on the fourth-root
/// scale, so the effective floor on the returned value is also one.
const PRIOR_FLOOR: f64 = 1.0;

////////////////
// Public API //
////////////////

/// Per-gene adjusted deviance, degrees of freedom and their ratio.
///
/// All three vectors have one entry per gene, in input order.
#[derive(Clone, Debug)]
pub struct AdjustedDeviance {
    /// Adjusted residual deviance, summed over samples.
    pub deviance: Vec<f64>,
    /// Adjusted residual degrees of freedom, summed over samples. Fractional,
    /// because each sample contributes its own `kappa` rather than one.
    pub df: Vec<f64>,
    /// Quasi-likelihood dispersion, `deviance / df`. Zero where `df` has
    /// collapsed below [`THRESHOLD_ZERO`].
    pub s2: Vec<f64>,
}

/// Adjusted deviance and degrees of freedom for every gene.
///
/// A port of edgeR's `compute_adjust_vec`. For one gene and one sample the
/// contribution is
///
/// ```text
/// deviance += alpha * d(y, mu, phi w / prior) * w
/// df       += kappa * (1 - h)
/// ```
///
/// with `(alpha, kappa)` from [`compute_weight`], `d` the unit deviance, and `h`
/// the leverage of that sample under the working weights
/// `mu / (1 + mu phi w / prior)`. Observations whose complementary leverage
/// falls below [`THRESHOLD_ZERO`] drop out of both sums.
///
/// ### Params
///
/// * `counts` - Counts, row-major `n_genes * n_samples`
/// * `n_genes` - Number of genes
/// * `n_samples` - Number of samples
/// * `fitted` - Fitted means, row-major `n_genes * n_samples`
/// * `design` - Design matrix, row-major `n_samples * n_coef`
/// * `n_coef` - Number of coefficients
/// * `dispersion` - Negative binomial dispersions, recycled over genes and
///   samples
/// * `prior` - Average quasi-dispersion. Length one for edgeR's scalar prior,
///   or `n_genes` for a per-gene one. Every entry must be finite and positive.
/// * `weights` - Optional observation weights, recycled the same way
///
/// ### Returns
///
/// An [`AdjustedDeviance`], or [`EdgeErrors`] if any shape disagrees, the design
/// has fewer rows than columns, or a dispersion or prior is out of domain.
#[allow(clippy::too_many_arguments)]
pub fn compute_adjust_vec<T: EdgeFloat>(
    counts: &[T],
    n_genes: usize,
    n_samples: usize,
    fitted: &[f64],
    design: &[f64],
    n_coef: usize,
    dispersion: &Recycled<f64>,
    prior: &[f64],
    weights: Option<&Recycled<f64>>,
) -> Result<AdjustedDeviance, EdgeErrors> {
    validate(
        counts.len(),
        n_genes,
        n_samples,
        fitted.len(),
        design.len(),
        n_coef,
    )?;
    dispersion.validate(n_genes, n_samples)?;
    if let Some(w) = weights {
        w.validate(n_genes, n_samples)?;
    }
    validate_prior(prior, n_genes)?;

    let per_gene: Vec<(f64, f64, f64)> = (0..n_genes)
        .into_par_iter()
        .map_init(
            || Scratch::new(n_samples, n_coef),
            |scratch, gene| {
                let start = gene * n_samples;
                for (slot, value) in scratch.y.iter_mut().zip(&counts[start..start + n_samples]) {
                    *slot = value.to_f64().unwrap_or(0.0);
                }

                adjust_one_gene(
                    &scratch.y,
                    &fitted[start..start + n_samples],
                    design,
                    n_coef,
                    dispersion.row(gene, n_samples),
                    weights.map(|w| w.row(gene, n_samples)),
                    if prior.len() == 1 {
                        prior[0]
                    } else {
                        prior[gene]
                    },
                    &mut scratch.weighted_design,
                )
            },
        )
        .collect();

    let mut out = AdjustedDeviance {
        deviance: Vec::with_capacity(n_genes),
        df: Vec::with_capacity(n_genes),
        s2: Vec::with_capacity(n_genes),
    };
    for (deviance, df, s2) in per_gene {
        out.deviance.push(deviance);
        out.df.push(df);
        out.s2.push(s2);
    }

    Ok(out)
}

/// Average quasi-dispersion from one round of quasi-likelihood dispersions.
///
/// A port of edgeR's `compute_prior`. Genes with usable residual degrees of
/// freedom are smoothed on the fourth-root scale against average log-CPM, the
/// 90th percentile of that trend is taken, floored at one, and raised back to
/// the fourth power.
///
/// ### Deviation from edgeR
///
/// edgeR smooths with R's `lowess` (`f = 0.5`, `iter = 3`,
/// `delta = 0.01 * diff(range(x))`); this uses limma's `weightedLowess`, which
/// is a different algorithm with a different window rule and a different seed
/// spacing. On 500 genes with a heavily overdispersed fit the two priors differ
/// by 5.2e-5 relative, so the whole quasi-likelihood path inherits an error of
/// that order.
///
/// ### Params
///
/// * `ave_log_cpm` - Average log-CPM, one per gene
/// * `s2` - Quasi-likelihood dispersions, one per gene
/// * `df` - Adjusted residual degrees of freedom, one per gene
///
/// ### Returns
///
/// A single-element vector holding the average quasi-dispersion, which is what
/// [`compute_adjust_vec`] takes as its `prior`. Errors with
/// [`EdgeErrors::LengthMismatch`] if the three inputs disagree.
pub fn compute_prior(ave_log_cpm: &[f64], s2: &[f64], df: &[f64]) -> Result<Vec<f64>, EdgeErrors> {
    let n = ave_log_cpm.len();
    if s2.len() != n {
        return Err(EdgeErrors::LengthMismatch {
            name: "s2",
            expected: n,
            got: s2.len(),
        });
    }
    if df.len() != n {
        return Err(EdgeErrors::LengthMismatch {
            name: "df",
            expected: n,
            got: df.len(),
        });
    }

    let mut abundance = Vec::with_capacity(n);
    let mut root_s2 = Vec::with_capacity(n);
    for gene in 0..n {
        if df[gene] > MIN_DF {
            abundance.push(ave_log_cpm[gene]);
            root_s2.push(s2[gene].max(0.0).sqrt().sqrt());
        }
    }

    let trend = match root_s2.len() {
        0 => return Ok(vec![PRIOR_FLOOR]),
        1 => root_s2,
        _ => {
            // Cleveland's lowess, not limma's `weightedLowess`. edgeR's
            // `compute_ave_qd` calls `lowess(f = 0.5, iter = 3)`; reaching for
            // the other smoother costs up to 7e-4 on the prior and on every
            // quasi-likelihood dispersion downstream of it.
            lowess(&abundance, &root_s2, PRIOR_SPAN, PRIOR_ITERATIONS)?
        }
    };

    let p = quantile_type7(&trend, PRIOR_QUANTILE)?.max(PRIOR_FLOOR);

    Ok(vec![p * p * p * p])
}

/// Average quasi-dispersion, found by two rounds of adjustment.
///
/// A port of edgeR's `update_prior`. The adjustment needs a prior and the prior
/// needs an adjustment, so edgeR starts at one, adjusts, re-estimates, adjusts
/// again and re-estimates once more. Two rounds, not iterated to convergence:
/// the second move is already small and edgeR stops there.
///
/// ### Params
///
/// * `counts` - Counts, row-major `n_genes * n_samples`
/// * `n_genes` - Number of genes
/// * `n_samples` - Number of samples
/// * `fitted` - Fitted means, row-major `n_genes * n_samples`
/// * `design` - Design matrix, row-major `n_samples * n_coef`
/// * `n_coef` - Number of coefficients
/// * `dispersion` - Negative binomial dispersions, recycled over genes and
///   samples
/// * `weights` - Optional observation weights, recycled the same way
/// * `ave_log_cpm` - Average log-CPM, one per gene
///
/// ### Returns
///
/// A single-element vector holding the average quasi-dispersion, or
/// [`EdgeErrors`] if any shape disagrees.
#[allow(clippy::too_many_arguments)]
pub fn update_prior<T: EdgeFloat>(
    counts: &[T],
    n_genes: usize,
    n_samples: usize,
    fitted: &[f64],
    design: &[f64],
    n_coef: usize,
    dispersion: &Recycled<f64>,
    weights: Option<&Recycled<f64>>,
    ave_log_cpm: &[f64],
) -> Result<Vec<f64>, EdgeErrors> {
    if ave_log_cpm.len() != n_genes {
        return Err(EdgeErrors::LengthMismatch {
            name: "ave_log_cpm",
            expected: n_genes,
            got: ave_log_cpm.len(),
        });
    }

    let mut prior = vec![PRIOR_FLOOR];
    for _ in 0..2 {
        let adjusted = compute_adjust_vec(
            counts, n_genes, n_samples, fitted, design, n_coef, dispersion, &prior, weights,
        )?;
        prior = compute_prior(ave_log_cpm, &adjusted.s2, &adjusted.df)?;
    }

    Ok(prior)
}

/////////////
// Scratch //
/////////////

/// Per-thread buffers for the gene fan-out.
struct Scratch {
    /// Counts for the current gene, widened to `f64`.
    y: Vec<f64>,
    /// `sqrt(W) X` for the current gene, row-major `n_samples * n_coef`.
    weighted_design: Vec<f64>,
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
    /// Buffers sized for a single gene.
    fn new(n_samples: usize, n_coef: usize) -> Self {
        Self {
            y: vec![0.0; n_samples],
            weighted_design: vec![0.0; n_samples * n_coef],
        }
    }
}

/// Checks the shapes [`compute_adjust_vec`] cannot recover from.
///
/// Pulled out so that the per-gene kernel can treat [`hat_diagonal`] as
/// infallible: everything that routine rejects is rejected here first.
///
/// ### Params
///
/// * `n_counts` - Length of the counts slice
/// * `n_genes` - Number of genes
/// * `n_samples` - Number of samples
/// * `n_fitted` - Length of the fitted slice
/// * `n_design` - Length of the design slice
/// * `n_coef` - Number of coefficients
///
/// ### Returns
///
/// `Ok(())` when every shape is consistent, otherwise [`EdgeErrors`].
fn validate(
    n_counts: usize,
    n_genes: usize,
    n_samples: usize,
    n_fitted: usize,
    n_design: usize,
    n_coef: usize,
) -> Result<(), EdgeErrors> {
    if n_genes == 0 || n_samples == 0 {
        return Err(EdgeErrors::EmptyCounts { n_genes, n_samples });
    }
    if n_coef == 0 {
        return Err(EdgeErrors::MustBePositive("n_coef".to_string()));
    }
    if n_counts != n_genes * n_samples {
        return Err(EdgeErrors::LengthMismatch {
            name: "counts",
            expected: n_genes * n_samples,
            got: n_counts,
        });
    }
    if n_fitted != n_genes * n_samples {
        return Err(EdgeErrors::LengthMismatch {
            name: "fitted",
            expected: n_genes * n_samples,
            got: n_fitted,
        });
    }
    if n_design != n_samples * n_coef {
        return Err(EdgeErrors::LengthMismatch {
            name: "design",
            expected: n_samples * n_coef,
            got: n_design,
        });
    }
    if n_samples < n_coef {
        return Err(EdgeErrors::DesignNotFullRank {
            n_cols: n_coef,
            rank: n_samples,
        });
    }
    Ok(())
}

/// Checks the prior is either scalar or one value per gene, and in domain.
///
/// ### Params
///
/// * `prior` - Average quasi-dispersion, length one or `n_genes`
/// * `n_genes` - Number of genes
///
/// ### Returns
///
/// `Ok(())` when usable, otherwise [`EdgeErrors`]. The prior divides a
/// dispersion, so zero and negative values are rejected rather than allowed to
/// produce an infinity halfway through the sum.
fn validate_prior(prior: &[f64], n_genes: usize) -> Result<(), EdgeErrors> {
    if prior.len() != 1 && prior.len() != n_genes {
        return Err(EdgeErrors::LengthMismatch {
            name: "prior",
            expected: n_genes,
            got: prior.len(),
        });
    }
    if let Some(&bad) = prior.iter().find(|p| !p.is_finite() || **p <= 0.0) {
        return Err(EdgeErrors::InvalidArgument(format!(
            "prior must be finite and positive; got {bad}."
        )));
    }
    Ok(())
}

/// Adjusted deviance, degrees of freedom and their ratio for one gene.
///
/// The leverages come from the QR of `sqrt(W) X`, where the working weight is
/// `mu / (1 + mu phi w / prior)`. edgeR's `qr_hat` uses LINPACK's `dqrdc2`,
/// which truncates the factorisation at the numerical rank; [`hat_diagonal`]
/// does not. The two differ only when the weighted design is rank deficient
/// while its columns still carry non-negligible norm, which needs a fitted mean
/// that is tiny but not zero. edgeR's own GLM returns exact zeros for a
/// structurally zero group, and [`compute_weight`] reports `(0, 0)` there, so
/// the difference cancels on the path that actually produces these inputs. See
/// `test_structural_zeros_match_edger`.
///
/// ### Params
///
/// * `y` - Counts for this gene, one per sample
/// * `mu` - Fitted means for this gene, same length
/// * `design` - Design matrix, row-major `n_samples * n_coef`
/// * `n_coef` - Number of coefficients
/// * `dispersion` - Dispersion row for this gene
/// * `weights` - Optional weight row for this gene
/// * `prior` - Average quasi-dispersion for this gene
/// * `weighted_design` - Scratch of `n_samples * n_coef`, overwritten
///
/// ### Returns
///
/// `(deviance, df, s2)`.
#[allow(clippy::too_many_arguments)]
fn adjust_one_gene(
    y: &[f64],
    mu: &[f64],
    design: &[f64],
    n_coef: usize,
    dispersion: RecycledRow<'_, f64>,
    weights: Option<RecycledRow<'_, f64>>,
    prior: f64,
    weighted_design: &mut [f64],
) -> (f64, f64, f64) {
    let n_samples = y.len();

    for (sample, &mu_j) in mu.iter().enumerate() {
        let phi = dispersion.get(sample);
        let w = weights.map_or(1.0, |w| w.get(sample));
        let working = (mu_j / (1.0 + (mu_j * phi * w / prior))).sqrt();
        let row = &design[sample * n_coef..(sample + 1) * n_coef];
        let out = &mut weighted_design[sample * n_coef..(sample + 1) * n_coef];
        for (slot, &x) in out.iter_mut().zip(row) {
            *slot = x * working;
        }
    }

    // Infallible: `validate` has already rejected every shape `hat_diagonal`
    // checks, and the scratch is sized from those same dimensions.
    let hat = hat_diagonal(weighted_design, n_samples, n_coef)
        .expect("weighted design shape was validated on entry");

    let mut deviance = 0.0;
    let mut df = 0.0;
    for sample in 0..n_samples {
        let phi = dispersion.get(sample);
        let w = weights.map_or(1.0, |w| w.get(sample));
        let (alpha, kappa) = compute_weight(mu[sample], phi, prior / w);
        let mut unit = unit_nb_deviance(y[sample], mu[sample], phi * w / prior);
        let mut complement = 1.0 - hat[sample];
        if complement < THRESHOLD_ZERO {
            unit = 0.0;
            complement = 0.0;
        }
        deviance += (unit * alpha) * w;
        df += complement * kappa;
    }

    let s2 = if df < THRESHOLD_ZERO {
        0.0
    } else {
        deviance / df
    };

    (deviance, df, s2)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Relative tolerance for the deviance and degrees of freedom against
    /// edgeR's C. Everything on that path is arithmetic the port reproduces
    /// operation for operation, so the only slack is the QR.
    const TOL: f64 = 1e-12;

    /// Six genes by six samples. Every count and every fitted mean is exactly
    /// representable in binary, so R and Rust see identical inputs and any
    /// disagreement is the algorithm rather than the fixture.
    ///
    /// Gene 3 is zero in the first group and gene 4 is zero everywhere, which is
    /// what `glmFit` returns exact zero fitted values for. Gene 5 sits in the
    /// thousands, where the Chebyshev fits have switched panels.
    fn counts() -> Vec<f64> {
        vec![
            10.0, 12.0, 11.0, 40.0, 44.0, 38.0, //
            50.0, 48.0, 52.0, 49.0, 51.0, 50.0, //
            2.0, 0.0, 5.0, 1.0, 3.0, 0.0, //
            0.0, 0.0, 0.0, 20.0, 22.0, 18.0, //
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            1000.0, 1100.0, 900.0, 3000.0, 3200.0, 2800.0,
        ]
    }

    /// Fitted means for [`counts`], all dyadic rationals.
    fn fitted() -> Vec<f64> {
        vec![
            11.0, 11.5, 10.75, 40.5, 41.25, 39.5, //
            50.0, 49.5, 50.25, 49.75, 50.5, 50.0, //
            2.5, 2.25, 2.75, 1.5, 1.25, 1.75, //
            0.0, 0.0, 0.0, 20.0, 20.0, 20.0, //
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            1000.0, 1050.0, 950.0, 3000.0, 3100.0, 2900.0,
        ]
    }

    /// Two-group design, row-major.
    fn two_group() -> Vec<f64> {
        vec![
            1.0, 0.0, //
            1.0, 0.0, //
            1.0, 0.0, //
            1.0, 1.0, //
            1.0, 1.0, //
            1.0, 1.0,
        ]
    }

    /// Continuous covariate design, row-major.
    fn continuous() -> Vec<f64> {
        vec![
            1.0, -1.5, //
            1.0, -0.5, //
            1.0, 0.5, //
            1.0, 1.5, //
            1.0, 2.5, //
            1.0, 3.5,
        ]
    }

    /// Runs the six-gene fixture and compares against a reference triple.
    fn check(
        design: &[f64],
        dispersion: Recycled<f64>,
        prior: &[f64],
        weights: Option<&Recycled<f64>>,
        deviance: &[f64],
        df: &[f64],
        s2: &[f64],
    ) {
        let got = compute_adjust_vec(
            &counts(),
            6,
            6,
            &fitted(),
            design,
            2,
            &dispersion,
            prior,
            weights,
        )
        .unwrap();

        for gene in 0..6 {
            assert_relative_eq!(got.deviance[gene], deviance[gene], max_relative = TOL);
            assert_relative_eq!(got.df[gene], df[gene], max_relative = TOL);
            assert_relative_eq!(got.s2[gene], s2[gene], max_relative = TOL);
        }
    }

    /// The shared preamble of every `Rscript` block below.
    ///
    /// ```r
    /// # Rscript, edgeR 4.8.2
    /// library(edgeR)
    /// y <- matrix(c(10,12,11,40,44,38, 50,48,52,49,51,50, 2,0,5,1,3,0,
    ///               0,0,0,20,22,18, 0,0,0,0,0,0,
    ///               1000,1100,900,3000,3200,2800), nrow = 6, byrow = TRUE)
    /// mu <- matrix(c(11,11.5,10.75,40.5,41.25,39.5, 50,49.5,50.25,49.75,50.5,50,
    ///                2.5,2.25,2.75,1.5,1.25,1.75, 0,0,0,20,20,20, 0,0,0,0,0,0,
    ///                1000,1050,950,3000,3100,2900), nrow = 6, byrow = TRUE)
    /// storage.mode(y) <- "double"
    /// X  <- cbind(1, c(0,0,0,1,1,1));                 storage.mode(X)  <- "double"
    /// Xc <- cbind(1, c(-1.5,-0.5,0.5,1.5,2.5,3.5));   storage.mode(Xc) <- "double"
    /// dm <- edgeR:::.compressDispersions(y, 0.125)
    /// wm <- edgeR:::.compressWeights(y, NULL)
    /// ```
    ///
    /// `.Call(edgeR:::.cxx_compute_adj_vec, y, mu, X, dm, 1.0, wm)`
    #[test]
    fn test_two_group_matches_edger() {
        check(
            &two_group(),
            Recycled::scalar(0.125),
            &[1.0],
            None,
            &[
                0.087_678_574_155_993_14,
                0.016_535_319_784_658_36,
                11.401_625_661_199_109,
                0.111_548_751_889_321_45,
                0.0,
                0.056_972_440_150_977_24,
            ],
            &[
                3.969_634_764_628_127,
                3.999_429_434_054_224,
                5.340_746_228_772_16,
                1.992_229_288_606_397_7,
                0.0,
                4.001_942_003_921_18,
            ],
            &[
                0.022_087_315_169_965_472,
                0.004_134_419_685_934_175,
                2.134_837_562_544_204,
                0.055_991_924_487_442_87,
                0.0,
                0.014_236_198_349_489_956,
            ],
        );
    }

    /// The same fixture at a prior above one, which is the branch `update_prior`
    /// actually lands on. The prior enters the working weights, the moments and
    /// the unit deviance separately, so this pins more than a rescaling.
    ///
    /// `.Call(edgeR:::.cxx_compute_adj_vec, y, mu, X, dm, 2.5, wm)`
    #[test]
    fn test_prior_above_one_matches_edger() {
        check(
            &two_group(),
            Recycled::scalar(0.125),
            &[2.5],
            None,
            &[
                0.148_404_728_650_759_82,
                0.034_023_627_382_698_46,
                23.944_790_215_814_272,
                0.187_876_878_253_122_5,
                0.0,
                0.141_051_250_719_609_08,
            ],
            &[
                3.944_608_502_302_763_5,
                3.984_625_332_836_51,
                9.508_568_001_899_926,
                1.949_129_177_816_912_6,
                0.0,
                4.001_937_312_693_502,
            ],
            &[
                0.037_622_169_237_866_03,
                0.008_538_726_866_567_972,
                2.518_233_051_604_596_5,
                0.096_390_162_535_841_08,
                0.0,
                0.035_245_742_173_975_886,
            ],
        );
    }

    /// A dispersion per gene, spanning the Chebyshev cases: gene 5 sits at two,
    /// past the case-1 boundary, and gene 4 carries zero, which is inert because
    /// the gene has no counts.
    ///
    /// ```r
    /// dg <- edgeR:::.compressDispersions(y, c(0.05,0.125,0.5,0.25,0,2))
    /// .Call(edgeR:::.cxx_compute_adj_vec, y, mu, X, dg, 1.5, wm)
    /// ```
    #[test]
    fn test_per_gene_dispersion_matches_edger() {
        check(
            &two_group(),
            Recycled::by_gene(vec![0.05, 0.125, 0.5, 0.25, 0.0, 2.0]),
            &[1.5],
            None,
            &[
                0.185_285_564_462_572_72,
                0.023_166_881_640_206_673,
                14.989_239_815_832_821,
                0.086_845_649_266_648_48,
                0.0,
                0.005_175_385_846_847_329_5,
            ],
            &[
                3.954_646_135_693_23,
                3.996_255_259_784_583_3,
                8.051_884_164_459_716,
                1.968_403_109_458_039_6,
                0.0,
                4.722_405_574_548_103_5,
            ],
            &[
                0.046_852_628_049_385_024,
                0.005_797_147_613_001_947,
                1.861_581_650_912_709_4,
                0.044_119_849_663_598_47,
                0.0,
                0.001_095_921_509_736_607_7,
            ],
        );
    }

    /// Explicit observation weights, which enter in three places at once: the
    /// working weight, the dispersion handed to the moments and the deviance,
    /// and the multiplier on the adjusted unit deviance.
    ///
    /// ```r
    /// w <- matrix(c(1,1,1,1,1,1, 0.5,0.5,2,2,1,1, 1,0.25,1,1,4,1,
    ///               2,2,2,0.5,0.5,0.5, 1,1,1,1,1,1, 1,2,1,0.5,1,1),
    ///             nrow = 6, byrow = TRUE)
    /// wm2 <- edgeR:::.compressWeights(y, w)
    /// .Call(edgeR:::.cxx_compute_adj_vec, y, mu, X, dm, 1.25, wm2)
    /// ```
    #[test]
    fn test_weights_match_edger() {
        let weights = Recycled::full(
            vec![
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, //
                0.5, 0.5, 2.0, 2.0, 1.0, 1.0, //
                1.0, 0.25, 1.0, 1.0, 4.0, 1.0, //
                2.0, 2.0, 2.0, 0.5, 0.5, 0.5, //
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, //
                1.0, 2.0, 1.0, 0.5, 1.0, 1.0,
            ],
            6,
            6,
        )
        .unwrap();

        check(
            &two_group(),
            Recycled::scalar(0.125),
            &[1.25],
            Some(&weights),
            &[
                0.101_079_410_203_347_93,
                0.019_878_961_611_844_97,
                12.984_719_805_109_156,
                0.093_938_439_126_561_25,
                0.0,
                0.071_201_367_293_380_21,
            ],
            &[
                3.954_058_331_932_422_7,
                3.995_405_261_291_514,
                6.298_973_859_851_691,
                1.949_129_177_816_912_6,
                0.0,
                4.001_941_748_985_114,
            ],
            &[
                0.025_563_459_544_095_425,
                0.004_975_455_632_608_117,
                2.061_402_395_693_523_5,
                0.048_195_081_267_920_54,
                0.0,
                0.017_791_705_067_030_712,
            ],
        );
    }

    /// A continuous covariate. Only the leverages change, so the deviances are
    /// bit-identical to the two-group case while the degrees of freedom are not.
    /// Gene 3, whose first group is structurally zero, drops to roughly one
    /// residual degree of freedom here against two under the group design.
    ///
    /// `.Call(edgeR:::.cxx_compute_adj_vec, y, mu, Xc, dm, 1.0, wm)`
    #[test]
    fn test_continuous_design_matches_edger() {
        check(
            &continuous(),
            Recycled::scalar(0.125),
            &[1.0],
            None,
            &[
                0.087_678_574_155_993_14,
                0.016_535_319_784_658_36,
                11.401_625_661_199_109,
                0.111_548_751_889_321_45,
                0.0,
                0.056_972_440_150_977_24,
            ],
            &[
                3.968_981_768_078_912_3,
                3.999_429_410_347_023_6,
                5.390_514_507_550_15,
                0.996_114_644_303_199,
                0.0,
                4.001_941_999_230_005,
            ],
            &[
                0.022_090_949_084_513_38,
                0.004_134_419_710_441_55,
                2.115_127_534_714_836,
                0.111_983_848_974_885_72,
                0.0,
                0.014_236_198_366_177_98,
            ],
        );
    }

    /// One gene on its own must give exactly what it gives inside a batch, since
    /// the gene axis is the parallel one.
    ///
    /// ```r
    /// y1 <- y[2,,drop=FALSE]; mu1 <- mu[2,,drop=FALSE]
    /// dm1 <- edgeR:::.compressDispersions(y1, 0.125)
    /// wm1 <- edgeR:::.compressWeights(y1, NULL)
    /// .Call(edgeR:::.cxx_compute_adj_vec, y1, mu1, X, dm1, 1.0, wm1)
    /// ```
    #[test]
    fn test_single_gene_matches_edger() {
        let out = compute_adjust_vec(
            &counts()[6..12],
            1,
            6,
            &fitted()[6..12],
            &two_group(),
            2,
            &Recycled::scalar(0.125),
            &[1.0],
            None,
        )
        .unwrap();

        assert_relative_eq!(
            out.deviance[0],
            0.016_535_319_784_658_36,
            max_relative = TOL
        );
        assert_relative_eq!(out.df[0], 3.999_429_434_054_224, max_relative = TOL);
        assert_relative_eq!(out.s2[0], 0.004_134_419_685_934_175, max_relative = TOL);
    }

    /// Genes 3 and 4 are the structural-zero cases, and they are why the
    /// unpivoted QR here is safe.
    ///
    /// edgeR's `dqrdc2` finds rank 1 for gene 3 and rank 0 for gene 4;
    /// [`hat_diagonal`] keeps both columns and therefore reports different
    /// leverages. `test_two_group_matches_edger` still passes on those two genes,
    /// because [`compute_weight`] returns `(0, 0)` wherever the fitted mean is
    /// zero, so those samples contribute nothing under either set of leverages.
    /// This test states the consequence directly: gene 4 is entirely inert, and
    /// gene 3 keeps exactly the three samples that carry counts.
    #[test]
    fn test_structural_zeros_collapse_the_degrees_of_freedom() {
        let out = compute_adjust_vec(
            &counts(),
            6,
            6,
            &fitted(),
            &two_group(),
            2,
            &Recycled::scalar(0.125),
            &[1.0],
            None,
        )
        .unwrap();

        assert_eq!(out.deviance[4], 0.0);
        assert_eq!(out.df[4], 0.0);
        assert_eq!(out.s2[4], 0.0);
        // Three live samples against one fitted coefficient.
        assert!(out.df[3] > 1.9 && out.df[3] < 2.1);
        // Genes with no structural zeros keep close to four.
        assert!(out.df[0] > 3.9);
    }

    /// The three `Recycled` forms of one dispersion must not differ, which pins
    /// the dispatch inside the kernel.
    #[test]
    fn test_recycled_dispersion_forms_agree() {
        let run = |dispersion: Recycled<f64>| {
            compute_adjust_vec(
                &counts(),
                6,
                6,
                &fitted(),
                &two_group(),
                2,
                &dispersion,
                &[1.0],
                None,
            )
            .unwrap()
        };

        let scalar = run(Recycled::scalar(0.125));
        let per_gene = run(Recycled::by_gene(vec![0.125; 6]));
        let by_sample = run(Recycled::by_sample(vec![0.125; 6]));
        let full = run(Recycled::full(vec![0.125; 36], 6, 6).unwrap());

        assert_eq!(scalar.deviance, per_gene.deviance);
        assert_eq!(scalar.deviance, by_sample.deviance);
        assert_eq!(scalar.deviance, full.deviance);
        assert_eq!(scalar.df, per_gene.df);
        assert_eq!(scalar.df, full.df);
    }

    /// A per-gene prior of a repeated value must equal the scalar prior.
    #[test]
    fn test_per_gene_prior_matches_scalar_prior() {
        let run = |prior: &[f64]| {
            compute_adjust_vec(
                &counts(),
                6,
                6,
                &fitted(),
                &two_group(),
                2,
                &Recycled::scalar(0.125),
                prior,
                None,
            )
            .unwrap()
        };

        let scalar = run(&[2.5]);
        let vector = run(&[2.5; 6]);
        assert_eq!(scalar.deviance, vector.deviance);
        assert_eq!(scalar.df, vector.df);
    }

    /// `f32` counts must widen to the same answer, since every count in the
    /// fixture is exactly representable in single precision.
    #[test]
    fn test_f32_counts_agree_with_f64() {
        let wide = compute_adjust_vec(
            &counts(),
            6,
            6,
            &fitted(),
            &two_group(),
            2,
            &Recycled::scalar(0.125),
            &[1.0],
            None,
        )
        .unwrap();

        let narrow: Vec<f32> = counts().iter().map(|&v| v as f32).collect();
        let got = compute_adjust_vec(
            &narrow,
            6,
            6,
            &fitted(),
            &two_group(),
            2,
            &Recycled::scalar(0.125),
            &[1.0],
            None,
        )
        .unwrap();

        assert_eq!(wide.deviance, got.deviance);
        assert_eq!(wide.df, got.df);
    }

    ////////////////////
    // Prior fixtures //
    ////////////////////

    /// Relative tolerance for anything that passes through the smoother.
    ///
    /// This was 1e-3 while `compute_prior` used limma's `weightedLowess`, which
    /// is a different algorithm from the `lowess` edgeR actually calls: 3.0e-4
    /// on the 25-point grid, 7.1e-4 once two genes are filtered out of it, and
    /// 5.2e-5 on 500 genes. Pointing it at
    /// [`crate::limma::lowess::lowess`] with `iter = 3` closed the gap
    /// entirely, so the tolerance is now rounding.
    ///
    /// Mind the iteration convention if this is ever touched: R's `lowess`
    /// counts robustness passes *after* the first fit, limma's `weightedLowess`
    /// counts total passes. Using limma's 4 here instead of edgeR's 3 costs
    /// 3e-3, worse than the original mismatch.
    const PRIOR_TOL: f64 = 1e-12;

    /// Twenty-five genes on a dyadic grid of average log-CPM, with dyadic `s2`
    /// and three degrees of freedom each. Exactly representable, so R and Rust
    /// smooth identical numbers.
    fn prior_fixture() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let ave: Vec<f64> = (0..25).map(|i| i as f64 / 4.0 - 2.0).collect();
        let s2: Vec<f64> = (0..25)
            .map(|i| ((i * 7 % 13) + 1) as f64 / 8.0 + i as f64 / 16.0)
            .collect();
        (ave, s2, vec![3.0; 25])
    }

    /// The prior against edgeR's `compute_prior`, replicated in R because the C
    /// function is not exported on its own.
    ///
    /// ```r
    /// # Rscript, R 4.5.1 / edgeR 4.8.2
    /// ave <- (0:24)/4 - 2
    /// s2  <- ((0:24 * 7) %% 13 + 1)/8 + (0:24)/16
    /// ans <- lowess(ave, sqrt(sqrt(s2)), f = 0.5, iter = 3,
    ///               delta = 0.01 * diff(range(ave)))$y
    /// max(quantile(ans, 0.9, type = 7, names = FALSE), 1)^4   # 2.423348828030909
    /// ```
    #[test]
    fn test_compute_prior_matches_edger() {
        let (ave, s2, df) = prior_fixture();
        let got = compute_prior(&ave, &s2, &df).unwrap();

        assert_eq!(got.len(), 1);
        assert_relative_eq!(got[0], 2.423_348_828_030_909, max_relative = PRIOR_TOL);
    }

    /// Genes below the degrees of freedom threshold must never reach the
    /// smoother. Dropping the two end genes changes the answer, and it must
    /// change it to exactly what fitting the survivors alone gives.
    ///
    /// ```r
    /// a2 <- lowess(ave[2:24], sqrt(sqrt(s2[2:24])), f = 0.5, iter = 3,
    ///              delta = 0.01 * diff(range(ave[2:24])))$y
    /// max(quantile(a2, 0.9, type = 7, names = FALSE), 1)^4  # 2.2385492590160405
    /// ```
    #[test]
    fn test_compute_prior_drops_low_df_genes() {
        let (ave, s2, mut df) = prior_fixture();
        df[0] = 0.0;
        df[24] = MIN_DF;
        let filtered = compute_prior(&ave, &s2, &df).unwrap();

        let direct = compute_prior(&ave[1..24], &s2[1..24], &[3.0; 23]).unwrap();
        assert_eq!(filtered, direct);
        assert_relative_eq!(
            filtered[0],
            2.238_549_259_016_040_5,
            max_relative = PRIOR_TOL
        );
    }

    /// With no gene left the prior is inert, and with one gene it is that gene's
    /// own value. Both sit below the smoother's two-point minimum.
    #[test]
    fn test_compute_prior_degenerate_gene_counts() {
        let none = compute_prior(&[1.0, 2.0], &[4.0, 9.0], &[0.0, 0.0]).unwrap();
        assert_eq!(none, vec![1.0]);

        // The fourth root of 16 is 2, so the fourth power brings 16 straight back.
        let one = compute_prior(&[1.0, 2.0], &[16.0, 9.0], &[3.0, 0.0]).unwrap();
        assert_relative_eq!(one[0], 16.0, max_relative = 1e-12);
    }

    /// A trend that sits entirely below one is floored, not inverted.
    #[test]
    fn test_compute_prior_floors_at_one() {
        let ave: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let got = compute_prior(&ave, &[0.0625; 10], &[2.0; 10]).unwrap();
        assert_eq!(got, vec![1.0]);
    }

    /// Twenty-four genes by six samples, overdispersed enough that the prior
    /// lands well above its floor. Counts are integers and every fitted mean is
    /// a group mean rounded to a quarter, so the fixture is exact.
    ///
    /// Generated by the script quoted on [`test_update_prior_matches_edger`].
    #[allow(clippy::type_complexity)]
    fn prior_counts() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let counts = vec![
            0.0, 11.0, 1.0, 2.0, 2.0, 2.0, //
            1.0, 5.0, 5.0, 8.0, 3.0, 3.0, //
            1.0, 3.0, 5.0, 7.0, 3.0, 3.0, //
            7.0, 6.0, 0.0, 6.0, 3.0, 6.0, //
            3.0, 9.0, 1.0, 4.0, 8.0, 0.0, //
            18.0, 11.0, 9.0, 36.0, 11.0, 5.0, //
            19.0, 9.0, 7.0, 1.0, 34.0, 17.0, //
            20.0, 3.0, 7.0, 22.0, 3.0, 4.0, //
            19.0, 14.0, 40.0, 64.0, 37.0, 26.0, //
            12.0, 20.0, 27.0, 9.0, 28.0, 54.0, //
            41.0, 8.0, 2.0, 4.0, 22.0, 68.0, //
            32.0, 38.0, 43.0, 22.0, 13.0, 46.0, //
            46.0, 92.0, 8.0, 53.0, 49.0, 66.0, //
            13.0, 7.0, 88.0, 28.0, 156.0, 51.0, //
            294.0, 31.0, 19.0, 17.0, 158.0, 20.0, //
            22.0, 73.0, 14.0, 74.0, 160.0, 42.0, //
            76.0, 154.0, 67.0, 123.0, 28.0, 265.0, //
            196.0, 111.0, 29.0, 150.0, 18.0, 96.0, //
            7.0, 133.0, 102.0, 174.0, 310.0, 73.0, //
            133.0, 492.0, 362.0, 14.0, 71.0, 58.0, //
            250.0, 551.0, 40.0, 46.0, 54.0, 56.0, //
            45.0, 288.0, 209.0, 376.0, 182.0, 152.0, //
            458.0, 157.0, 192.0, 524.0, 480.0, 574.0, //
            172.0, 271.0, 340.0, 874.0, 456.0, 210.0,
        ];

        // Group means, three samples each.
        let group_means: [(f64, f64); 24] = [
            (4.0, 2.0),
            (3.75, 4.75),
            (3.0, 4.25),
            (4.25, 5.0),
            (4.25, 4.0),
            (12.75, 17.25),
            (11.75, 17.25),
            (10.0, 9.75),
            (24.25, 42.25),
            (19.75, 30.25),
            (17.0, 31.25),
            (37.75, 27.0),
            (48.75, 56.0),
            (36.0, 78.25),
            (114.75, 65.0),
            (36.25, 92.0),
            (99.0, 138.75),
            (112.0, 88.0),
            (80.75, 185.75),
            (329.0, 47.75),
            (280.25, 52.0),
            (180.75, 236.75),
            (269.0, 526.0),
            (261.0, 513.25),
        ];
        let mut fitted = Vec::with_capacity(144);
        for (first, second) in group_means {
            fitted.extend_from_slice(&[first, first, first, second, second, second]);
        }

        let ave_log_cpm = vec![
            11.171875, 11.484375, 11.375, 11.59375, 11.46875, 12.921875, 12.9375, 12.4375,
            14.015625, 13.6875, 13.65625, 14.046875, 14.59375, 14.828125, 15.453125, 14.84375,
            15.8125, 15.546875, 15.890625, 16.515625, 16.203125, 16.546875, 17.53125, 17.421875,
        ];

        (counts, fitted, ave_log_cpm)
    }

    /// One round of the adjustment followed by one `compute_prior`, which is the
    /// first half of [`update_prior`] with its intermediate exposed.
    ///
    /// ```r
    /// o1 <- .Call(edgeR:::.cxx_compute_adj_vec, y, mu, X, dm, 1.0, wm)
    /// cp(ave, o1$s2, o1$df)   # 5.2505569919690807
    /// ```
    #[test]
    fn test_prior_after_one_round_matches_edger() {
        let (counts, fitted, ave) = prior_counts();
        let adjusted = compute_adjust_vec(
            &counts,
            24,
            6,
            &fitted,
            &two_group(),
            2,
            &Recycled::scalar(0.125),
            &[1.0],
            None,
        )
        .unwrap();

        let prior = compute_prior(&ave, &adjusted.s2, &adjusted.df).unwrap();
        assert_relative_eq!(prior[0], 5.250_556_991_969_081, max_relative = PRIOR_TOL);
    }

    /// The whole two-round loop against edgeR's own `compute_ave_qd`, which is
    /// the exact entry point `glmQLFit(legacy = FALSE)` calls.
    ///
    /// ```r
    /// # Rscript, R 4.5.1 / edgeR 4.8.2
    /// set.seed(11)
    /// ng <- 24; ns <- 6
    /// X <- cbind(1, rep(c(0,1), each = 3)); storage.mode(X) <- "double"
    /// base <- round(exp(seq(log(4), log(400), length.out = ng)))
    /// y <- matrix(0, ng, ns)
    /// for (g in 1:ng) y[g,] <- rnbinom(ns, size = 1/0.6, mu = base[g])
    /// storage.mode(y) <- "double"
    /// mu <- t(apply(y, 1, function(r)
    ///   rep(round(c(mean(r[1:3]), mean(r[4:6])) * 4) / 4, each = 3)))
    /// ave <- round(aveLogCPM(y, dispersion = 0.1) * 64) / 64
    /// dm <- edgeR:::.compressDispersions(y, 0.125)
    /// wm <- edgeR:::.compressWeights(y, NULL)
    /// .Call(edgeR:::.cxx_compute_ave_qd, y, mu, X, dm, ave, wm)
    /// # 18.590485410598912
    /// ```
    #[test]
    fn test_update_prior_matches_edger() {
        let (counts, fitted, ave) = prior_counts();
        let prior = update_prior(
            &counts,
            24,
            6,
            &fitted,
            &two_group(),
            2,
            &Recycled::scalar(0.125),
            None,
            &ave,
        )
        .unwrap();

        assert_eq!(prior.len(), 1);
        assert_relative_eq!(prior[0], 18.590_485_410_598_912, max_relative = PRIOR_TOL);
    }

    /// `update_prior` must be exactly two rounds, not one and not three. Running
    /// the loop by hand reproduces it, and stopping after the first round does
    /// not.
    #[test]
    fn test_update_prior_is_exactly_two_rounds() {
        let (counts, fitted, ave) = prior_counts();
        let run = |prior: &[f64]| {
            let adjusted = compute_adjust_vec(
                &counts,
                24,
                6,
                &fitted,
                &two_group(),
                2,
                &Recycled::scalar(0.125),
                prior,
                None,
            )
            .unwrap();
            compute_prior(&ave, &adjusted.s2, &adjusted.df).unwrap()
        };

        let first = run(&[1.0]);
        let second = run(&first);
        let got = update_prior(
            &counts,
            24,
            6,
            &fitted,
            &two_group(),
            2,
            &Recycled::scalar(0.125),
            None,
            &ave,
        )
        .unwrap();

        assert_eq!(got, second);
        assert!(first[0] != second[0]);
    }

    /// A dispersion of `2^-20`, which is the near-Poisson corner: small enough
    /// that [`unit_nb_deviance`] takes its Poisson expansion, while
    /// [`compute_weight`] still uses the case-1 Chebyshev fit, exactly as edgeR
    /// does.
    ///
    /// A dispersion of exactly zero is deliberately not tested here. edgeR's
    /// `compute_weight` has no Poisson branch at all: `phi = 0` falls into
    /// `anbinomdevc_1(mu, 0)`, whose fit already carries `pois_alpha` as a
    /// factor. [`compute_weight`] in this crate diverts `phi <= 0` to the bare
    /// Poisson fit instead, which disagrees with edgeR by 4.7e-4 relative on
    /// gene 0 of this fixture. That belongs to [`crate::ql::chebyshev`], not
    /// here.
    ///
    /// ```r
    /// d0 <- edgeR:::.compressDispersions(y, 2^-20)
    /// .Call(edgeR:::.cxx_compute_adj_vec, y, mu, X, d0, 1.0, wm)
    /// ```
    #[test]
    fn test_near_poisson_dispersion_matches_edger() {
        check(
            &two_group(),
            Recycled::scalar(2.0_f64.powi(-20)),
            &[1.0],
            None,
            &[
                0.360_893_151_979_798_77,
                0.122_110_248_576_896_45,
                12.329_306_966_107_474,
                0.396_987_455_932_764_3,
                0.0,
                11.686_733_868_487_11,
            ],
            &[
                3.991_092_303_143_825_7,
                3.999_478_990_707_440_4,
                4.976_310_706_206_089,
                1.997_955_076_334_263_3,
                0.0,
                4.000_035_625_925_281,
            ],
            &[
                0.090_424_656_852_841_87,
                0.030_531_538_948_101_14,
                2.477_599_911_663_728_6,
                0.198_696_887_950_621_38,
                0.0,
                2.921_657_445_434_316,
            ],
        );
    }

    /////////////////////
    // Error branches  //
    /////////////////////

    /// Every rejection [`compute_adjust_vec`] can produce, in one table.
    #[test]
    fn test_rejects_bad_shapes() {
        let run = |n_genes: usize,
                   n_samples: usize,
                   counts: &[f64],
                   fitted: &[f64],
                   design: &[f64],
                   n_coef: usize| {
            compute_adjust_vec(
                counts,
                n_genes,
                n_samples,
                fitted,
                design,
                n_coef,
                &Recycled::scalar(0.125),
                &[1.0],
                None,
            )
            .unwrap_err()
        };

        assert!(matches!(
            run(0, 6, &[], &[], &two_group(), 2),
            EdgeErrors::EmptyCounts { .. }
        ));
        assert!(matches!(
            run(6, 0, &[], &[], &[], 2),
            EdgeErrors::EmptyCounts { .. }
        ));
        assert!(matches!(
            run(6, 6, &counts(), &fitted(), &[], 0),
            EdgeErrors::MustBePositive(_)
        ));
        assert!(matches!(
            run(5, 6, &counts(), &fitted(), &two_group(), 2),
            EdgeErrors::LengthMismatch { name: "counts", .. }
        ));
        assert!(matches!(
            run(6, 6, &counts(), &fitted()[..30], &two_group(), 2),
            EdgeErrors::LengthMismatch { name: "fitted", .. }
        ));
        assert!(matches!(
            run(6, 6, &counts(), &fitted(), &two_group(), 3),
            EdgeErrors::LengthMismatch { name: "design", .. }
        ));
        // Two samples, three coefficients: no residual degrees of freedom to
        // adjust, and the QR would be of a wide matrix.
        assert!(matches!(
            run(
                1,
                2,
                &[1.0, 2.0],
                &[1.0, 2.0],
                &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                3
            ),
            EdgeErrors::DesignNotFullRank { .. }
        ));
    }

    #[test]
    fn test_rejects_mismatched_dispersion() {
        let err = compute_adjust_vec(
            &counts(),
            6,
            6,
            &fitted(),
            &two_group(),
            2,
            &Recycled::by_gene(vec![0.125; 3]),
            &[1.0],
            None,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            EdgeErrors::LengthMismatch {
                name: "by_gene",
                ..
            }
        ));
    }

    #[test]
    fn test_rejects_mismatched_weights() {
        let weights = Recycled::by_sample(vec![1.0; 3]);
        let err = compute_adjust_vec(
            &counts(),
            6,
            6,
            &fitted(),
            &two_group(),
            2,
            &Recycled::scalar(0.125),
            &[1.0],
            Some(&weights),
        )
        .unwrap_err();
        assert!(matches!(
            err,
            EdgeErrors::LengthMismatch {
                name: "by_sample",
                ..
            }
        ));
    }

    #[test]
    fn test_rejects_wrong_prior_length() {
        let err = compute_adjust_vec(
            &counts(),
            6,
            6,
            &fitted(),
            &two_group(),
            2,
            &Recycled::scalar(0.125),
            &[1.0, 2.0],
            None,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            EdgeErrors::LengthMismatch { name: "prior", .. }
        ));
    }

    #[test]
    fn test_rejects_non_positive_prior() {
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let err = compute_adjust_vec(
                &counts(),
                6,
                6,
                &fitted(),
                &two_group(),
                2,
                &Recycled::scalar(0.125),
                &[bad],
                None,
            )
            .unwrap_err();
            assert!(matches!(err, EdgeErrors::InvalidArgument(_)));
        }
    }

    #[test]
    fn test_compute_prior_rejects_mismatched_lengths() {
        let err = compute_prior(&[1.0, 2.0], &[1.0], &[1.0, 1.0]).unwrap_err();
        assert!(matches!(err, EdgeErrors::LengthMismatch { name: "s2", .. }));

        let err = compute_prior(&[1.0, 2.0], &[1.0, 1.0], &[1.0]).unwrap_err();
        assert!(matches!(err, EdgeErrors::LengthMismatch { name: "df", .. }));
    }

    #[test]
    fn test_update_prior_rejects_mismatched_ave_log_cpm() {
        let err = update_prior(
            &counts(),
            6,
            6,
            &fitted(),
            &two_group(),
            2,
            &Recycled::scalar(0.125),
            None,
            &[0.0, 1.0],
        )
        .unwrap_err();
        assert!(matches!(
            err,
            EdgeErrors::LengthMismatch {
                name: "ave_log_cpm",
                ..
            }
        ));
    }

    /// Regression guard for the zero-dispersion dispatch.
    ///
    /// `phi = 0` must reach the case 1 Chebyshev fits, not the Poisson ones.
    /// An earlier version of `ql/chebyshev.rs` short circuited to the Poisson
    /// branch and was wrong by 1.8e-4 relative here. edgeR reaches this through
    /// `glmQLFit(dispersion = 0)` and through any zero entry in a tagwise
    /// dispersion vector, so it is not a corner case.
    ///
    /// ```r
    /// y <- matrix(c(8,16,32,64,128,256, 4,4,8,8,16,16, 64,64,64,128,128,128),
    ///             nrow = 3, byrow = TRUE)
    /// X <- cbind(1, c(0,0,0,1,1,1))
    /// f <- glmQLFit(y, design = X, dispersion = 0,
    ///               offset = matrix(log(1024), 3, 6), legacy = FALSE)
    /// f$deviance.adj      # 147.92726615926262 11.90464907383647 0
    /// f$df.residual.adj   # 5.3435958730008828 9.1582038968715125 3.9057308640743664
    /// ```
    #[test]
    fn test_zero_dispersion_matches_edger() {
        let counts: Vec<f64> = vec![
            8.0, 16.0, 32.0, 64.0, 128.0, 256.0, //
            4.0, 4.0, 8.0, 8.0, 16.0, 16.0, //
            64.0, 64.0, 64.0, 128.0, 128.0, 128.0, //
        ];
        let fitted: Vec<f64> = vec![
            18.666_666_666_666_668,
            18.666_666_666_666_668,
            18.666_666_666_666_668,
            149.333_333_333_333_4,
            149.333_333_333_333_4,
            149.333_333_333_333_4,
            5.333_333_333_333_334,
            5.333_333_333_333_334,
            5.333_333_333_333_334,
            13.333_333_333_333_334,
            13.333_333_333_333_334,
            13.333_333_333_333_334,
            64.000_000_000_000_03,
            64.000_000_000_000_03,
            64.000_000_000_000_03,
            128.000_000_000_000_09,
            128.000_000_000_000_09,
            128.000_000_000_000_09,
        ];
        let design = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let out = compute_adjust_vec(
            &counts,
            3,
            6,
            &fitted,
            &design,
            2,
            &Recycled::scalar(0.0),
            &[15.741_184_694_250_476],
            None,
        )
        .unwrap();

        let expected_deviance = [147.927_266_159_262_62, 11.904_649_073_836_47, 0.0];
        let expected_df = [
            5.343_595_873_000_883,
            9.158_203_896_871_512,
            3.905_730_864_074_366_4,
        ];
        for (got, want) in out.deviance.iter().zip(expected_deviance.iter()) {
            assert_relative_eq!(got, want, epsilon = 1e-12, max_relative = 1e-12);
        }
        for (got, want) in out.df.iter().zip(expected_df.iter()) {
            assert_relative_eq!(got, want, max_relative = 1e-12);
        }
    }
}
