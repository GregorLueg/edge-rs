//! Common dispersion by maximising the summed Cox-Reid adjusted profile
//! likelihood.
//!
//! edgeR's `dispCoxReid`, reached through `estimateGLMCommonDisp`. Unlike
//! [`crate::dispersion::estimate`], which reads a precomputed grid, this
//! optimises the dispersion directly with a bounded scalar search.
//!
//! The search runs over the fourth root of the dispersion, which spreads the
//! interval so a bounded Brent search resolves small dispersions as well as
//! large ones. And on large experiments a systematic subset of genes stratified
//! by abundance stands in for the whole set, which changes the answer, so it
//! has to be reproduced rather than skipped.

use crate::dispersion::apl::apl_at;
use crate::numeric::optimise::brent_fmin;
use crate::prelude::*;

////////////
// Consts //
////////////

/// Lower end of the search interval when the caller asks for zero.
///
/// The objective is optimised over `dispersion^(1/4)`, and a true zero would put
/// the search on the boundary where the derivative is undefined.
const MIN_ROOT: f64 = 1e-10;

///////////////////
// CoxReidParams //
///////////////////

/// Tuning knobs for [`common_dispersion_cox_reid`].
#[derive(Clone, Copy, Debug)]
pub struct CoxReidParams {
    /// Search interval for the dispersion itself, not its fourth root.
    pub interval: (f64, f64),
    /// Convergence tolerance passed to R's `optimize`, on the fourth root of
    /// the dispersion. edgeR's default is 1e-5.
    pub tol: f64,
    /// Genes whose total count falls below this are excluded.
    pub min_row_sum: f64,
    /// Number of genes to subsample on large experiments.
    ///
    /// Subsetting only kicks in when there are at least twice this many genes,
    /// matching edgeR. `None` disables it and uses every gene, which is more
    /// accurate and slower.
    pub subset: Option<usize>,
}

impl Default for CoxReidParams {
    /// edgeR's defaults.
    fn default() -> Self {
        Self {
            interval: (0.0, 4.0),
            tol: 1e-5,
            min_row_sum: 5.0,
            subset: Some(10_000),
        }
    }
}

///////////////
// Front-end //
///////////////

/// A systematic subset of indices stratified by a ranking variable.
///
/// Sorts by the ranking variable, then takes every `n_total / n`th element
/// starting from the middle of the first stratum. That spreads the subset evenly
/// across the abundance range instead of sampling it at random, so the estimate
/// is reproducible and not dominated by the low-count genes that make up most of
/// a typical experiment.
///
/// ### Params
///
/// * `n` - Desired subset size
/// * `order_by` - Ranking variable, one value per candidate
///
/// ### Returns
///
/// Indices into `order_by`, in ascending order of the ranking variable. Returns
/// everything when the requested size is not smaller than the input.
pub fn systematic_subset(n: usize, order_by: &[f64]) -> Vec<usize> {
    let total = order_by.len();
    if n == 0 || total == 0 {
        return Vec::new();
    }
    let ratio = total / n;
    if ratio <= 1 {
        return (0..total).collect();
    }

    let mut order: Vec<usize> = (0..total).collect();
    order.sort_by(|&a, &b| order_by[a].total_cmp(&order_by[b]));

    let start = ratio / 2;
    (start..total).step_by(ratio).map(|i| order[i]).collect()
}

/// Estimates a single dispersion shared by every gene.
///
/// Maximises the sum over genes of the Cox-Reid adjusted profile likelihood.
///
/// ### Params
///
/// * `counts` - Counts, row-major `n_genes * n_samples`
/// * `n_genes` - Number of genes
/// * `n_samples` - Number of samples
/// * `design` - Design matrix, row-major `n_samples * n_coef`
/// * `n_coef` - Number of coefficients
/// * `offset` - Log-scale offsets
/// * `weights` - Optional observation weights
/// * `ave_log_cpm` - Abundance per gene, required only when subsetting applies
/// * `params` - Tuning knobs, or [`CoxReidParams::default`]
///
/// ### Returns
///
/// The common dispersion, or [`EdgeErrors`] if the shapes disagree, no gene
/// survives filtering, or subsetting was requested without an abundance to
/// stratify by.
#[allow(clippy::too_many_arguments)]
pub fn common_dispersion_cox_reid<T: EdgeFloat>(
    counts: &[T],
    n_genes: usize,
    n_samples: usize,
    design: &[f64],
    n_coef: usize,
    offset: &Recycled<f64>,
    weights: Option<&Recycled<f64>>,
    ave_log_cpm: Option<&[f64]>,
    params: Option<CoxReidParams>,
) -> Result<f64, EdgeErrors> {
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
    if params.interval.0 < 0.0 || params.interval.1 <= params.interval.0 {
        return Err(EdgeErrors::InvalidArgument(format!(
            "dispersion interval must be non-negative and increasing, got {:?}",
            params.interval
        )));
    }

    // Drop genes carrying no information about the dispersion.
    let kept = crate::dispersion::filter_by_min_row_sum(
        counts,
        n_genes,
        n_samples,
        params.min_row_sum,
    )?;

    // Subsample on large experiments, exactly as edgeR does.
    let chosen: Vec<usize> = match params.subset {
        Some(target) if target <= kept.len() / 2 => {
            let abundance = ave_log_cpm.ok_or_else(|| {
                EdgeErrors::InvalidArgument(
                    "subsetting needs ave_log_cpm to stratify by; supply it or set subset to None"
                        .to_string(),
                )
            })?;
            if abundance.len() != n_genes {
                return Err(EdgeErrors::LengthMismatch {
                    name: "ave_log_cpm",
                    expected: n_genes,
                    got: abundance.len(),
                });
            }
            let kept_abundance: Vec<f64> = kept.iter().map(|&g| abundance[g]).collect();
            systematic_subset(target, &kept_abundance)
                .into_iter()
                .map(|i| kept[i])
                .collect()
        }
        _ => kept,
    };

    let mut selected = Vec::with_capacity(chosen.len() * n_samples);
    for &gene in &chosen {
        selected.extend_from_slice(&counts[gene * n_samples..(gene + 1) * n_samples]);
    }
    let selected_offset = offset.subset(&chosen, n_samples);
    let selected_weights = weights.map(|w| w.subset(&chosen, n_samples));

    // Optimise over the fourth root of the dispersion.
    let lower = params.interval.0.powf(0.25).max(MIN_ROOT);
    let upper = params.interval.1.powf(0.25);

    let mut evaluation_error = None;
    let objective = |root: f64| -> f64 {
        let dispersion = root.powi(4);
        match apl_at(
            &selected,
            chosen.len(),
            n_samples,
            design,
            n_coef,
            dispersion,
            &selected_offset,
            selected_weights.as_ref(),
        ) {
            Ok(apl) => -apl.iter().sum::<f64>(),
            Err(e) => {
                evaluation_error = Some(e);
                f64::INFINITY
            }
        }
    };

    let root = brent_fmin(lower, upper, objective, params.tol);
    if let Some(e) = evaluation_error {
        return Err(e);
    }

    Ok(root.powi(4))
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn fixture() -> (Vec<f64>, Vec<f64>, Recycled<f64>) {
        let counts = vec![
            10.0, 12.0, 11.0, 40.0, 44.0, 38.0, //
            50.0, 48.0, 52.0, 49.0, 51.0, 50.0, //
            2.0, 0.0, 5.0, 1.0, 3.0, 0.0, //
            100.0, 90.0, 110.0, 300.0, 280.0, 320.0, //
            7.0, 9.0, 6.0, 8.0, 7.0, 10.0, //
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
    /// y <- matrix(c(10,12,11,40,44,38, 50,48,52,49,51,50, 2,0,5,1,3,0,
    ///               100,90,110,300,280,320, 7,9,6,8,7,10), nrow = 5, byrow = TRUE)
    /// X <- cbind(1, c(0,0,0,1,1,1))
    /// lib <- c(1e6,1.2e6,0.9e6,1.1e6,1e6,1.3e6)
    /// off <- matrix(rep(log(lib), each = 5), nrow = 5)
    /// edgeR:::dispCoxReid(y, design = X, offset = off)
    /// ```
    #[test]
    fn test_matches_edger_disp_cox_reid() {
        let (counts, design, offset) = fixture();
        let got = common_dispersion_cox_reid(&counts, 5, 6, &design, 2, &offset, None, None, None)
            .unwrap();

        assert_relative_eq!(got, 0.013_404_409_326_629_5, max_relative = 1e-9);
    }

    /// The optimiser should land where the summed adjusted profile likelihood
    /// actually peaks, so a coarse grid search must agree with it.
    #[test]
    fn test_agrees_with_a_grid_search() {
        let (counts, design, offset) = fixture();
        let fitted =
            common_dispersion_cox_reid(&counts, 5, 6, &design, 2, &offset, None, None, None)
                .unwrap();

        let summed = |d: f64| -> f64 {
            apl_at(&counts, 5, 6, &design, 2, d, &offset, None)
                .unwrap()
                .iter()
                .sum()
        };
        let at_optimum = summed(fitted);
        for factor in [0.5, 0.8, 1.25, 2.0] {
            assert!(
                summed(fitted * factor) <= at_optimum,
                "dispersion {} beat the fitted {}",
                fitted * factor,
                fitted
            );
        }
    }

    #[test]
    fn test_systematic_subset_spreads_across_the_range() {
        let order_by: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let picked = systematic_subset(10, &order_by);
        assert_eq!(picked.len(), 10);
        // Ratio is 10, so the first pick is index 5 and they step by 10.
        assert_eq!(picked[0], 5);
        assert_eq!(picked[1], 15);
        assert_eq!(picked[9], 95);
    }

    #[test]
    fn test_systematic_subset_returns_everything_when_not_smaller() {
        let order_by: Vec<f64> = (0..10).map(|i| i as f64).collect();
        assert_eq!(systematic_subset(10, &order_by).len(), 10);
        assert_eq!(systematic_subset(20, &order_by).len(), 10);
    }

    #[test]
    fn test_subsetting_needs_an_abundance_to_stratify_by() {
        let (counts, design, offset) = fixture();
        let params = CoxReidParams {
            subset: Some(1),
            ..Default::default()
        };
        let err = common_dispersion_cox_reid(
            &counts,
            5,
            6,
            &design,
            2,
            &offset,
            None,
            None,
            Some(params),
        )
        .unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));
    }

    #[test]
    fn test_rejects_an_inverted_interval() {
        let (counts, design, offset) = fixture();
        let params = CoxReidParams {
            interval: (4.0, 1.0),
            ..Default::default()
        };
        let err = common_dispersion_cox_reid(
            &counts,
            5,
            6,
            &design,
            2,
            &offset,
            None,
            None,
            Some(params),
        )
        .unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));
    }

    #[test]
    fn test_rejects_when_no_gene_survives_filtering() {
        let counts = vec![0.0; 12];
        let design = vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let err = common_dispersion_cox_reid(
            &counts,
            3,
            4,
            &design,
            2,
            &Recycled::scalar(0.0),
            None,
            None,
            None,
        )
        .unwrap_err();
        assert!(matches!(err, EdgeErrors::NoGenesAfterFiltering { .. }));
    }
}
