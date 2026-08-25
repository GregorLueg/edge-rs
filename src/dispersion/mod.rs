//! Dispersion estimation.
//!
//! The Cox-Reid adjusted profile likelihood and the estimators built on it:
//! common, trended and tagwise dispersions, and the weighted likelihood
//! empirical Bayes that shrinks between them.

use crate::prelude::*;

pub mod apl;
pub mod cox_reid;
pub mod estimate;

/// Drops genes carrying too little information about the dispersion.
///
/// Shared between [`cox_reid::disp_cox_reid`] and [`estimate::estimate_disp`],
/// which both filter on the same row-sum floor before fitting.
///
/// ### Params
///
/// * `counts` - Row-major counts, `n_genes * n_samples`
/// * `n_genes` - Number of genes
/// * `n_samples` - Number of samples
/// * `min_row_sum` - Minimum total count for a gene to be kept
///
/// ### Returns
///
/// Indices of the kept genes, or [`EdgeErrors::NoGenesAfterFiltering`] if none
/// survive.
pub(crate) fn filter_by_min_row_sum<T: EdgeFloat>(
    counts: &[T],
    n_genes: usize,
    n_samples: usize,
    min_row_sum: f64,
) -> Result<Vec<usize>, EdgeErrors> {
    let kept: Vec<usize> = counts
        .chunks_exact(n_samples)
        .enumerate()
        .filter(|(_, row)| row.iter().map(|v| v.to_f64().unwrap_or(0.0)).sum::<f64>() >= min_row_sum)
        .map(|(gene, _)| gene)
        .collect();
    if kept.is_empty() {
        return Err(EdgeErrors::NoGenesAfterFiltering { n_genes });
    }
    Ok(kept)
}
