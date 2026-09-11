//! The limma routines edgeR depends on, and the linear model stack on top.
//!
//! Not a general limma port. It covers what `estimateDisp`, `glmQLFit` and
//! `voomLmFit` reach for, plus the chain that turns a fit into a table:
//! `lmFit` -> `contrasts.fit` -> `eBayes` -> `topTable`, carried through by
//! [`marray::MArrayLm`], and `removeBatchEffect` on top of `lmFit`.

use crate::prelude::*;

pub mod array_weights;
pub mod contrasts;
pub mod ebayes;
pub mod lm_fit;
pub mod lowess;
pub mod marray;
pub mod remove_batch_effect;
pub mod smoothing;
pub mod squeeze_var;
pub mod toptable;
pub mod voom;

/// Checks that prior weights are non-negative.
///
/// ### Params
///
/// * `w` - Prior weights to check
///
/// ### Returns
///
/// `Ok(())` when every entry is non-negative, otherwise
/// [`EdgeErrors::InvalidArgument`] naming the first offending index.
pub(crate) fn check_nonneg_weights(w: &[f64]) -> Result<(), EdgeErrors> {
    if let Some(i) = w.iter().position(|v| *v < 0.0) {
        return Err(EdgeErrors::InvalidArgument(format!(
            "prior weights must be non-negative; weights[{i}] is {}",
            w[i]
        )));
    }
    Ok(())
}
