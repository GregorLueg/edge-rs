//! The limma routines edgeR depends on.
//!
//! Not a general limma port: only the pieces `estimateDisp`, `glmQLFit` and
//! `voomLmFit` actually reach for.

use crate::prelude::*;

pub mod array_weights;
pub mod lm_fit;
pub mod lowess;
pub mod smoothing;
pub mod squeeze_var;
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
