//! The limma routines edgeR depends on.
//!
//! Not a general limma port: only the pieces `estimateDisp`, `glmQLFit` and
//! `voomLmFit` actually reach for.

pub mod array_weights;
pub mod lm_fit;
pub mod lowess;
pub mod smoothing;
pub mod squeeze_var;
pub mod voom;
