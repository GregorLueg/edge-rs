//! Dispersion estimation.
//!
//! The Cox-Reid adjusted profile likelihood and the estimators built on it:
//! common, trended and tagwise dispersions, and the weighted likelihood
//! empirical Bayes that shrinks between them.

pub mod apl;
pub mod cox_reid;
pub mod estimate;
