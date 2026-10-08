//! Negative binomial generalised linear models, fitted per gene.
//!
//! Each gene's working set is `n_samples` by `n_coef`, small enough to stay in
//! L1, so the fits run as rayon over genes with a per-thread scratch buffer.

pub mod deviance;
pub mod fit;
pub mod levenberg;
pub mod one_group;
pub mod one_way;
pub mod ql_fit;
pub mod test;

////////////
// Consts //
////////////

/// Bound on the linear predictor before exponentiating.
///
/// `exp(710)` overflows a double. Clamping at 500 keeps the fitted mean finite
/// through rejected steps and never binds at convergence. Same as edgeR.
pub(crate) const ETA_CLAMP: f64 = 500.0;

/// Floor applied to fitted means and working weights.
///
/// Both appear in denominators. Never binds on real data; keeps reciprocals finite.
pub(crate) const MIN_POSITIVE: f64 = 1e-300;

/// Coefficient, or starting log-rate, assigned to a gene with nothing to fit.
///
/// `log(0)` would poison every downstream sum. edgeR substitutes this value
/// (fitted mean about `2e-9`). Used as the empty-gene coefficient in
/// [`crate::glm::one_group`] and the empty-gene starting log-rate in
/// [`crate::glm::levenberg`].
pub(crate) const EMPTY_GENE_COEF: f64 = -20.0;
