//! Negative binomial generalised linear models, fitted per gene.
//!
//! Genes are the parallel axis and each gene's working set is `n_samples` by
//! `n_coef`, small enough to stay in L1, so the fan-out is rayon over genes with
//! a per-thread scratch buffer rather than the batched matrix operations
//! edgePython needs to go fast in NumPy.

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
/// through the wild excursions a rejected step can produce, without touching
/// any value a converged fit would reach. edgeR clamps at the same place.
pub(crate) const ETA_CLAMP: f64 = 500.0;

/// Floor applied to fitted means and working weights.
///
/// Both appear in denominators. This is small enough never to bind on real data
/// and large enough to keep the reciprocal finite.
pub(crate) const MIN_POSITIVE: f64 = 1e-300;

/// Coefficient, or starting log-rate, assigned to a gene with nothing to fit.
///
/// `log(0)` is negative infinity, which would poison every downstream sum.
/// edgeR substitutes a large negative number, giving a fitted mean of about
/// `2e-9` counts, indistinguishable from zero in practice and finite in
/// arithmetic. The same value serves as both the empty-gene coefficient in
/// [`crate::glm::one_group`] and the empty-gene starting log-rate in
/// [`crate::glm::levenberg`].
pub(crate) const EMPTY_GENE_COEF: f64 = -20.0;
