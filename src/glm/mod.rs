//! Negative binomial generalised linear models, fitted per gene.
//!
//! Genes are the parallel axis and each gene's working set is `n_samples` by
//! `n_coef`, small enough to stay in L1, so the fan-out is rayon over genes with
//! a per-thread scratch buffer rather than the batched matrix operations
//! edgePython needs to go fast in NumPy.

pub mod deviance;
pub mod levenberg;
pub mod one_group;
pub mod one_way;
