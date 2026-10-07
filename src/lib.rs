//! Negative binomial differential expression for bulk and single-cell RNA-seq.
//!
//! A Rust port of the edgeR numerical stack (normalisation, Cox-Reid dispersion
//! estimation, the Levenberg-damped negative binomial GLM, quasi-likelihood
//! weights, the exact test), the limma linear model stack (`squeezeVar`, the
//! F-distribution fits, voom, `lmFit`, `contrasts.fit`, `eBayes`, `topTable`),
//! and NEBULA, a negative binomial gamma mixed model for single cell.
//!
//! The bulk stack follows edgePython, itself a port of edgeR and limma. Where
//! the two disagree, edgeR and limma win. NEBULA is ported from the `nebula`
//! package's own C++.
//!
//! ### Numeric policy
//!
//! Public containers and fits are generic over [`prelude::EdgeFloat`], so
//! single-cell counts can be held as `f32` and halve the memory. Likelihood
//! evaluation, Cox-Reid log-determinants, the optimisers and every p-value run
//! in `f64` regardless of `T`: an NB deviance is a difference of large logs and
//! loses edgeR parity in `f32` long before it saves anything.
//!
//! ### Parallelism
//!
//! Genes are the parallel axis almost everywhere. Counts are gene-major, so one
//! gene is a contiguous slice, and the fan-out is a rayon iterator over genes
//! with a per-thread scratch buffer.
//!
//! Normalisation is the exception: TMM's unit of work is one sample against a
//! reference column, so it parallelises over samples and transposes the counts
//! once on entry. Any module departing from the gene-major rule says so in its
//! own header.

#![warn(missing_docs)]

pub mod core;
pub mod dispersion;
pub mod errors;
pub mod exact;
pub mod glm;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod limma;
pub mod numeric;
pub mod prelude;
pub mod ql;
pub mod results;
pub mod sc;
pub mod splicing;
pub mod utils;
