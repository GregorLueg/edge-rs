//! Negative binomial differential expression for bulk and single-cell RNA-seq.
//!
//! A Rust port of the edgeR numerical stack (normalisation, Cox-Reid dispersion
//! estimation, the Levenberg-damped negative binomial GLM, quasi-likelihood
//! weights, the exact test), the parts of limma edgeR leans on (`squeezeVar`,
//! the F-distribution fits, voom), and NEBULA, a negative binomial gamma mixed
//! model for single cell.
//!
//! The bulk stack follows edgePython, itself a port of edgeR and limma, with the
//! R winning wherever the two disagree. NEBULA is ported from the `nebula`
//! package's own C++ instead, for the reasons in `UPSTREAM_DEVIATIONS.md`.
//!
//! ### Numeric policy
//!
//! Public containers and fits are generic over [`prelude::EdgeFloat`], so
//! single-cell counts can be held as `f32` and halve the memory. Likelihood
//! evaluation, Cox-Reid log-determinants, the optimisers and every p-value run
//! in `f64` regardless of `T`. A negative binomial deviance is a difference of
//! large logs, and accumulating one in `f32` loses parity with edgeR well before
//! it saves anything worth having.
//!
//! ### Parallelism
//!
//! Genes are the parallel axis almost everywhere. Counts are stored gene-major,
//! so one gene is a contiguous slice, and the fan-out is a rayon iterator over
//! genes with a per-thread scratch buffer.
//!
//! Normalisation is the exception, and deliberately so: TMM has no gene axis to
//! fan out over, since the unit of work is one sample compared against a
//! reference column. It parallelises over samples and transposes the counts once
//! on entry so each sample is contiguous. Any module that departs from the
//! gene-major rule says so in its own header.

#![warn(missing_docs)]

pub mod errors;
pub mod numeric;
pub mod prelude;
pub mod utils;
