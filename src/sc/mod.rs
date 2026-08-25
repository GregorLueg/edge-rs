//! NEBULA: negative binomial mixed models for single cell.
//!
//! Ported from the `nebula` R package sources rather than from edgePython.
//! edgePython implements only NEBULA-LN and its standard errors are 6 to 89 per
//! cent away from the R package, which is not usable for inference.

pub mod nebula;
pub mod pml;
pub mod ptmg;
pub mod shrink;
pub mod test;
