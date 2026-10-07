//! NEBULA: negative binomial mixed models for single cell.
//!
//! Ported from the `nebula` package sources, not edgePython, whose standard
//! errors are 6 to 89 per cent off. See `UPSTREAM_DEVIATIONS.md` A20.

pub mod nebula;
pub mod pml;
pub mod ptmg;
pub mod shrink;
pub mod test;
