//! Shared trait boundaries for the crate.
//!
//! `EdgeFloat` is the bound every algorithm in `edge-rs` is generic over. It
//! deliberately mirrors `AnnSearchFloat` and `BixverseFloat` in the sister
//! crates so that wiring `edge-rs` into `bixverse-rs` later is mechanical.
//!
//! Note the split documented in the crate root: `EdgeFloat` covers the data
//! layer (counts, offsets, fitted values), while likelihood evaluation,
//! Cox-Reid determinants, the optimisers and every p-value run in `f64`
//! regardless of `T`. Single-cell counts are worth holding as `f32` to halve
//! memory; a negative binomial deviance accumulated in `f32` is not.

use std::fmt::Display;
use std::iter::Sum;
use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};

use faer::traits::{ComplexField, RealField};
use num_traits::float::TotalOrder;
use num_traits::{Float, FromPrimitive, ToPrimitive};

use crate::utils::simd::EdgeSimd;

/// Floating-point types usable as the numeric type of an `edge-rs` algorithm.
///
/// This is a blanket-implemented marker: it requires no methods of its own and
/// exists purely to collapse a long bound list into one name. `f32` and `f64`
/// both satisfy it.
///
/// The bounds break down as: `Float`/`FromPrimitive`/`ToPrimitive` for the
/// numerics, `ComplexField`/`RealField` for faer interoperability, `Send`/`Sync`
/// for the rayon fan-out over genes, `EdgeSimd` for the vectorised kernels, and
/// `TotalOrder` for the sorts in normalisation, which must be deterministic in
/// the presence of NaN rather than merely fast.
pub trait EdgeFloat:
    Float
    + FromPrimitive
    + ToPrimitive
    + Send
    + Sync
    + Sum
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + EdgeSimd
    + ComplexField
    + RealField
    + TotalOrder
    + Display
    + Default
    + 'static
{
}

impl<T> EdgeFloat for T where
    T: Float
        + FromPrimitive
        + ToPrimitive
        + Send
        + Sync
        + Sum
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + EdgeSimd
        + ComplexField
        + RealField
        + TotalOrder
        + Display
        + Default
        + 'static
{
}
