//! Shared trait boundaries for the crate.
//!
//! `EdgeFloat` is the bound every algorithm in `edge-rs` is generic over. It
//! covers the data layer only; the `f64` policy for likelihoods is in the crate
//! root.

use std::fmt::Display;
use std::iter::Sum;
use std::ops::{AddAssign, DivAssign, MulAssign, SubAssign};

use faer::traits::{ComplexField, RealField};
use num_traits::float::TotalOrder;
use num_traits::{Float, FromPrimitive, ToPrimitive};

use crate::utils::simd::EdgeSimd;

/// Floating-point types usable as the numeric type of an `edge-rs` algorithm.
///
/// A blanket-implemented marker that collapses the bound list into one name;
/// `f32` and `f64` satisfy it.
///
/// `Float`/`FromPrimitive`/`ToPrimitive` cover the numerics,
/// `ComplexField`/`RealField` faer, `Send`/`Sync` rayon, `EdgeSimd` the
/// vectorised kernels, and `TotalOrder` the sorts in normalisation, which must
/// be deterministic with NaN.
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
