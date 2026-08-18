//! SIMD kernels behind the `EdgeSimd` trait.
//!
//! Algorithm code stays generic over `T: EdgeFloat` and calls `T::dot_simd(..)`
//! and friends. The vector width lives here and nowhere else; no intrinsics or
//! `wide` types appear outside this file.
//!
//! The primitives come from `wide`, which compiles to the SSE2/NEON baseline
//! and lets LLVM widen further under `-C target-cpu`. There is deliberately no
//! runtime AVX-512 dispatch yet: it belongs in this file when a benchmark asks
//! for it, not before.
//!
//! Only general primitives live here. Fused kernels (negative binomial unit
//! deviance, the working-weight outer product) arrive with the modules that
//! consume them, so they can be measured against the scalar form rather than
//! assumed faster.

use wide::{f32x8, f64x4};

/// Number of independent accumulators in the reduction loops.
///
/// A single accumulator serialises on the FMA latency chain. Four independent
/// ones keep the pipeline fed and is the point of diminishing returns on both
/// Apple Silicon and x86: eight costs register pressure without buying
/// throughput. This matches the constant `bixverse-rs` settled on.
const UNROLL: usize = 4;

/// SIMD-accelerated primitives over slices of a float type.
///
/// Implemented once per concrete float type; algorithm code never dispatches on
/// the type itself.
pub trait EdgeSimd: Sized + Copy {
    /// Dot product of two equal-length slices.
    ///
    /// ### Params
    ///
    /// * `a` - Left operand
    /// * `b` - Right operand, must be the same length as `a`
    ///
    /// ### Returns
    ///
    /// The sum of the elementwise products. Excess elements beyond `b`'s length
    /// are ignored, so a length mismatch truncates rather than panics.
    fn dot_simd(a: &[Self], b: &[Self]) -> Self;

    /// Sum of a slice.
    ///
    /// ### Params
    ///
    /// * `x` - Values to reduce
    ///
    /// ### Returns
    ///
    /// The sum. Note that the multi-accumulator layout changes the summation
    /// order relative to a scalar left fold, so results can differ in the last
    /// ulp. Nothing in the crate depends on that ordering.
    fn sum_simd(x: &[Self]) -> Self;

    /// Scaled vector addition, `y += a * x`, in place.
    ///
    /// ### Params
    ///
    /// * `y` - Destination, modified in place
    /// * `a` - Scalar multiplier
    /// * `x` - Source, must be at least as long as `y`
    fn axpy_simd(y: &mut [Self], a: Self, x: &[Self]);

    /// Exponentiates a slice in place.
    ///
    /// This is the `mu = exp(eta)` step of every GLM fit in the crate, so it
    /// sits on the hottest path there is.
    ///
    /// ### Params
    ///
    /// * `x` - Values to exponentiate, modified in place
    fn exp_in_place_simd(x: &mut [Self]);
}

//////////////////
// f64 kernels  //
//////////////////

/// Dot product of two `f64` slices.
///
/// ### Params
///
/// * `a` - Left operand
/// * `b` - Right operand
///
/// ### Returns
///
/// The sum of elementwise products.
#[inline]
fn dot_simd_f64(a: &[f64], b: &[f64]) -> f64 {
    const LANES: usize = 4;
    const BLOCK: usize = LANES * UNROLL;

    let n = a.len().min(b.len());
    let mut acc = [f64x4::ZERO; UNROLL];
    let mut i = 0;

    while i + BLOCK <= n {
        for (u, acc_u) in acc.iter_mut().enumerate() {
            let off = i + u * LANES;
            let va = f64x4::from(&a[off..off + LANES]);
            let vb = f64x4::from(&b[off..off + LANES]);
            *acc_u += va * vb;
        }
        i += BLOCK;
    }

    let mut total = f64x4::ZERO;
    for acc_u in acc.iter() {
        total += *acc_u;
    }
    let mut sum = total.reduce_add();

    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

/// Sum of an `f64` slice.
///
/// ### Params
///
/// * `x` - Values to reduce
///
/// ### Returns
///
/// The sum.
#[inline]
fn sum_simd_f64(x: &[f64]) -> f64 {
    const LANES: usize = 4;
    const BLOCK: usize = LANES * UNROLL;

    let n = x.len();
    let mut acc = [f64x4::ZERO; UNROLL];
    let mut i = 0;

    while i + BLOCK <= n {
        for (u, acc_u) in acc.iter_mut().enumerate() {
            let off = i + u * LANES;
            *acc_u += f64x4::from(&x[off..off + LANES]);
        }
        i += BLOCK;
    }

    let mut total = f64x4::ZERO;
    for acc_u in acc.iter() {
        total += *acc_u;
    }
    let mut sum = total.reduce_add();

    while i < n {
        sum += x[i];
        i += 1;
    }
    sum
}

/// In-place `y += a * x` over `f64` slices.
///
/// ### Params
///
/// * `y` - Destination
/// * `a` - Scalar multiplier
/// * `x` - Source
#[inline]
fn axpy_simd_f64(y: &mut [f64], a: f64, x: &[f64]) {
    const LANES: usize = 4;

    let n = y.len().min(x.len());
    let va = f64x4::splat(a);
    let mut i = 0;

    while i + LANES <= n {
        let vy = f64x4::from(&y[i..i + LANES]);
        let vx = f64x4::from(&x[i..i + LANES]);
        let out = vy + va * vx;
        y[i..i + LANES].copy_from_slice(out.as_array());
        i += LANES;
    }

    while i < n {
        y[i] += a * x[i];
        i += 1;
    }
}

/// In-place exponential over an `f64` slice.
///
/// ### Params
///
/// * `x` - Values to exponentiate
#[inline]
fn exp_in_place_f64(x: &mut [f64]) {
    const LANES: usize = 4;

    let n = x.len();
    let mut i = 0;

    while i + LANES <= n {
        let v = f64x4::from(&x[i..i + LANES]).exp();
        x[i..i + LANES].copy_from_slice(v.as_array());
        i += LANES;
    }

    while i < n {
        x[i] = x[i].exp();
        i += 1;
    }
}

//////////////////
// f32 kernels  //
//////////////////

/// Dot product of two `f32` slices.
///
/// ### Params
///
/// * `a` - Left operand
/// * `b` - Right operand
///
/// ### Returns
///
/// The sum of elementwise products.
#[inline]
fn dot_simd_f32(a: &[f32], b: &[f32]) -> f32 {
    const LANES: usize = 8;
    const BLOCK: usize = LANES * UNROLL;

    let n = a.len().min(b.len());
    let mut acc = [f32x8::ZERO; UNROLL];
    let mut i = 0;

    while i + BLOCK <= n {
        for (u, acc_u) in acc.iter_mut().enumerate() {
            let off = i + u * LANES;
            let va = f32x8::from(&a[off..off + LANES]);
            let vb = f32x8::from(&b[off..off + LANES]);
            *acc_u += va * vb;
        }
        i += BLOCK;
    }

    let mut total = f32x8::ZERO;
    for acc_u in acc.iter() {
        total += *acc_u;
    }
    let mut sum = total.reduce_add();

    while i < n {
        sum += a[i] * b[i];
        i += 1;
    }
    sum
}

/// Sum of an `f32` slice.
///
/// ### Params
///
/// * `x` - Values to reduce
///
/// ### Returns
///
/// The sum.
#[inline]
fn sum_simd_f32(x: &[f32]) -> f32 {
    const LANES: usize = 8;
    const BLOCK: usize = LANES * UNROLL;

    let n = x.len();
    let mut acc = [f32x8::ZERO; UNROLL];
    let mut i = 0;

    while i + BLOCK <= n {
        for (u, acc_u) in acc.iter_mut().enumerate() {
            let off = i + u * LANES;
            *acc_u += f32x8::from(&x[off..off + LANES]);
        }
        i += BLOCK;
    }

    let mut total = f32x8::ZERO;
    for acc_u in acc.iter() {
        total += *acc_u;
    }
    let mut sum = total.reduce_add();

    while i < n {
        sum += x[i];
        i += 1;
    }
    sum
}

/// In-place `y += a * x` over `f32` slices.
///
/// ### Params
///
/// * `y` - Destination
/// * `a` - Scalar multiplier
/// * `x` - Source
#[inline]
fn axpy_simd_f32(y: &mut [f32], a: f32, x: &[f32]) {
    const LANES: usize = 8;

    let n = y.len().min(x.len());
    let va = f32x8::splat(a);
    let mut i = 0;

    while i + LANES <= n {
        let vy = f32x8::from(&y[i..i + LANES]);
        let vx = f32x8::from(&x[i..i + LANES]);
        let out = vy + va * vx;
        y[i..i + LANES].copy_from_slice(out.as_array());
        i += LANES;
    }

    while i < n {
        y[i] += a * x[i];
        i += 1;
    }
}

/// In-place exponential over an `f32` slice.
///
/// ### Params
///
/// * `x` - Values to exponentiate
#[inline]
fn exp_in_place_f32(x: &mut [f32]) {
    const LANES: usize = 8;

    let n = x.len();
    let mut i = 0;

    while i + LANES <= n {
        let v = f32x8::from(&x[i..i + LANES]).exp();
        x[i..i + LANES].copy_from_slice(v.as_array());
        i += LANES;
    }

    while i < n {
        x[i] = x[i].exp();
        i += 1;
    }
}

/////////////////////
// Implementations //
/////////////////////

impl EdgeSimd for f64 {
    #[inline]
    fn dot_simd(a: &[f64], b: &[f64]) -> f64 {
        dot_simd_f64(a, b)
    }

    #[inline]
    fn sum_simd(x: &[f64]) -> f64 {
        sum_simd_f64(x)
    }

    #[inline]
    fn axpy_simd(y: &mut [f64], a: f64, x: &[f64]) {
        axpy_simd_f64(y, a, x)
    }

    #[inline]
    fn exp_in_place_simd(x: &mut [f64]) {
        exp_in_place_f64(x)
    }
}

impl EdgeSimd for f32 {
    #[inline]
    fn dot_simd(a: &[f32], b: &[f32]) -> f32 {
        dot_simd_f32(a, b)
    }

    #[inline]
    fn sum_simd(x: &[f32]) -> f32 {
        sum_simd_f32(x)
    }

    #[inline]
    fn axpy_simd(y: &mut [f32], a: f32, x: &[f32]) {
        axpy_simd_f32(y, a, x)
    }

    #[inline]
    fn exp_in_place_simd(x: &mut [f32]) {
        exp_in_place_f32(x)
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Lengths chosen to straddle the block boundaries: below one block, exactly
    /// one block, and a block plus an awkward remainder.
    const SIZES: [usize; 6] = [0, 1, 7, 16, 32, 45];

    fn ramp_f64(n: usize) -> Vec<f64> {
        (0..n).map(|i| (i as f64) * 0.5 - 3.0).collect()
    }

    fn ramp_f32(n: usize) -> Vec<f32> {
        (0..n).map(|i| (i as f32) * 0.5 - 3.0).collect()
    }

    #[test]
    fn test_dot_simd_f64_matches_scalar() {
        for n in SIZES {
            let a = ramp_f64(n);
            let b: Vec<f64> = ramp_f64(n).iter().map(|v| v * 1.7 + 0.3).collect();
            let expected: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            assert_relative_eq!(f64::dot_simd(&a, &b), expected, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_dot_simd_f32_matches_scalar() {
        for n in SIZES {
            let a = ramp_f32(n);
            let b: Vec<f32> = ramp_f32(n).iter().map(|v| v * 1.7 + 0.3).collect();
            let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            assert_relative_eq!(f32::dot_simd(&a, &b), expected, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_sum_simd_matches_scalar() {
        for n in SIZES {
            let x = ramp_f64(n);
            let expected: f64 = x.iter().sum();
            assert_relative_eq!(f64::sum_simd(&x), expected, epsilon = 1e-12);

            let x = ramp_f32(n);
            let expected: f32 = x.iter().sum();
            assert_relative_eq!(f32::sum_simd(&x), expected, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_axpy_simd_matches_scalar() {
        for n in SIZES {
            let x = ramp_f64(n);
            let mut y = ramp_f64(n);
            let expected: Vec<f64> = y.iter().zip(x.iter()).map(|(a, b)| a + 2.5 * b).collect();
            f64::axpy_simd(&mut y, 2.5, &x);
            for (got, want) in y.iter().zip(expected.iter()) {
                assert_relative_eq!(got, want, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_exp_in_place_simd_matches_scalar() {
        for n in SIZES {
            let mut x = ramp_f64(n);
            let expected: Vec<f64> = x.iter().map(|v| v.exp()).collect();
            f64::exp_in_place_simd(&mut x);
            for (got, want) in x.iter().zip(expected.iter()) {
                assert_relative_eq!(got, want, max_relative = 1e-12);
            }

            let mut x = ramp_f32(n);
            let expected: Vec<f32> = x.iter().map(|v| v.exp()).collect();
            f32::exp_in_place_simd(&mut x);
            for (got, want) in x.iter().zip(expected.iter()) {
                assert_relative_eq!(got, want, max_relative = 1e-5);
            }
        }
    }

    /// A dot product of unequal lengths truncates to the shorter operand rather
    /// than reading out of bounds. Documented behaviour, so pin it.
    #[test]
    fn test_dot_simd_truncates_on_length_mismatch() {
        let a = vec![1.0_f64; 20];
        let b = vec![2.0_f64; 5];
        assert_relative_eq!(f64::dot_simd(&a, &b), 10.0, epsilon = 1e-12);
    }
}
