//! SIMD kernels behind the `EdgeSimd` trait.
//!
//! The optimiser core (`glm::levenberg`) calls `f64::dot_simd(..)` and
//! `f64::exp_in_place_simd(..)` directly, since likelihoods and optimisers work
//! in f64 regardless of the crate's `T: EdgeFloat` data layer. The vector width
//! lives here and nowhere else; no intrinsics or `wide` types appear outside
//! this file.
//!
//! Each operation exists at four widths: scalar, 128-bit, 256-bit and 512-bit.
//! [`detect_simd_level`] caches what the machine supports and the trait impls
//! dispatch on it. A tier vectorises what it can at its own width, then hands
//! the remainder down to the tier below. That ladder matters here because the
//! hot caller is `linear_predictor`, which dots design rows of two to ten
//! elements: a single-width kernel would drop those straight to scalar.
//!
//! The reductions run one accumulator, not several. Multiple accumulators break
//! the `fadd` dependency chain and pay from a few hundred elements up, but the
//! block loop that drives them needs `lanes * accumulators` elements before it
//! can fire at all, which at two to ten coefficients it never does. Dropping the
//! four-accumulator version moved `glm_fit` by under three per cent either way
//! at two and four coefficients and made it five per cent faster at eight, so it
//! was buying code and nothing else.

use std::sync::OnceLock;

use wide::{f32x4, f32x8, f64x2, f64x4};

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use wide::{f32x16, f64x8};

//////////////
// Dispatch //
//////////////

/// Widest vector width available on the running machine.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimdLevel {
    /// No vector unit worth dispatching to.
    Scalar,
    /// 128-bit. Also covers NEON, which every aarch64 target has.
    Sse,
    /// 256-bit.
    Avx2,
    /// 512-bit.
    Avx512,
}

/// Cached result of the CPU feature probe.
static SIMD_LEVEL: OnceLock<SimdLevel> = OnceLock::new();

/// Detects the widest vector width this machine supports.
///
/// The probe runs once and is cached, so the `match` in the trait impls costs a
/// load and a branch rather than a `cpuid`.
///
/// ### Returns
///
/// The [`SimdLevel`] to dispatch on.
pub fn detect_simd_level() -> SimdLevel {
    *SIMD_LEVEL.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                return SimdLevel::Avx512;
            }
            if is_x86_feature_detected!("avx2") {
                return SimdLevel::Avx2;
            }
            if is_x86_feature_detected!("sse4.1") {
                return SimdLevel::Sse;
            }
            SimdLevel::Scalar
        }

        #[cfg(target_arch = "aarch64")]
        {
            SimdLevel::Sse
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            SimdLevel::Scalar
        }
    })
}

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

    /// Exponentiates a slice in place.
    ///
    /// This is the `mu = exp(eta)` step of every GLM fit in the crate, so it
    /// sits on the hottest path there is.
    ///
    /// ### Params
    ///
    /// * `x` - Values to exponentiate, modified in place
    ///
    /// Note that each tier vectorises what it can and hands the remainder down,
    /// and the last tier calls libm. `wide`'s polynomial and libm disagree in
    /// the last ulp, so the same input value can come back with two different
    /// bit patterns depending on where in the slice it sits, on the slice
    /// length, and on which tier the machine dispatched to: `vec![0.37; n]`
    /// gives two distinct patterns for every odd `n`. This is elementwise rather
    /// than a reassociation, so it is a stronger caveat than the one on
    /// [`EdgeSimd::dot_simd`], but the magnitude is still one ulp and nothing in
    /// the crate depends on it. The vector lanes also flush a deep underflow to
    /// zero where libm returns a subnormal: `exp(-745.0)` is exactly 0 in a
    /// lane, 5e-324 in the tail.
    fn exp_in_place_simd(x: &mut [Self]);
}

/////////////////
// Dot product //
/////////////////

//-----------//
// f64 tiers //
//-----------//

/// Dot product of two `f64` slices, scalar.
///
/// ### Params
///
/// * `a` - Left operand
/// * `b` - Right operand
///
/// ### Returns
///
/// The sum of elementwise products, truncated to the shorter operand.
#[inline]
fn dot_f64_scalar(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Dot product of two `f64` slices, 128-bit.
///
/// ### Params
///
/// * `a` - Left operand
/// * `b` - Right operand
///
/// ### Returns
///
/// The sum of elementwise products, truncated to the shorter operand.
#[inline]
fn dot_f64_sse(a: &[f64], b: &[f64]) -> f64 {
    const LANES: usize = 2;

    let n = a.len().min(b.len());
    let mut total = f64x2::ZERO;
    let mut i = 0;

    while i + LANES <= n {
        total += f64x2::from(&a[i..i + LANES]) * f64x2::from(&b[i..i + LANES]);
        i += LANES;
    }

    total.reduce_add() + dot_f64_scalar(&a[i..n], &b[i..n])
}

/// Dot product of two `f64` slices, 256-bit.
///
/// ### Params
///
/// * `a` - Left operand
/// * `b` - Right operand
///
/// ### Returns
///
/// The sum of elementwise products, truncated to the shorter operand.
#[inline]
fn dot_f64_avx2(a: &[f64], b: &[f64]) -> f64 {
    const LANES: usize = 4;

    let n = a.len().min(b.len());
    let mut total = f64x4::ZERO;
    let mut i = 0;

    while i + LANES <= n {
        total += f64x4::from(&a[i..i + LANES]) * f64x4::from(&b[i..i + LANES]);
        i += LANES;
    }

    total.reduce_add() + dot_f64_sse(&a[i..n], &b[i..n])
}

/// Dot product of two `f64` slices, 512-bit.
///
/// ### Params
///
/// * `a` - Left operand
/// * `b` - Right operand
///
/// ### Returns
///
/// The sum of elementwise products, truncated to the shorter operand.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline]
fn dot_f64_avx512(a: &[f64], b: &[f64]) -> f64 {
    const LANES: usize = 8;

    let n = a.len().min(b.len());
    let mut total = f64x8::ZERO;
    let mut i = 0;

    while i + LANES <= n {
        total += f64x8::from(&a[i..i + LANES]) * f64x8::from(&b[i..i + LANES]);
        i += LANES;
    }

    total.reduce_add() + dot_f64_avx2(&a[i..n], &b[i..n])
}

/// Fallback when the crate is not built with AVX-512, so `f64x8` would be two
/// 256-bit halves and buy nothing over the 256-bit tier.
///
/// ### Params
///
/// * `a` - Left operand
/// * `b` - Right operand
///
/// ### Returns
///
/// The sum of elementwise products, truncated to the shorter operand.
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
#[inline]
fn dot_f64_avx512(a: &[f64], b: &[f64]) -> f64 {
    dot_f64_avx2(a, b)
}

//-----------//
// f32 tiers //
//-----------//

/// Dot product of two `f32` slices, scalar.
///
/// ### Params
///
/// * `a` - Left operand
/// * `b` - Right operand
///
/// ### Returns
///
/// The sum of elementwise products, truncated to the shorter operand.
#[inline]
fn dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Dot product of two `f32` slices, 128-bit.
///
/// ### Params
///
/// * `a` - Left operand
/// * `b` - Right operand
///
/// ### Returns
///
/// The sum of elementwise products, truncated to the shorter operand.
#[inline]
fn dot_f32_sse(a: &[f32], b: &[f32]) -> f32 {
    const LANES: usize = 4;

    let n = a.len().min(b.len());
    let mut total = f32x4::ZERO;
    let mut i = 0;

    while i + LANES <= n {
        total += f32x4::from(&a[i..i + LANES]) * f32x4::from(&b[i..i + LANES]);
        i += LANES;
    }

    total.reduce_add() + dot_f32_scalar(&a[i..n], &b[i..n])
}

/// Dot product of two `f32` slices, 256-bit.
///
/// ### Params
///
/// * `a` - Left operand
/// * `b` - Right operand
///
/// ### Returns
///
/// The sum of elementwise products, truncated to the shorter operand.
#[inline]
fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    const LANES: usize = 8;

    let n = a.len().min(b.len());
    let mut total = f32x8::ZERO;
    let mut i = 0;

    while i + LANES <= n {
        total += f32x8::from(&a[i..i + LANES]) * f32x8::from(&b[i..i + LANES]);
        i += LANES;
    }

    total.reduce_add() + dot_f32_sse(&a[i..n], &b[i..n])
}

/// Dot product of two `f32` slices, 512-bit.
///
/// ### Params
///
/// * `a` - Left operand
/// * `b` - Right operand
///
/// ### Returns
///
/// The sum of elementwise products, truncated to the shorter operand.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline]
fn dot_f32_avx512(a: &[f32], b: &[f32]) -> f32 {
    const LANES: usize = 16;

    let n = a.len().min(b.len());
    let mut total = f32x16::ZERO;
    let mut i = 0;

    while i + LANES <= n {
        total += f32x16::from(&a[i..i + LANES]) * f32x16::from(&b[i..i + LANES]);
        i += LANES;
    }

    total.reduce_add() + dot_f32_avx2(&a[i..n], &b[i..n])
}

/// Fallback when the crate is not built with AVX-512.
///
/// ### Params
///
/// * `a` - Left operand
/// * `b` - Right operand
///
/// ### Returns
///
/// The sum of elementwise products, truncated to the shorter operand.
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
#[inline]
fn dot_f32_avx512(a: &[f32], b: &[f32]) -> f32 {
    dot_f32_avx2(a, b)
}

/////////////////
// Exponential //
/////////////////

//-----------//
// f64 tiers //
//-----------//

/// In-place exponential over an `f64` slice, scalar.
///
/// ### Params
///
/// * `x` - Values to exponentiate
#[inline]
fn exp_f64_scalar(x: &mut [f64]) {
    for v in x.iter_mut() {
        *v = v.exp();
    }
}

/// In-place exponential over an `f64` slice, 128-bit.
///
/// Worth taking down to two lanes: `wide`'s polynomial beats a pair of libm
/// calls, and `n_samples` in the GLM path is often odd and small.
///
/// ### Params
///
/// * `x` - Values to exponentiate
#[inline]
fn exp_f64_sse(x: &mut [f64]) {
    const LANES: usize = 2;

    let n = x.len();
    let mut i = 0;

    while i + LANES <= n {
        let v = f64x2::from(&x[i..i + LANES]).exp();
        x[i..i + LANES].copy_from_slice(v.as_array());
        i += LANES;
    }

    exp_f64_scalar(&mut x[i..]);
}

/// In-place exponential over an `f64` slice, 256-bit.
///
/// ### Params
///
/// * `x` - Values to exponentiate
#[inline]
fn exp_f64_avx2(x: &mut [f64]) {
    const LANES: usize = 4;

    let n = x.len();
    let mut i = 0;

    while i + LANES <= n {
        let v = f64x4::from(&x[i..i + LANES]).exp();
        x[i..i + LANES].copy_from_slice(v.as_array());
        i += LANES;
    }

    exp_f64_sse(&mut x[i..]);
}

/// In-place exponential over an `f64` slice, 512-bit.
///
/// ### Params
///
/// * `x` - Values to exponentiate
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline]
fn exp_f64_avx512(x: &mut [f64]) {
    const LANES: usize = 8;

    let n = x.len();
    let mut i = 0;

    while i + LANES <= n {
        let v = f64x8::from(&x[i..i + LANES]).exp();
        x[i..i + LANES].copy_from_slice(v.as_array());
        i += LANES;
    }

    exp_f64_avx2(&mut x[i..]);
}

/// Fallback when the crate is not built with AVX-512.
///
/// ### Params
///
/// * `x` - Values to exponentiate
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
#[inline]
fn exp_f64_avx512(x: &mut [f64]) {
    exp_f64_avx2(x)
}

//-----------//
// f32 tiers //
//-----------//

/// In-place exponential over an `f32` slice, scalar.
///
/// ### Params
///
/// * `x` - Values to exponentiate
#[inline]
fn exp_f32_scalar(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = v.exp();
    }
}

/// In-place exponential over an `f32` slice, 128-bit.
///
/// ### Params
///
/// * `x` - Values to exponentiate
#[inline]
fn exp_f32_sse(x: &mut [f32]) {
    const LANES: usize = 4;

    let n = x.len();
    let mut i = 0;

    while i + LANES <= n {
        let v = f32x4::from(&x[i..i + LANES]).exp();
        x[i..i + LANES].copy_from_slice(v.as_array());
        i += LANES;
    }

    exp_f32_scalar(&mut x[i..]);
}

/// In-place exponential over an `f32` slice, 256-bit.
///
/// ### Params
///
/// * `x` - Values to exponentiate
#[inline]
fn exp_f32_avx2(x: &mut [f32]) {
    const LANES: usize = 8;

    let n = x.len();
    let mut i = 0;

    while i + LANES <= n {
        let v = f32x8::from(&x[i..i + LANES]).exp();
        x[i..i + LANES].copy_from_slice(v.as_array());
        i += LANES;
    }

    exp_f32_sse(&mut x[i..]);
}

/// In-place exponential over an `f32` slice, 512-bit.
///
/// ### Params
///
/// * `x` - Values to exponentiate
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline]
fn exp_f32_avx512(x: &mut [f32]) {
    const LANES: usize = 16;

    let n = x.len();
    let mut i = 0;

    while i + LANES <= n {
        let v = f32x16::from(&x[i..i + LANES]).exp();
        x[i..i + LANES].copy_from_slice(v.as_array());
        i += LANES;
    }

    exp_f32_avx2(&mut x[i..]);
}

/// Fallback when the crate is not built with AVX-512.
///
/// ### Params
///
/// * `x` - Values to exponentiate
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
#[inline]
fn exp_f32_avx512(x: &mut [f32]) {
    exp_f32_avx2(x)
}

/////////////////////
// Implementations //
/////////////////////

impl EdgeSimd for f64 {
    #[inline]
    fn dot_simd(a: &[f64], b: &[f64]) -> f64 {
        match detect_simd_level() {
            SimdLevel::Avx512 => dot_f64_avx512(a, b),
            SimdLevel::Avx2 => dot_f64_avx2(a, b),
            SimdLevel::Sse => dot_f64_sse(a, b),
            SimdLevel::Scalar => dot_f64_scalar(a, b),
        }
    }

    #[inline]
    fn exp_in_place_simd(x: &mut [f64]) {
        match detect_simd_level() {
            SimdLevel::Avx512 => exp_f64_avx512(x),
            SimdLevel::Avx2 => exp_f64_avx2(x),
            SimdLevel::Sse => exp_f64_sse(x),
            SimdLevel::Scalar => exp_f64_scalar(x),
        }
    }
}

impl EdgeSimd for f32 {
    #[inline]
    fn dot_simd(a: &[f32], b: &[f32]) -> f32 {
        match detect_simd_level() {
            SimdLevel::Avx512 => dot_f32_avx512(a, b),
            SimdLevel::Avx2 => dot_f32_avx2(a, b),
            SimdLevel::Sse => dot_f32_sse(a, b),
            SimdLevel::Scalar => dot_f32_scalar(a, b),
        }
    }

    #[inline]
    fn exp_in_place_simd(x: &mut [f32]) {
        match detect_simd_level() {
            SimdLevel::Avx512 => exp_f32_avx512(x),
            SimdLevel::Avx2 => exp_f32_avx2(x),
            SimdLevel::Sse => exp_f32_sse(x),
            SimdLevel::Scalar => exp_f32_scalar(x),
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Lengths chosen to straddle every rung of the ladder: below a vector, one
    /// vector, and vectors plus awkward remainders that force the delegation
    /// down through 256, 128 and scalar.
    const SIZES: [usize; 16] = [0, 1, 2, 3, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 45];

    fn ramp_f64(n: usize) -> Vec<f64> {
        (0..n).map(|i| (i as f64) * 0.5 - 3.0).collect()
    }

    fn ramp_f32(n: usize) -> Vec<f32> {
        (0..n).map(|i| (i as f32) * 0.5 - 3.0).collect()
    }

    /// Every tier must agree with the scalar reference. Dispatch on this host
    /// only ever reaches one tier, so the tier functions are called directly.
    #[test]
    fn test_dot_tiers_match_scalar_f64() {
        for n in SIZES {
            let a = ramp_f64(n);
            let b: Vec<f64> = ramp_f64(n).iter().map(|v| v * 1.7 + 0.3).collect();
            let expected = dot_f64_scalar(&a, &b);
            assert_relative_eq!(dot_f64_sse(&a, &b), expected, epsilon = 1e-12);
            assert_relative_eq!(dot_f64_avx2(&a, &b), expected, epsilon = 1e-12);
            assert_relative_eq!(dot_f64_avx512(&a, &b), expected, epsilon = 1e-12);
            assert_relative_eq!(f64::dot_simd(&a, &b), expected, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_dot_tiers_match_scalar_f32() {
        for n in SIZES {
            let a = ramp_f32(n);
            let b: Vec<f32> = ramp_f32(n).iter().map(|v| v * 1.7 + 0.3).collect();
            let expected = dot_f32_scalar(&a, &b);
            assert_relative_eq!(dot_f32_sse(&a, &b), expected, epsilon = 1e-4);
            assert_relative_eq!(dot_f32_avx2(&a, &b), expected, epsilon = 1e-4);
            assert_relative_eq!(dot_f32_avx512(&a, &b), expected, epsilon = 1e-4);
            assert_relative_eq!(f32::dot_simd(&a, &b), expected, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_exp_tiers_match_scalar() {
        for n in SIZES {
            let mut expected = ramp_f64(n);
            exp_f64_scalar(&mut expected);

            for kernel in [
                exp_f64_sse as fn(&mut [f64]),
                exp_f64_avx2,
                exp_f64_avx512,
                f64::exp_in_place_simd,
            ] {
                let mut x = ramp_f64(n);
                kernel(&mut x);
                for (got, want) in x.iter().zip(expected.iter()) {
                    assert_relative_eq!(got, want, max_relative = 1e-12);
                }
            }

            let mut expected = ramp_f32(n);
            exp_f32_scalar(&mut expected);

            for kernel in [
                exp_f32_sse as fn(&mut [f32]),
                exp_f32_avx2,
                exp_f32_avx512,
                f32::exp_in_place_simd,
            ] {
                let mut x = ramp_f32(n);
                kernel(&mut x);
                for (got, want) in x.iter().zip(expected.iter()) {
                    assert_relative_eq!(got, want, max_relative = 1e-5);
                }
            }
        }
    }

    /// A dot product of unequal lengths truncates to the shorter operand rather
    /// than reading out of bounds. Documented behaviour, and the delegation has
    /// to preserve it at every rung, so pin all tiers.
    #[test]
    fn test_dot_truncates_on_length_mismatch() {
        let a = vec![1.0_f64; 20];
        let b = vec![2.0_f64; 5];
        assert_relative_eq!(dot_f64_scalar(&a, &b), 10.0, epsilon = 1e-12);
        assert_relative_eq!(dot_f64_sse(&a, &b), 10.0, epsilon = 1e-12);
        assert_relative_eq!(dot_f64_avx2(&a, &b), 10.0, epsilon = 1e-12);
        assert_relative_eq!(dot_f64_avx512(&a, &b), 10.0, epsilon = 1e-12);
        assert_relative_eq!(f64::dot_simd(&a, &b), 10.0, epsilon = 1e-12);
    }

    /// The probe is cached, so repeated calls must agree.
    #[test]
    fn test_detect_simd_level_is_stable() {
        assert_eq!(detect_simd_level(), detect_simd_level());
    }
}
