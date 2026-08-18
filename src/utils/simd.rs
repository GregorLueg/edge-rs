//! SIMD kernels behind the `EdgeSimd` trait.
//!
//! Algorithm code stays generic over `T: EdgeFloat` and calls `T::dot_simd(..)`
//! and friends. The vector width lives here and nowhere else; no intrinsics or
//! `wide` types appear outside this file.
//!
//! Each operation exists at four widths: scalar, 128-bit, 256-bit and 512-bit.
//! [`detect_simd_level`] caches what the machine supports and the trait impls
//! dispatch on it. A tier does its own unrolled block loop, then single vectors
//! at its own width, then hands the remainder down to the tier below. That
//! ladder matters here because the hot caller is `linear_predictor`, which dots
//! design rows of two to ten elements: a single-width kernel would drop those
//! straight to scalar.
//!
//! The primitives come from `wide`, whose `f64x8` and `f32x16` compile to real
//! `__m512d`/`__m512` under `target_feature = "avx512f"` and to two 256-bit
//! halves otherwise. The 512-bit bodies are `cfg`-gated on that feature and fall
//! back to the 256-bit kernel when it is off, so the runtime check guards
//! correctness while the `cfg` decides whether the wide bodies exist at all.
//! Nothing in this crate sets `target-cpu`, so a stock build compiles every tier
//! to the SSE2/NEON baseline; consumers who want the 512-bit path build with
//! `-C target-cpu=x86-64-v4` (or `native`). Use the `target-cpu` form, not a
//! bare `-C target-feature=+avx512f`: `wide` 1.6.1 picks `i32x16`'s layout on
//! `avx512f` but its `abs` on `avx512bw`, so the narrower flag fails to compile
//! inside `wide` itself. None of this is benchmarked yet, there being no
//! AVX-512 hardware to hand.
//!
//! Only general primitives live here. Fused kernels (negative binomial unit
//! deviance, the working-weight outer product) arrive with the modules that
//! consume them, so they can be measured against the scalar form rather than
//! assumed faster.

use std::sync::OnceLock;

use wide::{f32x4, f32x8, f64x2, f64x4};

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use wide::{f32x16, f64x8};

/// Number of independent accumulators in the reduction loops.
///
/// A single accumulator serialises on the FMA latency chain. Four independent
/// ones keep the pipeline fed and is the point of diminishing returns on both
/// Apple Silicon and x86: eight costs register pressure without buying
/// throughput. This matches the constant `bixverse-rs` settled on. Four 512-bit
/// accumulators still leave 28 of the 32 zmm registers free.
const UNROLL: usize = 4;

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

    /// Sum of a slice.
    ///
    /// ### Params
    ///
    /// * `x` - Values to reduce
    ///
    /// ### Returns
    ///
    /// The sum. Note that the multi-accumulator layout and the width ladder both
    /// change the summation order relative to a scalar left fold, so results can
    /// differ in the last ulp, and can differ between two machines that dispatch
    /// to different tiers. Nothing in the crate depends on that ordering. The
    /// same caveat applies to [`EdgeSimd::dot_simd`].
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
    const BLOCK: usize = LANES * UNROLL;

    let n = a.len().min(b.len());
    let mut acc = [f64x2::ZERO; UNROLL];
    let mut i = 0;

    while i + BLOCK <= n {
        for (u, acc_u) in acc.iter_mut().enumerate() {
            let off = i + u * LANES;
            *acc_u += f64x2::from(&a[off..off + LANES]) * f64x2::from(&b[off..off + LANES]);
        }
        i += BLOCK;
    }

    let mut total = f64x2::ZERO;
    for acc_u in acc.iter() {
        total += *acc_u;
    }

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
    const BLOCK: usize = LANES * UNROLL;

    let n = a.len().min(b.len());
    let mut acc = [f64x4::ZERO; UNROLL];
    let mut i = 0;

    while i + BLOCK <= n {
        for (u, acc_u) in acc.iter_mut().enumerate() {
            let off = i + u * LANES;
            *acc_u += f64x4::from(&a[off..off + LANES]) * f64x4::from(&b[off..off + LANES]);
        }
        i += BLOCK;
    }

    let mut total = f64x4::ZERO;
    for acc_u in acc.iter() {
        total += *acc_u;
    }

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
    const BLOCK: usize = LANES * UNROLL;

    let n = a.len().min(b.len());
    let mut acc = [f64x8::ZERO; UNROLL];
    let mut i = 0;

    while i + BLOCK <= n {
        for (u, acc_u) in acc.iter_mut().enumerate() {
            let off = i + u * LANES;
            *acc_u += f64x8::from(&a[off..off + LANES]) * f64x8::from(&b[off..off + LANES]);
        }
        i += BLOCK;
    }

    let mut total = f64x8::ZERO;
    for acc_u in acc.iter() {
        total += *acc_u;
    }

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
    const BLOCK: usize = LANES * UNROLL;

    let n = a.len().min(b.len());
    let mut acc = [f32x4::ZERO; UNROLL];
    let mut i = 0;

    while i + BLOCK <= n {
        for (u, acc_u) in acc.iter_mut().enumerate() {
            let off = i + u * LANES;
            *acc_u += f32x4::from(&a[off..off + LANES]) * f32x4::from(&b[off..off + LANES]);
        }
        i += BLOCK;
    }

    let mut total = f32x4::ZERO;
    for acc_u in acc.iter() {
        total += *acc_u;
    }

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
    const BLOCK: usize = LANES * UNROLL;

    let n = a.len().min(b.len());
    let mut acc = [f32x8::ZERO; UNROLL];
    let mut i = 0;

    while i + BLOCK <= n {
        for (u, acc_u) in acc.iter_mut().enumerate() {
            let off = i + u * LANES;
            *acc_u += f32x8::from(&a[off..off + LANES]) * f32x8::from(&b[off..off + LANES]);
        }
        i += BLOCK;
    }

    let mut total = f32x8::ZERO;
    for acc_u in acc.iter() {
        total += *acc_u;
    }

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
    const BLOCK: usize = LANES * UNROLL;

    let n = a.len().min(b.len());
    let mut acc = [f32x16::ZERO; UNROLL];
    let mut i = 0;

    while i + BLOCK <= n {
        for (u, acc_u) in acc.iter_mut().enumerate() {
            let off = i + u * LANES;
            *acc_u += f32x16::from(&a[off..off + LANES]) * f32x16::from(&b[off..off + LANES]);
        }
        i += BLOCK;
    }

    let mut total = f32x16::ZERO;
    for acc_u in acc.iter() {
        total += *acc_u;
    }

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

/////////
// Sum //
/////////

//-----------//
// f64 tiers //
//-----------//

/// Sum of an `f64` slice, scalar.
///
/// ### Params
///
/// * `x` - Values to reduce
///
/// ### Returns
///
/// The sum.
#[inline]
fn sum_f64_scalar(x: &[f64]) -> f64 {
    x.iter().sum()
}

/// Sum of an `f64` slice, 128-bit.
///
/// ### Params
///
/// * `x` - Values to reduce
///
/// ### Returns
///
/// The sum.
#[inline]
fn sum_f64_sse(x: &[f64]) -> f64 {
    const LANES: usize = 2;
    const BLOCK: usize = LANES * UNROLL;

    let n = x.len();
    let mut acc = [f64x2::ZERO; UNROLL];
    let mut i = 0;

    while i + BLOCK <= n {
        for (u, acc_u) in acc.iter_mut().enumerate() {
            let off = i + u * LANES;
            *acc_u += f64x2::from(&x[off..off + LANES]);
        }
        i += BLOCK;
    }

    let mut total = f64x2::ZERO;
    for acc_u in acc.iter() {
        total += *acc_u;
    }

    while i + LANES <= n {
        total += f64x2::from(&x[i..i + LANES]);
        i += LANES;
    }

    total.reduce_add() + sum_f64_scalar(&x[i..])
}

/// Sum of an `f64` slice, 256-bit.
///
/// ### Params
///
/// * `x` - Values to reduce
///
/// ### Returns
///
/// The sum.
#[inline]
fn sum_f64_avx2(x: &[f64]) -> f64 {
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

    while i + LANES <= n {
        total += f64x4::from(&x[i..i + LANES]);
        i += LANES;
    }

    total.reduce_add() + sum_f64_sse(&x[i..])
}

/// Sum of an `f64` slice, 512-bit.
///
/// ### Params
///
/// * `x` - Values to reduce
///
/// ### Returns
///
/// The sum.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline]
fn sum_f64_avx512(x: &[f64]) -> f64 {
    const LANES: usize = 8;
    const BLOCK: usize = LANES * UNROLL;

    let n = x.len();
    let mut acc = [f64x8::ZERO; UNROLL];
    let mut i = 0;

    while i + BLOCK <= n {
        for (u, acc_u) in acc.iter_mut().enumerate() {
            let off = i + u * LANES;
            *acc_u += f64x8::from(&x[off..off + LANES]);
        }
        i += BLOCK;
    }

    let mut total = f64x8::ZERO;
    for acc_u in acc.iter() {
        total += *acc_u;
    }

    while i + LANES <= n {
        total += f64x8::from(&x[i..i + LANES]);
        i += LANES;
    }

    total.reduce_add() + sum_f64_avx2(&x[i..])
}

/// Fallback when the crate is not built with AVX-512.
///
/// ### Params
///
/// * `x` - Values to reduce
///
/// ### Returns
///
/// The sum.
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
#[inline]
fn sum_f64_avx512(x: &[f64]) -> f64 {
    sum_f64_avx2(x)
}

//-----------//
// f32 tiers //
//-----------//

/// Sum of an `f32` slice, scalar.
///
/// ### Params
///
/// * `x` - Values to reduce
///
/// ### Returns
///
/// The sum.
#[inline]
fn sum_f32_scalar(x: &[f32]) -> f32 {
    x.iter().sum()
}

/// Sum of an `f32` slice, 128-bit.
///
/// ### Params
///
/// * `x` - Values to reduce
///
/// ### Returns
///
/// The sum.
#[inline]
fn sum_f32_sse(x: &[f32]) -> f32 {
    const LANES: usize = 4;
    const BLOCK: usize = LANES * UNROLL;

    let n = x.len();
    let mut acc = [f32x4::ZERO; UNROLL];
    let mut i = 0;

    while i + BLOCK <= n {
        for (u, acc_u) in acc.iter_mut().enumerate() {
            let off = i + u * LANES;
            *acc_u += f32x4::from(&x[off..off + LANES]);
        }
        i += BLOCK;
    }

    let mut total = f32x4::ZERO;
    for acc_u in acc.iter() {
        total += *acc_u;
    }

    while i + LANES <= n {
        total += f32x4::from(&x[i..i + LANES]);
        i += LANES;
    }

    total.reduce_add() + sum_f32_scalar(&x[i..])
}

/// Sum of an `f32` slice, 256-bit.
///
/// ### Params
///
/// * `x` - Values to reduce
///
/// ### Returns
///
/// The sum.
#[inline]
fn sum_f32_avx2(x: &[f32]) -> f32 {
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

    while i + LANES <= n {
        total += f32x8::from(&x[i..i + LANES]);
        i += LANES;
    }

    total.reduce_add() + sum_f32_sse(&x[i..])
}

/// Sum of an `f32` slice, 512-bit.
///
/// ### Params
///
/// * `x` - Values to reduce
///
/// ### Returns
///
/// The sum.
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline]
fn sum_f32_avx512(x: &[f32]) -> f32 {
    const LANES: usize = 16;
    const BLOCK: usize = LANES * UNROLL;

    let n = x.len();
    let mut acc = [f32x16::ZERO; UNROLL];
    let mut i = 0;

    while i + BLOCK <= n {
        for (u, acc_u) in acc.iter_mut().enumerate() {
            let off = i + u * LANES;
            *acc_u += f32x16::from(&x[off..off + LANES]);
        }
        i += BLOCK;
    }

    let mut total = f32x16::ZERO;
    for acc_u in acc.iter() {
        total += *acc_u;
    }

    while i + LANES <= n {
        total += f32x16::from(&x[i..i + LANES]);
        i += LANES;
    }

    total.reduce_add() + sum_f32_avx2(&x[i..])
}

/// Fallback when the crate is not built with AVX-512.
///
/// ### Params
///
/// * `x` - Values to reduce
///
/// ### Returns
///
/// The sum.
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
#[inline]
fn sum_f32_avx512(x: &[f32]) -> f32 {
    sum_f32_avx2(x)
}

//////////
// Axpy //
//////////

//-----------//
// f64 tiers //
//-----------//

/// In-place `y += a * x` over `f64` slices, scalar.
///
/// ### Params
///
/// * `y` - Destination
/// * `a` - Scalar multiplier
/// * `x` - Source
#[inline]
fn axpy_f64_scalar(y: &mut [f64], a: f64, x: &[f64]) {
    for (y_i, x_i) in y.iter_mut().zip(x.iter()) {
        *y_i += a * x_i;
    }
}

/// In-place `y += a * x` over `f64` slices, 128-bit.
///
/// No accumulator chain to break here, so this tier does not unroll.
///
/// ### Params
///
/// * `y` - Destination
/// * `a` - Scalar multiplier
/// * `x` - Source
#[inline]
fn axpy_f64_sse(y: &mut [f64], a: f64, x: &[f64]) {
    const LANES: usize = 2;

    let n = y.len().min(x.len());
    let va = f64x2::splat(a);
    let mut i = 0;

    while i + LANES <= n {
        let out = f64x2::from(&y[i..i + LANES]) + va * f64x2::from(&x[i..i + LANES]);
        y[i..i + LANES].copy_from_slice(out.as_array());
        i += LANES;
    }

    axpy_f64_scalar(&mut y[i..n], a, &x[i..n]);
}

/// In-place `y += a * x` over `f64` slices, 256-bit.
///
/// ### Params
///
/// * `y` - Destination
/// * `a` - Scalar multiplier
/// * `x` - Source
#[inline]
fn axpy_f64_avx2(y: &mut [f64], a: f64, x: &[f64]) {
    const LANES: usize = 4;

    let n = y.len().min(x.len());
    let va = f64x4::splat(a);
    let mut i = 0;

    while i + LANES <= n {
        let out = f64x4::from(&y[i..i + LANES]) + va * f64x4::from(&x[i..i + LANES]);
        y[i..i + LANES].copy_from_slice(out.as_array());
        i += LANES;
    }

    axpy_f64_sse(&mut y[i..n], a, &x[i..n]);
}

/// In-place `y += a * x` over `f64` slices, 512-bit.
///
/// ### Params
///
/// * `y` - Destination
/// * `a` - Scalar multiplier
/// * `x` - Source
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline]
fn axpy_f64_avx512(y: &mut [f64], a: f64, x: &[f64]) {
    const LANES: usize = 8;

    let n = y.len().min(x.len());
    let va = f64x8::splat(a);
    let mut i = 0;

    while i + LANES <= n {
        let out = f64x8::from(&y[i..i + LANES]) + va * f64x8::from(&x[i..i + LANES]);
        y[i..i + LANES].copy_from_slice(out.as_array());
        i += LANES;
    }

    axpy_f64_avx2(&mut y[i..n], a, &x[i..n]);
}

/// Fallback when the crate is not built with AVX-512.
///
/// ### Params
///
/// * `y` - Destination
/// * `a` - Scalar multiplier
/// * `x` - Source
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
#[inline]
fn axpy_f64_avx512(y: &mut [f64], a: f64, x: &[f64]) {
    axpy_f64_avx2(y, a, x)
}

//-----------//
// f32 tiers //
//-----------//

/// In-place `y += a * x` over `f32` slices, scalar.
///
/// ### Params
///
/// * `y` - Destination
/// * `a` - Scalar multiplier
/// * `x` - Source
#[inline]
fn axpy_f32_scalar(y: &mut [f32], a: f32, x: &[f32]) {
    for (y_i, x_i) in y.iter_mut().zip(x.iter()) {
        *y_i += a * x_i;
    }
}

/// In-place `y += a * x` over `f32` slices, 128-bit.
///
/// ### Params
///
/// * `y` - Destination
/// * `a` - Scalar multiplier
/// * `x` - Source
#[inline]
fn axpy_f32_sse(y: &mut [f32], a: f32, x: &[f32]) {
    const LANES: usize = 4;

    let n = y.len().min(x.len());
    let va = f32x4::splat(a);
    let mut i = 0;

    while i + LANES <= n {
        let out = f32x4::from(&y[i..i + LANES]) + va * f32x4::from(&x[i..i + LANES]);
        y[i..i + LANES].copy_from_slice(out.as_array());
        i += LANES;
    }

    axpy_f32_scalar(&mut y[i..n], a, &x[i..n]);
}

/// In-place `y += a * x` over `f32` slices, 256-bit.
///
/// ### Params
///
/// * `y` - Destination
/// * `a` - Scalar multiplier
/// * `x` - Source
#[inline]
fn axpy_f32_avx2(y: &mut [f32], a: f32, x: &[f32]) {
    const LANES: usize = 8;

    let n = y.len().min(x.len());
    let va = f32x8::splat(a);
    let mut i = 0;

    while i + LANES <= n {
        let out = f32x8::from(&y[i..i + LANES]) + va * f32x8::from(&x[i..i + LANES]);
        y[i..i + LANES].copy_from_slice(out.as_array());
        i += LANES;
    }

    axpy_f32_sse(&mut y[i..n], a, &x[i..n]);
}

/// In-place `y += a * x` over `f32` slices, 512-bit.
///
/// ### Params
///
/// * `y` - Destination
/// * `a` - Scalar multiplier
/// * `x` - Source
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline]
fn axpy_f32_avx512(y: &mut [f32], a: f32, x: &[f32]) {
    const LANES: usize = 16;

    let n = y.len().min(x.len());
    let va = f32x16::splat(a);
    let mut i = 0;

    while i + LANES <= n {
        let out = f32x16::from(&y[i..i + LANES]) + va * f32x16::from(&x[i..i + LANES]);
        y[i..i + LANES].copy_from_slice(out.as_array());
        i += LANES;
    }

    axpy_f32_avx2(&mut y[i..n], a, &x[i..n]);
}

/// Fallback when the crate is not built with AVX-512.
///
/// ### Params
///
/// * `y` - Destination
/// * `a` - Scalar multiplier
/// * `x` - Source
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
#[inline]
fn axpy_f32_avx512(y: &mut [f32], a: f32, x: &[f32]) {
    axpy_f32_avx2(y, a, x)
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
    fn sum_simd(x: &[f64]) -> f64 {
        match detect_simd_level() {
            SimdLevel::Avx512 => sum_f64_avx512(x),
            SimdLevel::Avx2 => sum_f64_avx2(x),
            SimdLevel::Sse => sum_f64_sse(x),
            SimdLevel::Scalar => sum_f64_scalar(x),
        }
    }

    #[inline]
    fn axpy_simd(y: &mut [f64], a: f64, x: &[f64]) {
        match detect_simd_level() {
            SimdLevel::Avx512 => axpy_f64_avx512(y, a, x),
            SimdLevel::Avx2 => axpy_f64_avx2(y, a, x),
            SimdLevel::Sse => axpy_f64_sse(y, a, x),
            SimdLevel::Scalar => axpy_f64_scalar(y, a, x),
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
    fn sum_simd(x: &[f32]) -> f32 {
        match detect_simd_level() {
            SimdLevel::Avx512 => sum_f32_avx512(x),
            SimdLevel::Avx2 => sum_f32_avx2(x),
            SimdLevel::Sse => sum_f32_sse(x),
            SimdLevel::Scalar => sum_f32_scalar(x),
        }
    }

    #[inline]
    fn axpy_simd(y: &mut [f32], a: f32, x: &[f32]) {
        match detect_simd_level() {
            SimdLevel::Avx512 => axpy_f32_avx512(y, a, x),
            SimdLevel::Avx2 => axpy_f32_avx2(y, a, x),
            SimdLevel::Sse => axpy_f32_sse(y, a, x),
            SimdLevel::Scalar => axpy_f32_scalar(y, a, x),
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
    /// vector, one block, and blocks plus awkward remainders that force the
    /// delegation down through 256, 128 and scalar.
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
    fn test_sum_tiers_match_scalar() {
        for n in SIZES {
            let x = ramp_f64(n);
            let expected = sum_f64_scalar(&x);
            assert_relative_eq!(sum_f64_sse(&x), expected, epsilon = 1e-12);
            assert_relative_eq!(sum_f64_avx2(&x), expected, epsilon = 1e-12);
            assert_relative_eq!(sum_f64_avx512(&x), expected, epsilon = 1e-12);
            assert_relative_eq!(f64::sum_simd(&x), expected, epsilon = 1e-12);

            let x = ramp_f32(n);
            let expected = sum_f32_scalar(&x);
            assert_relative_eq!(sum_f32_sse(&x), expected, epsilon = 1e-4);
            assert_relative_eq!(sum_f32_avx2(&x), expected, epsilon = 1e-4);
            assert_relative_eq!(sum_f32_avx512(&x), expected, epsilon = 1e-4);
            assert_relative_eq!(f32::sum_simd(&x), expected, epsilon = 1e-4);
        }
    }

    #[test]
    fn test_axpy_tiers_match_scalar() {
        for n in SIZES {
            let x = ramp_f64(n);
            let mut expected = ramp_f64(n);
            axpy_f64_scalar(&mut expected, 2.5, &x);

            for kernel in [
                axpy_f64_sse as fn(&mut [f64], f64, &[f64]),
                axpy_f64_avx2,
                axpy_f64_avx512,
                f64::axpy_simd,
            ] {
                let mut y = ramp_f64(n);
                kernel(&mut y, 2.5, &x);
                for (got, want) in y.iter().zip(expected.iter()) {
                    assert_relative_eq!(got, want, epsilon = 1e-12);
                }
            }

            let x = ramp_f32(n);
            let mut expected = ramp_f32(n);
            axpy_f32_scalar(&mut expected, 2.5, &x);

            for kernel in [
                axpy_f32_sse as fn(&mut [f32], f32, &[f32]),
                axpy_f32_avx2,
                axpy_f32_avx512,
                f32::axpy_simd,
            ] {
                let mut y = ramp_f32(n);
                kernel(&mut y, 2.5, &x);
                for (got, want) in y.iter().zip(expected.iter()) {
                    assert_relative_eq!(got, want, epsilon = 1e-4);
                }
            }
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

    /// `axpy` truncates the same way, and the destination past the shorter
    /// operand must be left untouched.
    #[test]
    fn test_axpy_truncates_on_length_mismatch() {
        let x = vec![2.0_f64; 5];
        let mut y = vec![1.0_f64; 20];
        f64::axpy_simd(&mut y, 3.0, &x);
        for v in y.iter().take(5) {
            assert_relative_eq!(*v, 7.0, epsilon = 1e-12);
        }
        for v in y.iter().skip(5) {
            assert_relative_eq!(*v, 1.0, epsilon = 1e-12);
        }
    }

    /// The probe is cached, so repeated calls must agree.
    #[test]
    fn test_detect_simd_level_is_stable() {
        assert_eq!(detect_simd_level(), detect_simd_level());
    }
}
