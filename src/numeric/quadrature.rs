//! Adaptive numerical integration.
//!
//! edgePython reaches for `scipy.integrate.quad` in exactly one place, the Huber
//! weighting inside `estimate_glm_robust_disp` (`dispersion.py` line 999), with
//! `limit = 50`. `quad` is QUADPACK's `QAGS`, which is adaptive Gauss-Kronrod
//! with epsilon extrapolation. The extrapolation only earns its keep on
//! endpoint singularities, and the integrand there is a smooth deviance
//! function on a finite interval, so this implements plain adaptive
//! Gauss-Kronrod 15/7 without it.
//!
//! ### References
//!
//! Piessens, de Doncker-Kapenga, Uberhuber and Kahaner, QUADPACK, 1983

use crate::errors::EdgeErrors;

/// Abscissae of the 15-point Kronrod rule on `[-1, 1]`, symmetric about zero.
///
/// Only the non-negative half is stored; the rule is mirrored at evaluation.
/// The odd-indexed entries are the 7-point Gauss nodes, which is what lets one
/// set of evaluations produce both estimates and hence the error estimate for
/// free.
const KRONROD_NODES: [f64; 8] = [
    0.000_000_000_000_000_0,
    0.207_784_955_007_898_5,
    0.405_845_151_377_397_2,
    0.586_087_235_467_691_1,
    0.741_531_185_599_394_4,
    0.864_864_423_359_769_1,
    0.949_107_912_342_758_5,
    0.991_455_371_120_812_6,
];

/// Weights of the 15-point Kronrod rule, aligned with [`KRONROD_NODES`].
const KRONROD_WEIGHTS: [f64; 8] = [
    0.209_482_141_084_727_8,
    0.204_432_940_075_298_9,
    0.190_350_578_064_785_4,
    0.169_004_726_639_267_9,
    0.140_653_259_715_525_9,
    0.104_790_010_322_250_2,
    0.063_092_092_629_978_6,
    0.022_935_322_010_529_2,
];

/// Weights of the embedded 7-point Gauss rule.
///
/// Indexed against the even entries of [`KRONROD_NODES`], that is the centre
/// plus nodes 2, 4 and 6. Sharing nodes with the Kronrod rule is the whole
/// point: one set of evaluations yields both estimates, and their difference is
/// the error estimate.
const GAUSS_WEIGHTS: [f64; 4] = [
    0.417_959_183_673_469_4,
    0.381_830_050_505_118_9,
    0.279_705_391_489_276_7,
    0.129_484_966_168_869_7,
];

/// Default subdivision budget, matching the `limit = 50` edgePython passes.
const DEFAULT_LIMIT: usize = 50;

/// A subinterval and what the rule made of it.
#[derive(Clone, Copy, Debug)]
struct Segment {
    /// Left endpoint.
    a: f64,
    /// Right endpoint.
    b: f64,
    /// Kronrod estimate of the integral over `[a, b]`.
    integral: f64,
    /// Estimated absolute error of `integral`.
    error: f64,
}

/// Result of an adaptive integration.
#[derive(Clone, Copy, Debug)]
pub struct QuadResult {
    /// Estimated value of the integral.
    pub value: f64,
    /// Estimated absolute error.
    pub abserr: f64,
    /// Number of subintervals used.
    pub subintervals: usize,
    /// Whether both tolerances were met before the subdivision budget ran out.
    pub converged: bool,
}

/// Applies the 15-point Kronrod rule with its embedded 7-point Gauss rule.
///
/// ### Params
///
/// * `f` - Integrand
/// * `a` - Left endpoint
/// * `b` - Right endpoint
///
/// ### Returns
///
/// The Kronrod estimate and an error estimate derived from its disagreement
/// with the Gauss estimate. The exponent of 1.5 in the error scaling is
/// QUADPACK's, chosen so the estimate is conservative on smooth integrands.
fn gauss_kronrod<F>(f: &mut F, a: f64, b: f64) -> (f64, f64)
where
    F: FnMut(f64) -> f64,
{
    let centre = 0.5 * (a + b);
    let half_width = 0.5 * (b - a);

    let f_centre = f(centre);
    let mut kronrod = KRONROD_WEIGHTS[0] * f_centre;
    let mut gauss = GAUSS_WEIGHTS[0] * f_centre;
    let mut abs_integral = KRONROD_WEIGHTS[0] * f_centre.abs();

    for j in 1..8 {
        let offset = half_width * KRONROD_NODES[j];
        let f_left = f(centre - offset);
        let f_right = f(centre + offset);
        let paired = f_left + f_right;

        kronrod += KRONROD_WEIGHTS[j] * paired;
        abs_integral += KRONROD_WEIGHTS[j] * (f_left.abs() + f_right.abs());
        // The 7-point Gauss nodes are the even-indexed Kronrod nodes, so every
        // second pair contributes to both estimates.
        if j.is_multiple_of(2) {
            gauss += GAUSS_WEIGHTS[j / 2] * paired;
        }
    }

    let integral = kronrod * half_width;
    let abs_integral = abs_integral * half_width.abs();
    let raw_error = ((kronrod - gauss) * half_width).abs();

    let mut error = raw_error;
    if abs_integral != 0.0 && error != 0.0 {
        error = abs_integral * 1.0_f64.min((200.0 * raw_error / abs_integral).powf(1.5));
    }
    (integral, error)
}

/// Integrates a function over a finite interval to a requested tolerance.
///
/// Globally adaptive: repeatedly bisects whichever subinterval currently carries
/// the largest error estimate.
///
/// ### Params
///
/// * `f` - Integrand, must be finite across `[a, b]`
/// * `a` - Left endpoint
/// * `b` - Right endpoint. May be less than `a`, in which case the sign of the
///   result flips, as it does in `scipy.integrate.quad`.
/// * `epsabs` - Absolute tolerance
/// * `epsrel` - Relative tolerance
/// * `limit` - Maximum subintervals, or `None` for the QUADPACK default of 50
///
/// ### Returns
///
/// The integral with its error estimate, or [`EdgeErrors`] if a tolerance is
/// non-positive. Running out of subintervals is not an error: the result comes
/// back with `converged` false, matching `quad`'s behaviour of returning its
/// best estimate with a warning.
pub fn integrate<F>(
    mut f: F,
    a: f64,
    b: f64,
    epsabs: f64,
    epsrel: f64,
    limit: Option<usize>,
) -> Result<QuadResult, EdgeErrors>
where
    F: FnMut(f64) -> f64,
{
    let limit = limit.unwrap_or(DEFAULT_LIMIT);
    if epsabs <= 0.0 && epsrel <= 0.0 {
        return Err(EdgeErrors::InvalidArgument(
            "integrate needs a positive epsabs or epsrel".to_string(),
        ));
    }
    if limit == 0 {
        return Err(EdgeErrors::MustBePositive("limit".to_string()));
    }
    if !a.is_finite() || !b.is_finite() {
        return Err(EdgeErrors::InvalidArgument(format!(
            "integrate needs finite limits, got [{a}, {b}]"
        )));
    }
    if a == b {
        return Ok(QuadResult {
            value: 0.0,
            abserr: 0.0,
            subintervals: 0,
            converged: true,
        });
    }

    let (integral, error) = gauss_kronrod(&mut f, a, b);
    let mut segments = vec![Segment {
        a,
        b,
        integral,
        error,
    }];

    for _ in 1..limit {
        let total: f64 = segments.iter().map(|s| s.integral).sum();
        let total_error: f64 = segments.iter().map(|s| s.error).sum();
        let tolerance = epsabs.max(epsrel * total.abs());

        if total_error <= tolerance {
            return Ok(QuadResult {
                value: total,
                abserr: total_error,
                subintervals: segments.len(),
                converged: true,
            });
        }

        // Bisect the worst offender.
        let (worst, _) =
            segments
                .iter()
                .enumerate()
                .fold((0usize, f64::NEG_INFINITY), |(bi, be), (i, s)| {
                    if s.error > be { (i, s.error) } else { (bi, be) }
                });

        let segment = segments.swap_remove(worst);
        let mid = 0.5 * (segment.a + segment.b);
        if mid == segment.a || mid == segment.b {
            // The interval has collapsed to adjacent floats; no further
            // refinement is representable.
            break;
        }

        for (lo, hi) in [(segment.a, mid), (mid, segment.b)] {
            let (integral, error) = gauss_kronrod(&mut f, lo, hi);
            segments.push(Segment {
                a: lo,
                b: hi,
                integral,
                error,
            });
        }
    }

    let value: f64 = segments.iter().map(|s| s.integral).sum();
    let abserr: f64 = segments.iter().map(|s| s.error).sum();
    let converged = abserr <= epsabs.max(epsrel * value.abs());
    Ok(QuadResult {
        value,
        abserr,
        subintervals: segments.len(),
        converged,
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    const EPSABS: f64 = 1.49e-8;
    const EPSREL: f64 = 1.49e-8;

    #[test]
    fn test_integrates_a_polynomial_exactly() {
        // The 15-point rule is exact for polynomials up to degree 22, so a
        // single panel should nail this.
        let res = integrate(|x: f64| x * x * x, 0.0, 2.0, EPSABS, EPSREL, None).unwrap();
        assert_relative_eq!(res.value, 4.0, max_relative = 1e-14);
        assert_eq!(res.subintervals, 1);
        assert!(res.converged);
    }

    /// scipy: `quad(np.sin, 0, np.pi)` gives 2.0.
    #[test]
    fn test_integrates_sine_over_a_half_period() {
        let res = integrate(f64::sin, 0.0, std::f64::consts::PI, EPSABS, EPSREL, None).unwrap();
        assert_relative_eq!(res.value, 2.0, max_relative = 1e-12);
    }

    /// scipy: `quad(lambda x: np.exp(-x*x), 0, 5)` gives 0.8862269254514.
    /// Just short of `sqrt(pi)/2`, since the tail beyond 5 is around 1.5e-12.
    #[test]
    fn test_integrates_a_gaussian_tail() {
        let res = integrate(|x: f64| (-x * x).exp(), 0.0, 5.0, EPSABS, EPSREL, None).unwrap();
        assert_relative_eq!(res.value, 0.886_226_925_451_4, max_relative = 1e-12);
    }

    /// A peaked integrand: one panel is nowhere near enough, so the adaptive
    /// bisection is what produces the answer.
    ///
    /// The reference is analytic rather than numerical here, since the
    /// antiderivative is known exactly: the integral is `200 * arctan(50)`,
    /// or 310.1597985643. scipy's `quad` agrees to 310.1597985644.
    #[test]
    fn test_adapts_to_a_sharp_peak() {
        let res = integrate(
            |x: f64| 1.0 / (1e-4 + (x - 0.5).powi(2)),
            0.0,
            1.0,
            EPSABS,
            EPSREL,
            Some(200),
        )
        .unwrap();
        assert_relative_eq!(res.value, 200.0 * 50.0_f64.atan(), max_relative = 1e-10);
        assert!(res.subintervals > 1, "the peak should have forced a split");
    }

    #[test]
    fn test_reversed_limits_flip_the_sign() {
        let forward = integrate(f64::sin, 0.0, 1.0, EPSABS, EPSREL, None).unwrap();
        let backward = integrate(f64::sin, 1.0, 0.0, EPSABS, EPSREL, None).unwrap();
        assert_relative_eq!(forward.value, -backward.value, max_relative = 1e-13);
    }

    #[test]
    fn test_degenerate_interval_is_zero() {
        let res = integrate(|x: f64| x.exp(), 2.0, 2.0, EPSABS, EPSREL, None).unwrap();
        assert_eq!(res.value, 0.0);
        assert!(res.converged);
    }

    #[test]
    fn test_reports_non_convergence_rather_than_failing() {
        // One subinterval on a peak it cannot resolve: best effort, flagged.
        let res = integrate(
            |x: f64| 1.0 / (1e-8 + (x - 0.5).powi(2)),
            0.0,
            1.0,
            1e-14,
            1e-14,
            Some(2),
        )
        .unwrap();
        assert!(!res.converged);
    }

    #[test]
    fn test_rejects_non_finite_limits() {
        let err = integrate(f64::sin, 0.0, f64::INFINITY, EPSABS, EPSREL, None).unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));
    }

    #[test]
    fn test_rejects_useless_tolerances() {
        let err = integrate(f64::sin, 0.0, 1.0, 0.0, 0.0, None).unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));
    }

    #[test]
    fn test_rejects_zero_limit() {
        let err = integrate(f64::sin, 0.0, 1.0, EPSABS, EPSREL, Some(0)).unwrap_err();
        assert!(matches!(err, EdgeErrors::MustBePositive(_)));
    }
}
