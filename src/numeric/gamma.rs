//! The polygamma family in `f64`: the `scipy.special` calls edgeR and limma
//! make.
//!
//! `ln_gamma` is delegated to `statrs`. Everything else is implemented here as
//! upward recurrence to a fixed threshold followed by the asymptotic Bernoulli
//! series. [`trigamma`] is literally [`polygamma`] at order one rather than a
//! second implementation, so the two cannot drift apart.
//!
//! [`logmdigamma`] exists because `ln(x) - digamma(x)` cancels catastrophically
//! for large `x`: both terms are `ln(x)` to leading order and the answer is
//! `1/(2x)`. At `x = 1e6` the naive form keeps about four digits. statmod
//! carries the function for exactly this reason and limma leans on it.
//!
//! ### Domain
//!
//! Everything here is defined for `x > 0` only, which is all edgeR ever asks
//! for. Non-positive arguments return `NaN` rather than reflecting across the
//! poles.

use statrs::function::gamma;

////////////
// Consts //
////////////

/// Argument at (or above) which the asymptotic Bernoulli series is used
/// directly; below it the recurrence walks `x` up in unit steps first.
const RECURRENCE_THRESHOLD: f64 = 20.0;

/// Highest polygamma order [`polygamma`] will evaluate.
///
/// The asymptotic series carries factorials of `2k + n`, and past this order the
/// truncation stops being negligible at the shifted argument. edgeR needs orders
/// zero through two; ten is slack, not a target.
const POLYGAMMA_MAX_ORDER: usize = 10;

/// Bernoulli numbers `B_2, B_4, ..., B_20`, used by the general polygamma
/// asymptotic series.
const BERNOULLI_EVEN: [f64; 10] = [
    1.0 / 6.0,
    -1.0 / 30.0,
    1.0 / 42.0,
    -1.0 / 30.0,
    5.0 / 66.0,
    -691.0 / 2730.0,
    7.0 / 6.0,
    -3617.0 / 510.0,
    43867.0 / 798.0,
    -174611.0 / 330.0,
];

/// Coefficients `B_2k / (2k)` for `k = 1..=8`, the asymptotic expansion of
/// `ln(z) - digamma(z)` in powers of `z^-2`.
///
/// Held separately from [`BERNOULLI_EVEN`] so the `digamma` path is a plain
/// Horner evaluation with no factorial bookkeeping.
const LOG_MINUS_DIGAMMA_COEFFS: [f64; 8] = [
    1.0 / 12.0,
    -1.0 / 120.0,
    1.0 / 252.0,
    -1.0 / 240.0,
    1.0 / 132.0,
    -691.0 / 32760.0,
    1.0 / 12.0,
    -3617.0 / 8160.0,
];

/// Newton iteration budget for [`trigamma_inverse`], matching limma.
const TRIGAMMA_INVERSE_MAX_ITER: usize = 50;

/// Relative step size at which [`trigamma_inverse`] declares convergence.
///
/// limma stops at `1e-8`; the extra two digits cost at most one more iteration
/// and put the round trip inside `f64` rounding.
const TRIGAMMA_INVERSE_TOL: f64 = 1e-10;

/// Above this, `trigamma_inverse(x)` is `1/sqrt(x)` to full precision, since
/// `trigamma(y) ~ 1/y^2` as `y -> 0`.
const TRIGAMMA_INVERSE_LARGE: f64 = 1e7;

/// Below this, `trigamma_inverse(x)` is `1/x` to full precision, since
/// `trigamma(y) ~ 1/y` as `y -> inf`.
const TRIGAMMA_INVERSE_SMALL: f64 = 1e-6;

/////////////
// Helpers //
/////////////

/// Whether an argument falls outside the `x > 0` domain this module supports.
///
/// Written out rather than as `!(x > 0.0)` so the `NaN` case is explicit: a
/// `NaN` argument propagates, it does not silently take the positive branch.
///
/// ### Params
///
/// * `x` - Argument to check
///
/// ### Returns
///
/// `true` when `x` is `NaN` or non-positive.
#[inline]
fn is_outside_domain(x: f64) -> bool {
    x.is_nan() || x <= 0.0
}

/// Walks `x` up in unit steps until it reaches `target`, accumulating the
/// recurrence correction on the way.
///
/// Every function here shares the same shape: shift the argument into the range
/// where the asymptotic series is good, and carry a sum of `(x + j)^-power` for
/// the steps taken. The sum is accumulated from the largest `j` downwards so the
/// small terms are not rounded away against the first one.
///
/// ### Params
///
/// * `x` - Starting argument, strictly positive
/// * `target` - Argument to reach
/// * `power` - Exponent of the correction terms: one for the digamma family,
///   `n + 1` for polygamma order `n`
///
/// ### Returns
///
/// The shifted argument `z >= target` and `sum_{j=0}^{m-1} (x + j)^-power`,
/// where `m` is the number of steps taken.
fn shift_to_threshold(x: f64, target: f64, power: i32) -> (f64, f64) {
    let mut z = x;
    let mut steps = 0_u32;
    while z < target {
        z += 1.0;
        steps += 1;
    }
    let mut correction = 0.0;
    for j in (0..steps).rev() {
        correction += (x + f64::from(j)).powi(-power);
    }
    (z, correction)
}

/// Asymptotic expansion of `ln(z) - psi(z)` for `z >= RECURRENCE_THRESHOLD`.
///
/// `1/(2z) + sum_k B_2k / (2k z^2k)`, evaluated by Horner in `z^-2`. Both terms
/// are positive-definite small quantities, so there is no cancellation to lose.
///
/// ### Params
///
/// * `z` - Argument, expected at or above [`RECURRENCE_THRESHOLD`]
///
/// ### Returns
///
/// `ln(z) - psi(z)`.
#[inline]
fn log_minus_digamma_asymptotic(z: f64) -> f64 {
    let inv_z = 1.0 / z;
    let inv_z2 = inv_z * inv_z;
    let mut tail = 0.0;
    for &c in LOG_MINUS_DIGAMMA_COEFFS.iter().rev() {
        tail = tail * inv_z2 + c;
    }
    0.5 * inv_z + tail * inv_z2
}

/// Asymptotic Bernoulli series for `psi^(n)(z)`, order `n >= 1`.
///
/// The factorial ratio `(2k + n - 1)! / (2k)!` is formed as the product of the
/// `n - 1` integers above `2k`, which keeps it out of overflow range for every
/// order this module accepts.
///
/// ### Params
///
/// * `n` - Order, at least one and at most [`POLYGAMMA_MAX_ORDER`]
/// * `z` - Argument, expected at or above `RECURRENCE_THRESHOLD + n`
///
/// ### Returns
///
/// `psi^(n)(z)`.
fn polygamma_asymptotic(n: usize, z: f64) -> f64 {
    let inv_z = 1.0 / z;
    let inv_z2 = inv_z * inv_z;
    let inv_z_n = inv_z.powi(n as i32);

    let mut sum = factorial(n - 1) * inv_z_n + 0.5 * factorial(n) * inv_z_n * inv_z;
    let mut inv_z_pow = inv_z_n * inv_z2;
    for (i, &b) in BERNOULLI_EVEN.iter().enumerate() {
        let two_k = 2 * (i + 1);
        let mut ratio = 1.0;
        for j in 1..n {
            ratio *= (two_k + j) as f64;
        }
        sum += b * ratio * inv_z_pow;
        inv_z_pow *= inv_z2;
    }

    // (-1)^(n-1)
    if n.is_multiple_of(2) { -sum } else { sum }
}

/// Factorial of a small integer as `f64`.
///
/// Only ever called with arguments up to [`POLYGAMMA_MAX_ORDER`], well inside
/// the exactly representable range.
///
/// ### Params
///
/// * `n` - Argument
///
/// ### Returns
///
/// `n!`.
#[inline]
fn factorial(n: usize) -> f64 {
    (1..=n).map(|i| i as f64).product()
}

///////////////
// Front end //
///////////////

/// Natural logarithm of the gamma function.
///
/// ### Params
///
/// * `x` - Argument. Poles at zero and the negative integers give `inf`.
///
/// ### Returns
///
/// `ln|Gamma(x)|`, via the Lanczos approximation in `statrs`.
#[inline]
pub fn ln_gamma(x: f64) -> f64 {
    gamma::ln_gamma(x)
}

/// Digamma function, the logarithmic derivative of the gamma function.
///
/// Recurrence up to [`RECURRENCE_THRESHOLD`], then `ln(z)` minus the asymptotic
/// expansion of `ln(z) - psi(z)`. Sharing that expansion with [`logmdigamma`]
/// keeps the two mutually consistent.
///
/// ### Params
///
/// * `x` - Argument, must be strictly positive.
///
/// ### Returns
///
/// `psi(x)`, or `NaN` for `x <= 0`.
pub fn digamma(x: f64) -> f64 {
    if is_outside_domain(x) {
        return f64::NAN;
    }
    let (z, shifted) = shift_to_threshold(x, RECURRENCE_THRESHOLD, 1);
    // psi(x) = psi(z) - sum_j 1/(x + j)
    z.ln() - log_minus_digamma_asymptotic(z) - shifted
}

/// Trigamma function, the first derivative of the digamma function.
///
/// ### Params
///
/// * `x` - Argument, must be strictly positive.
///
/// ### Returns
///
/// `psi'(x)`, or `NaN` for `x <= 0`. Bit-identical to `polygamma(1, x)`.
#[inline]
pub fn trigamma(x: f64) -> f64 {
    polygamma(1, x)
}

/// Polygamma function of arbitrary order.
///
/// Order zero delegates to [`digamma`]. Higher orders walk the recurrence
/// `psi^(n)(x) = psi^(n)(x + 1) + (-1)^(n+1) n! / x^(n+1)` up to
/// `RECURRENCE_THRESHOLD + n`, then evaluate the Bernoulli series
///
/// ```text
/// psi^(n)(z) = (-1)^(n-1) [ (n-1)!/z^n + n!/(2 z^(n+1))
///                           + sum_k B_2k (2k+n-1)!/(2k)! z^-(2k+n) ]
/// ```
///
/// The shift target grows with `n` because the series is in `n/z` as much as in
/// `1/z`.
///
/// ### Params
///
/// * `n` - Order of differentiation. Orders above [`POLYGAMMA_MAX_ORDER`] give
///   `NaN`; edgeR never asks past two.
/// * `x` - Argument, must be strictly positive.
///
/// ### Returns
///
/// `psi^(n)(x)`, or `NaN` outside the supported domain.
///
/// ### References
///
/// Abramowitz and Stegun, Handbook of Mathematical Functions, 6.4.11-6.4.12.
pub fn polygamma(n: usize, x: f64) -> f64 {
    if n == 0 {
        return digamma(x);
    }
    if n > POLYGAMMA_MAX_ORDER || is_outside_domain(x) {
        return f64::NAN;
    }

    let target = RECURRENCE_THRESHOLD + n as f64;
    let (z, shifted) = shift_to_threshold(x, target, n as i32 + 1);
    // (-1)^(n+1), the sign the recurrence attaches to n! sum_j (x + j)^-(n+1)
    let sign = if n.is_multiple_of(2) { -1.0 } else { 1.0 };
    sign * factorial(n) * shifted + polygamma_asymptotic(n, z)
}

/// Inverse of the trigamma function.
///
/// A port of limma's `trigammaInverse()`. Newton's method is applied to
/// `f(y) = 1/psi'(y) - 1/x` rather than to `psi'(y) - x`, because `1/psi'` is
/// close to linear in `y`, which is what makes the `0.5 + 1/x` start good enough
/// to converge in a handful of steps across the whole range. The two clamps
/// cover the tails, where the starting value is already the answer.
///
/// Note that edgePython's `_trigamma_inverse` returns `1/x` in the `x > 1e7`
/// branch. That is wrong: for large `x` the root is small, where
/// `psi'(y) ~ 1/y^2`, so the answer is `1/sqrt(x)`. limma has it right and this
/// follows limma.
///
/// ### Params
///
/// * `x` - Target trigamma value, must be strictly positive.
///
/// ### Returns
///
/// The `y > 0` with `trigamma(y) = x`, or `NaN` for `x <= 0`.
///
/// ### References
///
/// Smyth, limma, `R/fitFDist.R`.
pub fn trigamma_inverse(x: f64) -> f64 {
    if is_outside_domain(x) {
        return f64::NAN;
    }
    if x > TRIGAMMA_INVERSE_LARGE {
        return 1.0 / x.sqrt();
    }
    if x < TRIGAMMA_INVERSE_SMALL {
        return 1.0 / x;
    }

    let mut y = 0.5 + 1.0 / x;
    for _ in 0..TRIGAMMA_INVERSE_MAX_ITER {
        let tri = trigamma(y);
        let dif = tri * (1.0 - tri / x) / polygamma(2, y);
        y += dif;
        // A Newton step can overshoot past the pole at zero on a bad start.
        // limma has no such reset; this guard is ours, and restarting from `x`
        // is cheaper than returning a root the function has no business having.
        if y <= 0.0 {
            y = x;
        }
        if (dif / y).abs() < TRIGAMMA_INVERSE_TOL {
            break;
        }
    }
    y
}

/// `ln(x) - digamma(x)`, computed without subtractive cancellation.
///
/// The naive difference loses roughly `log10(x)` digits, because both terms are
/// `ln(x)` to leading order while the result is `1/(2x)`. Here the shift is
/// applied to the recurrence identity
///
/// ```text
/// ln(x) - psi(x) = ln(x/z) + [ln(z) - psi(z)] + sum_j 1/(x + j)
/// ```
///
/// so nothing large is ever subtracted from anything large.
///
/// ### Params
///
/// * `x` - Argument, must be strictly positive.
///
/// ### Returns
///
/// `ln(x) - psi(x)`, always positive, or `NaN` for `x <= 0`.
///
/// ### References
///
/// Smyth, statmod, `src/logmdigamma.c`.
pub fn logmdigamma(x: f64) -> f64 {
    if is_outside_domain(x) {
        return f64::NAN;
    }
    let (z, shifted) = shift_to_threshold(x, RECURRENCE_THRESHOLD, 1);
    (x / z).ln() + log_minus_digamma_asymptotic(z) + shifted
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// `scipy.special.gammaln(x)` for each `x`.
    ///
    /// Every reference table here is the shortest round-tripping `repr()` of
    /// the scipy `float64`, so the literal is the same bit pattern scipy
    /// produced.
    const LN_GAMMA_REF: [(f64, f64); 11] = [
        (0.001, 6.907178885383853),
        (0.1, 2.252712651734206),
        (0.5, 0.5723649429247),
        (1.0, 0.0),
        (2.5, 0.2846828704729192),
        (7.0, 6.579251212010101),
        (19.9, 39.04308858123621),
        (20.0, 39.339884187199495),
        (20.1, 39.637192503636946),
        (100.0, 359.1342053695754),
        (1000000.0, 12815504.569147611),
    ];

    /// `scipy.special.digamma(x)` for each `x`.
    const DIGAMMA_REF: [(f64, f64); 11] = [
        (0.001, -1000.5755719318103),
        (0.1, -10.423754940411076),
        (0.5, -1.9635100260214235),
        (1.0, -0.5772156649015329),
        (2.5, 0.7031566406452432),
        (7.0, 1.872784335098467),
        (19.9, 2.965383724267679),
        (20.0, 2.970523992242149),
        (20.1, 2.9756379786475406),
        (100.0, 4.600161852738088),
        (1000000.0, 13.81551005796419),
    ];

    /// `scipy.special.polygamma(1, x)` for each `x`.
    const TRIGAMMA_REF: [(f64, f64); 11] = [
        (0.001, 1000001.6425331959),
        (0.1, 101.43329915079275),
        (0.5, 4.93480220054468),
        (1.0, 1.6449340668482266),
        (2.5, 0.4903577561002349),
        (7.0, 0.15354517795933756),
        (19.9, 0.05153498898307066),
        (20.0, 0.051270822935203124),
        (20.1, 0.051009350700253045),
        (100.0, 0.010050166663333573),
        (1000000.0, 1.0000005000001667e-6),
    ];

    /// `scipy.special.polygamma(2, x)` for each `x`.
    const POLYGAMMA_2_REF: [(f64, f64); 7] = [
        (0.5, -16.828796644234316),
        (1.0, -2.404113806319188),
        (2.5, -0.23620405164172736),
        (7.0, -0.023530472985855234),
        (19.9, -0.0026552682774901074),
        (20.1, -0.0026013906050221915),
        (100.0, -0.00010100499983335001),
    ];

    /// `scipy.special.polygamma(3, x)` for each `x`.
    const POLYGAMMA_3_REF: [(f64, f64); 7] = [
        (0.5, 97.40909103400242),
        (1.0, 6.493939402266829),
        (2.5, 0.2239058488172521),
        (7.0, 0.007198198563125444),
        (19.9, 0.00027355760534575216),
        (20.1, 0.00026527568555815585),
        (100.0, 2.030199990001333e-6),
    ];

    /// `scipy.special.polygamma(4, x)` for each `x`.
    const POLYGAMMA_4_REF: [(f64, f64); 7] = [
        (0.5, -771.4742498266678),
        (1.0, -24.88626612344089),
        (2.5, -0.31375599950673133),
        (7.0, -0.00329677158902638),
        (19.9, -4.226537963454322e-5),
        (20.1, -4.056830388512717e-5),
        (100.0, -6.120999930011997e-8),
    ];

    /// `mpmath.log(x) - mpmath.digamma(x)` at 40 decimal digits, rounded to
    /// `f64`. mpmath rather than scipy because the point of `logmdigamma` is
    /// that the naive `log(x) - digamma(x)` scipy would compute is itself the
    /// thing under test.
    const LOGMDIGAMMA_REF: [(f64, f64); 9] = [
        (0.001, 993.6678166528282),
        (0.5, 1.2703628454614782),
        (1.0, 0.5772156649015329),
        (2.5, 0.2131340912289119),
        (7.0, 0.07312581395684617),
        (100.0, 0.005008333250003967),
        (1000000.0, 5.000000833333334e-7),
        (100000000.0, 5.0000000083333336e-9),
        (10000000000.0, 5.0000000000833335e-11),
    ];

    #[test]
    fn test_ln_gamma_matches_scipy_across_magnitudes() {
        // The Lanczos approximation in statrs is the loosest thing in this
        // module: about 1e-14 relative, against 1e-15 for everything else.
        for (x, expected) in LN_GAMMA_REF {
            assert_relative_eq!(ln_gamma(x), expected, max_relative = 1e-13, epsilon = 1e-15);
        }
    }

    #[test]
    fn test_digamma_matches_scipy_across_magnitudes() {
        for (x, expected) in DIGAMMA_REF {
            assert_relative_eq!(digamma(x), expected, max_relative = 1e-14);
        }
    }

    #[test]
    fn test_trigamma_matches_scipy_across_magnitudes() {
        for (x, expected) in TRIGAMMA_REF {
            assert_relative_eq!(trigamma(x), expected, max_relative = 1e-14);
        }
    }

    #[test]
    fn test_polygamma_matches_scipy_for_orders_zero_to_four() {
        for (x, expected) in DIGAMMA_REF {
            assert_relative_eq!(polygamma(0, x), expected, max_relative = 1e-14);
        }
        for (x, expected) in TRIGAMMA_REF {
            assert_relative_eq!(polygamma(1, x), expected, max_relative = 1e-14);
        }
        for (x, expected) in POLYGAMMA_2_REF {
            assert_relative_eq!(polygamma(2, x), expected, max_relative = 1e-14);
        }
        for (x, expected) in POLYGAMMA_3_REF {
            assert_relative_eq!(polygamma(3, x), expected, max_relative = 1e-14);
        }
        for (x, expected) in POLYGAMMA_4_REF {
            assert_relative_eq!(polygamma(4, x), expected, max_relative = 1e-14);
        }
    }

    #[test]
    fn test_polygamma_order_one_is_bit_identical_to_trigamma() {
        for x in [1e-3, 0.5, 1.0, 2.5, 7.0, 19.9, 20.0, 20.1, 100.0, 1e6] {
            assert_eq!(polygamma(1, x).to_bits(), trigamma(x).to_bits());
            assert_eq!(polygamma(0, x).to_bits(), digamma(x).to_bits());
        }
    }

    #[test]
    fn test_recurrence_boundary_agrees_from_both_sides() {
        // 19.5 takes a recurrence step, 20.5 goes straight to the series. The
        // exact recurrence relating them is the sharpest available test that
        // the two paths meet: a straddle test cannot distinguish a seam from
        // the genuine change in the function over the interval.
        let lo = RECURRENCE_THRESHOLD - 0.5;
        let hi = lo + 1.0;
        assert!(lo < RECURRENCE_THRESHOLD && hi >= RECURRENCE_THRESHOLD);

        assert_relative_eq!(digamma(hi) - digamma(lo), 1.0 / lo, max_relative = 1e-13);
        assert_relative_eq!(
            trigamma(lo) - trigamma(hi),
            1.0 / (lo * lo),
            max_relative = 1e-13
        );
        for n in 2..=4_usize {
            let sign = if n.is_multiple_of(2) { -1.0 } else { 1.0 };
            assert_relative_eq!(
                polygamma(n, lo) - polygamma(n, hi),
                sign * factorial(n) * lo.powi(-(n as i32 + 1)),
                max_relative = 1e-13
            );
        }
        assert_relative_eq!(
            logmdigamma(lo) - logmdigamma(hi),
            (lo / hi).ln() + 1.0 / lo,
            max_relative = 1e-12
        );
    }

    #[test]
    fn test_digamma_satisfies_recurrence_and_reflection_identities() {
        // psi(x + 1) - psi(x) = 1/x, and psi(1) = -gamma.
        for x in [1e-2, 0.3, 1.7, 6.0, 19.5, 55.0] {
            assert_relative_eq!(digamma(x + 1.0) - digamma(x), 1.0 / x, max_relative = 1e-13);
        }
        assert_relative_eq!(
            digamma(1.0),
            -0.577_215_664_901_532_9_f64,
            max_relative = 1e-15
        );
        // psi(1/2) = -gamma - 2 ln 2
        assert_relative_eq!(
            digamma(0.5),
            -0.577_215_664_901_532_9_f64 - 2.0 * std::f64::consts::LN_2,
            max_relative = 1e-15
        );
    }

    #[test]
    fn test_trigamma_hits_closed_form_values() {
        // psi'(1) = pi^2/6, psi'(1/2) = pi^2/2
        let pi2 = std::f64::consts::PI * std::f64::consts::PI;
        assert_relative_eq!(trigamma(1.0), pi2 / 6.0, max_relative = 1e-14);
        assert_relative_eq!(trigamma(0.5), pi2 / 2.0, max_relative = 1e-14);
    }

    #[test]
    fn test_logmdigamma_beats_naive_form_at_large_x() {
        for (x, expected) in LOGMDIGAMMA_REF {
            assert_relative_eq!(logmdigamma(x), expected, max_relative = 1e-14);
        }

        // The whole reason the function exists: at 1e10 the naive difference
        // has nothing left. Assert the failure so the comparison is not
        // wishful thinking.
        let x: f64 = 1e10;
        let expected: f64 = 5.0000000000833335e-11;
        let naive = x.ln() - digamma(x);
        assert!(
            ((naive - expected) / expected).abs() > 1e-5,
            "naive form was unexpectedly accurate: {naive:e}"
        );
        assert_relative_eq!(logmdigamma(x), expected, max_relative = 1e-13);
    }

    #[test]
    fn test_logmdigamma_agrees_with_naive_form_where_naive_is_safe() {
        for x in [1e-3, 0.25, 1.0, 3.0, 19.5, 21.0, 100.0] {
            assert_relative_eq!(logmdigamma(x), x.ln() - digamma(x), max_relative = 1e-11);
        }
    }

    #[test]
    fn test_trigamma_inverse_round_trips_through_trigamma() {
        // Every y here lands the Newton branch: trigamma(y) stays inside
        // [TRIGAMMA_INVERSE_SMALL, TRIGAMMA_INVERSE_LARGE].
        for y in [1e-3, 0.05, 0.5, 1.0, 2.5, 10.0, 100.0, 5000.0, 1e5] {
            let x = trigamma(y);
            assert!(x > TRIGAMMA_INVERSE_SMALL && x < TRIGAMMA_INVERSE_LARGE);
            assert_relative_eq!(trigamma_inverse(x), y, max_relative = 1e-13);
        }
    }

    #[test]
    fn test_trigamma_inverse_recovers_target_value() {
        for x in [1e-5, 0.01, 0.2, 1.0, 5.0, 500.0, 1e6] {
            let y = trigamma_inverse(x);
            assert!(y > 0.0, "trigamma_inverse({x:e}) returned {y:e}");
            assert_relative_eq!(trigamma(y), x, max_relative = 1e-13);
        }
    }

    #[test]
    fn test_trigamma_inverse_clamps_in_the_tails() {
        // Above 1e7 the root is 1/sqrt(x); below 1e-6 it is 1/x.
        assert_relative_eq!(trigamma_inverse(1e12), 1e-6, max_relative = 1e-12);
        assert_relative_eq!(trigamma_inverse(1e-9), 1e9, max_relative = 1e-12);

        // limma's clamps drop the next term of the expansion, so the round trip
        // through the tails is only good to the size of that term: psi'(1/sqrt(x))
        // carries an extra pi^2/6 on top of x, and psi'(1/x) an extra x^2/2.
        // At the clamp boundaries that is ~1e-7 relative, not machine precision.
        for x in [2e7, 1e9, 1e14] {
            assert_relative_eq!(trigamma(trigamma_inverse(x)), x, max_relative = 1e-7);
        }
        for x in [1e-7, 1e-9, 1e-14] {
            assert_relative_eq!(trigamma(trigamma_inverse(x)), x, max_relative = 1e-7);
        }
    }

    #[test]
    fn test_non_positive_arguments_return_nan() {
        for x in [0.0, -1.0, -0.5] {
            assert!(digamma(x).is_nan());
            assert!(trigamma(x).is_nan());
            assert!(polygamma(2, x).is_nan());
            assert!(logmdigamma(x).is_nan());
            assert!(trigamma_inverse(x).is_nan());
        }
        assert!(polygamma(POLYGAMMA_MAX_ORDER + 1, 3.0).is_nan());
    }
}
