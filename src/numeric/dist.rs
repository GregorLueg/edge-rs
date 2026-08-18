//! Distribution tails: everything edgePython calls `scipy.stats` for.
//!
//! Normal, chi-squared, Student's t, F, beta, gamma and the negative binomial,
//! each as plain functions rather than distribution objects. edgeR only ever
//! wants a tail probability or a quantile, so there is nothing worth carrying
//! state for.
//!
//! ### Survival functions, not `1 - cdf`
//!
//! Every p-value edgeR reports is an upper tail. `1 - cdf` is exact only down
//! to about 1e-16 and returns a flat `0.0` below that, so `chisq_sf(200, 1)`
//! would come back as zero instead of 2.09e-45. Each `_sf` here goes through
//! the complement directly: the upper regularised incomplete gamma for the
//! normal and chi-squared, and the regularised incomplete beta with its
//! arguments swapped for t, F and beta. None of them subtracts from one.
//!
//! ### Numeric policy
//!
//! `f64` throughout, per the crate policy in `lib.rs`. These sit inside
//! likelihood evaluation where the arithmetic is a difference of large logs.
//!
//! ### What comes from `statrs` and what does not
//!
//! The incomplete beta `beta_reg` and `ln_gamma` come from `statrs`, and so
//! does `norm_ppf`, through `statrs::function::erf::erfc_inv`. How well they
//! hold depends on which end of the distribution is being read. Measured
//! against R 4.5:
//!
//! * The small tails, which is where every p-value and every quantile edgeR
//!   reports comes from, are good to 1e-14 or better: `chisq_sf(200, 1)`
//!   1.1e-14 relative, `norm_sf(8)` 6.3e-16, `beta_cdf(1e-12, 0.3, 0.7)`
//!   8.8e-16, `beta_ppf(1e-12, 0.3, 0.7)` 8.6e-16, `gamma_ppf(1e-12, 2.5, 1)`
//!   2.9e-15, `t_ppf(1e-12, 5)` 8.7e-16, and `norm_ppf` 2e-16 from `p = 1e-300`
//!   upwards.
//! * The near-one tails are three to five orders worse, because they come out
//!   of `beta_reg`'s internal symmetry swap `1 - I(1-x)` and lose the
//!   cancellation: `f_sf(2, 1, 1e5)` is 4.2e-10 relative and
//!   `beta_sf(1e-12, 0.3, 0.7)` 1.4e-9. Nothing in edgeR is read from that end,
//!   but do not quote these functions as full precision either way.
//!
//! Three things are not taken from `statrs`:
//!
//! * `statrs::function::erf` carries Boost's *single* precision coefficient
//!   tables (`b = 0.3440242112` and friends, ten digits) on the `f64` path, so
//!   `erfc` is only good to about 4e-11 for arguments above 0.5. Since
//!   `erfc(z) = Q(1/2, z^2)`, the normal tails are built from the incomplete
//!   gamma below instead, which costs a handful of iterations and buys three
//!   digits. `erfc_inv` is a different routine and does not inherit that, which
//!   is why `norm_ppf` is left on it.
//! * `statrs::function::gamma::gamma_lr` returns a flat `0.0` for any
//!   `x < 1.1e-15`, which puts a floor under `gamma_cdf` and strands the gamma
//!   quantile solver on a plateau for small shapes.
//! * `statrs::function::beta::inv_beta_reg` is Algorithm AS 109, whose
//!   published floor is `FPU = 1e-30`. It gives up around there and returns
//!   answers wrong by orders of magnitude for a quantile deeper in the tail
//!   (`beta_ppf(1e-12, 0.3, 0.7)` comes back as 1.4e-16 against a true
//!   1.66e-40). It is used only as a starting point here, polished by Newton on
//!   `ln I` against `ln p`.
//!
//! `statrs`'s distribution types are not used at all: their `inverse_cdf`
//! panics on out-of-range input, `Gamma::inverse_cdf` is a truncated bisection
//! nowhere near `f64` precision, and `FisherSnedecor::sf` forms
//! `1 - d1 x / (d1 x + d2)`, which cancels in exactly the tail edgeR reads.

use statrs::function::beta::{beta_reg, inv_beta_reg, ln_beta};
use statrs::function::erf::erfc_inv;
use statrs::function::gamma::ln_gamma;

use crate::errors::EdgeErrors;

/// `sqrt(2)`, the scale between the standard normal and the error function.
const SQRT_2: f64 = std::f64::consts::SQRT_2;

/// Relative convergence tolerance for the incomplete gamma series and
/// continued fraction. One ulp; both recurrences reach it in a few tens of
/// terms over the range edgeR uses.
const INC_GAMMA_EPS: f64 = 2.220446049250313e-16;

/// Iteration budget for the incomplete gamma recurrences.
///
/// The series needs roughly `sqrt(a)` terms near the mean, so this covers
/// shapes into the millions. It is a runaway guard, not a working limit: on
/// exhaustion the best available partial sum is returned rather than an error,
/// because every caller here is a tail probability that is already converged to
/// well past `f64` resolution by then.
const INC_GAMMA_MAX_ITER: usize = 10_000;

/// Floor for the modified Lentz continued fraction, guarding a zero pivot.
///
/// ### References
///
/// Press et al., Numerical Recipes, 3rd ed., section 6.2
const LENTZ_TINY: f64 = 1e-300;

/// Iteration budget for the polished incomplete beta inverse.
///
/// The bracket is expanded with a doubling step and then halved whenever
/// Newton leaves it, so this is a runaway guard rather than a working limit;
/// convergence from an AS 109 start is normally two or three steps.
const INV_BETA_MAX_ITER: usize = 128;

/// Relative convergence tolerance for the incomplete beta inverse, in `ln(x)`.
const INV_BETA_REL_TOL: f64 = 1e-15;

/// Iteration budget for the hand-rolled gamma quantile.
///
/// The safeguarded step always halves a bracket in log-space when Newton
/// misbehaves, so a bracket 40 wide in `ln(x)` is exhausted long before this.
/// The cap only exists so a pathological shape cannot spin forever.
const GAMMA_PPF_MAX_ITER: usize = 128;

/// Crossover on `x^2` between the two incomplete beta forms of the t tails.
///
/// The mass inside `(0, |x|)` and the mass beyond `|x|` are equal at the upper
/// quartile, which runs from 1.0 at `df = 1` down to 0.4549 as `df` grows. Take
/// the inner form below this and the outer form above it and the quantity being
/// evaluated directly is always the smaller of the two, so the other one is
/// recovered from `0.5 - .` while it is still the larger. Sitting between the
/// two extremes, 0.7 leaves neither branch worse than a factor of about two
/// from balanced for any `df`.
const T_INNER_OUTER_SWITCH: f64 = 0.7;

/// Relative convergence tolerance for the gamma quantile, in `ln(x)`.
///
/// Roughly 4 ulp. Tightening further just cycles on the last bit of the
/// incomplete gamma, which is itself only good to a few ulp.
const GAMMA_PPF_REL_TOL: f64 = 1e-15;

/////////////////////////////////
// Regularised incomplete gamma //
/////////////////////////////////

/// The shared prefactor `exp(a ln x - x - ln Gamma(a))` of both branches.
///
/// Formed in logs so it underflows cleanly to zero in the far tail rather than
/// overflowing on `x^a` first.
///
/// ### Params
///
/// * `a` - Shape, strictly positive
/// * `x` - Argument, strictly positive
///
/// ### Returns
///
/// The prefactor, or 0.0 where the exponent underflows.
fn inc_gamma_prefactor(a: f64, x: f64) -> f64 {
    (a * x.ln() - x - ln_gamma(a)).exp()
}

/// Regularised lower incomplete gamma `P(a, x)`.
///
/// Series below `x = a + 1`, complement of the continued fraction above it,
/// which is where each converges fastest and neither cancels.
///
/// Unlike `statrs::function::gamma::gamma_lr` there is no cutoff at small `x`:
/// the series prefactor is evaluated in logs, so `P(0.3, 1e-30)` comes back as
/// 1e-9 rather than zero.
///
/// ### Params
///
/// * `a` - Shape, strictly positive
/// * `x` - Argument, non-negative
///
/// ### Returns
///
/// `P(a, x)` in `[0, 1]`.
///
/// ### References
///
/// Press et al., Numerical Recipes, 3rd ed., section 6.2
fn reg_gamma_lower(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x < a + 1.0 {
        gamma_series(a, x)
    } else {
        1.0 - gamma_cont_frac(a, x)
    }
}

/// Regularised upper incomplete gamma `Q(a, x)`.
///
/// The continued fraction above `x = a + 1`, evaluated directly, which is what
/// keeps `chisq_sf(1000, 1)` at 1.8e-219 rather than zero.
///
/// ### Params
///
/// * `a` - Shape, strictly positive
/// * `x` - Argument, non-negative
///
/// ### Returns
///
/// `Q(a, x) = 1 - P(a, x)` in `[0, 1]`.
///
/// ### References
///
/// Press et al., Numerical Recipes, 3rd ed., section 6.2
fn reg_gamma_upper(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    if x < a + 1.0 {
        1.0 - gamma_series(a, x)
    } else {
        gamma_cont_frac(a, x)
    }
}

/// Series representation of `P(a, x)`, for `x < a + 1`.
///
/// `P(a, x) = exp(a ln x - x - ln Gamma(a)) * sum_{n>=0} x^n / (a (a+1)...(a+n))`.
/// Every term is positive, so there is nothing to cancel.
///
/// ### Params
///
/// * `a` - Shape, strictly positive
/// * `x` - Argument, strictly positive and below `a + 1`
///
/// ### Returns
///
/// `P(a, x)`.
fn gamma_series(a: f64, x: f64) -> f64 {
    let mut ap = a;
    let mut term = 1.0 / a;
    let mut sum = term;
    for _ in 0..INC_GAMMA_MAX_ITER {
        ap += 1.0;
        term *= x / ap;
        sum += term;
        if term.abs() < sum.abs() * INC_GAMMA_EPS {
            break;
        }
    }
    sum * inc_gamma_prefactor(a, x)
}

/// Continued fraction representation of `Q(a, x)`, for `x >= a + 1`.
///
/// Evaluated by modified Lentz, which is stable where the naive recurrence
/// overflows.
///
/// ### Params
///
/// * `a` - Shape, strictly positive
/// * `x` - Argument, at least `a + 1`
///
/// ### Returns
///
/// `Q(a, x)`.
fn gamma_cont_frac(a: f64, x: f64) -> f64 {
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / LENTZ_TINY;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..=INC_GAMMA_MAX_ITER {
        let i = i as f64;
        let an = -i * (i - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < LENTZ_TINY {
            d = LENTZ_TINY;
        }
        c = b + an / c;
        if c.abs() < LENTZ_TINY {
            c = LENTZ_TINY;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() <= INC_GAMMA_EPS {
            break;
        }
    }
    h * inc_gamma_prefactor(a, x)
}

//////////////////
// Validation   //
//////////////////

/// Checks that a degrees-of-freedom-like parameter is finite and positive.
///
/// ### Params
///
/// * `name` - Parameter name, used verbatim in the error message
/// * `value` - Value supplied by the caller
///
/// ### Returns
///
/// `Ok(())`, or [`EdgeErrors::InvalidArgument`] naming the parameter and value.
fn check_positive(name: &str, value: f64) -> Result<(), EdgeErrors> {
    if !(value.is_finite() && value > 0.0) {
        return Err(EdgeErrors::InvalidArgument(format!(
            "'{name}' must be finite and strictly positive; got {value}."
        )));
    }
    Ok(())
}

/// Checks that a probability lies in the closed unit interval.
///
/// ### Params
///
/// * `name` - Parameter name, used verbatim in the error message
/// * `value` - Value supplied by the caller
///
/// ### Returns
///
/// `Ok(())`, or [`EdgeErrors::InvalidArgument`] naming the parameter and value.
fn check_probability(name: &str, value: f64) -> Result<(), EdgeErrors> {
    if !(value.is_finite() && (0.0..=1.0).contains(&value)) {
        return Err(EdgeErrors::InvalidArgument(format!(
            "'{name}' must be a probability in [0, 1]; got {value}."
        )));
    }
    Ok(())
}

////////////////////////////////
// Incomplete beta inverse    //
////////////////////////////////

/// Inverts the regularised incomplete beta on its lower half.
///
/// `inv_beta_reg` from `statrs` is Algorithm AS 109, which carries an explicit
/// `1e-30` floor and drifts by orders of magnitude below roughly `p = 1e-12`
/// for small shapes. It is taken here only as a starting point, then polished
/// by Newton on `ln I(x; a, b) = ln p` in `t = ln x`.
///
/// The log-log form is what makes the polish cheap. For small `x`,
/// `I ~ x^a / (a B(a, b))`, so `d ln I / d ln x -> a`: the residual is close to
/// linear in `t` however many decades the start is out by, and two or three
/// steps land it. A bracket doubled outwards from the start keeps a bad start
/// from diverging.
///
/// Only the lower half is solved. Callers wanting a quantile above the median
/// flip the shapes and the probability and take the complement, which is what
/// keeps the returned value free of cancellation on whichever side is small.
///
/// ### Params
///
/// * `a` - First shape, strictly positive
/// * `b` - Second shape, strictly positive
/// * `p` - Target probability in `[0, 0.5]`
///
/// ### Returns
///
/// `x` with `I(x; a, b) = p`, or [`EdgeErrors::NoConvergence`].
///
/// ### References
///
/// Cran, Martin & Robson, Algorithm AS 109, Applied Statistics, 1977
fn inv_beta_reg_lower(a: f64, b: f64, p: f64) -> Result<f64, EdgeErrors> {
    debug_assert!(p <= 0.5, "inv_beta_reg_lower solves the lower half only");
    if p <= 0.0 {
        return Ok(0.0);
    }
    let ln_b = ln_beta(a, b);
    let ln_p = p.ln();

    // g(t) = ln I(e^t; a, b) - ln p, increasing in t. -inf where I underflows,
    // which still gives the right sign for the bracket.
    let g = |t: f64| beta_reg(a, b, t.exp().min(1.0)).ln() - ln_p;

    let start = inv_beta_reg(a, b, p);
    let mut t = if start > 0.0 && start < 1.0 {
        start.ln()
    } else {
        // Small-x asymptote I ~ x^a / (a B(a, b)), inverted.
        (ln_p + a.ln() + ln_b) / a
    };

    // x = 1 always satisfies I = 1 >= p, so t = 0 is a standing upper bracket.
    let mut lo = f64::NEG_INFINITY;
    let mut hi = 0.0_f64;
    if t >= hi {
        t = -1.0;
    }
    let mut step = 1.0;
    if g(t) < 0.0 {
        lo = t;
        for _ in 0..INV_BETA_MAX_ITER {
            let probe = (t + step).min(0.0);
            if g(probe) >= 0.0 {
                hi = probe;
                break;
            }
            lo = probe;
            step *= 2.0;
        }
    } else {
        hi = t;
        for _ in 0..INV_BETA_MAX_ITER {
            let probe = t - step;
            if g(probe) < 0.0 {
                lo = probe;
                break;
            }
            hi = probe;
            step *= 2.0;
        }
    }

    for _ in 0..INV_BETA_MAX_ITER {
        if !(t > lo && t < hi) {
            t = if lo.is_finite() {
                0.5 * (lo + hi)
            } else {
                hi - 1.0
            };
        }
        let x = t.exp();
        let ln_i = beta_reg(a, b, x).ln();
        let residual = ln_i - ln_p;
        if residual < 0.0 {
            lo = t;
        } else {
            hi = t;
        }
        let tol = INV_BETA_REL_TOL * (1.0 + t.abs());
        if lo.is_finite() && hi - lo <= tol {
            return Ok(t.exp());
        }
        // d(ln I)/dt = x^a (1 - x)^(b - 1) / (B(a, b) I), formed in logs.
        let d = (a * t + (b - 1.0) * (-x).ln_1p() - ln_b - ln_i).exp();
        let t_next = if d > 0.0 && d.is_finite() {
            t - residual / d
        } else if lo.is_finite() {
            0.5 * (lo + hi)
        } else {
            hi - 1.0
        };
        let delta = (t_next - t).abs();
        t = t_next;
        if delta <= tol {
            return Ok(t.clamp(lo, hi).exp());
        }
    }

    Err(EdgeErrors::NoConvergence {
        routine: "inv_beta_reg_lower",
        iterations: INV_BETA_MAX_ITER,
        last_delta: hi - lo,
    })
}

/// Inverts the regularised incomplete beta, returning both `x` and `1 - x`.
///
/// Solving whichever side is below the median and complementing gives the
/// small quantity to full relative accuracy, which is what the F and t
/// quantiles need: both divide by it.
///
/// ### Params
///
/// * `a` - First shape, strictly positive
/// * `b` - Second shape, strictly positive
/// * `p` - Target probability in `[0, 1]`
///
/// ### Returns
///
/// `(x, 1 - x)` with `I(x; a, b) = p`, or [`EdgeErrors::NoConvergence`].
fn inv_beta_reg_pair(a: f64, b: f64, p: f64) -> Result<(f64, f64), EdgeErrors> {
    // 1 - p is exact for p >= 0.5 by Sterbenz, so the flip is free.
    if p <= 0.5 {
        let x = inv_beta_reg_lower(a, b, p)?;
        Ok((x, 1.0 - x))
    } else {
        let y = inv_beta_reg_lower(b, a, 1.0 - p)?;
        Ok((1.0 - y, y))
    }
}

////////////
// Normal //
////////////

/// Standard normal CDF.
///
/// `erfc(z) = Q(1/2, z^2)`, so this is the incomplete gamma at `x^2 / 2`,
/// halved. Below zero it reads the upper branch and above zero the lower one,
/// so the near tail is always the one being computed and the far tail is the
/// one being added to a half.
///
/// ### Params
///
/// * `x` - Quantile
///
/// ### Returns
///
/// `P(Z <= x)`. Total, so no `Result`: every `f64` is in the domain, and `NaN`
/// propagates. For the upper tail use [`norm_sf`].
pub fn norm_cdf(x: f64) -> f64 {
    let h = 0.5 * x * x;
    if x < 0.0 {
        0.5 * reg_gamma_upper(0.5, h)
    } else {
        0.5 * (1.0 + reg_gamma_lower(0.5, h))
    }
}

/// Standard normal survival function.
///
/// The mirror of [`norm_cdf`], never `1 - norm_cdf(x)`. The difference shows
/// from about `x = 8` onwards, where the tail drops below `f64` resolution
/// near one; `norm_sf(37)` is 5.7e-300 and the subtraction gives a flat zero.
///
/// ### Params
///
/// * `x` - Quantile
///
/// ### Returns
///
/// `P(Z > x)`, accurate down to roughly 1e-308.
pub fn norm_sf(x: f64) -> f64 {
    let h = 0.5 * x * x;
    if x > 0.0 {
        0.5 * reg_gamma_upper(0.5, h)
    } else {
        0.5 * (1.0 + reg_gamma_lower(0.5, h))
    }
}

/// Standard normal quantile function.
///
/// `-sqrt(2) * erfc_inv(2p)`, so a tiny `p` is handled in the tail branch of
/// the inverse rather than by inverting a CDF value that has already rounded
/// to zero.
///
/// ### Params
///
/// * `p` - Probability in `[0, 1]`
///
/// ### Returns
///
/// `z` with `P(Z <= z) = p`. `p = 0` gives `-inf` and `p = 1` gives `+inf`,
/// matching `scipy`. Anything outside `[0, 1]` is
/// [`EdgeErrors::InvalidArgument`].
pub fn norm_ppf(p: f64) -> Result<f64, EdgeErrors> {
    check_probability("p", p)?;
    Ok(-SQRT_2 * erfc_inv(2.0 * p))
}

/////////////////
// Chi-squared //
/////////////////

/// Chi-squared survival function.
///
/// The upper regularised incomplete gamma `Q(df/2, x/2)` directly, so the far
/// tail stays accurate: `chisq_sf(200, 1)` is 2.09e-45, where `1 - cdf` returns
/// a flat zero. This is the tail every LRT p-value in edgeR comes out of.
///
/// ### Params
///
/// * `x` - Test statistic, non-negative. Negative values give 1.0.
/// * `df` - Degrees of freedom, finite and strictly positive
///
/// ### Returns
///
/// `P(X > x)`, or [`EdgeErrors::InvalidArgument`] for a non-positive `df`.
pub fn chisq_sf(x: f64, df: f64) -> Result<f64, EdgeErrors> {
    check_positive("df", df)?;
    if x <= 0.0 {
        return Ok(1.0);
    }
    if x.is_infinite() {
        return Ok(0.0);
    }
    Ok(reg_gamma_upper(0.5 * df, 0.5 * x))
}

////////////////////
// Student's t    //
////////////////////

/// The two halves of the t distribution's mass either side of `|x|`.
///
/// With `h = df / (df + x^2)` and `z = x^2 / (df + x^2)`, the outer tail beyond
/// `|x|` is `I(h; df/2, 1/2) / 2` and the mass between zero and `|x|` is
/// `I(z; 1/2, df/2) / 2`. Whichever of the two is the smaller is the one
/// evaluated, per [`T_INNER_OUTER_SWITCH`], and the larger is recovered by
/// subtraction, where it can absorb the loss.
///
/// Both halves of this matter. Forming the inner mass as `0.5 - outer` for a
/// large `|x|` is the obvious blunder, but the reverse costs just as much: `h`
/// at `x = 1e-4, df = 100` is `1 - 1e-10`, whose complement carries only six
/// significant digits, and the tail that comes out of it is wrong in the
/// eleventh.
///
/// ### Params
///
/// * `x` - Quantile
/// * `df` - Degrees of freedom, assumed already validated
///
/// ### Returns
///
/// `(inner, outer)`, the mass in `(0, |x|)` and the mass beyond `|x|`. They sum
/// to a half.
fn t_half_masses(x: f64, df: f64) -> (f64, f64) {
    let x2 = x * x;
    if x2 < T_INNER_OUTER_SWITCH {
        let inner = 0.5 * beta_reg(0.5, 0.5 * df, x2 / (df + x2));
        (inner, 0.5 - inner)
    } else {
        let outer = 0.5 * beta_reg(0.5 * df, 0.5, df / (df + x2));
        (0.5 - outer, outer)
    }
}

/// Student's t CDF.
///
/// ### Params
///
/// * `x` - Quantile
/// * `df` - Degrees of freedom, finite and strictly positive
///
/// ### Returns
///
/// `P(T <= x)`, or [`EdgeErrors::InvalidArgument`] for a non-positive `df`.
/// Accurate in the lower tail; for the upper tail use [`t_sf`].
pub fn t_cdf(x: f64, df: f64) -> Result<f64, EdgeErrors> {
    check_positive("df", df)?;
    if x.is_infinite() {
        return Ok(if x > 0.0 { 1.0 } else { 0.0 });
    }
    let (inner, outer) = t_half_masses(x, df);
    Ok(if x <= 0.0 { outer } else { 0.5 + inner })
}

/// Student's t survival function.
///
/// For `x > 0` this is the incomplete beta itself, never `1 - cdf`, which is
/// what keeps `t_sf(50, 100)` at 7.24e-73 instead of zero. edgeR's moderated t
/// p-values and the t-to-z conversion in `glmTreat` both live here.
///
/// ### Params
///
/// * `x` - Quantile
/// * `df` - Degrees of freedom, finite and strictly positive
///
/// ### Returns
///
/// `P(T > x)`, or [`EdgeErrors::InvalidArgument`] for a non-positive `df`.
pub fn t_sf(x: f64, df: f64) -> Result<f64, EdgeErrors> {
    check_positive("df", df)?;
    if x.is_infinite() {
        return Ok(if x > 0.0 { 0.0 } else { 1.0 });
    }
    let (inner, outer) = t_half_masses(x, df);
    Ok(if x <= 0.0 { 0.5 + inner } else { outer })
}

/// Student's t quantile function.
///
/// Inverts the incomplete beta on whichever side of the median is smaller, so
/// a `p` of 1e-8 is resolved in that tail rather than against a CDF pinned at
/// one.
///
/// ### Params
///
/// * `p` - Probability in `[0, 1]`
/// * `df` - Degrees of freedom, finite and strictly positive
///
/// ### Returns
///
/// `t` with `P(T <= t) = p`. `p = 0` gives `-inf`, `p = 1` gives `+inf`.
/// [`EdgeErrors::InvalidArgument`] for `p` outside `[0, 1]` or non-positive
/// `df`.
pub fn t_ppf(p: f64, df: f64) -> Result<f64, EdgeErrors> {
    check_probability("p", p)?;
    check_positive("df", df)?;
    if p == 0.0 {
        return Ok(f64::NEG_INFINITY);
    }
    if p == 1.0 {
        return Ok(f64::INFINITY);
    }
    // 1 - p is exact for p >= 0.5 (Sterbenz), so the reflection costs nothing.
    let upper = p >= 0.5;
    let tail = if upper { 1.0 - p } else { p };
    // y = I^-1(2 * tail; df/2, 1/2), and t = sqrt(df (1 - y) / y). y is the
    // divisor, so it is the one that has to survive to full relative accuracy.
    let (y, one_minus_y) = inv_beta_reg_pair(0.5 * df, 0.5, 2.0 * tail)?;
    let t = (df * one_minus_y / y).sqrt();
    Ok(if upper { t } else { -t })
}

///////
// F //
///////

/// F survival function.
///
/// `I(df2 / (df1 x + df2); df2/2, df1/2)`, formed from the ratio directly
/// rather than as `1 - df1 x / (df1 x + df2)`. That subtraction is where
/// `statrs`'s own `FisherSnedecor::sf` loses the tail, and the quasi-likelihood
/// F test in `glmQLFTest` reads exactly that tail.
///
/// ### Params
///
/// * `x` - Test statistic, non-negative. Negative values give 1.0.
/// * `df1` - Numerator degrees of freedom, finite and strictly positive
/// * `df2` - Denominator degrees of freedom, finite and strictly positive
///
/// ### Returns
///
/// `P(F > x)`, or [`EdgeErrors::InvalidArgument`] for a non-positive `df`.
pub fn f_sf(x: f64, df1: f64, df2: f64) -> Result<f64, EdgeErrors> {
    check_positive("df1", df1)?;
    check_positive("df2", df2)?;
    if x <= 0.0 {
        return Ok(1.0);
    }
    if x.is_infinite() {
        return Ok(0.0);
    }
    Ok(beta_reg(0.5 * df2, 0.5 * df1, df2 / (df1 * x + df2)))
}

/// F quantile function.
///
/// Inverts the incomplete beta on the smaller tail and recovers `x` from the
/// side that has not cancelled, so `f_ppf(1 - eps, ..)` does not divide by a
/// rounded-off `1 - z`.
///
/// ### Params
///
/// * `p` - Probability in `[0, 1]`
/// * `df1` - Numerator degrees of freedom, finite and strictly positive
/// * `df2` - Denominator degrees of freedom, finite and strictly positive
///
/// ### Returns
///
/// `x` with `P(F <= x) = p`. `p = 0` gives 0.0 and `p = 1` gives `+inf`.
/// [`EdgeErrors::InvalidArgument`] for `p` outside `[0, 1]` or a non-positive
/// `df`.
pub fn f_ppf(p: f64, df1: f64, df2: f64) -> Result<f64, EdgeErrors> {
    check_probability("p", p)?;
    check_positive("df1", df1)?;
    check_positive("df2", df2)?;
    if p == 0.0 {
        return Ok(0.0);
    }
    if p == 1.0 {
        return Ok(f64::INFINITY);
    }
    // z = df1 x / (df1 x + df2), so x = df2 z / (df1 (1 - z)). Both z and its
    // complement come back from the inverse at full relative accuracy, so
    // neither end of the ratio is a cancelled difference.
    let (z, one_minus_z) = inv_beta_reg_pair(0.5 * df1, 0.5 * df2, p)?;
    Ok(df2 * z / (df1 * one_minus_z))
}

//////////
// Beta //
//////////

/// Beta CDF.
///
/// ### Params
///
/// * `x` - Quantile. Clamped: below 0 gives 0.0, above 1 gives 1.0, as `scipy`
///   does.
/// * `a` - First shape, finite and strictly positive
/// * `b` - Second shape, finite and strictly positive
///
/// ### Returns
///
/// `P(X <= x)`, or [`EdgeErrors::InvalidArgument`] for a non-positive shape.
pub fn beta_cdf(x: f64, a: f64, b: f64) -> Result<f64, EdgeErrors> {
    check_positive("a", a)?;
    check_positive("b", b)?;
    if x <= 0.0 {
        return Ok(0.0);
    }
    if x >= 1.0 {
        return Ok(1.0);
    }
    Ok(beta_reg(a, b, x))
}

/// Beta survival function.
///
/// `I(1 - x; b, a)`, the reflected incomplete beta, not `1 - beta_cdf(x)`. The
/// beta approximation in `exactTestBetaApprox` doubles this for its right-tail
/// p-value, so the precision carries straight into the reported value.
///
/// ### Params
///
/// * `x` - Quantile. Clamped: below 0 gives 1.0, above 1 gives 0.0.
/// * `a` - First shape, finite and strictly positive
/// * `b` - Second shape, finite and strictly positive
///
/// ### Returns
///
/// `P(X > x)`, or [`EdgeErrors::InvalidArgument`] for a non-positive shape.
pub fn beta_sf(x: f64, a: f64, b: f64) -> Result<f64, EdgeErrors> {
    check_positive("a", a)?;
    check_positive("b", b)?;
    if x <= 0.0 {
        return Ok(1.0);
    }
    if x >= 1.0 {
        return Ok(0.0);
    }
    Ok(beta_reg(b, a, 1.0 - x))
}

/// Beta quantile function.
///
/// ### Params
///
/// * `p` - Probability in `[0, 1]`
/// * `a` - First shape, finite and strictly positive
/// * `b` - Second shape, finite and strictly positive
///
/// ### Returns
///
/// `x` with `P(X <= x) = p`. [`EdgeErrors::InvalidArgument`] for `p` outside
/// `[0, 1]` or a non-positive shape.
pub fn beta_ppf(p: f64, a: f64, b: f64) -> Result<f64, EdgeErrors> {
    check_probability("p", p)?;
    check_positive("a", a)?;
    check_positive("b", b)?;
    Ok(inv_beta_reg_pair(a, b, p)?.0)
}

///////////
// Gamma //
///////////

/// Gamma CDF, shape-and-scale parameterisation.
///
/// Matches `scipy.stats.gamma.cdf(x, a=shape, scale=scale)`: the density is
/// `x^(shape-1) exp(-x/scale) / (Gamma(shape) scale^shape)`, so `scale` is the
/// reciprocal of the rate. `q2qnbinom` uses it for its gamma approximation.
///
/// ### Params
///
/// * `x` - Quantile. Non-positive values give 0.0.
/// * `shape` - Shape `a`, finite and strictly positive
/// * `scale` - Scale, finite and strictly positive
///
/// ### Returns
///
/// `P(X <= x)`, or [`EdgeErrors::InvalidArgument`] for a non-positive `shape`
/// or `scale`.
pub fn gamma_cdf(x: f64, shape: f64, scale: f64) -> Result<f64, EdgeErrors> {
    check_positive("shape", shape)?;
    check_positive("scale", scale)?;
    if x <= 0.0 {
        return Ok(0.0);
    }
    if x.is_infinite() {
        return Ok(1.0);
    }
    Ok(reg_gamma_lower(shape, x / scale))
}

/// Gamma quantile function, shape-and-scale parameterisation.
///
/// Hand-rolled. `statrs`'s `Gamma::inverse_cdf` is eight bisection steps plus
/// four Newton steps from a fixed bracket and lands around 1e-8 relative, which
/// is not enough for the quantile-to-quantile mapping in `q2qnbinom` where the
/// output feeds straight back into a pseudo-count. This solves
/// `gamma_lr(shape, x) = p` by Newton in `ln(x)`, bisecting whenever the Newton
/// step leaves the bracket, and inverts the upper tail instead for `p > 0.5` so
/// the residual never cancels.
///
/// ### Params
///
/// * `p` - Probability in `[0, 1]`
/// * `shape` - Shape `a`, finite and strictly positive
/// * `scale` - Scale, finite and strictly positive
///
/// ### Returns
///
/// `x` with `P(X <= x) = p`. `p = 0` gives 0.0 and `p = 1` gives `+inf`.
/// [`EdgeErrors::InvalidArgument`] for `p` outside `[0, 1]` or a non-positive
/// `shape` or `scale`, and [`EdgeErrors::NoConvergence`] if the bracket has not
/// closed within [`GAMMA_PPF_MAX_ITER`].
pub fn gamma_ppf(p: f64, shape: f64, scale: f64) -> Result<f64, EdgeErrors> {
    check_probability("p", p)?;
    check_positive("shape", shape)?;
    check_positive("scale", scale)?;
    if p == 0.0 {
        return Ok(0.0);
    }
    if p == 1.0 {
        return Ok(f64::INFINITY);
    }
    Ok(gamma_lr_inv(p, shape)? * scale)
}

/// Starting point for the gamma quantile, in `ln(x)` with unit rate.
///
/// Wilson-Hilferty where its cube root is positive, which covers everything
/// with `shape` above roughly 1. Below that the distribution piles up at zero
/// and Wilson-Hilferty goes negative, so the small-`x` series
/// `P(a, x) ~ x^a / (a Gamma(a))` is inverted instead.
///
/// The result only has to land in the right decade: the caller brackets it and
/// bisects, so a poor guess costs iterations, never correctness.
///
/// ### Params
///
/// * `p` - Target lower-tail probability, strictly inside `(0, 1)`
/// * `shape` - Shape, strictly positive
///
/// ### Returns
///
/// An initial `ln(x)`.
///
/// ### References
///
/// Wilson & Hilferty, PNAS, 1931
fn gamma_ppf_start(p: f64, shape: f64) -> f64 {
    let z = -SQRT_2 * erfc_inv(2.0 * p);
    let base = 1.0 - 1.0 / (9.0 * shape) + z / (3.0 * shape.sqrt());
    if base > 0.0 {
        let x = shape * base * base * base;
        if x > 0.0 && x.is_finite() {
            return x.ln();
        }
    }
    (p.ln() + ln_gamma(shape) + shape.ln()) / shape
}

/// Inverts the regularised lower incomplete gamma at unit rate.
///
/// Newton in `t = ln(x)` on a bracketed, monotonically increasing residual. In
/// log-space the derivative is `x f(x) = exp(shape ln x - x - ln_gamma(shape))`,
/// which stays representable for the tiny quantiles small shapes produce, where
/// the density itself would overflow.
///
/// For `p > 0.5` the residual is built from `gamma_ur` and the complement
/// `1 - p`, which is exact by Sterbenz, so no accuracy is lost switching tails.
///
/// ### Params
///
/// * `p` - Target lower-tail probability, strictly inside `(0, 1)`
/// * `shape` - Shape, strictly positive
///
/// ### Returns
///
/// `x` with `gamma_lr(shape, x) = p`, or [`EdgeErrors::NoConvergence`].
fn gamma_lr_inv(p: f64, shape: f64) -> Result<f64, EdgeErrors> {
    let upper = p > 0.5;
    let target = if upper { 1.0 - p } else { p };
    let ln_norm = ln_gamma(shape);

    // Increasing in t on both branches: P rises, and target - Q rises too
    // because Q falls.
    let residual = |t: f64| {
        let x = t.exp();
        if upper {
            target - reg_gamma_upper(shape, x)
        } else {
            reg_gamma_lower(shape, x) - target
        }
    };
    // d(residual)/dt = x * pdf(x), the same on both branches.
    let slope = |t: f64| (shape * t - t.exp() - ln_norm).exp();

    let mut t = gamma_ppf_start(p, shape);
    let mut lo = t;
    let mut hi = t;
    if residual(t) < 0.0 {
        // Walk the upper end out until it overshoots.
        for _ in 0..GAMMA_PPF_MAX_ITER {
            hi += 1.0;
            if residual(hi) >= 0.0 {
                break;
            }
            lo = hi;
        }
    } else {
        for _ in 0..GAMMA_PPF_MAX_ITER {
            lo -= 1.0;
            if residual(lo) < 0.0 {
                break;
            }
            hi = lo;
        }
    }

    for _ in 0..GAMMA_PPF_MAX_ITER {
        if !(t > lo && t < hi) {
            t = 0.5 * (lo + hi);
        }
        let f = residual(t);
        if f < 0.0 {
            lo = t;
        } else {
            hi = t;
        }
        let tol = GAMMA_PPF_REL_TOL * (1.0 + t.abs());
        if hi - lo <= tol {
            return Ok(t.exp());
        }
        let d = slope(t);
        let t_next = if d > 0.0 && d.is_finite() {
            t - f / d
        } else {
            0.5 * (lo + hi)
        };
        let step = (t_next - t).abs();
        t = t_next;
        if step <= tol {
            // One more residual evaluation would only re-tighten the bracket.
            return Ok(t.clamp(lo, hi).exp());
        }
    }

    Err(EdgeErrors::NoConvergence {
        routine: "gamma_ppf",
        iterations: GAMMA_PPF_MAX_ITER,
        last_delta: hi - lo,
    })
}

///////////////////////
// Negative binomial //
///////////////////////

/// Negative binomial log-PMF, "number of successes" parameterisation.
///
/// Matches `scipy.stats.nbinom.logpmf(k, size, prob)` and the `_nb_logpmf`
/// helper in `edgepython/exact_test.py`: `prob` is the success probability and
/// `size` the target number of successes, so `k` counts *failures* before the
/// `size`-th success and the mean is `size (1 - prob) / prob`. Every call site
/// in edgePython reaches it as `prob = size / (size + mu)`, which is the same
/// convention read backwards from a mean and a dispersion.
///
/// This is the parameterisation to get wrong. The other common one puts `prob`
/// on the failures and gives a mean of `size prob / (1 - prob)`, which is a
/// different distribution for the same numbers.
///
/// ```text
/// ln P(K = k) = lgamma(k + size) - lgamma(k + 1) - lgamma(size)
///               + size ln(prob) + k ln(1 - prob)
/// ```
///
/// The last term is evaluated as `k * ln_1p(-prob)`, which is what `scipy`
/// does and is more accurate than `ln(1 - prob)` for a small `prob`.
///
/// ### Params
///
/// * `k` - Number of failures. Non-integer values are evaluated on the
///   continuous extension, as `_nb_logpmf` does; negative values give `-inf`.
/// * `size` - Number of successes, finite and strictly positive. Need not be an
///   integer.
/// * `prob` - Success probability in `[0, 1]`
///
/// ### Returns
///
/// The log probability mass, or [`EdgeErrors::InvalidArgument`] for a
/// non-positive `size` or a `prob` outside `[0, 1]`.
pub fn nbinom_ln_pmf(k: f64, size: f64, prob: f64) -> Result<f64, EdgeErrors> {
    check_positive("size", size)?;
    check_probability("prob", prob)?;
    if k < 0.0 {
        return Ok(f64::NEG_INFINITY);
    }
    // Degenerate at k = 0; the general formula would give 0 * -inf = NaN.
    if prob == 1.0 {
        return Ok(if k == 0.0 { 0.0 } else { f64::NEG_INFINITY });
    }
    if prob == 0.0 {
        return Ok(f64::NEG_INFINITY);
    }
    Ok(ln_gamma(k + size) - ln_gamma(k + 1.0) - ln_gamma(size)
        + size * prob.ln()
        + k * (-prob).ln_1p())
}

/// Negative binomial CDF, "number of successes" parameterisation.
///
/// The same convention as [`nbinom_ln_pmf`]. `scipy` evaluates this as the
/// regularised incomplete beta `I(prob; size, floor(k) + 1)` rather than
/// summing the PMF, and so does this.
///
/// ### Params
///
/// * `k` - Number of failures. Floored, as `scipy` floors a discrete quantile.
/// * `size` - Number of successes, finite and strictly positive
/// * `prob` - Success probability in `[0, 1]`
///
/// ### Returns
///
/// `P(K <= k)`, or [`EdgeErrors::InvalidArgument`] for a non-positive `size` or
/// a `prob` outside `[0, 1]`.
pub fn nbinom_cdf(k: f64, size: f64, prob: f64) -> Result<f64, EdgeErrors> {
    check_positive("size", size)?;
    check_probability("prob", prob)?;
    if k < 0.0 {
        return Ok(0.0);
    }
    if k.is_infinite() {
        return Ok(1.0);
    }
    Ok(beta_reg(size, k.floor() + 1.0, prob))
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Target relative accuracy against scipy.
    ///
    /// A sweep of roughly 1500 points across every function here holds to this
    /// or better, apart from the mid-tail incomplete beta noted in
    /// [`BETA_REG_TOL`].
    const TOL: f64 = 1e-12;

    /// Relaxed tolerance where `statrs`'s `beta_reg` is the limiting factor.
    ///
    /// For arguments past its internal symmetry swap it returns `1 - I(1-x)`,
    /// and the subtraction costs about a decimal digit when the tail being
    /// asked for is a tenth or so of the mass. That is the whole of the gap
    /// between this and [`TOL`], and it only bites in the mid-tail of t and F
    /// at large `df`, nowhere near where a p-value is read.
    const BETA_REG_TOL: f64 = 1e-11;

    ////////////
    // Normal //
    ////////////

    #[test]
    fn test_norm_cdf_matches_scipy() {
        // scipy.stats.norm.cdf(-3.0)
        assert_relative_eq!(norm_cdf(-3.0), 0.001349898031630093, max_relative = TOL);
        // scipy.stats.norm.cdf(0.0)
        assert_relative_eq!(norm_cdf(0.0), 0.5, max_relative = TOL);
        // scipy.stats.norm.cdf(1.959963984540054)
        assert_relative_eq!(norm_cdf(1.959963984540054), 0.975, max_relative = TOL);
        // scipy.stats.norm.cdf(8.0)
        assert_relative_eq!(norm_cdf(8.0), 0.9999999999999993, max_relative = TOL);
    }

    #[test]
    fn test_norm_sf_holds_the_far_tail() {
        // scipy.stats.norm.sf(1.959963984540054)
        assert_relative_eq!(
            norm_sf(1.959963984540054),
            0.024999999999999998,
            max_relative = TOL
        );
        // scipy.stats.norm.sf(5.0)
        assert_relative_eq!(norm_sf(5.0), 2.8665157187919344e-07, max_relative = TOL);
        // scipy.stats.norm.sf(8.0)
        assert_relative_eq!(norm_sf(8.0), 6.22096057427174e-16, max_relative = TOL);
        // scipy.stats.norm.sf(37.0). 1 - cdf would be exactly 0.0 here.
        assert_relative_eq!(norm_sf(37.0), 5.725571222523923e-300, max_relative = TOL);
        assert!(1.0 - norm_cdf(37.0) == 0.0);
    }

    #[test]
    fn test_norm_ppf_matches_scipy() {
        // scipy.stats.norm.ppf(0.025)
        assert_relative_eq!(
            norm_ppf(0.025).unwrap(),
            -1.9599639845400545,
            max_relative = TOL
        );
        // scipy.stats.norm.ppf(0.5)
        assert_eq!(norm_ppf(0.5).unwrap(), 0.0);
        // scipy.stats.norm.ppf(0.975)
        assert_relative_eq!(
            norm_ppf(0.975).unwrap(),
            1.959963984540054,
            max_relative = TOL
        );
        // scipy.stats.norm.ppf(1e-10)
        assert_relative_eq!(
            norm_ppf(1e-10).unwrap(),
            -6.361340902404056,
            max_relative = TOL
        );
        // scipy.stats.norm.ppf(1e-300)
        assert_relative_eq!(
            norm_ppf(1e-300).unwrap(),
            -37.0470962993612,
            max_relative = TOL
        );
        assert_eq!(norm_ppf(0.0).unwrap(), f64::NEG_INFINITY);
        assert_eq!(norm_ppf(1.0).unwrap(), f64::INFINITY);
    }

    #[test]
    fn test_norm_sf_ppf_round_trip() {
        // Only the lower tail: ppf(cdf(x)) for x well above zero is genuinely
        // ill-conditioned, since p there carries no information below 1e-16.
        for &x in &[-8.0_f64, -4.0, -0.5, 0.0] {
            let p = norm_cdf(x);
            assert_relative_eq!(
                norm_ppf(p).unwrap(),
                x,
                max_relative = 1e-10,
                epsilon = 1e-12
            );
            // sf is the mirror of cdf, to the last few ulp.
            assert_relative_eq!(norm_sf(-x), p, max_relative = 1e-14);
        }
    }

    #[test]
    fn test_norm_sf_beats_scipy_into_the_subnormals() {
        // scipy.stats.norm.sf(37.7) is 0.0: cephes gives up before the
        // subnormals. mpmath: erfc(37.7 / sqrt(2)) / 2 = 2.4834853102778557e-311.
        assert_relative_eq!(
            norm_sf(37.7),
            2.483_485_310_277_6e-311,
            max_relative = 1e-11
        );
        // mpmath: erfc(38.15 / sqrt(2)) / 2 = 9.5090546401577420e-319.
        assert_relative_eq!(norm_cdf(-38.15), 9.509_03e-319, max_relative = 1e-4);
    }

    #[test]
    fn test_norm_ppf_rejects_out_of_range() {
        assert!(matches!(
            norm_ppf(-0.1),
            Err(EdgeErrors::InvalidArgument(_))
        ));
        assert!(matches!(norm_ppf(1.5), Err(EdgeErrors::InvalidArgument(_))));
        assert!(matches!(
            norm_ppf(f64::NAN),
            Err(EdgeErrors::InvalidArgument(_))
        ));
    }

    /////////////////
    // Chi-squared //
    /////////////////

    #[test]
    fn test_chisq_sf_matches_scipy() {
        // scipy.stats.chi2.sf(6.0485, 1)
        assert_relative_eq!(
            chisq_sf(6.0485, 1.0).unwrap(),
            0.013918115674714049,
            max_relative = TOL
        );
        // scipy.stats.chi2.sf(3.841458820694124, 1)
        assert_relative_eq!(
            chisq_sf(3.841458820694124, 1.0).unwrap(),
            0.04999999999999994,
            max_relative = TOL
        );
        // scipy.stats.chi2.sf(100.0, 10)
        assert_relative_eq!(
            chisq_sf(100.0, 10.0).unwrap(),
            5.4497019829205215e-17,
            max_relative = TOL
        );
        // scipy.stats.chi2.sf(0.5, 2.5), a non-integer df
        assert_relative_eq!(
            chisq_sf(0.5, 2.5).unwrap(),
            0.8638836284585889,
            max_relative = TOL
        );
    }

    #[test]
    fn test_chisq_sf_far_upper_tail() {
        // scipy.stats.chi2.sf(200.0, 1). A `1 - cdf` implementation returns 0.0.
        assert_relative_eq!(
            chisq_sf(200.0, 1.0).unwrap(),
            2.0884875837625688e-45,
            max_relative = TOL
        );
        // scipy.stats.chi2.sf(1000.0, 1)
        assert_relative_eq!(
            chisq_sf(1000.0, 1.0).unwrap(),
            1.7958327848007363e-219,
            max_relative = TOL
        );
    }

    #[test]
    fn test_chisq_sf_edges_and_errors() {
        assert_eq!(chisq_sf(-1.0, 1.0).unwrap(), 1.0);
        assert_eq!(chisq_sf(0.0, 3.0).unwrap(), 1.0);
        assert_eq!(chisq_sf(f64::INFINITY, 3.0).unwrap(), 0.0);
        assert!(matches!(
            chisq_sf(1.0, 0.0),
            Err(EdgeErrors::InvalidArgument(_))
        ));
        assert!(matches!(
            chisq_sf(1.0, -2.0),
            Err(EdgeErrors::InvalidArgument(_))
        ));
    }

    /////////////////
    // Student's t //
    /////////////////

    #[test]
    fn test_t_cdf_matches_scipy() {
        // scipy.stats.t.cdf(-2.5, 10)
        assert_relative_eq!(
            t_cdf(-2.5, 10.0).unwrap(),
            0.01572342211830441,
            max_relative = TOL
        );
        // scipy.stats.t.cdf(0.0, 5)
        assert_relative_eq!(t_cdf(0.0, 5.0).unwrap(), 0.5, max_relative = TOL);
        // scipy.stats.t.cdf(1.5, 3.7), a non-integer df
        assert_relative_eq!(
            t_cdf(1.5, 3.7).unwrap(),
            0.8932009153989933,
            max_relative = TOL
        );
        // scipy.stats.t.cdf(2.5, 10)
        assert_relative_eq!(
            t_cdf(2.5, 10.0).unwrap(),
            0.9842765778816955,
            max_relative = TOL
        );
    }

    #[test]
    fn test_t_sf_holds_the_far_tail() {
        // scipy.stats.t.sf(2.5, 10)
        assert_relative_eq!(
            t_sf(2.5, 10.0).unwrap(),
            0.01572342211830441,
            max_relative = TOL
        );
        // scipy.stats.t.sf(20.0, 5)
        assert_relative_eq!(
            t_sf(20.0, 5.0).unwrap(),
            2.8877581866120858e-06,
            max_relative = TOL
        );
        // scipy.stats.t.sf(50.0, 100). Well past where `1 - cdf` is all zeroes.
        assert_relative_eq!(
            t_sf(50.0, 100.0).unwrap(),
            7.236081839880731e-73,
            max_relative = TOL
        );
        // scipy.stats.t.sf(-2.5, 10)
        assert_relative_eq!(
            t_sf(-2.5, 10.0).unwrap(),
            0.9842765778816955,
            max_relative = TOL
        );
        assert!(1.0 - t_cdf(50.0, 100.0).unwrap() == 0.0);
    }

    #[test]
    fn test_t_tails_near_the_median() {
        // The naive h = df / (df + x^2) form loses six digits to cancellation
        // here, because 1 - h is 1e-10 formed from numbers of order one.
        // scipy.stats.t.cdf(0.0001, 100)
        assert_relative_eq!(
            t_cdf(1e-4, 100.0).unwrap(),
            0.5000397946186266,
            max_relative = TOL
        );
        // scipy.stats.t.sf(0.0001, 100)
        assert_relative_eq!(
            t_sf(1e-4, 100.0).unwrap(),
            0.4999602053813734,
            max_relative = TOL
        );
        // scipy.stats.t.cdf(0.001, 3)
        assert_relative_eq!(
            t_cdf(1e-3, 3.0).unwrap(),
            0.5003675525152695,
            max_relative = TOL
        );
        // scipy.stats.t.sf(0.5, 12). Past the inner/outer crossover, where
        // statrs's beta_reg is the limiting factor rather than the formulation.
        assert_relative_eq!(
            t_sf(0.5, 12.0).unwrap(),
            0.31305873811266205,
            max_relative = BETA_REG_TOL
        );
    }

    #[test]
    fn test_t_ppf_matches_scipy() {
        // scipy.stats.t.ppf(0.975, 10)
        assert_relative_eq!(
            t_ppf(0.975, 10.0).unwrap(),
            2.228138851986274,
            max_relative = TOL
        );
        // scipy.stats.t.ppf(0.025, 3)
        assert_relative_eq!(
            t_ppf(0.025, 3.0).unwrap(),
            -3.1824463052837086,
            max_relative = TOL
        );
        // scipy.stats.t.ppf(1e-08, 20)
        assert_relative_eq!(
            t_ppf(1e-8, 20.0).unwrap(),
            -8.942883259935037,
            max_relative = TOL
        );
        // scipy.stats.t.ppf(0.5, 7)
        assert_relative_eq!(t_ppf(0.5, 7.0).unwrap(), 0.0, epsilon = 1e-14);
        assert_eq!(t_ppf(0.0, 7.0).unwrap(), f64::NEG_INFINITY);
        assert_eq!(t_ppf(1.0, 7.0).unwrap(), f64::INFINITY);
    }

    #[test]
    fn test_t_ppf_far_tail_beats_as109() {
        // statrs's inv_beta_reg alone returns -9.2e7 for the first of these,
        // seven orders out, because AS 109 floors at 1e-30.
        // scipy.stats.t.ppf(1e-15, 1)
        assert_relative_eq!(
            t_ppf(1e-15, 1.0).unwrap(),
            -318309886183790.7,
            max_relative = TOL
        );
        // scipy.stats.t.ppf(1e-15, 3)
        assert_relative_eq!(
            t_ppf(1e-15, 3.0).unwrap(),
            -103311.08359284987,
            max_relative = TOL
        );
        // scipy.stats.t.ppf(1e-30, 5)
        assert_relative_eq!(
            t_ppf(1e-30, 5.0).unwrap(),
            -1568392.5590979713,
            max_relative = TOL
        );
    }

    #[test]
    fn test_t_ppf_sf_round_trip() {
        for &(p, df) in &[(1e-6, 4.0_f64), (0.01, 12.0), (0.3, 30.0), (1e-12, 8.0)] {
            let x = t_ppf(p, df).unwrap();
            assert_relative_eq!(t_cdf(x, df).unwrap(), p, max_relative = 1e-11);
            let x_up = t_ppf(1.0 - p, df).unwrap();
            assert_relative_eq!(t_sf(x_up, df).unwrap(), p, max_relative = 1e-11);
        }
    }

    #[test]
    fn test_t_rejects_bad_arguments() {
        assert!(matches!(
            t_cdf(0.0, 0.0),
            Err(EdgeErrors::InvalidArgument(_))
        ));
        assert!(matches!(
            t_sf(0.0, -1.0),
            Err(EdgeErrors::InvalidArgument(_))
        ));
        assert!(matches!(
            t_ppf(0.5, f64::INFINITY),
            Err(EdgeErrors::InvalidArgument(_))
        ));
        assert!(matches!(
            t_ppf(1.2, 5.0),
            Err(EdgeErrors::InvalidArgument(_))
        ));
    }

    ///////
    // F //
    ///////

    #[test]
    fn test_f_sf_matches_scipy() {
        // scipy.stats.f.sf(4.964602743243619, 1, 10)
        assert_relative_eq!(
            f_sf(4.964602743243619, 1.0, 10.0).unwrap(),
            0.050000000009265785,
            max_relative = TOL
        );
        // scipy.stats.f.sf(100.0, 3, 7)
        assert_relative_eq!(
            f_sf(100.0, 3.0, 7.0).unwrap(),
            4.130422101383355e-06,
            max_relative = TOL
        );
        // scipy.stats.f.sf(1.0, 5, 5)
        assert_relative_eq!(
            f_sf(1.0, 5.0, 5.0).unwrap(),
            0.5000000000000002,
            max_relative = TOL
        );
    }

    #[test]
    fn test_f_sf_far_upper_tail() {
        // scipy.stats.f.sf(1e6, 2, 4)
        assert_relative_eq!(
            f_sf(1e6, 2.0, 4.0).unwrap(),
            3.999984000048e-12,
            max_relative = TOL
        );
        // scipy.stats.f.sf(500.0, 1, 200). This is the case the `1 - d1x/(d1x+d2)`
        // form in statrs's own FisherSnedecor::sf cannot reach.
        assert_relative_eq!(
            f_sf(500.0, 1.0, 200.0).unwrap(),
            2.607870124613832e-56,
            max_relative = TOL
        );
    }

    #[test]
    fn test_f_ppf_matches_scipy() {
        // scipy.stats.f.ppf(0.95, 1, 10)
        assert_relative_eq!(
            f_ppf(0.95, 1.0, 10.0).unwrap(),
            4.964602743730711,
            max_relative = TOL
        );
        // scipy.stats.f.ppf(0.5, 3, 7)
        assert_relative_eq!(
            f_ppf(0.5, 3.0, 7.0).unwrap(),
            0.870944253187285,
            max_relative = TOL
        );
        // scipy.stats.f.ppf(0.99, 4, 20)
        assert_relative_eq!(
            f_ppf(0.99, 4.0, 20.0).unwrap(),
            4.430690161437775,
            max_relative = TOL
        );
        // scipy.stats.f.ppf(0.5, 1, 4)
        assert_relative_eq!(
            f_ppf(0.5, 1.0, 4.0).unwrap(),
            0.5486321704130301,
            max_relative = TOL
        );
        assert_eq!(f_ppf(0.0, 1.0, 4.0).unwrap(), 0.0);
        assert_eq!(f_ppf(1.0, 1.0, 4.0).unwrap(), f64::INFINITY);
    }

    #[test]
    fn test_f_ppf_far_tail_beats_as109() {
        // scipy.stats.f.ppf(1e-10, 1, 1). AS 109 alone gives 1.2e-16.
        assert_relative_eq!(
            f_ppf(1e-10, 1.0, 1.0).unwrap(),
            2.4674011002723397e-20,
            max_relative = TOL
        );
        // scipy.stats.f.ppf(1e-6, 2, 7)
        assert_relative_eq!(
            f_ppf(1e-6, 2.0, 7.0).unwrap(),
            1.0000006428576329e-06,
            max_relative = TOL
        );
        // scipy.stats.f.ppf(1 - 1e-9, 3, 5)
        assert_relative_eq!(
            f_ppf(1.0 - 1e-9, 3.0, 5.0).unwrap(),
            8817.937012202383,
            max_relative = TOL
        );
    }

    #[test]
    fn test_f_ppf_sf_round_trip() {
        for &(p, d1, d2) in &[
            (0.05, 1.0_f64, 10.0),
            (0.001, 4.0, 20.0),
            (1e-8, 2.0, 6.0),
            (0.4, 7.0, 7.0),
        ] {
            let x = f_ppf(1.0 - p, d1, d2).unwrap();
            assert_relative_eq!(f_sf(x, d1, d2).unwrap(), p, max_relative = 1e-9);
        }
    }

    #[test]
    fn test_f_rejects_bad_arguments() {
        assert!(matches!(
            f_sf(1.0, 0.0, 5.0),
            Err(EdgeErrors::InvalidArgument(_))
        ));
        assert!(matches!(
            f_sf(1.0, 5.0, -1.0),
            Err(EdgeErrors::InvalidArgument(_))
        ));
        assert!(matches!(
            f_ppf(-0.5, 5.0, 5.0),
            Err(EdgeErrors::InvalidArgument(_))
        ));
        assert!(matches!(
            f_ppf(0.5, f64::NAN, 5.0),
            Err(EdgeErrors::InvalidArgument(_))
        ));
    }

    //////////
    // Beta //
    //////////

    #[test]
    fn test_beta_cdf_matches_scipy() {
        // scipy.stats.beta.cdf(0.3, 2, 2)
        assert_relative_eq!(
            beta_cdf(0.3, 2.0, 2.0).unwrap(),
            0.21599999999999994,
            max_relative = TOL
        );
        // scipy.stats.beta.cdf(0.5, 0.5, 0.5)
        assert_relative_eq!(
            beta_cdf(0.5, 0.5, 0.5).unwrap(),
            0.5000000000000001,
            max_relative = TOL
        );
        // scipy.stats.beta.cdf(0.9, 5, 1)
        assert_relative_eq!(
            beta_cdf(0.9, 5.0, 1.0).unwrap(),
            0.5904900000000001,
            max_relative = TOL
        );
        // scipy.stats.beta.cdf(0.25, 2, 2)
        assert_relative_eq!(
            beta_cdf(0.25, 2.0, 2.0).unwrap(),
            0.15625,
            max_relative = TOL
        );
        assert_eq!(beta_cdf(-1.0, 2.0, 2.0).unwrap(), 0.0);
        assert_eq!(beta_cdf(2.0, 2.0, 2.0).unwrap(), 1.0);
    }

    #[test]
    fn test_beta_sf_holds_the_far_tail() {
        // scipy.stats.beta.sf(0.99, 2, 3)
        assert_relative_eq!(
            beta_sf(0.99, 2.0, 3.0).unwrap(),
            3.97000000000001e-06,
            max_relative = TOL
        );
        // scipy.stats.beta.sf(0.999999, 2, 3). `1 - cdf` gives 0.0 here.
        assert_relative_eq!(
            beta_sf(0.999999, 2.0, 3.0).unwrap(),
            3.9999970003450675e-18,
            max_relative = TOL
        );
        assert!(1.0 - beta_cdf(0.999999, 2.0, 3.0).unwrap() == 0.0);
        // scipy.stats.beta.sf(0.5, 0.5, 0.5)
        assert_relative_eq!(
            beta_sf(0.5, 0.5, 0.5).unwrap(),
            0.5000000000000001,
            max_relative = TOL
        );
        // scipy.stats.beta.sf(1e-06, 3, 2)
        assert_relative_eq!(beta_sf(1e-6, 3.0, 2.0).unwrap(), 1.0, max_relative = TOL);
        assert_eq!(beta_sf(-1.0, 2.0, 2.0).unwrap(), 1.0);
        assert_eq!(beta_sf(2.0, 2.0, 2.0).unwrap(), 0.0);
    }

    #[test]
    fn test_beta_ppf_matches_scipy() {
        // scipy.stats.beta.ppf(0.5, 2, 2)
        assert_relative_eq!(beta_ppf(0.5, 2.0, 2.0).unwrap(), 0.5, max_relative = TOL);
        // scipy.stats.beta.ppf(0.975, 3, 5)
        assert_relative_eq!(
            beta_ppf(0.975, 3.0, 5.0).unwrap(),
            0.7095791362626572,
            max_relative = TOL
        );
        // scipy.stats.beta.ppf(1e-08, 2, 3)
        assert_relative_eq!(
            beta_ppf(1e-8, 2.0, 3.0).unwrap(),
            4.082594021609242e-05,
            max_relative = TOL
        );
    }

    #[test]
    fn test_beta_ppf_far_tail_beats_as109() {
        // scipy.stats.beta.ppf(1e-12, 0.3, 0.7). AS 109 alone returns
        // 1.4e-16, twenty-four orders out, because it floors at 1e-30.
        assert_relative_eq!(
            beta_ppf(1e-12, 0.3, 0.7).unwrap(),
            1.6635847967496953e-40,
            max_relative = TOL
        );
        // scipy.stats.beta.ppf(1e-20, 2.0, 0.5)
        assert_relative_eq!(
            beta_ppf(1e-20, 2.0, 0.5).unwrap(),
            1.6329931618110077e-10,
            max_relative = TOL
        );
    }

    #[test]
    fn test_beta_ppf_cdf_round_trip() {
        for &(p, a, b) in &[(0.1_f64, 2.0_f64, 3.0), (0.5, 0.7, 4.0), (0.999, 5.0, 5.0)] {
            let x = beta_ppf(p, a, b).unwrap();
            assert_relative_eq!(beta_cdf(x, a, b).unwrap(), p, max_relative = 1e-10);
        }
    }

    #[test]
    fn test_solvers_converge_across_the_grid() {
        // The two hand-rolled inverses both report NoConvergence rather than
        // returning a wrong answer, so a sweep that never errs is the check
        // that the bracketing holds over shapes spanning five decades.
        for &shape in &[1e-2_f64, 0.5, 1.0, 13.0, 1e3] {
            for &b in &[1e-2_f64, 0.5, 1.0, 13.0, 1e3] {
                for &p in &[1e-30_f64, 1e-8, 0.1, 0.5, 0.9, 1.0 - 1e-8] {
                    assert!(beta_ppf(p, shape, b).is_ok(), "beta {p} {shape} {b}");
                    assert!(f_ppf(p, shape, b).is_ok(), "f {p} {shape} {b}");
                }
            }
            for &p in &[1e-30_f64, 1e-8, 0.1, 0.5, 0.9, 1.0 - 1e-8] {
                assert!(gamma_ppf(p, shape, 1.0).is_ok(), "gamma {p} {shape}");
                assert!(t_ppf(p, shape).is_ok(), "t {p} {shape}");
            }
        }
    }

    #[test]
    fn test_beta_rejects_bad_arguments() {
        assert!(matches!(
            beta_cdf(0.5, 0.0, 2.0),
            Err(EdgeErrors::InvalidArgument(_))
        ));
        assert!(matches!(
            beta_sf(0.5, 2.0, -1.0),
            Err(EdgeErrors::InvalidArgument(_))
        ));
        assert!(matches!(
            beta_ppf(1.5, 2.0, 2.0),
            Err(EdgeErrors::InvalidArgument(_))
        ));
        assert!(matches!(
            beta_ppf(0.5, f64::INFINITY, 2.0),
            Err(EdgeErrors::InvalidArgument(_))
        ));
    }

    ///////////
    // Gamma //
    ///////////

    #[test]
    fn test_gamma_cdf_matches_scipy() {
        // scipy.stats.gamma.cdf(2.0, 3.0, scale=1.0)
        assert_relative_eq!(
            gamma_cdf(2.0, 3.0, 1.0).unwrap(),
            0.3233235838169364,
            max_relative = TOL
        );
        // scipy.stats.gamma.cdf(0.5, 0.1, scale=2.0)
        assert_relative_eq!(
            gamma_cdf(0.5, 0.1, 2.0).unwrap(),
            0.8955592438575553,
            max_relative = TOL
        );
        // scipy.stats.gamma.cdf(10.0, 2.0, scale=1.5)
        assert_relative_eq!(
            gamma_cdf(10.0, 2.0, 1.5).unwrap(),
            0.9902431408563948,
            max_relative = TOL
        );
        // scipy.stats.gamma.cdf(0.001, 4.0, scale=1.0)
        assert_relative_eq!(
            gamma_cdf(1e-3, 4.0, 1.0).unwrap(),
            4.163334721825485e-14,
            max_relative = TOL
        );
        assert_eq!(gamma_cdf(-1.0, 2.0, 1.0).unwrap(), 0.0);
        assert_eq!(gamma_cdf(f64::INFINITY, 2.0, 1.0).unwrap(), 1.0);
    }

    #[test]
    fn test_gamma_ppf_matches_scipy() {
        // scipy.stats.gamma.ppf(0.5, 3.0, scale=1.0)
        assert_relative_eq!(
            gamma_ppf(0.5, 3.0, 1.0).unwrap(),
            2.6740603137235617,
            max_relative = TOL
        );
        // scipy.stats.gamma.ppf(0.975, 2.5, scale=1.5)
        assert_relative_eq!(
            gamma_ppf(0.975, 2.5, 1.5).unwrap(),
            9.624376495522519,
            max_relative = TOL
        );
        // scipy.stats.gamma.ppf(1e-06, 0.5, scale=2.0)
        assert_relative_eq!(
            gamma_ppf(1e-6, 0.5, 2.0).unwrap(),
            1.5707963267957187e-12,
            max_relative = TOL
        );
        // scipy.stats.gamma.ppf(0.999999, 10.0, scale=0.5)
        assert_relative_eq!(
            gamma_ppf(0.999999, 10.0, 0.5).unwrap(),
            16.355170258742405,
            max_relative = TOL
        );
        // scipy.stats.gamma.ppf(0.3, 0.05, scale=3.0), a shape well below one
        assert_relative_eq!(
            gamma_ppf(0.3, 0.05, 3.0).unwrap(),
            6.11369156619802e-11,
            max_relative = TOL
        );
        assert_eq!(gamma_ppf(0.0, 2.0, 1.0).unwrap(), 0.0);
        assert_eq!(gamma_ppf(1.0, 2.0, 1.0).unwrap(), f64::INFINITY);
    }

    #[test]
    fn test_gamma_ppf_cdf_round_trip() {
        for &(p, shape, scale) in &[
            (1e-9_f64, 0.3_f64, 2.0_f64),
            (0.2, 1.0, 1.0),
            (0.5, 7.5, 0.25),
            (0.99, 50.0, 3.0),
            (0.999999999, 2.0, 1.0),
        ] {
            let x = gamma_ppf(p, shape, scale).unwrap();
            assert_relative_eq!(gamma_cdf(x, shape, scale).unwrap(), p, max_relative = 1e-11);
        }
    }

    #[test]
    fn test_gamma_rejects_bad_arguments() {
        assert!(matches!(
            gamma_cdf(1.0, 0.0, 1.0),
            Err(EdgeErrors::InvalidArgument(_))
        ));
        assert!(matches!(
            gamma_cdf(1.0, 1.0, -2.0),
            Err(EdgeErrors::InvalidArgument(_))
        ));
        assert!(matches!(
            gamma_ppf(1.5, 1.0, 1.0),
            Err(EdgeErrors::InvalidArgument(_))
        ));
        assert!(matches!(
            gamma_ppf(0.5, -1.0, 1.0),
            Err(EdgeErrors::InvalidArgument(_))
        ));
        assert!(matches!(
            gamma_ppf(0.5, 1.0, 0.0),
            Err(EdgeErrors::InvalidArgument(_))
        ));
    }

    ///////////////////////
    // Negative binomial //
    ///////////////////////

    #[test]
    fn test_nbinom_ln_pmf_matches_scipy_size_5_prob_03() {
        // scipy.stats.nbinom.logpmf(0, 5.0, 0.3)
        assert_relative_eq!(
            nbinom_ln_pmf(0.0, 5.0, 0.3).unwrap(),
            -6.01986402162968,
            max_relative = TOL
        );
        // scipy.stats.nbinom.logpmf(1, 5.0, 0.3)
        assert_relative_eq!(
            nbinom_ln_pmf(1.0, 5.0, 0.3).unwrap(),
            -4.767101053134312,
            max_relative = TOL
        );
        // scipy.stats.nbinom.logpmf(10, 5.0, 0.3)
        assert_relative_eq!(
            nbinom_ln_pmf(10.0, 5.0, 0.3).unwrap(),
            -2.6778586817017827,
            max_relative = TOL
        );
        // scipy.stats.nbinom.logpmf(500, 5.0, 0.3)
        assert_relative_eq!(
            nbinom_ln_pmf(500.0, 5.0, 0.3).unwrap(),
            -162.65701716239616,
            max_relative = TOL
        );
    }

    #[test]
    fn test_nbinom_ln_pmf_matches_scipy_size_25_prob_07() {
        // scipy.stats.nbinom.logpmf(0, 2.5, 0.7)
        assert_relative_eq!(
            nbinom_ln_pmf(0.0, 2.5, 0.7).unwrap(),
            -0.8916873598468311,
            max_relative = TOL
        );
        // scipy.stats.nbinom.logpmf(1, 2.5, 0.7)
        assert_relative_eq!(
            nbinom_ln_pmf(1.0, 2.5, 0.7).unwrap(),
            -1.179369432298612,
            max_relative = TOL
        );
        // scipy.stats.nbinom.logpmf(10, 2.5, 0.7)
        assert_relative_eq!(
            nbinom_ln_pmf(10.0, 2.5, 0.7).unwrap(),
            -9.586163334718176,
            max_relative = TOL
        );
        // scipy.stats.nbinom.logpmf(500, 2.5, 0.7)
        assert_relative_eq!(
            nbinom_ln_pmf(500.0, 2.5, 0.7).unwrap(),
            -593.8371152363,
            max_relative = TOL
        );
    }

    #[test]
    fn test_nbinom_ln_pmf_uses_the_successes_convention() {
        // prob = size / (size + mu) is how every edgePython call site builds it,
        // so the mean must come back as mu. mu = 4, size = 3 -> prob = 3/7.
        let (size, mu) = (3.0_f64, 4.0_f64);
        let prob = size / (size + mu);
        let mean: f64 = (0..4000)
            .map(|k| {
                let k = k as f64;
                k * nbinom_ln_pmf(k, size, prob).unwrap().exp()
            })
            .sum();
        assert_relative_eq!(mean, mu, max_relative = 1e-9);
    }

    #[test]
    fn test_nbinom_ln_pmf_sums_to_one() {
        let total: f64 = (0..20000)
            .map(|k| nbinom_ln_pmf(k as f64, 2.5, 0.7).unwrap().exp())
            .sum();
        assert_relative_eq!(total, 1.0, max_relative = 1e-12);
    }

    #[test]
    fn test_nbinom_cdf_matches_scipy() {
        // scipy.stats.nbinom.cdf(0, 5.0, 0.3)
        assert_relative_eq!(
            nbinom_cdf(0.0, 5.0, 0.3).unwrap(),
            0.0024299999999999994,
            max_relative = TOL
        );
        // scipy.stats.nbinom.cdf(10, 5.0, 0.3)
        assert_relative_eq!(
            nbinom_cdf(10.0, 5.0, 0.3).unwrap(),
            0.48450894077315665,
            max_relative = TOL
        );
        // scipy.stats.nbinom.cdf(100, 5.0, 0.3)
        assert_relative_eq!(
            nbinom_cdf(100.0, 5.0, 0.3).unwrap(),
            0.9999999999903741,
            max_relative = TOL
        );
        // scipy.stats.nbinom.cdf(3, 2.5, 0.7)
        assert_relative_eq!(
            nbinom_cdf(3.0, 2.5, 0.7).unwrap(),
            0.9514994588636262,
            max_relative = TOL
        );
        // scipy.stats.nbinom.cdf(500, 2.5, 0.7)
        assert_relative_eq!(
            nbinom_cdf(500.0, 2.5, 0.7).unwrap(),
            1.0,
            max_relative = TOL
        );
        assert_eq!(nbinom_cdf(-1.0, 5.0, 0.3).unwrap(), 0.0);
    }

    #[test]
    fn test_nbinom_cdf_agrees_with_summed_pmf() {
        let (size, prob) = (4.0_f64, 0.35_f64);
        let summed: f64 = (0..=12)
            .map(|k| nbinom_ln_pmf(k as f64, size, prob).unwrap().exp())
            .sum();
        assert_relative_eq!(
            nbinom_cdf(12.0, size, prob).unwrap(),
            summed,
            max_relative = 1e-12
        );
    }

    #[test]
    fn test_nbinom_degenerate_probabilities() {
        assert_eq!(nbinom_ln_pmf(0.0, 3.0, 1.0).unwrap(), 0.0);
        assert_eq!(nbinom_ln_pmf(2.0, 3.0, 1.0).unwrap(), f64::NEG_INFINITY);
        assert_eq!(nbinom_ln_pmf(0.0, 3.0, 0.0).unwrap(), f64::NEG_INFINITY);
        assert_eq!(nbinom_ln_pmf(-1.0, 3.0, 0.5).unwrap(), f64::NEG_INFINITY);
    }

    #[test]
    fn test_nbinom_rejects_bad_arguments() {
        assert!(matches!(
            nbinom_ln_pmf(1.0, 0.0, 0.5),
            Err(EdgeErrors::InvalidArgument(_))
        ));
        assert!(matches!(
            nbinom_ln_pmf(1.0, 5.0, 1.5),
            Err(EdgeErrors::InvalidArgument(_))
        ));
        assert!(matches!(
            nbinom_cdf(1.0, -5.0, 0.5),
            Err(EdgeErrors::InvalidArgument(_))
        ));
        assert!(matches!(
            nbinom_cdf(1.0, 5.0, -0.1),
            Err(EdgeErrors::InvalidArgument(_))
        ));
    }
}
