//! Distribution tails: everything edgePython calls `scipy.stats` for.
//!
//! Normal, chi-squared, Student's t, F, beta, gamma and negative binomial, as
//! plain functions (edgeR only needs tail probabilities and quantiles).
//!
//! ### Survival functions, not `1 - cdf`
//!
//! Every edgeR p-value is an upper tail, and `1 - cdf` returns a flat `0.0`
//! below about 1e-16: `chisq_sf(200, 1)` is 2.09e-45. Each `_sf` evaluates the
//! upper tail directly: the upper incomplete gamma for normal and chi-squared,
//! the incomplete beta with swapped arguments for t, F and beta.

use statrs::function::beta::{beta_reg, inv_beta_reg, ln_beta};
use statrs::function::erf::erfc_inv;
use statrs::function::gamma::ln_gamma;

use crate::errors::EdgeErrors;

////////////
// Consts //
////////////

/// `sqrt(2)`, the scale between the standard normal and the error function.
const SQRT_2: f64 = std::f64::consts::SQRT_2;

/// Relative convergence tolerance for the incomplete gamma series and continued
/// fraction: one ulp.
const INC_GAMMA_EPS: f64 = 2.220446049250313e-16;

/// Iteration budget for the incomplete gamma recurrences.
///
/// A runaway guard: the series needs about `sqrt(a)` terms near the mean. On
/// exhaustion the partial sum is returned, not an error.
const INC_GAMMA_MAX_ITER: usize = 10_000;

/// Floor for the modified Lentz continued fraction, guarding a zero pivot.
///
/// ### References
///
/// Press et al., Numerical Recipes, 3rd ed., section 6.2
const LENTZ_TINY: f64 = 1e-300;

/// Iteration budget for the polished incomplete beta inverse.
///
/// A runaway guard: convergence from an AS 109 start takes two or three steps.
const INV_BETA_MAX_ITER: usize = 128;

/// Relative convergence tolerance for the incomplete beta inverse, in `ln(x)`.
const INV_BETA_REL_TOL: f64 = 1e-15;

/// Iteration budget for the hand-rolled gamma quantile.
///
/// A runaway guard against pathological shapes.
const GAMMA_PPF_MAX_ITER: usize = 128;

/// Crossover on `x^2` between the two incomplete beta forms of the t tails.
///
/// The masses inside `(0, |x|)` and beyond `|x|` are equal at the upper
/// quartile, which runs from 1.0 at `df = 1` down to 0.4549 as `df` grows.
/// Using the inner form below this and the outer form above evaluates the
/// smaller mass directly; the larger comes from `0.5 - .`. At 0.7 neither branch
/// is worse than about a factor of two from balanced for any `df`.
const T_INNER_OUTER_SWITCH: f64 = 0.7;

/// Relative convergence tolerance for the gamma quantile, in `ln(x)`.
///
/// About 4 ulp; tighter just cycles on the last bit of the incomplete gamma.
const GAMMA_PPF_REL_TOL: f64 = 1e-15;

////////////////
// Validation //
////////////////

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

//////////////////////////////////
// Regularised incomplete gamma //
//////////////////////////////////

/// The shared prefactor `exp(a ln x - x - ln Gamma(a))` of both branches.
///
/// Formed in logs so it underflows to zero in the far tail instead of
/// overflowing on `x^a`.
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
/// `P(0.3, 1e-30)` is 1e-9, not zero.
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
/// The continued fraction above `x = a + 1`, evaluated directly, so
/// `chisq_sf(1000, 1)` is 1.8e-219, not zero.
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
/// Evaluated by modified Lentz.
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

/////////////////////////
// Log incomplete beta //
/////////////////////////

/// Iteration budget for the incomplete beta continued fraction.
const BETA_CF_MAX_ITER: usize = 300;

/// Relative tolerance at which the continued fraction stops.
const BETA_CF_EPS: f64 = 3e-16;

/// Log of the smallest positive probability an `f64` can hold.
///
/// Below this, `exp(ln_p)` is zero and AS 109 cannot seed the search; the
/// small-`x` asymptote does instead.
const MIN_REPRESENTABLE_LN: f64 = -745.0;

/// Shape above which the beta log-normaliser goes through Stirling.
///
/// `ln_beta` forms `lnGamma(a) + lnGamma(b) - lnGamma(a + b)`, where the first
/// and last cancel for large `a`: at `a = 5e5` both are about 6.0e6 and differ
/// by 6.5, leaving barely nine digits. Above this a series is used instead.
/// Below it the plain form loses at most `lnGamma(30) * eps`, 7e-15.
const LN_BETA_STIRLING_MIN: f64 = 30.0;

/// Stirling's correction, `lnGamma(z) - [(z - 1/2) ln z - z + ln(2 pi) / 2]`.
///
/// Asymptotic series, used above [`LN_BETA_STIRLING_MIN`].
///
/// ### Params
///
/// * `z` - Argument, at least [`LN_BETA_STIRLING_MIN`]
///
/// ### Returns
///
/// The correction term.
fn stirlerr(z: f64) -> f64 {
    let z2 = z * z;
    (1.0 / 12.0 - (1.0 / 360.0 - (1.0 / 1260.0 - 1.0 / (1680.0 * z2)) / z2) / z2) / z
}

/// Log of the beta function, stable for a large shape.
///
/// Symmetric in its arguments. For a large shape, `lnGamma(a + b) - lnGamma(a)`
/// comes from Stirling's series, cancelling the leading terms analytically.
///
/// ### Params
///
/// * `a` - First shape, strictly positive
/// * `b` - Second shape, strictly positive
///
/// ### Returns
///
/// `ln B(a, b)`.
fn ln_beta_stable(a: f64, b: f64) -> f64 {
    let (big, small) = if a >= b { (a, b) } else { (b, a) };
    if big < LN_BETA_STIRLING_MIN {
        return ln_beta(a, b);
    }
    // lnGamma(big + small) - lnGamma(big), with the (big - 1/2) ln(big) terms
    // cancelled by hand.
    let ratio = small * big.ln() + (big + small - 0.5) * (small / big).ln_1p() - small
        + stirlerr(big + small)
        - stirlerr(big);
    ln_gamma(small) - ratio
}

/// Modified Lentz evaluation of the incomplete beta continued fraction.
///
/// Numerical Recipes' `betacf`, valid where `x < (a + 1) / (a + b + 2)`. The
/// value is `O(1)`; the dynamic range is in the caller's prefactor.
///
/// ### Params
///
/// * `a` - First shape, strictly positive
/// * `b` - Second shape, strictly positive
/// * `x` - Argument in `(0, 1)`, on the convergent side
///
/// ### Returns
///
/// The continued fraction value.
///
/// ### References
///
/// Press, Teukolsky, Vetterling & Flannery, Numerical Recipes, 3rd ed., 6.4
fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;

    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < LENTZ_TINY {
        d = LENTZ_TINY;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=BETA_CF_MAX_ITER {
        let m = m as f64;
        let m2 = 2.0 * m;

        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < LENTZ_TINY {
            d = LENTZ_TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < LENTZ_TINY {
            c = LENTZ_TINY;
        }
        d = 1.0 / d;
        h *= d * c;

        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < LENTZ_TINY {
            d = LENTZ_TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < LENTZ_TINY {
            c = LENTZ_TINY;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;

        if (del - 1.0).abs() < BETA_CF_EPS {
            break;
        }
    }

    h
}

/// Log of the regularised incomplete beta.
///
/// `ln I(x; a, b)` without forming `I`, so a tail of 1e-4000 is about `-9210`,
/// not `-inf`. R's `pbeta(log.p = TRUE)`. Needed by the unequal-df conversion in
/// `tmixture.vector` for a moderated t of 40 against a near-infinite `df.total`.
///
/// ### Params
///
/// * `x` - Argument in `[0, 1]`
/// * `a` - First shape, strictly positive
/// * `b` - Second shape, strictly positive
///
/// ### Returns
///
/// `ln I(x; a, b)`. `x = 0` gives `-inf` and `x = 1` gives `0`.
fn ln_beta_reg(x: f64, a: f64, b: f64) -> f64 {
    ln_beta_reg_pair(x, 1.0 - x, a, b)
}

/// Log of the regularised incomplete beta, given both `x` and `1 - x`.
///
/// The prefactor is `a ln x + b ln(1 - x)`; when one of `x`, `1 - x` is near one,
/// its log is taken from the other. `a` is half the degrees of freedom, which
/// runs into the millions for a moderated t, so an absolute error of 1e-16 in
/// `ln x` is multiplied by that.
///
/// Callers should compute the complement without cancellation where they can, as
/// [`t_sf_log`] does: `1 - df / (df + x^2)` is `x^2 / (df + x^2)`.
///
/// ### Params
///
/// * `x` - Argument in `[0, 1]`
/// * `one_minus_x` - Its complement, computed independently where possible
/// * `a` - First shape, strictly positive
/// * `b` - Second shape, strictly positive
///
/// ### Returns
///
/// `ln I(x; a, b)`.
fn ln_beta_reg_pair(x: f64, one_minus_x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if one_minus_x <= 0.0 {
        return 0.0;
    }

    // `ln1p` of the complement is more accurate when the value is near one.
    let ln_x = if x > 0.5 {
        (-one_minus_x).ln_1p()
    } else {
        x.ln()
    };
    let ln_1mx = if one_minus_x > 0.5 {
        (-x).ln_1p()
    } else {
        one_minus_x.ln()
    };

    // The continued fraction converges quickly only below this switchover;
    // past it, evaluate the complement and take `ln(1 - .)`.
    if x < (a + 1.0) / (a + b + 2.0) {
        let ln_pre = a * ln_x + b * ln_1mx - ln_beta_stable(a, b);
        ln_pre + (beta_cf(a, b, x) / a).ln()
    } else {
        let ln_other =
            b * ln_1mx + a * ln_x - ln_beta_stable(b, a) + (beta_cf(b, a, one_minus_x) / b).ln();
        (-ln_other.exp()).ln_1p()
    }
}

/////////////////////////////
// Incomplete beta inverse //
/////////////////////////////

/// Inverts the regularised incomplete beta on its lower half.
///
/// `statrs`' `inv_beta_reg` (AS 109) has a `1e-30` floor and drifts by orders of
/// magnitude below about `p = 1e-12` for small shapes. It is used as a start,
/// then polished by Newton on `ln I(x; a, b) = ln p` in `t = ln x`. For small
/// `x`, `I ~ x^a / (a B(a, b))`, so the residual is near-linear in `t` and two
/// or three steps land it. A bracket doubled outwards guards a bad start.
///
/// Only the lower half is solved. Callers wanting a quantile above the median
/// flip the shapes and probability and take the complement.
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
    inv_ln_beta_reg_lower(a, b, p.ln())
}

/// Inverts the regularised incomplete beta from a log probability.
///
/// The body of [`inv_beta_reg_lower`]. Taking the target in logs reaches
/// probabilities below `ln p = -745`, which are not representable otherwise.
///
/// ### Params
///
/// * `a` - First shape, strictly positive
/// * `b` - Second shape, strictly positive
/// * `ln_p` - Log of the target probability, at or below `ln(0.5)`
///
/// ### Returns
///
/// `x` with `ln I(x; a, b) = ln_p`, or [`EdgeErrors::NoConvergence`].
fn inv_ln_beta_reg_lower(a: f64, b: f64, ln_p: f64) -> Result<f64, EdgeErrors> {
    if ln_p == f64::NEG_INFINITY {
        return Ok(0.0);
    }
    let ln_b = ln_beta(a, b);

    // g(t) = ln I(e^t; a, b) - ln p, increasing in t.
    let g = |t: f64| ln_beta_reg(t.exp().min(1.0), a, b) - ln_p;

    let start = if ln_p > MIN_REPRESENTABLE_LN {
        inv_beta_reg(a, b, ln_p.exp())
    } else {
        0.0
    };
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
        let ln_i = ln_beta_reg(x, a, b);
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
        routine: "inv_ln_beta_reg_lower",
        iterations: INV_BETA_MAX_ITER,
        last_delta: hi - lo,
    })
}

/// Inverts the regularised incomplete beta, returning both `x` and `1 - x`.
///
/// Solving the side below the median and complementing gives the small quantity
/// to full relative accuracy, which the F and t quantiles divide by.
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
/// halved. Below zero it uses the upper branch, above zero the lower, so the
/// small tail is always computed directly.
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
/// The mirror of [`norm_cdf`], not `1 - norm_cdf(x)`: `norm_sf(37)` is 5.7e-300
/// and the subtraction gives zero from about `x = 8`.
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
/// `-sqrt(2) * erfc_inv(2p)`, so a tiny `p` uses the tail branch of the inverse.
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
/// The upper incomplete gamma `Q(df/2, x/2)` directly: `chisq_sf(200, 1)` is
/// 2.09e-45. Every edgeR LRT p-value comes from this tail.
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

/////////////////
// Student's t //
/////////////////

/// The two halves of the t distribution's mass either side of `|x|`.
///
/// With `h = df / (df + x^2)` and `z = x^2 / (df + x^2)`, the outer tail beyond
/// `|x|` is `I(h; df/2, 1/2) / 2` and the mass in `(0, |x|)` is
/// `I(z; 1/2, df/2) / 2`. The smaller one is evaluated, per
/// [`T_INNER_OUTER_SWITCH`], and the larger recovered by subtraction.
///
/// Both directions matter: at `x = 1e-4, df = 100`, `h` is `1 - 1e-10`, so
/// the tail derived from its complement is wrong in the eleventh digit.
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
/// For `x > 0` this is the incomplete beta itself, so `t_sf(50, 100)` is
/// 7.24e-73. Used for moderated t p-values and the t-to-z conversion in `glmTreat`.
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
/// Inverts the incomplete beta on the smaller side of the median.
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

/// Log of Student's t survival function.
///
/// `ln P(T > x)` from the log incomplete beta, so it survives where the
/// probability underflows. R's `pt(lower.tail = FALSE, log.p = TRUE)`.
///
/// `tmixture.vector` sends the top `proportion / 2` of genes by moderated t
/// through here. With infinite prior df, `df.total` is in the millions and a t
/// of 40 is already past 1e-350.
///
/// ### Params
///
/// * `x` - Quantile
/// * `df` - Degrees of freedom, finite and strictly positive
///
/// ### Returns
///
/// `ln P(T > x)`, or [`EdgeErrors::InvalidArgument`] for a non-positive `df`.
pub fn t_sf_log(x: f64, df: f64) -> Result<f64, EdgeErrors> {
    check_positive("df", df)?;
    if x.is_infinite() {
        return Ok(if x > 0.0 { f64::NEG_INFINITY } else { 0.0 });
    }
    if x <= 0.0 {
        // At or below the median the plain survival function is accurate.
        return Ok(t_sf(x, df)?.ln());
    }
    // Both halves formed directly: `1 - df / (df + x^2)` is `x^2 / (df + x^2)`,
    // so neither is a subtraction of near-equal numbers.
    let x2 = x * x;
    let denom = df + x2;
    Ok(-std::f64::consts::LN_2 + ln_beta_reg_pair(df / denom, x2 / denom, 0.5 * df, 0.5))
}

/// Student's t upper quantile from a log probability.
///
/// The inverse of [`t_sf_log`]: given `ln p`, returns the `t` with
/// `P(T > t) = p`. R's `qt(lower.tail = FALSE, log.p = TRUE)`.
///
/// ### Params
///
/// * `log_p` - Log of the upper tail probability, at or below zero
/// * `df` - Degrees of freedom, finite and strictly positive
///
/// ### Returns
///
/// `t` with `ln P(T > t) = log_p`. `log_p = 0` gives `-inf` and
/// `log_p = -inf` gives `+inf`. [`EdgeErrors::InvalidArgument`] for a positive
/// `log_p` or a non-positive `df`.
pub fn t_isf_log(log_p: f64, df: f64) -> Result<f64, EdgeErrors> {
    check_positive("df", df)?;
    if log_p > 0.0 || log_p.is_nan() {
        return Err(EdgeErrors::InvalidArgument(format!(
            "log_p must be at most 0; got {log_p}"
        )));
    }
    if log_p == 0.0 {
        return Ok(f64::NEG_INFINITY);
    }
    if log_p == f64::NEG_INFINITY {
        return Ok(f64::INFINITY);
    }
    if log_p > -std::f64::consts::LN_2 {
        // Above the median the log scale rescues nothing; `t_ppf` resolves
        // the other tail.
        return t_ppf(1.0 - log_p.exp(), df);
    }
    // P(T > t) = I(df / (df + t^2); df/2, 1/2) / 2, so invert on the half.
    let z = inv_ln_beta_reg_lower(0.5 * df, 0.5, log_p + std::f64::consts::LN_2)?;
    if z <= 0.0 {
        return Ok(f64::INFINITY);
    }
    Ok((df * (1.0 - z) / z).sqrt())
}

///////
// F //
///////

/// F survival function.
///
/// `I(df2 / (df1 x + df2); df2/2, df1/2)`, from the ratio directly. The
/// `1 - df1 x / (df1 x + df2)` form is where `statrs`' `FisherSnedecor::sf` loses
/// the tail that `glmQLFTest` reads.
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
/// uncancelled side, so `f_ppf(1 - eps, ..)` never divides by a rounded `1 - z`.
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
    // complement are accurate, so no cancelled difference.
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
/// `I(1 - x; b, a)`, the reflected incomplete beta. `exactTestBetaApprox`
/// doubles this for its right-tail p-value.
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
/// Matches `scipy.stats.gamma.cdf(x, a=shape, scale=scale)`: `scale` is the
/// reciprocal of the rate. Used by `q2qnbinom`'s gamma approximation.
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
/// Hand-rolled: `statrs`' `Gamma::inverse_cdf` is only good to about 1e-8
/// relative, too coarse for `q2qnbinom`. Solves `gamma_lr(shape, x) = p` by
/// Newton in `ln(x)`, bisecting when Newton leaves the bracket, and inverts the
/// upper tail for `p > 0.5` so the residual never cancels.
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
/// closed within `GAMMA_PPF_MAX_ITER`.
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
/// Wilson-Hilferty where its cube root is positive (`shape` above about 1).
/// Below that it goes negative, so the small-`x` series
/// `P(a, x) ~ x^a / (a Gamma(a))` is inverted instead. A poor guess costs
/// iterations, not correctness: the caller brackets and bisects.
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
/// Newton in `t = ln(x)` on a bracketed, increasing residual. The derivative
/// `x f(x) = exp(shape ln x - x - ln_gamma(shape))` stays representable for the
/// tiny quantiles of small shapes, where the density would overflow.
///
/// For `p > 0.5` the residual uses `gamma_ur` and `1 - p` (exact by Sterbenz).
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
/// Matches `scipy.stats.nbinom.logpmf(k, size, prob)` and `_nb_logpmf` in
/// `edgepython/exact_test.py`: `prob` is the success probability, `size` the
/// target number of successes, `k` counts failures, and the mean is
/// `size (1 - prob) / prob`. edgePython always calls it with
/// `prob = size / (size + mu)`. The other common convention puts `prob` on the
/// failures, a different distribution for the same numbers.
///
/// ```text
/// ln P(K = k) = lgamma(k + size) - lgamma(k + 1) - lgamma(size)
///               + size ln(prob) + k ln(1 - prob)
/// ```
///
/// The last term is `k * ln_1p(-prob)`, as in `scipy`, which is more accurate
/// than `ln(1 - prob)` for small `prob`.
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
/// Same convention as [`nbinom_ln_pmf`]. Like `scipy`, evaluated as the
/// incomplete beta `I(prob; size, floor(k) + 1)`, not by summing the PMF.
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
    // Reference values are pasted verbatim from R's 17-digit output, so they
    // can be checked against the quoted `Rscript` line.
    #![allow(clippy::excessive_precision)]

    use super::*;
    use approx::assert_relative_eq;

    /// Target relative accuracy against scipy.
    ///
    /// A sweep of about 1500 points holds to this, apart from the mid-tail
    /// incomplete beta noted in [`BETA_REG_TOL`].
    const TOL: f64 = 1e-12;

    /// Relaxed tolerance where `statrs`'s `beta_reg` is the limiting factor.
    ///
    /// Past its internal symmetry swap it returns `1 - I(1-x)`, costing about a
    /// digit when the tail is a tenth of the mass. This is the whole gap to
    /// [`TOL`]; it only bites in the mid-tail of t and F at large `df`.
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

    /// `Rscript -e 'pt(x, df, lower.tail=FALSE, log.p=TRUE)'` for each pair.
    ///
    /// The last two are below what an `f64` probability can hold.
    const T_SF_LOG: [(f64, f64, f64); 8] = [
        (2.5, 10.0, -4.1526038236684464),
        (8.0, 5.0, -8.3083379118742755),
        (50.0, 100.0, -166.10963191099665),
        (300.0, 3.0, -17.013663984295441),
        (0.5, 7.0, -1.1513690707642816),
        (-2.5, 10.0, -0.015848346341060398),
        (40.0, 1e6, -803.96832475034205),
        (120.0, 1e5, -6732.1838846408191),
    ];

    #[test]
    fn test_t_sf_log_matches_r() {
        for &(x, df, want) in &T_SF_LOG {
            // Worst observed is one ULP. The `df = 1e6` case pins both
            // accuracy fixes: without the paired form it is out by 2.8e-13,
            // without the Stirling `ln_beta` by 2.6e-13.
            let got = t_sf_log(x, df).unwrap();
            assert_relative_eq!(got, want, max_relative = 1e-15);
        }
    }

    #[test]
    fn test_t_sf_log_reaches_past_underflow() {
        // The plain survival function has underflowed here.
        assert_eq!(t_sf(120.0, 1e5).unwrap(), 0.0);
        assert!(t_sf_log(120.0, 1e5).unwrap() < -6000.0);
    }

    #[test]
    fn test_t_isf_log_round_trips() {
        for &(x, df, log_p) in &T_SF_LOG {
            let back = t_isf_log(log_p, df).unwrap();
            assert_relative_eq!(back, x, max_relative = 1e-10);
        }
    }

    #[test]
    fn test_t_isf_log_edges() {
        assert_eq!(t_isf_log(0.0, 5.0).unwrap(), f64::NEG_INFINITY);
        assert_eq!(t_isf_log(f64::NEG_INFINITY, 5.0).unwrap(), f64::INFINITY);
        assert!(t_isf_log(0.5, 5.0).is_err());
        assert!(t_isf_log(-1.0, 0.0).is_err());
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
        // The naive h = df / (df + x^2) form loses six digits here: 1 - h is 1e-10.
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
        // The hand-rolled inverses return NoConvergence rather than a wrong
        // answer, so a sweep that never errs checks the bracketing over five decades.
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
