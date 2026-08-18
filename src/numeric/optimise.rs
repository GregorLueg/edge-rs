//! Scalar minimisation, root finding and derivative-free simplex search.
//!
//! Between them these cover every `scipy.optimize` call in edgePython except the
//! bounded quasi-Newton, which lives in [`crate::numeric::lbfgsb`]:
//!
//! * `minimize_scalar(method = 'bounded')` becomes [`brent_minimise_bounded`],
//!   used by `disp_cox_reid` and the TMM-to-ChIP normalisation
//! * `brentq` becomes [`brentq`], used by `disp_pearson`, `disp_deviance` and
//!   limma's `fitFDist`
//! * `minimize(method = 'Nelder-Mead')` becomes [`nelder_mead`], used by the
//!   spline and power trend fitters
//!
//! The initial simplex in [`nelder_mead`] reproduces scipy's construction
//! exactly, because a derivative-free search on a flat likelihood surface is
//! sensitive to it and the trend fits are exactly that shape.

use std::cmp::Ordering;

use crate::errors::EdgeErrors;

/// Squared inverse of the golden ratio, the golden section step.
const GOLDEN: f64 = 0.381_966_011_250_105_2;

/// Default `tol` of R's `optimize`, `.Machine$double.eps^0.25`, which is exactly
/// `2^-13`.
pub const OPTIMIZE_TOL: f64 = 1.220_703_125e-4;

/// `sqrt(DBL_EPSILON)`, exactly `2^-26`, the relative half of `Brent_fmin`'s
/// convergence test.
const BRENT_SQRT_EPS: f64 = 1.490_116_119_384_765_6e-8;

/// Golden section step `(3 - sqrt(5)) / 2`, as `Brent_fmin` spells it.
const BRENT_GOLDEN: f64 = 0.381_966_011_250_105_15;

/// Relative step used to build scipy's initial Nelder-Mead simplex.
///
/// scipy perturbs each coordinate by 5% of its value. Matching this matters:
/// the trend fitters minimise a nearly flat objective, where a different
/// starting simplex converges to a visibly different point.
const NELDER_MEAD_STEP: f64 = 0.05;

/// Absolute perturbation used where a starting coordinate is exactly zero.
///
/// A relative step cannot move a zero, so scipy substitutes this constant.
const NELDER_MEAD_ZERO_STEP: f64 = 0.000_25;

/////////////////////////////
// Bounded scalar minimum  //
/////////////////////////////

/// Minimises a scalar function on a closed interval.
///
/// Brent's method: golden section search with parabolic interpolation accepted
/// whenever it stays inside the bracket and improves on the previous step. This
/// is the algorithm behind `scipy.optimize.minimize_scalar(method = 'bounded')`,
/// down to the convergence test.
///
/// ### Params
///
/// * `f` - Objective, evaluated only inside `[lower, upper]`
/// * `lower` - Lower end of the interval
/// * `upper` - Upper end of the interval
/// * `xatol` - Absolute tolerance on the location of the minimum
/// * `max_iter` - Evaluation budget
///
/// ### Returns
///
/// The minimiser and its objective value, or [`EdgeErrors`] if the interval is
/// inverted or the budget runs out.
///
/// ### References
///
/// Brent, Algorithms for Minimization without Derivatives, 1973, chapter 5
pub fn brent_minimise_bounded<F>(
    mut f: F,
    lower: f64,
    upper: f64,
    xatol: f64,
    max_iter: usize,
) -> Result<(f64, f64), EdgeErrors>
where
    F: FnMut(f64) -> f64,
{
    // Written as an explicit ordering test so a NaN bound is rejected rather
    // than slipping through a negated comparison.
    if lower.partial_cmp(&upper) != Some(Ordering::Less) {
        return Err(EdgeErrors::InvalidArgument(format!(
            "brent_minimise_bounded needs lower < upper, got [{lower}, {upper}]"
        )));
    }
    if xatol <= 0.0 {
        return Err(EdgeErrors::MustBePositive("xatol".to_string()));
    }

    let (mut a, mut b) = (lower, upper);
    let mut x = a + GOLDEN * (b - a);
    let (mut w, mut v) = (x, x);
    let mut fx = f(x);
    let (mut fw, mut fv) = (fx, fx);
    let mut step_before_last: f64 = 0.0;
    let mut last_step: f64 = 0.0;

    for _ in 0..max_iter {
        let midpoint = 0.5 * (a + b);
        // scipy's test: the bracket is small relative to |x| plus the absolute floor.
        let tol1 = xatol * x.abs() + xatol / 3.0;
        let tol2 = 2.0 * tol1;
        if (x - midpoint).abs() <= tol2 - 0.5 * (b - a) {
            return Ok((x, fx));
        }

        let mut golden_step = true;
        if step_before_last.abs() > tol1 {
            // Fit a parabola through (x, fx), (w, fw), (v, fv).
            let r = (x - w) * (fx - fv);
            let mut q = (x - v) * (fx - fw);
            let mut p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p;
            }
            q = q.abs();
            let prev = step_before_last;
            step_before_last = last_step;

            let acceptable = p.abs() < (0.5 * q * prev).abs() && p > q * (a - x) && p < q * (b - x);
            if acceptable {
                last_step = p / q;
                let u = x + last_step;
                // Never evaluate closer than tol2 to an endpoint.
                if (u - a) < tol2 || (b - u) < tol2 {
                    last_step = if x < midpoint { tol1 } else { -tol1 };
                }
                golden_step = false;
            }
        }

        if golden_step {
            step_before_last = if x >= midpoint { a - x } else { b - x };
            last_step = GOLDEN * step_before_last;
        }

        let u = if last_step.abs() >= tol1 {
            x + last_step
        } else {
            x + tol1.copysign(last_step)
        };
        let fu = f(u);

        if fu <= fx {
            if u < x {
                b = x;
            } else {
                a = x;
            }
            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        } else {
            if u < x {
                a = u;
            } else {
                b = u;
            }
            if fu <= fw || w == x {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu;
            }
        }
    }

    Err(EdgeErrors::NoConvergence {
        routine: "brent_minimise_bounded",
        iterations: max_iter,
        last_delta: (b - a).abs(),
    })
}

//////////////////
// Root finding //
//////////////////

/// Finds a root of a continuous function on a bracketing interval.
///
/// Brent's method: bisection guaranteed, accelerated by inverse quadratic
/// interpolation and the secant rule where they behave. This is
/// `scipy.optimize.brentq`.
///
/// ### Params
///
/// * `f` - Function whose root is wanted
/// * `lower` - Lower end of the bracket
/// * `upper` - Upper end of the bracket
/// * `xtol` - Absolute tolerance on the root
/// * `rtol` - Relative tolerance on the root
/// * `max_iter` - Iteration budget
///
/// ### Returns
///
/// The root, or [`EdgeErrors::RootNotBracketed`] if the endpoints share a sign,
/// or [`EdgeErrors::NoConvergence`] if the budget runs out.
///
/// ### References
///
/// Brent, Algorithms for Minimization without Derivatives, 1973, chapter 4
pub fn brentq<F>(
    mut f: F,
    lower: f64,
    upper: f64,
    xtol: f64,
    rtol: f64,
    max_iter: usize,
) -> Result<f64, EdgeErrors>
where
    F: FnMut(f64) -> f64,
{
    if lower.partial_cmp(&upper) != Some(Ordering::Less) {
        return Err(EdgeErrors::InvalidArgument(format!(
            "brentq needs lower < upper, got [{lower}, {upper}]"
        )));
    }
    if xtol <= 0.0 {
        return Err(EdgeErrors::MustBePositive("xtol".to_string()));
    }

    let (mut xpre, mut xcur) = (lower, upper);
    let (mut fpre, mut fcur) = (f(xpre), f(xcur));

    if fpre == 0.0 {
        return Ok(xpre);
    }
    if fcur == 0.0 {
        return Ok(xcur);
    }
    if fpre.signum() == fcur.signum() {
        return Err(EdgeErrors::RootNotBracketed {
            lower,
            upper,
            f_lower: fpre,
            f_upper: fcur,
        });
    }

    let (mut xblk, mut fblk) = (0.0, 0.0);
    let (mut spre, mut scur) = (0.0, 0.0);

    for _ in 0..max_iter {
        if fpre != 0.0 && fcur != 0.0 && fpre.signum() != fcur.signum() {
            xblk = xpre;
            fblk = fpre;
            spre = xcur - xpre;
            scur = spre;
        }
        if fblk.abs() < fcur.abs() {
            xpre = xcur;
            xcur = xblk;
            xblk = xpre;
            fpre = fcur;
            fcur = fblk;
            fblk = fpre;
        }

        let delta = (xtol + rtol * xcur.abs()) / 2.0;
        let sbis = (xblk - xcur) / 2.0;
        if fcur == 0.0 || sbis.abs() < delta {
            return Ok(xcur);
        }

        if spre.abs() > delta && fcur.abs() < fpre.abs() {
            let stry = if xpre == xblk {
                // Secant.
                -fcur * (xcur - xpre) / (fcur - fpre)
            } else {
                // Inverse quadratic interpolation.
                let dpre = (fpre - fcur) / (xpre - xcur);
                let dblk = (fblk - fcur) / (xblk - xcur);
                -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))
            };
            if 2.0 * stry.abs() < (spre.abs()).min(3.0 * sbis.abs() - delta) {
                spre = scur;
                scur = stry;
            } else {
                spre = sbis;
                scur = sbis;
            }
        } else {
            spre = sbis;
            scur = sbis;
        }

        xpre = xcur;
        fpre = fcur;
        if scur.abs() > delta {
            xcur += scur;
        } else {
            xcur += if sbis > 0.0 { delta } else { -delta };
        }
        fcur = f(xcur);
    }

    Err(EdgeErrors::NoConvergence {
        routine: "brentq",
        iterations: max_iter,
        last_delta: (xblk - xcur).abs(),
    })
}

/////// Minimises a scalar function on a closed interval, as R's `optimize` does.
///
/// A line-by-line port of R's `Brent_fmin`. This is deliberately not
/// [`brent_minimise_bounded`]: that one is scipy's
/// `minimize_scalar(method = 'bounded')`, and although the algorithm is the
/// same Brent search, its convergence test is `xatol * |x| + xatol / 3` where
/// R's is `sqrt(eps) * |x| + tol / 3`. The two constants cannot be recovered
/// from a single `xatol`, so the iterates diverge and, on a likelihood as flat
/// as this one, so does the answer. Matching limma to more than its own `tol`
/// requires stopping where limma stops.
///
/// ### Params
///
/// * `ax` - Lower end of the interval
/// * `bx` - Upper end of the interval
/// * `f` - Objective
/// * `tol` - Convergence tolerance, R's `optimize(tol = )`
///
/// ### Returns
///
/// The minimiser.
///
/// ### References
///
/// Brent, Algorithms for Minimization without Derivatives, 1973, chapter 5
pub fn brent_fmin<F>(ax: f64, bx: f64, mut f: F, tol: f64) -> f64
where
    F: FnMut(f64) -> f64,
{
    let (mut a, mut b) = (ax, bx);
    let mut v = a + BRENT_GOLDEN * (b - a);
    let (mut w, mut x) = (v, v);
    let mut d = 0.0_f64;
    let mut e = 0.0_f64;
    let mut fx = f(x);
    let (mut fv, mut fw) = (fx, fx);
    let tol3 = tol / 3.0;

    loop {
        let xm = 0.5 * (a + b);
        let tol1 = BRENT_SQRT_EPS * x.abs() + tol3;
        let t2 = 2.0 * tol1;
        if (x - xm).abs() <= t2 - 0.5 * (b - a) {
            return x;
        }

        let (mut p, mut q, mut r) = (0.0_f64, 0.0_f64, 0.0_f64);
        if e.abs() > tol1 {
            r = (x - w) * (fx - fv);
            q = (x - v) * (fx - fw);
            p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p;
            } else {
                q = -q;
            }
            r = e;
            e = d;
        }

        if p.abs() >= (0.5 * q * r).abs() || p <= q * (a - x) || p >= q * (b - x) {
            e = if x < xm { b - x } else { a - x };
            d = BRENT_GOLDEN * e;
        } else {
            d = p / q;
            let u = x + d;
            // f must not be evaluated too close to either end of the interval.
            if u - a < t2 || b - u < t2 {
                d = if x >= xm { -tol1 } else { tol1 };
            }
        }

        // Nor too close to the incumbent.
        let u = if d.abs() >= tol1 {
            x + d
        } else if d > 0.0 {
            x + tol1
        } else {
            x - tol1
        };
        let fu = f(u);

        if fu <= fx {
            if u < x {
                b = x;
            } else {
                a = x;
            }
            v = w;
            fv = fw;
            w = x;
            fw = fx;
            x = u;
            fx = fu;
        } else {
            if u < x {
                a = u;
            } else {
                b = u;
            }
            if fu <= fw || w == x {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu;
            }
        }
    }
}

//////////////
// Nelder-Mead  //
//////////////////

/// Tuning knobs for [`nelder_mead`].
#[derive(Clone, Copy, Debug)]
pub struct NelderMeadParams {
    /// Absolute tolerance on the spread of the simplex vertices.
    pub xatol: f64,
    /// Absolute tolerance on the spread of the objective across the simplex.
    pub fatol: f64,
    /// Iteration budget.
    pub max_iter: usize,
}

impl Default for NelderMeadParams {
    /// scipy's defaults.
    fn default() -> Self {
        Self {
            xatol: 1e-4,
            fatol: 1e-4,
            max_iter: 1000,
        }
    }
}

/// Outcome of a simplex search.
#[derive(Clone, Debug)]
pub struct NelderMeadResult {
    /// Best vertex found.
    pub x: Vec<f64>,
    /// Objective at that vertex.
    pub f: f64,
    /// Iterations performed.
    pub iterations: usize,
    /// Whether both tolerances were met before the budget ran out.
    pub converged: bool,
}

/// Derivative-free minimisation by the Nelder-Mead simplex method.
///
/// The initial simplex is scipy's: the starting point, plus one vertex per
/// coordinate perturbed by 5%, or by a small absolute step where the coordinate
/// is zero. Reflection, expansion, contraction and shrink use the textbook
/// coefficients 1, 2, 0.5 and 0.5.
///
/// ### Params
///
/// * `f` - Objective
/// * `x0` - Starting point
/// * `params` - Tuning knobs, or [`NelderMeadParams::default`]
///
/// ### Returns
///
/// The best vertex and its value, or [`EdgeErrors::MustBePositive`] if `x0` is
/// empty.
pub fn nelder_mead<F>(
    mut f: F,
    x0: &[f64],
    params: Option<NelderMeadParams>,
) -> Result<NelderMeadResult, EdgeErrors>
where
    F: FnMut(&[f64]) -> f64,
{
    let params = params.unwrap_or_default();
    let n = x0.len();
    if n == 0 {
        return Err(EdgeErrors::MustBePositive("x0.len()".to_string()));
    }

    // scipy's initial simplex.
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(x0.to_vec());
    for i in 0..n {
        let mut vertex = x0.to_vec();
        if vertex[i] != 0.0 {
            vertex[i] *= 1.0 + NELDER_MEAD_STEP;
        } else {
            vertex[i] = NELDER_MEAD_ZERO_STEP;
        }
        simplex.push(vertex);
    }

    let mut values: Vec<f64> = simplex.iter().map(|v| f(v)).collect();
    let mut order: Vec<usize> = (0..=n).collect();
    let mut iterations = 0;
    let mut converged = false;

    for iter in 0..params.max_iter {
        iterations = iter + 1;

        order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
        let best = order[0];
        let worst = order[n];
        let second_worst = order[n - 1];

        // Convergence: both the simplex and the objective values have collapsed.
        let spread_x = order[1..]
            .iter()
            .map(|&i| {
                simplex[i]
                    .iter()
                    .zip(simplex[best].iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f64, f64::max)
            })
            .fold(0.0_f64, f64::max);
        let spread_f = order[1..]
            .iter()
            .map(|&i| (values[i] - values[best]).abs())
            .fold(0.0_f64, f64::max);
        if spread_x <= params.xatol && spread_f <= params.fatol {
            converged = true;
            break;
        }

        // Centroid of everything but the worst vertex.
        let mut centroid = vec![0.0; n];
        for &i in &order[..n] {
            for (c, v) in centroid.iter_mut().zip(simplex[i].iter()) {
                *c += *v;
            }
        }
        for c in centroid.iter_mut() {
            *c /= n as f64;
        }

        let reflected: Vec<f64> = (0..n)
            .map(|j| centroid[j] + (centroid[j] - simplex[worst][j]))
            .collect();
        let f_reflected = f(&reflected);

        if f_reflected < values[best] {
            let expanded: Vec<f64> = (0..n)
                .map(|j| centroid[j] + 2.0 * (centroid[j] - simplex[worst][j]))
                .collect();
            let f_expanded = f(&expanded);
            if f_expanded < f_reflected {
                simplex[worst] = expanded;
                values[worst] = f_expanded;
            } else {
                simplex[worst] = reflected;
                values[worst] = f_reflected;
            }
        } else if f_reflected < values[second_worst] {
            simplex[worst] = reflected;
            values[worst] = f_reflected;
        } else {
            let inside = f_reflected >= values[worst];
            let contracted: Vec<f64> = if inside {
                (0..n)
                    .map(|j| centroid[j] - 0.5 * (centroid[j] - simplex[worst][j]))
                    .collect()
            } else {
                (0..n)
                    .map(|j| centroid[j] + 0.5 * (centroid[j] - simplex[worst][j]))
                    .collect()
            };
            let f_contracted = f(&contracted);
            let accept = if inside {
                f_contracted < values[worst]
            } else {
                f_contracted <= f_reflected
            };

            if accept {
                simplex[worst] = contracted;
                values[worst] = f_contracted;
            } else {
                // Shrink every vertex towards the best one.
                let anchor = simplex[best].clone();
                for &i in &order[1..] {
                    for j in 0..n {
                        simplex[i][j] = anchor[j] + 0.5 * (simplex[i][j] - anchor[j]);
                    }
                    values[i] = f(&simplex[i]);
                }
            }
        }
    }

    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let best = order[0];
    Ok(NelderMeadResult {
        x: simplex[best].clone(),
        f: values[best],
        iterations,
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

    #[test]
    fn test_brent_minimise_finds_a_smooth_interior_minimum() {
        // (x - 1.234)^2 + 3, minimum at 1.234.
        let (x, fx) =
            brent_minimise_bounded(|x| (x - 1.234).powi(2) + 3.0, -5.0, 5.0, 1e-10, 200).unwrap();
        assert_relative_eq!(x, 1.234, max_relative = 1e-7);
        assert_relative_eq!(fx, 3.0, max_relative = 1e-10);
    }

    /// scipy: `minimize_scalar(lambda x: (x-2)**2*np.exp(x), bounds=(0,5), method='bounded')`
    /// gives x = 2.0 to the default tolerance.
    #[test]
    fn test_brent_minimise_matches_scipy_on_a_skewed_objective() {
        let (x, _) =
            brent_minimise_bounded(|x: f64| (x - 2.0).powi(2) * x.exp(), 0.0, 5.0, 1e-8, 200)
                .unwrap();
        assert_relative_eq!(x, 2.0, max_relative = 1e-6);
    }

    /// The minimum lies outside the interval, so the answer must be the endpoint.
    #[test]
    fn test_brent_minimise_returns_an_endpoint_when_the_minimum_is_outside() {
        let (x, _) =
            brent_minimise_bounded(|x: f64| (x - 10.0).powi(2), 0.0, 1.0, 1e-8, 200).unwrap();
        assert_relative_eq!(x, 1.0, max_relative = 1e-4);
    }

    #[test]
    fn test_brent_minimise_rejects_an_inverted_interval() {
        let err = brent_minimise_bounded(|x: f64| x, 2.0, 1.0, 1e-8, 100).unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));
    }

    #[test]
    fn test_brentq_finds_a_polynomial_root() {
        // x^3 - 2x - 5 has its real root near 2.0945514815423265.
        let root = brentq(
            |x: f64| x.powi(3) - 2.0 * x - 5.0,
            1.0,
            3.0,
            1e-14,
            1e-15,
            200,
        )
        .unwrap();
        assert_relative_eq!(root, 2.094_551_481_542_326_5, max_relative = 1e-12);
    }

    #[test]
    fn test_brentq_handles_a_root_at_an_endpoint() {
        let root = brentq(|x: f64| x - 1.0, 1.0, 3.0, 1e-14, 1e-15, 200).unwrap();
        assert_relative_eq!(root, 1.0, max_relative = 1e-14);
    }

    #[test]
    fn test_brentq_rejects_an_unbracketed_interval() {
        let err = brentq(|x: f64| x * x + 1.0, -1.0, 1.0, 1e-12, 1e-14, 100).unwrap_err();
        assert!(matches!(err, EdgeErrors::RootNotBracketed { .. }));
    }

    /// A steep function where bisection alone would be slow, so the
    /// interpolation branches actually run.
    #[test]
    fn test_brentq_converges_on_a_steep_function() {
        let root = brentq(|x: f64| x.exp() - 1e6, 0.0, 30.0, 1e-12, 1e-14, 200).unwrap();
        assert_relative_eq!(root, 1e6_f64.ln(), max_relative = 1e-10);
    }

    #[test]
    fn test_nelder_mead_minimises_rosenbrock() {
        let res = nelder_mead(
            |x: &[f64]| 100.0 * (x[1] - x[0] * x[0]).powi(2) + (1.0 - x[0]).powi(2),
            &[-1.2, 1.0],
            Some(NelderMeadParams {
                xatol: 1e-10,
                fatol: 1e-12,
                max_iter: 5000,
            }),
        )
        .unwrap();
        assert!(res.converged);
        assert_relative_eq!(res.x[0], 1.0, max_relative = 1e-4);
        assert_relative_eq!(res.x[1], 1.0, max_relative = 1e-4);
    }

    #[test]
    fn test_nelder_mead_handles_a_zero_starting_coordinate() {
        // The zero coordinate exercises the absolute-step branch of the simplex.
        let res = nelder_mead(
            |x: &[f64]| (x[0] - 3.0).powi(2) + (x[1] + 2.0).powi(2),
            &[0.0, 0.0],
            Some(NelderMeadParams {
                xatol: 1e-10,
                fatol: 1e-12,
                max_iter: 5000,
            }),
        )
        .unwrap();
        assert_relative_eq!(res.x[0], 3.0, max_relative = 1e-5);
        assert_relative_eq!(res.x[1], -2.0, max_relative = 1e-5);
    }

    #[test]
    fn test_nelder_mead_rejects_empty_start() {
        let err = nelder_mead(|_: &[f64]| 0.0, &[], None).unwrap_err();
        assert!(matches!(err, EdgeErrors::MustBePositive(_)));
    }
}
