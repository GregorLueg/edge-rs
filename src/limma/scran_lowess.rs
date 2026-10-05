//! libscran's `WeightedLowess`.
//!
//! The smoother behind scrapper's `fitVarianceTrend`. It descends from the
//! same C code as limma's `weightedLowess` and shares its seed selection, so
//! [`resolve_delta`] and [`find_seeds`] are reused from [`super::lowess`]. The
//! rest differs in ways that move the fit, which is why this is a sibling
//! function rather than a flag on [`super::lowess::weighted_lowess`]:
//!
//! - the span can be a point count instead of a proportion, and every window
//!   can be forced to a minimum width in `x`
//! - equal left and right distances grow the window on both sides at once
//! - the tricube bandwidth is measured after the tie and width extensions
//! - degeneracy tests are exact (`dist <= 0`, `var == 0`), with no `1e-7`
//! - a window whose robustness weights are all zero falls back to the prior
//!   weights alone instead of returning zero
//! - `iterations` counts robustness passes, there is no early stop on a small
//!   MAD, and the weights are not recomputed after the final fit
//!
//! ### References
//!
//! Cleveland, Journal of the American Statistical Association, 1979

use crate::errors::EdgeErrors;
use crate::limma::lowess::{LowessFit, PARALLEL_WORK_THRESHOLD, fill, find_seeds, resolve_delta};

////////////
// Consts //
////////////

/// Lower bound on the robustness scale, as a fraction of the range of `y`.
///
/// libscran's `threshold_multiplier`. Stops a residual scale of zero from
/// dividing by zero when the fit is exact on most points.
const MIN_THRESHOLD_MULTIPLIER: f64 = 1e-8;

////////////////
// Public API //
////////////////

/// Tuning knobs for [`scran_lowess`].
///
/// The defaults are libscran's: `span = 0.3` as a proportion, no minimum
/// width, three robustness passes and 200 seed points.
#[derive(Clone, Copy, Debug)]
pub struct ScranLowessParams {
    /// Window size around each seed. A proportion of the total prior weight in
    /// `(0, 1]` when `span_as_proportion`, otherwise the prior weight itself
    /// (a point count with unit weights), which must be positive.
    pub span: f64,
    /// How `span` is read, see above.
    pub span_as_proportion: bool,
    /// Minimum width in `x` of every window. Zero disables it.
    pub minimum_width: f64,
    /// Number of robustness passes after the first fit. Zero is a single
    /// unweighted fit.
    pub iterations: usize,
    /// Approximate number of seed points to fit at. Only consulted when
    /// `delta` is `None`.
    pub npts: usize,
    /// Minimum distance in `x` between seed points. `None` derives it from
    /// `npts`; `Some(0.0)` fits at every point.
    pub delta: Option<f64>,
}

impl ScranLowessParams {
    /// Builds a parameter set.
    ///
    /// ### Params
    ///
    /// * `span` - Window size, as a proportion or a weight, see
    ///   `span_as_proportion`
    /// * `span_as_proportion` - Whether `span` is a proportion of the total
    ///   weight
    /// * `minimum_width` - Minimum window width in `x`, zero to disable
    /// * `iterations` - Number of robustness passes
    /// * `npts` - Approximate seed count, used only when `delta` is `None`
    /// * `delta` - Minimum seed spacing, or `None` to derive it from `npts`
    ///
    /// ### Returns
    ///
    /// The parameter set. Nothing is validated here; [`scran_lowess`] checks
    /// the values against the data it is given.
    pub fn new(
        span: f64,
        span_as_proportion: bool,
        minimum_width: f64,
        iterations: usize,
        npts: usize,
        delta: Option<f64>,
    ) -> Self {
        Self {
            span,
            span_as_proportion,
            minimum_width,
            iterations,
            npts,
            delta,
        }
    }
}

impl Default for ScranLowessParams {
    fn default() -> Self {
        Self {
            span: 0.3,
            span_as_proportion: true,
            minimum_width: 0.0,
            iterations: 3,
            npts: 200,
            delta: None,
        }
    }
}

/// The window around one seed point, as indices into the sorted data.
#[derive(Clone, Copy, Debug)]
struct Window {
    /// First index inside the window, inclusive.
    start: usize,
    /// Last index inside the window, inclusive.
    end: usize,
    /// Largest distance from the seed to either end of the final window. This
    /// is the tricube bandwidth.
    dist: f64,
}

///////////////
// Internals //
///////////////

/// Grows the window around one seed point.
///
/// Port of libscran's `find_limits`. Extends towards whichever neighbour is
/// closer, both at once on a tie in distance, until `span_weight` is enclosed
/// or one end of the data is hit; then tops up from the other side alone. The
/// window is widened over runs of tied `x` at either edge and, if it is still
/// narrower than `min_width`, out to `seed +- min_width / 2`.
///
/// ### Params
///
/// * `xs` - Sorted covariate values
/// * `ws` - Prior weights in the same order
/// * `curpt` - Index of the seed
/// * `span_weight` - Weight the window must enclose
/// * `min_width` - Minimum window width in `x`
///
/// ### Returns
///
/// The window bounds and the tricube bandwidth.
fn scran_window(xs: &[f64], ws: &[f64], curpt: usize, span_weight: f64, min_width: f64) -> Window {
    let last = xs.len() - 1;
    let curx = xs[curpt];
    let mut left = curpt;
    let mut right = curpt;
    let mut curw = ws[curpt];

    if curpt > 0 && curpt < last {
        let mut next_ldist = curx - xs[left - 1];
        let mut next_rdist = xs[right + 1] - curx;

        while curw < span_weight {
            if next_ldist < next_rdist {
                left -= 1;
                curw += ws[left];
                if left == 0 {
                    break;
                }
                next_ldist = curx - xs[left - 1];
            } else if next_ldist > next_rdist {
                right += 1;
                curw += ws[right];
                if right == last {
                    break;
                }
                next_rdist = xs[right + 1] - curx;
            } else {
                // Equal distances: take both, or one of them is skipped on a
                // break.
                left -= 1;
                right += 1;
                curw += ws[left] + ws[right];
                if left == 0 || right == last {
                    break;
                }
                next_ldist = curx - xs[left - 1];
                next_rdist = xs[right + 1] - curx;
            }
        }
    }

    while left > 0 && curw < span_weight {
        left -= 1;
        curw += ws[left];
    }
    while right < last && curw < span_weight {
        right += 1;
        curw += ws[right];
    }

    while left > 0 && xs[left] == xs[left - 1] {
        left -= 1;
    }
    while right < last && xs[right] == xs[right + 1] {
        right += 1;
    }

    let half_width = min_width / 2.0;
    let mut dist = (curx - xs[left]).max(xs[right] - curx);
    if dist < half_width {
        let lo = curx - half_width;
        let hi = curx + half_width;
        left = xs[..left].partition_point(|&v| v < lo);
        // First index past the window, minus one. `right` itself is inside.
        right += xs[right + 1..].partition_point(|&v| v <= hi);
        dist = (curx - xs[left]).max(xs[right] - curx);
    }

    Window {
        start: left,
        end: right,
        dist,
    }
}

/// Evaluates the tricube-weighted local line at one seed.
///
/// Port of libscran's `fit_point`. A zero bandwidth gives the weighted mean of
/// `y`, a zero weighted variance in `x` gives the intercept alone. If the
/// robustness weights zero out the whole window, they are dropped and the fit
/// is repeated with the prior weights only.
///
/// ### Params
///
/// * `xs` - Sorted covariate values
/// * `ys` - Responses in the same order
/// * `ws` - Prior weights in the same order
/// * `rw` - Current robustness weights in the same order
/// * `curpt` - Index of the seed being fitted
/// * `window` - Window and bandwidth from [`scran_window`]
///
/// ### Returns
///
/// The fitted value at `xs[curpt]`.
fn scran_fit_point(
    xs: &[f64],
    ys: &[f64],
    ws: &[f64],
    rw: &[f64],
    curpt: usize,
    window: Window,
) -> f64 {
    let span = window.start..=window.end;

    if window.dist <= 0.0 {
        let mut ymean = 0.0;
        let mut allweight = 0.0;
        for i in span.clone() {
            let w = rw[i] * ws[i];
            ymean += ys[i] * w;
            allweight += w;
        }
        if allweight == 0.0 {
            for i in span {
                ymean += ys[i] * ws[i];
                allweight += ws[i];
            }
        }
        return ymean / allweight;
    }

    let kernel = |i: usize, robust: bool| {
        let u = (xs[curpt] - xs[i]).abs() / window.dist;
        let t = 1.0 - u * u * u;
        let t = t * t * t;
        if robust { t * rw[i] * ws[i] } else { t * ws[i] }
    };

    let mut robust = true;
    let mut xmean = 0.0;
    let mut ymean = 0.0;
    let mut allweight = 0.0;
    for i in span.clone() {
        let w = kernel(i, true);
        xmean += w * xs[i];
        ymean += w * ys[i];
        allweight += w;
    }
    if allweight == 0.0 {
        robust = false;
        for i in span.clone() {
            let w = kernel(i, false);
            xmean += w * xs[i];
            ymean += w * ys[i];
            allweight += w;
        }
    }
    xmean /= allweight;
    ymean /= allweight;

    let mut var = 0.0;
    let mut covar = 0.0;
    for i in span {
        let w = kernel(i, robust);
        let centred = xs[i] - xmean;
        var += centred * centred * w;
        covar += centred * (ys[i] - ymean) * w;
    }
    if var == 0.0 {
        return ymean;
    }

    let slope = covar / var;
    slope * xs[curpt] + (ymean - slope * xmean)
}

/// Prior-weighted median of the absolute residuals.
///
/// Port of libscran's `compute_mad`. On an exact half-weight split it returns
/// the midpoint of the two straddling residuals.
///
/// ### Params
///
/// * `abs_dev` - Absolute residuals
/// * `ws` - Prior weights in the same order
/// * `half_weight` - Half the total prior weight
/// * `order` - Scratch buffer for the sort permutation
///
/// ### Returns
///
/// The weighted median.
fn weighted_mad(abs_dev: &[f64], ws: &[f64], half_weight: f64, order: &mut Vec<usize>) -> f64 {
    let n = abs_dev.len();
    order.clear();
    order.extend(0..n);
    order.sort_unstable_by(|&a, &b| abs_dev[a].total_cmp(&abs_dev[b]));

    let mut curweight = 0.0;
    for (i, &pt) in order.iter().enumerate() {
        curweight += ws[pt];
        if curweight == half_weight && i + 1 < n {
            let next = abs_dev[order[i + 1]];
            return abs_dev[pt] + (next - abs_dev[pt]) / 2.0;
        }
        if curweight > half_weight {
            return abs_dev[pt];
        }
    }
    0.0
}

/// Range of `y` over the points that still carry robustness weight.
///
/// ### Params
///
/// * `ys` - Responses
/// * `rw` - Robustness weights in the same order
///
/// ### Returns
///
/// `max - min` over the points with non-zero weight, or zero if there are none.
fn robust_range(ys: &[f64], rw: &[f64]) -> f64 {
    let (lo, hi) = ys
        .iter()
        .zip(rw)
        .filter(|(_, w)| **w != 0.0)
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), (&y, _)| {
            (lo.min(y), hi.max(y))
        });
    if lo > hi { 0.0 } else { hi - lo }
}

/// Runs the fit-interpolate-reweight loop over sorted data.
///
/// Port of libscran's `fit_trend`. Performs `iterations + 1` fits. The
/// robustness weights returned are the ones the final fit used.
///
/// ### Params
///
/// * `xs` - Sorted covariate values
/// * `ys` - Responses in the same order
/// * `ws` - Prior weights in the same order
/// * `seeds` - Seed indices from [`find_seeds`]
/// * `windows` - Matching windows from [`scran_window`]
/// * `total_weight` - Sum of `ws`
/// * `iterations` - Number of robustness passes
/// * `parallel` - Whether to fan the seed fits out over rayon
///
/// ### Returns
///
/// Fitted values and robustness weights, both in sorted order.
#[allow(clippy::too_many_arguments)]
fn scran_iterations(
    xs: &[f64],
    ys: &[f64],
    ws: &[f64],
    seeds: &[usize],
    windows: &[Window],
    total_weight: f64,
    iterations: usize,
    parallel: bool,
) -> (Vec<f64>, Vec<f64>) {
    let n = xs.len();
    let mut rob_w = vec![1.0; n];

    let mut min_threshold = 0.0;
    if iterations > 0 {
        let (lo, hi) = ys
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &y| {
                (lo.min(y), hi.max(y))
            });
        let range = hi - lo;
        if range == 0.0 {
            return (ys.to_vec(), rob_w);
        }
        min_threshold = range * MIN_THRESHOLD_MULTIPLIER;
    }

    let mut fitted = vec![0.0; n];
    let mut seed_fits = vec![0.0; seeds.len()];
    let mut abs_dev = vec![0.0; n];
    let mut order: Vec<usize> = Vec::with_capacity(n);
    let half_weight = total_weight / 2.0;

    let mut it = 0;
    loop {
        fill(&mut seed_fits, parallel, |s| {
            scran_fit_point(xs, ys, ws, &rob_w, seeds[s], windows[s])
        });
        for (&pt, &f) in seeds.iter().zip(&seed_fits) {
            fitted[pt] = f;
        }

        for pair in seeds.windows(2) {
            let (l, r) = (pair[0], pair[1]);
            if r - l <= 1 {
                continue;
            }
            let xdiff = xs[r] - xs[l];
            let ydiff = fitted[r] - fitted[l];
            let interior = l + 1..r;
            if xdiff > 0.0 {
                let slope = ydiff / xdiff;
                let intercept = fitted[r] - slope * xs[r];
                for (f, &xj) in fitted[interior.clone()].iter_mut().zip(&xs[interior]) {
                    *f = slope * xj + intercept;
                }
            } else {
                let ave = fitted[l] + ydiff / 2.0;
                fitted[interior].fill(ave);
            }
        }

        if it == iterations {
            break;
        }

        if it > 0 {
            // Re-derive the floor from the points still in play, so an outlier
            // that has been weighted out stops inflating it.
            let range = robust_range(ys, &rob_w);
            if range == 0.0 {
                break;
            }
            min_threshold = range * MIN_THRESHOLD_MULTIPLIER;
        }

        for ((d, &yi), &fi) in abs_dev.iter_mut().zip(ys).zip(&fitted) {
            *d = (yi - fi).abs();
        }
        let cmad = (6.0 * weighted_mad(&abs_dev, ws, half_weight, &mut order)).max(min_threshold);
        for (w, &d) in rob_w.iter_mut().zip(&abs_dev) {
            *w = if d < cmad {
                let u = d / cmad;
                (1.0 - u * u) * (1.0 - u * u)
            } else {
                0.0
            };
        }
        it += 1;
    }

    (fitted, rob_w)
}

//////////////
// Frontend //
//////////////

/// Locally weighted regression of `y` on `x`, libscran flavour.
///
/// Port of libscran's `WeightedLowess::compute`, the smoother scrapper fits its
/// mean-variance trend with. See the module documentation for how it differs
/// from [`super::lowess::weighted_lowess`]. Inputs need not be sorted:
/// `fitted[i]` and `robust_weights[i]` correspond to `x[i]` and `y[i]` in the
/// caller's original order. Prior weights act as frequency weights, so they
/// count towards the span as well as the local regressions.
///
/// ### Params
///
/// * `x` - Covariate values, in any order
/// * `y` - Responses, one per `x`
/// * `weights` - Prior weights, one per `x`, or `None` for all ones. Must be
///   non-negative.
/// * `params` - Tuning knobs, or `None` for [`ScranLowessParams::default`]
///
/// ### Returns
///
/// The fitted values and the robustness weights of the final fit, or
/// [`EdgeErrors`] if the lengths disagree, fewer than two points were
/// supplied, `span` is out of range, `minimum_width` is negative, a prior
/// weight is negative, or `npts` is zero while `delta` is left to be derived.
///
/// ### References
///
/// Cleveland, Journal of the American Statistical Association, 1979
pub fn scran_lowess(
    x: &[f64],
    y: &[f64],
    weights: Option<&[f64]>,
    params: Option<ScranLowessParams>,
) -> Result<LowessFit, EdgeErrors> {
    let params = params.unwrap_or_default();
    let n = x.len();

    if y.len() != n {
        return Err(EdgeErrors::LengthMismatch {
            name: "y",
            expected: n,
            got: y.len(),
        });
    }
    if let Some(w) = weights {
        if w.len() != n {
            return Err(EdgeErrors::LengthMismatch {
                name: "weights",
                expected: n,
                got: w.len(),
            });
        }
        crate::limma::check_nonneg_weights(w)?;
    }
    if n < 2 {
        return Err(EdgeErrors::InvalidArgument(format!(
            "scran_lowess needs at least two points; got {n}"
        )));
    }
    let span_ok = if params.span_as_proportion {
        params.span > 0.0 && params.span <= 1.0
    } else {
        params.span > 0.0
    };
    if !span_ok {
        return Err(EdgeErrors::InvalidArgument(format!(
            "span must lie in {}; got {}",
            if params.span_as_proportion {
                "(0, 1]"
            } else {
                "(0, inf)"
            },
            params.span
        )));
    }
    if params.minimum_width.is_nan() || params.minimum_width < 0.0 {
        return Err(EdgeErrors::InvalidArgument(format!(
            "minimum_width must be non-negative; got {}",
            params.minimum_width
        )));
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
    let xs: Vec<f64> = order.iter().map(|&i| x[i]).collect();
    let ys: Vec<f64> = order.iter().map(|&i| y[i]).collect();
    let ws: Vec<f64> = match weights {
        Some(w) => order.iter().map(|&i| w[i]).collect(),
        None => vec![1.0; n],
    };

    // libscran treats a negative delta as unset.
    let delta = match params.delta {
        Some(d) if d >= 0.0 => d,
        _ => resolve_delta(&xs, params.npts)?,
    };

    let total_weight: f64 = ws.iter().sum();
    let span_weight = if params.span_as_proportion {
        params.span * total_weight
    } else {
        params.span
    };
    let seeds = find_seeds(&xs, delta);
    let parallel = seeds.len().saturating_mul(n) >= PARALLEL_WORK_THRESHOLD;

    let mut windows = vec![
        Window {
            start: 0,
            end: 0,
            dist: 0.0,
        };
        seeds.len()
    ];
    fill(&mut windows, parallel, |s| {
        scran_window(&xs, &ws, seeds[s], span_weight, params.minimum_width)
    });

    let (fitted_sorted, rob_sorted) = scran_iterations(
        &xs,
        &ys,
        &ws,
        &seeds,
        &windows,
        total_weight,
        params.iterations,
        parallel,
    );

    let mut fitted = vec![0.0; n];
    let mut robust_weights = vec![0.0; n];
    for (k, &i) in order.iter().enumerate() {
        fitted[i] = fitted_sorted[k];
        robust_weights[i] = rob_sorted[k];
    }

    Ok(LowessFit {
        fitted,
        robust_weights,
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    /// Relative tolerance against scrapper. Same arithmetic, so only summation
    /// order separates the two.
    const TOL: f64 = 1e-12;

    /// The 40-point fixture: `x = ((j * 37) %% 41) / 10`, unsorted and unique,
    /// `y` a parabola plus deterministic jitter. The generator rebuilds it the
    /// same way, so no float input crosses as text.
    fn fixture() -> (Vec<f64>, Vec<f64>) {
        (1..=40u64)
            .map(|j| {
                let x = ((j * 37) % 41) as f64 / 10.0;
                let y = 0.1 * x * x - 0.5 * x + ((j * j * 13) % 97) as f64 / 300.0;
                (x, y)
            })
            .unzip()
    }

    /// 40 points on 8 tied `x` values, five apiece.
    fn ties_fixture() -> (Vec<f64>, Vec<f64>) {
        (1..=40u64)
            .map(|j| {
                let x = ((j - 1) / 5) as f64 / 8.0;
                let y = 2.0 * x + (((j * j * 13) % 97) as f64 / 300.0 - 0.16);
                (x, y)
            })
            .unzip()
    }

    /// 500 points, enough that the default 200 seeds leave most of them to the
    /// interpolation.
    fn anchors_fixture() -> (Vec<f64>, Vec<f64>) {
        (1..=500u64)
            .map(|j| {
                let x = ((j * 263) % 509) as f64 / 50.0;
                let y = 0.1 * x * x - 0.5 * x + ((j * j * 13) % 97) as f64 / 300.0;
                (x, y)
            })
            .unzip()
    }

    /// Asserts two slices agree elementwise, naming the offending index.
    fn assert_close(got: &[f64], want: &[f64]) {
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() <= TOL * w.abs().max(g.abs()).max(1.0),
                "element {i}: got {g}, want {w}"
            );
        }
    }

    // Generated against scrapper 1.2.1. `fit` is
    // `fitVarianceTrend(x, y, mean.filter = FALSE, transform = FALSE, ...)$fitted`,
    // which hands `x` and `y` straight to WeightedLowess.
    //
    //   base <- function(n, a, m) { j <- 1:n; x <- ((j*a) %% m)/10;
    //     y <- 0.1*x*x - 0.5*x + ((j*j*13) %% 97)/300; list(x=x, y=y) }
    //   SPAN03    fit(base(40, 37, 41))
    //   SPAN05    fit(base(40, 37, 41), span = 0.5)
    //   MINWIDTH  fit(base(40, 37, 41), use.min.width = TRUE, min.width = 1,
    //                 min.window.count = 5)
    //   OUTLIER   base(40, 37, 41) with y[20] + 5, default fit
    //   TIES      j <- 1:40; x <- floor((j-1)/5)/8;
    //             y <- 2*x + (((j*j*13) %% 97)/300 - 0.16)
    //   ANCHORS   j <- 1:500; x <- ((j*263) %% 509)/50, y as in base;
    //             fit(...)[seq(1, 500, by = 10)]

    #[rustfmt::skip]
    const SPAN03: [f64; 40] = [
        -0.37048934459774413, -0.36738900429725146, -0.4302885735931157,
        -0.5139215428116822, -0.4613488766229576, -0.3652917052511622,
        -0.29851169170050595, -0.18453507599659136, -0.025659004066628993,
        0.10634011689080113, -0.3656393108760705, -0.3690932507894886,
        -0.40481049544327347, -0.5053259069811267, -0.48353330483415147,
        -0.3904889673573016, -0.31614854759303007, -0.21748149560932123,
        -0.06011516204751732, 0.07360517455963696, -0.3599013700482884,
        -0.3758314081852323, -0.38408353660652267, -0.48689303546831136,
        -0.501565301641106, -0.4152629018884954, -0.3300870539166255,
        -0.246422136116648, -0.10301717241633994, 0.04078003538655426,
        -0.35257238721173834, -0.3746092614340269, -0.3711953460335296,
        -0.46012961962892385, -0.5120283059261183, -0.4385659981916277,
        -0.34474974202877245, -0.2749279762464706, -0.14612911930921377,
        0.0078069109159408494,
    ];

    #[rustfmt::skip]
    const SPAN05: [f64; 40] = [
        -0.34653184081602034, -0.3915830501018503, -0.434507146424833,
        -0.46702775267090624, -0.44696653310995255, -0.3746768491654038,
        -0.2832667935582989, -0.1663114029134699, -0.03262761013532847,
        0.10593774414526602, -0.3366087898674316, -0.3797767048805584,
        -0.42349781539730547, -0.4618252443032201, -0.45956396906169145,
        -0.39382908351696333, -0.3090458871965266, -0.1992479283465557,
        -0.06658217919308287, 0.07099654789229873, -0.3271448713568338,
        -0.3681802826552677, -0.41237302995989106, -0.45418291999897364,
        -0.46725025540990806, -0.4128501379714132, -0.3327864388965311,
        -0.22847311390935615, -0.10018107997564427, 0.03621896422942752,
        -0.3179332499512702, -0.3570518129900488, -0.40293630381687645,
        -0.4449702321755983, -0.4693723391109186, -0.4308726626848552,
        -0.35458906049935934, -0.2564212450442628, -0.13339933678085353,
        0.001661870501674493,
    ];

    #[rustfmt::skip]
    const MINWIDTH: [f64; 40] = [
        -0.37241184188221405, -0.35851656115426445, -0.4264247317434183,
        -0.5244374651323725, -0.4621384021194385, -0.36396825615736106,
        -0.30247565494149936, -0.18860172882339127, -0.022334212663937703,
        0.10124891727273447, -0.34556127173359047, -0.3668739509367521,
        -0.39543300471634885, -0.5170903961140638, -0.4852530907119797,
        -0.3892572054622052, -0.3118384654220606, -0.22515076953692173,
        -0.0562147353525723, 0.072237926724065, -0.2969889101623592,
        -0.382166114828749, -0.3759858826409881, -0.5000630351086105,
        -0.5089233448481549, -0.41691256131241416, -0.32443229302431714,
        -0.2570560308909309, -0.0967891276030946, 0.04096738037866923,
        -0.20919607105726934, -0.3871019620705632, -0.36361249193798284,
        -0.4668546719517906, -0.5244074617436006, -0.43971375398035484,
        -0.3431676641447329, -0.2843184636418529, -0.14343616611461762,
        0.008283893712701221,
    ];

    #[rustfmt::skip]
    const OUTLIER: [f64; 40] = [
        -0.36614307906437016, -0.36666918371591517, -0.42959156676086685,
        -0.5127138481897803, -0.4609561958767805, -0.36535607458215086,
        -0.29927622611035765, -0.18516587445517335, -0.02902502968990986,
        0.09377074678924927, -0.35957898551017453, -0.3682945329367195,
        -0.40409791162680136, -0.5041622251925605, -0.4830844610656757,
        -0.39052476939934483, -0.316753422450847, -0.21829189453143993,
        -0.060999604618679655, 0.06350859927018183, -0.35211489068396884,
        -0.37481064351083593, -0.3834791264703514, -0.4859831657963208,
        -0.500987258500204, -0.4151633273361019, -0.3304623406359828,
        -0.2472961477291814, -0.10304801293824596, 0.03296057229185959,
        -0.3429702629349591, -0.372001066849515, -0.37061310855318147,
        -0.45947112255576894, -0.5110840637024572, -0.4382998862976809,
        -0.3448859808767391, -0.2757867307853341, -0.1464992066131303,
        0.0021710792689871123,
    ];

    #[rustfmt::skip]
    const TIES: [f64; 40] = [
        -0.07436276517411594, -0.07436276517411594, -0.07436276517411594,
        -0.07436276517411594, -0.07436276517411594, 0.29864705494139104,
        0.29864705494139104, 0.29864705494139104, 0.29864705494139104,
        0.29864705494139104, 0.4353974617425126, 0.4353974617425126,
        0.4353974617425126, 0.4353974617425126, 0.4353974617425126,
        0.7467673690473674, 0.7467673690473674, 0.7467673690473674,
        0.7467673690473674, 0.7467673690473674, 1.058504960817956,
        1.058504960817956, 1.058504960817956, 1.058504960817956,
        1.058504960817956, 1.2877774449988402, 1.2877774449988402,
        1.2877774449988402, 1.2877774449988402, 1.2877774449988402,
        1.5710302550894324, 1.5710302550894324, 1.5710302550894324,
        1.5710302550894324, 1.5710302550894324, 1.7795260528315602,
        1.7795260528315602, 1.7795260528315602, 1.7795260528315602,
        1.7795260528315602,
    ];

    #[rustfmt::skip]
    const ANCHORS_EVERY_10TH: [f64; 50] = [
        0.3407421744144293, 1.555293759233642, 3.3600203883130706,
        0.025996805355744355, -0.3817705410709116, -0.30301230079203123,
        0.3518481280777753, 1.5730144036184535, 3.385176372770055,
        0.02034156086216459, -0.3840686837692054, -0.29863556481857567,
        0.36295408174112065, 1.5907350480032647, 3.410829146475189,
        0.014722453178870934, -0.3863668264674992, -0.29425882884512,
        0.3740600354044663, 1.6087205140449423, 3.4364819201803254,
        0.009103345495577276, -0.388664969165793, -0.28959907528804385,
        0.3853542108559675, 1.6267059800866208, 3.4621346938854605,
        0.003484237812283614, -0.39078326132428437, -0.28493932173096764,
        0.3966483863074692, 1.6446914461282987, 3.487908489415362,
        -0.0020989599602828676, -0.39290155348277567, -0.2802795681738914,
        0.4079425617589705, 1.6629426468582855, 3.5136822849452636,
        -0.00768215773284934, -0.395019845641267, -0.27533708907221494,
        0.41915187082115024, 1.6811938475882713, 3.5394560804751674,
        -0.013265355505415834, -0.39694448358942663, -0.27039460997053855,
        0.43036117988332945, 1.6994450483182577,
    ];

    #[test]
    fn test_matches_scrapper_at_span_03() {
        let (x, y) = fixture();
        let fit = scran_lowess(&x, &y, None, None).unwrap();
        assert_close(&fit.fitted, &SPAN03);
    }

    #[test]
    fn test_matches_scrapper_at_span_05() {
        let (x, y) = fixture();
        let params = ScranLowessParams {
            span: 0.5,
            ..Default::default()
        };
        let fit = scran_lowess(&x, &y, None, Some(params)).unwrap();
        assert_close(&fit.fitted, &SPAN05);
    }

    #[test]
    fn test_matches_scrapper_with_count_span_and_minimum_width() {
        let (x, y) = fixture();
        let params = ScranLowessParams {
            span: 5.0,
            span_as_proportion: false,
            minimum_width: 1.0,
            ..Default::default()
        };
        let fit = scran_lowess(&x, &y, None, Some(params)).unwrap();
        assert_close(&fit.fitted, &MINWIDTH);
    }

    #[test]
    fn test_matches_scrapper_with_an_outlier() {
        let (x, mut y) = fixture();
        y[19] += 5.0;
        let fit = scran_lowess(&x, &y, None, None).unwrap();
        assert_close(&fit.fitted, &OUTLIER);
        assert_eq!(fit.robust_weights[19], 0.0);
    }

    #[test]
    fn test_matches_scrapper_with_ties() {
        let (x, y) = ties_fixture();
        let fit = scran_lowess(&x, &y, None, None).unwrap();
        assert_close(&fit.fitted, &TIES);
    }

    #[test]
    fn test_matches_scrapper_with_seed_interpolation() {
        let (x, y) = anchors_fixture();
        let fit = scran_lowess(&x, &y, None, None).unwrap();
        let every_10th: Vec<f64> = fit.fitted.iter().step_by(10).copied().collect();
        assert_close(&every_10th, &ANCHORS_EVERY_10TH);
    }

    #[test]
    fn test_minimum_width_changes_the_fit() {
        let (x, y) = fixture();
        let narrow = ScranLowessParams {
            span: 5.0,
            span_as_proportion: false,
            ..Default::default()
        };
        let wide = ScranLowessParams {
            minimum_width: 2.0,
            ..narrow
        };
        let a = scran_lowess(&x, &y, None, Some(narrow)).unwrap();
        let b = scran_lowess(&x, &y, None, Some(wide)).unwrap();
        assert!(a.fitted.iter().zip(&b.fitted).any(|(p, q)| p != q));
    }

    #[test]
    fn test_constant_y_is_returned_as_is() {
        let x: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let y = vec![2.5; 10];
        let fit = scran_lowess(&x, &y, None, None).unwrap();
        assert_eq!(fit.fitted, y);
        assert!(fit.robust_weights.iter().all(|&w| w == 1.0));
    }

    #[test]
    fn test_all_x_identical_gives_the_mean() {
        let x = vec![1.0; 10];
        let y: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let params = ScranLowessParams {
            iterations: 0,
            ..Default::default()
        };
        let fit = scran_lowess(&x, &y, None, Some(params)).unwrap();
        assert!(fit.fitted.iter().all(|&f| (f - 5.5).abs() < 1e-12));
    }

    #[test]
    fn test_unsorted_input_matches_sorted() {
        let (x, y) = fixture();
        let mut order: Vec<usize> = (0..x.len()).collect();
        order.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
        let xs: Vec<f64> = order.iter().map(|&i| x[i]).collect();
        let ys: Vec<f64> = order.iter().map(|&i| y[i]).collect();
        let unsorted = scran_lowess(&x, &y, None, None).unwrap();
        let sorted = scran_lowess(&xs, &ys, None, None).unwrap();
        for (k, &i) in order.iter().enumerate() {
            assert_eq!(unsorted.fitted[i], sorted.fitted[k]);
        }
    }

    #[test]
    fn test_count_span_above_one_is_accepted() {
        let (x, y) = fixture();
        let params = ScranLowessParams {
            span: 20.0,
            span_as_proportion: false,
            ..Default::default()
        };
        assert!(scran_lowess(&x, &y, None, Some(params)).is_ok());
    }

    #[test]
    fn test_rejects_bad_arguments() {
        let (x, y) = fixture();
        assert!(scran_lowess(&x, &y[..5], None, None).is_err());
        assert!(scran_lowess(&x[..1], &y[..1], None, None).is_err());
        let bad = [
            ScranLowessParams {
                span: 1.5,
                ..Default::default()
            },
            ScranLowessParams {
                span: 0.0,
                span_as_proportion: false,
                ..Default::default()
            },
            ScranLowessParams {
                minimum_width: -1.0,
                ..Default::default()
            },
            ScranLowessParams {
                npts: 0,
                ..Default::default()
            },
        ];
        for params in bad {
            assert!(scran_lowess(&x, &y, None, Some(params)).is_err());
        }
        let w = vec![-1.0; x.len()];
        assert!(scran_lowess(&x, &y, Some(&w), None).is_err());
    }
}
