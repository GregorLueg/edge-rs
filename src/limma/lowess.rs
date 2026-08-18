//! limma's `weightedLowess`.
//!
//! Locally weighted linear regression with prior weights, ported from limma's
//! `src/weighted_lowess.c` and its R wrapper. This is the trend smoother behind
//! the quasi-likelihood prior and voom's mean-variance curve, so every QL
//! analysis runs through it.
//!
//! The shape of the algorithm, and the reason it is cheap:
//!
//! 1. sort by `x`
//! 2. pick roughly `npts` seed points, spaced at least `delta` apart
//! 3. grow a window around each seed until it holds `span` of the total weight
//! 4. fit a tricube-weighted line at the seed and evaluate it there
//! 5. linearly interpolate the non-seed points between consecutive seeds
//! 6. reweight by the bisquare of the scaled absolute residual and repeat
//!
//! Step 2 is the whole trick: a full local fit at every point is `O(n^2)`, while
//! fitting at ~200 seeds and interpolating is `O(npts * span * n)`. limma
//! defaults to that, and so does this, which means the fit is an approximation
//! to the exact loess by construction. Pass `delta = Some(0.0)` for the exact
//! version.
//!
//! ### Parallelism
//!
//! Seeds are the parallel axis: both the window search and the local fits fan
//! out over rayon once the work crosses `PARALLEL_WORK_THRESHOLD`, since each
//! seed is independent of the rest. The interpolation and the robustness pass
//! stay sequential; both are single passes over `n` with loop-carried state and
//! neither is where the time goes.
//!
//! ### References
//!
//! Cleveland, Journal of the American Statistical Association, 1979

use rayon::prelude::*;

use crate::errors::EdgeErrors;

/// Degeneracy threshold lifted verbatim from limma's `weighted_lowess.c`.
///
/// Guards three separate comparisons: a window whose points are effectively
/// coincident (fall back to a weighted mean), a local regression with no spread
/// in `x` (fall back to the weighted mean of `y`), and the residual scale below
/// which the robustness loop stops early. limma uses `1e-7` for all three, and
/// the fits only agree to the last digit if this matches.
const THRESHOLD: f64 = 1e-7;

/// Work below which the fan-out over seed points stays sequential.
///
/// One iteration costs roughly `n_seeds * span * n` weighted sums, and
/// `n_seeds * n` is the part of that a caller can vary. With the default
/// `npts = 200` on a few hundred points the whole pass is a few thousand flops
/// and rayon's fork costs more than it saves; a 200-seed fit over 20 000 genes,
/// which is the voom case, is worth splitting.
const PARALLEL_WORK_THRESHOLD: usize = 1 << 16;

/////////////////
// Public API  //
/////////////////

/// Tuning knobs for [`weighted_lowess`].
///
/// The defaults are limma's: `span = 0.3`, four passes, and 200 seed points.
#[derive(Clone, Copy, Debug)]
pub struct LowessParams {
    /// Proportion of the total prior weight held inside each local window.
    /// Must lie in `(0, 1]`.
    pub span: f64,
    /// Total number of fitting passes. One means no robustness iteration; the
    /// robustness weights are still computed and returned.
    pub iterations: usize,
    /// Approximate number of seed points to fit at. Only consulted when
    /// `delta` is `None`.
    pub npts: usize,
    /// Minimum distance in `x` between seed points. `None` derives it from
    /// `npts` and the spacing of `x`, which is what the R wrapper does.
    pub delta: Option<f64>,
}

impl LowessParams {
    /// Builds a parameter set.
    ///
    /// ### Params
    ///
    /// * `span` - Proportion of total weight per window, in `(0, 1]`
    /// * `iterations` - Number of fitting passes, at least one
    /// * `npts` - Approximate seed count, used only when `delta` is `None`
    /// * `delta` - Minimum seed spacing, or `None` to derive it from `npts`
    ///
    /// ### Returns
    ///
    /// The parameter set. Nothing is validated here; [`weighted_lowess`]
    /// checks the values against the data it is given.
    pub fn new(span: f64, iterations: usize, npts: usize, delta: Option<f64>) -> Self {
        Self {
            span,
            iterations,
            npts,
            delta,
        }
    }
}

impl Default for LowessParams {
    fn default() -> Self {
        Self {
            span: 0.3,
            iterations: 4,
            npts: 200,
            delta: None,
        }
    }
}

/// What [`weighted_lowess`] produces.
///
/// Both vectors are in the caller's original order, not sorted order.
#[derive(Clone, Debug)]
pub struct LowessFit {
    /// Fitted value for each input point.
    pub fitted: Vec<f64>,
    /// Bisquare robustness weight for each input point, in `[0, 1]`.
    pub robust_weights: Vec<f64>,
}

/// The span window around one seed point, as indices into the sorted data.
#[derive(Clone, Copy, Debug)]
struct Window {
    /// First index inside the window, inclusive.
    start: usize,
    /// Last index inside the window, inclusive.
    end: usize,
    /// Largest distance from the seed to a point admitted while the window was
    /// growing. This is the tricube bandwidth, and it deliberately excludes the
    /// points added afterwards by the tie extension, exactly as limma does.
    max_dist: f64,
}

///////////////
// Internals //
///////////////

/// Fills `out` by evaluating `f` at every index, optionally over rayon.
///
/// The two seed loops differ only in what they compute, so the dispatch on
/// [`PARALLEL_WORK_THRESHOLD`] lives here rather than being written out twice.
///
/// ### Params
///
/// * `out` - Slice to fill; its length sets the index range
/// * `parallel` - Whether to fan out over rayon
/// * `f` - Value for index `i`
fn fill<T, F>(out: &mut [T], parallel: bool, f: F)
where
    T: Send,
    F: Fn(usize) -> T + Send + Sync,
{
    if parallel {
        out.par_iter_mut().enumerate().for_each(|(i, o)| *o = f(i));
    } else {
        out.iter_mut().enumerate().for_each(|(i, o)| *o = f(i));
    }
}

/// Derives the seed spacing `delta` from a target number of seed points.
///
/// limma's R wrapper asks: if the points were to be grouped into `npts - k`
/// clusters, and the `k` widest gaps were the ones separating them, how wide
/// would the average remaining gap be? The answer for cluster count `k` is
/// `cumsum(sort(diff(x)))[n - 1 - k] / (npts - k)`, and `delta` is the smallest
/// such width over `k = 0..npts`. Taking the minimum is what keeps the seeds
/// from collapsing onto a dense patch of `x`.
///
/// ### Params
///
/// * `xs` - Sorted covariate values
/// * `npts` - Target number of seed points
///
/// ### Returns
///
/// The spacing, or zero when `npts` already covers every point so that no
/// binning is needed. Errors with [`EdgeErrors::MustBePositive`] if `npts` is
/// zero, matching the R wrapper's own check.
fn resolve_delta(xs: &[f64], npts: usize) -> Result<f64, EdgeErrors> {
    if npts == 0 {
        return Err(EdgeErrors::MustBePositive("npts".to_string()));
    }
    if npts >= xs.len() {
        return Ok(0.0);
    }

    // Gap widths, sorted then accumulated in place into a cumulative range.
    let mut cumrange: Vec<f64> = xs.windows(2).map(|w| w[1] - w[0]).collect();
    cumrange.sort_by(f64::total_cmp);
    let mut acc = 0.0;
    for gap in cumrange.iter_mut() {
        acc += *gap;
        *gap = acc;
    }

    // `npts < xs.len()` above, so `npts <= cumrange.len()` and the index holds.
    let last = cumrange.len() - 1;
    Ok((0..npts)
        .map(|k| cumrange[last - k] / (npts - k) as f64)
        .fold(f64::INFINITY, f64::min))
}

/// Picks the points at which a local fit is actually performed.
///
/// The first and last points are always seeds; an interior point becomes one
/// when it sits more than `delta` beyond the last seed taken. A non-positive
/// `delta` makes every point a seed, which is the exact, `O(n^2)` fit.
///
/// ### Params
///
/// * `xs` - Sorted covariate values
/// * `delta` - Minimum spacing between seeds
///
/// ### Returns
///
/// Seed indices in increasing order, always starting at `0` and ending at
/// `xs.len() - 1`.
fn find_seeds(xs: &[f64], delta: f64) -> Vec<usize> {
    let n = xs.len();
    if delta <= 0.0 || n <= 2 {
        return (0..n).collect();
    }

    let mut seeds = vec![0usize];
    let mut last = 0usize;
    for pt in 1..n - 1 {
        if xs[pt] - xs[last] > delta {
            seeds.push(pt);
            last = pt;
        }
    }
    seeds.push(n - 1);
    seeds
}

/// Grows the local window around one seed point.
///
/// Extends one index at a time, always towards whichever neighbour is closer in
/// `x`, until the prior weight enclosed reaches `span_weight` or both ends of
/// the data are hit. The window is then widened over any run of tied `x` at
/// either edge, so that tied points are never split across the boundary. The
/// tie extension deliberately does not update the bandwidth, which is limma's
/// behaviour.
///
/// ### Params
///
/// * `xs` - Sorted covariate values
/// * `ws` - Prior weights in the same order
/// * `curpt` - Index of the seed
/// * `span_weight` - Weight the window must enclose, `span * sum(ws)`
///
/// ### Returns
///
/// The window bounds and the tricube bandwidth.
fn seed_window(xs: &[f64], ws: &[f64], curpt: usize, span_weight: f64) -> Window {
    let n = xs.len();
    let mut left = curpt;
    let mut right = curpt;
    let mut cur_w = ws[curpt];
    let mut at_start = left == 0;
    let mut at_end = right == n - 1;
    let mut max_dist = 0.0_f64;

    while cur_w < span_weight && !(at_start && at_end) {
        if at_end {
            // Loop guard rules out both ends at once, so `left > 0` here.
            left -= 1;
            cur_w += ws[left];
            at_start = left == 0;
            max_dist = max_dist.max(xs[curpt] - xs[left]);
        } else if at_start {
            right += 1;
            cur_w += ws[right];
            at_end = right == n - 1;
            max_dist = max_dist.max(xs[right] - xs[curpt]);
        } else {
            let ldist = xs[curpt] - xs[left - 1];
            let rdist = xs[right + 1] - xs[curpt];
            if ldist < rdist {
                left -= 1;
                cur_w += ws[left];
                at_start = left == 0;
                max_dist = max_dist.max(ldist);
            } else {
                right += 1;
                cur_w += ws[right];
                at_end = right == n - 1;
                max_dist = max_dist.max(rdist);
            }
        }
    }

    while left > 0 && xs[left] == xs[left - 1] {
        left -= 1;
    }
    while right < n - 1 && xs[right] == xs[right + 1] {
        right += 1;
    }

    Window {
        start: left,
        end: right,
        max_dist,
    }
}

/// Evaluates the tricube-weighted local line at one seed.
///
/// Two degenerate paths, both limma's: a bandwidth below [`THRESHOLD`] means
/// the window has collapsed onto one `x` value and the answer is the weighted
/// mean of `y`; a window whose weighted variance in `x` is below [`THRESHOLD`]
/// cannot support a slope, so the intercept alone is returned. A window with no
/// weight at all returns zero rather than a NaN, which is how limma keeps a
/// fully downweighted neighbourhood from poisoning the interpolation.
///
/// ### Params
///
/// * `xs` - Sorted covariate values
/// * `ys` - Responses in the same order
/// * `ws` - Prior weights in the same order
/// * `rw` - Current robustness weights in the same order
/// * `curpt` - Index of the seed being fitted
/// * `window` - Window and bandwidth from [`seed_window`]
///
/// ### Returns
///
/// The fitted value at `xs[curpt]`.
fn lowess_fit(xs: &[f64], ys: &[f64], ws: &[f64], rw: &[f64], curpt: usize, window: Window) -> f64 {
    let span = window.start..=window.end;

    if window.max_dist < THRESHOLD {
        let allweight: f64 = span.clone().map(|i| ws[i] * rw[i]).sum();
        if allweight == 0.0 {
            return 0.0;
        }
        let val: f64 = span.map(|i| ys[i] * ws[i] * rw[i]).sum();
        return val / allweight;
    }

    let kernel = |i: usize| {
        let u = (xs[curpt] - xs[i]).abs() / window.max_dist;
        let t = 1.0 - u * u * u;
        t * t * t * ws[i] * rw[i]
    };

    let mut allweight = 0.0;
    let mut xmean = 0.0;
    let mut ymean = 0.0;
    for i in span.clone() {
        let w = kernel(i);
        allweight += w;
        xmean += w * xs[i];
        ymean += w * ys[i];
    }
    if allweight == 0.0 {
        return 0.0;
    }
    xmean /= allweight;
    ymean /= allweight;

    let mut var = 0.0;
    let mut covar = 0.0;
    for i in span {
        let w = kernel(i);
        let centred = xs[i] - xmean;
        var += centred * centred * w;
        covar += centred * (ys[i] - ymean) * w;
    }
    if var < THRESHOLD {
        return ymean;
    }

    let slope = covar / var;
    slope * xs[curpt] + ymean - slope * xmean
}

/// Runs the fit-interpolate-reweight loop over sorted data.
///
/// Each pass fits every seed against the current robustness weights, fills the
/// gaps between consecutive seeds by linear interpolation, then rescales the
/// absolute residuals by six times their weighted median and squares the
/// bisquare of that. The loop exits early once that scale collapses relative to
/// the mean absolute residual, which is what makes an exact fit return weights
/// of one rather than zero.
///
/// Note that the robustness weights are recomputed after the final fit, so they
/// describe the returned fit rather than the one before it. limma does the same,
/// which is why `iterations = 1` still returns non-trivial weights.
///
/// ### Params
///
/// * `xs` - Sorted covariate values
/// * `ys` - Responses in the same order
/// * `ws` - Prior weights in the same order
/// * `seeds` - Seed indices from [`find_seeds`]
/// * `windows` - Matching windows from [`seed_window`]
/// * `total_weight` - Sum of `ws`
/// * `iterations` - Number of fitting passes
/// * `parallel` - Whether to fan the seed fits out over rayon
///
/// ### Returns
///
/// Fitted values and robustness weights, both in sorted order.
#[allow(clippy::too_many_arguments)]
fn lowess_iterations(
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
    let mut fitted = vec![0.0; n];
    let mut rob_w = vec![1.0; n];
    let mut seed_fits = vec![0.0; seeds.len()];
    let mut abs_resid = vec![0.0; n];
    let mut order: Vec<usize> = Vec::with_capacity(n);

    // Scale for the "are two seeds at the same x" test in the interpolation.
    let subrange = (xs[n - 1] - xs[0]) / n as f64;
    let half_weight = total_weight / 2.0;

    for _ in 0..iterations {
        fill(&mut seed_fits, parallel, |s| {
            lowess_fit(xs, ys, ws, &rob_w, seeds[s], windows[s])
        });

        // Scatter the seed fits back and interpolate the points between them.
        // `find_seeds` guarantees `seeds[0] == 0`.
        fitted[0] = seed_fits[0];
        let mut last_pt = 0usize;
        for (s, &pt) in seeds.iter().enumerate().skip(1) {
            fitted[pt] = seed_fits[s];
            if pt - last_pt > 1 {
                let gap = xs[pt] - xs[last_pt];
                let interior = last_pt + 1..pt;
                if gap > THRESHOLD * subrange {
                    let slope = (fitted[pt] - fitted[last_pt]) / gap;
                    let intercept = fitted[pt] - slope * xs[pt];
                    for (f, &xj) in fitted[interior.clone()].iter_mut().zip(&xs[interior]) {
                        *f = slope * xj + intercept;
                    }
                } else {
                    let avg = 0.5 * (fitted[pt] + fitted[last_pt]);
                    fitted[interior].fill(avg);
                }
            }
            last_pt = pt;
        }

        for ((r, &yi), &fi) in abs_resid.iter_mut().zip(ys).zip(fitted.iter()) {
            *r = (yi - fi).abs();
        }
        let resid_scale = abs_resid.iter().sum::<f64>() / n as f64;

        order.clear();
        order.extend(0..n);
        order.sort_by(|&a, &b| abs_resid[a].total_cmp(&abs_resid[b]));

        // Six times the prior-weighted median absolute residual, with limma's
        // exact-tie interpolation between the two straddling residuals.
        let mut cumw = 0.0;
        let mut cmad = 0.0;
        for (i, &idx) in order.iter().enumerate() {
            cumw += ws[idx];
            if cumw == half_weight && i + 1 < n {
                cmad = 3.0 * (abs_resid[idx] + abs_resid[order[i + 1]]);
                break;
            }
            if cumw > half_weight {
                cmad = 6.0 * abs_resid[idx];
                break;
            }
        }
        if cmad <= THRESHOLD * resid_scale {
            break;
        }

        // Residuals at or beyond `cmad` get zero; `order` is ascending, so the
        // first one that qualifies ends the scan.
        rob_w.fill(0.0);
        for &idx in &order {
            if abs_resid[idx] >= cmad {
                break;
            }
            let u = abs_resid[idx] / cmad;
            rob_w[idx] = (1.0 - u * u) * (1.0 - u * u);
        }
    }

    (fitted, rob_w)
}

/// Locally weighted regression of `y` on `x` with prior weights.
///
/// Port of limma's `weightedLowess`. Inputs need not be sorted: the data is
/// sorted by `x` internally and the results are mapped back, so `fitted[i]` and
/// `robust_weights[i]` correspond to `x[i]` and `y[i]` **in the caller's
/// original order**. Ties in `x` keep their input order and are never split
/// across a window boundary.
///
/// The fit is an approximation by default. Local regressions are performed only
/// at seed points spaced `delta` apart and the rest is linear interpolation; see
/// the module documentation. Set `params.delta = Some(0.0)` to fit at every
/// point instead.
///
/// ### Params
///
/// * `x` - Covariate values, in any order
/// * `y` - Responses, one per `x`
/// * `weights` - Prior weights, one per `x`, or `None` for all ones. Must be
///   non-negative.
/// * `params` - Tuning knobs, or `None` for [`LowessParams::default`]
///
/// ### Returns
///
/// The fitted values and bisquare robustness weights, or [`EdgeErrors`] if the
/// lengths disagree, fewer than two points were supplied, `span` falls outside
/// `(0, 1]`, `iterations` is zero, a prior weight is negative, or `npts` is zero
/// while `delta` is left to be derived.
///
/// ### References
///
/// Cleveland, Journal of the American Statistical Association, 1979
pub fn weighted_lowess(
    x: &[f64],
    y: &[f64],
    weights: Option<&[f64]>,
    params: Option<LowessParams>,
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
        if let Some(i) = w.iter().position(|v| *v < 0.0) {
            return Err(EdgeErrors::InvalidArgument(format!(
                "prior weights must be non-negative; weights[{i}] is {}",
                w[i]
            )));
        }
    }
    if n < 2 {
        return Err(EdgeErrors::InvalidArgument(format!(
            "weighted_lowess needs at least two points; got {n}"
        )));
    }
    if !(params.span > 0.0 && params.span <= 1.0) {
        return Err(EdgeErrors::InvalidArgument(format!(
            "span must lie in (0, 1]; got {}",
            params.span
        )));
    }
    if params.iterations == 0 {
        return Err(EdgeErrors::MustBePositive("iterations".to_string()));
    }

    // Stable sort by x, matching R's `order`, so tied x keep their input order.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
    let xs: Vec<f64> = order.iter().map(|&i| x[i]).collect();
    let ys: Vec<f64> = order.iter().map(|&i| y[i]).collect();
    let ws: Vec<f64> = match weights {
        Some(w) => order.iter().map(|&i| w[i]).collect(),
        None => vec![1.0; n],
    };

    let delta = match params.delta {
        Some(d) => d,
        None => resolve_delta(&xs, params.npts)?,
    };

    let total_weight: f64 = ws.iter().sum();
    let span_weight = total_weight * params.span;
    let seeds = find_seeds(&xs, delta);
    let parallel = seeds.len().saturating_mul(n) >= PARALLEL_WORK_THRESHOLD;

    let mut windows = vec![
        Window {
            start: 0,
            end: 0,
            max_dist: 0.0,
        };
        seeds.len()
    ];
    fill(&mut windows, parallel, |s| {
        seed_window(&xs, &ws, seeds[s], span_weight)
    });

    let (fitted_sorted, rob_sorted) = lowess_iterations(
        &xs,
        &ys,
        &ws,
        &seeds,
        &windows,
        total_weight,
        params.iterations,
        parallel,
    );

    // Back to the caller's order.
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

////////////////////////////
// Cleveland's lowess     //
////////////////////////////

/// Fraction of the covariate range used as the lowess interpolation cell width,
/// which is `stats::lowess`'s `delta = 0.01 * diff(range(x))` default.
const LOWESS_DELTA_FRACTION: f64 = 0.01;

/// Fitted values of R's `stats::lowess`, in the caller's original order.
///
/// Distinct from [`weighted_lowess`] in this same module, which is limma's own
/// `weightedLowess`. Both are needed: limma's `loessFit` returns early to this
/// one whenever it has no prior weights, and edgeR's `compute_ave_qd` uses this
/// one too. Reaching for the wrong one costs up to 7e-4 on the quasi-likelihood
/// prior.
///
/// limma's `fitFDistRobustly` smooths the log variances with
/// `loessFit(z, covariate, span = 0.4)`, and `loessFit` with no prior weights is
/// a straight call to `stats::lowess`. That is Cleveland's original lowess, not
/// limma's own `weightedLowess` in [`crate::limma::lowess`]: the two differ in
/// how the local window is chosen and in the robustness weighting, so the trend
/// only matches limma's if this one is used.
///
/// ### Params
///
/// * `x` - Covariate, in any order
/// * `y` - Response, one per `x`
/// * `span` - Fraction of the points in each local window, in `(0, 1]`
/// * `n_steps` - Robustness iterations after the first fit, R's `lowess(iter = )`
///
/// ### Returns
///
/// One fitted value per input point, in the input order.
///
/// ### References
///
/// Cleveland, Journal of the American Statistical Association, 1979
pub fn lowess(x: &[f64], y: &[f64], span: f64, n_steps: usize) -> Result<Vec<f64>, EdgeErrors> {
    let n = x.len();
    if n < 2 {
        return Err(EdgeErrors::InvalidArgument(
            "lowess needs at least two points.".to_string(),
        ));
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
    let xs: Vec<f64> = order.iter().map(|&i| x[i]).collect();
    let ys: Vec<f64> = order.iter().map(|&i| y[i]).collect();
    let delta = LOWESS_DELTA_FRACTION * (xs[n - 1] - xs[0]);

    let fitted = clowess(&xs, &ys, span, n_steps, delta);
    let mut out = vec![0.0; n];
    for (k, &i) in order.iter().enumerate() {
        out[i] = fitted[k];
    }
    Ok(out)
}

/// Cleveland's lowess on data already sorted by `x`.
///
/// A local linear fit is computed at a subset of the points, spaced at least
/// `delta` apart, and the rest are linearly interpolated between them; the whole
/// thing repeats `n_steps` times with bisquare weights on the residuals. The
/// `delta` skipping is what keeps the cost near linear, and reproducing it
/// exactly is necessary because the interpolated points are genuinely different
/// from what a fit at every point would give.
///
/// ### Params
///
/// * `x` - Covariate, sorted ascending
/// * `y` - Response, in the same order
/// * `f` - Span, the fraction of points in each window
/// * `n_steps` - Robustness iterations after the first fit
/// * `delta` - Minimum spacing between points that are actually fitted
///
/// ### Returns
///
/// One fitted value per point, in sorted order.
fn clowess(x: &[f64], y: &[f64], f: f64, n_steps: usize, delta: f64) -> Vec<f64> {
    let n = x.len();
    let ns = (f * n as f64 + 1e-7) as usize;
    let ns = ns.clamp(2, n);

    let mut ys = vec![0.0; n];
    let mut rw = vec![0.0; n];
    let mut res = vec![0.0; n];
    let mut work = vec![0.0; n];

    let mut iter = 1;
    loop {
        let mut nleft = 0;
        let mut nright = ns - 1;
        let mut last: Option<usize> = None;
        let mut i = 0;

        loop {
            while nright < n - 1 {
                let d1 = x[i] - x[nleft];
                let d2 = x[nright + 1] - x[i];
                if d1 <= d2 {
                    break;
                }
                nleft += 1;
                nright += 1;
            }

            match lowest(x, y, x[i], nleft, nright, &mut work, iter > 1, &rw) {
                Some(v) => ys[i] = v,
                None => ys[i] = y[i],
            }

            if let Some(prev) = last
                && prev + 1 < i
            {
                let denom = x[i] - x[prev];
                for j in prev + 1..i {
                    let alpha = (x[j] - x[prev]) / denom;
                    ys[j] = alpha * ys[i] + (1.0 - alpha) * ys[prev];
                }
            }

            let mut last_i = i;
            let cut = x[last_i] + delta;
            i = last_i + 1;
            while i < n {
                if x[i] > cut {
                    break;
                }
                if x[i] == x[last_i] {
                    ys[i] = ys[last_i];
                    last_i = i;
                }
                i += 1;
            }
            i = (last_i + 1).max(i - 1);
            last = Some(last_i);
            if last_i >= n - 1 {
                break;
            }
        }

        for j in 0..n {
            res[j] = y[j] - ys[j];
        }
        if iter > n_steps {
            break;
        }
        let sc = res.iter().map(|v| v.abs()).sum::<f64>() / n as f64;

        // cmad is 6 * median(|res|), taken as the average of the two central
        // order statistics for an even count, exactly as the C does.
        let mut abs_res: Vec<f64> = res.iter().map(|v| v.abs()).collect();
        abs_res.sort_by(f64::total_cmp);
        let m1 = n / 2;
        let cmad = if n.is_multiple_of(2) {
            3.0 * (abs_res[m1] + abs_res[n - m1 - 1])
        } else {
            6.0 * abs_res[m1]
        };
        if cmad < 1e-7 * sc {
            break;
        }
        let c9 = 0.999 * cmad;
        let c1 = 0.001 * cmad;
        for j in 0..n {
            let r = res[j].abs();
            rw[j] = if r <= c1 {
                1.0
            } else if r <= c9 {
                let t = 1.0 - (r / cmad) * (r / cmad);
                t * t
            } else {
                0.0
            };
        }
        iter += 1;
    }
    ys
}

/// One tricube-weighted local linear fit, the `lowest` of Cleveland's code.
///
/// The window runs from `nleft` and is extended past `nright` to pick up ties in
/// `x`. When the window has no spread the fit falls back to a weighted mean,
/// which is what the `sqrt(c) > 0.001 * range` guard is for.
///
/// ### Params
///
/// * `x` - Covariate, sorted ascending
/// * `y` - Response
/// * `xs` - Point to fit at
/// * `nleft` - First index of the window
/// * `nright` - Nominal last index of the window
/// * `w` - Scratch buffer of length `n` for the local weights
/// * `use_rw` - Whether to fold in the robustness weights
/// * `rw` - Robustness weights
///
/// ### Returns
///
/// The fitted value, or `None` when every weight in the window is zero.
#[allow(clippy::too_many_arguments)]
fn lowest(
    x: &[f64],
    y: &[f64],
    xs: f64,
    nleft: usize,
    nright: usize,
    w: &mut [f64],
    use_rw: bool,
    rw: &[f64],
) -> Option<f64> {
    let n = x.len();
    let range = x[n - 1] - x[0];
    let h = (xs - x[nleft]).max(x[nright] - xs);
    let h9 = 0.999 * h;
    let h1 = 0.001 * h;

    let mut a = 0.0;
    let mut j = nleft;
    while j < n {
        w[j] = 0.0;
        let r = (x[j] - xs).abs();
        if r <= h9 {
            if r <= h1 {
                w[j] = 1.0;
            } else {
                let t = 1.0 - (r / h) * (r / h) * (r / h);
                w[j] = t * t * t;
            }
            if use_rw {
                w[j] *= rw[j];
            }
            a += w[j];
        } else if x[j] > xs {
            break;
        }
        j += 1;
    }
    if a <= 0.0 || j == nleft {
        return None;
    }
    let nrt = j - 1;

    for wj in w.iter_mut().take(nrt + 1).skip(nleft) {
        *wj /= a;
    }
    if h > 0.0 {
        let mut centre = 0.0;
        for j in nleft..=nrt {
            centre += w[j] * x[j];
        }
        let mut b = xs - centre;
        let mut c = 0.0;
        for j in nleft..=nrt {
            c += w[j] * (x[j] - centre) * (x[j] - centre);
        }
        if c.sqrt() > 0.001 * range {
            b /= c;
            for j in nleft..=nrt {
                w[j] *= b * (x[j] - centre) + 1.0;
            }
        }
    }
    Some((nleft..=nrt).map(|j| w[j] * y[j]).sum())
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Relative tolerance against limma, applied with no absolute floor.
    ///
    /// The worst disagreement actually observed across every case below is
    /// 1.5e-15, a couple of ulps, which is what one expects when the same
    /// arithmetic is performed in a marginally different order. The tolerance
    /// is loosened to 1e-13 only so a different optimiser or libm cannot make
    /// the suite flaky.
    const TOL: f64 = 1e-13;

    /// The shared 50-point fixture, built from exactly representable arithmetic
    /// so that R and Rust construct bit-identical inputs.
    ///
    /// R: `i <- 1:50; x <- i/64; noise <- ((i*i*37) %% 101)/500 - 0.1;`
    /// `y <- 3*x - 2*x*x + noise`
    fn fixture() -> (Vec<f64>, Vec<f64>) {
        let mut x = Vec::with_capacity(50);
        let mut y = Vec::with_capacity(50);
        for i in 1..=50u64 {
            let xi = i as f64 / 64.0;
            let noise = ((i * i * 37) % 101) as f64 / 500.0 - 0.1;
            x.push(xi);
            y.push(3.0 * xi - 2.0 * xi * xi + noise);
        }
        (x, y)
    }

    /// Rscript -e 'suppressMessages(library(limma)); i <- 1:50; x <- i/64;
    /// noise <- ((i*i*37) %% 101)/500 - 0.1; y <- 3*x - 2*x*x + noise;
    /// f <- weightedLowess(x, y, span=0.3, iterations=4); cat(f$fitted, sep=",")'
    #[rustfmt::skip]
    const LIMMA_SPAN03_FITTED: [f64; 50] = [
        0.02884617638158539, 0.07488153747159955, 0.12089455051667254, 0.1668882703760661,
        0.21287802704861467, 0.2586900519509724, 0.30417000568490377, 0.3492852738819449,
        0.39492218636530535, 0.43942896960951655, 0.482281763218411, 0.5221738876113334,
        0.5595035894877518, 0.5945434168031167, 0.6263966419442515, 0.6550233456550328,
        0.680519544814015, 0.7038870095147743, 0.7259934729659039, 0.7487015519987938,
        0.7716627156900544, 0.7924464699346567, 0.8111020622726952, 0.8289963524672412,
        0.847064022565787, 0.8639718998482121, 0.8801145417766755, 0.8969965708828094,
        0.9151828298879751, 0.9361805081577184, 0.9596393749919402, 0.9853751996891378,
        1.0119634950324796, 1.03494548040943, 1.0526991144982325, 1.0668330481398405,
        1.0773335747275383, 1.082636407857981, 1.0842669833033778, 1.0854201205535288,
        1.0854713621828038, 1.0849065164984981, 1.0841589267201792, 1.086847462834226,
        1.0931490310395207, 1.1001711408196604, 1.1071684657965764, 1.1140229752752147,
        1.1209028595074588, 1.1281426144633149
    ];

    /// Same call as [`LIMMA_SPAN03_FITTED`], with `cat(f$weights, sep=",")`.
    #[rustfmt::skip]
    const LIMMA_SPAN03_WEIGHTS: [f64; 50] = [
        0.9982602643374002, 0.9971043448119014, 0.9852596495241089, 0.8251547234921383,
        0.9179245006387058, 0.922492639268141, 0.8046211529296002, 0.9941387898671831,
        0.9867246551780985, 0.998246485743474, 0.9150873028619894, 0.9882486348243016,
        0.9368926350581316, 0.9810183117941136, 0.9466952247996472, 0.9810544578382142,
        0.9371576148518962, 0.9886222046691591, 0.9097319226756649, 0.99994629822593,
        0.9978611322200113, 0.9695420857354156, 0.8884352169874018, 0.8385562169994693,
        0.7105982103724293, 0.9285523952282887, 0.9186439479817102, 0.984481673942221,
        0.9437175263360283, 0.8758623200186673, 0.8976130831462985, 0.916095205234191,
        0.8000054743875572, 0.998577680240991, 0.9539240530367198, 0.9537718891569258,
        0.9986629880337708, 0.7985348592236604, 0.917629558078525, 0.9038344465316576,
        0.8518229780122982, 0.9729163214547771, 0.9999902226249788, 0.9864314133557722,
        0.7906639674414591, 0.9552810228565608, 0.9710012331210384, 0.8467102955813733,
        0.9906701350391008, 0.9001691460681668
    ];

    /// Rscript -e 'suppressMessages(library(limma)); i <- 1:50; x <- i/64;
    /// noise <- ((i*i*37) %% 101)/500 - 0.1; y <- 3*x - 2*x*x + noise;
    /// f <- weightedLowess(x, y, span=0.5, iterations=4); cat(f$fitted, sep=",")'
    #[rustfmt::skip]
    const LIMMA_SPAN05_FITTED: [f64; 50] = [
        0.0502040025424237, 0.09233312634615892, 0.1342483975020733, 0.17593715074892777,
        0.2173941929412161, 0.2585994728892008, 0.2995500992990607, 0.3402673622654965,
        0.3807673441699271, 0.42107873858241845, 0.4612403905565409, 0.5012307957149007,
        0.5405033131563842, 0.5743811498491763, 0.6066977934466802, 0.6375483484333743,
        0.6666170789356634, 0.6938200073253665, 0.7191488661665506, 0.7426359427582109,
        0.7643724973873608, 0.784570852933584, 0.8039400789167546, 0.8232657476821141,
        0.8430109122492775, 0.8633409075726487, 0.8841713069707374, 0.9053862026363093,
        0.9262518313285918, 0.9459580947742551, 0.9643346137321928, 0.9813688120972042,
        0.9970618925672503, 1.0116025850140906, 1.0254533424627585, 1.038599196196896,
        1.0508441500506736, 1.0614617968192341, 1.0666131576221753, 1.0721665436911023,
        1.077947306787132, 1.0837200534202618, 1.089363518850383, 1.094836270014361,
        1.100135346462288, 1.1052750877434108, 1.1102609535353394, 1.1151155388982341,
        1.1198822859465956, 1.1246245137565172
    ];

    /// Rscript -e 'suppressMessages(library(limma)); i <- 1:50; x <- i/64;
    /// noise <- ((i*i*37) %% 101)/500 - 0.1; y <- 3*x - 2*x*x + noise;
    /// f <- weightedLowess(x, y, span=0.3, iterations=4, npts=10);
    /// cat(f$delta, f$fitted, sep=",")'
    /// (delta came back as 0.0765625)
    #[rustfmt::skip]
    const LIMMA_NPTS10_FITTED: [f64; 50] = [
        0.029206226861471576, 0.07516262717420194, 0.12111902748693232, 0.16707542779966267,
        0.21303182811239305, 0.2589882284251234, 0.3036152279106658, 0.3482422273962082,
        0.3928692268817505, 0.4374962263672929, 0.4821232258528353, 0.5166454429451267,
        0.5511676600374181, 0.5856898771297097, 0.6202120942220011, 0.6547343113142925,
        0.6781022198566342, 0.7014701283989758, 0.7248380369413175, 0.7482059454836592,
        0.7715738540260009, 0.7901487131659873, 0.8087235723059738, 0.8272984314459602,
        0.8458732905859466, 0.8644481497259331, 0.8834850738923911, 0.902521998058849,
        0.921558922225307, 0.940595846391765, 0.9596327705582229, 0.9809465753166268,
        1.0022603800750307, 1.0235741848334345, 1.0448879895918384, 1.0662017943502422,
        1.070087244525965, 1.073972694701688, 1.077858144877411, 1.081743595053134,
        1.0856290452288568, 1.0886159757724936, 1.0916029063161306, 1.0945898368597673,
        1.0975767674034043, 1.100563697947041, 1.1074882486565683, 1.1144127993660953,
        1.1213373500756223, 1.1282619007851495
    ];

    /// Rscript -e 'suppressMessages(library(limma)); i <- 1:50; x <- i/64;
    /// noise <- ((i*i*37) %% 101)/500 - 0.1; y <- 3*x - 2*x*x + noise;
    /// y[25] <- y[25] + 5; f <- weightedLowess(x, y, span=0.3, iterations=1);
    /// cat(f$fitted, sep=",")'
    #[rustfmt::skip]
    const LIMMA_OUTLIER_IT1_FITTED: [f64; 50] = [
        0.03106855202949773, 0.07717316010753585, 0.12320167864925471, 0.16916673546900665,
        0.2150985462638953, 0.2608203145156239, 0.30617887534209315, 0.35108940230035507,
        0.39626100980627305, 0.44022049564093946, 0.4826106654267944, 0.522124291056273,
        0.5591600817086909, 0.5940819904449881, 0.6259099012768223, 0.6545351154449882,
        0.6800663317086908, 0.703483666056273, 0.7570184160051099, 0.9073451133632405,
        1.104565465787236, 1.2763776803326319, 1.3882348740363488, 1.4436172241743854,
        1.4673023774949454, 1.4787929183621724, 1.4580138165122363, 1.3826385854919376,
        1.2504190822272683, 1.097243096892691, 0.9929513416163951, 0.9873334099826038,
        1.014038720464609, 1.0369688883832606, 1.054596070045016, 1.068803101295016,
        1.0795899821332606, 1.0850738767146089, 1.0867826287326035, 1.088219372288079,
        1.0886614479203902, 1.0882590691904972, 1.087225975598379, 1.0892594959295647,
        1.095057358064819, 1.1016213588821133, 1.108144950698158, 1.1144820987791833,
        1.120790920417942, 1.12741844513999
    ];

    /// Same call as [`LIMMA_OUTLIER_IT1_FITTED`], with `cat(f$weights, sep=",")`.
    #[rustfmt::skip]
    const LIMMA_OUTLIER_IT1_WEIGHTS: [f64; 50] = [
        0.998268858186368, 0.9988715161747013, 0.9889889302567727, 0.8945077289629702,
        0.944443098909428, 0.9476023796865143, 0.8807808270272274, 0.9954417134553554,
        0.9926249743835063, 0.9990999463188386, 0.94598382006912, 0.9926249743835063,
        0.9598480968256157, 0.987739430288326, 0.9672434919990645, 0.9877394302883262,
        0.9598480968256154, 0.9926249743835063, 0.8739173307137105, 0.6602483438534276,
        0.04229477328190964, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8847871984079918,
        0.8565883333007306, 0.9437167312008538, 0.8780026346228988, 0.9985805871945471,
        0.9735495556344034, 0.9735495556344034, 0.9985805871945471, 0.8780026346228984,
        0.9437167312008544, 0.9340629565631526, 0.913337665005293, 0.9795091411356777,
        0.9999101524908519, 0.9897090366925873, 0.8715800946658386, 0.9700694279504515,
        0.9808202909704252, 0.9017637368521244, 0.9941056390523088, 0.9356932600012816
    ];

    /// Rscript -e 'suppressMessages(library(limma)); i <- 1:50; x <- i/64;
    /// noise <- ((i*i*37) %% 101)/500 - 0.1; y <- 3*x - 2*x*x + noise;
    /// y[25] <- y[25] + 5; f <- weightedLowess(x, y, span=0.3, iterations=4);
    /// cat(f$fitted, sep=",")'
    #[rustfmt::skip]
    const LIMMA_OUTLIER_IT4_FITTED: [f64; 50] = [
        0.028818571078165194, 0.07485364691555507, 0.12086702984979603, 0.16686165584526727,
        0.21285270090851283, 0.25866658218483113, 0.3041489780622304, 0.34926742758113394,
        0.3949100647098492, 0.43942361469480534, 0.4822813105330778, 0.5221775179225341,
        0.5595084303065795, 0.5945404551019211, 0.6263859440545082, 0.6550032102822706,
        0.680472430199714, 0.7037621525438478, 0.7251819722884809, 0.7451729273177546,
        0.7643570789948314, 0.7820629679714798, 0.7987062475450487, 0.815768636359257,
        0.8339302683070688, 0.8509629696394199, 0.867702309224105, 0.8861733958319391,
        0.9072155656991749, 0.9321521313113953, 0.9588223335420013, 0.9854615355925457,
        1.0120639222433077, 1.0350480792661543, 1.0527988882251693, 1.0669061807709028,
        1.0773614027213367, 1.0826327666719797, 1.084254777658199, 1.0854021396875264,
        1.0854448903478169, 1.0848736265574255, 1.0841269823105504, 1.086823365578022,
        1.0931297417083892, 1.1001560813156253, 1.1071581636933958, 1.1140184501514818,
        1.1209053019456447, 1.1281529164517266
    ];

    /// Same call as [`LIMMA_OUTLIER_IT4_FITTED`], with `cat(f$weights, sep=",")`.
    #[rustfmt::skip]
    const LIMMA_OUTLIER_IT4_WEIGHTS: [f64; 50] = [
        0.9980424329544307, 0.9967037819397565, 0.9833496125051744, 0.8031084402231133,
        0.9073790989672791, 0.9125164643851644, 0.7802035265935129, 0.9933778040934986,
        0.9849548162484397, 0.9980114919368172, 0.904107938435841, 0.9866993953538796,
        0.9286890833076206, 0.9785095149301034, 0.9397608918427356, 0.978524211461719,
        0.9288464414464999, 0.9869693622060156, 0.9006747726482551, 0.9993074103419125,
        0.9923449406181417, 0.9826636749057164, 0.8269478977198058, 0.8695049668692422, 0.0,
        0.8776956618754143, 0.9424742591544403, 0.994227738504275, 0.9553580077609584,
        0.8447209783309053, 0.887215133599026, 0.9049724063577086, 0.7756042096289071,
        0.99834559442311, 0.948127728845395, 0.9478937494688799, 0.9984743170441658,
        0.7734944956716602, 0.9070066230141562, 0.8915026780247038, 0.8329533118039492,
        0.9694112873754656, 0.9999877816431798, 0.9846696433980241, 0.7646513792079273,
        0.9494607069874093, 0.9672059133675933, 0.8273521672854732, 0.9894384285089609,
        0.8873554694836855
    ];

    /// Rscript -e 'suppressMessages(library(limma)); i <- 1:50; x <- i/64;
    /// noise <- ((i*i*37) %% 101)/500 - 0.1; y <- 3*x - 2*x*x + noise;
    /// w <- 1 + ((i*7) %% 5);
    /// f <- weightedLowess(x, y, weights=w, span=0.3, iterations=4);
    /// cat(f$fitted, sep=",")'
    #[rustfmt::skip]
    const LIMMA_PRIORW_FITTED: [f64; 50] = [
        0.03994998581530459, 0.08711566214810951, 0.1339449147709428, 0.1804874991900608,
        0.22686129944573435, 0.2730844392403633, 0.3193274923048413, 0.36548945542441225,
        0.40847523924848167, 0.45045799490307237, 0.4916207315759508, 0.5308125516414022,
        0.5678952309933928, 0.602807093305044, 0.6343048525141035, 0.6611535646797024,
        0.6846095833651786, 0.7057973537115118, 0.7231429070959012, 0.737667462510629,
        0.7517188545410529, 0.7661944480915583, 0.7812698788270172, 0.7975669860044925,
        0.8136674814730263, 0.8286356292850393, 0.8441834398623734, 0.8617284894544015,
        0.8820610191636362, 0.9053956588303979, 0.9313272211707209, 0.957665445333203,
        0.983399379323739, 1.0067914677051422, 1.0278667081385229, 1.0454899604534134,
        1.058719418395293, 1.0677455045157376, 1.070890547557146, 1.0710636167032437,
        1.0697628314665355, 1.0687092146300567, 1.0689379209868493, 1.0728070282205489,
        1.0785077209642524, 1.0846989192740135, 1.0909804993987746, 1.097338525575952,
        1.103965367644119, 1.1111092473305215
    ];

    /// Rscript -e 'suppressMessages(library(limma)); j <- 1:40;
    /// x <- floor((j-1)/5)/8; y <- 2*x + (((j*j*13) %% 97)/300 - 0.16);
    /// f <- weightedLowess(x, y, span=0.3, iterations=4); cat(f$fitted, sep=",")'
    #[rustfmt::skip]
    const LIMMA_TIES_FITTED: [f64; 40] = [
        -0.07436276517411591, -0.07436276517411594, -0.07436276517411594,
        -0.07436276517411594, -0.07436276517411594, 0.29864705494139104, 0.29864705494139104,
        0.29864705494139104, 0.29864705494139104, 0.29864705494139104, 0.4353974617425126,
        0.4353974617425126, 0.4353974617425126, 0.4353974617425126, 0.4353974617425126,
        0.7467673690473675, 0.7467673690473675, 0.7467673690473675, 0.7467673690473675,
        0.7467673690473675, 1.058504960817956, 1.058504960817956, 1.058504960817956,
        1.058504960817956, 1.058504960817956, 1.2877774449988402, 1.2877774449988402,
        1.2877774449988402, 1.2877774449988402, 1.2877774449988402, 1.5710302550894326,
        1.5710302550894326, 1.5710302550894326, 1.5710302550894326, 1.5710302550894326,
        1.7795260528315604, 1.7795260528315604, 1.7795260528315604, 1.7795260528315604,
        1.7795260528315604
    ];

    /// Rscript -e 'suppressMessages(library(limma)); i <- 1:10;
    /// x <- rep(1, 10); y <- i*(i %% 7)/10;
    /// f <- weightedLowess(x, y, span=0.3, iterations=4); cat(f$fitted, sep=",")'
    /// (every fitted value is this one number)
    const LIMMA_FLAT_X_FITTED: f64 = 1.4297213751835516;

    /// Asserts two slices agree elementwise, naming the offending index.
    fn assert_close(got: &[f64], want: &[f64]) {
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() <= TOL * w.abs().max(g.abs()),
                "element {i}: got {g}, want {w}"
            );
        }
    }

    #[test]
    fn test_matches_limma_at_span_03() {
        let (x, y) = fixture();
        let fit = weighted_lowess(&x, &y, None, None).unwrap();
        assert_close(&fit.fitted, &LIMMA_SPAN03_FITTED);
        assert_close(&fit.robust_weights, &LIMMA_SPAN03_WEIGHTS);
    }

    #[test]
    fn test_matches_limma_at_span_05() {
        let (x, y) = fixture();
        let params = LowessParams::new(0.5, 4, 200, None);
        let fit = weighted_lowess(&x, &y, None, Some(params)).unwrap();
        assert_close(&fit.fitted, &LIMMA_SPAN05_FITTED);
    }

    /// With `npts = 10` the derived delta is 0.0765625, so only a handful of
    /// seeds are fitted and everything else is interpolated. This is the test
    /// that exercises the binning shortcut against limma rather than around it.
    #[test]
    fn test_matches_limma_with_delta_binning() {
        let (x, y) = fixture();
        let params = LowessParams::new(0.3, 4, 10, None);
        let fit = weighted_lowess(&x, &y, None, Some(params)).unwrap();
        assert_close(&fit.fitted, &LIMMA_NPTS10_FITTED);

        let mut xs = x.clone();
        xs.sort_by(f64::total_cmp);
        assert_relative_eq!(
            resolve_delta(&xs, 10).unwrap(),
            0.0765625,
            max_relative = 1e-15
        );
        // Ten seeds is far fewer than fifty points, so the two fits differ.
        let exact = weighted_lowess(&x, &y, None, None).unwrap();
        let worst = fit
            .fitted
            .iter()
            .zip(&exact.fitted)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(worst > 1e-4, "binning should visibly change the fit");
    }

    /// One pass leaves the outlier's neighbourhood dragged towards it; four
    /// passes zero the outlier's weight and pull the curve back onto the trend.
    #[test]
    fn test_robustness_downweights_an_outlier() {
        let (x, mut y) = fixture();
        y[24] += 5.0;

        let one = LowessParams::new(0.3, 1, 200, None);
        let fit1 = weighted_lowess(&x, &y, None, Some(one)).unwrap();
        assert_close(&fit1.fitted, &LIMMA_OUTLIER_IT1_FITTED);
        assert_close(&fit1.robust_weights, &LIMMA_OUTLIER_IT1_WEIGHTS);

        let fit4 = weighted_lowess(&x, &y, None, None).unwrap();
        assert_close(&fit4.fitted, &LIMMA_OUTLIER_IT4_FITTED);
        assert_close(&fit4.robust_weights, &LIMMA_OUTLIER_IT4_WEIGHTS);

        assert_eq!(fit4.robust_weights[24], 0.0, "the outlier must be rejected");

        // The robustified fit sits on the outlier-free curve; the single-pass
        // one does not come close.
        let clean = fixture();
        let clean_fit = weighted_lowess(&clean.0, &clean.1, None, None).unwrap();
        let worst = |f: &[f64]| {
            f.iter()
                .zip(&clean_fit.fitted)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
        };
        assert!(worst(&fit4.fitted) < 0.02, "got {}", worst(&fit4.fitted));
        assert!(worst(&fit1.fitted) > 0.5, "got {}", worst(&fit1.fitted));

        // One pass leaves a whole run of neighbours looking like outliers.
        let zeros = |f: &[f64]| f.iter().filter(|w| **w == 0.0).count();
        assert!(zeros(&fit1.robust_weights) > zeros(&fit4.robust_weights));
    }

    #[test]
    fn test_prior_weights_change_the_fit() {
        let (x, y) = fixture();
        let w: Vec<f64> = (1..=50u64).map(|i| 1.0 + ((i * 7) % 5) as f64).collect();
        let fit = weighted_lowess(&x, &y, Some(&w), None).unwrap();
        assert_close(&fit.fitted, &LIMMA_PRIORW_FITTED);

        let unweighted = weighted_lowess(&x, &y, None, None).unwrap();
        let worst = fit
            .fitted
            .iter()
            .zip(&unweighted.fitted)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(worst > 1e-3, "prior weights should move the fit");
    }

    /// Unsorted input must give the same answer, returned in the caller's order.
    #[test]
    fn test_unsorted_input_matches_sorted() {
        let (x, y) = fixture();
        let perm: Vec<usize> = (1..=50usize).map(|i| (i * 17) % 50).collect();
        let xp: Vec<f64> = perm.iter().map(|&i| x[i]).collect();
        let yp: Vec<f64> = perm.iter().map(|&i| y[i]).collect();

        let fit = weighted_lowess(&xp, &yp, None, None).unwrap();
        let want: Vec<f64> = perm.iter().map(|&i| LIMMA_SPAN03_FITTED[i]).collect();
        assert_close(&fit.fitted, &want);

        let want_w: Vec<f64> = perm.iter().map(|&i| LIMMA_SPAN03_WEIGHTS[i]).collect();
        assert_close(&fit.robust_weights, &want_w);
    }

    /// Eight distinct x values, five points each. A tied group is never split
    /// across a window boundary, so every point in a group gets the same fit.
    #[test]
    fn test_ties_in_x() {
        let mut x = Vec::with_capacity(40);
        let mut y = Vec::with_capacity(40);
        for j in 1..=40u64 {
            let xj = ((j - 1) / 5) as f64 / 8.0;
            x.push(xj);
            y.push(2.0 * xj + (((j * j * 13) % 97) as f64 / 300.0 - 0.16));
        }
        let fit = weighted_lowess(&x, &y, None, None).unwrap();
        assert_close(&fit.fitted, &LIMMA_TIES_FITTED);
    }

    /// Every x identical: the bandwidth collapses, so the fit is the robustly
    /// weighted mean of y repeated.
    #[test]
    fn test_all_x_identical() {
        let x = vec![1.0; 10];
        let y: Vec<f64> = (1..=10u64).map(|i| (i * (i % 7)) as f64 / 10.0).collect();
        let fit = weighted_lowess(&x, &y, None, None).unwrap();
        for value in &fit.fitted {
            assert_relative_eq!(*value, LIMMA_FLAT_X_FITTED, max_relative = TOL);
        }
    }

    /// Rscript -e 'suppressMessages(library(limma));
    /// f <- weightedLowess((1:5)/5, c(1,3,2,5,4), span=0.3, iterations=4);
    /// cat(f$fitted, f$weights, sep=",")'
    /// gives the y values back with all weights at one: span 0.3 of five points
    /// admits two, of which the tricube kills one, so the fit interpolates.
    #[test]
    fn test_fewer_points_than_the_span_implies() {
        let x: Vec<f64> = (1..=5u64).map(|i| i as f64 / 5.0).collect();
        let y = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        let fit = weighted_lowess(&x, &y, None, None).unwrap();
        assert_close(&fit.fitted, &y);
        // An exact fit means zero residuals, so the robustness loop bails out
        // before it can zero anything.
        assert_close(&fit.robust_weights, &[1.0; 5]);

        // Rscript -e 'suppressMessages(library(limma));
        // cat(weightedLowess(c(1,2), c(3,7), span=0.3, iterations=4)$fitted, sep=",")'
        let two = weighted_lowess(&[1.0, 2.0], &[3.0, 7.0], None, None).unwrap();
        assert_close(&two.fitted, &[3.0, 7.0]);
    }

    #[test]
    fn test_rejects_length_mismatch() {
        let err = weighted_lowess(&[1.0, 2.0, 3.0], &[1.0, 2.0], None, None).unwrap_err();
        assert!(matches!(err, EdgeErrors::LengthMismatch { name: "y", .. }));

        let err = weighted_lowess(&[1.0, 2.0], &[1.0, 2.0], Some(&[1.0]), None).unwrap_err();
        assert!(matches!(
            err,
            EdgeErrors::LengthMismatch {
                name: "weights",
                ..
            }
        ));
    }

    #[test]
    fn test_rejects_too_few_points() {
        let err = weighted_lowess(&[], &[], None, None).unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));

        // R errors on a single point too.
        let err = weighted_lowess(&[1.0], &[2.0], None, None).unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));
    }

    #[test]
    fn test_rejects_span_outside_the_unit_interval() {
        let (x, y) = fixture();
        for span in [0.0, -0.1, 1.5, f64::NAN] {
            let params = LowessParams::new(span, 4, 200, None);
            let err = weighted_lowess(&x, &y, None, Some(params)).unwrap_err();
            assert!(matches!(err, EdgeErrors::InvalidArgument(_)), "span {span}");
        }
        // The closed upper end is legal.
        let params = LowessParams::new(1.0, 4, 200, None);
        assert!(weighted_lowess(&x, &y, None, Some(params)).is_ok());
    }

    #[test]
    fn test_rejects_negative_weights() {
        let (x, y) = fixture();
        let mut w = vec![1.0; 50];
        w[7] = -0.5;
        let err = weighted_lowess(&x, &y, Some(&w), None).unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));
    }

    #[test]
    fn test_rejects_zero_iterations_and_zero_npts() {
        let (x, y) = fixture();
        let params = LowessParams::new(0.3, 0, 200, None);
        let err = weighted_lowess(&x, &y, None, Some(params)).unwrap_err();
        assert!(matches!(err, EdgeErrors::MustBePositive(_)));

        let params = LowessParams::new(0.3, 4, 0, None);
        let err = weighted_lowess(&x, &y, None, Some(params)).unwrap_err();
        assert!(matches!(err, EdgeErrors::MustBePositive(_)));

        // npts is irrelevant once delta is given.
        let params = LowessParams::new(0.3, 4, 0, Some(0.0));
        assert!(weighted_lowess(&x, &y, None, Some(params)).is_ok());
    }

    /// `delta = Some(0.0)` fits every point, which is what the default does here
    /// anyway since `npts = 200` exceeds the fifty points on offer.
    #[test]
    fn test_explicit_zero_delta_matches_the_default() {
        let (x, y) = fixture();
        let params = LowessParams::new(0.3, 4, 200, Some(0.0));
        let fit = weighted_lowess(&x, &y, None, Some(params)).unwrap();
        assert_close(&fit.fitted, &LIMMA_SPAN03_FITTED);
    }

    /// The rayon fan-out must be bit-for-bit the sequential answer: each seed
    /// fit is independent, so splitting them changes no summation order.
    #[test]
    fn test_parallel_path_agrees_with_sequential() {
        let n = 400usize;
        let x: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let y: Vec<f64> = (0..n as u64)
            .map(|i| {
                let xi = i as f64 / n as f64;
                3.0 * xi - 2.0 * xi * xi + ((i * i * 37) % 101) as f64 / 500.0
            })
            .collect();
        let ws = vec![1.0; n];

        // 400 seeds over 400 points crosses the gate; the same run forced
        // sequential is the reference.
        assert!(n * n >= PARALLEL_WORK_THRESHOLD);
        let par =
            weighted_lowess(&x, &y, None, Some(LowessParams::new(0.3, 4, n, Some(0.0)))).unwrap();

        let seeds: Vec<usize> = (0..n).collect();
        let windows: Vec<Window> = seeds
            .iter()
            .map(|&s| seed_window(&x, &ws, s, 0.3 * n as f64))
            .collect();
        let (fitted, rob) = lowess_iterations(&x, &y, &ws, &seeds, &windows, n as f64, 4, false);

        assert_eq!(par.fitted, fitted);
        assert_eq!(par.robust_weights, rob);
    }
}
