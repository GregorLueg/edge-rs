//! Interpolation: the layer edgeR gets from R's C sources.
//!
//! Three things live here: the Forsythe-Malcolm-Moler cubic spline, which is
//! what R's `spline(method = "fmm")` fits and what edgeR's
//! `maximizeInterpolant` is built on. The natural spline basis behind
//! `dispCoxReidSplineTrend`. And plain piecewise-linear interpolation, which
//! voom leans on for its mean-variance trend.
//!
//! [`maximize_interpolant`] is the key one: every tagwise dispersion in
//! the package is the abscissa it returns. It reproduces edgeR's C `find_max`
//! step for step rather than handing the curve to a general optimiser, because
//! the grid is coarse and the answer has to be the same one edgeR would give,
//! not merely a nearby stationary point.
//!
//! Per the crate numeric policy this module is `f64` only.

use rayon::prelude::*;

use crate::prelude::*;

////////////
// Consts //
////////////

/// Fewest knots the FMM spline accepts.
///
/// Two knots degenerate to a straight line, which R still calls a spline, so
/// that is the floor rather than the four knots the full end conditions need.
const MIN_SPLINE_KNOTS: usize = 2;

/// Fewest columns [`natural_spline_basis`] will produce.
///
/// One intercept plus one linear term. Anything smaller is not a spline basis,
/// it is a rank-deficient design, so it is rejected rather than silently
/// truncated.
const MIN_BASIS_DF: usize = 2;

/// Fewest points the linear interpolators accept.
///
/// One point defines no segment and therefore no slope.
const MIN_INTERP_POINTS: usize = 2;

////////////////
// Validation //
////////////////

/// Checks that a knot vector can carry a spline.
///
/// ### Params
///
/// * `x` - Knot abscissae
///
/// ### Returns
///
/// `Ok(())`, or [`EdgeErrors::InvalidArgument`] if there are fewer than
/// [`MIN_SPLINE_KNOTS`] knots or they are not strictly increasing.
fn validate_knots(x: &[f64]) -> Result<(), EdgeErrors> {
    if x.len() < MIN_SPLINE_KNOTS {
        return Err(EdgeErrors::InvalidArgument(format!(
            "a spline needs at least {MIN_SPLINE_KNOTS} knots; got {}",
            x.len()
        )));
    }
    strictly_increasing(x, "knots")
}

/// Checks the knot vectors handed to the linear interpolators.
///
/// ### Params
///
/// * `xp` - Knot abscissae
/// * `fp` - Knot ordinates
///
/// ### Returns
///
/// `Ok(())`, or [`EdgeErrors::InvalidArgument`] / [`EdgeErrors::LengthMismatch`].
fn validate_interp_knots(xp: &[f64], fp: &[f64]) -> Result<(), EdgeErrors> {
    if xp.len() < MIN_INTERP_POINTS {
        return Err(EdgeErrors::InvalidArgument(format!(
            "linear interpolation needs at least {MIN_INTERP_POINTS} points; got {}",
            xp.len()
        )));
    }
    if fp.len() != xp.len() {
        return Err(EdgeErrors::LengthMismatch {
            name: "fp",
            expected: xp.len(),
            got: fp.len(),
        });
    }
    strictly_increasing(xp, "xp")
}

/// Checks that a vector is strictly increasing.
///
/// Ties matter as much as inversions: a repeated abscissa makes a segment of
/// zero width, and every routine here divides by that width.
///
/// ### Params
///
/// * `x` - Values to check
/// * `name` - Argument name, for the error message
///
/// ### Returns
///
/// `Ok(())`, or [`EdgeErrors::InvalidArgument`] naming the first bad position.
fn strictly_increasing(x: &[f64], name: &str) -> Result<(), EdgeErrors> {
    for (i, pair) in x.windows(2).enumerate() {
        // NaN fails the comparison silently, so it is rejected explicitly.
        if pair[1] <= pair[0] || pair[0].is_nan() || pair[1].is_nan() {
            return Err(EdgeErrors::InvalidArgument(format!(
                "{name} must be strictly increasing; {} at index {} is not above {} at index {i}",
                pair[1],
                i + 1,
                pair[0],
            )));
        }
    }
    Ok(())
}

////////////////
// FMM spline //
////////////////

/// Cubic spline coefficients in the Forsythe, Malcolm and Moler form.
///
/// On segment `i` the spline is
///
/// ```text
/// S(t) = y[i] + b[i] * t + c[i] * t^2 + d[i] * t^3,   t = x_eval - x[i]
/// ```
///
/// which is the layout R's `splines.c` writes and the one edgeR's `find_max`
/// reads. All five vectors have the same length, one entry per knot; the last
/// entry of `b`, `c` and `d` describes the extrapolating polynomial to the right
/// of the final knot.
///
/// ### References
///
/// Forsythe, Malcolm and Moler, *Computer Methods for Mathematical
/// Computations*, Prentice-Hall, 1977, chapter 4.
#[derive(Clone, Debug)]
pub struct FmmSpline {
    /// Knot abscissae, strictly increasing.
    pub x: Vec<f64>,
    /// Knot ordinates, i.e. the constant term of each segment polynomial.
    pub y: Vec<f64>,
    /// Linear coefficients, one per knot.
    pub b: Vec<f64>,
    /// Quadratic coefficients, one per knot.
    pub c: Vec<f64>,
    /// Cubic coefficients, one per knot.
    pub d: Vec<f64>,
}

/// Fits an FMM cubic spline through the given knots.
///
/// Not-a-knot end conditions: the third derivative is matched across the second
/// and second-to-last knots, so the end segments are cubics fitted to the four
/// outermost points. With fewer than four knots the end conditions degenerate
/// and the routine falls back to a parabola (three knots) or a straight line
/// (two knots), exactly as R does.
///
/// ### Params
///
/// * `x` - Knot abscissae, strictly increasing
/// * `y` - Knot ordinates, same length as `x`
///
/// ### Returns
///
/// The fitted [`FmmSpline`], or [`EdgeErrors::InvalidArgument`] if `x` is too
/// short or not strictly increasing, or [`EdgeErrors::LengthMismatch`] if `y`
/// disagrees with `x`.
pub fn fmm_spline(x: &[f64], y: &[f64]) -> Result<FmmSpline, EdgeErrors> {
    validate_knots(x)?;
    if y.len() != x.len() {
        return Err(EdgeErrors::LengthMismatch {
            name: "y",
            expected: x.len(),
            got: y.len(),
        });
    }

    let n = x.len();
    let mut b = vec![0.0; n];
    let mut c = vec![0.0; n];
    let mut d = vec![0.0; n];
    fmm_coefficients(x, y, &mut b, &mut c, &mut d);

    Ok(FmmSpline {
        x: x.to_vec(),
        y: y.to_vec(),
        b,
        c,
        d,
    })
}

impl FmmSpline {
    /// Evaluates the spline at one abscissa.
    ///
    /// Outside the knot range the nearest end segment's cubic is extended, which
    /// is what R's `spline_eval` does for `method = "fmm"`. That polynomial
    /// diverges quickly, so treat values far outside the range as meaningless.
    ///
    /// ### Params
    ///
    /// * `x` - Abscissa to evaluate at
    ///
    /// ### Returns
    ///
    /// The spline value.
    pub fn eval(&self, x: f64) -> f64 {
        // Largest index whose knot is at or below `x`, clamped into the last
        // segment so points beyond the final knot extrapolate rather than panic.
        let i = self.x.partition_point(|&knot| knot <= x).saturating_sub(1);
        let i = i.min(self.x.len() - 1);
        let dx = x - self.x[i];
        self.y[i] + dx * (self.b[i] + dx * (self.c[i] + dx * self.d[i]))
    }
}

/// Writes FMM spline coefficients into caller-supplied buffers.
///
/// A direct transcription of R's `fmm_spline` in
/// `src/library/stats/src/splines.c`. The tridiagonal system is assembled in
/// place: `d` holds the knot spacings and then the off-diagonal, `b` the
/// diagonal, `c` the right-hand side and then the solution. Every buffer entry
/// is written before it is read, so the buffers can be reused across genes
/// without clearing.
///
/// ### Params
///
/// * `x` - Knot abscissae, strictly increasing, at least [`MIN_SPLINE_KNOTS`] long
/// * `y` - Knot ordinates, same length as `x`
/// * `b` - Scratch for the linear coefficients, same length as `x`
/// * `c` - Scratch for the quadratic coefficients, same length as `x`
/// * `d` - Scratch for the cubic coefficients, same length as `x`
fn fmm_coefficients(x: &[f64], y: &[f64], b: &mut [f64], c: &mut [f64], d: &mut [f64]) {
    let n = x.len();

    if n < 3 {
        // Two knots: a straight line through both.
        let t = (y[1] - y[0]) / (x[1] - x[0]);
        b[0] = t;
        b[1] = t;
        c[0] = 0.0;
        c[1] = 0.0;
        d[0] = 0.0;
        d[1] = 0.0;
        return;
    }

    let nm1 = n - 1;

    // Tridiagonal system for the second derivatives.
    d[0] = x[1] - x[0];
    c[1] = (y[1] - y[0]) / d[0];
    for i in 1..nm1 {
        d[i] = x[i + 1] - x[i];
        b[i] = 2.0 * (d[i - 1] + d[i]);
        c[i + 1] = (y[i + 1] - y[i]) / d[i];
        c[i] = c[i + 1] - c[i];
    }

    // End conditions: third derivatives at x[1] and x[n-2] computed from
    // divided differences. With exactly three knots there is no room for them
    // and the spline collapses to a single parabola.
    b[0] = -d[0];
    b[nm1] = -d[nm1 - 1];
    c[0] = 0.0;
    c[nm1] = 0.0;
    if n > 3 {
        c[0] = c[2] / (x[3] - x[1]) - c[1] / (x[2] - x[0]);
        c[nm1] = c[nm1 - 1] / (x[nm1] - x[nm1 - 2]) - c[nm1 - 2] / (x[nm1 - 1] - x[nm1 - 3]);
        c[0] = c[0] * d[0] * d[0] / (x[3] - x[0]);
        c[nm1] = -c[nm1] * d[nm1 - 1] * d[nm1 - 1] / (x[nm1] - x[nm1 - 3]);
    }

    // Forward elimination.
    for i in 1..n {
        let t = d[i - 1] / b[i - 1];
        b[i] -= t * d[i - 1];
        c[i] -= t * c[i - 1];
    }

    // Back substitution.
    c[nm1] /= b[nm1];
    for i in (0..nm1).rev() {
        c[i] = (c[i] - d[i] * c[i + 1]) / b[i];
    }

    // Convert second derivatives into polynomial coefficients.
    b[nm1] = (y[nm1] - y[nm1 - 1]) / d[nm1 - 1] + d[nm1 - 1] * (c[nm1 - 1] + 2.0 * c[nm1]);
    for i in 0..nm1 {
        b[i] = (y[i + 1] - y[i]) / d[i] - d[i] * (c[i + 1] + 2.0 * c[i]);
        d[i] = (c[i + 1] - c[i]) / d[i];
        c[i] *= 3.0;
    }
    c[nm1] *= 3.0;
    d[nm1] = d[nm1 - 1];
}

//////////////////////////
// maximize_interpolant //
//////////////////////////

/// Locates the maximum of an FMM spline through a log-likelihood grid.
///
/// edgeR's `maximizeInterpolant` for a single curve. The procedure is three
/// steps and reproducing it exactly matters, because the tagwise dispersion is
/// whatever this returns:
///
/// 1. Take the grid point with the largest ordinate.
/// 2. Fit the FMM cubic spline through the whole grid.
/// 3. On each of the (at most two) segments touching that grid point, solve
///    `b + 2ct + 3dt^2 = 0` analytically and keep the root that is a maximum.
///    If it falls strictly inside the segment and beats the grid value, it
///    wins.
///
/// A general-purpose optimiser would land somewhere else on a coarse grid, so
/// it is not a substitute.
///
/// ### Params
///
/// * `x` - Grid abscissae, strictly increasing
/// * `y` - Curve ordinates, same length as `x`
///
/// ### Returns
///
/// The abscissa of the largest value found, or [`EdgeErrors::InvalidArgument`] /
/// [`EdgeErrors::LengthMismatch`] on a malformed grid.
pub fn maximize_interpolant(x: &[f64], y: &[f64]) -> Result<f64, EdgeErrors> {
    validate_knots(x)?;
    if y.len() != x.len() {
        return Err(EdgeErrors::LengthMismatch {
            name: "y",
            expected: x.len(),
            got: y.len(),
        });
    }

    let n = x.len();
    let mut b = vec![0.0; n];
    let mut c = vec![0.0; n];
    let mut d = vec![0.0; n];
    Ok(find_max(x, y, &mut b, &mut c, &mut d))
}

/// Locates the spline maximum of one log-likelihood curve per gene.
///
/// The parallel axis is genes because that is where the work is: a dataset has
/// tens of thousands of genes and a grid of a couple of dozen points, each gene
/// is an independent spline fit, and each gene's row is contiguous in `y`. The
/// grid itself is far too short to parallelise over and shared by everyone.
/// Scratch buffers for the tridiagonal solve are allocated once per chunk
/// rather than once per gene.
///
/// ### Params
///
/// * `x` - Grid abscissae, strictly increasing, shared by every gene
/// * `y` - Row-major `n_genes` by `x.len()` matrix, one curve per gene
/// * `n_genes` - Number of rows in `y`
///
/// ### Returns
///
/// One maximising abscissa per gene, in gene order, or [`EdgeErrors`] if the
/// grid is malformed or `y` is not `n_genes * x.len()` long.
pub fn maximize_interpolant_many(
    x: &[f64],
    y: &[f64],
    n_genes: usize,
) -> Result<Vec<f64>, EdgeErrors> {
    validate_knots(x)?;
    if n_genes == 0 {
        return Err(EdgeErrors::MustBePositive("n_genes".to_string()));
    }
    let npts = x.len();
    if y.len() != n_genes * npts {
        return Err(EdgeErrors::LengthMismatch {
            name: "y",
            expected: n_genes * npts,
            got: y.len(),
        });
    }

    let chunk = n_genes.div_ceil(rayon::current_num_threads().max(1)).max(1);
    let mut out = vec![0.0; n_genes];
    out.par_chunks_mut(chunk)
        .zip(y.par_chunks(chunk * npts))
        .for_each(|(out_chunk, y_chunk)| {
            let mut b = vec![0.0; npts];
            let mut c = vec![0.0; npts];
            let mut d = vec![0.0; npts];
            for (slot, row) in out_chunk.iter_mut().zip(y_chunk.chunks_exact(npts)) {
                *slot = find_max(x, row, &mut b, &mut c, &mut d);
            }
        });

    Ok(out)
}

/// The body of edgeR's C `find_max`, with caller-supplied scratch.
///
/// ### Params
///
/// * `x` - Grid abscissae, strictly increasing
/// * `y` - Curve ordinates, same length as `x`
/// * `b` - Scratch, same length as `x`
/// * `c` - Scratch, same length as `x`
/// * `d` - Scratch, same length as `x`
///
/// ### Returns
///
/// The abscissa of the largest value found.
fn find_max(x: &[f64], y: &[f64], b: &mut [f64], c: &mut [f64], d: &mut [f64]) -> f64 {
    let n = x.len();

    let mut maxed = y[0];
    let mut maxed_at = 0;
    for (i, &yi) in y.iter().enumerate().skip(1) {
        if yi > maxed {
            maxed = yi;
            maxed_at = i;
        }
    }
    let mut x_max = x[maxed_at];

    fmm_coefficients(x, y, b, c, d);

    if maxed_at > 0 {
        refine_on_segment(maxed_at - 1, x, y, b, c, d, &mut maxed, &mut x_max);
    }
    if maxed_at + 1 < n {
        refine_on_segment(maxed_at, x, y, b, c, d, &mut maxed, &mut x_max);
    }

    x_max
}

/// Solves for the spline maximum inside one segment and keeps it if it wins.
///
/// The derivative on segment `seg` is `b + 2ct + 3dt^2`, so the stationary
/// points are `t = (-c +/- sqrt(c^2 - 3db)) / (3d)`. The negative branch is the
/// maximum for the sign convention edgeR uses; a zero cubic coefficient is
/// treated as "no interior root", matching the reference implementation, since
/// the linear case has no interior stationary point to find anyway.
///
/// ### Params
///
/// * `seg` - Index of the segment, i.e. the knot at its left end
/// * `x` - Grid abscissae
/// * `y` - Curve ordinates
/// * `b` - Linear spline coefficients
/// * `c` - Quadratic spline coefficients
/// * `d` - Cubic spline coefficients
/// * `maxed` - Best ordinate so far, updated in place
/// * `x_max` - Abscissa of `maxed`, updated in place
#[allow(clippy::too_many_arguments)]
fn refine_on_segment(
    seg: usize,
    x: &[f64],
    y: &[f64],
    b: &[f64],
    c: &[f64],
    d: &[f64],
    maxed: &mut f64,
    x_max: &mut f64,
) {
    let (sb, sc, sd) = (b[seg], c[seg], d[seg]);

    let delta = sc * sc - 3.0 * sd * sb;
    if delta < 0.0 {
        return;
    }
    if sd == 0.0 {
        return;
    }

    let root = (-sc - delta.sqrt()) / (3.0 * sd);
    let width = x[seg + 1] - x[seg];
    if root <= 0.0 || root >= width {
        return;
    }

    let value = ((sd * root + sc) * root + sb) * root + y[seg];
    if value > *maxed {
        *maxed = value;
        *x_max = root + x[seg];
    }
}

//////////////////////////
// Natural spline basis //
//////////////////////////

/// Builds a natural cubic spline basis over `x`.
///
/// The truncated power basis of Hastie, Tibshirani and Friedman, equations 5.4
/// and 5.5: an intercept, a linear term, and `df - 2` terms of the form
///
/// ```text
/// d_j(x) - d_{K-1}(x),   d_j(x) = ((x - k_j)_+^3 - (x - k_K)_+^3) / (k_K - k_j)
/// ```
///
/// Boundary knots sit at the range of `x` and the `df - 2` internal knots at
/// equally spaced quantiles, the same placement R's `ns(x, df = df,
/// intercept = TRUE)` uses. This spans the same space as R's basis but is not
/// column-for-column equal to it: R returns a B-spline parametrisation, this is
/// the truncated power one. Fitted values from a linear model agree, individual
/// coefficients do not.
///
/// `x` need not be sorted. If every value of `x` is identical there is no range
/// to place knots in, and the basis degenerates to `[1, x]`; the returned
/// column count is then 2 rather than `df`, which is why the count is returned
/// at all.
///
/// ### Params
///
/// * `x` - Covariate values, in any order
/// * `df` - Requested number of basis columns, at least [`MIN_BASIS_DF`]
///
/// ### Returns
///
/// The flattened row-major basis and its column count, or
/// [`EdgeErrors::InvalidArgument`] if `x` is empty or `df` is below the
/// minimum.
///
/// ### References
///
/// Hastie, Tibshirani and Friedman, *The Elements of Statistical Learning*,
/// 2nd edition, Springer, 2009, section 5.2.1.
pub fn natural_spline_basis(x: &[f64], df: usize) -> Result<(Vec<f64>, usize), EdgeErrors> {
    if x.is_empty() {
        return Err(EdgeErrors::InvalidArgument(
            "natural_spline_basis needs at least one covariate value".to_string(),
        ));
    }
    if df < MIN_BASIS_DF {
        return Err(EdgeErrors::InvalidArgument(format!(
            "natural_spline_basis needs df >= {MIN_BASIS_DF}; got {df}"
        )));
    }

    let n = x.len();
    let mut sorted = x.to_vec();
    sorted.sort_by(f64::total_cmp);
    let lower = sorted[0];
    let upper = sorted[n - 1];

    let n_internal = df - MIN_BASIS_DF;
    if n_internal == 0 || lower == upper {
        let mut basis = vec![0.0; n * MIN_BASIS_DF];
        for (row, &xi) in basis.chunks_exact_mut(MIN_BASIS_DF).zip(x) {
            row[0] = 1.0;
            row[1] = xi;
        }
        return Ok((basis, MIN_BASIS_DF));
    }

    // Knots: boundaries at the range, internal ones at equally spaced quantiles.
    let mut knots = Vec::with_capacity(df);
    knots.push(lower);
    for j in 1..=n_internal {
        knots.push(quantile_type7(&sorted, j as f64 / (n_internal + 1) as f64));
    }
    knots.push(upper);
    knots.sort_by(f64::total_cmp);

    let n_knots = knots.len();
    let xi_last = knots[n_knots - 1];
    let xi_prev = knots[n_knots - 2];

    let mut basis = vec![0.0; n * df];
    for (row, &xi) in basis.chunks_exact_mut(df).zip(x) {
        row[0] = 1.0;
        row[1] = xi;
        let tail = truncated_cube(xi, xi_last);
        let d_prev = truncated_power_term(xi, xi_prev, xi_last, tail);
        for j in 0..n_knots - 2 {
            row[2 + j] = truncated_power_term(xi, knots[j], xi_last, tail) - d_prev;
        }
    }

    Ok((basis, df))
}

/// The cube of the positive part of `x - knot`.
///
/// ### Params
///
/// * `x` - Covariate value
/// * `knot` - Knot position
///
/// ### Returns
///
/// `max(x - knot, 0)^3`.
#[inline]
fn truncated_cube(x: f64, knot: f64) -> f64 {
    let t = (x - knot).max(0.0);
    t * t * t
}

/// One `d_j` term of the truncated power basis.
///
/// Coincident knots would divide by zero. R's knot placement cannot produce
/// that unless `x` is so tied that a quantile lands on the upper boundary, in
/// which case the term carries no information and is set to zero rather than to
/// NaN.
///
/// ### Params
///
/// * `x` - Covariate value
/// * `knot` - Knot `k_j` for this term
/// * `xi_last` - Upper boundary knot `k_K`
/// * `tail` - Precomputed `max(x - xi_last, 0)^3`, shared across terms
///
/// ### Returns
///
/// `(max(x - knot, 0)^3 - tail) / (xi_last - knot)`, or zero on a degenerate knot.
#[inline]
fn truncated_power_term(x: f64, knot: f64, xi_last: f64, tail: f64) -> f64 {
    let span = xi_last - knot;
    if span == 0.0 {
        return 0.0;
    }
    (truncated_cube(x, knot) - tail) / span
}

/// The type 7 sample quantile, R's default and numpy's.
///
/// ### Params
///
/// * `sorted` - Values in ascending order, non-empty
/// * `p` - Probability in `[0, 1]`
///
/// ### Returns
///
/// The interpolated quantile.
fn quantile_type7(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let h = (n - 1) as f64 * p;
    let lo = h.floor();
    let idx = (lo as usize).min(n - 2);
    sorted[idx] + (h - lo) * (sorted[idx + 1] - sorted[idx])
}

//////////////////////////
// Linear interpolation //
//////////////////////////

/// Piecewise-linear interpolation inside the knot range.
///
/// ### Params
///
/// * `x` - Abscissae to evaluate at, in any order
/// * `xp` - Knot abscissae, strictly increasing
/// * `fp` - Knot ordinates, same length as `xp`
///
/// ### Returns
///
/// One value per entry of `x`, or [`EdgeErrors::OutsideKnotRange`] if any query
/// point lies outside `[xp[0], xp[last]]`. Use [`interp_linear_extrap`] when
/// points outside the range are expected.
pub fn interp_linear(x: &[f64], xp: &[f64], fp: &[f64]) -> Result<Vec<f64>, EdgeErrors> {
    validate_interp_knots(xp, fp)?;
    let lower = xp[0];
    let upper = xp[xp.len() - 1];

    x.iter()
        .map(|&xi| {
            if xi < lower || xi > upper {
                Err(EdgeErrors::OutsideKnotRange {
                    x: xi,
                    lower,
                    upper,
                })
            } else {
                Ok(interp_one(xi, xp, fp))
            }
        })
        .collect()
}

/// Piecewise-linear interpolation with constant extension outside the range.
///
/// Points below the first knot take `fp[0]` and points above the last take
/// `fp[last]`, which is `numpy.interp`'s behaviour and therefore what voom's
/// mean-variance trend lookup does in edgePython.
///
/// ### Params
///
/// * `x` - Abscissae to evaluate at, in any order
/// * `xp` - Knot abscissae, strictly increasing
/// * `fp` - Knot ordinates, same length as `xp`
///
/// ### Returns
///
/// One value per entry of `x`.
pub fn interp_linear_extrap(x: &[f64], xp: &[f64], fp: &[f64]) -> Result<Vec<f64>, EdgeErrors> {
    validate_interp_knots(xp, fp)?;
    Ok(x.iter().map(|&xi| interp_one(xi, xp, fp)).collect())
}

/// Interpolates one point, clamping to the end values outside the knot range.
///
/// ### Params
///
/// * `x` - Abscissa to evaluate at
/// * `xp` - Knot abscissae, strictly increasing, at least two of them
/// * `fp` - Knot ordinates, same length as `xp`
///
/// ### Returns
///
/// The interpolated value.
#[inline]
fn interp_one(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let last = xp.len() - 1;
    if x <= xp[0] {
        return fp[0];
    }
    if x >= xp[last] {
        return fp[last];
    }
    // `x` is strictly inside, so the partition point is in 1..=last.
    let hi = xp.partition_point(|&knot| knot <= x);
    let lo = hi - 1;
    let t = (x - xp[lo]) / (xp[hi] - xp[lo]);
    fp[lo] + t * (fp[hi] - fp[lo])
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Rscript -e 'options(digits=17); x <- c(1,2,3,4,5); y <- c(1,4,2,8,3);
    ///   cat(spline(x, y, method="fmm", xout=c(1.5,2.5,3.5))$y, sep=",")'
    const R_FMM_FIVE: [f64; 3] = [4.15972222222222, 2.36805555555556, 4.74305555555556];

    /// Rscript -e 'options(digits=17); x <- c(1,2,3,4,5); y <- c(1,4,2,8,3);
    ///   cat(spline(x, y, method="fmm", xout=c(0.5,5.5))$y, sep=",")'
    const R_FMM_OUTSIDE: [f64; 2] = [-7.10416666666667, -10.2291666666667];

    /// Rscript -e 'options(digits=17);
    ///   cat(spline(c(0,1,3), c(2,0,5), method="fmm", xout=c(0.5,2.0))$y, sep=",")'
    const R_FMM_THREE: [f64; 2] = [0.625, 1.0];

    /// Rscript -e 'options(digits=17);
    ///   cat(spline(c(0,2), c(1,5), method="fmm", xout=c(0.5,1.5,3.0))$y, sep=",")'
    const R_FMM_TWO: [f64; 3] = [2.0, 4.0, 7.0];

    /// Grid shared by the maximizeInterpolant references below.
    const MI_GRID: [f64; 5] = [0.0, 1.0, 2.0, 3.0, 4.0];

    /// Rscript -e 'options(digits=17); library(edgeR);
    ///   cat(maximizeInterpolant(c(0,1,2,3,4),
    ///     matrix(c(-4,-1,0.5,0.2,-3), nrow=1)))'
    const R_MI_INTERIOR: f64 = 2.3887669447483653;

    /// Rscript -e 'options(digits=17); library(edgeR);
    ///   cat(maximizeInterpolant(c(0,1,2,3,4),
    ///     matrix(c(0,2.5,2.4,1,-1), nrow=1)))'
    const R_MI_INTERIOR_EARLY: f64 = 1.4164937870948595;

    /// Rscript -e 'options(digits=17); library(edgeR);
    ///   cat(maximizeInterpolant(c(-2,-0.5,0.25,1.5,4),
    ///     matrix(c(-3,0.4,1.1,0.9,-2.2), nrow=1)))'
    const R_MI_NONUNIFORM: f64 = 0.6884019949364506;

    /// Rscript -e 'options(digits=17); library(splines);
    ///   cat(ns(1:10, df=5, intercept=TRUE), sep=",")'   [column-major, 10 x 5]
    const R_NS_DF5: [f64; 50] = [
        -0.267261241912424,
        0.243465876939416,
        0.563319734782542,
        0.534154130643652,
        0.282669497191026,
        0.0747677460437111,
        0.00588550770606472,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0146319158664838,
        0.11705532693187,
        0.37037037037037,
        0.622770919067215,
        0.622770919067215,
        0.37037037037037,
        0.11705532693187,
        0.0146319158664838,
        0.0,
        -0.214285714285714,
        -0.137678080611314,
        -0.0733299370748607,
        -0.0262752385762756,
        0.0613409777969527,
        0.272581217827764,
        0.525783731386229,
        0.575413155660069,
        0.304069501600366,
        -0.142857142857143,
        0.642857142857143,
        0.413034241833943,
        0.219989811224582,
        0.0973442342473452,
        0.0512308388999513,
        0.0732028485743209,
        0.144871028063535,
        0.238095238095238,
        0.333333333333333,
        0.428571428571429,
        -0.428571428571429,
        -0.275356161222629,
        -0.146659874149721,
        -0.0648961561648968,
        -0.0341538925999675,
        -0.0469729095662368,
        -0.047197969326307,
        0.0694362793128225,
        0.347965249199817,
        0.714285714285714,
    ];

    /// uv run --with numpy python -c "... truncated power basis of 1:10, df=5 ..."
    /// (the port of edgePython's `_natural_spline_basis`, row-major 10 x 5)
    const NUMPY_TP_DF5: [f64; 50] = [
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        2.0,
        0.1111111111111111,
        0.0,
        0.0,
        1.0,
        3.0,
        0.8888888888888888,
        0.0,
        0.0,
        1.0,
        4.0,
        3.0,
        0.0625,
        0.0,
        1.0,
        5.0,
        7.111111111111111,
        0.7939814814814815,
        0.0,
        1.0,
        6.0,
        13.88888888888889,
        3.0810185185185186,
        0.027777777777777776,
        1.0,
        7.0,
        24.0,
        7.8125,
        0.75,
        1.0,
        8.0,
        38.10416666666667,
        15.87037037037037,
        3.4652777777777777,
        1.0,
        9.0,
        56.02083333333333,
        27.296296296296294,
        8.659722222222223,
        1.0,
        10.0,
        75.9375,
        40.5,
        15.1875,
    ];

    /// uv run --with numpy python -c "import numpy as np;
    ///   print(np.interp([0.5,1.0,1.5,3.0,3.999],[0,1,2,4],[0,2,3,-1]))"
    const NP_INTERP_INSIDE: [f64; 5] = [1.0, 2.0, 2.5, 1.0, -0.9980000000000002];

    /// uv run --with numpy python -c "import numpy as np;
    ///   print(np.interp([-3.0,-0.001,4.5,10.0],[0,1,2,4],[0,2,3,-1]))"
    const NP_INTERP_OUTSIDE: [f64; 4] = [0.0, 0.0, -1.0, -1.0];

    // -- FMM spline --

    #[test]
    fn test_fmm_spline_matches_r_at_interior_points() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [1.0, 4.0, 2.0, 8.0, 3.0];
        let s = fmm_spline(&x, &y).unwrap();
        for (&at, &want) in [1.5, 2.5, 3.5].iter().zip(&R_FMM_FIVE) {
            assert_relative_eq!(s.eval(at), want, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_fmm_spline_matches_r_outside_the_knots() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [1.0, 4.0, 2.0, 8.0, 3.0];
        let s = fmm_spline(&x, &y).unwrap();
        for (&at, &want) in [0.5, 5.5].iter().zip(&R_FMM_OUTSIDE) {
            assert_relative_eq!(s.eval(at), want, epsilon = 1e-11);
        }
    }

    #[test]
    fn test_fmm_spline_three_knots_is_a_parabola() {
        let x = [0.0, 1.0, 3.0];
        let y = [2.0, 0.0, 5.0];
        let s = fmm_spline(&x, &y).unwrap();
        for (&at, &want) in [0.5, 2.0].iter().zip(&R_FMM_THREE) {
            assert_relative_eq!(s.eval(at), want, epsilon = 1e-12);
        }
        // A parabola has no cubic term anywhere.
        assert!(s.d.iter().all(|&di| di.abs() < 1e-12));
    }

    #[test]
    fn test_fmm_spline_two_knots_is_a_line() {
        let x = [0.0, 2.0];
        let y = [1.0, 5.0];
        let s = fmm_spline(&x, &y).unwrap();
        for (&at, &want) in [0.5, 1.5, 3.0].iter().zip(&R_FMM_TWO) {
            assert_relative_eq!(s.eval(at), want, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_fmm_spline_passes_through_its_knots() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [1.0, 4.0, 2.0, 8.0, 3.0];
        let s = fmm_spline(&x, &y).unwrap();
        for (&at, &want) in x.iter().zip(&y) {
            assert_relative_eq!(s.eval(at), want, epsilon = 1e-12);
        }
    }

    // -- maximize_interpolant --

    #[test]
    fn test_maximize_interpolant_finds_max_between_grid_points() {
        let y = [-4.0, -1.0, 0.5, 0.2, -3.0];
        let got = maximize_interpolant(&MI_GRID, &y).unwrap();
        assert_relative_eq!(got, R_MI_INTERIOR, epsilon = 1e-10);
        // The coarse grid argmax is 2.0; a routine that stopped there fails here.
        assert!((got - 2.0).abs() > 0.3);
        // And the spline really is higher there than at the grid point.
        let s = fmm_spline(&MI_GRID, &y).unwrap();
        assert!(s.eval(got) > s.eval(2.0));
    }

    #[test]
    fn test_maximize_interpolant_at_first_grid_point() {
        let y = [5.0, 3.0, 1.0, -2.0, -6.0];
        // Rscript: maximizeInterpolant(c(0,1,2,3,4), matrix(c(5,3,1,-2,-6), nrow=1)) -> 0
        assert_relative_eq!(
            maximize_interpolant(&MI_GRID, &y).unwrap(),
            0.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_maximize_interpolant_at_last_grid_point() {
        let y = [-6.0, -2.0, 1.0, 3.0, 5.0];
        // Rscript: maximizeInterpolant(c(0,1,2,3,4), matrix(c(-6,-2,1,3,5), nrow=1)) -> 4
        assert_relative_eq!(
            maximize_interpolant(&MI_GRID, &y).unwrap(),
            4.0,
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_maximize_interpolant_on_a_non_uniform_grid() {
        let x = [-2.0, -0.5, 0.25, 1.5, 4.0];
        let y = [-3.0, 0.4, 1.1, 0.9, -2.2];
        assert_relative_eq!(
            maximize_interpolant(&x, &y).unwrap(),
            R_MI_NONUNIFORM,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_maximize_interpolant_many_matches_row_by_row() {
        let rows: [[f64; 5]; 4] = [
            [-4.0, -1.0, 0.5, 0.2, -3.0],
            [5.0, 3.0, 1.0, -2.0, -6.0],
            [-6.0, -2.0, 1.0, 3.0, 5.0],
            [0.0, 2.5, 2.4, 1.0, -1.0],
        ];
        let flat: Vec<f64> = rows.iter().flatten().copied().collect();
        let many = maximize_interpolant_many(&MI_GRID, &flat, rows.len()).unwrap();

        for (got, row) in many.iter().zip(&rows) {
            assert_relative_eq!(
                *got,
                maximize_interpolant(&MI_GRID, row).unwrap(),
                epsilon = 1e-14
            );
        }
        // Rscript: maximizeInterpolant on the same four rows.
        let want = [R_MI_INTERIOR, 0.0, 4.0, R_MI_INTERIOR_EARLY];
        for (got, wanted) in many.iter().zip(&want) {
            assert_relative_eq!(*got, *wanted, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_maximize_interpolant_many_over_many_genes() {
        // Enough genes that rayon actually splits the work into chunks.
        let n_genes = 500;
        let mut flat = Vec::with_capacity(n_genes * MI_GRID.len());
        for g in 0..n_genes {
            let peak = 0.5 + 3.0 * (g as f64) / (n_genes as f64);
            for &xi in &MI_GRID {
                flat.push(-(xi - peak) * (xi - peak));
            }
        }
        let many = maximize_interpolant_many(&MI_GRID, &flat, n_genes).unwrap();
        for (g, got) in many.iter().enumerate() {
            let row = &flat[g * MI_GRID.len()..(g + 1) * MI_GRID.len()];
            assert_relative_eq!(*got, maximize_interpolant(&MI_GRID, row).unwrap());
        }
    }

    // -- natural spline basis --

    #[test]
    fn test_natural_spline_basis_matches_truncated_power_reference() {
        let x: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let (basis, ncols) = natural_spline_basis(&x, 5).unwrap();
        assert_eq!(ncols, 5);
        assert_eq!(basis.len(), 50);
        for (got, want) in basis.iter().zip(&NUMPY_TP_DF5) {
            assert_relative_eq!(*got, *want, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_natural_spline_basis_spans_the_same_space_as_r_ns() {
        // R's ns() returns a B-spline parametrisation, ours the truncated power
        // one. They are different matrices spanning the same column space, so
        // the check is that each column of R's basis is an exact linear
        // combination of ours.
        let n = 10;
        let df = 5;
        let x: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let (basis, ncols) = natural_spline_basis(&x, df).unwrap();
        assert_eq!(ncols, df);

        // R printed column-major; transpose into row-major.
        let mut r_basis = vec![0.0; n * df];
        for j in 0..df {
            for i in 0..n {
                r_basis[i * df + j] = R_NS_DF5[j * n + i];
            }
        }

        let gram = gram_matrix(&basis, n, df);
        for j in 0..df {
            let target: Vec<f64> = (0..n).map(|i| r_basis[i * df + j]).collect();
            let rhs = cross_product(&basis, n, df, &target);
            let coef = solve_spd(&gram, df, &rhs);
            for i in 0..n {
                let fitted: f64 = (0..df).map(|k| basis[i * df + k] * coef[k]).sum();
                assert_relative_eq!(fitted, target[i], epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn test_natural_spline_basis_degenerates_when_x_is_constant() {
        let x = [3.0; 6];
        let (basis, ncols) = natural_spline_basis(&x, 5).unwrap();
        assert_eq!(ncols, 2);
        assert_eq!(
            basis,
            vec![1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0]
        );
    }

    #[test]
    fn test_natural_spline_basis_df_two_is_linear() {
        let x = [1.0, 4.0, 9.0];
        let (basis, ncols) = natural_spline_basis(&x, 2).unwrap();
        assert_eq!(ncols, 2);
        assert_eq!(basis, vec![1.0, 1.0, 1.0, 4.0, 1.0, 9.0]);
    }

    #[test]
    fn test_natural_spline_basis_ignores_input_order() {
        let ordered: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let shuffled: Vec<f64> = vec![7.0, 1.0, 10.0, 4.0, 2.0, 9.0, 3.0, 6.0, 8.0, 5.0];
        let (a, df) = natural_spline_basis(&ordered, 5).unwrap();
        let (b, _) = natural_spline_basis(&shuffled, 5).unwrap();
        for (i, &xi) in shuffled.iter().enumerate() {
            let src = (xi as usize) - 1;
            for k in 0..df {
                assert_relative_eq!(b[i * df + k], a[src * df + k], epsilon = 1e-12);
            }
        }
    }

    // -- linear interpolation --

    #[test]
    fn test_interp_linear_inside_the_range() {
        let xp = [0.0, 1.0, 2.0, 4.0];
        let fp = [0.0, 2.0, 3.0, -1.0];
        let q = [0.5, 1.0, 1.5, 3.0, 3.999];
        let got = interp_linear(&q, &xp, &fp).unwrap();
        for (g, w) in got.iter().zip(&NP_INTERP_INSIDE) {
            assert_relative_eq!(*g, *w, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_interp_linear_extrap_agrees_inside_the_range() {
        let xp = [0.0, 1.0, 2.0, 4.0];
        let fp = [0.0, 2.0, 3.0, -1.0];
        let q = [0.5, 1.0, 1.5, 3.0, 3.999];
        assert_eq!(
            interp_linear(&q, &xp, &fp).unwrap(),
            interp_linear_extrap(&q, &xp, &fp).unwrap()
        );
    }

    #[test]
    fn test_interp_linear_extrap_clamps_outside_the_range() {
        let xp = [0.0, 1.0, 2.0, 4.0];
        let fp = [0.0, 2.0, 3.0, -1.0];
        let q = [-3.0, -0.001, 4.5, 10.0];
        let got = interp_linear_extrap(&q, &xp, &fp).unwrap();
        for (g, w) in got.iter().zip(&NP_INTERP_OUTSIDE) {
            assert_relative_eq!(*g, *w, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_interp_linear_hits_the_knots_exactly() {
        let xp = [0.0, 1.0, 2.0, 4.0];
        let fp = [0.0, 2.0, 3.0, -1.0];
        assert_eq!(interp_linear(&xp, &xp, &fp).unwrap(), fp.to_vec());
    }

    #[test]
    fn test_interp_linear_empty_query_is_empty() {
        let xp = [0.0, 1.0];
        let fp = [0.0, 2.0];
        assert!(interp_linear(&[], &xp, &fp).unwrap().is_empty());
        assert!(interp_linear_extrap(&[], &xp, &fp).unwrap().is_empty());
    }

    // -- error branches --

    #[test]
    fn test_fmm_spline_rejects_empty_input() {
        assert!(matches!(
            fmm_spline(&[], &[]).unwrap_err(),
            EdgeErrors::InvalidArgument(_)
        ));
    }

    #[test]
    fn test_fmm_spline_rejects_a_single_knot() {
        assert!(matches!(
            fmm_spline(&[1.0], &[2.0]).unwrap_err(),
            EdgeErrors::InvalidArgument(_)
        ));
    }

    #[test]
    fn test_fmm_spline_rejects_unsorted_knots() {
        assert!(matches!(
            fmm_spline(&[0.0, 2.0, 1.0], &[1.0, 2.0, 3.0]).unwrap_err(),
            EdgeErrors::InvalidArgument(_)
        ));
    }

    #[test]
    fn test_fmm_spline_rejects_duplicated_knots() {
        assert!(matches!(
            fmm_spline(&[0.0, 1.0, 1.0], &[1.0, 2.0, 3.0]).unwrap_err(),
            EdgeErrors::InvalidArgument(_)
        ));
    }

    #[test]
    fn test_fmm_spline_rejects_mismatched_y() {
        assert!(matches!(
            fmm_spline(&[0.0, 1.0, 2.0], &[1.0, 2.0]).unwrap_err(),
            EdgeErrors::LengthMismatch {
                name: "y",
                expected: 3,
                got: 2
            }
        ));
    }

    #[test]
    fn test_maximize_interpolant_rejects_mismatched_y() {
        assert!(matches!(
            maximize_interpolant(&[0.0, 1.0, 2.0], &[1.0]).unwrap_err(),
            EdgeErrors::LengthMismatch { .. }
        ));
    }

    #[test]
    fn test_maximize_interpolant_many_rejects_zero_genes() {
        assert!(matches!(
            maximize_interpolant_many(&[0.0, 1.0], &[], 0).unwrap_err(),
            EdgeErrors::MustBePositive(_)
        ));
    }

    #[test]
    fn test_maximize_interpolant_many_rejects_wrong_matrix_size() {
        assert!(matches!(
            maximize_interpolant_many(&[0.0, 1.0, 2.0], &[1.0, 2.0, 3.0, 4.0], 2).unwrap_err(),
            EdgeErrors::LengthMismatch {
                expected: 6,
                got: 4,
                ..
            }
        ));
    }

    #[test]
    fn test_maximize_interpolant_many_rejects_bad_grid() {
        assert!(matches!(
            maximize_interpolant_many(&[1.0], &[1.0], 1).unwrap_err(),
            EdgeErrors::InvalidArgument(_)
        ));
    }

    #[test]
    fn test_natural_spline_basis_rejects_empty_x() {
        assert!(matches!(
            natural_spline_basis(&[], 5).unwrap_err(),
            EdgeErrors::InvalidArgument(_)
        ));
    }

    #[test]
    fn test_natural_spline_basis_rejects_df_below_two() {
        for df in [0, 1] {
            assert!(matches!(
                natural_spline_basis(&[1.0, 2.0, 3.0], df).unwrap_err(),
                EdgeErrors::InvalidArgument(_)
            ));
        }
    }

    #[test]
    fn test_interp_linear_rejects_points_outside_the_range() {
        let xp = [0.0, 1.0, 2.0];
        let fp = [0.0, 1.0, 4.0];
        assert!(matches!(
            interp_linear(&[0.5, 2.5], &xp, &fp).unwrap_err(),
            EdgeErrors::OutsideKnotRange { .. }
        ));
        assert!(matches!(
            interp_linear(&[-0.5], &xp, &fp).unwrap_err(),
            EdgeErrors::OutsideKnotRange { .. }
        ));
    }

    #[test]
    fn test_interp_linear_rejects_too_few_points() {
        assert!(matches!(
            interp_linear(&[0.5], &[0.0], &[1.0]).unwrap_err(),
            EdgeErrors::InvalidArgument(_)
        ));
        assert!(matches!(
            interp_linear_extrap(&[0.5], &[], &[]).unwrap_err(),
            EdgeErrors::InvalidArgument(_)
        ));
    }

    #[test]
    fn test_interp_linear_rejects_mismatched_fp() {
        assert!(matches!(
            interp_linear_extrap(&[0.5], &[0.0, 1.0, 2.0], &[1.0, 2.0]).unwrap_err(),
            EdgeErrors::LengthMismatch { name: "fp", .. }
        ));
    }

    #[test]
    fn test_interp_linear_rejects_unsorted_xp() {
        assert!(matches!(
            interp_linear_extrap(&[0.5], &[0.0, 2.0, 1.0], &[1.0, 2.0, 3.0]).unwrap_err(),
            EdgeErrors::InvalidArgument(_)
        ));
        assert!(matches!(
            interp_linear_extrap(&[0.5], &[0.0, 1.0, 1.0], &[1.0, 2.0, 3.0]).unwrap_err(),
            EdgeErrors::InvalidArgument(_)
        ));
    }

    // -- tiny dense helpers, tests only --

    /// `A^T A` for a row-major `n` by `p` matrix.
    fn gram_matrix(a: &[f64], n: usize, p: usize) -> Vec<f64> {
        let mut g = vec![0.0; p * p];
        for i in 0..n {
            for j in 0..p {
                for k in 0..p {
                    g[j * p + k] += a[i * p + j] * a[i * p + k];
                }
            }
        }
        g
    }

    /// `A^T v` for a row-major `n` by `p` matrix.
    fn cross_product(a: &[f64], n: usize, p: usize, v: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; p];
        for i in 0..n {
            for j in 0..p {
                out[j] += a[i * p + j] * v[i];
            }
        }
        out
    }

    /// Gaussian elimination with partial pivoting on a small dense system.
    fn solve_spd(a: &[f64], p: usize, rhs: &[f64]) -> Vec<f64> {
        let mut m = a.to_vec();
        let mut b = rhs.to_vec();
        for col in 0..p {
            let pivot = (col..p)
                .max_by(|&r1, &r2| m[r1 * p + col].abs().total_cmp(&m[r2 * p + col].abs()))
                .unwrap();
            if pivot != col {
                for k in 0..p {
                    m.swap(col * p + k, pivot * p + k);
                }
                b.swap(col, pivot);
            }
            let d = m[col * p + col];
            for row in (col + 1)..p {
                let f = m[row * p + col] / d;
                for k in col..p {
                    m[row * p + k] -= f * m[col * p + k];
                }
                b[row] -= f * b[col];
            }
        }
        let mut out = vec![0.0; p];
        for col in (0..p).rev() {
            let mut s = b[col];
            for k in (col + 1)..p {
                s -= m[col * p + k] * out[k];
            }
            out[col] = s / m[col * p + col];
        }
        out
    }
}
