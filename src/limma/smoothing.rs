//! edgeR's column-wise smoothers, `locfitByCol` and `loessByCol`.
//!
//! Both take an `(n_rows, n_cols)` matrix and smooth **down each column
//! independently** against one shared covariate of length `n_rows`. In
//! `estimateDisp` the rows are genes and the columns are the points of the
//! dispersion grid, so a single call smooths every gene's adjusted profile
//! likelihood curve into a trend.
//!
//! The two are different algorithms despite the similar names:
//!
//! * [`locfit_by_col`] is a port of the `locfit` package as edgeR drives it:
//!   `locfit(y ~ x, weights, alpha = span, deg = degree)` with locfit's default
//!   `rbox()` evaluation structure. locfit does **not** fit at every row. It
//!   builds an adaptive binary subdivision of the covariate range, fits at the
//!   cell corners only, and interpolates in between, linearly for `degree = 0`
//!   and by cubic Hermite on the fitted slope for `degree = 1`. Reproducing
//!   that structure is what buys parity with edgeR; a local fit evaluated at
//!   every row instead differs from edgeR by a couple of percent.
//! * [`loess_by_col`] is a port of edgeR's own `src/R_loess_by_col.cpp`. It is
//!   a tricube-weighted moving average with a frame that slides forwards once
//!   across the data, and it also returns the leverage of each row in its own
//!   fit, which `estimateDisp` needs for the weighted likelihood.
//!
//! ### No grid variants
//!
//! Neither function takes a set of evaluation points. Both edgeR routines are
//! only ever asked for fitted values at the rows they were given, and
//! `locfit_by_col` already evaluates on a grid internally, locfit's adaptive
//! tree. Bolting a second, differently spaced grid on top of it would be a
//! knob nothing calls that quietly stops agreeing with edgeR.
//!
//! ### Parallelism
//!
//! Genes are the crate's usual parallel axis. Not here: the rows are the data
//! points of one smooth, not independent units of work, and the columns are the
//! independent ones. Columns are the wrong axis too, because there are only a
//! couple of dozen of them and they share the expensive part of the work, the
//! window search and the bandwidth. So both routines fuse all columns into one
//! pass over the rows and split the rows instead:
//!
//! * `locfit_by_col` fans the vertex fits out over the tree's vertices, each of
//!   which scans the rows once for every column at once, then fans the final
//!   interpolation out over rows.
//! * `loess_by_col` computes the sliding frame in one sequential `O(n_rows)`
//!   sweep, because it is loop-carried, then fans the weighted sums out over
//!   rows. Those sums are the `O(n_rows * span * n_rows * n_cols)` part and the
//!   only part worth splitting.
//!
//! ### References
//!
//! Loader, *Local Regression and Likelihood*, Springer, 1999

use rayon::prelude::*;

use crate::errors::EdgeErrors;

/////////////////////////
// Constants and knobs //
/////////////////////////

/// Cell refinement threshold of locfit's `rbox()` evaluation structure.
///
/// A cell is split while its width exceeds `cut` times the smallest bandwidth
/// at its two corners, so a smaller `cut` means more vertices and a finer
/// interpolation. locfit's default is `0.8` and edgeR never overrides it.
const LOCFIT_CELL_CUT: f64 = 0.8;

/// Roundoff guard inside locfit's nearest-neighbour count.
///
/// locfit takes `k = (int)(n * alpha + 1e-12)`, so a span that ought to land on
/// a whole number of points but comes out a hair under it is not truncated down
/// by one. Dropping this changes `k` for spans like `0.3` on some `n`, and with
/// it the bandwidth at every vertex.
const LOCFIT_NN_EPS: f64 = 1e-12;

/// Degeneracy threshold lifted verbatim from edgeR's `utils.h` (`low_value`).
///
/// Guards two comparisons in `loess_by_col`: the relative tolerance with which
/// the sliding frame is allowed to move forwards through a run of near-equal
/// covariate values, and the frame half-width below which every point in the
/// frame is treated as coincident and given equal weight.
const LOESS_LOW_VALUE: f64 = 1e-10;

/// Work below which the fan-out over rows stays sequential.
///
/// Measured as `n_rows * n_cols`, the size of the output. A rayon fork costs
/// more than it saves on the few hundred rows a unit test uses; the 20 000
/// genes by 21 grid points that `estimateDisp` produces is worth splitting.
const PARALLEL_WORK_THRESHOLD: usize = 1 << 14;

////////////////
// Public API //
////////////////

/// What [`loess_by_col`] produces.
#[derive(Clone, Debug)]
pub struct LoessFit {
    /// Smoothed matrix, row-major `n_rows * n_cols`, same shape as the input.
    pub fitted: Vec<f64>,
    /// Weight each row gave to itself in its own local fit, one per row. This
    /// is the diagonal of the smoother matrix, which `estimateDisp` uses as the
    /// effective number of observations behind each smoothed value.
    pub leverages: Vec<f64>,
}

///////////////////////
// Shared validation //
///////////////////////

/// Checks the matrix shape and returns the covariate to smooth against.
///
/// ### Params
///
/// * `y` - Row-major matrix, expected to hold `n_rows * n_cols` values
/// * `n_rows` - Number of rows
/// * `n_cols` - Number of columns
/// * `x` - Covariate, one value per row, or `None` for `1..=n_rows`
/// * `span` - Proportion of the rows inside each local window
///
/// ### Returns
///
/// The covariate as an owned vector, or [`EdgeErrors`] if the matrix is empty,
/// its length disagrees with the stated shape, the covariate is the wrong
/// length, or `span` falls outside `(0, 1]`.
fn validate(
    y: &[f64],
    n_rows: usize,
    n_cols: usize,
    x: Option<&[f64]>,
    span: f64,
) -> Result<Vec<f64>, EdgeErrors> {
    if n_rows == 0 || n_cols == 0 {
        return Err(EdgeErrors::EmptyCounts {
            n_genes: n_rows,
            n_samples: n_cols,
        });
    }
    if y.len() != n_rows * n_cols {
        return Err(EdgeErrors::ShapeMismatch {
            expected: (n_rows, n_cols),
            got: (y.len(), 1),
        });
    }
    if !(span > 0.0 && span <= 1.0) {
        return Err(EdgeErrors::InvalidArgument(format!(
            "span must lie in (0, 1]; got {span}"
        )));
    }
    match x {
        Some(x) if x.len() != n_rows => Err(EdgeErrors::LengthMismatch {
            name: "x",
            expected: n_rows,
            got: x.len(),
        }),
        Some(x) => Ok(x.to_vec()),
        // edgeR's own default covariate is the row number, `1:ntags`.
        None => Ok((1..=n_rows).map(|i| i as f64).collect()),
    }
}

/// Checks prior weights and expands `None` into all ones.
///
/// ### Params
///
/// * `weights` - Prior weights, one per row, or `None`
/// * `n_rows` - Number of rows
///
/// ### Returns
///
/// The weights as an owned vector, or [`EdgeErrors`] if the length disagrees or
/// any weight is negative.
fn validate_weights(weights: Option<&[f64]>, n_rows: usize) -> Result<Vec<f64>, EdgeErrors> {
    let Some(w) = weights else {
        return Ok(vec![1.0; n_rows]);
    };
    if w.len() != n_rows {
        return Err(EdgeErrors::LengthMismatch {
            name: "weights",
            expected: n_rows,
            got: w.len(),
        });
    }
    crate::limma::check_nonneg_weights(w)?;
    Ok(w.to_vec())
}

////////////////////////////
// locfit: local fitting  //
////////////////////////////

/// One evaluation vertex of locfit's adaptive tree.
#[derive(Clone, Copy, Debug)]
struct Vertex {
    /// Position of the vertex on the covariate axis.
    x: f64,
    /// Nearest-neighbour bandwidth of the local fit performed here.
    h: f64,
}

/// The two halves a split cell was divided into.
///
/// The vertex added at the midpoint is not recorded separately; it is the right
/// end of the lower child and the left end of the upper one.
#[derive(Clone, Copy, Debug)]
struct Split {
    /// Child cell covering the lower half.
    low: usize,
    /// Child cell covering the upper half.
    high: usize,
}

/// One cell of locfit's adaptive tree: an interval between two vertices.
#[derive(Clone, Copy, Debug)]
struct Cell {
    /// Vertex at the left end of the interval.
    left: usize,
    /// Vertex at the right end of the interval.
    right: usize,
    /// `None` for a terminal cell, which is where interpolation happens.
    split: Option<Split>,
}

/// locfit's tricube kernel, `W()` with `ker = WTCUB`.
///
/// A zero bandwidth is not an error: it means every neighbour has collapsed
/// onto the fitting point, and locfit then admits exactly the coincident
/// observations at full weight.
///
/// ### Params
///
/// * `dist` - Non-negative distance from the fitting point
/// * `h` - Bandwidth
///
/// ### Returns
///
/// `(1 - u^3)^3` for `u = dist / h` below one, and zero at or beyond it.
#[inline]
fn tricube(dist: f64, h: f64) -> f64 {
    if h == 0.0 {
        return if dist == 0.0 { 1.0 } else { 0.0 };
    }
    let u = dist / h;
    if u > 1.0 {
        return 0.0;
    }
    let t = 1.0 - u * u * u;
    t * t * t
}

/// Nearest-neighbour bandwidth at one fitting point.
///
/// locfit's `compbandwid`: the distance to the `k`-th nearest observation. The
/// greedy window walk in locfit's `nbhd1` gives the same number, since a window
/// grown towards whichever neighbour is closer holds exactly the `k` nearest
/// points and its half-width is their largest distance.
///
/// ### Params
///
/// * `xs` - Covariate values, in any order
/// * `xv` - Point the local fit is centred on
/// * `k` - Number of neighbours the bandwidth must reach
/// * `scratch` - Reused buffer for the distances; contents are overwritten
///
/// ### Returns
///
/// The bandwidth, which is zero when the `k` nearest observations all sit on
/// `xv`.
fn nn_bandwidth(xs: &[f64], xv: f64, k: usize, scratch: &mut Vec<f64>) -> f64 {
    scratch.clear();
    scratch.extend(xs.iter().map(|xi| (xi - xv).abs()));
    if k < xs.len() {
        // `k >= 1` by construction, so `k - 1` is a valid index.
        let (_, kth, _) = scratch.select_nth_unstable_by(k - 1, f64::total_cmp);
        *kth
    } else {
        // `span <= 1` forces `k <= n`, so this is `k == n`, where locfit's
        // inflation factor `(k / n)^(1 / d)` is exactly one.
        scratch.iter().copied().fold(0.0_f64, f64::max)
    }
}

/// How many vertices locfit will allocate room for.
///
/// `atree_guessnv` with locfit's defaults (`cut = 0.8`, `maxk = 100`) in one
/// dimension. Exceeding it makes locfit abort with "out of vertex space", so
/// the same budget is enforced here rather than subdividing forever on a
/// covariate with a pathologically tight cluster.
///
/// ### Params
///
/// * `span` - Nearest-neighbour fraction
///
/// ### Returns
///
/// The vertex budget.
fn vertex_capacity(span: f64) -> usize {
    let a0 = if span > 1.0 { 1.0 } else { 1.0 / span };
    ((5.0 * a0 / LOCFIT_CELL_CUT + 1.0) * 2.0) as usize
}

/// Decides whether a cell still needs splitting.
///
/// locfit's `atree_split` in one dimension: score the cell by its width in
/// units of the smaller of the two corner bandwidths and split while that
/// exceeds [`LOCFIT_CELL_CUT`]. A cell whose corners both have zero bandwidth
/// is scored against the full covariate range instead, which is locfit's way of
/// still refining where the data is degenerate.
///
/// ### Params
///
/// * `vertices` - All vertices built so far
/// * `cell` - Cell under test
/// * `range` - Width of the whole covariate range
///
/// ### Returns
///
/// Whether to split. A non-finite score (an empty covariate range) is a `false`,
/// matching C's comparison against `NaN`.
fn needs_split(vertices: &[Vertex], cell: &Cell, range: f64) -> bool {
    let (hl, hr) = (vertices[cell.left].h, vertices[cell.right].h);
    let hmin = match (hl > 0.0, hr > 0.0) {
        (true, true) => hl.min(hr),
        (true, false) => hl,
        (false, true) => hr,
        (false, false) => 0.0,
    };
    let width = vertices[cell.right].x - vertices[cell.left].x;
    let score = if hmin == 0.0 {
        2.0 * width / range
    } else {
        width / hmin
    };
    LOCFIT_CELL_CUT < score
}

/// Builds locfit's adaptive tree over the covariate range.
///
/// The root cell spans `[min(x), max(x)]` with a vertex at each end, and every
/// split adds one vertex at the midpoint. Vertices are numbered depth-first,
/// lower half before upper, exactly as locfit's `atree_grow` numbers them, so
/// the vertex budget bites at the same point.
///
/// The tree depends only on the covariate, the span and the degree, never on
/// the responses, so it is built once and shared by every column.
///
/// ### Params
///
/// * `xs` - Covariate values, in any order
/// * `k` - Nearest-neighbour count behind every bandwidth
/// * `capacity` - Vertex budget from [`vertex_capacity`]
///
/// ### Returns
///
/// The vertices and the cells, cell zero being the root, or
/// [`EdgeErrors::InvalidArgument`] once the budget is exhausted.
fn build_tree(
    xs: &[f64],
    k: usize,
    capacity: usize,
) -> Result<(Vec<Vertex>, Vec<Cell>), EdgeErrors> {
    let lower = xs.iter().copied().fold(f64::INFINITY, f64::min);
    let upper = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = upper - lower;

    let mut scratch = Vec::with_capacity(xs.len());
    let mut vertices = vec![
        Vertex {
            x: lower,
            h: nn_bandwidth(xs, lower, k, &mut scratch),
        },
        Vertex {
            x: upper,
            h: nn_bandwidth(xs, upper, k, &mut scratch),
        },
    ];
    let mut cells = vec![Cell {
        left: 0,
        right: 1,
        split: None,
    }];

    // Depth-first, lower half first: pushing the upper child before the lower
    // one makes the stack pop them in locfit's order.
    let mut stack = vec![0usize];
    while let Some(c) = stack.pop() {
        if !needs_split(&vertices, &cells[c], range) {
            continue;
        }
        if vertices.len() >= capacity {
            return Err(EdgeErrors::InvalidArgument(format!(
                "locfit ran out of evaluation vertices after {capacity}; the covariate has a \
                 cluster too tight for this span"
            )));
        }

        let (left, right) = (cells[c].left, cells[c].right);
        let mid_x = (vertices[left].x + vertices[right].x) / 2.0;
        let mid = vertices.len();
        vertices.push(Vertex {
            x: mid_x,
            h: nn_bandwidth(xs, mid_x, k, &mut scratch),
        });

        let low = cells.len();
        cells.push(Cell {
            left,
            right: mid,
            split: None,
        });
        let high = cells.len();
        cells.push(Cell {
            left: mid,
            right,
            split: None,
        });
        cells[c].split = Some(Split { low, high });
        stack.push(high);
        stack.push(low);
    }

    Ok((vertices, cells))
}

/// Fits the local polynomial at one vertex, for every column at once.
///
/// The window search, the tricube weights and the moment sums `s0`, `s1`, `s2`
/// depend only on the covariate, so they are computed once and shared across
/// the columns; only the two right-hand sides are per column. That is the whole
/// reason columns are not the parallel axis.
///
/// Degree zero is a weighted mean, degree one the weighted least-squares line
/// in `x - xv` evaluated at `xv`, whose intercept and slope are what locfit
/// stores at the vertex. A singular normal matrix, which happens when every
/// point in the window sits on `xv`, falls back to the weighted mean and a zero
/// slope; that is what locfit's eigenvalue-based solve produces there.
///
/// ### Params
///
/// * `xs` - Covariate values, in any order
/// * `y` - Row-major responses, `xs.len() * n_cols`
/// * `n_cols` - Number of columns
/// * `weights` - Prior weights, one per row
/// * `vertex` - Vertex to fit at
/// * `degree` - Zero or one
/// * `coef0` - Output slice of `n_cols` fitted values
/// * `coef1` - Output slice of `n_cols` fitted slopes, zeroed for degree zero
#[allow(clippy::too_many_arguments)]
fn fit_vertex(
    xs: &[f64],
    y: &[f64],
    n_cols: usize,
    weights: &[f64],
    vertex: Vertex,
    degree: usize,
    coef0: &mut [f64],
    coef1: &mut [f64],
) {
    coef0.fill(0.0);
    coef1.fill(0.0);

    let (mut s0, mut s1, mut s2) = (0.0, 0.0, 0.0);
    for (i, row) in y.chunks_exact(n_cols).enumerate() {
        let dx = xs[i] - vertex.x;
        let kernel = tricube(dx.abs(), vertex.h);
        if kernel == 0.0 {
            continue;
        }
        let w = kernel * weights[i];
        s0 += w;
        s1 += w * dx;
        s2 += w * dx * dx;
        for ((r0, r1), &v) in coef0.iter_mut().zip(coef1.iter_mut()).zip(row) {
            *r0 += w * v;
            *r1 += w * dx * v;
        }
    }

    if s0 <= 0.0 {
        coef0.fill(0.0);
        coef1.fill(0.0);
        return;
    }

    // Cauchy-Schwarz makes the determinant non-negative, so a non-positive one
    // means the window has no spread in x and only an intercept is estimable.
    let det = if degree == 0 { 0.0 } else { s0 * s2 - s1 * s1 };
    if det <= 0.0 {
        for c in coef0.iter_mut() {
            *c /= s0;
        }
        coef1.fill(0.0);
        return;
    }
    for (c0, c1) in coef0.iter_mut().zip(coef1.iter_mut()) {
        let (r0, r1) = (*c0, *c1);
        *c0 = (s2 * r0 - s1 * r1) / det;
        *c1 = (s0 * r1 - s1 * r0) / det;
    }
}

/// locfit's `hermite2`: the cubic Hermite basis on `[0, z]`.
///
/// ### Params
///
/// * `x` - Offset into the interval
/// * `z` - Width of the interval
///
/// ### Returns
///
/// `[phi0, phi1, phi2, phi3]`, weighting the left value, right value, left
/// derivative and right derivative. Outside `[0, z]` it degenerates to a linear
/// extrapolation off the nearer end, and a zero-width interval returns the left
/// value alone.
#[inline]
fn hermite2(x: f64, z: f64) -> [f64; 4] {
    if z == 0.0 {
        return [1.0, 0.0, 0.0, 0.0];
    }
    let h = x / z;
    if h < 0.0 {
        return [1.0, 0.0, h, 0.0];
    }
    if h > 1.0 {
        return [0.0, 1.0, 0.0, h - 1.0];
    }
    let upper = h * h * (3.0 - 2.0 * h);
    [
        1.0 - upper,
        upper,
        h * (1.0 - h) * (1.0 - h),
        h * h * (h - 1.0),
    ]
}

/// Interpolates the vertex fits at one covariate value, for every column.
///
/// locfit's `atree_int` followed by `rectcell_interp`: descend to the terminal
/// cell containing `xq`, then blend its two corners. Degree zero has no
/// derivative to blend with and falls back to linear interpolation, which is
/// why an edgeR `locfitByCol` curve of degree zero is piecewise linear between
/// tree vertices.
///
/// ### Params
///
/// * `vertices` - Tree vertices
/// * `cells` - Tree cells, cell zero being the root
/// * `coef0` - Fitted values, vertex-major `n_vertices * n_cols`
/// * `coef1` - Fitted slopes, same layout
/// * `n_cols` - Number of columns
/// * `degree` - Zero or one
/// * `xq` - Covariate value to evaluate at
/// * `out` - Output slice of `n_cols` values
#[allow(clippy::too_many_arguments)]
fn interpolate_row(
    vertices: &[Vertex],
    cells: &[Cell],
    coef0: &[f64],
    coef1: &[f64],
    n_cols: usize,
    degree: usize,
    xq: f64,
    out: &mut [f64],
) {
    let mut cell = &cells[0];
    while let Some(split) = cell.split {
        let (ll, ur) = (vertices[cell.left].x, vertices[cell.right].x);
        cell = if 2.0 * (xq - ll) < ur - ll {
            &cells[split.low]
        } else {
            &cells[split.high]
        };
    }

    let (l, r) = (cell.left, cell.right);
    let (ll, ur) = (vertices[l].x, vertices[r].x);
    let width = ur - ll;
    let offset = xq - ll;
    let (lo0, hi0) = (&coef0[l * n_cols..], &coef0[r * n_cols..]);

    if degree == 0 {
        for (j, o) in out.iter_mut().enumerate() {
            *o = if width == 0.0 {
                lo0[j]
            } else {
                ((width - offset) * lo0[j] + offset * hi0[j]) / width
            };
        }
        return;
    }

    let mut phi = hermite2(offset, width);
    phi[2] *= width;
    phi[3] *= width;
    let (lo1, hi1) = (&coef1[l * n_cols..], &coef1[r * n_cols..]);
    for (j, o) in out.iter_mut().enumerate() {
        *o = phi[0] * lo0[j] + phi[1] * hi0[j] + phi[2] * lo1[j] + phi[3] * hi1[j];
    }
}

/// Local regression smoother for the columns of a matrix.
///
/// Port of edgeR's `locfitByCol`, which is the `locfit` package driven with its
/// default adaptive-tree evaluation structure. Each column of `y` is smoothed
/// against the shared covariate `x`; the columns never interact.
///
/// edgeR's own escape hatch is reproduced: when `span * n_rows < 2` or there is
/// only one row, there is nothing to smooth and `y` is returned unchanged.
///
/// The smoother is an approximation to the exact local fit by construction, and
/// deliberately so, because that is what edgeR gets from locfit. Local fits are
/// performed only at the corners of an adaptively refined subdivision of the
/// covariate range, typically a couple of dozen of them regardless of how many
/// rows there are, and everything between is interpolated. See the module
/// documentation.
///
/// ### Params
///
/// * `y` - Row-major matrix of responses, `n_rows * n_cols`
/// * `n_rows` - Number of rows, the points of each smooth
/// * `n_cols` - Number of columns, each smoothed independently
/// * `x` - Covariate, one value per row, in any order, or `None` for `1..=n_rows`
/// * `weights` - Prior weights, one per row, or `None` for all ones. Must be
///   non-negative.
/// * `span` - Proportion of the rows behind each local bandwidth, in `(0, 1]`
/// * `degree` - `0` for a local constant, `1` for a local linear fit
///
/// ### Returns
///
/// The smoothed matrix, row-major and the same shape as `y`, or [`EdgeErrors`]
/// if the shape is inconsistent, the input is empty, a length disagrees, a
/// weight is negative, `span` falls outside `(0, 1]`, or `degree` exceeds one.
///
/// ### References
///
/// Loader, *Local Regression and Likelihood*, Springer, 1999
pub fn locfit_by_col(
    y: &[f64],
    n_rows: usize,
    n_cols: usize,
    x: Option<&[f64]>,
    weights: Option<&[f64]>,
    span: f64,
    degree: usize,
) -> Result<Vec<f64>, EdgeErrors> {
    let xs = validate(y, n_rows, n_cols, x, span)?;
    let weights = validate_weights(weights, n_rows)?;
    if degree > 1 {
        return Err(EdgeErrors::InvalidArgument(format!(
            "degree must be 0 or 1; got {degree}"
        )));
    }

    // edgeR bails out here rather than calling locfit at all.
    if span * (n_rows as f64) < 2.0 || n_rows <= 1 {
        return Ok(y.to_vec());
    }

    let k = (n_rows as f64 * span + LOCFIT_NN_EPS) as usize;
    let (vertices, cells) = build_tree(&xs, k, vertex_capacity(span))?;

    let n_vertices = vertices.len();
    let mut coef0 = vec![0.0; n_vertices * n_cols];
    let mut coef1 = vec![0.0; n_vertices * n_cols];
    let parallel = n_rows * n_cols >= PARALLEL_WORK_THRESHOLD;
    let fit = |(v, (c0, c1)): (usize, (&mut [f64], &mut [f64]))| {
        fit_vertex(&xs, y, n_cols, &weights, vertices[v], degree, c0, c1);
    };
    if parallel {
        coef0
            .par_chunks_mut(n_cols)
            .zip(coef1.par_chunks_mut(n_cols))
            .enumerate()
            .for_each(fit);
    } else {
        coef0
            .chunks_mut(n_cols)
            .zip(coef1.chunks_mut(n_cols))
            .enumerate()
            .for_each(fit);
    }

    let mut fitted = vec![0.0; n_rows * n_cols];
    let evaluate = |(i, out): (usize, &mut [f64])| {
        interpolate_row(
            &vertices, &cells, &coef0, &coef1, n_cols, degree, xs[i], out,
        );
    };
    if parallel {
        fitted.par_chunks_mut(n_cols).enumerate().for_each(evaluate);
    } else {
        fitted.chunks_mut(n_cols).enumerate().for_each(evaluate);
    }
    Ok(fitted)
}

////////////////////////////
// loess: moving average  //
////////////////////////////

/// The local frame around one row, as edgeR's C++ leaves it.
#[derive(Clone, Copy, Debug)]
struct Frame {
    /// Last row inside the frame, inclusive.
    end: usize,
    /// Half-width of the frame, which is the tricube bandwidth.
    max_dist: f64,
}

/// Slides edgeR's frame once across the sorted covariate.
///
/// This is the loop-carried part of `R_loess_by_col.cpp` and the reason the
/// routine is `O(n_rows)` here rather than `O(n_rows * span * n_rows)`: the
/// frame only ever advances, because the frame that minimises the largest
/// distance for one row can only be at or ahead of the one that minimised it
/// for the previous row. The relative tolerance on the advance is edgeR's
/// protection against a run of near-equal covariate values walling the frame
/// off behind a rounding difference.
///
/// ### Params
///
/// * `xs` - Covariate values, sorted ascending
/// * `nspan` - Number of rows the frame holds, at least two
///
/// ### Returns
///
/// One [`Frame`] per row, in row order.
fn slide_frames(xs: &[f64], nspan: usize) -> Vec<Frame> {
    let total = xs.len();
    let mut frames = Vec::with_capacity(total);
    let mut end = nspan - 1;

    for cur in 0..total {
        if cur > end {
            end = cur;
        }
        let point = xs[cur];
        // `end >= nspan - 1` holds throughout, so both indices are in range.
        let back = point - xs[end + 1 - nspan];
        let front = xs[end] - point;
        let mut max_dist = if back > front { back } else { front };

        while end < total - 1 && cur + nspan - 1 > end {
            let back = point - xs[end + 2 - nspan];
            let front = xs[end + 1] - point;
            let next_max = if back > front { back } else { front };
            let diff = (next_max - max_dist) / max_dist;
            if diff > LOESS_LOW_VALUE {
                break;
            }
            if diff < 0.0 {
                max_dist = next_max;
            }
            end += 1;
        }
        frames.push(Frame { end, max_dist });
    }
    frames
}

/// Fills one row of the fitted matrix and its leverage.
///
/// Sums are accumulated from the far end of the frame back towards row zero,
/// which is edgeR's order and therefore the one that reproduces its rounding.
/// Rows below the frame's reach are skipped rather than visited and discarded:
/// a point further than `max_dist` from the fitting point gets a negative
/// tricube weight, which edgeR drops, so trimming the scan changes nothing but
/// the running time.
///
/// ### Params
///
/// * `xs` - Covariate values, sorted ascending
/// * `y` - Row-major responses in the same order
/// * `n_cols` - Number of columns
/// * `cur` - Row being fitted
/// * `frame` - Frame for this row from [`slide_frames`]
/// * `out` - Output slice of `n_cols` fitted values
///
/// ### Returns
///
/// The leverage of `cur` in its own fit.
fn loess_row(
    xs: &[f64],
    y: &[f64],
    n_cols: usize,
    cur: usize,
    frame: Frame,
    out: &mut [f64],
) -> f64 {
    let point = xs[cur];
    let coincident = frame.max_dist <= LOESS_LOW_VALUE;
    let start = if coincident {
        0
    } else {
        xs.partition_point(|v| point - v > frame.max_dist)
    };

    out.fill(0.0);
    let mut total_weight = 0.0;
    let mut leverage = -1.0;
    for m in (start..=frame.end).rev() {
        let rel = if coincident {
            0.0
        } else {
            (xs[m] - point).abs() / frame.max_dist
        };
        let t = 1.0 - rel * rel * rel;
        let weight = t * t * t;
        if weight < 0.0 {
            continue;
        }
        total_weight += weight;
        for (o, &v) in out.iter_mut().zip(&y[m * n_cols..(m + 1) * n_cols]) {
            *o += weight * v;
        }
        if m == cur {
            leverage = weight;
        }
    }

    for o in out.iter_mut() {
        *o /= total_weight;
    }
    leverage / total_weight
}

/// Fits a degree-zero lowess curve to each column of a matrix.
///
/// Port of edgeR's `loessByCol` and the `R_loess_by_col.cpp` behind it: a
/// tricube-weighted moving average whose window is the tightest run of
/// `floor(span * n_rows)` consecutive rows around each row. Alongside the
/// smoothed matrix it returns the leverages, the weight each row gave itself,
/// which is the diagonal of the smoother matrix.
///
/// edgeR's C++ requires `x` to be sorted and does not check. This does not: the
/// rows are sorted by `x` internally and the results mapped back, so `fitted`
/// and `leverages` line up with the caller's own row order. On sorted input,
/// which is how edgeR always calls it, that is the identity permutation and the
/// answer is bit for bit edgeR's.
///
/// As in edgeR, a span too small to hold two rows leaves `y` untouched and
/// gives every row a leverage of one.
///
/// ### Params
///
/// * `y` - Row-major matrix of responses, `n_rows * n_cols`
/// * `n_rows` - Number of rows, the points of each smooth
/// * `n_cols` - Number of columns, each smoothed independently
/// * `x` - Covariate, one value per row, in any order, or `None` for `1..=n_rows`
/// * `span` - Proportion of the rows inside each window, in `(0, 1]`
///
/// ### Returns
///
/// The smoothed matrix and the leverages, or [`EdgeErrors`] if the shape is
/// inconsistent, the input is empty, `x` is the wrong length, or `span` falls
/// outside `(0, 1]`.
///
/// ### References
///
/// Cleveland, Journal of the American Statistical Association, 1979
pub fn loess_by_col(
    y: &[f64],
    n_rows: usize,
    n_cols: usize,
    x: Option<&[f64]>,
    span: f64,
) -> Result<LoessFit, EdgeErrors> {
    let x = validate(y, n_rows, n_cols, x, span)?;

    let nspan = ((span * n_rows as f64).floor() as usize).min(n_rows);
    if nspan <= 1 {
        return Ok(LoessFit {
            fitted: y.to_vec(),
            leverages: vec![1.0; n_rows],
        });
    }

    // Stable sort by x, matching R's `order`, so tied x keep their input order.
    let mut order: Vec<usize> = (0..n_rows).collect();
    order.sort_by(|&a, &b| x[a].total_cmp(&x[b]));
    let xs: Vec<f64> = order.iter().map(|&i| x[i]).collect();
    let ys: Vec<f64> = order
        .iter()
        .flat_map(|&i| y[i * n_cols..(i + 1) * n_cols].iter().copied())
        .collect();

    let frames = slide_frames(&xs, nspan);
    let mut sorted_fit = vec![0.0; n_rows * n_cols];
    let mut sorted_lev = vec![0.0; n_rows];

    let smooth = |(i, (out, lev)): (usize, (&mut [f64], &mut f64))| {
        *lev = loess_row(&xs, &ys, n_cols, i, frames[i], out);
    };
    if n_rows * n_cols >= PARALLEL_WORK_THRESHOLD {
        sorted_fit
            .par_chunks_mut(n_cols)
            .zip(sorted_lev.par_iter_mut())
            .enumerate()
            .for_each(smooth);
    } else {
        sorted_fit
            .chunks_mut(n_cols)
            .zip(sorted_lev.iter_mut())
            .enumerate()
            .for_each(smooth);
    }

    // Back to the caller's row order.
    let mut fitted = vec![0.0; n_rows * n_cols];
    let mut leverages = vec![0.0; n_rows];
    for (k, &i) in order.iter().enumerate() {
        fitted[i * n_cols..(i + 1) * n_cols]
            .copy_from_slice(&sorted_fit[k * n_cols..(k + 1) * n_cols]);
        leverages[i] = sorted_lev[k];
    }

    Ok(LoessFit { fitted, leverages })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Relative tolerance against edgeR, applied with no absolute floor.
    ///
    /// The worst disagreement actually observed across every case below is
    /// 2.6e-15 for `locfit_by_col` and 4.0e-16 for `loess_by_col`.
    /// `loess_by_col` performs the same additions in the same order as edgeR's
    /// C++ and comes back bit for bit identical on most of these cases;
    /// `locfit_by_col` solves each local regression directly where locfit goes
    /// through an eigendecomposition, which costs a few ulps. The tolerance is
    /// loosened to 1e-12 only so a different optimiser or libm cannot make the
    /// suite flaky.
    const TOL: f64 = 1e-12;

    /// The shared 24-row, two-column fixture, built from exactly representable
    /// arithmetic so that R and Rust construct bit-identical inputs.
    ///
    /// R: `i <- 1:24; x <- i/32;`
    /// `y1 <- 3*x - 2*x*x + ((i*i*37) %% 101)/500 - 0.1;`
    /// `y2 <- 1 + ((i*i*53) %% 89)/400; Y <- cbind(y1, y2);`
    /// `W <- 1 + ((i*7) %% 5)`
    ///
    /// ### Returns
    ///
    /// The covariate, the row-major matrix and the prior weights.
    fn fixture() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut x = Vec::with_capacity(24);
        let mut y = Vec::with_capacity(48);
        let mut w = Vec::with_capacity(24);
        for i in 1..=24u64 {
            let xi = i as f64 / 32.0;
            x.push(xi);
            y.push(3.0 * xi - 2.0 * xi * xi + ((i * i * 37) % 101) as f64 / 500.0 - 0.1);
            y.push(1.0 + ((i * i * 53) % 89) as f64 / 400.0);
            w.push(1.0 + ((i * 7) % 5) as f64);
        }
        (x, y, w)
    }

    /// Eight distinct covariate values, three rows each.
    ///
    /// R: `j <- 1:24; xt <- floor((j-1)/3)/8;`
    /// `yt1 <- 2*xt + ((j*j*13) %% 97)/300 - 0.16;`
    /// `yt2 <- 0.5 + ((j*j*29) %% 71)/200; Yt <- cbind(yt1, yt2)`
    ///
    /// ### Returns
    ///
    /// The covariate and the row-major matrix.
    fn ties_fixture() -> (Vec<f64>, Vec<f64>) {
        let mut x = Vec::with_capacity(24);
        let mut y = Vec::with_capacity(48);
        for j in 1..=24u64 {
            let xj = ((j - 1) / 3) as f64 / 8.0;
            x.push(xj);
            y.push(2.0 * xj + ((j * j * 13) % 97) as f64 / 300.0 - 0.16);
            y.push(0.5 + ((j * j * 29) % 71) as f64 / 200.0);
        }
        (x, y)
    }

    /// Ten rows sharing one covariate value, where every bandwidth collapses.
    ///
    /// R: `xf <- rep(1, 10); k <- 1:10; yf1 <- k*(k %% 7)/10;`
    /// `yf2 <- 2 + (k %% 3)/4; Yf <- cbind(yf1, yf2)`
    ///
    /// ### Returns
    ///
    /// The covariate and the row-major matrix.
    fn flat_fixture() -> (Vec<f64>, Vec<f64>) {
        let mut x = Vec::with_capacity(10);
        let mut y = Vec::with_capacity(20);
        for k in 1..=10u64 {
            x.push(1.0);
            y.push((k * (k % 7)) as f64 / 10.0);
            y.push(2.0 + (k % 3) as f64 / 4.0);
        }
        (x, y)
    }

    /// Asserts two slices agree elementwise, naming the offending index.
    ///
    /// ### Params
    ///
    /// * `got` - Values produced here
    /// * `want` - Values produced by edgeR
    fn assert_close(got: &[f64], want: &[f64]) {
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!(
                (g - w).abs() <= TOL * w.abs().max(g.abs()),
                "element {i}: got {g}, want {w}"
            );
        }
    }

    /// Builds a wide fixture large enough to cross [`PARALLEL_WORK_THRESHOLD`].
    ///
    /// ### Params
    ///
    /// * `n_rows` - Number of rows
    /// * `n_cols` - Number of columns
    ///
    /// ### Returns
    ///
    /// The covariate and the row-major matrix.
    fn wide_fixture(n_rows: usize, n_cols: usize) -> (Vec<f64>, Vec<f64>) {
        let mut x = Vec::with_capacity(n_rows);
        let mut y = Vec::with_capacity(n_rows * n_cols);
        for i in 0..n_rows as u64 {
            let xi = i as f64 / n_rows as f64;
            x.push(xi);
            for j in 0..n_cols as u64 {
                y.push(3.0 * xi - 2.0 * xi * xi + ((i * i * 37 + j * 13) % 101) as f64 / 500.0);
            }
        }
        (x, y)
    }

    /// Rscript, after [`fixture`]'s preamble:
    /// cat(t(edgeR:::locfitByCol(Y, x, span=0.3, degree=0)), sep=",")
    #[rustfmt::skip]
    const LOCFIT_SPAN03_DEG0: [f64; 48] = [
        0.21938794942242046, 1.111419365226463, 0.2346805090923122,
        1.1134200715540645, 0.27186949894883394, 1.1168789648122928,
        0.3293239734113493, 1.1203657662392044, 0.4094229424330573,
        1.1161089071134846, 0.4853316576486573, 1.0975713962284916,
        0.5615104026645877, 1.0748217786499998, 0.6442132195443843,
        1.0589094570263837, 0.7009607347478165, 1.0609158473946163,
        0.7552702659454068, 1.073657638255966, 0.8154992988036569,
        1.0973999509679544, 0.8736840412943477, 1.1195503228598618,
        0.9268216329726164, 1.142722767836228, 0.9741775090363658,
        1.1642328604640992, 1.0084706227010969, 1.1438846307775865,
        1.0383894263982714, 1.1346223394470678, 1.0628560945602163,
        1.1326762215495478, 1.0750700479851465, 1.133610082136104,
        1.0809699803991726, 1.1352375675045725, 1.088164870601552,
        1.1246152657390927, 1.102838950012972, 1.0865347741055233,
        1.1022763480614508, 1.063874423192388, 1.1003901864757724,
        1.0499245349311213, 1.0996008794883507, 1.044341542191973
    ];

    /// Rscript, after [`fixture`]'s preamble:
    /// cat(t(edgeR:::locfitByCol(Y, x, span=0.5, degree=0)), sep=",")
    #[rustfmt::skip]
    const LOCFIT_SPAN05_DEG0: [f64; 48] = [
        0.34400480722818716, 1.1029588005340973, 0.35833257650888856,
        1.1013796357372314, 0.3726603457895899, 1.0998004709403653,
        0.39190515977841633, 1.0979200496483033, 0.4455692867241183,
        1.09393083288987, 0.4992334136698203, 1.0899416161314368,
        0.5565120057118113, 1.0863322229747645, 0.6246339930426689,
        1.0838623006233754, 0.6927559803735266, 1.0813923782719863,
        0.7558001047882574, 1.0856802306437383, 0.810381124342777,
        1.1012310408873927, 0.8649621438972966, 1.1167818511310468,
        0.9131430573336986, 1.1277548845028598, 0.9549238646519829,
        1.1341501410028318, 0.9967046719702672, 1.140545397502804,
        1.0248966058330207, 1.139493441496038, 1.0449352156224554,
        1.1339731579852292, 1.0649738254118903, 1.1284528744744204,
        1.074414139747434, 1.1178220114661872, 1.080321688931681,
        1.1054876219588126, 1.0862292381159278, 1.0931532324514377,
        1.0877527006785577, 1.0887923332729303, 1.088649865152385,
        1.0855705041414043, 1.0895470296262126, 1.082348675009878
    ];

    /// Rscript, after [`fixture`]'s preamble:
    /// cat(t(edgeR:::locfitByCol(Y, x, span=0.5, degree=1)), sep=",")
    #[rustfmt::skip]
    const LOCFIT_SPAN05_DEG1: [f64; 48] = [
        0.08519658517015832, 1.123840262884358, 0.1655078767321026,
        1.1175766569875414, 0.24586323245552177, 1.111233425131652,
        0.32556496474351015, 1.104160744196573, 0.4052128060336144,
        1.0989889875188859, 0.48345965903709087, 1.0938452229888076,
        0.557739806428577, 1.0840238823881043, 0.6284984920125529,
        1.0756297091052642, 0.6956862557026362, 1.0749170123959095,
        0.7581952678550901, 1.0846179202587196, 0.8147962275181707,
        1.0996962979409786, 0.8670747156071963, 1.1163720408469187,
        0.9168537972771138, 1.1315290906286912, 0.9624913995707429,
        1.1400810608125467, 1.0006316774463186, 1.1429580415992393,
        1.0289716775756732, 1.1415751504033262, 1.050213216713417,
        1.1363117496859327, 1.0664596822121086, 1.1290012990468974,
        1.078439619081621, 1.1179153420307804, 1.0857860990245636,
        1.0985904071469075, 1.0913069979012222, 1.0776842764216772,
        1.0960847055706102, 1.0578017083447342, 1.0993685421715442,
        1.0359325226705396, 1.1028230391830598, 1.0141026845422463
    ];

    /// Rscript, after [`fixture`]'s preamble:
    /// cat(t(edgeR:::locfitByCol(Y, x, span=1.0, degree=1)), sep=",")
    #[rustfmt::skip]
    const LOCFIT_SPAN10_DEG1: [f64; 48] = [
        0.1533357226872843, 1.0935180801736515, 0.21618805130629376,
        1.0950331198172125, 0.2798132731867089, 1.0964117432417602,
        0.3435547456331891, 1.0977817209900351, 0.4067558259503931,
        1.0992708236047777, 0.46875987144298015, 1.101006821628728,
        0.5291114044900106, 1.1030602737958715, 0.5911614676798688,
        1.1044366428337822, 0.6546094506780106, 1.1052232978879317,
        0.7171937967931601, 1.1060716010944722, 0.776652949334041,
        1.1076329145895554, 0.8307253516093772, 1.1105586005093329,
        0.8770407902822764, 1.114564887994291, 0.9145525107841271,
        1.1149715725990779, 0.9456608626576256, 1.1122084748005474,
        0.9730822678684581, 1.1076715737814966, 0.9995331483823101,
        1.1027568487247221, 1.027729926164868, 1.0988602788130206,
        1.058006778575636, 1.0961439547816327, 1.0862770614457309,
        1.0924692958757287, 1.1130928876618358, 1.0880901442594906,
        1.1392510680059353, 1.0833868614629456, 1.1655484132600145,
        1.078739809016122, 1.1927817342060578, 1.0745293484490468
    ];

    /// Rscript, after [`fixture`]'s preamble:
    /// cat(t(edgeR:::locfitByCol(Y, x, weights=W, span=0.5, degree=1)), sep=",")
    #[rustfmt::skip]
    const LOCFIT_WEIGHTED_DEG1: [f64; 48] = [
        0.09111704809496862, 1.1178182745107725, 0.17468701987468724,
        1.1097581001777581, 0.25825806515957916, 1.1016073288784236,
        0.3407173296416437, 1.0931625371674154, 0.42390633407438794,
        1.0863167163458296, 0.5048510305182696, 1.0795818460362796,
        0.577676914712255, 1.0697621140284246, 0.6441858195182493,
        1.0657593059451478, 0.7064601728932193, 1.0711962394596937,
        0.7646057712329637, 1.084569253866259, 0.8194695122142369,
        1.0996220383774602, 0.8724991968606948, 1.113980593262524,
        0.9241933322733106, 1.125671879150522, 0.9695064291717435,
        1.1293876673439116, 1.0068825511060304, 1.1274254420294614,
        1.0358609374042416, 1.123283662907688, 1.0569641734423234,
        1.1195489615835612, 1.0703138490529143, 1.117601487424323,
        1.0767311684822123, 1.1128002012238984, 1.0792753537946802,
        1.0964723932654796, 1.0802763834931752, 1.0781693207465575,
        1.0807338770254236, 1.0617668522161239, 1.0802128097285069,
        1.042819663955161, 1.0797906311775403, 1.0238192779735789
    ];

    /// Rscript, after [`fixture`]'s preamble:
    /// cat(t(edgeR:::locfitByCol(Y, span=0.5, degree=0)), sep=",")
    #[rustfmt::skip]
    const LOCFIT_DEFAULT_X_DEG0: [f64; 48] = [
        0.34400480722818716, 1.1029588005340973, 0.35833257650888856,
        1.1013796357372314, 0.3726603457895899, 1.0998004709403653,
        0.39190515977841633, 1.0979200496483033, 0.4455692867241183,
        1.09393083288987, 0.4992334136698203, 1.0899416161314368,
        0.5565120057118113, 1.0863322229747645, 0.6246339930426689,
        1.0838623006233754, 0.6927559803735266, 1.0813923782719863,
        0.7558001047882574, 1.0856802306437383, 0.810381124342777,
        1.1012310408873927, 0.8649621438972966, 1.1167818511310468,
        0.9131430573336986, 1.1277548845028598, 0.9549238646519829,
        1.1341501410028318, 0.9967046719702672, 1.140545397502804,
        1.0248966058330207, 1.139493441496038, 1.0449352156224554,
        1.1339731579852292, 1.0649738254118903, 1.1284528744744204,
        1.074414139747434, 1.1178220114661872, 1.080321688931681,
        1.1054876219588126, 1.0862292381159278, 1.0931532324514377,
        1.0877527006785577, 1.0887923332729303, 1.088649865152385,
        1.0855705041414043, 1.0895470296262126, 1.082348675009878
    ];

    /// Rscript, after [`ties_fixture`]'s preamble:
    /// cat(t(edgeR:::locfitByCol(Yt, xt, span=0.5, degree=0)), sep=",")
    #[rustfmt::skip]
    const LOCFIT_TIES_DEG0: [f64; 48] = [
        0.14963877177871465, 0.6635106230261268, 0.14963877177871465,
        0.6635106230261268, 0.14963877177871465, 0.6635106230261268,
        0.24888657869138886, 0.6407111119024703, 0.24888657869138886,
        0.6407111119024703, 0.24888657869138886, 0.6407111119024703,
        0.5020892191045134, 0.6205772152338217, 0.5020892191045134,
        0.6205772152338217, 0.5020892191045134, 0.6205772152338217,
        0.7167458258298545, 0.6602127697108336, 0.7167458258298545,
        0.6602127697108336, 0.7167458258298545, 0.6602127697108336,
        0.9685650129212061, 0.6456789884083054, 0.9685650129212061,
        0.6456789884083054, 0.9685650129212061, 0.6456789884083054,
        1.2307436342645328, 0.6145830932498177, 1.2307436342645328,
        0.6145830932498177, 1.2307436342645328, 0.6145830932498177,
        1.4859468434569383, 0.6152866917306774, 1.4859468434569383,
        0.6152866917306774, 1.4859468434569383, 0.6152866917306774,
        1.5783054535563716, 0.6167920878855191, 1.5783054535563716,
        0.6167920878855191, 1.5783054535563716, 0.6167920878855191
    ];

    /// Rscript, after [`ties_fixture`]'s preamble:
    /// cat(t(edgeR:::locfitByCol(Yt, xt, span=0.5, degree=1)), sep=",")
    #[rustfmt::skip]
    const LOCFIT_TIES_DEG1: [f64; 48] = [
        -0.06861640251480788, 0.7167079037422399, -0.06861640251480788,
        0.7167079037422399, -0.06861640251480788, 0.7167079037422399,
        0.2392909748472049, 0.6399827562900017, 0.2392909748472049,
        0.6399827562900017, 0.2392909748472049, 0.6399827562900017,
        0.5133014643172152, 0.6087563749672513, 0.5133014643172152,
        0.6087563749672513, 0.5133014643172152, 0.6087563749672513,
        0.708516887204566, 0.6953885210669604, 0.708516887204566,
        0.6953885210669604, 0.708516887204566, 0.6953885210669604,
        0.9593175885660604, 0.6350679150876141, 0.9593175885660604,
        0.6350679150876141, 0.9593175885660604, 0.6350679150876141,
        1.22948966411524, 0.6131602292844193, 1.22948966411524,
        0.6131602292844193, 1.22948966411524, 0.6131602292844193,
        1.4962036630451512, 0.6152729497854399, 1.4962036630451512,
        0.6152729497854399, 1.4962036630451512, 0.6152729497854399,
        1.785141474801969, 0.6208681848512095, 1.785141474801969,
        0.6208681848512095, 1.785141474801969, 0.6208681848512095
    ];

    /// Rscript, after [`flat_fixture`]'s preamble:
    /// cat(t(edgeR:::locfitByCol(Yf, xf, span=0.5, degree=0)), sep=",")
    #[rustfmt::skip]
    const LOCFIT_FLATX_DEG0: [f64; 20] = [
        1.47, 2.25, 1.47, 2.25, 1.47, 2.25, 1.47, 2.25, 1.47, 2.25, 1.47, 2.25, 1.47,
        2.25, 1.47, 2.25, 1.47, 2.25, 1.47, 2.25
    ];

    /// Rscript, after [`flat_fixture`]'s preamble:
    /// cat(t(edgeR:::locfitByCol(Yf, xf, span=0.5, degree=1)), sep=",")
    #[rustfmt::skip]
    const LOCFIT_FLATX_DEG1: [f64; 20] = [
        1.47, 2.25, 1.47, 2.25, 1.47, 2.25, 1.47, 2.25, 1.47, 2.25, 1.47, 2.25, 1.47,
        2.25, 1.47, 2.25, 1.47, 2.25, 1.47, 2.25
    ];

    /// Rscript -e 'suppressMessages(library(edgeR));
    /// cat(t(edgeR:::locfitByCol(cbind(c(1,3,2,5,4), c(2,1,4,3,6)), (1:5)/5,
    /// span=1.0, degree=1)), sep=",")'
    #[rustfmt::skip]
    const LOCFIT_N5_SPAN10_DEG1: [f64; 10] = [
        1.2917927706870431, 1.4893016570834954, 2.1671842902989855, 2.1979423747461504,
        3.1452420701168613, 2.8547579298831383, 3.80205762525385, 4.19794237474615,
        4.5106983429165055, 5.489301657083496
    ];

    /// Rscript -e 'suppressMessages(library(edgeR));
    /// cat(t(edgeR:::locfitByCol(cbind(c(1,3,2), c(4,5,6)), c(1,2,3),
    /// span=0.5, degree=0)), sep=",")'
    #[rustfmt::skip]
    const LOCFIT_N3_IDENTITY: [f64; 6] = [
        1.0, 4.0, 3.0, 5.0, 2.0, 6.0
    ];

    /// Rscript, after [`fixture`]'s preamble:
    /// cat(t(edgeR:::loessByCol(Y, x, span=0.3)$fitted.values), sep=",")
    #[rustfmt::skip]
    const LOESS_SPAN03_FITTED: [f64; 48] = [
        0.2193879494224205, 1.111419365226463, 0.233018824684498,
        1.113327179121195, 0.25913910815266084, 1.1155358883898008,
        0.32793878641343194, 1.1231435531632459, 0.40611223683682696,
        1.1238357548174402, 0.48437786183682696, 1.103143553163246,
        0.5627356614134319, 1.0679055621198195, 0.648979459323443,
        1.0529055621198196, 0.7037364491251294, 1.058143553163246,
        0.7523770741251294, 1.071097763774014, 0.8151122368368269,
        1.0974513515090512, 0.8779395741251295, 1.1160977637740144,
        0.9346505466117456, 1.1481435531632458, 0.9738770741251294,
        1.1656435531632459, 1.006987236836827, 1.1463357548174404,
        1.0401895741251292, 1.1279055621198197, 1.0672755466117456,
        1.129713360465625, 1.0768770741251292, 1.1363357548174404,
        1.0803622368368269, 1.1438357548174405, 1.0839395741251292,
        1.1299513515090513, 1.1016114491251294, 1.0860977637740141,
        1.103362080738569, 1.0584969844251852, 1.0999216972854096,
        1.0494141662544008, 1.0996008794883505, 1.0443415421919728
    ];

    /// Rscript, after [`fixture`]'s preamble:
    /// cat(edgeR:::loessByCol(Y, x, span=0.3)$leverages, sep=",")
    #[rustfmt::skip]
    const LOESS_SPAN03_LEV: [f64; 24] = [
        0.2517433276700617, 0.22880607073977888, 0.22518582658361702,
        0.2871209137455691, 0.2871209137455691, 0.2871209137455691,
        0.2871209137455691, 0.2871209137455691, 0.2871209137455691,
        0.2871209137455691, 0.2871209137455691, 0.2871209137455691,
        0.2871209137455691, 0.2871209137455691, 0.2871209137455691,
        0.2871209137455691, 0.2871209137455691, 0.2871209137455691,
        0.2871209137455691, 0.2871209137455691, 0.2871209137455691,
        0.22518582658361702, 0.22880607073977888, 0.2517433276700617
    ];

    /// Rscript, after [`fixture`]'s preamble:
    /// cat(t(edgeR:::loessByCol(Y, x, span=0.5)$fitted.values), sep=",")
    #[rustfmt::skip]
    const LOESS_SPAN05_FITTED: [f64; 48] = [
        0.3440048072281873, 1.1029588005340976, 0.355603784968213,
        1.1018365202507963, 0.36918123763120314, 1.1003002226230953,
        0.38808794972895844, 1.0980981144936464, 0.42157591358761254,
        1.0949605011110026, 0.48482622010274584, 1.0908443818258182,
        0.5559342347865345, 1.085813555540867, 0.6241788239578752,
        1.0819783903012228, 0.6919043087586447, 1.080868806433601,
        0.7578561461306359, 1.0866438050954679, 0.8155605951027461,
        1.1010747402900192, 0.8661074337586446, 1.11568742381822,
        0.914880624985765, 1.124679547065462, 0.9598698946655838,
        1.1318805652368376, 0.9972994305113214, 1.139577839950598,
        1.0261823946655835, 1.143417782282421, 1.047505624985765,
        1.1396843241576413, 1.0650449337586447, 1.1295778399505978,
        1.0808105951027458, 1.1156140233616172, 1.0855494332091025,
        1.1001336919868945, 1.0868513928912291, 1.092257620343311,
        1.0877476311411738, 1.0881288431788272, 1.0886537031655168,
        1.085132713419842, 1.0895470296262126, 1.082348675009878
    ];

    /// Rscript, after [`fixture`]'s preamble:
    /// cat(edgeR:::loessByCol(Y, x, span=0.5)$leverages, sep=",")
    #[rustfmt::skip]
    const LOESS_SPAN05_LEV: [f64; 24] = [
        0.14567844901709437, 0.13730769515957314, 0.1303655435738728,
        0.12615196163492806, 0.1283101039884125, 0.14399677784987835,
        0.14399677784987835, 0.14399677784987835, 0.14399677784987835,
        0.14399677784987835, 0.14399677784987835, 0.14399677784987835,
        0.14399677784987835, 0.14399677784987835, 0.14399677784987835,
        0.14399677784987835, 0.14399677784987835, 0.14399677784987835,
        0.14399677784987835, 0.1283101039884125, 0.12615196163492806,
        0.13036554357387284, 0.13730769515957314, 0.14567844901709437
    ];

    /// Rscript, after [`fixture`]'s preamble:
    /// cat(t(edgeR:::loessByCol(Y, x, span=1.0)$fitted.values), sep=",")
    #[rustfmt::skip]
    const LOESS_SPAN10_FITTED: [f64; 48] = [
        0.5919999382275462, 1.1049756108083535, 0.6003845203435453,
        1.1052926518997015, 0.6091098710543424, 1.1056595061555687,
        0.6183007919694515, 1.1060810476896294, 0.6282556399542248,
        1.1065680219288918, 0.6395195553319977, 1.1071273863302644,
        0.6530423281572642, 1.1077404072037507, 0.6703778450343311,
        1.1083745116112864, 0.6938726408710348, 1.1090113879941412,
        0.7267233665082738, 1.109683926241704, 0.7718915709544079,
        1.1105202462851993, 0.8272477827470099, 1.1117937685185284,
        0.8729649770087896, 1.1129143097576741, 0.9036326022233584,
        1.1109506051456721, 0.9261023541783554, 1.1081885198135022,
        0.9420181008601146, 1.1059722935531684, 0.953657476740288,
        1.1044706929531403, 0.9626515001567053, 1.1035364856253052,
        0.9699950786297831, 1.1030027276105538, 0.9762711492673746,
        1.1027041890057676, 0.9818254388090998, 1.1025013888945876,
        0.9868528051637, 1.1023088398873693, 0.9914575647672103,
        1.1020723633941227, 0.9956945899565433, 1.1017692594072268
    ];

    /// Rscript, after [`fixture`]'s preamble:
    /// cat(edgeR:::loessByCol(Y, x, span=1.0)$leverages, sep=",")
    #[rustfmt::skip]
    const LOESS_SPAN10_LEV: [f64; 24] = [
        0.07242619451620279, 0.0702824236768799, 0.06827286169305258,
        0.06640992317236669, 0.06472718591632619, 0.06328962803140528,
        0.062209505663947076, 0.0616703240153657, 0.06195983515122872,
        0.06349939673090867, 0.06679105474641003, 0.07201527819790013,
        0.07201527819790013, 0.06679105474641003, 0.06349939673090868,
        0.06195983515122872, 0.0616703240153657, 0.06220950566394705,
        0.06328962803140527, 0.0647271859163262, 0.06640992317236669,
        0.06827286169305258, 0.07028242367687988, 0.07242619451620277
    ];

    /// Rscript, after [`fixture`]'s preamble:
    /// cat(t(edgeR:::loessByCol(Y, span=0.5)$fitted.values), sep=",")
    #[rustfmt::skip]
    const LOESS_DEFAULT_X_FITTED: [f64; 48] = [
        0.3440048072281873, 1.1029588005340976, 0.355603784968213,
        1.1018365202507963, 0.36918123763120314, 1.1003002226230953,
        0.38808794972895844, 1.0980981144936464, 0.42157591358761254,
        1.0949605011110026, 0.48482622010274584, 1.0908443818258182,
        0.5559342347865345, 1.085813555540867, 0.6241788239578752,
        1.0819783903012228, 0.6919043087586447, 1.080868806433601,
        0.7578561461306359, 1.0866438050954679, 0.8155605951027461,
        1.1010747402900192, 0.8661074337586446, 1.11568742381822,
        0.914880624985765, 1.124679547065462, 0.9598698946655838,
        1.1318805652368376, 0.9972994305113214, 1.139577839950598,
        1.0261823946655835, 1.143417782282421, 1.047505624985765,
        1.1396843241576413, 1.0650449337586447, 1.1295778399505978,
        1.0808105951027458, 1.1156140233616172, 1.0855494332091025,
        1.1001336919868945, 1.0868513928912291, 1.092257620343311,
        1.0877476311411738, 1.0881288431788272, 1.0886537031655168,
        1.085132713419842, 1.0895470296262126, 1.082348675009878
    ];

    /// Rscript, after [`fixture`]'s preamble:
    /// cat(edgeR:::loessByCol(Y, span=0.5)$leverages, sep=",")
    #[rustfmt::skip]
    const LOESS_DEFAULT_X_LEV: [f64; 24] = [
        0.14567844901709437, 0.13730769515957314, 0.1303655435738728,
        0.12615196163492806, 0.1283101039884125, 0.14399677784987835,
        0.14399677784987835, 0.14399677784987835, 0.14399677784987835,
        0.14399677784987835, 0.14399677784987835, 0.14399677784987835,
        0.14399677784987835, 0.14399677784987835, 0.14399677784987835,
        0.14399677784987835, 0.14399677784987835, 0.14399677784987835,
        0.14399677784987835, 0.1283101039884125, 0.12615196163492806,
        0.13036554357387284, 0.13730769515957314, 0.14567844901709437
    ];

    /// Rscript, after [`ties_fixture`]'s preamble:
    /// cat(t(edgeR:::loessByCol(Yt, xt, span=0.5)$fitted.values), sep=",")
    #[rustfmt::skip]
    const LOESS_TIES_FITTED: [f64; 48] = [
        0.14963877177871465, 0.6635106230261268, 0.14963877177871465,
        0.6635106230261268, 0.14963877177871465, 0.6635106230261268,
        0.23953904655907998, 0.6396953255425709, 0.23953904655907998,
        0.6396953255425709, 0.23953904655907998, 0.6396953255425709,
        0.5011565572250046, 0.6318823038397329, 0.5011565572250046,
        0.6318823038397329, 0.5011565572250046, 0.6318823038397329,
        0.7270636245594508, 0.6469351697273235, 0.7270636245594508,
        0.6469351697273235, 0.7270636245594508, 0.6469351697273235,
        0.9628723798924134, 0.6524554813578186, 0.9628723798924134,
        0.6524554813578186, 0.9628723798924134, 0.6524554813578186,
        1.2242366907809314, 0.6135754034501947, 1.2242366907809314,
        0.6135754034501947, 1.2242366907809314, 0.6135754034501947,
        1.4959525134483402, 0.6152420701168615, 1.4959525134483402,
        0.6152420701168615, 1.4959525134483402, 0.6152420701168615,
        1.5783054535563716, 0.6167920878855192, 1.5783054535563716,
        0.6167920878855192, 1.5783054535563716, 0.6167920878855192
    ];

    /// Rscript, after [`ties_fixture`]'s preamble:
    /// cat(edgeR:::loessByCol(Yt, xt, span=0.5)$leverages, sep=",")
    #[rustfmt::skip]
    const LOESS_TIES_LEV: [f64; 24] = [
        0.14871481028151773, 0.14871481028151773, 0.14871481028151773,
        0.1424596549805231, 0.1424596549805231, 0.1424596549805231,
        0.1424596549805231, 0.1424596549805231, 0.1424596549805231,
        0.1424596549805231, 0.1424596549805231, 0.1424596549805231,
        0.1424596549805231, 0.1424596549805231, 0.1424596549805231,
        0.1424596549805231, 0.1424596549805231, 0.1424596549805231,
        0.1424596549805231, 0.1424596549805231, 0.1424596549805231,
        0.14871481028151773, 0.14871481028151773, 0.14871481028151773
    ];

    /// Rscript, after [`flat_fixture`]'s preamble:
    /// cat(t(edgeR:::loessByCol(Yf, xf, span=0.5)$fitted.values), sep=",")
    #[rustfmt::skip]
    const LOESS_FLATX_FITTED: [f64; 20] = [
        1.1, 2.3, 1.5166666666666666, 2.25,
        1.3, 2.25, 1.2375, 2.28125,
        1.2999999999999998, 2.25, 1.47, 2.25,
        1.47, 2.25, 1.47, 2.25,
        1.47, 2.25, 1.47, 2.25
    ];

    /// Rscript, after [`flat_fixture`]'s preamble:
    /// cat(edgeR:::loessByCol(Yf, xf, span=0.5)$leverages, sep=",")
    #[rustfmt::skip]
    const LOESS_FLATX_LEV: [f64; 10] = [
        0.2, 0.16666666666666666, 0.14285714285714285,
        0.125, 0.1111111111111111, 0.1,
        0.1, 0.1, 0.1, 0.1
    ];

    /// Rscript -e 'suppressMessages(library(edgeR));
    /// cat(t(edgeR:::loessByCol(cbind(c(1,3,2), c(4,5,6)), c(1,2,3),
    /// span=0.5)$fitted.values), sep=",")'
    #[rustfmt::skip]
    const LOESS_N3_IDENTITY_FITTED: [f64; 6] = [
        1.0, 4.0, 3.0, 5.0, 2.0, 6.0
    ];

    /// Rscript -e 'suppressMessages(library(edgeR));
    /// cat(edgeR:::loessByCol(cbind(c(1,3,2), c(4,5,6)), c(1,2,3), span=0.5)$leverages,
    /// sep=",")'
    #[rustfmt::skip]
    const LOESS_N3_IDENTITY_LEV: [f64; 3] = [
        1.0, 1.0, 1.0
    ];

    ////////////////////
    // locfit_by_col  //
    ////////////////////

    #[test]
    fn test_locfit_matches_edger_at_degree_0() {
        let (x, y, _) = fixture();
        let fit = locfit_by_col(&y, 24, 2, Some(&x), None, 0.3, 0).unwrap();
        assert_close(&fit, &LOCFIT_SPAN03_DEG0);

        let fit = locfit_by_col(&y, 24, 2, Some(&x), None, 0.5, 0).unwrap();
        assert_close(&fit, &LOCFIT_SPAN05_DEG0);

        // The two spans must actually disagree, or the test proves nothing.
        let worst = LOCFIT_SPAN03_DEG0
            .iter()
            .zip(&LOCFIT_SPAN05_DEG0)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(worst > 1e-2, "the span must move the fit");
    }

    #[test]
    fn test_locfit_matches_edger_at_degree_1() {
        let (x, y, _) = fixture();
        let fit = locfit_by_col(&y, 24, 2, Some(&x), None, 0.5, 1).unwrap();
        assert_close(&fit, &LOCFIT_SPAN05_DEG1);

        let fit = locfit_by_col(&y, 24, 2, Some(&x), None, 1.0, 1).unwrap();
        assert_close(&fit, &LOCFIT_SPAN10_DEG1);
    }

    /// A wider span means a coarser tree: the bandwidths grow, so cells stop
    /// being refined sooner and fewer local fits are performed.
    #[test]
    fn test_locfit_wider_spans_need_fewer_vertices() {
        let (x, _, _) = fixture();
        let mut counts = Vec::new();
        for span in [0.3, 0.5, 1.0] {
            let k = (24.0 * span + LOCFIT_NN_EPS) as usize;
            let (vertices, _) = build_tree(&x, k, vertex_capacity(span)).unwrap();
            counts.push(vertices.len());
        }
        assert!(counts[0] > counts[1], "{counts:?}");
        assert!(counts[1] > counts[2], "{counts:?}");
    }

    #[test]
    fn test_locfit_prior_weights_change_the_fit() {
        let (x, y, w) = fixture();
        let fit = locfit_by_col(&y, 24, 2, Some(&x), Some(&w), 0.5, 1).unwrap();
        assert_close(&fit, &LOCFIT_WEIGHTED_DEG1);

        let plain = locfit_by_col(&y, 24, 2, Some(&x), None, 0.5, 1).unwrap();
        let worst = fit
            .iter()
            .zip(&plain)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(worst > 1e-3, "prior weights should move the fit");

        // All-ones weights must be exactly the same as supplying none.
        let ones = vec![1.0; 24];
        let same = locfit_by_col(&y, 24, 2, Some(&x), Some(&ones), 0.5, 1).unwrap();
        assert_eq!(same, plain);
    }

    /// `None` means edgeR's `1:ntags`. Here that is an affine map of `i/32`,
    /// and locfit is equivariant under an affine change of covariate, so the
    /// answer is the same one the explicit covariate gives.
    #[test]
    fn test_locfit_default_covariate() {
        let (_, y, _) = fixture();
        let fit = locfit_by_col(&y, 24, 2, None, None, 0.5, 0).unwrap();
        assert_close(&fit, &LOCFIT_DEFAULT_X_DEG0);
    }

    #[test]
    fn test_locfit_columns_are_independent() {
        let (x, y, w) = fixture();
        let both = locfit_by_col(&y, 24, 2, Some(&x), Some(&w), 0.5, 1).unwrap();
        for col in 0..2 {
            let single: Vec<f64> = y.chunks_exact(2).map(|r| r[col]).collect();
            let alone = locfit_by_col(&single, 24, 1, Some(&x), Some(&w), 0.5, 1).unwrap();
            let sliced: Vec<f64> = both.chunks_exact(2).map(|r| r[col]).collect();
            assert_eq!(alone, sliced, "column {col} was not independent");
        }
    }

    /// Tied covariate values share one fitted value, since the fit only ever
    /// depends on the covariate.
    #[test]
    fn test_locfit_ties_in_x() {
        let (x, y) = ties_fixture();
        let fit = locfit_by_col(&y, 24, 2, Some(&x), None, 0.5, 0).unwrap();
        assert_close(&fit, &LOCFIT_TIES_DEG0);
        for group in fit.chunks_exact(6) {
            assert_eq!(group[0], group[2]);
            assert_eq!(group[0], group[4]);
        }

        let fit = locfit_by_col(&y, 24, 2, Some(&x), None, 0.5, 1).unwrap();
        assert_close(&fit, &LOCFIT_TIES_DEG1);
    }

    /// Every covariate value identical: the bandwidth collapses, the tree never
    /// splits, and the fit is the weighted column mean repeated.
    #[test]
    fn test_locfit_all_x_identical() {
        let (x, y) = flat_fixture();
        let fit = locfit_by_col(&y, 10, 2, Some(&x), None, 0.5, 0).unwrap();
        assert_close(&fit, &LOCFIT_FLATX_DEG0);

        let fit = locfit_by_col(&y, 10, 2, Some(&x), None, 0.5, 1).unwrap();
        assert_close(&fit, &LOCFIT_FLATX_DEG1);
    }

    /// `span * n_rows < 2` is edgeR's escape hatch and returns the input.
    #[test]
    fn test_locfit_fewer_rows_than_the_span_implies() {
        let y = vec![1.0, 4.0, 3.0, 5.0, 2.0, 6.0];
        let x = vec![1.0, 2.0, 3.0];
        let fit = locfit_by_col(&y, 3, 2, Some(&x), None, 0.5, 0).unwrap();
        assert_close(&fit, &LOCFIT_N3_IDENTITY);
        assert_eq!(fit, y);

        // A single row is returned unchanged whatever the span.
        let one = locfit_by_col(&[7.0, 8.0], 1, 2, None, None, 1.0, 1).unwrap();
        assert_eq!(one, vec![7.0, 8.0]);

        // Five rows at span one is the smallest case that actually smooths.
        let y = vec![1.0, 2.0, 3.0, 1.0, 2.0, 4.0, 5.0, 3.0, 4.0, 6.0];
        let x: Vec<f64> = (1..=5u64).map(|i| i as f64 / 5.0).collect();
        let fit = locfit_by_col(&y, 5, 2, Some(&x), None, 1.0, 1).unwrap();
        assert_close(&fit, &LOCFIT_N5_SPAN10_DEG1);
    }

    /// The tree depends on the covariate range and the bandwidths, none of
    /// which care about row order, so permuting the rows permutes the answer.
    #[test]
    fn test_locfit_unsorted_covariate_matches_sorted() {
        let (x, y, _) = fixture();
        let perm: Vec<usize> = (1..=24usize).map(|i| (i * 7) % 24).collect();
        let xp: Vec<f64> = perm.iter().map(|&i| x[i]).collect();
        let yp: Vec<f64> = perm
            .iter()
            .flat_map(|&i| y[i * 2..i * 2 + 2].iter().copied())
            .collect();

        let fit = locfit_by_col(&yp, 24, 2, Some(&xp), None, 0.5, 0).unwrap();
        let want: Vec<f64> = perm
            .iter()
            .flat_map(|&i| LOCFIT_SPAN05_DEG0[i * 2..i * 2 + 2].iter().copied())
            .collect();
        assert_close(&fit, &want);
    }

    /// The rayon fan-out must be bit-for-bit the sequential answer: every
    /// vertex fit and every interpolated row is independent of the rest, so
    /// splitting them changes no summation order. Eight columns crosses the
    /// gate, four does not, and the columns are independent.
    #[test]
    fn test_locfit_parallel_path_agrees_with_sequential() {
        let (rows, cols) = (2048usize, 8usize);
        let (x, y) = wide_fixture(rows, cols);
        assert!(rows * cols >= PARALLEL_WORK_THRESHOLD);
        assert!(rows * (cols / 2) < PARALLEL_WORK_THRESHOLD);

        let par = locfit_by_col(&y, rows, cols, Some(&x), None, 0.3, 1).unwrap();
        let half: Vec<f64> = y.chunks_exact(cols).flat_map(|r| r[..4].to_vec()).collect();
        let seq = locfit_by_col(&half, rows, 4, Some(&x), None, 0.3, 1).unwrap();
        let sliced: Vec<f64> = par
            .chunks_exact(cols)
            .flat_map(|r| r[..4].to_vec())
            .collect();
        assert_eq!(seq, sliced);
    }

    ////////////////////
    // loess_by_col   //
    ////////////////////

    #[test]
    fn test_loess_matches_edger() {
        let (x, y, _) = fixture();
        for (span, want_fit, want_lev) in [
            (0.3, &LOESS_SPAN03_FITTED, &LOESS_SPAN03_LEV),
            (0.5, &LOESS_SPAN05_FITTED, &LOESS_SPAN05_LEV),
            (1.0, &LOESS_SPAN10_FITTED, &LOESS_SPAN10_LEV),
        ] {
            let fit = loess_by_col(&y, 24, 2, Some(&x), span).unwrap();
            assert_close(&fit.fitted, want_fit);
            assert_close(&fit.leverages, want_lev);
        }
    }

    /// A wider span means each row counts for less of its own fit.
    #[test]
    fn test_loess_leverages_shrink_with_the_span() {
        let (x, y, _) = fixture();
        let narrow = loess_by_col(&y, 24, 2, Some(&x), 0.3).unwrap();
        let wide = loess_by_col(&y, 24, 2, Some(&x), 1.0).unwrap();
        for (n, w) in narrow.leverages.iter().zip(&wide.leverages) {
            assert!(n > w, "leverage {n} should exceed {w}");
            assert!(*w > 0.0 && *n <= 1.0);
        }
    }

    #[test]
    fn test_loess_default_covariate() {
        let (_, y, _) = fixture();
        let fit = loess_by_col(&y, 24, 2, None, 0.5).unwrap();
        assert_close(&fit.fitted, &LOESS_DEFAULT_X_FITTED);
        assert_close(&fit.leverages, &LOESS_DEFAULT_X_LEV);
    }

    #[test]
    fn test_loess_ties_in_x() {
        let (x, y) = ties_fixture();
        let fit = loess_by_col(&y, 24, 2, Some(&x), 0.5).unwrap();
        assert_close(&fit.fitted, &LOESS_TIES_FITTED);
        assert_close(&fit.leverages, &LOESS_TIES_LEV);
        for group in fit.fitted.chunks_exact(6) {
            assert_eq!(group[0], group[2]);
            assert_eq!(group[0], group[4]);
        }
    }

    /// Every covariate value identical. edgeR's frame still walks forwards
    /// through the run, so row `i` is averaged over the first `max(i, nspan)`
    /// rows and its leverage is one over that count. It is a quirk of the
    /// C++, not a smoothing decision, and it is reproduced here deliberately.
    #[test]
    fn test_loess_all_x_identical() {
        let (x, y) = flat_fixture();
        let fit = loess_by_col(&y, 10, 2, Some(&x), 0.5).unwrap();
        assert_close(&fit.fitted, &LOESS_FLATX_FITTED);
        assert_close(&fit.leverages, &LOESS_FLATX_LEV);
        for (i, lev) in fit.leverages.iter().enumerate() {
            assert_relative_eq!(*lev, 1.0 / (i + 5).min(10) as f64, max_relative = TOL);
        }
    }

    /// `floor(span * n_rows) <= 1` is edgeR's escape hatch.
    #[test]
    fn test_loess_span_too_small_returns_the_input() {
        let y = vec![1.0, 4.0, 3.0, 5.0, 2.0, 6.0];
        let x = vec![1.0, 2.0, 3.0];
        let fit = loess_by_col(&y, 3, 2, Some(&x), 0.5).unwrap();
        assert_close(&fit.fitted, &LOESS_N3_IDENTITY_FITTED);
        assert_close(&fit.leverages, &LOESS_N3_IDENTITY_LEV);
        assert_eq!(fit.fitted, y);
    }

    #[test]
    fn test_loess_columns_are_independent() {
        let (x, y, _) = fixture();
        let both = loess_by_col(&y, 24, 2, Some(&x), 0.5).unwrap();
        for col in 0..2 {
            let single: Vec<f64> = y.chunks_exact(2).map(|r| r[col]).collect();
            let alone = loess_by_col(&single, 24, 1, Some(&x), 0.5).unwrap();
            let sliced: Vec<f64> = both.fitted.chunks_exact(2).map(|r| r[col]).collect();
            assert_eq!(alone.fitted, sliced, "column {col} was not independent");
            assert_eq!(alone.leverages, both.leverages);
        }
    }

    /// edgeR's C++ assumes a sorted covariate; this sorts internally, so an
    /// unsorted input gives the same answer back in the caller's row order.
    #[test]
    fn test_loess_unsorted_covariate_matches_sorted() {
        let (x, y, _) = fixture();
        let perm: Vec<usize> = (1..=24usize).map(|i| (i * 7) % 24).collect();
        let xp: Vec<f64> = perm.iter().map(|&i| x[i]).collect();
        let yp: Vec<f64> = perm
            .iter()
            .flat_map(|&i| y[i * 2..i * 2 + 2].iter().copied())
            .collect();

        let fit = loess_by_col(&yp, 24, 2, Some(&xp), 0.5).unwrap();
        let want: Vec<f64> = perm
            .iter()
            .flat_map(|&i| LOESS_SPAN05_FITTED[i * 2..i * 2 + 2].iter().copied())
            .collect();
        assert_close(&fit.fitted, &want);
        let want_lev: Vec<f64> = perm.iter().map(|&i| LOESS_SPAN05_LEV[i]).collect();
        assert_close(&fit.leverages, &want_lev);
    }

    #[test]
    fn test_loess_parallel_path_agrees_with_sequential() {
        let (rows, cols) = (2048usize, 8usize);
        let (x, y) = wide_fixture(rows, cols);
        assert!(rows * cols >= PARALLEL_WORK_THRESHOLD);
        assert!(rows * (cols / 2) < PARALLEL_WORK_THRESHOLD);

        let par = loess_by_col(&y, rows, cols, Some(&x), 0.1).unwrap();
        let half: Vec<f64> = y.chunks_exact(cols).flat_map(|r| r[..4].to_vec()).collect();
        let seq = loess_by_col(&half, rows, 4, Some(&x), 0.1).unwrap();
        let sliced: Vec<f64> = par
            .fitted
            .chunks_exact(cols)
            .flat_map(|r| r[..4].to_vec())
            .collect();
        assert_eq!(seq.fitted, sliced);
        assert_eq!(seq.leverages, par.leverages);
    }

    ////////////////////
    // Error branches //
    ////////////////////

    #[test]
    fn test_rejects_shape_mismatch() {
        let y = vec![1.0; 6];
        assert!(matches!(
            locfit_by_col(&y, 4, 2, None, None, 0.5, 0).unwrap_err(),
            EdgeErrors::ShapeMismatch { .. }
        ));
        assert!(matches!(
            loess_by_col(&y, 4, 2, None, 0.5).unwrap_err(),
            EdgeErrors::ShapeMismatch { .. }
        ));
    }

    #[test]
    fn test_rejects_empty_input() {
        assert!(matches!(
            locfit_by_col(&[], 0, 2, None, None, 0.5, 0).unwrap_err(),
            EdgeErrors::EmptyCounts { .. }
        ));
        assert!(matches!(
            locfit_by_col(&[], 3, 0, None, None, 0.5, 0).unwrap_err(),
            EdgeErrors::EmptyCounts { .. }
        ));
        assert!(matches!(
            loess_by_col(&[], 0, 0, None, 0.5).unwrap_err(),
            EdgeErrors::EmptyCounts { .. }
        ));
    }

    #[test]
    fn test_rejects_covariate_length_mismatch() {
        let (_, y, _) = fixture();
        let short = vec![0.0; 23];
        assert!(matches!(
            locfit_by_col(&y, 24, 2, Some(&short), None, 0.5, 0).unwrap_err(),
            EdgeErrors::LengthMismatch { name: "x", .. }
        ));
        assert!(matches!(
            loess_by_col(&y, 24, 2, Some(&short), 0.5).unwrap_err(),
            EdgeErrors::LengthMismatch { name: "x", .. }
        ));
    }

    #[test]
    fn test_rejects_span_outside_the_unit_interval() {
        let (x, y, _) = fixture();
        for span in [0.0, -0.1, 1.5, f64::NAN] {
            assert!(
                matches!(
                    locfit_by_col(&y, 24, 2, Some(&x), None, span, 0).unwrap_err(),
                    EdgeErrors::InvalidArgument(_)
                ),
                "locfit span {span}"
            );
            assert!(
                matches!(
                    loess_by_col(&y, 24, 2, Some(&x), span).unwrap_err(),
                    EdgeErrors::InvalidArgument(_)
                ),
                "loess span {span}"
            );
        }
        // The closed upper end is legal for both.
        assert!(locfit_by_col(&y, 24, 2, Some(&x), None, 1.0, 0).is_ok());
        assert!(loess_by_col(&y, 24, 2, Some(&x), 1.0).is_ok());
    }

    #[test]
    fn test_rejects_degree_above_one() {
        let (x, y, _) = fixture();
        for degree in [2, 3, 17] {
            assert!(matches!(
                locfit_by_col(&y, 24, 2, Some(&x), None, 0.5, degree).unwrap_err(),
                EdgeErrors::InvalidArgument(_)
            ));
        }
        assert!(locfit_by_col(&y, 24, 2, Some(&x), None, 0.5, 1).is_ok());
    }

    #[test]
    fn test_rejects_bad_weights() {
        let (x, y, mut w) = fixture();
        w[9] = -0.25;
        assert!(matches!(
            locfit_by_col(&y, 24, 2, Some(&x), Some(&w), 0.5, 0).unwrap_err(),
            EdgeErrors::InvalidArgument(_)
        ));

        let short = vec![1.0; 23];
        assert!(matches!(
            locfit_by_col(&y, 24, 2, Some(&x), Some(&short), 0.5, 0).unwrap_err(),
            EdgeErrors::LengthMismatch {
                name: "weights",
                ..
            }
        ));

        // Zero is legal; it drops the row from every local fit.
        let mut zeroed = vec![1.0; 24];
        zeroed[9] = 0.0;
        assert!(locfit_by_col(&y, 24, 2, Some(&x), Some(&zeroed), 0.5, 0).is_ok());
    }

    /// locfit sizes its vertex array up front and aborts with "out of vertex
    /// space" if the refinement wants more. The same budget is enforced here,
    /// so a covariate whose clusters are too tight to tile is a refusal rather
    /// than an endless subdivision.
    #[test]
    fn test_tree_respects_locfits_vertex_budget() {
        assert_eq!(vertex_capacity(0.3), 43);
        assert_eq!(vertex_capacity(0.5), 27);
        assert_eq!(vertex_capacity(1.0), 14);

        let (x, _, _) = fixture();
        let k = (24.0 * 0.5 + LOCFIT_NN_EPS) as usize;
        let (vertices, _) = build_tree(&x, k, vertex_capacity(0.5)).unwrap();
        assert!(vertices.len() > 4);
        assert!(matches!(
            build_tree(&x, k, 4).unwrap_err(),
            EdgeErrors::InvalidArgument(_)
        ));
    }
}
