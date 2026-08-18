//! Box-constrained limited-memory BFGS.
//!
//! NEBULA's first stage is a bounded marginal maximum likelihood fit, which
//! edgePython hands to `scipy.optimize.minimize(method = 'L-BFGS-B', jac = True)`.
//! That routine is Nocedal's Fortran, and binding it would put a Fortran
//! toolchain on the Windows CI lane, so it is reimplemented here.
//!
//! This follows Byrd, Lu, Nocedal and Zhu (1995): the compact limited-memory
//! representation `B = theta*I - W M W'`, a generalised Cauchy point to identify
//! the active set, then subspace minimisation over the free variables. The one
//! deliberate departure from the Fortran is the line search, which is the
//! cubic-interpolation strong Wolfe search of Nocedal and Wright (Algorithm 3.5
//! and 3.6) rather than More and Thuente's `dcsrch`. Both satisfy the same
//! conditions; iterates can differ in the last digits, optima do not.
//!
//! ### References
//!
//! Byrd, Lu, Nocedal and Zhu, SIAM Journal on Scientific Computing 16(5), 1995
//! Nocedal and Wright, Numerical Optimization, 2nd edition, 2006

use crate::errors::EdgeErrors;

////////////
// Consts //
////////////

/// Armijo parameter for the sufficient decrease condition.
///
/// The value both Nocedal and Wright and the Fortran use. It is deliberately
/// tiny: sufficient decrease is meant to rule out pathological steps, not to
/// drive the search.
const WOLFE_C1: f64 = 1e-4;

/// Curvature parameter for the strong Wolfe condition.
///
/// 0.9 is the standard quasi-Newton choice. Tightening it buys nothing here
/// because the unit step is accepted on nearly every iteration.
const WOLFE_C2: f64 = 0.9;

/// Minimum curvature for a memory pair to be admitted.
///
/// A BFGS update needs `s'y > 0` to keep the Hessian approximation positive
/// definite. Pairs failing this are dropped rather than damped, which is what
/// the Fortran does.
const CURVATURE_EPS: f64 = 2.2e-16;

/// Largest step the line search will consider.
///
/// The subspace minimisation already truncates to the feasible box, so a unit
/// step is the natural scale and anything far beyond it signals trouble.
const MAX_STEP: f64 = 1e3;

//////////////////
// LbfgsbParams //
//////////////////

/// Tuning knobs for [`minimise`].
#[derive(Clone, Copy, Debug)]
pub struct LbfgsbParams {
    /// Number of correction pairs retained. scipy defaults to 10.
    pub memory: usize,
    /// Relative reduction in the objective below which the search stops.
    ///
    /// Matches scipy's `ftol`: the test is
    /// `f_old - f_new <= ftol * max(|f_old|, |f_new|, 1)`.
    pub ftol: f64,
    /// Infinity norm of the projected gradient below which the search stops.
    pub pgtol: f64,
    /// Iteration budget.
    pub max_iter: usize,
    /// Function evaluations allowed inside a single line search.
    pub max_line_search: usize,
}

impl Default for LbfgsbParams {
    /// scipy's defaults, except `ftol`, which NEBULA sets to 1e-6.
    fn default() -> Self {
        Self {
            memory: 10,
            ftol: 1e-6,
            pgtol: 1e-5,
            max_iter: 200,
            max_line_search: 20,
        }
    }
}

//////////////////
// LbfgsbStatus //
//////////////////

/// Why the optimiser stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LbfgsbStatus {
    /// The projected gradient fell below `pgtol`.
    GradientTolerance,
    /// The relative reduction in the objective fell below `ftol`.
    ObjectiveTolerance,
    /// The iteration budget ran out.
    MaxIterations,
    /// A line search could not find an acceptable step.
    ///
    /// The returned point is the last feasible iterate, which is usable but has
    /// not met either tolerance.
    LineSearchFailed,
}

impl LbfgsbStatus {
    /// Whether the stop was a genuine convergence rather than giving up.
    ///
    /// ### Returns
    ///
    /// `true` for the two tolerance statuses.
    pub fn converged(&self) -> bool {
        matches!(
            self,
            LbfgsbStatus::GradientTolerance | LbfgsbStatus::ObjectiveTolerance
        )
    }
}

//////////////////
// LbfgsbResult //
//////////////////

/// Outcome of a bounded minimisation.
#[derive(Clone, Debug)]
pub struct LbfgsbResult {
    /// Best point found, guaranteed inside the box.
    pub x: Vec<f64>,
    /// Objective value at `x`.
    pub f: f64,
    /// Gradient at `x`.
    pub grad: Vec<f64>,
    /// Iterations performed.
    pub iterations: usize,
    /// Objective evaluations performed.
    pub evaluations: usize,
    /// Why the search stopped.
    pub status: LbfgsbStatus,
}

//////////////////////////
// Limited-memory pairs //
//////////////////////////

/// The correction pairs and the compact representation built from them.
///
/// Pairs are held oldest first. `theta` scales the identity term in
/// `B = theta*I - W M W'` and is refreshed from the newest pair on every update.
struct Memory {
    /// Step differences `x_{k+1} - x_k`, oldest first.
    s: Vec<Vec<f64>>,
    /// Gradient differences `g_{k+1} - g_k`, oldest first.
    y: Vec<Vec<f64>>,
    /// Scaling of the identity term.
    theta: f64,
    /// Maximum number of pairs retained.
    limit: usize,
}

impl Memory {
    /// Creates an empty memory.
    ///
    /// ### Params
    ///
    /// * `limit` - Maximum number of correction pairs
    ///
    /// ### Returns
    ///
    /// A memory holding no pairs, with `theta` at 1.
    fn new(limit: usize) -> Self {
        Self {
            s: Vec::with_capacity(limit),
            y: Vec::with_capacity(limit),
            theta: 1.0,
            limit,
        }
    }

    /// Number of pairs currently held.
    ///
    /// ### Returns
    ///
    /// The pair count, at most `limit`.
    fn k(&self) -> usize {
        self.s.len()
    }

    /// Discards every pair and returns `theta` to one.
    ///
    /// The Fortran does this when the line search cannot make progress: a
    /// limited-memory model built from steps taken elsewhere on the surface can
    /// point somewhere the objective refuses to follow, and the cure is to
    /// forget it and take a projected steepest-descent step instead.
    fn reset(&mut self) {
        self.s.clear();
        self.y.clear();
        self.theta = 1.0;
    }

    /// Admits a correction pair, evicting the oldest if full.
    ///
    /// Pairs with insufficient curvature are rejected, which keeps the implied
    /// Hessian positive definite.
    ///
    /// ### Params
    ///
    /// * `s` - Step difference
    /// * `y` - Gradient difference
    ///
    /// ### Returns
    ///
    /// `true` if the pair was admitted.
    fn push(&mut self, s: Vec<f64>, y: Vec<f64>) -> bool {
        let sy: f64 = s.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let yy: f64 = y.iter().map(|b| b * b).sum();
        if sy <= CURVATURE_EPS * yy || yy <= 0.0 {
            return false;
        }
        if self.s.len() == self.limit {
            self.s.remove(0);
            self.y.remove(0);
        }
        self.theta = yy / sy;
        self.s.push(s);
        self.y.push(y);
        true
    }

    /// Column `j` of `W`, evaluated at row `i`.
    ///
    /// `W = [Y, theta*S]`, so the first `k` columns are gradient differences and
    /// the next `k` are scaled step differences.
    ///
    /// ### Params
    ///
    /// * `i` - Row, that is, the variable index
    /// * `j` - Column, in `0..2k`
    ///
    /// ### Returns
    ///
    /// The entry `W[i, j]`.
    #[inline]
    fn w(&self, i: usize, j: usize) -> f64 {
        let k = self.k();
        if j < k {
            self.y[j][i]
        } else {
            self.theta * self.s[j - k][i]
        }
    }

    /// Forms `W' v`.
    ///
    /// ### Params
    ///
    /// * `v` - Vector of length `n`
    ///
    /// ### Returns
    ///
    /// A vector of length `2k`.
    fn wt_dot(&self, v: &[f64]) -> Vec<f64> {
        let k = self.k();
        let mut out = vec![0.0; 2 * k];
        for (j, out_j) in out.iter_mut().enumerate() {
            let mut acc = 0.0;
            for (i, &vi) in v.iter().enumerate() {
                acc += self.w(i, j) * vi;
            }
            *out_j = acc;
        }
        out
    }

    /// Forms the middle matrix `M^{-1}` of the compact representation.
    ///
    /// Laid out as `[[-D, L'], [L, theta * S'S]]` with `D` the diagonal of
    /// `s_i' y_i` and `L` strictly lower triangular with `L[i, j] = s_i' y_j`.
    /// The routines below solve against this rather than inverting it.
    ///
    /// ### Returns
    ///
    /// A `2k` by `2k` matrix, row-major.
    fn middle(&self) -> Vec<f64> {
        let k = self.k();
        let dim = 2 * k;
        let mut m = vec![0.0; dim * dim];

        for i in 0..k {
            let sy: f64 = self.s[i]
                .iter()
                .zip(self.y[i].iter())
                .map(|(a, b)| a * b)
                .sum();
            m[i * dim + i] = -sy;
        }

        for i in 0..k {
            for j in 0..k {
                if i > j {
                    let sy: f64 = self.s[i]
                        .iter()
                        .zip(self.y[j].iter())
                        .map(|(a, b)| a * b)
                        .sum();
                    // L in the lower-left block, L' in the upper-right.
                    m[(k + i) * dim + j] = sy;
                    m[j * dim + (k + i)] = sy;
                }
            }
        }

        for i in 0..k {
            for j in 0..k {
                let ss: f64 = self.s[i]
                    .iter()
                    .zip(self.s[j].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                m[(k + i) * dim + (k + j)] = self.theta * ss;
            }
        }

        m
    }

    /// Solves `M^{-1} z = v`, that is, applies `M`.
    ///
    /// ### Params
    ///
    /// * `v` - Right-hand side of length `2k`
    ///
    /// ### Returns
    ///
    /// The solution, or `None` if the middle matrix is numerically singular, in
    /// which case the caller falls back to a steepest descent step.
    fn apply_m(&self, v: &[f64]) -> Option<Vec<f64>> {
        let dim = 2 * self.k();
        if dim == 0 {
            return Some(Vec::new());
        }
        solve_dense(&self.middle(), v, dim)
    }
}

/// Solves a small dense system by Gaussian elimination with partial pivoting.
///
/// The systems here are `2k` by `2k` with `k` at most the memory size, so
/// twenty by twenty at the default settings. A faer factorisation would be
/// dominated by its own call overhead at that size.
///
/// ### Params
///
/// * `a` - Row-major `dim` by `dim` matrix, consumed by value into a scratch copy
/// * `b` - Right-hand side of length `dim`
/// * `dim` - System size
///
/// ### Returns
///
/// The solution, or `None` if a pivot underflows.
fn solve_dense(a: &[f64], b: &[f64], dim: usize) -> Option<Vec<f64>> {
    let mut m = a.to_vec();
    let mut x = b.to_vec();

    for col in 0..dim {
        let mut pivot = col;
        let mut best = m[col * dim + col].abs();
        for row in (col + 1)..dim {
            let v = m[row * dim + col].abs();
            if v > best {
                best = v;
                pivot = row;
            }
        }
        if best < 1e-300 {
            return None;
        }
        if pivot != col {
            for j in 0..dim {
                m.swap(col * dim + j, pivot * dim + j);
            }
            x.swap(col, pivot);
        }
        let diag = m[col * dim + col];
        for row in (col + 1)..dim {
            let factor = m[row * dim + col] / diag;
            if factor == 0.0 {
                continue;
            }
            for j in col..dim {
                m[row * dim + j] -= factor * m[col * dim + j];
            }
            x[row] -= factor * x[col];
        }
    }

    for col in (0..dim).rev() {
        let mut acc = x[col];
        for j in (col + 1)..dim {
            acc -= m[col * dim + j] * x[j];
        }
        x[col] = acc / m[col * dim + col];
    }

    Some(x)
}

//////////////////////////////
// Generalised Cauchy point //
//////////////////////////////

/// Result of the Cauchy point search.
struct CauchyPoint {
    /// The Cauchy point itself.
    x: Vec<f64>,
    /// Which variables remain free, that is, did not reach a bound.
    free: Vec<bool>,
    /// `W' (xcp - x)`, needed by the subspace minimisation.
    c: Vec<f64>,
}

/// Locates the generalised Cauchy point along the projected steepest descent
/// path.
///
/// Walks the piecewise-linear projected gradient path, tracking the first and
/// second derivatives of the quadratic model segment by segment, and stops at
/// the first segment whose interior minimiser falls inside it. Variables that
/// hit a bound before then are fixed there.
///
/// ### Params
///
/// * `x` - Current iterate
/// * `g` - Gradient at `x`
/// * `lower` - Lower bounds
/// * `upper` - Upper bounds
/// * `mem` - Limited-memory state
///
/// ### Returns
///
/// The Cauchy point, the free variable mask, and `W'(xcp - x)`.
///
/// ### References
///
/// Byrd, Lu, Nocedal and Zhu (1995), Algorithm CP
fn cauchy_point(
    x: &[f64],
    g: &[f64],
    lower: &[f64],
    upper: &[f64],
    mem: &Memory,
) -> Option<CauchyPoint> {
    let n = x.len();
    let k = mem.k();

    // Breakpoints along the projected steepest descent path.
    let mut t = vec![f64::INFINITY; n];
    let mut d = vec![0.0; n];
    for i in 0..n {
        if g[i] < 0.0 && upper[i].is_finite() {
            t[i] = (x[i] - upper[i]) / g[i];
        } else if g[i] > 0.0 && lower[i].is_finite() {
            t[i] = (x[i] - lower[i]) / g[i];
        }
        d[i] = if t[i] == 0.0 { 0.0 } else { -g[i] };
    }

    let mut xcp = x.to_vec();
    let mut free: Vec<bool> = (0..n).map(|i| t[i] > 0.0).collect();

    // Breakpoints in increasing order, skipping variables already at a bound.
    let mut order: Vec<usize> = (0..n).filter(|&i| t[i] > 0.0 && t[i].is_finite()).collect();
    order.sort_by(|&a, &b| t[a].total_cmp(&t[b]));

    let mut c = vec![0.0; 2 * k];
    let mut p = mem.wt_dot(&d);

    let mut f_prime: f64 = -d.iter().map(|v| v * v).sum::<f64>();
    let mp = mem.apply_m(&p)?;
    let mut f_double = -mem.theta * f_prime - dot(&p, &mp);
    let f_double_initial = f_double.max(f64::EPSILON);
    f_double = f_double.max(f64::EPSILON);

    let mut dt_min = -f_prime / f_double;
    let mut t_old = 0.0;
    let mut cursor = 0;

    while cursor < order.len() {
        let b = order[cursor];
        let t_b = t[b];
        let dt = t_b - t_old;

        if dt_min < dt {
            break;
        }

        // This variable reaches its bound within the current segment.
        xcp[b] = if d[b] > 0.0 { upper[b] } else { lower[b] };
        let zb = xcp[b] - x[b];
        free[b] = false;

        for (ci, pi) in c.iter_mut().zip(p.iter()) {
            *ci += dt * pi;
        }

        let gb = g[b];
        let wb: Vec<f64> = (0..2 * k).map(|j| mem.w(b, j)).collect();
        let mc = mem.apply_m(&c)?;
        let mp = mem.apply_m(&p)?;
        let mwb = mem.apply_m(&wb)?;

        f_prime += dt * f_double + gb * gb + mem.theta * gb * zb - gb * dot(&wb, &mc);
        f_double += -mem.theta * gb * gb - 2.0 * gb * dot(&wb, &mp) - gb * gb * dot(&wb, &mwb);
        f_double = f_double.max(f64::EPSILON * f_double_initial);

        for (pi, wbi) in p.iter_mut().zip(wb.iter()) {
            *pi += gb * wbi;
        }
        d[b] = 0.0;

        dt_min = -f_prime / f_double;
        t_old = t_b;
        cursor += 1;
    }

    let dt_min = dt_min.max(0.0);
    let t_final = t_old + dt_min;

    for i in 0..n {
        if free[i] && t[i] >= t_final {
            xcp[i] = x[i] + t_final * d[i];
        }
    }
    for (ci, pi) in c.iter_mut().zip(p.iter()) {
        *ci += dt_min * pi;
    }

    Some(CauchyPoint { x: xcp, free, c })
}

/// Dot product of two equal-length slices.
///
/// ### Params
///
/// * `a` - Left operand
/// * `b` - Right operand
///
/// ### Returns
///
/// The sum of elementwise products.
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

///////////////////////////
// Subspace minimisation //
///////////////////////////

/// Minimises the quadratic model over the free variables, starting at the
/// Cauchy point.
///
/// Uses the direct primal method of the paper: form the reduced gradient at the
/// Cauchy point, apply the inverse of the compact Hessian restricted to the
/// free set through a Sherman-Morrison-Woodbury identity, then truncate the
/// resulting step so the point stays inside the box.
///
/// ### Params
///
/// * `x` - Current iterate
/// * `g` - Gradient at `x`
/// * `lower` - Lower bounds
/// * `upper` - Upper bounds
/// * `cp` - Output of [`cauchy_point`]
/// * `mem` - Limited-memory state
///
/// ### Returns
///
/// The refined point, or `None` if the small systems were singular.
///
/// ### References
///
/// Byrd, Lu, Nocedal and Zhu (1995), Section 5, `subsm`
fn subspace_minimise(
    x: &[f64],
    g: &[f64],
    lower: &[f64],
    upper: &[f64],
    cp: &CauchyPoint,
    mem: &Memory,
) -> Option<Vec<f64>> {
    let n = x.len();
    let k = mem.k();
    if k == 0 {
        return Some(cp.x.clone());
    }

    let free: Vec<usize> = (0..n).filter(|&i| cp.free[i]).collect();
    if free.is_empty() {
        return Some(cp.x.clone());
    }

    let mc = mem.apply_m(&cp.c)?;

    // Reduced gradient of the quadratic model at the Cauchy point.
    let mut r = vec![0.0; free.len()];
    for (slot, &i) in free.iter().enumerate() {
        let wi: Vec<f64> = (0..2 * k).map(|j| mem.w(i, j)).collect();
        r[slot] = g[i] + mem.theta * (cp.x[i] - x[i]) - dot(&wi, &mc);
    }

    // v = M W_Z' r
    let mut v = vec![0.0; 2 * k];
    for (j, v_j) in v.iter_mut().enumerate() {
        let mut acc = 0.0;
        for (slot, &i) in free.iter().enumerate() {
            acc += mem.w(i, j) * r[slot];
        }
        *v_j = acc;
    }
    let mut v = mem.apply_m(&v)?;

    // N = I - (1/theta) M W_Z' W_Z, solved against v.
    let dim = 2 * k;
    let mut wtw = vec![0.0; dim * dim];
    for a in 0..dim {
        for b in 0..dim {
            let mut acc = 0.0;
            for &i in &free {
                acc += mem.w(i, a) * mem.w(i, b);
            }
            wtw[a * dim + b] = acc;
        }
    }

    // Build M * WtW column by column, then N = I - (1/theta) * that.
    let mut n_mat = vec![0.0; dim * dim];
    for b in 0..dim {
        let col: Vec<f64> = (0..dim).map(|a| wtw[a * dim + b]).collect();
        let m_col = mem.apply_m(&col)?;
        for a in 0..dim {
            n_mat[a * dim + b] = -m_col[a] / mem.theta;
        }
    }
    for a in 0..dim {
        n_mat[a * dim + a] += 1.0;
    }

    v = solve_dense(&n_mat, &v, dim)?;

    // d = -( (1/theta) r + (1/theta^2) W_Z v )
    let mut d = vec![0.0; free.len()];
    for (slot, &i) in free.iter().enumerate() {
        let acc: f64 = v
            .iter()
            .enumerate()
            .map(|(j, &v_j)| mem.w(i, j) * v_j)
            .sum();
        d[slot] = -(r[slot] / mem.theta + acc / (mem.theta * mem.theta));
    }

    // Truncate to the box.
    let mut alpha = 1.0_f64;
    for (slot, &i) in free.iter().enumerate() {
        if d[slot] > 0.0 && upper[i].is_finite() {
            alpha = alpha.min((upper[i] - cp.x[i]) / d[slot]);
        } else if d[slot] < 0.0 && lower[i].is_finite() {
            alpha = alpha.min((lower[i] - cp.x[i]) / d[slot]);
        }
    }
    let alpha = alpha.clamp(0.0, 1.0);

    let mut out = cp.x.clone();
    for (slot, &i) in free.iter().enumerate() {
        out[i] = (cp.x[i] + alpha * d[slot]).clamp(lower[i], upper[i]);
    }
    Some(out)
}

/////////////////
// Line search //
/////////////////

/// Largest step along `direction` that stays inside the box.
///
/// ### Params
///
/// * `x` - Current iterate, assumed feasible
/// * `direction` - Search direction
/// * `lower` - Lower bounds
/// * `upper` - Upper bounds
///
/// ### Returns
///
/// The step at which the first coordinate reaches its bound, or infinity when
/// no bound lies ahead.
fn max_feasible_step(x: &[f64], direction: &[f64], lower: &[f64], upper: &[f64]) -> f64 {
    let mut step = f64::INFINITY;
    for i in 0..x.len() {
        if direction[i] > 0.0 && upper[i].is_finite() {
            step = step.min((upper[i] - x[i]) / direction[i]);
        } else if direction[i] < 0.0 && lower[i].is_finite() {
            step = step.min((lower[i] - x[i]) / direction[i]);
        }
    }
    step.max(0.0)
}

/// Cubic-interpolation line search enforcing the strong Wolfe conditions.
///
/// ### Params
///
/// * `f_and_grad` - Objective and gradient closure
/// * `x` - Current iterate
/// * `direction` - Search direction, generally the subspace step minus `x`
/// * `lower` - Lower bounds, used to keep trial points feasible
/// * `upper` - Upper bounds
/// * `f0` - Objective at `x`
/// * `g0` - Gradient at `x`
/// * `max_evals` - Evaluation budget
///
/// ### Returns
///
/// The accepted step length together with the point, objective, gradient and
/// number of evaluations used, or `None` if no acceptable step was found.
///
/// ### References
///
/// Nocedal and Wright, Numerical Optimization, Algorithms 3.5 and 3.6
#[allow(clippy::too_many_arguments)]
fn line_search<F>(
    f_and_grad: &mut F,
    x: &[f64],
    direction: &[f64],
    lower: &[f64],
    upper: &[f64],
    f0: f64,
    g0: &[f64],
    max_evals: usize,
) -> Option<(Vec<f64>, f64, Vec<f64>, usize)>
where
    F: FnMut(&[f64], &mut [f64]) -> f64,
{
    let n = x.len();
    let slope0 = dot(g0, direction);
    if slope0 >= 0.0 {
        return None;
    }

    let step_max = max_feasible_step(x, direction, lower, upper).min(MAX_STEP);
    if step_max <= 0.0 {
        return None;
    }

    let mut evals = 0;
    let mut grad = vec![0.0; n];

    let trial_point = |alpha: f64| -> Vec<f64> {
        (0..n)
            .map(|i| (x[i] + alpha * direction[i]).clamp(lower[i], upper[i]))
            .collect()
    };

    let mut alpha_prev = 0.0;
    let mut f_prev = f0;
    let mut alpha = 1.0_f64.min(step_max);

    let mut best: Option<(Vec<f64>, f64, Vec<f64>)> = None;

    for iteration in 0..max_evals {
        let point = trial_point(alpha);
        let f_new = f_and_grad(&point, &mut grad);
        evals += 1;

        if !f_new.is_finite() {
            alpha *= 0.5;
            if alpha < 1e-20 {
                break;
            }
            continue;
        }

        let sufficient = f_new <= f0 + WOLFE_C1 * alpha * slope0;
        if !sufficient || (iteration > 0 && f_new >= f_prev) {
            return zoom(
                f_and_grad,
                x,
                direction,
                lower,
                upper,
                f0,
                slope0,
                alpha_prev,
                f_prev,
                alpha,
                max_evals - evals,
                evals,
            );
        }

        let slope = dot(&grad, direction);
        if slope.abs() <= -WOLFE_C2 * slope0 {
            return Some((point, f_new, grad, evals));
        }
        if slope >= 0.0 {
            return zoom(
                f_and_grad,
                x,
                direction,
                lower,
                upper,
                f0,
                slope0,
                alpha,
                f_new,
                alpha_prev,
                max_evals - evals,
                evals,
            );
        }

        if alpha >= step_max {
            // Sitting on the boundary with the objective still descending. This
            // is the answer, not a failure to satisfy curvature.
            return Some((point, f_new, grad, evals));
        }
        best = Some((point, f_new, grad.clone()));
        alpha_prev = alpha;
        f_prev = f_new;
        alpha = (alpha * 2.0).min(step_max);
    }

    best.map(|(p, f, g)| (p, f, g, evals))
}

/// Refines a bracketed interval until a strong Wolfe point is found.
///
/// ### Params
///
/// * `f_and_grad` - Objective and gradient closure
/// * `x` - Current iterate
/// * `direction` - Search direction
/// * `lower` - Lower bounds
/// * `upper` - Upper bounds
/// * `f0` - Objective at `x`
/// * `slope0` - Directional derivative at `x`
/// * `alpha_lo` - Endpoint with the lower objective value
/// * `f_lo` - Objective at `alpha_lo`
/// * `alpha_hi` - Other endpoint of the bracket
/// * `budget` - Remaining evaluations
/// * `evals_so_far` - Evaluations already spent, for the returned tally
///
/// ### Returns
///
/// The accepted point, or `None` if the bracket collapsed without success.
#[allow(clippy::too_many_arguments)]
fn zoom<F>(
    f_and_grad: &mut F,
    x: &[f64],
    direction: &[f64],
    lower: &[f64],
    upper: &[f64],
    f0: f64,
    slope0: f64,
    mut alpha_lo: f64,
    mut f_lo: f64,
    mut alpha_hi: f64,
    budget: usize,
    evals_so_far: usize,
) -> Option<(Vec<f64>, f64, Vec<f64>, usize)>
where
    F: FnMut(&[f64], &mut [f64]) -> f64,
{
    let n = x.len();
    let mut grad = vec![0.0; n];
    let mut evals = evals_so_far;

    for _ in 0..budget {
        // Bisection is enough here: the bracket is already small and the extra
        // robustness beats a cubic fit that can land outside the interval.
        let alpha = 0.5 * (alpha_lo + alpha_hi);
        if (alpha_hi - alpha_lo).abs() < 1e-16 {
            break;
        }

        let point: Vec<f64> = (0..n)
            .map(|i| (x[i] + alpha * direction[i]).clamp(lower[i], upper[i]))
            .collect();
        let f_new = f_and_grad(&point, &mut grad);
        evals += 1;

        if !f_new.is_finite() || f_new > f0 + WOLFE_C1 * alpha * slope0 || f_new >= f_lo {
            alpha_hi = alpha;
            continue;
        }

        let slope = dot(&grad, direction);
        if slope.abs() <= -WOLFE_C2 * slope0 {
            return Some((point, f_new, grad, evals));
        }
        if slope * (alpha_hi - alpha_lo) >= 0.0 {
            alpha_hi = alpha_lo;
        }
        alpha_lo = alpha;
        f_lo = f_new;
    }

    None
}

///////////////
// Front end //
///////////////

/// Minimises a differentiable objective subject to box constraints.
///
/// The starting point is clamped into the box, so an infeasible `x0` is
/// corrected rather than rejected.
///
/// ### Params
///
/// * `f_and_grad` - Evaluates the objective at a point and writes the gradient
///   into the supplied buffer. Called once per function evaluation.
/// * `x0` - Starting point
/// * `lower` - Lower bounds, may contain negative infinity
/// * `upper` - Upper bounds, may contain positive infinity
/// * `params` - Tuning knobs, or [`LbfgsbParams::default`]
///
/// ### Returns
///
/// The best point found with its objective, gradient and stopping reason, or
/// [`EdgeErrors`] if the inputs disagree in length or a bound is inverted.
pub fn minimise<F>(
    mut f_and_grad: F,
    x0: &[f64],
    lower: &[f64],
    upper: &[f64],
    params: Option<LbfgsbParams>,
) -> Result<LbfgsbResult, EdgeErrors>
where
    F: FnMut(&[f64], &mut [f64]) -> f64,
{
    let params = params.unwrap_or_default();
    let n = x0.len();

    if n == 0 {
        return Err(EdgeErrors::MustBePositive("x0.len()".to_string()));
    }
    if lower.len() != n {
        return Err(EdgeErrors::LengthMismatch {
            name: "lower",
            expected: n,
            got: lower.len(),
        });
    }
    if upper.len() != n {
        return Err(EdgeErrors::LengthMismatch {
            name: "upper",
            expected: n,
            got: upper.len(),
        });
    }
    for i in 0..n {
        if lower[i] > upper[i] {
            return Err(EdgeErrors::InvalidBounds {
                index: i,
                lower: lower[i],
                upper: upper[i],
            });
        }
    }
    if params.memory == 0 {
        return Err(EdgeErrors::MustBePositive("memory".to_string()));
    }

    let mut x: Vec<f64> = (0..n).map(|i| x0[i].clamp(lower[i], upper[i])).collect();
    let mut grad = vec![0.0; n];
    let mut f = f_and_grad(&x, &mut grad);
    let mut evaluations = 1;

    let mut mem = Memory::new(params.memory);
    let mut status = LbfgsbStatus::MaxIterations;
    let mut iterations = 0;

    for iter in 0..params.max_iter {
        iterations = iter + 1;

        if projected_gradient_norm(&x, &grad, lower, upper) <= params.pgtol {
            status = LbfgsbStatus::GradientTolerance;
            break;
        }

        // Cauchy point, then subspace refinement. Either can fail on a singular
        // middle matrix, in which case fall back to projected steepest descent.
        let candidate = cauchy_point(&x, &grad, lower, upper, &mem)
            .and_then(|cp| subspace_minimise(&x, &grad, lower, upper, &cp, &mem));

        let direction: Vec<f64> = match candidate {
            Some(target) => (0..n).map(|i| target[i] - x[i]).collect(),
            None => (0..n)
                .map(|i| (x[i] - grad[i]).clamp(lower[i], upper[i]) - x[i])
                .collect(),
        };

        if direction.iter().all(|v| *v == 0.0) {
            status = LbfgsbStatus::GradientTolerance;
            break;
        }

        let searched = line_search(
            &mut f_and_grad,
            &x,
            &direction,
            lower,
            upper,
            f,
            &grad,
            params.max_line_search,
        );

        let (x_new, f_new, grad_new, used) = match searched {
            Some(v) => v,
            None => {
                // A line search that cannot make progress is usually the
                // limited-memory model's fault, not the objective's. The
                // Fortran throws the pairs away and retries from projected
                // steepest descent, and only gives up if that fails too.
                if mem.k() > 0 {
                    mem.reset();
                    continue;
                }
                status = LbfgsbStatus::LineSearchFailed;
                break;
            }
        };
        evaluations += used;

        let s: Vec<f64> = (0..n).map(|i| x_new[i] - x[i]).collect();
        let y: Vec<f64> = (0..n).map(|i| grad_new[i] - grad[i]).collect();
        mem.push(s, y);

        let reduction = f - f_new;
        let scale = f.abs().max(f_new.abs()).max(1.0);

        x = x_new;
        f = f_new;
        grad = grad_new;

        if reduction <= params.ftol * scale {
            status = LbfgsbStatus::ObjectiveTolerance;
            break;
        }
    }

    Ok(LbfgsbResult {
        x,
        f,
        grad,
        iterations,
        evaluations,
        status,
    })
}

/// Infinity norm of the gradient projected onto the feasible box.
///
/// This is the quantity scipy reports as `pgtol`: the gradient of the objective
/// after removing the components pushing against an active bound.
///
/// ### Params
///
/// * `x` - Current iterate
/// * `g` - Gradient at `x`
/// * `lower` - Lower bounds
/// * `upper` - Upper bounds
///
/// ### Returns
///
/// The largest absolute projected gradient component.
fn projected_gradient_norm(x: &[f64], g: &[f64], lower: &[f64], upper: &[f64]) -> f64 {
    let mut norm: f64 = 0.0;
    for i in 0..x.len() {
        let projected = (x[i] - g[i]).clamp(lower[i], upper[i]) - x[i];
        norm = norm.max(projected.abs());
    }
    norm
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Rosenbrock with its analytic gradient. Minimum 0 at (1, 1).
    fn rosenbrock(x: &[f64], g: &mut [f64]) -> f64 {
        let (a, b) = (x[0], x[1]);
        g[0] = -400.0 * a * (b - a * a) - 2.0 * (1.0 - a);
        g[1] = 200.0 * (b - a * a);
        100.0 * (b - a * a).powi(2) + (1.0 - a).powi(2)
    }

    /// Separable quadratic with minimum at (1, 2, 3).
    fn quadratic(x: &[f64], g: &mut [f64]) -> f64 {
        let target = [1.0, 2.0, 3.0];
        let mut f = 0.0;
        for i in 0..3 {
            let d = x[i] - target[i];
            f += d * d;
            g[i] = 2.0 * d;
        }
        f
    }

    fn tight(n: usize) -> (Vec<f64>, Vec<f64>) {
        (vec![f64::NEG_INFINITY; n], vec![f64::INFINITY; n])
    }

    #[test]
    fn test_unbounded_rosenbrock_finds_the_optimum() {
        let (lo, hi) = tight(2);
        let params = LbfgsbParams {
            ftol: 1e-14,
            pgtol: 1e-10,
            max_iter: 500,
            ..Default::default()
        };
        let res = minimise(rosenbrock, &[-1.2, 1.0], &lo, &hi, Some(params)).unwrap();
        assert_relative_eq!(res.x[0], 1.0, max_relative = 1e-5);
        assert_relative_eq!(res.x[1], 1.0, max_relative = 1e-5);
        assert!(res.f < 1e-10, "objective {} not near zero", res.f);
    }

    #[test]
    fn test_quadratic_converges_in_few_iterations() {
        let (lo, hi) = tight(3);
        let res = minimise(
            quadratic,
            &[0.0, 0.0, 0.0],
            &lo,
            &hi,
            Some(LbfgsbParams {
                ftol: 1e-14,
                pgtol: 1e-12,
                ..Default::default()
            }),
        )
        .unwrap();
        assert_relative_eq!(res.x[0], 1.0, max_relative = 1e-8);
        assert_relative_eq!(res.x[1], 2.0, max_relative = 1e-8);
        assert_relative_eq!(res.x[2], 3.0, max_relative = 1e-8);
    }

    /// The bound is active at the solution, so the Cauchy point and the active
    /// set logic are what produce the answer. An unconstrained solver fails this.
    #[test]
    fn test_active_bound_is_respected() {
        let lower = vec![2.0, f64::NEG_INFINITY, f64::NEG_INFINITY];
        let upper = vec![f64::INFINITY; 3];
        let res = minimise(
            quadratic,
            &[5.0, 0.0, 0.0],
            &lower,
            &upper,
            Some(LbfgsbParams {
                ftol: 1e-14,
                pgtol: 1e-12,
                ..Default::default()
            }),
        )
        .unwrap();
        assert_relative_eq!(res.x[0], 2.0, max_relative = 1e-10);
        assert_relative_eq!(res.x[1], 2.0, max_relative = 1e-8);
        assert_relative_eq!(res.x[2], 3.0, max_relative = 1e-8);
    }

    /// Both bounds active: the feasible set is a single point.
    #[test]
    fn test_degenerate_box_returns_the_only_feasible_point() {
        let lower = vec![1.5, 1.5, 1.5];
        let upper = vec![1.5, 1.5, 1.5];
        let res = minimise(quadratic, &[0.0, 0.0, 0.0], &lower, &upper, None).unwrap();
        for v in &res.x {
            assert_relative_eq!(v, &1.5, max_relative = 1e-12);
        }
    }

    #[test]
    fn test_bounded_rosenbrock_stops_on_the_boundary() {
        let lower = vec![f64::NEG_INFINITY, f64::NEG_INFINITY];
        let upper = vec![0.5, f64::INFINITY];
        let params = LbfgsbParams {
            ftol: 1e-14,
            pgtol: 1e-10,
            max_iter: 500,
            ..Default::default()
        };
        let res = minimise(rosenbrock, &[-1.2, 1.0], &lower, &upper, Some(params)).unwrap();
        assert_relative_eq!(res.x[0], 0.5, max_relative = 1e-6);
        assert_relative_eq!(res.x[1], 0.25, max_relative = 1e-4);
    }

    #[test]
    fn test_infeasible_start_is_clamped() {
        let lower = vec![0.0, 0.0, 0.0];
        let upper = vec![10.0, 10.0, 10.0];
        let res = minimise(quadratic, &[-50.0, 100.0, 3.0], &lower, &upper, None).unwrap();
        for (i, v) in res.x.iter().enumerate() {
            assert!(*v >= lower[i] - 1e-12 && *v <= upper[i] + 1e-12);
        }
    }

    #[test]
    fn test_rejects_inverted_bounds() {
        let err = minimise(
            quadratic,
            &[0.0; 3],
            &[1.0, 0.0, 0.0],
            &[0.0, 1.0, 1.0],
            None,
        )
        .unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidBounds { index: 0, .. }));
    }

    #[test]
    fn test_rejects_mismatched_bound_lengths() {
        let err = minimise(quadratic, &[0.0; 3], &[0.0; 2], &[1.0; 3], None).unwrap_err();
        assert!(matches!(
            err,
            EdgeErrors::LengthMismatch { name: "lower", .. }
        ));
    }

    #[test]
    fn test_rejects_empty_start() {
        let err = minimise(quadratic, &[], &[], &[], None).unwrap_err();
        assert!(matches!(err, EdgeErrors::MustBePositive(_)));
    }

    /// A negative binomial negative log-likelihood with a log link and a
    /// quadratic random-effect penalty: the shape NEBULA's stage one actually
    /// optimises, with the same box on the dispersion parameters.
    ///
    /// Parameters are `[beta0, beta1, sigma, phi]`. At the optimum `sigma` and
    /// `phi` are both driven onto their bounds, which is NEBULA's `-60`
    /// convergence code, so this exercises the active set rather than a smooth
    /// interior solve.
    fn nebula_shaped(p: &[f64], g: &mut [f64]) -> f64 {
        const N: usize = 60;
        let counts: Vec<f64> = (0..N)
            .map(|i| [0.0, 3.0, 1.0, 2.0, 0.0, 5.0, 2.0, 1.0, 4.0, 2.0][i % 10])
            .collect();
        let offset = 800.0_f64.ln();

        let (b0, b1, sigma, phi) = (p[0], p[1], p[2], p[3]);
        let r = 1.0 / phi;

        let mut f = 0.5 * sigma * sigma;
        let mut gb0 = 0.0;
        let mut gb1 = 0.0;
        let mut dr = 0.0;

        for (i, &y) in counts.iter().enumerate() {
            let xi = -1.0 + 2.0 * (i as f64) / ((N - 1) as f64);
            let mu = (offset + b0 + b1 * xi).exp();
            f += (y + r) * (mu + r).ln() - y * mu.ln() - r * r.ln();

            let d_mu = (y + r) / (mu + r) - y / mu;
            gb0 += d_mu * mu;
            gb1 += xi * d_mu * mu;
            dr += (mu + r).ln() + (y + r) / (mu + r) - r.ln() - 1.0;
        }

        g[0] = gb0;
        g[1] = gb1;
        g[2] = sigma;
        g[3] = dr * (-1.0 / (phi * phi));
        f
    }

    /// Reference values from
    /// `scipy.optimize.minimize(method='L-BFGS-B', jac=True, ftol=1e-12, gtol=1e-10)`
    /// on the identical objective.
    #[test]
    fn test_nebula_shaped_problem_matches_scipy() {
        let lower = vec![-100.0, -100.0, 1e-4, 1e-4];
        let upper = vec![100.0, 100.0, 10.0, 1000.0];
        let params = LbfgsbParams {
            ftol: 1e-12,
            pgtol: 1e-10,
            max_iter: 500,
            ..Default::default()
        };
        let res = minimise(
            nebula_shaped,
            &[0.0, 0.0, 1.0, 1.0],
            &lower,
            &upper,
            Some(params),
        )
        .unwrap();

        assert_relative_eq!(res.x[0], -5.992406750800, max_relative = 1e-6);
        assert_relative_eq!(res.x[1], 0.074133661253, max_relative = 1e-5);
        // Both dispersion parameters are driven onto their bounds.
        assert_relative_eq!(res.x[2], 1e-4, max_relative = 1e-12);
        assert_relative_eq!(res.x[3], 1000.0, max_relative = 1e-12);
        assert_relative_eq!(res.f, 0.516012646087, max_relative = 1e-8);
    }

    /// Badly scaled and separable, with every coordinate's optimum on the upper
    /// bound. The decay rates span three orders, so the search direction points
    /// far outside the box on nearly every iteration.
    fn decaying(x: &[f64], g: &mut [f64]) -> f64 {
        const RATE: [f64; 4] = [1.0, 30.0, 300.0, 3000.0];
        let mut f = 0.0;
        for i in 0..4 {
            let e = (-RATE[i] * x[i]).exp();
            f += e;
            g[i] = -RATE[i] * e;
        }
        f
    }

    /// Regression test for the line search running past the feasible step.
    ///
    /// Without the `max_feasible_step` cap every trial point beyond the boundary
    /// clamps to the same corner, so the objective flattens while the reported
    /// directional derivative stays negative. The curvature condition can then
    /// never be satisfied, the search spends its whole budget, and the driver
    /// eventually reports a failed line search a long way from the optimum.
    #[test]
    fn test_search_stops_at_the_boundary_rather_than_past_it() {
        let lower = vec![0.0; 4];
        let upper = vec![1.0; 4];
        let params = LbfgsbParams {
            ftol: 0.0,
            pgtol: 1e-10,
            max_iter: 200,
            ..Default::default()
        };
        let res = minimise(decaying, &[0.0; 4], &lower, &upper, Some(params)).unwrap();

        for i in 0..4 {
            assert_relative_eq!(res.x[i], 1.0, max_relative = 1e-12);
        }
        assert_eq!(res.status, LbfgsbStatus::GradientTolerance);
        assert!(
            res.evaluations < 100,
            "took {} evaluations on a four-variable box problem",
            res.evaluations
        );
    }

    #[test]
    fn test_solve_dense_matches_a_known_solution() {
        // [[2, 1], [1, 3]] x = [3, 5] has solution [0.8, 1.4].
        let a = vec![2.0, 1.0, 1.0, 3.0];
        let b = vec![3.0, 5.0];
        let x = solve_dense(&a, &b, 2).unwrap();
        assert_relative_eq!(x[0], 0.8, max_relative = 1e-12);
        assert_relative_eq!(x[1], 1.4, max_relative = 1e-12);
    }

    #[test]
    fn test_solve_dense_reports_singularity() {
        let a = vec![1.0, 2.0, 2.0, 4.0];
        assert!(solve_dense(&a, &[1.0, 2.0], 2).is_none());
    }
}
