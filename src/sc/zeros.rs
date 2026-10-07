//! Sums over cells with a zero count, as functions of one scalar.
//!
//! Within a subject, cells that share a design row differ only in their
//! offset. At a count of zero, every per-cell term NEBULA's kernels sum is a
//! function of `v = tau * O`, with `O = exp(offset)` the cell's own and `tau`
//! shared by the group: `exp(x'beta + log_w) / gamma` in the penalised fit,
//! `ym * exp(x'beta) / gamma` in the marginal likelihood. So the sum over a
//! group is a function of `s = ln(tau)` alone, the same for every gene, and is
//! tabulated once per run. A gene then pays one lookup per group plus a sweep
//! over its positive counts, instead of a sweep over every cell.
//!
//! The six sums are `L = sum ln(1 + v)` and its first five derivatives in `s`:
//! `A = sum v/(1+v)`, `B = sum v/(1+v)^2`, `C`, `D` and `E` (the curvature and
//! the higher-order Laplace terms). Each is a piecewise Chebyshev interpolant.
//! `L`, `A` and `B` are positive and are interpolated on the log scale, so the
//! error is relative; `C`, `D` and `E` change sign and are interpolated as
//! ratios to `B`, all bounded by one. Past the tabulated range every `v` is
//! below `exp(-ASYMPTOTE_DEPTH)` or above its inverse, and closed forms from
//! the group's offset sums take over.
//!
//! Groups smaller than [`MIN_GROUP_CELLS`] are cheaper to sweep and are left
//! out. A design with a continuous cell-level column has one cell per group, so
//! [`ZeroCells::build`] returns `None` and the kernels sweep every cell.
//!
//! ### References
//!
//! Trefethen. *Approximation Theory and Approximation Practice.* SIAM (2013).

#![allow(clippy::needless_range_loop)]

use std::collections::HashMap;

use rayon::prelude::*;

////////////
// Consts //
////////////

/// Number of tabulated sums: `L`, `A`, `B`, `C`, `D`, `E`.
pub(crate) const N_SUMS: usize = 6;

/// Chebyshev degree of each panel.
const DEGREE: usize = 16;

/// Points per panel, the Chebyshev-Lobatto nodes of [`DEGREE`].
const POINTS: usize = DEGREE + 1;

/// Width of a base panel in `s`.
///
/// Every sum has poles at `s + ln O = +-i pi`, so a panel of width one
/// converges to `1e-18` at degree 16; a base of four lets the near-linear
/// tails of the log forms pass unsplit and refines only the transition.
const BASE_WIDTH: f64 = 4.0;

/// Rounding allowance per unit of a form's size, in ulps, added to
/// [`PANEL_TOL`].
const NOISE_ULPS: f64 = 32.0;

/// Deepest halving of a base panel.
const MAX_DEPTH: u8 = 6;

/// Largest tolerated sum of the last two Chebyshev coefficients of any of the
/// six interpolated forms; a panel above it is halved.
///
/// Measured 2026-10-07 on 500 offsets spread over two orders of magnitude:
/// worst relative error against the compensated direct sum `8e-14`, at the far
/// small-`v` tail where `ln L` is near `-35` and its rounding dominates; inside
/// the transition `1e-14`. 33 panels per group.
const PANEL_TOL: f64 = 1e-14;

/// How far into either tail, in `ln v`, the table reaches before the closed
/// forms take over. `exp(-40)` is `4e-18`, under the rounding of a sum.
const ASYMPTOTE_DEPTH: f64 = 40.0;

/// Smallest group worth a table. Below it a lookup costs more than the sweep
/// over the group's cells.
pub(crate) const MIN_GROUP_CELLS: usize = 32;

/// Share of the cells that must sit in tabled groups for the tables to be built
/// at all.
const MIN_TABLED_SHARE: f64 = 0.5;

///////////
// Table //
///////////

/// Chebyshev coefficients of the six interpolated forms on one panel.
type Panel = [[f64; POINTS]; N_SUMS];

/// Piecewise Chebyshev interpolants over `[lo, lo + n_base * BASE_WIDTH)`.
///
/// Each base panel is split uniformly into `2^depth` panels, so a lookup is
/// two divisions and no search.
struct Table {
    /// Left end of the domain.
    lo: f64,
    /// Number of base panels.
    n_base: usize,
    /// Per base panel: index of its first panel in `panels`, and its depth.
    index: Vec<(u32, u8)>,
    /// The panels, base panel by base panel, left to right.
    panels: Vec<Panel>,
}

///////////
// Group //
///////////

/// The cells of one subject that share a design row.
pub(crate) struct Group {
    /// A cell of the group, whose design row every member shares.
    pub(crate) row: usize,
    /// Number of cells.
    pub(crate) n: f64,
    /// `sum O`, exact: the small-`v` tail and the marginal likelihood's
    /// subject sums.
    pub(crate) sum_o: f64,
    /// `sum 1 / O`, for the large-`v` tail.
    sum_inv_o: f64,
    /// `sum ln O`, for the large-`v` tail.
    sum_log_o: f64,
    /// Below this `s` every `v` is in the small tail.
    s_lo: f64,
    /// Above this `s` every `v` is in the large tail.
    s_hi: f64,
    /// The interpolants over `[s_lo, s_hi]`.
    table: Table,
}

impl Group {
    /// `M` consecutive sums at `s = ln(tau)`, starting at sum `FROM`.
    ///
    /// Sums are numbered `L, A, B, C, D, E`. Asking for `C`, `D` or `E`
    /// requires `B` in the same call, as they are stored as ratios to it.
    ///
    /// ### Params
    ///
    /// * `s` - `ln(tau)`
    ///
    /// ### Returns
    ///
    /// The sums over the group's cells at a count of zero.
    #[inline]
    pub(crate) fn eval<const FROM: usize, const M: usize>(&self, s: f64) -> [f64; M] {
        debug_assert!(FROM + M <= N_SUMS && (FROM + M <= 3 || FROM <= 2));
        let mut out = [0.0; M];
        if s < self.s_lo {
            // Every sum is `sum v` to first order, and `v^2` is below rounding.
            out.fill(s.exp() * self.sum_o);
            return out;
        }
        if s > self.s_hi {
            let r = (-s).exp() * self.sum_inv_o;
            for (j, o) in out.iter_mut().enumerate() {
                *o = match FROM + j {
                    0 => self.n * s + self.sum_log_o + r,
                    1 => self.n - r,
                    2 | 4 => r,
                    _ => -r,
                };
            }
            return out;
        }

        let t = &self.table;
        let pos = (s - t.lo) / BASE_WIDTH;
        let base = (pos as usize).min(t.n_base - 1);
        let (first, depth) = t.index[base];
        let parts = (1usize << depth) as f64;
        let within = ((pos - base as f64) * parts).clamp(0.0, parts - 1e-9);
        let sub = within as usize;
        let x = 2.0 * (within - sub as f64) - 1.0;
        let panel = &t.panels[first as usize + sub];

        let mut b_value = 0.0;
        for j in 0..M {
            let f = clenshaw(&panel[FROM + j], x);
            out[j] = if FROM + j < 3 { f.exp() } else { f };
            if FROM + j == 2 {
                b_value = out[j];
            }
        }
        for j in 0..M {
            if FROM + j >= 3 {
                out[j] *= b_value;
            }
        }
        out
    }
}

///////////////
// ZeroCells //
///////////////

/// The run's tabled groups, and the cells left to sweep, per subject.
pub(crate) struct ZeroCells {
    /// Every tabled group, subject by subject.
    pub(crate) groups: Vec<Group>,
    /// Subject `s` owns `groups[subject_groups[s]..subject_groups[s + 1]]`.
    pub(crate) subject_groups: Vec<usize>,
    /// Cells in no tabled group, ascending, subject by subject.
    pub(crate) loose: Vec<usize>,
    /// Subject `s` owns `loose[subject_loose[s]..subject_loose[s + 1]]`.
    pub(crate) subject_loose: Vec<usize>,
}

impl ZeroCells {
    /// Groups the cells and tabulates every group large enough to pay.
    ///
    /// ### Params
    ///
    /// * `design` - Row-major `n_cells * n_coef` design
    /// * `log_offset` - Log offset per cell
    /// * `fid` - Subject boundaries, length `n_subjects + 1`
    /// * `n_coef` - Number of design columns
    ///
    /// ### Returns
    ///
    /// The tables, or `None` when under [`MIN_TABLED_SHARE`] of the cells
    /// would be covered and sweeping every cell is the better plan.
    pub(crate) fn build(
        design: &[f64],
        log_offset: &[f64],
        fid: &[usize],
        n_coef: usize,
    ) -> Option<Self> {
        let n_cells = log_offset.len();
        let k = fid.len() - 1;

        // Members of each candidate group, subject by subject.
        let mut members: Vec<Vec<usize>> = Vec::new();
        let mut subject_groups = vec![0usize; k + 1];
        let mut loose = Vec::new();
        let mut subject_loose = vec![0usize; k + 1];
        let mut tabled = 0usize;
        for s in 0..k {
            let mut by_row: HashMap<Vec<u64>, Vec<usize>> = HashMap::new();
            for r in fid[s]..fid[s + 1] {
                let key = design[r * n_coef..(r + 1) * n_coef]
                    .iter()
                    .map(|v| v.to_bits())
                    .collect();
                by_row.entry(key).or_default().push(r);
            }
            let mut subject_loose_cells = Vec::new();
            let mut kept: Vec<Vec<usize>> = Vec::new();
            for (_, cells) in by_row {
                if cells.len() >= MIN_GROUP_CELLS {
                    tabled += cells.len();
                    kept.push(cells);
                } else {
                    subject_loose_cells.extend(cells);
                }
            }
            kept.sort_by_key(|c| c[0]);
            members.extend(kept);
            subject_groups[s + 1] = members.len();
            subject_loose_cells.sort_unstable();
            loose.extend(subject_loose_cells);
            subject_loose[s + 1] = loose.len();
        }
        if (tabled as f64) < MIN_TABLED_SHARE * n_cells as f64 {
            return None;
        }

        let groups = members
            .par_iter()
            .map(|cells| {
                let logs: Vec<f64> = cells.iter().map(|&r| log_offset[r]).collect();
                build_group(cells[0], &logs)
            })
            .collect();

        Some(Self {
            groups,
            subject_groups,
            loose,
            subject_loose,
        })
    }
}

/// Tabulates one group.
///
/// ### Params
///
/// * `row` - A cell of the group
/// * `log_o` - `ln O` of every cell in the group
///
/// ### Returns
///
/// The group with its interpolants.
fn build_group(row: usize, log_o: &[f64]) -> Group {
    let mut sorted = log_o.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mut distinct: Vec<(f64, f64)> = Vec::new();
    for &l in &sorted {
        match distinct.last_mut() {
            Some((v, m)) if *v == l => *m += 1.0,
            _ => distinct.push((l, 1.0)),
        }
    }
    let lo_o = sorted[0];
    let hi_o = sorted[sorted.len() - 1];
    let s_lo = -ASYMPTOTE_DEPTH - hi_o;
    let s_hi = ASYMPTOTE_DEPTH - lo_o;

    let n_base = ((s_hi - s_lo) / BASE_WIDTH).ceil() as usize;
    let mut index = Vec::with_capacity(n_base);
    let mut panels = Vec::new();
    for b in 0..n_base {
        let left = s_lo + b as f64 * BASE_WIDTH;
        let mut depth = 0u8;
        let mut fitted = loop {
            let parts = 1usize << depth;
            let width = BASE_WIDTH / parts as f64;
            let fitted: Vec<(Panel, bool)> = (0..parts)
                .map(|p| fit_panel(&distinct, left + p as f64 * width, width))
                .collect();
            if depth == MAX_DEPTH || fitted.iter().all(|(_, ok)| *ok) {
                break fitted;
            }
            depth += 1;
        };
        index.push((panels.len() as u32, depth));
        panels.extend(fitted.drain(..).map(|(p, _)| p));
    }

    Group {
        row,
        n: log_o.len() as f64,
        sum_o: compensated(log_o.iter().map(|l| l.exp())),
        sum_inv_o: compensated(log_o.iter().map(|l| (-l).exp())),
        sum_log_o: compensated(log_o.iter().copied()),
        s_lo,
        s_hi,
        table: Table {
            lo: s_lo,
            n_base,
            index,
            panels,
        },
    }
}

/// Fits one panel and reports whether every form converged.
///
/// ### Params
///
/// * `distinct` - The group's distinct `ln O`, each with its multiplicity
/// * `left` - Left end of the panel in `s`
/// * `width` - Width of the panel
///
/// ### Returns
///
/// The coefficients, and whether their tails are under [`PANEL_TOL`].
fn fit_panel(distinct: &[(f64, f64)], left: f64, width: f64) -> (Panel, bool) {
    let mut values = [[0.0; POINTS]; N_SUMS];
    for j in 0..POINTS {
        // Lobatto node `cos(pi j / DEGREE)`, mapped onto the panel.
        let x = (std::f64::consts::PI * j as f64 / DEGREE as f64).cos();
        let sums = direct_sums(distinct, left + 0.5 * width * (x + 1.0));
        values[0][j] = sums[0].ln();
        values[1][j] = sums[1].ln();
        values[2][j] = sums[2].ln();
        for f in 3..N_SUMS {
            values[f][j] = sums[f] / sums[2];
        }
    }
    let mut panel = [[0.0; POINTS]; N_SUMS];
    let mut ok = true;
    for f in 0..N_SUMS {
        panel[f] = chebyshev_coefficients(&values[f]);
        // A log form of size 35 carries rounding of `35 eps` at every node,
        // which no refinement removes.
        let size = values[f].iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let tail = panel[f][DEGREE - 1].abs() + panel[f][DEGREE].abs();
        ok &= tail <= PANEL_TOL + NOISE_ULPS * f64::EPSILON * size;
    }
    (panel, ok)
}

/// The six sums at one `s`, summed directly and compensated.
///
/// ### Params
///
/// * `distinct` - Distinct `ln O`, each with its multiplicity
/// * `s` - `ln(tau)`
///
/// ### Returns
///
/// `[L, A, B, C, D, E]`.
fn direct_sums(distinct: &[(f64, f64)], s: f64) -> [f64; N_SUMS] {
    let mut acc = [Neumaier::default(); N_SUMS];
    for &(l, m) in distinct {
        let v = (s + l).exp();
        let w = 1.0 / (1.0 + v);
        let u = v * w;
        let uw = u * w;
        acc[0].add(m * v.ln_1p());
        acc[1].add(m * u);
        acc[2].add(m * uw);
        acc[3].add(m * uw * (w - u));
        acc[4].add(m * uw * (w * w - 4.0 * u * w + u * u));
        acc[5].add(m * uw * (w * w * w - 11.0 * w * w * u + 11.0 * w * u * u - u * u * u));
    }
    acc.map(|a| a.total())
}

/////////////
// Helpers //
/////////////

/// Neumaier's compensated sum.
#[derive(Clone, Copy, Default)]
struct Neumaier {
    /// Running sum.
    sum: f64,
    /// Accumulated rounding error.
    carry: f64,
}

impl Neumaier {
    /// Adds one term.
    ///
    /// ### Params
    ///
    /// * `x` - The term
    fn add(&mut self, x: f64) {
        let t = self.sum + x;
        if self.sum.abs() >= x.abs() {
            self.carry += (self.sum - t) + x;
        } else {
            self.carry += (x - t) + self.sum;
        }
        self.sum = t;
    }

    /// The compensated total.
    ///
    /// ### Returns
    ///
    /// `sum + carry`.
    fn total(self) -> f64 {
        self.sum + self.carry
    }
}

/// Compensated sum of an iterator.
///
/// ### Params
///
/// * `values` - The terms
///
/// ### Returns
///
/// Their sum.
fn compensated(values: impl Iterator<Item = f64>) -> f64 {
    let mut acc = Neumaier::default();
    for v in values {
        acc.add(v);
    }
    acc.total()
}

/// Chebyshev coefficients from values at the Lobatto nodes `cos(pi j / N)`.
///
/// ### Params
///
/// * `values` - Values at the nodes, `j = 0..=DEGREE`
///
/// ### Returns
///
/// Coefficients of `T_0..T_DEGREE`.
fn chebyshev_coefficients(values: &[f64; POINTS]) -> [f64; POINTS] {
    let n = DEGREE as f64;
    let mut c = [0.0; POINTS];
    for (k, ck) in c.iter_mut().enumerate() {
        let mut acc = 0.0;
        for (j, &f) in values.iter().enumerate() {
            let weight = if j == 0 || j == DEGREE { 0.5 } else { 1.0 };
            acc += weight * f * (std::f64::consts::PI * (j * k) as f64 / n).cos();
        }
        *ck = 2.0 * acc / n;
    }
    c[0] *= 0.5;
    c[DEGREE] *= 0.5;
    c
}

/// Evaluates a Chebyshev series by Clenshaw's recurrence.
///
/// ### Params
///
/// * `c` - Coefficients of `T_0..T_DEGREE`
/// * `x` - Point in `[-1, 1]`
///
/// ### Returns
///
/// `sum c_k T_k(x)`.
#[inline(always)]
fn clenshaw(c: &[f64; POINTS], x: f64) -> f64 {
    let two_x = 2.0 * x;
    let mut b1 = 0.0;
    let mut b2 = 0.0;
    for k in (1..POINTS).rev() {
        let b0 = c[k] + two_x * b1 - b2;
        b2 = b1;
        b1 = b0;
    }
    c[0] + x * b1 - b2
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    use rand::rngs::SmallRng;

    /// Log offsets spread like library sizes over two orders of magnitude.
    fn offsets(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = SmallRng::seed_from_u64(seed);
        (0..n).map(|_| rng.random_range(6.0..10.6)).collect()
    }

    #[test]
    fn test_group_matches_direct_sums_over_the_whole_range() {
        let log_o = offsets(500, 7);
        let group = build_group(0, &log_o);
        let distinct: Vec<(f64, f64)> = log_o.iter().map(|&l| (l, 1.0)).collect();
        let mut worst = [0.0f64; N_SUMS];
        let mut s = group.s_lo - 5.0;
        while s < group.s_hi + 5.0 {
            let want = direct_sums(&distinct, s);
            let got = group.eval::<0, 6>(s);
            for f in 0..N_SUMS {
                // C, D and E are judged against B, the scale they are stored at.
                let scale = if f < 3 { want[f].abs() } else { want[2] };
                worst[f] = worst[f].max((got[f] - want[f]).abs() / scale);
            }
            s += 0.0137;
        }
        for (f, w) in worst.iter().enumerate() {
            assert!(*w < 1e-13, "sum {f}: worst relative error {w:e}");
        }
    }

    #[test]
    fn test_build_skips_a_continuous_design() {
        let n = 400;
        let log_o = offsets(n, 3);
        let mut rng = SmallRng::seed_from_u64(9);
        let design: Vec<f64> = (0..n)
            .flat_map(|_| [1.0, rng.random_range(-1.0..1.0)])
            .collect();
        assert!(ZeroCells::build(&design, &log_o, &[0, 200, 400], 2).is_none());
    }

    #[test]
    fn test_build_groups_a_categorical_design() {
        let n = 400;
        let log_o = offsets(n, 3);
        let design: Vec<f64> = (0..n)
            .flat_map(|r| [1.0, (r % 3) as f64, if r % 50 == 0 { 0.5 } else { 0.0 }])
            .collect();
        let zeros = ZeroCells::build(&design, &log_o, &[0, 200, 400], 3).expect("tabled");
        assert_eq!(zeros.subject_groups, vec![0, 3, 6]);
        // The rows with the odd third column are too few to table.
        assert_eq!(zeros.loose, vec![0, 50, 100, 150, 200, 250, 300, 350]);
        let covered: f64 = zeros.groups.iter().map(|g| g.n).sum();
        assert_eq!(covered as usize + zeros.loose.len(), n);
    }
}
