//! Chebyshev approximations to the moments of the negative binomial unit
//! deviance, and the unit deviance itself.
//!
//! Port of edgeR's `ql_weights.c` and the deviance kernel of `compute_nbdev.c`.
//! The unit deviance of one observation behaves like a scaled chi-square, and
//! the quasi-likelihood machinery needs its first two moments. Matching
//! moments gives
//!
//! ```text
//! alpha = 2 * E[d] / Var[d]        kappa = 2 * E[d]^2 / Var[d]
//! ```
//!
//! so `alpha` is the reciprocal scale onto a chi-square and `kappa` its
//! degrees of freedom. Neither has a closed form, so edgeR ships piecewise
//! Chebyshev fits: in `mu` alone for the Poisson limit, in `(mu, phi)` jointly
//! for the negative binomial. Only above `phi = 4.001` does the C sum the
//! probability mass function directly.
//!
//! The coefficient tables are copied verbatim from `ql_weights.c` by way of
//! edgePython. Panel boundaries and the affine maps onto `[-1, 1]` live in
//! `Panel` tables next to the coefficients they index.
//!
//! ### Deviations from edgePython
//!
//! [`unit_nb_deviance`] follows edgeR's C: the small-`phi` series has
//! `2/3*resid` with integer operands, which is identically zero. edgePython
//! "corrects" it to `2.0/3.0*resid`. See `UPSTREAM_DEVIATIONS.md` A4.
//!
//! ### References
//!
//! Lund, Nettleton, McCarthy and Smyth, SAGMB 11(5), 2012

use crate::numeric::gamma::ln_gamma;
use crate::ql::chebyshev_tables::*;

////////////
// Consts //
////////////

/// Below this mean every weight is reported as an exact zero.
///
/// edgeR's `low_value`. The fits lose meaning once `mu` underflows their range,
/// and such a gene contributes nothing to the quasi-likelihood.
const MIN_MU: f64 = 1e-32;

/// Upper bound of the case 1 negative binomial fits.
const PHI_CASE1_MAX: f64 = 0.736;

/// Upper bound of the case 2 negative binomial fits.
///
/// Above this the fits are abandoned for a direct sum over the probability
/// mass function.
const PHI_CASE2_MAX: f64 = 4.001;

/// Half-width of the case 1 `phi` domain, which maps `[0, 0.736]` onto `[-1, 1]`.
const PHI_CASE1_HALF: f64 = 0.368;

/// Half-width of the case 2 `phi` domain, which maps `[0, 4]` onto `[-1, 1]`.
///
/// The fitted range stops at 4.001, so the largest `phi` reaching case 2 maps
/// marginally past 1.
const PHI_CASE2_HALF: f64 = 2.0;

/// Mean above which the Poisson fits give way to their asymptotic expansions.
const POIS_ASYMPTOTIC_MU: f64 = 20.0;

/// Mean below which the Poisson fits carry an explicit `log(mu)` factor.
///
/// Both moments vanish as `mu -> 0` like `mu * log(mu)^2`, which no polynomial
/// can follow, so the first panel fits the smooth remainder.
const POIS_LOG_MU: f64 = 0.02;

/// Coefficients per panel in the one-dimensional Poisson fits.
const POIS_PANEL_LEN: usize = 10;

/// Mean above which case 1 drops the two-dimensional fit for a `phi`-only one
/// times the Poisson asymptote.
const NB1_LARGE_MU: f64 = 60.0;

/// Mean above which case 1 switches from the joint `(mu, phi)` fit to the
/// separable edge interpolation.
const NB1_MID_MU: f64 = 20.0;

/// Side length of a case 1 two-dimensional panel, which holds `7 * 7` terms.
const NB1_PANEL_SIDE: usize = 7;

/// Coefficients per block in the case 1 separable and `phi`-only fits.
const NB1_MID_LEN: usize = 7;

/// Coefficients per block in the case 1 large-`mu` fit.
const NB1_LARGE_LEN: usize = 6;

/// Edge series in the case 1 separable table: four panels sharing five edges.
const NB1_MID_EDGES: usize = 5;

/// Mean above which case 2 leaves the joint `(mu, phi)` fit.
const NB2_MID_MU: f64 = 50.0;

/// Mean above which case 2 collapses to a `phi`-only fit.
///
/// Both moments are flat in `mu` here, so the `mu -> inf` edge of the separable
/// fit is used unchanged.
const NB2_LARGE_MU: f64 = 5000.0;

/// Breakpoints in `mu` for the case 2 reciprocal-`mu` panels.
///
/// The separable fit above [`NB2_MID_MU`] is polynomial in `1/mu`, so these
/// panels carry their own maps rather than a [`Panel`] table.
const NB2_MID_KNOTS: [f64; 2] = [100.0, 1000.0];

/// Side length of a case 2 two-dimensional panel, which holds `10 * 10` terms.
const NB2_PANEL_SIDE: usize = 10;

/// Coefficients per block in the case 2 separable and `phi`-only fits.
const NB2_MID_LEN: usize = 10;

/// Edge series in the case 2 separable table: three panels sharing four edges.
///
/// The last is the `mu -> inf` edge, the whole answer above [`NB2_LARGE_MU`].
const NB2_MID_EDGES: usize = 4;

/// Terms kept in the direct probability mass function sum beyond `mu^2 * phi`.
const NB_SUM_SLACK: f64 = 10.0;

/// Hard cap on terms in the direct probability mass function sum.
///
/// Above `phi = 4.001` almost all mass sits on the first few counts.
const NB_SUM_MAX: usize = 50;

/// Blocks of the case 1 large-`mu` fit for alpha.
const NB1_ALPHA_LARGE_BLOCKS: [(f64, usize); 3] = [(80.0, 0), (120.0, 6), (f64::INFINITY, 12)];

/// Blocks of the case 1 large-`mu` fit for kappa, which needs one more than
/// alpha does before it settles.
const NB1_KAPPA_LARGE_BLOCKS: [(f64, usize); 4] =
    [(80.0, 0), (120.0, 6), (250.0, 12), (f64::INFINITY, 18)];

////////////
// Panels //
////////////

/// Panels of the [`pois_alpha`] fit.
#[rustfmt::skip]
const POIS_ALPHA_PANELS: [Panel; 5] = [
    Panel { upper: 0.02, centre2: 0.02, width2: 0.02, offset: 0 },
    Panel { upper: 0.4249, centre2: 0.4449, width2: 0.4049, offset: 10 },
    Panel { upper: 1.5, centre2: 1.9249, width2: 1.0751, offset: 20 },
    Panel { upper: 3.544, centre2: 5.044, width2: 2.044, offset: 30 },
    Panel { upper: 20.0, centre2: 23.544, width2: 16.456, offset: 40 },
];

/// Panels of the [`pois_kappa`] fit.
#[rustfmt::skip]
const POIS_KAPPA_PANELS: [Panel; 5] = [
    Panel { upper: 0.02, centre2: 0.02, width2: 0.02, offset: 0 },
    Panel { upper: 0.4966, centre2: 0.5166, width2: 0.4766, offset: 10 },
    Panel { upper: 1.5, centre2: 1.9966, width2: 1.0034, offset: 20 },
    Panel { upper: 4.2714, centre2: 5.7714, width2: 2.7714, offset: 30 },
    Panel { upper: 20.0, centre2: 24.2714, width2: 15.7286, offset: 40 },
];

/// Panels of the case 1 joint fit for alpha, valid up to [`NB1_MID_MU`].
#[rustfmt::skip]
const NB1_ALPHA_PANELS: [Panel; 6] = [
    Panel { upper: 0.01, centre2: 0.01, width2: 0.01, offset: 0 },
    Panel { upper: 0.33, centre2: 0.34, width2: 0.32, offset: 49 },
    Panel { upper: 1.77, centre2: 2.1, width2: 1.44, offset: 98 },
    Panel { upper: 4.0, centre2: 5.77, width2: 2.23, offset: 147 },
    Panel { upper: 10.0, centre2: 14.0, width2: 6.0, offset: 196 },
    Panel { upper: f64::INFINITY, centre2: 30.0, width2: 10.0, offset: 245 },
];

/// Panels of the case 1 joint fit for kappa, valid up to [`NB1_MID_MU`].
#[rustfmt::skip]
const NB1_KAPPA_PANELS: [Panel; 6] = [
    Panel { upper: 0.01, centre2: 0.01, width2: 0.01, offset: 0 },
    Panel { upper: 0.33, centre2: 0.34, width2: 0.32, offset: 49 },
    Panel { upper: 1.3, centre2: 1.63, width2: 0.97, offset: 98 },
    Panel { upper: 4.0, centre2: 5.3, width2: 2.7, offset: 147 },
    Panel { upper: 10.0, centre2: 14.0, width2: 6.0, offset: 196 },
    Panel { upper: f64::INFINITY, centre2: 30.0, width2: 10.0, offset: 245 },
];

/// Panels of the case 1 separable fit, shared by both moments.
#[rustfmt::skip]
const NB1_MID_PANELS: [Panel; 4] = [
    Panel { upper: 25.0, centre2: 45.0, width2: 5.0, offset: 0 },
    Panel { upper: 30.0, centre2: 55.0, width2: 5.0, offset: 7 },
    Panel { upper: 40.0, centre2: 70.0, width2: 10.0, offset: 14 },
    Panel { upper: f64::INFINITY, centre2: 100.0, width2: 20.0, offset: 21 },
];

/// Panels of the case 2 joint fit for alpha, valid up to [`NB2_MID_MU`].
///
/// The first panel maps `[0, 0.02]` rather than `[0, 0.01]` onto `[-1, 1]`, so
/// only the left half of its domain is used. This mirrors the C.
#[rustfmt::skip]
const NB2_ALPHA_PANELS: [Panel; 6] = [
    Panel { upper: 0.01, centre2: 0.02, width2: 0.02, offset: 0 },
    Panel { upper: 0.43, centre2: 0.44, width2: 0.42, offset: 100 },
    Panel { upper: 3.62, centre2: 4.05, width2: 3.19, offset: 200 },
    Panel { upper: 10.0, centre2: 13.62, width2: 6.38, offset: 300 },
    Panel { upper: 30.0, centre2: 40.0, width2: 20.0, offset: 400 },
    Panel { upper: f64::INFINITY, centre2: 80.0, width2: 20.0, offset: 500 },
];

/// Panels of the case 2 joint fit for kappa, valid up to [`NB2_MID_MU`].
#[rustfmt::skip]
const NB2_KAPPA_PANELS: [Panel; 6] = [
    Panel { upper: 0.01, centre2: 0.02, width2: 0.02, offset: 0 },
    Panel { upper: 0.5, centre2: 0.51, width2: 0.49, offset: 100 },
    Panel { upper: 3.88, centre2: 4.38, width2: 3.38, offset: 200 },
    Panel { upper: 10.0, centre2: 13.88, width2: 6.12, offset: 300 },
    Panel { upper: 30.0, centre2: 40.0, width2: 20.0, offset: 400 },
    Panel { upper: f64::INFINITY, centre2: 80.0, width2: 20.0, offset: 500 },
];

////////////////////
// Panel dispatch //
////////////////////

/// One panel of a piecewise Chebyshev fit in `mu`.
///
/// Each panel has its own coefficient block and affine map onto `[-1, 1]`,
/// written `(2 * mu - centre2) / width2` as in the C source.
#[derive(Clone, Copy, Debug)]
struct Panel {
    /// Exclusive upper bound in `mu`; the last panel of a table uses infinity
    upper: f64,
    /// Twice the panel centre
    centre2: f64,
    /// Twice the panel half-width
    width2: f64,
    /// Index of the panel's first coefficient in its table
    offset: usize,
}

/// Picks the panel covering `mu` and maps `mu` onto `[-1, 1]`.
#[inline]
fn locate(panels: &[Panel], mu: f64) -> (f64, usize) {
    let panel = panels
        .iter()
        .find(|p| mu < p.upper)
        .unwrap_or_else(|| panels.last().expect("panel tables are never empty"));
    ((2.0 * mu - panel.centre2) / panel.width2, panel.offset)
}

/// Picks a coefficient offset from a table of `mu` breakpoints.
///
/// For the large-`mu` case 1 fits, which switch the `phi` coefficient block on
/// `mu` but apply no map in `mu`.
///
/// ### Params
///
/// * `blocks` - `(upper bound, offset)` pairs in ascending order, the last
///   entry unbounded. The bound is inclusive, matching the `mu > bound` tests
///   in the C source.
/// * `mu` - Mean to locate
///
/// ### Returns
///
/// The offset of the block holding `mu`.
#[inline]
fn locate_block(blocks: &[(f64, usize)], mu: f64) -> usize {
    match blocks.iter().find(|(upper, _)| mu <= *upper) {
        Some((_, offset)) => *offset,
        None => blocks.last().expect("block tables are never empty").1,
    }
}

/// Evaluates the three series a separable panel is built from.
///
/// A separable table holds one `phi` series per panel edge, then one `mu`
/// series per panel. Panel `p` reads its lower edge at `offset`, its upper edge
/// at `offset + len`, and its blend series `edges * len` further along.
///
/// ### Params
///
/// * `table` - Flat coefficient table
/// * `offset` - Offset of the panel's lower edge series
/// * `len` - Terms per series
/// * `edges` - Number of edge series in the table, which is the stride to the
///   blend series
/// * `x` - Mapped `mu`, in `[-1, 1]`
/// * `y` - Mapped `phi`, in `[-1, 1]`
///
/// ### Returns
///
/// `(lower edge, upper edge, blend)`. Case 1 and case 2 blend in opposite
/// directions, so the caller combines them.
#[inline]
fn edge_series(
    table: &[f64],
    offset: usize,
    len: usize,
    edges: usize,
    x: f64,
    y: f64,
) -> (f64, f64, f64) {
    let blend = offset + edges * len;
    (
        cheb_eval(&table[offset..offset + len], y),
        cheb_eval(&table[offset + len..offset + 2 * len], y),
        cheb_eval(&table[blend..blend + len], x),
    )
}

//////////////////////////
// Chebyshev evaluation //
//////////////////////////

/// Evaluates a Chebyshev series of the first kind at `x`.
///
/// Three-term recurrence, accumulated in ascending order to match the
/// reference term for term. No allocation.
///
/// ### Params
///
/// * `coefficients` - Series coefficients, lowest order first. The degree is
///   `coefficients.len() - 1`.
/// * `x` - Abscissa, nominally in `[-1, 1]`
///
/// ### Returns
///
/// `sum_i coefficients[i] * T_i(x)`, or zero for an empty slice.
#[inline]
pub fn cheb_eval(coefficients: &[f64], x: f64) -> f64 {
    match coefficients {
        [] => 0.0,
        [c0] => *c0,
        [c0, c1, rest @ ..] => {
            let mut previous = 1.0;
            let mut current = x;
            let mut out = c0 + c1 * x;
            for c in rest {
                let next = 2.0 * x * current - previous;
                out += c * next;
                previous = current;
                current = next;
            }
            out
        }
    }
}

/// Evaluates a tensor product Chebyshev series at `(x, y)`.
///
/// Coefficients are `y`-major: term `(i, j)` multiplying `T_j(x) * T_i(y)` sits
/// at `i * nx + j`. Both recurrences run inline, so no basis is materialised.
///
/// ### Params
///
/// * `coefficients` - At least `nx * ny` coefficients in `y`-major order
/// * `x` - First abscissa, nominally in `[-1, 1]`
/// * `y` - Second abscissa, nominally in `[-1, 1]`
/// * `nx` - Number of terms in `x`
/// * `ny` - Number of terms in `y`
///
/// ### Returns
///
/// `sum_i sum_j coefficients[i * nx + j] * T_j(x) * T_i(y)`.
///
/// ### Panics
///
/// If `coefficients` holds fewer than `nx * ny` values.
#[inline]
pub fn cheb_eval_2d(coefficients: &[f64], x: f64, y: f64, nx: usize, ny: usize) -> f64 {
    let coefficients = &coefficients[..nx * ny];
    let mut out = 0.0;
    let mut y_previous = 1.0;
    let mut y_current = y;

    for (i, row) in coefficients.chunks_exact(nx).enumerate() {
        let y_term = match i {
            0 => 1.0,
            1 => y,
            _ => {
                let next = 2.0 * y * y_current - y_previous;
                y_previous = y_current;
                y_current = next;
                next
            }
        };

        let mut x_previous = 1.0;
        let mut x_current = x;
        for (j, c) in row.iter().enumerate() {
            let x_term = match j {
                0 => 1.0,
                1 => x,
                _ => {
                    let next = 2.0 * x * x_current - x_previous;
                    x_previous = x_current;
                    x_current = next;
                    next
                }
            };
            out += c * x_term * y_term;
        }
    }
    out
}

///////////////////
// Poisson limit //
///////////////////

/// Large-`mu` expansion of the Poisson deviance scale.
#[inline]
fn pois_alpha_tail(mu: f64) -> f64 {
    1.0 - 1.0 / (6.0 * mu) - 1.0 / (2.0 * mu * mu)
}

/// Large-`mu` expansion of the Poisson deviance degrees of freedom.
#[inline]
fn pois_kappa_tail(mu: f64) -> f64 {
    1.0 - 1.0 / (2.5 * mu * mu)
}

/// Reciprocal scale of the Poisson unit deviance.
///
/// Piecewise Chebyshev below `POIS_ASYMPTOTIC_MU`, asymptotic expansion
/// above. The first panel divides out the `log(mu)` singularity at `mu -> 0`.
///
/// ### Params
///
/// * `mu` - Fitted mean
///
/// ### Returns
///
/// `2 E[d] / Var[d]` for `d` the Poisson unit deviance at `mu`, or zero when
/// `mu` is below `MIN_MU`.
pub fn pois_alpha(mu: f64) -> f64 {
    if mu < MIN_MU {
        return 0.0;
    }
    if mu >= POIS_ASYMPTOTIC_MU {
        return pois_alpha_tail(mu);
    }

    let (x, offset) = locate(&POIS_ALPHA_PANELS, mu);
    let out = cheb_eval(&POIS_ALPHA_COEF[offset..offset + POIS_PANEL_LEN], x);
    if mu < POIS_LOG_MU {
        let log_mu = mu.ln();
        -out * log_mu / ((1.0 + log_mu) * (1.0 + log_mu))
    } else {
        out
    }
}

/// Degrees of freedom of the Poisson unit deviance.
///
/// ### Params
///
/// * `mu` - Fitted mean
///
/// ### Returns
///
/// `2 E[d]^2 / Var[d]` for `d` the Poisson unit deviance at `mu`, or zero when
/// `mu` is below `MIN_MU`.
pub fn pois_kappa(mu: f64) -> f64 {
    if mu < MIN_MU {
        return 0.0;
    }
    if mu >= POIS_ASYMPTOTIC_MU {
        return pois_kappa_tail(mu);
    }

    let (x, offset) = locate(&POIS_KAPPA_PANELS, mu);
    let out = cheb_eval(&POIS_KAPPA_COEF[offset..offset + POIS_PANEL_LEN], x);
    if mu < POIS_LOG_MU {
        let log_mu = mu.ln() / (1.0 + mu.ln());
        out * mu * log_mu * log_mu
    } else {
        out
    }
}

////////////////////////////////////
// Negative binomial, phi < 0.736 //
////////////////////////////////////

/// Reciprocal scale of the negative binomial unit deviance, case 1.
///
/// Three regimes in `mu`. Below [`NB1_MID_MU`], a joint `7 x 7` fit in
/// `(mu, phi)` multiplies the Poisson answer. Up to [`NB1_LARGE_MU`], a
/// separable fit: two `phi` series give the panel edges and an `x` series
/// interpolates. Above, only a `phi` correction to the Poisson asymptote.
///
/// ### Params
///
/// * `mu` - Fitted mean
/// * `phi` - Dispersion, expected in `[0, 0.736)`
///
/// ### Returns
///
/// `2 E[d] / Var[d]`, or zero when `mu` is below [`MIN_MU`].
fn nb_alpha_case1(mu: f64, phi: f64) -> f64 {
    if mu < MIN_MU {
        return 0.0;
    }
    let y = phi / PHI_CASE1_HALF - 1.0;

    if mu > NB1_LARGE_MU {
        let offset = locate_block(&NB1_ALPHA_LARGE_BLOCKS, mu);
        let out = cheb_eval(&NB_A_1_3[offset..offset + NB1_LARGE_LEN], y);
        out * pois_alpha_tail(mu)
    } else if mu > NB1_MID_MU {
        let (x, offset) = locate(&NB1_MID_PANELS, mu);
        let (lower, upper, blend) =
            edge_series(&NB_A_1_2, offset, NB1_MID_LEN, NB1_MID_EDGES, x, y);
        (upper + (lower - upper) * blend) * pois_alpha_tail(mu)
    } else {
        let (x, offset) = locate(&NB1_ALPHA_PANELS, mu);
        let block = NB1_PANEL_SIDE * NB1_PANEL_SIDE;
        let out = cheb_eval_2d(
            &NB_A_1_1[offset..offset + block],
            x,
            y,
            NB1_PANEL_SIDE,
            NB1_PANEL_SIDE,
        );
        out * pois_alpha(mu)
    }
}

/// Degrees of freedom of the negative binomial unit deviance, case 1.
///
/// Same regimes as [`nb_alpha_case1`], with its own breakpoints and one extra
/// large-`mu` block.
///
/// ### Params
///
/// * `mu` - Fitted mean
/// * `phi` - Dispersion, expected in `[0, 0.736)`
///
/// ### Returns
///
/// `2 E[d]^2 / Var[d]`, or zero when `mu` is below [`MIN_MU`].
fn nb_kappa_case1(mu: f64, phi: f64) -> f64 {
    if mu < MIN_MU {
        return 0.0;
    }
    let y = phi / PHI_CASE1_HALF - 1.0;

    if mu > NB1_LARGE_MU {
        let offset = locate_block(&NB1_KAPPA_LARGE_BLOCKS, mu);
        let out = cheb_eval(&NB_K_1_3[offset..offset + NB1_LARGE_LEN], y);
        out * pois_kappa_tail(mu)
    } else if mu > NB1_MID_MU {
        let (x, offset) = locate(&NB1_MID_PANELS, mu);
        let (lower, upper, blend) =
            edge_series(&NB_K_1_2, offset, NB1_MID_LEN, NB1_MID_EDGES, x, y);
        (upper + (lower - upper) * blend) * pois_kappa_tail(mu)
    } else {
        let (x, offset) = locate(&NB1_KAPPA_PANELS, mu);
        let block = NB1_PANEL_SIDE * NB1_PANEL_SIDE;
        let out = cheb_eval_2d(
            &NB_K_1_1[offset..offset + block],
            x,
            y,
            NB1_PANEL_SIDE,
            NB1_PANEL_SIDE,
        );
        out * pois_kappa(mu)
    }
}

/////////////////////////////////////////////
// Negative binomial, 0.736 <= phi < 4.001 //
/////////////////////////////////////////////

/// Reciprocal scale of the negative binomial unit deviance, case 2.
///
/// Below [`NB2_MID_MU`], a joint `10 x 10` fit in `(mu, phi)` with the Poisson
/// first-panel `log(mu)` factor. Up to [`NB2_LARGE_MU`], a separable fit in
/// `1/mu` and `phi`. Above, the `mu -> inf` edge alone.
///
/// ### Params
///
/// * `mu` - Fitted mean
/// * `phi` - Dispersion, expected in `[0.736, 4.001)`
///
/// ### Returns
///
/// `2 E[d] / Var[d]`, or zero when `mu` is below [`MIN_MU`].
fn nb_alpha_case2(mu: f64, phi: f64) -> f64 {
    if mu < MIN_MU {
        return 0.0;
    }
    let y = phi / PHI_CASE2_HALF - 1.0;

    if mu < NB2_MID_MU {
        let (x, offset) = locate(&NB2_ALPHA_PANELS, mu);
        let block = NB2_PANEL_SIDE * NB2_PANEL_SIDE;
        let out = cheb_eval_2d(
            &NB_A_2_1[offset..offset + block],
            x,
            y,
            NB2_PANEL_SIDE,
            NB2_PANEL_SIDE,
        );
        if mu < NB2_ALPHA_PANELS[0].upper {
            let log_mu = mu.ln();
            out * (log_mu / ((1.0 + log_mu) * (1.0 + log_mu)))
        } else {
            out
        }
    } else if mu < NB2_LARGE_MU {
        let (x, offset) = locate_reciprocal(mu);
        let (lower, upper, blend) =
            edge_series(&NB_A_2_2, offset, NB2_MID_LEN, NB2_MID_EDGES, x, y);
        lower + (upper - lower) * blend
    } else {
        let last = (NB2_MID_EDGES - 1) * NB2_MID_LEN;
        cheb_eval(&NB_A_2_2[last..last + NB2_MID_LEN], y)
    }
}

/// Degrees of freedom of the negative binomial unit deviance, case 2.
///
/// ### Params
///
/// * `mu` - Fitted mean
/// * `phi` - Dispersion, expected in `[0.736, 4.001)`
///
/// ### Returns
///
/// `2 E[d]^2 / Var[d]`, or zero when `mu` is below [`MIN_MU`].
fn nb_kappa_case2(mu: f64, phi: f64) -> f64 {
    if mu < MIN_MU {
        return 0.0;
    }
    let y = phi / PHI_CASE2_HALF - 1.0;

    if mu < NB2_MID_MU {
        let (x, offset) = locate(&NB2_KAPPA_PANELS, mu);
        let block = NB2_PANEL_SIDE * NB2_PANEL_SIDE;
        let out = cheb_eval_2d(
            &NB_K_2_1[offset..offset + block],
            x,
            y,
            NB2_PANEL_SIDE,
            NB2_PANEL_SIDE,
        );
        if mu < NB2_KAPPA_PANELS[0].upper {
            let log_mu = mu.ln() / (1.0 + mu.ln());
            out * mu * log_mu * log_mu
        } else {
            out
        }
    } else if mu < NB2_LARGE_MU {
        let (x, offset) = locate_reciprocal(mu);
        let (lower, upper, blend) =
            edge_series(&NB_K_2_2, offset, NB2_MID_LEN, NB2_MID_EDGES, x, y);
        lower + (upper - lower) * blend
    } else {
        let last = (NB2_MID_EDGES - 1) * NB2_MID_LEN;
        cheb_eval(&NB_K_2_2[last..last + NB2_MID_LEN], y)
    }
}

/// Maps `mu` onto `[-1, 1]` for the case 2 reciprocal-`mu` panels.
#[inline]
fn locate_reciprocal(mu: f64) -> (f64, usize) {
    if mu < NB2_MID_KNOTS[0] {
        (200.0 / mu - 3.0, 0)
    } else if mu < NB2_MID_KNOTS[1] {
        ((2000.0 / mu - 11.0) / 9.0, NB2_MID_LEN)
    } else {
        (2500.0 / mu - 1.5, 2 * NB2_MID_LEN)
    }
}

////////////////////////////////
// Negative binomial, big phi //
////////////////////////////////

/// Both deviance moments by direct summation over the probability mass function.
///
/// Above [`PHI_CASE2_MAX`] the distribution sits on small counts. The term
/// count grows with `mu^2 phi`, capped at [`NB_SUM_MAX`].
///
/// ### Params
///
/// * `mu` - Fitted mean
/// * `phi` - Dispersion, expected to be at least 4.001
///
/// ### Returns
///
/// `(alpha, kappa)`, both zero when `mu` is below [`MIN_MU`].
fn nb_moments_large_phi(mu: f64, phi: f64) -> (f64, f64) {
    if mu < MIN_MU {
        return (0.0, 0.0);
    }

    let n = (mu * mu * phi + NB_SUM_SLACK).min(NB_SUM_MAX as f64) as usize;
    let size = 1.0 / phi;
    let p = size / (mu + size);
    let log_p = p.ln();
    let log_1mp = (1.0 - p).ln();
    let ln_gamma_size = ln_gamma(size);

    // The mean must be known before the variance can be accumulated, so terms
    // are held; the cap keeps them on the stack.
    let mut mass = [0.0_f64; NB_SUM_MAX];
    let mut deviance = [0.0_f64; NB_SUM_MAX];
    for i in 0..n {
        let count = i as f64;
        let log_pmf = ln_gamma(count + size) - ln_gamma(count + 1.0) - ln_gamma_size
            + size * log_p
            + count * log_1mp;
        mass[i] = log_pmf.exp();
        deviance[i] = if i == 0 {
            2.0 * (-size * log_p)
        } else {
            2.0 * (count * (count / mu).ln() - (count + size) * ((count + size) / (mu + size)).ln())
        };
    }
    let (mass, deviance) = (&mass[..n], &deviance[..n]);

    let mean: f64 = mass.iter().zip(deviance).map(|(m, d)| m * d).sum();
    let variance: f64 = mass
        .iter()
        .zip(deviance)
        .map(|(m, d)| m * (d - mean) * (d - mean))
        .sum();
    (2.0 * mean / variance, 2.0 * mean * mean / variance)
}

//////////////
// Dispatch //
//////////////

/// Reciprocal scale of the negative binomial unit deviance.
///
/// Dispatches on `phi`: case 1 fits below `PHI_CASE1_MAX` (including zero),
/// case 2 below `PHI_CASE2_MAX`, direct summation above.
///
/// ### Params
///
/// * `mu` - Fitted mean
/// * `phi` - Dispersion, non-negative
///
/// ### Returns
///
/// `2 E[d] / Var[d]` for `d` the unit deviance at `(mu, phi)`, or zero when
/// `mu` is below `MIN_MU`.
pub fn nb_alpha(mu: f64, phi: f64) -> f64 {
    compute_weight(mu, phi, 1.0).0
}

/// Degrees of freedom of the negative binomial unit deviance.
///
/// Dispatches on `phi` as [`nb_alpha`] does. For both moments above
/// `PHI_CASE2_MAX` use [`compute_weight`], which sums the mass function once.
///
/// ### Params
///
/// * `mu` - Fitted mean
/// * `phi` - Dispersion, non-negative
///
/// ### Returns
///
/// `2 E[d]^2 / Var[d]` for `d` the unit deviance at `(mu, phi)`, or zero when
/// `mu` is below `MIN_MU`.
pub fn nb_kappa(mu: f64, phi: f64) -> f64 {
    compute_weight(mu, phi, 1.0).1
}

/// Both quasi-likelihood weights for one observation.
///
/// The prior divides the fitted value before the weights are taken, as in edgeR.
///
/// ### Params
///
/// * `u` - Fitted value including the prior count
/// * `phi` - Dispersion, non-negative
/// * `prior` - Prior scaling applied to the fitted value
///
/// ### Returns
///
/// `(alpha, kappa)`: the reciprocal scale for the adjusted deviance and the
/// degrees of freedom it carries. Both are zero for a mean below `MIN_MU`.
pub fn compute_weight(u: f64, phi: f64, prior: f64) -> (f64, f64) {
    let mu = u / prior;
    if phi < PHI_CASE1_MAX {
        (nb_alpha_case1(mu, phi), nb_kappa_case1(mu, phi))
    } else if phi < PHI_CASE2_MAX {
        (nb_alpha_case2(mu, phi), nb_kappa_case2(mu, phi))
    } else {
        nb_moments_large_phi(mu, phi)
    }
}

///////////////////
// Unit deviance //
///////////////////

/// The unit deviance, re-exported from [`crate::glm::deviance`] so the weights
/// and the GLM fitter share one implementation.
pub use crate::glm::deviance::unit_nb_deviance;

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::glm::deviance::{
        GAMMA_REGIME as DEVIANCE_GAMMA_MU_PHI, POISSON_REGIME as DEVIANCE_POISSON_PHI,
    };
    use approx::assert_relative_eq;

    /// edgePython `pois_alpha` and `pois_kappa` as `(mu, alpha, kappa)`.
    ///
    /// From `uv run --with numpy --with numba --with scipy --with pandas
    /// --with statsmodels python -c "from edgepython.ql_weights import
    /// pois_alpha, pois_kappa; ..."` over a grid straddling every panel boundary.
    const POIS_REF: [(f64, f64, f64); 28] = [
        (1e-33, 0.0, 0.0),
        (1e-06, 0.08411973860813003, 2.3243108858558407e-06),
        (0.0001, 0.13662860690354295, 0.00025168104987943003),
        (0.0199, 0.4524106923971172, 0.07077727410144066),
        (0.02, 0.4538062434771763, 0.07115202560244337),
        (0.1, 1.1019064888783496, 0.5224780420238604),
        (0.3, 2.5517169558940798, 2.1429520495294474),
        (0.4248, 2.8297281687994826, 2.7085850296729377),
        (0.4249, 2.8295378415518226, 2.7087974103312935),
        (0.45, 2.821057617775071, 2.75130991540634),
        (0.4965, 2.7658968295136646, 2.779722894471397),
        (0.4966, 2.7657344474156047, 2.77965429404501),
        (0.5, 2.7601200565800057, 2.779553338123332),
        (1.0, 1.6808389172227731, 1.9275619202958036),
        (1.4999, 1.2077621328315011, 1.3992213015745867),
        (1.5, 1.2077394505451682, 1.399164198743138),
        (2.0, 1.0205254820288545, 1.1627908734922283),
        (3.5439, 0.9020041592740922, 0.9708540534886082),
        (3.544, 0.9018811631795514, 0.9708514193339274),
        (4.0, 0.9054775442611375, 0.9637450081210828),
        (4.2713, 0.9095574078678059, 0.9628800783650602),
        (4.2714, 0.9095590843738127, 0.9628233088325225),
        (10.0, 0.9759975881849726, 0.9943817607706557),
        (19.999, 0.9903002730296325, 0.9990313096537103),
        (20.0, 0.9904166666666667, 0.999),
        (100.0, 0.9982833333333333, 0.99996),
        (10000.0, 0.9999833283333334, 0.999999996),
        (100000000.0, 0.9999999983333333, 1.0),
    ];

    /// edgePython `anbinomdevc_1` and `knbinomdevc_1` as
    /// `(mu, phi, alpha, kappa)`, over every case 1 panel boundary at three
    /// dispersions spanning the fitted range.
    const NB_CASE1_REF: [(f64, f64, f64, f64); 132] = [
        (1e-33, 1e-08, 0.0, 0.0),
        (1e-06, 1e-08, 0.08411973711533399, 2.3243108453075036e-06),
        (0.005, 1e-08, 0.2859666516622465, 0.01516132124673088),
        (0.0099, 1e-08, 0.3507321518094188, 0.03209793967932352),
        (0.01, 1e-08, 0.35192002927835225, 0.03245910724889308),
        (0.2, 1e-08, 1.9113485787128708, 1.3323356071219385),
        (0.3299, 1e-08, 2.6739023653790137, 2.3339089541106315),
        (0.33, 1e-08, 2.6741795802455655, 2.334495244093451),
        (1.0, 1e-08, 1.6808517704667618, 1.9275525196297503),
        (1.2999, 1e-08, 1.3444532355178287, 1.5598639202386815),
        (1.3, 1e-08, 1.344371131290927, 1.5597857137019517),
        (1.5, 1e-08, 1.2077625473746227, 1.3991690070566458),
        (1.7699, 1e-08, 1.087054196951265, 1.2497135527703154),
        (1.77, 1e-08, 1.0870166207676357, 1.2496689076522995),
        (2.0, 1e-08, 1.0205363552186837, 1.1627951424515064),
        (3.999, 1e-08, 0.9054151078166579, 0.9637713200653486),
        (4.0, 1e-08, 0.9054124942001593, 0.9637996334936467),
        (7.0, 1e-08, 0.9559006319062971, 0.9836882303674708),
        (9.999, 1e-08, 0.9761703674164365, 0.9943384282774658),
        (10.0, 1e-08, 0.9761726672261825, 0.9943339918360141),
        (15.0, 1e-08, 0.9865715613256096, 0.9980344230429046),
        (19.999, 1e-08, 0.9907133072612831, 0.9990089699337787),
        (20.0, 1e-08, 0.9908296103614397, 0.9989775601529741),
        (20.001, 1e-08, 0.9907754518980809, 0.9990424923426241),
        (24.999, 1e-08, 0.9930687211504843, 0.9994583783234792),
        (25.0, 1e-08, 0.9930690783254952, 0.9994584378757246),
        (29.999, 1e-08, 0.9945005378675396, 0.9996543227300508),
        (30.0, 1e-08, 0.9945007720425627, 0.9996543524106946),
        (39.999, 1e-08, 0.9961810329112905, 0.9998133751211902),
        (40.0, 1e-08, 0.996181155967532, 0.9998133851702397),
        (59.999, 1e-08, 0.9977430076694364, 0.9999001525789578),
        (60.0, 1e-08, 0.9977430585850229, 0.9999001548850044),
        (60.001, 1e-08, 0.9979848718649733, 0.9999384658629935),
        (79.999, 1e-08, 0.9987406840091856, 0.9999870741177826),
        (80.0, 1e-08, 0.9987407120296506, 0.9999870756803895),
        (80.001, 1e-08, 0.9986230651222, 0.9999591171216367),
        (119.999, 1e-08, 0.999361452257172, 0.9999938380689627),
        (120.0, 1e-08, 0.9993614644196079, 0.9999938385319416),
        (120.001, 1e-08, 0.9992963018085976, 0.9999738456057758),
        (249.999, 1e-08, 1.0000457712932904, 0.9999952229040907),
        (250.0, 1e-08, 1.0000457740259368, 0.9999952229552911),
        (250.001, 1e-08, 1.0000457767585609, 1.0000027296213105),
        (1000.0, 1e-08, 1.000553639896429, 1.0000087296248883),
        (1000000.0, 1e-08, 1.0007207602907506, 1.00000912962814),
        (1e-33, 0.2, 0.0, 0.0),
        (1e-06, 0.2, 0.0851418595048363, 2.329535578748824e-06),
        (0.005, 0.2, 0.29312712221714254, 0.01526763911431901),
        (0.0099, 0.2, 0.3609523844444957, 0.032371785287676855),
        (0.01, 0.2, 0.36237923582298665, 0.03274381618563231),
        (0.2, 0.2, 1.9914911710789422, 1.3325150926660232),
        (0.3299, 0.2, 2.7769758470099806, 2.318622175309517),
        (0.33, 0.2, 2.7777101053205753, 2.319523905309066),
        (1.0, 0.2, 1.8908926276817466, 2.094290839853621),
        (1.2999, 0.2, 1.5342707920193888, 1.7357015801122924),
        (1.3, 0.2, 1.5341811900814208, 1.7356335134393488),
        (1.5, 0.2, 1.3823579265039123, 1.5712839268508862),
        (1.7699, 0.2, 1.2412724531947241, 1.4118661923585956),
        (1.77, 0.2, 1.2411106985940603, 1.411817233977641),
        (2.0, 0.2, 1.1582269753687877, 1.3142996702315624),
        (3.999, 0.2, 0.9323408018952687, 1.017735443765317),
        (4.0, 0.2, 0.9323116442083943, 1.0177439186533679),
        (7.0, 0.2, 0.9189704899814328, 0.973259198444797),
        (9.999, 0.2, 0.9342430101384062, 0.9782325710618338),
        (10.0, 0.2, 0.9342519371077113, 0.9782362761059787),
        (15.0, 0.2, 0.9506524940267247, 0.988408177980256),
        (19.999, 0.2, 0.9582278728283652, 0.9936675787702245),
        (20.0, 0.2, 0.9583408423486351, 0.9936369632298262),
        (20.001, 0.2, 0.9582877429933285, 0.9937010021007308),
        (24.999, 0.2, 0.962131473750013, 0.9964223823371361),
        (25.0, 0.2, 0.9621320618462792, 0.9964227854236689),
        (29.999, 0.2, 0.9642738030178329, 0.9979403762193102),
        (30.0, 0.2, 0.9642741482976434, 0.9979406127073094),
        (39.999, 0.2, 0.9664107033506327, 0.9994420900850555),
        (40.0, 0.2, 0.9664108567775016, 0.9994421913004482),
        (59.999, 0.2, 0.9679165459585851, 1.0004807268900027),
        (60.0, 0.2, 0.9679165950912278, 1.000480754721559),
        (60.001, 0.2, 0.9679417458322541, 1.000726847951335),
        (79.999, 0.2, 0.9686748051671346, 1.0007754945303597),
        (80.0, 0.2, 0.9686748323440774, 1.0007754960941986),
        (80.001, 0.2, 0.9683175617127306, 1.0010157494509064),
        (119.999, 0.2, 0.969033540799443, 1.001050507087008),
        (120.0, 0.2, 0.9690335525927821, 1.0010505075504759),
        (120.001, 0.2, 0.9688606298055131, 1.0011605757308057),
        (249.999, 0.2, 0.9695872726197068, 1.001181978398868),
        (250.0, 0.2, 0.9695872752691246, 1.0011819784501292),
        (250.001, 0.2, 0.969587277918521, 1.0012701475821775),
        (1000.0, 0.2, 0.9700796730156701, 1.0012761551902467),
        (1000000.0, 0.2, 0.9702417034066588, 1.0012765557004684),
        (1e-33, 0.7359, 0.0, 0.0),
        (1e-06, 0.7359, 0.08748157296864943, 2.3414670178473616e-06),
        (0.005, 0.7359, 0.3100554943335558, 0.015512929049477896),
        (0.0099, 0.7359, 0.38525582834290084, 0.0329991521893798),
        (0.01, 0.7359, 0.38727146431422793, 0.03339684192942361),
        (0.2, 0.7359, 2.1493893155401853, 1.3111287608734894),
        (0.3299, 0.7359, 2.946024651005732, 2.2273932856665786),
        (0.33, 0.7359, 2.94817560793377, 2.228746920316303),
        (1.0, 0.7359, 2.370461450243509, 2.4233677675011593),
        (1.2999, 0.7359, 2.004287898724038, 2.1273850591500874),
        (1.3, 0.7359, 2.004190161044609, 2.1273556673552294),
        (1.5, 0.7359, 1.833210032060164, 1.9761774444821723),
        (1.7699, 0.7359, 1.6620675870570185, 1.818112454127341),
        (1.77, 0.7359, 1.6617352713529625, 1.8180618409192122),
        (2.0, 0.7359, 1.553363162262038, 1.714124307661069),
        (3.999, 0.7359, 1.1624431450302615, 1.3123896981250125),
        (4.0, 0.7359, 1.1623684227335906, 1.3123811345932006),
        (7.0, 0.7359, 1.0208087657986669, 1.1537303517285158),
        (9.999, 0.7359, 0.9730436439196942, 1.0976216431087553),
        (10.0, 0.7359, 0.9730438165673113, 1.097619403521147),
        (15.0, 0.7359, 0.9417382536139418, 1.0595538999933836),
        (19.999, 0.7359, 0.928778603024082, 1.0432792243075353),
        (20.0, 0.7359, 0.9288853306528594, 1.043244055029873),
        (20.001, 0.7359, 0.9288306873550932, 1.0433080961667016),
        (24.999, 0.7359, 0.9222780775610203, 1.034822068034685),
        (25.0, 0.7359, 0.9222771135261173, 1.0348208010832913),
        (29.999, 0.7359, 0.9185228907295467, 1.0298275526537235),
        (30.0, 0.7359, 0.9185223104880965, 1.0298267703313024),
        (39.999, 0.7359, 0.914673147755435, 1.0245141753159444),
        (40.0, 0.7359, 0.914672891175915, 1.024513813404661),
        (59.999, 0.7359, 0.9119896450808093, 1.0205101172069393),
        (60.0, 0.7359, 0.9119895784069596, 1.0205100098101918),
        (60.001, 0.7359, 0.9112901346272888, 1.0196609019259733),
        (79.999, 0.7359, 0.9119802895285, 1.0197104689129541),
        (80.0, 0.7359, 0.911980315114833, 1.0197104705063813),
        (80.001, 0.7359, 0.9105722427687016, 1.0185643941441649),
        (119.999, 0.7359, 0.9112455246635482, 1.0185997611107456),
        (120.0, 0.7359, 0.9112455357535951, 1.0185997615823386),
        (120.001, 0.7359, 0.9108920980950233, 1.0181326818360383),
        (249.999, 0.7359, 0.9115752646694761, 1.0181544473313653),
        (250.0, 0.7359, 0.911575267160375, 1.0181544473834956),
        (250.001, 0.7359, 0.9115752696512536, 1.0183264592333696),
        (1000.0, 0.7359, 0.9120382039364705, 1.0183325691790905),
        (1000000.0, 0.7359, 0.9121905397815472, 1.0183329765118736),
    ];

    /// edgePython `anbinomdevc_2` and `knbinomdevc_2` as
    /// `(mu, phi, alpha, kappa)`, over every case 2 panel boundary at three
    /// dispersions spanning the fitted range.
    const NB_CASE2_REF: [(f64, f64, f64, f64); 93] = [
        (1e-33, 0.736, 0.0, 0.0),
        (1e-06, 0.736, 0.08720075119731019, 2.3390990416287487e-06),
        (0.0099, 0.736, 0.3817069803376364, 0.03294340522463196),
        (0.01, 0.736, 0.389017592899089, 0.03325203060843815),
        (0.2, 0.736, 2.14904208073921, 1.31098109472119),
        (0.4299, 0.736, 3.175969069837466, 2.638532556272056),
        (0.43, 0.736, 3.178669792739246, 2.6388062088141377),
        (0.4999, 0.736, 3.1802350892404547, 2.7734059915105886),
        (0.5, 0.736, 3.180179784254812, 2.7758796996240043),
        (1.0, 0.736, 2.3677898741494894, 2.4232468321125245),
        (3.6199, 0.736, 1.2014458961836507, 1.3542266443657134),
        (3.62, 0.736, 1.2004643973765836, 1.3542141074851228),
        (3.8799, 0.736, 1.1736180949670831, 1.3253102643384904),
        (3.88, 0.736, 1.1736085463814705, 1.3246191273005616),
        (5.0, 0.736, 1.0933654339458845, 1.236284473291776),
        (9.999, 0.736, 0.9722043883358596, 1.0969292621204199),
        (10.0, 0.736, 0.9721937170617059, 1.096916351996782),
        (20.0, 0.736, 0.9282314649297286, 1.0427909534843451),
        (29.999, 0.736, 0.9186195992017568, 1.0299868730400976),
        (30.0, 0.736, 0.918619262718803, 1.0299863466095793),
        (49.999, 0.736, 0.9135352443841256, 1.022616396510499),
        (50.0, 0.736, 0.9135351311126704, 1.0226162202670954),
        (99.999, 0.736, 0.9115054701194938, 1.0191695834911865),
        (100.0, 0.736, 0.9115054457706042, 1.0191695583294713),
        (500.0, 0.736, 0.911444416622152, 1.018503707964735),
        (999.999, 0.736, 0.9114261152765833, 1.0182864369148552),
        (1000.0, 0.736, 0.9114261034510196, 1.0182862844157379),
        (4999.0, 0.736, 0.9117448886318072, 1.0186086245814898),
        (5000.0, 0.736, 0.9117449229066154, 1.018608660875892),
        (100000.0, 0.736, 0.9117449229066154, 1.018608660875892),
        (100000000.0, 0.736, 0.9117449229066154, 1.018608660875892),
        (1e-33, 2.0, 0.0, 0.0),
        (1e-06, 2.0, 0.09103355605960478, 2.3574205243010194e-06),
        (0.0099, 2.0, 0.42148432596724833, 0.03388960898317476),
        (0.01, 2.0, 0.4366639170256536, 0.03436227998127385),
        (0.2, 2.0, 2.3861505187080003, 1.2359443103023804),
        (0.4299, 2.0, 3.4485820653173405, 2.401037218783481),
        (0.43, 2.0, 3.449469209652979, 2.401344440022518),
        (0.4999, 2.0, 3.53213951968122, 2.5802183418305846),
        (0.5, 2.0, 3.532208771105753, 2.5807878924941647),
        (1.0, 2.0, 3.201456260675761, 2.7947918287261486),
        (3.6199, 2.0, 1.9708700199419207, 2.0853512772634417),
        (3.62, 2.0, 1.9700994969340586, 2.085337295123102),
        (3.8799, 2.0, 1.924553447235037, 2.0514101148473523),
        (3.88, 2.0, 1.924536920906725, 2.0505258432608695),
        (5.0, 2.0, 1.7755265604064279, 1.935229523856651),
        (9.999, 2.0, 1.4797260391249814, 1.6914345936457849),
        (10.0, 2.0, 1.4796920259862476, 1.6914053563640283),
        (20.0, 2.0, 1.2929848740467758, 1.526411309579495),
        (29.999, 2.0, 1.2155426207329052, 1.4550560743772925),
        (30.0, 2.0, 1.2155372457997626, 1.455050990945262),
        (49.999, 2.0, 1.1402159117742696, 1.3837226896543302),
        (50.0, 2.0, 1.1402133524391729, 1.3837202305764418),
        (99.999, 2.0, 1.065628480486857, 1.3109159230720784),
        (100.0, 2.0, 1.065629957808923, 1.3109179639467436),
        (500.0, 2.0, 0.9681644946802791, 1.212871583073621),
        (999.999, 2.0, 0.9389366721238712, 1.1808784488239272),
        (1000.0, 2.0, 0.9389188188161797, 1.1808561795859103),
        (4999.0, 2.0, 0.9025780689864683, 1.1414950797452963),
        (5000.0, 2.0, 0.9025741617401436, 1.141490647819847),
        (100000.0, 2.0, 0.9025741617401436, 1.141490647819847),
        (100000000.0, 2.0, 0.9025741617401436, 1.141490647819847),
        (1e-33, 4.0009, 0.0, 0.0),
        (1e-06, 4.0009, 0.09542006862477274, 2.378279757554487e-06),
        (0.0099, 4.0009, 0.47133482553352535, 0.034992500032877474),
        (0.01, 4.0009, 0.49716138965462475, 0.035625828455354565),
        (0.2, 4.0009, 2.6199430930557424, 1.12800413440163),
        (0.4299, 4.0009, 3.690485302289965, 2.10306095833732),
        (0.43, 4.0009, 3.691164701397179, 2.1033565705758237),
        (0.4999, 4.0009, 3.8316856639854247, 2.2882306613199144),
        (0.5, 4.0009, 3.831849305549054, 2.288570964945464),
        (1.0, 4.0009, 3.9954338733309225, 2.864187841788698),
        (3.6199, 4.0009, 3.2042387235694507, 2.894422972006017),
        (3.62, 4.0009, 3.1431505564006867, 2.8944218478000914),
        (3.8799, 4.0009, 3.092222584091328, 2.8931790932683237),
        (3.88, 4.0009, 3.0922038008776007, 2.8234423197574503),
        (5.0, 4.0009, 2.913385823495221, 2.749739080717844),
        (9.999, 4.0009, 2.4941566559480393, 2.5416659663676544),
        (10.0, 4.0009, 2.4941028970046855, 2.541636562223157),
        (20.0, 4.0009, 2.1714312678765553, 2.350963802396969),
        (29.999, 4.0009, 2.0192586499346485, 2.2518283361361284),
        (30.0, 4.0009, 2.0192473297070124, 2.2518206245202106),
        (49.999, 4.0009, 1.8580705737994083, 2.1400239121597213),
        (50.0, 4.0009, 1.8580648440821574, 2.1400198081216364),
        (99.999, 4.0009, 1.6822940935551878, 2.0094933897375857),
        (100.0, 4.0009, 1.6822988241189156, 2.009498686519693),
        (500.0, 4.0009, 1.4024665933908467, 1.7802465683306532),
        (999.999, 4.0009, 1.3185508500829985, 1.7054401988457881),
        (1000.0, 4.0009, 1.3184996639959827, 1.7053882208278182),
        (4999.0, 4.0009, 1.1759208641311245, 1.5704952783240638),
        (5000.0, 4.0009, 1.1759055344944394, 1.5704800898393099),
        (100000.0, 4.0009, 1.1759055344944394, 1.5704800898393099),
        (100000000.0, 4.0009, 1.1759055344944394, 1.5704800898393099),
    ];

    /// edgePython's dispatch chain as `(mu, phi, alpha, kappa)`, sweeping `phi`
    /// across both sides of 0.736 and 4.001 at three means.
    const NB_PHI_SWEEP_REF: [(f64, f64, f64, f64); 48] = [
        (0.7, 1e-08, 2.28411901146112, 2.4944231765986906),
        (0.7, 0.1, 2.391416313976551, 2.555062631174394),
        (0.7, 0.368, 2.631767077156171, 2.668711971766397),
        (0.7, 0.5, 2.733072621414845, 2.707431753140268),
        (0.7, 0.7, 2.871197703994407, 2.751229453757565),
        (0.7, 0.7359, 2.8943110020198635, 2.7575177041976042),
        (0.7, 0.736, 2.8977005666167552, 2.758136768833591),
        (0.7, 0.8, 2.937554379063828, 2.7682901831199116),
        (0.7, 1.5, 3.2995887801198283, 2.8167847673341098),
        (0.7, 2.5, 3.6598528637503884, 2.776586793886604),
        (0.7, 4.0, 4.010649018693627, 2.6330318109184527),
        (0.7, 4.0009, 4.010816339571991, 2.6329364906444415),
        (0.7, 4.001, 6.446738691460531, 4.080219641607852),
        (0.7, 4.5, 6.5174477764329195, 3.963799410958874),
        (0.7, 10.0, 13.157420047726275, 5.4302422597248485),
        (0.7, 50.0, 30.90545629575493, 4.641844539764744),
        (7.0, 1e-08, 0.9559006319062971, 0.9836882303674708),
        (7.0, 0.1, 0.9327301882732201, 0.9714705835097631),
        (7.0, 0.368, 0.9252382766197058, 1.0065438548431205),
        (7.0, 0.5, 0.9499422903769478, 1.051273214161724),
        (7.0, 0.7, 1.0084692603848333, 1.136938348172976),
        (7.0, 0.7359, 1.0208087657986669, 1.1537303517285158),
        (7.0, 0.736, 1.0205421063051512, 1.1536117241611046),
        (7.0, 0.8, 1.0438443738598935, 1.1844245649706002),
        (7.0, 1.5, 1.3577062693332584, 1.54799825005515),
        (7.0, 2.5, 1.8828201252523853, 2.0462432817068827),
        (7.0, 4.0, 2.6963588209388734, 2.6477289079674384),
        (7.0, 4.0009, 2.6968399423757305, 2.6480336736525287),
        (7.0, 4.001, 3.454198231950373, 3.103251056083398),
        (7.0, 4.5, 3.929783117675869, 3.3882953681542123),
        (7.0, 10.0, 10.720050022277201, 6.454968707239349),
        (7.0, 50.0, 133.93187714162994, 28.273049115094636),
        (300.0, 1e-08, 1.0001594114467671, 1.0000046851435194),
        (300.0, 0.1, 0.9845108687528138, 1.0003085451374627),
        (300.0, 0.368, 0.9470181979475886, 1.0045551787912181),
        (300.0, 0.5, 0.9314493604842934, 1.0084585105455548),
        (300.0, 0.7, 0.91369911244961, 1.0165195650451504),
        (300.0, 0.7359, 0.9116788514811565, 1.0183284505879409),
        (300.0, 0.736, 0.9114605272724586, 1.0186877690788854),
        (300.0, 0.8, 0.9079083657643666, 1.0232085340062957),
        (300.0, 1.5, 0.9267221440528636, 1.1226812938734534),
        (300.0, 2.5, 1.0894520662967082, 1.3785220073555513),
        (300.0, 4.0, 1.4760773680644994, 1.8433365060865063),
        (300.0, 4.0009, 1.4763375112736943, 1.8436187650432214),
        (300.0, 4.001, 1.697105454460408, 1.7426800014223915),
        (300.0, 4.5, 1.9564576703514682, 2.007518229158112),
        (300.0, 10.0, 6.138627653953748, 5.286146362905386),
        (300.0, 50.0, 105.0136391075532, 34.30128865732315),
    ];

    /// edgePython `compute_weight` as `(u, phi, prior, alpha, kappa)`.
    const COMPUTE_WEIGHT_REF: [(f64, f64, f64, f64, f64); 72] = [
        (1e-33, 1e-06, 1.0, 0.0, 0.0),
        (1e-33, 1e-06, 0.125, 0.0, 0.0),
        (1e-33, 0.5, 1.0, 0.0, 0.0),
        (1e-33, 0.5, 0.125, 0.0, 0.0),
        (1e-33, 0.736, 1.0, 0.0, 0.0),
        (1e-33, 0.736, 0.125, 0.0, 0.0),
        (1e-33, 2.0, 1.0, 0.0, 0.0),
        (1e-33, 2.0, 0.125, 0.0, 0.0),
        (1e-33, 4.001, 1.0, 0.0, 0.0),
        (1e-33, 4.001, 0.125, 0.0, 0.0),
        (1e-33, 20.0, 1.0, 0.0, 0.0),
        (1e-33, 20.0, 0.125, 0.0, 0.0),
        (
            1e-06,
            1e-06,
            1.0,
            0.08411974243947394,
            2.324310872542318e-06,
        ),
        (
            1e-06,
            1e-06,
            0.125,
            0.10182018610613065,
            1.9119482550355137e-05,
        ),
        (1e-06, 0.5, 1.0, 0.0865109560866671, 2.3365221063071616e-06),
        (
            1e-06,
            0.5,
            0.125,
            0.10472981791230808,
            1.922092500523333e-05,
        ),
        (
            1e-06,
            0.736,
            1.0,
            0.08720075119731019,
            2.3390990416287487e-06,
        ),
        (
            1e-06,
            0.736,
            0.125,
            0.10556807308107764,
            1.9242287643955312e-05,
        ),
        (1e-06, 2.0, 1.0, 0.09103355605960478, 2.3574205243010194e-06),
        (
            1e-06,
            2.0,
            0.125,
            0.11023268506046996,
            1.9394482776608432e-05,
        ),
        (
            1e-06,
            4.001,
            1.0,
            0.09189724031299797,
            2.353224935108577e-06,
        ),
        (
            1e-06,
            4.001,
            0.125,
            0.11340795456086798,
            1.9459258217706814e-05,
        ),
        (1e-06, 20.0, 1.0, 0.10303906553069346, 2.394375179919102e-06),
        (
            1e-06,
            20.0,
            0.125,
            0.13079359408748534,
            1.9963108877687755e-05,
        ),
        (0.5, 1e-06, 1.0, 2.760152213157573, 2.7795703663924196),
        (0.5, 1e-06, 0.125, 0.9054124748538324, 0.9637997075592926),
        (0.5, 0.5, 1.0, 3.0822925311205593, 2.79865294131596),
        (0.5, 0.5, 0.125, 1.0453915032993721, 1.1718624092406238),
        (0.5, 0.736, 1.0, 3.180179784254812, 2.7758796996240043),
        (0.5, 0.736, 0.125, 1.1625396573740578, 1.3125857051704735),
        (0.5, 2.0, 1.0, 3.532208771105753, 2.5807878924941647),
        (0.5, 2.0, 0.125, 1.9052120745536059, 2.035850146025959),
        (0.5, 4.001, 1.0, 4.841570527546001, 2.8473293546917393),
        (0.5, 4.001, 0.125, 3.848915441813864, 3.4168511819001726),
        (0.5, 20.0, 1.0, 16.98849328783584, 4.329855548037571),
        (0.5, 20.0, 0.125, 36.7092218542188, 13.744364640141697),
        (3.0, 1e-06, 1.0, 0.9100269785894167, 0.9960746558151207),
        (3.0, 1e-06, 0.125, 0.9926951945027321, 0.9993951613144001),
        (3.0, 0.5, 1.0, 1.1439060061378115, 1.2872240752458572),
        (3.0, 0.5, 0.125, 0.9205833678681944, 1.0002468294550202),
        (3.0, 0.736, 1.0, 1.2849942611603908, 1.4442849966225744),
        (3.0, 0.736, 0.125, 0.9230142104291075, 1.0359455968477709),
        (3.0, 2.0, 1.0, 2.1045515358231235, 2.1835524647801248),
        (3.0, 2.0, 0.125, 1.2558661044064239, 1.4924446660503583),
        (3.0, 4.001, 1.0, 3.9327145913158064, 3.3961728498221255),
        (3.0, 4.001, 0.125, 2.227752532008505, 2.0585798240298296),
        (3.0, 20.0, 1.0, 35.29597250931814, 12.803374362839046),
        (3.0, 20.0, 0.125, 21.87640443764169, 10.272550336036362),
        (25.0, 1e-06, 1.0, 0.9930689674774528, 0.9994584153445328),
        (25.0, 1e-06, 0.125, 0.9998743257183312, 0.9999916216321536),
        (25.0, 0.5, 1.0, 0.9208048861657141, 1.000199390574298),
        (25.0, 0.5, 0.125, 0.9311840072598463, 1.0072868537170627),
        (25.0, 0.736, 1.0, 0.922066700564147, 1.0346792836670013),
        (25.0, 0.736, 0.125, 0.9114752683522817, 1.0188507518645833),
        (25.0, 2.0, 1.0, 1.2480922517988704, 1.485277110744023),
        (25.0, 2.0, 0.125, 1.017435674791169, 1.263973752849699),
        (25.0, 4.001, 1.0, 2.2123682722703464, 2.0496138715348526),
        (25.0, 4.001, 0.125, 1.8323262408314094, 1.8848212574247385),
        (25.0, 20.0, 1.0, 21.794433340143826, 10.28684238288027),
        (25.0, 20.0, 0.125, 20.262013567632117, 11.955964206658317),
        (3000.0, 1e-06, 1.0, 1.0006651175079986, 1.0000090856891826),
        (3000.0, 1e-06, 0.125, 1.0007138183833943, 1.000009129439582),
        (3000.0, 0.5, 1.0, 0.9319204724821948, 1.0084629477827223),
        (3000.0, 0.5, 0.125, 0.9319658276584819, 1.0084629919029782),
        (3000.0, 0.736, 1.0, 0.911658089791928, 1.018518338303882),
        (3000.0, 0.736, 0.125, 0.9117449229066154, 1.018608660875892),
        (3000.0, 2.0, 1.0, 0.912472929587479, 1.1525199769338557),
        (3000.0, 2.0, 0.125, 0.9025741617401436, 1.141490647819847),
        (3000.0, 4.001, 1.0, 0.9546464226100999, 0.8562810267167016),
        (
            3000.0,
            4.001,
            0.125,
            0.6105365976191981,
            0.43126657276547314,
        ),
        (3000.0, 20.0, 1.0, 14.90425894459297, 10.576823101039968),
        (3000.0, 20.0, 0.125, 10.304665585085806, 7.981198378549229),
    ];

    /// `nbinomUnitDeviance(y, mean, dispersion)` from edgeR 4.8.2, as
    /// `(y, mu, phi, deviance)`. Produced by
    /// `Rscript -e 'suppressMessages(library(edgeR)); cat(sprintf("%.17g",
    /// nbinomUnitDeviance(y, mu, phi)))'` over the grid below.
    const DEVIANCE_REF: [(f64, f64, f64, f64); 16] = [
        (0.0, 5.0, 0.1, 8.10930176300355),
        (0.0, 1e-6, 0.1, 1.907697488390024e-06),
        (1e-8, 5.0, 0.1, 8.109301378236367),
        (0.5, 5.0, 1e-6, 6.6973946289643775),
        (3.0, 5.0, 1e-5, 0.935006256387544),
        (5.0, 5.0, 0.5, 0.0),
        (100.0, 100.0, 2.0, 0.0),
        (1e5, 1e5 + 10.0, 1e-5, 0.0009999333330125001),
        (1e6, 1e6 + 1.0, 1e-6, 9.998995586213226e-07),
        (1e6, 1e6, 5.0, 0.0),
        (0.0, 0.001, 0.2, 0.001999569767954525),
        (7.0, 7.0, 1e-6, 0.0),
        (1e4, 1.0, 1e-6, 164109.8270387379),
        (1e4, 1e4 + 3.0, 1e-5, 0.0008188200407417032),
        (2.0, 1e8, 0.001, 22997.00853998624),
        (1e6, 1.001e6, 1e-5, 90.99933383298236),
    ];

    /// Closed-form unit deviance with no numerical care. The authority away from
    /// the branch crossovers.
    fn exact_deviance(y: f64, mu: f64, phi: f64) -> f64 {
        let inv_phi = 1.0 / phi;
        let first = if y == 0.0 { 0.0 } else { y * (y / mu).ln() };
        2.0 * (first - (y + inv_phi) * ((y + inv_phi) / (mu + inv_phi)).ln())
    }

    /// Worst relative error over a table; an expected zero is compared absolutely.
    fn worst_relative(errors: &[(f64, f64)]) -> f64 {
        errors.iter().fold(0.0_f64, |worst, (got, expected)| {
            let e = if *expected == 0.0 {
                got.abs()
            } else {
                (got - expected).abs() / expected.abs()
            };
            worst.max(e)
        })
    }

    // -------------------- //
    // Chebyshev evaluation //
    // -------------------- //

    #[test]
    fn test_cheb_eval_reproduces_the_basis() {
        // T_0 = 1, T_1 = x, T_2 = 2x^2 - 1, T_3 = 4x^3 - 3x.
        let x = 0.37;
        assert_relative_eq!(cheb_eval(&[1.0], x), 1.0, max_relative = 1e-15);
        assert_relative_eq!(cheb_eval(&[0.0, 1.0], x), x, max_relative = 1e-15);
        assert_relative_eq!(
            cheb_eval(&[0.0, 0.0, 1.0], x),
            2.0 * x * x - 1.0,
            max_relative = 1e-15
        );
        assert_relative_eq!(
            cheb_eval(&[0.0, 0.0, 0.0, 1.0], x),
            4.0 * x * x * x - 3.0 * x,
            max_relative = 1e-14
        );
    }

    #[test]
    fn test_cheb_eval_handles_degenerate_slices() {
        assert_eq!(cheb_eval(&[], 0.5), 0.0);
        assert_eq!(cheb_eval(&[2.5], 0.5), 2.5);
    }

    #[test]
    fn test_cheb_eval_at_the_endpoints() {
        // T_i(1) = 1 and T_i(-1) = (-1)^i, so the series collapses to sums.
        let c = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_relative_eq!(cheb_eval(&c, 1.0), 15.0, max_relative = 1e-15);
        assert_relative_eq!(cheb_eval(&c, -1.0), 3.0, max_relative = 1e-15);
    }

    #[test]
    fn test_cheb_eval_2d_is_the_tensor_product() {
        // Coefficients are y-major, so this picks T_1(x) * T_2(y).
        let mut c = [0.0; 9];
        c[2 * 3 + 1] = 1.0;
        let (x, y) = (0.3, -0.7);
        assert_relative_eq!(
            cheb_eval_2d(&c, x, y, 3, 3),
            x * (2.0 * y * y - 1.0),
            max_relative = 1e-14
        );
    }

    #[test]
    fn test_cheb_eval_2d_factorises() {
        // A rank one coefficient block must factorise into the two 1d fits.
        let cx = [0.5, -1.5, 2.0, 0.25];
        let cy = [1.0, 0.75, -0.5];
        let mut c = [0.0; 12];
        for (i, wy) in cy.iter().enumerate() {
            for (j, wx) in cx.iter().enumerate() {
                c[i * 4 + j] = wx * wy;
            }
        }
        let (x, y) = (-0.4, 0.9);
        assert_relative_eq!(
            cheb_eval_2d(&c, x, y, 4, 3),
            cheb_eval(&cx, x) * cheb_eval(&cy, y),
            max_relative = 1e-13
        );
    }

    // --------------------------- //
    // coefficient table integrity //
    // --------------------------- //

    #[test]
    fn test_coefficient_tables_have_the_expected_shapes() {
        assert_eq!(POIS_ALPHA_COEF.len(), 5 * POIS_PANEL_LEN);
        assert_eq!(POIS_KAPPA_COEF.len(), 5 * POIS_PANEL_LEN);
        assert_eq!(NB_A_1_1.len(), 6 * NB1_PANEL_SIDE * NB1_PANEL_SIDE);
        assert_eq!(NB_K_1_1.len(), 6 * NB1_PANEL_SIDE * NB1_PANEL_SIDE);
        assert_eq!(NB_A_1_2.len(), 9 * NB1_MID_LEN);
        assert_eq!(NB_K_1_2.len(), 9 * NB1_MID_LEN);
        assert_eq!(NB_A_1_3.len(), 3 * NB1_LARGE_LEN);
        assert_eq!(NB_K_1_3.len(), 4 * NB1_LARGE_LEN);
        assert_eq!(NB_A_2_1.len(), 6 * NB2_PANEL_SIDE * NB2_PANEL_SIDE);
        assert_eq!(NB_K_2_1.len(), 6 * NB2_PANEL_SIDE * NB2_PANEL_SIDE);
        assert_eq!(NB_A_2_2.len(), 7 * NB2_MID_LEN);
        assert_eq!(NB_K_2_2.len(), 7 * NB2_MID_LEN);
    }

    #[test]
    fn test_leading_coefficients_match_the_c_source() {
        // First entry of every table against ql_weights.c, which catches a
        // mistranscribed sign or exponent.
        assert_eq!(POIS_ALPHA_COEF[0], 0.992269079723461);
        assert_eq!(POIS_KAPPA_COEF[0], 1.98775180998087);
        assert_eq!(NB_A_1_1[0], 1.04049914108557);
        assert_eq!(NB_A_1_2[0], 0.955633454636176);
        assert_eq!(NB_A_1_3[0], 0.951987668582991);
        assert_eq!(NB_K_1_1[0], 1.01093193289832);
        assert_eq!(NB_K_1_2[0], 1.00834766392583);
        assert_eq!(NB_K_1_3[0], 1.00583242222469);
        assert_eq!(NB_A_2_1[0], -1.16722391014247);
        assert_eq!(NB_A_2_2[0], 1.27901923265303);
        assert_eq!(NB_K_2_1[0], 2.08206884879209);
        assert_eq!(NB_K_2_2[0], 1.47107357007529);
    }

    // ------- //
    // Poisson //
    // ------- //

    /// edgePython: `pois_alpha(mu), pois_kappa(mu)` for each `mu` in the table.
    #[test]
    fn test_poisson_moments_match_edgepython() {
        let mut errors = Vec::new();
        for (mu, alpha, kappa) in POIS_REF {
            errors.push((pois_alpha(mu), alpha));
            errors.push((pois_kappa(mu), kappa));
        }
        let worst = worst_relative(&errors);
        // Bit exact against edgePython over the whole grid.
        assert!(worst < 1e-14, "worst relative error {worst:e}");
    }

    #[test]
    fn test_poisson_moments_are_zero_below_the_floor() {
        assert_eq!(pois_alpha(0.0), 0.0);
        assert_eq!(pois_kappa(0.0), 0.0);
        assert_eq!(pois_alpha(1e-33), 0.0);
        assert_eq!(pois_kappa(1e-33), 0.0);
    }

    #[test]
    fn test_poisson_moments_approach_one() {
        // Both moments tend to one as the deviance becomes chi-square with one
        // degree of freedom.
        assert_relative_eq!(pois_alpha(1e8), 1.0, max_relative = 1e-8);
        assert_relative_eq!(pois_kappa(1e8), 1.0, max_relative = 1e-8);
    }

    #[test]
    fn test_poisson_panels_join_up() {
        // Panels are fitted independently. The bound is the worst jump in
        // edgePython, 1.0e-3 for alpha at mu = 0.02; a misplaced boundary or
        // wrong offset moves it by order one.
        for knot in [0.02, 0.4249, 0.4966, 1.5, 3.544, 4.2714, 20.0] {
            let below = knot * (1.0 - 1e-9);
            assert_relative_eq!(pois_alpha(below), pois_alpha(knot), max_relative = 2e-3);
            assert_relative_eq!(pois_kappa(below), pois_kappa(knot), max_relative = 2e-3);
        }
    }

    // ------------------------- //
    // negative binomial, case 1 //
    // ------------------------- //

    /// edgePython: `anbinomdevc_1(mu, phi), knbinomdevc_1(mu, phi)`.
    #[test]
    fn test_case1_matches_edgepython() {
        let mut errors = Vec::new();
        for (mu, phi, alpha, kappa) in NB_CASE1_REF {
            errors.push((nb_alpha(mu, phi), alpha));
            errors.push((nb_kappa(mu, phi), kappa));
        }
        let worst = worst_relative(&errors);
        // Bit exact against edgePython; the bound leaves room for a differing
        // `log` on other platforms.
        assert!(worst < 1e-14, "worst relative error {worst:e}");
    }

    #[test]
    fn test_case1_panels_join_up() {
        // Worst jump in edgePython is 1.8e-3, for alpha at mu = 0.01 and the
        // top of the case 1 dispersion range.
        for knot in [
            0.01, 0.33, 1.3, 1.77, 4.0, 10.0, 20.0, 25.0, 30.0, 40.0, 60.0,
        ] {
            for phi in [1e-6, 0.3, 0.7359] {
                let below = knot * (1.0 - 1e-9);
                assert_relative_eq!(
                    nb_alpha(below, phi),
                    nb_alpha(knot, phi),
                    max_relative = 2e-3
                );
                assert_relative_eq!(
                    nb_kappa(below, phi),
                    nb_kappa(knot, phi),
                    max_relative = 2e-3
                );
            }
        }
    }

    // ------------------------- //
    // negative binomial, case 2 //
    // ------------------------- //

    /// edgePython: `anbinomdevc_2(mu, phi), knbinomdevc_2(mu, phi)`.
    #[test]
    fn test_case2_matches_edgepython() {
        let mut errors = Vec::new();
        for (mu, phi, alpha, kappa) in NB_CASE2_REF {
            errors.push((nb_alpha(mu, phi), alpha));
            errors.push((nb_kappa(mu, phi), kappa));
        }
        let worst = worst_relative(&errors);
        // Worst observed 2.9e-16.
        assert!(worst < 1e-14, "worst relative error {worst:e}");
    }

    #[test]
    fn test_case2_panels_join_up() {
        // Case 2 is rougher than case 1: edgePython jumps by 4.8e-2 for alpha at
        // mu = 0.01 (log factor off) and 2.5e-2 at the mu = 3.88 knot with phi at
        // the top of the range. The bound is that behaviour.
        for knot in [
            0.01, 0.43, 0.5, 3.62, 3.88, 10.0, 30.0, 50.0, 100.0, 1000.0, 5000.0,
        ] {
            for phi in [0.736, 2.0, 4.0009] {
                let below = knot * (1.0 - 1e-9);
                assert_relative_eq!(
                    nb_alpha(below, phi),
                    nb_alpha(knot, phi),
                    max_relative = 6e-2
                );
                assert_relative_eq!(
                    nb_kappa(below, phi),
                    nb_kappa(knot, phi),
                    max_relative = 6e-2
                );
            }
        }
    }

    // -------- //
    // dispatch //
    // -------- //

    /// edgePython: the same `phi` dispatch chain, evaluated at three means.
    #[test]
    fn test_phi_dispatch_matches_edgepython() {
        let mut errors = Vec::new();
        for (mu, phi, alpha, kappa) in NB_PHI_SWEEP_REF {
            errors.push((nb_alpha(mu, phi), alpha));
            errors.push((nb_kappa(mu, phi), kappa));
        }
        let worst = worst_relative(&errors);
        // Worst observed 1.1e-14, in the summation branch above phi = 4.001,
        // where `ln_gamma` comes from `statrs` rather than the platform libm.
        assert!(worst < 1e-13, "worst relative error {worst:e}");
    }

    /// edgePython: `compute_weight(u, phi, prior)`.
    #[test]
    fn test_compute_weight_matches_edgepython() {
        let mut errors = Vec::new();
        for (u, phi, prior, alpha, kappa) in COMPUTE_WEIGHT_REF {
            let (got_alpha, got_kappa) = compute_weight(u, phi, prior);
            errors.push((got_alpha, alpha));
            errors.push((got_kappa, kappa));
        }
        let worst = worst_relative(&errors);
        // Worst observed 1.6e-14.
        assert!(worst < 1e-13, "worst relative error {worst:e}");
    }

    #[test]
    fn test_compute_weight_agrees_with_the_scalar_entry_points() {
        for phi in [0.0, 1e-6, 0.5, 0.736, 2.0, 4.001, 10.0] {
            for u in [0.5, 4.0, 250.0] {
                let (alpha, kappa) = compute_weight(u, phi, 0.5);
                assert_eq!(alpha, nb_alpha(u / 0.5, phi));
                assert_eq!(kappa, nb_kappa(u / 0.5, phi));
            }
        }
    }

    /// `phi = 0` must take the case 1 fit, not the Poisson one.
    ///
    /// edgeR's `compute_weight` has no Poisson branch: `phi = 0` falls into
    /// `anbinomdevc_1`, which carries [`pois_alpha`] as a factor. Short
    /// circuiting to Poisson costs 1.8e-4 relative on the adjusted deviance
    /// against `glmQLFit(dispersion = 0)`.
    #[test]
    fn test_zero_dispersion_takes_the_case_one_fit() {
        for mu in [0.005, 0.1, 1.0, 7.0, 15.0, 30.0, 100.0, 500.0] {
            assert_eq!(nb_alpha(mu, 0.0), nb_alpha_case1(mu, 0.0));
            assert_eq!(nb_kappa(mu, 0.0), nb_kappa_case1(mu, 0.0));

            // Continuous into the interior of case 1.
            assert_relative_eq!(nb_alpha(mu, 0.0), nb_alpha(mu, 1e-12), max_relative = 1e-9);
            assert_relative_eq!(nb_kappa(mu, 0.0), nb_kappa(mu, 1e-12), max_relative = 1e-9);

            // Close to the Poisson fit (a factor of it), but not equal.
            assert_relative_eq!(nb_alpha(mu, 0.0), pois_alpha(mu), max_relative = 2e-3);
        }
    }

    #[test]
    fn test_weights_are_zero_below_the_mean_floor() {
        for phi in [0.0, 1e-6, 0.5, 2.0, 10.0] {
            assert_eq!(nb_alpha(1e-33, phi), 0.0);
            assert_eq!(nb_kappa(1e-33, phi), 0.0);
        }
    }

    #[test]
    fn test_large_phi_moments_are_a_sane_chi_square() {
        // Both are positive and kappa = alpha * E[d] by construction.
        for phi in [4.001, 10.0, 100.0] {
            for mu in [0.1, 2.0, 50.0] {
                let (alpha, kappa) = compute_weight(mu, phi, 1.0);
                assert!(alpha > 0.0, "alpha {alpha} at mu {mu} phi {phi}");
                assert!(kappa > 0.0, "kappa {kappa} at mu {mu} phi {phi}");
            }
        }
    }

    // ------------- //
    // unit deviance //
    // ------------- //

    /// edgeR 4.8.2 via `nbinomUnitDeviance`.
    #[test]
    fn test_unit_deviance_matches_edger() {
        let errors: Vec<(f64, f64)> = DEVIANCE_REF
            .iter()
            .map(|(y, mu, phi, expected)| (unit_nb_deviance(*y, *mu, *phi), *expected))
            .collect();
        let worst = worst_relative(&errors);
        // Worst observed 8.6e-11, on `y = 1e6, mu = 1e6 + 1, phi = 1e-6`, where
        // the series differences two quantities of order 1e6 to give one of
        // order 1e-6. Every other case agrees to 5e-13 or better.
        assert!(worst < 1e-9, "worst relative error {worst:e}");
    }

    #[test]
    fn test_unit_deviance_matches_the_closed_form() {
        // Away from both crossovers the closed form is well conditioned.
        for (y, mu, phi) in [
            (0.0, 5.0, 0.1),
            (1.0, 5.0, 0.1),
            (3.0, 5.0, 0.1),
            (20.0, 5.0, 0.5),
            (100.0, 3.0, 0.05),
            (2.0, 1000.0, 0.2),
            (1e4, 100.0, 1.5),
        ] {
            assert_relative_eq!(
                unit_nb_deviance(y, mu, phi),
                exact_deviance(y, mu, phi),
                max_relative = 1e-6
            );
        }
    }

    #[test]
    fn test_unit_deviance_is_zero_at_the_fitted_value() {
        for (mu, phi) in [
            (1.0, 0.1),
            (5.0, 1e-6),
            (1e6, 2.0),
            (1e6, 1e-6),
            (0.001, 0.5),
        ] {
            assert_eq!(unit_nb_deviance(mu, mu, phi), 0.0);
        }
    }

    #[test]
    fn test_unit_deviance_is_symmetric_and_quadratic_near_the_fit() {
        // Locally d ~ (y - mu)^2 / (mu (1 + mu phi)), so halving the residual
        // must quarter the deviance and the two sides must almost agree.
        let (mu, phi) = (50.0, 0.1);
        let curvature = 1.0 / (mu * (1.0 + mu * phi));
        for step in [1.0, 0.5, 0.25] {
            let up = unit_nb_deviance(mu + step, mu, phi);
            let down = unit_nb_deviance(mu - step, mu, phi);
            assert_relative_eq!(up, step * step * curvature, max_relative = 5e-2);
            assert_relative_eq!(down, step * step * curvature, max_relative = 5e-2);
        }
    }

    #[test]
    fn test_unit_deviance_crosses_the_poisson_branch_smoothly() {
        // phi = 1e-4 switches from the series to the exact form. The series
        // stops after the cubic term, so the step is the first dropped term,
        // order phi^2 (y - mu)^3 (9e-7 relative here). Both sides agree with the
        // closed form.
        let (y, mu) = (30.0, 25.0);
        let below = unit_nb_deviance(y, mu, DEVIANCE_POISSON_PHI * (1.0 - 1e-12));
        let above = unit_nb_deviance(y, mu, DEVIANCE_POISSON_PHI);
        assert_relative_eq!(below, above, max_relative = 1e-5);
        assert_relative_eq!(
            above,
            exact_deviance(y, mu, DEVIANCE_POISSON_PHI),
            max_relative = 1e-8
        );
        assert_relative_eq!(
            below,
            exact_deviance(y, mu, DEVIANCE_POISSON_PHI),
            max_relative = 1e-5
        );
    }

    #[test]
    fn test_unit_deviance_crosses_the_gamma_branch_smoothly() {
        // mu * phi = 1e6 switches to the gamma limit. The exact form has lost
        // most digits by there, so the branches are compared to each other and
        // the gamma side to its own limit.
        let phi = 1.0;
        let mu = DEVIANCE_GAMMA_MU_PHI;
        for ratio in [0.5, 0.9, 1.1, 2.0] {
            let y = mu * ratio;
            let below = unit_nb_deviance(y, mu * (1.0 - 1e-9), phi);
            let above = unit_nb_deviance(y, mu * (1.0 + 1e-9), phi);
            assert_relative_eq!(below, above, max_relative = 1e-6);
            // The gamma limit: 2 * ((y - mu)/mu - log(y/mu)) / phi.
            assert_relative_eq!(
                above,
                2.0 * ((y - mu) / mu - (y / mu).ln()) / phi,
                max_relative = 1e-5
            );
        }
    }

    #[test]
    fn test_unit_deviance_series_beats_the_exact_form_near_the_poisson_limit() {
        // At phi = 1e-12 the exact form takes a log of two numbers agreeing to
        // twelve digits and loses about twelve; the series never forms either.
        // The phi -> 0 limit is the Poisson deviance, the authority here.
        let (y, mu, phi) = (12.0, 10.0, 1e-12);
        let series = unit_nb_deviance(y, mu, phi);
        let exact = exact_deviance(y, mu, phi);
        let poisson = 2.0 * (y * (y / mu).ln() - (y - mu));
        assert_relative_eq!(series, poisson, max_relative = 1e-8);
        assert!(
            (exact - poisson).abs() / poisson > 1e-6,
            "the exact form was expected to have lost digits here, got {exact}"
        );
    }

    #[test]
    fn test_unit_deviance_is_never_negative() {
        for y in [0.0, 1e-12, 0.5, 1.0, 100.0, 1e7] {
            for mu in [1e-6, 0.5, 1.0, 100.0, 1e7] {
                for phi in [0.0, 1e-9, 1e-4, 0.1, 2.0, 1e3] {
                    let d = unit_nb_deviance(y, mu, phi);
                    assert!(d >= 0.0, "negative deviance {d} at ({y}, {mu}, {phi})");
                    assert!(d.is_finite(), "non finite deviance at ({y}, {mu}, {phi})");
                }
            }
        }
    }

    #[test]
    fn test_unit_deviance_handles_a_zero_count() {
        // At y = 0 the deviance is 2/phi * log(1 + mu phi), the y -> 0 limit of
        // the exact form.
        for (mu, phi) in [(5.0_f64, 0.1_f64), (1.0, 2.0), (100.0, 0.5)] {
            let expected = 2.0 / phi * (1.0 + mu * phi).ln();
            assert_relative_eq!(
                unit_nb_deviance(0.0, mu, phi),
                expected,
                max_relative = 1e-6
            );
        }
    }
}
