//! Chebyshev approximations to the moments of the negative binomial unit
//! deviance, and the unit deviance itself.
//!
//! A port of edgeR's `ql_weights.c` and the deviance kernel of
//! `compute_nbdev.c`. Under the fitted model the unit deviance of one
//! observation behaves like a scaled chi-square, and the quasi-likelihood
//! machinery needs its first two moments. Matching moments gives
//!
//! ```text
//! alpha = 2 * E[d] / Var[d]        kappa = 2 * E[d]^2 / Var[d]
//! ```
//!
//! so `alpha` is the reciprocal scale that maps the deviance onto a chi-square
//! and `kappa` is that chi-square's degrees of freedom. Neither has a closed
//! form, so edgeR ships piecewise Chebyshev fits: in `mu` alone for the Poisson
//! limit, and in `(mu, phi)` jointly for the negative binomial. Only above
//! `phi = 4.001` does the C fall back on summing the probability mass function
//! directly, which is cheap there because the distribution has collapsed onto a
//! handful of small counts.
//!
//! The coefficient tables are copied verbatim from `ql_weights.c` by way of
//! edgePython. Panel boundaries and the affine maps that take each panel onto
//! `[-1, 1]` live in [`Panel`] tables next to the coefficients they index, so
//! the fit and its domain cannot drift apart.
//!
//! ### Deviations from edgePython
//!
//! [`unit_nb_deviance`] follows edgeR's C, not edgePython. The small-`phi`
//! series in the C writes `2/3*resid` with integer operands, so that term is
//! identically zero; edgePython "corrected" it to `2.0/3.0*resid` and drifts
//! from edgeR by up to 7e-4 relative on large counts. See the tests.
//!
//! ### References
//!
//! Lund, Nettleton, McCarthy and Smyth, SAGMB 11(5), 2012

use crate::numeric::gamma::ln_gamma;

////////////
// Consts //
////////////

/// Below this mean every weight is reported as an exact zero.
///
/// edgeR's `low_value`. The fits are all multiplied by `log(mu)`-shaped
/// factors that lose meaning once `mu` underflows the fitted range, and a gene
/// with a mean this small contributes nothing to the quasi-likelihood anyway.
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
/// The fitted range stops at 4.001 rather than 4, so the largest `phi` reaching
/// case 2 maps to a hair over 1. Chebyshev polynomials are well behaved there.
const PHI_CASE2_HALF: f64 = 2.0;

/// Mean above which the Poisson fits give way to their asymptotic expansions.
const POIS_ASYMPTOTIC_MU: f64 = 20.0;

/// Mean below which the Poisson fits carry an explicit `log(mu)` factor.
///
/// Both moments vanish as `mu -> 0` like `mu * log(mu)^2`, which no polynomial
/// in `mu` can follow, so the first panel fits the smooth remainder instead.
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
/// Both moments have flattened out in `mu` by here, so the `mu -> inf` edge of
/// the separable fit is used unchanged.
const NB2_LARGE_MU: f64 = 5000.0;

/// Breakpoints in `mu` for the case 2 reciprocal-`mu` panels.
///
/// The separable fit above [`NB2_MID_MU`] is a polynomial in `1/mu`, not in
/// `mu`, so these panels carry their own maps rather than a [`Panel`] table.
const NB2_MID_KNOTS: [f64; 2] = [100.0, 1000.0];

/// Side length of a case 2 two-dimensional panel, which holds `10 * 10` terms.
const NB2_PANEL_SIDE: usize = 10;

/// Coefficients per block in the case 2 separable and `phi`-only fits.
const NB2_MID_LEN: usize = 10;

/// Edge series in the case 2 separable table: three panels sharing four edges.
///
/// The last of the four is the `mu -> inf` edge, which is also the whole answer
/// above [`NB2_LARGE_MU`].
const NB2_MID_EDGES: usize = 4;

/// Terms kept in the direct probability mass function sum beyond `mu^2 * phi`.
const NB_SUM_SLACK: f64 = 10.0;

/// Hard cap on terms in the direct probability mass function sum.
///
/// Above `phi = 4.001` the negative binomial has almost all of its mass on the
/// first few counts, so fifty terms is a generous tail.
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
/// only the left half of its Chebyshev domain is ever used. That is how the C
/// writes it.
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
/// The fit is stitched from panels, each holding its own block of coefficients
/// and its own affine map onto `[-1, 1]`. Storing twice the centre and twice
/// the half-width lets the map be written `(2 * mu - centre2) / width2`, which
/// is the form the C source uses for most panels.
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
///
/// ### Params
///
/// * `panels` - Panels in ascending order of `upper`, the last unbounded
/// * `mu` - Mean to locate
///
/// ### Returns
///
/// The mapped abscissa and the coefficient offset of the panel holding `mu`.
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
/// Used by the large-`mu` case 1 fits, which vary the block of `phi`
/// coefficients with `mu` but apply no map in `mu` at all.
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
/// series per panel. Consecutive panels share an edge, so panel `p` reads its
/// lower edge at `offset` and its upper edge at `offset + len`. The `mu` series
/// that blends the two sits `edges * len` further along.
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
/// directions, so the combination is left to the caller.
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
/// The basis is generated by the three-term recurrence and accumulated in
/// ascending order, matching the reference term for term. Nothing is
/// allocated: the recurrence carries two values through the loop.
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
/// Coefficients are laid out `y`-major: term `(i, j)` multiplying
/// `T_j(x) * T_i(y)` sits at `i * nx + j`. Both recurrences are run inline, the
/// `x` one restarted for each `y` term, so no basis is materialised.
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
///
/// ### Params
///
/// * `mu` - Mean, expected to be at least [`POIS_ASYMPTOTIC_MU`]
///
/// ### Returns
///
/// `1 - 1/(6 mu) - 1/(2 mu^2)`, the tail edgeR uses in place of a fit.
#[inline]
fn pois_alpha_tail(mu: f64) -> f64 {
    1.0 - 1.0 / (6.0 * mu) - 1.0 / (2.0 * mu * mu)
}

/// Large-`mu` expansion of the Poisson deviance degrees of freedom.
///
/// ### Params
///
/// * `mu` - Mean, expected to be at least [`POIS_ASYMPTOTIC_MU`]
///
/// ### Returns
///
/// `1 - 1/(2.5 mu^2)`.
#[inline]
fn pois_kappa_tail(mu: f64) -> f64 {
    1.0 - 1.0 / (2.5 * mu * mu)
}

/// Reciprocal scale of the Poisson unit deviance.
///
/// Piecewise Chebyshev below [`POIS_ASYMPTOTIC_MU`] and an asymptotic
/// expansion above it. The first panel divides out the `log(mu)` singularity
/// that the deviance carries as `mu -> 0`.
///
/// ### Params
///
/// * `mu` - Fitted mean
///
/// ### Returns
///
/// `2 E[d] / Var[d]` for `d` the Poisson unit deviance at `mu`, or zero when
/// `mu` is below [`MIN_MU`].
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
/// `mu` is below [`MIN_MU`].
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
/// Three regimes in `mu`. Below [`NB1_MID_MU`] a joint `7 x 7` fit in
/// `(mu, phi)` multiplies the Poisson answer. Between there and
/// [`NB1_LARGE_MU`] the fit is separable: two `phi` series give the value at
/// the panel edges and an `x` series interpolates between them. Above
/// [`NB1_LARGE_MU`] the `mu` dependence has collapsed onto the Poisson
/// asymptote and only a `phi` correction remains.
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
/// Same three regimes as [`nb_alpha_case1`], with its own panel breakpoints
/// and one extra large-`mu` block.
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
/// Below [`NB2_MID_MU`] a joint `10 x 10` fit in `(mu, phi)`, with the same
/// `log(mu)` factor the Poisson fit uses on its first panel. Up to
/// [`NB2_LARGE_MU`] a separable fit in `1/mu` and `phi`. Above that the
/// `mu -> inf` edge alone.
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
///
/// ### Params
///
/// * `mu` - Fitted mean, expected in `[50, 5000)`
///
/// ### Returns
///
/// The mapped abscissa and the panel's coefficient offset.
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
/// Once `phi` clears [`PHI_CASE2_MAX`] the negative binomial is concentrated on
/// small counts, so the moments can be summed outright. The term count grows
/// with `mu^2 phi` and is capped at [`NB_SUM_MAX`], which covers the mass to
/// well past any precision the fits offer elsewhere.
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

    // Both moments are needed before the variance can be accumulated, so the
    // terms are held rather than folded. The cap keeps them on the stack.
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
/// Dispatches on `phi`: the exact Poisson limit at zero, the case 1 fits below
/// [`PHI_CASE1_MAX`], the case 2 fits below [`PHI_CASE2_MAX`], and direct
/// summation above that.
///
/// ### Params
///
/// * `mu` - Fitted mean
/// * `phi` - Dispersion, non-negative
///
/// ### Returns
///
/// `2 E[d] / Var[d]` for `d` the unit deviance at `(mu, phi)`, or zero when
/// `mu` is below [`MIN_MU`].
pub fn nb_alpha(mu: f64, phi: f64) -> f64 {
    compute_weight(mu, phi, 1.0).0
}

/// Degrees of freedom of the negative binomial unit deviance.
///
/// Dispatches on `phi` exactly as [`nb_alpha`] does. Callers wanting both
/// moments above [`PHI_CASE2_MAX`] should use [`compute_weight`], which sums
/// the probability mass function once instead of twice.
///
/// ### Params
///
/// * `mu` - Fitted mean
/// * `phi` - Dispersion, non-negative
///
/// ### Returns
///
/// `2 E[d]^2 / Var[d]` for `d` the unit deviance at `(mu, phi)`, or zero when
/// `mu` is below [`MIN_MU`].
pub fn nb_kappa(mu: f64, phi: f64) -> f64 {
    compute_weight(mu, phi, 1.0).1
}

/// Both quasi-likelihood weights for one observation.
///
/// The prior divides the fitted value before the weights are taken, which is
/// how edgeR folds the prior count out of the mean it fitted with it.
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
/// degrees of freedom it carries. Both are zero for a mean below [`MIN_MU`].
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

/// The unit deviance, re-exported from its canonical home.
///
/// The quasi-likelihood weights and the GLM fitter must agree on this to the
/// last bit, so there is exactly one implementation and it lives in
/// [`crate::glm::deviance`].
pub use crate::glm::deviance::unit_nb_deviance;

////////////////////////
// Coefficient tables //
////////////////////////

/// Chebyshev coefficients of the Poisson alpha fit, five panels of ten.
const POIS_ALPHA_COEF: [f64; 50] = [
    0.992269079723461,
    -0.00876330120996393,
    -0.000899675388042544,
    7.89660557196009e-05,
    -2.57354549725262e-05,
    1.08697519751391e-05,
    -5.35069556616911e-06,
    2.84372162217423e-06,
    -1.50386201954269e-06,
    6.57650652677658e-07,
    1.85780813766284,
    1.24247480214702,
    -0.21715340185502,
    -0.0631511240632299,
    0.00340555750851111,
    0.00803518029869897,
    -0.00193332946300635,
    0.000422285597551136,
    -0.000358792369207728,
    0.00018078402690016,
    1.91315214581754,
    -0.856602731401166,
    0.14506703274623,
    0.0282393880322191,
    -0.0342290401769809,
    0.0164475718752771,
    -0.00537219169999174,
    0.00111151176883841,
    3.53510678861024e-06,
    -0.000112100033404627,
    0.993770062727655,
    -0.137524078134838,
    0.0577356484590452,
    -0.0146150722546239,
    0.00321203741534916,
    -0.000697567780801707,
    0.000148410349572635,
    -2.99784860990367e-05,
    5.64558922538649e-06,
    -9.49347957962532e-07,
    0.963842126602874,
    0.0400326805227385,
    -0.0189806647831021,
    0.00660614625198861,
    -0.000644918979770783,
    -0.00133133783618231,
    0.00141013711041267,
    -0.000921896816221545,
    0.000464099586562025,
    -0.00017597576489865,
];

/// Chebyshev coefficients of the Poisson kappa fit, five panels of ten.
const POIS_KAPPA_COEF: [f64; 50] = [
    1.98775180998087,
    -0.0140162756693573,
    -0.00156029275453937,
    0.000123049848636432,
    -4.08304177779774e-05,
    1.71720371481344e-05,
    -8.42336547993185e-06,
    4.4646802776689e-06,
    -2.35658416200667e-06,
    1.02938173103251e-06,
    1.60458875341805,
    1.477085480242,
    -0.19980775848221,
    -0.131588940571592,
    0.0220740847667339,
    0.00967036309540797,
    -0.0016317349486263,
    -0.000935545763694168,
    0.000214605308912541,
    5.45674582952849e-05,
    2.03462160780017,
    -0.732803114491053,
    0.0825292616057215,
    0.0319556185536422,
    -0.0247880343267009,
    0.0100493097517216,
    -0.00293285060419899,
    0.000582324399808305,
    -2.62735420803323e-05,
    -3.47213262170909e-05,
    1.08644201311408,
    -0.190240854290349,
    0.0875963322357097,
    -0.026205913685883,
    0.0065783791460467,
    -0.00160331441788305,
    0.000386861436014592,
    -8.86429171986961e-05,
    1.855123830804e-05,
    -3.33626166532652e-06,
    0.989053963537709,
    0.015979160846486,
    -0.00859775363058498,
    0.0032276071532244,
    -0.000399428679513422,
    -0.000588586841491422,
    0.00065224130088336,
    -0.000431002058576289,
    0.000218230833245675,
    -8.32345704255971e-05,
];

/// Case 1 alpha, joint `(mu, phi)` fit: six panels of `7 x 7`.
const NB_A_1_1: [f64; 294] = [
    1.04049914108557,
    0.0127829261351696,
    -0.00360455781917675,
    0.00169216027426875,
    -0.000932212854791133,
    0.000515955109346805,
    -0.000233370296542144,
    0.039204768073801,
    0.0124949086947064,
    -0.00351407730783974,
    0.00164497502188393,
    -0.000905424668193214,
    0.000500833449545262,
    -0.000226229529813676,
    -0.00120807372808612,
    -0.000268829996423627,
    8.48980124941129e-05,
    -4.40956191245684e-05,
    2.5209920352366e-05,
    -1.42256360918374e-05,
    6.49146898845057e-06,
    7.90086546632299e-05,
    1.7370412236139e-05,
    -5.25206676548285e-06,
    2.74417720188475e-06,
    -1.573608026899e-06,
    8.89879048604737e-07,
    -4.06545231382484e-07,
    -6.55455243538993e-06,
    -1.55533745363278e-06,
    4.05582620479861e-07,
    -2.17596237102604e-07,
    1.25249172035386e-07,
    -7.09433688382276e-08,
    3.24347786296094e-08,
    6.22832528839386e-07,
    1.69292458244868e-07,
    -3.4362367794877e-08,
    1.94602405615533e-08,
    -1.12653831527225e-08,
    6.39525950737642e-09,
    -2.92661653851364e-09,
    -6.46069021535491e-08,
    -2.10435790040596e-08,
    2.92723268357146e-09,
    -1.8425346846335e-09,
    1.07695805239289e-09,
    -6.13657623352106e-10,
    2.8124615388151e-10,
    1.06424701129544,
    -0.00445227760352371,
    -0.00812278333788542,
    0.0065250301476989,
    -0.00155168300170678,
    0.000534749169753732,
    -0.000199852018701397,
    0.0598095363563134,
    -0.00724465687163299,
    -0.00792637800128617,
    0.00654673850950836,
    -0.0015993633481645,
    0.000574403119281069,
    -0.000265049420968128,
    -0.00386234266184615,
    -0.00222288913294229,
    0.000243267365215804,
    1.70577521913158e-05,
    -3.76973638690368e-05,
    -6.81694428657795e-06,
    7.5387948087131e-06,
    0.000486626937461981,
    0.000453286454186866,
    4.68454695450085e-05,
    -1.96077488303009e-05,
    -3.76665810544259e-06,
    1.99716527656491e-06,
    -2.13931539156521e-07,
    -7.63345309731219e-05,
    -8.39766671322446e-05,
    -1.41022939211181e-05,
    2.7766199703885e-06,
    1.28985391127772e-06,
    -8.59865871701407e-08,
    -6.65035381168432e-08,
    1.32416700591369e-05,
    1.5636556863828e-05,
    2.99560885605302e-06,
    -3.91923794531312e-07,
    -2.3791530594303e-07,
    -1.18988827428514e-08,
    1.26538480275952e-08,
    -2.36309694132414e-06,
    -2.88685662926369e-06,
    -5.83128390212439e-07,
    6.00861732871828e-08,
    3.99610130695816e-08,
    3.44600487993849e-09,
    -1.88365327478652e-09,
    1.18525897060528,
    0.110622667675842,
    -0.0312709289670913,
    -0.00471845164893182,
    0.00568273171751003,
    -0.00266120079391508,
    0.000863763918906479,
    0.179615115315884,
    0.115586391897639,
    -0.0286043133961031,
    -0.00567849513547171,
    0.00582982419494136,
    -0.0025515830961685,
    0.000765890449348615,
    -0.00500183038775851,
    0.00401040800281335,
    0.00249871059318371,
    -0.000834536966280148,
    5.3685139567558e-05,
    0.000139880414808427,
    -9.38061585820301e-05,
    0.000465335006171737,
    -0.000843463634004466,
    -0.000144241295748217,
    0.000112764065432697,
    -7.49013718670456e-05,
    2.16222577426405e-05,
    -6.31043059849805e-07,
    -0.000120696524771701,
    0.000101920368955321,
    1.81287018034102e-05,
    -1.25019661425765e-05,
    1.48471005362438e-05,
    -7.100360218617e-06,
    2.04104488493934e-06,
    3.35975385446848e-05,
    -6.91384424492635e-06,
    -3.42736805195046e-06,
    1.90216717199321e-06,
    -2.58282226787259e-06,
    1.34105601946279e-06,
    -4.62285763372737e-07,
    -8.86279025545242e-06,
    -1.61640138437832e-06,
    4.87589207033766e-07,
    -3.7120970648758e-07,
    4.4000491267876e-07,
    -2.15051162638944e-07,
    7.50558076331363e-08,
    1.19544438789949,
    -0.0773100100554574,
    -0.00492889189508714,
    0.00349917759124764,
    -0.000630069859470966,
    7.5664290041388e-05,
    3.19847626071802e-06,
    0.212049412445435,
    -0.0623720849879723,
    -0.00565013585624764,
    0.00323979845792352,
    -0.000615217145014969,
    7.53241210873651e-05,
    -5.15454196224282e-06,
    0.0135242642247185,
    0.0122443868397377,
    -0.000645509764621205,
    -0.000154889669365253,
    1.14861054572146e-05,
    3.85150436726167e-06,
    -6.24947181321617e-07,
    -0.0026722443881629,
    -0.00223650358184969,
    5.14601588431917e-05,
    8.10922321881989e-05,
    -2.1460567026664e-06,
    -2.55578951600511e-06,
    4.0720062807818e-07,
    0.000397676639522537,
    0.000399714773806229,
    -2.95662441419623e-05,
    -2.84024606483782e-05,
    5.42937934241391e-07,
    9.21938274203052e-07,
    -9.7304942841531e-08,
    -2.51256487247361e-05,
    -3.99600654832094e-05,
    1.72966752433537e-05,
    8.82368754936036e-06,
    -1.11695321988844e-07,
    -3.3375599713366e-07,
    2.15921415421398e-08,
    -1.27833636664884e-05,
    -7.42633007900127e-06,
    -7.09805069751622e-06,
    -2.44469403463076e-06,
    3.93701365634999e-08,
    1.06064543202022e-07,
    -4.14675873596401e-09,
    1.0225567220461,
    -0.0673925936177268,
    0.0212753698769062,
    -0.00458336079276947,
    0.000456790781947487,
    7.56225994568141e-05,
    -3.29624763155598e-05,
    0.0541208982148202,
    -0.0716914555311019,
    0.0175474809277264,
    -0.00315179676762023,
    0.00030597647545073,
    4.83869507480045e-05,
    -3.23169564224085e-05,
    0.0301271743662228,
    0.000905455043074157,
    -0.00327082016952318,
    0.000883937996216929,
    -0.00010416961367834,
    -9.71017918700334e-06,
    7.80270018611677e-06,
    -0.00294828536697095,
    0.0029539363538755,
    0.00069829678072009,
    -0.000382241782614151,
    5.4590469528186e-05,
    7.08704105324809e-06,
    -4.96234618603406e-06,
    -0.000651426477171804,
    -0.00156161830675502,
    1.25268568367179e-05,
    0.000152447697936549,
    -3.42978590767859e-05,
    -1.65415392077584e-06,
    2.4698572262237e-06,
    0.000515651163499266,
    0.000508236817598213,
    -0.000113592264780958,
    -4.91891828540802e-05,
    1.862504151202e-05,
    -4.22600136996305e-07,
    -1.09139557488312e-06,
    -0.000186298449474523,
    -0.000111791314945222,
    6.43037922830365e-05,
    1.16263302845459e-05,
    -7.75343230668744e-06,
    5.73350297349161e-07,
    3.91899950830608e-07,
    0.961649944581677,
    -0.00769554332509738,
    0.00221807977830475,
    -0.000584596744438663,
    0.000162468816478291,
    -5.69505081476282e-05,
    -6.02670291160918e-06,
    -0.0229737788535761,
    -0.0154055501813675,
    0.00373539125095005,
    -0.00079289204340887,
    0.000157278185698056,
    -2.94080741251818e-05,
    5.85366352375027e-06,
    0.0208591900989835,
    -0.00684424897975748,
    0.000831366984970052,
    -3.10246373042132e-05,
    -1.81120619589943e-05,
    6.89709338033865e-06,
    -2.02099126676063e-06,
    0.0028249579856111,
    0.00169019911470862,
    -0.000630034157757312,
    0.000115692406348481,
    -1.15422520759204e-05,
    -8.79578182834161e-07,
    6.35910698929679e-07,
    -0.00192392363209026,
    0.000401300963184934,
    0.000159381778478925,
    -6.81190348617846e-05,
    1.42468124967748e-05,
    -1.59916578470436e-06,
    -2.35284913434789e-08,
    0.000437508313134998,
    -0.000437086420173263,
    3.41143137077315e-05,
    2.01189582481101e-05,
    -8.31369698659812e-06,
    1.6763257238898e-06,
    -1.75898013783907e-07,
    -3.32063933269639e-05,
    0.000156510669826146,
    -4.45195427650763e-05,
    -1.08635372960636e-06,
    3.07547659524779e-06,
    -8.92310634907793e-07,
    1.36516152111499e-07,
];

/// Case 1 alpha, separable fit: five edge series then four blends, each of seven.
const NB_A_1_2: [f64; 63] = [
    0.955633454636176,
    -0.0353017999327798,
    0.014801527359565,
    0.00398866639936046,
    -0.00141861487166827,
    4.78412981759927e-05,
    8.0003971219361e-05,
    0.95405198091524,
    -0.0393730367060334,
    0.0116733341309116,
    0.00379776138108256,
    -0.000896666497542174,
    -8.33239067378239e-05,
    5.25286614834842e-05,
    0.953241775830876,
    -0.0415342139986733,
    0.00968827990430808,
    0.00342743877143424,
    -0.000542803055687517,
    -0.000113755226167609,
    7.86348198228029e-06,
    0.952500080216221,
    -0.043546376957867,
    0.00743653256730117,
    0.0026934491727019,
    -0.000152524833664607,
    -8.40494492726932e-05,
    -5.77702773907796e-05,
    0.952046538229577,
    -0.0447035472302332,
    0.00560242980315084,
    0.00170281148403691,
    0.000105282375830592,
    -2.62704164066942e-06,
    -9.59567582148542e-05,
    0.463662750451396,
    -0.49762028390154,
    0.0361887823199828,
    -0.00237069760928119,
    0.000147928207122997,
    -8.98649934616215e-06,
    5.3527662624564e-07,
    0.469749219769089,
    -0.498363361657382,
    0.0301667671216273,
    -0.0016324504404238,
    8.38080956855595e-05,
    -4.17796121517893e-06,
    2.04063316593712e-07,
    0.451256313241208,
    -0.495787088893,
    0.048399714975517,
    -0.00418571540776048,
    0.000341864340877962,
    -2.70336724540148e-05,
    2.08303944586914e-06,
    0.428749253234344,
    -0.491146072493907,
    0.0702137207158178,
    -0.00873816129301612,
    0.00102411089556573,
    -0.000114374459178271,
    1.26178552522901e-05,
];

/// Case 1 alpha, large `mu`: three `phi` series of six.
const NB_A_1_3: [f64; 18] = [
    0.951987668582991,
    -0.0448581648629627,
    0.0052764825617196,
    0.00142363147629304,
    0.000164539457990479,
    -4.09021449015654e-05,
    0.951872601228331,
    -0.0449472596488782,
    0.00461714691814746,
    0.00079816777628508,
    0.000173515645685499,
    2.61593127098628e-05,
    0.951844993678891,
    -0.0448948939017227,
    0.00445172575177597,
    0.000584124087073856,
    0.000156610785751354,
    4.31713532349729e-05,
];

/// Case 1 kappa, joint `(mu, phi)` fit: six panels of `7 x 7`.
const NB_K_1_1: [f64; 294] = [
    1.01093193289832,
    0.00462853770318509,
    -0.0013257790514778,
    0.00058718464473768,
    -0.000313667356834079,
    0.00017051235847544,
    -7.64887186619954e-05,
    0.0105133161944017,
    0.00444339389041176,
    -0.00128494354041815,
    0.000567492953176159,
    -0.00030302590164197,
    0.000164652911822336,
    -7.36629415499648e-05,
    -0.000390213876528697,
    -0.000172250152098295,
    3.83388390407595e-05,
    -1.83281385721324e-05,
    1.00412565538404e-05,
    -5.52709717036985e-06,
    2.48893880146899e-06,
    2.58914015036293e-05,
    1.16187466558406e-05,
    -2.39058520599079e-06,
    1.17751559814183e-06,
    -6.47018264245274e-07,
    3.56892441907046e-07,
    -1.60890095810609e-07,
    -2.22600165151887e-06,
    -1.07967380032374e-06,
    1.79197682909908e-07,
    -9.37820851908349e-08,
    5.18811661464574e-08,
    -2.87007407590861e-08,
    1.29549707151856e-08,
    2.25050796907116e-07,
    1.23206553274927e-07,
    -1.38646567634652e-08,
    8.30202137219825e-09,
    -4.64772123276187e-09,
    2.58359957155045e-09,
    -1.16853096810167e-09,
    -2.55165450489502e-08,
    -1.61020691268932e-08,
    9.34469288478596e-10,
    -7.66536134276264e-10,
    4.38164982807883e-10,
    -2.45613031618184e-10,
    1.11458233555504e-10,
    0.999782466356686,
    -0.0217395164752148,
    -0.00136065410605654,
    0.00410671347791474,
    -0.000724838429757106,
    0.000211477224137636,
    -0.000119969954131525,
    -0.00262479261783484,
    -0.0232863554628768,
    -0.00113706519554179,
    0.00402867715254928,
    -0.000759468050041847,
    0.000211484497171724,
    -0.000114981890208271,
    -0.00192052067361352,
    -0.0010287206211033,
    0.00027383989199694,
    -0.000102862787249167,
    -3.57645126906077e-05,
    2.2555624227282e-07,
    5.8836599538702e-06,
    0.000398533720169022,
    0.000407594606430905,
    3.33336751051403e-05,
    -1.48913633591519e-05,
    -1.04378380498225e-06,
    1.44004221776175e-06,
    -3.02312905541102e-07,
    -7.43659848081565e-05,
    -8.69927910677015e-05,
    -1.39500576738242e-05,
    3.12001526702217e-06,
    1.16203313780644e-06,
    -1.07204723581969e-07,
    -5.75913133290787e-08,
    1.38674400564893e-05,
    1.70416555369113e-05,
    3.27002078732208e-06,
    -4.89104373764075e-07,
    -2.62864492674841e-07,
    -9.00083988528769e-09,
    1.44493039230895e-08,
    -2.55070621493684e-06,
    -3.20438480135503e-06,
    -6.61829726017853e-07,
    7.55100519434572e-08,
    4.78887143731464e-08,
    3.75453287591688e-09,
    -2.33418331617108e-09,
    1.08657653584548,
    0.112966369812148,
    -0.00431235744212862,
    -0.00704453550109759,
    0.00311144953809285,
    -0.000810352751102421,
    9.10051060002878e-05,
    0.0783190959635853,
    0.110472726637074,
    -0.00212212587266686,
    -0.00713052053585931,
    0.00289293936980041,
    -0.000680201162981986,
    6.10057779217409e-05,
    -0.00736838089287954,
    -0.00305250670560969,
    0.00190730229152324,
    3.84905399852992e-05,
    -0.000244737412736932,
    0.000116218158658951,
    -2.72785127527176e-05,
    0.000721259935939989,
    -0.000439099378093384,
    -0.000238226623782394,
    9.444636263816e-05,
    -1.50171511032126e-05,
    -7.09683143743382e-06,
    5.1115190463917e-06,
    -0.000127298147240348,
    0.000102881239728336,
    3.03038562803951e-05,
    -2.07513430795027e-05,
    8.28724587764598e-06,
    -9.80197156702065e-07,
    -5.59126101957378e-07,
    2.95186782478303e-05,
    -1.49379889428777e-05,
    -5.36546584476294e-06,
    3.7232010505347e-06,
    -1.81484639404335e-06,
    3.83589252838927e-07,
    3.71028581367066e-08,
    -6.9843979155793e-06,
    1.37119335749273e-06,
    1.07793979273086e-06,
    -6.38299779029065e-07,
    3.18614295018979e-07,
    -7.49778956768421e-08,
    -1.18446714391017e-09,
    1.20349910269204,
    -0.0244285226929874,
    -0.027501476565759,
    0.00863082537382384,
    -0.00114178819870073,
    5.21429464931338e-05,
    4.6461808467283e-07,
    0.212726278839578,
    -0.00550329429150672,
    -0.0273239985214072,
    0.0079148273577492,
    -0.00110121363752523,
    6.42002092030218e-05,
    1.02120877488363e-06,
    0.00572878785085981,
    0.015138654588278,
    0.000186881606676356,
    -0.000534786691047413,
    5.63841270982302e-05,
    1.37319358571942e-06,
    1.29552102139137e-06,
    -0.00280610191434446,
    -0.00303300736111316,
    -7.79574386808508e-06,
    0.000117623331500726,
    1.03707325614439e-05,
    -7.75851575797932e-06,
    8.56880629246779e-07,
    0.000580804735087941,
    0.00062228609668188,
    -2.08594742776028e-06,
    -4.23367598301475e-05,
    -4.03213665647869e-06,
    2.56484736050404e-06,
    -1.21546393508457e-07,
    -0.000103367003595746,
    -0.000119351261790355,
    7.36539298671015e-06,
    1.39637757399001e-05,
    1.37647347766318e-06,
    -8.0227960245283e-07,
    -2.45588464651666e-08,
    1.43304418609481e-05,
    1.87725861273021e-05,
    -3.91716443155408e-06,
    -4.05137380291927e-06,
    -4.1185511627974e-07,
    2.36555685248085e-07,
    2.01211479970673e-08,
    1.07343427008151,
    -0.0636462581170872,
    0.0186914102849887,
    -0.00343711271748352,
    0.000140987066879751,
    0.00012864664524751,
    -4.32557212947592e-05,
    0.105450092513193,
    -0.0653821427116433,
    0.0143032552996045,
    -0.00192784293680669,
    -6.99805732116574e-05,
    0.00014016744333778,
    -4.8971167852551e-05,
    0.0281573112395602,
    0.00287946115242847,
    -0.00371090944608537,
    0.000921031225858324,
    -9.43314889210288e-05,
    -1.5424326890036e-05,
    9.14235914447912e-06,
    -0.00426825705072003,
    0.00269799221060197,
    0.000813968607467724,
    -0.000389659058420281,
    5.11423453428474e-05,
    8.24758004262836e-06,
    -5.10616154165635e-06,
    -4.41310251759636e-05,
    -0.00143123961752165,
    -4.64600411910262e-05,
    0.000156900251006438,
    -3.22511334916637e-05,
    -2.38255307642868e-06,
    2.58069618072368e-06,
    0.000254902314608232,
    0.00043876655814503,
    -8.52908834165834e-05,
    -5.21272798938455e-05,
    1.7681708408662e-05,
    -1.44885769036637e-08,
    -1.16121307068839e-06,
    -9.15081196127163e-05,
    -8.2862934808353e-05,
    5.38220725879876e-05,
    1.29701644294864e-05,
    -7.40854488867587e-06,
    3.95539249126622e-07,
    4.2439391478743e-07,
    1.01431155927191,
    -0.00785380365724673,
    0.00231880183982156,
    -0.000598760225672643,
    0.000153584007173812,
    -4.19478550416061e-05,
    -6.13826727855979e-06,
    0.0320223464107577,
    -0.0157934953086863,
    0.00370906775192677,
    -0.000766195919174174,
    0.000148805336054292,
    -2.77366843767371e-05,
    4.42658380611198e-06,
    0.0206954884551707,
    -0.00678012662842718,
    0.000707777463459694,
    2.76744400985432e-06,
    -2.497550964731e-05,
    8.30393455164392e-06,
    -2.19368419964827e-06,
    0.00163633293770298,
    0.00197896823988567,
    -0.000633332853909624,
    0.000107976661250467,
    -9.19773408543018e-06,
    -1.30235141865011e-06,
    7.34263614429709e-07,
    -0.00134463758440543,
    0.000297151243551216,
    0.000163990395737169,
    -6.59069485923075e-05,
    1.32723114990257e-05,
    -1.37731593330268e-06,
    -7.65643730530063e-08,
    0.000156726293081861,
    -0.000412607162128543,
    3.31923919247524e-05,
    1.98143522185518e-05,
    -8.02903105479738e-06,
    1.57552693133545e-06,
    -1.48898592466148e-07,
    8.10664711169916e-05,
    0.000155532303123849,
    -4.5011318484676e-05,
    -1.18679606570382e-06,
    3.04070004784726e-06,
    -8.60163521395389e-07,
    1.2540732452363e-07,
];

/// Case 1 kappa, separable fit: five edge series then four blends, each of seven.
const NB_K_1_2: [f64; 63] = [
    1.00834766392583,
    0.0192979279059682,
    0.0146083035144376,
    0.00308022583666966,
    -0.000937596125903907,
    -0.000209491597665304,
    0.000192710909226761,
    1.00686253063177,
    0.0149356810248778,
    0.01126012476528,
    0.00308772268183088,
    -0.000480799720749079,
    -0.000320327887887822,
    0.000159721286563437,
    1.00619751243379,
    0.0125737930820398,
    0.00907164985240881,
    0.0028592201089205,
    -0.000175761102535182,
    -0.00033094253440056,
    0.00010751046243034,
    1.00575021163364,
    0.0103183760283486,
    0.00650375639074266,
    0.00230727584847106,
    0.000144113576127633,
    -0.000265049112222215,
    2.59224051219644e-05,
    1.00574866572579,
    0.00894649307842045,
    0.00429575570667532,
    0.00149501855868558,
    0.000315522538881028,
    -0.000130001158361778,
    -3.71662558734308e-05,
    0.463732255482877,
    -0.497650142070071,
    0.0361230879882763,
    -0.00234119665174697,
    0.000144146492079203,
    -8.63146398489516e-06,
    5.06598335804829e-07,
    0.469597928838205,
    -0.49836556211003,
    0.0303189893820078,
    -0.00163034251955743,
    8.28836559655311e-05,
    -4.08587908107867e-06,
    1.97228693297826e-07,
    0.450567753089914,
    -0.495732824123936,
    0.049085905164632,
    -0.0042400073062975,
    0.000344254834029005,
    -2.70096212096378e-05,
    2.06319779323077e-06,
    0.426448639619048,
    -0.490793472420116,
    0.0724732969298092,
    -0.00908656217111574,
    0.00106474893150343,
    -0.000118537962073022,
    1.30117552146679e-05,
];

/// Case 1 kappa, large `mu`: four `phi` series of six.
const NB_K_1_3: [f64; 24] = [
    1.00583242222469,
    0.00873709502448643,
    0.00376552611974505,
    0.00123376716719861,
    0.000319027732575897,
    -0.000103464972872556,
    1.00607786894243,
    0.00858542384423476,
    0.00297488242801692,
    0.000739257572744861,
    0.000276314028663275,
    -1.72329691626533e-05,
    1.00635045350281,
    0.00868250398415398,
    0.00254061639483328,
    0.000370445832422752,
    0.00019376009123299,
    3.02571933150642e-05,
    1.00684587759935,
    0.0091522827500655,
    0.00230701638763935,
    -1.80996899546194e-05,
    2.07822890340898e-05,
    3.03635924785072e-05,
];

/// Case 2 alpha, joint `(mu, phi)` fit: six panels of `10 x 10`.
const NB_A_2_1: [f64; 600] = [
    -1.16722391014247,
    -0.0546256908019699,
    0.0170793200599762,
    -0.00804740645688377,
    0.0047630509804499,
    -0.00308300749457653,
    0.00206330505315553,
    -0.00136533690951409,
    0.000837415658420806,
    -0.00039913902158864,
    -0.154153127597451,
    -0.0543377608208736,
    0.0155408219510873,
    -0.00734130700460622,
    0.00433312771788251,
    -0.00279916581200137,
    0.00187065417877222,
    -0.00123660134310969,
    0.000757945664203257,
    -0.0003611214907502,
    0.0130418509478454,
    0.00307967307978365,
    -0.00100469607476531,
    0.000547873443316112,
    -0.000341035002520363,
    0.000227001270109255,
    -0.000154652291396383,
    0.000103544598040902,
    -6.3985911235002e-05,
    3.06260412358941e-05,
    -0.00238914014014019,
    -0.000513276027378282,
    0.000178889289036894,
    -9.6502127610301e-05,
    6.01698555590128e-05,
    -4.01790992063982e-05,
    2.74415257013936e-05,
    -1.84057968086555e-05,
    1.13875686248569e-05,
    -5.45426398265434e-06,
    0.000550637258888052,
    0.000117163382182506,
    -3.99077972713885e-05,
    2.15961963902439e-05,
    -1.35049723021659e-05,
    9.03327356889918e-06,
    -6.17646485351526e-06,
    4.14591638834687e-06,
    -2.56635381652109e-06,
    1.22955163885351e-06,
    -0.000142736810440748,
    -3.09044348983343e-05,
    1.00470294542678e-05,
    -5.4766710992874e-06,
    3.43097178962954e-06,
    -2.29712573511776e-06,
    1.57162341759151e-06,
    -1.05537889845464e-06,
    6.53461918993409e-07,
    -3.13123699004237e-07,
    3.98631473379949e-05,
    8.94900589386699e-06,
    -2.71894875886667e-06,
    1.49803733616335e-06,
    -9.40083243180891e-07,
    6.29908669141569e-07,
    -4.31161944121561e-07,
    2.89617503223381e-07,
    -1.79355253509363e-07,
    8.5951314135875e-08,
    -1.17292434382893e-05,
    -2.77584992207106e-06,
    7.69690506960468e-07,
    -4.30696076793736e-07,
    2.70838964847677e-07,
    -1.8163125399774e-07,
    1.24378856150003e-07,
    -8.3568650111873e-08,
    5.17606052706602e-08,
    -2.48069583119167e-08,
    3.56220236168492e-06,
    9.00169962347319e-07,
    -2.22743872534993e-07,
    1.27323211073609e-07,
    -8.02748554451852e-08,
    5.38885944984316e-08,
    -3.69206790911975e-08,
    2.48134289808225e-08,
    -1.53713315160413e-08,
    7.36752550649388e-09,
    -1.01947738840646e-06,
    -2.74576384660108e-07,
    6.05968248201805e-08,
    -3.54690524186074e-08,
    2.24245042013127e-08,
    -1.50693249190148e-08,
    1.03296512094647e-08,
    -6.94418808958736e-09,
    4.30242026432798e-09,
    -2.06232665644554e-09,
    2.19177644367907,
    1.46946112268895,
    -0.287261982947043,
    -0.0110204727590795,
    0.000108717599429569,
    0.00708322092644729,
    -0.00402479139611959,
    0.00205267004136486,
    -0.00114413903311179,
    0.000518374751056418,
    0.285045603833933,
    0.11185463606765,
    -0.0497741492825227,
    0.0516503393146531,
    -0.00350286245590367,
    -0.00315705919773758,
    -6.78264538763366e-05,
    0.000555506505402722,
    -0.000233675644466058,
    7.95613246334069e-05,
    -0.0526010558138137,
    -0.0396868283573155,
    0.00624829284325377,
    -0.00445869833218103,
    -0.00265506452425111,
    0.00116303057697719,
    0.000297552236101003,
    -0.000195169619671923,
    1.66896012995388e-05,
    3.14372243349244e-06,
    0.0162142328873837,
    0.0156424084129644,
    -0.000559652732798628,
    -0.000504125908805775,
    0.000646472953550888,
    -1.10300198742975e-05,
    -0.000164658282787258,
    3.08384415503524e-05,
    1.34699763366627e-05,
    -3.44272174655889e-06,
    -0.00612089420814747,
    -0.00672214548470423,
    -0.00029120923838331,
    0.000606131329944741,
    -3.18524718565219e-05,
    -7.72489390523521e-05,
    3.26445136384769e-05,
    6.51910226128158e-06,
    -5.58344029462565e-06,
    -5.18312174733047e-07,
    0.00252484919195595,
    0.00298752694551629,
    0.000292460029201985,
    -0.000322806731059286,
    -6.09371070597273e-05,
    3.88511587300224e-05,
    2.1293052415021e-06,
    -5.10727697728961e-06,
    3.98114080277897e-07,
    8.32646755986661e-07,
    -0.0010827657123351,
    -0.0013394439606634,
    -0.000180783078674982,
    0.000147302746725034,
    4.6488629281248e-05,
    -1.45343441991703e-05,
    -5.45593188669291e-06,
    1.73714165178067e-06,
    5.66476978573623e-07,
    -3.35910555635441e-07,
    0.000471317019243641,
    0.000599093493869607,
    9.57268453069892e-05,
    -6.37830175484136e-05,
    -2.51721584296757e-05,
    4.91043578184438e-06,
    3.39307120227866e-06,
    -3.7955780177606e-07,
    -4.11324963973882e-07,
    7.14125669417352e-08,
    -0.000202192413300886,
    -0.000261296950522389,
    -4.59749048757893e-05,
    2.66174455281915e-05,
    1.18286376489109e-05,
    -1.59560908034422e-06,
    -1.6269543353966e-06,
    2.57453293400333e-08,
    1.96597338545049e-07,
    1.95498311998077e-09,
    7.51971927407524e-05,
    9.80775705378845e-05,
    1.81795331751011e-05,
    -9.63366759392903e-06,
    -4.55062717044374e-06,
    4.74746246515653e-07,
    6.20301380558489e-07,
    2.10084676717168e-08,
    -7.2297246369547e-08,
    -8.43880485993081e-09,
    2.56813447698693,
    -0.735435030513319,
    0.181907489726715,
    -0.00232003127043097,
    -0.0365247447481438,
    0.0345069157183025,
    -0.0235680035713836,
    0.013725541243189,
    -0.00703887125750823,
    0.00287807123930044,
    1.08414492797523,
    0.26850094769957,
    -0.263727264648089,
    0.134388795947091,
    -0.0447053294814637,
    0.00561014692581434,
    0.00668253547945574,
    -0.00784330647108019,
    0.00548048490661248,
    -0.00263068139028896,
    -0.0676993337635559,
    0.0999051824668334,
    0.0221216016847372,
    -0.0285790870906516,
    0.0191623723933489,
    -0.00838882790868529,
    0.00181580553543879,
    0.000810376762688633,
    -0.0012561828495928,
    0.000776988891986595,
    -0.00179625480432948,
    -0.0352986016849718,
    0.00874591646627203,
    0.00222525678470606,
    -0.00404861276064511,
    0.00309817214709717,
    -0.00166029454645659,
    0.00056871565700848,
    -3.83266079181114e-05,
    -7.73605921550287e-05,
    0.00334045993385561,
    0.0139511231194863,
    -0.00287404648937032,
    0.00119797785999628,
    0.00015068737160496,
    -0.000780932422718831,
    0.000679304789816955,
    -0.000402098303682325,
    0.000185743233470867,
    -6.33417365776938e-05,
    -0.001104007996255,
    -0.00537626415760712,
    0.00092773047796782,
    -0.000711063194802622,
    0.000189548736857684,
    0.000124383610285577,
    -0.000208393079051185,
    0.000179061614308504,
    -0.000111480726104121,
    4.94348126689509e-05,
    0.000479972047772159,
    0.00244352306905941,
    -0.000307681755542364,
    0.000241992733934785,
    -0.000135712792476898,
    -5.80607948387789e-06,
    6.81383749926438e-05,
    -6.90021842025716e-05,
    4.83856022338225e-05,
    -2.40845748327752e-05,
    -0.000229970198322359,
    -0.00117018063751512,
    7.39321046462887e-05,
    -9.19974783162457e-05,
    6.78340944687064e-05,
    1.23516625991582e-06,
    -2.41405317004791e-05,
    2.54174683407126e-05,
    -1.96047953146037e-05,
    1.03343998353437e-05,
    9.19612817412685e-05,
    0.000522987018644589,
    -1.55945058044682e-05,
    3.76617507368171e-05,
    -2.67180538380174e-05,
    -9.13003134069557e-07,
    9.10631866206234e-06,
    -1.01677393679741e-05,
    7.80663609360799e-06,
    -4.02335126942807e-06,
    -2.8581569730477e-05,
    -0.000194455947587304,
    4.84864216231173e-06,
    -1.18076098844933e-05,
    9.2920080057024e-06,
    6.34882175884477e-08,
    -3.60135805047082e-06,
    3.75925092441567e-06,
    -2.66277077343599e-06,
    1.37350197870992e-06,
    1.74454187246023,
    -0.187598343804154,
    0.0382711278100216,
    -0.0088192387602456,
    0.00212102130818851,
    -0.0005087984048742,
    0.000118729409860525,
    -2.68805884123034e-05,
    5.98580432942804e-06,
    -1.27931757615315e-06,
    0.966824819981767,
    -0.169017612227891,
    0.0242209724162969,
    -0.00292215350597323,
    8.77293385636511e-05,
    9.76513246098106e-05,
    -4.25117535519911e-05,
    1.28884954271538e-05,
    -3.56598117860342e-06,
    9.31740559574186e-07,
    0.0882593511403611,
    0.0493593219848723,
    -0.0125792305741734,
    0.00253056934877706,
    -0.000422797520529407,
    6.6610877964794e-05,
    -1.40752631538471e-05,
    4.32445655698079e-06,
    -1.33631602015216e-06,
    3.31471747064483e-07,
    -0.0497383355609957,
    -0.0100990502955599,
    0.0038901807858901,
    -0.000684437380315692,
    3.28383922387991e-05,
    1.6384392122421e-05,
    -3.69597093861809e-06,
    -5.7618732632183e-07,
    5.89867317183099e-07,
    -2.01744910877495e-07,
    0.0196971878199526,
    0.00338391360223226,
    -0.00230173364844542,
    0.000407774584919418,
    1.52767301522715e-05,
    -2.78450737413495e-05,
    7.51332517221607e-06,
    -7.59919309939514e-07,
    -1.67204712941793e-07,
    8.92812510706885e-08,
    -0.00807898935483216,
    -0.000244300925170411,
    0.00148121454802224,
    -0.000331468769807696,
    -2.86903966173194e-06,
    2.09716695729223e-05,
    -6.08365932896113e-06,
    6.80876541341752e-07,
    1.0921605102604e-07,
    -6.34516009295832e-08,
    0.00291123867191853,
    -0.00108467868663886,
    -0.000872583527960933,
    0.000252992286999821,
    -4.74232735813076e-06,
    -1.46780336607586e-05,
    4.49606281850305e-06,
    -4.87981098641842e-07,
    -9.6629302154971e-08,
    5.2112491020213e-08,
    -0.00069946993816816,
    0.00129993195419112,
    0.000475011021810713,
    -0.000179764825656301,
    7.18705714651002e-06,
    9.87920979891111e-06,
    -3.18833421353199e-06,
    3.42695800982589e-07,
    7.58558947739466e-08,
    -3.96631117082906e-08,
    -6.10783391032221e-05,
    -0.00100612342561093,
    -0.000234968842541174,
    0.000114645206259347,
    -6.49091376239518e-06,
    -6.05164406006307e-06,
    2.04524698576172e-06,
    -2.21035685526503e-07,
    -5.10136086302114e-08,
    2.64390790646883e-08,
    0.000156265116218659,
    0.000527797812255377,
    9.38173383589774e-05,
    -5.56883530036052e-05,
    3.75740782707781e-06,
    2.86259048827464e-06,
    -9.98809414951321e-07,
    1.08707533173154e-07,
    2.5482418813393e-08,
    -1.3178749078964e-08,
    1.44661823459421,
    -0.112132427725358,
    0.0225665033477026,
    -0.00514309093235508,
    0.00123550836774177,
    -0.000305357644675104,
    7.68108042368605e-05,
    -1.95727645613383e-05,
    5.03034835047846e-06,
    -1.2300787121587e-06,
    0.660748521290301,
    -0.127651361243069,
    0.0241371692705357,
    -0.00519212600386762,
    0.00117428995572317,
    -0.00027219631724216,
    6.38735454701334e-05,
    -1.5011167354352e-05,
    3.47585681041505e-06,
    -7.47839170020245e-07,
    0.143054830440596,
    0.00948432702827647,
    -0.00437325069952857,
    0.00140828370484909,
    -0.000418654399478355,
    0.000120481849322971,
    -3.40160480697017e-05,
    9.41656576774665e-06,
    -2.52660582372642e-06,
    6.17361831955686e-07,
    -0.0511899526106159,
    0.00572681063870222,
    -6.7137717880131e-05,
    -0.000224662781644157,
    0.000111126102912362,
    -4.17298210174797e-05,
    1.40407616800403e-05,
    -4.38109322211039e-06,
    1.2659741394666e-06,
    -3.19887666558302e-07,
    0.0146267407169202,
    -0.00581124673824278,
    0.000787266421308082,
    -3.17152546177197e-05,
    -3.8394473929288e-05,
    2.30572922598525e-05,
    -9.53751718513942e-06,
    3.34419928697886e-06,
    -1.03701624359654e-06,
    2.73220980497092e-07,
    -0.00169736187122299,
    0.00428548833635551,
    -0.00101365251030567,
    0.000174702553277576,
    -9.41095160444763e-06,
    -9.81722324312262e-06,
    6.22770865970777e-06,
    -2.57771847471041e-06,
    8.7312810281181e-07,
    -2.42488360924254e-07,
    -0.00224845991800643,
    -0.00228831421906382,
    0.000899147657518651,
    -0.000223601015470803,
    3.59427992666547e-05,
    6.82979150050647e-07,
    -3.53657995093706e-06,
    1.85050311506636e-06,
    -6.90443929103337e-07,
    2.0177444992352e-07,
    0.00256555618565994,
    0.000764676711476169,
    -0.000642428604711898,
    0.000205189784488846,
    -4.30180157680216e-05,
    3.93556826682856e-06,
    1.69832932382305e-06,
    -1.23670892531046e-06,
    5.08099897465846e-07,
    -1.55282796546843e-07,
    -0.00176161364431881,
    8.19344393344523e-06,
    0.00038275439914222,
    -0.000149840404312014,
    3.59920088722745e-05,
    -4.81502613958368e-06,
    -6.40560300828965e-07,
    7.42889464217726e-07,
    -3.32417268250669e-07,
    1.05245673425879e-07,
    0.000833264968912353,
    -0.000166620735565239,
    -0.000171721965783591,
    7.77923806682282e-05,
    -2.01074126347173e-05,
    3.07856313982085e-06,
    1.64964358884078e-07,
    -3.44883244445955e-07,
    1.63934163244125e-07,
    -5.31003617196067e-08,
    1.31256031837516,
    -0.0365824361416769,
    0.00335961417072592,
    -0.000353645740028721,
    3.94478528682562e-05,
    -4.54295361332742e-06,
    5.33889296249662e-07,
    -6.36348618395466e-08,
    7.6630450062659e-09,
    -9.16768764813204e-10,
    0.50455844504986,
    -0.0440928900317907,
    0.00389683263168004,
    -0.000398864386748602,
    4.3437668259404e-05,
    -4.89157269026202e-06,
    5.62115191117916e-07,
    -6.54368570302802e-08,
    7.6810711153492e-09,
    -8.94304260851634e-10,
    0.148393046200453,
    -0.00107548115272704,
    -0.000172912808120591,
    3.8891604389596e-05,
    -6.25359699014342e-06,
    9.12434951289837e-07,
    -1.27274797814475e-07,
    1.72978663889585e-08,
    -2.31068509148419e-09,
    3.00020724383349e-10,
    -0.0420314798531724,
    0.003384525760562,
    -0.0002400940719526,
    1.76383742985389e-05,
    -1.12496509951055e-06,
    3.47646581656643e-08,
    6.71319954070153e-09,
    -2.04250364740116e-09,
    3.90236261070008e-10,
    -6.29693064357921e-11,
    0.00718559011387008,
    -0.00212572827993254,
    0.000213499199760115,
    -2.15052744726246e-05,
    2.1515892048549e-06,
    -2.0999990376758e-07,
    1.94351105581905e-08,
    -1.60607246543168e-09,
    9.75700859545674e-11,
    7.00417188683951e-13,
    0.00270124713274945,
    0.000814629352105269,
    -0.000132247946196851,
    1.75726513889312e-05,
    -2.21823472295491e-06,
    2.71987962514597e-07,
    -3.24318167454666e-08,
    3.7368646960407e-09,
    -4.10702230925885e-10,
    4.17420482262914e-11,
    -0.00377119596947872,
    0.000109104742403079,
    4.17772212687551e-05,
    -9.780806025548e-06,
    1.64703626508212e-06,
    -2.44966238308555e-07,
    3.38345157679561e-08,
    -4.41586625313133e-09,
    5.4726512213929e-10,
    -6.35259032847587e-11,
    0.00234402561667888,
    -0.000495927001694854,
    1.87754230757549e-05,
    2.8349129824347e-06,
    -9.30120380918586e-07,
    1.7661155065858e-07,
    -2.7929848665835e-08,
    3.9902142374732e-09,
    -5.30240639338096e-10,
    6.54216074177113e-11,
    -0.000960939494931763,
    0.000488779862466886,
    -3.93379111011428e-05,
    9.30858074028693e-07,
    3.87526175897263e-07,
    -1.05039056118358e-07,
    1.89617265713342e-08,
    -2.91237648749815e-09,
    4.06196146861105e-10,
    -5.2010918515583e-11,
    0.000246180651864663,
    -0.000280135078152641,
    2.80369038325828e-05,
    -1.43782067474354e-06,
    -1.06114810587099e-07,
    4.67071659293539e-08,
    -9.38884370778449e-09,
    1.51548951641865e-09,
    -2.17827110782867e-10,
    2.84960079580858e-11,
];

/// Case 2 alpha, separable fit: four edge series then three blends, each of ten.
const NB_A_2_2: [f64; 70] = [
    1.27901923265303,
    0.464002572908361,
    0.147178090663608,
    -0.0388704949853813,
    0.00525381527471972,
    0.00339922587609729,
    -0.00362866283366029,
    0.001868931031525,
    -0.000510267789729385,
    -7.4228162943224e-06,
    1.20144061121365,
    0.367790084558894,
    0.140146369159432,
    -0.0307135196248815,
    0.00132417869560944,
    0.00406375305848214,
    -0.00266728096612815,
    0.000648279409651754,
    0.000341828144492967,
    -0.000397494193264381,
    1.0511541649247,
    0.171662964840304,
    0.110209630663404,
    -0.0140094523635252,
    -0.00235574124081925,
    0.00193367244900671,
    -4.61020316163428e-05,
    -0.000489868019092097,
    0.000283854974371295,
    -6.17988948719228e-05,
    0.996857562766985,
    0.0964033749846483,
    0.0921530532066982,
    -0.00874158811443212,
    -0.00158096839378969,
    0.000356616153774879,
    0.000500311195000911,
    -0.000281475017127144,
    -4.90682313524948e-05,
    0.000116474304115738,
    0.472255377928953,
    -0.497302789017012,
    0.0274301408074467,
    -0.0026570054977465,
    0.000309042550261815,
    -3.94411074873225e-05,
    5.32827497327774e-06,
    -7.48075989387594e-07,
    1.07938043200639e-07,
    -1.55591509787278e-08,
    0.407666978519895,
    -0.471653746300708,
    0.082061239878469,
    -0.024293763963617,
    0.00858707424343456,
    -0.00332602096993726,
    0.00136136261435912,
    -0.000575597530018397,
    0.000243191107591659,
    -8.98792430236828e-05,
    0.428767300408055,
    -0.483858534063321,
    0.0669338363306574,
    -0.0148958981516649,
    0.00391868666415752,
    -0.00112533022034369,
    0.000341112038849306,
    -0.000107158553608014,
    3.42232770820627e-05,
    -1.01304384733658e-05,
];

/// Case 2 kappa, joint `(mu, phi)` fit: six panels of `10 x 10`.
const NB_K_2_1: [f64; 600] = [
    2.08206884879209,
    0.0308557221218032,
    -0.0121647054052645,
    0.00531555584096633,
    -0.0030349624533247,
    0.00191929115105982,
    -0.00126408730036736,
    0.000827269113943132,
    -0.000503705656442577,
    0.000239083119260642,
    0.0764572189330207,
    0.0315905993674644,
    -0.0105652858563205,
    0.0047287900329908,
    -0.00270045429411021,
    0.00170705751991458,
    -0.00112372941900901,
    0.000735089208641621,
    -0.000447434945423754,
    0.000212333055634876,
    -0.00841491860065422,
    -0.00380601737770928,
    0.000881154205686593,
    -0.000431191682421155,
    0.000256087428209791,
    -0.000165221617256669,
    0.00011016890632675,
    -7.26775946817839e-05,
    4.44766261906645e-05,
    -2.11705388441743e-05,
    0.00153801661508973,
    0.000668460038440809,
    -0.000160088559924449,
    7.96779041178785e-05,
    -4.72741780989762e-05,
    3.05493897268033e-05,
    -2.04015152136304e-05,
    1.34744634959531e-05,
    -8.25261624969471e-06,
    3.93002077166865e-06,
    -0.000355819680275117,
    -0.000154773625858464,
    3.62944263109156e-05,
    -1.81830237142726e-05,
    1.08143319807279e-05,
    -6.99968102479944e-06,
    4.67925514213939e-06,
    -3.09255414219332e-06,
    1.89489755442309e-06,
    -9.02599597767152e-07,
    9.31977337088433e-05,
    4.14412681567896e-05,
    -9.13933052512646e-06,
    4.64914264032397e-06,
    -2.77259551822682e-06,
    1.79691062867971e-06,
    -1.20213554811934e-06,
    7.9487826358562e-07,
    -4.87189188109112e-07,
    2.32101542860918e-07,
    -2.64592078452712e-05,
    -1.22139861788059e-05,
    2.44514715112521e-06,
    -1.27481175881946e-06,
    7.62733519794553e-07,
    -4.95026848472098e-07,
    3.3142727562645e-07,
    -2.19246083017232e-07,
    1.34414510059876e-07,
    -6.40456506167351e-08,
    7.96665322309577e-06,
    3.86865796453358e-06,
    -6.76371911508484e-07,
    3.65969933066178e-07,
    -2.19908898895806e-07,
    1.42979605397787e-07,
    -9.58144543524396e-08,
    6.34161558450848e-08,
    -3.88906179152261e-08,
    1.85334929420932e-08,
    -2.4908741350127e-06,
    -1.28373241370841e-06,
    1.88785460074441e-07,
    -1.07693417515566e-07,
    6.50765927845062e-08,
    -4.24072014629673e-08,
    2.8450353168597e-08,
    -1.88420832328891e-08,
    1.15592387023517e-08,
    -5.50963451477655e-09,
    7.33973184221526e-07,
    3.99613081709105e-07,
    -4.91515236197837e-08,
    2.98196689847707e-08,
    -1.81284348896998e-08,
    1.18417614969335e-08,
    -7.95385339043275e-09,
    5.27109003141749e-09,
    -3.23489603421163e-09,
    1.54218671370981e-09,
    1.43325194104243,
    1.33961237580868,
    -0.152286582746308,
    -0.0726878787229801,
    0.0206950011861293,
    0.000742859733458131,
    -0.000406301049489568,
    -0.000383442186935807,
    0.000178491234435621,
    -3.48051008349293e-05,
    -0.163505671438694,
    -0.187829486288863,
    0.0301022879930871,
    0.0570813358623534,
    -0.00395512774270033,
    -0.00718102928552259,
    0.00143561702877113,
    0.000354199392311261,
    -9.01665764522882e-05,
    -3.9135530135717e-05,
    0.000130421630831756,
    -0.000314190708535859,
    -0.00940232582498807,
    -0.0121516962501225,
    -0.00126852257097601,
    0.0025083017162688,
    5.61803138680095e-05,
    -0.00034141708433001,
    3.01004426511432e-05,
    3.26418451017582e-05,
    0.00758325909171979,
    0.00944065351906358,
    0.00284318959634758,
    0.00175865834940007,
    0.000524483141192739,
    -0.000510745053335503,
    -0.000154601689840876,
    0.000105272438435609,
    1.68377153889253e-05,
    -1.82366187092739e-05,
    -0.00450324870149408,
    -0.00561353445790241,
    -0.000901909650317802,
    0.0001672089516862,
    -6.67387215716642e-05,
    3.70505793733457e-05,
    5.38727975894809e-05,
    -1.45516186572481e-05,
    -1.15505571520472e-05,
    4.59540236846976e-06,
    0.00224931910705733,
    0.00282653727496252,
    0.000341932124064466,
    -0.000315799012720499,
    -4.59204634193846e-05,
    3.17100088984378e-05,
    -6.70299983884861e-06,
    -3.27469348498512e-06,
    3.2416904808306e-06,
    -4.40355148019668e-08,
    -0.00107194647529833,
    -0.00136073152859038,
    -0.000153787491699216,
    0.000200128119979271,
    4.73217056570614e-05,
    -2.48156573845673e-05,
    -4.54672177442266e-06,
    3.49601360387238e-06,
    -8.90435116132203e-07,
    -5.35680793960882e-07,
    0.000498721661001113,
    0.000639269036497244,
    7.48948434133422e-05,
    -0.000102746433910623,
    -2.95504022519054e-05,
    1.30168461314045e-05,
    4.56495354777602e-06,
    -1.79501648991575e-06,
    -4.9108845514763e-07,
    3.15478799650456e-07,
    -0.000223503088458065,
    -0.000288816403078443,
    -3.60763199800596e-05,
    4.74055899025227e-05,
    1.52290975830091e-05,
    -5.89611076959762e-06,
    -2.69891622025134e-06,
    7.47184578729233e-07,
    3.70479087093096e-07,
    -1.2633185902102e-07,
    8.5320618885687e-05,
    0.000110865293805915,
    1.46009322477225e-05,
    -1.81363351940922e-05,
    -6.19844108568454e-06,
    2.20472177248771e-06,
    1.15451773678936e-06,
    -2.56138714298084e-07,
    -1.70535642465769e-07,
    3.89456974521124e-08,
    2.27493407924473,
    -0.371416526926058,
    0.0572609682633895,
    0.0314971839001834,
    -0.0388155974570845,
    0.0286639666025479,
    -0.0177077320654452,
    0.00975223227708291,
    -0.00483439429468741,
    0.00194048406812581,
    0.665798873158015,
    0.527645890527456,
    -0.289447247207714,
    0.117443099625076,
    -0.0303678052152874,
    -0.000718078917131843,
    0.00800019390186596,
    -0.00727256910158515,
    0.00466122255152426,
    -0.0021517082530369,
    -0.124872051835562,
    0.023462317928128,
    0.0626024663361809,
    -0.0407866347234605,
    0.0203658354631932,
    -0.00735424654223463,
    0.00102869167400011,
    0.00106228724110721,
    -0.00122286135614939,
    0.000701510334078165,
    0.0089372446884,
    -0.0312032747827991,
    -0.000580697798952387,
    0.00748177990249174,
    -0.00596597230441212,
    0.00336622471924274,
    -0.00148879112994981,
    0.000415967117169688,
    2.75057635488177e-05,
    -9.356682573777e-05,
    0.00411890144838968,
    0.015823725016081,
    -0.00198078899367366,
    -0.000116065117867908,
    0.000754295816296197,
    -0.000945266614807064,
    0.000667108329721887,
    -0.000344706842744642,
    0.000141655204896463,
    -4.36704147815743e-05,
    -0.00247763223896593,
    -0.00658040733674968,
    0.000954708947718426,
    -0.000546218412213323,
    9.12617613502802e-05,
    0.000161340022456332,
    -0.00019451716365904,
    0.000151203489652689,
    -9.01421396895714e-05,
    3.88572906226195e-05,
    0.0012335381644497,
    0.00296777109211048,
    -0.000424781802905229,
    0.000239789011471104,
    -0.000134405684957213,
    8.0300606863358e-06,
    5.18823350599202e-05,
    -5.53564701027564e-05,
    3.81227010671426e-05,
    -1.85721436419973e-05,
    -0.000627324354276201,
    -0.00145510214666981,
    0.000148619100213682,
    -8.87971420446909e-05,
    8.35330275088902e-05,
    -1.38411226382881e-05,
    -1.47807468748785e-05,
    1.80019822154572e-05,
    -1.43390119422428e-05,
    7.78024399950445e-06,
    0.000295642447593285,
    0.000687568533331209,
    -4.64465173446049e-05,
    3.65288262044161e-05,
    -3.71764318171453e-05,
    6.37938370259661e-06,
    3.94870471337721e-06,
    -6.33339904072622e-06,
    5.50459593445539e-06,
    -2.96369674830103e-06,
    -0.00011337892964832,
    -0.000268164215466733,
    1.56613344036316e-05,
    -1.21541100992961e-05,
    1.2850613989519e-05,
    -2.96929481884511e-06,
    -1.2408670697403e-06,
    2.37717452586966e-06,
    -1.8439720551057e-06,
    9.38223770959312e-07,
    1.80796250637391,
    -0.118993464553825,
    0.0215852701986281,
    -0.00458293181495866,
    0.00103143315653826,
    -0.000230962687788554,
    4.94412121810203e-05,
    -1.00032707889067e-05,
    1.94232371000971e-06,
    -3.61784436775266e-07,
    0.899121892459448,
    -0.0666397846880035,
    0.00276513801905683,
    0.00143570633842226,
    -0.00073752811269604,
    0.000240476147680184,
    -6.452350789293e-05,
    1.55361441762383e-05,
    -3.57791923686531e-06,
    7.90651262813944e-07,
    -0.0111142433368148,
    0.0576829820058678,
    -0.0115256918672587,
    0.00187868220696845,
    -0.000219974584486338,
    1.07782825719954e-05,
    1.79682828723522e-06,
    -3.39931969782325e-07,
    -2.45909863784757e-08,
    1.17716583012805e-08,
    -0.0456399150377906,
    -0.0141907150094285,
    0.00454780753702893,
    -0.00080926721468158,
    7.19184666025495e-05,
    5.45277830483067e-06,
    -2.2542061744278e-06,
    -1.96047125955837e-07,
    2.86802800157748e-07,
    -9.79780960670767e-08,
    0.022765098725876,
    0.00329527102818271,
    -0.00227095045434458,
    0.000424875147464019,
    -1.02920853039813e-05,
    -1.64769560376613e-05,
    4.77731599711788e-06,
    -4.83448954134864e-07,
    -1.06380643099696e-07,
    5.64142273723895e-08,
    -0.00979783930376944,
    0.000111824264498265,
    0.00135947452494467,
    -0.000310718408708645,
    7.89493826266284e-06,
    1.39807934480407e-05,
    -4.27902972209711e-06,
    5.2334265993931e-07,
    5.28667963934928e-08,
    -3.77854669660342e-08,
    0.00382460209409474,
    -0.00125814636180862,
    -0.00079138047643941,
    0.000232186052618335,
    -1.03731904821343e-05,
    -9.98041221638849e-06,
    3.25582688954736e-06,
    -4.05713764614176e-07,
    -4.19321359148361e-08,
    2.96538080017074e-08,
    -0.0012212970789824,
    0.00135420476424938,
    0.000429659218422553,
    -0.000163701235162064,
    1.02364362622419e-05,
    6.7081255921485e-06,
    -2.32451635510323e-06,
    2.916870982122e-07,
    3.28901681314855e-08,
    -2.24470987479532e-08,
    0.000233414518919018,
    -0.00100922100301286,
    -0.000212263990703416,
    0.000103782296471003,
    -7.96907482476237e-06,
    -4.09954579594512e-06,
    1.49592269905224e-06,
    -1.89771446624749e-07,
    -2.22392769726627e-08,
    1.49962578987404e-08,
    2.16717403914579e-05,
    0.000520314799673662,
    8.4716906013436e-05,
    -5.02162736887102e-05,
    4.32588826633155e-06,
    1.93646985450978e-06,
    -7.32047757815658e-07,
    9.3732170874448e-08,
    1.11272562951818e-08,
    -7.4881021527244e-09,
    1.60154877837244,
    -0.0845210628161747,
    0.016096150451567,
    -0.00354680331942509,
    0.000831578909341359,
    -0.000201648315020195,
    4.99383895846401e-05,
    -1.2568739189512e-05,
    3.20440093845721e-06,
    -7.81468887003263e-07,
    0.73996668080051,
    -0.0802549019691381,
    0.0129355929372237,
    -0.00240445061936414,
    0.000464301315707479,
    -8.91219239812355e-05,
    1.63185760794742e-05,
    -2.60395094494651e-06,
    2.42529669958316e-07,
    4.52974449740036e-08,
    0.0701754552026199,
    0.0238789469815556,
    -0.00697316524487234,
    0.00190789624065943,
    -0.000514118866726677,
    0.000137713935997741,
    -3.6700433893059e-05,
    9.67110223995536e-06,
    -2.48239778706464e-06,
    5.8315228542214e-07,
    -0.0530641970188518,
    0.0041713095658793,
    0.000564833458746284,
    -0.000412067332267503,
    0.000162155632076054,
    -5.49965544063627e-05,
    1.73678388864487e-05,
    -5.18729235343706e-06,
    1.45419803220482e-06,
    -3.60495952226814e-07,
    0.0168587540316855,
    -0.00660429334780659,
    0.000810714865676047,
    -8.86029654557825e-06,
    -4.89659945358914e-05,
    2.64330641724163e-05,
    -1.04491037473789e-05,
    3.56043631879922e-06,
    -1.08157656202052e-06,
    2.80718995227825e-07,
    -0.00247385150127898,
    0.00490356637179088,
    -0.00108591067554912,
    0.000172941383554542,
    -5.03043319395579e-06,
    -1.1554515856349e-05,
    6.72025543114217e-06,
    -2.68742430366812e-06,
    8.9049598034828e-07,
    -2.43375941558341e-07,
    -0.00186756600602412,
    -0.00265236565651471,
    0.000956255205082565,
    -0.000224571244190689,
    3.2991589761139e-05,
    2.01266217072606e-06,
    -3.93618883080324e-06,
    1.94269280807538e-06,
    -7.05392445722651e-07,
    2.02561855054149e-07,
    0.00230558319570477,
    0.00096474181547603,
    -0.000680723501651049,
    0.000206473012696603,
    -4.09761406865301e-05,
    2.93964481301306e-06,
    2.01068373294297e-06,
    -1.31125440522742e-06,
    5.20756085408237e-07,
    -1.56138322400053e-07,
    -0.00158334834157499,
    -9.54626068814757e-05,
    0.000405338733951418,
    -0.000150791601927849,
    3.46899392539704e-05,
    -4.14848095842703e-06,
    -8.55554415476213e-07,
    7.95360114310164e-07,
    -3.41586964127945e-07,
    1.05947808533661e-07,
    0.000741433488928838,
    -0.000122333996846629,
    -0.000182053106192765,
    7.82750281579638e-05,
    -1.94714477792787e-05,
    2.74405128528395e-06,
    2.74656724144948e-07,
    -3.72011851296212e-07,
    1.68755949943737e-07,
    -5.34945462065923e-08,
    1.49807273163734,
    -0.0293224670219557,
    0.00255869626860635,
    -0.000260946359611808,
    2.84373531827947e-05,
    -3.21470304483813e-06,
    3.7201178601762e-07,
    -4.3764394611793e-08,
    5.21153829156109e-09,
    -6.17642761086121e-10,
    0.636121041198279,
    -0.0317099798666925,
    0.00253335354798408,
    -0.000240302644035193,
    2.44963878577344e-05,
    -2.59385382632916e-06,
    2.80649224028023e-07,
    -3.0728361514621e-08,
    3.38181807752523e-09,
    -3.67991407451815e-10,
    0.0931887209904959,
    0.00387231631445795,
    -0.000633758528491274,
    8.59953228976375e-05,
    -1.12699593023798e-05,
    1.45792883504897e-06,
    -1.87231179614486e-07,
    2.39178607912262e-08,
    -3.04169054837776e-09,
    3.79542885031772e-10,
    -0.0449438459792192,
    0.00345818045942173,
    -0.00020652433132965,
    1.108421672435e-05,
    -1.302427251037e-07,
    -1.04367239858538e-07,
    2.54245691251454e-08,
    -4.50195701244914e-09,
    7.08606525738887e-10,
    -1.03110574689015e-10,
    0.00818620304737732,
    -0.00256535828359749,
    0.000247859286949875,
    -2.40802888232489e-05,
    2.31216002102924e-06,
    -2.14045189393106e-07,
    1.82994330263047e-08,
    -1.29318845200506e-09,
    4.05488872962646e-11,
    9.52630625341982e-12,
    0.00272365171149271,
    0.0010379573328512,
    -0.000157899800759635,
    2.01327423633803e-05,
    -2.45173689670571e-06,
    2.90709726646909e-07,
    -3.35457667649852e-08,
    3.73468938744831e-09,
    -3.94345121712383e-10,
    3.7994934423454e-11,
    -0.00380695807547749,
    1.91804377083341e-05,
    5.66525723571923e-05,
    -1.15718166595488e-05,
    1.83619392516913e-06,
    -2.6272737586943e-07,
    3.52263252453383e-08,
    -4.48180773440372e-09,
    5.42056093679028e-10,
    -6.13317973208957e-11,
    0.00228891832970032,
    -0.000464271006874102,
    1.11630088282949e-05,
    3.93618622645726e-06,
    -1.06213999342901e-06,
    1.90453412684967e-07,
    -2.91682949498514e-08,
    4.07108657318181e-09,
    -5.3027503631325e-10,
    6.41923440026709e-11,
    -0.000880576251740988,
    0.00047786707349005,
    -3.58679228448552e-05,
    3.31447017738358e-07,
    4.67913807140891e-07,
    -1.14221473683187e-07,
    1.98561483269185e-08,
    -2.97969416326819e-09,
    4.08030506488687e-10,
    -5.13873126567838e-11,
    0.000194570507801508,
    -0.000276330190633692,
    2.67466944393796e-05,
    -1.18072955036708e-06,
    -1.43577987822575e-07,
    5.12395763315701e-08,
    -9.85336120575911e-09,
    1.55301762469516e-09,
    -2.19292345930665e-10,
    2.84960079580858e-11,
];

/// Case 2 kappa, separable fit: four edge series then three blends, each of ten.
const NB_K_2_2: [f64; 70] = [
    1.47107357007529,
    0.606726267737109,
    0.0965032960556014,
    -0.0416813187048704,
    0.00584673893094613,
    0.00362165078856726,
    -0.00374109218485856,
    0.00183884926199942,
    -0.000437874723663312,
    -5.62945505165113e-05,
    1.40633587142072,
    0.532573697441418,
    0.0996264069268105,
    -0.0326665009374596,
    0.000948940588515306,
    0.0046158235979294,
    -0.00283430769938751,
    0.000606830944577483,
    0.000422247979015644,
    -0.000446107066080925,
    1.27052060401718,
    0.362954588828754,
    0.0858725564025334,
    -0.0122201511044634,
    -0.00407959697139479,
    0.00238702791287164,
    3.99744277180342e-05,
    -0.000610781628625445,
    0.000327618942421801,
    -6.40824868670675e-05,
    1.21669061346608,
    0.289888567583572,
    0.0711581420994488,
    -0.00507535645846862,
    -0.00333014317433887,
    0.000538258828628294,
    0.000665877776956594,
    -0.000366504466632854,
    -4.58025954888364e-05,
    0.000130223821210906,
    0.467719207492528,
    -0.496755967004945,
    0.0318951452197283,
    -0.00319405933867979,
    0.000378817692004189,
    -4.90059108703262e-05,
    6.68882206374488e-06,
    -9.46845206413818e-07,
    1.37551493080423e-07,
    -1.99391020361261e-08,
    0.39713599018272,
    -0.467511720087748,
    0.090898855310456,
    -0.0277149805374783,
    0.00996413458793538,
    -0.00390464645944911,
    0.00161220679652104,
    -0.000686314291188118,
    0.000291492761933432,
    -0.000108109237944853,
    0.423902395218573,
    -0.482402109471717,
    0.0713555166369812,
    -0.0162127148240246,
    0.00431669915734149,
    -0.00124997765166743,
    0.000381284118282679,
    -0.000120378689085621,
    3.86007377873893e-05,
    -1.14599548166949e-05,
];

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    // The deviance regime boundaries live with the deviance itself.
    use super::*;
    use crate::glm::deviance::{
        GAMMA_REGIME as DEVIANCE_GAMMA_MU_PHI, POISSON_REGIME as DEVIANCE_POISSON_PHI,
    };
    use approx::assert_relative_eq;

    /// edgePython `pois_alpha` and `pois_kappa` as `(mu, alpha, kappa)`.
    ///
    /// Produced by `uv run --with numpy --with numba --with scipy --with pandas
    /// --with statsmodels python -c "from edgepython.ql_weights import
    /// pois_alpha, pois_kappa; ..."` over a grid straddling every panel
    /// boundary of both fits.
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

    /// Closed form of the negative binomial unit deviance, with no attempt at
    /// numerical care. The authority away from the branch crossovers.
    fn exact_deviance(y: f64, mu: f64, phi: f64) -> f64 {
        let inv_phi = 1.0 / phi;
        let first = if y == 0.0 { 0.0 } else { y * (y / mu).ln() };
        2.0 * (first - (y + inv_phi) * ((y + inv_phi) / (mu + inv_phi)).ln())
    }

    /// Worst relative error over a table, treating an expected zero as absolute.
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

    // -- Chebyshev evaluation --

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

    // -- coefficient table integrity --

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
        // Spot check of the first entry of every table against ql_weights.c,
        // which is what a mistranscribed sign or exponent would show up in.
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

    // -- Poisson --

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
        // Panels are fitted independently, so the joins are only as continuous
        // as the fits are accurate. The bound is the worst jump edgePython
        // itself shows, 1.0e-3 for alpha at mu = 0.02; a misplaced boundary or
        // a coefficient block read at the wrong offset moves it by order one.
        for knot in [0.02, 0.4249, 0.4966, 1.5, 3.544, 4.2714, 20.0] {
            let below = knot * (1.0 - 1e-9);
            assert_relative_eq!(pois_alpha(below), pois_alpha(knot), max_relative = 2e-3);
            assert_relative_eq!(pois_kappa(below), pois_kappa(knot), max_relative = 2e-3);
        }
    }

    // -- negative binomial, case 1 --

    /// edgePython: `anbinomdevc_1(mu, phi), knbinomdevc_1(mu, phi)`.
    #[test]
    fn test_case1_matches_edgepython() {
        let mut errors = Vec::new();
        for (mu, phi, alpha, kappa) in NB_CASE1_REF {
            errors.push((nb_alpha(mu, phi), alpha));
            errors.push((nb_kappa(mu, phi), kappa));
        }
        let worst = worst_relative(&errors);
        // Bit exact against edgePython over the whole grid at the time of
        // writing; the bound leaves room for a differing `log` on another
        // platform.
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

    // -- negative binomial, case 2 --

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
        // The case 2 fits are visibly rougher than case 1: edgePython jumps by
        // 4.8e-2 for alpha at mu = 0.01, where the log factor is switched off,
        // and by 2.5e-2 at the mu = 3.88 knot with phi at the top of the range.
        // The bound is that behaviour, not this port's.
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

    // -- dispatch --

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
    /// `anbinomdevc_1`, which carries [`pois_alpha`] as an internal factor.
    /// An earlier version of this module short circuited to [`pois_alpha`]
    /// directly, which cost 1.8e-4 relative on the adjusted deviance at
    /// `dispersion = 0` measured against `glmQLFit(dispersion = 0)`.
    ///
    /// The two are close but not equal, so asserting equality with the Poisson
    /// fit is what pinned the bug in place. Assert the dispatch instead.
    #[test]
    fn test_zero_dispersion_takes_the_case_one_fit() {
        for mu in [0.005, 0.1, 1.0, 7.0, 15.0, 30.0, 100.0, 500.0] {
            assert_eq!(nb_alpha(mu, 0.0), nb_alpha_case1(mu, 0.0));
            assert_eq!(nb_kappa(mu, 0.0), nb_kappa_case1(mu, 0.0));

            // Continuous into the interior of case 1.
            assert_relative_eq!(nb_alpha(mu, 0.0), nb_alpha(mu, 1e-12), max_relative = 1e-9);
            assert_relative_eq!(nb_kappa(mu, 0.0), nb_kappa(mu, 1e-12), max_relative = 1e-9);

            // Close to the Poisson fit, since that is a factor of it, but the
            // two are not the same number.
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
        // alpha is a reciprocal scale and kappa a degrees of freedom, so both
        // are positive and kappa = alpha * E[d] must hold by construction.
        for phi in [4.001, 10.0, 100.0] {
            for mu in [0.1, 2.0, 50.0] {
                let (alpha, kappa) = compute_weight(mu, phi, 1.0);
                assert!(alpha > 0.0, "alpha {alpha} at mu {mu} phi {phi}");
                assert!(kappa > 0.0, "kappa {kappa} at mu {mu} phi {phi}");
            }
        }
    }

    // -- unit deviance --

    /// edgeR 4.8.2 via `nbinomUnitDeviance`.
    #[test]
    fn test_unit_deviance_matches_edger() {
        let errors: Vec<(f64, f64)> = DEVIANCE_REF
            .iter()
            .map(|(y, mu, phi, expected)| (unit_nb_deviance(*y, *mu, *phi), *expected))
            .collect();
        let worst = worst_relative(&errors);
        // Worst observed 8.6e-11, on `y = 1e6, mu = 1e6 + 1, phi = 1e-6`, where
        // the series differences two quantities of order 1e6 to produce one of
        // order 1e-6. Every other case in the table agrees to 5e-13 or better.
        assert!(worst < 1e-9, "worst relative error {worst:e}");
    }

    #[test]
    fn test_unit_deviance_matches_the_closed_form() {
        // Away from both crossovers the exact expression is well conditioned,
        // so it is the authority rather than a reference dump.
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
        // phi = 1e-4 switches from the series to the exact form. The series is
        // truncated after the cubic term, so the step across the boundary is
        // the first term it drops: order phi^2 (y - mu)^3, which is 9e-7
        // relative here. Both sides still agree with the closed form, which is
        // well conditioned at this dispersion.
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
        // most of its digits by there, so the two branches are compared to each
        // other and the gamma side is checked against its own limit.
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
        // This is the point of the small phi branch. At phi = 1e-12 the exact
        // form forms `y + 1/phi` and a log of two numbers that agree to twelve
        // digits, so it loses about twelve of them; the series never forms
        // either. The phi -> 0 limit is the Poisson deviance, which is well
        // conditioned and therefore the authority.
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
