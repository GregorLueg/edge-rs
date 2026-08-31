//! Negative binomial deviance.
//!
//! This is the crate's single implementation, used by the Levenberg fit, the
//! residual deviance and the quasi-likelihood weights alike.
//!
//! It follows edgeR's `compute_nbdev.c` rather than edgePython's
//! `glm_levenberg._unit_nb_deviance`. edgePython carries two versions and only
//! the one in `ql_weights.py` matches edgeR; the naive one its GLM path uses
//! disagrees with edgeR by around 1e-8 relative on ordinary values and
//! completely near `y == mu`.
//!
//! Two details carry that accuracy. Both `y` and `mu` are nudged by
//! [`MILDLY_LOW_VALUE`] before anything else, which is what keeps the
//! zero-count case finite without a special branch. And the formula switches by
//! regime, so the difference of large logarithms is never evaluated where it
//! would cancel.

////////////
// Consts //
////////////

/// Nudge added to both `y` and `mu` before any logarithm is taken.
///
/// edgeR's `mildly_low_value`. It removes the `y == 0` and `mu == 0` special
/// cases at the cost of a bias around 1e-8 relative, and reproducing it is
/// required for parity: edgeR's published deviances carry this bias.
pub const MILDLY_LOW_VALUE: f64 = 1e-8;

/// Dispersion below which the Poisson expansion is used.
///
/// Under this, `1/phi` is large enough that the negative binomial form loses
/// precision, and the Poisson limit plus a first-order correction in `phi` is
/// both cheaper and more accurate.
pub const POISSON_REGIME: f64 = 1e-4;

/// Value of `mu * phi` above which the gamma limit is used.
///
/// The negative binomial tends to a gamma as `mu * phi` grows, and past this the
/// `log((mu + 1/phi) / (y + 1/phi))` term is all cancellation.
pub const GAMMA_REGIME: f64 = 1e6;

/// Unit deviance of one observation under a negative binomial model.
///
/// Three regimes, selected exactly as edgeR does: a Poisson expansion for small
/// dispersion, a gamma limit for large `mu * phi`, and otherwise the rearranged
/// exact form, which is algebraically identical to the textbook expression but
/// groups the logarithms so they do not cancel.
///
/// ### Params
///
/// * `y` - Observed count, non-negative
/// * `mu` - Fitted mean, non-negative
/// * `phi` - Dispersion, non-negative. Zero gives the Poisson deviance.
///
/// ### Returns
///
/// The unit deviance, clamped at zero. A negative value can only arise from
/// rounding, since the deviance is non-negative by construction.
///
/// ### References
///
/// McCarthy, Chen and Smyth, Nucleic Acids Research 40(10), 2012
#[inline]
pub fn unit_nb_deviance(y: f64, mu: f64, phi: f64) -> f64 {
    let y = y + MILDLY_LOW_VALUE;
    let mu = mu + MILDLY_LOW_VALUE;

    let out = if phi < POISSON_REGIME {
        // Poisson limit with the leading correction in phi.
        //
        // The cubic term is `-phi * y`, not `phi * (2/3 * resid - y)`.
        // edgeR's C writes `2/3` with integer operands, so it truncates to zero
        // and the `resid` contribution never enters. edgePython treats that as
        // a typo and writes `2.0/3.0`, which changes the answer by up to 7e-4
        // relative on large counts. Reproducing the C is what matches published
        // edgeR results.
        let resid = y - mu;
        2.0 * (y * (y / mu).ln() - resid - 0.5 * resid * resid * phi * (1.0 - phi * y))
    } else {
        let product = mu * phi;
        if product > GAMMA_REGIME {
            // Gamma limit.
            2.0 * ((y - mu) / mu - (y / mu).ln()) * mu / (1.0 + product)
        } else {
            let inv_phi = 1.0 / phi;
            2.0 * (y * (y / mu).ln() + (y + inv_phi) * ((mu + inv_phi) / (y + inv_phi)).ln())
        }
    };

    out.max(0.0)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Reference values from the installed edgeR 4.8.2:
    /// `nbinomUnitDeviance(c(10,10.001,10.1,12,0,3,100,1), c(10,10,10,10,10,10,90,0.5), 0.1)`
    #[test]
    fn test_unit_deviance_matches_edger() {
        let cases: [(f64, f64, f64); 8] = [
            (10.0, 10.0, 0.0),
            (10.001, 10.0, 4.999_749_966_85e-08),
            (10.1, 10.0, 0.000_497_514_489_479),
            (12.0, 10.0, 0.182_069_451_405),
            (0.0, 10.0, 13.862_943_200_6),
            (3.0, 10.0, 3.976_518_983_98),
            (100.0, 90.0, 0.103_863_574_593),
            (1.0, 0.5, 0.362_854_011_037),
        ];
        for (y, mu, expected) in cases {
            let got = unit_nb_deviance(y, mu, 0.1);
            if expected == 0.0 {
                assert!(got.abs() < 1e-14, "y={y} mu={mu} gave {got}");
            } else {
                assert_relative_eq!(got, expected, max_relative = 1e-9);
            }
        }
    }

    /// The naive textbook formula, which edgePython's GLM path uses. It must
    /// disagree with ours, otherwise the regime switching is not doing anything
    /// and this module has silently reverted to the Python behaviour.
    fn naive(y: f64, mu: f64, phi: f64) -> f64 {
        if y > 0.0 {
            2.0 * (y * (y / mu).ln() - (y + 1.0 / phi) * ((1.0 + phi * y) / (1.0 + phi * mu)).ln())
        } else {
            2.0 / phi * (1.0 + phi * mu).ln()
        }
    }

    #[test]
    fn test_regime_switch_beats_the_naive_form_near_the_mean() {
        // Right next to y == mu the naive form is all cancellation.
        let (y, mu, phi) = (10.000_001, 10.0, 0.1);
        let ours = unit_nb_deviance(y, mu, phi);
        let theirs = naive(y, mu, phi);
        let relative = (ours - theirs).abs() / ours.abs();
        assert!(
            relative > 1e-6,
            "expected a visible disagreement with the naive form, got {relative:e}"
        );
    }

    #[test]
    fn test_zero_count_is_finite_without_a_special_case() {
        let d = unit_nb_deviance(0.0, 10.0, 0.1);
        assert_relative_eq!(d, 13.862_943_200_6, max_relative = 1e-9);
        assert!(d.is_finite());
    }

    /// The Poisson-regime correction term, checked against edgeR where it
    /// actually bites: large counts and a dispersion just under the 1e-4
    /// threshold. This is the case that separates edgeR's integer-truncated
    /// `2/3` from edgePython's `2.0/3.0`.
    ///
    /// ```r
    /// nbinomUnitDeviance(c(1e6, 1e6, 100, 5, 0),
    ///                    c(1.001e6, 1e6 + 1, 110, 7, 10),
    ///                    c(1e-5, 1e-6, 1e-5, 1e-6, 1e-5))
    /// ```
    #[test]
    fn test_poisson_correction_matches_edger() {
        let cases: [(f64, f64, f64, f64); 5] = [
            (1e6, 1.001e6, 1e-5, 90.999_333_833),
            (1e6, 1e6 + 1.0, 1e-6, 9.998_995_586_21e-7),
            (100.0, 110.0, 1e-5, 0.936_965_039_047),
            (5.0, 7.0, 1e-6, 0.635_273_632_793),
            (0.0, 10.0, 1e-5, 19.998_999_585_5),
        ];
        for (y, mu, phi, expected) in cases {
            assert_relative_eq!(unit_nb_deviance(y, mu, phi), expected, max_relative = 1e-10);
        }
    }

    /// edgeR: `nbinomUnitDeviance(c(0,1,3,5,100), c(5,5,5,5,100), 1e-8)`
    /// exercises the Poisson regime, since the dispersion is below 1e-4.
    #[test]
    fn test_poisson_regime_matches_the_poisson_deviance() {
        let phi = 1e-10;
        for (y, mu) in [
            (0.0, 5.0),
            (1.0, 5.0),
            (3.0, 5.0),
            (5.0, 5.0),
            (100.0, 90.0),
        ] {
            let ours = unit_nb_deviance(y, mu, phi);
            let yn = y + MILDLY_LOW_VALUE;
            let mn = mu + MILDLY_LOW_VALUE;
            let poisson = 2.0 * (yn * (yn / mn).ln() - (yn - mn));
            assert_relative_eq!(ours, poisson, max_relative = 1e-8);
        }
    }

    /// Beyond `mu * phi > 1e6` the gamma limit takes over.
    ///
    /// The switch is genuinely discontinuous, by around 0.2%: the gamma form is
    /// a limit, not an identity, and edgeR accepts that jump. The assertion is
    /// therefore that the two branches agree to within a few tenths of a
    /// percent, which is loose enough to allow the real discontinuity and tight
    /// enough to catch a wrong formula on either side.
    #[test]
    fn test_gamma_regime_agrees_across_the_boundary() {
        let mu = 1e7;
        let below = unit_nb_deviance(1.2e7, mu, GAMMA_REGIME / mu * 0.999);
        let above = unit_nb_deviance(1.2e7, mu, GAMMA_REGIME / mu * 1.001);
        assert_relative_eq!(below, above, max_relative = 5e-3);
    }

    #[test]
    fn test_deviance_is_never_negative() {
        for y in [0.0, 0.5, 1.0, 10.0, 1e6] {
            for mu in [1e-8, 0.5, 10.0, 1e6] {
                for phi in [0.0, 1e-6, 0.01, 0.5, 10.0] {
                    let d = unit_nb_deviance(y, mu, phi);
                    assert!(d >= 0.0, "y={y} mu={mu} phi={phi} gave {d}");
                    assert!(d.is_finite(), "y={y} mu={mu} phi={phi} gave {d}");
                }
            }
        }
    }
}
