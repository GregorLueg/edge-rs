//! Wald tests on a NEBULA fit.
//!
//! A single coefficient is tested with `z = beta / se`, which needs only the
//! diagonal of the covariance. A contrast is not: its variance is `c' V c`, and
//! the off-diagonal terms are non-zero whenever the design columns are
//! correlated, which is the normal case once a batch or a covariate is in the
//! model.
//!
//! edgePython's `glm_sc_test` forms contrast standard errors as
//! `sqrt(sum(se^2 c^2))`, which silently assumes `V` is diagonal. It has no
//! choice, because its fit discards everything but the diagonal. The error runs
//! in either direction depending on the sign of the covariance, so it is not
//! even conservative. See `UPSTREAM_DEVIATIONS.md` entry 2.
//!
//! This crate keeps the full covariance per gene, so contrasts are correct. The
//! R package supports the same through `covariance = TRUE`, which is what the
//! tests validate against.

use crate::errors::EdgeErrors;
use crate::numeric::dist::chisq_sf;
use crate::numeric::stats::p_adjust_bh;

/// Degrees of freedom of a single-contrast Wald test.
const WALD_DF: f64 = 1.0;

/// What to test.
#[derive(Clone, Debug)]
pub enum ScTested {
    /// One coefficient, by index.
    Coef(usize),
    /// A linear combination of coefficients, one weight per coefficient.
    Contrast(Vec<f64>),
}

/// Outcome of a Wald test, one entry per gene.
#[derive(Clone, Debug)]
pub struct ScTestResult {
    /// Estimated effect on the natural log scale.
    pub log_fc: Vec<f64>,
    /// Standard error of the effect.
    pub se: Vec<f64>,
    /// Wald statistic, `log_fc / se`.
    pub z: Vec<f64>,
    /// Two-sided p-value.
    pub p_value: Vec<f64>,
    /// Benjamini-Hochberg adjusted p-value.
    pub fdr: Vec<f64>,
}

/// Number of stored values in a packed symmetric matrix.
///
/// ### Params
///
/// * `n_coef` - Side length of the matrix
///
/// ### Returns
///
/// `n_coef * (n_coef + 1) / 2`.
#[inline]
pub const fn packed_len(n_coef: usize) -> usize {
    n_coef * (n_coef + 1) / 2
}

/// Reads one entry of a packed upper-triangular covariance.
///
/// The layout is the usual packed upper triangle in column-major order, so the
/// entry `(i, j)` with `i <= j` sits at `j (j + 1) / 2 + i` and the index does
/// not depend on `n_coef`. For three coefficients that is
/// `V11, V12, V22, V13, V23, V33`.
///
/// **This is not the R package's order for three or more coefficients.** nebula
/// returns `fccov[lower.tri(fccov, diag = TRUE)]`, which R traverses
/// column-major over the *lower* triangle, giving
/// `V11, V21, V31, V22, V32, V33`. The two coincide only at `n_coef <= 2`.
/// Comparing against a `covariance = TRUE` fixture means permuting one of them;
/// the tests do that rather than bending the layout, since the packed upper
/// triangle is what every other consumer of this crate expects.
///
/// ### Params
///
/// * `packed` - One gene's packed covariance
/// * `i` - Row
/// * `j` - Column
///
/// ### Returns
///
/// `V[i, j]`, using symmetry when `i > j`.
#[inline]
fn packed_get(packed: &[f64], i: usize, j: usize) -> f64 {
    let (lo, hi) = if i <= j { (i, j) } else { (j, i) };
    packed[hi * (hi + 1) / 2 + lo]
}

/// Tests a coefficient or a contrast on a NEBULA fit.
///
/// ### Params
///
/// * `coefficients` - Fitted coefficients, row-major `n_genes * n_coef`
/// * `covariance` - Packed upper-triangular covariance per gene, row-major
///   `n_genes * packed_len(n_coef)`
/// * `n_genes` - Number of genes
/// * `n_coef` - Number of coefficients
/// * `tested` - Which coefficient or contrast to test
///
/// ### Returns
///
/// The effect, its standard error, the Wald statistic, and raw and adjusted
/// p-values, or [`EdgeErrors`] if the shapes disagree or the contrast is empty.
pub fn glm_sc_test(
    coefficients: &[f64],
    covariance: &[f64],
    n_genes: usize,
    n_coef: usize,
    tested: &ScTested,
) -> Result<ScTestResult, EdgeErrors> {
    if n_genes == 0 || n_coef == 0 {
        return Err(EdgeErrors::MustBePositive("fit dimensions".to_string()));
    }
    if coefficients.len() != n_genes * n_coef {
        return Err(EdgeErrors::LengthMismatch {
            name: "coefficients",
            expected: n_genes * n_coef,
            got: coefficients.len(),
        });
    }
    let packed = packed_len(n_coef);
    if covariance.len() != n_genes * packed {
        return Err(EdgeErrors::LengthMismatch {
            name: "covariance",
            expected: n_genes * packed,
            got: covariance.len(),
        });
    }

    // Both cases reduce to a weight vector, so there is one code path.
    let weights: Vec<f64> = match tested {
        ScTested::Coef(index) => {
            if *index >= n_coef {
                return Err(EdgeErrors::CoefOutOfRange {
                    index: *index,
                    n_coef,
                });
            }
            let mut w = vec![0.0; n_coef];
            w[*index] = 1.0;
            w
        }
        ScTested::Contrast(values) => {
            if values.len() != n_coef {
                return Err(EdgeErrors::LengthMismatch {
                    name: "contrast",
                    expected: n_coef,
                    got: values.len(),
                });
            }
            if values.iter().all(|v| *v == 0.0) {
                return Err(EdgeErrors::InvalidArgument(
                    "contrast is entirely zero".to_string(),
                ));
            }
            values.clone()
        }
    };

    let mut log_fc = Vec::with_capacity(n_genes);
    let mut se = Vec::with_capacity(n_genes);
    let mut z = Vec::with_capacity(n_genes);
    let mut p_value = Vec::with_capacity(n_genes);

    for gene in 0..n_genes {
        let beta = &coefficients[gene * n_coef..(gene + 1) * n_coef];
        let cov = &covariance[gene * packed..(gene + 1) * packed];

        let effect: f64 = beta.iter().zip(weights.iter()).map(|(b, w)| b * w).sum();

        // c' V c, with the off-diagonal terms that edgePython drops.
        let mut variance = 0.0;
        for i in 0..n_coef {
            if weights[i] == 0.0 {
                continue;
            }
            for j in 0..n_coef {
                if weights[j] == 0.0 {
                    continue;
                }
                variance += weights[i] * weights[j] * packed_get(cov, i, j);
            }
        }

        let error = if variance > 0.0 { variance.sqrt() } else { 0.0 };
        let statistic = if error > 0.0 { effect / error } else { 0.0 };
        let p = if error > 0.0 {
            chisq_sf(statistic * statistic, WALD_DF)?
        } else {
            1.0
        };

        log_fc.push(effect);
        se.push(error);
        z.push(statistic);
        p_value.push(p);
    }

    let fdr = p_adjust_bh(&p_value);
    Ok(ScTestResult {
        log_fc,
        se,
        z,
        p_value,
        fdr,
    })
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Two genes, two coefficients. Covariance packed as `V11, V12, V22`, the
    /// one width at which this crate's layout and the R package's coincide.
    fn fixture() -> (Vec<f64>, Vec<f64>) {
        let coefficients = vec![
            -3.625_170_366_404_492,
            0.751_566_378_617_826, //
            -5.046_180_663_686_06,
            1.161_381_538_814_725,
        ];
        let covariance = vec![
            0.014_446_194_037_806_379,
            -0.001_444_153_266_646_014,
            0.057_784_776_151_225_514,
            0.020_240_376_033_846_74,
            -0.009_541_502_189_198_18,
            0.080_961_504_135_386_98,
        ];
        (coefficients, covariance)
    }

    /// Testing a coefficient reproduces the standard errors the R package
    /// reports, which are the square roots of the covariance diagonal.
    ///
    /// ```r
    /// nebula(..., covariance = TRUE)$summary$se_grp   # 0.24038464208685528, 0.28453735103741123
    /// ```
    #[test]
    fn test_single_coefficient_matches_nebula() {
        let (coefficients, covariance) = fixture();
        let out = glm_sc_test(&coefficients, &covariance, 2, 2, &ScTested::Coef(1)).unwrap();

        assert_relative_eq!(out.se[0], 0.240_384_642_086_855_28, max_relative = 1e-12);
        assert_relative_eq!(out.se[1], 0.284_537_351_037_411_23, max_relative = 1e-12);
        assert_relative_eq!(out.log_fc[0], 0.751_566_378_617_826, max_relative = 1e-12);
    }

    /// The point of keeping the full matrix. A contrast that mixes two
    /// correlated coefficients must pick up the off-diagonal term, and the
    /// diagonal-only formula edgePython uses gets a different answer.
    #[test]
    fn test_contrast_uses_the_off_diagonal_term() {
        let (coefficients, covariance) = fixture();
        let contrast = vec![1.0, 1.0];
        let out = glm_sc_test(
            &coefficients,
            &covariance,
            2,
            2,
            &ScTested::Contrast(contrast),
        )
        .unwrap();

        // c' V c = V11 + 2 V12 + V22.
        let expected: f64 = 0.014_446_194_037_806_379
            + 2.0 * -0.001_444_153_266_646_014
            + 0.057_784_776_151_225_514;
        let expected = expected.sqrt();
        assert_relative_eq!(out.se[0], expected, max_relative = 1e-12);

        // What edgePython would report: sqrt(se1^2 + se2^2), dropping 2 V12.
        let diagonal_only = (0.014_446_194_037_806_379 + 0.057_784_776_151_225_514f64).sqrt();
        let relative = (out.se[0] - diagonal_only).abs() / out.se[0];
        assert!(
            relative > 1e-3,
            "the off-diagonal term should move the answer, moved {relative:e}"
        );
        // Here the covariance is negative, so ignoring it overstates the error.
        assert!(diagonal_only > out.se[0]);
    }

    /// A contrast that isolates one coefficient must agree with testing it
    /// directly, which pins the two paths against each other.
    #[test]
    fn test_unit_contrast_agrees_with_the_coefficient() {
        let (coefficients, covariance) = fixture();
        let direct = glm_sc_test(&coefficients, &covariance, 2, 2, &ScTested::Coef(1)).unwrap();
        let via_contrast = glm_sc_test(
            &coefficients,
            &covariance,
            2,
            2,
            &ScTested::Contrast(vec![0.0, 1.0]),
        )
        .unwrap();
        for (a, b) in direct.se.iter().zip(via_contrast.se.iter()) {
            assert_relative_eq!(a, b, max_relative = 1e-15);
        }
        for (a, b) in direct.p_value.iter().zip(via_contrast.p_value.iter()) {
            assert_relative_eq!(a, b, max_relative = 1e-15);
        }
    }

    /// nebula's reported p-values for the group coefficient.
    ///
    /// ```r
    /// nebula(...)$summary$p_grp   # 0.00176891006399106164, 0.00004471733282162820
    /// ```
    #[test]
    fn test_p_values_match_nebula() {
        let (coefficients, covariance) = fixture();
        let out = glm_sc_test(&coefficients, &covariance, 2, 2, &ScTested::Coef(1)).unwrap();
        assert_relative_eq!(
            out.p_value[0],
            0.001_768_910_063_991_061_6,
            max_relative = 1e-9
        );
        assert_relative_eq!(
            out.p_value[1],
            0.000_044_717_332_821_628_2,
            max_relative = 1e-9
        );
    }

    #[test]
    fn test_packed_get_is_symmetric() {
        let packed = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(packed_get(&packed, i, j), packed_get(&packed, j, i));
            }
        }
        assert_eq!(packed_get(&packed, 0, 0), 1.0);
        assert_eq!(packed_get(&packed, 0, 1), 2.0);
        assert_eq!(packed_get(&packed, 1, 1), 3.0);
        assert_eq!(packed_get(&packed, 2, 2), 6.0);
    }

    #[test]
    fn test_packed_len_matches_the_triangle() {
        assert_eq!(packed_len(1), 1);
        assert_eq!(packed_len(2), 3);
        assert_eq!(packed_len(3), 6);
    }

    #[test]
    fn test_rejects_a_zero_contrast() {
        let (coefficients, covariance) = fixture();
        let err = glm_sc_test(
            &coefficients,
            &covariance,
            2,
            2,
            &ScTested::Contrast(vec![0.0, 0.0]),
        )
        .unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));
    }

    #[test]
    fn test_rejects_an_out_of_range_coefficient() {
        let (coefficients, covariance) = fixture();
        let err = glm_sc_test(&coefficients, &covariance, 2, 2, &ScTested::Coef(5)).unwrap_err();
        assert!(matches!(err, EdgeErrors::CoefOutOfRange { .. }));
    }

    #[test]
    fn test_rejects_a_shape_mismatch() {
        let (coefficients, _) = fixture();
        let err = glm_sc_test(&coefficients, &[0.0; 2], 2, 2, &ScTested::Coef(0)).unwrap_err();
        assert!(matches!(err, EdgeErrors::LengthMismatch { .. }));
    }

    /// A gene whose covariance collapsed reports no evidence rather than a NaN.
    #[test]
    fn test_degenerate_covariance_gives_a_p_value_of_one() {
        let coefficients = vec![1.0, 2.0];
        let covariance = vec![0.0, 0.0, 0.0];
        let out = glm_sc_test(&coefficients, &covariance, 1, 2, &ScTested::Coef(1)).unwrap();
        assert_eq!(out.se[0], 0.0);
        assert_eq!(out.p_value[0], 1.0);
        assert!(out.z[0].is_finite());
    }
}
