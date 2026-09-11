//! limma's `removeBatchEffect`: regress batch out of a log-expression matrix.
//!
//! Batch factors get sum-to-zero contrasts, so the corrected values stay centred
//! on the grand mean rather than on the first batch. The batch columns are fitted
//! jointly with the design of interest through [`lm_fit`], and only the batch
//! part of the fit is subtracted: the design protects the biology from being
//! absorbed into the batch coefficients.
//!
//! Meant for plotting and unsupervised work. For testing, put batch into the
//! design instead, as limma's own documentation says.
//!
//! ### References
//!
//! Smyth, Statistical Applications in Genetics and Molecular Biology 3(1), 2004

use rayon::prelude::*;

use crate::limma::lm_fit::lm_fit;
use crate::prelude::*;

/////////////
// Helpers //
/////////////

/// Sum-to-zero encoding of one batch factor, R's `contr.sum` on
/// `factor(labels)`.
///
/// Levels are the distinct labels in ascending order, which is how R orders the
/// levels of a numeric vector. The last level carries `-1` in every column.
///
/// ### Params
///
/// * `labels` - Batch label per sample
///
/// ### Returns
///
/// Row-major `n_samples * (n_levels - 1)` columns and their count, or
/// [`EdgeErrors::InvalidArgument`] when fewer than two levels are present, the
/// same case R's `contrasts<-` refuses.
fn sum_contrast_columns(labels: &[usize]) -> Result<(Vec<f64>, usize), EdgeErrors> {
    let mut levels = labels.to_vec();
    levels.sort_unstable();
    levels.dedup();
    if levels.len() < 2 {
        return Err(EdgeErrors::InvalidArgument(
            "a batch factor needs at least two levels".to_string(),
        ));
    }

    let n_cols = levels.len() - 1;
    let mut out = vec![0.0; labels.len() * n_cols];
    for (row, label) in out.chunks_exact_mut(n_cols).zip(labels) {
        // binary_search cannot fail: every label is one of the levels.
        let lvl = levels.binary_search(label).expect("label is a level");
        if lvl == n_cols {
            row.fill(-1.0);
        } else {
            row[lvl] = 1.0;
        }
    }
    Ok((out, n_cols))
}

/// Appends row-major columns onto a row-major matrix, `cbind` for this layout.
///
/// ### Params
///
/// * `left` - Row-major `n_rows * n_left`
/// * `n_left` - Columns in `left`
/// * `right` - Row-major `n_rows * n_right`
/// * `n_right` - Columns in `right`
/// * `n_rows` - Number of rows
///
/// ### Returns
///
/// Row-major `n_rows * (n_left + n_right)`.
fn cbind(left: &[f64], n_left: usize, right: &[f64], n_right: usize, n_rows: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(n_rows * (n_left + n_right));
    for i in 0..n_rows {
        out.extend_from_slice(&left[i * n_left..(i + 1) * n_left]);
        out.extend_from_slice(&right[i * n_right..(i + 1) * n_right]);
    }
    out
}

/// Checks a per-sample vector against the sample count.
///
/// ### Params
///
/// * `name` - Argument name for the error
/// * `len` - Supplied length
/// * `n_samples` - Required length
///
/// ### Returns
///
/// `Ok(())` or [`EdgeErrors::LengthMismatch`].
fn check_len(name: &'static str, len: usize, n_samples: usize) -> Result<(), EdgeErrors> {
    if len != n_samples {
        return Err(EdgeErrors::LengthMismatch {
            name,
            expected: n_samples,
            got: len,
        });
    }
    Ok(())
}

///////////////////////
// removeBatchEffect //
///////////////////////

/// Removes batch effects and covariates from a log-expression matrix.
///
/// Port of limma's `removeBatchEffect` without `group`. Builds
/// `X_batch = cbind(contr.sum(batch), contr.sum(batch2), covariates)`, fits
/// `cbind(design, X_batch)` per gene with [`lm_fit`] and returns
/// `x - beta_batch %*% t(X_batch)`. Coefficients the fit cannot estimate count
/// as zero, as in limma. Covariates are column-centred first, also as in limma,
/// so the correction leaves the grand mean alone.
///
/// With no batch, no second batch and no covariates the input comes back
/// unchanged, again as in limma.
///
/// ### Params
///
/// * `x` - Log-expression, row-major `n_genes * n_samples`. Non-finite entries
///   are treated as missing by the fit and stay non-finite in the output.
/// * `n_genes` - Number of genes
/// * `n_samples` - Number of samples
/// * `batch` - Optional batch label per sample
/// * `batch2` - Optional second batch label per sample
/// * `covariates` - Optional numeric covariates, row-major `n_samples * n_cov`,
///   with `n_cov`
/// * `design` - Optional design of interest, row-major `n_samples * n_coef`, with
///   `n_coef`. `None` is an intercept only.
/// * `weights` - Optional observation weights, forwarded to [`lm_fit`] as limma's
///   `...` does
///
/// ### Returns
///
/// The corrected matrix, row-major `n_genes * n_samples`, or [`EdgeErrors`] if a
/// shape disagrees or a batch factor has fewer than two levels.
///
/// ### References
///
/// Smyth, Statistical Applications in Genetics and Molecular Biology 3(1), 2004
#[allow(clippy::too_many_arguments)]
pub fn remove_batch_effect<T: EdgeFloat>(
    x: &[T],
    n_genes: usize,
    n_samples: usize,
    batch: Option<&[usize]>,
    batch2: Option<&[usize]>,
    covariates: Option<(&[f64], usize)>,
    design: Option<(&[f64], usize)>,
    weights: Option<&Recycled<f64>>,
) -> Result<Vec<f64>, EdgeErrors> {
    if x.len() != n_genes * n_samples {
        return Err(EdgeErrors::LengthMismatch {
            name: "x",
            expected: n_genes * n_samples,
            got: x.len(),
        });
    }
    let y: Vec<f64> = x.iter().map(|v| v.to_f64().unwrap_or(f64::NAN)).collect();

    let mut x_batch: Vec<f64> = vec![0.0; 0];
    let mut n_batch = 0;
    for (name, labels) in [("batch", batch), ("batch2", batch2)] {
        if let Some(labels) = labels {
            check_len(name, labels.len(), n_samples)?;
            let (cols, n_cols) = sum_contrast_columns(labels)?;
            x_batch = cbind(&x_batch, n_batch, &cols, n_cols, n_samples);
            n_batch += n_cols;
        }
    }
    if let Some((cov, n_cov)) = covariates {
        check_len("covariates", cov.len(), n_samples * n_cov)?;
        let mut centred = cov.to_vec();
        for j in 0..n_cov {
            let mean = (0..n_samples).map(|i| cov[i * n_cov + j]).sum::<f64>() / n_samples as f64;
            (0..n_samples).for_each(|i| centred[i * n_cov + j] -= mean);
        }
        x_batch = cbind(&x_batch, n_batch, &centred, n_cov, n_samples);
        n_batch += n_cov;
    }
    if n_batch == 0 {
        return Ok(y);
    }

    let intercept = vec![1.0; n_samples];
    let (design, n_coef) = design.unwrap_or((&intercept, 1));
    check_len("design", design.len(), n_samples * n_coef)?;

    let n_full = n_coef + n_batch;
    let full = cbind(design, n_coef, &x_batch, n_batch, n_samples);
    let fit = lm_fit(&y, n_genes, n_samples, &full, n_full, weights, None, None)?;

    let mut out = y;
    out.par_chunks_mut(n_samples)
        .zip(fit.coefficients.par_chunks(n_full))
        .for_each(|(row, coef)| {
            let beta = &coef[n_coef..];
            for (s, v) in row.iter_mut().enumerate() {
                let x_s = &x_batch[s * n_batch..(s + 1) * n_batch];
                // Aliased coefficients are NaN from the fit; limma zeroes them.
                let effect: f64 = beta
                    .iter()
                    .zip(x_s)
                    .map(|(b, xb)| if b.is_nan() { 0.0 } else { b * xb })
                    .sum();
                *v -= effect;
            }
        });
    Ok(out)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    // References are pasted verbatim from R's 17-digit output of
    // `limma::removeBatchEffect` (limma 3.66.0) on the fixture below, same
    // convention as `lm_fit`'s tests.
    #![allow(clippy::excessive_precision)]

    use super::*;
    use approx::assert_relative_eq;

    const N_GENES: usize = 8;
    const N_SAMPLES: usize = 6;

    /// Absolute tolerance. Every reference is a small dyadic combination, so
    /// anything past round-off is a real difference.
    const TOL: f64 = 1e-13;

    /// `y[i] = ((i * 7) mod 23) / 64`, row-major 8 by 6. Dyadic, so R and Rust
    /// see bit-identical inputs.
    fn fixture_y() -> Vec<f64> {
        (0..N_GENES * N_SAMPLES)
            .map(|i| ((i * 7) % 23) as f64 / 64.0)
            .collect()
    }

    const BATCH: [usize; 6] = [0, 0, 1, 1, 2, 2];
    const BATCH2: [usize; 6] = [0, 1, 0, 1, 1, 0];

    /// Intercept plus a group indicator that is not orthogonal to `BATCH`.
    const DESIGN: [f64; 12] = [1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0];

    /// `c(1, 3, 2, 5, 4, 7) / 8`.
    const COV: [f64; 6] = [0.125, 0.375, 0.25, 0.625, 0.5, 0.875];

    fn assert_matches(got: &[f64], expected: &[f64]) {
        assert_eq!(got.len(), expected.len());
        for (g, e) in got.iter().zip(expected) {
            assert_relative_eq!(*g, *e, epsilon = TOL);
        }
    }

    #[test]
    fn test_remove_batch_effect_batch_only() {
        // removeBatchEffect(y, batch = c(0, 0, 1, 1, 2, 2))
        let expected = [
            0.098958333333333343,
            0.208333333333333343,
            0.098958333333333287,
            0.208333333333333287,
            0.098958333333333370,
            0.208333333333333370,
            0.276041666666666630,
            0.026041666666666657,
            0.096354166666666657,
            0.205729166666666657,
            0.096354166666666685,
            0.205729166666666685,
            0.153645833333333287,
            0.263020833333333259,
            0.153645833333333370,
            0.263020833333333370,
            0.333333333333333370,
            0.083333333333333343,
            0.091145833333333287,
            0.200520833333333287,
            0.091145833333333370,
            0.200520833333333370,
            0.270833333333333315,
            0.020833333333333336,
            0.148437499999999972,
            0.257812500000000000,
            0.328125000000000000,
            0.078125000000000000,
            0.148437500000000028,
            0.257812500000000000,
            0.085937500000000000,
            0.195312500000000000,
            0.265625000000000000,
            0.015625000000000007,
            0.085937500000000000,
            0.195312500000000000,
            0.322916666666666630,
            0.072916666666666630,
            0.143229166666666685,
            0.252604166666666685,
            0.143229166666666685,
            0.252604166666666685,
            0.260416666666666685,
            0.010416666666666661,
            0.080729166666666657,
            0.190104166666666657,
            0.080729166666666685,
            0.190104166666666685,
        ];
        let got = remove_batch_effect(
            &fixture_y(),
            N_GENES,
            N_SAMPLES,
            Some(&BATCH),
            None,
            None,
            None,
            None,
        )
        .unwrap();
        assert_matches(&got, &expected);
    }

    #[test]
    fn test_remove_batch_effect_with_design() {
        // removeBatchEffect(y, batch, design = cbind(1, c(0, 1, 1, 1, 0, 0)))
        let expected = [
            0.098958333333333329,
            0.208333333333333315,
            0.153645833333333315,
            0.263020833333333315,
            0.044270833333333370,
            0.153645833333333370,
            0.276041666666666685,
            0.026041666666666689,
            -0.028645833333333481,
            0.080729166666666519,
            0.221354166666666796,
            0.330729166666666796,
            0.153645833333333315,
            0.263020833333333315,
            0.208333333333333343,
            0.317708333333333370,
            0.278645833333333370,
            0.028645833333333343,
            0.091145833333333315,
            0.200520833333333315,
            0.145833333333333343,
            0.255208333333333370,
            0.216145833333333343,
            -0.033854166666666657,
            0.148437499999999944,
            0.257812499999999944,
            0.382812500000000000,
            0.132812500000000000,
            0.093750000000000042,
            0.203125000000000056,
            0.085937499999999958,
            0.195312499999999944,
            0.320312500000000056,
            0.070312500000000056,
            0.031249999999999986,
            0.140625000000000000,
            0.322916666666666685,
            0.072916666666666671,
            0.018229166666666519,
            0.127604166666666519,
            0.268229166666666852,
            0.377604166666666852,
            0.260416666666666685,
            0.010416666666666678,
            -0.044270833333333426,
            0.065104166666666574,
            0.205729166666666741,
            0.315104166666666741,
        ];
        let got = remove_batch_effect(
            &fixture_y(),
            N_GENES,
            N_SAMPLES,
            Some(&BATCH),
            None,
            None,
            Some((&DESIGN, 2)),
            None,
        )
        .unwrap();
        assert_matches(&got, &expected);
    }

    #[test]
    fn test_remove_batch_effect_two_batches_and_design() {
        // removeBatchEffect(y, batch, batch2 = c(0, 1, 0, 1, 1, 0), design)
        let expected = [
            0.0989583333333333148,
            0.2083333333333333426,
            0.1536458333333333148,
            0.2630208333333333148,
            0.0442708333333333634,
            0.1536458333333333426,
            0.2760416666666666852,
            0.0260416666666666644,
            -0.0286458333333334814,
            0.0807291666666664631,
            0.2213541666666667962,
            0.3307291666666668517,
            0.2434895833333333148,
            0.1731770833333333148,
            0.2083333333333332871,
            0.1380208333333332871,
            0.2786458333333333703,
            0.2083333333333333981,
            0.1809895833333333148,
            0.1106770833333333148,
            0.1458333333333332871,
            0.0755208333333332593,
            0.2161458333333333981,
            0.1458333333333334259,
            0.0585937499999999514,
            0.3476562499999999445,
            0.3828125000000000555,
            0.3125000000000000555,
            0.0937500000000000000,
            0.0234375000000000000,
            -0.0039062500000000486,
            0.2851562500000000000,
            0.3203125000000001110,
            0.2500000000000001110,
            0.0312499999999999306,
            -0.0390625000000001110,
            0.3229166666666666852,
            0.0729166666666666713,
            0.0182291666666665186,
            0.1276041666666665186,
            0.2682291666666668517,
            0.3776041666666668517,
            0.2604166666666666852,
            0.0104166666666666748,
            -0.0442708333333334259,
            0.0651041666666665741,
            0.2057291666666667407,
            0.3151041666666667407,
        ];
        let got = remove_batch_effect(
            &fixture_y(),
            N_GENES,
            N_SAMPLES,
            Some(&BATCH),
            Some(&BATCH2),
            None,
            Some((&DESIGN, 2)),
            None,
        )
        .unwrap();
        assert_matches(&got, &expected);
    }

    #[test]
    fn test_remove_batch_effect_covariates_and_design() {
        // removeBatchEffect(y, batch, covariates = c(1, 3, 2, 5, 4, 7) / 8, design)
        let expected = [
            0.135416666666666657,
            0.171875000000000000,
            0.171874999999999972,
            0.171874999999999972,
            0.135416666666666657,
            0.135416666666666685,
            0.312500000000000056,
            -0.010416666666666650,
            -0.010416666666666824,
            -0.010416666666666852,
            0.312500000000000111,
            0.312500000000000111,
            0.130208333333333315,
            0.286458333333333315,
            0.196614583333333343,
            0.376302083333333370,
            0.220052083333333343,
            0.040364583333333343,
            0.067708333333333315,
            0.223958333333333343,
            0.134114583333333315,
            0.313802083333333370,
            0.157552083333333315,
            -0.022135416666666647,
            0.124999999999999944,
            0.281249999999999944,
            0.371093750000000000,
            0.191406250000000028,
            0.035156250000000028,
            0.214843750000000028,
            0.062499999999999958,
            0.218749999999999944,
            0.308593750000000056,
            0.128906250000000056,
            -0.027343750000000000,
            0.152343750000000000,
            0.359375000000000000,
            0.036458333333333329,
            0.036458333333333176,
            0.036458333333333148,
            0.359375000000000167,
            0.359375000000000167,
            0.296875000000000000,
            -0.026041666666666657,
            -0.026041666666666768,
            -0.026041666666666796,
            0.296875000000000111,
            0.296875000000000111,
        ];
        let got = remove_batch_effect(
            &fixture_y(),
            N_GENES,
            N_SAMPLES,
            Some(&BATCH),
            None,
            Some((&COV, 1)),
            Some((&DESIGN, 2)),
            None,
        )
        .unwrap();
        assert_matches(&got, &expected);
    }

    #[test]
    fn test_remove_batch_effect_aliased_batch_is_zeroed() {
        // Batch identical to the design group: the batch coefficient is not
        // estimable, limma zeroes it and hands back the input.
        let batch = [0, 0, 0, 1, 1, 1];
        let design = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let y = fixture_y();
        let got = remove_batch_effect(
            &y,
            N_GENES,
            N_SAMPLES,
            Some(&batch),
            None,
            None,
            Some((&design, 2)),
            None,
        )
        .unwrap();
        assert_matches(&got, &y);
    }

    #[test]
    fn test_remove_batch_effect_non_contiguous_labels() {
        // removeBatchEffect(y, batch = c(7, 7, 3, 3, 7, 3)): levels sort to 3, 7
        let expected = [
            0.091145833333333329,
            0.200520833333333315,
            0.127604166666666685,
            0.236979166666666685,
            0.169270833333333315,
            0.096354166666666671,
            0.328125000000000000,
            0.078125000000000000,
            0.125000000000000000,
            0.234375000000000000,
            0.046875000000000000,
            0.093750000000000000,
            0.145833333333333315,
            0.255208333333333315,
            0.182291666666666685,
            0.291666666666666685,
            0.223958333333333315,
            0.151041666666666685,
            0.083333333333333329,
            0.192708333333333315,
            0.119791666666666671,
            0.229166666666666685,
            0.161458333333333315,
            0.088541666666666671,
            0.140624999999999972,
            0.249999999999999972,
            0.296875000000000000,
            0.046875000000000028,
            0.218749999999999972,
            0.265625000000000000,
            0.078124999999999986,
            0.187500000000000000,
            0.234375000000000000,
            -0.015624999999999990,
            0.156250000000000000,
            0.203125000000000000,
            0.375000000000000000,
            0.124999999999999972,
            0.171875000000000028,
            0.281250000000000000,
            0.093749999999999972,
            0.140625000000000028,
            0.312500000000000000,
            0.062500000000000000,
            0.109375000000000000,
            0.218750000000000000,
            0.031250000000000000,
            0.078125000000000000,
        ];
        let batch = [7, 7, 3, 3, 7, 3];
        let got = remove_batch_effect(
            &fixture_y(),
            N_GENES,
            N_SAMPLES,
            Some(&batch),
            None,
            None,
            None,
            None,
        )
        .unwrap();
        assert_matches(&got, &expected);
    }

    #[test]
    fn test_remove_batch_effect_nothing_to_remove_returns_input() {
        let y = fixture_y();
        let got =
            remove_batch_effect(&y, N_GENES, N_SAMPLES, None, None, None, None, None).unwrap();
        assert_matches(&got, &y);
    }

    #[test]
    fn test_remove_batch_effect_f32_input() {
        let y32: Vec<f32> = fixture_y().iter().map(|v| *v as f32).collect();
        let got32 = remove_batch_effect(
            &y32,
            N_GENES,
            N_SAMPLES,
            Some(&BATCH),
            None,
            None,
            None,
            None,
        )
        .unwrap();
        let got64 = remove_batch_effect(
            &fixture_y(),
            N_GENES,
            N_SAMPLES,
            Some(&BATCH),
            None,
            None,
            None,
            None,
        )
        .unwrap();
        // Dyadic fixture, so the f32 cast is exact and the two agree.
        assert_matches(&got32, &got64);
    }

    #[test]
    fn test_remove_batch_effect_single_level_errors() {
        let batch = [1; 6];
        let res = remove_batch_effect(
            &fixture_y(),
            N_GENES,
            N_SAMPLES,
            Some(&batch),
            None,
            None,
            None,
            None,
        );
        assert!(matches!(res, Err(EdgeErrors::InvalidArgument(_))));
    }

    #[test]
    fn test_remove_batch_effect_length_mismatch_errors() {
        let batch = [0, 1, 0];
        let res = remove_batch_effect(
            &fixture_y(),
            N_GENES,
            N_SAMPLES,
            Some(&batch),
            None,
            None,
            None,
            None,
        );
        assert!(matches!(
            res,
            Err(EdgeErrors::LengthMismatch { name: "batch", .. })
        ));
    }

    #[test]
    fn test_remove_batch_effect_nan_input() {
        // y[1, 3:4] <- NA; removeBatchEffect(y, batch = c(0, 0, 1, 1, 2, 2))
        let expected = [
            0.039062500000000000,
            0.148437500000000000,
            f64::NAN,
            f64::NAN,
            0.039062500000000000,
            0.148437500000000000,
            0.276041666666666630,
            0.026041666666666657,
            0.096354166666666657,
            0.205729166666666657,
            0.096354166666666685,
            0.205729166666666685,
        ];
        let mut y = fixture_y();
        y[2] = f64::NAN;
        y[3] = f64::NAN;
        let got = remove_batch_effect(&y, N_GENES, N_SAMPLES, Some(&BATCH), None, None, None, None)
            .unwrap();
        for (g, e) in got[..12].iter().zip(&expected) {
            if e.is_nan() {
                assert!(g.is_nan());
            } else {
                assert_relative_eq!(*g, *e, epsilon = TOL);
            }
        }
    }

    #[test]
    fn test_remove_batch_effect_two_covariates() {
        // cv <- cbind(c(1, 3, 2, 5, 4, 7) / 8, c(4, 1, 3, 0, 2, 5) / 8)
        // removeBatchEffect(y, batch, covariates = cv, design)
        let expected = [
            0.135416666666666657,
            0.171875000000000000,
            0.171874999999999972,
            0.171874999999999972,
            0.135416666666666657,
            0.135416666666666685,
            0.312500000000000056,
            -0.010416666666666685,
            -0.010416666666666824,
            -0.010416666666666907,
            0.312500000000000111,
            0.312500000000000111,
            0.220052083333333287,
            0.196614583333333343,
            0.196614583333333315,
            0.196614583333333370,
            0.220052083333333426,
            0.220052083333333287,
            0.157552083333333287,
            0.134114583333333315,
            0.134114583333333315,
            0.134114583333333315,
            0.157552083333333398,
            0.157552083333333343,
            0.035156249999999944,
            0.371093749999999944,
            0.371093750000000056,
            0.371093750000000056,
            0.035156249999999972,
            0.035156250000000056,
            -0.027343750000000056,
            0.308593750000000000,
            0.308593750000000111,
            0.308593750000000111,
            -0.027343750000000083,
            -0.027343750000000056,
            0.359375000000000000,
            0.036458333333333322,
            0.036458333333333176,
            0.036458333333333148,
            0.359375000000000167,
            0.359375000000000167,
            0.296875000000000056,
            -0.026041666666666678,
            -0.026041666666666768,
            -0.026041666666666852,
            0.296875000000000111,
            0.296875000000000111,
        ];
        let cov = [
            0.125, 0.5, 0.375, 0.125, 0.25, 0.375, 0.625, 0.0, 0.5, 0.25, 0.875, 0.625,
        ];
        let got = remove_batch_effect(
            &fixture_y(),
            N_GENES,
            N_SAMPLES,
            Some(&BATCH),
            None,
            Some((&cov, 2)),
            Some((&DESIGN, 2)),
            None,
        )
        .unwrap();
        assert_matches(&got, &expected);
    }

    #[test]
    fn test_remove_batch_effect_covariates_and_design_length_errors() {
        let res = remove_batch_effect(
            &fixture_y(),
            N_GENES,
            N_SAMPLES,
            None,
            None,
            Some((&COV, 2)),
            None,
            None,
        );
        assert!(matches!(
            res,
            Err(EdgeErrors::LengthMismatch {
                name: "covariates",
                ..
            })
        ));

        // The design is only checked once there is something to remove.
        let res = remove_batch_effect(
            &fixture_y(),
            N_GENES,
            N_SAMPLES,
            Some(&BATCH),
            None,
            None,
            Some((&DESIGN, 3)),
            None,
        );
        assert!(matches!(
            res,
            Err(EdgeErrors::LengthMismatch { name: "design", .. })
        ));
    }

    #[test]
    fn test_remove_batch_effect_weights() {
        // w <- matrix(1 + ((0:47 * 5) %% 7) / 4, 8, 6, byrow = TRUE)
        // removeBatchEffect(y, batch, design, weights = w)
        let expected = [
            0.093894675925925930,
            0.203269675925925930,
            0.157696759259259189,
            0.267071759259259189,
            0.045283564814814867,
            0.154658564814814881,
            0.279839409722222265,
            0.029839409722222252,
            -0.018012152777777873,
            0.091362847222222127,
            0.206922743055555636,
            0.316297743055555636,
            0.167601495726495797,
            0.276976495726495797,
            0.201255341880341887,
            0.310630341880341887,
            0.271768162393162316,
            0.021768162393162316,
            0.092708333333333323,
            0.202083333333333337,
            0.158333333333333326,
            0.267708333333333326,
            0.202083333333333337,
            -0.047916666666666649,
            0.161401098901098883,
            0.270776098901098883,
            0.377918956043956089,
            0.127918956043956089,
            0.085679945054945028,
            0.195054945054945028,
            0.086921296296296302,
            0.196296296296296302,
            0.307407407407407407,
            0.057407407407407407,
            0.043171296296296291,
            0.152546296296296291,
            0.326388888888888840,
            0.076388888888888867,
            0.003472222222222154,
            0.112847222222222154,
            0.279513888888888951,
            0.388888888888888951,
            0.255353009259259300,
            0.005353009259259283,
            -0.040219907407407413,
            0.069155092592592587,
            0.206741898148148140,
            0.316116898148148140,
        ];
        let w = (0..N_GENES * N_SAMPLES)
            .map(|i| 1.0 + ((i * 5) % 7) as f64 / 4.0)
            .collect();
        let w = Recycled::full(w, N_GENES, N_SAMPLES).unwrap();
        let got = remove_batch_effect(
            &fixture_y(),
            N_GENES,
            N_SAMPLES,
            Some(&BATCH),
            None,
            None,
            Some((&DESIGN, 2)),
            Some(&w),
        )
        .unwrap();
        assert_matches(&got, &expected);
    }
}
