//! Small statistical utilities: multiple testing, ranks, quantiles, trimmed
//! means and the column-wise moving average.
//!
//! These are the pieces edgePython reaches into `scipy.stats`, `numpy` and R's
//! base library for. Every one of them is defined to match a specific reference
//! implementation bit for bit, because the port's parity tests compare against
//! edgeR output rather than against a mathematical ideal:
//!
//! * [`p_adjust_bh`] is R's `p.adjust(method = "BH")`
//! * [`rank_average`] is `scipy.stats.rankdata(method = "average")`
//! * [`quantile_type7`] is R's `quantile(type = 7)`, the default
//! * [`trimmed_mean`] is R's `mean(x, trim = ...)`
//! * [`moving_average_by_col`] is edgeR's `movingAverageByCol`
//!
//! Per the crate numeric policy in `lib.rs` this module is `f64` only. It sits
//! under the likelihood machinery, not next to the count data, so there is no
//! memory argument for a generic float and plenty of accuracy argument against
//! one.
//!
//! ### Ordering of `f64`
//!
//! Ranking and quantiles need a total order, so everything here sorts with
//! [`f64::total_cmp`]. That places a quiet positive NaN above `+inf`, a
//! negative NaN below `-inf`, and orders `-0.0` strictly below `+0.0`. In
//! practice: a NaN in the input sorts to the end and poisons any quantile or
//! trimmed mean that touches it, rather than being silently dropped. The one
//! deliberate exception is tie detection in [`rank_average`], which uses `==`
//! so that it agrees with `scipy` on `-0.0` and on NaN.

use crate::errors::EdgeErrors;

//////////////////////
// Multiple testing //
//////////////////////

/// Benjamini-Hochberg adjusted p-values, exactly as R's `p.adjust`.
///
/// The definition that matters is the monotone one. Sort the p-values in
/// decreasing order, form `n / i * p` where `i` is the descending rank, take a
/// running minimum down the sorted sequence, clamp at 1, and scatter back to
/// the input order. The naive `p * n / rank` without the running minimum agrees
/// on well-behaved input and disagrees the moment the raw ratios are
/// non-monotone, which is most real gene lists.
///
/// Ties are handled by the running minimum itself: two equal p-values always
/// come out equal, whichever order the sort happens to place them in.
///
/// Sequential on purpose. The cost is dominated by one sort of at most a few
/// hundred thousand `f64`, which is single-digit milliseconds, and it runs once
/// at the very end of an analysis that has already spent seconds to minutes in
/// the GLM. The running minimum is a prefix scan, so parallelising it means a
/// two-pass block scan; `par_sort_unstable` would shave a few milliseconds off
/// a wall clock nobody is looking at. Not worth the extra code path.
///
/// ### Params
///
/// * `p` - Raw p-values, in whatever order the caller holds its genes. A NaN
///   sorts above `+inf` under [`f64::total_cmp`] and is then skipped by
///   [`f64::min`], so it comes back as 1.0 and leaves the rest untouched. R's
///   `p.adjust` instead drops NA and adjusts against the reduced count, so
///   filter before calling if that distinction matters.
///
/// ### Returns
///
/// Adjusted p-values in the same order as `p`, each in `[0, 1]`. An empty input
/// gives an empty output; there is nothing to fail on.
///
/// ### References
///
/// Benjamini and Hochberg, JRSS-B, 1995
pub fn p_adjust_bh(p: &[f64]) -> Vec<f64> {
    let n = p.len();
    if n == 0 {
        return Vec::new();
    }

    // Stable, so that ties keep the caller's order, as R's `order` does.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| p[b].total_cmp(&p[a]));

    let n_f = n as f64;
    let mut out = vec![0.0_f64; n];
    let mut running = f64::INFINITY;
    for (j, &idx) in order.iter().enumerate() {
        // Descending rank: n for the largest p-value, 1 for the smallest.
        let rank = (n - j) as f64;
        let candidate = n_f / rank * p[idx];
        running = running.min(candidate);
        out[idx] = running.min(1.0);
    }
    out
}

///////////
// Ranks //
///////////

/// Fractional ranks with ties averaged, as `scipy.stats.rankdata`.
///
/// Ranks are 1-based. A run of tied values all receive the mean of the ranks
/// that run spans, so three values tied at positions 4, 5, 6 all rank 5.0.
///
/// Sorting uses [`f64::total_cmp`], but tie detection uses `==`. That pairing
/// is what reproduces `scipy`: `-0.0` and `+0.0` tie despite sorting apart, and
/// two NaNs never tie despite comparing equal under `total_cmp`. NaN sorts last
/// and so takes the top ranks.
///
/// ### Params
///
/// * `x` - Values to rank.
///
/// ### Returns
///
/// One rank per input element, in the input order. Empty in, empty out.
pub fn rank_average(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| x[a].total_cmp(&x[b]));

    let mut out = vec![0.0_f64; n];
    let mut start = 0_usize;
    while start < n {
        let mut end = start + 1;
        while end < n && x[order[end]] == x[order[start]] {
            end += 1;
        }
        // Mean of the 1-based ranks start+1 ..= end.
        let mean_rank = (start + 1 + end) as f64 / 2.0;
        for &idx in &order[start..end] {
            out[idx] = mean_rank;
        }
        start = end;
    }
    out
}

///////////////
// Quantiles //
///////////////

/// Sample quantile by R's type 7 rule, the default of `quantile()`.
///
/// Type 7 places the `k`-th order statistic at probability `(k - 1) / (n - 1)`
/// and interpolates linearly between neighbours. With `h = (n - 1) * prob`,
/// `lo = floor(h)` and `frac = h - lo`, the result is
/// `(1 - frac) * x[lo] + frac * x[lo + 1]`.
///
/// The exact arithmetic form matters. `(1 - frac) * a + frac * b` and
/// `a + frac * (b - a)` differ in the last bits, and this function feeds the
/// 90th percentile of the quasi-likelihood prior trend, where the result is
/// then raised to the fourth power. R uses the first form, so this does too.
///
/// ### Params
///
/// * `x` - Sample values, in any order. Copied and sorted internally with
///   [`f64::total_cmp`], so a NaN sorts to the top and will be returned for
///   probabilities near 1.
/// * `prob` - Probability in `[0, 1]`.
///
/// ### Returns
///
/// The interpolated quantile, or [`EdgeErrors::InvalidArgument`] if `x` is
/// empty or `prob` is outside `[0, 1]` (NaN included).
pub fn quantile_type7(x: &[f64], prob: f64) -> Result<f64, EdgeErrors> {
    if x.is_empty() {
        return Err(EdgeErrors::InvalidArgument(
            "quantile_type7 needs at least one value; got an empty slice.".to_string(),
        ));
    }
    if !(0.0..=1.0).contains(&prob) {
        return Err(EdgeErrors::InvalidArgument(format!(
            "quantile_type7 probability must lie in [0, 1]; got {prob}."
        )));
    }

    let mut sorted = x.to_vec();
    sorted.sort_unstable_by(f64::total_cmp);

    let n = sorted.len();
    let h = (n - 1) as f64 * prob;
    let lo = h.floor() as usize;
    let frac = h - lo as f64;
    // `lo` is already n - 1 when prob is 1, and then `frac` is 0, so the upper
    // neighbour is only there to keep the index in bounds.
    let hi = (lo + 1).min(n - 1);
    Ok((1.0 - frac) * sorted[lo] + frac * sorted[hi])
}

////////////////////
// Trimmed means  //
////////////////////

/// Mean after dropping a fraction from each tail, as R's `mean(x, trim = ...)`.
///
/// The trim count is `floor(trim * n)` values from each end of the sorted
/// sample, which is why `trim = 0.15` on eight observations drops one from each
/// side rather than 1.2. limma calls this with `trim = 0.15` when it summarises
/// a genewise quantity into a single robust centre.
///
/// Non-finite values are dropped before anything else, matching the reference
/// port. If nothing finite survives, that is an error rather than a NaN: a NaN
/// returned here would travel a long way before anyone noticed.
///
/// ### Params
///
/// * `x` - Sample values, in any order.
/// * `trim` - Fraction to remove from each tail, in `[0, 0.5)`.
///
/// ### Returns
///
/// The trimmed mean, or [`EdgeErrors::InvalidArgument`] if `x` is empty, `trim`
/// is outside `[0, 0.5)` (NaN included), or no finite values remain.
pub fn trimmed_mean(x: &[f64], trim: f64) -> Result<f64, EdgeErrors> {
    if x.is_empty() {
        return Err(EdgeErrors::InvalidArgument(
            "trimmed_mean needs at least one value; got an empty slice.".to_string(),
        ));
    }
    if !(0.0..0.5).contains(&trim) {
        return Err(EdgeErrors::InvalidArgument(format!(
            "trimmed_mean trim must lie in [0, 0.5); got {trim}."
        )));
    }

    let mut finite: Vec<f64> = x.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return Err(EdgeErrors::InvalidArgument(
            "trimmed_mean found no finite values in its input.".to_string(),
        ));
    }
    finite.sort_unstable_by(f64::total_cmp);

    let n = finite.len();
    let k = (trim * n as f64).floor() as usize;
    // `trim < 0.5` makes 2k < n for every n, but the guard is free and keeps
    // the slice below provably non-empty.
    let kept = if 2 * k >= n {
        &finite[..]
    } else {
        &finite[k..n - k]
    };
    Ok(kept.iter().sum::<f64>() / kept.len() as f64)
}

////////////////////
// Moving average //
////////////////////

/// Column-wise moving average, edgeR's `movingAverageByCol`.
///
/// Each column of a row-major `n_rows` by `n_cols` matrix is smoothed with a
/// running mean of `width` consecutive rows. Computed from a per-column prefix
/// sum, so the cost is one pass over the matrix regardless of `width`.
///
/// The `full_length` flag decides what happens at the ends, and it changes the
/// shape of the result:
///
/// * `true` - the column is zero-padded by `ceil(width / 2)` above and
///   `floor(width / 2)` below, and the divisor near each end is reduced to the
///   number of real observations in the window. The result has `n_rows` rows.
/// * `false` - only complete windows are kept, and the result has
///   `n_rows - width + 1` rows. The exception is `width == n_rows`, which
///   collapses to a single row of column means.
///
/// A `width` above `n_rows` is clamped to `n_rows`, as edgeR does (with a
/// warning it is not worth reproducing). A `width` of 1 returns the input
/// unchanged.
///
/// Sequential: this runs on binned data, tens to hundreds of rows, after the
/// expensive per-gene work is already done. Parallelising over columns would
/// mean strided writes from several threads for no measurable gain.
///
/// ### Params
///
/// * `x` - Matrix values, row-major, `n_rows * n_cols` of them
/// * `n_rows` - Number of rows
/// * `n_cols` - Number of columns
/// * `width` - Window width in rows, at least 1
/// * `full_length` - Whether to keep the result the same height as the input
///
/// ### Returns
///
/// The smoothed matrix, row-major with `n_cols` columns and a row count as
/// described above. Errors are [`EdgeErrors::MustBePositive`] for a zero
/// `width`, [`EdgeErrors::EmptyCounts`] for a zero dimension, and
/// [`EdgeErrors::LengthMismatch`] when `x` disagrees with `n_rows * n_cols`.
pub fn moving_average_by_col(
    x: &[f64],
    n_rows: usize,
    n_cols: usize,
    width: usize,
    full_length: bool,
) -> Result<Vec<f64>, EdgeErrors> {
    if width == 0 {
        return Err(EdgeErrors::MustBePositive("width".to_string()));
    }
    if n_rows == 0 || n_cols == 0 {
        return Err(EdgeErrors::EmptyCounts {
            n_genes: n_rows,
            n_samples: n_cols,
        });
    }
    if x.len() != n_rows * n_cols {
        return Err(EdgeErrors::LengthMismatch {
            name: "x",
            expected: n_rows * n_cols,
            got: x.len(),
        });
    }
    if width == 1 {
        return Ok(x.to_vec());
    }

    let width = width.min(n_rows);
    let half1 = width.div_ceil(2);
    let half2 = width / 2;

    let n_out = if full_length {
        n_rows
    } else if width == n_rows {
        1
    } else {
        n_rows - width + 1
    };

    // Divisors: constant `width` except near the padded ends of a full-length
    // result, where the window only covers part of the column.
    let mut divisor = vec![width as f64; n_out];
    if full_length {
        for (i, d) in divisor.iter_mut().take(half1.saturating_sub(1)).enumerate() {
            *d = (width - (half1 - 1 - i)) as f64;
        }
        for (t, d) in divisor[n_out - half2..].iter_mut().enumerate() {
            *d = (width - (t + 1)) as f64;
        }
    }

    let mut out = vec![0.0_f64; n_out * n_cols];
    // Inclusive prefix sums of one column: prefix[k] is the sum of its first k
    // entries, so a window sum is a single subtraction.
    let mut prefix = vec![0.0_f64; n_rows + 1];
    for col in 0..n_cols {
        let mut acc = 0.0_f64;
        for row in 0..n_rows {
            acc += x[row * n_cols + col];
            prefix[row + 1] = acc;
        }

        if full_length {
            let shift = half1 as isize - 1;
            for i in 0..n_out {
                let hi = ((width + i) as isize - shift).clamp(0, n_rows as isize) as usize;
                let lo = (i as isize - shift).clamp(0, n_rows as isize) as usize;
                out[i * n_cols + col] = (prefix[hi] - prefix[lo]) / divisor[i];
            }
        } else if width == n_rows {
            out[col] = prefix[n_rows] / width as f64;
        } else {
            for i in 0..n_out {
                out[i * n_cols + col] = (prefix[width + i] - prefix[i]) / divisor[i];
            }
        }
    }

    Ok(out)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Rscript -e 'cat(p.adjust(c(0.01,0.005,0.3,0.31,0.32), method="BH"), sep=",")'
    const BH_NON_MONOTONE: [f64; 5] = [0.025, 0.025, 0.32, 0.32, 0.32];

    /// Rscript -e 'cat(p.adjust(c(0.01,0.02,0.03,0.5), method="BH"), sep=",")'
    const BH_SIMPLE: [f64; 4] = [0.04, 0.04, 0.04, 0.5];

    /// Rscript -e 'cat(p.adjust(c(0.5,0.6,0.7,0.9), method="BH"), sep=",")'
    const BH_CLAMPED: [f64; 4] = [0.9, 0.9, 0.9, 0.9];

    /// uv run --with scipy python -c "from scipy.stats import rankdata; print(list(rankdata([3,1,4,1,5], method='average')))"
    const RANK_SIMPLE: [f64; 5] = [3.0, 1.5, 4.0, 1.5, 5.0];

    /// uv run --with scipy python -c "from scipy.stats import rankdata; print(list(rankdata([2.0,2.0,2.0,2.0], method='average')))"
    const RANK_ALL_EQUAL: [f64; 4] = [2.5, 2.5, 2.5, 2.5];

    /// uv run --with scipy python -c "from scipy.stats import rankdata; print(list(rankdata([5.0,1.0,5.0,3.0,1.0,5.0,2.0], method='average')))"
    const RANK_MULTI_TIE: [f64; 7] = [6.0, 1.5, 6.0, 4.0, 1.5, 6.0, 3.0];

    /// Ten values with a deliberately awkward spread, used for the quantiles.
    const Q_SAMPLE: [f64; 10] = [5.2, 1.1, 9.8, 3.3, 7.7, 2.2, 8.4, 4.6, 6.1, 0.5];

    /// Rscript -e 'cat(format(quantile(c(5.2,1.1,9.8,3.3,7.7,2.2,8.4,4.6,6.1,0.5), 0.9, type=7), digits=17))'
    const Q7_SAMPLE_090: f64 = 8.54;

    /// Rscript -e 'cat(format(quantile(c(5.2,1.1,9.8,3.3,7.7,2.2,8.4,4.6,6.1,0.5), 0.25, type=7), digits=17))'
    const Q7_SAMPLE_025: f64 = 2.475;

    /// Rscript -e 'cat(format(quantile(c(1,2,3,4,5,6,7,8,9,10), 0.9, type=7), digits=17))'
    const Q7_ONE_TO_TEN_090: f64 = 9.1;

    /// Rscript -e 'cat(format(mean(c(1,2,3,4,5,6,7,8,9,100), trim=0.15), digits=17))'
    const TRIM_TEN_015: f64 = 5.5;

    /// Rscript -e 'cat(format(mean(c(2.5,-1.0,3.75,10.0,0.25,7.5,4.0), trim=0.15), digits=17))'
    const TRIM_SEVEN_015: f64 = 3.6;

    /// Rscript -e 'cat(format(mean(c(1,2,3,4,5), trim=0.15), digits=17))'
    const TRIM_FIVE_015: f64 = 3.0;

    /// The 5 by 2 matrix used for every moving average case, row-major.
    /// R: matrix(c(1,2,3,4,5,10,20,30,40,50), nrow=5, ncol=2)
    const MA_INPUT: [f64; 10] = [1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 5.0, 50.0];

    /// Rscript -e 'library(edgeR); x <- matrix(c(1,2,3,4,5,10,20,30,40,50), nrow=5, ncol=2);
    /// cat(as.vector(t(movingAverageByCol(x, width=3, full.length=TRUE))), sep=",")'
    const MA_W3_FULL: [f64; 10] = [1.5, 15.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 4.5, 45.0];

    /// Rscript -e 'library(edgeR); x <- matrix(c(1,2,3,4,5,10,20,30,40,50), nrow=5, ncol=2);
    /// cat(as.vector(t(movingAverageByCol(x, width=3, full.length=FALSE))), sep=",")'
    const MA_W3_NOFULL: [f64; 6] = [2.0, 20.0, 3.0, 30.0, 4.0, 40.0];

    /// Rscript -e 'library(edgeR); x <- matrix(c(1,2,3,4,5,10,20,30,40,50), nrow=5, ncol=2);
    /// cat(as.vector(t(movingAverageByCol(x, width=4, full.length=TRUE))), sep=",")'
    const MA_W4_FULL: [f64; 10] = [2.0, 20.0, 2.5, 25.0, 3.5, 35.0, 4.0, 40.0, 4.5, 45.0];

    /// Rscript -e 'library(edgeR); x <- matrix(c(1,2,3,4,5,10,20,30,40,50), nrow=5, ncol=2);
    /// cat(as.vector(t(movingAverageByCol(x, width=8, full.length=TRUE))), sep=",")'
    const MA_W8_FULL: [f64; 10] = [2.0, 20.0, 2.5, 25.0, 3.0, 30.0, 3.5, 35.0, 4.0, 40.0];

    /// Rscript -e 'library(edgeR); x <- matrix(c(1,2,3,4,5,10,20,30,40,50), nrow=5, ncol=2);
    /// cat(as.vector(t(movingAverageByCol(x, width=8, full.length=FALSE))), sep=",")'
    const MA_W8_NOFULL: [f64; 2] = [3.0, 30.0];

    // -- p_adjust_bh --

    #[test]
    fn test_p_adjust_bh_cumulative_minimum_bites() {
        // Raw n/i * p here is 0.32, 0.3875, 0.5, ... descending, so the running
        // minimum pulls the three large values down to 0.32. A naive
        // p * n / rank returns 0.5 for the middle one.
        let got = p_adjust_bh(&[0.01, 0.005, 0.3, 0.31, 0.32]);
        for (g, e) in got.iter().zip(BH_NON_MONOTONE.iter()) {
            assert_relative_eq!(g, e, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_p_adjust_bh_matches_r_on_easy_case() {
        let got = p_adjust_bh(&[0.01, 0.02, 0.03, 0.5]);
        for (g, e) in got.iter().zip(BH_SIMPLE.iter()) {
            assert_relative_eq!(g, e, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_p_adjust_bh_clamps_at_one() {
        let got = p_adjust_bh(&[0.5, 0.6, 0.7, 0.9]);
        for (g, e) in got.iter().zip(BH_CLAMPED.iter()) {
            assert_relative_eq!(g, e, epsilon = 1e-12);
        }
        assert!(got.iter().all(|&v| v <= 1.0));
    }

    #[test]
    fn test_p_adjust_bh_ties_agree() {
        let got = p_adjust_bh(&[0.3, 0.3, 0.3]);
        assert_relative_eq!(got[0], 0.3, epsilon = 1e-12);
        assert_relative_eq!(got[1], 0.3, epsilon = 1e-12);
        assert_relative_eq!(got[2], 0.3, epsilon = 1e-12);
    }

    #[test]
    fn test_p_adjust_bh_singleton_and_empty() {
        // Rscript -e 'cat(p.adjust(c(0.42), method="BH"))' -> 0.42
        assert_relative_eq!(p_adjust_bh(&[0.42])[0], 0.42, epsilon = 1e-12);
        assert!(p_adjust_bh(&[]).is_empty());
    }

    // -- rank_average --

    #[test]
    fn test_rank_average_ties_take_mean_rank() {
        assert_eq!(rank_average(&[3.0, 1.0, 4.0, 1.0, 5.0]), RANK_SIMPLE);
    }

    #[test]
    fn test_rank_average_all_equal() {
        assert_eq!(rank_average(&[2.0, 2.0, 2.0, 2.0]), RANK_ALL_EQUAL);
    }

    #[test]
    fn test_rank_average_several_tie_groups() {
        let got = rank_average(&[5.0, 1.0, 5.0, 3.0, 1.0, 5.0, 2.0]);
        assert_eq!(got, RANK_MULTI_TIE);
    }

    #[test]
    fn test_rank_average_singleton_and_empty() {
        assert_eq!(rank_average(&[7.0]), vec![1.0]);
        assert!(rank_average(&[]).is_empty());
    }

    #[test]
    fn test_rank_average_descending_input() {
        // uv run --with scipy python -c "from scipy.stats import rankdata; print(list(rankdata([4.0,3.0,2.0,1.0], method='average')))"
        assert_eq!(
            rank_average(&[4.0, 3.0, 2.0, 1.0]),
            vec![4.0, 3.0, 2.0, 1.0]
        );
    }

    #[test]
    fn test_rank_average_signed_zeros_tie() {
        // total_cmp separates -0.0 and 0.0, but tie detection uses `==`, so
        // scipy's answer (all 1.5) is what comes out.
        assert_eq!(rank_average(&[-0.0, 0.0]), vec![1.5, 1.5]);
    }

    // -- quantile_type7 --

    #[test]
    fn test_quantile_type7_endpoints_are_min_and_max() {
        // Rscript -e 'cat(quantile(c(1,2,3,4,5,6,7,8,9,10), 0, type=7))' -> 1
        // Rscript -e 'cat(quantile(c(1,2,3,4,5,6,7,8,9,10), 1, type=7))' -> 10
        let x: Vec<f64> = (1..=10).map(f64::from).collect();
        assert_relative_eq!(quantile_type7(&x, 0.0).unwrap(), 1.0, epsilon = 1e-12);
        assert_relative_eq!(quantile_type7(&x, 1.0).unwrap(), 10.0, epsilon = 1e-12);
    }

    #[test]
    fn test_quantile_type7_median() {
        // Rscript -e 'cat(quantile(c(1,2,3,4,5,6,7,8,9,10), 0.5, type=7))' -> 5.5
        let x: Vec<f64> = (1..=10).map(f64::from).collect();
        assert_relative_eq!(quantile_type7(&x, 0.5).unwrap(), 5.5, epsilon = 1e-12);
    }

    #[test]
    fn test_quantile_type7_ninetieth_percentile() {
        // This is the one the quasi-likelihood prior leans on; an off-by-one in
        // the interpolation index moves every QL result.
        let x: Vec<f64> = (1..=10).map(f64::from).collect();
        assert_relative_eq!(
            quantile_type7(&x, 0.9).unwrap(),
            Q7_ONE_TO_TEN_090,
            epsilon = 1e-14
        );
        assert_relative_eq!(
            quantile_type7(&Q_SAMPLE, 0.9).unwrap(),
            Q7_SAMPLE_090,
            epsilon = 1e-14
        );
    }

    #[test]
    fn test_quantile_type7_sorts_unsorted_input() {
        assert_relative_eq!(
            quantile_type7(&Q_SAMPLE, 0.25).unwrap(),
            Q7_SAMPLE_025,
            epsilon = 1e-14
        );
    }

    #[test]
    fn test_quantile_type7_length_one() {
        // Rscript -e 'cat(quantile(c(3.5), 0.9, type=7))' -> 3.5
        for prob in [0.0, 0.5, 0.9, 1.0] {
            assert_relative_eq!(quantile_type7(&[3.5], prob).unwrap(), 3.5, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_quantile_type7_rejects_empty_and_bad_prob() {
        assert!(matches!(
            quantile_type7(&[], 0.5),
            Err(EdgeErrors::InvalidArgument(_))
        ));
        assert!(matches!(
            quantile_type7(&Q_SAMPLE, -0.001),
            Err(EdgeErrors::InvalidArgument(_))
        ));
        assert!(matches!(
            quantile_type7(&Q_SAMPLE, 1.001),
            Err(EdgeErrors::InvalidArgument(_))
        ));
        assert!(matches!(
            quantile_type7(&Q_SAMPLE, f64::NAN),
            Err(EdgeErrors::InvalidArgument(_))
        ));
    }

    // -- trimmed_mean --

    #[test]
    fn test_trimmed_mean_fractional_trim_count() {
        // n = 10, trim = 0.15 gives floor(1.5) = 1 dropped from each end, so
        // the 100 outlier goes and the answer is mean(2..9).
        let x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0];
        assert_relative_eq!(
            trimmed_mean(&x, 0.15).unwrap(),
            TRIM_TEN_015,
            epsilon = 1e-14
        );
    }

    #[test]
    fn test_trimmed_mean_odd_length_unsorted() {
        // n = 7, trim = 0.15 gives floor(1.05) = 1 from each end.
        let x = [2.5, -1.0, 3.75, 10.0, 0.25, 7.5, 4.0];
        assert_relative_eq!(
            trimmed_mean(&x, 0.15).unwrap(),
            TRIM_SEVEN_015,
            epsilon = 1e-14
        );
    }

    #[test]
    fn test_trimmed_mean_trims_nothing_when_count_rounds_to_zero() {
        // n = 5, trim = 0.15 gives floor(0.75) = 0, so this is the plain mean.
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_relative_eq!(
            trimmed_mean(&x, 0.15).unwrap(),
            TRIM_FIVE_015,
            epsilon = 1e-14
        );
        assert_relative_eq!(trimmed_mean(&x, 0.0).unwrap(), 3.0, epsilon = 1e-14);
    }

    #[test]
    fn test_trimmed_mean_drops_non_finite() {
        let x = [1.0, f64::NAN, 2.0, f64::INFINITY, 3.0, 4.0, 5.0];
        assert_relative_eq!(trimmed_mean(&x, 0.15).unwrap(), 3.0, epsilon = 1e-14);
    }

    #[test]
    fn test_trimmed_mean_rejects_empty_and_bad_trim() {
        assert!(matches!(
            trimmed_mean(&[], 0.15),
            Err(EdgeErrors::InvalidArgument(_))
        ));
        assert!(matches!(
            trimmed_mean(&[1.0, 2.0], -0.01),
            Err(EdgeErrors::InvalidArgument(_))
        ));
        assert!(matches!(
            trimmed_mean(&[1.0, 2.0], 0.5),
            Err(EdgeErrors::InvalidArgument(_))
        ));
        assert!(matches!(
            trimmed_mean(&[1.0, 2.0], f64::NAN),
            Err(EdgeErrors::InvalidArgument(_))
        ));
    }

    #[test]
    fn test_trimmed_mean_rejects_all_non_finite() {
        assert!(matches!(
            trimmed_mean(&[f64::NAN, f64::INFINITY], 0.15),
            Err(EdgeErrors::InvalidArgument(_))
        ));
    }

    // -- moving_average_by_col --

    #[test]
    fn test_moving_average_odd_width_full_length() {
        let got = moving_average_by_col(&MA_INPUT, 5, 2, 3, true).unwrap();
        assert_eq!(got.len(), MA_W3_FULL.len());
        for (g, e) in got.iter().zip(MA_W3_FULL.iter()) {
            assert_relative_eq!(g, e, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_moving_average_odd_width_complete_windows_only() {
        let got = moving_average_by_col(&MA_INPUT, 5, 2, 3, false).unwrap();
        assert_eq!(got.len(), MA_W3_NOFULL.len());
        for (g, e) in got.iter().zip(MA_W3_NOFULL.iter()) {
            assert_relative_eq!(g, e, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_moving_average_even_width_full_length() {
        // Even widths split the padding unevenly, so this catches a half1/half2
        // swap that the odd case does not.
        let got = moving_average_by_col(&MA_INPUT, 5, 2, 4, true).unwrap();
        for (g, e) in got.iter().zip(MA_W4_FULL.iter()) {
            assert_relative_eq!(g, e, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_moving_average_width_larger_than_n_rows_is_clamped() {
        let full = moving_average_by_col(&MA_INPUT, 5, 2, 8, true).unwrap();
        assert_eq!(full.len(), MA_W8_FULL.len());
        for (g, e) in full.iter().zip(MA_W8_FULL.iter()) {
            assert_relative_eq!(g, e, epsilon = 1e-12);
        }

        // width == n_rows without full length collapses to the column means.
        let collapsed = moving_average_by_col(&MA_INPUT, 5, 2, 8, false).unwrap();
        assert_eq!(collapsed.len(), MA_W8_NOFULL.len());
        for (g, e) in collapsed.iter().zip(MA_W8_NOFULL.iter()) {
            assert_relative_eq!(g, e, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_moving_average_width_one_is_identity() {
        for full in [true, false] {
            let got = moving_average_by_col(&MA_INPUT, 5, 2, 1, full).unwrap();
            assert_eq!(got, MA_INPUT.to_vec());
        }
    }

    #[test]
    fn test_moving_average_single_column() {
        // Rscript -e 'library(edgeR); cat(as.vector(movingAverageByCol(matrix(c(1,2,3,4,5)), width=3)), sep=",")'
        // -> 1.5,2,3,4,4.5
        let got = moving_average_by_col(&[1.0, 2.0, 3.0, 4.0, 5.0], 5, 1, 3, true).unwrap();
        for (g, e) in got.iter().zip([1.5, 2.0, 3.0, 4.0, 4.5].iter()) {
            assert_relative_eq!(g, e, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_moving_average_sweep_against_edger() {
        // Every width from 2 to one past 2 * n_rows, both end treatments, on a
        // 7 by 2 matrix with negative and fractional entries. This is the test
        // that pins the padding split and the tapered divisors; the single
        // width cases above only pin the shapes.
        //
        // Rscript -e 'suppressMessages(library(edgeR));
        //   x <- matrix(c(3,-1,4,1,5,9,2, 0.5,2.5,1.5,3.5,0.25,7.25,6.75), nrow=7, ncol=2);
        //   for (w in 2:9) for (fl in c(TRUE, FALSE)) {
        //     cat(as.vector(t(movingAverageByCol(x, width=w, full.length=fl))), sep=","); cat("\n") }'
        #[rustfmt::skip]
        const X: [f64; 14] = [
            3.0, 0.5, -1.0, 2.5, 4.0, 1.5, 1.0, 3.5, 5.0, 0.25, 9.0, 7.25, 2.0, 6.75,
        ];
        #[rustfmt::skip]
        let cases: &[(usize, bool, &[f64])] = &[
            (2, true,  &[1.0, 1.5, 1.5, 2.0, 2.5, 2.5, 3.0, 1.875, 7.0, 3.75, 5.5, 7.0, 2.0, 6.75]),
            (2, false, &[1.0, 1.5, 1.5, 2.0, 2.5, 2.5, 3.0, 1.875, 7.0, 3.75, 5.5, 7.0]),
            (3, true,  &[1.0, 1.5, 2.0, 1.5, 1.3333333333333333, 2.5, 3.3333333333333335, 1.75,
                         5.0, 3.6666666666666665, 5.333333333333333, 4.75, 5.5, 7.0]),
            (3, false, &[2.0, 1.5, 1.3333333333333333, 2.5, 3.3333333333333335, 1.75,
                         5.0, 3.6666666666666665, 5.333333333333333, 4.75]),
            (4, true,  &[2.0, 1.5, 1.75, 2.0, 2.25, 1.9375, 4.75, 3.125, 4.25, 4.4375,
                         5.333333333333333, 4.75, 5.5, 7.0]),
            (4, false, &[1.75, 2.0, 2.25, 1.9375, 4.75, 3.125, 4.25, 4.4375]),
            (5, true,  &[2.0, 1.5, 1.75, 2.0, 2.4, 1.65, 3.6, 3.0, 4.2, 3.85, 4.25, 4.4375,
                         5.333333333333333, 4.75]),
            (5, false, &[2.4, 1.65, 3.6, 3.0, 4.2, 3.85]),
            (6, true,  &[1.75, 2.0, 2.4, 1.65, 3.5, 2.5833333333333335, 3.3333333333333335,
                         3.625, 4.2, 3.85, 4.25, 4.4375, 5.333333333333333, 4.75]),
            (6, false, &[3.5, 2.5833333333333335, 3.3333333333333335, 3.625]),
            (7, true,  &[1.75, 2.0, 2.4, 1.65, 3.5, 2.5833333333333335, 3.2857142857142856,
                         3.1785714285714284, 3.3333333333333335, 3.625, 4.2, 3.85, 4.25, 4.4375]),
            (7, false, &[3.2857142857142856, 3.1785714285714284]),
            (9, true,  &[1.75, 2.0, 2.4, 1.65, 3.5, 2.5833333333333335, 3.2857142857142856,
                         3.1785714285714284, 3.3333333333333335, 3.625, 4.2, 3.85, 4.25, 4.4375]),
            (9, false, &[3.2857142857142856, 3.1785714285714284]),
        ];

        for &(width, full_length, expected) in cases {
            let got = moving_average_by_col(&X, 7, 2, width, full_length).unwrap();
            assert_eq!(
                got.len(),
                expected.len(),
                "wrong shape for width {width}, full_length {full_length}"
            );
            for (g, e) in got.iter().zip(expected.iter()) {
                assert_relative_eq!(g, e, epsilon = 1e-14);
            }
        }
    }

    #[test]
    fn test_moving_average_rejects_bad_shape() {
        assert!(matches!(
            moving_average_by_col(&MA_INPUT, 5, 2, 0, true),
            Err(EdgeErrors::MustBePositive(_))
        ));
        assert!(matches!(
            moving_average_by_col(&[], 0, 2, 3, true),
            Err(EdgeErrors::EmptyCounts { .. })
        ));
        assert!(matches!(
            moving_average_by_col(&[], 5, 0, 3, true),
            Err(EdgeErrors::EmptyCounts { .. })
        ));
        assert!(matches!(
            moving_average_by_col(&MA_INPUT, 4, 2, 3, true),
            Err(EdgeErrors::LengthMismatch {
                expected: 8,
                got: 10,
                ..
            })
        ));
    }
}
