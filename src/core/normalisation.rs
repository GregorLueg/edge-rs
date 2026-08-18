//! `calcNormFactors`: TMM, TMMwsp, RLE and upper-quartile scaling factors.
//!
//! A port of edgeR's `normLibSizes.default` and the four `.calcFactor*` helpers
//! it dispatches to. The reference is the R source, not the intermediate Python
//! port, wherever the two disagree; the one place they do is noted at
//! [`calc_factor_tmmwsp`].
//!
//! ### Access pattern
//!
//! Counts arrive gene-major, as everywhere else in the crate, but every method
//! here works *down a column*. TMM compares one sample against a reference
//! sample element by element, RLE takes a per-column median of ratios, and the
//! quantile rule reduces a column to one number. So the first thing this module
//! does is drop the all-zero genes and transpose what survives into a
//! column-major `f64` buffer, once, in a single pass over the row-major input.
//!
//! That costs one `n_kept * n_samples` allocation and buys contiguous reads for
//! everything downstream, including the two rank sorts per sample that dominate
//! TMM's cost. Reading the gene-major layout with a stride of `n_samples`
//! instead would touch a fresh cache line per element and would do so twice per
//! sample, once for the observed column and once for the reference. The
//! geometric means RLE needs are also computed off the transposed buffer, by
//! looping samples on the outside and genes on the inside, so both the read and
//! the accumulator stay sequential.
//!
//! ### Numerics
//!
//! Counts carry the generic `T` and are converted to `f64` on read. Library
//! sizes, factors and every intermediate are `f64`, per the crate numeric
//! policy.

use rayon::prelude::*;

use crate::errors::EdgeErrors;
use crate::numeric::stats::{quantile_type7, rank_average};
use crate::utils::traits::EdgeFloat;

////////////
// Consts //
////////////

/// Below this the observed and reference libraries are proportional.
///
/// edgeR short-circuits to a factor of exactly 1 when `max(|M|)` falls under
/// this, which keeps two identical or exactly proportional samples from picking
/// up a factor of `1 + 1e-17` out of the weighted mean.
const TMM_NULL_TOLERANCE: f64 = 1e-6;

/// Counts at or below this count as zero for TMMwsp's singleton pairing.
///
/// edgeR uses a threshold rather than `> 0` because the same routine is handed
/// fractional counts from transcript-level quantification, where a "zero" comes
/// back as a denormal rather than as an exact zero.
const TMMWSP_ZERO_EPS: f64 = 1e-14;

/// Ridge added to the TMMwsp asymptotic variance before inverting it.
///
/// TMMwsp keeps genes that are zero in one library, whose approximate variance
/// is unbounded. The ridge caps the weight at `(1 + eps) / eps` instead of
/// letting a single pair dominate the mean.
const TMMWSP_WEIGHT_RIDGE: f64 = 1e-6;

/// Median upper-quartile factor below which the reference column is picked by
/// root-count totals instead.
///
/// If more than half the samples have a zero upper quartile, `f75` carries no
/// information and edgeR falls back to `which.max(colSums(sqrt(x)))`, the same
/// rule TMMwsp uses unconditionally.
const F75_DEGENERATE: f64 = 1e-20;

/// Probability of the quantile used to choose the TMM reference column.
///
/// Fixed at the upper quartile regardless of [`NormParams::quantile`], which
/// only affects the `upperquartile` method itself. edgeR hardcodes it the same
/// way.
const REF_COLUMN_QUANTILE: f64 = 0.75;

/// The strings edgeR accepts, for the error message.
const NORM_METHOD_NAMES: &str = "TMM, TMMwsp, RLE, upperquartile, none";

/////////////
// Methods //
/////////////

/// Which scaling rule [`calc_norm_factors`] applies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormMethod {
    /// Trimmed mean of M-values, edgeR's default.
    Tmm,
    /// TMM with singleton pairing, for data with many zeros.
    TmmwSp,
    /// Relative log expression, the median ratio to the gene-wise geometric mean.
    Rle,
    /// Ratio of a column quantile to the library size.
    UpperQuartile,
    /// No scaling; every factor is 1.
    None,
}

/// Parses edgeR's `method` string.
///
/// Matching is case-insensitive, and `TMMwzp` is accepted as the pre-3.36 spelling
/// of `TMMwsp`, as edgeR still does.
///
/// ### Params
///
/// * `s` - Method name, e.g. `"TMM"` or `"upperquartile"`
///
/// ### Returns
///
/// The variant, or `None` if the string matches nothing. Use
/// `NormMethod::try_from` if you want the error rather than the `Option`.
pub fn parse_norm_method(s: &str) -> Option<NormMethod> {
    match s.to_ascii_lowercase().as_str() {
        "tmm" => Some(NormMethod::Tmm),
        "tmmwsp" | "tmmwzp" => Some(NormMethod::TmmwSp),
        "rle" => Some(NormMethod::Rle),
        "upperquartile" => Some(NormMethod::UpperQuartile),
        "none" => Some(NormMethod::None),
        _ => None,
    }
}

impl TryFrom<&str> for NormMethod {
    type Error = EdgeErrors;

    /// Fallible form of [`parse_norm_method`].
    ///
    /// ### Params
    ///
    /// * `s` - Method name
    ///
    /// ### Returns
    ///
    /// The variant, or [`EdgeErrors::UnknownMethod`] naming the accepted set.
    fn try_from(s: &str) -> Result<Self, Self::Error> {
        parse_norm_method(s).ok_or_else(|| EdgeErrors::UnknownMethod {
            kind: "normalisation",
            name: s.to_string(),
            expected: NORM_METHOD_NAMES,
        })
    }
}

////////////////
// NormParams //
////////////////

/// Tuning knobs for [`calc_norm_factors`].
///
/// The names mirror edgeR's arguments: `logratio_trim` is `logratioTrim`,
/// `sum_trim` is `sumTrim`, `do_weighting` is `doWeighting`, `a_cutoff` is
/// `Acutoff` and `quantile` is `p`.
#[derive(Clone, Copy, Debug)]
pub struct NormParams {
    /// Fraction trimmed from each tail of the log-ratios. TMM and TMMwsp only.
    pub logratio_trim: f64,
    /// Fraction trimmed from each tail of the abundances. TMM and TMMwsp only.
    pub sum_trim: f64,
    /// Weight each retained log-ratio by the inverse of its approximate
    /// asymptotic variance. TMM and TMMwsp only.
    pub do_weighting: bool,
    /// Abundance floor; genes with a mean log-abundance at or below this are
    /// dropped before trimming. TMM only, as in edgeR, where TMMwsp accepts the
    /// argument and ignores it.
    pub a_cutoff: f64,
    /// Probability of the column quantile. [`NormMethod::UpperQuartile`] only.
    pub quantile: f64,
}

impl NormParams {
    /// Builds a parameter set.
    ///
    /// No validation happens here; [`calc_norm_factors`] checks the domains so
    /// that a bad value surfaces as an [`EdgeErrors`] at the call that uses it
    /// rather than at construction.
    ///
    /// ### Params
    ///
    /// * `logratio_trim` - Tail fraction trimmed from the log-ratios, in `[0, 1)`
    /// * `sum_trim` - Tail fraction trimmed from the abundances, in `[0, 1)`
    /// * `do_weighting` - Whether to use inverse-variance weights
    /// * `a_cutoff` - Abundance floor for TMM
    /// * `quantile` - Column quantile probability, in `[0, 1]`
    ///
    /// ### Returns
    ///
    /// The parameter set.
    pub fn new(
        logratio_trim: f64,
        sum_trim: f64,
        do_weighting: bool,
        a_cutoff: f64,
        quantile: f64,
    ) -> Self {
        Self {
            logratio_trim,
            sum_trim,
            do_weighting,
            a_cutoff,
            quantile,
        }
    }
}

impl Default for NormParams {
    /// edgeR's defaults.
    fn default() -> Self {
        Self {
            logratio_trim: 0.3,
            sum_trim: 0.05,
            do_weighting: true,
            a_cutoff: -1e10,
            quantile: 0.75,
        }
    }
}

/// Rejects parameter values whose downstream behaviour is undefined.
///
/// ### Params
///
/// * `params` - The resolved parameter set
///
/// ### Returns
///
/// `Ok(())`, or [`EdgeErrors::InvalidArgument`] naming the offending knob.
fn validate_params(params: &NormParams) -> Result<(), EdgeErrors> {
    for (name, trim) in [
        ("logratio_trim", params.logratio_trim),
        ("sum_trim", params.sum_trim),
    ] {
        if !(0.0..1.0).contains(&trim) {
            return Err(EdgeErrors::InvalidArgument(format!(
                "{name} must lie in [0, 1); got {trim}."
            )));
        }
    }
    if !(0.0..=1.0).contains(&params.quantile) {
        return Err(EdgeErrors::InvalidArgument(format!(
            "quantile must lie in [0, 1]; got {}.",
            params.quantile
        )));
    }
    if params.a_cutoff.is_nan() {
        return Err(EdgeErrors::InvalidArgument(
            "a_cutoff must not be NaN.".to_string(),
        ));
    }
    Ok(())
}

/////////////
// Helpers //
/////////////

/// Resolves the library sizes, defaulting to the column sums.
///
/// edgeR recycles a mismatched `lib.size` with a warning. This errors instead:
/// the crate has no warning channel, and silently recycling a length-3 vector
/// across six samples is not something a caller wants to discover downstream.
///
/// ### Params
///
/// * `counts` - Row-major counts, `n_genes * n_samples`
/// * `n_genes` - Number of genes
/// * `n_samples` - Number of samples
/// * `lib_size` - Supplied library sizes, or `None` for the column sums
///
/// ### Returns
///
/// One library size per sample, each finite and strictly positive, or an
/// [`EdgeErrors`] if the supplied vector is the wrong length or an entry is not
/// usable as a denominator.
fn resolve_lib_size<T: EdgeFloat>(
    counts: &[T],
    n_genes: usize,
    n_samples: usize,
    lib_size: Option<&[f64]>,
) -> Result<Vec<f64>, EdgeErrors> {
    let lib = match lib_size {
        Some(ls) => {
            if ls.len() != n_samples {
                return Err(EdgeErrors::LengthMismatch {
                    name: "lib_size",
                    expected: n_samples,
                    got: ls.len(),
                });
            }
            ls.to_vec()
        }
        None => {
            let mut sums = vec![0.0_f64; n_samples];
            for row in counts.chunks_exact(n_samples).take(n_genes) {
                for (acc, v) in sums.iter_mut().zip(row.iter()) {
                    *acc += v.to_f64().unwrap_or(f64::NAN);
                }
            }
            sums
        }
    };

    // edgeR carries a zero library size through to a NaN factor. Every method
    // here divides by it, so refuse up front and say which sample is dead.
    if let Some(j) = lib.iter().position(|v| !v.is_finite() || *v <= 0.0) {
        return Err(EdgeErrors::InvalidArgument(format!(
            "library size for sample {j} must be finite and positive; got {}.",
            lib[j]
        )));
    }
    Ok(lib)
}

/// Drops the all-zero genes and transposes the survivors to column-major `f64`.
///
/// Two passes over the row-major input: one to mark which genes have a positive
/// count anywhere, one to scatter the survivors into the transposed buffer. The
/// scatter reads sequentially and writes with a stride of `n_kept`, which is the
/// cheaper of the two directions because the write side is what the rest of the
/// module then reads sequentially, repeatedly.
///
/// ### Params
///
/// * `counts` - Row-major counts, `n_genes * n_samples`
/// * `n_genes` - Number of genes
/// * `n_samples` - Number of samples
///
/// ### Returns
///
/// The column-major `n_kept * n_samples` buffer, sample-major, together with
/// `n_kept`. Errors with [`EdgeErrors::InvalidArgument`] on a negative or
/// non-finite count.
fn to_column_major_nonzero<T: EdgeFloat>(
    counts: &[T],
    n_genes: usize,
    n_samples: usize,
) -> Result<(Vec<f64>, usize), EdgeErrors> {
    let mut keep = vec![false; n_genes];
    let mut n_kept = 0_usize;
    for (g, row) in counts.chunks_exact(n_samples).enumerate() {
        let mut any_positive = false;
        for v in row {
            if !v.is_finite() || *v < T::zero() {
                return Err(EdgeErrors::InvalidArgument(format!(
                    "counts must be non-negative and finite, found {v} for gene {g}"
                )));
            }
            any_positive |= *v > T::zero();
        }
        keep[g] = any_positive;
        n_kept += usize::from(any_positive);
    }

    let mut x = vec![0.0_f64; n_kept * n_samples];
    let mut k = 0_usize;
    for (g, row) in counts.chunks_exact(n_samples).enumerate() {
        if !keep[g] {
            continue;
        }
        for (j, v) in row.iter().enumerate() {
            x[j * n_kept + k] = v.to_f64().unwrap_or(f64::NAN);
        }
        k += 1;
    }
    Ok((x, n_kept))
}

/// Rescales factors to a geometric mean of one.
///
/// `f / exp(mean(log(f)))`, edgeR's final line. Without it the factors are only
/// identified up to a constant and the effective library sizes drift.
///
/// ### Params
///
/// * `f` - Factors, rescaled in place
///
/// ### Returns
///
/// `Ok(())`, or [`EdgeErrors::InvalidArgument`] if a factor is not finite and
/// strictly positive, which would make the geometric mean meaningless.
fn normalise_to_unit_product(f: &mut [f64]) -> Result<(), EdgeErrors> {
    if let Some(j) = f.iter().position(|v| !v.is_finite() || *v <= 0.0) {
        return Err(EdgeErrors::InvalidArgument(format!(
            "normalisation factor for sample {j} is not finite and positive ({}); \
             the counts are too sparse for this method",
            f[j]
        )));
    }
    let log_mean = f.iter().map(|v| v.ln()).sum::<f64>() / f.len() as f64;
    let scale = log_mean.exp();
    for v in f.iter_mut() {
        *v /= scale;
    }
    Ok(())
}

///////////////////////
// Reference columns //
///////////////////////

/// Picks the reference sample when the caller does not name one.
///
/// TMM takes the sample whose upper-quartile factor sits closest to the mean of
/// those factors, `which.min(abs(f75 - mean(f75)))`, breaking ties towards the
/// first sample. If more than half the samples have a zero upper quartile, that
/// statistic is uninformative and it falls back to `which.max(colSums(sqrt(x)))`.
/// TMMwsp uses the root-count rule unconditionally, because it exists for data
/// sparse enough that the quartile rule has already broken down.
///
/// ### Params
///
/// * `x` - Column-major filtered counts, `n_kept * n_samples`
/// * `n_kept` - Genes surviving the all-zero filter
/// * `n_samples` - Number of samples
/// * `lib_size` - Library size per sample
/// * `method` - [`NormMethod::Tmm`] or [`NormMethod::TmmwSp`]
///
/// ### Returns
///
/// The zero-based reference column, or an [`EdgeErrors`] if the quantile could
/// not be formed.
fn resolve_ref_column(
    x: &[f64],
    n_kept: usize,
    n_samples: usize,
    lib_size: &[f64],
    method: NormMethod,
) -> Result<usize, EdgeErrors> {
    if method == NormMethod::TmmwSp {
        return Ok(arg_max_sqrt_colsum(x, n_kept, n_samples));
    }

    let mut f75 = vec![0.0_f64; n_samples];
    for (j, f) in f75.iter_mut().enumerate() {
        *f = quantile_type7(&x[j * n_kept..(j + 1) * n_kept], REF_COLUMN_QUANTILE)? / lib_size[j];
    }

    let median = quantile_type7(&f75, 0.5)?;
    if median < F75_DEGENERATE {
        return Ok(arg_max_sqrt_colsum(x, n_kept, n_samples));
    }

    let mean = f75.iter().sum::<f64>() / n_samples as f64;
    // `which.min` keeps the first minimum, so the comparison must be strict.
    let mut best = 0_usize;
    let mut best_gap = f64::INFINITY;
    for (j, f) in f75.iter().enumerate() {
        let gap = (f - mean).abs();
        if gap < best_gap {
            best_gap = gap;
            best = j;
        }
    }
    if !best_gap.is_finite() {
        return Err(EdgeErrors::InvalidReferenceColumn(
            "upper-quartile factors are not finite for any sample".to_string(),
        ));
    }
    Ok(best)
}

/// Index of the sample with the largest sum of square-rooted counts.
///
/// The variance-stabilised total, which is edgeR's sparse-data proxy for "the
/// deepest library". Ties go to the first sample, as `which.max` does.
///
/// ### Params
///
/// * `x` - Column-major filtered counts, `n_kept * n_samples`
/// * `n_kept` - Genes surviving the all-zero filter
/// * `n_samples` - Number of samples
///
/// ### Returns
///
/// The zero-based column index.
fn arg_max_sqrt_colsum(x: &[f64], n_kept: usize, n_samples: usize) -> usize {
    let mut best = 0_usize;
    let mut best_sum = f64::NEG_INFINITY;
    for j in 0..n_samples {
        let s: f64 = x[j * n_kept..(j + 1) * n_kept]
            .iter()
            .map(|v| v.sqrt())
            .sum();
        if s > best_sum {
            best_sum = s;
            best = j;
        }
    }
    best
}

/////////////////////////
// Column-wise kernels //
/////////////////////////

/// Upper-quartile factors: a column quantile over the library size.
///
/// `.calcFactorQuantile`. The quantile is R's type 7, which is also numpy's
/// default, so [`quantile_type7`] matches both references.
///
/// ### Params
///
/// * `x` - Column-major filtered counts, `n_kept * n_samples`
/// * `n_kept` - Genes surviving the all-zero filter
/// * `n_samples` - Number of samples
/// * `lib_size` - Library size per sample
/// * `p` - Quantile probability
///
/// ### Returns
///
/// One unnormalised factor per sample, or an [`EdgeErrors`] if a column
/// quantile is zero. edgeR only warns there and returns NaN once the geometric
/// mean is taken, which is not a useful answer to hand back.
fn calc_factor_quantile(
    x: &[f64],
    n_kept: usize,
    n_samples: usize,
    lib_size: &[f64],
    p: f64,
) -> Result<Vec<f64>, EdgeErrors> {
    let mut f = vec![0.0_f64; n_samples];
    for (j, out) in f.iter_mut().enumerate() {
        let q = quantile_type7(&x[j * n_kept..(j + 1) * n_kept], p)?;
        if q <= 0.0 {
            return Err(EdgeErrors::InvalidArgument(format!(
                "the {p} quantile of sample {j} is zero; upper-quartile \
                 normalisation is undefined for counts this sparse"
            )));
        }
        *out = q / lib_size[j];
    }
    Ok(f)
}

/// RLE factors: the median ratio to the gene-wise geometric mean.
///
/// `.calcFactorRLE(x) / lib.size`. Genes whose geometric mean is zero, meaning
/// at least one sample has no count for them, contribute nothing, which is
/// what `[gm > 0]` does in the R.
///
/// The geometric means are accumulated with samples on the outside and genes on
/// the inside, so both the count read and the accumulator write run
/// sequentially over the column-major buffer.
///
/// ### Params
///
/// * `x` - Column-major filtered counts, `n_kept * n_samples`
/// * `n_kept` - Genes surviving the all-zero filter
/// * `n_samples` - Number of samples
/// * `lib_size` - Library size per sample
///
/// ### Returns
///
/// One unnormalised factor per sample, or [`EdgeErrors::NoGenesAfterFiltering`]
/// if no gene is positive in every sample, which leaves the median undefined.
///
/// ### References
///
/// Anders and Huber, Genome Biology, 2010
fn calc_factor_rle(
    x: &[f64],
    n_kept: usize,
    n_samples: usize,
    lib_size: &[f64],
) -> Result<Vec<f64>, EdgeErrors> {
    let mut log_gm = vec![0.0_f64; n_kept];
    for j in 0..n_samples {
        for (acc, v) in log_gm.iter_mut().zip(&x[j * n_kept..(j + 1) * n_kept]) {
            *acc += v.ln();
        }
    }
    let positive: Vec<usize> = log_gm
        .iter()
        .enumerate()
        .filter(|(_, v)| (*v / n_samples as f64).exp() > 0.0)
        .map(|(g, _)| g)
        .collect();
    if positive.is_empty() {
        return Err(EdgeErrors::NoGenesAfterFiltering { n_genes: n_kept });
    }

    let gm: Vec<f64> = positive
        .iter()
        .map(|&g| (log_gm[g] / n_samples as f64).exp())
        .collect();

    let mut f = vec![0.0_f64; n_samples];
    let mut ratios = vec![0.0_f64; positive.len()];
    for (j, out) in f.iter_mut().enumerate() {
        let col = &x[j * n_kept..(j + 1) * n_kept];
        for (r, (&g, &m)) in ratios.iter_mut().zip(positive.iter().zip(gm.iter())) {
            *r = col[g] / m;
        }
        *out = quantile_type7(&ratios, 0.5)? / lib_size[j];
    }
    Ok(f)
}

///////////////////////
// TMM between pairs //
///////////////////////

/// TMM factor for one sample against the reference.
///
/// `.calcFactorTMM`. Genes that are zero in *either* library give a
/// non-finite `M` or `A` and are dropped outright: TMM has nothing to say about
/// a ratio with a zero in it. That is the whole reason TMMwsp exists.
///
/// Trimming is by rank, on both axes at once. With `n` usable genes the kept
/// set is the rank window `[floor(n * logratio_trim) + 1, n + 1 - that]` on the
/// log-ratios intersected with the same window at `sum_trim` on the abundances.
/// The `floor(...) + 1` is the part that has to be exact: computing the bound
/// as `ceil` or dropping the `+ 1` shifts the window by one gene and moves
/// every factor in the last few digits, sometimes further on small matrices.
///
/// Ranks are averaged over ties, matching R's `rank()` default, so a run of
/// equal log-ratios straddling the trim boundary is either wholly kept or
/// wholly dropped rather than split arbitrarily.
///
/// ### Params
///
/// * `obs` - Observed sample's filtered counts, contiguous
/// * `reference` - Reference sample's filtered counts, contiguous
/// * `lib_obs` - Library size of the observed sample
/// * `lib_ref` - Library size of the reference sample
/// * `params` - Trim fractions, weighting flag and abundance cutoff
///
/// ### Returns
///
/// The unnormalised factor, `2^TMM`. Exactly 1 when the two libraries are
/// proportional or nothing survives the finite filter.
///
/// ### References
///
/// Robinson and Oshlack, Genome Biology, 2010
fn calc_factor_tmm(
    obs: &[f64],
    reference: &[f64],
    lib_obs: f64,
    lib_ref: f64,
    params: &NormParams,
) -> f64 {
    let n_all = obs.len();
    let mut log_r = Vec::with_capacity(n_all);
    let mut abs_e = Vec::with_capacity(n_all);
    let mut var = Vec::with_capacity(n_all);

    for (&o, &r) in obs.iter().zip(reference.iter()) {
        let p_obs = o / lib_obs;
        let p_ref = r / lib_ref;
        let lr = (p_obs / p_ref).log2();
        let ae = (p_obs.log2() + p_ref.log2()) / 2.0;
        if lr.is_finite() && ae.is_finite() && ae > params.a_cutoff {
            log_r.push(lr);
            abs_e.push(ae);
            var.push((lib_obs - o) / lib_obs / o + (lib_ref - r) / lib_ref / r);
        }
    }

    let n = log_r.len();
    if n == 0 || log_r.iter().fold(0.0_f64, |m, v| m.max(v.abs())) < TMM_NULL_TOLERANCE {
        return 1.0;
    }

    let lo_l = (n as f64 * params.logratio_trim).floor() + 1.0;
    let hi_l = n as f64 + 1.0 - lo_l;
    let lo_s = (n as f64 * params.sum_trim).floor() + 1.0;
    let hi_s = n as f64 + 1.0 - lo_s;

    let rank_r = rank_average(&log_r);
    let rank_e = rank_average(&abs_e);

    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    let mut kept = 0_usize;
    for i in 0..n {
        let keep = rank_r[i] >= lo_l && rank_r[i] <= hi_l && rank_e[i] >= lo_s && rank_e[i] <= hi_s;
        if !keep {
            continue;
        }
        kept += 1;
        if params.do_weighting {
            num += log_r[i] / var[i];
            den += 1.0 / var[i];
        } else {
            num += log_r[i];
        }
    }

    let f = if params.do_weighting {
        num / den
    } else if kept == 0 {
        f64::NAN
    } else {
        num / kept as f64
    };

    // edgeR maps a missing weighted mean to a factor of one rather than
    // failing.
    let f = if f.is_nan() { 0.0 } else { f };

    f.exp2()
}

/// TMMwsp factor for one sample against the reference.
///
/// `.calcFactorTMMwsp`, TMM with singleton pairing. Where TMM throws away every
/// gene that is zero in one library, TMMwsp keeps that information: it pairs
/// the largest counts among the observed-only positives with the largest among
/// the reference-only positives, so `k` genes seen only in the observed sample
/// and `k` seen only in the reference offset each other instead of both
/// vanishing. Genes zero in both are still dropped. On data where most genes
/// are zero somewhere, that is most of the matrix.
///
/// ### Notes
///
/// **On tie-breaking:**
///
/// The M-values are ordered by `order(M, M.shrunk)`, R's stable sort keyed on
/// `M` with the add-0.5 shrunk log-ratio as the tiebreak. This is not cosmetic.
/// On sparse integer counts the same `(obs, ref)` pair recurs thousands of
/// times, and after the singleton pairing many pairs share an `M` exactly, so
/// the trim boundary lands in the middle of a tied run. Without a tiebreak,
/// which members of that run get trimmed depends on the sort's internal order
/// and the answer stops being reproducible; with it, the run is ordered by the
/// shrunk ratio, which is monotone in the raw counts, so the pairs carrying the
/// least evidence are trimmed first. Ties in *both* keys fall back to input
/// order, which is why the sort must be stable and why the paired singletons
/// are appended in edgeR's order rather than interleaved.
///
/// Note the trim window here is `[floor(n * trim) + 1, n + 1 - that]` in
/// 1-based positions, the same as [`calc_factor_tmm`].
///
/// ### Params
///
/// * `obs` - Observed sample's filtered counts, contiguous
/// * `reference` - Reference sample's filtered counts, contiguous
/// * `lib_obs` - Library size of the observed sample
/// * `lib_ref` - Library size of the reference sample
/// * `params` - Trim fractions and weighting flag. `a_cutoff` is ignored, as in
///   edgeR.
///
/// ### Returns
///
/// The unnormalised factor, `2^TMM`. Exactly 1 when no pair survives or the two
/// libraries are proportional.
///
/// ### References
///
/// Chen et al., F1000Research, 2016
fn calc_factor_tmmwsp(
    obs: &[f64],
    reference: &[f64],
    lib_obs: f64,
    lib_ref: f64,
    params: &NormParams,
) -> f64 {
    let n_all = obs.len();
    let mut pair_obs = Vec::with_capacity(n_all);
    let mut pair_ref = Vec::with_capacity(n_all);
    // Positive in the reference only, and positive in the observed only.
    let mut single_ref = Vec::new();
    let mut single_obs = Vec::new();

    for (&o, &r) in obs.iter().zip(reference.iter()) {
        match (o > TMMWSP_ZERO_EPS, r > TMMWSP_ZERO_EPS) {
            (true, true) => {
                pair_obs.push(o);
                pair_ref.push(r);
            }
            (true, false) => single_obs.push(o),
            (false, true) => single_ref.push(r),
            (false, false) => {}
        }
    }

    let n_eligible = single_obs.len().min(single_ref.len());
    if n_eligible > 0 {
        single_obs.sort_unstable_by(|a, b| b.total_cmp(a));
        single_ref.sort_unstable_by(|a, b| b.total_cmp(a));
        pair_obs.extend_from_slice(&single_obs[..n_eligible]);
        pair_ref.extend_from_slice(&single_ref[..n_eligible]);
    }

    let n = pair_obs.len();
    if n == 0 {
        return 1.0;
    }

    let p_obs: Vec<f64> = pair_obs.iter().map(|o| o / lib_obs).collect();
    let p_ref: Vec<f64> = pair_ref.iter().map(|r| r / lib_ref).collect();
    let m: Vec<f64> = p_obs
        .iter()
        .zip(p_ref.iter())
        .map(|(o, r)| (o / r).log2())
        .collect();
    let a: Vec<f64> = p_obs
        .iter()
        .zip(p_ref.iter())
        .map(|(o, r)| 0.5 * (o * r).log2())
        .collect();

    if m.iter().fold(0.0_f64, |acc, v| acc.max(v.abs())) < TMM_NULL_TOLERANCE {
        return 1.0;
    }

    let m_shrunk: Vec<f64> = pair_obs
        .iter()
        .zip(pair_ref.iter())
        .map(|(o, r)| (((o + 0.5) / (lib_obs + 0.5)) / ((r + 0.5) / (lib_ref + 0.5))).log2())
        .collect();

    let mut order_m: Vec<usize> = (0..n).collect();
    // Stable, so a tie in both keys keeps the input order, as R's `order` does.
    order_m.sort_by(|&i, &j| {
        m[i].total_cmp(&m[j])
            .then(m_shrunk[i].total_cmp(&m_shrunk[j]))
    });
    let mut order_a: Vec<usize> = (0..n).collect();
    order_a.sort_by(|&i, &j| a[i].total_cmp(&a[j]));

    let keep_m = trim_window(&order_m, n, params.logratio_trim);
    let keep_a = trim_window(&order_a, n, params.sum_trim);

    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    let mut kept = 0_usize;
    for i in 0..n {
        if !(keep_m[i] && keep_a[i]) {
            continue;
        }
        kept += 1;
        if params.do_weighting {
            let v = (1.0 - p_obs[i]) / p_obs[i] / lib_obs + (1.0 - p_ref[i]) / p_ref[i] / lib_ref;
            let w = (1.0 + TMMWSP_WEIGHT_RIDGE) / (v + TMMWSP_WEIGHT_RIDGE);
            num += w * m[i];
            den += w;
        } else {
            num += m[i];
        }
    }

    if kept == 0 {
        return 1.0;
    }
    let tmm = if params.do_weighting {
        num / den
    } else {
        num / kept as f64
    };
    if tmm.is_nan() { 1.0 } else { tmm.exp2() }
}

/// Marks the central `n + 2 - 2 * (floor(n * trim) + 1)` entries of an
/// ordering.
///
/// The window is stated in R's 1-based positions, `[lo, n + 1 - lo]` with
/// `lo = floor(n * trim) + 1`, and translated to a half-open 0-based slice of
/// the order vector. R would silently count *down* if `lo` exceeded the upper
/// bound, which only happens for trims at or above 0.5 on small `n`; that keeps
/// nothing here instead of keeping a reversed window.
///
/// ### Params
///
/// * `order` - Indices sorted by the statistic being trimmed
/// * `n` - Length of `order`
/// * `trim` - Fraction removed from each tail
///
/// ### Returns
///
/// A mask over the *original* positions, true for the retained entries.
fn trim_window(order: &[usize], n: usize, trim: f64) -> Vec<bool> {
    let mut keep = vec![false; n];
    let lo = (n as f64 * trim) as usize + 1;
    let hi = n + 1 - lo.min(n + 1);
    if lo > hi {
        return keep;
    }
    for &idx in &order[lo - 1..hi] {
        keep[idx] = true;
    }
    keep
}

///////////////
// Front end //
///////////////

/// Normalisation factors for a count matrix, as edgeR's `calcNormFactors`.
///
/// Genes that are zero in every sample are dropped first, exactly as edgeR
/// does, and the library sizes are *not* recomputed afterwards: they stay
/// anchored to whatever was supplied or to the column sums of the unfiltered
/// matrix. If no gene survives, or there is only one sample, the method
/// degrades to [`NormMethod::None`] rather than failing, again matching edgeR.
///
/// The returned factors are rescaled to have a geometric mean of one, so they
/// multiply to one across samples.
///
/// ### Params
///
/// * `counts` - Row-major counts, `n_genes * n_samples`. Must be non-negative
///   and finite.
/// * `n_genes` - Number of genes, that is, rows
/// * `n_samples` - Number of samples, that is, columns
/// * `lib_size` - Library size per sample. Defaults to the column sums. Every
///   method divides by it, so each entry must be finite and strictly positive.
/// * `method` - Which scaling rule to apply
/// * `ref_column` - Zero-based reference sample for TMM and TMMwsp. `None`
///   picks one by the rule described at [`resolve_ref_column`]. Ignored by the
///   other methods.
/// * `params` - Tuning knobs, or `None` for [`NormParams::default`]
///
/// ### Returns
///
/// One factor per sample, with geometric mean one, or an [`EdgeErrors`] if the
/// shape is inconsistent, a library size is not positive, the reference column
/// is out of range, a parameter is outside its domain, or the chosen method
/// cannot be computed on this matrix.
///
/// ### References
///
/// Robinson and Oshlack, Genome Biology, 2010 (TMM); Anders and Huber, Genome
/// Biology, 2010 (RLE); Bullard et al., BMC Bioinformatics, 2010 (upper
/// quartile)
pub fn calc_norm_factors<T: EdgeFloat>(
    counts: &[T],
    n_genes: usize,
    n_samples: usize,
    lib_size: Option<&[f64]>,
    method: NormMethod,
    ref_column: Option<usize>,
    params: Option<NormParams>,
) -> Result<Vec<f64>, EdgeErrors> {
    let params = params.unwrap_or_default();
    validate_params(&params)?;

    if n_genes == 0 || n_samples == 0 {
        return Err(EdgeErrors::EmptyCounts { n_genes, n_samples });
    }
    if counts.len() != n_genes * n_samples {
        return Err(EdgeErrors::LengthMismatch {
            name: "counts",
            expected: n_genes * n_samples,
            got: counts.len(),
        });
    }
    if let Some(r) = ref_column
        && r >= n_samples
    {
        return Err(EdgeErrors::InvalidReferenceColumn(format!(
            "column {r} requested but there are only {n_samples} samples"
        )));
    }

    let lib = resolve_lib_size(counts, n_genes, n_samples, lib_size)?;
    let (x, n_kept) = to_column_major_nonzero(counts, n_genes, n_samples)?;

    // edgeR degrades rather than failing: nothing to normalise against.
    let method = if n_kept == 0 || n_samples == 1 {
        NormMethod::None
    } else {
        method
    };

    let mut f = match method {
        NormMethod::Tmm | NormMethod::TmmwSp => {
            let reference = match ref_column {
                Some(r) => r,
                None => resolve_ref_column(&x, n_kept, n_samples, &lib, method)?,
            };
            let kernel = if method == NormMethod::Tmm {
                calc_factor_tmm
            } else {
                calc_factor_tmmwsp
            };
            let ref_col = &x[reference * n_kept..(reference + 1) * n_kept];
            (0..n_samples)
                .into_par_iter()
                .map(|j| {
                    kernel(
                        &x[j * n_kept..(j + 1) * n_kept],
                        ref_col,
                        lib[j],
                        lib[reference],
                        &params,
                    )
                })
                .collect()
        }
        NormMethod::Rle => calc_factor_rle(&x, n_kept, n_samples, &lib)?,
        NormMethod::UpperQuartile => {
            calc_factor_quantile(&x, n_kept, n_samples, &lib, params.quantile)?
        }
        NormMethod::None => vec![1.0; n_samples],
    };

    normalise_to_unit_product(&mut f)?;
    Ok(f)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::path::PathBuf;

    /// Relative tolerance against the R reference values.
    const TOL: f64 = 1e-14;

    /// Reads a headed, all-numeric CSV from `tests/data` into a row-major matrix.
    ///
    /// The fixtures are R `write.csv` output without row names: one quoted
    /// header line, then one comma-separated record per row. Everything is
    /// parsed as `f64`.
    ///
    /// ### Params
    ///
    /// * `name` - File name inside `tests/data`
    ///
    /// ### Returns
    ///
    /// The values row-major, the row count and the column count.
    fn read_csv(name: &str) -> (Vec<f64>, usize, usize) {
        let path: PathBuf = [env!("CARGO_MANIFEST_DIR"), "tests", "data", name]
            .iter()
            .collect();
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));

        let mut lines = text.lines().filter(|l| !l.trim().is_empty());
        let header = lines.next().expect("fixture has no header");
        let n_cols = header.split(',').count();

        let mut values = Vec::new();
        let mut n_rows = 0_usize;
        for line in lines {
            for field in line.split(',') {
                values.push(
                    field
                        .trim()
                        .trim_matches('"')
                        .parse::<f64>()
                        .unwrap_or_else(|e| panic!("bad field {field:?} in {name}: {e}")),
                );
            }
            n_rows += 1;
        }
        assert_eq!(values.len(), n_rows * n_cols, "ragged fixture {name}");
        (values, n_rows, n_cols)
    }

    /// Runs [`calc_norm_factors`] and compares against R, element by element.
    ///
    /// ### Params
    ///
    /// * `counts` - Row-major counts
    /// * `n_genes` - Number of genes
    /// * `n_samples` - Number of samples
    /// * `lib_size` - Library sizes, or `None` for column sums
    /// * `method` - Method under test
    /// * `ref_column` - Zero-based reference column, or `None`
    /// * `params` - Tuning knobs, or `None`
    /// * `expected` - `calcNormFactors` output from edgeR 4.8.2
    #[allow(clippy::too_many_arguments)]
    fn check(
        counts: &[f64],
        n_genes: usize,
        n_samples: usize,
        lib_size: Option<&[f64]>,
        method: NormMethod,
        ref_column: Option<usize>,
        params: Option<NormParams>,
        expected: &[f64],
    ) {
        let got = calc_norm_factors(
            counts, n_genes, n_samples, lib_size, method, ref_column, params,
        )
        .expect("calc_norm_factors failed");
        assert_eq!(got.len(), expected.len());
        for (g, e) in got.iter().zip(expected.iter()) {
            assert_relative_eq!(g, e, max_relative = TOL, epsilon = 0.0);
        }
    }

    // -- method parsing and parameters --

    #[test]
    fn test_parse_norm_method_accepts_every_edger_spelling() {
        assert_eq!(parse_norm_method("TMM"), Some(NormMethod::Tmm));
        assert_eq!(parse_norm_method("tmm"), Some(NormMethod::Tmm));
        assert_eq!(parse_norm_method("TMMwsp"), Some(NormMethod::TmmwSp));
        assert_eq!(parse_norm_method("TMMwzp"), Some(NormMethod::TmmwSp));
        assert_eq!(parse_norm_method("RLE"), Some(NormMethod::Rle));
        assert_eq!(
            parse_norm_method("upperquartile"),
            Some(NormMethod::UpperQuartile)
        );
        assert_eq!(parse_norm_method("none"), Some(NormMethod::None));
        assert_eq!(parse_norm_method("quantile"), None);
    }

    #[test]
    fn test_try_from_reports_the_unknown_method() {
        let err = NormMethod::try_from("wibble").unwrap_err();
        assert!(matches!(err, EdgeErrors::UnknownMethod { .. }));
    }

    #[test]
    fn test_default_params_are_edgers() {
        let p = NormParams::default();
        assert_eq!(p.logratio_trim, 0.3);
        assert_eq!(p.sum_trim, 0.05);
        assert!(p.do_weighting);
        assert_eq!(p.a_cutoff, -1e10);
        assert_eq!(p.quantile, 0.75);
        let q = NormParams::new(0.2, 0.1, false, -5.0, 0.5);
        assert_eq!(q.logratio_trim, 0.2);
        assert!(!q.do_weighting);
    }

    // -- R parity, tests/data/test_data_part1.csv, 22 genes by 4 samples --
    //
    // Row one is all zeros, so this fixture also exercises the all-zero filter.
    // Every expected vector below is `calcNormFactors(x, ...)` from edgeR 4.8.2.

    #[test]
    fn test_part1_tmm_matches_r() {
        let (x, g, s) = read_csv("test_data_part1.csv");
        check(
            &x,
            g,
            s,
            None,
            NormMethod::Tmm,
            None,
            None,
            &[
                1.09214845928324,
                0.881664551382546,
                1.03839386653584,
                1.0001216498394,
            ],
        );
    }

    #[test]
    fn test_part1_tmmwsp_matches_r() {
        let (x, g, s) = read_csv("test_data_part1.csv");
        check(
            &x,
            g,
            s,
            None,
            NormMethod::TmmwSp,
            None,
            None,
            &[
                1.11707573778036,
                0.875057108000525,
                1.03061167368031,
                0.992626639707476,
            ],
        );
    }

    #[test]
    fn test_part1_rle_matches_r() {
        let (x, g, s) = read_csv("test_data_part1.csv");
        check(
            &x,
            g,
            s,
            None,
            NormMethod::Rle,
            None,
            None,
            &[
                1.08551277222965,
                0.956217074652334,
                0.91748548770939,
                1.05004851312078,
            ],
        );
    }

    #[test]
    fn test_part1_upper_quartile_matches_r() {
        let (x, g, s) = read_csv("test_data_part1.csv");
        check(
            &x,
            g,
            s,
            None,
            NormMethod::UpperQuartile,
            None,
            None,
            &[
                0.917795511372727,
                1.20110064867884,
                0.913136650908906,
                0.993433714094184,
            ],
        );
    }

    #[test]
    fn test_part1_upper_quartile_at_the_median_matches_r() {
        let (x, g, s) = read_csv("test_data_part1.csv");
        let params = NormParams {
            quantile: 0.5,
            ..Default::default()
        };
        check(
            &x,
            g,
            s,
            None,
            NormMethod::UpperQuartile,
            None,
            Some(params),
            &[
                1.04104095814556,
                0.864593338120892,
                1.0357564862768,
                1.07266136107989,
            ],
        );
    }

    #[test]
    fn test_part1_none_is_all_ones() {
        let (x, g, s) = read_csv("test_data_part1.csv");
        let f = calc_norm_factors(&x, g, s, None, NormMethod::None, None, None).unwrap();
        assert_eq!(f, vec![1.0; 4]);
    }

    #[test]
    fn test_part1_tmm_with_an_explicit_reference_matches_r() {
        let (x, g, s) = read_csv("test_data_part1.csv");
        // R's refColumn = 1 and 3, zero-based here.
        check(
            &x,
            g,
            s,
            None,
            NormMethod::Tmm,
            Some(0),
            None,
            &[
                1.12488124746577,
                0.866533389465521,
                0.995933161741123,
                1.03009630195075,
            ],
        );
        check(
            &x,
            g,
            s,
            None,
            NormMethod::Tmm,
            Some(2),
            None,
            &[
                1.04654677428291,
                1.15554356186379,
                0.926578374534831,
                0.892427355851779,
            ],
        );
    }

    #[test]
    fn test_part1_tmm_with_custom_trims_matches_r() {
        let (x, g, s) = read_csv("test_data_part1.csv");
        let params = NormParams {
            logratio_trim: 0.2,
            sum_trim: 0.1,
            ..Default::default()
        };
        check(
            &x,
            g,
            s,
            None,
            NormMethod::Tmm,
            None,
            Some(params),
            &[
                1.13220983123328,
                0.969765698086251,
                0.916554640595877,
                0.993683089958524,
            ],
        );
    }

    #[test]
    fn test_part1_tmm_without_weighting_matches_r() {
        let (x, g, s) = read_csv("test_data_part1.csv");
        let params = NormParams {
            do_weighting: false,
            ..Default::default()
        };
        check(
            &x,
            g,
            s,
            None,
            NormMethod::Tmm,
            None,
            Some(params),
            &[
                1.11187880471527,
                0.87234553380242,
                0.996922506288526,
                1.03417159694611,
            ],
        );
    }

    #[test]
    fn test_part1_tmm_with_supplied_library_sizes_matches_r() {
        let (x, g, s) = read_csv("test_data_part1.csv");
        check(
            &x,
            g,
            s,
            Some(&[1000.0, 2000.0, 1500.0, 1200.0]),
            NormMethod::Tmm,
            None,
            None,
            &[
                1.44466458873241,
                0.720343929088734,
                0.855991617053877,
                1.12259618110301,
            ],
        );
    }

    #[test]
    fn test_part1_tmmwsp_variants_match_r() {
        let (x, g, s) = read_csv("test_data_part1.csv");
        let params = NormParams {
            do_weighting: false,
            ..Default::default()
        };
        check(
            &x,
            g,
            s,
            None,
            NormMethod::TmmwSp,
            None,
            Some(params),
            &[
                1.14857890712681,
                0.862953537941521,
                0.986189268494682,
                1.02303732161413,
            ],
        );
        check(
            &x,
            g,
            s,
            None,
            NormMethod::TmmwSp,
            Some(1),
            None,
            &[
                1.24531629866136,
                0.95930860926261,
                0.769226281876988,
                1.08819787022886,
            ],
        );
    }

    // -- R parity, tests/data/test_data_part2.csv, 1000 genes by 3 samples --

    #[test]
    fn test_part2_all_methods_match_r() {
        let (x, g, s) = read_csv("test_data_part2.csv");
        check(
            &x,
            g,
            s,
            None,
            NormMethod::Tmm,
            None,
            None,
            &[1.00087299037568, 1.00141716151291, 0.997713849403509],
        );
        check(
            &x,
            g,
            s,
            None,
            NormMethod::TmmwSp,
            None,
            None,
            &[0.999243678236565, 1.00770066012757, 0.993109297052868],
        );
        check(
            &x,
            g,
            s,
            None,
            NormMethod::Rle,
            None,
            None,
            &[1.00777567277904, 0.990633895481675, 1.00166603054436],
        );
        check(
            &x,
            g,
            s,
            None,
            NormMethod::UpperQuartile,
            None,
            None,
            &[0.952168937473405, 1.09472066633077, 0.959362356794534],
        );
    }

    #[test]
    fn test_part2_trim_and_weight_variants_match_r() {
        let (x, g, s) = read_csv("test_data_part2.csv");
        check(
            &x,
            g,
            s,
            None,
            NormMethod::Tmm,
            None,
            Some(NormParams {
                do_weighting: false,
                ..Default::default()
            }),
            &[0.998440905548369, 1.00708634636271, 0.99451405794568],
        );
        check(
            &x,
            g,
            s,
            None,
            NormMethod::TmmwSp,
            None,
            Some(NormParams {
                logratio_trim: 0.2,
                sum_trim: 0.1,
                ..Default::default()
            }),
            &[1.0147225769176, 0.998674566234331, 0.986798969146451],
        );
    }

    // -- R parity, small and sparse matrices generated for this test module --

    /// A 3 by 3 matrix with one dominant count, `matrix(1:9 with x[3,3] = 900)`.
    fn tiny() -> Vec<f64> {
        // Column-major in R, row-major here.
        vec![1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 900.0]
    }

    #[test]
    fn test_tiny_matrix_matches_r() {
        let x = tiny();
        check(
            &x,
            3,
            3,
            None,
            NormMethod::Tmm,
            None,
            None,
            &[1.31487541222201, 1.24518272726712, 0.610776484663282],
        );
        check(
            &x,
            3,
            3,
            None,
            NormMethod::TmmwSp,
            None,
            None,
            &[1.21548697491053, 1.45714699120165, 0.56460708913077],
        );
        check(
            &x,
            3,
            3,
            None,
            NormMethod::Rle,
            None,
            None,
            &[2.49099606900441, 3.51095868773469, 0.114340803167416],
        );
        check(
            &x,
            3,
            3,
            None,
            NormMethod::UpperQuartile,
            None,
            None,
            &[0.984518260707848, 0.866376069422906, 1.17238371242325],
        );
    }

    /// 20 genes by 3 samples of `rpois(60, 0.6)`, R `set.seed(7)`, row-major.
    ///
    /// Two thirds of the entries are zero, so this is the fixture that actually
    /// exercises TMMwsp's singleton pairing and the M-value tie-break: the
    /// non-zero counts are almost all 1, so the M-values come in long tied runs.
    fn sparse() -> Vec<f64> {
        vec![
            3.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 4.0, 1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 2.0, 1.0, 0.0,
            0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 2.0, 2.0, 0.0, 0.0, 2.0, 0.0, 1.0, 1.0, 0.0,
            0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        ]
    }

    #[test]
    fn test_sparse_matrix_matches_r() {
        let x = sparse();
        check(
            &x,
            20,
            3,
            None,
            NormMethod::Tmm,
            None,
            None,
            &[1.86860983013238, 0.718566958550494, 0.744756200069697],
        );
        check(
            &x,
            20,
            3,
            None,
            NormMethod::TmmwSp,
            None,
            None,
            &[1.01633695449891, 0.900162048144607, 1.09305391560417],
        );
        check(
            &x,
            20,
            3,
            None,
            NormMethod::TmmwSp,
            None,
            Some(NormParams {
                do_weighting: false,
                ..Default::default()
            }),
            &[1.06398557538257, 0.879775142410609, 1.06829838721288],
        );
        check(
            &x,
            20,
            3,
            None,
            NormMethod::Rle,
            None,
            None,
            &[2.60622175336887, 0.562126260530541, 0.682581887787085],
        );
        check(
            &x,
            20,
            3,
            None,
            NormMethod::UpperQuartile,
            None,
            None,
            &[1.25294073464069, 0.810726357708683, 0.984453434360544],
        );
    }

    // -- degenerate inputs --

    #[test]
    fn test_single_sample_degrades_to_no_normalisation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let f = calc_norm_factors(&x, 5, 1, None, NormMethod::Tmm, None, None).unwrap();
        assert_eq!(f, vec![1.0]);
    }

    #[test]
    fn test_all_zero_matrix_degrades_to_no_normalisation() {
        let x = vec![0.0_f64; 12];
        for method in [
            NormMethod::Tmm,
            NormMethod::TmmwSp,
            NormMethod::Rle,
            NormMethod::UpperQuartile,
        ] {
            let f = calc_norm_factors(&x, 4, 3, None, method, None, None);
            // Every library size is zero, so this is rejected before the method
            // ever runs. edgeR reaches the same input with lib.size supplied and
            // returns ones, which the next assertion pins.
            assert!(f.is_err());
        }
        let f = calc_norm_factors(
            &x,
            4,
            3,
            Some(&[10.0, 10.0, 10.0]),
            NormMethod::Tmm,
            None,
            None,
        )
        .unwrap();
        assert_eq!(f, vec![1.0; 3]);
    }

    #[test]
    fn test_proportional_columns_give_exactly_one() {
        // c(10..100) and twice that: M is constant, so TMM short-circuits.
        let mut x = Vec::new();
        for i in 1..=10 {
            x.push(10.0 * i as f64);
            x.push(20.0 * i as f64);
        }
        for method in [NormMethod::Tmm, NormMethod::TmmwSp, NormMethod::Rle] {
            let f = calc_norm_factors(&x, 10, 2, None, method, None, None).unwrap();
            assert_relative_eq!(f[0], 1.0, max_relative = 1e-14);
            assert_relative_eq!(f[1], 1.0, max_relative = 1e-14);
        }
    }

    #[test]
    fn test_identical_columns_give_exactly_one() {
        let mut x = Vec::new();
        for i in 1..=10 {
            x.push(i as f64);
            x.push(i as f64);
        }
        for method in [NormMethod::Tmm, NormMethod::TmmwSp] {
            let f = calc_norm_factors(&x, 10, 2, None, method, None, None).unwrap();
            assert_eq!(f, vec![1.0, 1.0]);
        }
    }

    #[test]
    fn test_factors_have_geometric_mean_one() {
        let (x, g, s) = read_csv("test_data_part1.csv");
        for method in [
            NormMethod::Tmm,
            NormMethod::TmmwSp,
            NormMethod::Rle,
            NormMethod::UpperQuartile,
            NormMethod::None,
        ] {
            let f = calc_norm_factors(&x, g, s, None, method, None, None).unwrap();
            let product: f64 = f.iter().product();
            assert_relative_eq!(product, 1.0, max_relative = 1e-12);
        }
    }

    // -- reference column selection --

    #[test]
    fn test_reference_column_is_the_one_nearest_the_mean_f75() {
        let (x, g, s) = read_csv("test_data_part1.csv");
        let lib = resolve_lib_size(&x, g, s, None).unwrap();
        let (cm, n_kept) = to_column_major_nonzero(&x, g, s).unwrap();
        // R: f75 = 0.0561, 0.0734, 0.0558, 0.0607; which.min(abs(f75 - mean)) = 4.
        let chosen = resolve_ref_column(&cm, n_kept, s, &lib, NormMethod::Tmm).unwrap();
        assert_eq!(chosen, 3);
        // TMMwsp uses the root-count rule unconditionally.
        let wsp = resolve_ref_column(&cm, n_kept, s, &lib, NormMethod::TmmwSp).unwrap();
        assert_eq!(wsp, arg_max_sqrt_colsum(&cm, n_kept, s));
    }

    #[test]
    fn test_explicit_reference_column_overrides_the_heuristic() {
        let (x, g, s) = read_csv("test_data_part1.csv");
        let auto = calc_norm_factors(&x, g, s, None, NormMethod::Tmm, None, None).unwrap();
        let pinned = calc_norm_factors(&x, g, s, None, NormMethod::Tmm, Some(3), None).unwrap();
        assert_eq!(auto, pinned);
    }

    // -- generic float and layout --

    #[test]
    fn test_f32_counts_agree_with_f64() {
        let (x, g, s) = read_csv("test_data_part1.csv");
        let x32: Vec<f32> = x.iter().map(|v| *v as f32).collect();
        for method in [
            NormMethod::Tmm,
            NormMethod::TmmwSp,
            NormMethod::Rle,
            NormMethod::UpperQuartile,
        ] {
            let a = calc_norm_factors(&x, g, s, None, method, None, None).unwrap();
            let b = calc_norm_factors(&x32, g, s, None, method, None, None).unwrap();
            for (p, q) in a.iter().zip(b.iter()) {
                assert_relative_eq!(p, q, max_relative = 1e-12);
            }
        }
    }

    #[test]
    fn test_transpose_drops_only_all_zero_genes() {
        // Three genes, two samples; the middle gene is all zero.
        let x = vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0];
        let (cm, n_kept) = to_column_major_nonzero(&x, 3, 2).unwrap();
        assert_eq!(n_kept, 2);
        assert_eq!(cm, vec![1.0, 3.0, 2.0, 4.0]);
    }

    // -- error paths --

    #[test]
    fn test_rejects_an_out_of_range_reference_column() {
        let x = vec![1.0_f64; 6];
        let err = calc_norm_factors(&x, 3, 2, None, NormMethod::Tmm, Some(2), None).unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidReferenceColumn(_)));
    }

    #[test]
    fn test_rejects_a_library_size_of_the_wrong_length() {
        let x = vec![1.0_f64; 6];
        let err =
            calc_norm_factors(&x, 3, 2, Some(&[10.0]), NormMethod::Tmm, None, None).unwrap_err();
        assert!(matches!(
            err,
            EdgeErrors::LengthMismatch {
                name: "lib_size",
                ..
            }
        ));
    }

    #[test]
    fn test_rejects_a_dead_sample() {
        // Sample two has no counts at all, so its library size is zero.
        let x = vec![5.0, 0.0, 4.0, 7.0, 0.0, 6.0, 3.0, 0.0, 2.0, 9.0, 0.0, 8.0];
        let err = calc_norm_factors(&x, 4, 3, None, NormMethod::Tmm, None, None).unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));
    }

    #[test]
    fn test_rejects_a_shape_mismatch_and_empty_counts() {
        let err =
            calc_norm_factors(&[1.0_f64; 5], 3, 2, None, NormMethod::Tmm, None, None).unwrap_err();
        assert!(matches!(
            err,
            EdgeErrors::LengthMismatch { name: "counts", .. }
        ));
        let err =
            calc_norm_factors::<f64>(&[], 0, 2, None, NormMethod::Tmm, None, None).unwrap_err();
        assert!(matches!(err, EdgeErrors::EmptyCounts { .. }));
    }

    #[test]
    fn test_rejects_negative_counts() {
        let x = vec![1.0, 2.0, -3.0, 4.0];
        let err = calc_norm_factors(&x, 2, 2, None, NormMethod::Tmm, None, None).unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));
    }

    #[test]
    fn test_rejects_parameters_outside_their_domain() {
        let x = vec![1.0_f64; 6];
        for params in [
            NormParams {
                logratio_trim: 1.0,
                ..Default::default()
            },
            NormParams {
                sum_trim: -0.1,
                ..Default::default()
            },
            NormParams {
                quantile: 1.5,
                ..Default::default()
            },
        ] {
            let err =
                calc_norm_factors(&x, 3, 2, None, NormMethod::Tmm, None, Some(params)).unwrap_err();
            assert!(matches!(err, EdgeErrors::InvalidArgument(_)));
        }
    }

    #[test]
    fn test_upper_quartile_rejects_a_zero_quantile() {
        // Eight genes, two samples. Every gene survives the all-zero filter on
        // the strength of sample two, but sample one is zero in seven of eight,
        // so its upper quartile is zero and the factor would come back NaN.
        let mut x = vec![0.0_f64; 16];
        for g in 0..8 {
            x[g * 2 + 1] = 1.0;
        }
        x[7 * 2] = 10.0;
        let err =
            calc_norm_factors(&x, 8, 2, None, NormMethod::UpperQuartile, None, None).unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));
    }

    #[test]
    fn test_rle_rejects_a_matrix_with_no_gene_positive_everywhere() {
        // Every gene is zero in at least one sample, so every geometric mean is
        // zero and the median ratio is undefined.
        let x = vec![1.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 4.0];
        let err = calc_norm_factors(&x, 4, 2, None, NormMethod::Rle, None, None).unwrap_err();
        assert!(matches!(err, EdgeErrors::NoGenesAfterFiltering { .. }));
    }
}
