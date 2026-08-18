//! `cpm`, `rpkm`, `tpm` and `aveLogCPM`.
//!
//! The per-gene expression summaries edgeR hangs off a count matrix. All four
//! are a division by a library size, so the only interesting question is which
//! library size: an explicit `lib_size`, the column sums of the counts, or
//! `exp(offset)` when offsets are supplied. Offsets win outright, as in edgeR,
//! where `lib.size` is silently discarded in their presence.
//!
//! Library sizes are carried as a [`Recycled`] rather than materialised. A
//! per-sample library size stays `n_samples` values all the way into the inner
//! loop, and only a genuinely gene-varying offset ever costs
//! `n_genes * n_samples`.
//!
//! [`ave_log_cpm`] is the one that matters for speed: `estimateDisp` calls it on
//! every gene of every analysis. It adds a library-size-scaled prior count and
//! then fits an intercept-only negative binomial GLM with
//! [`mglm_one_group`], rather than averaging log-CPMs, which is what keeps it
//! stable for genes that are zero in most samples.
//!
//! Genes are the parallel axis throughout, per the crate rule: one gene is a
//! contiguous row of the input and a contiguous row of the output.

use std::f64::consts::LN_2;

use rayon::prelude::*;

use crate::errors::EdgeErrors;
use crate::glm::fit::add_prior_count;
use crate::glm::one_group::mglm_one_group;
use crate::utils::recycled::{Recycled, RecycledRow};
use crate::utils::traits::EdgeFloat;

///////////////
// Constants //
///////////////

/// Counts per million: the scale every summary in this module reports on.
pub(crate) const PER_MILLION: f64 = 1e6;

/// Bases per kilobase, the unit RPKM normalises gene length to.
const BASES_PER_KB: f64 = 1000.0;

/// Dispersion `aveLogCPM` assumes when the caller has not estimated one yet.
///
/// edgeR's default. The abundance *ranking* is close to insensitive to it, and
/// ranking is all `estimateDisp` uses the result for when it bins genes by
/// abundance, so a fixed guess is good enough to start from.
const DEFAULT_DISPERSION: f64 = 0.05;

////////////////
// Validation //
////////////////

/// Checks a count matrix against its stated shape.
///
/// ### Params
///
/// * `counts` - Row-major counts, `n_genes * n_samples`
/// * `n_genes` - Number of genes
/// * `n_samples` - Number of samples
///
/// ### Returns
///
/// `Ok(())`, or [`EdgeErrors::EmptyCounts`] for a zero dimension,
/// [`EdgeErrors::LengthMismatch`] for the wrong length, or
/// [`EdgeErrors::InvalidArgument`] for a negative or non-finite count.
pub(crate) fn check_counts<T: EdgeFloat>(
    counts: &[T],
    n_genes: usize,
    n_samples: usize,
) -> Result<(), EdgeErrors> {
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
    if let Some(bad) = counts.iter().find(|v| !v.is_finite() || **v < T::zero()) {
        return Err(EdgeErrors::InvalidArgument(format!(
            "counts must be non-negative and finite, found {bad}"
        )));
    }
    Ok(())
}

/// Column sums of a row-major matrix.
///
/// ### Params
///
/// * `values` - Row-major values, a whole number of rows of `n_samples`
/// * `n_samples` - Number of columns
///
/// ### Returns
///
/// One sum per column, in `f64` regardless of `T`.
pub(crate) fn column_sums<T: EdgeFloat>(values: &[T], n_samples: usize) -> Vec<f64> {
    let mut sums = vec![0.0_f64; n_samples];
    for row in values.chunks_exact(n_samples) {
        for (acc, v) in sums.iter_mut().zip(row.iter()) {
            *acc += v.to_f64().unwrap_or(f64::NAN);
        }
    }
    sums
}

/// Rejects a prior count that cannot be scaled.
///
/// ### Params
///
/// * `prior_count` - Prior count before library-size scaling
///
/// ### Returns
///
/// `Ok(())`, or [`EdgeErrors::InvalidArgument`] if it is negative or non-finite.
fn check_prior_count(prior_count: f64) -> Result<(), EdgeErrors> {
    if !prior_count.is_finite() || prior_count < 0.0 {
        return Err(EdgeErrors::InvalidArgument(format!(
            "prior_count must be non-negative and finite, got {prior_count}"
        )));
    }
    Ok(())
}

/// Rejects library sizes that cannot be divided by.
///
/// Only the stored values are examined, so a `Scalar` costs one comparison
/// rather than `n_genes * n_samples`.
///
/// ### Params
///
/// * `lib_sizes` - Library sizes, recycled over genes and samples
///
/// ### Returns
///
/// `Ok(())`, or [`EdgeErrors::InvalidArgument`] if any entry is not strictly
/// positive and finite, as in edgeR's "library sizes should be greater than
/// zero".
fn check_lib_sizes(lib_sizes: &Recycled<f64>) -> Result<(), EdgeErrors> {
    let stored: &[f64] = match lib_sizes {
        Recycled::Scalar(v) => std::slice::from_ref(v),
        Recycled::ByGene(v) | Recycled::BySample(v) | Recycled::Full(v) => v,
    };
    if let Some(bad) = stored.iter().find(|v| !v.is_finite() || **v <= 0.0) {
        return Err(EdgeErrors::InvalidArgument(format!(
            "library sizes must be positive and finite, found {bad}"
        )));
    }
    Ok(())
}

/// Resolves the library sizes every summary here divides by.
///
/// edgeR's precedence, from `cpm.default`: an offset replaces the library sizes
/// outright (`lib.size <- exp(offset)`), an explicit `lib_size` comes next, and
/// the column sums are the fallback.
///
/// ### Params
///
/// * `counts` - Row-major counts, `n_genes * n_samples`
/// * `n_genes` - Number of genes
/// * `n_samples` - Number of samples
/// * `lib_size` - Library size per sample, or `None` for the column sums
/// * `offset` - Log-scale offsets, which override `lib_size` when present
///
/// ### Returns
///
/// The library sizes on the natural scale, in the same recycled form the offset
/// arrived in, or [`EdgeErrors`] if a length disagrees or a size is not
/// positive.
fn resolve_lib_sizes<T: EdgeFloat>(
    counts: &[T],
    n_genes: usize,
    n_samples: usize,
    lib_size: Option<&[f64]>,
    offset: Option<&Recycled<f64>>,
) -> Result<Recycled<f64>, EdgeErrors> {
    let lib_sizes = match offset {
        Some(o) => {
            o.validate(n_genes, n_samples)?;
            match o {
                Recycled::Scalar(v) => Recycled::Scalar(v.exp()),
                Recycled::ByGene(v) => Recycled::ByGene(v.iter().map(|x| x.exp()).collect()),
                Recycled::BySample(v) => Recycled::BySample(v.iter().map(|x| x.exp()).collect()),
                Recycled::Full(v) => Recycled::Full(v.iter().map(|x| x.exp()).collect()),
            }
        }
        None => match lib_size {
            Some(l) => {
                if l.len() != n_samples {
                    return Err(EdgeErrors::LengthMismatch {
                        name: "lib_size",
                        expected: n_samples,
                        got: l.len(),
                    });
                }
                Recycled::BySample(l.to_vec())
            }
            None => Recycled::BySample(column_sums(counts, n_samples)),
        },
    };

    check_lib_sizes(&lib_sizes)?;
    Ok(lib_sizes)
}

//////////////////
// Prior counts //
//////////////////

/// Adds library-size-scaled prior counts to one gene.
///
/// edgeR's `addPriorCount` rule, per gene: the prior is scaled by each library's
/// size relative to the mean *of that gene's row*, so every sample is perturbed
/// by the same relative amount. With per-sample library sizes the row mean is
/// the same for every gene and this collapses to the usual formula; with a
/// gene-varying offset it does not, and edgeR really does use the row mean.
///
/// ### Params
///
/// * `y` - Counts for this gene, `n_samples` long
/// * `lib` - Library size row for this gene
/// * `prior_count` - Prior count before scaling
/// * `augmented` - Filled with `y + prior * lib / mean(lib)`
/// * `adjusted` - Filled with `lib + 2 * prior * lib / mean(lib)`
#[inline]
fn add_prior_count_row<T: EdgeFloat>(
    y: &[T],
    lib: RecycledRow<'_, f64>,
    prior_count: f64,
    augmented: &mut [f64],
    adjusted: &mut [f64],
) {
    let n_samples = y.len();
    let mut total = 0.0;
    for j in 0..n_samples {
        total += lib.get(j);
    }
    let mean_lib = total / n_samples as f64;

    for j in 0..n_samples {
        let size = lib.get(j);
        let prior = prior_count * size / mean_lib;
        augmented[j] = y[j].to_f64().unwrap_or(f64::NAN) + prior;
        adjusted[j] = size + 2.0 * prior;
    }
}

/// Adds prior counts to the whole matrix and reports the matching log offsets.
///
/// The per-sample case is handed to [`add_prior_count`], which already
/// implements exactly this rule; only a gene-varying library size needs the
/// per-gene loop, and only that case pays `n_genes * n_samples` for the offsets.
///
/// ### Params
///
/// * `counts` - Row-major counts, `n_genes * n_samples`
/// * `n_genes` - Number of genes
/// * `n_samples` - Number of samples
/// * `lib_sizes` - Library sizes on the natural scale
/// * `prior_count` - Prior count before scaling
///
/// ### Returns
///
/// The augmented counts, row-major, and the log of the adjusted library sizes.
fn augment_counts<T: EdgeFloat>(
    counts: &[T],
    n_genes: usize,
    n_samples: usize,
    lib_sizes: &Recycled<f64>,
    prior_count: f64,
) -> Result<(Vec<f64>, Recycled<f64>), EdgeErrors> {
    if let Recycled::BySample(libs) = lib_sizes {
        let (augmented, offsets) = add_prior_count(counts, n_genes, n_samples, libs, prior_count)?;
        return Ok((augmented, Recycled::by_sample(offsets)));
    }

    let mut augmented = vec![0.0_f64; n_genes * n_samples];
    let mut offsets = vec![0.0_f64; n_genes * n_samples];
    augmented
        .par_chunks_mut(n_samples)
        .zip(offsets.par_chunks_mut(n_samples))
        .enumerate()
        .for_each(|(gene, (y_out, offset_out))| {
            let y = &counts[gene * n_samples..(gene + 1) * n_samples];
            add_prior_count_row(
                y,
                lib_sizes.row(gene, n_samples),
                prior_count,
                y_out,
                offset_out,
            );
            for v in offset_out.iter_mut() {
                *v = v.ln();
            }
        });

    Ok((augmented, Recycled::Full(offsets)))
}

///////////////
// Front end //
///////////////

/// Counts per million, as edgeR's `cpm`.
///
/// On the log scale a library-size-scaled prior count is added first, which is
/// what stops a zero count reporting minus infinity and damps the variance of
/// low-count genes.
///
/// ### Params
///
/// * `counts` - Row-major counts, `n_genes * n_samples`
/// * `n_genes` - Number of genes
/// * `n_samples` - Number of samples
/// * `lib_size` - Library size per sample. `None` uses the column sums. Ignored
///   when `offset` is given, as in edgeR.
/// * `offset` - Log-scale offsets. When present the library sizes are
///   `exp(offset)`, which is the only way to get a gene-varying denominator.
/// * `log` - Return log2-CPM rather than CPM
/// * `prior_count` - Prior count before library-size scaling. Only used when
///   `log` is set; edgeR's default is 2.
///
/// ### Returns
///
/// Row-major `n_genes * n_samples` values, or [`EdgeErrors`] if a shape
/// disagrees, a count is negative, or a library size is not positive.
pub fn cpm<T: EdgeFloat>(
    counts: &[T],
    n_genes: usize,
    n_samples: usize,
    lib_size: Option<&[f64]>,
    offset: Option<&Recycled<f64>>,
    log: bool,
    prior_count: f64,
) -> Result<Vec<f64>, EdgeErrors> {
    check_counts(counts, n_genes, n_samples)?;
    if log {
        check_prior_count(prior_count)?;
    }
    let lib_sizes = resolve_lib_sizes(counts, n_genes, n_samples, lib_size, offset)?;

    let mut out = vec![0.0_f64; n_genes * n_samples];
    if log {
        out.par_chunks_mut(n_samples).enumerate().for_each_init(
            || vec![0.0_f64; n_samples],
            |adjusted, (gene, row)| {
                let y = &counts[gene * n_samples..(gene + 1) * n_samples];
                add_prior_count_row(
                    y,
                    lib_sizes.row(gene, n_samples),
                    prior_count,
                    row,
                    adjusted,
                );
                for (v, size) in row.iter_mut().zip(adjusted.iter()) {
                    *v = (*v / size * PER_MILLION).log2();
                }
            },
        );
    } else {
        out.par_chunks_mut(n_samples)
            .enumerate()
            .for_each(|(gene, row)| {
                let y = &counts[gene * n_samples..(gene + 1) * n_samples];
                let lib = lib_sizes.row(gene, n_samples);
                for (j, v) in row.iter_mut().enumerate() {
                    *v = y[j].to_f64().unwrap_or(f64::NAN) / lib.get(j) * PER_MILLION;
                }
            });
    }

    Ok(out)
}

/// Reads per kilobase of gene length per million reads, as edgeR's `rpkm`.
///
/// [`cpm`] divided by the gene length in kilobases, or minus its log2 when `log`
/// is set.
///
/// ### Params
///
/// * `counts` - Row-major counts, `n_genes * n_samples`
/// * `n_genes` - Number of genes
/// * `n_samples` - Number of samples
/// * `gene_length` - Gene length in bases, one per gene. Must be positive.
/// * `lib_size` - Library size per sample. `None` uses the column sums.
/// * `offset` - Log-scale offsets, which override `lib_size`
/// * `log` - Return log2-RPKM rather than RPKM
/// * `prior_count` - Prior count before library-size scaling; `log` only
///
/// ### Returns
///
/// Row-major `n_genes * n_samples` values, or [`EdgeErrors`] as for [`cpm`],
/// plus [`EdgeErrors::LengthMismatch`] if `gene_length` is not one per gene.
#[allow(clippy::too_many_arguments)]
pub fn rpkm<T: EdgeFloat>(
    counts: &[T],
    n_genes: usize,
    n_samples: usize,
    gene_length: &[f64],
    lib_size: Option<&[f64]>,
    offset: Option<&Recycled<f64>>,
    log: bool,
    prior_count: f64,
) -> Result<Vec<f64>, EdgeErrors> {
    check_counts(counts, n_genes, n_samples)?;
    check_gene_length(gene_length, n_genes, "gene_length")?;

    let mut out = cpm(
        counts,
        n_genes,
        n_samples,
        lib_size,
        offset,
        log,
        prior_count,
    )?;
    out.par_chunks_mut(n_samples)
        .zip(gene_length.par_iter())
        .for_each(|(row, length)| {
            let kilobases = length / BASES_PER_KB;
            if log {
                let shift = kilobases.log2();
                for v in row.iter_mut() {
                    *v -= shift;
                }
            } else {
                for v in row.iter_mut() {
                    *v /= kilobases;
                }
            }
        });

    Ok(out)
}

/// Transcripts per million, as edgeR's `tpm`.
///
/// CPM divided by the effective transcript length, then rescaled so the column
/// totals are one million on average. edgeR uses the *geometric* mean of the
/// column totals for that rescaling, not the arithmetic one, so the result is
/// invariant to a per-sample scale factor.
///
/// ### Params
///
/// * `counts` - Row-major counts, `n_genes * n_samples`
/// * `n_genes` - Number of genes
/// * `n_samples` - Number of samples
/// * `effective_length` - Effective transcript length per gene. Must be
///   positive. Unlike [`rpkm`] this is used in bases, not kilobases: the
///   rescaling absorbs any constant factor.
///
/// ### Returns
///
/// Row-major `n_genes * n_samples` values, or [`EdgeErrors`] if a shape
/// disagrees or a whole sample ends up empty, which would make the geometric
/// mean undefined.
pub fn tpm<T: EdgeFloat>(
    counts: &[T],
    n_genes: usize,
    n_samples: usize,
    effective_length: &[f64],
) -> Result<Vec<f64>, EdgeErrors> {
    check_counts(counts, n_genes, n_samples)?;
    check_gene_length(effective_length, n_genes, "effective_length")?;

    let mut out = cpm(counts, n_genes, n_samples, None, None, false, 0.0)?;
    out.par_chunks_mut(n_samples)
        .zip(effective_length.par_iter())
        .for_each(|(row, length)| {
            for v in row.iter_mut() {
                *v /= length;
            }
        });

    let totals = column_sums(&out, n_samples);
    if let Some(bad) = totals.iter().find(|v| !v.is_finite() || **v <= 0.0) {
        return Err(EdgeErrors::InvalidArgument(format!(
            "tpm needs every sample to have a positive length-adjusted total, found {bad}"
        )));
    }
    let geometric_mean = (totals.iter().map(|v| v.ln()).sum::<f64>() / n_samples as f64).exp();

    out.par_iter_mut()
        .for_each(|v| *v = *v / geometric_mean * PER_MILLION);
    Ok(out)
}

/// Average log2-CPM per gene, as edgeR's `aveLogCPM`.
///
/// Not a mean of [`cpm`] values. edgeR adds a library-size-scaled prior count
/// and fits an intercept-only negative binomial GLM per gene with
/// [`mglm_one_group`], then reports the coefficient as log2-CPM. The GLM weights
/// samples by their information rather than equally, which is what makes the
/// answer usable for genes that are zero in most samples, and it is the
/// abundance covariate the whole dispersion trend is built on.
///
/// ### Params
///
/// * `counts` - Row-major counts, `n_genes * n_samples`
/// * `n_genes` - Number of genes
/// * `n_samples` - Number of samples
/// * `lib_size` - Library size per sample. `None` uses the column sums.
/// * `offset` - Log-scale offsets, which override `lib_size`
/// * `prior_count` - Prior count before library-size scaling; edgeR's default
///   is 2
/// * `dispersion` - Dispersion, recycled over genes and samples. `None` uses
///   [`DEFAULT_DISPERSION`].
///
/// ### Returns
///
/// One log2-CPM per gene, or [`EdgeErrors`] if a shape disagrees, a library size
/// is not positive, or a dispersion is negative.
pub fn ave_log_cpm<T: EdgeFloat>(
    counts: &[T],
    n_genes: usize,
    n_samples: usize,
    lib_size: Option<&[f64]>,
    offset: Option<&Recycled<f64>>,
    prior_count: f64,
    dispersion: Option<&Recycled<f64>>,
) -> Result<Vec<f64>, EdgeErrors> {
    check_counts(counts, n_genes, n_samples)?;
    check_prior_count(prior_count)?;

    // edgeR's escape hatch for a matrix that is empty and carries no library
    // information either: there is no abundance to estimate, so it reports the
    // CPM of one count spread over every gene rather than log(0).
    if lib_size.is_none() && offset.is_none() && counts.iter().all(|v| *v == T::zero()) {
        let abundance = -(n_genes as f64).ln();
        return Ok(vec![(abundance + PER_MILLION.ln()) / LN_2; n_genes]);
    }

    let fallback = Recycled::scalar(DEFAULT_DISPERSION);
    let dispersion = match dispersion {
        Some(d) => {
            d.validate(n_genes, n_samples)?;
            check_dispersion(d)?;
            d
        }
        None => &fallback,
    };

    let lib_sizes = resolve_lib_sizes(counts, n_genes, n_samples, lib_size, offset)?;
    let (augmented, offsets) = augment_counts(counts, n_genes, n_samples, &lib_sizes, prior_count)?;

    let coefficients = mglm_one_group(
        &augmented, n_genes, n_samples, dispersion, &offsets, None, None, None,
    )?;

    let shift = PER_MILLION.ln();
    Ok(coefficients.iter().map(|a| (a + shift) / LN_2).collect())
}

/// Rejects gene lengths that cannot be divided by.
///
/// ### Params
///
/// * `lengths` - One length per gene
/// * `n_genes` - Number of genes
/// * `name` - Argument name, for the error message
///
/// ### Returns
///
/// `Ok(())`, or [`EdgeErrors::LengthMismatch`] on the wrong length and
/// [`EdgeErrors::InvalidArgument`] on a non-positive entry.
fn check_gene_length(
    lengths: &[f64],
    n_genes: usize,
    name: &'static str,
) -> Result<(), EdgeErrors> {
    if lengths.len() != n_genes {
        return Err(EdgeErrors::LengthMismatch {
            name,
            expected: n_genes,
            got: lengths.len(),
        });
    }
    if let Some(bad) = lengths.iter().find(|v| !v.is_finite() || **v <= 0.0) {
        return Err(EdgeErrors::InvalidArgument(format!(
            "{name} must be positive and finite, found {bad}"
        )));
    }
    Ok(())
}

/// Rejects a dispersion outside `[0, inf)`.
///
/// Only the stored values are examined, as in [`check_lib_sizes`].
///
/// ### Params
///
/// * `dispersion` - Dispersion, recycled over genes and samples
///
/// ### Returns
///
/// `Ok(())`, or [`EdgeErrors::InvalidDispersion`] naming the offending value.
fn check_dispersion(dispersion: &Recycled<f64>) -> Result<(), EdgeErrors> {
    let stored: &[f64] = match dispersion {
        Recycled::Scalar(v) => std::slice::from_ref(v),
        Recycled::ByGene(v) | Recycled::BySample(v) | Recycled::Full(v) => v,
    };
    match stored.iter().find(|v| !v.is_finite() || **v < 0.0) {
        Some(bad) => Err(EdgeErrors::InvalidDispersion(*bad)),
        None => Ok(()),
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    // Every reference below is pasted verbatim from R's 17-digit output. The
    // last digit or two is past what an f64 can hold and clippy would rather
    // they were rounded, but keeping them exactly as printed is what makes them
    // checkable against the `Rscript` line quoted above each test.
    #![allow(clippy::excessive_precision)]

    use super::*;
    use approx::assert_relative_eq;

    /// 3 genes by 6 samples, the fixture every reference below was generated
    /// from:
    /// ```r
    /// y <- matrix(c(10,12,11,40,44,38, 50,48,52,49,51,50, 2,0,5,1,3,0),
    ///             nrow = 3, byrow = TRUE)
    /// ```
    const COUNTS: [f64; 18] = [
        10.0, 12.0, 11.0, 40.0, 44.0, 38.0, //
        50.0, 48.0, 52.0, 49.0, 51.0, 50.0, //
        2.0, 0.0, 5.0, 1.0, 3.0, 0.0, //
    ];

    /// `ls <- c(1e6, 1.2e6, 0.9e6, 1.1e6, 1e6, 1.3e6)`
    const LIB: [f64; 6] = [1e6, 1.2e6, 0.9e6, 1.1e6, 1e6, 1.3e6];

    /// The same fixture with gene 2 zeroed out.
    /// ```r
    /// yz <- matrix(c(10,12,11,40,44,38, 0,0,0,0,0,0, 2,0,5,1,3,0),
    ///              nrow = 3, byrow = TRUE)
    /// ```
    const COUNTS_WITH_ZERO_GENE: [f64; 18] = [
        10.0, 12.0, 11.0, 40.0, 44.0, 38.0, //
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
        2.0, 0.0, 5.0, 1.0, 3.0, 0.0, //
    ];

    /// One sample, four genes: `ys <- matrix(c(5, 0, 100, 3), nrow = 4)`.
    const COUNTS_SINGLE: [f64; 4] = [5.0, 0.0, 100.0, 3.0];

    /// A gene-varying offset. Gene `g` gets `log(ls) + c(0, 0.1, -0.2)[g]`.
    /// ```r
    /// off <- matrix(log(rep(ls, each = 3)), 3, 6) + matrix(c(0, 0.1, -0.2), 3, 6)
    /// ```
    fn varying_offset() -> Recycled<f64> {
        let shifts = [0.0, 0.1, -0.2];
        let mut values = Vec::with_capacity(18);
        for shift in shifts {
            for l in LIB {
                values.push(l.ln() + shift);
            }
        }
        Recycled::full(values, 3, 6).unwrap()
    }

    /// Agreement for the closed-form summaries. These are the same arithmetic
    /// edgeR does in a different order, so the only gap is rounding; the worst
    /// observed here is 3e-14 relative, on the smallest log2-CPM.
    fn assert_close(got: &[f64], want: &[f64]) {
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want.iter()) {
            assert_relative_eq!(g, w, epsilon = 1e-14, max_relative = 1e-13);
        }
    }

    /// Agreement for [`ave_log_cpm`], which ends in a Fisher scoring iteration.
    /// Both sides stop at the same `tol = 1e-10` on the coefficient step, so the
    /// gap is the convergence tolerance rather than rounding: the worst observed
    /// here is 1.4e-11 relative, and 1.3e-11 absolute on the log2 scale.
    fn assert_close_fit(got: &[f64], want: &[f64]) {
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want.iter()) {
            assert_relative_eq!(g, w, epsilon = 1e-10, max_relative = 1e-10);
        }
    }

    // -- cpm --

    /// `cat(format(cpm(y), digits = 17), sep = ", ")`
    #[test]
    fn test_cpm_matches_edger_with_derived_library_sizes() {
        let want = [
            161290.322580645152,
            200000.0,
            161764.705882352951,
            444444.444444444438,
            448979.591836734675,
            431818.181818181823,
            806451.612903225818,
            800000.0,
            764705.882352941204,
            544444.444444444496,
            520408.163265306095,
            568181.818181818235,
            32258.064516129034,
            0.0,
            73529.411764705888,
            11111.111111111111,
            30612.244897959183,
            0.0,
        ];
        let got = cpm(&COUNTS, 3, 6, None, None, false, 0.0).unwrap();
        assert_close(&got, &want);
    }

    /// `cat(format(cpm(y, lib.size = ls), digits = 17), sep = ", ")`
    #[test]
    fn test_cpm_matches_edger_with_explicit_library_sizes() {
        let want = [
            10.0,
            10.0,
            12.222222222222221,
            36.363636363636367,
            44.0,
            29.230769230769230,
            50.0,
            40.0,
            57.777777777777779,
            44.545454545454547,
            51.0,
            38.461538461538460,
            2.0,
            0.0,
            5.5555555555555554,
            0.90909090909090906,
            3.0,
            0.0,
        ];
        let got = cpm(&COUNTS, 3, 6, Some(&LIB), None, false, 0.0).unwrap();
        assert_close(&got, &want);
    }

    /// `cat(format(cpm(y, log = TRUE), digits = 17), sep = ", ")`
    #[test]
    fn test_cpm_log_matches_edger_with_derived_library_sizes() {
        let want = [
            17.440546279260413,
            17.711921524726545,
            17.444200678906679,
            18.770449465257201,
            18.784297922447628,
            18.731178797180288,
            19.594123624656348,
            19.582895619204283,
            19.519875582209934,
            19.048645377656534,
            18.986510319551922,
            19.107488609288882,
            15.751540753634080,
            14.579893131042759,
            16.526770744486953,
            15.097402137916776,
            15.710017124427095,
            14.579893131042759,
        ];
        let got = cpm(&COUNTS, 3, 6, None, None, true, 2.0).unwrap();
        assert_close(&got, &want);
    }

    /// `cat(format(cpm(y, lib.size = ls, log = TRUE, prior.count = 0.125),`
    /// `digits = 17), sep = ", ")`
    #[test]
    fn test_cpm_log_matches_edger_at_prior_count_one_eighth() {
        let want = [
            3.3384789382216224,
            3.3384789382216224,
            3.6249903397229910,
            5.1889947730485728,
            5.4632096249451472,
            4.8750991957638519,
            5.6471813174974557,
            5.3260833914490862,
            5.8553207273744832,
            5.4809381210851829,
            5.6756853382054953,
            5.2696658395415001,
            1.0809196624539799,
            -3.1154775503495236,
            2.5035876028572019,
            0.034885184714270906,
            1.6394099518139453,
            -3.1154775503495236,
        ];
        let got = cpm(&COUNTS, 3, 6, Some(&LIB), None, true, 0.125).unwrap();
        assert_close(&got, &want);
    }

    /// `cat(format(cpm(y, lib.size = ls, log = TRUE, prior.count = 1),`
    /// `digits = 17), sep = ", ")`
    #[test]
    fn test_cpm_log_matches_edger_at_prior_count_one() {
        let want = [
            3.4493047379290478,
            3.4493047379290505,
            3.7164724049438820,
            5.2205870648760433,
            5.4893821773043845,
            4.9142674625395761,
            5.6702450252315844,
            5.3548400539255594,
            5.8753069409841885,
            5.5067938404774264,
            5.6983013103625604,
            5.2995576184243678,
            1.5474851318679521,
            -0.11547988085447503,
            2.6956866551531466,
            0.87354900132451929,
            1.9719829603958630,
            -0.11547988085447503,
        ];
        let got = cpm(&COUNTS, 3, 6, Some(&LIB), None, true, 1.0).unwrap();
        assert_close(&got, &want);
    }

    /// `cat(format(cpm(y, lib.size = ls, log = TRUE, prior.count = 2),`
    /// `digits = 17), sep = ", ")`
    #[test]
    fn test_cpm_log_matches_edger_at_prior_count_two() {
        let want = [
            3.5663414956896449,
            3.5663414956896449,
            3.8143785739740554,
            5.2558651046475378,
            5.5187234754569063,
            4.9577664377465398,
            5.6961597361766039,
            5.3870177962450843,
            5.8978142783293563,
            5.5357866331454702,
            5.7237212438128440,
            5.3329779565321962,
            1.9434111447694709,
            0.88451745571590135,
            2.8878531682791548,
            1.4621751558138234,
            2.2768348784946606,
            0.88451745571590135,
        ];
        let got = cpm(&COUNTS, 3, 6, Some(&LIB), None, true, 2.0).unwrap();
        assert_close(&got, &want);
    }

    /// A per-sample offset is the same thing as `lib.size` on the log scale.
    /// `cat(format(cpm(y, offset = log(ls)), digits = 17), sep = ", ")`
    #[test]
    fn test_cpm_with_a_per_sample_offset_matches_the_library_sizes() {
        let offset = Recycled::by_sample(LIB.iter().map(|l| l.ln()).collect());
        let got = cpm(&COUNTS, 3, 6, None, Some(&offset), false, 0.0).unwrap();
        let want = cpm(&COUNTS, 3, 6, Some(&LIB), None, false, 0.0).unwrap();
        assert_close(&got, &want);
    }

    /// An offset overrides `lib_size` outright, as in edgeR.
    #[test]
    fn test_cpm_offset_overrides_the_library_sizes() {
        let offset = Recycled::by_sample(LIB.iter().map(|l| l.ln()).collect());
        let bogus = [1.0_f64; 6];
        let got = cpm(&COUNTS, 3, 6, Some(&bogus), Some(&offset), false, 0.0).unwrap();
        let want = cpm(&COUNTS, 3, 6, Some(&LIB), None, false, 0.0).unwrap();
        assert_close(&got, &want);
    }

    /// `cat(format(cpm(y, offset = off), digits = 17), sep = ", ")`
    #[test]
    fn test_cpm_matches_edger_with_a_gene_varying_offset() {
        let want = [
            10.000000000000005,
            10.000000000000002,
            12.222222222222213,
            36.363636363636346,
            44.000000000000021,
            29.230769230769237,
            45.241870901798016,
            36.193496721438400,
            52.279495264299861,
            40.306394076147292,
            46.146708319833977,
            34.801439155229232,
            2.4428055163203388,
            0.0,
            6.7855708786675999,
            1.1103661437819714,
            3.6642082744805085,
            0.0,
        ];
        let offset = varying_offset();
        let got = cpm(&COUNTS, 3, 6, None, Some(&offset), false, 0.0).unwrap();
        assert_close(&got, &want);
    }

    /// The prior count is scaled by the *row* mean library size, so a
    /// gene-varying offset does not reduce to the per-sample formula.
    /// `cat(format(cpm(y, offset = off, log = TRUE, prior.count = 2),`
    /// `digits = 17), sep = ", ")`
    #[test]
    fn test_cpm_log_matches_edger_with_a_gene_varying_offset() {
        let want = [
            3.5663414956896449,
            3.5663414956896449,
            3.8143785739740581,
            5.2558651046475378,
            5.5187234754569063,
            4.9577664377465398,
            5.5518907390050094,
            5.2427487990734898,
            5.7535452811577619,
            5.3915176359738757,
            5.5794522466412495,
            5.1887089593606008,
            2.2319489735675027,
            1.1730552845139333,
            3.1763909970771866,
            1.7507129846118552,
            2.5653727072926924,
            1.1730552845139333,
        ];
        let offset = varying_offset();
        let got = cpm(&COUNTS, 3, 6, None, Some(&offset), true, 2.0).unwrap();
        assert_close(&got, &want);
    }

    /// `cat(format(cpm(yz, lib.size = ls), digits = 17), sep = ", ")` and the
    /// matching `log = TRUE, prior.count = 2` call.
    #[test]
    fn test_cpm_handles_an_all_zero_gene() {
        let raw = cpm(&COUNTS_WITH_ZERO_GENE, 3, 6, Some(&LIB), None, false, 0.0).unwrap();
        assert_close(&raw[6..12], &[0.0; 6]);

        let want_log = [
            0.88451745571590135,
            0.88451745571590135,
            0.88451745571589879,
            0.88451745571590135,
            0.88451745571590135,
            0.88451745571590135,
        ];
        let logged = cpm(&COUNTS_WITH_ZERO_GENE, 3, 6, Some(&LIB), None, true, 2.0).unwrap();
        assert_close(&logged[6..12], &want_log);
    }

    /// `cat(format(cpm(ys), digits = 17), sep = ", ")` and the `log = TRUE`
    /// version, on `ys <- matrix(c(5, 0, 100, 3), nrow = 4)`.
    #[test]
    fn test_cpm_matches_edger_for_a_single_sample() {
        let want = [
            46296.296296296299,
            0.0,
            925925.925925925956,
            27777.777777777777,
        ];
        let got = cpm(&COUNTS_SINGLE, 4, 1, None, None, false, 0.0).unwrap();
        assert_close(&got, &want);

        let want_log = [
            15.931568569324174,
            14.124213647266568,
            19.796638989238065,
            15.446141742153934,
        ];
        let got_log = cpm(&COUNTS_SINGLE, 4, 1, None, None, true, 2.0).unwrap();
        assert_close(&got_log, &want_log);
    }

    // -- rpkm and tpm --

    /// `glen <- c(1500, 2500, 800)`
    /// `cat(format(rpkm(y, gene.length = glen, lib.size = ls), digits = 17), sep = ", ")`
    #[test]
    fn test_rpkm_matches_edger() {
        let want = [
            6.6666666666666670,
            6.6666666666666670,
            8.1481481481481470,
            24.242424242424246,
            29.333333333333332,
            19.487179487179485,
            20.0,
            16.0,
            23.111111111111111,
            17.818181818181820,
            20.399999999999999,
            15.384615384615383,
            2.5,
            0.0,
            6.9444444444444438,
            1.1363636363636362,
            3.75,
            0.0,
        ];
        let lengths = [1500.0, 2500.0, 800.0];
        let got = rpkm(&COUNTS, 3, 6, &lengths, Some(&LIB), None, false, 0.0).unwrap();
        assert_close(&got, &want);
    }

    /// `cat(format(rpkm(y, gene.length = glen, lib.size = ls, log = TRUE,`
    /// `prior.count = 2), digits = 17), sep = ", ")`
    #[test]
    fn test_rpkm_log_matches_edger() {
        let want = [
            2.9813789949684888,
            2.9813789949684888,
            3.2294160732528994,
            4.6709026039263817,
            4.9337609747357503,
            4.3728039370253837,
            4.3742316412892412,
            4.0650897013577216,
            4.5758861834419937,
            4.2138585382581075,
            4.4017931489254813,
            4.0110498616448336,
            2.2653392396568330,
            1.2064455506032636,
            3.2097812631665170,
            1.7841032507011856,
            2.5987629733820228,
            1.2064455506032636,
        ];
        let lengths = [1500.0, 2500.0, 800.0];
        let got = rpkm(&COUNTS, 3, 6, &lengths, Some(&LIB), None, true, 2.0).unwrap();
        assert_close(&got, &want);
    }

    /// `cat(format(tpm(y, effective.tx.length = glen), digits = 17), sep = ", ")`
    #[test]
    fn test_tpm_matches_edger() {
        let want = [
            214192.014390340802,
            265598.097844022675,
            214821.990903253609,
            590217.995208939188,
            596240.627813111991,
            573450.438526867074,
            642576.043171022437,
            637435.434825654374,
            609313.283289228450,
            433810.226478570316,
            414658.254797300615,
            452724.030415947665,
            80322.005396377819,
            0.0,
            183086.924065272964,
            27666.468525419023,
            76223.943896562603,
            0.0,
        ];
        let lengths = [1500.0, 2500.0, 800.0];
        let got = tpm(&COUNTS, 3, 6, &lengths).unwrap();
        assert_close(&got, &want);
    }

    // -- aveLogCPM --

    /// `cat(format(aveLogCPM(y), digits = 17), sep = ", ")`
    #[test]
    fn test_ave_log_cpm_matches_edger_with_the_defaults() {
        let want = [18.321647698392781, 19.316473215279711, 15.523372838343663];
        let got = ave_log_cpm(&COUNTS, 3, 6, None, None, 2.0, None).unwrap();
        assert_close_fit(&got, &want);
    }

    /// `cat(format(aveLogCPM(y, lib.size = ls, dispersion = 0.1), digits = 17),`
    /// `sep = ", ")`
    #[test]
    fn test_ave_log_cpm_matches_edger_with_a_dispersion() {
        let want = [4.6750928102298239, 5.6052509696052537, 1.8473870176899603];
        let dispersion = Recycled::scalar(0.1);
        let got = ave_log_cpm(&COUNTS, 3, 6, Some(&LIB), None, 2.0, Some(&dispersion)).unwrap();
        assert_close_fit(&got, &want);
    }

    /// `cat(format(aveLogCPM(y, lib.size = ls, prior.count = 0.125,`
    /// `dispersion = 0.1), digits = 17), sep = ", ")`
    #[test]
    fn test_ave_log_cpm_matches_edger_at_prior_count_one_eighth() {
        let want = [4.5740830265787151, 5.5529050289506570, 0.88224321795430982];
        let dispersion = Recycled::scalar(0.1);
        let got = ave_log_cpm(&COUNTS, 3, 6, Some(&LIB), None, 0.125, Some(&dispersion)).unwrap();
        assert_close_fit(&got, &want);
    }

    /// `cat(format(aveLogCPM(y, lib.size = ls, prior.count = 1,`
    /// `dispersion = 0.1), digits = 17), sep = ", ")`
    #[test]
    fn test_ave_log_cpm_matches_edger_at_prior_count_one() {
        let want = [4.6221000788436140, 5.5775706311233373, 1.4131875437323824];
        let dispersion = Recycled::scalar(0.1);
        let got = ave_log_cpm(&COUNTS, 3, 6, Some(&LIB), None, 1.0, Some(&dispersion)).unwrap();
        assert_close_fit(&got, &want);
    }

    /// `cat(format(aveLogCPM(y, lib.size = ls, dispersion = c(0.1, 0.2, 0.3)),`
    /// `digits = 17), sep = ", ")`
    #[test]
    fn test_ave_log_cpm_matches_edger_with_a_tagwise_dispersion() {
        let want = [4.6750928102298239, 5.6070127619390506, 1.8700119444011387];
        let dispersion = Recycled::by_gene(vec![0.1, 0.2, 0.3]);
        let got = ave_log_cpm(&COUNTS, 3, 6, Some(&LIB), None, 2.0, Some(&dispersion)).unwrap();
        assert_close_fit(&got, &want);
    }

    /// A `Full` offset whose rows are all equal must give the same answer as the
    /// per-sample form. It does here.
    ///
    /// edgeR does not: `aveLogCPM(y, offset = matrix(log(ls), nrow(y), ncol(y),
    /// byrow = TRUE), dispersion = 0.1)` returns 24.62 for the *first* gene and
    /// the correct value for every other one, against 4.675 from the identical
    /// `offset = log(ls)` call. The first row of a matrix offset is dropped
    /// somewhere in edgeR 4.8.2's `.cxx_ave_log_cpm`. This crate does not
    /// reproduce that, so the reference for this path is the vector-offset call
    /// rather than the matrix one.
    #[test]
    fn test_ave_log_cpm_with_a_constant_full_offset_matches_the_per_sample_form() {
        let dispersion = Recycled::scalar(0.1);
        let mut values = Vec::with_capacity(18);
        for _ in 0..3 {
            values.extend(LIB.iter().map(|l| l.ln()));
        }
        let full = Recycled::full(values, 3, 6).unwrap();

        let got = ave_log_cpm(&COUNTS, 3, 6, None, Some(&full), 2.0, Some(&dispersion)).unwrap();
        let want = [4.6750928102298239, 5.6052509696052537, 1.8473870176899603];
        assert_close_fit(&got, &want);
    }

    /// Genes 2 and 3 of `aveLogCPM(y, offset = off, dispersion = 0.1)`, the
    /// gene-varying offset. Gene 1 is omitted for the edgeR bug documented at
    /// [`test_ave_log_cpm_with_a_constant_full_offset_matches_the_per_sample_form`].
    #[test]
    fn test_ave_log_cpm_matches_edger_with_a_gene_varying_offset() {
        let dispersion = Recycled::scalar(0.1);
        let offset = varying_offset();
        let got = ave_log_cpm(&COUNTS, 3, 6, None, Some(&offset), 2.0, Some(&dispersion)).unwrap();
        assert_close_fit(&got[1..], &[5.4609819724336592, 2.1359248464879919]);
        // Gene 1's offset is exactly log(ls), so its answer must be the
        // per-sample one.
        assert_relative_eq!(got[0], 4.6750928102298239, max_relative = 1e-10);
    }

    /// `cat(format(aveLogCPM(yz, lib.size = ls, dispersion = 0.1), digits = 17),`
    /// `sep = ", ")`
    #[test]
    fn test_ave_log_cpm_handles_an_all_zero_gene() {
        let want = [4.6750928102298239, 0.88451745571590135, 1.8473870176899603];
        let dispersion = Recycled::scalar(0.1);
        let got = ave_log_cpm(
            &COUNTS_WITH_ZERO_GENE,
            3,
            6,
            Some(&LIB),
            None,
            2.0,
            Some(&dispersion),
        )
        .unwrap();
        assert_close_fit(&got, &want);
    }

    /// `cat(format(aveLogCPM(ys, dispersion = 0.1), digits = 17), sep = ", ")`
    #[test]
    fn test_ave_log_cpm_matches_edger_for_a_single_sample() {
        let want = [
            15.931568569324174,
            14.124213647266568,
            19.796638989238065,
            15.446141742153934,
        ];
        let dispersion = Recycled::scalar(0.1);
        let got = ave_log_cpm(&COUNTS_SINGLE, 4, 1, None, None, 2.0, Some(&dispersion)).unwrap();
        assert_close_fit(&got, &want);
    }

    /// An entirely empty matrix with no library sizes has no abundance to
    /// estimate, and edgeR reports `(-log(nrow) + log(1e6)) / log(2)`.
    /// `cat(format(aveLogCPM(matrix(0, 4, 3)), digits = 17), sep = ", ")`
    #[test]
    fn test_ave_log_cpm_of_an_empty_matrix_matches_edger() {
        let got = ave_log_cpm(&[0.0_f64; 12], 4, 3, None, None, 2.0, None).unwrap();
        assert_close_fit(&got, &[17.931568569324174; 4]);
    }

    /// Given library sizes, the empty-matrix short circuit does not apply and
    /// the fit runs as usual.
    #[test]
    fn test_ave_log_cpm_of_an_empty_matrix_with_library_sizes_fits_normally() {
        let got = ave_log_cpm(
            &[0.0_f64; 12],
            4,
            3,
            Some(&[1e6, 1e6, 1e6]),
            None,
            2.0,
            None,
        )
        .unwrap();
        for v in &got {
            assert!(v.is_finite());
            assert!(*v < 2.0);
        }
    }

    /// `f32` counts must land on the same answer to `f32` precision, since every
    /// derived quantity is `f64` per the crate numeric policy.
    #[test]
    fn test_cpm_is_generic_over_the_count_type() {
        let counts32: Vec<f32> = COUNTS.iter().map(|v| *v as f32).collect();
        let got = cpm(&counts32, 3, 6, Some(&LIB), None, false, 0.0).unwrap();
        let want = cpm(&COUNTS, 3, 6, Some(&LIB), None, false, 0.0).unwrap();
        for (g, w) in got.iter().zip(want.iter()) {
            assert_relative_eq!(g, w, epsilon = 1e-6);
        }
    }

    // -- errors --

    #[test]
    fn test_rejects_empty_counts() {
        let err = cpm::<f64>(&[], 0, 6, None, None, false, 0.0).unwrap_err();
        assert!(matches!(err, EdgeErrors::EmptyCounts { .. }));
    }

    #[test]
    fn test_rejects_a_shape_mismatch() {
        let err = cpm(&COUNTS[..12], 3, 6, None, None, false, 0.0).unwrap_err();
        assert!(matches!(
            err,
            EdgeErrors::LengthMismatch { name: "counts", .. }
        ));
    }

    #[test]
    fn test_rejects_negative_counts() {
        let mut bad = COUNTS;
        bad[0] = -1.0;
        let err = cpm(&bad, 3, 6, None, None, false, 0.0).unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));
    }

    #[test]
    fn test_rejects_library_sizes_of_the_wrong_length() {
        let err = cpm(&COUNTS, 3, 6, Some(&[1e6; 3]), None, false, 0.0).unwrap_err();
        assert!(matches!(
            err,
            EdgeErrors::LengthMismatch {
                name: "lib_size",
                ..
            }
        ));
    }

    #[test]
    fn test_rejects_a_non_positive_library_size() {
        let err = cpm(
            &COUNTS,
            3,
            6,
            Some(&[1e6, 0.0, 1e6, 1e6, 1e6, 1e6]),
            None,
            false,
            0.0,
        )
        .unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));
    }

    #[test]
    fn test_rejects_an_offset_of_the_wrong_length() {
        let offset = Recycled::by_sample(vec![1.0, 2.0]);
        let err = cpm(&COUNTS, 3, 6, None, Some(&offset), false, 0.0).unwrap_err();
        assert!(matches!(err, EdgeErrors::LengthMismatch { .. }));
    }

    #[test]
    fn test_rejects_a_negative_prior_count() {
        let err = cpm(&COUNTS, 3, 6, None, None, true, -1.0).unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));
        let err = ave_log_cpm(&COUNTS, 3, 6, None, None, -1.0, None).unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));
    }

    /// The prior count is unused when `log` is not set, so it is not checked
    /// there either, matching edgeR.
    #[test]
    fn test_ignores_the_prior_count_on_the_raw_scale() {
        assert!(cpm(&COUNTS, 3, 6, None, None, false, -1.0).is_ok());
    }

    #[test]
    fn test_rejects_a_negative_dispersion() {
        let dispersion = Recycled::by_gene(vec![0.1, -0.2, 0.3]);
        let err = ave_log_cpm(&COUNTS, 3, 6, None, None, 2.0, Some(&dispersion)).unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidDispersion(_)));
    }

    #[test]
    fn test_rejects_a_dispersion_of_the_wrong_length() {
        let dispersion = Recycled::by_gene(vec![0.1, 0.2]);
        let err = ave_log_cpm(&COUNTS, 3, 6, None, None, 2.0, Some(&dispersion)).unwrap_err();
        assert!(matches!(err, EdgeErrors::LengthMismatch { .. }));
    }

    #[test]
    fn test_rejects_gene_lengths_of_the_wrong_length() {
        let err = rpkm(&COUNTS, 3, 6, &[1000.0, 2000.0], None, None, false, 0.0).unwrap_err();
        assert!(matches!(
            err,
            EdgeErrors::LengthMismatch {
                name: "gene_length",
                ..
            }
        ));
        let err = tpm(&COUNTS, 3, 6, &[1000.0, 2000.0]).unwrap_err();
        assert!(matches!(
            err,
            EdgeErrors::LengthMismatch {
                name: "effective_length",
                ..
            }
        ));
    }

    #[test]
    fn test_rejects_a_non_positive_gene_length() {
        let err = rpkm(&COUNTS, 3, 6, &[1000.0, 0.0, 800.0], None, None, false, 0.0).unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));
    }

    /// A sample with no counts at all has no geometric mean to rescale by. It
    /// trips the library size guard first, since the library sizes are derived
    /// from the same column sums, but either way it is an error rather than a
    /// column of NaN.
    #[test]
    fn test_tpm_rejects_an_empty_sample() {
        let counts = [0.0_f64, 1.0, 0.0, 2.0];
        let err = tpm(&counts, 2, 2, &[1000.0, 1000.0]).unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));
    }
}
