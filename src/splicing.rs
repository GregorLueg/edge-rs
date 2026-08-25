//! `diffSpliceDGE`: differential exon usage.
//!
//! Where the rest of the crate asks whether a gene changed, this asks whether
//! one exon changed *relative to the rest of its gene*. The gene's own
//! log-fold-change is estimated from the summed counts, folded into the offsets
//! as a fixed shift, and the exon is then tested for whatever fold change is
//! left over. That leftover is alternative splicing.
//!
//! Three entry points: [`diff_splice`] takes a fitted GLM and does the work,
//! [`diff_splice_dge`] is the [`DgeList`] wrapper that fits first, and
//! [`splice_variants`] is edgeR's older per-gene interaction test, which unrolls
//! each gene into one row of an exon-by-group layout and runs a single
//! likelihood ratio test on the interaction.
//!
//! ### Parallelism
//!
//! Both axes are used, each where its work lives. The two refits fan out over
//! rows inside [`glm_fit`], and those rows are exons for the exon-level fit and
//! genes for the gene-level one. On top of that the per-exon tail probabilities
//! fan out over exons, and the per-gene aggregation, which has to sort each
//! gene's p-values for the Simes step, fans out over genes. The scatter that
//! sums exon quantities into genes stays sequential: it is one pass of `+=` over
//! an index vector, and a parallel reduction over gene groups costs more in
//! synchronisation than the additions are worth.
//!
//! ### Layout
//!
//! Exons take the place of genes in the crate's row-major convention: counts
//! are `n_exons * n_samples`, and `gene_id` labels each row. Exons need not be
//! sorted or even contiguous by gene; genes come out in first-appearance order.
//! edgeR sorts by gene id first, so the two agree whenever the input is sorted,
//! which is how exon-level counts arrive in practice.

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::core::dgelist::DgeList;
use crate::glm::fit::{DEFAULT_PRIOR_COUNT, GlmFit, glm_fit, glm_fit_dge};
use crate::glm::test::{GlmTestInput, Tested, glm_lrt};
use crate::limma::squeeze_var::{SqueezeVarParams, squeeze_var};
use crate::numeric::dist::{chisq_sf, f_sf};
use crate::prelude::*;
use crate::utils::design::contrast_as_coef;

////////////
// Consts //
////////////

/// Dispersion the gene-level fit runs at.
///
/// edgeR hard-codes this. The gene-level coefficient is only ever used as a
/// fixed offset for the exon-level test, so its standard error never reaches a
/// p-value and a plausible round number does the job. Estimating one per gene
/// would double the cost of the routine and change nothing downstream.
const GENE_LEVEL_DISPERSION: f64 = 0.05;

//////////////////////
// DiffSpliceParams //
//////////////////////

/// Tuning knobs for [`diff_splice`] and [`diff_splice_dge`].
#[derive(Clone, Copy, Debug)]
pub struct DiffSpliceParams {
    /// Prior count for the gene-level fit's log-fold-change shrinkage. Only the
    /// gene-level fit sees it; the exon-level fit was done by the caller and the
    /// reduced-model refit never needs shrunk coefficients.
    pub prior_count: f64,
    /// Whether the gene-level empirical Bayes step Winsorises its moments.
    /// `None` follows edgeR: robust exactly when the quasi-likelihood fit
    /// produced a per-gene `df_prior`. Ignored on the likelihood ratio path,
    /// which does no squeezing.
    pub robust: Option<bool>,
}

impl DiffSpliceParams {
    /// Builds a parameter set.
    ///
    /// ### Params
    ///
    /// * `prior_count` - Shrinkage prior for the gene-level fit, non-negative
    /// * `robust` - Robust empirical Bayes, or `None` to let the fit decide
    ///
    /// ### Returns
    ///
    /// The parameter set. Validation happens in [`diff_splice`].
    pub fn new(prior_count: f64, robust: Option<bool>) -> Self {
        Self {
            prior_count,
            robust,
        }
    }
}

impl Default for DiffSpliceParams {
    /// edgeR's defaults: a prior count of 0.125 and an automatic `robust`.
    fn default() -> Self {
        Self {
            prior_count: DEFAULT_PRIOR_COUNT,
            robust: None,
        }
    }
}

//////////////////
// DiffSpliceQl //
//////////////////

/// The quasi-likelihood quantities [`diff_splice`] needs from `glmQLFit`.
///
/// Passing `Some` switches the exon and gene tests from chi-squared to
/// moderated F. Every field maps onto one of `crate::glm::ql_fit::QlFit`:
///
/// | field | legacy pipeline | current pipeline |
/// |---|---|---|
/// | `df_residual` | `df_residual_zeros` | `df_residual_adj` |
/// | `deviance` | `deviance` | `deviance_adj` |
/// | `legacy_zeros` | `true` | `false` |
/// | `average_ql_dispersion` | `None` | `average_ql_dispersion` |
///
/// The fit's own `s2_post` is deliberately absent: `diffSpliceDGE` throws the
/// exon-level squeezing away and squeezes again at the gene level, because the
/// unit being tested is a gene's worth of exons rather than a single row.
#[derive(Clone, Copy, Debug)]
pub struct DiffSpliceQl<'a> {
    /// Prior degrees of freedom from the fit, one value or one per exon. Only
    /// its length is read, and only to default `robust`.
    pub df_prior: &'a [f64],
    /// Residual degrees of freedom per exon that go with `deviance`.
    pub df_residual: &'a [f64],
    /// Residual deviance per exon that goes with `df_residual`.
    pub deviance: &'a [f64],
    /// Whether `df_residual` came from `df.residual.zeros`.
    ///
    /// Switches on edgeR's chi-squared floor: an exon in a gene whose posterior
    /// quasi-dispersion fell below one is not allowed a p-value smaller than the
    /// unmoderated chi-squared would give it. Only the legacy pipeline, which
    /// drops structural zeros from the degrees of freedom, can push a gene there
    /// spuriously, so only the legacy pipeline gets the floor.
    pub legacy_zeros: bool,
    /// `average.ql.dispersion`, which the negative binomial dispersion is
    /// divided by before the reduced model is refitted so that the two models
    /// sit on the same scale. `None` on the legacy pipeline, which has none.
    pub average_ql_dispersion: Option<f64>,
}

//////////////////////
// DiffSpliceResult //
//////////////////////

/// Result of a differential exon usage analysis.
///
/// Exon-level vectors have one entry per input exon, in input order. Gene-level
/// vectors have one entry per *testable* gene, in first-appearance order: a gene
/// with a single exon has no within-gene contrast to test and is dropped rather
/// than errored on.
#[derive(Clone, Debug)]
pub struct DiffSpliceResult {
    /// Exon-level log fold change relative to the gene.
    ///
    /// The exon's coefficient minus its gene's, on the **natural** log scale,
    /// matching edgeR's `coefficients` slot. `topSpliceDGE` prints this column
    /// as `logFC`, which is a misnomer there and stays one here rather than
    /// silently disagreeing with edgeR by a factor of `ln 2`.
    ///
    /// Zero for an exon whose gene was dropped: such an exon *is* its gene, so
    /// its usage relative to the gene cannot move.
    pub exon_log_fc: Vec<f64>,
    /// Exon-level test statistic and p-value.
    ///
    /// The statistic is a likelihood ratio when `ql` was `None` and a moderated
    /// F otherwise. An exon whose gene was dropped carries a zero statistic and
    /// a p-value of one.
    pub exon_statistic: Vec<f64>,
    /// Exon-level p-value. See [`DiffSpliceResult::exon_statistic`].
    pub exon_p_value: Vec<f64>,
    /// Gene-level identifiers, one per gene, in first-appearance order.
    pub gene_id: Vec<usize>,
    /// Gene-level Simes and F tests.
    ///
    /// The Simes p-value is `min over k of (n * p_(k) / k)` on the gene's
    /// sorted exon p-values. It answers "does this gene contain any
    /// differentially used exon", which is a different question from the joint
    /// test below and is usually the more sensitive of the two.
    pub gene_simes_p: Vec<f64>,
    /// Joint gene-level statistic: the summed exon statistics, read off a
    /// chi-squared on the likelihood ratio path and an F on the
    /// quasi-likelihood path. Named for the quasi-likelihood case, which is
    /// edgeR's default.
    pub gene_f_statistic: Vec<f64>,
    /// P-value for [`DiffSpliceResult::gene_f_statistic`].
    pub gene_f_p_value: Vec<f64>,
    /// Number of exons per gene.
    pub gene_n_exons: Vec<usize>,
}

/// Tests for differential exon usage.
///
/// Port of edgeR's `diffSpliceDGE`. Given a negative binomial GLM already fitted
/// at the exon level, it
///
/// 1. sums each gene's exon counts and fits the same design to the totals,
///    which gives the gene's own log-fold-change `betabar`,
/// 2. folds `betabar * design[, coef]` into the exon-level offsets, so the
///    gene-level change is now a fixed, known part of every exon's expectation,
/// 3. refits each exon under the reduced design with `coef` dropped, and
/// 4. reads the deviance difference as the evidence that this exon moved by
///    something other than its gene's amount.
///
/// The exon-level test therefore compares each exon's coefficient against the
/// gene's average, which is why the design is rewritten per gene rather than
/// once. Genes with a single exon have nothing to compare against and are
/// dropped.
///
/// ### Params
///
/// * `input` - Counts, design and recycled matrices behind the fit, with exons
///   in the place of genes: `input.n_genes` is the exon count and
///   `input.counts` is `n_exons * n_samples`
/// * `fit` - The exon-level fit, from [`glm_fit`] or `glmQLFit`. Only its
///   coefficients, deviance and residual degrees of freedom are read
/// * `gene_id` - Gene label per exon, one per row of `input.counts`
/// * `tested` - Which coefficient carries the comparison. Only one is testable:
///   extra entries of [`Tested::Coef`], and extra columns of
///   [`Tested::Contrast`], are dropped, as edgeR does with a warning
/// * `ql` - Quasi-likelihood quantities from `glmQLFit`, or `None` for the
///   likelihood ratio flavour
/// * `params` - Tuning knobs, or [`DiffSpliceParams::default`]
///
/// ### Returns
///
/// The exon and gene level tests, or [`EdgeErrors`] if a shape disagrees, the
/// design has fewer than two columns, the tested coefficient is out of range,
/// the contrast is rank deficient, or no gene has more than one exon.
///
/// ### References
///
/// Chen, Lun and Smyth, F1000Research, 2016
pub fn diff_splice<T: EdgeFloat>(
    input: &GlmTestInput<'_, T>,
    fit: &GlmFit,
    gene_id: &[usize],
    tested: &Tested,
    ql: Option<&DiffSpliceQl<'_>>,
    params: Option<DiffSpliceParams>,
) -> Result<DiffSpliceResult, EdgeErrors> {
    let params = params.unwrap_or_default();
    validate(input, fit, gene_id, ql, &params)?;

    let n_exons = input.n_genes;
    let n_samples = input.n_samples;
    let n_coef = input.n_coef;

    let resolved = resolve_tested(input, fit, tested)?;
    let groups = group_by_gene(gene_id);

    let kept_genes: Vec<usize> = (0..groups.ids.len())
        .filter(|g| groups.members[*g].len() > 1)
        .collect();
    if kept_genes.is_empty() {
        return Err(EdgeErrors::InvalidArgument(
            "no gene has more than one exon, so there is no within-gene contrast to test"
                .to_string(),
        ));
    }

    let mut kept_exons: Vec<usize> = Vec::new();
    let mut exon_gene: Vec<usize> = Vec::new();
    let mut gene_start: Vec<usize> = Vec::with_capacity(kept_genes.len());
    for (slot, &gene) in kept_genes.iter().enumerate() {
        gene_start.push(kept_exons.len());
        for &exon in &groups.members[gene] {
            kept_exons.push(exon);
            exon_gene.push(slot);
        }
    }
    let n_kept_genes = kept_genes.len();
    let n_kept_exons = kept_exons.len();
    let gene_n_exons: Vec<usize> = kept_genes
        .iter()
        .map(|g| groups.members[*g].len())
        .collect();

    let mut gene_counts = vec![0.0_f64; n_kept_genes * n_samples];
    for (exon, &gene) in kept_exons.iter().zip(exon_gene.iter()) {
        let source = &input.counts[exon * n_samples..(exon + 1) * n_samples];
        let target = &mut gene_counts[gene * n_samples..(gene + 1) * n_samples];
        for (acc, v) in target.iter_mut().zip(source.iter()) {
            *acc += v.to_f64().unwrap_or(f64::NAN);
        }
    }

    let gene_offset: Vec<f64> = input.offset.row(kept_exons[0], n_samples).to_vec(n_samples);
    let gene_fit = glm_fit(
        &gene_counts,
        n_kept_genes,
        n_samples,
        &resolved.design,
        n_coef,
        &Recycled::scalar(GENE_LEVEL_DISPERSION),
        &Recycled::by_sample(gene_offset),
        None,
        params.prior_count,
    )?;
    let betabar: Vec<f64> = (0..n_kept_genes)
        .map(|g| gene_fit.coefficients[g * n_coef + resolved.coef])
        .collect();

    let mut counts_kept: Vec<T> = Vec::with_capacity(n_kept_exons * n_samples);
    let mut offset_new: Vec<f64> = Vec::with_capacity(n_kept_exons * n_samples);
    for (exon, &gene) in kept_exons.iter().zip(exon_gene.iter()) {
        counts_kept.extend_from_slice(&input.counts[exon * n_samples..(exon + 1) * n_samples]);
        let row = input.offset.row(*exon, n_samples);
        for sample in 0..n_samples {
            offset_new.push(
                row.get(sample) + betabar[gene] * resolved.design[sample * n_coef + resolved.coef],
            );
        }
    }

    let dispersion = input.dispersion.subset(&kept_exons, n_samples);
    let dispersion = match ql.and_then(|q| q.average_ql_dispersion) {
        Some(average) => dispersion.map(|v| v / average),
        None => dispersion,
    };
    let weights = input.weights.map(|w| w.subset(&kept_exons, n_samples));

    let design0 = drop_column(&resolved.design, n_samples, n_coef, resolved.coef);
    let fit0 = glm_fit(
        &counts_kept,
        n_kept_exons,
        n_samples,
        &design0,
        n_coef - 1,
        &dispersion,
        &Recycled::full(offset_new, n_kept_exons, n_samples)?,
        weights.as_ref(),
        // edgeR leaves `prior.count` at its default here, but glmFit keeps the
        // unshrunk deviance and only the deviance is read, so shrinking would
        // buy a second fit and change nothing.
        0.0,
    )?;

    let exon_df_test = fit0.df_residual as f64 - fit.df_residual as f64;
    let exon_lr: Vec<f64> = (0..n_kept_exons)
        .map(|i| fit0.deviance[i] - fit.deviance[kept_exons[i]])
        .collect();

    let mut gene_lr = vec![0.0_f64; n_kept_genes];
    for (lr, &gene) in exon_lr.iter().zip(exon_gene.iter()) {
        gene_lr[gene] += lr;
    }
    let gene_df_test: Vec<f64> = gene_n_exons
        .iter()
        .map(|n| *n as f64 * exon_df_test)
        .collect();

    let tests = match ql {
        None => lrt_tests(&exon_lr, exon_df_test, &gene_lr, &gene_df_test)?,
        Some(q) => ql_tests(
            &exon_lr,
            exon_df_test,
            &gene_lr,
            &gene_df_test,
            &kept_exons,
            &exon_gene,
            n_kept_genes,
            q,
            &params,
        )?,
    };

    let gene_simes_p: Vec<f64> = gene_n_exons
        .par_iter()
        .zip(gene_start.par_iter())
        .map(|(n, start)| simes(&tests.exon_p[*start..start + n]))
        .collect();

    let mut exon_log_fc = vec![0.0_f64; n_exons];
    let mut exon_statistic = vec![0.0_f64; n_exons];
    let mut exon_p_value = vec![1.0_f64; n_exons];
    for (i, &exon) in kept_exons.iter().enumerate() {
        exon_log_fc[exon] = resolved.beta[exon] - betabar[exon_gene[i]];
        exon_statistic[exon] = tests.exon_statistic[i];
        exon_p_value[exon] = tests.exon_p[i];
    }

    Ok(DiffSpliceResult {
        exon_log_fc,
        exon_statistic,
        exon_p_value,
        gene_id: kept_genes.iter().map(|g| groups.ids[*g]).collect(),
        gene_simes_p,
        gene_f_statistic: tests.gene_statistic,
        gene_f_p_value: tests.gene_p,
        gene_n_exons,
    })
}

/// Fits a [`DgeList`] of exon counts and tests it for differential usage.
///
/// The [`DgeList`] wrapper around [`diff_splice`], in the same relation to it
/// as `glm_fit_dge` is to `glm_fit`: it fits the exon-level GLM from the
/// container's own offsets and dispersion and then hands the fit straight on.
/// The fit uses [`DEFAULT_PRIOR_COUNT`], as `glmFit` does;
/// `params.prior_count` belongs to the gene-level fit inside [`diff_splice`]
/// and is a separate knob.
///
/// This is the likelihood ratio flavour. For the quasi-likelihood one, run
/// `glm_ql_fit` yourself and call [`diff_splice`] with a [`DiffSpliceQl`].
///
/// ### Params
///
/// * `dge` - Exon-level counts, one row per exon, carrying a dispersion
///   estimate
/// * `design` - Design matrix, row-major `n_samples * n_coef`
/// * `n_coef` - Number of coefficients, at least two
/// * `gene_id` - Gene label per exon
/// * `tested` - Which coefficient carries the comparison
/// * `params` - Tuning knobs, or [`DiffSpliceParams::default`]
///
/// ### Returns
///
/// The exon and gene level tests, or [`EdgeErrors`] if the container has no
/// dispersion, a shape disagrees, or [`diff_splice`] rejects the input.
pub fn diff_splice_dge<T: EdgeFloat>(
    dge: &DgeList<T>,
    design: &[f64],
    n_coef: usize,
    gene_id: &[usize],
    tested: &Tested,
    params: Option<DiffSpliceParams>,
) -> Result<DiffSpliceResult, EdgeErrors> {
    let (dispersion, _) = dge.dispersion().ok_or_else(|| {
        EdgeErrors::InvalidArgument("no dispersion has been estimated for this DgeList".to_string())
    })?;
    let offset = dge.offset()?;
    let fit = glm_fit_dge(dge, design, n_coef, None)?;

    let input = GlmTestInput {
        counts: &dge.counts,
        n_genes: dge.n_genes,
        n_samples: dge.n_samples,
        design,
        n_coef,
        dispersion: &dispersion,
        offset: &offset,
        weights: dge.weights.as_ref(),
        log_cpm: None,
    };
    diff_splice(&input, &fit, gene_id, tested, None, params)
}

/// Identifies genes carrying splice variants.
///
/// Port of edgeR's `spliceVariants`, which predates `diffSpliceDGE` and asks
/// the question in one shot rather than exon by exon. Each gene is unrolled
/// into a single row of `n_exons * n_samples` counts, laid out exon block by
/// exon block, and fitted with `~ exon + group + exon:group`. The interaction
/// is the splice signal: it is exactly the claim that the exon profile differs
/// between groups. A likelihood ratio test on all
/// `(n_exons - 1) * (n_groups - 1)` interaction coefficients gives one
/// statistic per gene.
///
/// Genes are batched by exon count, since every gene with the same number of
/// exons shares a design and can be fitted in one call.
///
/// Two edgeR behaviours are kept deliberately. Exons whose counts are zero in
/// every sample are dropped first, which can shrink or empty a gene. And the
/// fit runs with no offset at all, so library sizes are not corrected for;
/// within a gene they are common to every exon and cancel out of the
/// interaction.
///
/// ### Params
///
/// * `counts` - Exon counts, row-major `n_exons * n_samples`
/// * `n_exons` - Number of exons, that is, rows
/// * `n_samples` - Number of samples, that is, columns
/// * `gene_id` - Gene label per exon
/// * `group` - Group label per sample. At least two distinct labels are needed
///   for the interaction to exist
/// * `dispersion` - Negative binomial dispersion, either one value shared by
///   every gene or one per gene in first-appearance order. edgeR's
///   `estimateExonGenewiseDisp` front end is not ported; supply the dispersion
///
/// ### Returns
///
/// A [`DiffSpliceResult`] carrying gene-level results only: `gene_f_statistic`
/// holds the likelihood ratio, `gene_f_p_value` its chi-squared p-value, and
/// `gene_simes_p` repeats that p-value, since `spliceVariants` has no Simes
/// stage. The exon-level vectors are empty. Unlike [`diff_splice`], single-exon
/// genes are reported rather than dropped, with a zero statistic and a p-value
/// of one, which is what edgeR does.
///
/// Or [`EdgeErrors`] if a shape disagrees, fewer than two groups are supplied,
/// a dispersion is negative, or a gene's unrolled design is too large for the
/// samples available.
pub fn splice_variants<T: EdgeFloat>(
    counts: &[T],
    n_exons: usize,
    n_samples: usize,
    gene_id: &[usize],
    group: &[usize],
    dispersion: &[f64],
) -> Result<DiffSpliceResult, EdgeErrors> {
    if n_exons == 0 || n_samples == 0 {
        return Err(EdgeErrors::EmptyCounts {
            n_genes: n_exons,
            n_samples,
        });
    }
    if counts.len() != n_exons * n_samples {
        return Err(EdgeErrors::LengthMismatch {
            name: "counts",
            expected: n_exons * n_samples,
            got: counts.len(),
        });
    }
    if gene_id.len() != n_exons {
        return Err(EdgeErrors::LengthMismatch {
            name: "gene_id",
            expected: n_exons,
            got: gene_id.len(),
        });
    }
    if group.len() != n_samples {
        return Err(EdgeErrors::LengthMismatch {
            name: "group",
            expected: n_samples,
            got: group.len(),
        });
    }
    if let Some(bad) = dispersion.iter().find(|d| !d.is_finite() || **d < 0.0) {
        return Err(EdgeErrors::InvalidDispersion(*bad));
    }

    let mut levels: Vec<usize> = group.to_vec();
    levels.sort_unstable();
    levels.dedup();
    let n_groups = levels.len();
    if n_groups < 2 {
        return Err(EdgeErrors::InvalidArgument(format!(
            "splice_variants needs at least two groups; got {n_groups}"
        )));
    }
    let group_level: Vec<usize> = group
        .iter()
        .map(|g| {
            levels
                .iter()
                .position(|l| l == g)
                .expect("level was taken from group")
        })
        .collect();

    // edgeR drops exons that are zero everywhere before anything else, which can
    // shrink a gene or remove it outright.
    let nonzero: Vec<usize> = (0..n_exons)
        .filter(|e| {
            counts[e * n_samples..(e + 1) * n_samples]
                .iter()
                .any(|v| *v > T::zero())
        })
        .collect();
    let kept_ids: Vec<usize> = nonzero.iter().map(|e| gene_id[*e]).collect();
    let groups = group_by_gene(&kept_ids);
    let n_genes = groups.ids.len();
    if n_genes == 0 {
        return Err(EdgeErrors::InvalidArgument(
            "every exon is zero in every sample".to_string(),
        ));
    }
    if dispersion.len() != 1 && dispersion.len() != n_genes {
        return Err(EdgeErrors::LengthMismatch {
            name: "dispersion",
            expected: n_genes,
            got: dispersion.len(),
        });
    }

    let mut gene_statistic = vec![0.0_f64; n_genes];
    let mut gene_p = vec![1.0_f64; n_genes];
    let gene_n_exons: Vec<usize> = groups.members.iter().map(Vec::len).collect();

    let mut sizes: Vec<usize> = gene_n_exons.clone();
    sizes.sort_unstable();
    sizes.dedup();
    for n_exon in sizes {
        // One exon means no exon factor and so no interaction to test. edgeR
        // still reports the gene, with no evidence.
        if n_exon < 2 {
            continue;
        }
        let batch: Vec<usize> = (0..n_genes)
            .filter(|g| gene_n_exons[*g] == n_exon)
            .collect();
        let (statistic, p_value) = fit_exon_group_interaction(
            counts,
            n_samples,
            &nonzero,
            &groups,
            &batch,
            n_exon,
            n_groups,
            &group_level,
            dispersion,
        )?;
        for (slot, &gene) in batch.iter().enumerate() {
            gene_statistic[gene] = statistic[slot];
            gene_p[gene] = p_value[slot];
        }
    }

    Ok(DiffSpliceResult {
        exon_log_fc: Vec::new(),
        exon_statistic: Vec::new(),
        exon_p_value: Vec::new(),
        gene_id: groups.ids,
        gene_simes_p: gene_p.clone(),
        gene_f_statistic: gene_statistic,
        gene_f_p_value: gene_p,
        gene_n_exons,
    })
}

////////////////////
// Gene groupings //
////////////////////

/// Exons grouped by gene label, in first-appearance order.
struct GeneGroups {
    /// The distinct gene labels, ordered by where each was first seen.
    ids: Vec<usize>,
    /// Row indices belonging to each gene, in input order.
    members: Vec<Vec<usize>>,
}

/// Groups row indices by their gene label.
///
/// Order of first appearance, which is what `rowsum(..., reorder = FALSE)`
/// gives. Rows need not be contiguous by gene.
///
/// ### Params
///
/// * `gene_id` - Gene label per row
///
/// ### Returns
///
/// The distinct labels and the rows belonging to each.
fn group_by_gene(gene_id: &[usize]) -> GeneGroups {
    let mut slot: FxHashMap<usize, usize> = FxHashMap::default();
    let mut ids = Vec::new();
    let mut members: Vec<Vec<usize>> = Vec::new();
    for (row, id) in gene_id.iter().enumerate() {
        let next = ids.len();
        let g = *slot.entry(*id).or_insert_with(|| {
            ids.push(*id);
            members.push(Vec::new());
            next
        });
        members[g].push(row);
    }
    GeneGroups { ids, members }
}

///////////////////////
// Statistical tests //
///////////////////////

/// Exon and gene level statistics and p-values, over kept exons and genes.
struct SpliceTests {
    /// Statistic per kept exon.
    exon_statistic: Vec<f64>,
    /// P-value per kept exon.
    exon_p: Vec<f64>,
    /// Statistic per kept gene.
    gene_statistic: Vec<f64>,
    /// P-value per kept gene.
    gene_p: Vec<f64>,
}

/// Chi-squared tests on the raw deviance differences.
///
/// The likelihood ratio flavour, taken when the fit carries no quasi-likelihood
/// summary. Both levels read straight off a chi-squared, the gene level with as
/// many degrees of freedom as it has exons.
///
/// ### Params
///
/// * `exon_lr` - Deviance difference per kept exon
/// * `exon_df_test` - Degrees of freedom under test per exon
/// * `gene_lr` - Summed deviance difference per kept gene
/// * `gene_df_test` - Summed degrees of freedom per kept gene
///
/// ### Returns
///
/// The statistics and their p-values, or [`EdgeErrors`] if a tail probability
/// is asked for at invalid degrees of freedom.
fn lrt_tests(
    exon_lr: &[f64],
    exon_df_test: f64,
    gene_lr: &[f64],
    gene_df_test: &[f64],
) -> Result<SpliceTests, EdgeErrors> {
    let exon_p = exon_lr
        .par_iter()
        .map(|lr| chisq_sf(lr.max(0.0), exon_df_test))
        .collect::<Result<Vec<f64>, EdgeErrors>>()?;
    let gene_p = gene_lr
        .par_iter()
        .zip(gene_df_test.par_iter())
        .map(|(lr, df)| chisq_sf(lr.max(0.0), *df))
        .collect::<Result<Vec<f64>, EdgeErrors>>()?;
    Ok(SpliceTests {
        exon_statistic: exon_lr.to_vec(),
        exon_p,
        gene_statistic: gene_lr.to_vec(),
        gene_p,
    })
}

/// Moderated F tests against a gene-level squeezed dispersion.
///
/// The quasi-likelihood flavour. Each gene's exon deviances are pooled into one
/// residual variance, those variances are squeezed towards a fitted prior
/// across genes, and the deviance differences are divided by the result. The
/// denominator degrees of freedom are capped at the experiment's total residual
/// degrees of freedom, so a large prior cannot claim more information than the
/// data hold.
///
/// ### Params
///
/// * `exon_lr` - Deviance difference per kept exon
/// * `exon_df_test` - Degrees of freedom under test per exon
/// * `gene_lr` - Summed deviance difference per kept gene
/// * `gene_df_test` - Summed degrees of freedom per kept gene
/// * `kept_exons` - Input row index of each kept exon
/// * `exon_gene` - Gene slot of each kept exon
/// * `n_genes` - Number of kept genes
/// * `ql` - The quasi-likelihood quantities from the fit
/// * `params` - Tuning knobs, for `robust`
///
/// ### Returns
///
/// The F statistics and their p-values, or [`EdgeErrors`] if the empirical Bayes
/// fit or a tail probability fails.
#[allow(clippy::too_many_arguments)]
fn ql_tests(
    exon_lr: &[f64],
    exon_df_test: f64,
    gene_lr: &[f64],
    gene_df_test: &[f64],
    kept_exons: &[usize],
    exon_gene: &[usize],
    n_genes: usize,
    ql: &DiffSpliceQl<'_>,
    params: &DiffSpliceParams,
) -> Result<SpliceTests, EdgeErrors> {
    let mut gene_df_residual = vec![0.0_f64; n_genes];
    let mut gene_deviance = vec![0.0_f64; n_genes];
    for (exon, &gene) in kept_exons.iter().zip(exon_gene.iter()) {
        gene_df_residual[gene] += ql.df_residual[*exon];
        gene_deviance[gene] += ql.deviance[*exon];
    }
    let gene_s2: Vec<f64> = gene_deviance
        .iter()
        .zip(gene_df_residual.iter())
        .map(|(dev, df)| if *df > 0.0 { dev / df } else { 0.0 })
        .collect();

    let robust = params.robust.unwrap_or(ql.df_prior.len() > 1);
    let squeezed = squeeze_var(
        &gene_s2,
        &gene_df_residual,
        None,
        Some(SqueezeVarParams {
            robust,
            ..Default::default()
        }),
    )?;
    let cap: f64 = gene_df_residual.iter().sum();
    let gene_df_total: Vec<f64> = (0..n_genes)
        .map(|g| {
            let prior = squeezed.df_prior[g.min(squeezed.df_prior.len() - 1)];
            (gene_df_residual[g] + prior).min(cap)
        })
        .collect();
    let s2_post = &squeezed.var_post;

    let exon_statistic: Vec<f64> = exon_lr
        .iter()
        .zip(exon_gene.iter())
        .map(|(lr, g)| lr / exon_df_test / s2_post[*g])
        .collect();
    let gene_statistic: Vec<f64> = (0..n_genes)
        .map(|g| gene_lr[g] / gene_df_test[g] / s2_post[g])
        .collect();

    let mut exon_p = exon_statistic
        .par_iter()
        .zip(exon_gene.par_iter())
        .map(|(f, g)| f_sf(f.max(0.0), exon_df_test, gene_df_total[*g]))
        .collect::<Result<Vec<f64>, EdgeErrors>>()?;
    let gene_p = gene_statistic
        .par_iter()
        .enumerate()
        .map(|(g, f)| f_sf(f.max(0.0), gene_df_test[g], gene_df_total[g]))
        .collect::<Result<Vec<f64>, EdgeErrors>>()?;

    if ql.legacy_zeros {
        for (i, g) in exon_gene.iter().enumerate() {
            if s2_post[*g] < 1.0 {
                exon_p[i] = exon_p[i].max(chisq_sf(exon_lr[i].max(0.0), exon_df_test)?);
            }
        }
    }

    Ok(SpliceTests {
        exon_statistic,
        exon_p,
        gene_statistic,
        gene_p,
    })
}

/// Simes' combined p-value.
///
/// `min over k of (n * p_(k) / k)` on the sorted p-values. The `k = n` term is
/// the largest p-value itself, so the result never exceeds one and needs no
/// clamp.
///
/// ### Params
///
/// * `p` - P-values to combine, in any order
///
/// ### Returns
///
/// The combined p-value, or one for an empty input.
///
/// ### References
///
/// Simes, Biometrika, 1986
fn simes(p: &[f64]) -> f64 {
    if p.is_empty() {
        return 1.0;
    }
    let mut sorted = p.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len() as f64;
    sorted
        .iter()
        .enumerate()
        .map(|(k, v)| v * n / (k + 1) as f64)
        .fold(f64::INFINITY, f64::min)
}

////////////////////////////
// spliceVariants kernels //
////////////////////////////

/// Fits and tests one batch of genes that share an exon count.
///
/// Every gene in the batch is unrolled into a row of `n_exon * n_samples`
/// counts, ordered exon block by exon block, and the shared design
/// `~ exon + group + exon:group` is built once. The interaction columns are
/// then dropped and the deviance difference read off a chi-squared, which is
/// exactly `glmLRT` on the trailing coefficients.
///
/// ### Params
///
/// * `counts` - Exon counts, row-major `n_exons * n_samples`
/// * `n_samples` - Number of samples
/// * `nonzero` - Input row index of each surviving exon
/// * `groups` - Gene grouping over the surviving exons
/// * `batch` - Gene slots in this batch, all with `n_exon` exons
/// * `n_exon` - Exons per gene in this batch, at least two
/// * `n_groups` - Number of distinct group levels
/// * `group_level` - Zero-based group level per sample
/// * `dispersion` - One dispersion, or one per gene slot
///
/// ### Returns
///
/// The likelihood ratio and p-value per gene of the batch, or [`EdgeErrors`] if
/// the unrolled design does not fit the samples available or a fit fails.
#[allow(clippy::too_many_arguments)]
fn fit_exon_group_interaction<T: EdgeFloat>(
    counts: &[T],
    n_samples: usize,
    nonzero: &[usize],
    groups: &GeneGroups,
    batch: &[usize],
    n_exon: usize,
    n_groups: usize,
    group_level: &[usize],
    dispersion: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), EdgeErrors> {
    let n_obs = n_exon * n_samples;
    let n_coef = n_exon * n_groups;
    let n_interaction = (n_exon - 1) * (n_groups - 1);
    if n_coef >= n_obs {
        return Err(EdgeErrors::InvalidArgument(format!(
            "a gene with {n_exon} exons needs more than {n_groups} samples for the \
             exon-by-group interaction; got {n_samples}"
        )));
    }

    let mut design = vec![0.0_f64; n_obs * n_coef];
    for exon in 0..n_exon {
        for sample in 0..n_samples {
            let row = &mut design[(exon * n_samples + sample) * n_coef..][..n_coef];
            let level = group_level[sample];
            row[0] = 1.0;
            if exon > 0 {
                row[exon] = 1.0;
            }
            if level > 0 {
                row[n_exon - 1 + level] = 1.0;
            }
            if exon > 0 && level > 0 {
                row[n_exon + n_groups - 2 + (level - 1) * (n_exon - 1) + exon] = 1.0;
            }
        }
    }

    let mut unrolled: Vec<T> = Vec::with_capacity(batch.len() * n_obs);
    for &gene in batch {
        for &member in &groups.members[gene] {
            let exon = nonzero[member];
            unrolled.extend_from_slice(&counts[exon * n_samples..(exon + 1) * n_samples]);
        }
    }
    let disp = if dispersion.len() == 1 {
        Recycled::scalar(dispersion[0])
    } else {
        Recycled::by_gene(batch.iter().map(|g| dispersion[*g]).collect())
    };

    let offset = Recycled::scalar(0.0);
    let fit = glm_fit(
        &unrolled,
        batch.len(),
        n_obs,
        &design,
        n_coef,
        &disp,
        &offset,
        None,
        0.0,
    )?;
    let input = GlmTestInput {
        counts: &unrolled,
        n_genes: batch.len(),
        n_samples: n_obs,
        design: &design,
        n_coef,
        dispersion: &disp,
        offset: &offset,
        weights: None,
        log_cpm: None,
    };
    let tested = Tested::Coef((n_coef - n_interaction..n_coef).collect());
    let test = glm_lrt(&input, &fit, &tested, None)?;
    Ok((test.statistic, test.p_value))
}

/////////////
// Helpers //
/////////////

/// The coefficient under test, resolved against a possibly rewritten design.
struct ResolvedCoef {
    /// Design the tests run against, row-major `n_samples * n_coef`. The input
    /// design unless a contrast forced a change of basis.
    design: Vec<f64>,
    /// Column of `design` carrying the comparison.
    coef: usize,
    /// The tested effect per exon, one per input row.
    beta: Vec<f64>,
}

/// Picks out the tested coefficient, rotating the design if it is a contrast.
///
/// edgeR takes only the first entry of `coef` and only the first column of
/// `contrast`, warning as it does so. There is no warning channel here, so the
/// same silent truncation happens and is documented on [`diff_splice`].
///
/// ### Params
///
/// * `input` - The test input, for the design and its shape
/// * `fit` - The exon-level fit, for its coefficients
/// * `tested` - Coefficient index or contrast
///
/// ### Returns
///
/// The design, the tested column and the per-exon effect, or [`EdgeErrors`] if
/// the index is out of range, the contrast is empty or rank deficient, or the
/// shapes disagree.
fn resolve_tested<T: EdgeFloat>(
    input: &GlmTestInput<'_, T>,
    fit: &GlmFit,
    tested: &Tested,
) -> Result<ResolvedCoef, EdgeErrors> {
    let n_coef = input.n_coef;
    match tested {
        Tested::Coef(coefs) => {
            let coef = *coefs.first().ok_or_else(|| {
                EdgeErrors::InvalidArgument("no coefficient was given to test".to_string())
            })?;
            if coef >= n_coef {
                return Err(EdgeErrors::CoefOutOfRange {
                    index: coef,
                    n_coef,
                });
            }
            let beta = (0..input.n_genes)
                .map(|exon| fit.coefficients[exon * n_coef + coef])
                .collect();
            Ok(ResolvedCoef {
                design: input.design.to_vec(),
                coef,
                beta,
            })
        }
        Tested::Contrast {
            values,
            n_contrasts,
        } => {
            if *n_contrasts == 0 {
                return Err(EdgeErrors::InvalidArgument(
                    "no contrast was given to test".to_string(),
                ));
            }
            if values.len() != n_coef * n_contrasts {
                return Err(EdgeErrors::LengthMismatch {
                    name: "contrast",
                    expected: n_coef * n_contrasts,
                    got: values.len(),
                });
            }
            let first = &values[..n_coef];
            let reform = contrast_as_coef(input.design, input.n_samples, n_coef, first, 1, true)?;
            let beta = (0..input.n_genes)
                .map(|exon| {
                    let row = &fit.coefficients[exon * n_coef..(exon + 1) * n_coef];
                    row.iter().zip(first.iter()).map(|(b, c)| b * c).sum()
                })
                .collect();
            Ok(ResolvedCoef {
                design: reform.design,
                coef: reform.coef[0],
                beta,
            })
        }
    }
}

/// Checks the shapes [`diff_splice`] depends on.
///
/// ### Params
///
/// * `input` - The test input
/// * `fit` - The exon-level fit
/// * `gene_id` - Gene label per exon
/// * `ql` - Quasi-likelihood quantities, if any
/// * `params` - Tuning knobs, for `prior_count`
///
/// ### Returns
///
/// `Ok(())`, or the first [`EdgeErrors`] the shapes trip.
fn validate<T: EdgeFloat>(
    input: &GlmTestInput<'_, T>,
    fit: &GlmFit,
    gene_id: &[usize],
    ql: Option<&DiffSpliceQl<'_>>,
    params: &DiffSpliceParams,
) -> Result<(), EdgeErrors> {
    let (n_exons, n_samples, n_coef) = (input.n_genes, input.n_samples, input.n_coef);
    if n_exons == 0 || n_samples == 0 {
        return Err(EdgeErrors::EmptyCounts {
            n_genes: n_exons,
            n_samples,
        });
    }
    if n_coef < 2 {
        return Err(EdgeErrors::InvalidArgument(format!(
            "diff_splice needs a design with at least two columns, usually an intercept and the \
             comparison; got {n_coef}"
        )));
    }
    if input.counts.len() != n_exons * n_samples {
        return Err(EdgeErrors::LengthMismatch {
            name: "counts",
            expected: n_exons * n_samples,
            got: input.counts.len(),
        });
    }
    if input.design.len() != n_samples * n_coef {
        return Err(EdgeErrors::LengthMismatch {
            name: "design",
            expected: n_samples * n_coef,
            got: input.design.len(),
        });
    }
    if gene_id.len() != n_exons {
        return Err(EdgeErrors::LengthMismatch {
            name: "gene_id",
            expected: n_exons,
            got: gene_id.len(),
        });
    }
    if fit.coefficients.len() != n_exons * n_coef {
        return Err(EdgeErrors::LengthMismatch {
            name: "fit.coefficients",
            expected: n_exons * n_coef,
            got: fit.coefficients.len(),
        });
    }
    if fit.deviance.len() != n_exons {
        return Err(EdgeErrors::LengthMismatch {
            name: "fit.deviance",
            expected: n_exons,
            got: fit.deviance.len(),
        });
    }
    if params.prior_count < 0.0 || params.prior_count.is_nan() {
        return Err(EdgeErrors::InvalidArgument(format!(
            "prior_count must be non-negative, got {}",
            params.prior_count
        )));
    }
    input.dispersion.validate(n_exons, n_samples)?;
    input.offset.validate(n_exons, n_samples)?;
    if let Some(w) = input.weights {
        w.validate(n_exons, n_samples)?;
    }
    if let Some(q) = ql {
        if q.df_residual.len() != n_exons {
            return Err(EdgeErrors::LengthMismatch {
                name: "ql.df_residual",
                expected: n_exons,
                got: q.df_residual.len(),
            });
        }
        if q.deviance.len() != n_exons {
            return Err(EdgeErrors::LengthMismatch {
                name: "ql.deviance",
                expected: n_exons,
                got: q.deviance.len(),
            });
        }
        if q.df_prior.is_empty() {
            return Err(EdgeErrors::InvalidArgument(
                "ql.df_prior is empty".to_string(),
            ));
        }
    }
    Ok(())
}

/// Drops one column from a row-major matrix.
///
/// ### Params
///
/// * `matrix` - Row-major values, `n_rows * n_cols`
/// * `n_rows` - Number of rows
/// * `n_cols` - Number of columns
/// * `drop` - Column to remove
///
/// ### Returns
///
/// A fresh `n_rows * (n_cols - 1)` matrix, row-major.
fn drop_column(matrix: &[f64], n_rows: usize, n_cols: usize, drop: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(n_rows * (n_cols - 1));
    for row in matrix.chunks_exact(n_cols) {
        for (col, v) in row.iter().enumerate() {
            if col != drop {
                out.push(*v);
            }
        }
    }
    out
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Two genes of three exons each, the fixture from the gate command.
    fn gate() -> (Vec<f64>, Vec<usize>, Vec<f64>) {
        let counts = vec![
            8.0, 16.0, 32.0, 64.0, 128.0, 256.0, //
            4.0, 4.0, 8.0, 8.0, 16.0, 16.0, //
            64.0, 64.0, 64.0, 128.0, 128.0, 128.0, //
            12.0, 20.0, 28.0, 40.0, 56.0, 72.0, //
            100.0, 110.0, 90.0, 220.0, 200.0, 240.0, //
            6.0, 6.0, 6.0, 12.0, 12.0, 12.0, //
        ];
        let gene_id = vec![1, 1, 1, 2, 2, 2];
        let design = vec![
            1.0, 0.0, //
            1.0, 0.0, //
            1.0, 0.0, //
            1.0, 1.0, //
            1.0, 1.0, //
            1.0, 1.0, //
        ];
        (counts, gene_id, design)
    }

    /// Five genes of one to four exons, including an exon that is zero
    /// everywhere. Gene 2 has a single exon and must be dropped.
    fn mixed() -> (Vec<f64>, Vec<usize>, Vec<f64>) {
        let counts = vec![
            8.0, 16.0, 32.0, 64.0, 128.0, 256.0, //
            4.0, 4.0, 8.0, 8.0, 16.0, 16.0, //
            64.0, 64.0, 64.0, 128.0, 128.0, 128.0, //
            32.0, 16.0, 8.0, 4.0, 2.0, 1.0, //
            12.0, 20.0, 28.0, 40.0, 56.0, 72.0, //
            100.0, 110.0, 90.0, 220.0, 200.0, 240.0, //
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            6.0, 6.0, 6.0, 12.0, 12.0, 12.0, //
            256.0, 128.0, 192.0, 64.0, 96.0, 32.0, //
            8.0, 8.0, 8.0, 8.0, 8.0, 8.0, //
            20.0, 24.0, 16.0, 48.0, 40.0, 44.0, //
            512.0, 256.0, 128.0, 64.0, 32.0, 16.0, //
            96.0, 80.0, 64.0, 48.0, 32.0, 16.0, //
        ];
        let gene_id = vec![1, 1, 1, 1, 2, 3, 3, 3, 4, 4, 5, 5, 5];
        let design = vec![
            1.0, 0.0, //
            1.0, 0.0, //
            1.0, 0.0, //
            1.0, 1.0, //
            1.0, 1.0, //
            1.0, 1.0, //
        ];
        (counts, gene_id, design)
    }

    /// Library-size offsets, as `glmFit` derives them when none are supplied.
    fn offsets_from(counts: &[f64], n_exons: usize, n_samples: usize) -> Recycled<f64> {
        let mut lib = vec![0.0_f64; n_samples];
        for row in counts.chunks_exact(n_samples) {
            for (acc, v) in lib.iter_mut().zip(row.iter()) {
                *acc += v;
            }
        }
        assert_eq!(counts.len(), n_exons * n_samples);
        Recycled::by_sample(lib.iter().map(|v| v.ln()).collect())
    }

    /// Fits the exon level GLM the way `glmFit(y, X, dispersion = 0.1)` does.
    fn exon_fit(counts: &[f64], n_exons: usize, design: &[f64]) -> (GlmFit, Recycled<f64>) {
        let offset = offsets_from(counts, n_exons, 6);
        let fit = glm_fit(
            counts,
            n_exons,
            6,
            design,
            2,
            &Recycled::scalar(0.1),
            &offset,
            None,
            DEFAULT_PRIOR_COUNT,
        )
        .unwrap();
        (fit, offset)
    }

    /// The exon and gene level likelihood ratio tests, against edgeR 4.8.2:
    /// ```r
    /// y <- matrix(c(8,16,32,64,128,256, 4,4,8,8,16,16, 64,64,64,128,128,128,
    ///               12,20,28,40,56,72, 100,110,90,220,200,240, 6,6,6,12,12,12),
    ///             nrow = 6, byrow = TRUE)
    /// gid <- c(1,1,1,2,2,2)
    /// X <- cbind(1, c(0,0,0,1,1,1))
    /// f <- glmFit(y, X, dispersion = 0.1)
    /// s <- diffSpliceDGE(f, geneid = gid)
    /// ```
    #[test]
    fn test_matches_edger_lrt_on_the_gate_fixture() {
        let (counts, gene_id, design) = gate();
        let (fit, offset) = exon_fit(&counts, 6, &design);
        let input = GlmTestInput {
            counts: &counts,
            n_genes: 6,
            n_samples: 6,
            design: &design,
            n_coef: 2,
            dispersion: &Recycled::scalar(0.1),
            offset: &offset,
            weights: None,
            log_cpm: None,
        };
        let out = diff_splice(&input, &fit, &gene_id, &Tested::Coef(vec![1]), None, None).unwrap();

        // s$coefficients
        let expected_log_fc = [
            0.855_174_395_193_151_3,
            -0.261_843_130_654_270_2,
            -0.461_416_768_452_753_1,
            0.192_177_299_143_341_5,
            -0.035_676_657_669_298_6,
            -0.128_554_161_416_470_9,
        ];
        // s$exon.LR
        let expected_lr = [
            8.436_569_219_765_81,
            0.440_940_993_487_607_8,
            2.839_571_646_837_012_7,
            0.413_720_073_978_469,
            0.017_921_690_140_715_3,
            0.115_727_330_674_815_6,
        ];
        // s$exon.p.value
        let expected_p = [
            0.003_677_493_961_314_914_4,
            0.506_668_627_239_259_9,
            0.091_968_728_075_895_53,
            0.520_087_356_281_412_4,
            0.893_503_857_408_516_5,
            0.733_715_557_310_116_2,
        ];
        for i in 0..6 {
            assert_relative_eq!(out.exon_log_fc[i], expected_log_fc[i], max_relative = 1e-9);
            assert_relative_eq!(out.exon_statistic[i], expected_lr[i], max_relative = 1e-9);
            assert_relative_eq!(out.exon_p_value[i], expected_p[i], max_relative = 1e-9);
        }

        assert_eq!(out.gene_id, vec![1, 2]);
        assert_eq!(out.gene_n_exons, vec![3, 3]);
        // s$gene.LR, s$gene.p.value, s$gene.Simes.p.value
        assert_relative_eq!(
            out.gene_f_statistic[0],
            11.717_081_860_090_431,
            max_relative = 1e-9
        );
        assert_relative_eq!(
            out.gene_f_statistic[1],
            0.547_369_094_793_999_9,
            max_relative = 1e-9
        );
        assert_relative_eq!(
            out.gene_f_p_value[0],
            0.008_417_910_555_119_631,
            max_relative = 1e-9
        );
        assert_relative_eq!(
            out.gene_f_p_value[1],
            0.908_367_971_898_078,
            max_relative = 1e-9
        );
        assert_relative_eq!(
            out.gene_simes_p[0],
            0.011_032_481_883_944_744,
            max_relative = 1e-9
        );
        assert_relative_eq!(
            out.gene_simes_p[1],
            0.893_503_857_408_516_5,
            max_relative = 1e-9
        );
    }

    /// Genes of one, two, three and four exons, one of which is zero in every
    /// sample. Against edgeR 4.8.2:
    /// ```r
    /// y <- matrix(c(8,16,32,64,128,256, 4,4,8,8,16,16, 64,64,64,128,128,128,
    ///               32,16,8,4,2,1, 12,20,28,40,56,72, 100,110,90,220,200,240,
    ///               0,0,0,0,0,0, 6,6,6,12,12,12, 256,128,192,64,96,32,
    ///               8,8,8,8,8,8, 20,24,16,48,40,44, 512,256,128,64,32,16,
    ///               96,80,64,48,32,16), nrow = 13, byrow = TRUE)
    /// gid <- c(1,1,1,1,2,3,3,3,4,4,5,5,5)
    /// X <- cbind(1, c(0,0,0,1,1,1))
    /// s <- diffSpliceDGE(glmFit(y, X, dispersion = 0.1), geneid = gid)
    /// ```
    #[test]
    fn test_matches_edger_lrt_with_mixed_exon_counts() {
        let (counts, gene_id, design) = mixed();
        let (fit, offset) = exon_fit(&counts, 13, &design);
        let input = GlmTestInput {
            counts: &counts,
            n_genes: 13,
            n_samples: 6,
            design: &design,
            n_coef: 2,
            dispersion: &Recycled::scalar(0.1),
            offset: &offset,
            weights: None,
            log_cpm: None,
        };
        let out = diff_splice(&input, &fit, &gene_id, &Tested::Coef(vec![1]), None, None).unwrap();

        // The single-exon gene is dropped, so its row carries the neutral values.
        assert_eq!(out.gene_id, vec![1, 3, 4, 5]);
        assert_eq!(out.gene_n_exons, vec![4, 3, 2, 3]);
        assert_eq!(out.exon_log_fc[4], 0.0);
        assert_eq!(out.exon_statistic[4], 0.0);
        assert_eq!(out.exon_p_value[4], 1.0);

        // s$coefficients, over the twelve kept exons in input order
        let expected_log_fc = [
            0.986_261_373_785_026_8,
            -0.092_813_816_117_244_93,
            -0.299_573_258_165_403_7,
            -2.932_713_900_334_851_7,
            0.002_284_551_340_094_198_6,
            -0.822_162_460_092_049_4,
            -0.078_425_809_463_148_25,
            -0.072_451_134_208_873_84,
            0.991_832_771_986_186_4,
            1.951_633_344_827_402_4,
            -0.785_350_396_075_128_2,
            0.286_727_193_468_867,
        ];
        // s$exon.LR
        let expected_lr = [
            11.203_352_118_219_136,
            0.039_045_494_218_443_72,
            1.189_460_502_071_776_1,
            49.591_886_657_154_48,
            0.000_124_411_095_185_239_52,
            0.0,
            0.029_691_603_087_216_08,
            0.073_511_269_031_998_34,
            6.456_369_674_304_806,
            39.738_828_764_140_41,
            7.975_958_887_749_272,
            0.991_332_488_893_542_4,
        ];
        // s$exon.p.value
        let expected_p = [
            0.000_816_497_019_046_571_5,
            0.843_358_561_026_85,
            0.275_438_406_419_161,
            1.892_949_108_888_925e-12,
            0.991_100_602_344_373_6,
            1.0,
            0.863_191_845_368_248_2,
            0.786_291_343_598_799_3,
            0.011_055_528_859_942_796,
            2.902_963_175_866_408e-10,
            0.004_740_264_033_567_157,
            0.319_416_920_511_383_43,
        ];
        let kept: Vec<usize> = (0..13).filter(|i| *i != 4).collect();
        for (slot, &exon) in kept.iter().enumerate() {
            assert_relative_eq!(
                out.exon_log_fc[exon],
                expected_log_fc[slot],
                max_relative = 1e-9,
                epsilon = 1e-12
            );
            assert_relative_eq!(
                out.exon_statistic[exon],
                expected_lr[slot],
                max_relative = 1e-9,
                epsilon = 1e-12
            );
            assert_relative_eq!(
                out.exon_p_value[exon],
                expected_p[slot],
                max_relative = 1e-9
            );
        }

        // s$gene.LR
        let expected_gene_lr = [
            62.023_744_771_663_84,
            0.029_816_014_182_401_318,
            6.529_880_943_336_805,
            48.706_120_140_783_22,
        ];
        // s$gene.p.value
        let expected_gene_p = [
            1.088_995_352_370_079_7e-12,
            0.998_642_900_101_906_2,
            0.038_199_208_951_176_705,
            1.506_569_551_839_668e-10,
        ];
        // s$gene.Simes.p.value
        let expected_simes = [
            7.571_796_435_555_7e-12,
            1.0,
            0.022_111_057_719_885_592,
            8.708_889_527_599_224e-10,
        ];
        for g in 0..4 {
            assert_relative_eq!(
                out.gene_f_statistic[g],
                expected_gene_lr[g],
                max_relative = 1e-9
            );
            assert_relative_eq!(
                out.gene_f_p_value[g],
                expected_gene_p[g],
                max_relative = 1e-9
            );
            assert_relative_eq!(out.gene_simes_p[g], expected_simes[g], max_relative = 1e-9);
        }
    }

    /// The quasi-likelihood path, against edgeR 4.8.2:
    /// ```r
    /// qf <- glmQLFit(y, X, dispersion = 0.1, legacy = FALSE)   # y, X as above
    /// s <- diffSpliceDGE(qf, geneid = gid)
    /// ```
    ///
    /// The fit is edgeR's own, embedded as literals rather than routed through
    /// `glm_ql_fit`, so that this exercises `diff_splice` and nothing else.
    #[test]
    fn test_matches_edger_quasi_likelihood_f_tests() {
        let (counts, gene_id, design) = mixed();
        let offset = offsets_from(&counts, 13, 6);

        // as.vector(t(qf$coefficients))
        let coefficients = vec![
            -3.716_649_123_803_044,
            2.053_338_438_767_273_3,
            -5.004_790_281_784_431,
            0.958_599_359_856_350_5,
            -2.521_384_664_641_309,
            0.736_712_308_560_846_2,
            -3.818_356_521_647_034_5,
            -1.922_703_330_975_371_8,
            -3.662_138_872_007_650_6,
            1.039_972_007_661_760_2,
            -2.075_371_926_749_308,
            0.829_195_942_258_679_3,
            -8.762_906_779_045_517,
            0.0,
            -4.898_543_552_754_313,
            0.753_138_951_659_528_8,
            -1.453_993_800_252_147,
            -1.010_073_952_288_818_5,
            -4.613_303_129_296_377,
            0.066_946_339_544_628_59,
            -3.698_761_496_409_288_5,
            0.848_147_053_835_922_5,
            -1.086_383_432_125_770_6,
            -1.908_243_414_438_834,
            -2.323_859_444_983_016,
            -0.829_704_616_425_026_7,
        ];
        // qf$deviance
        let deviance = vec![
            32.934_761_083_769_92,
            5.474_240_344_388_582_5,
            3.263_241_513_475_087_6,
            5.470_865_795_828_582_5,
            11.857_671_991_595_268,
            3.339_511_158_808_827_8,
            0.0,
            0.950_343_670_388_99,
            15.350_643_930_570_67,
            1.123_733_784_713_515_8,
            2.832_718_144_252_497_4,
            24.339_069_453_856_304,
            10.860_457_084_684_125,
        ];
        // qf$df.residual.adj
        let df_residual_adj = [
            3.949_399_184_578_337,
            4.429_910_119_524_08,
            3.995_636_528_129_935_5,
            6.215_119_099_483_442,
            3.947_839_985_398_613_5,
            3.999_088_419_001_173,
            0.0,
            4.298_601_889_799_675,
            3.996_644_613_146_141,
            4.160_749_445_488_15,
            3.941_931_767_768_654,
            3.987_982_672_430_048,
            3.979_648_238_150_599,
        ];
        // qf$deviance.adj
        let deviance_adj = [
            31.681_571_682_117_89,
            5.716_898_207_948_569,
            3.199_241_360_732_763_4,
            7.325_504_268_111_918,
            11.299_290_927_664_186,
            3.281_195_035_994_136,
            0.0,
            0.943_989_863_873_622_4,
            15.053_574_377_441_105,
            1.055_510_220_953_673_5,
            2.703_138_273_427_556_6,
            23.743_938_823_095_38,
            10.517_433_786_052_315,
        ];
        // qf$df.prior, qf$average.ql.dispersion
        let df_prior = [50.431_634_778_955_26];
        let average_ql_dispersion = 2.517_465_302_235_576;

        let fit = GlmFit {
            coefficients,
            unshrunk_coefficients: None,
            fitted: Vec::new(),
            deviance,
            df_residual: 4,
            method: crate::glm::fit::FitMethod::OneWay,
        };
        let ql = DiffSpliceQl {
            df_prior: &df_prior,
            df_residual: &df_residual_adj,
            deviance: &deviance_adj,
            legacy_zeros: false,
            average_ql_dispersion: Some(average_ql_dispersion),
        };
        let input = GlmTestInput {
            counts: &counts,
            n_genes: 13,
            n_samples: 6,
            design: &design,
            n_coef: 2,
            // glmQLFit stores the undivided dispersion; diff_splice divides it.
            dispersion: &Recycled::scalar(0.1),
            offset: &offset,
            weights: None,
            log_cpm: None,
        };
        let out = diff_splice(
            &input,
            &fit,
            &gene_id,
            &Tested::Coef(vec![1]),
            Some(&ql),
            None,
        )
        .unwrap();

        // s$exon.F over the twelve kept exons
        let expected_f = [
            10.923_983_891_986_724,
            0.012_553_476_455_043_623,
            1.147_597_224_575_699_3,
            33.801_175_852_216_82,
            0.000_892_201_161_999_136_9,
            0.0,
            0.013_982_948_088_757_496,
            0.080_179_046_492_054_87,
            4.186_641_136_888_011,
            35.647_046_431_592_5,
            8.176_517_864_533_706,
            0.884_633_969_063_336_8,
        ];
        // s$exon.p.value
        let expected_p = [
            0.001_823_606_690_634_109_5,
            0.911_267_482_829_818_5,
            0.289_528_269_680_133_8,
            5.158_549_168_775_326e-7,
            0.976_297_542_709_692_6,
            1.0,
            0.906_374_380_415_774_7,
            0.778_299_597_202_179_1,
            0.046_366_642_339_368_824,
            2.992_427_861_639_236_6e-7,
            0.006_311_974_788_731_110_5,
            0.351_749_366_873_619_46,
        ];
        // s$coefficients
        let expected_log_fc = [
            1.024_069_012_156_488_6,
            -0.070_670_066_754_434_21,
            -0.292_557_118_049_938_45,
            -2.951_972_757_586_156_5,
            0.007_033_482_166_629_912,
            -0.822_162_460_092_049_4,
            -0.069_023_508_432_520_59,
            -0.074_627_228_010_850_64,
            1.002_393_063_822_596_4,
            1.960_274_530_132_001,
            -0.796_115_938_142_755_4,
            0.282_422_859_871_052,
        ];
        let kept: Vec<usize> = (0..13).filter(|i| *i != 4).collect();
        for (slot, &exon) in kept.iter().enumerate() {
            assert_relative_eq!(
                out.exon_log_fc[exon],
                expected_log_fc[slot],
                max_relative = 1e-9,
                epsilon = 1e-12
            );
            assert_relative_eq!(
                out.exon_statistic[exon],
                expected_f[slot],
                max_relative = 1e-9,
                epsilon = 1e-12
            );
            assert_relative_eq!(
                out.exon_p_value[exon],
                expected_p[slot],
                max_relative = 1e-9
            );
        }

        // s$gene.F, s$gene.p.value, s$gene.Simes.p.value
        let expected_gene_f = [
            11.471_327_611_308_572,
            0.004_958_383_083_585_545,
            2.133_410_091_690_033,
            14.902_732_755_063_182,
        ];
        let expected_gene_p = [
            1.412_135_847_820_031_8e-6,
            0.999_512_145_784_596_1,
            0.129_771_944_348_671_37,
            5.987_503_169_646_61e-7,
        ];
        let expected_simes = [
            2.063_419_667_510_130_6e-6,
            1.0,
            0.092_733_284_678_737_65,
            8.977_283_584_917_71e-7,
        ];
        for g in 0..4 {
            assert_relative_eq!(
                out.gene_f_statistic[g],
                expected_gene_f[g],
                max_relative = 1e-9
            );
            assert_relative_eq!(
                out.gene_f_p_value[g],
                expected_gene_p[g],
                max_relative = 1e-9
            );
            assert_relative_eq!(out.gene_simes_p[g], expected_simes[g], max_relative = 1e-9);
        }
    }

    /// Testing a contrast that picks out the same column must reproduce the
    /// coefficient path exactly, since the rotation leaves that column alone.
    #[test]
    fn test_contrast_reproduces_the_coefficient_path() {
        let (counts, gene_id, design) = gate();
        let (fit, offset) = exon_fit(&counts, 6, &design);
        let input = GlmTestInput {
            counts: &counts,
            n_genes: 6,
            n_samples: 6,
            design: &design,
            n_coef: 2,
            dispersion: &Recycled::scalar(0.1),
            offset: &offset,
            weights: None,
            log_cpm: None,
        };
        let by_coef =
            diff_splice(&input, &fit, &gene_id, &Tested::Coef(vec![1]), None, None).unwrap();
        let by_contrast = diff_splice(
            &input,
            &fit,
            &gene_id,
            &Tested::Contrast {
                values: vec![0.0, 1.0],
                n_contrasts: 1,
            },
            None,
            None,
        )
        .unwrap();

        for i in 0..6 {
            assert_relative_eq!(
                by_contrast.exon_p_value[i],
                by_coef.exon_p_value[i],
                max_relative = 1e-9
            );
            assert_relative_eq!(
                by_contrast.exon_log_fc[i],
                by_coef.exon_log_fc[i],
                max_relative = 1e-9,
                epsilon = 1e-12
            );
        }
    }

    /// Genes are grouped by label, not by position, so shuffling the exon rows
    /// permutes the answers rather than changing them.
    #[test]
    fn test_gene_grouping_is_independent_of_row_order() {
        let (counts, gene_id, design) = gate();
        let (fit, offset) = exon_fit(&counts, 6, &design);
        let input = GlmTestInput {
            counts: &counts,
            n_genes: 6,
            n_samples: 6,
            design: &design,
            n_coef: 2,
            dispersion: &Recycled::scalar(0.1),
            offset: &offset,
            weights: None,
            log_cpm: None,
        };
        let sorted =
            diff_splice(&input, &fit, &gene_id, &Tested::Coef(vec![1]), None, None).unwrap();

        // Interleave the two genes: 1, 2, 1, 2, 1, 2.
        let order = [0_usize, 3, 1, 4, 2, 5];
        let shuffled_counts: Vec<f64> = order
            .iter()
            .flat_map(|r| counts[r * 6..(r + 1) * 6].iter().copied())
            .collect();
        let shuffled_ids: Vec<usize> = order.iter().map(|r| gene_id[*r]).collect();
        let (shuffled_fit, shuffled_offset) = exon_fit(&shuffled_counts, 6, &design);
        let shuffled_input = GlmTestInput {
            counts: &shuffled_counts,
            n_genes: 6,
            n_samples: 6,
            design: &design,
            n_coef: 2,
            dispersion: &Recycled::scalar(0.1),
            offset: &shuffled_offset,
            weights: None,
            log_cpm: None,
        };
        let shuffled = diff_splice(
            &shuffled_input,
            &shuffled_fit,
            &shuffled_ids,
            &Tested::Coef(vec![1]),
            None,
            None,
        )
        .unwrap();

        assert_eq!(shuffled.gene_id, sorted.gene_id);
        for g in 0..2 {
            assert_relative_eq!(
                shuffled.gene_simes_p[g],
                sorted.gene_simes_p[g],
                max_relative = 1e-9
            );
        }
        for (slot, &row) in order.iter().enumerate() {
            assert_relative_eq!(
                shuffled.exon_p_value[slot],
                sorted.exon_p_value[row],
                max_relative = 1e-9
            );
        }
    }

    /// The `DgeList` wrapper must reproduce the explicit call.
    #[test]
    fn test_dge_wrapper_matches_the_explicit_call() {
        let (counts, gene_id, design) = gate();
        let (fit, offset) = exon_fit(&counts, 6, &design);
        let input = GlmTestInput {
            counts: &counts,
            n_genes: 6,
            n_samples: 6,
            design: &design,
            n_coef: 2,
            dispersion: &Recycled::scalar(0.1),
            offset: &offset,
            weights: None,
            log_cpm: None,
        };
        let direct =
            diff_splice(&input, &fit, &gene_id, &Tested::Coef(vec![1]), None, None).unwrap();

        let mut dge = DgeList::new(counts.clone(), 6, 6, None).unwrap();
        dge.common_dispersion = Some(0.1);
        let wrapped =
            diff_splice_dge(&dge, &design, 2, &gene_id, &Tested::Coef(vec![1]), None).unwrap();

        for i in 0..6 {
            assert_relative_eq!(
                wrapped.exon_p_value[i],
                direct.exon_p_value[i],
                max_relative = 1e-9
            );
            assert_relative_eq!(
                wrapped.exon_log_fc[i],
                direct.exon_log_fc[i],
                max_relative = 1e-9
            );
        }
    }

    /// `spliceVariants` against edgeR 4.8.2, on the mixed fixture:
    /// ```r
    /// sv <- spliceVariants(y, geneID = gid, dispersion = 0.1,
    ///                      group = factor(c(1,1,1,2,2,2)))
    /// ```
    #[test]
    fn test_matches_edger_splice_variants() {
        let (counts, gene_id, _) = mixed();
        let group = vec![1, 1, 1, 2, 2, 2];
        let out = splice_variants(&counts, 13, 6, &gene_id, &group, &[0.1]).unwrap();

        assert_eq!(out.gene_id, vec![1, 2, 3, 4, 5]);
        // The all-zero exon is dropped, so gene 3 arrives with two exons.
        assert_eq!(out.gene_n_exons, vec![4, 1, 2, 2, 3]);
        assert!(out.exon_p_value.is_empty());

        // sv$table$LR
        let expected_lr = [
            64.964_266_458_512_41,
            0.0,
            0.040_945_652_052_297_28,
            5.370_218_472_941_118,
            47.582_028_134_328_2,
        ];
        // sv$table$PValue
        let expected_p = [
            5.105_088_859_960_058e-14,
            1.0,
            0.839_642_863_748_024_1,
            0.020_483_411_823_205_97,
            4.652_580_238_805_525e-11,
        ];
        // The unrolled full model is a one-way layout and fits in closed form,
        // but the null model that drops the interaction is not, so it goes
        // through the damped iteration. The likelihood ratio is a difference of
        // two deviances near 10^2 and inherits the fitter's own convergence
        // tolerance rather than machine precision; 1e-6 is what that buys. The
        // rest of this module agrees with edgeR to 1e-9.
        for g in 0..5 {
            assert_relative_eq!(
                out.gene_f_statistic[g],
                expected_lr[g],
                max_relative = 1e-6,
                epsilon = 1e-12
            );
            assert_relative_eq!(out.gene_f_p_value[g], expected_p[g], max_relative = 1e-6);
            assert_relative_eq!(out.gene_simes_p[g], expected_p[g], max_relative = 1e-6);
        }
    }

    /// A per-gene dispersion vector must be accepted and must move the answer.
    #[test]
    fn test_splice_variants_accepts_a_genewise_dispersion() {
        let (counts, gene_id, _) = mixed();
        let group = vec![1, 1, 1, 2, 2, 2];
        let common = splice_variants(&counts, 13, 6, &gene_id, &group, &[0.1]).unwrap();
        let genewise =
            splice_variants(&counts, 13, 6, &gene_id, &group, &[0.1, 0.1, 0.1, 0.5, 0.1]).unwrap();

        assert_relative_eq!(
            genewise.gene_f_statistic[0],
            common.gene_f_statistic[0],
            max_relative = 1e-12
        );
        assert!(genewise.gene_f_statistic[3] < common.gene_f_statistic[3]);
    }

    /// Simes on a single p-value is that p-value, and the largest term bounds
    /// the result by one.
    #[test]
    fn test_simes_bounds() {
        assert_relative_eq!(simes(&[0.4]), 0.4, max_relative = 1e-12);
        assert_relative_eq!(simes(&[0.5, 0.73, 0.89]), 0.89, max_relative = 1e-12);
        assert_relative_eq!(simes(&[0.01, 0.5, 0.9]), 0.03, max_relative = 1e-12);
        assert_eq!(simes(&[]), 1.0);
    }

    /// A design with one column has no comparison to make.
    #[test]
    fn test_rejects_a_single_column_design() {
        let (counts, gene_id, _) = gate();
        let design = vec![1.0; 6];
        let offset = offsets_from(&counts, 6, 6);
        let fit = glm_fit(
            &counts,
            6,
            6,
            &design,
            1,
            &Recycled::scalar(0.1),
            &offset,
            None,
            0.0,
        )
        .unwrap();
        let input = GlmTestInput {
            counts: &counts,
            n_genes: 6,
            n_samples: 6,
            design: &design,
            n_coef: 1,
            dispersion: &Recycled::scalar(0.1),
            offset: &offset,
            weights: None,
            log_cpm: None,
        };
        let err =
            diff_splice(&input, &fit, &gene_id, &Tested::Coef(vec![0]), None, None).unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));
    }

    #[test]
    fn test_rejects_a_gene_id_of_the_wrong_length() {
        let (counts, _, design) = gate();
        let (fit, offset) = exon_fit(&counts, 6, &design);
        let input = GlmTestInput {
            counts: &counts,
            n_genes: 6,
            n_samples: 6,
            design: &design,
            n_coef: 2,
            dispersion: &Recycled::scalar(0.1),
            offset: &offset,
            weights: None,
            log_cpm: None,
        };
        let err =
            diff_splice(&input, &fit, &[1, 1, 2], &Tested::Coef(vec![1]), None, None).unwrap_err();
        assert!(matches!(
            err,
            EdgeErrors::LengthMismatch {
                name: "gene_id",
                expected: 6,
                got: 3
            }
        ));
    }

    #[test]
    fn test_rejects_a_coefficient_out_of_range() {
        let (counts, gene_id, design) = gate();
        let (fit, offset) = exon_fit(&counts, 6, &design);
        let input = GlmTestInput {
            counts: &counts,
            n_genes: 6,
            n_samples: 6,
            design: &design,
            n_coef: 2,
            dispersion: &Recycled::scalar(0.1),
            offset: &offset,
            weights: None,
            log_cpm: None,
        };
        let err =
            diff_splice(&input, &fit, &gene_id, &Tested::Coef(vec![7]), None, None).unwrap_err();
        assert!(matches!(
            err,
            EdgeErrors::CoefOutOfRange {
                index: 7,
                n_coef: 2
            }
        ));
    }

    #[test]
    fn test_rejects_an_empty_coefficient_list() {
        let (counts, gene_id, design) = gate();
        let (fit, offset) = exon_fit(&counts, 6, &design);
        let input = GlmTestInput {
            counts: &counts,
            n_genes: 6,
            n_samples: 6,
            design: &design,
            n_coef: 2,
            dispersion: &Recycled::scalar(0.1),
            offset: &offset,
            weights: None,
            log_cpm: None,
        };
        let err = diff_splice(
            &input,
            &fit,
            &gene_id,
            &Tested::Coef(Vec::new()),
            None,
            None,
        )
        .unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));
    }

    #[test]
    fn test_rejects_a_contrast_of_the_wrong_length() {
        let (counts, gene_id, design) = gate();
        let (fit, offset) = exon_fit(&counts, 6, &design);
        let input = GlmTestInput {
            counts: &counts,
            n_genes: 6,
            n_samples: 6,
            design: &design,
            n_coef: 2,
            dispersion: &Recycled::scalar(0.1),
            offset: &offset,
            weights: None,
            log_cpm: None,
        };
        let err = diff_splice(
            &input,
            &fit,
            &gene_id,
            &Tested::Contrast {
                values: vec![1.0, 0.0, 0.0],
                n_contrasts: 1,
            },
            None,
            None,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            EdgeErrors::LengthMismatch {
                name: "contrast",
                ..
            }
        ));
    }

    /// Every gene having a single exon leaves nothing testable.
    #[test]
    fn test_rejects_when_no_gene_has_more_than_one_exon() {
        let (counts, _, design) = gate();
        let (fit, offset) = exon_fit(&counts, 6, &design);
        let input = GlmTestInput {
            counts: &counts,
            n_genes: 6,
            n_samples: 6,
            design: &design,
            n_coef: 2,
            dispersion: &Recycled::scalar(0.1),
            offset: &offset,
            weights: None,
            log_cpm: None,
        };
        let err = diff_splice(
            &input,
            &fit,
            &[1, 2, 3, 4, 5, 6],
            &Tested::Coef(vec![1]),
            None,
            None,
        )
        .unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));
    }

    #[test]
    fn test_rejects_a_negative_prior_count() {
        let (counts, gene_id, design) = gate();
        let (fit, offset) = exon_fit(&counts, 6, &design);
        let input = GlmTestInput {
            counts: &counts,
            n_genes: 6,
            n_samples: 6,
            design: &design,
            n_coef: 2,
            dispersion: &Recycled::scalar(0.1),
            offset: &offset,
            weights: None,
            log_cpm: None,
        };
        let err = diff_splice(
            &input,
            &fit,
            &gene_id,
            &Tested::Coef(vec![1]),
            None,
            Some(DiffSpliceParams::new(-1.0, None)),
        )
        .unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));
    }

    #[test]
    fn test_rejects_a_quasi_likelihood_summary_of_the_wrong_length() {
        let (counts, gene_id, design) = gate();
        let (fit, offset) = exon_fit(&counts, 6, &design);
        let input = GlmTestInput {
            counts: &counts,
            n_genes: 6,
            n_samples: 6,
            design: &design,
            n_coef: 2,
            dispersion: &Recycled::scalar(0.1),
            offset: &offset,
            weights: None,
            log_cpm: None,
        };
        let ql = DiffSpliceQl {
            df_prior: &[5.0],
            df_residual: &[4.0, 4.0],
            deviance: &[1.0; 6],
            legacy_zeros: false,
            average_ql_dispersion: None,
        };
        let err = diff_splice(
            &input,
            &fit,
            &gene_id,
            &Tested::Coef(vec![1]),
            Some(&ql),
            None,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            EdgeErrors::LengthMismatch {
                name: "ql.df_residual",
                ..
            }
        ));
    }

    #[test]
    fn test_rejects_a_fit_of_the_wrong_length() {
        let (counts, gene_id, design) = gate();
        let (mut fit, offset) = exon_fit(&counts, 6, &design);
        fit.deviance.pop();
        let input = GlmTestInput {
            counts: &counts,
            n_genes: 6,
            n_samples: 6,
            design: &design,
            n_coef: 2,
            dispersion: &Recycled::scalar(0.1),
            offset: &offset,
            weights: None,
            log_cpm: None,
        };
        let err =
            diff_splice(&input, &fit, &gene_id, &Tested::Coef(vec![1]), None, None).unwrap_err();
        assert!(matches!(
            err,
            EdgeErrors::LengthMismatch {
                name: "fit.deviance",
                ..
            }
        ));
    }

    #[test]
    fn test_splice_variants_rejects_a_single_group() {
        let (counts, gene_id, _) = mixed();
        let err = splice_variants(&counts, 13, 6, &gene_id, &[1; 6], &[0.1]).unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));
    }

    #[test]
    fn test_splice_variants_rejects_a_bad_dispersion_length() {
        let (counts, gene_id, _) = mixed();
        let group = vec![1, 1, 1, 2, 2, 2];
        let err = splice_variants(&counts, 13, 6, &gene_id, &group, &[0.1, 0.2]).unwrap_err();
        assert!(matches!(
            err,
            EdgeErrors::LengthMismatch {
                name: "dispersion",
                ..
            }
        ));
    }

    #[test]
    fn test_splice_variants_rejects_a_negative_dispersion() {
        let (counts, gene_id, _) = mixed();
        let group = vec![1, 1, 1, 2, 2, 2];
        let err = splice_variants(&counts, 13, 6, &gene_id, &group, &[-0.1]).unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidDispersion(_)));
    }

    #[test]
    fn test_splice_variants_rejects_an_all_zero_matrix() {
        let err = splice_variants(&[0.0_f64; 12], 2, 6, &[1, 1], &[1, 1, 1, 2, 2, 2], &[0.1])
            .unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));
    }

    #[test]
    fn test_splice_variants_rejects_a_group_of_the_wrong_length() {
        let (counts, gene_id, _) = mixed();
        let err = splice_variants(&counts, 13, 6, &gene_id, &[1, 2], &[0.1]).unwrap_err();
        assert!(matches!(
            err,
            EdgeErrors::LengthMismatch { name: "group", .. }
        ));
    }

    #[test]
    fn test_splice_variants_rejects_empty_counts() {
        let err =
            splice_variants(&[0.0_f64; 0], 0, 6, &[], &[1, 1, 1, 2, 2, 2], &[0.1]).unwrap_err();
        assert!(matches!(err, EdgeErrors::EmptyCounts { .. }));
    }

    /// `f32` counts must run the same algorithm; every derived statistic is
    /// `f64` either way, so only the count conversion differs.
    #[test]
    fn test_runs_on_f32_counts() {
        let (counts, gene_id, design) = gate();
        let counts32: Vec<f32> = counts.iter().map(|v| *v as f32).collect();
        let offset = offsets_from(&counts, 6, 6);
        let fit = glm_fit(
            &counts32,
            6,
            6,
            &design,
            2,
            &Recycled::scalar(0.1),
            &offset,
            None,
            DEFAULT_PRIOR_COUNT,
        )
        .unwrap();
        let input = GlmTestInput {
            counts: &counts32,
            n_genes: 6,
            n_samples: 6,
            design: &design,
            n_coef: 2,
            dispersion: &Recycled::scalar(0.1),
            offset: &offset,
            weights: None,
            log_cpm: None,
        };
        let out = diff_splice(&input, &fit, &gene_id, &Tested::Coef(vec![1]), None, None).unwrap();
        assert_relative_eq!(
            out.gene_simes_p[0],
            0.011_032_481_883_944_744,
            max_relative = 1e-6
        );
    }
}
