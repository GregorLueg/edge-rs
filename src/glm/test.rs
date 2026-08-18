//! Turning a fitted negative binomial GLM into p-values.
//!
//! Three tests, all of them nested-model comparisons against the same fit:
//! [`glm_lrt`] refits the null and reads the deviance difference off a
//! chi-squared, [`glm_ql_ftest`] divides that difference by a squeezed
//! quasi-likelihood dispersion and reads an F, and [`glm_treat`] shifts the
//! offsets by a fold-change threshold and tests against an interval null rather
//! than a point.
//!
//! Only [`glm_lrt`] and [`glm_treat`] refit anything of their own.
//! [`glm_ql_ftest`] is handed the quasi-likelihood quantities in a
//! [`QlSummary`], because the squeezing that produces them belongs to
//! `glmQLFit`, not here.
//!
//! Everything a test needs about the data it is testing travels in one
//! [`GlmTestInput`], so the three entry points stay narrow and the caller
//! assembles the shared half once.

use rayon::prelude::*;

use crate::errors::EdgeErrors;
use crate::glm::fit::{GlmFit, glm_fit};
use crate::glm::levenberg::{LevenbergParams, mglm_levenberg};
use crate::numeric::dist::{chisq_sf, f_sf, norm_cdf, norm_ppf, t_sf};
use crate::utils::design::{contrast_as_coef, design_as_factor, matrix_rank, non_estimable};
use crate::utils::recycled::Recycled;
use crate::utils::traits::EdgeFloat;

/// Iteration budget for a warm-started null fit.
///
/// Mirrors the budget `glm_fit` gives its own general path. The null fit is the
/// last thing standing between the caller and a missing p-value, so it gets the
/// same allowance rather than `mglmLevenberg`'s stingier default of 200.
const NULL_FIT_MAX_ITER: usize = 250;

/// Half-width of the interval null in [`glm_treat`], on the z scale.
///
/// edgeR's `glmTreat` hard-codes this. It is the constant that makes the
/// interval null's p-value agree with the point null's at the boundary; see
/// McCarthy and Smyth (2009).
const TREAT_INTERVAL_WIDTH: f64 = 1.470402;

/// Largest magnitude a treat z-score is allowed to reach.
///
/// `norm_sf(38)` is about 2.9e-316, so past this the tail is numerically zero
/// anyway. The clamp exists because the interval null divides by `b - a`: an
/// infinite z would turn the whole p-value into a NaN rather than into the zero
/// it is trying to express.
const MAX_TREAT_ZSCORE: f64 = 38.0;

/// `1 / sqrt(2 * pi)`, the standard normal density's normalising constant.
const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

//////////////////////
// Public interface //
//////////////////////

/// What a test is aimed at.
#[derive(Clone, Debug)]
pub enum Tested {
    /// Design columns to drop from the null model, as zero-based indices.
    ///
    /// Repeats are collapsed, keeping the first occurrence, as edgeR's
    /// `unique` does. The first surviving index is the one whose coefficient
    /// becomes the reported log-fold-change.
    Coef(Vec<usize>),
    /// A contrast, or several, over the coefficients.
    Contrast {
        /// Column-major `n_coef * n_contrasts` values.
        values: Vec<f64>,
        /// Number of contrast columns. Must have full column rank.
        n_contrasts: usize,
    },
}

/// Which null [`glm_treat`] tests against.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TreatNull {
    /// The whole interval `[-lfc, lfc]` is null. edgeR's default, and the one
    /// that keeps the test from being anti-conservative near the threshold.
    #[default]
    Interval,
    /// Only the worst point of the interval is null, giving the conservative
    /// `pnorm(-z_right) + pnorm(z_left)`.
    WorstCase,
}

/// Everything a test needs to know about the data behind the fit.
///
/// Layouts follow the crate's conventions: counts are row-major
/// `n_genes * n_samples`, the design is row-major `n_samples * n_coef`, and the
/// recycled matrices are logically `n_genes * n_samples`.
#[derive(Clone, Debug)]
pub struct GlmTestInput<'a, T: EdgeFloat> {
    /// Counts, row-major `n_genes * n_samples`.
    pub counts: &'a [T],
    /// Number of genes.
    pub n_genes: usize,
    /// Number of samples.
    pub n_samples: usize,
    /// Design matrix, row-major `n_samples * n_coef`. The same one the fit used.
    pub design: &'a [f64],
    /// Number of coefficients. At least two, since the null model must keep one.
    pub n_coef: usize,
    /// Dispersion the fit used, recycled over genes and samples.
    pub dispersion: &'a Recycled<f64>,
    /// Log-scale offsets, recycled over genes and samples.
    pub offset: &'a Recycled<f64>,
    /// Optional observation weights.
    pub weights: Option<&'a Recycled<f64>>,
    /// Optional average log-CPM per gene, copied straight into the result.
    pub log_cpm: Option<&'a [f64]>,
}

/// The quasi-likelihood quantities `glmQLFit` produces.
///
/// [`glm_ql_ftest`] and the quasi-likelihood flavour of [`glm_treat`] read these
/// rather than recomputing them, exactly as edgeR reads them off the `DGEGLM`.
#[derive(Clone, Debug)]
pub struct QlSummary<'a> {
    /// Posterior quasi-likelihood dispersion per gene, `s2.post`.
    pub s2_post: &'a [f64],
    /// Prior degrees of freedom, either one value shared by every gene or one
    /// per gene when the squeezing was robust.
    pub df_prior: &'a [f64],
    /// Residual degrees of freedom adjusted for the effective number of
    /// observations, `df.residual.adj`. One per gene.
    pub df_residual_adj: &'a [f64],
    /// Residual degrees of freedom after dropping structural zeros,
    /// `df.residual.zeros`. One per gene.
    ///
    /// `Some` marks the legacy quasi-likelihood pipeline, which is the only one
    /// where the Poisson bound applies; `None` takes its place *and* switches
    /// the bound off, as edgeR does.
    pub df_residual_zeros: Option<&'a [f64]>,
    /// Fitted means from the quasi-likelihood fit, row-major
    /// `n_genes * n_samples`. Only read by the Poisson bound.
    pub fitted: &'a [f64],
    /// Average quasi-likelihood dispersion the fit was divided by, if the
    /// non-legacy pipeline produced one. The null refit divides the dispersion
    /// by it again so that the two models see the same scale.
    pub average_ql_dispersion: Option<f64>,
}

/// Result of a genewise test.
#[derive(Clone, Debug)]
pub struct GlmTest {
    /// Log2 fold change per gene.
    ///
    /// The shrunk coefficient of the first tested column, or the contrast
    /// applied to the shrunk coefficients, divided by `ln 2`. When several
    /// coefficients are tested at once only the first one is reported, matching
    /// the first `logFC` column of edgeR's table.
    pub log_fc: Vec<f64>,
    /// Average log-CPM per gene, passed through from the input.
    pub log_cpm: Option<Vec<f64>>,
    /// Test statistic per gene.
    ///
    /// The likelihood ratio for [`glm_lrt`], the F statistic for
    /// [`glm_ql_ftest`], and `z_right` for [`glm_treat`], which is the larger of
    /// the two threshold z-scores. edgeR reports no statistic for `glmTreat`, so
    /// that last one is this crate's choice rather than a ported column.
    pub statistic: Vec<f64>,
    /// P-value per gene.
    pub p_value: Vec<f64>,
    /// Degrees of freedom under test, that is, how many columns the null drops.
    pub df_test: f64,
    /// Denominator degrees of freedom per gene. Only the quasi-likelihood tests
    /// set this.
    pub df_total: Option<Vec<f64>>,
}

///////////////////////////
// Likelihood ratio test //
///////////////////////////

/// Genewise likelihood ratio test.
///
/// Port of edgeR's `glmLRT`. Drops the tested columns from the design, refits
/// the null model at the fit's own dispersion and offsets, and reads the
/// deviance difference off a chi-squared with as many degrees of freedom as
/// columns were dropped. When a contrast is given the design is first rotated
/// through [`contrast_as_coef`] so that the contrast becomes the leading
/// coefficients; the likelihood ratio depends only on the column space the null
/// keeps, so the rotation's sign conventions do not reach the p-value.
///
/// ### Params
///
/// * `input` - Counts, design and the recycled matrices behind the fit
/// * `fit` - The full-model fit, from [`glm_fit`]
/// * `tested` - Coefficients or contrast under test
/// * `ql` - Quasi-likelihood summary when the fit came from `glmQLFit`, purely
///   so that `average_ql_dispersion` can be divided out before the null refit.
///   `None` for a plain `glmFit`.
///
/// ### Returns
///
/// The likelihood ratios and their p-values, or [`EdgeErrors`] if the shapes
/// disagree, the design has fewer than two columns, the contrast is rank
/// deficient, or every column is under test.
///
/// ### References
///
/// McCarthy, Chen and Smyth, Nucleic Acids Research, 2012
pub fn glm_lrt<T: EdgeFloat>(
    input: &GlmTestInput<'_, T>,
    fit: &GlmFit,
    tested: &Tested,
    ql: Option<&QlSummary<'_>>,
) -> Result<GlmTest, EdgeErrors> {
    validate(input, fit)?;

    let resolved = resolve_tested(input, fit, tested)?;
    let n_coef_null = input.n_coef - resolved.coef.len();
    if n_coef_null == 0 {
        return Err(EdgeErrors::InvalidArgument(
            "cannot test every coefficient: the null model would have no columns".to_string(),
        ));
    }

    let design_null = drop_columns(
        &resolved.design,
        input.n_samples,
        input.n_coef,
        &resolved.coef,
    );

    // edgeR stores the undivided dispersion on a quasi-likelihood fit but fits
    // with `dispersion / average.ql.dispersion`, so the null must be divided
    // again or the two models sit on different scales.
    let scaled = ql
        .and_then(|q| q.average_ql_dispersion)
        .map(|a| divide_dispersion(input.dispersion, a));
    let dispersion = scaled.as_ref().unwrap_or(input.dispersion);

    // Warm start only when the coefficients are still in the original basis. A
    // contrast has been rotated by `contrast_as_coef`, which does not hand back
    // the transformation, so there is nothing to project the full fit through.
    let start = match tested {
        Tested::Coef(_) => Some(drop_columns(
            fit.unshrunk_coefficients
                .as_deref()
                .unwrap_or(&fit.coefficients),
            input.n_genes,
            input.n_coef,
            &resolved.coef,
        )),
        Tested::Contrast { .. } => None,
    };

    let deviance_null = fit_null(
        input,
        &design_null,
        n_coef_null,
        dispersion,
        input.offset,
        start.as_deref(),
    )?;

    let df_test = resolved.coef.len() as f64;
    let statistic: Vec<f64> = deviance_null
        .iter()
        .zip(fit.deviance.iter())
        .map(|(null, full)| null - full)
        .collect();
    let p_value = statistic
        .par_iter()
        .map(|lr| chisq_sf(lr.max(0.0), df_test))
        .collect::<Result<Vec<f64>, EdgeErrors>>()?;

    Ok(GlmTest {
        log_fc: resolved.log_fc,
        log_cpm: input.log_cpm.map(<[f64]>::to_vec),
        statistic,
        p_value,
        df_test,
        df_total: None,
    })
}

/////////////////////////////
// Quasi-likelihood F test //
/////////////////////////////

/// Genewise quasi-likelihood F test.
///
/// Port of edgeR's `glmQLFTest`. Runs [`glm_lrt`] for the deviance difference,
/// then divides by the tested degrees of freedom and the posterior
/// quasi-likelihood dispersion to get an F statistic on `df_test` and
/// `df_prior + df_residual` degrees of freedom. The denominator is capped at the
/// total residual degrees of freedom in the experiment, which is what stops a
/// large `df_prior` claiming more information than the data hold.
///
/// The Poisson bound, when it applies, refits every gene at zero dispersion and
/// raises the p-value of any gene whose quasi-likelihood variance
/// `s2_post * (1 + dispersion * mu)` has fallen below the Poisson variance for
/// some library. Such a gene would otherwise look more significant than its own
/// counts can justify.
///
/// ### Params
///
/// * `input` - Counts, design and the recycled matrices behind the fit
/// * `fit` - The quasi-likelihood fit, from `glmQLFit`
/// * `ql` - The squeezed quantities that fit produced
/// * `tested` - Coefficients or contrast under test
/// * `poisson_bound` - Apply the Poisson bound. Silently ignored, as in edgeR,
///   when `ql.df_residual_zeros` is `None`, since the non-legacy pipeline has
///   no zero-adjusted degrees of freedom to bound against.
///
/// ### Returns
///
/// The F statistics, their p-values and the denominator degrees of freedom, or
/// [`EdgeErrors`] if a shape disagrees or the null refit fails.
///
/// ### References
///
/// Lund, Nettleton, McCarthy and Smyth, Statistical Applications in Genetics and
/// Molecular Biology, 2012
pub fn glm_ql_ftest<T: EdgeFloat>(
    input: &GlmTestInput<'_, T>,
    fit: &GlmFit,
    ql: &QlSummary<'_>,
    tested: &Tested,
    poisson_bound: bool,
) -> Result<GlmTest, EdgeErrors> {
    validate_ql(input, ql)?;
    let out = glm_lrt(input, fit, tested, Some(ql))?;

    // No zero-adjusted degrees of freedom means the non-legacy pipeline, which
    // edgeR does not bound.
    let (df_residual, poisson_bound) = match ql.df_residual_zeros {
        Some(zeros) => (zeros, poisson_bound),
        None => (ql.df_residual_adj, false),
    };
    let cap = (input.n_genes * fit.df_residual) as f64;

    let df_total: Vec<f64> = (0..input.n_genes)
        .map(|gene| (recycle(ql.df_prior, gene) + df_residual[gene]).min(cap))
        .collect();
    let statistic: Vec<f64> = out
        .statistic
        .iter()
        .zip(ql.s2_post.iter())
        .map(|(lr, s2)| lr / out.df_test / s2)
        .collect();
    let mut p_value = statistic
        .par_iter()
        .zip(df_total.par_iter())
        .map(|(f, df2)| f_sf(f.max(0.0), out.df_test, *df2))
        .collect::<Result<Vec<f64>, EdgeErrors>>()?;

    if poisson_bound {
        let below = below_poisson_bound(input, ql);
        if below.iter().any(|b| *b) {
            let poisson = poisson_refit_lrt(input, tested)?;
            for (gene, flagged) in below.iter().enumerate() {
                if *flagged {
                    p_value[gene] = p_value[gene].max(poisson.p_value[gene]);
                }
            }
        }
    }

    Ok(GlmTest {
        log_fc: out.log_fc,
        log_cpm: out.log_cpm,
        statistic,
        p_value,
        df_test: out.df_test,
        df_total: Some(df_total),
    })
}

///////////
// Treat //
///////////

/// Genewise test against a fold-change threshold.
///
/// Port of edgeR's `glmTreat`. Rather than asking whether the log-fold-change is
/// zero, it asks whether it is outside `[-lfc, lfc]`. Both bounds are tested by
/// shifting the offsets by `lfc * ln 2 * design[, coef]` and refitting, which
/// gives two deviance-difference z-scores; the smaller is signed negative when
/// the estimated fold change lies inside the interval, and the pair is fed to
/// either the interval null or the conservative worst-case null.
///
/// A zero `lfc` falls straight through to [`glm_lrt`] or [`glm_ql_ftest`], as
/// edgeR does.
///
/// Only one coefficient can be tested. Extra entries in `tested` are dropped,
/// which is edgeR's behaviour for a multi-column contrast, there with a warning.
///
/// ### Params
///
/// * `input` - Counts, design and the recycled matrices behind the fit
/// * `fit` - The fit, from [`glm_fit`] or `glmQLFit`
/// * `ql` - The squeezed quantities when the fit is a quasi-likelihood one,
///   which switches the test from the likelihood ratio flavour to the moderated
///   t flavour. `None` for a plain `glmFit`.
/// * `tested` - Coefficient or contrast under test
/// * `lfc` - Log2 fold-change threshold, non-negative
/// * `null` - Which null to test against
///
/// ### Returns
///
/// The threshold z-scores and their p-values, or [`EdgeErrors`] for a negative
/// `lfc`, a shape mismatch, or a failed refit.
///
/// ### References
///
/// McCarthy and Smyth, Bioinformatics, 2009
pub fn glm_treat<T: EdgeFloat>(
    input: &GlmTestInput<'_, T>,
    fit: &GlmFit,
    ql: Option<&QlSummary<'_>>,
    tested: &Tested,
    lfc: f64,
    null: TreatNull,
) -> Result<GlmTest, EdgeErrors> {
    if lfc < 0.0 || lfc.is_nan() {
        return Err(EdgeErrors::InvalidArgument(format!(
            "lfc must be non-negative, got {lfc}"
        )));
    }
    if lfc == 0.0 {
        return match ql {
            Some(q) => glm_ql_ftest(input, fit, q, tested, true),
            None => glm_lrt(input, fit, tested, None),
        };
    }
    validate(input, fit)?;
    if let Some(q) = ql {
        validate_ql(input, q)?;
    }

    let narrowed = narrow(tested, input.n_coef)?;
    let resolved = resolve_tested(input, fit, &narrowed)?;
    let coef = resolved.coef[0];
    let design_null = drop_columns(&resolved.design, input.n_samples, input.n_coef, &[coef]);

    let unshrunk_log_fc = match fit.unshrunk_coefficients.as_deref() {
        Some(unshrunk) => project(unshrunk, input.n_genes, input.n_coef, &narrowed, coef),
        None => resolved.log_fc.clone(),
    };

    let scaled = ql
        .and_then(|q| q.average_ql_dispersion)
        .map(|a| divide_dispersion(input.dispersion, a));
    let dispersion = scaled.as_ref().unwrap_or(input.dispersion);

    // The threshold enters as an offset shift along the tested column, so both
    // bounds are ordinary fits of the same two models on shifted data.
    let adjustment: Vec<f64> = (0..input.n_samples)
        .map(|sample| lfc * std::f64::consts::LN_2 * resolved.design[sample * input.n_coef + coef])
        .collect();

    let mut z_left = threshold_z(
        input,
        &resolved.design,
        &design_null,
        dispersion,
        &adjustment,
        1.0,
    )?;
    let mut z_right = threshold_z(
        input,
        &resolved.design,
        &design_null,
        dispersion,
        &adjustment,
        -1.0,
    )?;
    for (left, right) in z_left.iter_mut().zip(z_right.iter_mut()) {
        if *left > *right {
            std::mem::swap(left, right);
        }
    }

    // Under the quasi-likelihood pipeline the two deviance roots are moderated
    // t statistics, not z, so they are pushed through the t quantile first.
    let df_total = ql.map(|q| {
        let df_residual = q.df_residual_zeros.unwrap_or(q.df_residual_adj);
        let cap = (input.n_genes * (input.n_samples - input.n_coef)) as f64;
        (0..input.n_genes)
            .map(|gene| (recycle(q.df_prior, gene) + df_residual[gene]).min(cap))
            .collect::<Vec<f64>>()
    });
    if let (Some(q), Some(df)) = (ql, df_total.as_ref()) {
        for gene in 0..input.n_genes {
            let scale = q.s2_post[gene].sqrt();
            z_left[gene] = zscore_t(z_left[gene] / scale, df[gene])?;
            z_right[gene] = zscore_t(z_right[gene] / scale, df[gene])?;
        }
    }

    for (gene, left) in z_left.iter_mut().enumerate() {
        let within = unshrunk_log_fc[gene].abs() <= lfc;
        *left *= if within { 1.0 } else { -1.0 };
    }

    let mut p_value: Vec<f64> = z_left
        .par_iter()
        .zip(z_right.par_iter())
        .map(|(left, right)| treat_p_value(*left, *right, null))
        .collect();

    // edgeR's recursive Poisson bound. Note that it re-enters `glmTreat` without
    // passing `null` through, so the bound is always computed against the
    // interval null even when the caller asked for the worst case.
    if let Some(q) = ql
        && q.df_residual_zeros.is_some()
    {
        let below = below_poisson_bound(input, q);
        if below.iter().any(|b| *b) {
            let poisson = poisson_refit_treat(input, &narrowed, lfc)?;
            for (gene, flagged) in below.iter().enumerate() {
                if *flagged {
                    p_value[gene] = p_value[gene].max(poisson.p_value[gene]);
                }
            }
        }
    }

    Ok(GlmTest {
        log_fc: resolved.log_fc,
        log_cpm: input.log_cpm.map(<[f64]>::to_vec),
        statistic: z_right,
        p_value,
        df_test: 1.0,
        df_total,
    })
}

/////////////
// Helpers //
/////////////

/// The design a test works on, the columns it drops, and the fold change it
/// reports.
struct Resolved {
    /// Design to test against, row-major `n_samples * n_coef`. The input design
    /// for a coefficient test, the rotated one for a contrast.
    design: Vec<f64>,
    /// Columns the null model drops, ascending.
    coef: Vec<usize>,
    /// Log2 fold change per gene.
    log_fc: Vec<f64>,
}

/// Works out which columns a test drops and what fold change it reports.
///
/// For a contrast this is where [`contrast_as_coef`] runs, with `first = true`
/// so that the contrasts land in the leading columns. That is the placement
/// edgeR's `glmLRT` builds by hand and the one `glmTreat` asks limma for.
///
/// ### Params
///
/// * `input` - Test input
/// * `fit` - The full-model fit, for its coefficients
/// * `tested` - Coefficients or contrast under test
///
/// ### Returns
///
/// The resolved design, dropped columns and log-fold-changes, or
/// [`EdgeErrors`] if an index is out of range or the contrast is degenerate.
fn resolve_tested<T: EdgeFloat>(
    input: &GlmTestInput<'_, T>,
    fit: &GlmFit,
    tested: &Tested,
) -> Result<Resolved, EdgeErrors> {
    match tested {
        Tested::Coef(requested) => {
            let mut coef: Vec<usize> = Vec::with_capacity(requested.len());
            for index in requested {
                if *index >= input.n_coef {
                    return Err(EdgeErrors::CoefOutOfRange {
                        index: *index,
                        n_coef: input.n_coef,
                    });
                }
                if !coef.contains(index) {
                    coef.push(*index);
                }
            }
            if coef.is_empty() {
                return Err(EdgeErrors::InvalidArgument(
                    "no coefficients were selected for testing".to_string(),
                ));
            }
            let first = coef[0];
            let log_fc = (0..input.n_genes)
                .map(|gene| fit.coefficients[gene * input.n_coef + first] / std::f64::consts::LN_2)
                .collect();
            coef.sort_unstable();
            Ok(Resolved {
                design: input.design.to_vec(),
                coef,
                log_fc,
            })
        }
        Tested::Contrast {
            values,
            n_contrasts,
        } => {
            if values.len() != input.n_coef * n_contrasts {
                return Err(EdgeErrors::LengthMismatch {
                    name: "contrast",
                    expected: input.n_coef * n_contrasts,
                    got: values.len(),
                });
            }
            // The contrast is column-major `n_coef * n_contrasts`; read as
            // row-major it is its own transpose, whose rank is the same.
            let rank = matrix_rank(values, *n_contrasts, input.n_coef)?;
            if rank == 0 {
                return Err(EdgeErrors::InvalidArgument(
                    "contrasts are all zero".to_string(),
                ));
            }
            if rank < *n_contrasts {
                return Err(EdgeErrors::InvalidArgument(format!(
                    "contrast matrix has rank {rank} but {n_contrasts} columns; \
                     drop the redundant columns first"
                )));
            }
            let reform = contrast_as_coef(
                input.design,
                input.n_samples,
                input.n_coef,
                values,
                *n_contrasts,
                true,
            )?;
            let log_fc = (0..input.n_genes)
                .map(|gene| {
                    let beta = &fit.coefficients[gene * input.n_coef..(gene + 1) * input.n_coef];
                    let dot: f64 = beta
                        .iter()
                        .zip(values[..input.n_coef].iter())
                        .map(|(b, c)| b * c)
                        .sum();
                    dot / std::f64::consts::LN_2
                })
                .collect();
            Ok(Resolved {
                design: reform.design,
                coef: reform.coef,
                log_fc,
            })
        }
    }
}

/// Cuts a [`Tested`] down to the single column [`glm_treat`] can handle.
///
/// ### Params
///
/// * `tested` - Coefficients or contrast under test
/// * `n_coef` - Number of coefficients in the design
///
/// ### Returns
///
/// The first coefficient, or the first contrast column, or
/// [`EdgeErrors::InvalidArgument`] if nothing was selected.
fn narrow(tested: &Tested, n_coef: usize) -> Result<Tested, EdgeErrors> {
    match tested {
        Tested::Coef(v) => match v.first() {
            Some(first) => Ok(Tested::Coef(vec![*first])),
            None => Err(EdgeErrors::InvalidArgument(
                "no coefficients were selected for testing".to_string(),
            )),
        },
        Tested::Contrast { values, .. } => {
            if values.len() < n_coef {
                return Err(EdgeErrors::LengthMismatch {
                    name: "contrast",
                    expected: n_coef,
                    got: values.len(),
                });
            }
            Ok(Tested::Contrast {
                values: values[..n_coef].to_vec(),
                n_contrasts: 1,
            })
        }
    }
}

/// Log2 fold change of one column of a coefficient matrix, or of a contrast.
///
/// Used for the unshrunk fold change [`glm_treat`] compares against `lfc`; the
/// reported one comes from the shrunk coefficients instead.
///
/// ### Params
///
/// * `coefficients` - Row-major `n_genes * n_coef`
/// * `n_genes` - Number of genes
/// * `n_coef` - Number of coefficients
/// * `tested` - The narrowed selection, which decides column or contrast
/// * `column` - Column index to read for the coefficient case
///
/// ### Returns
///
/// One log2 fold change per gene.
fn project(
    coefficients: &[f64],
    n_genes: usize,
    n_coef: usize,
    tested: &Tested,
    column: usize,
) -> Vec<f64> {
    (0..n_genes)
        .map(|gene| {
            let beta = &coefficients[gene * n_coef..(gene + 1) * n_coef];
            let value = match tested {
                Tested::Coef(_) => beta[column],
                Tested::Contrast { values, .. } => {
                    beta.iter().zip(values.iter()).map(|(b, c)| b * c).sum()
                }
            };
            value / std::f64::consts::LN_2
        })
        .collect()
}

/// Refits the null model, warm-starting it where that is safe.
///
/// edgePython starts the null fit from the full fit's coefficients on the kept
/// columns, which cuts the Levenberg iteration count from roughly twenty to two
/// on wide designs, but only when the null design is *not* a one-way layout. A
/// one-way null goes through Fisher scoring in `mglm_one_way` instead, where a
/// warm start taken from a near-zero gene can push the iteration straight off a
/// cliff. This reproduces that condition: one-way nulls go through [`glm_fit`]
/// cold, everything else goes to the Levenberg fitter with the start.
///
/// ### Params
///
/// * `input` - Test input, for the counts and weights
/// * `design_null` - Null design, row-major `n_samples * n_coef_null`
/// * `n_coef_null` - Columns the null keeps
/// * `dispersion` - Dispersion to fit at
/// * `offset` - Offsets to fit at
/// * `start` - Warm start, row-major `n_genes * n_coef_null`, if one is available
///
/// ### Returns
///
/// One residual deviance per gene.
fn fit_null<T: EdgeFloat>(
    input: &GlmTestInput<'_, T>,
    design_null: &[f64],
    n_coef_null: usize,
    dispersion: &Recycled<f64>,
    offset: &Recycled<f64>,
    start: Option<&[f64]>,
) -> Result<Vec<f64>, EdgeErrors> {
    let (_, n_groups) = design_as_factor(design_null, input.n_samples, n_coef_null)?;
    let one_way = n_groups == n_coef_null;

    match start {
        Some(values) if !one_way => {
            if let Some(bad) = non_estimable(design_null, input.n_samples, n_coef_null)? {
                return Err(EdgeErrors::DesignNotFullRank {
                    n_cols: n_coef_null,
                    rank: n_coef_null - bad.len(),
                });
            }
            let params = LevenbergParams {
                max_iter: NULL_FIT_MAX_ITER,
                ..Default::default()
            };
            let fit = mglm_levenberg(
                input.counts,
                input.n_genes,
                input.n_samples,
                design_null,
                n_coef_null,
                dispersion,
                offset,
                input.weights,
                Some(values),
                Some(params),
            )?;
            Ok(fit.deviance)
        }
        _ => {
            let fit = glm_fit(
                input.counts,
                input.n_genes,
                input.n_samples,
                design_null,
                n_coef_null,
                dispersion,
                offset,
                input.weights,
                0.0,
            )?;
            Ok(fit.deviance)
        }
    }
}

/// One bound of the threshold test.
///
/// Fits the full and null models against offsets shifted by `sign * adjustment`
/// and returns the square root of the deviance difference, which is the z-score
/// for the shifted null.
///
/// ### Params
///
/// * `input` - Test input
/// * `design` - Full design, row-major `n_samples * n_coef`
/// * `design_null` - Null design, row-major `n_samples * (n_coef - 1)`
/// * `dispersion` - Dispersion to fit at
/// * `adjustment` - Per-sample offset shift, `lfc * ln 2 * design[, coef]`
/// * `sign` - `+1` for the left bound, `-1` for the right
///
/// ### Returns
///
/// One non-negative z-score per gene.
fn threshold_z<T: EdgeFloat>(
    input: &GlmTestInput<'_, T>,
    design: &[f64],
    design_null: &[f64],
    dispersion: &Recycled<f64>,
    adjustment: &[f64],
    sign: f64,
) -> Result<Vec<f64>, EdgeErrors> {
    let offset = shift_offset(
        input.offset,
        adjustment,
        input.n_genes,
        input.n_samples,
        sign,
    );
    let fit_null = glm_fit(
        input.counts,
        input.n_genes,
        input.n_samples,
        design_null,
        input.n_coef - 1,
        dispersion,
        &offset,
        input.weights,
        0.0,
    )?;
    let fit_full = glm_fit(
        input.counts,
        input.n_genes,
        input.n_samples,
        design,
        input.n_coef,
        dispersion,
        &offset,
        input.weights,
        0.0,
    )?;
    Ok(fit_null
        .deviance
        .iter()
        .zip(fit_full.deviance.iter())
        .map(|(null, full)| (null - full).max(0.0).sqrt())
        .collect())
}

/// The treat p-value for one gene.
///
/// ### Params
///
/// * `z_left` - Signed z-score of the nearer bound
/// * `z_right` - z-score of the further bound
/// * `null` - Which null to integrate against
///
/// ### Returns
///
/// The p-value, in `[0, 1]` up to rounding in the integral.
fn treat_p_value(z_left: f64, z_right: f64, null: TreatNull) -> f64 {
    match null {
        TreatNull::WorstCase => norm_cdf(-z_right) + norm_cdf(z_left),
        TreatNull::Interval => {
            if z_right + z_left > TREAT_INTERVAL_WIDTH {
                integrate_pnorm(-z_right, -z_right + TREAT_INTERVAL_WIDTH)
                    + integrate_pnorm(z_left - TREAT_INTERVAL_WIDTH, z_left)
            } else {
                2.0 * integrate_pnorm(-z_right, z_left)
            }
        }
    }
}

/// Mean of the standard normal CDF over `[a, b]`.
///
/// Port of edgeR's `.integratepnorm`. The closed form is
/// `(b F(b) + f(b) - a F(a) - f(a)) / (b - a)`, with the degenerate `a == b`
/// collapsing to `F(a)`. The equality test is exact, as edgeR's is.
///
/// ### Params
///
/// * `a` - Lower limit
/// * `b` - Upper limit
///
/// ### Returns
///
/// The average of `F` over the interval.
fn integrate_pnorm(a: f64, b: f64) -> f64 {
    if a == b {
        return norm_cdf(a);
    }
    (b * norm_cdf(b) + norm_pdf(b) - (a * norm_cdf(a) + norm_pdf(a))) / (b - a)
}

/// Standard normal density.
///
/// ### Params
///
/// * `x` - Quantile
///
/// ### Returns
///
/// `exp(-x^2 / 2) / sqrt(2 pi)`.
#[inline]
fn norm_pdf(x: f64) -> f64 {
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

/// Converts a t statistic to the z with the same tail probability.
///
/// Port of limma's `zscoreT` at its exact setting: the upper tail of the t is
/// evaluated directly and inverted through the normal quantile, never as
/// `1 - cdf`. limma works in logs to keep going past `f64`'s smallest normal;
/// here the tail probability is floored instead and the answer clamped to
/// [`MAX_TREAT_ZSCORE`], which is where the normal tail has already reached
/// 1e-316 and the treat integral would otherwise divide by an infinity.
///
/// ### Params
///
/// * `x` - t statistic
/// * `df` - Degrees of freedom, finite and strictly positive
///
/// ### Returns
///
/// The matching z-score, or [`EdgeErrors::InvalidArgument`] for a non-positive
/// `df`.
fn zscore_t(x: f64, df: f64) -> Result<f64, EdgeErrors> {
    let tail = t_sf(x.abs(), df)?;
    let z = -norm_ppf(tail.max(f64::MIN_POSITIVE))?;
    let z = z.min(MAX_TREAT_ZSCORE);
    Ok(if x < 0.0 { -z } else { z })
}

/// Genes whose quasi-likelihood variance has dropped below the Poisson variance.
///
/// Port of edgeR's `check_poisson_bound`: a gene is flagged as soon as one
/// library has `s2_post * (1 + dispersion * mu) < 1`. The dispersion read here
/// is the one stored on the fit, undivided by `average_ql_dispersion`, matching
/// edgeR.
///
/// ### Params
///
/// * `input` - Test input, for the dispersion
/// * `ql` - Quasi-likelihood summary, for `s2_post` and the fitted means
///
/// ### Returns
///
/// One flag per gene.
fn below_poisson_bound<T: EdgeFloat>(input: &GlmTestInput<'_, T>, ql: &QlSummary<'_>) -> Vec<bool> {
    (0..input.n_genes)
        .into_par_iter()
        .map(|gene| {
            let disp = input.dispersion.row(gene, input.n_samples);
            let s2 = ql.s2_post[gene];
            let mu = &ql.fitted[gene * input.n_samples..(gene + 1) * input.n_samples];
            mu.iter()
                .enumerate()
                .any(|(sample, m)| s2 * (1.0 + disp.get(sample) * m) < 1.0)
        })
        .collect()
}

/// Refits at zero dispersion and runs the likelihood ratio test on that fit.
///
/// The Poisson bound compares a gene's quasi-likelihood p-value against what its
/// Poisson likelihood alone would support. edgeR subsets to the flagged genes
/// first; this fits all of them, because the flag is usually set for a large
/// fraction of the low-count genes anyway and the subsetting would have to
/// rebuild every recycled matrix.
///
/// ### Params
///
/// * `input` - Test input
/// * `tested` - Coefficients or contrast under test
///
/// ### Returns
///
/// The Poisson likelihood ratio test, whose p-values are the bound.
fn poisson_refit_lrt<T: EdgeFloat>(
    input: &GlmTestInput<'_, T>,
    tested: &Tested,
) -> Result<GlmTest, EdgeErrors> {
    let dispersion = Recycled::scalar(0.0);
    let poisson = poisson_input(input, &dispersion);
    let fit = poisson_fit(&poisson)?;
    glm_lrt(&poisson, &fit, tested, None)
}

/// The same bound for [`glm_treat`].
///
/// edgeR re-enters `glmTreat` here without forwarding `null`, so the bound is
/// always taken against the interval null. That is reproduced rather than
/// corrected.
///
/// ### Params
///
/// * `input` - Test input
/// * `tested` - The narrowed selection under test
/// * `lfc` - Log2 fold-change threshold
///
/// ### Returns
///
/// The Poisson threshold test, whose p-values are the bound.
fn poisson_refit_treat<T: EdgeFloat>(
    input: &GlmTestInput<'_, T>,
    tested: &Tested,
    lfc: f64,
) -> Result<GlmTest, EdgeErrors> {
    let dispersion = Recycled::scalar(0.0);
    let poisson = poisson_input(input, &dispersion);
    let fit = poisson_fit(&poisson)?;
    glm_treat(&poisson, &fit, None, tested, lfc, TreatNull::Interval)
}

/// Rewrites a test input to sit at zero dispersion.
///
/// ### Params
///
/// * `input` - Test input
/// * `dispersion` - The zero dispersion to borrow
///
/// ### Returns
///
/// A copy of the input with the dispersion replaced and the log-CPM dropped,
/// since the bound only ever reads p-values off it.
fn poisson_input<'a, T: EdgeFloat>(
    input: &GlmTestInput<'a, T>,
    dispersion: &'a Recycled<f64>,
) -> GlmTestInput<'a, T> {
    GlmTestInput {
        counts: input.counts,
        n_genes: input.n_genes,
        n_samples: input.n_samples,
        design: input.design,
        n_coef: input.n_coef,
        dispersion,
        offset: input.offset,
        weights: input.weights,
        log_cpm: None,
    }
}

/// Fits the full model at zero dispersion.
///
/// ### Params
///
/// * `input` - Test input already carrying a zero dispersion
///
/// ### Returns
///
/// The Poisson fit. Shrinkage is off, since only the deviance is read.
fn poisson_fit<T: EdgeFloat>(input: &GlmTestInput<'_, T>) -> Result<GlmFit, EdgeErrors> {
    glm_fit(
        input.counts,
        input.n_genes,
        input.n_samples,
        input.design,
        input.n_coef,
        input.dispersion,
        input.offset,
        input.weights,
        0.0,
    )
}

/// Drops columns from a row-major design.
///
/// ### Params
///
/// * `design` - Row-major `n_rows * n_cols`
/// * `n_rows` - Number of samples
/// * `n_cols` - Number of coefficients
/// * `drop` - Column indices to remove
///
/// ### Returns
///
/// The design without those columns, still row-major.
fn drop_columns(design: &[f64], n_rows: usize, n_cols: usize, drop: &[usize]) -> Vec<f64> {
    let kept: Vec<usize> = (0..n_cols).filter(|c| !drop.contains(c)).collect();
    let mut out = Vec::with_capacity(n_rows * kept.len());
    for row in design.chunks_exact(n_cols) {
        out.extend(kept.iter().map(|c| row[*c]));
    }
    out
}

/// Divides every entry of a recycled dispersion by a scalar.
///
/// ### Params
///
/// * `dispersion` - Dispersion to scale
/// * `average` - Divisor, edgeR's `average.ql.dispersion`
///
/// ### Returns
///
/// A new recycled matrix in the same storage form.
fn divide_dispersion(dispersion: &Recycled<f64>, average: f64) -> Recycled<f64> {
    let scale = |v: &Vec<f64>| v.iter().map(|d| d / average).collect();
    match dispersion {
        Recycled::Scalar(d) => Recycled::Scalar(d / average),
        Recycled::ByGene(v) => Recycled::ByGene(scale(v)),
        Recycled::BySample(v) => Recycled::BySample(scale(v)),
        Recycled::Full(v) => Recycled::Full(scale(v)),
    }
}

/// Adds a per-sample shift to a recycled offset matrix.
///
/// Stays compressed where it can: a scalar or per-sample offset shifted by a
/// per-sample vector is still per-sample. A per-gene or full offset has to
/// expand, since the result varies along both axes.
///
/// ### Params
///
/// * `offset` - Offsets to shift
/// * `adjustment` - Per-sample shift, length `n_samples`
/// * `n_genes` - Number of genes
/// * `n_samples` - Number of samples
/// * `sign` - `+1` to add the shift, `-1` to subtract it
///
/// ### Returns
///
/// The shifted offsets.
fn shift_offset(
    offset: &Recycled<f64>,
    adjustment: &[f64],
    n_genes: usize,
    n_samples: usize,
    sign: f64,
) -> Recycled<f64> {
    match offset {
        Recycled::Scalar(v) => {
            Recycled::BySample(adjustment.iter().map(|a| v + sign * a).collect())
        }
        Recycled::BySample(values) => Recycled::BySample(
            values
                .iter()
                .zip(adjustment.iter())
                .map(|(v, a)| v + sign * a)
                .collect(),
        ),
        Recycled::ByGene(values) => {
            let mut out = Vec::with_capacity(n_genes * n_samples);
            for value in values {
                out.extend(adjustment.iter().map(|a| value + sign * a));
            }
            Recycled::Full(out)
        }
        Recycled::Full(values) => Recycled::Full(
            values
                .chunks_exact(n_samples)
                .flat_map(|row| {
                    row.iter()
                        .zip(adjustment.iter())
                        .map(|(v, a)| v + sign * a)
                        .collect::<Vec<f64>>()
                })
                .collect(),
        ),
    }
}

/// Reads a per-gene quantity that may have been supplied as a single value.
///
/// ### Params
///
/// * `values` - Either one value or one per gene
/// * `gene` - Gene index
///
/// ### Returns
///
/// The value for that gene.
#[inline]
fn recycle(values: &[f64], gene: usize) -> f64 {
    if values.len() == 1 {
        values[0]
    } else {
        values[gene]
    }
}

/// Checks the shapes a test relies on.
///
/// ### Params
///
/// * `input` - Test input
/// * `fit` - The full-model fit
///
/// ### Returns
///
/// `Ok(())`, or the first [`EdgeErrors`] the shapes trip.
fn validate<T: EdgeFloat>(input: &GlmTestInput<'_, T>, fit: &GlmFit) -> Result<(), EdgeErrors> {
    if input.n_genes == 0 || input.n_samples == 0 {
        return Err(EdgeErrors::EmptyCounts {
            n_genes: input.n_genes,
            n_samples: input.n_samples,
        });
    }
    if input.n_coef < 2 {
        return Err(EdgeErrors::InvalidArgument(
            "need at least two design columns; the first is usually the intercept".to_string(),
        ));
    }
    if input.counts.len() != input.n_genes * input.n_samples {
        return Err(EdgeErrors::LengthMismatch {
            name: "counts",
            expected: input.n_genes * input.n_samples,
            got: input.counts.len(),
        });
    }
    if input.design.len() != input.n_samples * input.n_coef {
        return Err(EdgeErrors::LengthMismatch {
            name: "design",
            expected: input.n_samples * input.n_coef,
            got: input.design.len(),
        });
    }
    if fit.coefficients.len() != input.n_genes * input.n_coef {
        return Err(EdgeErrors::LengthMismatch {
            name: "coefficients",
            expected: input.n_genes * input.n_coef,
            got: fit.coefficients.len(),
        });
    }
    if fit.deviance.len() != input.n_genes {
        return Err(EdgeErrors::LengthMismatch {
            name: "deviance",
            expected: input.n_genes,
            got: fit.deviance.len(),
        });
    }
    if let Some(cpm) = input.log_cpm
        && cpm.len() != input.n_genes
    {
        return Err(EdgeErrors::LengthMismatch {
            name: "log_cpm",
            expected: input.n_genes,
            got: cpm.len(),
        });
    }
    input.dispersion.validate(input.n_genes, input.n_samples)?;
    input.offset.validate(input.n_genes, input.n_samples)?;
    if let Some(w) = input.weights {
        w.validate(input.n_genes, input.n_samples)?;
    }
    Ok(())
}

/// Checks the shapes of a quasi-likelihood summary.
///
/// ### Params
///
/// * `input` - Test input
/// * `ql` - The summary to check
///
/// ### Returns
///
/// `Ok(())`, or the first [`EdgeErrors`] the shapes trip.
fn validate_ql<T: EdgeFloat>(
    input: &GlmTestInput<'_, T>,
    ql: &QlSummary<'_>,
) -> Result<(), EdgeErrors> {
    let shapes: [(&'static str, usize, usize); 3] = [
        ("s2_post", ql.s2_post.len(), input.n_genes),
        ("df_residual_adj", ql.df_residual_adj.len(), input.n_genes),
        ("fitted", ql.fitted.len(), input.n_genes * input.n_samples),
    ];
    for (name, got, expected) in shapes {
        if got != expected {
            return Err(EdgeErrors::LengthMismatch {
                name,
                expected,
                got,
            });
        }
    }
    if ql.df_prior.len() != 1 && ql.df_prior.len() != input.n_genes {
        return Err(EdgeErrors::LengthMismatch {
            name: "df_prior",
            expected: input.n_genes,
            got: ql.df_prior.len(),
        });
    }
    if let Some(zeros) = ql.df_residual_zeros
        && zeros.len() != input.n_genes
    {
        return Err(EdgeErrors::LengthMismatch {
            name: "df_residual_zeros",
            expected: input.n_genes,
            got: zeros.len(),
        });
    }
    Ok(())
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Three genes, two groups of three. The same fixture `glm::fit` uses.
    ///
    /// ```r
    /// y <- matrix(c(10,12,11,40,44,38, 50,48,52,49,51,50, 2,0,5,1,3,0),
    ///             nrow = 3, byrow = TRUE)
    /// X <- cbind(1, c(0,0,0,1,1,1))
    /// ```
    fn small() -> (Vec<f64>, Vec<f64>, Recycled<f64>) {
        let counts = vec![
            10.0, 12.0, 11.0, 40.0, 44.0, 38.0, //
            50.0, 48.0, 52.0, 49.0, 51.0, 50.0, //
            2.0, 0.0, 5.0, 1.0, 3.0, 0.0, //
        ];
        let design = vec![
            1.0, 0.0, //
            1.0, 0.0, //
            1.0, 0.0, //
            1.0, 1.0, //
            1.0, 1.0, //
            1.0, 1.0, //
        ];
        // glmFit defaults to lib.size = colSums(y).
        let libraries: [f64; 6] = [62.0, 60.0, 68.0, 90.0, 98.0, 88.0];
        let offset = Recycled::by_sample(libraries.iter().map(|v| v.ln()).collect());
        (counts, design, offset)
    }

    /// Eight genes, two groups of three, used for the quasi-likelihood and
    /// threshold tests.
    ///
    /// ```r
    /// y <- matrix(c(10,12,11,40,44,38, 50,48,52,49,51,50, 2,0,5,1,3,0,
    ///               200,190,210,205,195,215, 1,2,0,8,9,7, 0,1,0,0,0,1,
    ///               30,33,28,31,29,30, 5,6,4,20,22,19), nrow = 8, byrow = TRUE)
    /// X <- cbind(1, c(0,0,0,1,1,1))
    /// ```
    fn wide() -> (Vec<f64>, Vec<f64>, Recycled<f64>) {
        let counts = vec![
            10.0, 12.0, 11.0, 40.0, 44.0, 38.0, //
            50.0, 48.0, 52.0, 49.0, 51.0, 50.0, //
            2.0, 0.0, 5.0, 1.0, 3.0, 0.0, //
            200.0, 190.0, 210.0, 205.0, 195.0, 215.0, //
            1.0, 2.0, 0.0, 8.0, 9.0, 7.0, //
            0.0, 1.0, 0.0, 0.0, 0.0, 1.0, //
            30.0, 33.0, 28.0, 31.0, 29.0, 30.0, //
            5.0, 6.0, 4.0, 20.0, 22.0, 19.0, //
        ];
        let design = vec![
            1.0, 0.0, //
            1.0, 0.0, //
            1.0, 0.0, //
            1.0, 1.0, //
            1.0, 1.0, //
            1.0, 1.0, //
        ];
        let libraries: [f64; 6] = [298.0, 292.0, 310.0, 354.0, 353.0, 360.0];
        let offset = Recycled::by_sample(libraries.iter().map(|v| v.ln()).collect());
        (counts, design, offset)
    }

    /// Four genes over three groups of two, for the contrast paths.
    ///
    /// ```r
    /// y2 <- matrix(c(10,12,40,44,100,110, 50,48,49,51,52,50,
    ///                0,0,1,3,20,25, 5,7,6,8,7,6), nrow = 4, byrow = TRUE)
    /// X2 <- model.matrix(~ 0 + factor(c("A","A","B","B","C","C")))
    /// ```
    fn three_group() -> (Vec<f64>, Vec<f64>, Recycled<f64>) {
        let counts = vec![
            10.0, 12.0, 40.0, 44.0, 100.0, 110.0, //
            50.0, 48.0, 49.0, 51.0, 52.0, 50.0, //
            0.0, 0.0, 1.0, 3.0, 20.0, 25.0, //
            5.0, 7.0, 6.0, 8.0, 7.0, 6.0, //
        ];
        let design = vec![
            1.0, 0.0, 0.0, //
            1.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 1.0, 0.0, //
            0.0, 0.0, 1.0, //
            0.0, 0.0, 1.0, //
        ];
        let libraries: [f64; 6] = [65.0, 67.0, 96.0, 106.0, 179.0, 191.0];
        let offset = Recycled::by_sample(libraries.iter().map(|v| v.ln()).collect());
        (counts, design, offset)
    }

    /// Builds a test input over borrowed pieces.
    fn input<'a>(
        counts: &'a [f64],
        n_genes: usize,
        design: &'a [f64],
        n_coef: usize,
        dispersion: &'a Recycled<f64>,
        offset: &'a Recycled<f64>,
    ) -> GlmTestInput<'a, f64> {
        GlmTestInput {
            counts,
            n_genes,
            n_samples: 6,
            design,
            n_coef,
            dispersion,
            offset,
            weights: None,
            log_cpm: None,
        }
    }

    fn fit_of(input: &GlmTestInput<'_, f64>) -> GlmFit {
        glm_fit(
            input.counts,
            input.n_genes,
            input.n_samples,
            input.design,
            input.n_coef,
            input.dispersion,
            input.offset,
            input.weights,
            0.125,
        )
        .unwrap()
    }

    ///////////////
    // `glm_lrt` //
    ///////////////

    /// ```r
    /// f <- glmFit(y, X, dispersion = 0.1)
    /// glmLRT(f, coef = 2)$table
    /// ```
    #[test]
    fn test_glm_lrt_matches_edger_on_a_single_coefficient() {
        let (counts, design, offset) = small();
        let dispersion = Recycled::scalar(0.1);
        let inp = input(&counts, 3, &design, 2, &dispersion, &offset);
        let fit = fit_of(&inp);
        let out = glm_lrt(&inp, &fit, &Tested::Coef(vec![1]), None).unwrap();

        let log_fc = [
            1.335_986_011_350_581_5,
            -0.536_817_869_447_691_6,
            -1.250_904_046_511_814_2,
        ];
        let lr = [
            8.341_979_103_855_252,
            1.732_785_607_270_639,
            1.956_785_090_352_040_5,
        ];
        let p = [
            0.003_873_937_497_185_523_3,
            0.188_055_540_378_233_27,
            0.161_857_564_986_522_76,
        ];
        assert_eq!(out.df_test, 1.0);
        assert!(out.df_total.is_none());
        for gene in 0..3 {
            assert_relative_eq!(out.log_fc[gene], log_fc[gene], max_relative = 1e-10);
            assert_relative_eq!(out.statistic[gene], lr[gene], max_relative = 1e-10);
            assert_relative_eq!(out.p_value[gene], p[gene], max_relative = 1e-10);
        }
    }

    /// The intercept is testable too, and its null design is the group
    /// indicator alone, which is not a one-way layout. That sends the null fit
    /// down the warm-started Levenberg branch.
    ///
    /// ```r
    /// glmLRT(f, coef = 1)$table
    /// ```
    #[test]
    fn test_glm_lrt_on_the_intercept_uses_the_warm_started_branch() {
        let (counts, design, offset) = small();
        let dispersion = Recycled::scalar(0.1);
        let inp = input(&counts, 3, &design, 2, &dispersion, &offset);
        let fit = fit_of(&inp);
        let out = glm_lrt(&inp, &fit, &Tested::Coef(vec![0]), None).unwrap();

        let lr = [
            41.921_421_573_734_83,
            1.313_819_348_845_321_6,
            86.065_935_682_930_47,
        ];
        let p = [
            9.501_597_746_548_692e-11,
            2.517_043_040_693_706e-1,
            1.740_350_328_216_402_5e-20,
        ];
        for gene in 0..3 {
            assert_relative_eq!(out.statistic[gene], lr[gene], max_relative = 1e-10);
            assert_relative_eq!(out.p_value[gene], p[gene], max_relative = 1e-10);
        }
    }

    /// `glmLRT(f, contrast = c(0, 1))` is the same test as `coef = 2`, but it
    /// arrives through `contrast_as_coef` and a rotated design.
    #[test]
    fn test_glm_lrt_contrast_agrees_with_the_coefficient_path() {
        let (counts, design, offset) = small();
        let dispersion = Recycled::scalar(0.1);
        let inp = input(&counts, 3, &design, 2, &dispersion, &offset);
        let fit = fit_of(&inp);
        let by_coef = glm_lrt(&inp, &fit, &Tested::Coef(vec![1]), None).unwrap();
        let by_contrast = glm_lrt(
            &inp,
            &fit,
            &Tested::Contrast {
                values: vec![0.0, 1.0],
                n_contrasts: 1,
            },
            None,
        )
        .unwrap();

        for gene in 0..3 {
            assert_relative_eq!(
                by_contrast.log_fc[gene],
                by_coef.log_fc[gene],
                max_relative = 1e-12
            );
            assert_relative_eq!(
                by_contrast.statistic[gene],
                by_coef.statistic[gene],
                max_relative = 1e-10
            );
            assert_relative_eq!(
                by_contrast.p_value[gene],
                by_coef.p_value[gene],
                max_relative = 1e-10
            );
        }
    }

    /// A genuine group-versus-group contrast on a three-level factor.
    ///
    /// ```r
    /// f2 <- glmFit(y2, X2, dispersion = 0.15)
    /// glmLRT(f2, contrast = c(-1, 1, 0))$table
    /// ```
    #[test]
    fn test_glm_lrt_matches_edger_on_a_group_contrast() {
        let (counts, design, offset) = three_group();
        let dispersion = Recycled::scalar(0.15);
        let inp = input(&counts, 4, &design, 3, &dispersion, &offset);
        let fit = fit_of(&inp);
        let out = glm_lrt(
            &inp,
            &fit,
            &Tested::Contrast {
                values: vec![-1.0, 1.0, 0.0],
                n_contrasts: 1,
            },
            None,
        )
        .unwrap();

        let log_fc = [
            1.314_710_781_299_474,
            -0.582_462_832_327_925_1,
            4.284_529_874_638_284_5,
            -0.388_058_150_000_178_07,
        ];
        let lr = [
            4.032_180_454_464_596,
            0.956_446_195_492_408_8,
            3.679_427_818_235_808_5,
            0.243_122_673_596_927_35,
        ];
        let p = [
            0.044_640_210_361_232_81,
            0.328_083_883_367_401_46,
            0.055_087_751_666_934_24,
            0.621_959_796_729_172_5,
        ];
        assert_eq!(out.df_test, 1.0);
        for gene in 0..4 {
            assert_relative_eq!(out.log_fc[gene], log_fc[gene], max_relative = 1e-10);
            assert_relative_eq!(out.statistic[gene], lr[gene], max_relative = 1e-10);
            assert_relative_eq!(out.p_value[gene], p[gene], max_relative = 1e-10);
        }
    }

    /// A two-column contrast, which is the two-degree-of-freedom test.
    ///
    /// ```r
    /// glmLRT(f2, contrast = cbind(c(-1, 1, 0), c(-1, 0, 1)))$table
    /// ```
    #[test]
    fn test_glm_lrt_matches_edger_on_a_two_column_contrast() {
        let (counts, design, offset) = three_group();
        let dispersion = Recycled::scalar(0.15);
        let inp = input(&counts, 4, &design, 3, &dispersion, &offset);
        let fit = fit_of(&inp);
        let out = glm_lrt(
            &inp,
            &fit,
            &Tested::Contrast {
                values: vec![-1.0, 1.0, 0.0, -1.0, 0.0, 1.0],
                n_contrasts: 2,
            },
            None,
        )
        .unwrap();

        let lr = [
            7.489_094_796_878_978_5,
            5.604_306_004_386_178,
            22.625_269_055_571_128,
            3.060_853_924_032_122,
        ];
        let p = [
            2.364_632_899_149_205_6e-2,
            6.067_927_926_597_326e-2,
            1.221_758_161_337_463_6e-5,
            2.164_432_345_449_859_9e-1,
        ];
        assert_eq!(out.df_test, 2.0);
        for gene in 0..4 {
            // Gene 3 has an empty group, so its null maximum sits at minus
            // infinity and both fitters stop wherever their own tolerance runs
            // out. That is the one place the agreement drops to 1e-9, and the
            // chi-squared tail turns that into 1e-8 on the p-value.
            assert_relative_eq!(out.statistic[gene], lr[gene], max_relative = 1e-8);
            assert_relative_eq!(out.p_value[gene], p[gene], max_relative = 1e-7);
        }
        // The reported fold change is the first contrast, as in edgeR's table.
        assert_relative_eq!(out.log_fc[0], 1.314_710_781_299_474, max_relative = 1e-10);
    }

    /// Two coefficients at once, dropping both from the null.
    ///
    /// ```r
    /// glmLRT(f2, coef = c(2, 3))$table
    /// ```
    #[test]
    fn test_glm_lrt_matches_edger_on_two_coefficients() {
        let (counts, design, offset) = three_group();
        let dispersion = Recycled::scalar(0.15);
        let inp = input(&counts, 4, &design, 3, &dispersion, &offset);
        let fit = fit_of(&inp);
        let out = glm_lrt(&inp, &fit, &Tested::Coef(vec![1, 2]), None).unwrap();

        let lr = [
            10.585_850_031_983_432,
            19.010_202_865_950_3,
            86.152_535_750_910_06,
            92.074_489_192_513_13,
        ];
        let p = [
            5.027_034_577_232_391e-3,
            7.447_095_063_546_895e-5,
            1.959_812_668_604_125_9e-19,
            1.014_562_275_310_370_5e-20,
        ];
        assert_eq!(out.df_test, 2.0);
        for gene in 0..4 {
            // Dropping both group columns leaves a null design that is a single
            // indicator, and gene 3 has no counts in that group at all. Its
            // null coefficient diverges, so the two fitters part company in the
            // ninth digit.
            assert_relative_eq!(out.statistic[gene], lr[gene], max_relative = 1e-8);
            assert_relative_eq!(out.p_value[gene], p[gene], max_relative = 1e-8);
        }
    }

    /// The far tail and an all-zero gene in one fixture. Gene 3's p-value is
    /// 9.46e-204, which `1 - pchisq` would return as a flat zero.
    ///
    /// ```r
    /// y3 <- matrix(c(20,22,18,5000,5200,4800, 0,0,0,0,0,0,
    ///                100,110,90,105,95,100), nrow = 3, byrow = TRUE)
    /// glmLRT(glmFit(y3, X, dispersion = 0.01), coef = 2)$table
    /// ```
    #[test]
    fn test_glm_lrt_in_the_far_tail_and_on_a_zero_gene() {
        let counts = vec![
            20.0, 22.0, 18.0, 5000.0, 5200.0, 4800.0, //
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, //
            100.0, 110.0, 90.0, 105.0, 95.0, 100.0, //
        ];
        let design = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let libraries: [f64; 6] = [120.0, 132.0, 108.0, 5105.0, 5295.0, 4900.0];
        let offset = Recycled::by_sample(libraries.iter().map(|v| v.ln()).collect());
        let dispersion = Recycled::scalar(0.01);
        let inp = input(&counts, 3, &design, 2, &dispersion, &offset);
        let fit = fit_of(&inp);
        let out = glm_lrt(&inp, &fit, &Tested::Coef(vec![1]), None).unwrap();

        let lr = [172.284_890_441_011_5, 0.0, 927.674_565_354_018_5];
        let p = [2.344_998_823_345_881e-39, 1.0, 9.458_079_718_661_872e-204];
        for gene in 0..3 {
            assert_relative_eq!(out.statistic[gene], lr[gene], epsilon = 1e-9);
            assert_relative_eq!(out.p_value[gene], p[gene], max_relative = 1e-10);
        }
        // The zero gene carries no information, so its fold change is zero to
        // rounding and its p-value is exactly one.
        assert!(out.log_fc[1].abs() < 1e-12);
        assert_eq!(out.p_value[1], 1.0);
    }

    ////////////////////
    // `glm_ql_ftest` //
    ////////////////////

    /// Owned storage for the legacy quasi-likelihood fixture, so that a
    /// [`QlSummary`] can borrow from it.
    struct LegacyQl {
        /// Posterior quasi-likelihood dispersions.
        s2_post: Vec<f64>,
        /// Prior degrees of freedom, one value shared by every gene.
        df_prior: Vec<f64>,
        /// Zero-adjusted residual degrees of freedom.
        df_residual_zeros: Vec<f64>,
        /// Adjusted residual degrees of freedom.
        df_residual_adj: Vec<f64>,
        /// Fitted means, row-major.
        fitted: Vec<f64>,
    }

    impl LegacyQl {
        /// Borrows the fixture as a summary the tests can pass in.
        fn summary(&self) -> QlSummary<'_> {
            QlSummary {
                s2_post: &self.s2_post,
                df_prior: &self.df_prior,
                df_residual_adj: &self.df_residual_adj,
                df_residual_zeros: Some(&self.df_residual_zeros),
                fitted: &self.fitted,
                average_ql_dispersion: None,
            }
        }
    }

    /// The legacy quasi-likelihood fit of the eight-gene fixture.
    ///
    /// ```r
    /// q <- glmQLFit(y, X, dispersion = 0.1, legacy = TRUE)
    /// ```
    fn ql_legacy() -> LegacyQl {
        let s2_post = vec![
            0.049_281_563_700_074_56,
            0.011_659_677_647_637_47,
            1.728_524_387_495_482_8,
            0.007_316_196_288_206_033,
            0.561_520_580_113_145_8,
            2.054_690_133_109_537_5,
            0.042_104_651_793_427_66,
            0.108_585_097_222_579_13,
        ];
        let df_prior = vec![4.460_965_388_747_048];
        let df_residual_zeros = vec![4.0; 8];
        let df_residual_adj = vec![4.0; 8];
        let fitted = vec![
            10.933_744_628_513_18,
            10.713_602_119_214_256,
            11.374_029_647_111_028,
            40.492_974_243_000_01,
            40.378_587_310_110_184,
            41.179_295_840_339_01,
            //
            49.659_282_993_114_66,
            48.659_431_657_682_816,
            51.658_985_663_978_34,
            49.769_601_759_585_95,
            49.629_009_664_219_89,
            50.613_154_331_782_35,
            //
            2.308_489_259_791_181,
            2.262_009_610_265_183,
            2.401_448_558_843_174_7,
            1.328_158_182_691_277_4,
            1.324_406_323_418_138_2,
            1.350_669_338_330_113_2,
            //
            198.595_261_331_926_4,
            194.596_699_023_229_86,
            206.592_385_949_319_44,
            203.991_992_958_926_5,
            203.415_744_391_245_88,
            207.449_484_365_010_05,
            //
            0.995_197_722_682_345_2,
            0.975_160_184_641_760_9,
            1.035_272_798_763_513_5,
            7.965_675_239_409_282,
            7.943_173_331_953_323,
            8.100_686_684_145_035,
            //
            0.331_402_866_955_448_6,
            0.324_730_326_010_036_9,
            0.344_747_948_846_272_1,
            0.331_641_809_994_310_75,
            0.330_704_968_723_140_37,
            0.337_262_857_621_333_07,
            //
            30.181_552_543_631_07,
            29.573_870_277_651_917,
            31.396_917_075_589_37,
            29.860_397_478_017_26,
            29.776_046_072_712_123,
            30.366_505_909_848_07,
            //
            4.974_310_683_547_974,
            4.874_156_777_167_812,
            5.174_618_496_308_295,
            20.245_100_646_285_735,
            20.187_911_096_437_47,
            20.588_237_945_375_333,
        ];
        LegacyQl {
            s2_post,
            df_prior,
            df_residual_zeros,
            df_residual_adj,
            fitted,
        }
    }

    /// ```r
    /// glmQLFTest(q, coef = 2)$table       # poisson.bound = TRUE
    /// glmQLFTest(q, coef = 2, poisson.bound = FALSE)$table
    /// ```
    ///
    /// Genes 2, 4 and 7 sit below the Poisson bound, so their p-values are
    /// raised. The others are untouched, which is what makes the pair of
    /// expectations a real test of the bound rather than of the F tail.
    #[test]
    fn test_glm_ql_ftest_matches_edger_with_and_without_the_poisson_bound() {
        let (counts, design, offset) = wide();
        let dispersion = Recycled::scalar(0.1);
        let inp = input(&counts, 8, &design, 2, &dispersion, &offset);
        let fit = fit_of(&inp);
        let legacy = ql_legacy();
        let ql = legacy.summary();

        let f_stat = [
            2.520_057_973_402_189_2e2,
            3.095_392_962_353_460_2e1,
            6.853_203_261_465_175e-1,
            4.126_759_810_823_103_4e1,
            1.979_404_448_269_534e1,
            6.917_310_190_199_355e-3,
            8.944_754_825_397_515,
            9.876_119_655_893_109e1,
        ];
        let bounded = [
            1.364_107_878_599_685_5e-7,
            1.407_007_683_799_561e-1,
            4.304_904_955_542_642e-1,
            1.127_377_026_183_572_2e-2,
            1.865_737_655_809_165_6e-3,
            9.356_503_273_786_928e-1,
            2.229_724_201_299_175_8e-1,
            5.944_660_235_862_634e-6,
        ];
        let unbounded = [
            1.364_107_878_599_685_5e-7,
            4.366_386_781_336_41e-4,
            4.304_904_955_542_642e-1,
            1.598_459_203_229_021_3e-4,
            1.865_737_655_809_165_6e-3,
            9.356_503_273_786_928e-1,
            1.625_485_152_929_125_4e-2,
            5.944_660_235_862_634e-6,
        ];

        let with_bound = glm_ql_ftest(&inp, &fit, &ql, &Tested::Coef(vec![1]), true).unwrap();
        let without = glm_ql_ftest(&inp, &fit, &ql, &Tested::Coef(vec![1]), false).unwrap();

        assert_eq!(with_bound.df_test, 1.0);
        let df_total = with_bound.df_total.as_ref().unwrap();
        for gene in 0..8 {
            assert_relative_eq!(df_total[gene], 8.460_965_388_747_049, max_relative = 1e-12);
            assert_relative_eq!(
                with_bound.statistic[gene],
                f_stat[gene],
                max_relative = 1e-10
            );
            assert_relative_eq!(
                with_bound.p_value[gene],
                bounded[gene],
                max_relative = 1e-10
            );
            assert_relative_eq!(without.p_value[gene], unbounded[gene], max_relative = 1e-10);
        }
    }

    /// The non-legacy pipeline has no zero-adjusted degrees of freedom, so
    /// edgeR turns the bound off however the caller asked for it, and the
    /// denominator degrees of freedom hit the cap at the experiment's total.
    ///
    /// ```r
    /// q2 <- glmQLFit(y, X, dispersion = 0.1)
    /// glmQLFTest(q2, coef = 2)$table
    /// ```
    #[test]
    fn test_glm_ql_ftest_without_zero_adjusted_df_ignores_the_bound() {
        let (counts, design, offset) = wide();
        let dispersion = Recycled::scalar(0.1);
        let inp = input(&counts, 8, &design, 2, &dispersion, &offset);
        let fit = fit_of(&inp);
        let s2_post = [
            0.048_188_444_222_375_504,
            0.024_446_203_487_533_725,
            1.180_185_563_640_951_3,
            0.005_271_328_303_740_723,
            0.546_740_168_004_454_8,
            2.445_938_293_050_726_3,
            0.039_977_551_427_882_49,
            0.154_447_261_383_920_18,
        ];
        let df_prior = [8_134.844_780_382_662];
        let adj = [
            3.973_521_938_241_366,
            3.999_098_159_890_601,
            5.529_668_600_344_663,
            4.000_963_615_960_567_5,
            5.978_981_705_096_965,
            9.394_040_228_202_032,
            3.995_519_515_274_4,
            3.929_173_912_871_219,
        ];
        let ql = QlSummary {
            s2_post: &s2_post,
            df_prior: &df_prior,
            df_residual_adj: &adj,
            df_residual_zeros: None,
            fitted: &[0.0; 48],
            average_ql_dispersion: Some(1.0),
        };

        let out = glm_ql_ftest(&inp, &fit, &ql, &Tested::Coef(vec![1]), true).unwrap();
        let f_stat = [
            2.577_223_638_326_844_4e2,
            1.476_355_383_862_022_7e1,
            1.003_734_440_994_232_6,
            5.727_623_678_615_572e1,
            2.032_915_083_096_279_2e1,
            5.810_828_930_493_346e-3,
            9.420_681_703_839_223,
            6.943_466_678_578_658e1,
        ];
        let p = [
            7.252_743_073_151_62e-17,
            5.442_532_006_559_633e-4,
            3.239_246_833_697_202_5e-1,
            1.276_727_950_912_218_5e-8,
            8.231_984_735_990_331e-5,
            9.397_116_797_565_639e-1,
            4.348_612_379_217_248e-3,
            1.606_882_310_788_258_1e-9,
        ];
        let df_total = out.df_total.as_ref().unwrap();
        for gene in 0..8 {
            // pmin(df.prior + df.residual, sum(df.residual)) = min(8138.8, 32).
            assert_relative_eq!(df_total[gene], 32.0, max_relative = 1e-12);
            assert_relative_eq!(out.statistic[gene], f_stat[gene], max_relative = 1e-10);
            assert_relative_eq!(out.p_value[gene], p[gene], max_relative = 1e-10);
        }
    }

    /// The F tail, well past where `1 - pf` collapses. The squeezed quantities
    /// are chosen rather than fitted, since the point is the tail.
    ///
    /// ```r
    /// LR <- c(12.419239754410078547, 0.360912841338069723, 1.184592896990613653,
    ///         0.301921848102618240, 11.114763340708499584, 0.014212928995460672,
    ///         0.376615787300944194, 10.723994130169780092)
    /// pf(LR / 1e-4, df1 = 1, df2 = 32, lower.tail = FALSE)
    /// ```
    #[test]
    fn test_glm_ql_ftest_far_tail() {
        let (counts, design, offset) = wide();
        let dispersion = Recycled::scalar(0.1);
        let inp = input(&counts, 8, &design, 2, &dispersion, &offset);
        let fit = fit_of(&inp);
        let s2_post = [1e-4; 8];
        let df_prior = [1000.0];
        let adj = [4.0; 8];
        let ql = QlSummary {
            s2_post: &s2_post,
            df_prior: &df_prior,
            df_residual_adj: &adj,
            df_residual_zeros: None,
            fitted: &[0.0; 48],
            average_ql_dispersion: None,
        };
        let out = glm_ql_ftest(&inp, &fit, &ql, &Tested::Coef(vec![1]), false).unwrap();

        let p = [
            5.261_561_085_923_352_5e-59,
            1.779_918_291_545_363_5e-34,
            1.079_125_786_482_700_5e-42,
            3.012_683_792_206_176e-33,
            3.104_755_317_213_854_3e-58,
            2.604_261_088_389_738e-13,
            9.055_927_316_446_368e-35,
            5.503_704_196_958_632e-58,
        ];
        for (got, want) in out.p_value.iter().zip(p.iter()) {
            assert_relative_eq!(got, want, max_relative = 1e-10);
        }
    }

    /////////////////
    // `glm_treat` //
    /////////////////

    /// ```r
    /// f <- glmFit(y, X, dispersion = 0.1)
    /// glmTreat(f, coef = 2, lfc = 1)$table$PValue
    /// glmTreat(f, coef = 2, lfc = 1, null = "worst.case")$table$PValue
    /// ```
    #[test]
    fn test_glm_treat_lrt_at_lfc_one() {
        let (counts, design, offset) = wide();
        let dispersion = Recycled::scalar(0.1);
        let inp = input(&counts, 8, &design, 2, &dispersion, &offset);
        let fit = fit_of(&inp);

        let interval = [
            0.025_915_419_120_752_61,
            0.862_946_855_826_681_2,
            0.335_130_846_838_047_5,
            0.901_730_557_007_466_8,
            0.005_453_587_125_419_520_4,
            0.910_944_563_566_415_2,
            0.836_150_163_403_213_7,
            0.023_915_536_140_601_778,
        ];
        let worst = [
            0.084_605_550_556_336_65,
            0.968_617_766_474_667_2,
            0.497_500_093_784_695_2,
            0.981_164_069_793_248_7,
            0.021_394_389_526_003_44,
            0.917_471_298_505_725_4,
            0.957_971_652_323_224_3,
            0.078_981_717_742_023_32,
        ];
        let log_fc = [
            1.630_328_521_653_603_4,
            -0.244_625_377_904_330_73,
            -0.975_328_979_782_516_8,
            -0.209_626_606_575_486_35,
            2.620_328_451_924_648,
            -0.180_553_480_809_313_55,
            -0.262_787_305_134_895_23,
            1.753_538_627_526_939,
        ];

        let a = glm_treat(
            &inp,
            &fit,
            None,
            &Tested::Coef(vec![1]),
            1.0,
            TreatNull::Interval,
        )
        .unwrap();
        let b = glm_treat(
            &inp,
            &fit,
            None,
            &Tested::Coef(vec![1]),
            1.0,
            TreatNull::WorstCase,
        )
        .unwrap();

        assert_eq!(a.df_test, 1.0);
        assert!(a.df_total.is_none());
        for gene in 0..8 {
            assert_relative_eq!(a.log_fc[gene], log_fc[gene], max_relative = 1e-10);
            assert_relative_eq!(a.p_value[gene], interval[gene], max_relative = 1e-10);
            assert_relative_eq!(b.p_value[gene], worst[gene], max_relative = 1e-10);
        }
    }

    /// Two more thresholds, including edgeR's own default of `log2(1.2)`.
    ///
    /// ```r
    /// glmTreat(f, coef = 2, lfc = 0.5)$table$PValue
    /// glmTreat(f, coef = 2, lfc = log2(1.2))$table$PValue
    /// glmTreat(f, coef = 2, lfc = log2(1.2), null = "worst.case")$table$PValue
    /// ```
    #[test]
    fn test_glm_treat_lrt_at_other_thresholds() {
        let (counts, design, offset) = wide();
        let dispersion = Recycled::scalar(0.1);
        let inp = input(&counts, 8, &design, 2, &dispersion, &offset);
        let fit = fit_of(&inp);

        let half = [
            0.001_747_492_463_475_674_3,
            0.618_965_088_475_578_5,
            0.297_496_484_707_449_94,
            0.661_667_162_539_030_5,
            0.001_731_384_373_595_051_3,
            0.906_630_646_614_269_5,
            0.602_748_783_056_843_9,
            0.002_830_446_598_910_725,
        ];
        let default_interval = [
            0.000_772_184_486_707_990_3,
            0.574_647_381_380_633_3,
            0.282_377_207_940_779_64,
            0.611_261_200_442_731_6,
            0.001_073_803_287_152_239_8,
            0.905_530_268_033_189_1,
            0.563_833_195_684_254,
            0.001_600_134_051_293_08,
        ];
        let default_worst = [
            0.001_557_323_252_030_769_7,
            0.624_202_827_105_601_4,
            0.294_451_670_100_143_67,
            0.663_637_030_156_907_3,
            0.001_546_443_189_297_852,
            0.906_031_012_374_345_1,
            0.609_637_644_920_852_7,
            0.002_786_515_725_771_390_3,
        ];

        let a = glm_treat(
            &inp,
            &fit,
            None,
            &Tested::Coef(vec![1]),
            0.5,
            TreatNull::Interval,
        )
        .unwrap();
        let lfc = 1.2_f64.log2();
        let b = glm_treat(
            &inp,
            &fit,
            None,
            &Tested::Coef(vec![1]),
            lfc,
            TreatNull::Interval,
        )
        .unwrap();
        let c = glm_treat(
            &inp,
            &fit,
            None,
            &Tested::Coef(vec![1]),
            lfc,
            TreatNull::WorstCase,
        )
        .unwrap();
        for gene in 0..8 {
            assert_relative_eq!(a.p_value[gene], half[gene], max_relative = 1e-10);
            assert_relative_eq!(
                b.p_value[gene],
                default_interval[gene],
                max_relative = 1e-10
            );
            assert_relative_eq!(c.p_value[gene], default_worst[gene], max_relative = 1e-10);
        }
    }

    /// The contrast path, including a gene whose unshrunk fold change is
    /// effectively infinite because one group is all zeros.
    ///
    /// ```r
    /// glmTreat(f2, contrast = c(-1, 1, 0), lfc = 1)$table$PValue
    /// glmTreat(f2, contrast = c(-1, 1, 0), lfc = 1, null = "worst.case")$table$PValue
    /// ```
    #[test]
    fn test_glm_treat_matches_edger_on_a_contrast() {
        let (counts, design, offset) = three_group();
        let dispersion = Recycled::scalar(0.15);
        let inp = input(&counts, 4, &design, 3, &dispersion, &offset);
        let fit = fit_of(&inp);
        let contrast = Tested::Contrast {
            values: vec![-1.0, 1.0, 0.0],
            n_contrasts: 1,
        };

        let interval = [
            0.136_036_070_900_815_65,
            0.528_392_553_392_713_7,
            0.060_483_368_850_335_27,
            0.689_127_004_865_843_3,
        ];
        let worst = [
            0.313_021_256_154_164_54,
            0.762_143_061_238_379_5,
            0.079_816_920_784_910_93,
            0.818_329_089_039_554_8,
        ];
        let log_fc = [
            1.314_710_781_299_474,
            -0.582_462_832_327_925_1,
            4.284_529_874_638_284_5,
            -0.388_058_150_000_178_07,
        ];

        let a = glm_treat(&inp, &fit, None, &contrast, 1.0, TreatNull::Interval).unwrap();
        let b = glm_treat(&inp, &fit, None, &contrast, 1.0, TreatNull::WorstCase).unwrap();
        for gene in 0..4 {
            assert_relative_eq!(a.log_fc[gene], log_fc[gene], max_relative = 1e-10);
            // Gene 3's unshrunk fold change is infinite, so the shifted fits it
            // is compared against are only as reproducible as their stopping
            // rule; 1e-9 is where the two agree.
            assert_relative_eq!(a.p_value[gene], interval[gene], max_relative = 1e-8);
            assert_relative_eq!(b.p_value[gene], worst[gene], max_relative = 1e-8);
        }
    }

    /// The quasi-likelihood flavour, where the deviance roots become moderated
    /// t statistics and the Poisson bound applies. Genes 1 and 8 come out with
    /// the same p-value under both nulls, because the bound has caught them and
    /// edgeR's recursive call always uses the interval null.
    ///
    /// ```r
    /// q <- glmQLFit(y, X, dispersion = 0.1, legacy = TRUE)
    /// glmTreat(q, coef = 2, lfc = 1)$table$PValue
    /// glmTreat(q, coef = 2, lfc = 1, null = "worst.case")$table$PValue
    /// ```
    #[test]
    fn test_glm_treat_quasi_likelihood_with_the_poisson_bound() {
        let (counts, design, offset) = wide();
        let dispersion = Recycled::scalar(0.1);
        let inp = input(&counts, 8, &design, 2, &dispersion, &offset);
        let fit = fit_of(&inp);
        let legacy = ql_legacy();
        let ql = legacy.summary();

        let interval = [
            0.002_095_420_937_924_167_8,
            0.999_993_018_183_757_4,
            0.487_393_893_826_179_1,
            1.000_000_000_000_001,
            0.005_735_420_883_968_396_5,
            0.938_980_124_834_262_6,
            0.999_431_608_175_005_4,
            0.006_193_294_762_041_081_5,
        ];
        let worst = [
            0.002_095_420_937_924_167_8,
            0.999_999_963_126_753_1,
            0.556_398_596_920_132_5,
            1.000_000_000_000_001,
            0.012_897_636_268_926_604,
            0.941_055_590_649_069_9,
            0.999_988_456_782_335_7,
            0.006_193_294_762_041_081_5,
        ];

        let a = glm_treat(
            &inp,
            &fit,
            Some(&ql),
            &Tested::Coef(vec![1]),
            1.0,
            TreatNull::Interval,
        )
        .unwrap();
        let b = glm_treat(
            &inp,
            &fit,
            Some(&ql),
            &Tested::Coef(vec![1]),
            1.0,
            TreatNull::WorstCase,
        )
        .unwrap();
        for gene in 0..8 {
            assert_relative_eq!(a.p_value[gene], interval[gene], max_relative = 1e-10);
            assert_relative_eq!(b.p_value[gene], worst[gene], max_relative = 1e-10);
        }
        assert_eq!(a.p_value[0], b.p_value[0]);
        assert_eq!(a.p_value[7], b.p_value[7]);
        assert!(a.df_total.is_some());
    }

    /// The quasi-likelihood flavour at a second threshold.
    ///
    /// ```r
    /// glmTreat(q, coef = 2, lfc = 0.5)$table$PValue
    /// glmTreat(q, coef = 2, lfc = 0.5, null = "worst.case")$table$PValue
    /// ```
    #[test]
    fn test_glm_treat_quasi_likelihood_at_half_a_fold() {
        let (counts, design, offset) = wide();
        let dispersion = Recycled::scalar(0.1);
        let inp = input(&counts, 8, &design, 2, &dispersion, &offset);
        let fit = fit_of(&inp);
        let legacy = ql_legacy();
        let ql = legacy.summary();

        let interval = [
            1.156_930_720_415_820_5e-6,
            9.958_190_513_004_438e-1,
            4.457_622_253_174_958e-1,
            9.995_944_678_440_168e-1,
            2.585_377_345_926_488_7e-3,
            9.365_161_866_221_516e-1,
            9.136_827_161_461_856e-1,
            7.667_539_435_702_798e-5,
        ];
        let worst = [
            1.299_077_183_789_172_9e-6,
            9.998_319_316_629_676e-1,
            4.654_945_529_449_657e-1,
            9.999_928_996_412_859e-1,
            3.653_055_884_276_585e-3,
            9.370_713_782_972_866e-1,
            9.866_427_824_263_567e-1,
            7.667_539_435_702_798e-5,
        ];

        let a = glm_treat(
            &inp,
            &fit,
            Some(&ql),
            &Tested::Coef(vec![1]),
            0.5,
            TreatNull::Interval,
        )
        .unwrap();
        let b = glm_treat(
            &inp,
            &fit,
            Some(&ql),
            &Tested::Coef(vec![1]),
            0.5,
            TreatNull::WorstCase,
        )
        .unwrap();
        for gene in 0..8 {
            assert_relative_eq!(a.p_value[gene], interval[gene], max_relative = 1e-10);
            assert_relative_eq!(b.p_value[gene], worst[gene], max_relative = 1e-10);
        }
    }

    /// A zero threshold is not a threshold, so edgeR falls straight through to
    /// the ordinary test. Both flavours do.
    #[test]
    fn test_glm_treat_with_zero_lfc_falls_back() {
        let (counts, design, offset) = wide();
        let dispersion = Recycled::scalar(0.1);
        let inp = input(&counts, 8, &design, 2, &dispersion, &offset);
        let fit = fit_of(&inp);
        let tested = Tested::Coef(vec![1]);

        let lrt = glm_lrt(&inp, &fit, &tested, None).unwrap();
        let treat = glm_treat(&inp, &fit, None, &tested, 0.0, TreatNull::Interval).unwrap();
        for gene in 0..8 {
            assert_eq!(treat.p_value[gene], lrt.p_value[gene]);
        }

        let legacy = ql_legacy();
        let ql = legacy.summary();
        let ftest = glm_ql_ftest(&inp, &fit, &ql, &tested, true).unwrap();
        let treat_ql = glm_treat(&inp, &fit, Some(&ql), &tested, 0.0, TreatNull::Interval).unwrap();
        for gene in 0..8 {
            assert_eq!(treat_ql.p_value[gene], ftest.p_value[gene]);
        }
    }

    /// A larger threshold must make every gene less significant, since the null
    /// it is tested against has grown.
    #[test]
    fn test_glm_treat_is_monotone_in_the_threshold() {
        let (counts, design, offset) = wide();
        let dispersion = Recycled::scalar(0.1);
        let inp = input(&counts, 8, &design, 2, &dispersion, &offset);
        let fit = fit_of(&inp);
        let tested = Tested::Coef(vec![1]);

        let small = glm_treat(&inp, &fit, None, &tested, 0.25, TreatNull::Interval).unwrap();
        let large = glm_treat(&inp, &fit, None, &tested, 1.5, TreatNull::Interval).unwrap();
        for gene in 0..8 {
            assert!(
                large.p_value[gene] >= small.p_value[gene] - 1e-12,
                "gene {gene}: {} < {}",
                large.p_value[gene],
                small.p_value[gene]
            );
        }
    }

    ////////////////
    // Numerics   //
    ////////////////

    /// ```r
    /// limma::zscoreT(c(0.5, 2, 6, 20), df = 8.4609653887470486)
    /// ```
    #[test]
    fn test_zscore_t_matches_limma() {
        let df = 8.460_965_388_747_049;
        let expected = [
            0.481_981_798_779_442_4,
            1.759_032_448_322_397_1,
            3.653_010_684_919_440_4,
            5.609_979_465_791_493,
        ];
        for (x, want) in [0.5, 2.0, 6.0, 20.0].iter().zip(expected.iter()) {
            assert_relative_eq!(zscore_t(*x, df).unwrap(), want, max_relative = 1e-10);
        }
        // Odd symmetry, and zero maps to zero.
        assert_relative_eq!(
            zscore_t(-2.0, df).unwrap(),
            -zscore_t(2.0, df).unwrap(),
            max_relative = 1e-12
        );
        assert_eq!(zscore_t(0.0, df).unwrap(), 0.0);
    }

    /// ```r
    /// edgeR:::.integratepnorm(c(-1, -3, 0.5, 2), c(0.470402, -1.529598, 0.5, 3.470402))
    /// ```
    #[test]
    fn test_integrate_pnorm_matches_edger() {
        let a = [-1.0, -3.0, 0.5, 2.0];
        let b = [0.470_402, -1.529_598, 0.5, 3.470_402];
        let expected = [
            0.404_086_872_171_341_76,
            0.018_364_485_997_322_027,
            0.691_462_461_274_013,
            0.994_270_314_804_512_8,
        ];
        for gene in 0..4 {
            assert_relative_eq!(
                integrate_pnorm(a[gene], b[gene]),
                expected[gene],
                max_relative = 1e-10
            );
        }
    }

    //////////////////
    // Argument use //
    //////////////////

    #[test]
    fn test_log_cpm_is_passed_through() {
        let (counts, design, offset) = small();
        let dispersion = Recycled::scalar(0.1);
        let cpm = [
            18.305_644_059_893_947,
            19.322_293_068_742_58,
            15.526_879_626_455_655,
        ];
        let mut inp = input(&counts, 3, &design, 2, &dispersion, &offset);
        inp.log_cpm = Some(&cpm);
        let fit = fit_of(&inp);
        let out = glm_lrt(&inp, &fit, &Tested::Coef(vec![1]), None).unwrap();
        assert_eq!(out.log_cpm.unwrap(), cpm.to_vec());
    }

    #[test]
    fn test_rejects_an_out_of_range_coefficient() {
        let (counts, design, offset) = small();
        let dispersion = Recycled::scalar(0.1);
        let inp = input(&counts, 3, &design, 2, &dispersion, &offset);
        let fit = fit_of(&inp);
        let err = glm_lrt(&inp, &fit, &Tested::Coef(vec![7]), None).unwrap_err();
        assert!(matches!(err, EdgeErrors::CoefOutOfRange { .. }));
    }

    #[test]
    fn test_rejects_testing_every_coefficient() {
        let (counts, design, offset) = small();
        let dispersion = Recycled::scalar(0.1);
        let inp = input(&counts, 3, &design, 2, &dispersion, &offset);
        let fit = fit_of(&inp);
        let err = glm_lrt(&inp, &fit, &Tested::Coef(vec![0, 1]), None).unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));
    }

    #[test]
    fn test_rejects_a_zero_contrast() {
        let (counts, design, offset) = small();
        let dispersion = Recycled::scalar(0.1);
        let inp = input(&counts, 3, &design, 2, &dispersion, &offset);
        let fit = fit_of(&inp);
        let err = glm_lrt(
            &inp,
            &fit,
            &Tested::Contrast {
                values: vec![0.0, 0.0],
                n_contrasts: 1,
            },
            None,
        )
        .unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));
    }

    #[test]
    fn test_rejects_a_rank_deficient_contrast() {
        let (counts, design, offset) = three_group();
        let dispersion = Recycled::scalar(0.15);
        let inp = input(&counts, 4, &design, 3, &dispersion, &offset);
        let fit = fit_of(&inp);
        let err = glm_lrt(
            &inp,
            &fit,
            &Tested::Contrast {
                values: vec![-1.0, 1.0, 0.0, -2.0, 2.0, 0.0],
                n_contrasts: 2,
            },
            None,
        )
        .unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));
    }

    #[test]
    fn test_rejects_a_negative_lfc() {
        let (counts, design, offset) = small();
        let dispersion = Recycled::scalar(0.1);
        let inp = input(&counts, 3, &design, 2, &dispersion, &offset);
        let fit = fit_of(&inp);
        let err = glm_treat(
            &inp,
            &fit,
            None,
            &Tested::Coef(vec![1]),
            -0.5,
            TreatNull::Interval,
        )
        .unwrap_err();
        assert!(matches!(err, EdgeErrors::InvalidArgument(_)));
    }

    #[test]
    fn test_rejects_a_mismatched_quasi_likelihood_summary() {
        let (counts, design, offset) = small();
        let dispersion = Recycled::scalar(0.1);
        let inp = input(&counts, 3, &design, 2, &dispersion, &offset);
        let fit = fit_of(&inp);
        let ql = QlSummary {
            s2_post: &[1.0, 1.0],
            df_prior: &[4.0],
            df_residual_adj: &[4.0; 3],
            df_residual_zeros: None,
            fitted: &[0.0; 18],
            average_ql_dispersion: None,
        };
        let err = glm_ql_ftest(&inp, &fit, &ql, &Tested::Coef(vec![1]), false).unwrap_err();
        assert!(matches!(err, EdgeErrors::LengthMismatch { .. }));
    }

    /// A per-gene offset has to expand when the threshold shift is applied,
    /// but the answer must not depend on how the offsets were stored.
    #[test]
    fn test_treat_is_insensitive_to_the_offset_storage() {
        let (counts, design, _) = wide();
        let libraries: [f64; 6] = [298.0, 292.0, 310.0, 354.0, 353.0, 360.0];
        let by_sample = Recycled::by_sample(libraries.iter().map(|v| v.ln()).collect());
        let full = Recycled::Full(
            (0..8)
                .flat_map(|_| libraries.iter().map(|v| v.ln()))
                .collect(),
        );
        let dispersion = Recycled::scalar(0.1);

        let compact = input(&counts, 8, &design, 2, &dispersion, &by_sample);
        let dense = input(&counts, 8, &design, 2, &dispersion, &full);
        let fit = fit_of(&compact);
        let a = glm_treat(
            &compact,
            &fit,
            None,
            &Tested::Coef(vec![1]),
            1.0,
            TreatNull::Interval,
        )
        .unwrap();
        let b = glm_treat(
            &dense,
            &fit,
            None,
            &Tested::Coef(vec![1]),
            1.0,
            TreatNull::Interval,
        )
        .unwrap();
        for gene in 0..8 {
            assert_relative_eq!(a.p_value[gene], b.p_value[gene], max_relative = 1e-12);
        }
    }
}
