//! The classic two-group exact test, edgeR's `exactTest`.
//!
//! This is the pre-GLM path and still the one people reach for on a simple
//! two-group design. It has three parts:
//!
//! * [`q2q_nbinom`] maps a count from one negative binomial mean to another
//!   through matched quantiles, averaging a normal and a gamma approximation.
//! * [`equalize_lib_sizes`] runs that mapping over a whole matrix to produce
//!   pseudo-counts that all sit on one common library size.
//! * [`exact_test`] fits the fold change, builds the pseudo-counts and then
//!   sums the conditional negative binomial mass over one of three rejection
//!   regions.
//!
//! Genes are the parallel axis throughout, per the crate rule: one gene is a
//! contiguous row and every kernel here is a rayon fan-out over rows with a
//! per-thread scratch buffer for the convolution.
//!
//! ### Where the tails come from
//!
//! `q2qnbinom` is the fiddly part. edgeR evaluates both approximations with
//! `log.p = TRUE` so a quantile fifty standard deviations out still round-trips.
//! Two things are done differently here and both are documented at their call
//! sites: the normal round trip is replaced by the linear map it is algebraically
//! equal to, and the upper gamma tail goes through a log-space incomplete gamma
//! defined in this module, because [`crate::numeric::dist`] exposes only
//! `gamma_cdf` and `gamma_ppf` on the natural scale and `1 - cdf` is a flat zero
//! well before edgeR gives up.
//!
//! ### Where this departs from edgePython
//!
//! `edgepython/exact_test.py` stubs `exact_test_by_deviance` and
//! `exact_test_by_small_p` out to the double-tail test. They are genuinely
//! different tests whenever the two groups differ in size, and both are
//! implemented here against edgeR.

use std::f64::consts::LN_2;

use rayon::prelude::*;

use crate::core::dgelist::DgeList;
use crate::core::expression::{ave_log_cpm, column_sums};
use crate::glm::deviance::unit_nb_deviance;
use crate::glm::one_group::mglm_one_group;
use crate::numeric::dist::nbinom_ln_pmf;
use crate::numeric::dist::{beta_cdf, beta_ppf, beta_sf, chisq_sf, gamma_cdf, gamma_ppf};
use crate::numeric::gamma::ln_gamma;
use crate::prelude::*;

////////////
// Consts //
////////////

/// Row sum above which, in both groups, the beta approximation replaces the
/// convolution.
///
/// edgeR's `big.count` default. The convolution is `O(s)` per gene, so a gene
/// with a million reads in each group would otherwise dominate the whole
/// analysis; past this the beta approximation is accurate to well under the
/// resolution anyone reads a p-value at.
pub const BIG_COUNT_DEFAULT: f64 = 900.0;

/// Prior count added before the fold change is fitted.
///
/// edgeR's `exactTest` default. It is small on purpose: this prior only damps
/// the fold change of a gene that is zero in one group, and a larger value would
/// visibly shrink real effects.
pub const PRIOR_COUNT_DEFAULT: f64 = 0.125;

/// Prior count [`exact_test`] passes to `aveLogCPM` for the reported log-CPM.
///
/// edgeR's `aveLogCPM` default, and a different quantity from
/// [`PRIOR_COUNT_DEFAULT`]: this one only sets the floor of the abundance
/// covariate, where 2 counts per million is the usual choice.
const AVE_LOG_CPM_PRIOR: f64 = 2.0;

/// Below this, a mean is treated as zero by [`q2q_nbinom`] and nudged.
///
/// edgeR's `eps`. Both means are nudged together when either trips it, which is
/// what keeps the mapping the identity for a gene that is fitted at zero.
const Q2Q_EPS: f64 = 1e-14;

/// Nudge added to a mean that fell below [`Q2Q_EPS`].
///
/// edgeR's 0.25. It is a quarter of a count, small enough not to move a real
/// pseudo-count and large enough to keep the gamma shape away from zero.
const Q2Q_ZERO_NUDGE: f64 = 0.25;

/// Relative convergence tolerance for the upper gamma quantile, in `ln(x)`.
///
/// Roughly 4 ulp, matching the tolerance [`crate::numeric::dist::gamma_ppf`]
/// uses on the lower tail. Tightening further only cycles on the last bit of the
/// continued fraction.
const GAMMA_ISF_REL_TOL: f64 = 1e-15;

/// Iteration budget for the upper gamma quantile and its bracket search.
///
/// The bracket walks outwards one nat at a time and then halves, so a bracket
/// 128 wide in `ln(x)` covers everything an `f64` can represent. This is a
/// runaway guard, not a working limit.
const GAMMA_ISF_MAX_ITER: usize = 256;

/// Floor for the modified Lentz continued fraction, guarding a zero pivot.
///
/// ### References
///
/// Press et al., Numerical Recipes, 3rd ed., section 6.2
const LENTZ_TINY: f64 = 1e-300;

/// Relative convergence tolerance for the continued fraction.
const LENTZ_EPS: f64 = 2.220446049250313e-16;

/// Slack allowed when deciding whether an outcome is as improbable as the
/// observed one, in [`binom_test_one`].
///
/// R's own `binom.test` carries the same `1 + 1e-7`. It is not a fudge factor
/// for sloppy arithmetic: for a `p` of one third and a total of `3k + 2`, the
/// outcomes `k` and `k + 1` are *exactly* equiprobable, and which side of the
/// comparison each lands on is then decided by the last bit of whichever
/// routine evaluated the mass. Without the slack the p-value for two
/// mathematically identical situations differs by a whole term.
const BINOM_TIE_TOL: f64 = 1.0 + 1e-7;

/////////////////////
// RejectionRegion //
/////////////////////

/// Which set of outcomes counts as "at least as extreme as observed".
///
/// All three condition on the total count and sum negative binomial mass over a
/// rejection region; they differ only in how that region is chosen. When the
/// two groups have the same number of samples the regions coincide and edgeR
/// routes [`RejectionRegion::Deviance`] and [`RejectionRegion::SmallP`]
/// straight to the double-tail test, which this does too.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RejectionRegion {
    /// Double the smaller tail. edgeR's default, and the only one with the
    /// beta approximation for large counts.
    DoubleTail,
    /// Every split whose negative binomial deviance is at least the observed
    /// one.
    Deviance,
    /// Every split no more probable than the observed one.
    SmallP,
}

/////////////////////
// ExactTestResult //
/////////////////////

/// The three columns edgeR's `exactTest` reports.
#[derive(Clone, Debug)]
pub struct ExactTestResult {
    /// Log2 fold change of the second group over the first, one per gene.
    pub log_fc: Vec<f64>,
    /// Average log2 counts per million, one per gene.
    pub log_cpm: Vec<f64>,
    /// Raw exact-test p-value, one per gene.
    pub p_value: Vec<f64>,
}

/////////////////////
// ExactTestParams //
/////////////////////

/// Tuning knobs for [`exact_test`].
#[derive(Clone, Copy, Debug)]
pub struct ExactTestParams {
    /// Which rejection region to sum over.
    pub rejection_region: RejectionRegion,
    /// Row sum above which, in both groups, the beta approximation is used.
    /// Only consulted by [`RejectionRegion::DoubleTail`].
    pub big_count: f64,
    /// Prior count added before the fold change is fitted.
    pub prior_count: f64,
}

impl Default for ExactTestParams {
    /// edgeR's defaults: double tail, `big.count = 900`, `prior.count = 0.125`.
    fn default() -> Self {
        Self {
            rejection_region: RejectionRegion::DoubleTail,
            big_count: BIG_COUNT_DEFAULT,
            prior_count: PRIOR_COUNT_DEFAULT,
        }
    }
}

impl ExactTestParams {
    /// Builds a parameter set explicitly.
    ///
    /// ### Params
    ///
    /// * `rejection_region` - Which outcomes count as extreme
    /// * `big_count` - Beta approximation threshold, non-negative
    /// * `prior_count` - Prior count for the fold change, non-negative
    ///
    /// ### Returns
    ///
    /// The parameter set. Values are validated at the point of use in
    /// [`exact_test`], not here.
    pub fn new(rejection_region: RejectionRegion, big_count: f64, prior_count: f64) -> Self {
        Self {
            rejection_region,
            big_count,
            prior_count,
        }
    }
}

//////////////////////////////////////
// Log-space upper incomplete gamma //
//////////////////////////////////////

/// Modified Lentz evaluation of the continued fraction behind `Q(a, x)`.
///
/// Returns the fraction alone, without the `exp(a ln x - x - ln Gamma(a))`
/// prefactor, so the caller can keep that part in logs. The fraction itself is
/// of order `1 / x` and never underflows over the range this module uses.
///
/// Only valid for `x >= a + 1`, where the fraction converges quickly.
///
/// ### Params
///
/// * `a` - Shape, strictly positive
/// * `x` - Argument, at least `a + 1`
///
/// ### Returns
///
/// The continued fraction value.
///
/// ### References
///
/// Press et al., Numerical Recipes, 3rd ed., section 6.2
fn gamma_cont_frac(a: f64, x: f64) -> f64 {
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / LENTZ_TINY;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..=GAMMA_ISF_MAX_ITER * 4 {
        let i = i as f64;
        let an = -i * (i - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < LENTZ_TINY {
            d = LENTZ_TINY;
        }
        c = b + an / c;
        if c.abs() < LENTZ_TINY {
            c = LENTZ_TINY;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() <= LENTZ_EPS {
            break;
        }
    }
    h
}

/// Log of the regularised upper incomplete gamma, `ln Q(a, x)`.
///
/// The reason this exists rather than `gamma_sf(...).ln()`: `q2qnbinom` maps a
/// count that can sit hundreds of standard deviations above its fitted mean, and
/// R evaluates the whole round trip with `log.p = TRUE` for exactly that reason.
/// A count of 6000 against a fitted mean of 1000 at dispersion 0.001 has
/// `Q = e^-1605`, which is a flat zero on the natural scale and would send the
/// inverse to `+inf`.
///
/// Two branches, split where each is well conditioned. Below `x = a + 1` the
/// upper tail is a decent fraction of the mass, so `chisq_sf` computes it
/// directly and the logarithm is taken afterwards. Above it the prefactor is
/// kept in logs and only the continued fraction, which is order one, is
/// exponentiated.
///
/// ### Params
///
/// * `a` - Shape, strictly positive
/// * `x` - Argument, non-negative
///
/// ### Returns
///
/// `ln Q(a, x)` in `(-inf, 0]`, or [`EdgeErrors::InvalidArgument`] for a
/// non-positive shape.
fn ln_reg_gamma_upper(a: f64, x: f64) -> Result<f64, EdgeErrors> {
    if x <= 0.0 {
        return Ok(0.0);
    }
    if x < a + 1.0 {
        // Q is at least about 0.15 here, so there is nothing to lose to the log.
        return Ok(chisq_sf(2.0 * x, 2.0 * a)?.ln());
    }
    Ok(a * x.ln() - x - ln_gamma(a) + gamma_cont_frac(a, x).ln())
}

/// Starting point for [`ln_reg_gamma_upper_inv`], in `x` at unit scale.
///
/// For `x` well past the mean, `ln Q(a, x) ~ (a - 1) ln x - x - ln Gamma(a)`.
/// Rearranged that is `x = c + (a - 1) ln x` with `c = -ln q - ln Gamma(a)`,
/// which converges in a handful of fixed-point steps whenever `c` exceeds the
/// shape. When it does not, the quantile is at or below the mean and the shape
/// itself is a good enough start.
///
/// The result only has to land in roughly the right decade: the caller brackets
/// it and bisects, so a poor guess costs iterations, never correctness.
///
/// ### Params
///
/// * `ln_q` - Target log upper-tail probability, at most 0
/// * `a` - Shape, strictly positive
///
/// ### Returns
///
/// An initial `x`, strictly positive and finite.
fn gamma_isf_start(ln_q: f64, a: f64) -> f64 {
    let c = -ln_q - ln_gamma(a);
    if !c.is_finite() || c <= a {
        return a.max(f64::MIN_POSITIVE);
    }
    let mut x = c.max(a + 1.0);
    for _ in 0..8 {
        let next = c + (a - 1.0) * x.ln();
        if !(next.is_finite() && next > 0.0) {
            break;
        }
        x = next;
    }
    x
}

/// Inverts [`ln_reg_gamma_upper`] at unit scale.
///
/// Newton in `t = ln x` on a bracketed, monotonically decreasing residual, with
/// a bisection fallback whenever the Newton step leaves the bracket. The
/// derivative `d ln Q / dt = -x f(x) / Q` is formed entirely in logs, so it
/// stays representable at the same depth the residual does.
///
/// ### Params
///
/// * `ln_q` - Target log upper-tail probability, at most 0
/// * `a` - Shape, strictly positive
///
/// ### Returns
///
/// `x` with `ln Q(a, x) = ln_q`, or [`EdgeErrors::NoConvergence`] if the bracket
/// has not closed within [`GAMMA_ISF_MAX_ITER`].
fn ln_reg_gamma_upper_inv(ln_q: f64, a: f64) -> Result<f64, EdgeErrors> {
    if ln_q >= 0.0 {
        return Ok(0.0);
    }
    if ln_q == f64::NEG_INFINITY {
        return Ok(f64::INFINITY);
    }
    let ln_norm = ln_gamma(a);

    // Decreasing in t: Q falls as x grows, so the residual does too.
    let residual =
        |t: f64| -> Result<f64, EdgeErrors> { Ok(ln_reg_gamma_upper(a, t.exp())? - ln_q) };

    let mut t = gamma_isf_start(ln_q, a).ln();
    let mut lo = t;
    let mut hi = t;
    if residual(t)? > 0.0 {
        // Q is still too large: walk the upper end out until it undershoots.
        for _ in 0..GAMMA_ISF_MAX_ITER {
            hi += 1.0;
            if residual(hi)? <= 0.0 {
                break;
            }
            lo = hi;
        }
    } else {
        for _ in 0..GAMMA_ISF_MAX_ITER {
            lo -= 1.0;
            if residual(lo)? > 0.0 {
                break;
            }
            hi = lo;
        }
    }

    for _ in 0..GAMMA_ISF_MAX_ITER {
        if !(t > lo && t < hi) {
            t = 0.5 * (lo + hi);
        }
        let f = residual(t)?;
        // Residual falls with t, so a positive value sits below the root.
        if f > 0.0 {
            lo = t;
        } else {
            hi = t;
        }
        let tol = GAMMA_ISF_REL_TOL * (1.0 + t.abs());
        if hi - lo <= tol {
            return Ok(t.exp());
        }
        // -d(residual)/dt = exp(a t - e^t - ln Gamma(a) - ln Q), kept in logs.
        let slope = (a * t - t.exp() - ln_norm - (f + ln_q)).exp();
        let t_next = if slope > 0.0 && slope.is_finite() {
            t + f / slope
        } else {
            0.5 * (lo + hi)
        };
        let step = (t_next - t).abs();
        t = t_next;
        if step <= tol {
            return Ok(t.clamp(lo, hi).exp());
        }
    }

    Err(EdgeErrors::NoConvergence {
        routine: "ln_reg_gamma_upper_inv",
        iterations: GAMMA_ISF_MAX_ITER,
        last_delta: hi - lo,
    })
}

////////////////
// Validation //
////////////////

/// Rejects a dispersion outside `[0, inf)`.
///
/// ### Params
///
/// * `dispersion` - Value supplied by the caller
///
/// ### Returns
///
/// `Ok(())`, or [`EdgeErrors::InvalidDispersion`].
fn check_dispersion(dispersion: f64) -> Result<(), EdgeErrors> {
    if !dispersion.is_finite() || dispersion < 0.0 {
        return Err(EdgeErrors::InvalidDispersion(dispersion));
    }
    Ok(())
}

/// Rejects a slice holding a negative or non-finite value.
///
/// ### Params
///
/// * `name` - Argument name, used verbatim in the error message
/// * `values` - Values to check
///
/// ### Returns
///
/// `Ok(())`, or [`EdgeErrors::InvalidArgument`] naming the offending value.
fn check_non_negative(name: &str, values: &[f64]) -> Result<(), EdgeErrors> {
    if let Some(bad) = values.iter().find(|v| !v.is_finite() || **v < 0.0) {
        return Err(EdgeErrors::InvalidArgument(format!(
            "'{name}' must be non-negative and finite, found {bad}"
        )));
    }
    Ok(())
}

///////////////
// q2qnbinom //
///////////////

/// Maps one count from one negative binomial mean to another.
///
/// The scalar kernel behind [`q2q_nbinom`]; see there for the algorithm and for
/// where it departs from edgeR's arithmetic.
///
/// ### Params
///
/// * `x` - Count to map, non-negative
/// * `input_mean` - Mean the count was drawn under, non-negative
/// * `output_mean` - Mean to map it onto, non-negative
/// * `dispersion` - Dispersion, non-negative and finite
///
/// ### Returns
///
/// The mapped count, or [`EdgeErrors`] propagated from the quantile solvers.
#[inline]
fn q2q_one(x: f64, input_mean: f64, output_mean: f64, dispersion: f64) -> Result<f64, EdgeErrors> {
    // Either mean falling below eps nudges both, so a gene fitted at zero maps
    // to itself rather than dividing by zero.
    let (mi, mo) = if input_mean < Q2Q_EPS || output_mean < Q2Q_EPS {
        (input_mean + Q2Q_ZERO_NUDGE, output_mean + Q2Q_ZERO_NUDGE)
    } else {
        (input_mean, output_mean)
    };

    let ri = 1.0 + dispersion * mi;
    let vi = mi * ri;
    let ro = 1.0 + dispersion * mo;
    let vo = mo * ro;
    let q1 = mo + (vo / vi).sqrt() * (x - mi);

    let shape_i = mi / ri;
    let shape_o = mo / ro;
    let q2 = if x >= mi {
        let ln_q = ln_reg_gamma_upper(shape_i, x / ri)?;
        ro * ln_reg_gamma_upper_inv(ln_q, shape_o)?
    } else {
        let p = gamma_cdf(x, shape_i, ri)?;
        gamma_ppf(p, shape_o, ro)?
    };

    Ok(0.5 * (q1 + q2))
}

/// Quantile-to-quantile mapping between two negative binomial means.
///
/// edgeR's `q2qnbinom`. Each count is placed at its quantile under
/// `NB(input_mean, dispersion)` and read back off
/// `NB(output_mean, dispersion)`. The negative binomial quantile has no closed
/// form, so edgeR maps through two continuous approximations and averages them:
/// a normal with the matching mean and variance, and a gamma with the matching
/// mean and variance. Neither is good on its own, since the normal is symmetric
/// and the gamma has the wrong behaviour near zero, and the average of the two
/// is what edgeR's pseudo-counts are actually built from.
///
/// The tail is chosen per element by `x >= input_mean`, so whichever side of
/// the mean the count sits on is the side that is evaluated, and the small
/// probability is never formed as one minus a large one.
///
/// ### Notes
///
/// **Two departures from edgeR's arithmetic**
///
/// * The normal half is the linear map `output_mean + sd_out (x - input_mean) /
///   sd_in`, which is what `qnorm(pnorm(...))` reduces to exactly. edgeR forms
///   it as a round trip through log probabilities; the answers agree to the
///   last bit or two and this one cannot lose any.
/// * The upper gamma tail goes through [`ln_reg_gamma_upper`] and its inverse
///   rather than through `gamma_cdf` and `gamma_ppf`, because those work on the
///   natural scale and R uses `log.p = TRUE` here for a reason.
///
/// ### Params
///
/// * `x` - Counts to map, non-negative
/// * `input_mean` - Mean each count was drawn under, same length as `x`,
///   non-negative
/// * `output_mean` - Mean to map each count onto, same length as `x`,
///   non-negative
/// * `dispersion` - Dispersion shared by every element, non-negative and finite.
///   A per-gene dispersion is applied by calling this once per gene, which is
///   how [`equalize_lib_sizes`] uses it.
///
/// ### Returns
///
/// The mapped counts, same length as `x`. Errors are
/// [`EdgeErrors::LengthMismatch`] if the three slices disagree,
/// [`EdgeErrors::InvalidDispersion`] for a negative or non-finite dispersion,
/// and [`EdgeErrors::InvalidArgument`] for a negative count or mean.
///
/// ### References
///
/// Robinson and Smyth, Biostatistics 9(2), 2008
pub fn q2q_nbinom(
    x: &[f64],
    input_mean: &[f64],
    output_mean: &[f64],
    dispersion: f64,
) -> Result<Vec<f64>, EdgeErrors> {
    if input_mean.len() != x.len() {
        return Err(EdgeErrors::LengthMismatch {
            name: "input_mean",
            expected: x.len(),
            got: input_mean.len(),
        });
    }
    if output_mean.len() != x.len() {
        return Err(EdgeErrors::LengthMismatch {
            name: "output_mean",
            expected: x.len(),
            got: output_mean.len(),
        });
    }
    check_dispersion(dispersion)?;
    check_non_negative("x", x)?;
    check_non_negative("input_mean", input_mean)?;
    check_non_negative("output_mean", output_mean)?;

    x.iter()
        .zip(input_mean.iter())
        .zip(output_mean.iter())
        .map(|((v, im), om)| q2q_one(*v, *im, *om, dispersion))
        .collect()
}

///////////////////
// Pseudo-counts //
///////////////////

/// Distinct labels in first-appearance order, with a mask per label.
///
/// R's `unique` on a factor keeps first-appearance order, and
/// `equalizeLibSizes` loops over it, so this does the same. The order only
/// affects which columns are written first, never the values.
///
/// ### Params
///
/// * `group` - One label per sample
///
/// ### Returns
///
/// One entry per distinct label, holding the sample indices carrying it.
fn group_columns(group: &[usize]) -> Vec<Vec<usize>> {
    let mut labels: Vec<usize> = Vec::new();
    let mut columns: Vec<Vec<usize>> = Vec::new();
    for (sample, label) in group.iter().enumerate() {
        match labels.iter().position(|l| l == label) {
            Some(k) => columns[k].push(sample),
            None => {
                labels.push(*label);
                columns.push(vec![sample]);
            }
        }
    }
    columns
}

/// Copies a subset of columns out of a row-major matrix, as `f64`.
///
/// ### Params
///
/// * `counts` - Row-major counts, `n_genes * n_samples`
/// * `n_genes` - Number of genes
/// * `n_samples` - Number of samples in `counts`
/// * `columns` - Sample indices to keep, in the order wanted
///
/// ### Returns
///
/// A row-major `n_genes * columns.len()` matrix.
fn select_columns<T: EdgeFloat>(
    counts: &[T],
    n_genes: usize,
    n_samples: usize,
    columns: &[usize],
) -> Vec<f64> {
    let mut out = vec![0.0_f64; n_genes * columns.len()];
    out.par_chunks_mut(columns.len())
        .enumerate()
        .for_each(|(gene, row)| {
            let src = &counts[gene * n_samples..(gene + 1) * n_samples];
            for (dst, &col) in row.iter_mut().zip(columns.iter()) {
                *dst = src[col].to_f64().unwrap_or(f64::NAN);
            }
        });
    out
}

/// Expands a recycled dispersion into one value per gene.
///
/// Only the gene axis is meaningful for the exact test, so a
/// [`Recycled::BySample`] or [`Recycled::Full`] form is an error rather than
/// something to average over.
///
/// ### Params
///
/// * `dispersion` - Dispersion, recycled over genes and samples
/// * `n_genes` - Number of genes
///
/// ### Returns
///
/// One dispersion per gene, or [`EdgeErrors::LengthMismatch`] on a bad length,
/// [`EdgeErrors::InvalidDispersion`] on a negative or non-finite value and
/// [`EdgeErrors::InvalidArgument`] on a sample-varying form.
fn per_gene_dispersion(dispersion: &Recycled<f64>, n_genes: usize) -> Result<Vec<f64>, EdgeErrors> {
    let out = match dispersion {
        Recycled::Scalar(v) => vec![*v; n_genes],
        Recycled::ByGene(v) => {
            if v.len() != n_genes {
                return Err(EdgeErrors::LengthMismatch {
                    name: "by_gene",
                    expected: n_genes,
                    got: v.len(),
                });
            }
            v.clone()
        }
        Recycled::BySample(_) | Recycled::Full(_) => {
            return Err(EdgeErrors::InvalidArgument(
                "the exact test needs a dispersion that varies by gene only; got a \
                 sample-varying form"
                    .to_string(),
            ));
        }
    };
    for d in &out {
        check_dispersion(*d)?;
    }
    Ok(out)
}

/// Pseudo-counts on a common library size, edgeR's `equalizeLibSizes`.
///
/// The classic path cannot condition on the total count unless every sample has
/// the same library size, so edgeR manufactures that. Within each group it fits
/// an intercept-only negative binomial GLM to get a per-gene rate, forms the
/// fitted mean each observation was drawn under and the fitted mean it would
/// have had at the common library size, and maps between the two with
/// [`q2q_nbinom`]. The common size is the geometric mean of the library sizes,
/// which keeps the pseudo-counts on roughly the scale of the originals.
///
/// Negative pseudo-counts are clamped to zero, as edgeR does. The normal half
/// of the mapping is unbounded below and will go negative for a zero count
/// mapped onto a much larger library.
///
/// Genes are the parallel axis for the mapping. The per-group GLM fits are
/// already parallel over genes inside [`mglm_one_group`].
///
/// ### Params
///
/// * `counts` - Row-major counts, `n_genes * n_samples`
/// * `n_genes` - Number of genes
/// * `n_samples` - Number of samples
/// * `group` - One label per sample, or `None` to treat every sample as one
///   group
/// * `dispersion` - Dispersion, recycled over genes and samples. A
///   [`Recycled::BySample`] or [`Recycled::Full`] form is rejected: the mapping
///   is defined per gene, and edgeR has no notion of a sample-varying
///   dispersion here.
/// * `lib_size` - Library size per sample, or `None` for the column sums
///
/// ### Returns
///
/// Pseudo-counts, row-major `n_genes * n_samples`. The common library size they
/// sit on is `exp(mean(ln(lib_size)))` and is not returned, being a one-line
/// function of the input. Errors are [`EdgeErrors::EmptyCounts`] for a zero
/// dimension, [`EdgeErrors::LengthMismatch`] for a `group` or `lib_size` of the
/// wrong length, [`EdgeErrors::InvalidDispersion`] for a negative dispersion
/// and [`EdgeErrors::InvalidArgument`] for a non-positive library size.
pub fn equalize_lib_sizes<T: EdgeFloat>(
    counts: &[T],
    n_genes: usize,
    n_samples: usize,
    group: Option<&[usize]>,
    dispersion: &Recycled<f64>,
    lib_size: Option<&[f64]>,
) -> Result<Vec<f64>, EdgeErrors> {
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
    let per_gene = per_gene_dispersion(dispersion, n_genes)?;

    let ones = vec![0_usize; n_samples];
    let group = match group {
        Some(g) => {
            if g.len() != n_samples {
                return Err(EdgeErrors::LengthMismatch {
                    name: "group",
                    expected: n_samples,
                    got: g.len(),
                });
            }
            g
        }
        None => &ones,
    };

    let derived: Vec<f64>;
    let lib_size = match lib_size {
        Some(l) => {
            if l.len() != n_samples {
                return Err(EdgeErrors::LengthMismatch {
                    name: "lib_size",
                    expected: n_samples,
                    got: l.len(),
                });
            }
            l
        }
        None => {
            derived = column_sums(counts, n_samples);
            &derived
        }
    };
    if let Some(bad) = lib_size.iter().find(|v| !v.is_finite() || **v <= 0.0) {
        return Err(EdgeErrors::InvalidArgument(format!(
            "library sizes must be positive and finite, found {bad}"
        )));
    }

    let common_lib_size = (lib_size.iter().map(|l| l.ln()).sum::<f64>() / n_samples as f64).exp();

    // Fitted rate per gene, one fit per group. Written back into a full
    // n_genes by n_samples layout so the mapping below is one pass over rows.
    let mut input_mean = vec![0.0_f64; n_genes * n_samples];
    let mut output_mean = vec![0.0_f64; n_genes * n_samples];
    for columns in group_columns(group) {
        let sub = select_columns(counts, n_genes, n_samples, &columns);
        let offsets: Vec<f64> = columns.iter().map(|&j| lib_size[j].ln()).collect();
        let beta = mglm_one_group(
            &sub,
            n_genes,
            columns.len(),
            dispersion,
            &Recycled::by_sample(offsets),
            None,
            None,
            None,
        )?;
        for (gene, b) in beta.iter().enumerate() {
            let lambda = b.exp();
            for &j in &columns {
                input_mean[gene * n_samples + j] = lambda * lib_size[j];
                output_mean[gene * n_samples + j] = lambda * common_lib_size;
            }
        }
    }

    let mut pseudo = vec![0.0_f64; n_genes * n_samples];
    let outcome: Result<(), EdgeErrors> = pseudo
        .par_chunks_mut(n_samples)
        .enumerate()
        .try_for_each(|(gene, row)| {
            let y = &counts[gene * n_samples..(gene + 1) * n_samples];
            let im = &input_mean[gene * n_samples..(gene + 1) * n_samples];
            let om = &output_mean[gene * n_samples..(gene + 1) * n_samples];
            let d = per_gene[gene];
            for (j, out) in row.iter_mut().enumerate() {
                let count = y[j].to_f64().unwrap_or(f64::NAN);
                *out = q2q_one(count, im[j], om[j], d)?.max(0.0);
            }
            Ok(())
        });
    outcome?;

    Ok(pseudo)
}

/////////////////////
// Exact test core //
/////////////////////

/// Log of a sum of exponentials, shifted by the maximum.
///
/// The convolution below sums thousands of negative binomial masses whose
/// product can be `e^-800` apiece, so the sum is formed in logs rather than
/// accumulated on the natural scale as edgeR's R loop does.
///
/// ### Params
///
/// * `values` - Log-scale terms. An empty slice sums to `-inf`.
///
/// ### Returns
///
/// `ln(sum(exp(values)))`.
fn log_sum_exp(values: &[f64]) -> f64 {
    let mut max = f64::NEG_INFINITY;
    for v in values {
        if *v > max {
            max = *v;
        }
    }
    if max == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    let sum: f64 = values.iter().map(|v| (v - max).exp()).sum();
    max + sum.ln()
}

/// Binomial log-PMF, for the Poisson limit of the exact test.
///
/// ### Params
///
/// * `k` - Number of successes, in `0..=n`
/// * `n` - Number of trials
/// * `ln_p` - Log of the success probability
/// * `ln_q` - Log of one minus the success probability
///
/// ### Returns
///
/// `ln P(K = k)`.
#[inline]
fn binom_ln_pmf(k: f64, n: f64, ln_p: f64, ln_q: f64) -> f64 {
    ln_gamma(n + 1.0) - ln_gamma(k + 1.0) - ln_gamma(n - k + 1.0) + k * ln_p + (n - k) * ln_q
}

/// Exact binomial test on one gene's split, edgeR's `binomTest`.
///
/// The Poisson limit of the exact test: at zero dispersion the conditional
/// distribution of the first group's count given the total is exactly binomial,
/// so no convolution is needed.
///
/// Two branches, as in edgeR. At `p = 0.5` the distribution is symmetric and
/// twice the smaller tail is the answer in closed form. Otherwise it is the
/// small-probability rule: sum every outcome no more probable than the observed
/// one.
///
/// ### Notes
///
/// **Departure from edgeR**
///
/// edgeR replaces the enumeration with a Yates-corrected two-by-two chi-square
/// test once the total exceeds 10000, and the two-by-two it builds uses the
/// column totals of whichever subset of genes happened to be passed in, so the
/// answer for one gene depends on the others. That is a speed shortcut with a
/// bug attached. The enumeration is `O(total)` and is used here at every size.
///
/// Ties are admitted within [`BINOM_TIE_TOL`], as R's `binom.test` does, rather
/// than resolved by `order` on raw masses as edgeR's `binomTest` does. Where
/// the two disagree it is because two outcomes are exactly equiprobable and
/// edgeR's `dbinom` put them one ulp apart; the p-value then depends on which
/// of the two was observed, which it should not.
///
/// ### Params
///
/// * `y1` - First group's total, non-negative and integral
/// * `y2` - Second group's total, non-negative and integral
/// * `p` - Success probability, strictly inside `(0, 1)`
///
/// ### Returns
///
/// The two-sided p-value, or [`EdgeErrors`] propagated from [`beta_cdf`].
fn binom_test_one(y1: f64, y2: f64, p: f64) -> Result<f64, EdgeErrors> {
    let size = y1 + y2;
    if size <= 0.0 {
        return Ok(1.0);
    }
    if p == 0.5 {
        // pbinom(k, n, 0.5) is the regularised incomplete beta I(0.5; n-k, k+1).
        let k = y1.min(y2);
        let cdf = beta_cdf(0.5, size - k, k + 1.0)?;
        return Ok((2.0 * cdf).min(1.0));
    }

    let n = size as usize;
    let ln_p = p.ln();
    let ln_q = (-p).ln_1p();
    let masses: Vec<f64> = (0..=n)
        .map(|k| binom_ln_pmf(k as f64, size, ln_p, ln_q).exp())
        .collect();

    let threshold = masses[y1 as usize] * BINOM_TIE_TOL;
    let mut kept: Vec<f64> = masses.into_iter().filter(|m| *m <= threshold).collect();
    // Ascending, so the sum starts from the terms that would otherwise be lost
    // under the mode. This is also the order R's `cumsum(d[order(d)])` uses.
    kept.sort_by(f64::total_cmp);
    Ok(kept.iter().sum::<f64>().min(1.0))
}

/// Beta approximation to the double-tail test, edgeR's `exactTestBetaApprox`.
///
/// For a gene with thousands of reads in each group the convolution is
/// thousands of terms and the conditional distribution of the first group's
/// share is already indistinguishable from a beta with the matching first two
/// moments. Both tails carry a half-count continuity correction and are
/// compared against the beta median rather than its mean, so the split matches
/// the discrete distribution's own centre.
///
/// The sums used here are the raw pseudo-counts, not the rounded ones. edgeR
/// decides *whether* to take this path from the rounded sums and then hands the
/// unrounded matrix over, and the correction is a half count, so the difference
/// shows.
///
/// ### Params
///
/// * `raw1` - First group's unrounded pseudo-count sum, non-negative
/// * `raw2` - Second group's unrounded pseudo-count sum, non-negative
/// * `n1` - Number of samples in the first group, at least one
/// * `n2` - Number of samples in the second group, at least one
/// * `dispersion` - Dispersion for this gene, non-negative
///
/// ### Returns
///
/// The two-sided p-value, or [`EdgeErrors`] propagated from the beta routines.
fn beta_approx_one(
    raw1: f64,
    raw2: f64,
    n1: usize,
    n2: usize,
    dispersion: f64,
) -> Result<f64, EdgeErrors> {
    let total = raw1 + raw2;
    if total <= 0.0 {
        return Ok(1.0);
    }
    let mu = total / (n1 + n2) as f64;
    let alpha1 = n1 as f64 * mu / (1.0 + dispersion * mu);
    let alpha2 = (n2 as f64 / n1 as f64) * alpha1;
    let median = beta_ppf(0.5, alpha1, alpha2)?;

    let left = (raw1 + 0.5) / total;
    let right = (raw1 - 0.5) / total;
    if left < median {
        Ok((2.0 * beta_cdf(left, alpha1, alpha2)?).min(1.0))
    } else if right > median {
        Ok((2.0 * beta_sf(right, alpha1, alpha2)?).min(1.0))
    } else {
        Ok(1.0)
    }
}

/// Double-tail exact p-value for one gene, edgeR's `exactTestDoubleTail`.
///
/// Conditioning on the total, the first group's count follows a known
/// distribution; the p-value is twice the mass in whichever tail the observation
/// falls in. Doubling the smaller tail rather than summing two genuinely
/// two-sided tails is what makes this the "double tail" test, and it is what
/// edgeR reports by default.
///
/// Three paths, chosen exactly as edgeR chooses them: the exact binomial at zero
/// dispersion, the beta approximation when both rounded sums exceed `big_count`,
/// and the convolution otherwise.
///
/// The convolution is accumulated in logs rather than on the natural scale.
/// edgeR sums `dnbinom` products directly, which silently returns zero once
/// every term has underflowed; a gene with a few thousand reads and a small
/// dispersion gets there.
///
/// ### Params
///
/// * `y1` - First group's pseudo-counts for this gene
/// * `y2` - Second group's pseudo-counts for this gene
/// * `dispersion` - Dispersion for this gene, non-negative
/// * `big_count` - Beta approximation threshold
/// * `buf` - Scratch buffer for the convolution terms, cleared on entry
///
/// ### Returns
///
/// The p-value, capped at one, or [`EdgeErrors`] propagated from the
/// distribution routines.
fn double_tail_one(
    y1: &[f64],
    y2: &[f64],
    dispersion: f64,
    big_count: f64,
    buf: &mut Vec<f64>,
) -> Result<f64, EdgeErrors> {
    let (n1, n2) = (y1.len(), y2.len());
    let raw1: f64 = y1.iter().sum();
    let raw2: f64 = y2.iter().sum();
    // R's round is half-to-even, and a pseudo-count landing on a half is not
    // as rare as it looks: an all-zero group sums to exactly zero.
    let s1 = raw1.round_ties_even();
    let s2 = raw2.round_ties_even();

    if dispersion <= 0.0 {
        return binom_test_one(s1, s2, n1 as f64 / (n1 + n2) as f64);
    }
    if s1 > big_count && s2 > big_count {
        return beta_approx_one(raw1, raw2, n1, n2, dispersion);
    }

    let total = s1 + s2;
    if total == 0.0 {
        return Ok(1.0);
    }
    let n_total = (n1 + n2) as f64;
    let mu = total / n_total;
    let mu1 = n1 as f64 * mu;
    let mu2 = n2 as f64 * mu;
    let size1 = n1 as f64 / dispersion;
    let size2 = n2 as f64 / dispersion;
    let size_total = n_total / dispersion;
    // The same success probability everywhere, since mu1 / size1, mu2 / size2
    // and total / size_total are all the dispersion times the per-sample mean.
    let prob1 = size1 / (size1 + mu1);
    let prob2 = size2 / (size2 + mu2);
    let prob_total = size_total / (size_total + total);

    let (lo, hi) = if s1 < mu1 {
        (0.0, s1)
    } else if s1 > mu1 {
        (s1, total)
    } else {
        return Ok(1.0);
    };

    buf.clear();
    let mut x = lo;
    while x <= hi {
        buf.push(nbinom_ln_pmf(x, size1, prob1)? + nbinom_ln_pmf(total - x, size2, prob2)?);
        x += 1.0;
    }
    let ln_bottom = nbinom_ln_pmf(total, size_total, prob_total)?;
    Ok((LN_2 + log_sum_exp(buf) - ln_bottom).min(0.0).exp())
}

/// Deviance-based exact p-value for one gene, edgeR's `exactTestByDeviance`.
///
/// The rejection region is every split of the total whose negative binomial
/// deviance is at least the observed one. Unlike the double-tail test this is a
/// genuinely two-sided region rather than one tail doubled, so it can only
/// differ when the two groups have different numbers of samples: with equal
/// group sizes the deviance is symmetric about the midpoint and the two regions
/// coincide, which is why edgeR short-circuits that case.
///
/// The scan walks out from both ends and stops at the first split inside the
/// acceptance region, which is what edgeR's C++ does. That relies on the
/// deviance being unimodal in the split, which it is.
///
/// The conditional mass has the same success probability in every factor, so it
/// cancels between the numerator and the denominator; this keeps edgeR's form
/// anyway, since the cancellation is exact to the last bit or two and the
/// explicit form is easier to check against the reference.
///
/// ### Params
///
/// * `y1` - First group's pseudo-counts for this gene
/// * `y2` - Second group's pseudo-counts for this gene
/// * `dispersion` - Dispersion for this gene, strictly positive
/// * `buf` - Scratch buffer for the retained terms, cleared on entry
///
/// ### Returns
///
/// The p-value, capped at one, or [`EdgeErrors`] propagated from
/// [`nbinom_ln_pmf`].
fn deviance_one(
    y1: &[f64],
    y2: &[f64],
    dispersion: f64,
    buf: &mut Vec<f64>,
) -> Result<f64, EdgeErrors> {
    let (n1, n2) = (y1.len(), y2.len());
    let s1: f64 = y1.iter().sum::<f64>().round_ties_even();
    let s2: f64 = y2.iter().sum::<f64>().round_ties_even();

    if dispersion <= 0.0 {
        return binom_test_one(s1, s2, n1 as f64 / (n1 + n2) as f64);
    }
    let total = s1 + s2;
    if total == 0.0 {
        return Ok(1.0);
    }

    let n_total = (n1 + n2) as f64;
    let mu = total / n_total;
    let mu1 = n1 as f64 * mu;
    let mu2 = n2 as f64 * mu;
    let size1 = n1 as f64 / dispersion;
    let size2 = n2 as f64 / dispersion;
    let phi1 = 1.0 / size1;
    let phi2 = 1.0 / size2;
    let prob = size1 / (size1 + mu1);

    let observed = unit_nb_deviance(s1, mu1, phi1) + unit_nb_deviance(s2, mu2, phi2);
    buf.clear();

    // From the left: splits with the first group small.
    let mut j = 0.0_f64;
    while j <= total {
        if observed > unit_nb_deviance(j, mu1, phi1) + unit_nb_deviance(total - j, mu2, phi2) {
            break;
        }
        buf.push(nbinom_ln_pmf(j, size1, prob)? + nbinom_ln_pmf(total - j, size2, prob)?);
        j += 1.0;
    }
    // From the right, over whatever the left scan did not already cover.
    let mut k = 0.0_f64;
    while k <= total - j {
        if observed > unit_nb_deviance(k, mu2, phi2) + unit_nb_deviance(total - k, mu1, phi1) {
            break;
        }
        buf.push(nbinom_ln_pmf(k, size2, prob)? + nbinom_ln_pmf(total - k, size1, prob)?);
        k += 1.0;
    }

    let size_total = size1 + size2;
    let ln_bottom = nbinom_ln_pmf(total, size_total, size_total / (size_total + mu1 + mu2))?;
    Ok((log_sum_exp(buf) - ln_bottom).min(0.0).exp())
}

/// Small-probability exact p-value for one gene, edgeR's `exactTestBySmallP`.
///
/// The rejection region is every split no more probable than the observed one,
/// which is the textbook definition of an exact test and the most expensive of
/// the three: there is no monotonicity to exploit, so every split is evaluated.
/// As with the deviance region this only differs from the double-tail test when
/// the groups differ in size.
///
/// ### One departure from edgeR
///
/// `exactTestBySmallP` ends on `min(pvals, 1)`, and `min` in R reduces a vector
/// to a scalar. Every gene therefore comes back carrying the smallest p-value
/// in the whole matrix. It is a typo for `pmin`, and it is not reproduced: this
/// caps each gene's own value. Calling edgeR one gene at a time recovers what
/// the function was meant to return, and that is what the fixtures here compare
/// against.
///
/// ### Params
///
/// * `y1` - First group's pseudo-counts for this gene
/// * `y2` - Second group's pseudo-counts for this gene
/// * `dispersion` - Dispersion for this gene, strictly positive
/// * `buf` - Scratch buffer for the retained terms, cleared on entry
///
/// ### Returns
///
/// The p-value, capped at one, or [`EdgeErrors`] propagated from
/// [`nbinom_ln_pmf`].
fn small_p_one(
    y1: &[f64],
    y2: &[f64],
    dispersion: f64,
    buf: &mut Vec<f64>,
) -> Result<f64, EdgeErrors> {
    let (n1, n2) = (y1.len(), y2.len());
    let s1: f64 = y1.iter().sum::<f64>().round_ties_even();
    let s2: f64 = y2.iter().sum::<f64>().round_ties_even();

    if dispersion <= 0.0 {
        return binom_test_one(s1, s2, n1 as f64 / (n1 + n2) as f64);
    }
    let total = s1 + s2;
    if total == 0.0 {
        return Ok(1.0);
    }

    let n_total = (n1 + n2) as f64;
    let mu = total / n_total;
    let size1 = n1 as f64 / dispersion;
    let size2 = n2 as f64 / dispersion;
    let prob = size1 / (size1 + n1 as f64 * mu);

    let ln_observed = nbinom_ln_pmf(s1, size1, prob)? + nbinom_ln_pmf(s2, size2, prob)?;
    buf.clear();
    let mut x = 0.0_f64;
    while x <= total {
        let term = nbinom_ln_pmf(x, size1, prob)? + nbinom_ln_pmf(total - x, size2, prob)?;
        if term <= ln_observed {
            buf.push(term);
        }
        x += 1.0;
    }

    let size_total = size1 + size2;
    let ln_bottom = nbinom_ln_pmf(total, size_total, size_total / (size_total + total))?;
    Ok((log_sum_exp(buf) - ln_bottom).min(0.0).exp())
}

/// Runs one of the three rejection regions over every gene.
///
/// The rayon fan-out for the whole test. Each thread keeps one convolution
/// buffer, which grows to the largest total it sees and is then reused.
///
/// [`RejectionRegion::Deviance`] and [`RejectionRegion::SmallP`] fall back to
/// the double-tail test when the groups have the same number of samples, as
/// edgeR does: the regions are identical there, and the double-tail path is the
/// only one with the large-count shortcut.
///
/// ### Params
///
/// * `y1` - First group's pseudo-counts, row-major `n_genes * n1`
/// * `y2` - Second group's pseudo-counts, row-major `n_genes * n2`
/// * `n_genes` - Number of genes
/// * `n1` - Number of samples in the first group
/// * `n2` - Number of samples in the second group
/// * `dispersion` - One dispersion per gene, non-negative
/// * `region` - Which rejection region to sum over
/// * `big_count` - Beta approximation threshold, double tail only
///
/// ### Returns
///
/// One p-value per gene, or the first [`EdgeErrors`] any gene produced.
#[allow(clippy::too_many_arguments)]
fn exact_pvalues(
    y1: &[f64],
    y2: &[f64],
    n_genes: usize,
    n1: usize,
    n2: usize,
    dispersion: &[f64],
    region: RejectionRegion,
    big_count: f64,
) -> Result<Vec<f64>, EdgeErrors> {
    let region = if n1 == n2 {
        RejectionRegion::DoubleTail
    } else {
        region
    };

    let mut out = vec![1.0_f64; n_genes];
    let outcome: Result<(), EdgeErrors> =
        out.par_iter_mut()
            .enumerate()
            .try_for_each_init(Vec::<f64>::new, |buf, (gene, p)| {
                let row1 = &y1[gene * n1..(gene + 1) * n1];
                let row2 = &y2[gene * n2..(gene + 1) * n2];
                let d = dispersion[gene];
                *p = match region {
                    RejectionRegion::DoubleTail => double_tail_one(row1, row2, d, big_count, buf)?,
                    RejectionRegion::Deviance => deviance_one(row1, row2, d, buf)?,
                    RejectionRegion::SmallP => small_p_one(row1, row2, d, buf)?,
                };
                Ok(())
            });
    outcome?;
    Ok(out)
}

/// Fits the abundance of one group with prior counts added.
///
/// edgeR's rule: the prior is scaled by each library's size relative to the
/// mean over the pair, added to the counts, and the offset is raised to
/// `log(lib + 2 * prior)` so that adding the prior does not by itself move the
/// fitted level.
///
/// ### Params
///
/// * `counts` - This group's counts, row-major `n_genes * n_group`
/// * `n_genes` - Number of genes
/// * `lib` - Effective library size per sample of this group
/// * `mean_lib` - Mean effective library size over *both* groups
/// * `prior_count` - Prior count before scaling
/// * `dispersion` - Dispersion, recycled over genes
///
/// ### Returns
///
/// One log-scale abundance per gene, or whatever [`mglm_one_group`] propagates.
fn augmented_abundance(
    counts: &[f64],
    n_genes: usize,
    lib: &[f64],
    mean_lib: f64,
    prior_count: f64,
    dispersion: &Recycled<f64>,
) -> Result<Vec<f64>, EdgeErrors> {
    let n_group = lib.len();
    let prior: Vec<f64> = lib.iter().map(|l| prior_count * l / mean_lib).collect();
    let offset: Vec<f64> = lib
        .iter()
        .zip(prior.iter())
        .map(|(l, p)| (l + 2.0 * p).ln())
        .collect();

    let mut augmented = vec![0.0_f64; n_genes * n_group];
    augmented
        .par_chunks_mut(n_group)
        .enumerate()
        .for_each(|(gene, row)| {
            let src = &counts[gene * n_group..(gene + 1) * n_group];
            for (j, v) in row.iter_mut().enumerate() {
                *v = src[j] + prior[j];
            }
        });

    mglm_one_group(
        &augmented,
        n_genes,
        n_group,
        dispersion,
        &Recycled::by_sample(offset),
        None,
        None,
        None,
    )
}

/// Glues two per-group matrices back into one, first group's columns first.
///
/// ### Params
///
/// * `y1` - First group's counts, row-major `n_genes * n1`
/// * `y2` - Second group's counts, row-major `n_genes * n2`
/// * `n_genes` - Number of genes
/// * `n1` - Number of samples in the first group
/// * `n2` - Number of samples in the second group
///
/// ### Returns
///
/// A row-major `n_genes * (n1 + n2)` matrix.
fn interleave(y1: &[f64], y2: &[f64], n_genes: usize, n1: usize, n2: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; n_genes * (n1 + n2)];
    out.par_chunks_mut(n1 + n2)
        .enumerate()
        .for_each(|(gene, row)| {
            row[..n1].copy_from_slice(&y1[gene * n1..(gene + 1) * n1]);
            row[n1..].copy_from_slice(&y2[gene * n2..(gene + 1) * n2]);
        });
    out
}

///////////////
// Front end //
///////////////

/// Exact test for differential expression between two groups, edgeR's
/// `exactTest`.
///
/// The classic two-group path. It reports three things per gene, and they come
/// from three different places:
///
/// * the log2 fold change, from two intercept-only negative binomial GLMs with a
///   small prior count added, one per group;
/// * the average log2-CPM, from `aveLogCPM` over the whole object, both groups
///   and any samples outside the pair included;
/// * the p-value, from pseudo-counts equalised onto a common library size and
///   then summed over a rejection region.
///
/// The pseudo-counts are built the way `exactTest` builds them and not the way
/// [`equalize_lib_sizes`] does: one pooled fit across both groups rather than a
/// fit per group, and the common library size is the geometric mean over the
/// pair rather than over every sample. That difference is deliberate in edgeR
/// and reproduced here.
///
/// ### Params
///
/// * `y` - Counts and the per-sample quantities. Must carry a `group`.
/// * `pair` - The two group labels to compare, as `(first, second)`. The fold
///   change is reported for the second over the first, matching edgeR's
///   `pair = c(a, b)` giving `logFC` of `b` against `a`. Samples in neither
///   group are dropped for the test but still count towards the log-CPM.
/// * `dispersion` - Dispersion, recycled over genes. `None` takes the most
///   structured estimate on `y`, as edgeR's `dispersion = "auto"` does.
/// * `params` - Tuning knobs, or [`ExactTestParams::default`]
///
/// ### Returns
///
/// The three columns, one entry per gene in the input order. Errors are
/// [`EdgeErrors::InvalidArgument`] for a missing group, a degenerate pair or a
/// group with no samples, [`EdgeErrors::InvalidDispersion`] if none is available
/// or one is negative, and whatever the GLM fit and the distribution routines
/// propagate.
///
/// ### References
///
/// Robinson and Smyth, Bioinformatics 23(21), 2007
pub fn exact_test<T: EdgeFloat>(
    y: &DgeList<T>,
    pair: (usize, usize),
    dispersion: Option<&Recycled<f64>>,
    params: Option<ExactTestParams>,
) -> Result<ExactTestResult, EdgeErrors> {
    let params = params.unwrap_or_default();
    if !params.prior_count.is_finite() || params.prior_count < 0.0 {
        return Err(EdgeErrors::InvalidArgument(format!(
            "prior_count must be non-negative and finite, got {}",
            params.prior_count
        )));
    }
    if pair.0 == pair.1 {
        return Err(EdgeErrors::InvalidArgument(format!(
            "pair must name two different groups; got ({}, {})",
            pair.0, pair.1
        )));
    }

    let group = y.group.as_ref().ok_or_else(|| {
        EdgeErrors::InvalidArgument("the exact test needs a group per sample".to_string())
    })?;

    let owned;
    let dispersion = match dispersion {
        Some(d) => d,
        None => {
            owned = y
                .dispersion()
                .ok_or(EdgeErrors::InvalidDispersion(f64::NAN))?
                .0;
            &owned
        }
    };
    let per_gene = per_gene_dispersion(dispersion, y.n_genes)?;

    let columns1: Vec<usize> = (0..y.n_samples).filter(|&j| group[j] == pair.0).collect();
    let columns2: Vec<usize> = (0..y.n_samples).filter(|&j| group[j] == pair.1).collect();
    if columns1.is_empty() || columns2.is_empty() {
        return Err(EdgeErrors::InvalidArgument(format!(
            "both groups of the pair need at least one sample; got {} and {}",
            columns1.len(),
            columns2.len()
        )));
    }
    let (n1, n2) = (columns1.len(), columns2.len());

    // Effective library sizes for the pair, in group order.
    let effective = y.norm_lib_sizes();
    if let Some(bad) = effective.iter().find(|v| !v.is_finite() || **v <= 0.0) {
        return Err(EdgeErrors::InvalidArgument(format!(
            "library sizes must be positive and finite, found {bad}"
        )));
    }
    let lib1: Vec<f64> = columns1.iter().map(|&j| effective[j]).collect();
    let lib2: Vec<f64> = columns2.iter().map(|&j| effective[j]).collect();

    let n_pair = n1 + n2;
    let mean_lib = (lib1.iter().sum::<f64>() + lib2.iter().sum::<f64>()) / n_pair as f64;
    let ln_sum: f64 = lib1.iter().chain(lib2.iter()).map(|l| l.ln()).sum::<f64>();
    let lib_size_average = (ln_sum / n_pair as f64).exp();

    let y1 = select_columns(&y.counts, y.n_genes, y.n_samples, &columns1);
    let y2 = select_columns(&y.counts, y.n_genes, y.n_samples, &columns2);

    // Fold change: one intercept-only fit per group on prior-augmented counts,
    // with the offset raised to match so the prior does not shift the level.
    let abundance1 = augmented_abundance(
        &y1,
        y.n_genes,
        &lib1,
        mean_lib,
        params.prior_count,
        dispersion,
    )?;
    let abundance2 = augmented_abundance(
        &y2,
        y.n_genes,
        &lib2,
        mean_lib,
        params.prior_count,
        dispersion,
    )?;
    let log_fc: Vec<f64> = abundance1
        .iter()
        .zip(abundance2.iter())
        .map(|(a1, a2)| (a2 - a1) / LN_2)
        .collect();

    // Pseudo-counts: one pooled fit over both groups, then the quantile map onto
    // the geometric mean library size.
    let pooled = interleave(&y1, &y2, y.n_genes, n1, n2);
    let pooled_offset: Vec<f64> = lib1.iter().chain(lib2.iter()).map(|l| l.ln()).collect();
    let abundance = mglm_one_group(
        &pooled,
        y.n_genes,
        n_pair,
        dispersion,
        &Recycled::by_sample(pooled_offset),
        None,
        None,
        None,
    )?;

    let mut eq1 = vec![0.0_f64; y.n_genes * n1];
    let mut eq2 = vec![0.0_f64; y.n_genes * n2];
    let outcome: Result<(), EdgeErrors> = eq1
        .par_chunks_mut(n1)
        .zip(eq2.par_chunks_mut(n2))
        .enumerate()
        .try_for_each(|(gene, (row1, row2))| {
            let rate = abundance[gene].exp();
            let d = per_gene[gene];
            let output_mean = rate * lib_size_average;
            for (j, out) in row1.iter_mut().enumerate() {
                *out = q2q_one(y1[gene * n1 + j], rate * lib1[j], output_mean, d)?;
            }
            for (j, out) in row2.iter_mut().enumerate() {
                *out = q2q_one(y2[gene * n2 + j], rate * lib2[j], output_mean, d)?;
            }
            Ok(())
        });
    outcome?;

    let p_value = exact_pvalues(
        &eq1,
        &eq2,
        y.n_genes,
        n1,
        n2,
        &per_gene,
        params.rejection_region,
        params.big_count,
    )?;

    let log_cpm = match &y.ave_log_cpm {
        Some(v) => {
            if v.len() != y.n_genes {
                return Err(EdgeErrors::LengthMismatch {
                    name: "ave_log_cpm",
                    expected: y.n_genes,
                    got: v.len(),
                });
            }
            v.clone()
        }
        None => {
            // edgeR's aveLogCPM.DGEList: the *common* dispersion, not the one
            // the test itself runs on, and every sample rather than the pair.
            let alc_dispersion = y.common_dispersion.map(Recycled::scalar);
            ave_log_cpm(
                &y.counts,
                y.n_genes,
                y.n_samples,
                Some(&effective),
                None,
                AVE_LOG_CPM_PRIOR,
                alc_dispersion.as_ref(),
            )?
        }
    };

    Ok(ExactTestResult {
        log_fc,
        log_cpm,
        p_value,
    })
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

    /// Agreement target against edgeR.
    ///
    /// Far looser than what is achieved, and deliberately so: two of the three
    /// numbers here come out of an iterative fit, and pinning a test to the
    /// convergence tolerance of a routine in another module makes it a tripwire
    /// rather than a check.
    ///
    /// What is actually achieved, over every fixture in this module: every
    /// p-value, pseudo-count and quantile map holds to 1e-12 relative or better.
    /// The log fold changes hold to 2e-11, and that gap is
    /// [`mglm_one_group`]'s own `tol = 1e-10` on the coefficient step rather
    /// than anything here; edgeR stops on the same criterion and the two land a
    /// step apart.
    const TOL: f64 = 1e-9;

    /// 6 genes by 6 samples. Gene 4 is zero in the first group, gene 5 is above
    /// `big_count` in both, and gene 6 is flat.
    /// ```r
    /// y <- matrix(c(10,12,11,40,44,38, 50,48,52,49,51,50, 2,0,5,1,3,0,
    ///               0,0,0,7,9,8, 1200,1100,1300,2400,2500,2300, 5,7,6,5,6,7),
    ///             nrow = 6, byrow = TRUE)
    /// ls <- c(1e6,1.2e6,0.9e6,1.1e6,1e6,1.3e6)
    /// nf <- c(0.95,1.05,1.0,1.1,0.9,1.0)
    /// disp <- c(0.05,0.1,0.2,0.15,0.08,0.12)
    /// ```
    #[rustfmt::skip]
    const COUNTS: [f64; 36] = [
        10.0, 12.0, 11.0, 40.0, 44.0, 38.0,
        50.0, 48.0, 52.0, 49.0, 51.0, 50.0,
        2.0, 0.0, 5.0, 1.0, 3.0, 0.0,
        0.0, 0.0, 0.0, 7.0, 9.0, 8.0,
        1200.0, 1100.0, 1300.0, 2400.0, 2500.0, 2300.0,
        5.0, 7.0, 6.0, 5.0, 6.0, 7.0,
    ];

    const LIB: [f64; 6] = [1e6, 1.2e6, 0.9e6, 1.1e6, 1e6, 1.3e6];
    const NF: [f64; 6] = [0.95, 1.05, 1.0, 1.1, 0.9, 1.0];
    const DISP: [f64; 6] = [0.05, 0.1, 0.2, 0.15, 0.08, 0.12];

    /// Builds the fixture with the library sizes and factors R was given.
    ///
    /// ### Params
    ///
    /// * `group` - One label per sample
    ///
    /// ### Returns
    ///
    /// The container, matching `DGEList(counts=y, group=group, lib.size=ls,
    /// norm.factors=nf)`.
    fn fixture(group: Vec<usize>) -> DgeList<f64> {
        let mut y = DgeList::new(COUNTS.to_vec(), 6, 6, Some(group)).unwrap();
        y.lib_size = LIB.to_vec();
        y.norm_factors = NF.to_vec();
        y
    }

    /// Compares against a reference at [`TOL`].
    ///
    /// ### Params
    ///
    /// * `got` - Values produced here
    /// * `want` - Reference values from edgeR
    fn assert_close(got: &[f64], want: &[f64]) {
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want.iter()) {
            assert_relative_eq!(g, w, epsilon = 1e-12, max_relative = TOL);
        }
    }

    ////////////////
    // q2qnbinom  //
    ////////////////

    /// `x <- c(0,1,5,20,100,900,3,0.5)`
    /// `im <- c(2,2,10,10,50,500,3,1e-16)`
    /// `om <- c(4,4,20,20,25,800,3,1e-16)`
    const Q2Q_X: [f64; 8] = [0.0, 1.0, 5.0, 20.0, 100.0, 900.0, 3.0, 0.5];
    const Q2Q_IM: [f64; 8] = [2.0, 2.0, 10.0, 10.0, 50.0, 500.0, 3.0, 1e-16];
    const Q2Q_OM: [f64; 8] = [4.0, 4.0, 20.0, 20.0, 25.0, 800.0, 3.0, 1e-16];

    /// `cat(format(edgeR:::q2qnbinom(x, im, om, dispersion=0.1), digits=17), sep=", ")`
    #[test]
    fn test_q2q_nbinom_matches_edger_at_dispersion_one_tenth() {
        let want = [
            0.47247476834805369,
            2.48762930404220217,
            11.26359605754503690,
            37.16149924300670193,
            52.18843930258167774,
            1437.48127962309013128,
            3.00000000000000000,
            0.50000000000000011,
        ];
        let got = q2q_nbinom(&Q2Q_X, &Q2Q_IM, &Q2Q_OM, 0.1).unwrap();
        assert_close(&got, &want);
    }

    /// `cat(format(edgeR:::q2qnbinom(x, im, om, dispersion=0), digits=17), sep=", ")`
    #[test]
    fn test_q2q_nbinom_matches_edger_at_zero_dispersion() {
        let want = [
            0.58578643762690552,
            2.59345105979179325,
            12.74306177448739064,
            33.76355118458810267,
            61.97302934441376010,
            1295.83915428995442198,
            3.00000000000000000,
            0.50000000000000011,
        ];
        let got = q2q_nbinom(&Q2Q_X, &Q2Q_IM, &Q2Q_OM, 0.0).unwrap();
        assert_close(&got, &want);
    }

    /// `cat(format(edgeR:::q2qnbinom(x, im, om, dispersion=1.5), digits=17), sep=", ")`
    #[test]
    fn test_q2q_nbinom_matches_edger_at_a_large_dispersion() {
        let want = [
            0.12917130661302939,
            2.17053597295496514,
            10.19528650723413499,
            39.77185077921592438,
            50.12063954091983220,
            1439.89519676942063597,
            3.00000000000000044,
            0.49999999999999989,
        ];
        let got = q2q_nbinom(&Q2Q_X, &Q2Q_IM, &Q2Q_OM, 1.5).unwrap();
        assert_close(&got, &want);
    }

    /// The branch that needs a log-space upper incomplete gamma. At dispersion
    /// 0.001 the first element sits at `Q = e^-1605`, which is a flat zero on
    /// the natural scale and would send the inverse to infinity.
    ///
    /// `cat(format(edgeR:::q2qnbinom(c(6000,6000,2000), c(1000,200,40),`
    /// `c(1500,400,55), dispersion=0.001), digits=17), sep=", ")`
    #[test]
    fn test_q2q_nbinom_survives_the_far_upper_tail() {
        let x = [6000.0, 6000.0, 2000.0];
        let im = [1000.0, 200.0, 40.0];
        let om = [1500.0, 400.0, 55.0];

        let want = [8186.4210997511691, 8404.0259444343410, 2227.2071162950324];
        assert_close(&q2q_nbinom(&x, &im, &om, 0.001).unwrap(), &want);

        // `dispersion = 0.05`
        let want = [8968.9264289783969, 11625.9467408884921, 2572.2470886452957];
        assert_close(&q2q_nbinom(&x, &im, &om, 0.05).unwrap(), &want);
    }

    /// `cat(format(edgeR:::q2qnbinom(c(0,0,1), c(1000,200,40), c(1500,400,55),`
    /// `dispersion=0.001), digits=17), sep=", ")` and the `0.05` version.
    #[test]
    fn test_q2q_nbinom_matches_edger_in_the_far_lower_tail() {
        let x = [0.0, 0.0, 1.0];
        let im = [1000.0, 200.0, 40.0];
        let om = [1500.0, 400.0, 55.0];

        let want = [65.3468031185425104, 47.2474768348053402, 5.9130723310523443];
        assert_close(&q2q_nbinom(&x, &im, &om, 0.001).unwrap(), &want);

        let want = [2.4549984035979393, 4.5983158163212074, 2.8240360871062333];
        assert_close(&q2q_nbinom(&x, &im, &om, 0.05).unwrap(), &want);
    }

    /// Equal means must map a count onto itself.
    /// `cat(format(edgeR:::q2qnbinom(c(0,3,17,900), rep(5,4), rep(5,4),`
    /// `dispersion=0.1), digits=17), sep=", ")`
    #[test]
    fn test_q2q_nbinom_is_the_identity_when_the_means_agree() {
        let x = [0.0, 3.0, 17.0, 900.0];
        let means = [5.0_f64; 4];
        let got = q2q_nbinom(&x, &means, &means, 0.1).unwrap();
        // R reports 1.5365178734079724e-16 for the zero, which is its own gamma
        // round trip rounding rather than a real difference.
        for (g, w) in got.iter().zip(x.iter()) {
            assert_relative_eq!(g, w, epsilon = 1e-9, max_relative = TOL);
        }
    }

    /// A mean below `eps` nudges both sides by a quarter, so the mapping stays
    /// the identity rather than dividing by zero.
    #[test]
    fn test_q2q_nbinom_nudges_a_vanishing_mean() {
        let got = q2q_nbinom(&[0.5], &[1e-16], &[1e-16], 0.1).unwrap();
        assert_relative_eq!(got[0], 0.5, epsilon = 1e-12);
    }

    #[test]
    fn test_q2q_nbinom_rejects_bad_input() {
        assert!(matches!(
            q2q_nbinom(&[1.0], &[1.0, 2.0], &[1.0], 0.1),
            Err(EdgeErrors::LengthMismatch { .. })
        ));
        assert!(matches!(
            q2q_nbinom(&[1.0], &[1.0], &[1.0, 2.0], 0.1),
            Err(EdgeErrors::LengthMismatch { .. })
        ));
        assert!(matches!(
            q2q_nbinom(&[1.0], &[1.0], &[1.0], -0.1),
            Err(EdgeErrors::InvalidDispersion(_))
        ));
        assert!(matches!(
            q2q_nbinom(&[1.0], &[1.0], &[1.0], f64::NAN),
            Err(EdgeErrors::InvalidDispersion(_))
        ));
        assert!(matches!(
            q2q_nbinom(&[-1.0], &[1.0], &[1.0], 0.1),
            Err(EdgeErrors::InvalidArgument(_))
        ));
        assert!(matches!(
            q2q_nbinom(&[1.0], &[-1.0], &[1.0], 0.1),
            Err(EdgeErrors::InvalidArgument(_))
        ));
        assert!(matches!(
            q2q_nbinom(&[1.0], &[1.0], &[-1.0], 0.1),
            Err(EdgeErrors::InvalidArgument(_))
        ));
    }

    /// The log-space upper incomplete gamma against the natural-scale one it
    /// replaces, over the range where both are valid.
    #[test]
    fn test_ln_reg_gamma_upper_agrees_with_the_natural_scale() {
        for &(a, x) in &[
            (0.5_f64, 0.1_f64),
            (1.0, 0.5),
            (2.0, 3.0),
            (5.0, 5.0),
            (5.0, 40.0),
            (14.3, 285.0),
            (100.0, 120.0),
            (500.0, 600.0),
        ] {
            let direct = chisq_sf(2.0 * x, 2.0 * a).unwrap();
            let logged = ln_reg_gamma_upper(a, x).unwrap();
            assert_relative_eq!(logged.exp(), direct, max_relative = 1e-11);
        }
    }

    /// The inverse round-trips to the last few bits, including where the tail
    /// probability itself is unrepresentable.
    #[test]
    fn test_ln_reg_gamma_upper_inv_round_trips() {
        for &a in &[0.5_f64, 1.0, 3.0, 20.0, 500.0] {
            for &ln_q in &[-1e-3_f64, -0.7, -5.0, -50.0, -400.0, -1600.0] {
                let x = ln_reg_gamma_upper_inv(ln_q, a).unwrap();
                assert_relative_eq!(ln_reg_gamma_upper(a, x).unwrap(), ln_q, max_relative = 1e-9);
            }
        }
    }

    /////////////////////////
    // equalize_lib_sizes  //
    /////////////////////////

    /// `m <- matrix(c(10,12,11,40,44,38, 50,48,52,49,51,50, 2,0,5,1,3,0),`
    /// `nrow=3, byrow=TRUE)`
    /// `lsx <- c(1e6,1.2e6,0.9e6,1.1e6,1e6,1.3e6)`
    #[rustfmt::skip]
    const EQ_COUNTS: [f64; 18] = [
        10.0, 12.0, 11.0, 40.0, 44.0, 38.0,
        50.0, 48.0, 52.0, 49.0, 51.0, 50.0,
        2.0, 0.0, 5.0, 1.0, 3.0, 0.0,
    ];

    /// `eq <- equalizeLibSizes(m, group=c(1,1,1,2,2,2), dispersion=c(0.05,0.1,0.2),`
    /// `lib.size=lsx); cat(format(t(eq$pseudo.counts), digits=17), sep=", ")`
    #[test]
    fn test_equalize_lib_sizes_matches_edger_with_a_tagwise_dispersion() {
        let want = [
            10.77396968053937343,
            10.71756939170353817,
            13.06148638116767380,
            39.09381908827660368,
            47.20616010342799029,
            31.18877655466048182,
            53.75710417620476989,
            42.91780303131315577,
            61.99137485252995816,
            47.89006600316915296,
            54.79292199365470140,
            41.23952895466592850,
            2.16482265343957536,
            0.00000000000000000,
            5.76615714775634647,
            0.97296861050989225,
            3.17215790436234890,
            0.00000000000000000,
        ];
        let dispersion = Recycled::by_gene(vec![0.05, 0.1, 0.2]);
        let got = equalize_lib_sizes(
            &EQ_COUNTS,
            3,
            6,
            Some(&[1, 1, 1, 2, 2, 2]),
            &dispersion,
            Some(&LIB),
        )
        .unwrap();
        assert_close(&got, &want);

        // `cat(format(eq$pseudo.lib.size, digits=17))` -> 1075127.4874964589
        let common = (LIB.iter().map(|l| l.ln()).sum::<f64>() / 6.0).exp();
        assert_relative_eq!(common, 1075127.4874964589, max_relative = 1e-12);
    }

    /// `eq2 <- equalizeLibSizes(m, group=NULL, dispersion=0.05, lib.size=lsx)`
    #[test]
    fn test_equalize_lib_sizes_treats_a_missing_group_as_one_group() {
        let want = [
            10.96008842033811348,
            10.44646960127512614,
            13.59712013549795273,
            39.16595939215174837,
            46.92801915579765648,
            31.65690822951201966,
            53.72547236839000107,
            42.89019621190085019,
            61.80956593706329016,
            47.88264153989446470,
            54.78906141484367254,
            41.11794764355339282,
            2.14673596145325218,
            0.00000000000000000,
            5.61776154772958769,
            0.96792622133193928,
            3.18430182459055366,
            0.00000000000000000,
        ];
        let dispersion = Recycled::scalar(0.05);
        let got = equalize_lib_sizes(&EQ_COUNTS, 3, 6, None, &dispersion, Some(&LIB)).unwrap();
        assert_close(&got, &want);
    }

    /// `eq3 <- equalizeLibSizes(m, group=c(1,1,1,2,2,2), dispersion=0.05,`
    /// `lib.size=NULL)`. Without library sizes the column sums are tiny, the
    /// fitted means fall under `eps`, and the quarter-count nudge shows.
    #[test]
    fn test_equalize_lib_sizes_falls_back_to_the_column_sums() {
        let want = [
            12.37402438212434674,
            15.12339607660574714,
            12.37382629390799593,
            33.87607295258574425,
            34.23385182961500561,
            32.88971374206387566,
            61.46657328525719777,
            60.98805703840496051,
            58.33745565015052392,
            41.49462344661865387,
            39.58747634774626079,
            43.34659093904294025,
            2.50600579756254671,
            0.14125440085394836,
            5.46619301702612859,
            0.81569267802383960,
            2.47919721221450740,
            0.00000000000000000,
        ];
        let dispersion = Recycled::scalar(0.05);
        let got = equalize_lib_sizes(
            &EQ_COUNTS,
            3,
            6,
            Some(&[1, 1, 1, 2, 2, 2]),
            &dispersion,
            None,
        )
        .unwrap();
        assert_close(&got, &want);
    }

    #[test]
    fn test_equalize_lib_sizes_rejects_bad_input() {
        let d = Recycled::scalar(0.05);
        assert!(matches!(
            equalize_lib_sizes::<f64>(&[], 0, 6, None, &d, None),
            Err(EdgeErrors::EmptyCounts { .. })
        ));
        assert!(matches!(
            equalize_lib_sizes(&EQ_COUNTS[..12], 3, 6, None, &d, None),
            Err(EdgeErrors::LengthMismatch { name: "counts", .. })
        ));
        assert!(matches!(
            equalize_lib_sizes(&EQ_COUNTS, 3, 6, Some(&[1, 1, 2]), &d, None),
            Err(EdgeErrors::LengthMismatch { name: "group", .. })
        ));
        assert!(matches!(
            equalize_lib_sizes(&EQ_COUNTS, 3, 6, None, &d, Some(&[1e6, 1e6])),
            Err(EdgeErrors::LengthMismatch {
                name: "lib_size",
                ..
            })
        ));
        assert!(matches!(
            equalize_lib_sizes(
                &EQ_COUNTS,
                3,
                6,
                None,
                &d,
                Some(&[1e6, 0.0, 1e6, 1e6, 1e6, 1e6])
            ),
            Err(EdgeErrors::InvalidArgument(_))
        ));
        assert!(matches!(
            equalize_lib_sizes(&EQ_COUNTS, 3, 6, None, &Recycled::scalar(-0.1), None),
            Err(EdgeErrors::InvalidDispersion(_))
        ));
        assert!(matches!(
            equalize_lib_sizes(
                &EQ_COUNTS,
                3,
                6,
                None,
                &Recycled::by_gene(vec![0.1, 0.2]),
                None
            ),
            Err(EdgeErrors::LengthMismatch {
                name: "by_gene",
                ..
            })
        ));
        assert!(matches!(
            equalize_lib_sizes(
                &EQ_COUNTS,
                3,
                6,
                None,
                &Recycled::by_sample(vec![0.1; 6]),
                None
            ),
            Err(EdgeErrors::InvalidArgument(_))
        ));
    }

    ///////////////////////
    // Rejection regions //
    ///////////////////////

    /// Six genes split two against four, straight into the kernels with no
    /// pipeline in front.
    /// ```r
    /// y1 <- matrix(c(10,12, 50,48, 2,0, 0,0, 300,290, 5,7), nrow=6, byrow=TRUE)
    /// y2 <- matrix(c(40,44,38,41, 49,51,50,52, 1,3,0,2, 7,9,8,6,
    ///                100,110,105,99, 5,6,7,5), nrow=6, byrow=TRUE)
    /// disp <- c(0.05,0.1,0.2,0.15,0.08,0.12)
    /// ```
    #[rustfmt::skip]
    const K_Y1: [f64; 12] = [
        10.0, 12.0, 50.0, 48.0, 2.0, 0.0, 0.0, 0.0, 300.0, 290.0, 5.0, 7.0,
    ];
    #[rustfmt::skip]
    const K_Y2: [f64; 24] = [
        40.0, 44.0, 38.0, 41.0,
        49.0, 51.0, 50.0, 52.0,
        1.0, 3.0, 0.0, 2.0,
        7.0, 9.0, 8.0, 6.0,
        100.0, 110.0, 105.0, 99.0,
        5.0, 6.0, 7.0, 5.0,
    ];

    /// `cat(format(edgeR:::exactTestDoubleTail(y1, y2, dispersion=disp), digits=17), sep=", ")`
    #[test]
    fn test_double_tail_matches_edger_on_unequal_groups() {
        let want = [
            1.0877946590376498e-05,
            9.6474841698552705e-01,
            9.6034839393473215e-01,
            3.2619499340000606e-04,
            1.8490293273658116e-05,
            1.0000000000000000e+00,
        ];
        let got = exact_pvalues(
            &K_Y1,
            &K_Y2,
            6,
            2,
            4,
            &DISP,
            RejectionRegion::DoubleTail,
            BIG_COUNT_DEFAULT,
        )
        .unwrap();
        assert_close(&got, &want);
    }

    /// `cat(format(edgeR:::exactTestByDeviance(y1, y2, dispersion=disp), digits=17), sep=", ")`
    #[test]
    fn test_deviance_matches_edger_on_unequal_groups() {
        let want = [
            8.1344278851021571e-06,
            9.2093279234031145e-01,
            7.5779954527582549e-01,
            1.8059270195564577e-04,
            2.0166440897726341e-05,
            1.0000000000000000e+00,
        ];
        let got = exact_pvalues(
            &K_Y1,
            &K_Y2,
            6,
            2,
            4,
            &DISP,
            RejectionRegion::Deviance,
            BIG_COUNT_DEFAULT,
        )
        .unwrap();
        assert_close(&got, &want);
    }

    /// edgeR's `exactTestBySmallP` ends on `min(pvals, 1)` rather than `pmin`,
    /// so a whole matrix comes back carrying the smallest p-value in it. The
    /// reference here is that function called one gene at a time:
    /// `for (i in 1:6) edgeR:::exactTestBySmallP(y1[i,,drop=FALSE],`
    /// `y2[i,,drop=FALSE], dispersion=disp[i])`
    #[test]
    fn test_small_p_matches_edger_called_one_gene_at_a_time() {
        let want = [
            1.1812148275602017e-05,
            9.9999999999999922e-01,
            1.0000000000000000e+00,
            3.7544809480160240e-04,
            1.4354611746487754e-05,
            8.9101578715497332e-01,
        ];
        let got = exact_pvalues(
            &K_Y1,
            &K_Y2,
            6,
            2,
            4,
            &DISP,
            RejectionRegion::SmallP,
            BIG_COUNT_DEFAULT,
        )
        .unwrap();
        assert_close(&got, &want);
    }

    /// A common dispersion rather than one per gene.
    /// `cat(format(edgeR:::exactTestDoubleTail(y1, y2, dispersion=0.1), digits=17), sep=", ")`
    /// and the same for `exactTestByDeviance`.
    #[test]
    fn test_kernels_match_edger_at_a_common_dispersion() {
        let common = [0.1_f64; 6];
        let want_dt = [
            0.00043066236852219725,
            0.96474841698552704639,
            0.94916961181158332472,
            0.00014095556701738610,
            0.00011337582143935078,
            1.00000000000000000000,
        ];
        let got = exact_pvalues(
            &K_Y1,
            &K_Y2,
            6,
            2,
            4,
            &common,
            RejectionRegion::DoubleTail,
            BIG_COUNT_DEFAULT,
        )
        .unwrap();
        assert_close(&got, &want_dt);

        let want_dev = [
            0.00033859092769447587,
            0.92093279234031144576,
            0.74358178328280177816,
            0.00007467786708842068,
            0.00012974328406972557,
            0.99999999999999977796,
        ];
        let got = exact_pvalues(
            &K_Y1,
            &K_Y2,
            6,
            2,
            4,
            &common,
            RejectionRegion::Deviance,
            BIG_COUNT_DEFAULT,
        )
        .unwrap();
        assert_close(&got, &want_dev);
    }

    /// With equal group sizes all three regions coincide, and edgeR routes the
    /// other two to the double-tail test.
    /// ```r
    /// z1 <- matrix(c(10,12,11, 2,0,5, 300,290,310), nrow=3, byrow=TRUE)
    /// z2 <- matrix(c(40,44,38, 1,3,0, 100,110,105), nrow=3, byrow=TRUE)
    /// cat(format(edgeR:::exactTestDoubleTail(z1, z2, dispersion=c(0.05,0.2,0.08)), digits=17))
    /// ```
    #[test]
    fn test_regions_coincide_when_the_groups_are_the_same_size() {
        #[rustfmt::skip]
        let z1 = [10.0, 12.0, 11.0, 2.0, 0.0, 5.0, 300.0, 290.0, 310.0];
        #[rustfmt::skip]
        let z2 = [40.0, 44.0, 38.0, 1.0, 3.0, 0.0, 100.0, 110.0, 105.0];
        let d = [0.05, 0.2, 0.08];
        let want = [
            8.2785897767219603e-07,
            6.1008651331231956e-01,
            1.8992725372229388e-05,
        ];
        for region in [
            RejectionRegion::DoubleTail,
            RejectionRegion::Deviance,
            RejectionRegion::SmallP,
        ] {
            let got = exact_pvalues(&z1, &z2, 3, 3, 3, &d, region, BIG_COUNT_DEFAULT).unwrap();
            assert_close(&got, &want);
        }
    }

    /// The beta approximation, both sums well past `big_count`.
    /// ```r
    /// b1 <- matrix(c(1200,1100,1300, 2,3,4, 950,960,970), nrow=3, byrow=TRUE)
    /// b2 <- matrix(c(2400,2500,2300, 950,960,970, 950,960,970), nrow=3, byrow=TRUE)
    /// cat(format(edgeR:::exactTestBetaApprox(b1, b2, c(0.08,0.05,0.1)), digits=17))
    /// ```
    #[test]
    fn test_beta_approximation_matches_edger() {
        let want = [
            3.1654679012061791e-03,
            2.7141288810736955e-110,
            1.0000000000000000e+00,
        ];
        let got = [
            beta_approx_one(3600.0, 7200.0, 3, 3, 0.08).unwrap(),
            beta_approx_one(9.0, 2880.0, 3, 3, 0.05).unwrap(),
            beta_approx_one(2880.0, 2880.0, 3, 3, 0.1).unwrap(),
        ];
        assert_close(&got, &want);
    }

    /// Above `big_count` the double-tail test hands over to the beta
    /// approximation, and the p-value moves. Gene 2 of the pipeline fixture is
    /// the one that changes when the threshold is dropped to 100.
    #[test]
    fn test_big_count_threshold_switches_paths() {
        let y1 = [1200.0, 1100.0, 1300.0];
        let y2 = [2400.0, 2500.0, 2300.0];
        let mut buf = Vec::new();
        let exact = double_tail_one(&y1, &y2, 0.08, 1e12, &mut buf).unwrap();
        let approx = double_tail_one(&y1, &y2, 0.08, BIG_COUNT_DEFAULT, &mut buf).unwrap();
        assert_relative_eq!(approx, 3.1654679012061791e-03, max_relative = TOL);
        // The convolution and the approximation agree to about three digits,
        // which is the whole reason edgeR is willing to make the swap.
        assert_relative_eq!(exact, approx, max_relative = 1e-2);
        assert!(exact != approx);
    }

    /// At zero dispersion every region routes through the exact binomial test.
    /// `cat(format(edgeR:::exactTestDoubleTail(y1, y2, dispersion=0), digits=17), sep=", ")`
    ///
    /// Gene 6 is the one exception, and it is edgeR that is wrong. Its split is
    /// 12 against 23 at `p = 1/3`, and a total of 35 makes the outcomes 11 and
    /// 12 *exactly* equiprobable, so every outcome is at most as probable as the
    /// observed one and the p-value is 1. edgeR reports 0.86009048828607226,
    /// having dropped one of the two tied terms on a one-ulp difference in
    /// `dbinom`. R's own `binom.test(12, 35, 1/3)` returns 1, and so does this.
    #[test]
    fn test_zero_dispersion_uses_the_binomial_test() {
        let want = [
            2.3227510676278390e-11,
            8.5428783017505106e-01,
            1.0000000000000000e+00,
            6.6897430840626251e-06,
            8.1441784505498174e-61,
            1.0000000000000000e+00,
        ];
        let zero = [0.0_f64; 6];
        let got = exact_pvalues(
            &K_Y1,
            &K_Y2,
            6,
            2,
            4,
            &zero,
            RejectionRegion::DoubleTail,
            BIG_COUNT_DEFAULT,
        )
        .unwrap();
        assert_close(&got, &want);
    }

    /// `cat(format(binomTest(c(0,3,10,50,300), c(0,5,10,60,290), p=0.5), digits=17))`
    /// and the same at `p = 1/3` and `p = 0.25`.
    #[test]
    fn test_binom_test_matches_edger() {
        let y1 = [0.0, 3.0, 10.0, 50.0, 300.0];
        let y2 = [0.0, 5.0, 10.0, 60.0, 290.0];

        let want_half = [
            1.00000000000000000,
            0.72656250000000033,
            1.00000000000000000,
            0.39092745669950468,
            0.71102572604547820,
        ];
        for (i, w) in want_half.iter().enumerate() {
            assert_relative_eq!(
                binom_test_one(y1[i], y2[i], 0.5).unwrap(),
                w,
                max_relative = TOL
            );
        }

        // Index 1 is a 3 against 5 split at p = 1/3, where a total of 8 makes
        // the outcomes 2 and 3 exactly equiprobable. edgeR reports
        // 0.72687090382563635, having dropped the tied term; `binom.test(3, 8,
        // 1/3)` reports 1, and so does this. Every other entry agrees with
        // edgeR to the tolerance.
        let want_third = [
            1.0000000000000000e+00,
            1.0000000000000000e+00,
            1.5234223511142742e-01,
            8.4044576876146256e-03,
            2.2012099642833954e-18,
        ];
        for (i, w) in want_third.iter().enumerate() {
            assert_relative_eq!(
                binom_test_one(y1[i], y2[i], 1.0 / 3.0).unwrap(),
                w,
                max_relative = TOL
            );
        }

        // `binomTest(c(2,7,40), c(6,3,60), p=0.25)`
        let want_quarter = [
            1.0000000000000000000,
            0.0035057067871093776,
            0.0010801099946617424,
        ];
        let a = [2.0, 7.0, 40.0];
        let b = [6.0, 3.0, 60.0];
        for (i, w) in want_quarter.iter().enumerate() {
            assert_relative_eq!(
                binom_test_one(a[i], b[i], 0.25).unwrap(),
                w,
                max_relative = TOL
            );
        }
    }

    ////////////////
    // exact_test //
    ////////////////

    /// `cat(format(exactTest(d, dispersion=disp)$table$logCPM, digits=17), sep=", ")`
    const A_LOG_CPM: [f64; 6] = [
        4.6828570480490503,
        5.6094058355452656,
        1.8369026364109835,
        2.4631896748603697,
        10.7199768972231873,
        2.8848509178259678,
    ];

    /// The whole pipeline against `exactTest`, equal groups.
    /// ```r
    /// d <- DGEList(counts=y, group=c(1,1,1,2,2,2), lib.size=ls, norm.factors=nf)
    /// r <- exactTest(d, dispersion=disp)
    /// cat(format(r$table$logFC, digits=17), sep=", ")
    /// ```
    #[test]
    fn test_exact_test_matches_edger_on_equal_groups() {
        let want_fc = [
            1.76837964949810367,
            -0.13190914581416660,
            -0.89031602850510549,
            5.98706174338887909,
            0.86708062515865070,
            -0.12249104113230144,
        ];
        let want_p = [
            3.4349237412116053e-06,
            7.6299000470704081e-01,
            4.6743523769487572e-01,
            1.4972774802797630e-05,
            1.0337909917591625e-02,
            8.9951524706410912e-01,
        ];

        let y = fixture(vec![1, 1, 1, 2, 2, 2]);
        let dispersion = Recycled::by_gene(DISP.to_vec());
        for region in [
            RejectionRegion::DoubleTail,
            RejectionRegion::Deviance,
            RejectionRegion::SmallP,
        ] {
            let params = ExactTestParams::new(region, BIG_COUNT_DEFAULT, PRIOR_COUNT_DEFAULT);
            let got = exact_test(&y, (1, 2), Some(&dispersion), Some(params)).unwrap();
            assert_close(&got.log_fc, &want_fc);
            assert_close(&got.log_cpm, &A_LOG_CPM);
            assert_close(&got.p_value, &want_p);
        }
    }

    /// Unequal groups, where the three regions genuinely differ.
    /// `d <- DGEList(counts=y, group=c(1,1,2,2,2,2), lib.size=ls, norm.factors=nf)`
    #[test]
    fn test_exact_test_matches_edger_on_unequal_groups() {
        let want_fc = [
            1.617329610273079199,
            0.089998354795996716,
            1.144582075236848073,
            5.616562904007275314,
            0.900674912831392449,
            0.052897540463604717,
        ];
        let y = fixture(vec![1, 1, 2, 2, 2, 2]);
        let dispersion = Recycled::by_gene(DISP.to_vec());

        // `exactTest(d, dispersion=disp, rejection.region="doubletail")`
        let want_dt = [
            0.00017528190903009128,
            0.87366769045812053829,
            0.45299967984928735110,
            0.00158610472572541451,
            0.01631959757549012249,
            1.00000000000000000000,
        ];
        let got = exact_test(&y, (1, 2), Some(&dispersion), None).unwrap();
        assert_close(&got.log_fc, &want_fc);
        assert_close(&got.log_cpm, &A_LOG_CPM);
        assert_close(&got.p_value, &want_dt);

        // `rejection.region="deviance"`
        let want_dev = [
            0.00014592570324165402,
            0.84420783731055693000,
            0.32771177133536538717,
            0.00089863870338400599,
            0.01523232012423183027,
            1.00000000000000000000,
        ];
        let params = ExactTestParams::new(
            RejectionRegion::Deviance,
            BIG_COUNT_DEFAULT,
            PRIOR_COUNT_DEFAULT,
        );
        let got = exact_test(&y, (1, 2), Some(&dispersion), Some(params)).unwrap();
        assert_close(&got.p_value, &want_dev);
    }

    /// edgeR's `exactTestBySmallP` cannot be read straight off `exactTest` for
    /// more than one gene, per its `min` bug, so this checks the region does
    /// something different from the other two and stays a probability.
    #[test]
    fn test_exact_test_small_p_region_runs_on_unequal_groups() {
        let y = fixture(vec![1, 1, 2, 2, 2, 2]);
        let dispersion = Recycled::by_gene(DISP.to_vec());
        let params = ExactTestParams::new(
            RejectionRegion::SmallP,
            BIG_COUNT_DEFAULT,
            PRIOR_COUNT_DEFAULT,
        );
        let got = exact_test(&y, (1, 2), Some(&dispersion), Some(params)).unwrap();
        assert!(got.p_value.iter().all(|p| (0.0..=1.0).contains(p)));
        let double = exact_test(&y, (1, 2), Some(&dispersion), None).unwrap();
        assert!(got.p_value[0] != double.p_value[0]);
    }

    /// Lowering `big_count` moves gene 2 onto the beta approximation, and
    /// nothing else.
    /// `cat(format(exactTest(d, dispersion=disp, big.count=100)$table$PValue, digits=17))`
    #[test]
    fn test_exact_test_honours_a_lowered_big_count() {
        let want = [
            3.4349237412116053e-06,
            7.6605971775209569e-01,
            4.6743523769487572e-01,
            1.4972774802797630e-05,
            1.0337909917591625e-02,
            8.9951524706410912e-01,
        ];
        let y = fixture(vec![1, 1, 1, 2, 2, 2]);
        let dispersion = Recycled::by_gene(DISP.to_vec());
        let params = ExactTestParams::new(RejectionRegion::DoubleTail, 100.0, PRIOR_COUNT_DEFAULT);
        let got = exact_test(&y, (1, 2), Some(&dispersion), Some(params)).unwrap();
        assert_close(&got.p_value, &want);
    }

    /// The prior count only moves the fold change, and it moves gene 4 most,
    /// which is the one that is zero in the first group.
    /// `cat(format(exactTest(d, dispersion=disp, prior.count=0.5)$table$logFC, digits=17))`
    #[test]
    fn test_exact_test_honours_the_prior_count() {
        let want = [
            1.73637993337830987,
            -0.13094130138685259,
            -0.74393133794193356,
            4.05433589435307962,
            0.86689211695718937,
            -0.11516957613998995,
        ];
        let y = fixture(vec![1, 1, 1, 2, 2, 2]);
        let dispersion = Recycled::by_gene(DISP.to_vec());
        let params = ExactTestParams::new(RejectionRegion::DoubleTail, BIG_COUNT_DEFAULT, 0.5);
        let got = exact_test(&y, (1, 2), Some(&dispersion), Some(params)).unwrap();
        assert_close(&got.log_fc, &want);
    }

    /// Reversing the pair flips the sign of the fold change and leaves the
    /// p-value alone.
    #[test]
    fn test_exact_test_pair_order_flips_the_fold_change() {
        let y = fixture(vec![1, 1, 1, 2, 2, 2]);
        let dispersion = Recycled::by_gene(DISP.to_vec());
        let forward = exact_test(&y, (1, 2), Some(&dispersion), None).unwrap();
        let reverse = exact_test(&y, (2, 1), Some(&dispersion), None).unwrap();
        for (f, r) in forward.log_fc.iter().zip(reverse.log_fc.iter()) {
            assert_relative_eq!(*f, -r, max_relative = 1e-10);
        }
        assert_close(&forward.p_value, &reverse.p_value);
    }

    /// A dispersion carried on the container is used when none is passed.
    #[test]
    fn test_exact_test_falls_back_to_the_container_dispersion() {
        let mut y = fixture(vec![1, 1, 1, 2, 2, 2]);
        y.tagwise_dispersion = Some(DISP.to_vec());
        let got = exact_test(&y, (1, 2), None, None).unwrap();
        let dispersion = Recycled::by_gene(DISP.to_vec());
        let want = exact_test(&y, (1, 2), Some(&dispersion), None).unwrap();
        assert_close(&got.p_value, &want.p_value);
    }

    /// `f32` counts must land on the same answer to single precision.
    #[test]
    fn test_exact_test_is_generic_over_the_count_type() {
        let counts32: Vec<f32> = COUNTS.iter().map(|v| *v as f32).collect();
        let mut y32 = DgeList::new(counts32, 6, 6, Some(vec![1, 1, 1, 2, 2, 2])).unwrap();
        y32.lib_size = LIB.to_vec();
        y32.norm_factors = NF.to_vec();

        let dispersion = Recycled::by_gene(DISP.to_vec());
        let got = exact_test(&y32, (1, 2), Some(&dispersion), None).unwrap();
        let want = exact_test(
            &fixture(vec![1, 1, 1, 2, 2, 2]),
            (1, 2),
            Some(&dispersion),
            None,
        )
        .unwrap();
        for (g, w) in got.p_value.iter().zip(want.p_value.iter()) {
            assert_relative_eq!(g, w, epsilon = 1e-6, max_relative = 1e-5);
        }
    }

    #[test]
    fn test_exact_test_rejects_bad_input() {
        let dispersion = Recycled::by_gene(DISP.to_vec());

        // No group on the container.
        let mut ungrouped = fixture(vec![1, 1, 1, 2, 2, 2]);
        ungrouped.group = None;
        assert!(matches!(
            exact_test(&ungrouped, (1, 2), Some(&dispersion), None),
            Err(EdgeErrors::InvalidArgument(_))
        ));

        let y = fixture(vec![1, 1, 1, 2, 2, 2]);
        // A pair naming the same group twice.
        assert!(matches!(
            exact_test(&y, (1, 1), Some(&dispersion), None),
            Err(EdgeErrors::InvalidArgument(_))
        ));
        // A group with no samples.
        assert!(matches!(
            exact_test(&y, (1, 7), Some(&dispersion), None),
            Err(EdgeErrors::InvalidArgument(_))
        ));
        // No dispersion anywhere.
        assert!(matches!(
            exact_test(&y, (1, 2), None, None),
            Err(EdgeErrors::InvalidDispersion(_))
        ));
        // A negative dispersion.
        assert!(matches!(
            exact_test(&y, (1, 2), Some(&Recycled::scalar(-0.1)), None),
            Err(EdgeErrors::InvalidDispersion(_))
        ));
        // A dispersion of the wrong length.
        assert!(matches!(
            exact_test(&y, (1, 2), Some(&Recycled::by_gene(vec![0.1, 0.2])), None),
            Err(EdgeErrors::LengthMismatch { .. })
        ));
        // A negative prior count.
        let params = ExactTestParams::new(RejectionRegion::DoubleTail, BIG_COUNT_DEFAULT, -1.0);
        assert!(matches!(
            exact_test(&y, (1, 2), Some(&dispersion), Some(params)),
            Err(EdgeErrors::InvalidArgument(_))
        ));
        // A zero library size.
        let mut zero_lib = fixture(vec![1, 1, 1, 2, 2, 2]);
        zero_lib.lib_size[0] = 0.0;
        assert!(matches!(
            exact_test(&zero_lib, (1, 2), Some(&dispersion), None),
            Err(EdgeErrors::InvalidArgument(_))
        ));
        // A pre-set log-CPM of the wrong length.
        let mut bad_alc = fixture(vec![1, 1, 1, 2, 2, 2]);
        bad_alc.ave_log_cpm = Some(vec![1.0, 2.0]);
        assert!(matches!(
            exact_test(&bad_alc, (1, 2), Some(&dispersion), None),
            Err(EdgeErrors::LengthMismatch {
                name: "ave_log_cpm",
                ..
            })
        ));
    }

    #[test]
    fn test_log_sum_exp_handles_an_empty_and_an_all_infinite_input() {
        assert_eq!(log_sum_exp(&[]), f64::NEG_INFINITY);
        assert_eq!(
            log_sum_exp(&[f64::NEG_INFINITY, f64::NEG_INFINITY]),
            f64::NEG_INFINITY
        );
        assert_relative_eq!(log_sum_exp(&[0.0, 0.0]), LN_2, max_relative = 1e-15);
    }
}
