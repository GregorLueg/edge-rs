# Upstream deviations

Where `edge-rs` deliberately does not reproduce its upstreams, and why.

- **Section A: edgePython is wrong.** The port follows
  [`edgePython`](https://github.com/pachterlab/edgePython), but where it
  disagrees with edgeR, limma or `nebula`, upstream wins. Those are what users
  compare against and what the fixtures gate on.
- **Section B: edgeR or limma are wrong.** Rarer. Each fault is reproduced in
  the installed package. Copying a bug faithfully is not parity worth having.

Every entry was checked against the installed package (edgeR 4.8.2, limma
3.66.0, nebula 1.5.8), not inferred from reading, and names a test that fails if
the behaviour drifts back.

edgePython citations are against **version 0.2.6, commit `1e572ae`**
(2026-06-16). A line number without a commit rots, and any of these may have
been fixed upstream since. Re-pin this paragraph if you re-check the entries
against a newer checkout.

## Index

| ID | Summary | Severity | Ported to |
|---|---|---|---|
| [A1](#a1-tmmwsp-trims-one-gene-too-many-at-each-end) | TMMwsp trim window off by one | high | `core/normalisation.rs` |
| [A2](#a2-filterbyexpr-leverages-omit-the-intercept) | `filterByExpr` leverages omit the intercept | medium | `core/filtering.rs` |
| [A3](#a3-the-glm-path-uses-a-naive-unit-deviance) | GLM path uses the naive unit deviance | medium | `glm/deviance.rs` |
| [A4](#a4-the-integer-division-in-the-poisson-regime-is-corrected) | Integer `2/3` in edgeR's C "corrected" | low | `glm/deviance.rs` |
| [A5](#a5-add_prior_count-drops-the-library-scaling) | `add_prior_count` drops library scaling | low | `glm/fit.rs` |
| [A6](#a6-natural_spline_basis-is-not-rs-ns) | `natural_spline_basis` is not R's `ns()` | none | `numeric/interpolate.rs` |
| [A7](#a7-locfitbycol-is-a-per-point-fit-not-locfit) | `locfitByCol` is not locfit | high | `limma/smoothing.rs` |
| [A8](#a8-chooselowessspan-has-the-wrong-defaults) | `chooseLowessSpan` defaults wrong | medium | `utils/design.rs` |
| [A9](#a9-trigamma_inverse-uses-the-wrong-large-x-asymptotic) | `trigamma_inverse` large-x asymptotic | low | `numeric/gamma.rs` |
| [A10](#a10-squeezevar-and-fitfdist-clamp-and-use-the-wrong-smoother) | `fitFDist` clamps and wrong smoother | medium | `limma/squeeze_var.rs` |
| [A11](#a11-compute_prior-uses-the-wrong-smoother) | QL prior uses the wrong smoother | medium | `ql/weights.rs` |
| [A12](#a12-lmfit-loses-estimability-df-and-stdev_unscaled) | `lmFit` loses estimability and df | high | `limma/lm_fit.rs` |
| [A13](#a13-voom-uses-one-smoother-where-upstream-uses-two) | `voom` uses one smoother, not two | medium | `limma/voom.rs` |
| [A14](#a14-duplicatecorrelation-and-arrayweights-are-different-estimators) | `duplicateCorrelation` is a different estimator | high | `limma/array_weights.rs` |
| [A15](#a15-the-deviance-and-small-p-rejection-regions-are-stubs) | Exact test regions are stubs | high | `exact/mod.rs` |
| [A16](#a16-q2qnbinom-leaves-log-space) | `q2qnbinom` leaves log space | medium | `exact/mod.rs` |
| [A17](#a17-splicevariants-is-a-different-test) | `spliceVariants` is a different test | high | `splicing.rs` |
| [A18](#a18-diffsplicedge-is-not-edgers-procedure) | `diffSpliceDGE` is not edgeR's procedure | high | `splicing.rs` |
| [A19](#a19-glm_sc_test-drops-the-off-diagonal-covariance) | Single-cell contrasts drop covariance | high | `sc/test.rs` |
| [A20](#a20-nebula-has-no-marginal-likelihood-hessian) | NEBULA has no marginal Hessian | highest | `sc/ptmg.rs`, `sc/nebula.rs` |
| [A21](#a21-_opt_pml_nb-adds-a-ridge-a-floor-and-a-clamp) | `_opt_pml_nb` ridge, floor and clamp | high | `sc/pml.rs` |
| [B1](#b1-avelogcpm-corrupts-the-first-gene-under-a-matrix-offset) | `aveLogCPM` matrix offset corrupts gene 1 | high | `core/expression.rs` |
| [B2](#b2-fitfdistrobustly-stops-its-root-solve-early) | `fitFDistRobustly` stops `uniroot` early | low | `limma/squeeze_var.rs` |
| [B3](#b3-exacttestbysmallp-returns-one-p-value-for-every-gene) | `exactTestBySmallP` uses `min` for `pmin` | severe | `exact/mod.rs` |
| [B4](#b4-binomtest-mishandles-exact-ties) | `binomTest` mishandles exact ties | medium | `exact/mod.rs` |

---

# Section A: edgePython disagrees with upstream

## Normalisation and filtering

### A1. TMMwsp trims one gene too many at each end

- **Upstream:** `edgepython/normalization.py:299` (`_calc_factor_tmmwsp`)
- **Ported to:** `src/core/normalisation.rs`, `calc_factor_tmmwsp`
- **Severity:** high. Factors off by up to 11%, and every fold change with them.
- **Test:** `test_part1_tmmwsp_matches_r`, `test_part1_tmmwsp_variants_match_r`

R keeps positions `[loM, n + 1 - loM]` inclusive. The Python computes
`hiM = n - loM` and slices `o_M[loM:hiM]`, which in 1-based terms is
`[loM + 1, n - loM]`: one element short at each end. `_calc_factor_tmm` right
above it is correct, so this is a transcription slip.

On a 10 by 3 matrix with the zeros TMMwsp exists for:

| method | edgePython | edgeR 4.8.2 |
|---|---|---|
| TMM | 0.883046965512903, 1.059296727816462, 1.069051351250571 | identical to 15 figures |
| TMMwsp | 0.892298677901486, 1.070758413481783, 1.046642232997959 | 0.831890371585877, 1.193584170243712, 1.007119145059728 |

TMM agrees exactly, which isolates the fault to the TMMwsp trim. edgePython's
own tests miss it: they compare at `1e-3` on large matrices, where two genes out
of tens of thousands barely move the trimmed mean.

`edge-rs` follows the R.

### A2. `filterByExpr` leverages omit the intercept

- **Upstream:** `edgepython/filtering.py:93` (`_hat_values`), against edgeR's `filterByExpr.default`
- **Ported to:** `src/core/filtering.rs`, `design_leverage`
- **Severity:** wrong minimum group size, hence wrong CPM cutoff, on any design without an intercept in its span
- **Test:** `test_filter_by_a_design_without_an_intercept_matches_edger`

edgeR takes the minimum group size as `1 / max(stats::hat(design))`.
`stats::hat` defaults to `intercept = TRUE` and prepends a column of ones before
the QR. `_hat_values` runs a plain QR on the design as given.

For `design = cbind(c(0.1, 0.2, 0.3, 0.4, 0.5, 0.9))`, edgeR's maximum leverage
is 0.7917 (group size 1.2632). Without the extra column it is 0.5956 (1.679).
Different cutoff, different gene list. The two agree whenever the intercept is
already in the span, which is the common case.

`edge-rs` follows edgeR. `[1 | X]` is rank deficient exactly when the intercept
is already in the span, so the rank of the augmented matrix is checked first:
the QR has no rank truncation.

## GLM and deviance

### A3. The GLM path uses a naive unit deviance

- **Upstream:** `edgepython/glm_levenberg.py:310` (`_unit_nb_deviance`) and `nbinom_deviance` at line 224
- **Ported to:** `src/glm/deviance.rs`
- **Severity:** deviances wrong in the eighth figure, much worse near `y == mu`
- **Test:** `test_regime_switch_beats_the_naive_form_near_the_mean`

edgePython carries two unit deviances. `ql_weights.py:500`
(`compute_unit_nb_deviance`) is a faithful port of edgeR's `compute_nbdev.c`.
`glm_levenberg.py:310` is the textbook formula, and it is the one the Levenberg
fit and `nbinom_deviance` use.

edgeR does two things the textbook formula does not:

1. Adds `mildly_low_value = 1e-8` to both `y` and `mu`.
2. Switches by regime: a Poisson expansion for `phi < 1e-4`, a gamma limit for
   `mu * phi > 1e6`, otherwise the rearranged exact form
   `2*(y*log(y/mu) + (y + 1/phi)*log((mu + 1/phi)/(y + 1/phi)))`, which does
   not cancel.

`nbinomUnitDeviance` in edgeR 4.8.2 against the naive form at `mu = 10`,
`phi = 0.1`:

| y | edgeR | naive | relative difference |
|---|---|---|---|
| 12 | 0.182069451405 | 0.1820694517 | 1.4e-9 |
| 3 | 3.97651898398 | 3.976518992 | 2.1e-9 |
| 0 | 13.8629432006 | 13.86294361 | 3.0e-8 |
| 10.001 | 4.99974996685e-08 | 4.99975045900e-08 | 9.8e-8 |
| 10 | 0 | 5.0e-16 | total |

The nudge alone recovers edgeR to 1e-13 except next to `y == mu`, where the
regime switch carries the rest. Both are needed.

`edge-rs` has one implementation, edgeR's, in `src/glm/deviance.rs`, used by the
Levenberg fit, the residual deviance and the QL weights alike.

### A4. The integer division in the Poisson regime is "corrected"

- **Upstream:** `edgepython/ql_weights.py:514` (`compute_unit_nb_deviance`)
- **Ported to:** `src/glm/deviance.rs`, `unit_nb_deviance`
- **Severity:** up to 7e-4 relative on large counts at small dispersion
- **Test:** `test_unit_deviance_matches_edger` (`src/ql/chebyshev.rs`)

The Poisson-regime correction in edgeR's C reads

```c
2 * (y * log(y/mu) - resid - 0.5*resid*resid*phi*(1 + phi*(2/3*resid - y)))
```

`2/3` is integer division, so it is zero and the term is `-phi*y`. edgePython
writes `2.0 / 3.0`. Intended or not, the C is what every published edgeR result
reflects. Against `nbinomUnitDeviance`:

| y | mu | phi | edgeR | integer `2/3` | `2.0/3.0` |
|---|---|---|---|---|---|
| 1e6 | 1.001e6 | 1e-5 | 90.999333833 | 90.999333833 | 91.0660004996 |
| 1e6 | 1e6 + 1 | 1e-6 | 9.99899558621e-07 | 9.99899558707e-07 | 9.99900225374e-07 |
| 100 | 110 | 1e-5 | 0.936965039047 | 0.936965039047 | 0.936965105714 |
| 0 | 10 | 1e-5 | 19.9989995855 | 19.9989995855 | 19.9989996522 |

Only visible when `phi` is under the 1e-4 threshold *and* counts are large. A
test at `phi = 1e-10` passes either way. `edge-rs` reproduces the C.

### A5. `add_prior_count` drops the library scaling

- **Upstream:** `edgepython/utils.py:49` (`add_prior_count`)
- **Ported to:** `src/glm/fit.rs`, `add_prior_count`
- **Severity:** low, around 1e-8 relative at the default prior count
- **Test:** `test_prior_count_scales_with_library_size`

The first pass handles a matrix offset and gives
`log(lib + 2 * mean(scaled_prior) * mean(lib) / lib)`, which is
`log(lib + 2 * prior)`. The correct `log(lib + 2 * prior * lib / mean(lib))`
follows, but behind `if offset.ndim == 1`, and `glm_fit` always passes a matrix.

At `prior.count = 0.125` with libraries from 0.9e6 to 1.3e6, coefficients agree
with `glmFit` to 1.4e-8. The error grows with the prior count and with unequal
libraries. `edge-rs` scales by library size, as edgeR does.

## Dispersion and smoothing

### A6. `natural_spline_basis` is not R's `ns()`

- **Upstream:** `edgepython/limma_port.py:281`, `edgepython/dispersion_lowlevel.py:804`
- **Ported to:** `src/numeric/interpolate.rs`, `natural_spline_basis`
- **Severity:** none in practice
- **Test:** `test_natural_spline_basis_spans_the_same_space_as_r_ns`

edgePython builds a truncated power basis; `splines::ns()` is a constrained
B-spline basis. Different matrices, same span. The basis only ever feeds a
least-squares or GLM fit, so fitted values agree to machine precision while
coefficients do not.

`edge-rs` follows the Python and tests span equivalence against `ns()`. Recorded
so nobody "fixes" it with a real `ns()` and wonders why coefficients moved.

### A7. `locfitByCol` is a per-point fit, not locfit

- **Upstream:** `edgepython/smoothing.py` (`locfit_by_col` and its kernels)
- **Ported to:** `src/limma/smoothing.rs`, `locfit_by_col`
- **Severity:** 2 to 8% on every trended dispersion
- **Test:** `test_locfit_matches_edger_at_degree_0`, `test_locfit_matches_edger_at_degree_1`

`edgeR:::locfitByCol` calls the `locfit` package, which does not fit at every
point. It splits the covariate range into an adaptive tree (`rbox(cut = 0.8)`),
fits at the 9 to 43 cell corners, and interpolates: linearly at degree 0, cubic
Hermite on the fitted slope at degree 1. edgePython fits at every point instead.
On a 40-point fixture the curves differ by 0.024 absolute, 2 to 8% relative, and
`estimateDisp` runs this on the likelihood grid.

`edge-rs` ports locfit's 1-D tree. The nearest-neighbour bandwidth is the
`(int)(n * span + 1e-12)`-th nearest distance; that `1e-12` matters at exact
ties. Agrees with edgeR to 2.6e-15.

`loessByCol` is edgeR's own `src/R_loess_by_col.cpp` (not in the binary
install), a tricube moving average with a forward-only frame. Ported line for
line, including `low_value = 1e-10` and the descending summation order, it is
bit-identical.

### A8. `chooseLowessSpan` has the wrong defaults

- **Upstream:** `edgepython/limma_port.py:920` (`choose_lowess_span`)
- **Ported to:** `src/utils/design.rs`, `choose_lowess_span`
- **Severity:** a span around 0.15 too narrow wherever it is used
- **Test:** `test_choose_lowess_span_matches_limma`

limma: `chooseLowessSpan(n = 1000, small.n = 50, min.span = 0.3, power = 1/3)`.
edgePython: `small_n = 25, min_span = 0.2`. Same formula,
`min(min_span + (1 - min_span) * (small_n / n)^power, 1)`, different constants:

| n | limma | edgePython |
|---|---|---|
| 100 | 0.8555904 | 0.7039684 |
| 1000 | 0.5578822 | 0.4039684 |

`edge-rs` takes all four as arguments and carries limma's values in
`LIMMA_LOWESS_DEFAULTS`.

## Empirical Bayes and limma

### A9. `trigamma_inverse` uses the wrong large-x asymptotic

- **Upstream:** `edgepython/limma_port.py:615` (`_trigamma_inverse`)
- **Ported to:** `src/numeric/gamma.rs`, `trigamma_inverse`
- **Severity:** silently wrong in the far tail
- **Test:** `test_trigamma_inverse_clamps_in_the_tails`

edgePython returns `1/x` for `x > 1e7`; limma returns `1/sqrt(x)`. For small `y`,
`psi'(y) ~ 1/y^2`, so inverting gives `y ~ 1/sqrt(x)`. limma is right:

```
$ Rscript -e 'library(limma); b <- body(trigammaInverse); cat(deparse(b[[9]]), sep="\n")'
if (any(omit)) {
    y <- x
    y[omit] <- 1/sqrt(x[omit])
    ...
}
```

`edge-rs` follows limma, takes its better-conditioned `0.5 + 1/x` start, and
keeps the Python's tighter `1e-10` stopping rule. Reached through `squeeze_var`
when the prior df is large, i.e. designs with many residual df.

### A10. `squeezeVar` and `fitFDist` clamp and use the wrong smoother

- **Upstream:** `edgepython/limma_port.py`, `_fit_f_dist` and `_fit_f_dist_robustly`
- **Ported to:** `src/limma/squeeze_var.rs`
- **Severity:** varies; the smoother swap is the one that matters
- **Test:** `test_squeeze_var_robust_trended_matches_limma`

Five divergences from limma:

- `_fit_f_dist` clamps `df2 >= 1e-6`, treats `> 1e15` as infinite and floors the
  scale at `1e-15`. limma does none of this.
- `_fit_f_dist_robustly` smooths with `weightedLowess(span = 0.4, iterations = 4)`.
  limma uses `stats::lowess(f = 0.4, iter = 3)` via `loessFit`. Different
  algorithms, different trend.
- `squeeze_var` sets `var[df == 0] <- 0` always; limma only with more than one `df`.
- `_fit_f_dist_trend` places spline knots over every covariate, not just the
  genes surviving the filter, and interpolates linearly to dropped genes where
  limma predicts from the spline.
- Log tail probabilities are clipped to `[-500, 0]` before `qf`; limma passes
  them through with `log.p = TRUE`.

`edge-rs` follows limma throughout, using the Cleveland `lowess` port in
`src/limma/lowess.rs`, and agrees to 1e-12. One residual gap: the spline is
built over surviving genes as in limma, but dropped genes (zero `df1` or
non-finite variance) are read off the fitted trend rather than via
`predict.ns`, because the A6 basis exposes no knots. That costs about 1e-3 on
those genes only.

### A11. `compute_prior` uses the wrong smoother

- **Upstream:** `edgepython/ql_weights.py:666` (`compute_prior`)
- **Ported to:** `src/ql/weights.rs`, `compute_prior`
- **Severity:** up to 7e-4 on the QL prior and every dispersion downstream
- **Test:** `test_compute_prior_matches_edger`

The docstring claims it "wraps the same Cleveland/Grosse Fortran code as R's
lowess", then calls limma's `weightedLowess`. edgeR's `compute_ave_qd` uses
Cleveland's `lowess` with `f = 0.5, iter = 3`. The window is a neighbour count in
one and enclosed prior weight in the other, and the delta rule differs. Against
`.Call(edgeR:::.cxx_compute_ave_qd, ...)`:

| fixture | edgeR | via `weightedLowess` | relative |
|---|---|---|---|
| 25-point dyadic grid | 2.423348828030909 | 2.4240863018696586 | 3.0e-4 |
| same, two genes filtered | 2.2385492590160405 | 2.2369528500736866 | 7.1e-4 |
| 24 genes by 6 samples | 18.590485410598912 | 18.590485410599115 | 1.1e-14 |
| 500 genes, overdispersed | 148.62313987863917 | 148.61541890776701 | 5.2e-5 |

`edge-rs` uses `lowess` from `src/limma/lowess.rs` and agrees to 1e-12.

**Iteration trap:** R's `lowess` counts robustness passes *after* the initial
fit; limma's `weightedLowess` counts total passes. Handing limma's 4 to R's
smoother costs 3e-3, worse than the original mismatch.

### A12. `lmFit` loses estimability, df and `stdev_unscaled`

- **Upstream:** `edgepython/voom_lmfit.py`, `_lm_fit` and `_row_lm_fit_with_missing`
- **Ported to:** `src/limma/lm_fit.rs`
- **Severity:** high; several independent problems
- **Test:** `test_aliased_design_reports_rank_and_pivot`, `test_zero_weight_drops_the_observation`

- **No `stdev_unscaled`.** `eBayes` cannot be built without it.
- **Pseudo-inverse for rank deficiency.** Gives a finite minimum-norm solution;
  limma pivots and returns `NA` for the aliased column. Different numbers, and
  nothing flagged as inestimable.
- **Zero weights clipped, not dropped.** A zero-weight sample still costs a
  residual df: `df_residual` 3 where limma gives 2 on the fixture.
- **`correlation` clamped to plus or minus 0.95.** limma errors at `|r| >= 1`
  and, via `chol`, on an indefinite block covariance. edgePython fits a model
  that does not exist.
- **Rank from an SVD**, not the pivoted QR, so it differs near the boundary and
  cannot say *which* column is aliased.

`edge-rs` follows limma, including its choice of aliased column.

### A13. `voom` uses one smoother where upstream uses two

- **Upstream:** `edgepython/voom_lmfit.py:418` (`_weighted_lowess_trend`) and `voom`
- **Ported to:** `src/limma/voom.rs`
- **Severity:** same class of error as [A11](#a11-compute_prior-uses-the-wrong-smoother)
- **Test:** `test_voom_lmfit_and_voom_disagree_on_structural_zeros`

limma's `voom` smooths the mean-variance trend with `stats::lowess(f = span,
iter = 3)`. edgeR's `voomLmFit` does the same until structural zeros are
detected, then switches to `weightedLowess` weighted by residual df. edgePython
uses `weighted_lowess` always, and with `npts = 120` and 3 iterations instead
of 200 and 4.

`edge-rs` dispatches as upstream does. Mind the iteration trap in A11.

### A14. `duplicateCorrelation` and `arrayWeights` are different estimators

- **Upstream:** `edgepython/voom_lmfit.py:898` (`duplicate_correlation`) and `array_weights`
- **Ported to:** `src/limma/array_weights.rs`
- **Severity:** high; agrees only in the simplest case
- **Test:** `test_duplicate_correlation_unbalanced_blocks`, `test_duplicate_correlation_absorbed_block_is_zero`

limma fits a REML mixed model per gene (`statmod::mixedModel2Fit`) and takes a
trimmed mean of Fisher-z correlations. edgePython uses a one-way ANOVA moment
estimator. They agree only for balanced blocks and an intercept-only design.
On top of that:

- per-gene correlations clipped to `[-0.99, 0.99]` instead of limma's
  block-size bound, which is what keeps the block covariance positive definite
- the `nblocks < n_obs - 1` and `n_obs > n_coef + 2` admission rules are dropped
  on the vectorised path
- no check for a block factor spanned by the design or for all-singleton
  blocks; both are exact-zero returns in limma

`arrayWeights` has its own list:

- **`method = "reml"` with gene weights drops the weights.** limma has
  `.arrayWeightsPrWtsREML` for this; edgePython's REML takes no `weights`.
- a fully pivoted QR reorders columns by magnitude; R moves only negligible
  columns, and only to the end
- `pinv(..., rcond = 1e-12)` where limma uses `solve`
- genes admitted on `sum(good) > p`; limma also needs two residual df
- `nanmean` where limma's `colMeans` propagates NaN

`edge-rs` ports limma, including `glmgam.fit`, and agrees to 5e-15.

## Exact test

### A15. The deviance and small-p rejection regions are stubs

- **Upstream:** `edgepython/exact_test.py:287` and `:295`
- **Ported to:** `src/exact/mod.rs`
- **Severity:** two of three rejection regions do the wrong test
- **Test:** `test_deviance_matches_edger_on_unequal_groups`, `test_small_p_matches_edger_called_one_gene_at_a_time`

`exact_test_by_deviance` and `exact_test_by_small_p` both call
`exact_test_double_tail`. On a six-gene fixture at dispersion 0.1, gene 1:
doubletail 1.088e-05, deviance 8.134e-06, smallp 1.181e-05.

edgeR's `exactTestBySmallP` itself returns to `exactTestDoubleTail` when
`n1 == n2`, so the stub is right for balanced designs only. `edge-rs` implements
all three.

### A16. `q2qnbinom` leaves log space

- **Upstream:** `edgepython/exact_test.py:425` (`q2q_nbinom`)
- **Ported to:** `src/exact/mod.rs`, `q2q_nbinom`
- **Severity:** `+Inf` where edgeR is finite
- **Test:** `test_q2q_nbinom_survives_the_far_upper_tail`

edgePython computes `norm.isf(np.exp(p1))` and `gamma_dist.isf(np.exp(p2))`; R
keeps `log.p = TRUE`. At `x = 6000`, `input_mean = 1000`, `dispersion = 0.001`
the gamma upper tail is around `e^-1605`, which exponentiates to zero, so the
inverse is `+Inf`. edgeR returns 8186.42.

`edge-rs` stays in log space, hence the log-scale upper incomplete gamma in
`src/exact/mod.rs`.

## Splicing

### A17. `spliceVariants` is a different test

- **Upstream:** `edgepython/splicing.py:483` (`splice_variants`)
- **Ported to:** `src/splicing.rs`, `splice_variants`
- **Severity:** answers a different question
- **Test:** `test_matches_edger_splice_variants`

edgeR unrolls each gene into an exon-by-group layout, fits
`~ exon + group + exon:group` as an NB GLM and runs an LRT on the interaction:
differential exon usage *between conditions*. edgePython runs a Pearson
chi-squared homogeneity test on raw counts, with no dispersion and **no `group`
argument**, so it tests exon-by-*sample* heterogeneity. `edge-rs` ports edgeR.

### A18. `diffSpliceDGE` is not edgeR's procedure

- **Upstream:** `edgepython/splicing.py:410` (`diff_splice_dge`), against edgeR's `diffSpliceDGE`
- **Ported to:** `src/splicing.rs`, `diff_splice` and `diff_splice_dge`
- **Severity:** a different test under the same name
- **Test:** `test_matches_edger_lrt_on_the_gate_fixture`, `test_matches_edger_quasi_likelihood_f_tests`

edgeR's `diffSpliceDGE` takes a fitted `DGEGLM`, folds the gene-level fold change
into the offsets, refits, and tests each exon against the rest of its gene: a
likelihood ratio test, or a QL F-test with `squeezeVar` on the gene variances
when the fit is quasi-likelihood. Gene-level p-values come from the summed test
and from Simes. edgePython runs `exact_test` on the exon counts and aggregates
by Simes, with no GLM and no offset adjustment.

`edge-rs` ports edgeR as `diff_splice`; `diff_splice_dge` is the `DgeList`
wrapper that fits first (as `glm_fit_dge` wraps `glm_fit`). Genes come out in
first-appearance order. edgeR sorts by gene id, so the two agree on sorted
input; edgePython's `np.unique` always sorts.

## Single cell (NEBULA)

### A19. `glm_sc_test` drops the off-diagonal covariance

- **Upstream:** `edgepython/sc_fit.py:1494` (`glm_sc_test`)
- **Ported to:** `src/sc/test.rs`
- **Severity:** wrong contrast standard errors on correlated designs
- **Test:** `test_contrast_uses_the_off_diagonal_term`

Contrast SEs are `sqrt(sum(se^2 * c^2))`, correct only for a diagonal covariance,
because `glm_sc_fit` keeps only `sqrt(diag(cov))`. The right variance is
`c' V c`. Off-diagonals are non-zero whenever a batch or covariate enters the
model, and the error can go either way, so it is not even conservative.

`edge-rs` keeps the full `p x p` inverse Fisher information per gene. Single
coefficients are unaffected. The fix needs the fit to change, not just the test.

### A20. NEBULA has no marginal-likelihood Hessian

- **Upstream:** `edgepython/sc_fit.py`, the NEBULA-LN path
- **Ported to:** `src/sc/ptmg.rs`, `src/sc/nebula.rs`
- **Severity:** the highest here. Point estimates survive, inference does not.
- **Test:** `test_nebula_matches_the_r_package` (`tests/e2e_nebula.rs`)

`nebula` takes SEs from `ptmg_ll_der_hes_eigen`, the Hessian of the *marginal*
log-likelihood. edgePython has none: `grep trigamma sc_fit.py` is empty, and the
sigma-sigma block needs `trigamma(cumsumy + alpha)` and `trigamma(y + phi)`. Its
Hessians at lines 370-402 belong to the inner Laplace step of `_opt_pml_nb`:
the *penalised* Hessian, a different matrix.

Against `nebula` 1.5.8 on data with a real subject effect:

| | logFC (worst / median) | SE (worst / median) |
|---|---|---|
| 6 subjects | 1.9% / 0.16% | 289% / 24% |
| 20 subjects | 0.07% / 0.02% | 89% / 6.2% |

Fold changes converge as subjects are added; SEs, and so p-values, do not.
`edge-rs` ports the single-cell stack from `nebula`'s own C++
(`src/optimization.cpp`) and R driver.

Also: `_digamma_nb` is a hand-rolled asymptotic series ("~15 digits"). At the
lower `sigma` bound `alpha_pr` reaches `1e8` and `alpha_pr^2` reaches `1e16`, so
a one-ulp digamma error becomes an O(1) error in the sigma gradient. `edge-rs`
uses `numeric::gamma`, bit-identical to base R there.

### A21. `_opt_pml_nb` adds a ridge, a floor and a clamp

- **Upstream:** `edgepython/sc_fit.py`, `_opt_pml_nb`
- **Ported to:** `src/sc/pml.rs`
- **Severity:** high. Each item moves the information matrix, so the SE.
- **Test:** `test_opt_pml_well_behaved_matches_nebula`, `test_opt_pml_reml_log_determinant`

1. **Ridge.** `if abs(vb2[ii, ii]) < 1e-10: vb2[ii, ii] += 1e-8`, on exactly the
   matrix inverted for the SEs. Not in the C++.
2. **`vw` floored at `1e-15`** in the main loop and the `ord` block. Shifts
   `dw/vw`, `vwb/vw`, the Schur complement and the log-determinant together.
3. **REML dropped.** `logdet = sum(log(max(|vw|, 1e-300)))` ignores `reml`. The
   C++ adds `log|det(vb2)|` when `reml == 1`, a different outer objective.
4. **Linear predictor clamped at 500.** The C++ relies on `isinf(loglik)` to
   reject the step in backtracking. Clamping changes the surface and defeats the
   test, so the damping path diverges.
5. **Schur complement as `vb - vwb' diag(1/vw) vwb`.** The C++ forms
   `temp = vwb / sqrt(vw); vb - temp'temp`, symmetric by construction.
6. **`np.linalg.solve`** (partial-pivot LU) where Eigen uses `LDLT` with tiny
   pivots zeroed. They disagree on near-singular systems, the population the
   `-25` convergence code flags.
7. `extb < 1e-300 -> 0` guards. Harmless; listed for completeness.

`edge-rs` follows the C++ and matches `nebula:::opt_pml` to 1e-11 on `beta`,
`log_w`, the information matrix, `loglik`, `logdet` and `second`.

Kept on purpose: `sec_ord` in the C++ recomputes `vw` at the final iterate
*after* `logdet` was taken from the previous one. The outer objective is
calibrated against that, so both ports keep it.

---

# Section B: edgeR or limma are wrong

### B1. `aveLogCPM` corrupts the first gene under a matrix offset

- **Upstream:** edgeR 4.8.2, `aveLogCPM` with a matrix `offset`
- **Ported to:** `src/core/expression.rs`, `ave_log_cpm`
- **Severity:** gene 1 off by about 20 on the log2 scale; under `glmQLFit`, the QL prior for every gene
- **Test:** `test_ave_log_cpm_matches_edger` (`tests/e2e_bulk_classic.rs`), `test_matches_edger_glm_ql_fit` (`src/glm/ql_fit.rs`)

A vector offset and the equivalent matrix describe the same model. They agree on
every gene but the first:

```
y <- matrix(c(10,12,11,40,44,38, 50,48,52,49,51,50, 2,0,5,1,3,0), nrow = 3, byrow = TRUE)
ls <- c(1e6, 1.2e6, 0.9e6, 1.1e6, 1e6, 1.3e6)
aveLogCPM(y, offset = log(ls), dispersion = 0.1)
#  4.675093  5.605251  1.847387
aveLogCPM(y, offset = matrix(log(ls), 3, 6, byrow = TRUE), dispersion = 0.1)
# 24.622730  5.605251  1.847387
```

Reproduced on an 8 by 5 Poisson matrix (row 1: 4.838 against 24.708),
independent of `prior.count`. Row 1 of a matrix offset is not reaching the C
routine intact.

It spreads. `glmQLFit` computes `AveLogCPM` internally as the `squeezeVar` trend
covariate, so a matrix offset moves the QL prior for *every* gene. Eight genes at
`dispersion = 0.1`:

| offset form | `s2.prior` |
|---|---|
| none, or a vector | 3.90e-4, 4.08e-5, 2.09e-4, 1.31e-1, ... |
| the equivalent matrix | 2.61e+0, 1.10e-4, 4.79e-5, 6.85e-5, ... |

Gene 1's abundance is 26.32 instead of 16.35, dragging the trend across the whole
range. Feeding limma's `squeezeVar` the two covariates reproduces each output
exactly; everything upstream of the covariate agrees to 1e-9.

`edge-rs` treats both forms as the same model. The matrix-offset tests use the
vector call as the reference for gene 1 and edgeR's matrix call for the rest.
Worth reporting upstream, and worth knowing when comparing against pipelines
that pass matrix offsets, which `voomLmFit` and custom normalisation both do.

### B2. `fitFDistRobustly` stops its root solve early

- **Upstream:** limma 3.66, `fitFDistRobustly`
- **Ported to:** `src/limma/squeeze_var.rs`, `fit_f_dist_robustly`
- **Severity:** about 1e-4 relative on `df2`
- **Test:** `test_robust_fit_matches_stock_limma_to_its_own_root_tolerance`

limma solves for `df2` with `uniroot(..., tol = 1e-8)` on the `d / (1 + d)` link.
At `df2` near 100 that leaves the answer about 1e-4 out; limma's value sits
2.5e-8 from the root of its own objective. With `uniroot` shadowed at
`tol = 1e-14`, limma lands on `edge-rs` to 5e-14.

`edge-rs` converges at `xtol = 1e-15`. Robust references in the suite are
limma-with-converged-uniroot (override script in the test doc and
`tests/r/generate_fixtures.R`), plus one test pinning stock limma at 1e-7 and
asserting `edge-rs` is closer to the true root.

A related limit, not a bug: `fitFDistUnequalDF1` maximises a flat likelihood
with `optimize` at its default `tol` of 1.2e-4, which lands `df.prior` at 1.4e-8
relative. `tests/e2e_ebayes.rs` sizes its tolerances for that.

### B3. `exactTestBySmallP` returns one p-value for every gene

- **Upstream:** edgeR 4.8.2, `exactTestBySmallP`, final line
- **Ported to:** `src/exact/mod.rs`, the `SmallP` rejection region
- **Severity:** severe. Every gene reports the same p-value.
- **Test:** `test_small_p_matches_edger_called_one_gene_at_a_time`

The function ends with `min(pvals, 1)` where it needs `pmin(pvals, 1)`, so the
whole matrix collapses to its smallest p-value. Masked for balanced designs by
the early return to `exactTestDoubleTail`. With unequal groups:

```
y1 <- 6 genes by 2 samples, y2 <- the same 6 genes by 4 samples
exactTestBySmallP(y1, y2, dispersion = 0.1)
# 0.0002999433935          <- length 1, not 6

# one gene at a time:
# 0.0004995834196  1  1  0.0002999433935  1  0.9092143135841
```

Visible through the documented entry point:

```
exactTest(d, rejection.region = "smallp")   # unequal group sizes
# PValue: 0.008179571648 repeated for all six genes
```

`edge-rs` caps each gene on its own value. Worth reporting upstream.

### B4. `binomTest` mishandles exact ties

- **Upstream:** edgeR 4.8.2, `binomTest`
- **Ported to:** `src/exact/mod.rs`, the zero-dispersion limit
- **Severity:** 0.86 where the answer is 1
- **Test:** `test_zero_dispersion_uses_the_binomial_test`

At `p = 1/3` and a total of `3k + 2`, outcomes `k` and `k + 1` are equiprobable.
edgeR's `order`/`cumsum` keeps or drops the tied term on a one-ulp difference in
`dbinom`:

```
binomTest(12, 23, p = 1/3)      # 0.8600904883
binom.test(12, 35, 1/3)$p.value # 1
binomTest(10, 20, p = 1/3)      # 1, so tie-specific
```

`edge-rs` uses `binom.test`'s `1 + 1e-7` relative slack and gives 1.

Not reproduced either: above a total of 10000, `binomTest` switches to a
chi-square shortcut whose 2 by 2 table uses column totals of whichever genes were
passed in, so one gene's p-value depends on its neighbours. `edge-rs` enumerates
exactly at every size.
