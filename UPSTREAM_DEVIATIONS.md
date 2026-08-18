# Upstream deviations

Places where `edge-rs` deliberately does not reproduce its upstreams, and why.

The port's reference is [`edgePython`](https://github.com/pachterlab/edgePython),
itself a port of edgeR, limma and `nebula`. Where the Python disagrees with the R
and C++ it came from, upstream wins: those packages are what users compare
against, and their output is what the test suite gates on. Section A covers those
cases.

Section B is different and rarer: places where **edgeR or limma themselves are
wrong**, verified by reproducing the fault in the installed package. There the R
does not win, because reproducing a bug faithfully is not parity worth having.

Every entry is checked against the actual installed package, not inferred from
reading, and is covered by a test that would fail if the behaviour drifted back.

---

# Section A: edgePython disagrees with edgeR or limma

---

## 1. `trigamma_inverse` uses the wrong asymptotic for large arguments

**Where:** `edgepython/limma_port.py:615` (`_trigamma_inverse`)
**Ported to:** `src/numeric/gamma.rs`, `trigamma_inverse`
**Severity:** wrong answers, silently, in the far tail

edgePython returns `1/x` for `x > 1e7`. limma returns `1/sqrt(x)`.

limma is right. `trigamma_inverse` inverts `psi'(y) = x`, and for small `y` the
trigamma function behaves as `psi'(y) ~ 1/y^2`. Inverting that gives
`y ~ 1/sqrt(x)`, not `1/x`. A large `x` means a small root, so this is exactly
the regime where the approximation matters.

Verified against the installed limma:

```
$ Rscript -e 'library(limma); b <- body(trigammaInverse); cat(deparse(b[[9]]), sep="\n")'
if (any(omit)) {
    y <- x
    y[omit] <- 1/sqrt(x[omit])
    ...
}
```

`edge-rs` follows limma. It also takes limma's `0.5 + 1/x` starting value over
the Python's, which is better conditioned, while keeping the Python's tighter
`1e-10` stopping rule.

Reached through `squeeze_var` when the prior degrees of freedom are large, which
happens on designs with many residual degrees of freedom.

---

## 2. `glm_sc_test` discards the off-diagonal covariance

**Where:** `edgepython/sc_fit.py:1494` (`glm_sc_test`)
**Ported to:** `src/sc/test.rs` (phase 8, not yet written)
**Severity:** wrong standard errors for contrasts on correlated designs

Contrast standard errors are computed as `sqrt(sum(se^2 * c^2))`, which is only
correct when the coefficient covariance matrix is diagonal. It is that way
because `glm_sc_fit` keeps only `sqrt(diag(cov))` and throws the rest away.

For a contrast `c`, the correct variance is `c' V c`, which carries the
off-diagonal terms. Those are non-zero whenever design columns are correlated,
which is the normal case once a batch or covariate is in the model. The error
goes in either direction depending on the sign of the covariance, so it is not a
conservative approximation.

`edge-rs` keeps the full `p x p` inverse Fisher information per gene and forms
`c' V c` properly. Single-coefficient tests are unaffected; contrasts are not.

Not a numerical accident: it is a consequence of what `glm_sc_fit` chose to
return, so it needs the fit to change too, not just the test.

---

## 3. TMMwsp trims one gene too many at each end

**Where:** `edgepython/normalization.py:299` (`_calc_factor_tmmwsp`)
**Ported to:** `src/core/normalisation.rs`, `calc_factor_tmmwsp`
**Severity:** high. Normalisation factors wrong by several percent, and every
downstream fold change with them.

The trim window is mistranslated from R's 1-based indexing. R keeps positions
`[loM, n + 1 - loM]` inclusive. The Python computes `hiM = n - loM` and slices
`o_M[loM:hiM]`, which in 1-based terms is `[loM + 1, n - loM]`: one element
dropped at each end. `_calc_factor_tmm` right above it is correct, so this is a
transcription slip rather than a deliberate choice.

It is not a small effect. On a 10 by 3 matrix with the zeros TMMwsp exists to
handle:

| method | edgePython | edgeR 4.8.2 |
|---|---|---|
| TMM | 0.883046965512903, 1.059296727816462, 1.069051351250571 | identical to 15 figures |
| TMMwsp | 0.892298677901486, 1.070758413481783, 1.046642232997959 | 0.831890371585877, 1.193584170243712, 1.007119145059728 |

TMM agrees exactly, which rules out any difference in the surrounding
machinery and isolates the fault to the TMMwsp trim. The TMMwsp factors differ
by up to 11%.

edgePython's own tests do not catch this: they compare against R at a `1e-3`
tolerance on large matrices, where dropping two genes out of tens of thousands
moves the trimmed mean far less than on the small, sparse inputs TMMwsp is
actually recommended for.

`edge-rs` follows the R.

---

## 4. The GLM path uses a naive unit deviance where edgeR uses a careful one

**Where:** `edgepython/glm_levenberg.py:310` (`_unit_nb_deviance`) and
`nbinom_deviance` at line 224
**Ported to:** `src/glm/deviance.rs`
**Severity:** deviances wrong in the eighth significant figure, and much worse
near `y == mu`

edgePython carries two different unit deviance implementations and only one of
them matches edgeR:

- `ql_weights.py:500` (`compute_unit_nb_deviance`) is a faithful port of edgeR's
  `compute_nbdev.c`, with the input nudge and all three regimes.
- `glm_levenberg.py:310` (`_unit_nb_deviance`) is the textbook formula with none
  of that, and it is the one the Levenberg fitter and `nbinom_deviance` use.

edgeR does two things the naive version does not. First it adds
`mildly_low_value = 1e-8` to both `y` and `mu` before doing anything else.
Second it switches formula by regime: a Poisson expansion with a correction term
for `phi < 1e-4`, a gamma limit for `mu * phi > 1e6`, and otherwise the
rearranged exact form `2*(y*log(y/mu) + (y + 1/phi)*log((mu + 1/phi)/(y + 1/phi)))`,
which is algebraically the same as the naive expression but does not cancel.

Measured against the installed edgeR 4.8.2, `nbinomUnitDeviance` versus the
naive formula at `mu = 10`, `phi = 0.1`:

| y | edgeR | naive | relative difference |
|---|---|---|---|
| 12 | 0.182069451405 | 0.1820694517 | 1.4e-9 |
| 3 | 3.97651898398 | 3.976518992 | 2.1e-9 |
| 0 | 13.8629432006 | 13.86294361 | 3.0e-8 |
| 10.001 | 4.99974996685e-08 | 4.99975045900e-08 | 9.8e-8 |
| 10 | 0 | 5.0e-16 | total |

Adding the `1e-8` nudge to the naive formula recovers edgeR to 1e-13 or better
everywhere except right next to `y == mu`, where the regime switch is what
carries the remaining accuracy. So both halves are needed.

`edge-rs` implements edgeR's version once, in `src/glm/deviance.rs`, and uses it
for the Levenberg fit, the residual deviance and the quasi-likelihood weights
alike. There is no second implementation.

The size of this matters for the tighter fixtures: a deviance carried into a
likelihood ratio statistic at 1e-8 relative is fine against a 1e-3 tolerance and
not fine against the 1e-8 ones the Python suite uses in places.

---

## 5. edgePython "fixes" an integer division that edgeR relies on

**Where:** `edgepython/ql_weights.py:514` (`compute_unit_nb_deviance`)
**Ported to:** `src/glm/deviance.rs`, `unit_nb_deviance`
**Severity:** up to 7e-4 relative on large counts at small dispersion

In the Poisson regime the correction term in edgeR's C reads

```c
2 * (y * log(y/mu) - resid - 0.5*resid*resid*phi*(1 + phi*(2/3*resid - y)))
```

`2/3` there is integer division, so it is zero, and the whole `resid`
contribution vanishes: the term is `-phi*y`. edgePython writes `2.0 / 3.0`,
which reinstates it.

Whether or not the C is what its author intended, it is what edgeR computes and
therefore what every published edgeR result reflects. Measured against
`nbinomUnitDeviance` in edgeR 4.8.2:

| y | mu | phi | edgeR | integer `2/3` | edgePython's `2.0/3.0` |
|---|---|---|---|---|---|
| 1e6 | 1.001e6 | 1e-5 | 90.999333833 | 90.999333833 | 91.0660004996 |
| 1e6 | 1e6 + 1 | 1e-6 | 9.99899558621e-07 | 9.99899558707e-07 | 9.99900225374e-07 |
| 100 | 110 | 1e-5 | 0.936965039047 | 0.936965039047 | 0.936965105714 |
| 0 | 10 | 1e-5 | 19.9989995855 | 19.9989995855 | 19.9989996522 |

The integer form matches to twelve figures; the corrected form is out by 7e-4 in
the worst case. `edge-rs` reproduces the C.

This one is easy to miss: it only shows up when the dispersion is under the 1e-4
Poisson threshold *and* the counts are large, so a test at `phi = 1e-10`, where
the correction is negligible either way, passes under both versions.

---

## 6. `add_prior_count` drops the library scaling when offsets arrive as a matrix

**Where:** `edgepython/utils.py:49` (`add_prior_count`)
**Ported to:** `src/glm/fit.rs`, `add_prior_count`
**Severity:** low, around 1e-8 relative at edgeR's default prior count

The function computes its result twice. The first pass handles a matrix offset
and produces
`log(lib + 2 * mean(scaled_prior) * mean(lib) / lib)`, which simplifies to
`log(lib + 2 * prior)`. A second block then recomputes the correct
`log(lib + 2 * prior * lib / mean(lib))`, but it is guarded by
`if offset.ndim == 1`, and `glm_fit` always passes a matrix, so the correct
branch never runs.

The effect is small because the prior count is tiny next to a library size.
Comparing `glm_fit` against `glmFit` at the default `prior.count = 0.125` with
libraries spanning 0.9e6 to 1.3e6, coefficients agree to 1.4e-8, which is the
size this predicts. It grows with the prior count and with how unequal the
libraries are, so it is worth getting right rather than inheriting.

`edge-rs` scales the prior by library size, as edgeR does.

---

## 7. `natural_spline_basis` is not R's `ns()`

**Where:** `edgepython/limma_port.py:281`, `edgepython/dispersion_lowlevel.py:804`
**Ported to:** `src/numeric/interpolate.rs`, `natural_spline_basis`
**Severity:** none in practice, but worth knowing

edgePython builds a truncated power basis. R's `splines::ns()` returns a
B-spline parametrisation with the natural boundary constraints applied. These
are different matrices.

This is not a bug. Both span the same function space, and the basis is only ever
used as a design matrix for a least-squares or GLM fit, so fitted values agree
to machine precision even though coefficients do not. `edge-rs` follows the
Python so the rest of the port stays self-consistent, and the test asserts
span-equivalence against R's `ns()` rather than element-wise equality.

Flagged only so nobody later "fixes" it by swapping in a real `ns()` and then
wonders why the coefficients moved.

---

## 8. `locfitByCol` is the real locfit package, not a per-point local fit

**Where:** `edgepython/smoothing.py` (`locfit_by_col` and its kernels)
**Ported to:** `src/limma/smoothing.rs`, `locfit_by_col`
**Severity:** 2 to 8% on every trended dispersion

`edgeR:::locfitByCol` calls the `locfit` package. locfit does **not** fit a local
regression at every data point. It builds an adaptive binary subdivision of the
covariate range (`rbox(cut = 0.8)`), fits only at the resulting cell corners,
typically 9 to 43 of them, and interpolates between them: linearly for degree 0,
cubic Hermite on the fitted slope for degree 1.

edgePython reimplements it as a straightforward local fit evaluated at every
point. That is a different estimator. On a 40-point fixture the tree-interpolated
curve and the exact per-point curve differ by 0.024 absolute, 2 to 8% relative.
Since `estimateDisp` runs this on the log-likelihood grid, the error lands
directly on the trended and tagwise dispersions.

`edge-rs` ports locfit's 1-D adaptive tree: the nearest-neighbour bandwidth is
the `(int)(n * span + 1e-12)`-th nearest distance, and that `1e-12` is
load-bearing at exact ties. The result agrees with edgeR to 2.6e-15.

`loessByCol` is a separate story with a happier ending: it is edgeR's own
`src/R_loess_by_col.cpp`, a tricube moving average with a forward-only sliding
frame. That source is not shipped in the binary install and had to be fetched
separately. Ported line for line, including its `low_value = 1e-10` and its
descending summation order, it comes back bit-identical.

---

## 9. `squeezeVar` and `fitFDist` pick up clamps and the wrong smoother

**Where:** `edgepython/limma_port.py`, `_fit_f_dist` and `_fit_f_dist_robustly`
**Ported to:** `src/limma/squeeze_var.rs`
**Severity:** varies; the smoother swap is the material one

Five separate divergences from limma, all in the empirical Bayes fit:

- `_fit_f_dist` clamps `df2` to at least `1e-6`, treats anything above `1e15` as
  infinite, and floors the scale at `1e-15`. limma's `fitFDist` has none of these.
- `_fit_f_dist_robustly` smooths the trend with `weightedLowess(span = 0.4,
  iterations = 4)`. limma reaches `stats::lowess(f = 0.4, iter = 3)` through
  `loessFit`. Cleveland's lowess and limma's `weightedLowess` are different
  algorithms, so the fitted trend differs. This is the one that matters.
- `squeeze_var` applies `var[df == 0] <- 0` unconditionally; limma only does so
  when there is more than one `df`.
- `_fit_f_dist_trend` builds its spline over every covariate rather than only the
  genes that survive filtering, which moves the knots, and interpolates linearly
  to the dropped genes where limma predicts from the spline.
- `_fit_f_dist_robustly` clips log tail probabilities to `[-500, 0]` before
  `qf`; limma passes them through untouched with `log.p = TRUE`.

`edge-rs` follows limma throughout, including porting Cleveland's `clowess`
privately for the robust trended path, and agrees with it to 1e-12.

---

## 10. `filterByExpr` leverages include an intercept edgePython omits

**Where:** `edgepython/filtering.py:93` (`_hat_values`) versus edgeR's
`filterByExpr.default`
**Ported to:** `src/core/filtering.rs`, `design_leverage`
**Severity:** the wrong minimum group size, hence the wrong CPM cutoff, on any
design without an intercept in its column space

edgeR derives its minimum group size from `1 / max(stats::hat(design))`.
`stats::hat` defaults to `intercept = TRUE` and **prepends a column of ones**
before the QR. edgePython's `_hat_values` runs a plain QR on the design as given.

For `design = cbind(c(0.1, 0.2, 0.3, 0.4, 0.5, 0.9))`, edgeR's leverages max at
0.7917, giving a minimum group size of 1.2632; without the prepended column the
maximum is 0.5956 and the size is 1.679. Different cutoff, different gene list.

The two agree whenever the intercept is already in the design's span, which is
the common case, so this only bites on no-intercept designs.

`edge-rs` follows edgeR. Note that `[1 | X]` is rank deficient exactly when the
intercept is already in the span, so the implementation checks the rank of the
augmented matrix before computing leverages, rather than handing a singular
matrix to a QR that has no rank truncation.

---

## 13. `chooseLowessSpan` is given the wrong defaults

**Where:** `edgepython/limma_port.py:920` (`choose_lowess_span`)
**Ported to:** `src/utils/design.rs`, `choose_lowess_span`
**Severity:** the wrong smoothing span wherever it is used to pick one

The formula is right. The defaults are not. limma:

```r
chooseLowessSpan(n = 1000, small.n = 50, min.span = 0.3, power = 1/3)
```

edgePython declares `small_n = 25, min_span = 0.2`. Both feed the same
expression, `min(min_span + (1 - min_span) * (small_n / n)^power, 1)`, so the
difference is entirely in the constants:

| n | limma | edgePython |
|---|---|---|
| 100 | 0.8555904 | 0.7039684 |
| 1000 | 0.5578822 | 0.4039684 |

Around 0.15 narrower everywhere, which is a visibly tighter smooth. `edge-rs`
takes all four as explicit arguments and carries limma's values in
`LIMMA_LOWESS_DEFAULTS`, so a caller cannot inherit the wrong ones by accident.

---

## 14. The deviance and small-p rejection regions are stubs

**Where:** `edgepython/exact_test.py:287` and `:295`
**Ported to:** `src/exact/mod.rs`
**Severity:** two of the three rejection regions silently do the wrong test

`exact_test_by_deviance` and `exact_test_by_small_p` both just call
`exact_test_double_tail`. They are genuinely different tests whenever the two
groups have different sizes. Measured on a six-gene fixture at dispersion 0.1,
gene 1: doubletail 1.088e-05, deviance 8.134e-06, smallp 1.181e-05.

Note edgeR's own `exactTestBySmallP` returns early to `exactTestDoubleTail` when
`n1 == n2`, so the stub happens to be right for balanced designs. It is wrong
for everything else. `edge-rs` implements all three.

---

## 15. `q2qnbinom` exponentiates its log probabilities

**Where:** `edgepython/exact_test.py:425` (`q2q_nbinom`)
**Ported to:** `src/exact/mod.rs`, `q2q_nbinom`
**Severity:** infinities where edgeR returns a finite answer

edgePython computes `norm.isf(np.exp(p1))` and `gamma_dist.isf(np.exp(p2))`
where R keeps `log.p = TRUE` throughout. For `x = 6000`, `input_mean = 1000`,
`dispersion = 0.001` the gamma upper tail is around `e^-1605`, so exponentiating
gives exactly zero and the inverse returns `+Inf`. edgeR returns 8186.42.

`edge-rs` works in log space, which is why `src/exact/mod.rs` carries a
log-scale upper incomplete gamma.

---

## 16. `compute_prior` uses the wrong smoother

**Where:** `edgepython/ql_weights.py:666` (`compute_prior`)
**Ported to:** `src/ql/weights.rs`, `compute_prior`
**Severity:** up to 7e-4 on the quasi-likelihood prior, and on every dispersion
downstream of it

edgePython's docstring claims the smoother it uses "wraps the same
Cleveland/Grosse Fortran code as R's lowess", then calls limma's
`weightedLowess`. edgeR's `compute_ave_qd` uses Cleveland's `lowess` with
`f = 0.5, iter = 3`. They are different algorithms: nearest-neighbour count
versus enclosed prior weight for the window, and a different delta rule.

Measured against `.Call(edgeR:::.cxx_compute_ave_qd, ...)`:

| fixture | edgeR | via `weightedLowess` | relative |
|---|---|---|---|
| 25-point dyadic grid | 2.423348828030909 | 2.4240863018696586 | 3.0e-4 |
| same, two genes filtered | 2.2385492590160405 | 2.2369528500736866 | 7.1e-4 |
| 24 genes by 6 samples | 18.590485410598912 | 18.590485410599115 | 1.1e-14 |
| 500 genes, overdispersed | 148.62313987863917 | 148.61541890776701 | 5.2e-5 |

**Fixed.** The Cleveland `lowess` port was promoted out of
`src/limma/squeeze_var.rs` into `src/limma/lowess.rs` as a public `lowess`, and
both callers now use it. `compute_prior` agrees with edgeR to 1e-12.

Watch the iteration convention when touching this: R's `lowess` counts
robustness passes *after* the initial fit, limma's `weightedLowess` counts total
passes. Passing limma's 4 to R's smoother costs 3e-3, worse than the original
mismatch it was meant to fix.

---

## 19. `spliceVariants` is a different test entirely

**Where:** `edgepython/splicing.py:483` (`splice_variants`)
**Ported to:** `src/splicing.rs`, `splice_variants`
**Severity:** answers a different question from the function it is named after

edgeR's `spliceVariants` unrolls each gene into an exon-by-group layout, fits
`~ exon + group + exon:group` as a negative binomial GLM, and does a likelihood
ratio test on the interaction. That is a test for differential exon usage
*between conditions*.

edgePython instead runs a Pearson chi-squared test of homogeneity of proportions
on the raw counts. No dispersion, no negative binomial, and **no `group`
argument at all**, so what it tests is exon-by-*sample* heterogeneity. It cannot
answer edgeR's question, because the information needed is not passed in.

`edge-rs` ports edgeR.

---

## 20. `diffSpliceDGE` does not exist in edgeR

**Where:** `edgepython/splicing.py:410` (`diff_splice_dge`)
**Ported to:** `src/splicing.rs`, `diff_splice_dge`
**Severity:** an invented function presented as a port

edgeR has no `diffSpliceDGE`; limma's `diffSplice` takes a fitted `MArrayLM`.
edgePython's version runs `exact_test` on exon counts and aggregates by Simes,
which is neither limma's nor edgeR's procedure.

`edge-rs` keeps the name as the `DgeList` wrapper around `diff_splice`, standing
in the same relation to it that `glm_fit_dge` does to `glm_fit`, and does the
edgeR procedure underneath. It also uses first-appearance gene order, where
edgePython's `np.unique` silently sorts.

---

## 21. `lmFit` loses estimability, degrees of freedom and the unscaled errors

**Where:** `edgepython/voom_lmfit.py`, `_lm_fit` and `_row_lm_fit_with_missing`
**Ported to:** `src/limma/lm_fit.rs`
**Severity:** several independent problems, one of them structural

- **`stdev_unscaled` is not computed at all.** It is the one quantity `eBayes`
  cannot be built without, so this is a hole rather than a disagreement.
- **Rank deficiency is resolved with a pseudo-inverse**, giving a minimum-norm
  solution in which every coefficient is finite. limma pivots and returns `NA`
  for the aliased column. The coefficients are different numbers, not a rounding
  difference, and no coefficient is flagged as inestimable.
- **Zero-weight observations are clipped rather than dropped**, so a sample that
  should contribute nothing still consumes a residual degree of freedom. On the
  test fixture that is `df_residual` of 3 where limma gives 2, and it moves
  everything that divides by it.
- **`correlation` is silently clamped to plus or minus 0.95.** limma errors at
  `|r| >= 1` and, through `chol`, also refuses a correlation that makes the
  block covariance indefinite. edgePython reaches for a pseudo-inverse and
  returns a regularised fit for a model that does not exist.
- Per-gene rank comes from an SVD rather than the pivoted QR, so it differs near
  the boundary and has no notion of *which* column is aliased.

`edge-rs` follows limma, and reproduces its choice of aliased column.

---

## 22. `voom` uses one smoother where limma and edgeR use two

**Where:** `edgepython/voom_lmfit.py:418` (`_weighted_lowess_trend`) and `voom`
**Ported to:** `src/limma/voom.rs`
**Severity:** the same class of error as entry 16, in a second place

limma's `voom` smooths the mean-variance trend with Cleveland's
`stats::lowess(f = span, iter = 3)`. edgeR's `voomLmFit` uses the same, **except**
once structural zeros are detected, when it switches to `weightedLowess` with the
residual degrees of freedom as weights.

edgePython uses `weighted_lowess` unconditionally, and with `npts = 120` and 3
iterations rather than limma's 200 and 4. Two different mistakes in one call.

`edge-rs` dispatches as upstream does. Note the iteration convention trap
recorded in entry 16: R counts robustness passes after the initial fit, limma
counts total passes.
---

## 23. `duplicateCorrelation` uses a moment estimator, not a mixed model

**Where:** `edgepython/voom_lmfit.py:898` (`duplicate_correlation`)
**Ported to:** `src/limma/array_weights.rs`, `duplicate_correlation`
**Severity:** high. A different estimator, agreeing only in the simplest case.

limma fits a REML mixed model per gene through `statmod::mixedModel2Fit` and
takes a trimmed mean of the Fisher z-transformed correlations. edgePython
substitutes a one-way ANOVA moment estimator, the classical intraclass
correlation. The two agree only for balanced blocks with no fixed effect beyond
an intercept, which is the case nobody needs the function for.

On top of the estimator itself:

- per-gene correlations are clipped to `[-0.99, 0.99]` rather than to limma's
  block-size bound, which is what keeps the block covariance positive definite
- the `nblocks < n_obs - 1` and `n_obs > n_coef + 2` admission rules are dropped
  on the vectorised path, so genes limma excludes are averaged in
- there is no check for a block factor already spanned by the design, and none
  for all-singleton blocks. Both are exact-zero returns in limma.

`edge-rs` ports limma, including `glmgam.fit` underneath, and agrees to 5e-15.

`arrayWeights` in the same file has its own set:

- **`method = "reml"` with gene-level weights silently drops the weights.**
  limma has a third routine, `.arrayWeightsPrWtsREML`, for exactly that case;
  edgePython's REML function takes no `weights` argument at all. Masked in
  practice by limma's `auto` rule rather than fixed.
- a fully pivoted QR reorders design columns by magnitude, where R moves only
  negligible columns and only to the end, so a rank-deficient design keeps a
  different subset in a different order
- `pinv(..., rcond = 1e-12)` where limma uses `solve`, giving a minimum-norm
  answer where limma would stop
- a gene is admitted on `sum(good) > p` where limma also requires two residual
  degrees of freedom
- `nanmean` where limma uses `colMeans`, absorbing NaNs rather than propagating

---

## 24. NEBULA has no marginal-likelihood Hessian, so its standard errors are wrong

**Where:** `edgepython/sc_fit.py`, the whole NEBULA-LN path
**Ported to:** `src/sc/ptmg.rs`, `src/sc/nebula.rs`
**Severity:** the highest in this document. Point estimates survive, inference does not.

The `nebula` package computes standard errors from `ptmg_ll_der_hes_eigen`, the
Hessian of the *marginal* log-likelihood, evaluated at the fitted parameters.
edgePython has no equivalent. `grep trigamma sc_fit.py` returns nothing, and the
sigma-sigma block cannot be formed without `trigamma(cumsumy + alpha)` and
`trigamma(y + phi)`. What it has instead are the Hessians at lines 370-402, which
are the inner Laplace step of `_opt_pml_nb`: the *penalised* Hessian, a different
matrix answering a different question.

Measured against `nebula` 1.5.8 on synthetic data with a genuine subject-level
random effect:

| | logFC (worst / median) | standard error (worst / median) |
|---|---|---|
| 6 subjects | 1.9% / 0.16% | 289% / 24% |
| 20 subjects | 0.07% / 0.02% | 89% / 6.2% |

Log fold changes converge as subjects are added. Standard errors do not, and they
are what produces the p-values. A NEBULA that cannot be used for inference is not
NEBULA.

`edge-rs` therefore ports phase 8 from the `nebula` package's own C++
(`src/optimization.cpp`) and R driver, not from edgePython. See the decision log
in `plans/delegated-cuddling-riddle.md`.

A second, smaller problem in the same area: edgePython hand-rolls `_digamma_nb`
as an asymptotic series with a recurrence shift to `x >= 7`, documented as "~15
digits". At the lower `sigma` bound `alpha_pr` reaches `1e8` and `alpha_pr^2`
reaches `1e16`, so a one-ulp digamma error becomes an O(1) error in the sigma
gradient. `edge-rs` uses `numeric::gamma`, which matches base R bit for bit at
those arguments.

---

## 25. `_opt_pml_nb` adds a ridge, a floor and a clamp that the C++ does not have

**Where:** `edgepython/sc_fit.py`, `_opt_pml_nb`
**Ported to:** `src/sc/pml.rs`
**Severity:** high. Seven items, and every one of them moves the information
matrix, which is to say the standard error.

1. **A ridge on the information matrix.** `if abs(vb2[ii, ii]) < 1e-10: vb2[ii, ii] += 1e-8`.
   Nothing like it in the C++. It perturbs exactly the matrix that gets inverted
   for the standard errors.
2. **`vw` floored at `1e-15`**, in both the main loop and the `ord` block. The C++
   never floors it. The floor shifts `dw/vw`, `vwb/vw`, the Schur complement and
   the log-determinant together.
3. **REML is dropped.** edgePython takes `logdet = sum(log(max(|vw|, 1e-300)))`
   and ignores its own `reml` argument. The C++ adds `log|det(vb2)|` when
   `reml == 1`. Under `reml = 1` the outer overdispersion objective is a different
   function.
4. **The linear predictor is clamped at 500** before `exp`. The C++ does not clamp;
   it relies on `isinf(loglik)` to reject the step during backtracking. Clamping
   both changes the surface and defeats the rejection test, so the damping path
   diverges rather than merely rounding differently.
5. **The Schur complement is formed as `vb - vwb' diag(1/vw) vwb`** where the C++
   forms `temp = vwb / sqrt(vw); vb - temp'temp`. Algebraically identical,
   numerically not, and only the C++ form is symmetric by construction.
6. **`np.linalg.solve`** (partial-pivot LU) where Eigen uses `LDLT` with symmetric
   pivoting and tiny pivots zeroed. The two disagree on near-singular systems,
   which is precisely the population the `-25` convergence code exists to flag.
7. `extb < 1e-300 -> 0` guards. Harmless: IEEE reaches the same limit, and
   `extb + gamma` with `gamma > 0` never trips the second one. Listed for
   completeness.

`edge-rs` follows the C++ and agrees with `nebula:::opt_pml` to 1e-11 on `beta`,
`log_w`, the information matrix, `loglik`, `logdet` and `second`.

One upstream quirk worth recording rather than fixing: `sec_ord` in the C++
recomputes `vw` at the final iterate *after* `logdet` has already been taken from
the previous one. The outer objective is calibrated against that, so both
edgePython and `edge-rs` keep it.

---

# Section B: edgeR or limma are themselves wrong

---

## 11. `aveLogCPM` corrupts the first gene when the offset is a matrix

**Where:** edgeR 4.8.2, `aveLogCPM` with a matrix `offset`
**Affects:** `src/core/expression.rs`, `ave_log_cpm`
**Severity:** the first gene's abundance is wrong by around twenty on the log2
scale, so roughly a millionfold in CPM

`aveLogCPM(y, offset = log(lib))` and
`aveLogCPM(y, offset = matrix(log(lib), nrow(y), ncol(y), byrow = TRUE))` describe
exactly the same model, so they must agree. They agree on every gene but the
first.

```
y <- matrix(c(10,12,11,40,44,38, 50,48,52,49,51,50, 2,0,5,1,3,0), nrow = 3, byrow = TRUE)
ls <- c(1e6, 1.2e6, 0.9e6, 1.1e6, 1e6, 1.3e6)
aveLogCPM(y, offset = log(ls), dispersion = 0.1)
#  4.675093  5.605251  1.847387
aveLogCPM(y, offset = matrix(log(ls), 3, 6, byrow = TRUE), dispersion = 0.1)
# 24.622730  5.605251  1.847387
```

Reproduced independently on an 8 by 5 Poisson matrix: again only row 1 differs,
4.838 against 24.708, and it is unaffected by `prior.count`. The first row of a
matrix offset is evidently not reaching the C routine intact.

`edge-rs` treats the vector and matrix forms as the same model, because they are.
Every gene agrees with edgeR's *vector*-offset answer. The matrix-offset test
therefore uses the vector call as its reference for gene 1 and edgeR's matrix
call for the rest, with a comment saying why.

**This is not confined to `aveLogCPM`.** `glmQLFit` computes its own `AveLogCPM`
internally and hands it to `squeezeVar` as the trend covariate, so a matrix
offset corrupts the quasi-likelihood prior for *every* gene, not just the first.
On an eight-gene fixture at `dispersion = 0.1`:

| offset form | `s2.prior` |
|---|---|
| none, or a vector | 3.90e-4, 4.08e-5, 2.09e-4, 1.31e-1, ... |
| the equivalent matrix | 2.61e+0, 1.10e-4, 4.79e-5, 6.85e-5, ... |

The first gene's abundance comes out as 26.32 instead of 16.35, which drags the
fitted trend across the whole abundance range. Confirmed by feeding limma's own
`squeezeVar` the two covariates: the matrix one reproduces edgeR's output
exactly, the vector one reproduces `edge-rs`'s.

Everything upstream of the covariate agrees to 1e-9 either way, including the
adjusted deviances and degrees of freedom, so this is entirely the covariate.

Worth reporting upstream, and worth knowing if you compare against published
edgeR numbers from a pipeline that passes offsets as a matrix, which
`voomLmFit` and any custom normalisation both do.

---

## 12. `fitFDistRobustly` does not converge its own root solve

**Where:** limma 3.66, `fitFDistRobustly`
**Affects:** `src/limma/squeeze_var.rs`, `fit_f_dist_robustly`
**Severity:** around 1e-4 relative on `df2`, so a slightly wrong prior

limma solves for `df2` with `uniroot(..., tol = 1e-8)` in the `d / (1 + d)` link.
That tolerance is on the transformed variable, and at a `df2` near 100 it leaves
the answer roughly 1e-4 out. limma's returned value sits 2.5e-8 away from the
root of limma's own objective function.

Confirmed by re-running limma with its `uniroot` shadowed at `tol = 1e-14`: it
then lands on `edge-rs`'s values to 5e-14. So this is limma stopping early, not
a difference in the objective.

`edge-rs` converges the root properly, at `xtol = 1e-15`. The robust reference
values in the test suite are therefore limma-with-converged-uniroot, with the
override script recorded in the test doc comment, plus one test that pins stock
limma at 1e-7 and asserts `edge-rs` is strictly closer to the true root than
stock limma is.


---

## 17. `exactTestBySmallP` returns one p-value for the whole matrix

**Where:** edgeR 4.8.2, `exactTestBySmallP`, final line
**Affects:** `src/exact/mod.rs`, the `SmallP` rejection region
**Severity:** severe. Every gene reports the same p-value.

The function ends with `min(pvals, 1)` where it needs `pmin(pvals, 1)`. In R
`min` reduces a vector to a scalar, so a matrix of genes comes back as a single
number: the smallest p-value found anywhere in the matrix.

It is masked for balanced designs, because an earlier line returns to
`exactTestDoubleTail` when `n1 == n2`. With unequal group sizes it bites:

```
y1 <- 6 genes by 2 samples, y2 <- the same 6 genes by 4 samples
exactTestBySmallP(y1, y2, dispersion = 0.1)
# 0.0002999433935          <- length 1, not 6

# the same genes one at a time:
# 0.0004995834196  1  1  0.0002999433935  1  0.9092143135841
```

Visible to users through the documented entry point:

```
exactTest(d, rejection.region = "smallp")   # unequal group sizes
# PValue: 0.008179571648 repeated for all six genes, 1 distinct value of 6
```

`edge-rs` caps each gene on its own value. Worth reporting upstream.

---

## 18. `binomTest` mishandles exact ties

**Where:** edgeR 4.8.2, `binomTest`
**Affects:** `src/exact/mod.rs`, the zero-dispersion limit
**Severity:** a p-value of 0.86 where the answer is 1

When `p = 1/3` and the total is `3k + 2`, the outcomes `k` and `k + 1` are
exactly equiprobable. edgeR's `order`/`cumsum` construction keeps or drops the
tied term on a one-ulp difference in `dbinom`:

```
binomTest(12, 23, p = 1/3)      # 0.8600904883
binom.test(12, 35, 1/3)$p.value # 1
binomTest(10, 20, p = 1/3)      # 1, so it is tie-specific, not systematic
```

`edge-rs` uses `binom.test`'s `1 + 1e-7` relative slack on the tie comparison,
which is the standard remedy and gives 1 here.

There is a third, milder issue not reproduced either: above a total of 10000
`binomTest` switches to a chi-square shortcut whose 2 by 2 table is built from
the column totals of whatever subset of genes was passed in, so one gene's
p-value depends on which other genes accompanied it. `edge-rs` enumerates
exactly at every size.
