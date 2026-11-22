# Upstream deviations

Where `edge-rs` deliberately does not reproduce its upstreams, and why.

- **Section A: edgePython is wrong.** The port follows
  [`edgePython`](https://github.com/pachterlab/edgePython), but where it
  disagrees with edgeR, limma or `nebula`, upstream wins. Those are what users
  compare against and what the fixtures gate on.
- **Section B: edgeR, limma or nebula are wrong.** Rarer. Each fault is reproduced in
  the installed package. Copying a bug faithfully is not parity worth having.

Every entry was checked against the installed package, not inferred from
reading, and names a test that fails if the behaviour drifts back. Section A
was last checked against edgeR 4.10.7, limma 3.68.5 and nebula 1.5.9, Section B
against the versions each entry names. The committed fixtures still come from
edgeR 4.8.2 and limma 3.66.0.

edgePython citations are against **version 0.5.1, commit `4be67c8`**
(2026-10-24). A line number without a commit rots. Entries edgePython has since
fixed are removed; the remaining IDs keep their numbers because the source
cites them. Re-pin this paragraph if you re-check the entries against a newer
checkout.

## Index

| ID | Summary | Severity | Ported to |
|---|---|---|---|
| [A4](#a4-the-integer-division-in-the-poisson-regime-is-corrected) | Integer `2/3` in edgeR's C "corrected" | low | `glm/deviance.rs` |
| [A6](#a6-natural_spline_basis-is-not-rs-ns) | `natural_spline_basis` is not R's `ns()` | none | `numeric/interpolate.rs` |
| [A9](#a9-trigamma_inverse-uses-the-wrong-large-x-asymptotic) | `trigamma_inverse` large-x asymptotic | low | `numeric/gamma.rs` |
| [A10](#a10-squeezevar-and-fitfdist-clamp) | `fitFDist` and `squeezeVar` clamp | low | `limma/squeeze_var.rs` |
| [A12](#a12-lmfit-loses-estimability-and-df) | `lmFit` loses estimability and df | high | `limma/lm_fit.rs` |
| [A15](#a15-the-deviance-and-small-p-rejection-regions-are-stubs) | Exact test regions are stubs | high | `exact/mod.rs` |
| [A17](#a17-splicevariants-is-a-different-test) | `spliceVariants` is a different test | high | `splicing.rs` |
| [A18](#a18-diffsplicedge-is-not-edgers-procedure) | `diffSpliceDGE` is not edgeR's procedure | high | `splicing.rs` |
| [A19](#a19-glm_sc_test-drops-the-off-diagonal-covariance) | Single-cell contrasts drop covariance | high | `sc/test.rs` |
| [A20](#a20-nebula-has-no-marginal-likelihood-hessian) | NEBULA has no marginal Hessian | highest | `sc/ptmg.rs`, `sc/nebula.rs` |
| [A21](#a21-_opt_pml_nb-adds-a-ridge-a-floor-and-a-clamp) | `_opt_pml_nb` ridge, floor and clamp | high | `sc/pml.rs` |
| [B2](#b2-fitfdistrobustly-stops-its-root-solve-early) | `fitFDistRobustly` stops `uniroot` early | low | `limma/squeeze_var.rs` |
| [B3](#b3-exacttestbysmallp-returns-one-p-value-for-every-gene) | `exactTestBySmallP` uses `min` for `pmin` | severe | `exact/mod.rs` |
| [B4](#b4-binomtest-mishandles-exact-ties) | `binomTest` mishandles exact ties | medium | `exact/mod.rs` |
| [B5](#b5-nebula-ln-stage-one-stops-on-a-spurious-sigma-bound) | NEBULA-LN stage one stops on a spurious `sigma` bound | high | `sc/nebula.rs` |

---

# Section A: edgePython disagrees with upstream

## GLM and deviance

### A4. The integer division in the Poisson regime is "corrected"

- **Upstream:** `edgepython/glm_kernels.py:34` (`unit_nb_deviance`) and `edgepython/ql_weights.py:484` (`compute_unit_nb_deviance`)
- **Ported to:** `src/glm/deviance.rs`, `unit_nb_deviance`
- **Severity:** up to 7e-4 relative on large counts at small dispersion
- **Test:** `test_unit_deviance_matches_edger` (`src/ql/chebyshev.rs`)

The Poisson-regime correction in edgeR's C reads

```c
2 * (y * log(y/mu) - resid - 0.5*resid*resid*phi*(1 + phi*(2/3*resid - y)))
```

`2/3` is integer division, so it is zero and the term is `-phi*y`. edgePython
writes `2.0 / 3.0` in both of its unit deviances, and since 0.5 the one in
`glm_kernels.py` also drives the Levenberg fit and `nbinom_deviance`. Intended or
not, the C is what every published edgeR result reflects, and edgeR 4.10.7
still gives the integer-division values. Against `nbinomUnitDeviance`:

| y | mu | phi | edgeR | integer `2/3` | `2.0/3.0` |
|---|---|---|---|---|---|
| 1e6 | 1.001e6 | 1e-5 | 90.999333833 | 90.999333833 | 91.0660004996 |
| 1e6 | 1e6 + 1 | 1e-6 | 9.99899558621e-07 | 9.99899558707e-07 | 9.99900225374e-07 |
| 100 | 110 | 1e-5 | 0.936965039047 | 0.936965039047 | 0.936965105714 |
| 0 | 10 | 1e-5 | 19.9989995855 | 19.9989995855 | 19.9989996522 |

Only visible when `phi` is under the 1e-4 threshold *and* counts are large. A
test at `phi = 1e-10` passes either way. `edge-rs` reproduces the C.

## Smoothing and limma

### A6. `natural_spline_basis` is not R's `ns()`

- **Upstream:** `edgepython/limma_port.py:281` (`_natural_spline_basis`)
- **Ported to:** `src/numeric/interpolate.rs`, `natural_spline_basis`
- **Severity:** none in practice
- **Test:** `test_natural_spline_basis_spans_the_same_space_as_r_ns`

edgePython builds a truncated power basis; `splines::ns()` is a constrained
B-spline basis. Different matrices, same span. The basis only ever feeds a
least-squares or GLM fit, so fitted values agree to machine precision while
coefficients do not.

`edge-rs` follows the Python and tests span equivalence against `ns()`. Recorded
so nobody "fixes" it with a real `ns()` and wonders why coefficients moved.

### A9. `trigamma_inverse` uses the wrong large-x asymptotic

- **Upstream:** `edgepython/limma_port.py:646` (`_trigamma_inverse`)
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

### A10. `squeezeVar` and `fitFDist` clamp

- **Upstream:** `edgepython/limma_port.py`, `squeeze_var` (line 14) and `_fit_f_dist` (line 142)
- **Ported to:** `src/limma/squeeze_var.rs`
- **Severity:** low; edge cases and genes dropped from the fit
- **Test:** `test_squeeze_var_robust_trended_matches_limma`

Four divergences from limma remain (the robust smoother now matches):

- `_fit_f_dist` clamps `df2 >= 1e-6`, treats `> 1e15` as infinite and floors the
  scale at `1e-15`. limma does none of this.
- `squeeze_var` sets `var[df == 0] <- 0` always; limma only with more than one `df`.
- Genes dropped from the trended fit get the prior by linear interpolation
  (extrapolating in `squeeze_var`, `rule = 2` in the robust fit) where limma
  predicts from the spline.
- Log tail probabilities are clipped to `[-500, 0]` before `qf` in the
  unequal-`df1` step of the robust fit; limma passes them through with
  `log.p = TRUE`.

`edge-rs` follows limma throughout, using the Cleveland `lowess` port in
`src/limma/lowess.rs`, and agrees to 1e-12. One residual gap: dropped genes
(zero `df1` or non-finite variance) are read off the fitted trend rather than
via `predict.ns`, because the A6 basis exposes no knots. That costs about 1e-3
on those genes only.

### A12. `lmFit` loses estimability and df

- **Upstream:** `edgepython/voom_lmfit.py:253` (`_lm_fit`) and `:145` (`_row_lm_fit_with_missing`)
- **Ported to:** `src/limma/lm_fit.rs`
- **Severity:** high; several independent problems
- **Test:** `test_aliased_design_reports_rank_and_pivot`, `test_zero_weight_drops_the_observation`

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

## Exact test

### A15. The deviance and small-p rejection regions are stubs

- **Upstream:** `edgepython/exact_test.py:288` and `:296`
- **Ported to:** `src/exact/mod.rs`
- **Severity:** two of three rejection regions do the wrong test
- **Test:** `test_deviance_matches_edger_on_unequal_groups`, `test_small_p_matches_edger_called_one_gene_at_a_time`

`exact_test_by_deviance` and `exact_test_by_small_p` both call
`exact_test_double_tail`. On a six-gene fixture at dispersion 0.1, gene 1:
doubletail 1.088e-05, deviance 8.134e-06, smallp 1.181e-05.

edgeR's `exactTestBySmallP` itself returns to `exactTestDoubleTail` when
`n1 == n2`, so the stub is right for balanced designs only. `edge-rs` implements
all three.

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

# Section B: edgeR, limma or nebula are wrong

### B2. `fitFDistRobustly` stops its root solve early

- **Upstream:** limma 3.66 and 3.68.5, `fitFDistRobustly`
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

- **Upstream:** edgeR 4.8.2 and 4.10.7, `exactTestBySmallP`, final line
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

- **Upstream:** edgeR 4.8.2 and 4.10.7, `binomTest`
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

### B5. NEBULA-LN stage one stops on a spurious `sigma` bound

- **Upstream:** nebula 1.5.9, `R/nebula.R`, the stage-one `lbfgs` on `ptmg_ll_der`
- **Ported to:** `src/sc/nebula.rs`, `minimise_marginal`
- **Severity:** up to one gene in a hundred in simulation keeps `sigma^2 = 1e-4` where the interior
  optimum is far better; the subject effect is dropped and the SE shrinks up to
  3.6x on simulated data
- **Test:** `test_stage_one_restarts_off_a_spurious_sigma_bound`

The marginal likelihood can rise off the lower bound on `sigma` before falling
into an interior minimum. Any step from `sigma = 1` that jumps the interior basin
lands on the bound, where the outward gradient holds it. `kappa_obs` is above 800
on these genes, so the sigma-only HL refit never fires. nebula's own objective
shows it. Simulated, 130 subjects of 280 cells:

```
# nebula's lbfgs, nebula's setup, by start value of sigma
gene 148  start 1     -> sigma 1e-4     nll -135420.547
gene 148  start 1e-3  -> sigma 3.158e-3 nll -135454.927
gene 246  start 1     -> sigma 1e-4     nll  -29132.124
gene 246  ptmg_ll at the edge-rs point (sigma 4.663e-3):  -29164.884
```

Simulated `sigma^2` was 3.1e-3 and 6.0e-3. The same trap catches the port's own
projected Newton more often than nebula's L-BFGS, since its first step is longer.

`edge-rs` restarts stage one, up to three times, from the geometric mean of the
bound and the previous start, and keeps the lowest objective. That fixes every gene
where nebula finds the interior and the port did not, and also moves genes off the
bound where nebula stays on it. The second kind disagree with R by design.
