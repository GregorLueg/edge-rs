# edge-rs: porting edgePython to Rust

## Context

`edgePython` (`~/repos/others/edgePython`, v0.2.6) is Pachter's Python port of edgeR, which
itself is R plus C++. It carries the edgeR numerical stack (TMM, Cox-Reid dispersion, the
Levenberg-damped NB GLM, quasi-likelihood weights, the exact test), a large slice of limma
(`squeezeVar`, the `fitFDist` family, voom, `duplicateCorrelation`), and NEBULA-LN, a
negative-binomial gamma mixed model for single cell.

Python is the wrong floor for this. The hot paths are per-gene fits over a small design
matrix repeated tens of thousands of times, and edgePython already reaches for `numba` in
seven modules to make them tolerable. That is the tell: those loops want a compiled
language with SIMD and real thread fan-out.

The destination is `bixverse-rs`, which already has a single-cell streaming engine but
nothing model-based for differential expression (only Mann-Whitney and AUROC). NEBULA wired
into that engine gives mixed-effect DE over single-cell data without leaving Rust.

### Decisions taken

- Standalone crate in this repo. No Python bindings, Rust API only.
- Full numerical surface, delivered in phases.
- Public API generic over an `EdgeFloat` trait; likelihood evaluation, Cox-Reid
  determinants, optimisers and p-values run in `f64` internally and are documented as such.
- Hand-rolled box-constrained L-BFGS-B (Byrd, Lu, Nocedal and Zhu), no FFI, no `argmin`.
- NEBULA validated against the R `nebula` package.

### Hard constraint: no dependency on bixverse-rs

bixverse-rs will consume edge-rs, so edge-rs must not depend on it. That rules out reusing
`BixverseFloat`, `BixverseSimd`, `CompressedSparseData2`, `BixverseErrors` or
`SingleCellReading` directly. edge-rs defines its own equivalents, modelled on the house
versions so the later wiring is mechanical. The sparse type follows the `manifolds-rs`
`CompressedSparseData<T>` shape with `u32` indices as in bixverse.

The practical consequence for NEBULA: its entry point takes plain per-gene slices
(`indices: &[u32]`, `values: &[T]`) plus shared design, offset and subject-boundary arrays.
bixverse then feeds it straight from `CscGeneChunk` (already gene-major CSC, `u32` indices,
raw counts via `DataLayerReturn::Raw`) with no adapter type in between.

### Out of scope

Plotting (`visualization.py`, 409), format readers (`io.py`, 1887), pathway enrichment
(`gene_sets.py`, 1217), biomart and TSS lookups in `utils.py`, and the MCP server. bixverse
already covers h5ad/10x/mtx reading and GO enrichment. That drops roughly 3.5k of 17.4k
lines.

## Source map

| Python source | Lines | Rust destination |
|---|---|---|
| `glm_levenberg.py` | 375 | `glm/levenberg.rs`, `glm/deviance.rs` |
| `glm_fit.py` | 661 | `glm/one_group.rs`, `glm/one_way.rs`, `glm/fit.rs` |
| `glm_test.py` | 398 | `glm/test.rs` |
| `dispersion_lowlevel.py` | 1207 | `dispersion/apl.rs`, `dispersion/cox_reid.rs`, `numeric/interpolate.rs` |
| `dispersion.py` | 1125 | `dispersion/estimate.rs`, `dispersion/robust.rs` |
| `ql_weights.py` | 765 | `ql/chebyshev.rs`, `ql/weights.rs` |
| `limma_port.py` | 987 | `limma/squeeze_var.rs`, `numeric/gamma.rs` |
| `voom_lmfit.py` | 1371 | `limma/lm_fit.rs`, `limma/voom.rs` |
| `smoothing.py` | 474 | `limma/smoothing.rs` |
| `weighted_lowess.py` | 323 | `limma/lowess.rs` |
| `normalization.py` | 548 | `core/normalisation.rs` |
| `exact_test.py` | 534 | `exact/mod.rs` |
| `sc_fit.py` | 1554 | `sc/nebula.rs`, `sc/pml.rs`, `sc/shrink.rs`, `sc/test.rs` |
| `compressed_matrix.py` | 388 | `utils/recycled.rs` |
| `classes.py`, `dgelist.py` | 834 | `core/dgelist.rs` |
| `expression.py`, `filtering.py` | 419 | `core/expression.rs`, `core/filtering.rs` |
| `utils.py` | 1050 | `utils/*`, minus network lookups |
| `splicing.py` | 538 | `splicing.rs` |
| `results.py` | 236 | `results.rs` |

## Architecture

```
src/
  lib.rs                  // #![warn(missing_docs)], module docs
  prelude.rs
  errors.rs               // EdgeErrors, one thiserror enum, sectioned by subsystem
  utils/
    traits.rs             // EdgeFloat, EdgeSimd
    simd.rs               // wide-backed kernels + runtime dispatch
    recycled.rs           // Recycled<T> (the compressedMatrix equivalent)
    sparse.rs             // CompressedSparse<T>, u32 indices, CSR/CSC + transpose
    design.rs             // rank, QR, non_estimable, is_fullrank, contrast_as_coef
  numeric/
    gamma.rs              // lgamma, digamma, trigamma, polygamma, trigamma_inverse, logmdigamma
    dist.rs               // chi2/norm/t/F/beta/gamma/nbinom tails via statrs, f64
    optimise.rs           // Brent minimise + root, Nelder-Mead, L-BFGS-B
    interpolate.rs        // fmm_spline, maximize_interpolant, natural spline basis, interp1d
    quadrature.rs         // adaptive Gauss-Kronrod (robust dispersion only)
    stats.rs              // BH adjust, rankdata, trimmed mean, moving average
  core/
    dgelist.rs            // DgeList<T>
    normalisation.rs      // TMM, TMMwsp, RLE, upperquartile
    expression.rs         // cpm, rpkm, tpm, ave_log_cpm
    filtering.rs          // filter_by_expr
  glm/                    // one_group, one_way, levenberg, deviance, fit, test
  dispersion/             // apl, cox_reid, estimate, robust
  ql/                     // chebyshev, weights
  limma/                  // squeeze_var, lowess, smoothing, lm_fit, voom
  exact/
  sc/                     // nebula, pml, shrink, test
  results.rs
  splicing.rs
```

### Key types

`EdgeFloat`, mirroring `AnnSearchFloat`/`BixverseFloat`:

```rust
pub trait EdgeFloat:
    Float + FromPrimitive + ToPrimitive + Send + Sync + Sum
    + AddAssign + SubAssign + MulAssign + DivAssign
    + EdgeSimd + ComplexField + RealField + TotalOrder + Display + 'static
{}
```

Blanket impl, no required methods, exactly as the sister crates do it.

`Recycled<T>` replaces `CompressedMatrix`. edgePython expands it to a dense matrix almost
everywhere, which throws away the whole point. In Rust it stays compressed and the consumers
index through it:

```rust
pub enum Recycled<T> {
    Scalar(T),
    ByGene(Vec<T>),     // length n_genes
    BySample(Vec<T>),   // length n_samples
    Full(Vec<T>),       // n_genes * n_samples, row-major
}
```

Offsets, weights, dispersions and prior counts all flow through this. For a 30k gene by
200k cell single-cell run, holding offsets as `BySample` rather than `Full` is the
difference between a few megabytes and terabytes.

`EdgeSimd`, the crate's own SIMD trait backed by `wide` with runtime dispatch, in the shape
of `BixverseSimd`. Members are chosen from the actual inner loops, not from a generic
distance kernel:

- `unit_nb_deviance_sum`, the fused `y log(y/mu)` and `(y + 1/phi) log(...)` accumulation
- `xtwx_accumulate`, the working-weight outer product for small `p`
- `exp_mul_add`, computing `mu = exp(offset + X beta)` over a row
- `lgamma_sum`, `digamma_sum`, used by the APL and the NEBULA gradient
- `dot`, `sum`, `axpy` as the general fallbacks

### Data layout

Counts are genes by samples, row-major, so one gene is a contiguous slice. That matches the
parallel axis for every algorithm in the crate and matches edgePython's own internal
convention. Bulk uses dense `Vec<T>`; single cell uses `CompressedSparse<T>` in CSC form
(gene-major), which is already how bixverse stores its chunks, so no transpose is needed at
the boundary.

## Performance strategy

**Per-gene rayon, not batched GEMM.** edgePython's `mglm_levenberg` batches all genes into
one `(n_active, nlibs) @ (nlibs, p*p)` matmul plus a batched solve, because that is the only
way to go fast in NumPy. In Rust the better shape is `par_chunks` over genes with
`for_each_init` holding a per-thread scratch buffer. A gene's working set is `n` samples by
`p` coefficients, which sits in L1, and the `p x p` normal-equation solve is a hand-rolled
fixed-size Cholesky on a stack array rather than a faer call. faer has real per-call
overhead at `p = 3`. Keep a documented `const` threshold above which the port switches to
faer's `llt`.

**faer where it earns its place.** The Cox-Reid `XtWX` assembly is a 3-operand einsum in
edgePython (`dispersion_lowlevel.py` lines 181, 217, 323) and is the obvious fusion target.
The `-0.5 log|XtWX|` determinant needs an LDL fallback for near-singular genes to match
edgeR's C, which uses LDL throughout. `limma/lm_fit.rs` and the voom path want faer QR
properly.

**Fuse the grid.** `adjusted_profile_lik_grid` evaluates 21 dispersions across all genes.
edgePython loops the grid outside and vectorises genes inside, materialising a
`(ngenes, nsamples, ngrid)` `gammaln` broadcast in `_cond_log_lik_grid`. Invert it: one
parallel pass over genes, the 21 grid points in the inner loop, nothing intermediate
materialised.

**NEBULA fan-out.** The top-level gene loop is embarrassingly parallel with no shared
mutable state. Each gene reads the shared design, log-offsets and subject boundaries, plus
its own count row. Straight `par_iter`. The inner cost is `_ptmg_negll_and_grad` at roughly
100 to 200 calls per gene and `_compute_pml_loglik` at 50 to 500, so those two get the SIMD
attention.

**Keep the full covariance.** `glm_sc_test` builds contrast standard errors as
`sqrt(sum se^2 c^2)`, which is only correct when the covariance is diagonal. It is that way
because `glm_sc_fit` discards everything but the diagonal. The Rust port keeps the `p x p`
inverse Fisher information per gene, so contrast tests are correct. This is a deliberate
divergence from edgePython and needs a test asserting the difference on a correlated design.

## Phases

Each phase ends green against its fixtures before the next starts.

**Phase 0, scaffolding.** Cargo manifest, `lib.rs`, `prelude`, `EdgeErrors`, `EdgeFloat`,
`EdgeSimd` plus `wide` impls, `Recycled<T>`, `CompressedSparse<T>`, CI callers, fixture
loading helper.

**Phase 1, `numeric/`.** `lgamma`, `digamma`, `trigamma`, `polygamma`, `trigamma_inverse`,
`logmdigamma`. Distribution tails through `statrs`. Brent bounded minimisation and root
finding, Nelder-Mead, and the L-BFGS-B. `fmm_spline` and `maximize_interpolant` (edgeR's
`find_max`: locate the coarse grid maximum, then solve the quadratic derivative root on the
two neighbouring segments analytically). Natural spline basis, linear interpolation,
Gauss-Kronrod quadrature, BH adjustment, rank with average ties. Validated against values
dumped from scipy.

**Phase 2, containers and normalisation.** `DgeList<T>`, TMM, TMMwsp, RLE, upperquartile,
cpm/rpkm/tpm/`ave_log_cpm`, `filter_by_expr`. TMMwsp uses a lexsort tie-break, so ordering
must match exactly. Gates on `R_norm_*.csv` and `R_dgelist_*.csv`.

**Phase 3, GLM core.** `mglm_one_group` (NB Fisher scoring with a shrinking active mask),
`mglm_one_way`, `mglm_levenberg`, `nbinom_deviance`. `glm_fit` including the `design_as_factor`
one-way dispatch and `pred_fc` shrinkage. This is the hot path, so it gets the SIMD kernels
and a bench.

**Phase 4, dispersion.** Cox-Reid APL with the LDL determinant fallback, the grid, `WLEB`,
`estimate_disp` in both classic and GLM branches, common/trended/tagwise, the locfit and
loess column smoothers, `weighted_lowess`, the spline/power/bin trend fitters, robust
dispersion. Gates on the hardcoded R dispersions in `test_dispersion.py`.

**Phase 5, quasi-likelihood and `squeezeVar`.** The Chebyshev tables from edgeR's
`ql_weights.c`, `compute_adjust_vec`, `compute_prior`, `update_prior`, `squeeze_var` and the
`fitFDist` family, then `glm_ql_fit`, `glm_lrt`, `glm_ql_ftest`, `glm_treat`. Gates on
`R_treat_*.csv` and the hoxa1/mammary goldens.

**Phase 6, exact test and results.** The per-gene NB convolution with the log-sum-exp work
buffer, the beta approximation above `big_count`, `q2q_nbinom`, `equalize_lib_sizes`,
`top_tags`, `decide_tests`.

**Phase 7, voom and lmFit.** `lm_fit`, row-wise fitting with missing values,
`array_weights` in both gene-by-gene and REML forms, `duplicate_correlation`, `voom`,
`voom_lmfit`. Gates on the voom parity fixtures.

**Phase 8, NEBULA.** `_center_design`, pseudobulk TMM offsets, `cumsumy`/`posindy`
precomputation, the stage 1 marginal MLE via L-BFGS-B, the stage 2 PML Newton solver with
the Schur complement and backtracking, the third and fourth order Laplace corrections for
low-expression genes, the convergence codes, `shrink_sc_disp`, `glm_sc_test` with full
covariance. Note that only NEBULA-LN exists in edgePython; NEBULA-HL is detected and skipped
with a tracking code, so the port matches that and leaves HL for later.

**Phase 9, splicing.** `diff_splice`, `diff_splice_dge`, `splice_variants`.

## Execution model

The port is large enough to delegate, but only in the places where delegation is safe. The
split is by coupling, not by size.

**I write the spine.** Phase 0 first and by hand: `EdgeFloat`, `EdgeSimd`, `EdgeErrors`,
`Recycled<T>`, `CompressedSparse<T>`, and the module skeleton with every public signature
and doc comment already in place. Fixing the interface before anything else is what makes
the fan-out possible at all, and it sets the style template the agents copy rather than
invent.

After that I keep `glm/*`, `dispersion/apl.rs`, `dispersion/estimate.rs`, `ql/weights.rs`,
`limma/squeeze_var.rs`, `limma/lm_fit.rs`, `limma/voom.rs` and `sc/*`. These share types,
and when a dispersion is wrong in the fourth decimal the cause is usually two modules away
from the symptom. All numerical parity debugging stays here for the same reason.

**Agents take the leaves.** Twelve modules are self-contained: they depend only on the
traits, they have a known reference implementation in a single Python file, and their pass
condition is a number rather than a judgement.

| Unit | Reference | Gate |
|---|---|---|
| `numeric/gamma.rs` | `limma_port.py` 603-691 | scipy `special` dumps |
| `numeric/dist.rs` | scipy `stats` call sites | scipy tail dumps |
| `numeric/optimise.rs` | scipy `optimize` call sites | scipy on standard bounded problems |
| `numeric/interpolate.rs` | `dispersion_lowlevel.py` 333-499 | edgeR `maximize_interpolant` |
| `numeric/quadrature.rs` | `dispersion.py` 999 | scipy `quad` |
| `numeric/stats.rs` | `limma_port.py` 693, `normalization.py` 324 | scipy `rankdata`, R `p.adjust` |
| `utils/recycled.rs` | `compressed_matrix.py` | unit tests on recycling semantics |
| `utils/sparse.rs` | `manifolds-rs`, bixverse `sparse.rs` | round-trip and transpose tests |
| `limma/lowess.rs` | `weighted_lowess.py` | limma `weightedLowess` |
| `limma/smoothing.rs` | `smoothing.py` | edgeR `locfitByCol`, `loessByCol` |
| `ql/chebyshev.rs` | `ql_weights.py` 24-530 | edgeR `ql_weights.c` tables |
| `core/normalisation.rs` | `normalization.py` | `R_norm_*.csv` |

Each agent gets its Python source, the `rust-style` skill, the phase 0 trait definitions and
its golden fixture. Agents run several at a time within a phase, never across phases.

**Every return is reviewed and integrated by me.** Nothing an agent produces lands unread.
The two failure modes to watch are style drift across a dozen independent agents, which the
skeleton and the review pass are there to catch, and agents plateauing on iterative
numerical debugging, which is why that work is not delegated in the first place.

## Validation

Three tiers.

**Existing fixtures.** Copy the 115 CSVs from `~/repos/others/edgePython/tests/data/` into
`tests/data/`. Only the `R_*` files matter as truth; the `Py_*` files are frozen Python
output and `test_r_vs_py.py` only ever compares the two files against each other without
running edgepython, so it is not a regression suite. Do not replicate that pattern.

**Hardcoded R values.** The real parity anchors are the literal R numbers in `test_glm.py`,
`test_dispersion.py`, `test_treat.py` and `test_exact_test.py`, for example
`common.dispersion` of `0.210292` on `test_data_part1.csv` and `0.339898452813` on the
two-pass equalisation case. Port these as Rust tests with `approx::assert_relative_eq!` at
the same tolerances the Python tests use, mostly `1e-3` relative, tightening to `1e-8` where
the Python suite does.

**Regenerated goldens.** edgeR 4.8.2 is installed locally, so `scripts/gen_goldens.R` can
produce a richer and version-pinned set from `examples/hoxa1/data` and
`examples/mammary/data`. Record the edgeR version in the fixture header: edgePython targets
an older edgeR, so a mismatch against 4.8.2 may be a genuine upstream change rather than a
port bug. For NEBULA, `nebula` is **not** installed and needs
`install.packages("nebula")` once; `scripts/gen_nebula_goldens.R` then runs it on fixed
simulated data and writes per-gene coefficients, standard errors, sigma, dispersion and
convergence codes.

Benchmarks follow the house convention: `harness = false`, hand-rolled `Instant` timing, no
criterion, one `[[bench]]` block per file. Cover `mglm_levenberg`, the APL grid and NEBULA
per-gene cost, against edgePython wall-clock on the same input.

## CI

Two thin callers, matching `ann-search-rs`:

`.github/workflows/test.yml` calls
`GregorLueg/personal-actions/.github/workflows/rust-test.yml@v1` with `cpu-args: '--release'`
and no `gpu-args`, which skips the GPU lane. `.github/workflows/release.yml` calls
`rust-release.yml@v1` on `workflow_run` of "Test the package", with
`permissions: contents: write` and `secrets: inherit`.

`personal-templates` is for R packages carrying a Rust crate, so it does not apply here.

## Dependencies

`faer` 0.23.2, `faer-traits` 0.23.2, `num-traits` 0.2.19, `rayon` 1.11, `thiserror` 2.0,
`wide` 1.4, `rustc-hash` 2.1, `statrs` 0.18, `rand` 0.9 with `rand_distr` 0.5 for synthetic
data. `approx` 0.5 as a dev dependency. Versions match bixverse-rs so the later merge does
not force a resolver conflict. Edition 2024. Release profile `opt-level = 3`,
`lto = "thin"`, `codegen-units = 4`.

## Risks

- **L-BFGS-B parity.** The generalised Cauchy point and subspace minimisation are fiddly,
  and NEBULA estimates depend on the trajectory, not just the optimum. Mitigation: test the
  solver standalone against scipy on the standard bound-constrained problems before wiring
  it into NEBULA.
- **Cox-Reid determinant.** edgeR uses LDL everywhere; edgePython uses a fast 2x2 path with
  an LDL fallback below `1e-10`. Parity depends on reproducing the fallback threshold.
- **edgeR version drift.** edgePython targets an older edgeR than the locally installed
  4.8.2. Expect some fixtures to disagree for legitimate reasons; pin and record the version.
- **Scale.** Roughly 9.7k lines of dense numerical Python before the excluded modules, so
  expect 25k to 35k lines of documented Rust. The phases are the mitigation, not optimism.

## Verification

```bash
cargo fmt && cargo clippy --all-targets
cargo test --release                       # per-phase gates
cargo test --release -- --nocapture nebula # NEBULA against the R goldens
cargo bench --bench glm_levenberg
```

End to end, the phase 5 gate is the meaningful one: load `test_data_part1.csv`, run
`estimate_disp` then `glm_ql_fit` then `glm_ql_ftest`, and match the R
`common.dispersion`, `LR` and `PValue` columns within the tolerances the Python suite uses.

---

## Decision log: NEBULA is ported from the R package, not from edgePython

Recorded 2026-08-18, after phase 8 began. This supersedes the phase 8 text above.

The plan assumed NEBULA would be a port of `edgepython/sc_fit.py`, validated
against the R `nebula` package. Installing `nebula` 1.5.8 and actually running
that comparison showed the assumption was wrong.

edgePython implements only NEBULA-LN and has no equivalent of the R package's
`ptmg_ll_der_hes_eigen`, the marginal-likelihood Hessian. It derives standard
errors from the penalised Hessian instead. Measured on synthetic data with a
genuine subject-level random effect:

| | logFC (worst / median) | standard error (worst / median) |
|---|---|---|
| 6 subjects | 1.9% / 0.16% | 289% / 24% |
| 20 subjects | 0.07% / 0.02% | 89% / 6.2% |

Log fold changes converge with sample size. Standard errors do not, and they are
what produces the p-values. Porting edgePython faithfully would have shipped a
module that cannot be used for inference, under the name NEBULA.

So phase 8 ports from the `nebula` package's own sources (CRAN tarball, C++ in
`src/optimization.cpp`, driver in `R/nebula.R`), and keeps the **full covariance
matrix** per gene rather than only its diagonal. The latter also resolves
`UPSTREAM_DEVIATIONS.md` entry 2 properly: `covariance = TRUE` in the R package
gives contrast standard errors something real to validate against, instead of
the diagonal-only approximation edgePython is forced into.

Layout: `sc/ptmg.rs` (likelihood, gradient, Hessian), `sc/pml.rs` (the penalised
Newton solver), `sc/nebula.rs` (driver), `sc/test.rs` (Wald tests with `c'Vc`).

Two smaller corrections to the plan while here:

- The `R_*.csv` fixtures copied from edgePython are largely unusable. Their input
  matrices are not in the repo and cannot be regenerated locally. Every module
  has instead been validated against references generated fresh from the
  installed edgeR 4.8.2 and limma 3.66. That has been strictly better, since it
  also pins the package version. They have since been deleted; `tests/data` holds
  only the six files the suite reads.
- Section B of `UPSTREAM_DEVIATIONS.md` did not exist in the plan. It was added
  once upstream R itself turned out to be wrong in places, which the plan did not
  anticipate. Three entries so far.

### The L-BFGS-B risk materialised

The plan listed "L-BFGS-B parity" as a risk and mitigated it by testing the
solver standalone against scipy before wiring it into NEBULA. That happened, and
it was not enough: the solver matched scipy on every standard bounded problem
including one with two active bounds, and still failed on NEBULA's stage one.

Two defects, both in the line search:

- It never capped the step at the last feasible one. Past that point every trial
  clamps to the same corner of the box, so the objective flattens while
  `dot(grad, direction)` keeps reporting the unclamped slope. The strong Wolfe
  curvature condition is then unsatisfiable and the search burns its budget.
  NEBULA's `sigma` sits on its lower bound for most genes, so this fired
  constantly.
- On a failed line search it gave up, where the Fortran discards the
  limited-memory pairs and retries from projected steepest descent.

On the gene that exposed it, the fix takes the solver from stopping at a
non-stationary point with a projected gradient of 142, to the exact optimum in
29 iterations and 56 evaluations. The alternating quasi-Newton and simplex
workaround in `sc/nebula.rs` collapsed back to a single call.

Lesson for the risk register rather than the code: a solver validated only on
textbook problems is validated only for textbook problems. The regression test
added, `test_search_stops_at_the_boundary_rather_than_past_it`, is a badly scaled
separable problem with every optimum on a bound, which is the shape that broke it.
