# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`edge-rs` is a from-scratch Rust port of the edgeR numerical stack (normalisation,
Cox-Reid dispersion estimation, the Levenberg-damped NB GLM, quasi-likelihood
weights, the exact test, `diffSpliceDGE`), the parts of limma edgeR leans on
(`squeezeVar`, `fitFDist`, lowess/locfit smoothing, `voomLmFit`), and NEBULA
(negative binomial gamma mixed models for single cell). No R or Python runtime is
involved; the crate is pure Rust and CPU-only.

## Commands

```sh
cargo build --release
cargo test --release                       # parity tests are slow in debug
cargo test --release --test e2e_bulk_ql    # one E2E binary
cargo test --release glm::levenberg        # unit tests matching a path
cargo clippy --all-targets -- -D warnings
cargo doc --no-deps --open
```

`#![warn(missing_docs)]` is on at the crate root, so every public item needs a doc
comment or the build warns.

AVX-512 kernels are compile-time gated on `target_feature = "avx512f"`. `wide`'s
512-bit types only appear with `RUSTFLAGS="-C target-cpu=x86-64-v4"`; a bare
`+avx512f` is not enough. Without it, x86 tops out at AVX2 and aarch64 at NEON.
The runtime probe in `detect_simd_level` handles dispatch below that.

## Parity testing

The E2E suites in `tests/e2e_*.rs` gate against committed CSV fixtures under
`tests/data/e2e`, written by `tests/r/generate_fixtures.R` against edgeR 4.8.2,
limma 3.66.0, statmod 1.5.2 and nebula 1.5.8. CI never runs the R. Regenerate by
hand from the repository root after an upstream bump and inspect the diff:

```sh
Rscript tests/r/generate_fixtures.R
```

Input matrices are written once and read back forever, so a fixture diff always
means an expected value moved, never that the data was redrawn.

**Tolerances are measured, not guessed.** `tests/common/mod.rs` prints the worst
observed error per labelled quantity when `EDGE_RS_TOL_REPORT` is set, even on a
pass:

```sh
EDGE_RS_TOL_REPORT=1 cargo test --release --test 'e2e_*' -- --nocapture
```

Read a tolerance off the NEEDS column of that table, not off the worst relative
error, and record the measured worst case in the doc comment on the constant.
`src/sc/nebula.rs` is the pattern to follow.

## Upstream policy

The port's working reference is [`edgePython`](https://github.com/pachterlab/edgePython),
but **edgeR, limma and nebula win wherever the Python disagrees with them**.
Those are what users compare against. The rare exception is where edgeR or limma
are themselves wrong, verified by reproducing the fault in the installed package;
reproducing a bug faithfully is not parity worth having.

Every such divergence is recorded in `docs/UPSTREAM_DEVIATIONS.md` (Section A:
edgePython is wrong; Section B: edgeR/limma are wrong), each with the upstream
file and line, the ported location, and a test that would fail if the behaviour
drifted back. **Any new deliberate deviation belongs there.** NEBULA is ported
from the `nebula` package's own C++, not from edgePython, whose standard errors
are 6 to 89 per cent off.

The edgePython entries were read against version 0.2.6, commit `1e572ae`, and the
doc says so. A file-and-line citation with no commit behind it rots the moment
upstream edits the file. Re-pin the header if you check the claims against a
newer checkout.

## Architecture

Layered bottom-up; `src/lib.rs` carries the crate-level policy in its header.

- `utils/`: `EdgeFloat` (blanket-implemented bound over `f32`/`f64`),
  `EdgeSimd`, `Recycled`, `CompressedSparse`, design-matrix helpers.
- `numeric/`: everything edgePython reaches into `scipy` for. Special
  functions, distribution tails, L-BFGS-B, Nelder-Mead, interpolation.
- `core/`: `DgeList`, `filterByExpr`, `cpm`/`aveLogCPM`, TMM and friends.
- `glm/`, `dispersion/`, `ql/`, `exact/`: the edgeR stack proper.
- `limma/`: only what `estimateDisp`, `glmQLFit` and `voomLmFit` actually reach
  for. Not a general limma port.
- `sc/`: NEBULA. Driver (`nebula.rs`), kernels (`ptmg.rs`, `pml.rs`), Wald
  tests, dispersion shrinkage.
- `splicing.rs`, `results.rs` (`topTags`, `decideTests`), `errors.rs`.

### Numeric policy

Public containers and fits are generic over `EdgeFloat`, so single-cell counts
can be held as `f32` and halve the memory. **Likelihood evaluation, Cox-Reid
log-determinants, the optimisers and every p-value run in `f64` regardless of
`T`.** An NB deviance is a difference of large logs and loses edgeR parity in
`f32` long before it saves anything. Everything derived from counts (library
sizes, offsets, dispersions, weights) is `f64`.

### Layout and parallelism

Counts are dense and **gene-major**: `n_genes` rows of `n_samples`, row-major, so
one gene is a contiguous slice. Genes are the parallel axis almost everywhere:
rayon over genes with a per-thread scratch buffer, rather than the batched matrix
operations edgePython needs to go fast in NumPy.

Two documented exceptions: normalisation parallelises over samples (TMM's unit of
work is one sample against a reference column) and transposes once on entry;
`splicing.rs` uses both axes. Any module departing from the rule says so in its
own header.

The single-cell path does not go through `DgeList` at all. NEBULA takes per-gene
slices directly, which is what keeps the crate free of any particular sparse
container. `CompressedSparse::transpose` is free (a label flip); `convert` moves
data.

### Conventions

- `use crate::prelude::*;` over deep imports in new code.
- One `thiserror` enum, `EdgeErrors`, grouped by subsystem with section comments.
  Add variants to the matching section, not the bottom. Anything a caller could
  plausibly hit returns `Err`; panics are for broken invariants only.
- Unit tests live inline at the bottom of their module; E2E goes in `tests/*.rs`.
- SIMD width lives in `src/utils/simd.rs` and nowhere else. No `wide` types or
  intrinsics outside that file.
- `Recycled<T>` is edgeR's `compressedMatrix`: use it for anything logically
  `n_genes × n_samples` that varies along only one axis.
- Section banners (`//////////`) separate consts, types and impls within a file.
