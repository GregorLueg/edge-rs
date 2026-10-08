# Changelog

## v0.2.1

### Features

* NEBULA is several times faster. Against R nebula 1.5.8 on 100 genes of 20000
  cells, one thread against one core: NEBULA-LN 0.78 s against 1.94 s and
  NEBULA-HL 5.4 s against 6.7 s with a continuous cell-level covariate, and
  0.74 s against 1.92 s and 1.8 s against 6.9 s with a categorical one. On ten
  threads the four take 0.11 to 0.71 s.
  * Stage two searches the two variance components with BOBYQA, ported from
    NLopt 2.7.1 as `nloptr::bobyqa` runs it, so it makes about forty fits
    per gene where the simplex and polish made 120. It retraces nloptr point
    for point on seven bounded problems.
  * Stage one takes projected Newton steps on a fused Hessian on dense designs,
    falling back to L-BFGS-B.
  * The penalised fit sweeps every cell once per Newton step instead of a dozen
    times.
  * Designs whose cell-level columns are categorical sum each gene's zero
    counts from per-run tables, one lookup per subject and cell type, so a gene
    costs its positive counts rather than every cell.
* `nebula_sparse_gpu` runs 1.9x to 3.3x faster than the ten-thread CPU path
  (500 genes, 20000 to 100000 cells), up to 2.8x faster on the device at eight
  coefficients. A design with zero-count tables runs the CPU path from it,
  which is faster there and accepts designs wider than eight columns.

### Bug fixes

* The one-component NEBULA-LN refit no longer stops on a variance bound its
  simplex collapsed onto; it restarts once from between the bound and the
  start. This settles the `sc_high` gene the tests recorded as an open gap.

### Notes

* Results move against v0.2.0 within nebula's own reproducibility: stage two
  now takes R's optimiser path, and the `1e-6` jitter of the profile likelihood
  sends nearby runs to slightly different points. The R fixtures gate every
  change; three new categorical fixtures cover the zero-count tables.
* Repo clean-up: reduction in documentation and unnecessary details.

## v0.2.0

### Breaking changes

* `NebulaParams` gains a public `min_subjects` field. Code building the struct
  with an exhaustive literal no longer compiles; `..NebulaParams::default()`
  keeps working.

### Features

* NEBULA can drop genes that too few subjects express. `min_subjects` counts
  the subjects whose own mean count per cell clears `cpc` and drops the gene
  below the threshold. `cpc` and `mincp` pool every cell, so a single subject
  could carry a gene through and leave a subject-level coefficient resting on a
  handful of subjects. Applies to `nebula`, `nebula_sparse` and
  `nebula_sparse_gpu`. Defaults to `0`, off, which matches the R package.

## v0.1.2

### Features

* Expose the `scran_lowess()` function, a modified version of Limma's Lowess
  designed for HVG detection in single cell.

## v0.1.1

### Bug fixes

* The GPU NEBULA kernel no longer demands a 32-lane plane. It reads the plane
  width at run time, so it runs on AMD, Intel and software adapters such as
  lavapipe instead of refusing them. Checked on Metal (32 lanes) and lavapipe
  (4 lanes); no measurable change on an M1 Max.

## v0.1.0

* GPU-accelerated NEBULA added via the wgpu/cubecl framework, behind the `gpu`
  feature. `nebula_sparse_gpu` takes and returns what `nebula_sparse` does, runs
  stage two's fits on the device in `f32` and finishes each in `f64` on the
  host. 3.4x to 7.5x over the CPU path under NEBULA-HL on an M1 Max.
* `gpu-tests` feature and a GPU lane in CI.

## v0.0.5

* `AplWorkspace` added: the adjusted profile likelihood point by point, with the
  coefficients carried from one dispersion to the next. Callers running their
  own search over the dispersion, rather than a grid, no longer pay a cold
  restart per evaluation the way repeated `apl_at` calls do. `apl_grid` now runs
  on top of it.

## v0.0.4

* `remove_batch_effect` added.

## v0.0.3

* Ported in eBayes, contrasts.fit, MArrayLm to also enable the limma-voom
  workflow fully in Rust. E2E parity tests again fixtures generated via R.

## v0.0.2

* Public entry point to directly take `CompressedSparse` data.
* `QlFit::as_glm_fit()` and `QlFit::ql_summary()` accessors added.

## v0.0.1

* Pre-release version that should have most of the functionality ported from
  edgeR, edgePython and NEBULA, minus the visualisations and pathway enrichment
  tools.
