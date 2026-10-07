# Changelog

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
