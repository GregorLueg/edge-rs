# Changelog

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
