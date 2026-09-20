# Changelog

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
