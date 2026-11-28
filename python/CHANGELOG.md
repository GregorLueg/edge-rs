# Changelog

Versions here track the Python package, not the `edge-rs` crate. Each release
says which crate version it builds against.

## 0.2.0

Builds against the `edge-rs` crate at 0.4.0. NEBULA parity is tested against
nebula 1.5.9 on the crate's committed fixtures.

**Features**

- `nebula(adata, subject_key, design)`: NEBULA's negative binomial gamma mixed
  model straight on an `AnnData`, behind the new `anndata` extra. Takes an
  additive formula over `obs` and returns a pandas frame indexed by gene.
- Cells need not be sorted by subject.

## 0.1.0

Builds against the `edge-rs` crate at 0.4.0. Parity is tested against edgeR
4.8.2 and limma 3.66.0 on the crate's committed fixtures.

**Features**

- First release. The bulk edgeR quasi-likelihood stack (`filter_by_expr`,
  `calc_norm_factors`, `cpm`, `ave_log_cpm`, `glm_ql_fit`, `glm_ql_ftest`,
  `top_tags`) and limma-voom (`voom`, `voom_lm_fit`, `lm_fit`,
  `make_contrasts`, `contrasts_fit`, `ebayes`, `top_table`), plus
  `remove_batch_effect`.
- `run_edger_ql` and `run_limma_voom` chain the steps in one call.
- Optional `make_dgelist`/`DgeList` record: counts, library sizes and
  normalisation factors in one place, keeping the library sizes through a gene
  filter. Every function that takes counts also takes it.
- Counts are genes x samples, designs a NumPy matrix or a polars frame, results
  polars frames. Pseudobulk from bixverse goes in as `pb.counts.T`.
- Docs site with two vignettes: bulk DGE on GSE60450, and pseudobulk DGE on the
  Kang PBMCs with bixverse.
