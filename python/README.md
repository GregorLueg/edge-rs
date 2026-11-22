[![PyPI](https://img.shields.io/pypi/v/edge-rs.svg)](https://pypi.org/project/edge-rs/)
[![Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://gregorlueg.github.io/edge-rs/)
[![CI](https://github.com/GregorLueg/edge-rs/actions/workflows/python-test.yml/badge.svg)](https://github.com/GregorLueg/edge-rs/actions/workflows/python-test.yml)

# edge-rs

edgeR's quasi-likelihood pipeline and limma-voom for Python, with the numerics
in the [`edge-rs`](https://github.com/GregorLueg/edge-rs) Rust crate. No R, no
rpy2. NumPy in, polars out, and every function named after the R function it
reproduces.

Docs, API reference and vignettes: <https://gregorlueg.github.io/edge-rs/>

## Install

```bash
uv pip install edge-rs
```

The runtime dependencies are NumPy, polars and beartype.

## Use

```python
import edge_rs

# counts: (n_genes, n_samples), as in edgeR
res = edge_rs.run_edger_ql(counts, design, feature_ids=genes, coef="treated")
res = edge_rs.run_limma_voom(counts, design, feature_ids=genes, coef="treated")
```

Or step by step, as in edgeR and limma:

```python
keep = edge_rs.filter_by_expr(counts, design=design)
lib_size = counts.sum(axis=0)
nf = edge_rs.calc_norm_factors(counts[keep], lib_size=lib_size)

fit = edge_rs.glm_ql_fit(counts[keep], design, lib_size=lib_size, norm_factors=nf)
edge_rs.top_tags(edge_rs.glm_ql_ftest(fit, coef="treated"))

vfit = edge_rs.voom_lm_fit(counts[keep], design, lib_size=lib_size, norm_factors=nf)
edge_rs.top_table(edge_rs.ebayes(vfit), coef="treated")
```

If you would rather not carry `lib_size` around, `make_dgelist` bundles counts,
library sizes, normalisation factors and annotation. Subsetting it keeps the
library sizes of the full matrix, as edgeR's `keep.lib.sizes = TRUE` does, and
every function above takes it in place of the counts:

```python
dge = edge_rs.make_dgelist(counts, genes=pl.DataFrame({"gene_id": genes}))
dge = edge_rs.calc_norm_factors(dge[edge_rs.filter_by_expr(dge, design=design)])
fit = edge_rs.glm_ql_fit(dge, design)
edge_rs.top_tags(
    edge_rs.glm_ql_ftest(fit, coef="treated"), feature_ids=dge.genes["gene_id"]
)
```

Counts are genes x samples. Single cell tools are the other way round, so
pseudobulk from [bixverse](https://gregorlueg.github.io/bixverse-rs/) goes in as
`pb.counts.T`. A design is a NumPy matrix or a polars frame; with a frame, the
column names become coefficient names and `coef="treated"` works. Everything is
0-based.

## What's in it

| Python | R |
| --- | --- |
| `make_dgelist`, `DgeList` | `edgeR::DGEList` |
| `filter_by_expr` | `edgeR::filterByExpr` |
| `calc_norm_factors` | `edgeR::calcNormFactors` |
| `cpm`, `ave_log_cpm` | `edgeR::cpm`, `edgeR::aveLogCPM` |
| `remove_batch_effect` | `limma::removeBatchEffect` |
| `glm_ql_fit`, `glm_ql_ftest`, `top_tags` | `edgeR::glmQLFit`, `edgeR::glmQLFTest`, `edgeR::topTags` |
| `voom`, `voom_lm_fit`, `lm_fit` | `limma::voom`, `edgeR::voomLmFit`, `limma::lmFit` |
| `make_contrasts`, `contrasts_fit` | `limma::makeContrasts`, `limma::contrasts.fit` |
| `ebayes`, `top_table` | `limma::eBayes`, `limma::topTable` |
| `run_edger_ql`, `run_limma_voom` | the above, chained |

Not exposed yet: NEBULA, the exact test, `glmFit`/`glmLRT`/`glmTreat` and
`diffSpliceDGE`. The crate has all of them.

## How close to R?

The crate is gated against edgeR 4.8.2 and limma 3.66.0 on committed fixtures,
and the Python tests run the same fixtures through the bindings. The bulk
vignette reruns the edgeR QL F1000 workflow on GSE60450 and matches R on every
number it shows.

## Vignettes

Two notebooks: bulk DGE on GSE60450, and pseudobulk DGE on the Kang PBMCs with
bixverse. Each is a jupytext `.py` in `docs/vignettes/` with its executed
`.ipynb` next to it, and the data download on first use.

```bash
uv sync --group dev --group vignettes
uv run maturin develop --release
uv run python dev/vignettes.py execute             # both
uv run python dev/vignettes.py execute pseudobulk  # or by name
```

## Development

```bash
uv sync --group dev
uv run maturin develop --release
uv run pytest tests -q
```

The wheel builds against the crate in the parent directory, so it always tracks
this checkout.
