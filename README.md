[![CI](https://github.com/GregorLueg/edge-rs/actions/workflows/test.yml/badge.svg)](https://github.com/GregorLueg/edge-rs/actions/workflows/test.yml)
[![Crates.io Version](https://img.shields.io/crates/v/edge-rs.svg)](https://crates.io/crates/edge-rs)
[![docs.rs](https://img.shields.io/docsrs/edge-rs)](https://docs.rs/edge-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# edge-rs

## Description

Negative binomial differential expression for bulk and single-cell RNA-seq. This
is the edgeR numerical stack in Rust: normalisation, Cox-Reid dispersion
estimation, the Levenberg-damped NB GLM, quasi-likelihood weights, the exact
test and `diffSpliceDGE`. On top of that sits the limma linear model stack
(`squeezeVar`, the F-distribution fits, lowess and locfit smoothing, `voom`,
`lmFit`, `contrasts.fit`, `eBayes`, `topTable`, `removeBatchEffect`) and
NEBULA, a negative binomial gamma mixed model for single cell.

No R, no Python, no BLAS to hunt down. CPU by default, with an optional GPU path
for NEBULA.

## Install

```sh
cargo add edge-rs
```

SIMD dispatch happens at runtime, so a stock build already picks up NEON on
aarch64 and AVX2 on x86. The 512-bit kernels are the exception: they are gated
at compile time and need

```sh
RUSTFLAGS="-C target-cpu=x86-64-v4" cargo build --release
```

A bare `+avx512f` is not enough, since `wide` only ships its real 512-bit types
under the full feature set.

## Quick start

### The GLM chain

Filter, normalise, estimate dispersions, fit, test, rank.
Counts are dense and gene-major throughout, `n_genes` rows of `n_samples`
values, row-major, so one gene is a contiguous slice.

```rust
use edge_rs::core::dgelist::DgeList;
use edge_rs::core::expression::ave_log_cpm;
use edge_rs::core::filtering::filter_by_expr;
use edge_rs::core::normalisation::{NormMethod, calc_norm_factors};
use edge_rs::dispersion::estimate::estimate_disp;
use edge_rs::glm::fit::glm_fit;
use edge_rs::glm::test::{GlmTestInput, Tested, glm_lrt};
use edge_rs::prelude::*;
use edge_rs::results::{SortBy, top_tags};

let dge = DgeList::new(counts, n_genes, n_samples, Some(group))?;

let keep = filter_by_expr(
    &dge.counts, dge.n_genes, dge.n_samples,
    None, dge.group.as_deref(), None, None,
)?;
let mut dge = dge.subset_genes(&keep)?;

dge.norm_factors = calc_norm_factors(
    &dge.counts, dge.n_genes, dge.n_samples,
    None, NormMethod::Tmm, None, None,
)?;

// Log of lib.size * norm.factors, one value per sample.
let offset = dge.offset()?;
let abundance = ave_log_cpm(
    &dge.counts, dge.n_genes, dge.n_samples,
    None, Some(&offset), 2.0, None,
)?;

let disp = estimate_disp(
    &dge.counts, dge.n_genes, dge.n_samples,
    &design, n_coef, &offset, None, Some(&abundance), None,
)?;
let dispersion = Recycled::by_gene(disp.tagwise.expect("tagwise is on by default"));

let fit = glm_fit(
    &dge.counts, dge.n_genes, dge.n_samples,
    &design, n_coef, &dispersion, &offset, None, 0.125,
)?;

let input = GlmTestInput {
    counts: &dge.counts,
    n_genes: dge.n_genes,
    n_samples: dge.n_samples,
    design: &design,
    n_coef,
    dispersion: &dispersion,
    offset: &offset,
    weights: None,
    log_cpm: Some(&abundance),
};
let lrt = glm_lrt(&input, &fit, &Tested::Coef(vec![n_coef - 1]), None)?;

let top = top_tags(
    &lrt.log_fc,
    lrt.log_cpm.as_deref().unwrap(),
    Some(&lrt.statistic),
    &lrt.p_value,
    10,
    SortBy::PValue,
    1.0,
)?;
```

Swap `glm_fit` for `glm_ql_fit` and `glm_lrt` for `glm_ql_ftest` to get the
quasi-likelihood pipeline instead. `exact_test` covers the pre-GLM two-group
path, and `nebula` the single-cell one. Counts already in CSR? Hand them to
`nebula_sparse` and skip the densification.

### NEBULA on the GPU

NEBULA spends its time in stage two, the search over the two variance
components, where every evaluation is a penalised fit over every cell. The
`gpu` feature adds `nebula_sparse_gpu`, which runs those fits on the device via
[CubeCL](https://github.com/tracel-ai/cubecl) and wgpu (Metal, Vulkan, DX12).
Same inputs, same `NebulaFit` out as `nebula_sparse`.

```toml
edge-rs = { version = "0.1", features = ["gpu"] }
cubecl = { version = "0.10", features = ["wgpu"] }
```

```rust
use cubecl::Runtime;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use edge_rs::gpu::stage_two::nebula_sparse_gpu;

let client = WgpuRuntime::client(&WgpuDevice::default());
let fit = nebula_sparse_gpu(
    &counts, &subject_id, &design, n_coef, Some(&offset), None, &client,
)?;
```

wgpu has no `f64`, so the device fits in `f32` and the host finishes every fit
in `f64`. It's gated against the same R fixtures as the CPU path, at a looser
tolerance. Designs up to eight columns; no `reml`.

How much faster? Forced NEBULA-HL, 500 genes, 20 subjects, against the
10-thread CPU path on an M1 Max: 7.5x at 100000 cells, 4x to 6x at 20000 to
50000, 3.4x with eight coefficients, where every launch waits out one fit's
serial walk over the cells. Under the default NEBULA-LN most genes never reach
stage two, and on the bench data the GPU bought nothing.

### limma-voom

The other bulk route: transform the counts, fit a linear model against precision
weights, moderate the variances, rank. `MArrayLm` is limma's `MArrayLM`, one
object that each stage consumes and hands back with more on it.

```rust
use edge_rs::limma::contrasts::{contrasts_fit, make_contrasts};
use edge_rs::limma::ebayes::{EBayesParams, EBayesTrend, ebayes};
use edge_rs::limma::marray::MArrayLm;
use edge_rs::limma::toptable::{TopTableParams, TopTableSort, top_table};
use edge_rs::limma::voom::voom_lmfit;

// Mean-variance trend, precision weights and the weighted least squares fit,
// in one pass over the counts.
let (voom, lm) = voom_lmfit(
    &counts, n_genes, n_samples, &design, n_coef, None, None, None,
)?;
let fit = MArrayLm::from_lm_fit(lm, &design, n_coef, n_samples, Some(voom.amean))?;

// Rotate onto the comparisons of interest. Names are the design columns.
let (contrasts, n_contrasts) = make_contrasts(
    &["Int", "grpB", "batb2"],
    &["grpB", "grpB - 0.5 * batb2"],
)?;
let fit = contrasts_fit(fit, &contrasts, n_contrasts)?;

// Moderate, trending the prior against average log-expression.
let fit = ebayes(fit, Some(EBayesParams {
    trend: EBayesTrend::Amean,
    ..Default::default()
}))?;

// Rank the first contrast.
let top = top_table(&fit, 0, Some(TopTableParams {
    number: 10,
    sort_by: TopTableSort::PValue,
    ..Default::default()
}))?;
```

Prefer limma-trend? Skip `voom_lmfit`: put log-CPM from `cpm(.., log = true, ..)`
straight into `lm_fit`, hand `MArrayLm::from_lm_fit` the row means as `amean`,
and moderate with `EBayesTrend::Amean`. Cheaper, and what limma recommends when
the library sizes are not too variable. `robust: true` goes with either route
and Winsorises the prior so a handful of outlier genes cannot drag the prior
degrees of freedom down.

Batch effects messing up a PCA or heatmap? `remove_batch_effect` is limma's
`removeBatchEffect`: up to two batch factors plus numeric covariates, with the
design protecting the biology and precision weights if you have them. It's for
plotting only. For testing, put the batch in the design.

`ebayes` also fills in the moderated F across contrasts whenever the design is
full rank, and `top_table_f` ranks on it. Building the contrast matrix yourself
rather than through `make_contrasts`? `contrasts_fit` wants it column-major,
`n_coef` by `n_contrasts`.

Fits and containers are generic over `EdgeFloat`, so single-cell counts can be
held as `f32` and halve the memory. Likelihoods, Cox-Reid determinants, the
optimisers and every p-value run in `f64` regardless: an NB deviance is a
difference of large logs and loses edgeR parity in `f32` long before it saves
anything.

## What is not in here

Still not a general limma port. The linear model chain is complete through
`topTable`, but `treat` and `topTreat` are absent, and so is limma's
`decideTests`: only the F statistic `eBayes` needs is ported, not the
step-down classification around it. `voom_lmfit` takes no `block` or
`correlation`, though `duplicate_correlation` and `array_weights` are both
there to be called on their own. No `mrlm`, no `vooma`, no
`voomWithQualityWeights`. `make_contrasts` reads a linear expression over the
design column names rather than evaluating arbitrary R.

No visualisations and no pathway enrichment. On the single-cell side, NEBULA is
`NBGMM` only: `PMM` needs Poisson-gamma kernels this crate does not have, and
`NBLMM` has no golden to validate against.

## Parity

The end-to-end suites gate against committed CSV fixtures generated by
`tests/r/generate_fixtures.R` against R `4.5.1` with edgeR `4.8.2`, limma
`3.66.0`, statmod `1.5.2` and nebula `1.5.8`. CI never runs the R; the fixtures
are in the repository and the Rust reads them.

Tolerances are measured rather than guessed. Every one carries the worst error
actually observed, and the test harness prints that table on demand:

```sh
EDGE_RS_TOL_REPORT=1 cargo test --release --test 'e2e_*' -- --nocapture
```

## Deviations

The port's working reference was
[edgePython](https://github.com/pachterlab/edgePython), itself a port of edgeR,
limma and nebula. Where it disagreed with the R and C++ it came from, upstream
won: edgeR and limma are what users compare against, and their output is what
the test suite gates on. NEBULA is ported from the nebula package's own C++
rather than from the Python.

Rarer, and worth stating plainly: a handful of places where edgeR or limma are
themselves wrong, each verified by reproducing the fault in the installed
package. Reproducing a bug faithfully is not parity worth having.

Both sets are written up in [docs/UPSTREAM_DEVIATIONS.md](docs/UPSTREAM_DEVIATIONS.md),
with the upstream file and line, what this crate does instead, and a test that
would fail if the behaviour drifted back. The edgePython entries were read
against version 0.2.6, commit `1e572ae`, and any of them may well have been
fixed upstream since.

## Licence

MIT License

Copyright (c) 2026 Gregor Lueg

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
