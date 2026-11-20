"""edgeR and limma-voom differential expression, in Rust.

NumPy in, polars out. Counts are `(n_genes, n_samples)` as in edgeR, every
index is 0-based, and a design is a NumPy matrix or a polars frame whose column
names become coefficient names.

One call:

    >>> res = edge_rs.run_edger_ql(counts, design, feature_ids=genes, coef="treated")
    >>> res = edge_rs.run_limma_voom(counts, design, feature_ids=genes, coef="treated")

Step by step, as in edgeR and limma:

    >>> keep = edge_rs.filter_by_expr(counts, design=design)
    >>> lib = counts.sum(axis=0)
    >>> nf = edge_rs.calc_norm_factors(counts[keep], lib_size=lib)
    >>> fit = edge_rs.glm_ql_fit(counts[keep], design, lib_size=lib, norm_factors=nf)
    >>> edge_rs.top_tags(edge_rs.glm_ql_ftest(fit, coef="treated"))

Every function that takes counts also takes a `DgeList` from `make_dgelist`,
which carries `lib_size` and `norm_factors` and keeps the library sizes of the
full matrix through a gene subset.
"""

from importlib.metadata import version

from edge_rs._core import (
    ave_log_cpm,
    calc_norm_factors,
    cpm,
    filter_by_expr,
    remove_batch_effect,
)
from edge_rs._dgelist import DgeList, make_dgelist
from edge_rs._edge_rs import EdgeRsError, LinearFit, QLFit
from edge_rs._edger import QLTest, glm_ql_fit, glm_ql_ftest, top_tags
from edge_rs._limma import (
    Voom,
    contrasts_fit,
    ebayes,
    lm_fit,
    make_contrasts,
    top_table,
    voom,
    voom_lm_fit,
)
from edge_rs._pipelines import run_edger_ql, run_limma_voom

__version__ = version("edge-rs")

__all__ = [
    "DgeList",
    "EdgeRsError",
    "LinearFit",
    "QLFit",
    "QLTest",
    "Voom",
    "ave_log_cpm",
    "calc_norm_factors",
    "contrasts_fit",
    "cpm",
    "ebayes",
    "filter_by_expr",
    "glm_ql_fit",
    "glm_ql_ftest",
    "lm_fit",
    "make_contrasts",
    "make_dgelist",
    "remove_batch_effect",
    "run_edger_ql",
    "run_limma_voom",
    "top_table",
    "top_tags",
    "voom",
    "voom_lm_fit",
]
