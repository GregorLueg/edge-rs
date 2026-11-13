"""One-call edgeR quasi-likelihood and limma-voom runs.

Filter, normalise, fit, test, tabulate. The library sizes stay those of the
unfiltered matrix throughout, as edgeR keeps them on a subset `DGEList`;
recomputing them after the filter shifts every normalisation factor.

    >>> pb = bv.de.get_pseudobulked(sc, cell_list)
    >>> edge_rs.run_edger_ql(pb.counts.T, design, feature_ids=pb.gene_ids)
"""

from typing import Literal

import numpy as np
import polars as pl
from beartype import beartype

from edge_rs._core import calc_norm_factors, filter_by_expr
from edge_rs._edger import glm_ql_fit, glm_ql_ftest, top_tags
from edge_rs._limma import contrasts_fit, ebayes, top_table, voom_lm_fit
from edge_rs._validate import (
    CONF,
    Coef,
    Contrast,
    Design,
    Labels,
    as_counts,
    as_design,
    coef_positions,
)

NormMethod = Literal["TMM", "TMMwsp", "RLE", "upperquartile", "none"]


@beartype(conf=CONF)
def _filter_and_normalise(
    counts: np.ndarray,
    design: Design,
    feature_ids: Labels | None,
    norm_method: str,
    filter: bool,
) -> tuple[np.ndarray, np.ndarray, pl.Series]:
    """Filtered counts, their effective library sizes and their ids.

    Raises:
        ValueError: If no gene survives the filter.
    """
    y = as_counts(counts)
    as_design(design, y.shape[1])
    lib = y.sum(axis=0)
    keep = filter_by_expr(y, design=design) if filter else np.ones(y.shape[0], bool)
    if not keep.any():
        raise ValueError("No gene passed `filter_by_expr`.")
    kept = np.flatnonzero(keep)
    ids = (
        pl.Series("feature_id", kept, dtype=pl.UInt32)
        if feature_ids is None
        else pl.Series(feature_ids).gather(kept)
    )
    y = y[keep]
    nf = calc_norm_factors(y, method=norm_method, lib_size=lib)
    return y, lib * nf, ids


@beartype(conf=CONF)
def run_edger_ql(
    counts: np.ndarray,
    design: Design,
    *,
    feature_ids: Labels | None = None,
    coef: Coef | None = None,
    contrast: Contrast | None = None,
    norm_method: NormMethod = "TMM",
    filter: bool = True,
    robust: bool = False,
    legacy: bool = False,
) -> pl.DataFrame:
    """edgeR's quasi-likelihood pipeline in one call.

    `filter_by_expr`, `calc_norm_factors`, `glm_ql_fit`, `glm_ql_ftest` and
    `top_tags`.

    Args:
        counts: `(n_genes, n_samples)` raw counts. From a bixverse pseudobulk,
            `pb.counts.T`.
        design: `(n_samples, n_coef)` design with an intercept.
        feature_ids: One id per row of `counts`, e.g. `pb.gene_ids`. `None`
            uses the 0-based row.
        coef: 0-based design column(s) or names to test. `None` is the last.
        contrast: Weights over the design columns. Exclusive with `coef`.
        norm_method: Normalisation, see `calc_norm_factors`.
        filter: Run `filter_by_expr` first.
        robust: Robust empirical Bayes.
        legacy: edgeR's pre-4.0 pipeline.

    Returns:
        `feature_id`, `log_fc`, `log_cpm`, `f_stat`, `p_value` and `fdr`, one
        row per gene that passed the filter, in input order.

    References:
        Chen, Lun and Smyth, F1000Research, 2016.
    """
    y, lib, ids = _filter_and_normalise(
        counts, design, feature_ids, norm_method, filter
    )
    fit = glm_ql_fit(y, design, lib_size=lib, robust=robust, legacy=legacy)
    res = glm_ql_ftest(fit, coef=coef, contrast=contrast)
    return top_tags(res, feature_ids=ids, sort_by="none")


@beartype(conf=CONF)
def run_limma_voom(
    counts: np.ndarray,
    design: Design,
    *,
    feature_ids: Labels | None = None,
    coef: int | str | None = None,
    contrast: Contrast | None = None,
    norm_method: NormMethod = "TMM",
    filter: bool = True,
    robust: bool = False,
    span: float = 0.5,
    adaptive_span: bool = True,
    proportion: float = 0.01,
) -> pl.DataFrame:
    """limma-voom in one call.

    `filter_by_expr`, `calc_norm_factors`, `voom_lm_fit`, `contrasts_fit` when
    a contrast is given, `ebayes` and `top_table`.

    Args:
        counts: `(n_genes, n_samples)` raw counts. From a bixverse pseudobulk,
            `pb.counts.T`.
        design: `(n_samples, n_coef)` design with an intercept.
        feature_ids: One id per row of `counts`. `None` uses the 0-based row.
        coef: 0-based design column or name to test. `None` is the last.
        contrast: A single contrast over the design columns. Exclusive with
            `coef`.
        norm_method: Normalisation, see `calc_norm_factors`.
        filter: Run `filter_by_expr` first.
        robust: Robust empirical Bayes.
        span: Lowess span for the voom trend, used only without
            `adaptive_span`.
        adaptive_span: Pick the span from the number of genes.
        proportion: Assumed proportion of differentially expressed genes.

    Returns:
        `feature_id`, `log_fc`, `ci_lower`, `ci_upper`, `ave_expr`, `t`,
        `p_value`, `adj_p_value` and `b`, one row per gene that passed the
        filter, in input order.

    Raises:
        ValueError: On both `coef` and `contrast`, or a contrast with more
            than one column.

    References:
        Law, Chen, Shi and Smyth, Genome Biology, 2014.
    """
    if coef is not None and contrast is not None:
        raise ValueError("Supply either `coef` or `contrast`, not both.")
    y, lib, ids = _filter_and_normalise(
        counts, design, feature_ids, norm_method, filter
    )
    fit = voom_lm_fit(y, design, lib_size=lib, span=span, adaptive_span=adaptive_span)
    if contrast is not None:
        fit = contrasts_fit(fit, contrast)
        if fit.n_coef != 1:
            raise ValueError("`run_limma_voom` tests one contrast at a time.")
        j = 0
    else:
        (j,) = coef_positions(
            fit.n_coef - 1 if coef is None else coef, fit.coef_names, fit.n_coef
        )
    fit = ebayes(fit, proportion=proportion, robust=robust)
    return top_table(fit, coef=j, feature_ids=ids, sort_by="none", confint=True)
