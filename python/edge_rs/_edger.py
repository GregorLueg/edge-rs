"""edgeR's quasi-likelihood pipeline: `glmQLFit`, `glmQLFTest` and `topTags`.

>>> fit = edge_rs.glm_ql_fit(counts, design, norm_factors=nf)
>>> res = edge_rs.glm_ql_ftest(fit, coef="treated")
>>> edge_rs.top_tags(res, feature_ids=genes)
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
from beartype import beartype

from edge_rs import _edge_rs
from edge_rs._edge_rs import QLFit
from edge_rs._validate import (
    CONF,
    Coef,
    Contrast,
    Design,
    Labels,
    as_counts,
    as_design,
    as_vector,
    effective_lib_size,
    feature_column,
    tested,
)

Vector = np.ndarray | Sequence[float]


@beartype(conf=CONF)
@dataclass(frozen=True, kw_only=True, slots=True)
class QLTest:
    """A quasi-likelihood F-test, one entry per gene of the fit.

    Attributes:
        log_fc: Log2 fold change of the first tested coefficient or contrast.
        log_cpm: Average log2-CPM.
        f_stat: Quasi-likelihood F statistic.
        p_value: Raw p-value.
        df_test: Numerator degrees of freedom.
        df_total: Denominator degrees of freedom per gene.
    """

    log_fc: np.ndarray
    log_cpm: np.ndarray
    f_stat: np.ndarray
    p_value: np.ndarray
    df_test: float
    df_total: np.ndarray | None


@beartype(conf=CONF)
def glm_ql_fit(
    counts: np.ndarray,
    design: Design,
    *,
    lib_size: Vector | None = None,
    norm_factors: Vector | None = None,
    ave_log_cpm: Vector | None = None,
    robust: bool = False,
    legacy: bool = False,
) -> QLFit:
    """Fit a quasi-likelihood negative binomial GLM per gene, edgeR's `glmQLFit`.

    The current (edgeR 4) pipeline estimates the negative binomial dispersion
    itself, so there is no `estimateDisp` step. `legacy=True` takes the
    pre-4.0 pipeline, which runs `estimateDisp` first and uses its trend.

    Args:
        counts: `(n_genes, n_samples)` raw counts, already filtered.
        design: `(n_samples, n_coef)` design with an intercept. A polars frame's
            column names can later be used as `coef`.
        lib_size: Library size per sample. Defaults to the column sums. After a
            gene filter, pass the unfiltered column sums.
        norm_factors: Normalisation factors from `calc_norm_factors`.
        ave_log_cpm: Average log2-CPM per gene. `None` computes it.
        robust: Robust empirical Bayes, so outlier genes get a smaller prior.
        legacy: edgeR's pre-4.0 pipeline.

    Returns:
        The `QLFit`, to hand to `glm_ql_ftest`.

    Raises:
        ValueError: On mismatched shapes or negative counts.
        EdgeRsError: If the design is not full rank.

    References:
        Chen, Lun and Smyth, F1000Research, 2016.
    """
    y = as_counts(counts)
    mm, names = as_design(design, y.shape[1])
    lib = effective_lib_size(y, lib_size, norm_factors)
    lib = np.ascontiguousarray(y.sum(axis=0)) if lib is None else lib
    alc = (
        None
        if ave_log_cpm is None
        else as_vector(ave_log_cpm, y.shape[0], "ave_log_cpm")
    )
    return _edge_rs.glm_ql_fit(y, mm, lib, alc, robust, legacy, names)


@beartype(conf=CONF)
def glm_ql_ftest(
    fit: QLFit,
    *,
    coef: Coef | None = None,
    contrast: Contrast | None = None,
) -> QLTest:
    """Quasi-likelihood F-test per gene, edgeR's `glmQLFTest`.

    Args:
        fit: The `QLFit`.
        coef: 0-based design column(s) or names to drop from the null model.
            Several give an F-test on all of them. `None` tests the last
            column.
        contrast: Weights over the design columns, a vector or an
            `(n_coef, n_contrasts)` matrix (see `make_contrasts`). Exclusive
            with `coef`.

    Returns:
        The `QLTest`, to hand to `top_tags`.

    Raises:
        ValueError: On an unknown coefficient, or both `coef` and `contrast`.
    """
    coefs, c = tested(fit.coef_names, fit.n_coef, coef, contrast)
    res = _edge_rs.glm_ql_ftest(
        fit,
        coefs,
        None if c is None else c.ravel(order="F").tolist(),
        1 if c is None else c.shape[1],
    )
    return QLTest(
        log_fc=res["log_fc"],
        log_cpm=res["log_cpm"],
        f_stat=res["f_stat"],
        p_value=res["p_value"],
        df_test=res["df_test"],
        df_total=res.get("df_total"),
    )


@beartype(conf=CONF)
def top_tags(
    test: QLTest,
    *,
    feature_ids: Labels | None = None,
    n: int | None = None,
    sort_by: Literal["p_value", "log_fc", "none"] = "p_value",
    p_cutoff: float = 1.0,
) -> pl.DataFrame:
    """The test as a table, edgeR's `topTags`.

    The FDR is Benjamini-Hochberg over every gene, before `n` or `p_cutoff`
    thin the table.

    Args:
        test: The `QLTest`.
        feature_ids: One id per gene of the fit. `None` uses the 0-based row.
        n: Maximum number of rows. `None` keeps every gene.
        sort_by: Ascending p-value, descending absolute fold change, or input
            order.
        p_cutoff: Keep rows with FDR at most this.

    Returns:
        `feature_id`, `log_fc`, `log_cpm`, `f_stat`, `p_value` and `fdr`.
    """
    n_genes = test.p_value.shape[0]
    tt = _edge_rs.top_tags(
        test.log_fc,
        test.log_cpm,
        test.f_stat,
        test.p_value,
        n_genes if n is None else n,
        sort_by,
        p_cutoff,
    )
    return pl.DataFrame(
        [
            feature_column(feature_ids, n_genes, tt["index"]),
            pl.Series("log_fc", tt["log_fc"]),
            pl.Series("log_cpm", tt["log_cpm"]),
            pl.Series("f_stat", tt["statistic"]),
            pl.Series("p_value", tt["p_value"]),
            pl.Series("fdr", tt["fdr"]),
        ]
    )
