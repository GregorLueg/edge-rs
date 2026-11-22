"""limma's linear model pipeline: voom, `lmFit`, `contrasts.fit`, `eBayes` and
`topTable`.

    >>> fit = edge_rs.voom_lm_fit(counts, design, norm_factors=nf)
    >>> fit = edge_rs.ebayes(fit)
    >>> edge_rs.top_table(fit, coef="treated", feature_ids=genes)
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
from beartype import beartype

from edge_rs import _edge_rs
from edge_rs._dgelist import DgeList, unpack
from edge_rs._edge_rs import LinearFit
from edge_rs._validate import (
    CONF,
    Contrast,
    Design,
    Labels,
    as_counts,
    as_design,
    as_matrix,
    coef_positions,
    contrast_matrix,
    effective_lib_size,
    feature_column,
)

Vector = np.ndarray | Sequence[float]
TableSort = Literal["B", "log_fc", "ave_expr", "p_value", "t", "none"]


@beartype(conf=CONF)
@dataclass(frozen=True, kw_only=True, slots=True)
class Voom:
    """voom's log2-CPM and precision weights.

    Attributes:
        e: `(n_genes, n_samples)` log2-CPM. limma's `E`.
        weights: `(n_genes, n_samples)` precision weights.
        amean: Average log2-CPM per gene.
        trend_x: Abscissae of the fitted mean-variance trend.
        trend_y: Ordinates of the trend, sqrt of the residual standard
            deviation.
    """

    e: np.ndarray
    weights: np.ndarray
    amean: np.ndarray
    trend_x: np.ndarray
    trend_y: np.ndarray


@beartype(conf=CONF)
def voom(
    counts: np.ndarray | DgeList,
    design: Design,
    *,
    lib_size: Vector | None = None,
    norm_factors: Vector | None = None,
    span: float = 0.5,
    adaptive_span: bool = True,
    prior_count: float = 0.5,
) -> Voom:
    """Log2-CPM with precision weights from the mean-variance trend, limma's `voom`.

    For a normalised expression matrix. To test, `voom_lm_fit` is the one-step
    route and handles structural zeros; `lm_fit(v.e, design, weights=v.weights)`
    is limma's classic two-step one.

    Args:
        counts: `(n_genes, n_samples)` raw counts, already filtered, or a
            `DgeList`.
        design: `(n_samples, n_coef)` design.
        lib_size: Library size per sample. Defaults to the column sums.
        norm_factors: Normalisation factors, multiplied into `lib_size`.
        span: Lowess span, used only when `adaptive_span` is off.
        adaptive_span: Pick the span from the number of genes.
        prior_count: Count added before the log.

    Returns:
        The `Voom`.

    References:
        Law, Chen, Shi and Smyth, Genome Biology, 2014.
    """
    counts, lib_size, norm_factors = unpack(counts, lib_size, norm_factors)
    y = as_counts(counts)
    mm, _ = as_design(design, y.shape[1])
    lib = effective_lib_size(y, lib_size, norm_factors)
    v = _edge_rs.voom(y, mm, lib, span, adaptive_span, prior_count)
    return Voom(**v)


@beartype(conf=CONF)
def voom_lm_fit(
    counts: np.ndarray | DgeList,
    design: Design,
    *,
    lib_size: Vector | None = None,
    norm_factors: Vector | None = None,
    span: float = 0.5,
    adaptive_span: bool = True,
    prior_count: float = 0.5,
) -> LinearFit:
    """voom and the weighted linear model in one step, edgeR's `voomLmFit`.

    Genes with an all-zero group lose the residual degrees of freedom those
    zeros would otherwise inflate.

    Args:
        counts: `(n_genes, n_samples)` raw counts, already filtered, or a
            `DgeList`.
        design: `(n_samples, n_coef)` design with an intercept.
        lib_size: Library size per sample. Defaults to the column sums. After a
            gene filter, pass the unfiltered column sums.
        norm_factors: Normalisation factors from `calc_norm_factors`.
        span: Lowess span, used only when `adaptive_span` is off.
        adaptive_span: Pick the span from the number of genes.
        prior_count: Count added before the log.

    Returns:
        The `LinearFit`, to hand to `contrasts_fit` or `ebayes`.

    References:
        Law, Chen, Shi and Smyth, Genome Biology, 2014.
    """
    counts, lib_size, norm_factors = unpack(counts, lib_size, norm_factors)
    y = as_counts(counts)
    mm, names = as_design(design, y.shape[1])
    lib = effective_lib_size(y, lib_size, norm_factors)
    return _edge_rs.voom_lm_fit(y, mm, lib, span, adaptive_span, prior_count, names)


@beartype(conf=CONF)
def lm_fit(
    y: np.ndarray,
    design: Design,
    *,
    weights: np.ndarray | None = None,
) -> LinearFit:
    """Least squares fit per gene on log-expression, limma's `lmFit`.

    For limma-trend, fit `cpm(counts, log=True, ...)` and call
    `ebayes(fit, trend=True)`.

    Args:
        y: `(n_genes, n_samples)` log-expression. NaN is missing.
        design: `(n_samples, n_coef)` design.
        weights: `(n_genes, n_samples)` observation weights, e.g. `Voom.weights`.

    Returns:
        The `LinearFit`.
    """
    e = as_matrix(y, "y")
    mm, names = as_design(design, e.shape[1])
    w = None if weights is None else as_matrix(weights, "weights")
    return _edge_rs.lm_fit(e, mm, w, names)


@beartype(conf=CONF)
def make_contrasts(
    contrasts: Mapping[str, str] | Sequence[str],
    levels: Design | Sequence[str],
) -> pl.DataFrame:
    """A contrast matrix from expressions over the design columns, limma's
    `makeContrasts`.

    Supports `+ - * / ( )`, numbers and column names.

    Args:
        contrasts: Contrast name to expression, e.g. `{"BvsA": "B - A"}`. A
            plain list of expressions uses them as their own names.
        levels: The design column names, or the polars design itself.

    Returns:
        One column per contrast, one row per design column. Hand it to
        `contrasts_fit` or `glm_ql_ftest(contrast=...)`.

    Examples:
        >>> c = edge_rs.make_contrasts({"BvsA": "groupB - groupA"}, design)
    """
    named = (
        dict(contrasts) if isinstance(contrasts, Mapping) else {c: c for c in contrasts}
    )
    if isinstance(levels, pl.DataFrame):
        lv = list(levels.columns)
    elif isinstance(levels, np.ndarray):
        raise ValueError("A NumPy design has no column names; pass the names instead.")
    else:
        lv = list(levels)
    m = _edge_rs.make_contrasts(lv, list(named.values()))
    return pl.DataFrame(m, schema=list(named), orient="row")


@beartype(conf=CONF)
def contrasts_fit(fit: LinearFit, contrasts: Contrast) -> LinearFit:
    """Re-express the fit in terms of contrasts, limma's `contrasts.fit`.

    Run it before `ebayes`.

    Args:
        fit: The `LinearFit` from `voom_lm_fit` or `lm_fit`.
        contrasts: `(n_coef, n_contrasts)` weights, a vector for one contrast,
            or the frame `make_contrasts` returns.

    Returns:
        A new `LinearFit` whose coefficients are the contrasts.
    """
    c, names = contrast_matrix(contrasts, fit.n_coef)
    return _edge_rs.contrasts_fit(fit, c.ravel(order="F").tolist(), c.shape[1], names)


@beartype(conf=CONF)
def ebayes(
    fit: LinearFit,
    *,
    proportion: float = 0.01,
    trend: bool = False,
    robust: bool = False,
) -> LinearFit:
    """Empirical Bayes moderated statistics, limma's `eBayes`.

    Args:
        fit: The `LinearFit`.
        proportion: Assumed proportion of differentially expressed genes. Only
            the B-statistic uses it.
        trend: Trend the prior variance against average expression
            (limma-trend). Leave off after voom.
        robust: Robust against outlier genes.

    Returns:
        A new `LinearFit` with `t`, `p_value`, `lods` and friends set.

    References:
        Smyth, Statistical Applications in Genetics and Molecular Biology, 2004.
    """
    return _edge_rs.ebayes(fit, proportion, trend, robust)


@beartype(conf=CONF)
def top_table(
    fit: LinearFit,
    *,
    coef: int | str | None = None,
    feature_ids: Labels | None = None,
    number: int | None = None,
    sort_by: TableSort = "B",
    p_value: float = 1.0,
    lfc: float = 0.0,
    confint: bool | float = False,
) -> pl.DataFrame:
    """The moderated t-test for one coefficient as a table, limma's `topTable`.

    The adjusted p-value is Benjamini-Hochberg over every gene, before
    `number`, `p_value` or `lfc` thin the table.

    Args:
        fit: The `LinearFit`, after `ebayes`.
        coef: 0-based coefficient or its name. `None` is the last one.
        feature_ids: One id per gene of the fit. `None` uses the 0-based row.
        number: Maximum number of rows. `None` keeps every gene.
        sort_by: Ranking column. `"B"` is limma's default.
        p_value: Keep rows with adjusted p-value at most this.
        lfc: Keep rows with absolute log2 fold change at least this.
        confint: Add a 95% interval on the fold change, or the level to use.

    Returns:
        `feature_id`, `log_fc`, `ci_lower` and `ci_upper` (with `confint`),
        `ave_expr`, `t`, `p_value`, `adj_p_value` and `b`.

    Raises:
        ValueError: If `ebayes` has not run, or on an unknown coefficient.
    """
    if fit.t is None:
        raise ValueError("Run `ebayes` on the fit before `top_table`.")
    (j,) = coef_positions(
        fit.n_coef - 1 if coef is None else coef, fit.coef_names, fit.n_coef
    )
    level = (0.95 if confint else None) if isinstance(confint, bool) else confint
    tt = _edge_rs.top_table(
        fit, j, fit.n_genes if number is None else number, sort_by, p_value, lfc, level
    )
    cols = [feature_column(feature_ids, fit.n_genes, tt["index"])]
    for k in (
        "log_fc",
        "ci_lower",
        "ci_upper",
        "ave_expr",
        "t",
        "p_value",
        "adj_p_value",
        "b",
    ):
        if k in tt:
            cols.append(pl.Series(k, tt[k]))
    return pl.DataFrame(cols)
