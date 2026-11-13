"""Filtering, normalisation, CPM and batch removal.

Counts are `(n_genes, n_samples)`, as in edgeR. `lib_size` defaults to the
column sums of the counts passed in, so after subsetting genes pass the full
matrix's column sums to keep edgeR's `keep.lib.sizes = TRUE` behaviour.
"""

from collections.abc import Sequence

import numpy as np
from beartype import beartype

from edge_rs import _edge_rs
from edge_rs._validate import (
    CONF,
    Design,
    Labels,
    as_codes,
    as_counts,
    as_design,
    as_matrix,
    as_vector,
    effective_lib_size,
)

Vector = np.ndarray | Sequence[float]


@beartype(conf=CONF)
def filter_by_expr(
    counts: np.ndarray,
    *,
    group: Labels | None = None,
    design: Design | None = None,
    lib_size: Vector | None = None,
    norm_factors: Vector | None = None,
    min_count: float = 10.0,
    min_total_count: float = 15.0,
    large_n: float = 10.0,
    min_prop: float = 0.7,
) -> np.ndarray:
    """Keep genes with enough counts to be worth testing, edgeR's `filterByExpr`.

    Args:
        counts: `(n_genes, n_samples)` raw counts.
        group: One group label per sample. Takes precedence over `design`.
        design: `(n_samples, n_coef)` design. Without `group` or `design` all
            samples are one group.
        lib_size: Library size per sample. Defaults to the column sums.
        norm_factors: Normalisation factors, multiplied into `lib_size`.
        min_count: Count a gene must reach in the median-sized library.
        min_total_count: Total count across all samples a gene must reach.
        large_n: Group size past which the sample requirement is damped.
        min_prop: Fraction of samples beyond `large_n` that still count.

    Returns:
        A boolean mask over the genes.

    Raises:
        ValueError: On mismatched shapes or negative counts.

    References:
        Chen, Lun and Smyth, F1000Research, 2016.

    Examples:
        >>> keep = edge_rs.filter_by_expr(counts, design=design)
        >>> counts = counts[keep]
    """
    y = as_counts(counts)
    n = y.shape[1]
    return _edge_rs.filter_by_expr(
        y,
        effective_lib_size(y, lib_size, norm_factors),
        None if group is None else as_codes(group, n, "group"),
        None if design is None else as_design(design, n)[0],
        min_count,
        min_total_count,
        large_n,
        min_prop,
    )


@beartype(conf=CONF)
def calc_norm_factors(
    counts: np.ndarray,
    *,
    method: str = "TMM",
    lib_size: Vector | None = None,
    ref_column: int | None = None,
) -> np.ndarray:
    """Library size normalisation factors, edgeR's `calcNormFactors`.

    Args:
        counts: `(n_genes, n_samples)` raw counts.
        method: `"TMM"`, `"TMMwsp"`, `"RLE"`, `"upperquartile"` or `"none"`.
        lib_size: Library size per sample. Defaults to the column sums. Pass
            the unfiltered column sums after a gene filter, as edgeR does.
        ref_column: 0-based reference sample for TMM. `None` picks edgeR's.

    Returns:
        One factor per sample, with a geometric mean of one.

    Raises:
        ValueError: On an unknown method or mismatched shapes.

    References:
        Robinson and Oshlack, Genome Biology, 2010.
    """
    y = as_counts(counts)
    lib = None if lib_size is None else as_vector(lib_size, y.shape[1], "lib_size")
    return _edge_rs.calc_norm_factors(y, lib, method, ref_column)


@beartype(conf=CONF)
def cpm(
    counts: np.ndarray,
    *,
    lib_size: Vector | None = None,
    norm_factors: Vector | None = None,
    log: bool = False,
    prior_count: float = 2.0,
) -> np.ndarray:
    """Counts per million, edgeR's `cpm`.

    Args:
        counts: `(n_genes, n_samples)` raw counts.
        lib_size: Library size per sample. Defaults to the column sums.
        norm_factors: Normalisation factors, multiplied into `lib_size`.
        log: Return log2-CPM.
        prior_count: Count added before the log, scaled by library size.

    Returns:
        `(n_genes, n_samples)` CPM or log2-CPM.
    """
    y = as_counts(counts)
    lib = effective_lib_size(y, lib_size, norm_factors)
    return _edge_rs.cpm(y, lib, log, prior_count)


@beartype(conf=CONF)
def ave_log_cpm(
    counts: np.ndarray,
    *,
    lib_size: Vector | None = None,
    norm_factors: Vector | None = None,
    prior_count: float = 2.0,
    dispersion: float | None = None,
) -> np.ndarray:
    """Average log2-CPM per gene, edgeR's `aveLogCPM`.

    Not the mean of `cpm(log=True)`: an intercept-only negative binomial fit
    per gene.

    Args:
        counts: `(n_genes, n_samples)` raw counts.
        lib_size: Library size per sample. Defaults to the column sums.
        norm_factors: Normalisation factors, multiplied into `lib_size`.
        prior_count: Count added before the log, scaled by library size.
        dispersion: Negative binomial dispersion. `None` is edgeR's 0.05.

    Returns:
        One value per gene.
    """
    y = as_counts(counts)
    lib = effective_lib_size(y, lib_size, norm_factors)
    return _edge_rs.ave_log_cpm(y, lib, prior_count, dispersion)


@beartype(conf=CONF)
def remove_batch_effect(
    x: np.ndarray,
    batch: Labels | None = None,
    *,
    batch2: Labels | None = None,
    covariates: np.ndarray | None = None,
    design: Design | None = None,
) -> np.ndarray:
    """Regress batch effects out of log-expression, limma's `removeBatchEffect`.

    For plots and clustering. Do not feed the result into a linear model: put
    the batch in the design there instead.

    Args:
        x: `(n_genes, n_samples)` log-expression, e.g. `cpm(..., log=True)`.
        batch: One batch label per sample.
        batch2: A second batch label per sample.
        covariates: `(n_samples, n_cov)` numeric covariates to regress out.
        design: `(n_samples, n_coef)` design of the effects to keep.

    Returns:
        The corrected `(n_genes, n_samples)` matrix.
    """
    e = as_matrix(x, "x")
    n = e.shape[1]
    return _edge_rs.remove_batch_effect(
        e,
        None if batch is None else as_codes(batch, n, "batch"),
        None if batch2 is None else as_codes(batch2, n, "batch2"),
        None if covariates is None else as_matrix(covariates, "covariates"),
        None if design is None else as_design(design, n)[0],
    )
