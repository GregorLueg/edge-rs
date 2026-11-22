"""`DgeList`: counts with the library sizes that belong to them.

A lite edgeR `DGEList`. Its point is `keep.lib.sizes = TRUE`: subsetting genes
keeps the library sizes of the full matrix, so a filter followed by TMM cannot
silently renormalise against the filtered column sums.

    >>> dge = edge_rs.make_dgelist(counts, genes=pl.DataFrame({"gene_id": ids}))
    >>> dge = edge_rs.calc_norm_factors(dge[edge_rs.filter_by_expr(dge, design=design)])
    >>> fit = edge_rs.glm_ql_fit(dge, design)
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
from beartype import beartype

from edge_rs._validate import CONF, as_counts, as_vector

Vector = np.ndarray | Sequence[float]


@beartype(conf=CONF)
@dataclass(frozen=True, kw_only=True, slots=True)
class DgeList:
    """Counts, library sizes, normalisation factors and annotation.

    Build it with `make_dgelist`, which checks the shapes. Every function that
    takes counts takes a `DgeList` in their place and reads `lib_size` and
    `norm_factors` from it.

    Attributes:
        counts: `(n_genes, n_samples)` raw counts.
        lib_size: Library size per sample, of the matrix before any gene subset.
        norm_factors: Normalisation factor per sample.
        genes: One row per gene, or `None`.
        samples: One row per sample, or `None`.
    """

    counts: np.ndarray
    lib_size: np.ndarray
    norm_factors: np.ndarray
    genes: pl.DataFrame | None = None
    samples: pl.DataFrame | None = None

    @property
    def effective_lib_size(self) -> np.ndarray:
        """`lib_size * norm_factors`, edgeR's effective library size."""
        return self.lib_size * self.norm_factors

    @property
    def shape(self) -> tuple[int, int]:
        """`(n_genes, n_samples)`."""
        return self.counts.shape

    def __getitem__(self, keep: np.ndarray | Sequence[int] | slice) -> "DgeList":
        """The genes in `keep`, a boolean mask or 0-based rows, with the library
        sizes and normalisation factors untouched."""
        rows = np.arange(self.shape[0])[keep]
        return DgeList(
            counts=np.ascontiguousarray(self.counts[rows]),
            lib_size=self.lib_size,
            norm_factors=self.norm_factors,
            genes=None if self.genes is None else self.genes[rows],
            samples=self.samples,
        )


@beartype(conf=CONF)
def make_dgelist(
    counts: np.ndarray,
    *,
    lib_size: Vector | None = None,
    norm_factors: Vector | None = None,
    genes: pl.DataFrame | None = None,
    samples: pl.DataFrame | None = None,
) -> DgeList:
    """A `DgeList` from its parts, edgeR's `DGEList`.

    Args:
        counts: `(n_genes, n_samples)` raw counts, before any gene filter.
        lib_size: Library size per sample. Defaults to the column sums.
        norm_factors: Normalisation factor per sample. Defaults to ones.
        genes: Gene annotation, one row per gene.
        samples: Sample annotation, one row per sample.

    Returns:
        The `DgeList`.

    Raises:
        ValueError: On negative counts, a non-positive library size, or
            annotation of the wrong height.
    """
    y = as_counts(counts)
    n_genes, n = y.shape
    lib = y.sum(axis=0) if lib_size is None else as_vector(lib_size, n, "lib_size")
    if (lib <= 0).any():
        raise ValueError("Every library size must be positive.")
    nf = (
        np.ones(n)
        if norm_factors is None
        else as_vector(norm_factors, n, "norm_factors")
    )
    if genes is not None and genes.height != n_genes:
        raise ValueError(
            f"`genes` needs one row per gene ({n_genes}), got {genes.height}."
        )
    if samples is not None and samples.height != n:
        raise ValueError(
            f"`samples` needs one row per sample ({n}), got {samples.height}."
        )
    return DgeList(
        counts=y, lib_size=lib, norm_factors=nf, genes=genes, samples=samples
    )


@beartype(conf=CONF)
def unpack(
    counts: np.ndarray | DgeList,
    lib_size: Vector | None,
    norm_factors: Vector | None,
) -> tuple[np.ndarray, Vector | None, Vector | None]:
    """Counts, library sizes and normalisation factors from either input.

    Raises:
        ValueError: If a `DgeList` comes with its own `lib_size` or
            `norm_factors` as well.
    """
    if not isinstance(counts, DgeList):
        return counts, lib_size, norm_factors
    if lib_size is not None or norm_factors is not None:
        raise ValueError("A `DgeList` carries its own `lib_size` and `norm_factors`.")
    return counts.counts, counts.lib_size, counts.norm_factors
