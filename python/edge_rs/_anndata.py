"""NEBULA on `AnnData`, returning a pandas frame.

The counts are read straight out of the layer's CSR buffers: the Rust side
transposes them one gene block at a time, so peak memory is the matrix plus
one block. Scanpy itself is never imported.

Imported on first access of ``edge_rs.nebula``, so `anndata` stays an optional
dependency.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from beartype import beartype
from scipy import sparse

from edge_rs import _edge_rs
from edge_rs._validate import CONF, coef_positions

###########
# Globals #
###########

INTERCEPT: str = "(Intercept)"


###########
# Helpers #
###########


@beartype(conf=CONF)
def _counts(adata: AnnData, layer: str | None) -> tuple[sparse.csr_matrix, pd.Index]:
    """The count matrix as a canonical CSR, and its gene names.

    A layer of that name wins over ``"raw"``, which otherwise means
    ``adata.raw.X``. CSR with `float32` or `float64` data passes through
    uncopied; anything else costs one conversion.

    Raises:
        ValueError: If the layer does not exist.
        TypeError: If the matrix is neither sparse nor a NumPy array.
    """
    match layer:
        case None:
            x, var_names = adata.X, adata.var_names
        case str() if layer in adata.layers:
            x, var_names = adata.layers[layer], adata.var_names
        case "raw" if adata.raw is not None:
            x, var_names = adata.raw.X, adata.raw.var_names
        case _:
            raise ValueError(
                f"No layers[{layer!r}]. Layers are {list(adata.layers)}; "
                "`layer=None` reads X, `layer='raw'` reads adata.raw.X."
            )
    if not isinstance(x, np.ndarray | sparse.spmatrix | sparse.sparray):
        raise TypeError(
            f"The counts are a {type(x).__name__}; load them into memory as a "
            "scipy sparse matrix or a NumPy array."
        )
    # Shares the buffers of a CSR matrix or array; converts anything else.
    csr = sparse.csr_matrix(x)
    if not csr.has_canonical_format:
        csr = csr.copy()
        csr.sum_duplicates()
    if csr.dtype not in (np.float32, np.float64):
        csr = sparse.csr_matrix(
            (csr.data.astype(np.float32), csr.indices, csr.indptr), shape=csr.shape
        )
    return csr, var_names


@beartype(conf=CONF)
def _formula_terms(formula: str) -> list[str]:
    """The main effects of an additive formula, ``"~ a + b"``.

    Raises:
        ValueError: On anything but ``~`` and ``+``-separated column names.
    """
    lhs, sep, rhs = formula.partition("~")
    if not sep or lhs.strip():
        raise ValueError(f"The design must be a one-sided formula, got {formula!r}.")
    terms = [t.strip() for t in rhs.split("+")]
    terms = [t for t in terms if t != "1"]
    bad = [t for t in terms if not re.fullmatch(r"[A-Za-z_.][A-Za-z0-9_.]*", t)]
    if bad or not terms:
        raise ValueError(
            f"Only additive main effects of obs columns are supported, got {formula!r}."
        )
    return terms


@beartype(conf=CONF)
def _factor_codes(values: pd.Series) -> tuple[np.ndarray, list[str]]:
    """0-based factor codes and level names.

    A categorical keeps its level order, minus unused levels. Anything else is
    sorted; booleans are named ``FALSE`` / ``TRUE``, as in R.
    """
    if isinstance(values.dtype, pd.CategoricalDtype):
        values = values.cat.remove_unused_categories()
        levels = [str(c) for c in values.cat.categories]
        return values.cat.codes.to_numpy(dtype=np.int64), levels
    uniq, codes = np.unique(values.to_numpy(), return_inverse=True)
    if values.dtype == bool:
        return codes, [str(v).upper() for v in uniq]
    return codes, [str(v) for v in uniq]


@beartype(conf=CONF)
def _model_matrix(obs: pd.DataFrame, formula: str) -> tuple[np.ndarray, list[str]]:
    """The design for additive main effects with an intercept.

    Numeric columns enter as they are; any other column is treatment coded
    against its first level, one ``<col><level>`` column per other level.
    """
    cols = [np.ones(obs.shape[0])]
    names = [INTERCEPT]
    for term in _formula_terms(formula):
        values = obs[term]
        if pd.api.types.is_numeric_dtype(values) and values.dtype != bool:
            cols.append(values.to_numpy(dtype=np.float64))
            names.append(term)
            continue
        codes, levels = _factor_codes(values)
        for k, level in enumerate(levels[1:], start=1):
            cols.append((codes == k).astype(np.float64))
            names.append(f"{term}{level}")
    return np.ascontiguousarray(np.column_stack(cols)), names


@beartype(conf=CONF)
def _gene_indices(
    genes: Sequence[str] | np.ndarray | None, var_names: pd.Index
) -> np.ndarray:
    """0-based `uint32` genes from names or a boolean mask, all when `None`.

    Raises:
        ValueError: On an unknown or repeated name, or a mask of the wrong length.
    """
    if genes is None:
        return np.arange(len(var_names), dtype=np.uint32)
    if isinstance(genes, np.ndarray) and genes.dtype == bool:
        if genes.shape != (len(var_names),):
            raise ValueError(f"The gene mask needs {len(var_names)} entries.")
        return np.flatnonzero(genes).astype(np.uint32)
    idx = var_names.get_indexer(list(genes))
    if (idx < 0).any():
        missing = [g for g, i in zip(genes, idx, strict=True) if i < 0][:5]
        raise ValueError(f"Genes not in var_names, e.g. {missing}.")
    if np.unique(idx).size != idx.size:
        raise ValueError("`genes` lists a gene twice.")
    return idx.astype(np.uint32)


###############
# Entry point #
###############


@beartype(conf=CONF)
def nebula(
    adata: AnnData,
    subject_key: str,
    design: str,
    *,
    coef: int | str | None = None,
    contrast: np.ndarray | Sequence[float] | None = None,
    layer: str | None = "counts",
    genes: Sequence[str] | np.ndarray | None = None,
    offset: np.ndarray | pd.Series | Sequence[float] | None = None,
    method: Literal["ln", "hl"] = "ln",
    cpc: float = 0.005,
    mincp: int = 5,
    min_subjects: int = 0,
    shrink_dispersion: bool = True,
    gene_block_size: int = 1000,
) -> pd.DataFrame:
    """NEBULA's negative binomial gamma mixed model, per gene.

    The variance splits into a subject-level random effect and a cell-level
    overdispersion, so donors are modelled rather than pseudoreplicated. A fit
    takes milliseconds to seconds per gene, so pass the HVGs or a candidate set
    as `genes` rather than all of them. Cells need not be sorted by subject.

    Args:
        adata: Annotated data matrix with raw counts.
        subject_key: obs column with the subject (donor), the random effect.
        design: Additive formula over obs, e.g. ``"~ condition + age"``. Numeric
            columns enter as they are; anything else is treatment coded against
            its first level (category order, else sorted).
        coef: The coefficient the Wald test reports, a 0-based design column
            or its name, e.g. ``"conditiontreated"``. `None` takes the last.
        contrast: One weight per design column. Exclusive with `coef`.
        layer: Where the counts are. ``None`` reads ``X``, ``"raw"`` reads
            ``adata.raw.X`` unless a layer of that name exists.
        genes: Gene names or a boolean mask over var. `None` fits every gene.
        offset: Strictly positive scaling per cell, one per obs. `None` uses
            the library size over every gene of the layer.
        method: ``"ln"`` for NEBULA-LN, ``"hl"`` for NEBULA-HL. LN falls back
            to HL below 30 cells per subject, as in R.
        cpc: Drop a gene whose mean count per cell is at most this.
        mincp: Drop a gene expressed in fewer cells than this.
        min_subjects: Drop a gene fewer subjects than this express. ``0`` is off.
        shrink_dispersion: Also shrink the cell-level overdispersions towards
            an empirical Bayes prior. Does not change the test.
        gene_block_size: Genes transposed and fitted at a time. Bounds the
            memory on top of the counts.

    Returns:
        One row per gene that passed the expression filter, indexed by gene
        name in input order: ``log_fc`` (natural log), ``se``, ``z``,
        ``p_value``, ``fdr``, ``subject_overdispersion`` (sigma^2),
        ``cell_overdispersion`` (phi^-1), ``cell_overdispersion_shrunk`` when
        shrinking, ``convergence`` (at or below -20 is a likely failure) and
        ``sigma_at_bound``.

    Raises:
        ValueError: On a missing layer or column, non-integral counts, both
            `coef` and `contrast`, a bad offset, or no gene passing the filter.

    References:
        He et al., Communications Biology 4, 629, 2021.

    Examples:
        >>> res = edge_rs.nebula(adata, "donor", "~ condition", genes=hvgs)
        >>> res.sort_values("p_value").head()
    """
    x, var_names = _counts(adata, layer)
    terms = _formula_terms(design)
    needed = list(dict.fromkeys([*terms, subject_key]))
    obs_all = adata.obs
    if not isinstance(obs_all, pd.DataFrame):
        raise TypeError("obs is not a pandas DataFrame; load the AnnData into memory.")
    missing = [c for c in needed if c not in obs_all.columns]
    if missing:
        raise ValueError(f"obs has no column(s) {missing}.")

    complete = obs_all[needed].notna().all(axis=1).to_numpy()
    if not complete.all():
        warnings.warn(
            f"Dropping {int((~complete).sum())} cell(s) with a missing design or "
            "subject value.",
            stacklevel=2,
        )
    cells = np.flatnonzero(complete)
    if cells.size == 0:
        raise ValueError(
            "No cells left after dropping missing design or subject values."
        )

    subject_codes, subjects = pd.factorize(obs_all[subject_key].iloc[cells])
    if len(subjects) < 2:
        warnings.warn(
            "Only one subject found. The subject-level variance has nothing to "
            "estimate from and NEBULA collapses to a plain negative binomial.",
            stacklevel=2,
        )
    # Cells grouped by subject, the order the Rust side reads the subjects in.
    order = np.argsort(subject_codes, kind="stable")
    cells = cells[order]
    obs = obs_all.iloc[cells]
    mm, names = _model_matrix(obs, design)

    if coef is not None and contrast is not None:
        raise ValueError("Supply either `coef` or `contrast`, not both.")
    weights = None
    if contrast is not None:
        weights = np.asarray(contrast, dtype=np.float64)
        if weights.shape != (mm.shape[1],) or not np.isfinite(weights).all():
            raise ValueError(f"The contrast needs {mm.shape[1]} finite weights.")
    pos = coef_positions(mm.shape[1] - 1 if coef is None else coef, names, mm.shape[1])

    off = None
    if offset is not None:
        off = np.asarray(offset, dtype=np.float64).ravel()
        if off.shape != (adata.n_obs,) or not (np.isfinite(off) & (off > 0)).all():
            raise ValueError(
                f"`offset` needs {adata.n_obs} finite, strictly positive values."
            )
        off = np.ascontiguousarray(off[cells])

    res = _edge_rs.nebula_cell_major(
        x.data,
        x.indices,
        x.indptr.astype(x.indices.dtype, copy=False),
        x.shape[1],
        cells.astype(np.uint32),
        subject_codes[order].astype(np.uint32),
        _gene_indices(genes, var_names),
        mm,
        offset=off,
        coef=pos[0],
        contrast=None if weights is None else weights.tolist(),
        method=method,
        cpc=cpc,
        mincp=mincp,
        min_subjects=min_subjects,
        shrink_dispersion=shrink_dispersion,
        gene_block_size=gene_block_size,
    )
    cols = [
        "log_fc",
        "se",
        "z",
        "p_value",
        "fdr",
        "subject_overdispersion",
        "cell_overdispersion",
        "cell_overdispersion_shrunk",
        "convergence",
        "sigma_at_bound",
    ]
    return pd.DataFrame(
        {c: res[c] for c in cols if res[c] is not None},
        index=var_names[res["gene_idx"]],
    )
