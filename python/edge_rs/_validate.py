"""Argument checks and conversions shared by every public function.

Everything that crosses into Rust goes through here: C-contiguous `float64`
matrices, 0-based integer codes, effective library sizes, and the tested
coefficients or contrasts.
"""

from collections.abc import Sequence

import numpy as np
import polars as pl
from beartype import BeartypeConf, beartype

###########
# Globals #
###########

CONF = BeartypeConf(is_pep484_tower=True)

Design = np.ndarray | pl.DataFrame
Coef = int | str | Sequence[int] | Sequence[str]
Labels = Sequence[object] | np.ndarray | pl.Series
Contrast = np.ndarray | Sequence[float] | pl.DataFrame


###############
# Conversions #
###############


@beartype(conf=CONF)
def as_matrix(x: np.ndarray, what: str) -> np.ndarray:
    """A 2-D C-contiguous `float64` copy or view of `x`.

    Raises:
        ValueError: If `x` is not 2-D.
    """
    if x.ndim != 2:
        raise ValueError(f"`{what}` must be a 2-D matrix, got {x.ndim}-D.")
    return np.ascontiguousarray(x, dtype=np.float64)


@beartype(conf=CONF)
def as_counts(counts: np.ndarray) -> np.ndarray:
    """Counts as a `(n_genes, n_samples)` C-contiguous `float64` matrix.

    Raises:
        ValueError: If the counts are not 2-D, or hold negative values.
    """
    y = as_matrix(counts, "counts")
    if (y < 0).any():
        raise ValueError("`counts` holds negative values. This wants raw counts.")
    return y


@beartype(conf=CONF)
def as_vector(x: np.ndarray | Sequence[float], n: int, what: str) -> np.ndarray:
    """A 1-D C-contiguous `float64` vector of length `n`.

    Raises:
        ValueError: On the wrong length.
    """
    v = np.ascontiguousarray(x, dtype=np.float64).ravel()
    if v.shape[0] != n:
        raise ValueError(f"`{what}` needs {n} values, got {v.shape[0]}.")
    return v


@beartype(conf=CONF)
def as_design(design: Design, n_samples: int) -> tuple[np.ndarray, list[str]]:
    """A design as a contiguous `float64` matrix and its column names.

    An array has no names, so its coefficients can only be picked by position.

    Raises:
        ValueError: If the design does not have one row per sample.
    """
    if isinstance(design, pl.DataFrame):
        mm, names = design.to_numpy(), list(design.columns)
    else:
        mm, names = design, []
    mm = as_matrix(mm, "design")
    if mm.shape[0] != n_samples:
        raise ValueError(
            f"The design has {mm.shape[0]} rows against {n_samples} samples."
        )
    return mm, names


@beartype(conf=CONF)
def as_codes(labels: Labels, n_samples: int, what: str) -> list[int]:
    """0-based integer codes for per-sample labels, levels in sorted order.

    Raises:
        ValueError: If there is not one label per sample.
    """
    values = labels.to_numpy() if isinstance(labels, pl.Series) else labels
    _, codes = np.unique(np.asarray(values), return_inverse=True)
    if codes.shape[0] != n_samples:
        raise ValueError(f"`{what}` needs one label per sample ({n_samples}).")
    return codes.astype(int).tolist()


@beartype(conf=CONF)
def effective_lib_size(
    counts: np.ndarray,
    lib_size: np.ndarray | Sequence[float] | None,
    norm_factors: np.ndarray | Sequence[float] | None,
) -> np.ndarray | None:
    """`lib_size * norm_factors`, edgeR's effective library size.

    `lib_size` defaults to the column sums. Returns `None` when neither is
    given, which leaves the column sums to the Rust side.
    """
    if lib_size is None and norm_factors is None:
        return None
    n = counts.shape[1]
    lib = counts.sum(axis=0) if lib_size is None else as_vector(lib_size, n, "lib_size")
    if norm_factors is not None:
        lib = lib * as_vector(norm_factors, n, "norm_factors")
    return np.ascontiguousarray(lib, dtype=np.float64)


@beartype(conf=CONF)
def feature_column(feature_ids: Labels | None, n: int, index: np.ndarray) -> pl.Series:
    """The `feature_id` column of a result table.

    Args:
        feature_ids: One id per gene of the fit, or `None` for the row index.
        n: Number of genes in the fit.
        index: 0-based gene of each table row.

    Raises:
        ValueError: If `feature_ids` does not have one id per gene.
    """
    if feature_ids is None:
        return pl.Series("feature_id", index, dtype=pl.UInt32)
    ids = feature_ids if isinstance(feature_ids, pl.Series) else pl.Series(feature_ids)
    if ids.len() != n:
        raise ValueError(f"`feature_ids` needs one id per gene ({n}), got {ids.len()}.")
    return ids.gather(index).alias("feature_id")


##############
# Hypotheses #
##############


@beartype(conf=CONF)
def coef_positions(coef: Coef, names: Sequence[str], n_coef: int) -> list[int]:
    """0-based coefficients from positions or names.

    Raises:
        ValueError: On an unknown name or a position outside the design.
    """
    wanted = [coef] if isinstance(coef, int | str) else list(coef)
    out = []
    for c in wanted:
        if isinstance(c, str):
            if c not in names:
                raise ValueError(f"Coefficient {c!r} is not one of {list(names)}.")
            out.append(list(names).index(c))
        elif not 0 <= c < n_coef:
            raise ValueError(f"Coefficient {c} is outside 0..{n_coef - 1}.")
        else:
            out.append(c)
    return out


@beartype(conf=CONF)
def contrast_matrix(contrast: Contrast, n_coef: int) -> tuple[np.ndarray, list[str]]:
    """A contrast as an `(n_coef, n_contrasts)` matrix and its names.

    Raises:
        ValueError: If the contrast does not have one row per coefficient.
    """
    if isinstance(contrast, pl.DataFrame):
        c, names = contrast.to_numpy(), list(contrast.columns)
    else:
        c, names = np.asarray(contrast, dtype=np.float64), []
    c = c.reshape(-1, 1) if c.ndim == 1 else c
    if c.shape[0] != n_coef or np.isnan(c).any():
        raise ValueError(
            f"The contrast has {c.shape[0]} rows against {n_coef} coefficients."
        )
    return np.asarray(c, dtype=np.float64), names


@beartype(conf=CONF)
def tested(
    names: Sequence[str],
    n_coef: int,
    coef: Coef | None,
    contrast: Contrast | None,
) -> tuple[list[int] | None, np.ndarray | None]:
    """The tested `(coef, contrast)`, `coef` defaulting to the last column.

    The contrast comes back as an `(n_coef, n_contrasts)` matrix.

    Raises:
        ValueError: If both are given.
    """
    if coef is not None and contrast is not None:
        raise ValueError("Supply either `coef` or `contrast`, not both.")
    if contrast is not None:
        return None, contrast_matrix(contrast, n_coef)[0]
    return coef_positions(n_coef - 1 if coef is None else coef, names, n_coef), None
