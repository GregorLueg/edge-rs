"""Shared fixtures: the committed edgeR 4.8.2 / limma 3.66.0 parity tables.

They live with the Rust E2E suites in `tests/data/e2e` at the repository root
and are written by `tests/r/generate_fixtures.R`. The tolerances in the tests
are the ones measured for the matching Rust constants.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

DATA = Path(__file__).resolve().parents[2] / "tests" / "data" / "e2e"


def read(name: str) -> pl.DataFrame:
    """One fixture table."""
    return pl.read_csv(DATA / f"{name}.csv", infer_schema_length=None)


def assert_close(
    got: np.ndarray, want: np.ndarray, rel: float, eps: float = 0.0
) -> None:
    """`approx::relative_eq!` semantics: within `eps` absolutely or `rel`
    relatively, as the Rust E2E suites compare."""
    got, want = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    assert got.shape == want.shape
    diff = np.abs(got - want)
    ok = (diff <= eps) | (diff <= rel * np.maximum(np.abs(got), np.abs(want)))
    assert ok.all(), f"worst abs {diff.max():.3e} at {int(diff.argmax())}"


@pytest.fixture(scope="session")
def fac() -> dict[str, object]:
    """The 2x2 factorial dataset, unfiltered and filtered."""
    design = read("fac_design")
    kept_samples = read("fac_kept_samples")
    return {
        "counts": read("fac_counts").to_numpy().astype(np.float64),
        "kept": read("fac_kept_counts").to_numpy().astype(np.float64),
        "design": design,
        "samples": read("fac_samples"),
        "lib_size": kept_samples["lib_size"].to_numpy(),
        "norm_factors": kept_samples["norm_factors"].to_numpy(),
    }
