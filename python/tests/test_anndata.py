"""`edge_rs.nebula` on AnnData against nebula 1.5.9.

The numerics are gated in `tests/e2e_nebula.rs`; this checks the AnnData glue:
the layer read, the subject reorder, the formula, the offset alignment and the
gene-block transpose. Tolerances are the E2E ones.
"""

import numpy as np
import pandas as pd
import pytest
from conftest import DATA, assert_close, read
from scipy import sparse

anndata = pytest.importorskip("anndata")

import edge_rs  # noqa: E402

# Cells are shuffled so that the subject reorder and the offset alignment run.
SEED = 7


@pytest.fixture(scope="module")
def sc() -> "anndata.AnnData":
    """The 300-gene, 1005-cell, 15-subject fixture as cells x genes, shuffled."""
    counts = read("sc_counts").to_numpy().astype(np.float32).T
    meta = pd.read_csv(DATA / "sc_meta.csv")
    perm = np.random.default_rng(SEED).permutation(counts.shape[0])
    obs = meta.iloc[perm].reset_index(drop=True)
    obs.index = obs.index.astype(str)
    obs["subject"] = obs["subject"].astype(str)
    var = pd.DataFrame(index=[f"g{i + 1}" for i in range(counts.shape[1])])
    adata = anndata.AnnData(X=sparse.csr_matrix(counts[perm]), obs=obs, var=var)
    adata.layers["counts"] = adata.X.copy()
    return adata


def _fit(
    adata: "anndata.AnnData",
    *,
    layer: str | None = "counts",
    gene_block_size: int = 1000,
) -> pd.DataFrame:
    return edge_rs.nebula(
        adata,
        "subject",
        "~ grp + cov2",
        coef="grp",
        layer=layer,
        offset=adata.obs["offset"],
        shrink_dispersion=False,
        gene_block_size=gene_block_size,
    )


def test_nebula_matches_r(sc):
    want = pd.read_csv(DATA / "sc_nebula.csv")
    got = _fit(sc)
    assert list(got.index) == [f"g{int(i)}" for i in want["gene_id"]]

    # nebula's LN+HL sub-path is gated apart in the E2E suite, as here.
    pure = (want["algorithm"] != 2).to_numpy()
    mixed = ~pure
    assert_close(got["log_fc"][pure], want["logFC_grp"][pure], rel=3e-5, eps=1e-9)
    assert_close(got["se"][pure], want["se_grp"][pure], rel=1e-4)
    assert_close(got["p_value"][pure], want["p_grp"][pure], rel=1e-2, eps=1e-300)
    assert_close(
        got["subject_overdispersion"][pure], want["Subject"][pure], rel=2e-4, eps=1e-6
    )
    assert_close(got["cell_overdispersion"][pure], want["Cell"][pure], rel=1e-4)
    assert_close(got["log_fc"][mixed], want["logFC_grp"][mixed], rel=1e-5, eps=1e-2)


def test_nebula_gene_blocks_do_not_change_the_fit(sc):
    pd.testing.assert_frame_equal(
        _fit(sc, gene_block_size=64), _fit(sc, gene_block_size=1000)
    )


def test_nebula_rejects_normalised_counts(sc):
    adata = sc.copy()
    adata.X = adata.X.multiply(1.0 / 3.0).tocsr().astype(np.float32)
    with pytest.raises(ValueError, match="layer="):
        _fit(adata, layer=None)
