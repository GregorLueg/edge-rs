"""Behaviour of the Python layer itself: orientation, names, pipelines."""

import numpy as np
import polars as pl
import pytest

import edge_rs


def test_transposed_samples_by_genes_input_matches(fac):
    # `pb.counts.T` from a bixverse pseudobulk is F-contiguous.
    counts_f = np.ascontiguousarray(fac["counts"].T).T
    assert not counts_f.flags.c_contiguous
    a = edge_rs.run_edger_ql(counts_f, fac["design"], coef="grpB")
    b = edge_rs.run_edger_ql(fac["counts"], fac["design"], coef="grpB")
    assert a.equals(b)


def test_run_edger_ql_matches_the_steps(fac):
    counts, design = fac["counts"], fac["design"]
    genes = [f"g{i}" for i in range(counts.shape[0])]
    res = edge_rs.run_edger_ql(counts, design, feature_ids=genes, coef="grpB")

    keep = edge_rs.filter_by_expr(counts, design=design)
    lib = counts.sum(axis=0)
    nf = edge_rs.calc_norm_factors(counts[keep], lib_size=lib)
    fit = edge_rs.glm_ql_fit(counts[keep], design, lib_size=lib, norm_factors=nf)
    tt = edge_rs.top_tags(
        edge_rs.glm_ql_ftest(fit, coef=1),
        feature_ids=np.asarray(genes)[keep],
        sort_by="none",
    )
    assert res.equals(tt)
    assert res.columns == [
        "feature_id",
        "log_fc",
        "log_cpm",
        "f_stat",
        "p_value",
        "fdr",
    ]


def test_run_limma_voom_contrast_equals_coef(fac):
    design = fac["design"]
    con = edge_rs.make_contrasts({"B": "grpB"}, design)
    a = edge_rs.run_limma_voom(fac["counts"], design, contrast=con)
    b = edge_rs.run_limma_voom(fac["counts"], design, coef="grpB")
    for col in ["log_fc", "t", "p_value", "adj_p_value"]:
        np.testing.assert_allclose(a[col], b[col], rtol=1e-10)


def test_make_contrasts_layout():
    c = edge_rs.make_contrasts({"BvsA": "B - A", "C": "C"}, ["A", "B", "C"])
    assert c.columns == ["BvsA", "C"]
    np.testing.assert_array_equal(c.to_numpy(), [[-1, 0], [1, 0], [0, 1]])


def test_contrast_names_carry_through(fac):
    con = edge_rs.make_contrasts({"batch": "batb2", "group": "grpB"}, fac["design"])
    fit = edge_rs.contrasts_fit(edge_rs.voom_lm_fit(fac["kept"], fac["design"]), con)
    assert fit.coef_names == ["batch", "group"]
    tt = edge_rs.top_table(edge_rs.ebayes(fit), coef="group", number=5)
    assert tt.height == 5


def test_errors():
    rng = np.random.default_rng(1)
    counts = rng.poisson(20, size=(50, 6)).astype(float)
    design = np.column_stack([np.ones(6), [0, 0, 0, 1, 1, 1]])
    with pytest.raises(ValueError, match="negative"):
        edge_rs.cpm(-counts)
    with pytest.raises(ValueError, match="rows against"):
        edge_rs.glm_ql_fit(counts, design[:5])
    with pytest.raises(ValueError, match="not one of"):
        edge_rs.glm_ql_ftest(edge_rs.glm_ql_fit(counts, design), coef="grp")
    with pytest.raises(ValueError, match="ebayes"):
        edge_rs.top_table(edge_rs.lm_fit(edge_rs.cpm(counts, log=True), design))
    rank_deficient = np.column_stack([design, design[:, 1]])
    with pytest.raises(edge_rs.EdgeRsError):
        edge_rs.glm_ql_fit(counts, rank_deficient)


def test_polars_design_and_labels():
    rng = np.random.default_rng(2)
    counts = rng.poisson(30, size=(80, 8)).astype(float)
    design = pl.DataFrame({"intercept": [1.0] * 8, "treated": [0.0] * 4 + [1.0] * 4})
    corrected = edge_rs.remove_batch_effect(
        edge_rs.cpm(counts, log=True), pl.Series(["a", "b"] * 4), design=design
    )
    assert corrected.shape == counts.shape
    res = edge_rs.run_limma_voom(counts, design, filter=False)
    assert res.height == 80
    assert res["feature_id"].dtype == pl.UInt32
