"""Parity against edgeR and limma through the Python layer.

The numerics are gated in the Rust E2E suites; these check that the bindings
hand the right buffers, orientation and defaults across.
"""

import numpy as np
from conftest import assert_close, read

import edge_rs


def test_filter_by_expr_matches_edger(fac):
    want = read("fac_filter")
    got = edge_rs.filter_by_expr(fac["counts"], design=fac["design"])
    np.testing.assert_array_equal(got, want["design"].to_numpy().astype(bool))
    got = edge_rs.filter_by_expr(fac["counts"], group=fac["samples"]["grp"])
    np.testing.assert_array_equal(got, want["group"].to_numpy().astype(bool))


def test_calc_norm_factors_matches_edger(fac):
    want = read("fac_norm")
    for method, col in [("TMM", "tmm"), ("TMMwsp", "tmmwsp"), ("RLE", "rle")]:
        got = edge_rs.calc_norm_factors(fac["counts"], method=method)
        assert_close(got, want[col].to_numpy(), rel=1e-13)
    got = edge_rs.calc_norm_factors(fac["counts"], ref_column=2)
    assert_close(got, want["tmm_ref3"].to_numpy(), rel=1e-13)


def test_cpm_matches_edger(fac):
    kw = {"lib_size": fac["lib_size"], "norm_factors": fac["norm_factors"]}
    assert_close(edge_rs.cpm(fac["kept"], **kw), read("fac_cpm").to_numpy(), rel=1e-13)
    assert_close(
        edge_rs.cpm(fac["kept"], log=True, **kw),
        read("fac_logcpm").to_numpy(),
        rel=1e-12,
        eps=1e-13,
    )
    assert_close(
        edge_rs.ave_log_cpm(fac["kept"], **kw),
        read("fac_avelogcpm")["default"].to_numpy(),
        rel=2e-9,
    )


def test_glm_ql_ftest_matches_edger(fac):
    want = read("fac_ql")
    fit = edge_rs.glm_ql_fit(
        fac["kept"],
        fac["design"],
        lib_size=fac["lib_size"],
        norm_factors=fac["norm_factors"],
        ave_log_cpm=read("fac_disp")["ave_log_cpm"].to_numpy(),
    )
    res = edge_rs.glm_ql_ftest(fit, coef="grpB")
    assert_close(res.f_stat, want["F"].to_numpy(), rel=1e-5, eps=1e-4)
    assert_close(res.p_value, want["PValue"].to_numpy(), rel=1e-3, eps=1e-300)
    assert_close(fit.s2_post, want["s2_post"].to_numpy(), rel=2e-4, eps=1e-9)


def test_voom_ebayes_toptable_matches_limma(fac):
    fit = edge_rs.ebayes(edge_rs.voom_lm_fit(fac["kept"], fac["design"]))
    assert fit.t is not None
    assert_close(
        fit.t, read("fac_ebayes").select("t1", "t2", "t3").to_numpy(), rel=1e-8
    )

    want = read("fac_toptable_confint")
    tt = edge_rs.top_table(fit, coef="grpB", confint=True)
    np.testing.assert_array_equal(
        tt["feature_id"].to_numpy(), want["index"].to_numpy() - 1
    )
    for got, col in [
        ("log_fc", "logFC"),
        ("ave_expr", "AveExpr"),
        ("t", "t"),
        ("p_value", "P.Value"),
        ("adj_p_value", "adj.P.Val"),
        ("b", "B"),
    ]:
        assert_close(tt[got].to_numpy(), want[col].to_numpy(), rel=5e-6, eps=2e-6)
    assert_close(tt["ci_lower"].to_numpy(), want["CI.L"].to_numpy(), rel=1e-6, eps=5e-8)


def test_contrasts_fit_matches_limma(fac):
    con = read("fac_contrasts").to_numpy().T
    fit = edge_rs.contrasts_fit(edge_rs.voom_lm_fit(fac["kept"], fac["design"]), con)
    want = read("fac_contrasts_fit")
    assert_close(
        fit.coefficients, want.select("coef1", "coef2").to_numpy(), rel=1e-10, eps=1e-13
    )
