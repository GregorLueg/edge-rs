"""Types for the compiled extension. The public API is in `edge_rs`."""

from typing import Any

import numpy as np

class EdgeRsError(Exception): ...

class QLFit:
    """A fitted quasi-likelihood negative binomial GLM, edgeR's `DGEGLM`."""

    @property
    def n_genes(self) -> int: ...
    @property
    def n_samples(self) -> int: ...
    @property
    def n_coef(self) -> int: ...
    @property
    def coef_names(self) -> list[str]: ...
    @property
    def coefficients(self) -> np.ndarray: ...
    @property
    def deviance(self) -> np.ndarray: ...
    @property
    def s2_post(self) -> np.ndarray: ...
    @property
    def s2_prior(self) -> np.ndarray: ...
    @property
    def df_prior(self) -> np.ndarray: ...
    @property
    def ave_log_cpm(self) -> np.ndarray: ...
    @property
    def average_ql_dispersion(self) -> float | None: ...

class LinearFit:
    """A linear model fit, limma's `MArrayLM`."""

    @property
    def n_genes(self) -> int: ...
    @property
    def n_coef(self) -> int: ...
    @property
    def coef_names(self) -> list[str]: ...
    @property
    def coefficients(self) -> np.ndarray: ...
    @property
    def stdev_unscaled(self) -> np.ndarray: ...
    @property
    def sigma(self) -> np.ndarray: ...
    @property
    def df_residual(self) -> np.ndarray: ...
    @property
    def amean(self) -> np.ndarray | None: ...
    @property
    def df_prior(self) -> np.ndarray | None: ...
    @property
    def s2_prior(self) -> np.ndarray | None: ...
    @property
    def s2_post(self) -> np.ndarray | None: ...
    @property
    def df_total(self) -> np.ndarray | None: ...
    @property
    def t(self) -> np.ndarray | None: ...
    @property
    def p_value(self) -> np.ndarray | None: ...
    @property
    def lods(self) -> np.ndarray | None: ...
    @property
    def f_stat(self) -> np.ndarray | None: ...
    @property
    def f_p_value(self) -> np.ndarray | None: ...

def filter_by_expr(
    counts: np.ndarray,
    lib_size: np.ndarray | None,
    group: list[int] | None,
    design: np.ndarray | None,
    min_count: float,
    min_total_count: float,
    large_n: float,
    min_prop: float,
) -> np.ndarray: ...
def calc_norm_factors(
    counts: np.ndarray,
    lib_size: np.ndarray | None,
    method: str,
    ref_column: int | None,
) -> np.ndarray: ...
def cpm(
    counts: np.ndarray, lib_size: np.ndarray | None, log: bool, prior_count: float
) -> np.ndarray: ...
def ave_log_cpm(
    counts: np.ndarray,
    lib_size: np.ndarray | None,
    prior_count: float,
    dispersion: float | None,
) -> np.ndarray: ...
def remove_batch_effect(
    x: np.ndarray,
    batch: list[int] | None,
    batch2: list[int] | None,
    covariates: np.ndarray | None,
    design: np.ndarray | None,
) -> np.ndarray: ...
def glm_ql_fit(
    counts: np.ndarray,
    design: np.ndarray,
    lib_size: np.ndarray,
    ave_log_cpm: np.ndarray | None,
    robust: bool,
    legacy: bool,
    coef_names: list[str],
) -> QLFit: ...
def glm_ql_ftest(
    fit: QLFit,
    coef: list[int] | None,
    contrast: list[float] | None,
    n_contrasts: int,
) -> dict[str, Any]: ...
def top_tags(
    log_fc: np.ndarray,
    log_cpm: np.ndarray,
    statistic: np.ndarray,
    p_value: np.ndarray,
    n: int,
    sort_by: str,
    p_cutoff: float,
) -> dict[str, np.ndarray]: ...
def voom(
    counts: np.ndarray,
    design: np.ndarray,
    lib_size: np.ndarray | None,
    span: float,
    adaptive_span: bool,
    prior_count: float,
) -> dict[str, np.ndarray]: ...
def voom_lm_fit(
    counts: np.ndarray,
    design: np.ndarray,
    lib_size: np.ndarray | None,
    span: float,
    adaptive_span: bool,
    prior_count: float,
    coef_names: list[str],
) -> LinearFit: ...
def lm_fit(
    y: np.ndarray,
    design: np.ndarray,
    weights: np.ndarray | None,
    coef_names: list[str],
) -> LinearFit: ...
def make_contrasts(levels: list[str], contrasts: list[str]) -> np.ndarray: ...
def contrasts_fit(
    fit: LinearFit, contrasts: list[float], n_contrasts: int, names: list[str]
) -> LinearFit: ...
def ebayes(
    fit: LinearFit, proportion: float, trend: bool, robust: bool
) -> LinearFit: ...
def top_table(
    fit: LinearFit,
    coef: int,
    number: int,
    sort_by: str,
    p_value: float,
    lfc: float,
    confint: float | None,
) -> dict[str, np.ndarray]: ...
def nebula_cell_major(
    data: np.ndarray,
    indices: np.ndarray,
    indptr: np.ndarray,
    n_vars: int,
    cells: np.ndarray,
    subject_run: np.ndarray,
    genes: np.ndarray,
    design: np.ndarray,
    offset: np.ndarray | None = None,
    coef: int = 0,
    contrast: list[float] | None = None,
    method: str = "ln",
    cpc: float = 0.005,
    mincp: int = 5,
    min_subjects: int = 0,
    shrink_dispersion: bool = True,
    gene_block_size: int = 1000,
) -> dict[str, Any]: ...
