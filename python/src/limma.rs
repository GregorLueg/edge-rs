//! limma's linear model chain: `voom`, `voomLmFit`, `lmFit`, `makeContrasts`,
//! `contrasts.fit`, `eBayes` and `topTable`.
//!
//! The fit object is immutable from Python: `contrasts_fit` and `ebayes`
//! clone it and return a new one, as limma returns a new `MArrayLM`.

use edge_rs::core::normalisation::NormMethod;
use edge_rs::limma::contrasts;
use edge_rs::limma::ebayes::{self as eb, EBayesParams, EBayesTrend};
use edge_rs::limma::lm_fit as lm;
use edge_rs::limma::marray::MArrayLm;
use edge_rs::limma::toptable::{self, TopTableParams, TopTableSort};
use edge_rs::limma::voom::{self as vm, VoomParams};
use edge_rs::prelude::Recycled;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::core::{opt_slice, rows};
use crate::{EdgeErr, mat};

///////////////
// LinearFit //
///////////////

/// A linear model fit, limma's `MArrayLM`.
#[pyclass(frozen, name = "LinearFit", module = "edge_rs._edge_rs")]
pub struct LinearFitPy {
    /// The fit.
    inner: MArrayLm,
    /// Coefficient names: design columns, or contrasts after
    /// `contrasts_fit`. Empty when there were none.
    coef_names: Vec<String>,
}

/// An optional per-gene field as a NumPy array.
///
/// ### Params
///
/// * `py` - GIL token
/// * `v` - The field
///
/// ### Returns
///
/// The array, or `None` while the field is unset.
fn opt_vec<'py>(py: Python<'py>, v: &Option<Vec<f64>>) -> Option<Bound<'py, PyArray1<f64>>> {
    v.clone().map(|v| v.into_pyarray(py))
}

/// An optional `n_genes * n_coef` field as a NumPy matrix.
///
/// ### Params
///
/// * `py` - GIL token
/// * `v` - The field, row-major
/// * `fit` - The fit, for the shape
///
/// ### Returns
///
/// The `(n_genes, n_coef)` array, or `None` while the field is unset.
fn opt_mat<'py>(
    py: Python<'py>,
    v: &Option<Vec<f64>>,
    fit: &MArrayLm,
) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
    v.clone()
        .map(|v| mat(py, v, fit.n_genes, fit.n_coef))
        .transpose()
}

#[pymethods]
impl LinearFitPy {
    /// Number of genes.
    #[getter]
    fn n_genes(&self) -> usize {
        self.inner.n_genes
    }

    /// Width of the coefficient axis: design columns, or contrasts after
    /// `contrasts_fit`.
    #[getter]
    fn n_coef(&self) -> usize {
        self.inner.n_coef
    }

    /// Coefficient names, empty when there were none.
    #[getter]
    fn coef_names(&self) -> Vec<String> {
        self.coef_names.clone()
    }

    /// `(n_genes, n_coef)` coefficients.
    #[getter]
    fn coefficients<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        mat(
            py,
            self.inner.coefficients.clone(),
            self.inner.n_genes,
            self.inner.n_coef,
        )
    }

    /// `(n_genes, n_coef)` unscaled standard deviations.
    #[getter]
    fn stdev_unscaled<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        mat(
            py,
            self.inner.stdev_unscaled.clone(),
            self.inner.n_genes,
            self.inner.n_coef,
        )
    }

    /// Residual standard deviation per gene.
    #[getter]
    fn sigma<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.sigma.clone().into_pyarray(py)
    }

    /// Residual degrees of freedom per gene.
    #[getter]
    fn df_residual<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.df_residual.clone().into_pyarray(py)
    }

    /// Average log-expression per gene. limma's `Amean`.
    #[getter]
    fn amean<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        opt_vec(py, &self.inner.amean)
    }

    /// Prior degrees of freedom, set by `ebayes`.
    #[getter]
    fn df_prior<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        opt_vec(py, &self.inner.df_prior)
    }

    /// Prior variance, set by `ebayes`.
    #[getter]
    fn s2_prior<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        opt_vec(py, &self.inner.s2_prior)
    }

    /// Posterior variance per gene, set by `ebayes`.
    #[getter]
    fn s2_post<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        opt_vec(py, &self.inner.s2_post)
    }

    /// Total degrees of freedom per gene, set by `ebayes`.
    #[getter]
    fn df_total<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        opt_vec(py, &self.inner.df_total)
    }

    /// `(n_genes, n_coef)` moderated t, set by `ebayes`.
    #[getter]
    fn t<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        opt_mat(py, &self.inner.t, &self.inner)
    }

    /// `(n_genes, n_coef)` p-values, set by `ebayes`.
    #[getter]
    fn p_value<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        opt_mat(py, &self.inner.p_value, &self.inner)
    }

    /// `(n_genes, n_coef)` log-odds of differential expression, set by `ebayes`.
    #[getter]
    fn lods<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<f64>>>> {
        opt_mat(py, &self.inner.lods, &self.inner)
    }

    /// Moderated F per gene, set by `ebayes` on a full-rank design.
    #[getter]
    fn f_stat<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        opt_vec(py, &self.inner.f_stat)
    }

    /// P-value of the moderated F.
    #[getter]
    fn f_p_value<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        opt_vec(py, &self.inner.f_p_value)
    }

    /// Short summary.
    fn __repr__(&self) -> String {
        format!(
            "LinearFit(n_genes={}, n_coef={}, ebayes={})",
            self.inner.n_genes,
            self.inner.n_coef,
            self.inner.t.is_some()
        )
    }
}

///////////////
// Functions //
///////////////

/// Voom parameters with edgeR's normalisation left out: the factors are
/// already folded into `lib_size` by the caller.
///
/// ### Params
///
/// * `span` - Lowess span, read only when `adaptive_span` is off
/// * `adaptive_span` - Derive the span from the gene count
/// * `prior_count` - Count added before the log
///
/// ### Returns
///
/// The [`VoomParams`].
fn voom_params(span: f64, adaptive_span: bool, prior_count: f64) -> VoomParams {
    VoomParams::new(NormMethod::None, span, adaptive_span, prior_count, true)
}

/// limma's `voom`.
///
/// ### Params
///
/// * `counts` - `(n_genes, n_samples)` raw counts
/// * `design` - `(n_samples, n_coef)` design
/// * `lib_size` - Effective library sizes, `None` for the column sums
/// * `span` - Lowess span, read only when `adaptive_span` is off
/// * `adaptive_span` - Derive the span from the gene count
/// * `prior_count` - Count added before the log
///
/// ### Returns
///
/// Dict of `e` and `weights` (`(n_genes, n_samples)`), `amean`, and the
/// fitted trend `trend_x`, `trend_y`.
#[pyfunction]
#[pyo3(signature = (counts, design, lib_size, span, adaptive_span, prior_count))]
pub fn voom<'py>(
    py: Python<'py>,
    counts: PyReadonlyArray2<'py, f64>,
    design: PyReadonlyArray2<'py, f64>,
    lib_size: Option<numpy::PyReadonlyArray1<'py, f64>>,
    span: f64,
    adaptive_span: bool,
    prior_count: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let (y, n_genes, n_samples) = rows(&counts)?;
    let (design, _, n_coef) = rows(&design)?;
    let lib_size = opt_slice(&lib_size)?;
    let params = voom_params(span, adaptive_span, prior_count);
    let v = py
        .detach(|| {
            vm::voom(
                y,
                n_genes,
                n_samples,
                design,
                n_coef,
                lib_size,
                None,
                Some(params),
            )
        })
        .map_err(EdgeErr::from)?;

    let out = PyDict::new(py);
    out.set_item("e", mat(py, v.e, n_genes, n_samples)?)?;
    out.set_item("weights", mat(py, v.weights, n_genes, n_samples)?)?;
    out.set_item("amean", v.amean.into_pyarray(py))?;
    out.set_item("trend_x", v.trend_x.into_pyarray(py))?;
    out.set_item("trend_y", v.trend_y.into_pyarray(py))?;
    Ok(out)
}

/// edgeR's `voomLmFit`: voom weights and the weighted fit in one, with the
/// structural-zero degrees of freedom correction.
///
/// ### Params
///
/// * `counts` - `(n_genes, n_samples)` raw counts
/// * `design` - `(n_samples, n_coef)` design
/// * `lib_size` - Effective library sizes, `None` for the column sums
/// * `span` - Lowess span, read only when `adaptive_span` is off
/// * `adaptive_span` - Derive the span from the gene count
/// * `prior_count` - Count added before the log
/// * `coef_names` - Design column names, possibly empty
///
/// ### Returns
///
/// The [`LinearFitPy`], `amean` set.
#[pyfunction]
#[pyo3(signature = (counts, design, lib_size, span, adaptive_span, prior_count, coef_names))]
#[allow(clippy::too_many_arguments)]
pub fn voom_lm_fit<'py>(
    py: Python<'py>,
    counts: PyReadonlyArray2<'py, f64>,
    design: PyReadonlyArray2<'py, f64>,
    lib_size: Option<numpy::PyReadonlyArray1<'py, f64>>,
    span: f64,
    adaptive_span: bool,
    prior_count: f64,
    coef_names: Vec<String>,
) -> PyResult<LinearFitPy> {
    let (y, n_genes, n_samples) = rows(&counts)?;
    let (design, _, n_coef) = rows(&design)?;
    let lib_size = opt_slice(&lib_size)?;
    let mut params = voom_params(span, adaptive_span, prior_count);
    params.save_trend = false;
    let inner = py
        .detach(|| {
            let (v, fit) = vm::voom_lmfit(
                y,
                n_genes,
                n_samples,
                design,
                n_coef,
                lib_size,
                None,
                Some(params),
            )?;
            MArrayLm::from_lm_fit(fit, design, n_coef, n_samples, Some(v.amean))
        })
        .map_err(EdgeErr::from)?;
    Ok(LinearFitPy { inner, coef_names })
}

/// limma's `lmFit`, gene-wise least squares on log-expression.
///
/// ### Params
///
/// * `y` - `(n_genes, n_samples)` log-expression. Non-finite is missing
/// * `design` - `(n_samples, n_coef)` design
/// * `weights` - `(n_genes, n_samples)` observation weights, e.g. voom's
/// * `coef_names` - Design column names, possibly empty
///
/// ### Returns
///
/// The [`LinearFitPy`], `amean` the row means over the finite entries, as
/// limma's `rowMeans(y, na.rm = TRUE)`.
#[pyfunction]
#[pyo3(signature = (y, design, weights, coef_names))]
pub fn lm_fit<'py>(
    py: Python<'py>,
    y: PyReadonlyArray2<'py, f64>,
    design: PyReadonlyArray2<'py, f64>,
    weights: Option<PyReadonlyArray2<'py, f64>>,
    coef_names: Vec<String>,
) -> PyResult<LinearFitPy> {
    let (e, n_genes, n_samples) = rows(&y)?;
    let (design, _, n_coef) = rows(&design)?;
    let weights = weights
        .as_ref()
        .map(|w| {
            let (w, _, _) = rows(w)?;
            Recycled::full(w.to_vec(), n_genes, n_samples).map_err(|e| PyErr::from(EdgeErr(e)))
        })
        .transpose()?;
    let inner = py
        .detach(|| {
            let amean = e
                .chunks_exact(n_samples)
                .map(|row| {
                    let (sum, n) = row
                        .iter()
                        .filter(|v| v.is_finite())
                        .fold((0.0, 0usize), |(s, n), v| (s + v, n + 1));
                    sum / n as f64
                })
                .collect();
            let fit = lm::lm_fit(
                e,
                n_genes,
                n_samples,
                design,
                n_coef,
                weights.as_ref(),
                None,
                None,
            )?;
            MArrayLm::from_lm_fit(fit, design, n_coef, n_samples, Some(amean))
        })
        .map_err(EdgeErr::from)?;
    Ok(LinearFitPy { inner, coef_names })
}

/// limma's `makeContrasts` over named design columns.
///
/// ### Params
///
/// * `levels` - Design column names, in design order
/// * `contrasts` - One expression per contrast, e.g. `"B - A"`
///
/// ### Returns
///
/// The `(n_levels, n_contrasts)` contrast matrix.
#[pyfunction]
pub fn make_contrasts<'py>(
    py: Python<'py>,
    levels: Vec<String>,
    contrasts: Vec<String>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let lv: Vec<&str> = levels.iter().map(String::as_str).collect();
    let cs: Vec<&str> = contrasts.iter().map(String::as_str).collect();
    let (col_major, k) = contrasts::make_contrasts(&lv, &cs).map_err(EdgeErr::from)?;
    let n = lv.len();
    let row_major = (0..n * k)
        .map(|ij| col_major[(ij % k) * n + ij / k])
        .collect();
    mat(py, row_major, n, k)
}

/// limma's `contrasts.fit`. Must run before `ebayes`.
///
/// ### Params
///
/// * `fit` - The [`LinearFitPy`]
/// * `contrasts` - Column-major `n_coef * n_contrasts` weights
/// * `n_contrasts` - Contrast columns
/// * `names` - Contrast names, possibly empty
///
/// ### Returns
///
/// A new [`LinearFitPy`] over the contrasts.
#[pyfunction]
pub fn contrasts_fit(
    py: Python<'_>,
    fit: &LinearFitPy,
    contrasts: Vec<f64>,
    n_contrasts: usize,
    names: Vec<String>,
) -> PyResult<LinearFitPy> {
    let inner = fit.inner.clone();
    let inner = py
        .detach(|| contrasts::contrasts_fit(inner, &contrasts, n_contrasts))
        .map_err(EdgeErr::from)?;
    Ok(LinearFitPy {
        inner,
        coef_names: names,
    })
}

/// limma's `eBayes`.
///
/// ### Params
///
/// * `fit` - The [`LinearFitPy`]
/// * `proportion` - Assumed proportion of differentially expressed genes
/// * `trend` - Trend the prior variance against `amean` (limma-trend)
/// * `robust` - Winsorise the moments against outlier genes
///
/// ### Returns
///
/// A new [`LinearFitPy`] carrying the moderated statistics.
#[pyfunction]
pub fn ebayes(
    py: Python<'_>,
    fit: &LinearFitPy,
    proportion: f64,
    trend: bool,
    robust: bool,
) -> PyResult<LinearFitPy> {
    let inner = fit.inner.clone();
    let params = EBayesParams {
        proportion,
        trend: if trend {
            EBayesTrend::Amean
        } else {
            EBayesTrend::None
        },
        robust,
        ..Default::default()
    };
    let inner = py
        .detach(|| eb::ebayes(inner, Some(params)))
        .map_err(EdgeErr::from)?;
    Ok(LinearFitPy {
        inner,
        coef_names: fit.coef_names.clone(),
    })
}

/// limma's `topTable` for one coefficient.
///
/// ### Params
///
/// * `fit` - The [`LinearFitPy`], after `ebayes`
/// * `coef` - 0-based coefficient
/// * `number` - Maximum number of rows
/// * `sort_by` - `"B"`, `"log_fc"`, `"ave_expr"`, `"p_value"`, `"t"` or `"none"`
/// * `p_value` - Keep rows with adjusted p-value at most this
/// * `lfc` - Keep rows with absolute log fold change at least this
/// * `confint` - Confidence level for the interval on the fold change
///
/// ### Returns
///
/// Dict of `index`, `log_fc`, `t`, `p_value`, `adj_p_value`, `b`, and where
/// available `ave_expr`, `ci_lower`, `ci_upper`, in table order.
#[pyfunction]
#[pyo3(signature = (fit, coef, number, sort_by, p_value, lfc, confint))]
#[allow(clippy::too_many_arguments)]
pub fn top_table<'py>(
    py: Python<'py>,
    fit: &LinearFitPy,
    coef: usize,
    number: usize,
    sort_by: &str,
    p_value: f64,
    lfc: f64,
    confint: Option<f64>,
) -> PyResult<Bound<'py, PyDict>> {
    let sort_by = match sort_by {
        "B" => TopTableSort::B,
        "log_fc" => TopTableSort::LogFc,
        "ave_expr" => TopTableSort::AveExpr,
        "p_value" => TopTableSort::PValue,
        "t" => TopTableSort::T,
        "none" => TopTableSort::None,
        other => {
            return Err(PyValueError::new_err(format!(
                "sort_by must be 'B', 'log_fc', 'ave_expr', 'p_value', 't' or 'none', \
                 got '{other}'"
            )));
        }
    };
    let params = TopTableParams {
        number,
        sort_by,
        p_value,
        lfc,
        confint,
        ..Default::default()
    };
    let tt = toptable::top_table(&fit.inner, coef, Some(params)).map_err(EdgeErr::from)?;

    let out = PyDict::new(py);
    out.set_item("index", tt.index.into_pyarray(py))?;
    out.set_item("log_fc", tt.log_fc.into_pyarray(py))?;
    if let (Some(lo), Some(hi)) = (tt.ci_lower, tt.ci_upper) {
        out.set_item("ci_lower", lo.into_pyarray(py))?;
        out.set_item("ci_upper", hi.into_pyarray(py))?;
    }
    if let Some(a) = tt.ave_expr {
        out.set_item("ave_expr", a.into_pyarray(py))?;
    }
    out.set_item("t", tt.t.into_pyarray(py))?;
    out.set_item("p_value", tt.p_value.into_pyarray(py))?;
    out.set_item("adj_p_value", tt.adj_p_value.into_pyarray(py))?;
    out.set_item("b", tt.b.into_pyarray(py))?;
    Ok(out)
}
