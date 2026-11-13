//! Filtering, normalisation, CPM and batch removal.
//!
//! `lib_size` here is always the effective library size, `lib.size *
//! norm.factors` in edgeR's terms. The Python layer multiplies the factors in,
//! so none of these functions take them separately.

use edge_rs::core::expression;
use edge_rs::core::filtering::{self, FilterParams};
use edge_rs::core::normalisation::{self, NormMethod};
use edge_rs::limma::remove_batch_effect as rbe;
use edge_rs::prelude::Recycled;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::prelude::*;

use crate::{EdgeErr, c_slice, mat};

/// The `(n_genes, n_samples)` shape and buffer of a count matrix.
///
/// ### Params
///
/// * `counts` - The array
///
/// ### Returns
///
/// The row-major buffer, `n_genes` and `n_samples`.
pub fn rows<'a>(counts: &'a PyReadonlyArray2<'_, f64>) -> PyResult<(&'a [f64], usize, usize)> {
    let shape = counts.shape();
    Ok((c_slice(counts)?, shape[0], shape[1]))
}

/// An optional 1-D array as a slice.
///
/// ### Params
///
/// * `a` - The array, or `None`
///
/// ### Returns
///
/// The buffer, or `None`.
pub fn opt_slice<'a>(a: &'a Option<PyReadonlyArray1<'_, f64>>) -> PyResult<Option<&'a [f64]>> {
    a.as_ref().map(|a| c_slice(a)).transpose()
}

/// An optional 2-D array as a row-major buffer and its column count.
///
/// ### Params
///
/// * `a` - The array, or `None`
///
/// ### Returns
///
/// The buffer and its width, or `None`.
pub fn opt_mat<'a>(
    a: &'a Option<PyReadonlyArray2<'_, f64>>,
) -> PyResult<Option<(&'a [f64], usize)>> {
    a.as_ref()
        .map(|a| Ok((c_slice(a)?, a.shape()[1])))
        .transpose()
}

/// edgeR's `filterByExpr`.
///
/// ### Params
///
/// * `counts` - `(n_genes, n_samples)` raw counts
/// * `lib_size` - Effective library sizes, `None` for the column sums
/// * `group` - 0-based group code per sample. Wins over `design`
/// * `design` - `(n_samples, n_coef)` design
/// * `min_count` - Count a gene must reach in the median-sized library
/// * `min_total_count` - Total count across all samples
/// * `large_n` - Group size past which the sample requirement is damped
/// * `min_prop` - Damping fraction beyond `large_n`
///
/// ### Returns
///
/// One keep flag per gene.
#[pyfunction]
#[pyo3(signature = (counts, lib_size, group, design, min_count, min_total_count, large_n, min_prop))]
#[allow(clippy::too_many_arguments)]
pub fn filter_by_expr<'py>(
    py: Python<'py>,
    counts: PyReadonlyArray2<'py, f64>,
    lib_size: Option<PyReadonlyArray1<'py, f64>>,
    group: Option<Vec<usize>>,
    design: Option<PyReadonlyArray2<'py, f64>>,
    min_count: f64,
    min_total_count: f64,
    large_n: f64,
    min_prop: f64,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let (y, n_genes, n_samples) = rows(&counts)?;
    let lib_size = opt_slice(&lib_size)?;
    let design = opt_mat(&design)?;
    let params = FilterParams::new(min_count, min_total_count, large_n, min_prop);
    let keep = py
        .detach(|| {
            filtering::filter_by_expr(
                y,
                n_genes,
                n_samples,
                lib_size,
                group.as_deref(),
                design,
                Some(params),
            )
        })
        .map_err(EdgeErr::from)?;
    Ok(keep.into_pyarray(py))
}

/// edgeR's `calcNormFactors`.
///
/// ### Params
///
/// * `counts` - `(n_genes, n_samples)` raw counts
/// * `lib_size` - Library sizes, `None` for the column sums
/// * `method` - `"TMM"`, `"TMMwsp"`, `"RLE"`, `"upperquartile"` or `"none"`
/// * `ref_column` - 0-based TMM reference sample, `None` for edgeR's choice
///
/// ### Returns
///
/// One normalisation factor per sample.
#[pyfunction]
#[pyo3(signature = (counts, lib_size, method, ref_column))]
pub fn calc_norm_factors<'py>(
    py: Python<'py>,
    counts: PyReadonlyArray2<'py, f64>,
    lib_size: Option<PyReadonlyArray1<'py, f64>>,
    method: &str,
    ref_column: Option<usize>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (y, n_genes, n_samples) = rows(&counts)?;
    let lib_size = opt_slice(&lib_size)?;
    let method = NormMethod::try_from(method).map_err(EdgeErr::from)?;
    let nf = py
        .detach(|| {
            normalisation::calc_norm_factors(
                y, n_genes, n_samples, lib_size, method, ref_column, None,
            )
        })
        .map_err(EdgeErr::from)?;
    Ok(nf.into_pyarray(py))
}

/// edgeR's `cpm`.
///
/// ### Params
///
/// * `counts` - `(n_genes, n_samples)` raw counts
/// * `lib_size` - Effective library sizes, `None` for the column sums
/// * `log` - Return log2-CPM
/// * `prior_count` - Prior count before library-size scaling, log only
///
/// ### Returns
///
/// `(n_genes, n_samples)` CPM or log2-CPM.
#[pyfunction]
#[pyo3(signature = (counts, lib_size, log, prior_count))]
pub fn cpm<'py>(
    py: Python<'py>,
    counts: PyReadonlyArray2<'py, f64>,
    lib_size: Option<PyReadonlyArray1<'py, f64>>,
    log: bool,
    prior_count: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (y, n_genes, n_samples) = rows(&counts)?;
    let lib_size = opt_slice(&lib_size)?;
    let out = py
        .detach(|| expression::cpm(y, n_genes, n_samples, lib_size, None, log, prior_count))
        .map_err(EdgeErr::from)?;
    mat(py, out, n_genes, n_samples)
}

/// edgeR's `aveLogCPM`.
///
/// ### Params
///
/// * `counts` - `(n_genes, n_samples)` raw counts
/// * `lib_size` - Effective library sizes, `None` for the column sums
/// * `prior_count` - Prior count before library-size scaling
/// * `dispersion` - NB dispersion of the intercept fit, `None` for edgeR's 0.05
///
/// ### Returns
///
/// One average log2-CPM per gene.
#[pyfunction]
#[pyo3(signature = (counts, lib_size, prior_count, dispersion))]
pub fn ave_log_cpm<'py>(
    py: Python<'py>,
    counts: PyReadonlyArray2<'py, f64>,
    lib_size: Option<PyReadonlyArray1<'py, f64>>,
    prior_count: f64,
    dispersion: Option<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (y, n_genes, n_samples) = rows(&counts)?;
    let lib_size = opt_slice(&lib_size)?;
    let dispersion = dispersion.map(Recycled::scalar);
    let out = py
        .detach(|| {
            expression::ave_log_cpm(
                y,
                n_genes,
                n_samples,
                lib_size,
                None,
                prior_count,
                dispersion.as_ref(),
            )
        })
        .map_err(EdgeErr::from)?;
    Ok(out.into_pyarray(py))
}

/// limma's `removeBatchEffect`.
///
/// ### Params
///
/// * `x` - `(n_genes, n_samples)` log-expression
/// * `batch` - 0-based batch code per sample
/// * `batch2` - 0-based second batch code per sample
/// * `covariates` - `(n_samples, n_cov)` numeric covariates to regress out
/// * `design` - `(n_samples, n_coef)` design of interest to protect
///
/// ### Returns
///
/// The corrected `(n_genes, n_samples)` matrix.
#[pyfunction]
#[pyo3(signature = (x, batch, batch2, covariates, design))]
pub fn remove_batch_effect<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    batch: Option<Vec<usize>>,
    batch2: Option<Vec<usize>>,
    covariates: Option<PyReadonlyArray2<'py, f64>>,
    design: Option<PyReadonlyArray2<'py, f64>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (y, n_genes, n_samples) = rows(&x)?;
    let covariates = opt_mat(&covariates)?;
    let design = opt_mat(&design)?;
    let out = py
        .detach(|| {
            rbe::remove_batch_effect(
                y,
                n_genes,
                n_samples,
                batch.as_deref(),
                batch2.as_deref(),
                covariates,
                design,
                None,
            )
        })
        .map_err(EdgeErr::from)?;
    mat(py, out, n_genes, n_samples)
}
