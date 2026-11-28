//! Python bindings for the bulk edgeR and limma-voom stacks of edge-rs, and
//! NEBULA over AnnData's cell-major CSR.
//!
//! Thin by design: arrays in, arrays out, and the Python package on top does
//! the validation, the defaults and the polars tables. Every matrix crosses
//! row-major and C-contiguous, counts as `(n_genes, n_samples)` `float64`, and
//! every index is 0-based.

use edge_rs::prelude::EdgeErrors;
use numpy::{IntoPyArray, PyArray2, PyArrayMethods, PyReadonlyArray, PyUntypedArrayMethods};
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;

mod core;
mod edger;
mod limma;
mod sc;

////////////////
// Exceptions //
////////////////

create_exception!(
    _edge_rs,
    EdgeRsError,
    PyException,
    "Base class for every numerical error raised by the edge-rs Rust core."
);

/// Carries an [`EdgeErrors`] to the FFI boundary.
///
/// Both types are foreign here, so the `From` impl needs a local newtype.
/// `.map_err(EdgeErr::from)?` in any function returning [`PyResult`].
pub struct EdgeErr(
    /// The error the crate raised.
    pub EdgeErrors,
);

impl From<EdgeErrors> for EdgeErr {
    fn from(e: EdgeErrors) -> Self {
        Self(e)
    }
}

impl From<EdgeErr> for PyErr {
    /// Malformed arguments become `ValueError`, everything else (rank
    /// deficiency, failed factorisations, no convergence) `EdgeRsError`.
    fn from(e: EdgeErr) -> PyErr {
        let msg = e.0.to_string();
        match e.0 {
            EdgeErrors::InvalidArgument(_)
            | EdgeErrors::MustBePositive(_)
            | EdgeErrors::UnknownMethod { .. }
            | EdgeErrors::ShapeMismatch { .. }
            | EdgeErrors::LengthMismatch { .. }
            | EdgeErrors::EmptyCounts { .. }
            | EdgeErrors::CoefOutOfRange { .. }
            | EdgeErrors::InvalidReferenceColumn { .. } => PyValueError::new_err(msg),
            _ => EdgeRsError::new_err(msg),
        }
    }
}

/////////////
// Helpers //
/////////////

/// The flat buffer of a NumPy array, in C order only.
///
/// rust-numpy's `as_slice` also accepts an F-contiguous array and hands back
/// its column-major buffer, which a row-major reader would then take silently
/// transposed. `pb.counts.T` is exactly such an array.
///
/// ### Params
///
/// * `a` - The array
///
/// ### Returns
///
/// The buffer, or a `ValueError` unless the array is C-contiguous.
pub fn c_slice<'a, T: numpy::Element, D: numpy::ndarray::Dimension>(
    a: &'a PyReadonlyArray<'_, T, D>,
) -> PyResult<&'a [T]> {
    if !a.is_c_contiguous() {
        return Err(PyValueError::new_err(format!(
            "expected a C-contiguous (row-major) array of shape {:?}; pass \
             np.ascontiguousarray(x)",
            a.shape()
        )));
    }
    Ok(a.as_slice()?)
}

/// A row-major buffer as a 2-D NumPy array.
///
/// ### Params
///
/// * `py` - GIL token
/// * `v` - Row-major values, `rows * cols`
/// * `rows` - Number of rows
/// * `cols` - Number of columns
///
/// ### Returns
///
/// The `(rows, cols)` array.
pub fn mat<'py>(
    py: Python<'py>,
    v: Vec<f64>,
    rows: usize,
    cols: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    v.into_pyarray(py).reshape([rows, cols])
}

////////////
// Module //
////////////

/// The `edge_rs._edge_rs` extension module.
///
/// ### Params
///
/// * `m` - The module being initialised
///
/// ### Returns
///
/// Unit, or the error pyo3 raised while registering.
#[pymodule]
fn _edge_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("EdgeRsError", m.py().get_type::<EdgeRsError>())?;
    m.add_class::<edger::QlFitPy>()?;
    m.add_class::<limma::LinearFitPy>()?;
    for f in [
        wrap_pyfunction!(core::filter_by_expr, m)?,
        wrap_pyfunction!(core::calc_norm_factors, m)?,
        wrap_pyfunction!(core::cpm, m)?,
        wrap_pyfunction!(core::ave_log_cpm, m)?,
        wrap_pyfunction!(core::remove_batch_effect, m)?,
        wrap_pyfunction!(edger::glm_ql_fit, m)?,
        wrap_pyfunction!(edger::glm_ql_ftest, m)?,
        wrap_pyfunction!(edger::top_tags, m)?,
        wrap_pyfunction!(limma::voom, m)?,
        wrap_pyfunction!(limma::voom_lm_fit, m)?,
        wrap_pyfunction!(limma::lm_fit, m)?,
        wrap_pyfunction!(limma::make_contrasts, m)?,
        wrap_pyfunction!(limma::contrasts_fit, m)?,
        wrap_pyfunction!(limma::ebayes, m)?,
        wrap_pyfunction!(limma::top_table, m)?,
        wrap_pyfunction!(sc::nebula_cell_major, m)?,
    ] {
        m.add_function(f)?;
    }
    Ok(())
}
