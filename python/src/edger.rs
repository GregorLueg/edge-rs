//! edgeR's quasi-likelihood chain: `glmQLFit`, `glmQLFTest` and `topTags`.
//!
//! The fit object keeps the counts, design and offsets alongside the fit,
//! because the F-test refits the null model and needs them again.

use edge_rs::core::expression;
use edge_rs::dispersion::estimate::estimate_disp;
use edge_rs::glm::ql_fit::{self, QlFit, QlFitParams};
use edge_rs::glm::test::{GlmTestInput, Tested, glm_ql_ftest as ql_ftest};
use edge_rs::prelude::Recycled;
use edge_rs::results::{self, SortBy};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::core::{opt_slice, rows};
use crate::{EdgeErr, c_slice, mat};

////////////
// Consts //
////////////

/// Prior count `aveLogCPM` adds before the log. edgeR's default.
const AVE_LOG_CPM_PRIOR: f64 = 2.0;

///////////
// QLFit //
///////////

/// A fitted quasi-likelihood negative binomial GLM, edgeR's `DGEGLM`.
#[pyclass(frozen, name = "QLFit", module = "edge_rs._edge_rs")]
pub struct QlFitPy {
    /// The fit.
    fit: QlFit,
    /// Counts the fit ran on, row-major `n_genes * n_samples`.
    counts: Vec<f64>,
    /// Number of genes.
    n_genes: usize,
    /// Number of samples.
    n_samples: usize,
    /// Design, row-major `n_samples * n_coef`.
    design: Vec<f64>,
    /// Number of design columns.
    n_coef: usize,
    /// Log effective library size per sample.
    offset: Recycled<f64>,
    /// Average log2-CPM per gene.
    ave_log_cpm: Vec<f64>,
    /// Design column names, empty when the design had none.
    coef_names: Vec<String>,
}

#[pymethods]
impl QlFitPy {
    /// Number of genes.
    #[getter]
    fn n_genes(&self) -> usize {
        self.n_genes
    }

    /// Number of samples.
    #[getter]
    fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Number of design columns.
    #[getter]
    fn n_coef(&self) -> usize {
        self.n_coef
    }

    /// Design column names, empty when the design had none.
    #[getter]
    fn coef_names(&self) -> Vec<String> {
        self.coef_names.clone()
    }

    /// `(n_genes, n_coef)` coefficients on the natural log scale.
    #[getter]
    fn coefficients<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        mat(py, self.fit.coefficients.clone(), self.n_genes, self.n_coef)
    }

    /// Residual deviance per gene.
    #[getter]
    fn deviance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.fit.deviance.clone().into_pyarray(py)
    }

    /// Posterior quasi-likelihood dispersion per gene.
    #[getter]
    fn s2_post<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.fit.s2_post.clone().into_pyarray(py)
    }

    /// Prior quasi-likelihood dispersion, length one or one per gene.
    #[getter]
    fn s2_prior<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.fit.s2_prior.clone().into_pyarray(py)
    }

    /// Prior degrees of freedom, length one or one per gene.
    #[getter]
    fn df_prior<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.fit.df_prior.clone().into_pyarray(py)
    }

    /// Average log2-CPM per gene.
    #[getter]
    fn ave_log_cpm<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.ave_log_cpm.clone().into_pyarray(py)
    }

    /// Average quasi-dispersion the NB dispersion was divided by. `None` on
    /// the legacy pipeline.
    #[getter]
    fn average_ql_dispersion(&self) -> Option<f64> {
        self.fit.average_ql_dispersion
    }

    /// Short summary.
    fn __repr__(&self) -> String {
        format!(
            "QLFit(n_genes={}, n_samples={}, n_coef={})",
            self.n_genes, self.n_samples, self.n_coef
        )
    }
}

///////////////
// Functions //
///////////////

/// edgeR's `glmQLFit`.
///
/// On the current pipeline the NB dispersion is estimated from the most
/// abundant genes. The legacy one needs a trended dispersion first, which is
/// what `estimateDisp` is run for there and only there.
///
/// ### Params
///
/// * `counts` - `(n_genes, n_samples)` raw counts
/// * `design` - `(n_samples, n_coef)` design
/// * `lib_size` - Effective library size per sample
/// * `ave_log_cpm` - Average log2-CPM per gene, `None` to compute it
/// * `robust` - Robust empirical Bayes squeezing
/// * `legacy` - edgeR's pre-4.0 pipeline
/// * `coef_names` - Design column names, possibly empty
///
/// ### Returns
///
/// The [`QlFitPy`].
#[pyfunction]
#[pyo3(signature = (counts, design, lib_size, ave_log_cpm, robust, legacy, coef_names))]
#[allow(clippy::too_many_arguments)]
pub fn glm_ql_fit<'py>(
    py: Python<'py>,
    counts: PyReadonlyArray2<'py, f64>,
    design: PyReadonlyArray2<'py, f64>,
    lib_size: PyReadonlyArray1<'py, f64>,
    ave_log_cpm: Option<PyReadonlyArray1<'py, f64>>,
    robust: bool,
    legacy: bool,
    coef_names: Vec<String>,
) -> PyResult<QlFitPy> {
    let (y, n_genes, n_samples) = rows(&counts)?;
    let (design, _, n_coef) = rows(&design)?;
    let offset = Recycled::by_sample(c_slice(&lib_size)?.iter().map(|l| l.ln()).collect());
    let given = opt_slice(&ave_log_cpm)?;

    let (fit, abundance) = py
        .detach(|| {
            let abundance = match given {
                Some(a) => a.to_vec(),
                None => expression::ave_log_cpm(
                    y,
                    n_genes,
                    n_samples,
                    None,
                    Some(&offset),
                    AVE_LOG_CPM_PRIOR,
                    None,
                )?,
            };
            let dispersion = if legacy {
                let disp = estimate_disp(
                    y,
                    n_genes,
                    n_samples,
                    design,
                    n_coef,
                    &offset,
                    None,
                    Some(&abundance),
                    None,
                )?;
                let per_gene = disp.trended.unwrap_or_else(|| vec![disp.common; n_genes]);
                Some(Recycled::by_gene(per_gene))
            } else {
                None
            };
            let fit = ql_fit::glm_ql_fit(
                y,
                n_genes,
                n_samples,
                design,
                n_coef,
                dispersion.as_ref(),
                &offset,
                None,
                &abundance,
                Some(QlFitParams {
                    robust,
                    legacy,
                    ..Default::default()
                }),
            )?;
            Ok::<_, edge_rs::prelude::EdgeErrors>((fit, abundance))
        })
        .map_err(EdgeErr::from)?;

    Ok(QlFitPy {
        fit,
        counts: y.to_vec(),
        n_genes,
        n_samples,
        design: design.to_vec(),
        n_coef,
        offset,
        ave_log_cpm: abundance,
        coef_names,
    })
}

/// edgeR's `glmQLFTest`, with the Poisson bound on as upstream.
///
/// ### Params
///
/// * `fit` - The [`QlFitPy`]
/// * `coef` - 0-based design columns to drop from the null
/// * `contrast` - Column-major `n_coef * n_contrasts` weights; wins over `coef`
/// * `n_contrasts` - Contrast columns
///
/// ### Returns
///
/// Dict of `log_fc`, `log_cpm`, `f_stat`, `p_value`, `df_total` per gene and
/// the scalar `df_test`.
#[pyfunction]
#[pyo3(signature = (fit, coef, contrast, n_contrasts))]
pub fn glm_ql_ftest<'py>(
    py: Python<'py>,
    fit: &QlFitPy,
    coef: Option<Vec<usize>>,
    contrast: Option<Vec<f64>>,
    n_contrasts: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let tested = match (contrast, coef) {
        (Some(values), _) => Tested::Contrast {
            values,
            n_contrasts,
        },
        (None, Some(coef)) => Tested::Coef(coef),
        (None, None) => {
            return Err(PyValueError::new_err(
                "the QL F-test needs either `coef` or `contrast`",
            ));
        }
    };
    let test = py
        .detach(|| {
            let input = GlmTestInput {
                counts: &fit.counts,
                n_genes: fit.n_genes,
                n_samples: fit.n_samples,
                design: &fit.design,
                n_coef: fit.n_coef,
                dispersion: &fit.fit.dispersion,
                offset: &fit.offset,
                weights: None,
                log_cpm: Some(&fit.ave_log_cpm),
            };
            ql_ftest(
                &input,
                &fit.fit.as_glm_fit(),
                &fit.fit.ql_summary(),
                &tested,
                true,
            )
        })
        .map_err(EdgeErr::from)?;

    let out = PyDict::new(py);
    out.set_item("log_fc", test.log_fc.into_pyarray(py))?;
    out.set_item(
        "log_cpm",
        test.log_cpm
            .unwrap_or_else(|| fit.ave_log_cpm.clone())
            .into_pyarray(py),
    )?;
    out.set_item("f_stat", test.statistic.into_pyarray(py))?;
    out.set_item("p_value", test.p_value.into_pyarray(py))?;
    out.set_item("df_test", test.df_test)?;
    if let Some(df_total) = test.df_total {
        out.set_item("df_total", df_total.into_pyarray(py))?;
    }
    Ok(out)
}

/// edgeR's `topTags`. FDR is Benjamini-Hochberg over the whole table before
/// any thinning.
///
/// ### Params
///
/// * `log_fc` - Log2 fold change per gene
/// * `log_cpm` - Average log2-CPM per gene
/// * `statistic` - Test statistic per gene
/// * `p_value` - Raw p-value per gene
/// * `n` - Maximum number of rows
/// * `sort_by` - `"p_value"`, `"log_fc"` or `"none"`
/// * `p_cutoff` - Keep rows with FDR at most this
///
/// ### Returns
///
/// Dict of `index`, `log_fc`, `log_cpm`, `statistic`, `p_value`, `fdr`, in
/// table order.
#[pyfunction]
#[pyo3(signature = (log_fc, log_cpm, statistic, p_value, n, sort_by, p_cutoff))]
#[allow(clippy::too_many_arguments)]
pub fn top_tags<'py>(
    py: Python<'py>,
    log_fc: PyReadonlyArray1<'py, f64>,
    log_cpm: PyReadonlyArray1<'py, f64>,
    statistic: PyReadonlyArray1<'py, f64>,
    p_value: PyReadonlyArray1<'py, f64>,
    n: usize,
    sort_by: &str,
    p_cutoff: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let sort_by = match sort_by {
        "p_value" => SortBy::PValue,
        "log_fc" => SortBy::LogFc,
        "none" => SortBy::None,
        other => {
            return Err(PyValueError::new_err(format!(
                "sort_by must be 'p_value', 'log_fc' or 'none', got '{other}'"
            )));
        }
    };
    let tt = results::top_tags(
        c_slice(&log_fc)?,
        c_slice(&log_cpm)?,
        Some(c_slice(&statistic)?),
        c_slice(&p_value)?,
        n,
        sort_by,
        p_cutoff,
    )
    .map_err(EdgeErr::from)?;

    let out = PyDict::new(py);
    out.set_item("index", tt.index.into_pyarray(py))?;
    out.set_item("log_fc", tt.log_fc.into_pyarray(py))?;
    out.set_item("log_cpm", tt.log_cpm.into_pyarray(py))?;
    out.set_item("statistic", tt.statistic.into_pyarray(py))?;
    out.set_item("p_value", tt.p_value.into_pyarray(py))?;
    out.set_item("fdr", tt.fdr.into_pyarray(py))?;
    Ok(out)
}
