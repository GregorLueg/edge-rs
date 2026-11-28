//! NEBULA over a cell-major CSR, the layout AnnData hands over.
//!
//! NEBULA wants gene-major counts over subject-contiguous cells. Genes are
//! fitted independently, so the sweep runs in gene blocks: each block is
//! transposed out of the borrowed CSR, fitted and dropped. Peak memory is the
//! caller's matrix plus one block, and only the block is ever `f64`. The Wald
//! test and the dispersion shrinkage need every gene, so they run once the
//! sweep is in, as bixverse's `run_nebula` does.

use edge_rs::prelude::*;
use edge_rs::sc::nebula::{NebulaMethod, NebulaParams, nebula_sparse};
use edge_rs::sc::shrink::{sc_residual_df, shrink_sc_dispersion};
use edge_rs::sc::test::{ScTested, glm_sc_test};
use numpy::{Element, IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::core::{opt_slice, rows};
use crate::{EdgeErr, c_slice};

////////////
// Consts //
////////////

/// Added to the mean count per cell before the log, for the shrinkage
/// covariate. edgePython's `log2(mean + 0.5)`, as bixverse uses it.
const SHRINK_COVARIATE_OFFSET: f64 = 0.5;

/// Marks a gene outside the current block in the gene to slot map.
const NO_SLOT: u32 = u32::MAX;

///////////
// Types //
///////////

/// A cell-major CSR over `(n_obs, n_vars)`, borrowed from scipy.
struct CellMajor<'a, T, I> {
    /// Stored values, one per non-zero.
    data: &'a [T],
    /// Gene of each stored value.
    indices: &'a [I],
    /// Row offsets, `n_obs + 1`, already checked and widened.
    indptr: Vec<usize>,
    /// Number of genes.
    n_vars: usize,
}

/// Everything about one sweep except the counts.
struct Sweep<'a> {
    /// Rows of the CSR to fit, grouped by subject.
    cells: &'a [usize],
    /// Subject of each entry of `cells`.
    subject_run: &'a [usize],
    /// Genes to fit, as CSR columns. No duplicates.
    genes: &'a [usize],
    /// Design, row-major `cells.len() * n_coef`, in `cells` order.
    design: &'a [f64],
    /// Number of design columns.
    n_coef: usize,
    /// Scaling per cell in `cells` order, or `None` for the library sizes.
    offset: Option<&'a [f64]>,
    /// Upstream NEBULA knobs.
    nebula: NebulaParams,
    /// Coefficient or contrast the Wald test reports.
    tested: ScTested,
    /// Shrink the cell-level overdispersions once the sweep is done.
    shrink: bool,
    /// Genes transposed and fitted per block.
    block: usize,
}

/// The fit for every gene that survived the expression filter.
struct SweepOut {
    /// CSR column of each kept gene.
    gene_idx: Vec<usize>,
    /// Subject-level overdispersion, nebula's `sigma^2`.
    subject_overdispersion: Vec<f64>,
    /// Cell-level overdispersion, nebula's `phi^-1`.
    cell_overdispersion: Vec<f64>,
    /// Shrunken cell-level overdispersion, `None` unless asked for.
    cell_overdispersion_shrunk: Option<Vec<f64>>,
    /// nebula's convergence code.
    convergence: Vec<i32>,
    /// Whether `sigma^2` finished on its lower bound.
    sigma_at_bound: Vec<bool>,
    /// Tested effect, natural log scale.
    log_fc: Vec<f64>,
    /// Standard error of the tested effect.
    se: Vec<f64>,
    /// Wald statistic.
    z: Vec<f64>,
    /// Two-sided p-value.
    p_value: Vec<f64>,
    /// Benjamini-Hochberg adjusted p-value.
    fdr: Vec<f64>,
}

/////////////
// Helpers //
/////////////

/// A scipy index as `usize`.
///
/// ### Params
///
/// * `i` - The index, `int32` or `int64`
///
/// ### Returns
///
/// The index, or [`EdgeErrors::InvalidArgument`] if it is negative.
#[inline]
fn to_index<I: Copy + TryInto<usize>>(i: I) -> Result<usize, EdgeErrors> {
    i.try_into().map_err(|_| {
        EdgeErrors::InvalidArgument("The sparse matrix holds a negative index.".to_string())
    })
}

/// Checks the CSR buffers agree and widens `indptr`.
///
/// ### Params
///
/// * `data` - Stored values
/// * `indices` - Column of each stored value
/// * `indptr` - Row offsets
/// * `n_vars` - Number of columns
///
/// ### Returns
///
/// The borrowed matrix, or [`EdgeErrors::InvalidArgument`] if the offsets are
/// not monotone or disagree with the buffer lengths.
fn cell_major<'a, T, I: Copy + TryInto<usize>>(
    data: &'a [T],
    indices: &'a [I],
    indptr: &[I],
    n_vars: usize,
) -> Result<CellMajor<'a, T, I>, EdgeErrors> {
    let indptr = indptr
        .iter()
        .map(|&p| to_index(p))
        .collect::<Result<Vec<_>, _>>()?;
    let malformed = indptr.is_empty()
        || indptr[0] != 0
        || indptr.windows(2).any(|w| w[0] > w[1])
        || indptr[indptr.len() - 1] != data.len()
        || indices.len() != data.len();
    if malformed {
        return Err(EdgeErrors::InvalidArgument(
            "The CSR buffers disagree: indptr is not monotone or does not match \
             data and indices."
                .to_string(),
        ));
    }
    Ok(CellMajor {
        data,
        indices,
        indptr,
        n_vars,
    })
}

/// Transposes one gene block out of the cell-major CSR.
///
/// Two passes over the selected rows, count then fill, so the block is built
/// in flat buffers without per-gene allocation. Walking `cells` in order keeps
/// the cell indices of every gene sorted. Stored zeros are dropped.
///
/// Sequential on purpose: one pass over the rows per block, against a NEBULA
/// fit of every gene in it. Not measured against a parallel transpose.
///
/// ### Params
///
/// * `x` - The counts
/// * `cells` - Rows to read, in output order
/// * `slot` - Gene to block-local row map, [`NO_SLOT`] outside the block
/// * `n_block` - Number of genes in the block
///
/// ### Returns
///
/// The block as CSR over `(n_block, cells.len())`, or
/// [`EdgeErrors::InvalidArgument`] on a negative, non-finite or non-integral
/// count, or a block too large for `u32` offsets.
fn gene_block<T: EdgeFloat, I: Copy + TryInto<usize>>(
    x: &CellMajor<'_, T, I>,
    cells: &[usize],
    slot: &[u32],
    n_block: usize,
) -> Result<CompressedSparse<f64>, EdgeErrors> {
    let mut counts = vec![0usize; n_block + 1];
    for (local, &cell) in cells.iter().enumerate() {
        let (start, end) = (x.indptr[cell], x.indptr[cell + 1]);
        for (&g, &v) in x.indices[start..end].iter().zip(&x.data[start..end]) {
            let gene = to_index(g)?;
            let s = *slot.get(gene).ok_or_else(|| {
                EdgeErrors::InvalidArgument(format!(
                    "The sparse matrix has column {gene} past its {} genes.",
                    x.n_vars
                ))
            })?;
            if s == NO_SLOT {
                continue;
            }
            let v = v.to_f64().unwrap_or(f64::NAN);
            if !(v.is_finite() && v >= 0.0 && v.fract() == 0.0) {
                return Err(EdgeErrors::InvalidArgument(format!(
                    "Count {v} at cell {}, gene {gene} is not a non-negative integer. \
                     NEBULA wants raw counts; pick them with `layer=`.",
                    cells[local]
                )));
            }
            if v > 0.0 {
                counts[s as usize + 1] += 1;
            }
        }
    }
    for i in 0..n_block {
        counts[i + 1] += counts[i];
    }
    let nnz = counts[n_block];
    if nnz > u32::MAX as usize || cells.len() > u32::MAX as usize {
        return Err(EdgeErrors::InvalidArgument(format!(
            "A gene block holds {nnz} counts, past the u32 offsets; lower \
             `gene_block_size`."
        )));
    }

    let mut cursor = counts[..n_block].to_vec();
    let mut data = vec![0.0f64; nnz];
    let mut indices = vec![0u32; nnz];
    for (local, &cell) in cells.iter().enumerate() {
        let (start, end) = (x.indptr[cell], x.indptr[cell + 1]);
        for (&g, &v) in x.indices[start..end].iter().zip(&x.data[start..end]) {
            // Validated in the first pass.
            let s = slot[to_index(g)?];
            let v = v.to_f64().unwrap_or(0.0);
            if s == NO_SLOT || v == 0.0 {
                continue;
            }
            let at = &mut cursor[s as usize];
            data[*at] = v;
            indices[*at] = local as u32;
            *at += 1;
        }
    }
    let indptr = counts.iter().map(|&c| c as u32).collect();
    CompressedSparse::from_parts(
        data,
        indices,
        indptr,
        SparseFormat::Csr,
        (n_block, cells.len()),
    )
}

////////////
// Driver //
////////////

/// Fits NEBULA block by block, then tests and shrinks over every kept gene.
///
/// ### Params
///
/// * `x` - The counts, cell-major
/// * `s` - The cells, genes, design and knobs
///
/// ### Returns
///
/// The kept genes' fits, or [`EdgeErrors`] on inconsistent inputs, a bad count,
/// or when no gene survives the expression filter.
fn sweep<T: EdgeFloat, I: Copy + TryInto<usize>>(
    x: &CellMajor<'_, T, I>,
    s: &Sweep<'_>,
) -> Result<SweepOut, EdgeErrors> {
    let n_obs = x.indptr.len() - 1;
    let n_cells = s.cells.len();
    if n_cells == 0 || s.genes.is_empty() {
        return Err(EdgeErrors::EmptyCounts {
            n_genes: s.genes.len(),
            n_samples: n_cells,
        });
    }
    if s.block == 0 {
        return Err(EdgeErrors::MustBePositive("gene_block_size".to_string()));
    }
    if s.subject_run.len() != n_cells {
        return Err(EdgeErrors::LengthMismatch {
            name: "subject_run",
            expected: n_cells,
            got: s.subject_run.len(),
        });
    }
    if let Some(&c) = s.cells.iter().find(|&&c| c >= n_obs) {
        return Err(EdgeErrors::InvalidArgument(format!(
            "Cell {c} is past the {n_obs} rows of the matrix."
        )));
    }
    let mut seen = vec![false; x.n_vars];
    for &g in s.genes {
        if g >= x.n_vars || std::mem::replace(&mut seen[g], true) {
            return Err(EdgeErrors::InvalidArgument(format!(
                "Gene {g} is out of range or listed twice."
            )));
        }
    }

    let offsets: Vec<f64> = match s.offset {
        Some(o) => o.to_vec(),
        None => s
            .cells
            .iter()
            .map(|&c| {
                x.data[x.indptr[c]..x.indptr[c + 1]]
                    .iter()
                    .map(|v| v.to_f64().unwrap_or(f64::NAN))
                    .sum()
            })
            .collect(),
    };

    let mut out = SweepOut {
        gene_idx: Vec::new(),
        subject_overdispersion: Vec::new(),
        cell_overdispersion: Vec::new(),
        cell_overdispersion_shrunk: None,
        convergence: Vec::new(),
        sigma_at_bound: Vec::new(),
        log_fc: Vec::new(),
        se: Vec::new(),
        z: Vec::new(),
        p_value: Vec::new(),
        fdr: Vec::new(),
    };
    let mut coefficients = Vec::new();
    let mut covariance = Vec::new();
    let mut mean_count = Vec::new();
    let mut slot = vec![NO_SLOT; x.n_vars];

    for block_genes in s.genes.chunks(s.block) {
        for (i, &g) in block_genes.iter().enumerate() {
            slot[g] = i as u32;
        }
        let counts = gene_block(x, s.cells, &slot, block_genes.len());
        for &g in block_genes {
            slot[g] = NO_SLOT;
        }
        let counts = counts?;

        let fit = match nebula_sparse(
            &counts,
            s.subject_run,
            s.design,
            s.n_coef,
            Some(&offsets),
            Some(s.nebula),
        ) {
            Ok(fit) => fit,
            // A block can lose every gene to the filter; only an empty sweep
            // is an error.
            Err(EdgeErrors::NoGenesAfterFiltering { .. }) => continue,
            Err(e) => return Err(e),
        };
        for &row in &fit.gene_index {
            out.gene_idx.push(block_genes[row]);
            let (_, values) = counts.outer(row);
            mean_count.push(values.iter().sum::<f64>() / n_cells as f64);
        }
        coefficients.extend_from_slice(&fit.coefficients);
        covariance.extend_from_slice(&fit.covariance);
        out.subject_overdispersion
            .extend_from_slice(&fit.subject_overdispersion);
        out.cell_overdispersion
            .extend_from_slice(&fit.cell_overdispersion);
        out.convergence.extend_from_slice(&fit.convergence);
        out.sigma_at_bound.extend_from_slice(&fit.sigma_at_bound);
    }

    let n_kept = out.gene_idx.len();
    if n_kept == 0 {
        return Err(EdgeErrors::InvalidArgument(format!(
            "No gene of {} passed NEBULA's expression filter (`cpc`, `mincp`, \
             `min_subjects`).",
            s.genes.len()
        )));
    }

    let test = glm_sc_test(&coefficients, &covariance, n_kept, s.n_coef, &s.tested)?;
    out.log_fc = test.log_fc;
    out.se = test.se;
    out.z = test.z;
    out.p_value = test.p_value;
    out.fdr = test.fdr;

    if s.shrink {
        let n_subjects = s.subject_run.windows(2).filter(|w| w[0] != w[1]).count() + 1;
        let covariate: Vec<f64> = mean_count
            .iter()
            .map(|m| (m + SHRINK_COVARIATE_OFFSET).log2())
            .collect();
        let shrunk = shrink_sc_dispersion(
            &out.cell_overdispersion,
            &out.convergence,
            Some(&covariate),
            sc_residual_df(n_cells, s.n_coef, n_subjects),
            None,
        )?;
        out.cell_overdispersion_shrunk = Some(shrunk.dispersion_shrunk);
    }
    Ok(out)
}

/// Borrows the CSR buffers as `T` data and `I` indices and runs the sweep.
///
/// ### Params
///
/// * `py` - GIL token, released for the sweep
/// * `data` - Stored values, a 1-D `T` array
/// * `indices` - Column of each value, a 1-D `I` array
/// * `indptr` - Row offsets, a 1-D `I` array
/// * `n_vars` - Number of columns
/// * `s` - Everything else
///
/// ### Returns
///
/// The sweep's result, or the extraction or numerical error.
fn sweep_typed<T, I>(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    indices: &Bound<'_, PyAny>,
    indptr: &Bound<'_, PyAny>,
    n_vars: usize,
    s: &Sweep<'_>,
) -> PyResult<SweepOut>
where
    T: EdgeFloat + Element,
    I: Copy + TryInto<usize> + Element + Sync,
{
    let data = data.extract::<PyReadonlyArray1<T>>()?;
    let indices = indices.extract::<PyReadonlyArray1<I>>()?;
    let indptr = indptr.extract::<PyReadonlyArray1<I>>()?;
    let x = cell_major(
        c_slice(&data)?,
        c_slice(&indices)?,
        c_slice(&indptr)?,
        n_vars,
    )
    .map_err(EdgeErr::from)?;
    Ok(py.detach(|| sweep(&x, s)).map_err(EdgeErr::from)?)
}

/// A 1-D `uint32` array as `usize` indices.
///
/// ### Params
///
/// * `a` - The array
///
/// ### Returns
///
/// The widened indices, or a `ValueError` unless C-contiguous.
fn index_vec(a: &PyReadonlyArray1<'_, u32>) -> PyResult<Vec<usize>> {
    Ok(c_slice(a)?.iter().map(|&i| i as usize).collect())
}

//////////////
// Bindings //
//////////////

/// NEBULA over a cell-major CSR, in gene blocks.
///
/// ### Params
///
/// * `data` - CSR values, `float32` or `float64`
/// * `indices` - CSR columns, `int32` or `int64`, the dtype of `indptr`
/// * `indptr` - CSR row offsets
/// * `n_vars` - Number of columns (genes)
/// * `cells` - `uint32` rows to fit, grouped by subject
/// * `subject_run` - `uint32` subject code per entry of `cells`
/// * `genes` - `uint32` columns to fit, no duplicates
/// * `design` - `(len(cells), n_coef)` `float64` design in `cells` order
/// * `offset` - Positive scaling per entry of `cells`; `None` uses library
///   sizes over every gene
/// * `coef` - 0-based coefficient to test; ignored when `contrast` is given
/// * `contrast` - One weight per design column
/// * `method` - `"ln"` or `"hl"`
/// * `cpc` - Drop a gene whose mean count per cell is at most this
/// * `mincp` - Drop a gene expressed in fewer cells than this
/// * `min_subjects` - Drop a gene fewer subjects express; `0` is off
/// * `shrink_dispersion` - Shrink the cell-level overdispersions
/// * `gene_block_size` - Genes transposed and fitted per block
///
/// ### Returns
///
/// Dict of 1-D arrays over the kept genes, `cell_overdispersion_shrunk`
/// `None` unless shrinkage was on.
#[pyfunction]
#[pyo3(signature = (
    data, indices, indptr, n_vars, cells, subject_run, genes, design,
    offset = None, coef = 0, contrast = None, method = "ln", cpc = 0.005,
    mincp = 5, min_subjects = 0, shrink_dispersion = true, gene_block_size = 1000
))]
// Many arguments on purpose: they are the Python keyword arguments.
#[allow(clippy::too_many_arguments)]
pub fn nebula_cell_major<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    indices: &Bound<'py, PyAny>,
    indptr: &Bound<'py, PyAny>,
    n_vars: usize,
    cells: PyReadonlyArray1<'py, u32>,
    subject_run: PyReadonlyArray1<'py, u32>,
    genes: PyReadonlyArray1<'py, u32>,
    design: PyReadonlyArray2<'py, f64>,
    offset: Option<PyReadonlyArray1<'py, f64>>,
    coef: usize,
    contrast: Option<Vec<f64>>,
    method: &str,
    cpc: f64,
    mincp: usize,
    min_subjects: usize,
    shrink_dispersion: bool,
    gene_block_size: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let method = match method {
        "ln" => NebulaMethod::Ln,
        "hl" => NebulaMethod::Hl,
        other => {
            return Err(PyValueError::new_err(format!(
                "method {other:?} is not 'ln' or 'hl'"
            )));
        }
    };
    let cells = index_vec(&cells)?;
    let subject_run = index_vec(&subject_run)?;
    let genes = index_vec(&genes)?;
    let (design, n_rows, n_coef) = rows(&design)?;
    if n_rows != cells.len() {
        return Err(PyValueError::new_err(format!(
            "the design has {n_rows} rows against {} cells",
            cells.len()
        )));
    }
    let offset = opt_slice(&offset)?;
    let s = Sweep {
        cells: &cells,
        subject_run: &subject_run,
        genes: &genes,
        design,
        n_coef,
        offset,
        nebula: NebulaParams {
            method,
            cpc,
            mincp,
            min_subjects,
            ..NebulaParams::default()
        },
        tested: contrast.map_or(ScTested::Coef(coef), ScTested::Contrast),
        shrink: shrink_dispersion,
        block: gene_block_size,
    };

    let f32_data = data.extract::<PyReadonlyArray1<f32>>().is_ok();
    let i32_index = indices.extract::<PyReadonlyArray1<i32>>().is_ok();
    let res = match (f32_data, i32_index) {
        (true, true) => sweep_typed::<f32, i32>(py, data, indices, indptr, n_vars, &s),
        (true, false) => sweep_typed::<f32, i64>(py, data, indices, indptr, n_vars, &s),
        (false, true) => sweep_typed::<f64, i32>(py, data, indices, indptr, n_vars, &s),
        (false, false) => sweep_typed::<f64, i64>(py, data, indices, indptr, n_vars, &s),
    }?;

    let out = PyDict::new(py);
    out.set_item("gene_idx", res.gene_idx.into_pyarray(py))?;
    out.set_item("log_fc", res.log_fc.into_pyarray(py))?;
    out.set_item("se", res.se.into_pyarray(py))?;
    out.set_item("z", res.z.into_pyarray(py))?;
    out.set_item("p_value", res.p_value.into_pyarray(py))?;
    out.set_item("fdr", res.fdr.into_pyarray(py))?;
    out.set_item(
        "subject_overdispersion",
        res.subject_overdispersion.into_pyarray(py),
    )?;
    out.set_item(
        "cell_overdispersion",
        res.cell_overdispersion.into_pyarray(py),
    )?;
    out.set_item(
        "cell_overdispersion_shrunk",
        res.cell_overdispersion_shrunk.map(|v| v.into_pyarray(py)),
    )?;
    out.set_item("convergence", res.convergence.into_pyarray(py))?;
    out.set_item("sigma_at_bound", res.sigma_at_bound.into_pyarray(py))?;
    Ok(out)
}
