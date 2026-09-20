//! Host side of the GPU NEBULA path: staging, launch and read-back.
//!
//! The CPU fits one gene at a time; the device fits a whole batch at once, so
//! everything here is about turning the per-gene views
//! [`crate::sc::pml::PmlData`] borrows into one set of buffers laid out the way
//! the kernel indexes them.
//!
//! ### Layout
//!
//! Everything that varies per gene is **gene-minor**: entry `i` of gene `g`
//! lives at `i * n_genes + g`. Consecutive threads are consecutive genes, so a
//! gene-minor layout makes every read and write of a per-gene quantity
//! coalesced across the plane. The shared design, offsets and subject
//! boundaries stay in their natural row-major order, because every thread walks
//! them identically and the traffic is served once from cache.
//!
//! The positive counts of all genes are concatenated into one pair of buffers
//! with a `gene_ptr` index, which is the same CSR the crate already holds them
//! in.
//!
//! ### Precision
//!
//! The device is `f32`. Everything derived on the host, in particular the gamma
//! prior's `alpha` and `lambda`, is computed in `f64` and cast once, because
//! `alpha = 1 / (e^s - 1)` cancels badly for the small `s` most genes land on.

use cubecl::prelude::*;
use cubecl_utils_rs::prelude::*;
use rayon::prelude::*;

use crate::errors::EdgeErrors;
use crate::gpu::pml_kernel::{PmlGpuTensors, SUBJECT_SLOTS, launch_opt_pml};

///////////
// Input //
///////////

/// One gene's sparse counts, as the batch builder wants them.
///
/// The same three slices [`crate::sc::pml::PmlData`] borrows, minus the shared
/// design and offsets.
#[derive(Clone, Copy, Debug)]
pub struct GpuGene<'a> {
    /// The positive counts, in increasing cell order.
    pub counts: &'a [f64],
    /// Cell index of each entry of `counts`.
    pub cell_index: &'a [usize],
    /// Count total per subject, length `k`.
    pub subject_total: &'a [f64],
    /// Starting fixed effects, length `nb`.
    pub beta_init: &'a [f64],
    /// nebula's `sigma[0]`, `log(1 + sigma^2)`.
    pub sigma: f64,
    /// nebula's `sigma[1]`, the cell-level negative binomial size.
    pub gamma: f64,
}

////////////
// Output //
////////////

/// What one launch brings back, for every gene in the batch.
///
/// Laid out gene-major on the host, which is the opposite of the device
/// layout, because callers read one gene at a time.
#[derive(Clone, Debug)]
pub struct GpuPmlBatch {
    /// Fitted fixed effects, row-major `n_genes * nb`.
    pub beta: Vec<f64>,
    /// Schur complement of the observed information, `n_genes * nb * nb`.
    pub information: Vec<f64>,
    /// Penalised log-likelihood at the optimum, one per gene.
    pub log_likelihood: Vec<f64>,
    /// Log-determinant of the random-effect block, one per gene.
    pub log_det: Vec<f64>,
    /// Newton steps taken, one per gene.
    pub iterations: Vec<u32>,
    /// Backtracks used in the final step, one per gene.
    pub backtracks: Vec<u32>,
    /// Number of design columns, the stride of `beta`.
    pub n_coef: usize,
}

/////////////
// Staging //
/////////////

/// Fits a batch of genes by penalised maximum likelihood on the device.
///
/// One thread per gene. The design, offsets and subject boundaries are shared
/// across the batch and uploaded once; see the module doc for the layouts.
///
/// ### Params
///
/// * `design` - Shared design, row-major `n_cells * nb`
/// * `log_offset` - Shared log offset per cell, length `n_cells`
/// * `subject_start` - Shared subject boundaries, length `k + 1`
/// * `genes` - One entry per gene in the batch
/// * `eps` - nebula's absolute stopping tolerance
/// * `noise_scale` - Resolution floor on the objective, relative to its
///   magnitude. [`crate::gpu::pml_kernel::F32_NOISE_SCALE`] is the default; it
///   is a parameter because the right value scales with the cell count
/// * `max_iter` - Newton budget
/// * `max_backtrack` - Backtracking budget within one step
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// The fits, or [`EdgeErrors`] if the inputs disagree in shape or a device
/// limit rejects the work.
#[allow(clippy::too_many_arguments)]
pub fn opt_pml_batch<R: Runtime>(
    design: &[f64],
    log_offset: &[f64],
    subject_start: &[usize],
    genes: &[GpuGene<'_>],
    eps: f64,
    noise_scale: f64,
    max_iter: u32,
    max_backtrack: u32,
    client: &ComputeClient<R>,
) -> Result<GpuPmlBatch, EdgeErrors> {
    let mut resident = ResidentBatch::upload(design, log_offset, subject_start, genes, client)?;
    resident.solve(genes, eps, noise_scale, max_iter, max_backtrack, client)
}

/// A batch whose gene-independent buffers stay on the device between solves.
///
/// The counts and their cell indices are by far the largest upload, running to
/// hundreds of megabytes on a real batch, and NEBULA's stage two calls the
/// solver on the order of a hundred times per gene with only the two variance
/// components changed. Uploading once and solving many times is what makes that
/// affordable: measured at 16384 genes and 20000 cells, a single solve spends
/// 1.5 of its 4.0 seconds on staging, and every solve after the first pays none
/// of it.
pub struct ResidentBatch<R: Runtime> {
    /// The device buffers, including the ones re-written per solve.
    tensors: PmlGpuTensors<R, f32>,
    /// Genes in the batch.
    n_genes: usize,
    /// Subjects.
    k: usize,
    /// Design columns.
    nb: usize,
}

impl<R: Runtime> ResidentBatch<R> {
    /// Uploads everything that does not change between solves.
    ///
    /// ### Params
    ///
    /// * `design` - Shared design, row-major `n_cells * nb`
    /// * `log_offset` - Shared log offset per cell
    /// * `subject_start` - Shared subject boundaries, length `k + 1`
    /// * `genes` - One entry per gene; only the counts, their cell indices and
    ///   the subject totals are read here
    /// * `client` - CubeCL compute client
    ///
    /// ### Returns
    ///
    /// The resident batch, or [`EdgeErrors`] if the inputs disagree in shape or
    /// a device limit rejects the allocation.
    pub fn upload(
        design: &[f64],
        log_offset: &[f64],
        subject_start: &[usize],
        genes: &[GpuGene<'_>],
        client: &ComputeClient<R>,
    ) -> Result<Self, EdgeErrors> {
    let n_genes = genes.len();
    if n_genes == 0 {
        return Err(EdgeErrors::MustBePositive("n_genes".to_string()));
    }
    let n_cells = log_offset.len();
    if n_cells == 0 {
        return Err(EdgeErrors::MustBePositive("n_cells".to_string()));
    }
    let k = subject_start.len().saturating_sub(1);
    if k == 0 {
        return Err(EdgeErrors::MustBePositive("n_subjects".to_string()));
    }
    if !design.len().is_multiple_of(n_cells) {
        return Err(EdgeErrors::LengthMismatch {
            name: "design",
            expected: n_cells,
            got: design.len(),
        });
    }
    let nb = design.len() / n_cells;
    for gene in genes {
        if gene.counts.len() != gene.cell_index.len() {
            return Err(EdgeErrors::LengthMismatch {
                name: "counts",
                expected: gene.cell_index.len(),
                got: gene.counts.len(),
            });
        }
        if gene.subject_total.len() != k {
            return Err(EdgeErrors::LengthMismatch {
                name: "subject_total",
                expected: k,
                got: gene.subject_total.len(),
            });
        }
        if gene.beta_init.len() != nb {
            return Err(EdgeErrors::LengthMismatch {
                name: "beta_init",
                expected: nb,
                got: gene.beta_init.len(),
            });
        }
    }

    // -- Shared inputs --
    // The narrowing to f32 is over `n_cells * nb` and `nnz`, both of which run
    // to tens of millions on a real batch. Left sequential it is a serial third
    // of the time the caller spends in here, next to a kernel that is not.
    let design_f32: Vec<f32> = design.par_iter().map(|&v| v as f32).collect();
    let offset_f32: Vec<f32> = log_offset.par_iter().map(|&v| v as f32).collect();
    let start_u32: Vec<u32> = subject_start.iter().map(|&v| v as u32).collect();

    // -- Concatenated sparse counts --
    let mut gene_ptr = Vec::with_capacity(n_genes + 1);
    gene_ptr.push(0u32);
    let mut running = 0usize;
    for gene in genes {
        running += gene.counts.len();
        gene_ptr.push(running as u32);
    }
    let nnz = running;

    let mut counts_f32 = vec![0.0f32; nnz];
    let mut cells_u32 = vec![0u32; nnz];
    {
        // Split once into the per-gene runs `gene_ptr` already describes, then
        // fill them in parallel; the runs are disjoint so nothing is shared.
        let mut count_runs: Vec<&mut [f32]> = Vec::with_capacity(n_genes);
        let mut cell_runs: Vec<&mut [u32]> = Vec::with_capacity(n_genes);
        let mut count_rest = counts_f32.as_mut_slice();
        let mut cell_rest = cells_u32.as_mut_slice();
        for gene in genes {
            let (a, b) = count_rest.split_at_mut(gene.counts.len());
            count_runs.push(a);
            count_rest = b;
            let (c, d) = cell_rest.split_at_mut(gene.cell_index.len());
            cell_runs.push(c);
            cell_rest = d;
        }
        count_runs
            .into_par_iter()
            .zip(cell_runs)
            .zip(genes)
            .for_each(|((count_run, cell_run), gene)| {
                for (out, &v) in count_run.iter_mut().zip(gene.counts) {
                    *out = v as f32;
                }
                for (out, &c) in cell_run.iter_mut().zip(gene.cell_index) {
                    *out = c as u32;
                }
            });
    }

    // -- Gene-minor per-gene inputs --
    let mut subject_total = vec![0.0f32; k * n_genes];
    for (g, gene) in genes.iter().enumerate() {
        for (s, &t) in gene.subject_total.iter().enumerate() {
            subject_total[s * n_genes + g] = t as f32;
        }
    }
    let err = |e: CubeclUtilsErrors| EdgeErrors::Gpu(e.to_string());
    let tensors = PmlGpuTensors::<R, f32> {
        design: GpuTensor::from_slice(&design_f32, vec![n_cells * nb], client).map_err(err)?,
        log_offset: GpuTensor::from_slice(&offset_f32, vec![n_cells], client).map_err(err)?,
        subject_start: GpuTensor::from_slice(&start_u32, vec![k + 1], client).map_err(err)?,
        counts: GpuTensor::from_slice(&counts_f32, vec![nnz.max(1)], client).map_err(err)?,
        cells: GpuTensor::from_slice(&cells_u32, vec![nnz.max(1)], client).map_err(err)?,
        gene_ptr: GpuTensor::from_slice(&gene_ptr, vec![n_genes + 1], client).map_err(err)?,
        subject_total: GpuTensor::from_slice(&subject_total, vec![k * n_genes], client)
            .map_err(err)?,
        // Placeholders: `solve` writes all three, and they are small enough
        // that replacing them per solve costs nothing worth measuring.
        gene_params: GpuTensor::from_slice(&vec![0.0f32; 3 * n_genes], vec![3 * n_genes], client)
            .map_err(err)?,
        beta_init: GpuTensor::from_slice(&vec![0.0f32; nb * n_genes], vec![nb * n_genes], client)
            .map_err(err)?,
        tolerance: GpuTensor::from_slice(&[0.0f32, 0.0f32], vec![2], client).map_err(err)?,
        subject_scratch: GpuTensor::from_slice(
            &vec![0.0f32; SUBJECT_SLOTS as usize * k * n_genes],
            vec![SUBJECT_SLOTS as usize * k * n_genes],
            client,
        )
        .map_err(err)?,
        vwb_scratch: GpuTensor::from_slice(
            &vec![0.0f32; k * nb * n_genes],
            vec![k * nb * n_genes],
            client,
        )
        .map_err(err)?,
        out_beta: GpuTensor::from_slice(&vec![0.0f32; nb * n_genes], vec![nb * n_genes], client)
            .map_err(err)?,
        out_information: GpuTensor::from_slice(
            &vec![0.0f32; nb * nb * n_genes],
            vec![nb * nb * n_genes],
            client,
        )
        .map_err(err)?,
        out_scalars: GpuTensor::from_slice(&vec![0.0f32; 4 * n_genes], vec![4 * n_genes], client)
            .map_err(err)?,
        out_counts: GpuTensor::from_slice(&vec![0u32; 2 * n_genes], vec![2 * n_genes], client)
            .map_err(err)?,
    };

        Ok(Self {
            tensors,
            n_genes,
            k,
            nb,
        })
    }

    /// Fits the resident batch at the given variance components.
    ///
    /// Only the per-gene scalars are re-uploaded; the counts, the design and
    /// the offsets stay where they are.
    ///
    /// ### Params
    ///
    /// * `genes` - One entry per gene; only `beta_init`, `sigma` and `gamma`
    ///   are read here, and they must be in the same gene order as the upload
    /// * `eps` - nebula's absolute stopping tolerance
    /// * `noise_scale` - Resolution floor on the objective, relative to its
    ///   magnitude
    /// * `max_iter` - Newton budget
    /// * `max_backtrack` - Backtracking budget within one step
    /// * `client` - CubeCL compute client
    ///
    /// ### Returns
    ///
    /// The fits, or [`EdgeErrors`] if a variance component is out of range.
    pub fn solve(
        &mut self,
        genes: &[GpuGene<'_>],
        eps: f64,
        noise_scale: f64,
        max_iter: u32,
        max_backtrack: u32,
        client: &ComputeClient<R>,
    ) -> Result<GpuPmlBatch, EdgeErrors> {
        let (n_genes, k, nb) = (self.n_genes, self.k, self.nb);
        if genes.len() != n_genes {
            return Err(EdgeErrors::LengthMismatch {
                name: "genes",
                expected: n_genes,
                got: genes.len(),
            });
        }
        let err = |e: CubeclUtilsErrors| EdgeErrors::Gpu(e.to_string());

        let mut beta_init = vec![0.0f32; nb * n_genes];
        for (g, gene) in genes.iter().enumerate() {
            if gene.beta_init.len() != nb {
                return Err(EdgeErrors::LengthMismatch {
                    name: "beta_init",
                    expected: nb,
                    got: gene.beta_init.len(),
                });
            }
            for (j, &b) in gene.beta_init.iter().enumerate() {
                beta_init[j * n_genes + g] = b as f32;
            }
        }

        // The gamma prior, resolved in f64 exactly as `opt_pml` does. `e^s - 1`
        // cancels for the small `s` most genes sit on, so this cannot be left
        // to the device.
        let mut gene_params = vec![0.0f32; 3 * n_genes];
        for (g, gene) in genes.iter().enumerate() {
            let exps = gene.sigma.exp();
            if !(exps.is_finite() && exps > 1.0) {
                return Err(EdgeErrors::InvalidArgument(format!(
                    "PmlVariance::subject must be finite and strictly positive; gene {g} has {}.",
                    gene.sigma
                )));
            }
            if !(gene.gamma.is_finite() && gene.gamma > 0.0) {
                return Err(EdgeErrors::InvalidDispersion(gene.gamma));
            }
            gene_params[g] = (1.0 / (exps - 1.0)) as f32;
            gene_params[n_genes + g] = (1.0 / (exps.sqrt() * (exps - 1.0))) as f32;
            gene_params[2 * n_genes + g] = gene.gamma as f32;
        }

        self.tensors.beta_init =
            GpuTensor::from_slice(&beta_init, vec![nb * n_genes], client).map_err(err)?;
        self.tensors.gene_params =
            GpuTensor::from_slice(&gene_params, vec![3 * n_genes], client).map_err(err)?;
        self.tensors.tolerance =
            GpuTensor::from_slice(&[eps as f32, noise_scale as f32], vec![2], client)
                .map_err(err)?;

        launch_opt_pml::<R, f32>(
            &mut self.tensors,
            n_genes,
            k,
            nb,
            max_iter,
            max_backtrack,
            client,
        )?;

        let beta_dev = self.tensors.out_beta.clone().read(client).map_err(err)?;
        let info_dev = self
            .tensors
            .out_information
            .clone()
            .read(client)
            .map_err(err)?;
        let scalars = self.tensors.out_scalars.clone().read(client).map_err(err)?;
        let counts_dev = self.tensors.out_counts.clone().read(client).map_err(err)?;

        let mut beta = vec![0.0f64; n_genes * nb];
        for g in 0..n_genes {
            for j in 0..nb {
                beta[g * nb + j] = beta_dev[j * n_genes + g] as f64;
            }
        }
        let mut information = vec![0.0f64; n_genes * nb * nb];
        for g in 0..n_genes {
            for i in 0..nb * nb {
                information[g * nb * nb + i] = info_dev[i * n_genes + g] as f64;
            }
        }

        Ok(GpuPmlBatch {
            beta,
            information,
            log_likelihood: (0..n_genes).map(|g| scalars[g] as f64).collect(),
            log_det: (0..n_genes)
                .map(|g| scalars[2 * n_genes + g] as f64)
                .collect(),
            iterations: (0..n_genes).map(|g| counts_dev[g]).collect(),
            backtracks: (0..n_genes).map(|g| counts_dev[n_genes + g]).collect(),
            n_coef: nb,
        })
    }
}
