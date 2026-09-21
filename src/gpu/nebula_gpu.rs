//! Host side of the GPU NEBULA path: staging, launch and read-back.
//!
//! The CPU fits one gene at a time; the device fits a batch of *requests* at
//! once, where a request is one penalised fit of one gene at one pair of
//! variance components. A gene can carry several requests in the same launch,
//! which is what lets a stage-two polish stencil go out as one batch.
//!
//! ### Layout
//!
//! The per-gene data is uploaded once and stays resident: the positive counts of
//! every gene concatenated with a `gene_ptr` index (the same CSR the crate
//! already holds them in), and the per-subject count totals gene-minor. The
//! shared design, offsets and subject boundaries stay in their natural order,
//! because every thread walks them identically and the traffic is served once
//! from cache.
//!
//! Everything per request is **request-minor**: entry `i` of request `q` lives
//! at `i * n_req + q`, so consecutive threads touch consecutive addresses.
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

////////////
// Consts //
////////////

/// Scalars the kernel writes per request: the log-likelihood, the previous
/// one, the log-determinant, the final improvement and the Laplace correction.
const OUT_SCALARS: usize = 5;

///////////
// Input //
///////////

/// One gene's sparse counts, as the batch builder wants them.
///
/// The same three slices [`crate::sc::pml::PmlData`] borrows, minus the shared
/// design and offsets, plus the variance components and starting point for the
/// one-request-per-gene convenience in [`opt_pml_batch`].
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

/// One penalised fit to run on the device.
#[derive(Clone, Debug)]
pub struct PmlRequest {
    /// Index of the gene, in upload order.
    pub gene: usize,
    /// nebula's `sigma[0]`, `log(1 + sigma^2)`.
    pub subject: f64,
    /// nebula's `sigma[1]`, the cell-level negative binomial size.
    pub cell: f64,
    /// Starting fixed effects, length `nb`.
    pub beta_init: Vec<f64>,
    /// Laplace order, one to three.
    pub ord: u32,
}

////////////
// Output //
////////////

/// What one request brings back.
#[derive(Clone, Debug)]
pub struct PmlReply {
    /// Penalised log-likelihood at the optimum.
    pub log_likelihood: f64,
    /// At the previous iterate.
    pub log_likelihood_prev: f64,
    /// Log-determinant of the random-effect block.
    pub log_det: f64,
    /// Higher-order Laplace correction, zero at order one.
    pub second_order: f64,
    /// Newton steps taken.
    pub iterations: u32,
    /// Backtracks used in the final step.
    pub backtracks: u32,
    /// Fitted fixed effects, length `nb`; empty unless read back in full.
    pub beta: Vec<f64>,
    /// Schur complement, row-major `nb * nb`; empty unless read back in full.
    pub information: Vec<f64>,
}

/// What one launch of one request per gene brings back, laid out gene-major.
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

/// Knobs for one launch, shared by every request in it.
#[derive(Clone, Copy, Debug)]
pub struct GpuSolveParams {
    /// nebula's absolute stopping tolerance.
    pub eps: f64,
    /// Resolution floor on the objective, relative to its magnitude.
    /// [`crate::gpu::pml_kernel::F32_NOISE_SCALE`] is the default.
    pub noise_scale: f64,
    /// Newton budget.
    pub max_iter: u32,
    /// Backtracking budget within one step.
    pub max_backtrack: u32,
    /// Whether to read back the coefficients and the information as well as
    /// the scalars. Stage two needs only the scalars, and at a polish round of
    /// nine requests per gene the rest is most of the read-back.
    pub full: bool,
}

//////////////////
// Resident set //
//////////////////

/// A set of genes whose data stays on the device between launches.
///
/// The counts and their cell indices are by far the largest upload, running to
/// hundreds of megabytes on a real batch, and NEBULA's stage two fits every
/// gene on the order of a hundred times with only the variance components
/// changed. Uploading once and solving many times is what makes that
/// affordable: measured at 16384 genes and 20000 cells, a one-shot solve spent
/// 1.5 of its 4.0 seconds on staging, and every solve after the first pays
/// none of it.
pub struct ResidentBatch<R: Runtime> {
    /// Shared design, row-major `n_cells * nb`.
    design: GpuTensor<R, f32>,
    /// Shared log offset per cell.
    log_offset: GpuTensor<R, f32>,
    /// Shared subject boundaries.
    subject_start: GpuTensor<R, u32>,
    /// Concatenated positive counts.
    counts: GpuTensor<R, f32>,
    /// Cell index of each count.
    cells: GpuTensor<R, u32>,
    /// Start of each gene's block in `counts`.
    gene_ptr: GpuTensor<R, u32>,
    /// Count total per subject, gene-minor.
    subject_total: GpuTensor<R, f32>,
    /// Per-subject scratch, sized for `capacity` requests.
    subject_scratch: GpuTensor<R, f32>,
    /// Cross-block scratch, sized for `capacity` requests.
    vwb_scratch: GpuTensor<R, f32>,
    /// Requests the scratch is currently sized for.
    capacity: usize,
    /// Genes resident.
    n_genes: usize,
    /// Subjects.
    k: usize,
    /// Design columns.
    nb: usize,
}

impl<R: Runtime> ResidentBatch<R> {
    /// Uploads everything that does not change between launches.
    ///
    /// ### Params
    ///
    /// * `design` - Shared design, row-major `n_cells * nb`
    /// * `log_offset` - Shared log offset per cell
    /// * `subject_start` - Shared subject boundaries, length `k + 1`
    /// * `genes` - One entry per gene; only the counts, their cell indices and
    ///   the subject totals are read
    /// * `client` - CubeCL compute client
    ///
    /// ### Returns
    ///
    /// The resident set, or [`EdgeErrors`] if the inputs disagree in shape or a
    /// device limit rejects the allocation.
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
        }

        // The narrowing to f32 is over `n_cells * nb` and `nnz`, both of which
        // run to tens of millions on a real batch. Left sequential it is a
        // serial third of the staging time.
        let design_f32: Vec<f32> = design.par_iter().map(|&v| v as f32).collect();
        let offset_f32: Vec<f32> = log_offset.par_iter().map(|&v| v as f32).collect();
        let start_u32: Vec<u32> = subject_start.iter().map(|&v| v as u32).collect();

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
            // Split once into the per-gene runs `gene_ptr` already describes,
            // then fill them in parallel; the runs are disjoint.
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

        let mut subject_total = vec![0.0f32; k * n_genes];
        for (g, gene) in genes.iter().enumerate() {
            for (s, &t) in gene.subject_total.iter().enumerate() {
                subject_total[s * n_genes + g] = t as f32;
            }
        }

        let err = |e: CubeclUtilsErrors| EdgeErrors::Gpu(e.to_string());
        let capacity = n_genes;
        Ok(Self {
            design: GpuTensor::from_slice(&design_f32, vec![n_cells * nb], client).map_err(err)?,
            log_offset: GpuTensor::from_slice(&offset_f32, vec![n_cells], client).map_err(err)?,
            subject_start: GpuTensor::from_slice(&start_u32, vec![k + 1], client).map_err(err)?,
            counts: GpuTensor::from_slice(&counts_f32, vec![nnz.max(1)], client).map_err(err)?,
            cells: GpuTensor::from_slice(&cells_u32, vec![nnz.max(1)], client).map_err(err)?,
            gene_ptr: GpuTensor::from_slice(&gene_ptr, vec![n_genes + 1], client).map_err(err)?,
            subject_total: GpuTensor::from_slice(&subject_total, vec![k * n_genes], client)
                .map_err(err)?,
            subject_scratch: GpuTensor::empty(
                vec![SUBJECT_SLOTS as usize * k * capacity],
                client,
            )
            .map_err(err)?,
            vwb_scratch: GpuTensor::empty(vec![k * nb * capacity], client).map_err(err)?,
            capacity,
            n_genes,
            k,
            nb,
        })
    }

    /// Design width of the resident set.
    ///
    /// ### Returns
    ///
    /// The number of design columns.
    pub fn n_coef(&self) -> usize {
        self.nb
    }

    /// Runs a batch of penalised fits against the resident genes.
    ///
    /// Only the per-request scalars and starting points are uploaded; the
    /// counts, the design and the offsets stay where they are.
    ///
    /// ### Params
    ///
    /// * `requests` - The fits to run; any gene may appear more than once
    /// * `params` - Knobs shared by every request
    /// * `client` - CubeCL compute client
    ///
    /// ### Returns
    ///
    /// One reply per request, in request order, or [`EdgeErrors`] if a request
    /// is out of range.
    pub fn solve(
        &mut self,
        requests: &[PmlRequest],
        params: &GpuSolveParams,
        client: &ComputeClient<R>,
    ) -> Result<Vec<PmlReply>, EdgeErrors> {
        let n_req = requests.len();
        if n_req == 0 {
            return Ok(Vec::new());
        }
        let (n_genes, k, nb) = (self.n_genes, self.k, self.nb);
        let err = |e: CubeclUtilsErrors| EdgeErrors::Gpu(e.to_string());

        let mut request_gene = vec![0u32; n_req];
        let mut request_ord = vec![0u32; n_req];
        let mut request_params = vec![0.0f32; 3 * n_req];
        let mut beta_init = vec![0.0f32; nb * n_req];
        for (q, req) in requests.iter().enumerate() {
            if req.gene >= n_genes {
                return Err(EdgeErrors::InvalidArgument(format!(
                    "Request {q} names gene {}, but only {n_genes} are resident.",
                    req.gene
                )));
            }
            if req.beta_init.len() != nb {
                return Err(EdgeErrors::LengthMismatch {
                    name: "beta_init",
                    expected: nb,
                    got: req.beta_init.len(),
                });
            }
            // The gamma prior, resolved in f64 exactly as `opt_pml` does.
            let exps = req.subject.exp();
            if !(exps.is_finite() && exps > 1.0) {
                return Err(EdgeErrors::InvalidArgument(format!(
                    "PmlVariance::subject must be finite and strictly positive; request {q} has {}.",
                    req.subject
                )));
            }
            if !(req.cell.is_finite() && req.cell > 0.0) {
                return Err(EdgeErrors::InvalidDispersion(req.cell));
            }
            request_gene[q] = req.gene as u32;
            request_ord[q] = req.ord;
            request_params[q] = (1.0 / (exps - 1.0)) as f32;
            request_params[n_req + q] = (1.0 / (exps.sqrt() * (exps - 1.0))) as f32;
            request_params[2 * n_req + q] = req.cell as f32;
            for (j, &b) in req.beta_init.iter().enumerate() {
                beta_init[j * n_req + q] = b as f32;
            }
        }

        if n_req > self.capacity {
            self.subject_scratch =
                GpuTensor::empty(vec![SUBJECT_SLOTS as usize * k * n_req], client).map_err(err)?;
            self.vwb_scratch = GpuTensor::empty(vec![k * nb * n_req], client).map_err(err)?;
            self.capacity = n_req;
        }

        let tensors = PmlGpuTensors::<R, f32> {
            design: self.design.clone(),
            log_offset: self.log_offset.clone(),
            subject_start: self.subject_start.clone(),
            counts: self.counts.clone(),
            cells: self.cells.clone(),
            gene_ptr: self.gene_ptr.clone(),
            subject_total: self.subject_total.clone(),
            request_gene: GpuTensor::from_slice(&request_gene, vec![n_req], client)
                .map_err(err)?,
            request_ord: GpuTensor::from_slice(&request_ord, vec![n_req], client).map_err(err)?,
            request_params: GpuTensor::from_slice(&request_params, vec![3 * n_req], client)
                .map_err(err)?,
            beta_init: GpuTensor::from_slice(&beta_init, vec![nb * n_req], client).map_err(err)?,
            tolerance: GpuTensor::from_slice(
                &[params.eps as f32, params.noise_scale as f32],
                vec![2],
                client,
            )
            .map_err(err)?,
            subject_scratch: self.subject_scratch.clone(),
            vwb_scratch: self.vwb_scratch.clone(),
            out_beta: GpuTensor::empty(vec![nb * n_req], client).map_err(err)?,
            out_information: GpuTensor::empty(vec![nb * nb * n_req], client).map_err(err)?,
            out_scalars: GpuTensor::empty(vec![OUT_SCALARS * n_req], client).map_err(err)?,
            out_counts: GpuTensor::empty(vec![2 * n_req], client).map_err(err)?,
        };

        launch_opt_pml::<R, f32>(
            &tensors,
            n_genes,
            n_req,
            k,
            nb,
            params.max_iter,
            params.max_backtrack,
            client,
        )?;

        let scalars = tensors.out_scalars.read(client).map_err(err)?;
        let counts = tensors.out_counts.read(client).map_err(err)?;
        let (beta_dev, info_dev) = if params.full {
            (
                tensors.out_beta.read(client).map_err(err)?,
                tensors.out_information.read(client).map_err(err)?,
            )
        } else {
            (Vec::new(), Vec::new())
        };

        Ok((0..n_req)
            .map(|q| PmlReply {
                log_likelihood: scalars[q] as f64,
                log_likelihood_prev: scalars[n_req + q] as f64,
                log_det: scalars[2 * n_req + q] as f64,
                second_order: scalars[4 * n_req + q] as f64,
                iterations: counts[q],
                backtracks: counts[n_req + q],
                beta: if params.full {
                    (0..nb).map(|j| beta_dev[j * n_req + q] as f64).collect()
                } else {
                    Vec::new()
                },
                information: if params.full {
                    (0..nb * nb)
                        .map(|i| info_dev[i * n_req + q] as f64)
                        .collect()
                } else {
                    Vec::new()
                },
            })
            .collect())
    }
}

//////////////////////////
// One request per gene //
//////////////////////////

/// Fits every gene once, at its own variance components, on the device.
///
/// Uploads, solves one request per gene at order one, and reads back in full.
/// The one-shot form of [`ResidentBatch`].
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
    solve_per_gene(
        &mut resident,
        genes,
        &GpuSolveParams {
            eps,
            noise_scale,
            max_iter,
            max_backtrack,
            full: true,
        },
        client,
    )
}

/// One order-one request per gene against a resident set, laid out gene-major.
///
/// ### Params
///
/// * `resident` - The resident set, in the same gene order as `genes`
/// * `genes` - One entry per gene; `beta_init`, `sigma` and `gamma` are read
/// * `params` - Knobs shared by every request
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// The fits, gene-major.
pub fn solve_per_gene<R: Runtime>(
    resident: &mut ResidentBatch<R>,
    genes: &[GpuGene<'_>],
    params: &GpuSolveParams,
    client: &ComputeClient<R>,
) -> Result<GpuPmlBatch, EdgeErrors> {
    let nb = resident.n_coef();
    let requests: Vec<PmlRequest> = genes
        .iter()
        .enumerate()
        .map(|(g, gene)| PmlRequest {
            gene: g,
            subject: gene.sigma,
            cell: gene.gamma,
            beta_init: gene.beta_init.to_vec(),
            ord: 1,
        })
        .collect();
    let replies = resident.solve(&requests, params, client)?;
    Ok(GpuPmlBatch {
        beta: replies.iter().flat_map(|r| r.beta.iter().copied()).collect(),
        information: replies
            .iter()
            .flat_map(|r| r.information.iter().copied())
            .collect(),
        log_likelihood: replies.iter().map(|r| r.log_likelihood).collect(),
        log_det: replies.iter().map(|r| r.log_det).collect(),
        iterations: replies.iter().map(|r| r.iterations).collect(),
        backtracks: replies.iter().map(|r| r.backtracks).collect(),
        n_coef: nb,
    })
}
