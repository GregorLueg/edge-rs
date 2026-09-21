//! NEBULA with its stage two on the device.
//!
//! Stage two, the search over the two variance components, is where NEBULA
//! spends its time: every objective evaluation is a full penalised fit over
//! every cell, and a gene takes on the order of a hundred of them. This module
//! runs that search for every gene at once.
//!
//! ### Split
//!
//! * **Stage one** (L-BFGS-B on the marginal likelihood) stays on the CPU. It is
//!   a few per cent of the time and has an exact gradient.
//! * **Stage two** runs as one [`crate::sc::nebula::StageTwoSearch`] per gene,
//!   the same state machine the CPU path drives. Each round, every live search
//!   asks for its next points, all of them go out as one device launch of
//!   [`ResidentBatch::solve`], and the replies are assembled into the profile
//!   objective on the host in `f64`. Nelder-Mead, the polish least squares and
//!   the objective assembly never leave the host; only the penalised fits do.
//! * **Stage three**, the final fit whose information gives the standard
//!   errors, stays on the CPU in `f64`.
//!
//! So the only `f32` influence on the output is through the variance
//! components stage two lands on. The coefficients and standard errors are
//! then the `f64` answer at those components.
//!
//! ### Why lockstep rounds
//!
//! Nelder-Mead is sequential within a gene, so the round count is set by the
//! slowest gene, and on the R fixtures that is about three times the median.
//! The rounds thin out as searches finish, since only live searches send
//! requests, so the total device work is the total number of evaluations, not
//! the slowest gene's count times the gene count. A polish stencil goes out as
//! all of its points in one round.

use cubecl::prelude::*;
use rayon::prelude::*;

use crate::errors::EdgeErrors;
use crate::gpu::nebula_gpu::{GpuGene, GpuSolveParams, PmlReply, PmlRequest, ResidentBatch};
use crate::gpu::pml_kernel::F32_NOISE_SCALE;
use crate::numeric::gamma::ln_gamma;
use crate::prelude::*;
use crate::sc::nebula::{
    GeneOutcome, GenePlan, InnerFit, NebulaFit, NebulaParams, Shared, StageTwoSearch,
    finish_gene, gene_pml, nebula_sparse_with, plan_gene, variance_bounds, variance_objective,
};
use crate::sc::pml::{PmlParams, PmlVariance, opt_pml_from};
use crate::sc::ptmg::{GeneCounts, positive_indices};

/////////////////
// Entry point //
/////////////////

/// Fits NEBULA's negative binomial gamma mixed model with stage two on the
/// device.
///
/// Takes and returns exactly what [`crate::sc::nebula::nebula_sparse`] does.
/// The answers are not bit-identical to the CPU path: the penalised fits in
/// stage two run in `f32`, so the variance components land within a measured
/// tolerance of the CPU's rather than on them, and the coefficients and
/// standard errors follow. See `tests/e2e_nebula_gpu.rs` for the numbers.
///
/// ### Params
///
/// * `counts` - Raw counts, CSR over `(n_genes, n_cells)`
/// * `subject_id` - Subject of each cell, with each subject's cells contiguous
/// * `design` - Predictors, row-major `n_cells * n_coef`, including an intercept
/// * `n_coef` - Number of design columns
/// * `offset` - Strictly positive scaling factor per cell, or `None` for ones
/// * `params` - Tuning knobs, or [`NebulaParams::default`]
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// The per-gene fits, or [`EdgeErrors`] as for the CPU path, plus
/// [`EdgeErrors::InvalidArgument`] if `params.reml` is set, which the device
/// fit does not implement, and [`EdgeErrors::Gpu`] if the device rejects the
/// work.
///
/// ### References
///
/// He et al., Communications Biology 4, 629, 2021
pub fn nebula_sparse_gpu<T: EdgeFloat, R: Runtime>(
    counts: &CompressedSparse<f64>,
    subject_id: &[usize],
    design: &[T],
    n_coef: usize,
    offset: Option<&[T]>,
    params: Option<NebulaParams>,
    client: &ComputeClient<R>,
) -> Result<NebulaFit, EdgeErrors> {
    if params.is_some_and(|p| p.reml) {
        return Err(EdgeErrors::InvalidArgument(
            "The GPU NEBULA path does not implement `reml`; use the CPU path.".to_string(),
        ));
    }
    nebula_sparse_with(
        counts,
        subject_id,
        design,
        n_coef,
        offset,
        params,
        |shared, sparse, totals, kept| fit_all(shared, sparse, totals, kept, client),
    )
}

/// Fits every kept gene: stage one on the CPU, stage two on the device, stage
/// three on the CPU.
///
/// ### Params
///
/// * `shared` - The inputs common to every gene
/// * `sparse` - The counts
/// * `totals` - Count total per subject, gene-major
/// * `kept` - Genes that passed the expression filter
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// One outcome per kept gene, in order.
fn fit_all<R: Runtime>(
    shared: &Shared<'_>,
    sparse: &CompressedSparse<f64>,
    totals: &[f64],
    kept: &[usize],
    client: &ComputeClient<R>,
) -> Result<Vec<GeneOutcome>, EdgeErrors> {
    let k = shared.n_subjects;
    let genes: Vec<(GeneCounts, GenePlan)> = kept
        .par_iter()
        .map(|&g| {
            let counts = positive_indices(sparse, g)?;
            let plan = plan_gene(shared, &counts, &totals[g * k..(g + 1) * k])?;
            Ok((counts, plan))
        })
        .collect::<Result<_, EdgeErrors>>()?;

    let refits = search_all(shared, &genes, totals, kept, client)?;

    genes
        .into_par_iter()
        .zip(refits)
        .zip(kept.par_iter())
        .map(|(((counts, plan), refit), &g)| {
            finish_gene(shared, &counts, &totals[g * k..(g + 1) * k], plan, refit)
        })
        .collect()
}

///////////////
// Stage two //
///////////////

/// One gene's search, with everything a round needs to evaluate its points.
struct Live<'a> {
    /// Position of the gene among the kept genes.
    index: usize,
    /// Position of the gene in the resident set.
    resident: usize,
    /// The search.
    search: StageTwoSearch,
    /// Cell-level component held fixed, for the one-dimensional search.
    fixed_cell: Option<f64>,
    /// The gene in the layout the objective borrows.
    pml: crate::sc::pml::PmlData<'a>,
    /// Distinct positive counts other than one and two, with multiplicities.
    tail: Vec<(f64, f64)>,
}

/// Runs stage two for every gene that needs it, all searches in lockstep.
///
/// ### Params
///
/// * `shared` - The inputs common to every gene
/// * `genes` - Every kept gene's counts and plan
/// * `totals` - Count total per subject, gene-major
/// * `kept` - Genes that passed the expression filter
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// One entry per kept gene: `None` where no refit was called for, else the
/// search's result.
fn search_all<R: Runtime>(
    shared: &Shared<'_>,
    genes: &[(GeneCounts, GenePlan)],
    totals: &[f64],
    kept: &[usize],
    client: &ComputeClient<R>,
) -> Result<Vec<Option<Option<Vec<f64>>>>, EdgeErrors> {
    let k = shared.n_subjects;
    let mut refits: Vec<Option<Option<Vec<f64>>>> = vec![None; genes.len()];

    let mut live: Vec<Live<'_>> = Vec::new();
    for (index, (counts, plan)) in genes.iter().enumerate() {
        let Some((start, fixed_cell)) = plan.search() else {
            continue;
        };
        let (lower, upper) = variance_bounds(&shared.params, fixed_cell);
        let g = kept[index];
        live.push(Live {
            index,
            resident: live.len(),
            search: StageTwoSearch::new(&start, &lower, &upper, plan.ord),
            fixed_cell,
            pml: gene_pml(shared, counts, &totals[g * k..(g + 1) * k]),
            tail: count_histogram(&counts.counts),
        });
    }
    if live.is_empty() {
        return Ok(refits);
    }

    let resident_genes: Vec<GpuGene<'_>> = live
        .iter()
        .map(|l| GpuGene {
            counts: l.pml.counts,
            cell_index: l.pml.cell_index,
            subject_total: l.pml.subject_total,
            beta_init: &[],
            sigma: 0.0,
            gamma: 0.0,
        })
        .collect();
    let mut resident = ResidentBatch::upload(
        shared.design,
        shared.log_offset,
        shared.fid,
        &resident_genes,
        client,
    )?;

    let inner = PmlParams::default();
    let solve_params = GpuSolveParams {
        eps: shared.params.eps,
        noise_scale: f64::from(F32_NOISE_SCALE),
        max_iter: inner.max_iter as u32,
        max_backtrack: inner.max_backtrack as u32,
        full: true,
    };

    loop {
        // -- Gather: every live search's points, as device requests. Points
        //    outside the domain are infinite without a fit. --
        let mut requests = Vec::new();
        let mut routes: Vec<(usize, usize, f64, f64)> = Vec::new();
        let mut values: Vec<Vec<f64>> = vec![Vec::new(); live.len()];
        let mut any = false;
        for (i, l) in live.iter().enumerate() {
            let Some(points) = l.search.ask() else {
                continue;
            };
            any = true;
            values[i] = vec![f64::INFINITY; points.len()];
            let (counts, plan) = &genes[l.index];
            let objective = variance_objective(
                shared,
                &l.pml,
                &plan.beta_start,
                counts,
                l.search.order(),
                l.fixed_cell,
            );
            for (p, x) in points.iter().enumerate() {
                if let Some((subject, cell, beta_init)) = objective.request(x) {
                    requests.push(PmlRequest {
                        gene: l.resident,
                        subject,
                        cell,
                        beta_init,
                    });
                    routes.push((i, p, subject, cell));
                }
            }
        }
        if !any {
            break;
        }

        let replies = resident.solve(&requests, &solve_params, client)?;

        // -- Scatter: assemble the profile objective in f64, in parallel, then
        //    hand each search its values. --
        let assembled: Vec<f64> = routes
            .par_iter()
            .zip(replies.par_iter())
            .map(|(&(i, _, subject, cell), reply)| {
                let l = &live[i];
                let (counts, plan) = &genes[l.index];
                let objective = variance_objective(
                    shared,
                    &l.pml,
                    &plan.beta_start,
                    counts,
                    l.search.order(),
                    l.fixed_cell,
                );
                let Some(fit) = finish_at_argmax(&l.pml, reply, subject, cell, objective.params)
                else {
                    return f64::INFINITY;
                };
                objective.assemble(subject, cell, &fit, histogram_tail(&l.tail, cell))
            })
            .collect();
        for (&(i, p, _, _), v) in routes.iter().zip(assembled) {
            values[i][p] = v;
        }
        for (l, v) in live.iter_mut().zip(values) {
            if l.search.ask().is_some() {
                l.search.tell(&v);
            }
        }
    }

    for l in live {
        refits[l.index] = Some(l.search.result());
    }
    Ok(refits)
}

/// The four scalars the objective reads: the CPU's fit, started from the
/// device's optimum.
///
/// The device finds the optimum; its own `f32` value is not used. The profile
/// likelihood is compared across evaluations to well below what `f32` over every
/// cell can resolve, and the error is not only noise but biased: it grows as the
/// subject-level variance falls, which on the R fixtures pulled half the genes'
/// searches onto the lower bound. Evaluating the `f64` value at the device's
/// point fixed the value, but the log-determinant is not stationary at the
/// optimum and still carried the device's location error at first order. So
/// the `f64` fit is finished from the device's point instead: one Newton step
/// from there, and what comes back is exactly what the CPU path computes.
///
/// ### Params
///
/// * `pml` - The gene
/// * `reply` - The device fit, read back in full
/// * `subject` - nebula's `sigma[0]`
/// * `cell` - The cell-level negative binomial size
/// * `params` - The inner fit's knobs, including its Laplace order
///
/// ### Returns
///
/// The inner fit, or `None` if the device's point is not finite or the fit
/// fails.
fn finish_at_argmax(
    pml: &crate::sc::pml::PmlData<'_>,
    reply: &PmlReply,
    subject: f64,
    cell: f64,
    params: PmlParams,
) -> Option<InnerFit> {
    if !reply.beta.iter().chain(&reply.log_w).all(|v| v.is_finite()) {
        return None;
    }
    let fit = opt_pml_from(
        pml,
        &reply.beta,
        &reply.log_w,
        &PmlVariance { subject, cell },
        Some(params),
    )
    .ok()?;
    Some(InnerFit {
        log_likelihood: fit.log_likelihood,
        log_likelihood_prev: fit.log_likelihood_prev,
        log_det: fit.log_det,
        second_order: fit.second_order,
    })
}

/// Distinct positive counts other than one and two, with their multiplicities.
///
/// The profile objective sums `lgamma(y + cell)` over these on every
/// evaluation. Single-cell counts take few distinct values, so summing over the
/// distinct ones cuts that from one `lgamma` per positive count to one per
/// distinct count, which matters once the fits themselves are on the device.
///
/// ### Params
///
/// * `counts` - The gene's positive counts
///
/// ### Returns
///
/// `(value, multiplicity)` pairs, sorted by value.
fn count_histogram(counts: &[f64]) -> Vec<(f64, f64)> {
    let mut values: Vec<f64> = counts
        .iter()
        .copied()
        .filter(|&y| y != 1.0 && y != 2.0)
        .collect();
    values.sort_by(f64::total_cmp);
    let mut histogram: Vec<(f64, f64)> = Vec::new();
    for y in values {
        match histogram.last_mut() {
            Some((v, m)) if *v == y => *m += 1.0,
            _ => histogram.push((y, 1.0)),
        }
    }
    histogram
}

/// `sum lgamma(y + cell)` over a count histogram.
///
/// ### Params
///
/// * `histogram` - From [`count_histogram`]
/// * `cell` - Cell-level negative binomial size
///
/// ### Returns
///
/// The tail term of the profile objective.
fn histogram_tail(histogram: &[(f64, f64)], cell: f64) -> f64 {
    histogram
        .iter()
        .map(|&(y, m)| m * ln_gamma(y + cell))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_tail_matches_the_direct_sum() {
        let counts = [1.0, 2.0, 3.0, 3.0, 7.0, 1.0, 12.0, 3.0, 2.0, 7.0];
        let cell = 0.8;
        let direct: f64 = counts
            .iter()
            .filter(|&&y| y != 1.0 && y != 2.0)
            .map(|&y| ln_gamma(y + cell))
            .sum();
        let histogram = count_histogram(&counts);
        assert_eq!(histogram, vec![(3.0, 3.0), (7.0, 2.0), (12.0, 1.0)]);
        approx::assert_relative_eq!(histogram_tail(&histogram, cell), direct, max_relative = 1e-14);
    }
}
