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
//! * **Stage two** runs as one `StageTwoSearch` per gene,
//!   the same state machine the CPU path drives. Each round, every live search
//!   asks for its next points and all of them go out as one device launch of
//!   [`ResidentBatch::submit`]. The device finds each penalised fit's optimum;
//!   the host then finishes that fit in `f64` from the device's point and
//!   assembles the profile objective. Nelder-Mead, the polish least squares and
//!   the objective never leave the host.
//! * **Stage three**, the final fit whose information gives the standard
//!   errors, stays on the CPU in `f64`.
//!
//! ### Why the host finishes every fit
//!
//! The search compares profile likelihoods finely: the objective moves about
//! `6e-6` for a `1e-3` relative move in the subject-level variance. The device's
//! own value cannot resolve that, and not for want of better summation. With
//! the sum made exact, the `f32` rounding of one `exp` and one `ln` per cell
//! still leaves the value off by `1.5e-3` to `2.7e-3` at 20000 cells and
//! jittering by `2e-5` to `2e-4` between nearby variance components, so the
//! device resolves the variance to a few tenths of a per cent at best, worse
//! with more cells. Searching on the device's values, measured on the R
//! fixtures, drove half the genes' subject-level variance onto its lower bound.
//! So the host finishes each fit: a full Newton step in `f64` from the device's
//! optimum (`newton_finish` in [`crate::sc::pml`]), and the value that comes
//! back is the CPU path's.
//!
//! The log-determinant is where this path parts from the CPU's. nebula reads it
//! off the penultimate iterate, which on the CPU is an `f64` iterate a hair from
//! the last. Here the penultimate iterate is the device's `f32` point, so the
//! finish takes it at the stepped point instead, for one more division per
//! cell. Measured on the R fixtures that moved the worst `sigma^2` disagreement
//! with nebula on `sc_small` from `3.4e-2` relative to `1.7e-3`, and the median
//! disagreement with the CPU path on the bench shapes from `2.4e-6` to `4e-7`.
//!
//! ### Where a fit starts
//!
//! nebula starts every inner fit cold, from the gene's mean count and zero
//! random effects. Here only a gene's first fit does; every later one starts
//! from the point that first fit converged to. The optimum does not depend on
//! the start, and the search stays close enough to its own starting variance
//! that the device's Newton count fell from five or six to two or three.
//!
//! The anchor is fixed on purpose. Starting each fit from the *previous* one's
//! optimum saves a little more per fit and was slower overall: the device's
//! `f32` location error then depends on the search's history instead of being a
//! smooth function of the variance components, that reaches the objective at
//! the simplex's `1e-7` tolerance, and the searches took 24 to 39 per cent more
//! evaluations. With a fixed anchor they took as many as from cold, 73247
//! against 73423 on one shape.
//!
//! ### Cost, measured
//!
//! Forced HL, 20000 cells, against a 10-thread CPU:
//!
//! | genes | CPU | GPU | stage one | device | host finish |
//! |---|---|---|---|---|---|
//! | 500 | 30.6 s | 23.1 s | | | |
//! | 4000 | 236 s | 151 s | 25 s | 45 s | 80 s |
//!
//! The host finish is now the largest part and the ceiling: it costs about a
//! third of the CPU's own stage two. With one thread per fit the device took
//! 140 s at 4000 genes and the run only broke even; see the kernel's module doc
//! for the mapping that fixed it.
//!
//! ### Why lockstep rounds
//!
//! Nelder-Mead is sequential within a gene, so the round count is set by the
//! slowest gene, and on the R fixtures that is about three times the median.
//! The rounds thin out as searches finish, since only live searches send
//! requests, so the total device work is the total number of evaluations, not
//! the slowest gene's count times the gene count. A polish stencil goes out as
//! all of its points in one round.
//!
//! The rounds are not strictly lockstep. The device sits idle while the host
//! finishes a round in `f64` and the host while the device fits, so where it
//! pays the searches run as two cohorts that leapfrog: one on the device while
//! the other is being finished. Where it pays is decided at run time; see
//! `PROBE_ROUNDS`. The cohorts never change what a search is told, only when.

use std::time::{Duration, Instant};

use cubecl::prelude::*;
use rayon::prelude::*;

use crate::errors::EdgeErrors;
use crate::gpu::nebula_gpu::{
    GpuGene, GpuSolveParams, PendingSolve, PmlReply, PmlRequest, ResidentBatch, SolveTiming,
};
use crate::gpu::pml_kernel::F32_NOISE_SCALE;
use crate::numeric::gamma::ln_gamma;
use crate::prelude::*;
use crate::sc::nebula::{
    GeneOutcome, GenePlan, InnerFit, NebulaFit, NebulaParams, Shared, StageTwoSearch, finish_gene,
    gene_pml, nebula_sparse_with, plan_gene, variance_bounds, variance_objective,
};
use crate::sc::pml::{DeviceCurvature, PmlParams, PmlVariance, newton_finish, opt_pml_from};
use crate::sc::ptmg::{GeneCounts, positive_indices};

////////////
// Consts //
////////////

/// Environment variable that switches on the per-phase timing report.
const TIMING_ENV: &str = "EDGE_RS_GPU_TIMING";

/// Launches kept in flight at once, each carrying its own share of the searches.
///
/// Two is what it takes for the device to run one cohort while the host
/// finishes the other in `f64`. The two costs are of the same order on every
/// shape measured, so a third cohort would only queue behind the second.
const COHORTS: usize = 2;

/// Rounds, after the first, run as one cohort to decide whether to split.
///
/// A launch costs the device about the same for 250 fits as for 500: measured
/// 75 ms either way at eight coefficients and 20000 cells, because what a
/// launch costs is one fit's serial walk over the cells and the device has
/// lanes to spare. Splitting the searches in two therefore doubles the device's
/// launches, and a search then advances once per two launches instead of once
/// per launch plus finish. That wins exactly when a round's `f64` finish takes
/// longer than its launch, which depends on the design width, the cell count
/// and the machine: measured 1.25x and 1.16x end to end at three coefficients
/// with 20000 and 100000 cells, and nothing at eight, where the launch is the
/// longer of the two. So the two are timed rather than assumed.
const PROBE_ROUNDS: usize = 8;

/// Running searches below which the cohorts merge into one.
///
/// A launch of a few dozen fits costs the device its fixed latency, about 19 ms
/// at 20000 cells, whatever it carries, and the host finishes it in a
/// millisecond or two. There is nothing left to overlap then, and two cohorts
/// would make every straggler wait out the other cohort's launch as well.
const MERGE_BELOW: usize = 64;

/// Bins of the Newton-step histograms; the last one collects everything above.
const STEP_BINS: usize = 12;

////////////
// Timing //
////////////

/// Where one [`fit_all`] spent its wall clock, reported under [`TIMING_ENV`].
#[derive(Default)]
struct Timing {
    /// Stage one, every gene.
    stage_one: Duration,
    /// Resident upload.
    upload: Duration,
    /// Building the requests of every round.
    gather: Duration,
    /// [`ResidentBatch::solve`], every round.
    solve: Duration,
    /// The `f64` finish and the objective assembly, every round.
    finish: Duration,
    /// Handing the searches their values.
    tell: Duration,
    /// Stage three, every gene.
    stage_three: Duration,
    /// Requests per round.
    requests: Vec<usize>,
    /// Newton steps the device took, per request.
    device_steps: [usize; STEP_BINS],
    /// Newton steps the `f64` finish took, per request.
    finish_steps: [usize; STEP_BINS],
}

impl Timing {
    /// Prints the report to stderr.
    ///
    /// ### Params
    ///
    /// * `solve` - The phases inside the device solves
    fn report(&self, solve: Option<SolveTiming>) {
        let secs = |d: Duration| d.as_secs_f64();
        let mut sorted = self.requests.clone();
        sorted.sort_unstable();
        let at = |f: f64| {
            sorted
                .get(((sorted.len() as f64 - 1.0) * f) as usize)
                .copied()
                .unwrap_or(0)
        };
        eprintln!(
            "gpu nebula: stage one {:.2} s, upload {:.2} s, gather {:.2} s, solve {:.2} s, finish {:.2} s, tell {:.2} s, stage three {:.2} s",
            secs(self.stage_one),
            secs(self.upload),
            secs(self.gather),
            secs(self.solve),
            secs(self.finish),
            secs(self.tell),
            secs(self.stage_three),
        );
        if let Some(t) = solve {
            eprintln!(
                "  solve: staging {:.2} s, blocked on the device {:.2} s, scatter {:.2} s",
                secs(t.staging),
                secs(t.blocked),
                secs(t.scatter),
            );
        }
        eprintln!(
            "  rounds {}, requests {}, per round min {} / median {} / max {}",
            sorted.len(),
            sorted.iter().sum::<usize>(),
            at(0.0),
            at(0.5),
            at(1.0),
        );
        eprintln!(
            "  device newton steps (last bin is {}+): {:?}",
            STEP_BINS - 1,
            self.device_steps
        );
        eprintln!(
            "  finish newton steps (last bin is {}+): {:?}",
            STEP_BINS - 1,
            self.finish_steps
        );
    }
}

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
    let mut timing = std::env::var_os(TIMING_ENV).map(|_| Timing::default());
    let started = Instant::now();
    let genes: Vec<(GeneCounts, GenePlan)> = kept
        .par_iter()
        .map(|&g| {
            let counts = positive_indices(sparse, g)?;
            let plan = plan_gene(shared, &counts, &totals[g * k..(g + 1) * k])?;
            Ok((counts, plan))
        })
        .collect::<Result<_, EdgeErrors>>()?;

    if let Some(t) = timing.as_mut() {
        t.stage_one = started.elapsed();
    }

    let (refits, solve_timing) = search_all(shared, &genes, totals, kept, client, timing.as_mut())?;

    let started = Instant::now();
    let outcomes = genes
        .into_par_iter()
        .zip(refits)
        .zip(kept.par_iter())
        .map(|(((counts, plan), refit), &g)| {
            finish_gene(shared, &counts, &totals[g * k..(g + 1) * k], plan, refit)
        })
        .collect();
    if let Some(t) = timing.as_mut() {
        t.stage_three = started.elapsed();
        t.report(solve_timing);
    }
    outcomes
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
    /// The device's fitted `(beta, log_w)` for the search's first finite
    /// evaluation, which every later fit starts from. See the module doc for
    /// why it is never updated.
    warm: Option<(Vec<f64>, Vec<f64>)>,
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
/// * `timing` - Where to book the phases, when the report is on
///
/// ### Returns
///
/// One entry per kept gene: `None` where no refit was called for, else the
/// search's result. Then the phases inside the device solves, when timed.
#[allow(clippy::type_complexity)]
fn search_all<R: Runtime>(
    shared: &Shared<'_>,
    genes: &[(GeneCounts, GenePlan)],
    totals: &[f64],
    kept: &[usize],
    client: &ComputeClient<R>,
    mut timing: Option<&mut Timing>,
) -> Result<(Vec<Option<Option<Vec<f64>>>>, Option<SolveTiming>), EdgeErrors> {
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
            warm: None,
        });
    }
    if live.is_empty() {
        return Ok((refits, None));
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
    let started = Instant::now();
    let mut resident = ResidentBatch::upload(
        shared.design,
        shared.log_offset,
        shared.fid,
        &resident_genes,
        client,
    )?;
    if let Some(t) = timing.as_deref_mut() {
        t.upload = started.elapsed();
        resident.time_solves();
    }

    let inner = PmlParams::default();
    let solve_params = GpuSolveParams {
        eps: shared.params.eps,
        noise_scale: f64::from(F32_NOISE_SCALE),
        max_iter: inner.max_iter as u32,
        max_backtrack: inner.max_backtrack as u32,
        full: true,
        // The host recomputes both in `f64`; see `finish_at_argmax`.
        information: false,
    };

    // Two cohorts leapfrog: while the host finishes one cohort's fits in `f64`,
    // the device is already running the other's. Whether that pays is read off
    // the first rounds, which run as one cohort; see `PROBE_ROUNDS`.
    let mut split = false;
    let mut probed = 0usize;
    let mut probe_blocked = Duration::ZERO;
    let mut probe_finish = Duration::ZERO;
    let mut in_flight = vec![false; live.len()];
    let mut flights: [Option<Flight<'_>>; COHORTS] = std::array::from_fn(|_| None);
    for (cohort, flight) in flights.iter_mut().enumerate() {
        *flight = launch_cohort(
            shared,
            genes,
            &live,
            &mut in_flight,
            cohort,
            split,
            &mut resident,
            &solve_params,
            client,
            timing.as_deref_mut(),
        )?;
    }
    let mut cohort = 0;
    while flights.iter().any(Option::is_some) || live.iter().any(|l| l.search.ask().is_some()) {
        if let Some(flight) = flights[cohort].take() {
            let started = Instant::now();
            let replies = resident.collect(flight.pending)?;
            let solved = Instant::now();

            // -- Scatter: assemble the profile objective in f64, in parallel,
            //    then hand each search its values. --
            let assembled: Vec<(f64, usize)> = flight
                .routes
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
                    let Some((fit, steps)) =
                        finish_at_argmax(&l.pml, reply, subject, cell, objective.params)
                    else {
                        return (f64::INFINITY, 0);
                    };
                    (
                        objective.assemble(subject, cell, &fit, histogram_tail(&l.tail, cell)),
                        steps,
                    )
                })
                .collect();
            let finished = Instant::now();
            // The first round is left out: it starts cold and is not what the
            // rest look like.
            if probed <= PROBE_ROUNDS {
                if probed > 0 {
                    probe_blocked += solved - started;
                    probe_finish += finished - solved;
                }
                split = probed == PROBE_ROUNDS && probe_finish > probe_blocked;
                probed += 1;
            }

            let mut values = flight.values;
            for ((&(i, p, _, _), &(v, _)), reply) in
                flight.routes.iter().zip(&assembled).zip(&replies)
            {
                let slot = values
                    .binary_search_by_key(&i, |(index, _)| *index)
                    .expect("every route belongs to a search of its own flight");
                values[slot].1[p] = v;
                if v.is_finite() && live[i].warm.is_none() {
                    live[i].warm = Some((reply.beta.clone(), reply.log_w.clone()));
                }
            }
            for (i, v) in values {
                live[i].search.tell(&v);
                in_flight[i] = false;
            }
            if let Some(t) = timing.as_deref_mut() {
                t.solve += solved - started;
                t.finish += finished - solved;
                t.tell += finished.elapsed();
                for (reply, &(_, steps)) in replies.iter().zip(&assembled) {
                    t.device_steps[(reply.iterations as usize).min(STEP_BINS - 1)] += 1;
                    t.finish_steps[steps.min(STEP_BINS - 1)] += 1;
                }
            }
        }
        // Tried even when nothing of this cohort's came back: once the cohorts
        // merge, the searches the other one hands over have nowhere else to go.
        flights[cohort] = launch_cohort(
            shared,
            genes,
            &live,
            &mut in_flight,
            cohort,
            split,
            &mut resident,
            &solve_params,
            client,
            timing.as_deref_mut(),
        )?;
        cohort = (cohort + 1) % COHORTS;
    }

    let solve_timing = resident.timing();
    for l in live {
        refits[l.index] = Some(l.search.result());
    }
    Ok((refits, solve_timing))
}

/// One cohort's launch in flight.
struct Flight<'a> {
    /// The device work.
    pending: PendingSolve<'a>,
    /// Per request: the search, the point within its batch, and the two variance
    /// components it was asked at.
    routes: Vec<(usize, usize, f64, f64)>,
    /// Per search in the flight, in increasing search order: its values, infinite
    /// until a fit fills them in.
    values: Vec<(usize, Vec<f64>)>,
}

/// Gathers a cohort's next points and launches them.
///
/// A search belongs to the cohort of its index while `split` is set and at
/// least [`MERGE_BELOW`] searches are running, and to cohort zero otherwise.
/// Points outside the domain are infinite without a fit.
///
/// ### Params
///
/// * `shared` - The inputs common to every gene
/// * `genes` - Every kept gene's counts and plan
/// * `live` - Every search
/// * `in_flight` - Which searches are waiting on a launch; updated
/// * `cohort` - The cohort to launch
/// * `split` - Whether the searches are split across the cohorts at all
/// * `resident` - The resident genes
/// * `solve_params` - Knobs shared by every request
/// * `client` - CubeCL compute client
/// * `timing` - Where to book the phases, when the report is on
///
/// ### Returns
///
/// The flight, or `None` when no search of the cohort is asking.
#[allow(clippy::too_many_arguments)]
fn launch_cohort<'a, R: Runtime>(
    shared: &Shared<'_>,
    genes: &[(GeneCounts, GenePlan)],
    live: &[Live<'_>],
    in_flight: &mut [bool],
    cohort: usize,
    split: bool,
    resident: &mut ResidentBatch<R>,
    solve_params: &GpuSolveParams,
    client: &'a ComputeClient<R>,
    timing: Option<&mut Timing>,
) -> Result<Option<Flight<'a>>, EdgeErrors> {
    let started = Instant::now();
    let running = live.iter().filter(|l| l.search.ask().is_some()).count();
    let merged = !split || running < MERGE_BELOW;

    let mut requests = Vec::new();
    let mut routes = Vec::new();
    let mut values = Vec::new();
    for (i, l) in live.iter().enumerate() {
        let mine = if merged {
            cohort == 0
        } else {
            i % COHORTS == cohort
        };
        if in_flight[i] || !mine {
            continue;
        }
        let Some(points) = l.search.ask() else {
            continue;
        };
        in_flight[i] = true;
        values.push((i, vec![f64::INFINITY; points.len()]));
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
                let (beta_init, log_w_init) = match &l.warm {
                    Some((beta, log_w)) => (beta.clone(), log_w.clone()),
                    None => (beta_init, Vec::new()),
                };
                requests.push(PmlRequest {
                    gene: l.resident,
                    subject,
                    cell,
                    beta_init,
                    log_w_init,
                });
                routes.push((i, p, subject, cell));
            }
        }
    }
    if values.is_empty() {
        return Ok(None);
    }

    let gathered = Instant::now();
    let pending = resident.submit(&requests, solve_params, client)?;
    if let Some(t) = timing {
        t.gather += gathered - started;
        t.solve += gathered.elapsed();
        t.requests.push(requests.len());
    }
    Ok(Some(Flight {
        pending,
        routes,
        values,
    }))
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
/// the `f64` fit is finished from the device's point instead: a Newton step
/// from there, with the log-determinant taken at the stepped point.
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
/// The inner fit and the Newton steps it took, or `None` if the device's point
/// is not finite or the fit fails.
fn finish_at_argmax(
    pml: &crate::sc::pml::PmlData<'_>,
    reply: &PmlReply,
    subject: f64,
    cell: f64,
    params: PmlParams,
) -> Option<(InnerFit, usize)> {
    if !reply.beta.iter().chain(&reply.log_w).all(|v| v.is_finite()) {
        return None;
    }
    // The fused steps cover an order-one fit that needs no damping, which is
    // nearly all of them; anything else takes the general loop.
    if params.ord == 1
        && !params.reml
        && let Some(fit) = newton_finish(
            pml,
            &reply.beta,
            &reply.log_w,
            &PmlVariance { subject, cell },
            params.eps,
            params.max_iter,
            Some(DeviceCurvature {
                subject: &reply.subject_curvature,
                cross: &reply.cross_block,
                schur: &reply.information,
            }),
        )
    {
        return Some((
            InnerFit {
                log_likelihood: fit.log_likelihood,
                log_likelihood_prev: fit.log_likelihood_prev,
                log_det: fit.log_det,
                second_order: 0.0,
            },
            fit.iterations,
        ));
    }
    let fit = opt_pml_from(
        pml,
        &reply.beta,
        &reply.log_w,
        &PmlVariance { subject, cell },
        Some(params),
    )
    .ok()?;
    Some((
        InnerFit {
            log_likelihood: fit.log_likelihood,
            log_likelihood_prev: fit.log_likelihood_prev,
            log_det: fit.log_det,
            second_order: fit.second_order,
        },
        fit.iterations,
    ))
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
    histogram.iter().map(|&(y, m)| m * ln_gamma(y + cell)).sum()
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
        approx::assert_relative_eq!(
            histogram_tail(&histogram, cell),
            direct,
            max_relative = 1e-14
        );
    }
}
