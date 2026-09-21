//! NEBULA's penalised maximum likelihood inner solver, one fit per plane.
//!
//! A port of [`crate::sc::pml`]'s Newton loop to CubeCL. The CPU module is the
//! reference for every formula; this file changes where the work happens, what
//! precision it happens in, and — because of that precision — how three of the
//! quantities are arranged.
//!
//! ### Mapping
//!
//! One 32-lane plane owns one fit and every lane runs the whole Newton loop,
//! backtracking included, on the same values. Only the passes over cells are
//! split: each lane takes a contiguous chunk of every subject block, and the
//! partial sums are recombined with plane reductions, so the quantities the
//! Newton step and the stopping tests read are identical in every lane and the
//! lanes never disagree about what to do next. No barrier, no shared memory.
//!
//! The first version put one fit on one thread. That made a launch cost one
//! thread's serial walk over every cell whatever the batch size: measured at
//! 20000 cells, 455 fits took 152 ms and 3644 took 202 ms, and stage two's
//! lockstep rounds carry a few thousand fits at most. Spreading each fit over a
//! plane measured 15x faster at 455 fits and 4.2x at 3644 on the likelihood
//! pass.
//!
//! ### Precision: the part better summation cannot fix
//!
//! wgpu exposes no `f64`, so the device arithmetic is `f32`. The obvious
//! defence is compensated summation, and on this backend it **does not work**.
//! wgpu-hal compiles every Metal shader with fast-math left on (it builds
//! `MTLCompileOptions` and sets only the language version and invariance,
//! `wgpu-hal-29.0.4/src/metal/device.rs:227`), which licenses reassociation.
//! Kahan and double-single both recover a rounding error through a `two_sum`
//! whose exact-algebra value is zero, so the compiler folds them away. Measured:
//! scaling the recovered low part by a thousand changed not one digit, and a
//! bit-level round-trip, a multiply by a runtime one read from a buffer and a
//! trip through global memory are all folded too. Integer fixed-point
//! accumulation is exact and costs nothing measurable, but see below for why
//! exact summation would not be enough.
//!
//! So the arithmetic has to be arranged so that nothing cancels, and three
//! places needed it. All three are exact rewrites, not approximations.
//!
//! * **The fixed-effect gradient.** The CPU forms `db = X'y - X'phi`, two sums
//!   of order `1e4` whose difference near the optimum is of order `1e-3`. Here
//!   it is `db = X'(y - phi)`, a sum of residuals. `X'y` is never formed.
//! * **The random-effect gradient.** Likewise `dw = cumsumy - sum phi` becomes
//!   `dw = sum (y - phi)` over the subject's cells.
//! * **The Schur complement.** `vb - vwb' vw^-1 vwb` cancels because the
//!   subject random effects absorb almost all of the intercept. Per subject,
//!   with `p` the curvature weight, `P = sum p`, `A_j = sum x_j p` and
//!   `M_ij = sum x_i x_j p`, the contribution
//!   `gamma M_ij - gamma^2 A_i A_j / (gamma P + lambda w)` rearranges exactly to
//!   `gamma [ C_ij + (A_i A_j / P) lambda w / (gamma P + lambda w) ]`, where
//!   `C_ij = sum p (x_i - A_i/P)(x_j - A_j/P)` is a weighted within-subject
//!   covariance. Neither term cancels. The centred sum needs the centre, which
//!   would mean a second pass over each subject block and a second `exp` per
//!   cell. So the moments are taken about an anchor known up front, the
//!   subject's unweighted mean design row, and moved to the weighted centre
//!   afterwards: `C_ij = S_ij - A_i A_j / P` with `S` and `A` about the anchor.
//!   That subtraction is the square of the gap between two means of the same
//!   cells against the spread, where about the origin it was the whole of the
//!   intercept; columns constant within a subject come out exactly zero. A
//!   Welford online update does the same job with no subtraction at all, but
//!   costs two divisions and a branch per cell and a pairwise merge across the
//!   plane, and measured 1.3x to 1.45x slower on the whole kernel.
//!
//! What is left is the `f32` rounding of one `exp` and one `ln` per cell, which
//! nothing here can undo: it leaves the value of the objective off by about
//! `2e-3` at 20000 cells and jittering by `2e-5` to `2e-4` between nearby
//! variance components, even with exact summation. That, not the summation, is
//! why the host finishes every fit in `f64` (see [`crate::gpu::stage_two`]).
//! Long sums run as a three-level tree (within a lane, across the plane, over
//! the subjects), a change of association the compiler cannot fold.
//!
//! ### Consequence
//!
//! This path does not reproduce the CPU's iteration count and must not be held
//! to the `1e-6` parity gate in `tests/e2e_nebula.rs`. It carries its own
//! measured tolerance, swept in `tests/e2e_nebula_gpu.rs`.
//!
//! ### References
//!
//! He et al., Communications Biology 4, 629, 2021

// The nested `if ptr < ptr_hi { if cells[ptr] == r { .. } }` guards cannot be
// collapsed: `&&` in a `#[cube]` body is not a short-circuit, so the collapsed
// form indexes `cells` past the gene's block. `!(likdif > eps)` is deliberate
// too, because it is the NaN-safe reading: a NaN improvement settles the loop
// rather than spinning it to the iteration cap.
#![allow(
    clippy::collapsible_if,
    clippy::neg_cmp_op_on_partial_ord,
    missing_docs
)]

use cubecl::prelude::*;
use cubecl_utils_rs::prelude::*;

use crate::errors::EdgeErrors;

////////////
// Consts //
////////////

/// Magnitude past which a Newton component is treated as unbounded rather than
/// damped. `STEP_CUTOFF` in [`crate::sc::pml`].
const STEP_CUTOFF: f32 = 40.0;

/// Largest gradient component still called a critical point, nebula's `convd`.
const GRADIENT_TOLERANCE: f32 = 0.01;

/// Lanes per request: one plane.
///
/// Every reduction here is a plane reduction, which is only correct when a
/// plane is exactly this wide; the dispatch refuses to run anywhere else rather
/// than return wrong answers, which a plane straddling two requests would give
/// silently.
pub const PLANE: u32 = 32;

/// Requests per workgroup, one plane each.
///
/// The planes share nothing, so this is purely an occupancy knob: two planes
/// make a 64-thread workgroup, which is what the per-thread kernel measured
/// best at and no worse than wider.
const PLANES_PER_CUBE: u32 = 2;

/// Default resolution floor on the objective, relative to its magnitude.
///
/// The CPU path has no equivalent: in `f64` nebula's absolute `1e-6` sits far
/// above its own noise floor. In `f32` it sits below it, and the consequence is
/// not a slow loop but a wrong answer. A genuine improvement smaller than the
/// per-cell rounding reads as a *worsening*, the backtracking search then damps
/// a good step to nothing, exhausts its budget and leaves the gene short of the
/// optimum. So a trial step is rejected only when it worsens the objective by
/// more than this floor. The stopping test keeps nebula's absolute `eps`:
/// applying the floor there as well costs a Newton step, and the direction that
/// has not converged is the one confounded with the random effects, which is
/// usually the coefficient of interest.
///
/// Swept rather than guessed; see `tests/e2e_nebula_gpu.rs`. Overridable per
/// call because the right value scales with the cell count.
pub const F32_NOISE_SCALE: f32 = 1e-6;

/// Number of per-subject scratch slots the kernel keeps in global memory.
///
/// `log_w`, `new_log_w`, `step_log_w`, `damp_log_w`, `w`, `dw`, `vw` and
/// `dwvw`. They are too large for registers at NEBULA's subject counts and too
/// small to be worth staging in shared memory.
pub const SUBJECT_SLOTS: u32 = 8;

/// Slot index of `log_w` within the per-subject scratch.
const SLOT_LOG_W: u32 = 0;
/// Slot index of the trial `log_w`.
const SLOT_NEW_LOG_W: u32 = 1;
/// Slot index of the Newton step in `log_w`.
const SLOT_STEP_LOG_W: u32 = 2;
/// Slot index of the per-coordinate damping on `log_w`.
const SLOT_DAMP_LOG_W: u32 = 3;
/// Slot index of `exp(log_w)`.
const SLOT_W: u32 = 4;
/// Slot index of the random-effect gradient.
const SLOT_DW: u32 = 5;
/// Slot index of the random-effect curvature.
const SLOT_VW: u32 = 6;
/// Slot index of `dw / vw`.
const SLOT_DWVW: u32 = 7;

/// Rows of the packed output ahead of the fitted point: the log-likelihood,
/// the previous one, the log-determinant, the final improvement, the Newton
/// steps taken and the backtracks used.
///
/// Everything a request brings back sits in one buffer so the host pays one
/// read-back per launch; each read is a round trip to the device whatever its
/// size. The two counts ride along as floats, which is exact at their size.
pub const OUT_HEADER: u32 = 6;

/// Largest design width the kernel is compiled for.
///
/// The `n_beta`-sized and `n_beta`-squared working arrays are registers, so
/// their capacity is a compile-time constant. The dispatch compiles one shader
/// per width rather than rounding up to a tier: register pressure is what bounds
/// occupancy here, and at the common `n_beta` of three a padded capacity of four
/// wastes seven registers on the quadratic block alone. Measured, dropping three
/// such arrays moved the kernel 18 per cent.
pub const MAX_BETA_CAP: usize = 8;

////////////
// Kernel //
////////////

/// Fits one request per 32-lane plane by penalised maximum likelihood.
///
/// Mirrors `optimise` in [`crate::sc::pml`] with the gamma penalty, NEBULA's
/// NBGMM, at Laplace order one. Higher orders are left to the host, which has
/// the `f64` path for them.
///
/// Every lane of a plane runs the same control flow on the same request. The
/// passes over cells are split across the lanes, each lane taking a contiguous
/// chunk of every subject block, and recombined with plane reductions, so every
/// quantity the Newton step and the backtracking search read is bit-identical
/// in all 32 lanes and their decisions agree without a barrier.
///
/// ### Params
///
/// * `design` - Shared design, row-major `n_cells * nb`
/// * `log_offset` - Shared log offset per cell, length `n_cells`
/// * `subject_start` - Shared subject boundaries, length `k + 1`
/// * `counts` - Every gene's positive counts, concatenated
/// * `cells` - Cell index of each entry of `counts`
/// * `subject_ptr` - Start of each subject's run within each gene's block in
///   `counts`, `[gene * (k + 1) + s]`
/// * `subject_total` - Count total per subject, `[s * n_genes + gene]`
/// * `subject_mean` - Unweighted mean design row per subject, `[s * nb + j]`
/// * `varying` - The design columns that vary within some subject, first
///   `n_varying` entries of `nb`. The others are constant within every subject,
///   so their deviation from the anchor is exactly zero and they drop out of
///   the curvature moments: an intercept and donor-level covariates cost the
///   sweep nothing
/// * `request_gene` - The gene each request is for
/// * `request_params` - Three per request: the gamma prior's `alpha` and
///   `lambda`, then the cell-level size `gamma`
/// * `beta_init` - Starting fixed effects, `[j * n_req + q]`
/// * `log_w_init` - Starting random effects on the log scale, `[s * n_req + q]`
/// * `tolerance` - Two elements: nebula's absolute stopping tolerance, then the
///   resolution floor relative to the objective ([`F32_NOISE_SCALE`]). A buffer
///   rather than scalar arguments because a runtime float scalar would need a
///   `ScalarArgSettings` bound this module otherwise has no use for
/// * `subject_scratch` - Per-subject working store,
///   `[(slot * k + s) * n_req + q]`, [`SUBJECT_SLOTS`] slots
/// * `vwb_scratch` - Cross block of the information, `[(s * nb + j) * n_req + q]`
/// * `out` - Everything a request brings back, `[row * n_req + q]`: the
///   [`OUT_HEADER`] rows, then `nb` fitted fixed effects, the `nb * nb` Schur
///   complement, the `k` fitted random effects on the log scale, the `k`
///   random-effect curvatures and the `k * nb` cross block
/// * `n_genes` - Genes resident on the device
/// * `n_req` - Requests in the launch
/// * `k` - Subjects
/// * `nb` - Design width
/// * `max_iter` - Newton budget
/// * `max_backtrack` - Backtracking budget within one step
/// * `final_assembly` - Non-zero to assemble once more at the point the fit
///   returns, which is what makes the reported information and log-determinant
///   belong to it. Zero skips that pass, a whole sweep over the cells, and
///   leaves both rows of the output at the last Newton step's values
/// * `n_varying` - How many entries of `varying` are live
/// * `nb_cap` - Comptime capacity of the `n_beta`-sized register arrays
///
/// ### Grid mapping
///
/// * `(CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * PLANES_PER_CUBE + UNIT_POS_Y`
///   -> request
/// * `UNIT_POS_X` -> lane within the request's plane
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn opt_pml_gpu<F: Float + CubeElement>(
    design: &Tensor<F>,
    log_offset: &Tensor<F>,
    subject_start: &Tensor<u32>,
    counts: &Tensor<F>,
    cells: &Tensor<u32>,
    subject_ptr: &Tensor<u32>,
    subject_total: &Tensor<F>,
    subject_mean: &Tensor<F>,
    varying: &Tensor<u32>,
    request_gene: &Tensor<u32>,
    request_params: &Tensor<F>,
    beta_init: &Tensor<F>,
    log_w_init: &Tensor<F>,
    tolerance: &Tensor<F>,
    subject_scratch: &mut Tensor<F>,
    vwb_scratch: &mut Tensor<F>,
    out: &mut Tensor<F>,
    n_genes: u32,
    n_req: u32,
    k: u32,
    nb: u32,
    max_iter: u32,
    max_backtrack: u32,
    final_assembly: u32,
    n_varying: u32,
    #[comptime] nb_cap: u32,
) {
    let q = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * PLANES_PER_CUBE + UNIT_POS_Y;
    if q >= n_req {
        terminate!();
    }
    let lane = UNIT_POS_X;
    let gene = request_gene[q as usize];

    let zero = F::new(0.0_f32);
    let one = F::new(1.0_f32);
    let two = F::new(2.0_f32);
    let cutoff = F::new(STEP_CUTOFF);
    // `is_infinite` has no device counterpart. Anything past the largest finite
    // `f32` either way is an infinity, and a NaN compares false on both sides,
    // which is what the CPU's test does too.
    let huge = F::new(f32::MAX);

    let eps = tolerance[0];
    let noise_scale = tolerance[1];

    let alpha = request_params[q as usize];
    let lambda = request_params[(n_req + q) as usize];
    let gamma = request_params[(2u32 * n_req + q) as usize];

    let mut beta = Array::<F>::new(nb_cap as usize);
    let mut new_beta = Array::<F>::new(nb_cap as usize);
    let mut step_beta = Array::<F>::new(nb_cap as usize);
    let mut damp_beta = Array::<F>::new(nb_cap as usize);
    let mut db = Array::<F>::new(nb_cap as usize);
    let mut db_lane = Array::<F>::new(nb_cap as usize);
    // The cell's design row. Every use below reads it from here: left in
    // global memory the row is re-read once per use, which is quadratic in
    // `nb` through the cross-products.
    let mut x = Array::<F>::new(nb_cap as usize);
    let mut centre = Array::<F>::new(nb_cap as usize);
    let mut anchor = Array::<F>::new(nb_cap as usize);
    let mut first = Array::<F>::new(nb_cap as usize);
    let mut delta = Array::<F>::new(nb_cap as usize);
    let mut vary = Array::<u32>::new(nb_cap as usize);
    let mut v = 0u32;
    while v < n_varying {
        vary[v as usize] = varying[v as usize];
        v += 1u32;
    }
    let mut spread = Array::<F>::new((nb_cap * nb_cap) as usize);
    let mut vb2 = Array::<F>::new((nb_cap * nb_cap) as usize);

    let mut j = 0u32;
    while j < nb {
        beta[j as usize] = beta_init[(j * n_req + q) as usize];
        j += 1u32;
    }
    let mut s = 0u32;
    while s < k {
        subject_scratch[((SLOT_LOG_W * k + s) * n_req + q) as usize] =
            log_w_init[(s * n_req + q) as usize];
        s += 1u32;
    }

    let mut ll = evaluate_pass::<F>(
        design,
        log_offset,
        subject_start,
        counts,
        cells,
        subject_ptr,
        subject_total,
        subject_scratch,
        &beta,
        SLOT_LOG_W,
        alpha,
        lambda,
        gamma,
        n_genes,
        n_req,
        k,
        q,
        gene,
        lane,
        nb,
    );

    let mut ll_prev = zero;
    let mut likdif = zero;
    let mut step = 0u32;
    let mut backtracks = 0u32;
    // The loop assembles first and steps second, so the assembly runs once more
    // after the final update and the reported information belongs to the point
    // the fit actually returns. See the module doc for why this departs from
    // nebula, which reports the penultimate iterate's.
    let mut settled = false;
    // Whether the stopping test has passed once already. See below.
    let mut confirmed = false;
    let mut running = true;

    while running {
        j = 0u32;
        while j < nb {
            damp_beta[j as usize] = one;
            db[j as usize] = zero;
            j += 1u32;
        }
        s = 0u32;
        while s < k {
            subject_scratch[((SLOT_DAMP_LOG_W * k + s) * n_req + q) as usize] = one;
            s += 1u32;
        }
        let mut i = 0u32;
        while i < nb * nb {
            vb2[i as usize] = zero;
            i += 1u32;
        }

        ///////////////////////////////////
        // Gradient and curvature, fused //
        ///////////////////////////////////

        s = 0u32;
        while s < k {
            let log_w_s = subject_scratch[((SLOT_LOG_W * k + s) * n_req + q) as usize];
            let w_s = F::exp(log_w_s);
            subject_scratch[((SLOT_W * k + s) * n_req + q) as usize] = w_s;

            let begin = subject_start[s as usize];
            let end = subject_start[(s + 1u32) as usize];
            let sp_lo = subject_ptr[(gene * (k + 1u32) + s) as usize];
            let sp_hi = subject_ptr[(gene * (k + 1u32) + s + 1u32) as usize];

            // Every quantity here is linear in the count `y`: with
            // `u = 1 / (1 + gamma / extb)` and `h = u / (extb + gamma)`, the
            // gradient weight is `(gamma + y) u`, the residual is
            // `-gamma u + y (1 - u)` and the curvature weight is `(gamma + y) h`.
            // So each subject is a dense pass over every cell at `y = 0`, then a
            // sparse pass over the subject's positive counts adding the `y`
            // parts. Both stride across the lanes, so a plane's loads fall on
            // consecutive cells: a contiguous chunk per lane, which walked the
            // counts with one pointer, measured 2.4x slower for the scattered
            // loads alone. The moments are order-free sums, so a count's
            // curvature enters as one more observation at its cell's covariates.
            let mut resid = zero;
            let mut weight = zero;
            j = 0u32;
            while j < nb {
                db_lane[j as usize] = zero;
                first[j as usize] = zero;
                anchor[j as usize] = subject_mean[(s * nb + j) as usize];
                j += 1u32;
            }
            i = 0u32;
            while i < nb * nb {
                spread[i as usize] = zero;
                i += 1u32;
            }

            let mut r = begin + lane;
            while r < end {
                let mut eta = log_offset[r as usize];
                j = 0u32;
                while j < nb {
                    x[j as usize] = design[(r * nb + j) as usize];
                    eta += x[j as usize] * beta[j as usize];
                    j += 1u32;
                }
                let extb = F::exp(eta + log_w_s);
                let u = one / (one + gamma / extb);
                let d = zero - gamma * u;
                let phi_c = gamma * u / (extb + gamma);
                moment_step::<F>(
                    &x,
                    &anchor,
                    &vary,
                    n_varying,
                    nb,
                    d,
                    phi_c,
                    &mut resid,
                    &mut db_lane,
                    &mut weight,
                    &mut first,
                    &mut spread,
                    &mut delta,
                );
                r += PLANE;
            }

            let mut p = sp_lo + lane;
            while p < sp_hi {
                let c = cells[p as usize];
                let y = counts[p as usize];
                let mut eta = log_offset[c as usize];
                j = 0u32;
                while j < nb {
                    x[j as usize] = design[(c * nb + j) as usize];
                    eta += x[j as usize] * beta[j as usize];
                    j += 1u32;
                }
                let extb = F::exp(eta + log_w_s);
                let u = one / (one + gamma / extb);
                let d = y * (one - u);
                let phi_c = y * u / (extb + gamma);
                moment_step::<F>(
                    &x,
                    &anchor,
                    &vary,
                    n_varying,
                    nb,
                    d,
                    phi_c,
                    &mut resid,
                    &mut db_lane,
                    &mut weight,
                    &mut first,
                    &mut spread,
                    &mut delta,
                );
                p += PLANE;
            }

            // -- Across the plane. Every moment is taken about the same anchor in
            //    every lane, so all of them are plain sums. --
            let resid_s = plane_sum(resid);
            j = 0u32;
            while j < nb {
                db[j as usize] += plane_sum(db_lane[j as usize]);
                j += 1u32;
            }
            weight = plane_sum(weight);
            let mut va = 0u32;
            while va < n_varying {
                let a = vary[va as usize];
                first[a as usize] = plane_sum(first[a as usize]);
                let mut vb = va;
                while vb < n_varying {
                    let idx = (a * nb + vary[vb as usize]) as usize;
                    spread[idx] = plane_sum(spread[idx]);
                    vb += 1u32;
                }
                va += 1u32;
            }

            // Move the moments from the anchor to the weighted centre. The
            // anchor is the subject's unweighted mean, so what is subtracted is
            // the square of the gap between two means, small against the
            // spread; about the origin it would be the whole of the intercept.
            j = 0u32;
            while j < nb {
                centre[j as usize] = anchor[j as usize];
                j += 1u32;
            }
            if weight > zero {
                va = 0u32;
                while va < n_varying {
                    let a = vary[va as usize];
                    let shift = first[a as usize] / weight;
                    let mut vb = va;
                    while vb < n_varying {
                        let b = vary[vb as usize];
                        spread[(a * nb + b) as usize] -= shift * first[b as usize];
                        vb += 1u32;
                    }
                    centre[a as usize] = anchor[a as usize] + shift;
                    va += 1u32;
                }
            }

            let dw_s = resid_s + (alpha - lambda * w_s);
            subject_scratch[((SLOT_DW * k + s) * n_req + q) as usize] = dw_s;
            let vw_s = gamma * weight + lambda * w_s;
            subject_scratch[((SLOT_VW * k + s) * n_req + q) as usize] = vw_s;

            j = 0u32;
            while j < nb {
                vwb_scratch[((s * nb + j) * n_req + q) as usize] =
                    gamma * centre[j as usize] * weight;
                j += 1u32;
            }

            // The centred covariance, then what the centring leaves over, which
            // the prior's share of the subject curvature keeps from cancelling.
            let shrink = lambda * w_s / vw_s;
            let mut a = 0u32;
            while a < nb {
                let mut b = a;
                while b < nb {
                    vb2[(a * nb + b) as usize] += gamma * spread[(a * nb + b) as usize]
                        + gamma * centre[a as usize] * centre[b as usize] * weight * shrink;
                    b += 1u32;
                }
                a += 1u32;
            }

            s += 1u32;
        }

        let mut a = 0u32;
        while a < nb {
            let mut b = a + 1u32;
            while b < nb {
                vb2[(b * nb + a) as usize] = vb2[(a * nb + b) as usize];
                b += 1u32;
            }
            a += 1u32;
        }

        /////////////////
        // Newton step //
        /////////////////

        if settled {
            // The assembly above ran at the point the fit returns, which is
            // what the outputs want; nothing further to do.
            running = false;
        } else {
            step += 1u32;

                s = 0u32;
            while s < k {
                let dw_s = subject_scratch[((SLOT_DW * k + s) * n_req + q) as usize];
                let vw_s = subject_scratch[((SLOT_VW * k + s) * n_req + q) as usize];
                subject_scratch[((SLOT_DWVW * k + s) * n_req + q) as usize] = dw_s / vw_s;
                s += 1u32;
            }

            j = 0u32;
            while j < nb {
                let mut acc = zero;
                s = 0u32;
                while s < k {
                    acc += vwb_scratch[((s * nb + j) * n_req + q) as usize]
                        * subject_scratch[((SLOT_DWVW * k + s) * n_req + q) as usize];
                    s += 1u32;
                }
                step_beta[j as usize] = db[j as usize] - acc;
                j += 1u32;
            }

            // The solve destroys its matrix, so the Schur complement goes out
            // first. With a final assembly these rows are overwritten by the one
            // at the returned point; without, they are what the host gets.
            i = 0u32;
            while i < nb * nb {
                out[((OUT_HEADER + nb + i) * n_req + q) as usize] = vb2[i as usize];
                i += 1u32;
            }
            ldlt_solve::<F>(&mut vb2, &mut step_beta, nb, nb_cap);

            s = 0u32;
            while s < k {
                let mut acc = zero;
                j = 0u32;
                while j < nb {
                    acc += vwb_scratch[((s * nb + j) * n_req + q) as usize] * step_beta[j as usize];
                    j += 1u32;
                }
                let vw_s = subject_scratch[((SLOT_VW * k + s) * n_req + q) as usize];
                let dwvw = subject_scratch[((SLOT_DWVW * k + s) * n_req + q) as usize];
                subject_scratch[((SLOT_STEP_LOG_W * k + s) * n_req + q) as usize] = dwvw - acc / vw_s;
                s += 1u32;
            }

            j = 0u32;
            while j < nb {
                new_beta[j as usize] = beta[j as usize] + step_beta[j as usize];
                j += 1u32;
            }
            s = 0u32;
            while s < k {
                let base = subject_scratch[((SLOT_LOG_W * k + s) * n_req + q) as usize];
                let d = subject_scratch[((SLOT_STEP_LOG_W * k + s) * n_req + q) as usize];
                subject_scratch[((SLOT_NEW_LOG_W * k + s) * n_req + q) as usize] = base + d;
                s += 1u32;
            }

            ll_prev = ll;
            ll = evaluate_pass::<F>(
                design,
                log_offset,
                subject_start,
                counts,
                cells,
                subject_ptr,
                subject_total,
                subject_scratch,
                &new_beta,
                SLOT_NEW_LOG_W,
                alpha,
                lambda,
                gamma,
                n_genes,
                n_req,
                k,
                q,
                gene,
                lane,
                nb,
            );
            likdif = ll - ll_prev;

            /////////////////////////
            // Backtracking search //
            /////////////////////////

            // The floor is read off the previous iterate so it does not move while
            // the search damps.
            let noise = F::abs(ll_prev) * noise_scale;
            backtracks = 0u32;
            let mut min_step = cutoff;
            let mut searching = likdif < zero - noise || ll > huge || ll < zero - huge;
            while searching {
                backtracks += 1u32;
                min_step /= two;

                if backtracks > max_backtrack {
                    likdif = zero;
                    ll = ll_prev;
                    let mut worst = zero;
                    j = 0u32;
                    while j < nb {
                        let v = F::abs(db[j as usize]);
                        if v > worst {
                            worst = v;
                        }
                        j += 1u32;
                    }
                    s = 0u32;
                    while s < k {
                        let v = F::abs(subject_scratch[((SLOT_DW * k + s) * n_req + q) as usize]);
                        if v > worst {
                            worst = v;
                        }
                        s += 1u32;
                    }
                    if worst > F::new(GRADIENT_TOLERANCE) {
                        backtracks += 1u32;
                    }
                    searching = false;
                } else {
                    j = 0u32;
                    while j < nb {
                        let d = step_beta[j as usize];
                        if d < cutoff && d > zero - cutoff {
                            damp_beta[j as usize] = damp_beta[j as usize] / two;
                            new_beta[j as usize] = beta[j as usize] + d * damp_beta[j as usize];
                        } else if d > zero {
                            new_beta[j as usize] = beta[j as usize] + min_step;
                        } else {
                            new_beta[j as usize] = beta[j as usize] - min_step;
                        }
                        j += 1u32;
                    }
                    s = 0u32;
                    while s < k {
                        let d = subject_scratch[((SLOT_STEP_LOG_W * k + s) * n_req + q) as usize];
                        let base = subject_scratch[((SLOT_LOG_W * k + s) * n_req + q) as usize];
                        let damp_idx = ((SLOT_DAMP_LOG_W * k + s) * n_req + q) as usize;
                        let trial = if d < cutoff && d > zero - cutoff {
                            let damped = subject_scratch[damp_idx] / two;
                            subject_scratch[damp_idx] = damped;
                            base + d * damped
                        } else if d > zero {
                            base + min_step
                        } else {
                            base - min_step
                        };
                        subject_scratch[((SLOT_NEW_LOG_W * k + s) * n_req + q) as usize] = trial;
                        s += 1u32;
                    }

                    ll = evaluate_pass::<F>(
                        design,
                        log_offset,
                        subject_start,
                        counts,
                        cells,
                        subject_ptr,
                        subject_total,
                        subject_scratch,
                        &new_beta,
                        SLOT_NEW_LOG_W,
                        alpha,
                        lambda,
                        gamma,
                        n_genes,
                        n_req,
                        k,
                        q,
                        gene,
                        lane,
                        nb,
                    );
                    likdif = ll - ll_prev;
                    searching = likdif < zero - noise || ll > huge || ll < zero - huge;
                }
            }

            j = 0u32;
            while j < nb {
                beta[j as usize] = new_beta[j as usize];
                j += 1u32;
            }
            s = 0u32;
            while s < k {
                subject_scratch[((SLOT_LOG_W * k + s) * n_req + q) as usize] =
                    subject_scratch[((SLOT_NEW_LOG_W * k + s) * n_req + q) as usize];
                s += 1u32;
            }

            // In `f32` the improvement near the optimum is rounding, so the
            // stopping test passes as soon as one step happens not to improve,
            // which is early. The value the host reads off is insensitive to
            // that at first order, but the log-determinant is not stationary at
            // the optimum and carries the location error straight into the
            // profile objective. So the test has to pass twice: the second
            // pass is one more Newton step, which squares the location error.
            if step >= max_iter {
                settled = true;
            } else if !(likdif > eps) {
                settled = confirmed;
                confirmed = true;
            }
            if settled && final_assembly == 0u32 {
                running = false;
            }
        }
    }

    /////////////
    // Outputs //
    /////////////

    let mut log_det = zero;
    s = 0u32;
    while s < k {
        log_det += F::ln(F::abs(
            subject_scratch[((SLOT_VW * k + s) * n_req + q) as usize],
        ));
        s += 1u32;
    }

    out[q as usize] = ll;
    out[(n_req + q) as usize] = ll_prev;
    out[(2u32 * n_req + q) as usize] = log_det;
    out[(3u32 * n_req + q) as usize] = likdif;
    out[(4u32 * n_req + q) as usize] = F::cast_from(step);
    out[(5u32 * n_req + q) as usize] = F::cast_from(backtracks);
    // The rows are indexed rather than counted: a counter started from the
    // comptime header is itself comptime and cannot be advanced.
    j = 0u32;
    while j < nb {
        out[((OUT_HEADER + j) * n_req + q) as usize] = beta[j as usize];
        j += 1u32;
    }
    if final_assembly != 0u32 {
        let mut idx = 0u32;
        while idx < nb * nb {
            out[((OUT_HEADER + nb + idx) * n_req + q) as usize] = vb2[idx as usize];
            idx += 1u32;
        }
    }
    // The random-effect block and the cross block ride along from the same
    // assembly as the Schur complement, whichever that was.
    s = 0u32;
    while s < k {
        out[((OUT_HEADER + nb + nb * nb + s) * n_req + q) as usize] =
            subject_scratch[((SLOT_LOG_W * k + s) * n_req + q) as usize];
        out[((OUT_HEADER + nb + nb * nb + k + s) * n_req + q) as usize] =
            subject_scratch[((SLOT_VW * k + s) * n_req + q) as usize];
        j = 0u32;
        while j < nb {
            out[((OUT_HEADER + nb + nb * nb + 2u32 * k + s * nb + j) * n_req + q) as usize] =
                vwb_scratch[((s * nb + j) * n_req + q) as usize];
            j += 1u32;
        }
        s += 1u32;
    }
}

/////////////////
// Cell passes //
/////////////////

/// The penalised log-likelihood at `(beta, log_w)`, reduced across the plane.
///
/// Accumulates the four terms of the CPU's `Workspace::evaluate`: the linear
/// term over the positive counts, the subject term, `-gamma` times the sum of
/// `log(extb + gamma)` over all cells, and the count-weighted sum of the same
/// logs over the positive counts. One pass per subject covers the first, third
/// and fourth, each lane summing its own chunk, then the plane, then the
/// subjects: a three-level tree, which is what keeps a long `f32` sum honest
/// here without compensation.
///
/// ### Params
///
/// * `design` - Shared design, row-major
/// * `log_offset` - Shared log offset per cell
/// * `subject_start` - Shared subject boundaries
/// * `counts` - Concatenated positive counts
/// * `cells` - Cell index of each count
/// * `subject_ptr` - Start of each subject's run in each gene's counts
/// * `subject_total` - Count total per subject, gene-minor
/// * `subject_scratch` - Per-subject store holding the random effects
/// * `beta` - Fixed effects to evaluate at
/// * `slot_log_w` - Which scratch slot holds the random effects
/// * `alpha` - Shape of the gamma prior
/// * `lambda` - Rate of the gamma prior
/// * `gamma` - Cell-level negative binomial size
/// * `n_genes` - Genes resident on the device, the stride of `subject_total`
/// * `n_req` - Requests in the launch, the stride of `subject_scratch`
/// * `k` - Subjects
/// * `q` - This plane's request
/// * `gene` - The gene the request is for
/// * `lane` - This thread's lane within the plane
/// * `nb` - Design width
///
/// ### Returns
///
/// The penalised log-likelihood, identical in every lane.
#[cube]
#[allow(clippy::too_many_arguments)]
fn evaluate_pass<F: Float>(
    design: &Tensor<F>,
    log_offset: &Tensor<F>,
    subject_start: &Tensor<u32>,
    counts: &Tensor<F>,
    cells: &Tensor<u32>,
    subject_ptr: &Tensor<u32>,
    subject_total: &Tensor<F>,
    subject_scratch: &Tensor<F>,
    beta: &Array<F>,
    slot_log_w: u32,
    alpha: F,
    lambda: F,
    gamma: F,
    n_genes: u32,
    n_req: u32,
    k: u32,
    q: u32,
    gene: u32,
    lane: u32,
    nb: u32,
) -> F {
    let zero = F::new(0.0_f32);

    let mut linear = zero;
    let mut phil = zero;
    let mut weighted = zero;
    let mut subject_term = zero;
    let mut sum_log_w = zero;
    let mut sum_w = zero;

    let mut s = 0u32;
    while s < k {
        let log_w_s = subject_scratch[((slot_log_w * k + s) * n_req + q) as usize];
        sum_log_w += log_w_s;
        sum_w += F::exp(log_w_s);
        subject_term += log_w_s * subject_total[(s * n_genes + gene) as usize];

        let begin = subject_start[s as usize];
        let end = subject_start[(s + 1u32) as usize];
        let sp_lo = subject_ptr[(gene * (k + 1u32) + s) as usize];
        let sp_hi = subject_ptr[(gene * (k + 1u32) + s + 1u32) as usize];

        // Dense over every cell for `log(extb + gamma)`, then sparse over the
        // positive counts for the two count-weighted terms; both strided across
        // the lanes. See the assembly in `opt_pml_gpu` for why.
        let mut lin_lane = zero;
        let mut phil_lane = zero;
        let mut weighted_lane = zero;
        // Four of the lane's cells per trip, every load ahead of its use: the
        // sweep waits on memory, not arithmetic, and four loads in flight hide
        // most of that wait. The logarithms are taken on products of two, which
        // halves them; each term is at least `gamma` and a product of two stays
        // far inside the `f32` range for any count a fit can reach.
        let mut r = begin + lane;
        while r + 3u32 * PLANE < end {
            let r1 = r + PLANE;
            let r2 = r1 + PLANE;
            let r3 = r2 + PLANE;
            let mut eta0 = log_offset[r as usize];
            let mut eta1 = log_offset[r1 as usize];
            let mut eta2 = log_offset[r2 as usize];
            let mut eta3 = log_offset[r3 as usize];
            let mut j = 0u32;
            while j < nb {
                let b = beta[j as usize];
                eta0 += design[(r * nb + j) as usize] * b;
                eta1 += design[(r1 * nb + j) as usize] * b;
                eta2 += design[(r2 * nb + j) as usize] * b;
                eta3 += design[(r3 * nb + j) as usize] * b;
                j += 1u32;
            }
            let t0 = F::exp(eta0 + log_w_s) + gamma;
            let t1 = F::exp(eta1 + log_w_s) + gamma;
            let t2 = F::exp(eta2 + log_w_s) + gamma;
            let t3 = F::exp(eta3 + log_w_s) + gamma;
            phil_lane += F::ln(t0 * t1) + F::ln(t2 * t3);
            r += 4u32 * PLANE;
        }
        while r < end {
            let mut eta = log_offset[r as usize];
            let mut j = 0u32;
            while j < nb {
                eta += design[(r * nb + j) as usize] * beta[j as usize];
                j += 1u32;
            }
            phil_lane += F::ln(F::exp(eta + log_w_s) + gamma);
            r += PLANE;
        }
        let mut p = sp_lo + lane;
        while p < sp_hi {
            let c = cells[p as usize];
            let y = counts[p as usize];
            let mut eta = log_offset[c as usize];
            let mut j = 0u32;
            while j < nb {
                eta += design[(c * nb + j) as usize] * beta[j as usize];
                j += 1u32;
            }
            // The linear term takes the predictor without the random effect,
            // which the subject term adds back.
            lin_lane += eta * y;
            weighted_lane += y * F::ln(F::exp(eta + log_w_s) + gamma);
            p += PLANE;
        }
        linear += plane_sum(lin_lane);
        phil += plane_sum(phil_lane);
        weighted += plane_sum(weighted_lane);
        s += 1u32;
    }

    let mut acc = linear + subject_term;
    acc -= gamma * phil;
    acc -= weighted;
    acc + (alpha * sum_log_w - lambda * sum_w)
}

/// Folds one observation into a lane's residual and weighted moments.
///
/// The per-cell body of the assembly, shared by the dense and the sparse pass.
/// The moments are taken about a fixed anchor, so they are order-free sums with
/// no division and no branch per cell.
///
/// ### Params
///
/// * `x` - The design row of the cell the observation belongs to
/// * `anchor` - The point the moments are taken about
/// * `vary` - The columns that vary within some subject, in increasing order
/// * `n_varying` - How many of them there are
/// * `nb` - Design width
/// * `d` - Contribution to the residual
/// * `w` - Curvature weight of the observation
/// * `resid` - The lane's residual sum
/// * `db_lane` - The lane's fixed-effect gradient
/// * `weight` - The lane's total curvature weight
/// * `first` - The lane's weighted sum of `x - anchor`
/// * `spread` - The lane's weighted cross-products of `x - anchor`, upper
///   triangle
/// * `delta` - Scratch for `x - anchor` over the varying columns
#[cube]
#[allow(clippy::too_many_arguments)]
fn moment_step<F: Float>(
    x: &Array<F>,
    anchor: &Array<F>,
    vary: &Array<u32>,
    n_varying: u32,
    nb: u32,
    d: F,
    w: F,
    resid: &mut F,
    db_lane: &mut Array<F>,
    weight: &mut F,
    first: &mut Array<F>,
    spread: &mut Array<F>,
    delta: &mut Array<F>,
) {
    *resid += d;
    *weight += w;
    let mut j = 0u32;
    while j < nb {
        db_lane[j as usize] += x[j as usize] * d;
        j += 1u32;
    }
    let mut va = 0u32;
    while va < n_varying {
        let a = vary[va as usize];
        delta[va as usize] = x[a as usize] - anchor[a as usize];
        va += 1u32;
    }
    va = 0u32;
    while va < n_varying {
        let a = vary[va as usize];
        let wa = w * delta[va as usize];
        first[a as usize] += wa;
        let mut vb = va;
        while vb < n_varying {
            let b = vary[vb as usize];
            spread[(a * nb + b) as usize] += wa * delta[vb as usize];
            vb += 1u32;
        }
        va += 1u32;
    }
}

//////////////////
// Dense linear //
//////////////////

/// Pivoted `LDL'` solve of a small symmetric system, in place.
///
/// A transcription of `ldlt_solve` in [`crate::sc::pml`], which follows Eigen's
/// `LDLT`. Indefinite iterates are expected here, which is why the pivoting is
/// worth its cost on a matrix this small.
///
/// ### Params
///
/// * `a` - Row-major `n * n` symmetric matrix, overwritten by the factor
/// * `b` - Right-hand side, overwritten by the solution
/// * `n` - System size
/// * `n_cap` - Comptime capacity of the internal scratch
#[cube]
fn ldlt_solve<F: Float>(a: &mut Array<F>, b: &mut Array<F>, n: u32, #[comptime] n_cap: u32) {
    let zero = F::new(0.0_f32);
    let tiny = F::new(f32::MIN_POSITIVE);
    let mut perm = Array::<u32>::new(n_cap as usize);
    let mut tmp = Array::<F>::new(n_cap as usize);

    let mut step = 0u32;
    while step < n {
        let mut pivot = step;
        let mut best = F::abs(a[(step * n + step) as usize]);
        let mut i = step + 1u32;
        while i < n {
            let v = F::abs(a[(i * n + i) as usize]);
            if v > best {
                best = v;
                pivot = i;
            }
            i += 1u32;
        }
        perm[step as usize] = pivot;
        if pivot != step {
            let mut j = 0u32;
            while j < step {
                let t = a[(step * n + j) as usize];
                a[(step * n + j) as usize] = a[(pivot * n + j) as usize];
                a[(pivot * n + j) as usize] = t;
                j += 1u32;
            }
            i = pivot + 1u32;
            while i < n {
                let t = a[(i * n + step) as usize];
                a[(i * n + step) as usize] = a[(i * n + pivot) as usize];
                a[(i * n + pivot) as usize] = t;
                i += 1u32;
            }
            let t = a[(step * n + step) as usize];
            a[(step * n + step) as usize] = a[(pivot * n + pivot) as usize];
            a[(pivot * n + pivot) as usize] = t;
            i = step + 1u32;
            while i < pivot {
                let t2 = a[(i * n + step) as usize];
                a[(i * n + step) as usize] = a[(pivot * n + i) as usize];
                a[(pivot * n + i) as usize] = t2;
                i += 1u32;
            }
        }

        if step > 0u32 {
            let mut j = 0u32;
            while j < step {
                tmp[j as usize] = a[(j * n + j) as usize] * a[(step * n + j) as usize];
                j += 1u32;
            }
            let mut acc = zero;
            j = 0u32;
            while j < step {
                acc += a[(step * n + j) as usize] * tmp[j as usize];
                j += 1u32;
            }
            a[(step * n + step) as usize] -= acc;
            i = step + 1u32;
            while i < n {
                let mut acc2 = zero;
                j = 0u32;
                while j < step {
                    acc2 += a[(i * n + j) as usize] * tmp[j as usize];
                    j += 1u32;
                }
                a[(i * n + step) as usize] -= acc2;
                i += 1u32;
            }
        }

        let d = a[(step * n + step) as usize];
        if d != zero {
            i = step + 1u32;
            while i < n {
                a[(i * n + step) as usize] = a[(i * n + step) as usize] / d;
                i += 1u32;
            }
        }
        step += 1u32;
    }

    step = 0u32;
    while step < n {
        let pv = perm[step as usize];
        let t = b[step as usize];
        b[step as usize] = b[pv as usize];
        b[pv as usize] = t;
        step += 1u32;
    }
    let mut row = 0u32;
    while row < n {
        let mut acc = b[row as usize];
        let mut j = 0u32;
        while j < row {
            acc -= a[(row * n + j) as usize] * b[j as usize];
            j += 1u32;
        }
        b[row as usize] = acc;
        row += 1u32;
    }
    row = 0u32;
    while row < n {
        let d = a[(row * n + row) as usize];
        if F::abs(d) > tiny {
            b[row as usize] = b[row as usize] / d;
        } else {
            b[row as usize] = zero;
        }
        row += 1u32;
    }
    row = n;
    while row > 0u32 {
        row -= 1u32;
        let mut acc = b[row as usize];
        let mut j = row + 1u32;
        while j < n {
            acc -= a[(j * n + row) as usize] * b[j as usize];
            j += 1u32;
        }
        b[row as usize] = acc;
    }
    step = n;
    while step > 0u32 {
        step -= 1u32;
        let pv = perm[step as usize];
        let t = b[step as usize];
        b[step as usize] = b[pv as usize];
        b[pv as usize] = t;
    }
}

//////////////
// Dispatch //
//////////////

/// Launches [`fn@opt_pml_gpu`] over a batch of requests.
///
/// ### Params
///
/// * `tensors` - Every device buffer the kernel reads or writes
/// * `n_genes` - Genes resident on the device
/// * `n_req` - Requests in this launch, which is the thread count
/// * `k` - Subjects
/// * `nb` - Design width
/// * `max_iter` - Newton budget
/// * `max_backtrack` - Backtracking budget within one step
/// * `final_assembly` - Whether to assemble once more at the returned point;
///   see [`fn@opt_pml_gpu`]
/// * `n_varying` - How many entries of [`PmlGpuTensors::varying`] are live
/// * `client` - CubeCL compute client
///
/// ### Returns
///
/// `Ok(())`, with the outputs in `tensors` filled for the first `n_req`
/// requests.
///
/// ### Errors
///
/// * [`EdgeErrors::InvalidArgument`] if `nb` exceeds [`MAX_BETA_CAP`].
/// * [`EdgeErrors::Gpu`] if the grid busts the device's cube-count limit.
#[allow(clippy::too_many_arguments)]
pub fn launch_opt_pml<R, F>(
    tensors: &PmlGpuTensors<R, F>,
    n_genes: usize,
    n_req: usize,
    k: usize,
    nb: usize,
    max_iter: u32,
    max_backtrack: u32,
    final_assembly: bool,
    n_varying: usize,
    client: &ComputeClient<R>,
) -> Result<(), EdgeErrors>
where
    R: Runtime,
    F: Float + cubecl::CubeElement,
{
    if nb == 0 || nb > MAX_BETA_CAP {
        return Err(EdgeErrors::InvalidArgument(format!(
            "The GPU NEBULA path is compiled for designs of one to {MAX_BETA_CAP} columns; got {nb}."
        )));
    }
    if n_req == 0 {
        return Ok(());
    }

    let limits = GpuLimits::from_client(client);
    // Every reduction is a plane reduction; on a plane of any other width they
    // would mix requests and return wrong answers without an error.
    if !plane_uniform(PLANE, &limits) {
        return Err(EdgeErrors::Gpu(format!(
            "The GPU NEBULA kernel needs a plane of exactly {PLANE} lanes; this device reports {} to {}.",
            limits.plane_size_min, limits.plane_size_max
        )));
    }
    let blocks = (n_req as u32).div_ceil(PLANES_PER_CUBE);
    let (gx, gy) = grid_2d(blocks, &limits).map_err(|e| EdgeErrors::Gpu(e.to_string()))?;
    let count = checked_cube_count("opt_pml_gpu", gx, gy, 1, &limits)
        .map_err(|e| EdgeErrors::Gpu(e.to_string()))?;

    macro_rules! dispatch {
        ($cap:expr) => {
            unsafe {
                opt_pml_gpu::launch_unchecked::<F, R>(
                    client,
                    count,
                    CubeDim::new_2d(PLANE, PLANES_PER_CUBE),
                    tensors.design.clone().into_tensor_arg(),
                    tensors.log_offset.clone().into_tensor_arg(),
                    tensors.subject_start.clone().into_tensor_arg(),
                    tensors.counts.clone().into_tensor_arg(),
                    tensors.cells.clone().into_tensor_arg(),
                    tensors.subject_ptr.clone().into_tensor_arg(),
                    tensors.subject_total.clone().into_tensor_arg(),
                    tensors.subject_mean.clone().into_tensor_arg(),
                    tensors.varying.clone().into_tensor_arg(),
                    tensors.request_gene.clone().into_tensor_arg(),
                    tensors.request_params.clone().into_tensor_arg(),
                    tensors.beta_init.clone().into_tensor_arg(),
                    tensors.log_w_init.clone().into_tensor_arg(),
                    tensors.tolerance.clone().into_tensor_arg(),
                    tensors.subject_scratch.clone().into_tensor_arg(),
                    tensors.vwb_scratch.clone().into_tensor_arg(),
                    tensors.out.clone().into_tensor_arg(),
                    n_genes as u32,
                    n_req as u32,
                    k as u32,
                    nb as u32,
                    max_iter,
                    max_backtrack,
                    u32::from(final_assembly),
                    n_varying as u32,
                    $cap,
                );
            }
        };
    }

    match nb {
        1 => dispatch!(1),
        2 => dispatch!(2),
        3 => dispatch!(3),
        4 => dispatch!(4),
        5 => dispatch!(5),
        6 => dispatch!(6),
        7 => dispatch!(7),
        _ => dispatch!(8),
    }

    Ok(())
}

/// Every device buffer [`launch_opt_pml`] binds.
///
/// Two kinds, with two strides. The gene-resident buffers (the design, the
/// offsets, the subject boundaries, the counts and the per-gene totals) are
/// uploaded once and indexed through a request's gene. The request buffers are
/// rewritten per launch and indexed by request, request-minor: entry `i` of
/// request `q` lives at `i * n_req + q`, so consecutive threads touch
/// consecutive addresses.
pub struct PmlGpuTensors<R: Runtime, F: cubecl::CubeElement + Numeric> {
    /// Shared design, row-major `n_cells * nb`.
    pub design: GpuTensor<R, F>,
    /// Shared log offset per cell.
    pub log_offset: GpuTensor<R, F>,
    /// Shared subject boundaries, length `k + 1`.
    pub subject_start: GpuTensor<R, u32>,
    /// Concatenated positive counts for every resident gene.
    pub counts: GpuTensor<R, F>,
    /// Cell index of each entry of `counts`.
    pub cells: GpuTensor<R, u32>,
    /// Start of each subject's run within each gene's block in `counts`,
    /// `[gene * (k + 1) + s]`.
    pub subject_ptr: GpuTensor<R, u32>,
    /// Count total per subject, `[s * n_genes + gene]`.
    pub subject_total: GpuTensor<R, F>,
    /// Unweighted mean design row per subject, `[s * nb + j]`.
    pub subject_mean: GpuTensor<R, F>,
    /// Design columns that vary within some subject, length `nb`, the live ones
    /// first and in increasing order.
    pub varying: GpuTensor<R, u32>,
    /// The gene each request is for.
    pub request_gene: GpuTensor<R, u32>,
    /// `alpha`, `lambda` and `gamma` per request, `[i * n_req + q]`.
    pub request_params: GpuTensor<R, F>,
    /// Starting fixed effects, `[j * n_req + q]`.
    pub beta_init: GpuTensor<R, F>,
    /// Starting random effects on the log scale, `[s * n_req + q]`.
    pub log_w_init: GpuTensor<R, F>,
    /// nebula's absolute stopping tolerance, then the relative resolution
    /// floor. Two elements.
    pub tolerance: GpuTensor<R, F>,
    /// Per-subject working store, `[(slot * k + s) * n_req + q]`.
    pub subject_scratch: GpuTensor<R, F>,
    /// Cross block of the information, `[(s * nb + j) * n_req + q]`.
    pub vwb_scratch: GpuTensor<R, F>,
    /// Everything a request brings back, `[row * n_req + q]`: [`OUT_HEADER`]
    /// rows, then `nb` fixed effects, the `nb * nb` Schur complement, the `k`
    /// random effects, the `k` random-effect curvatures and the `k * nb` cross
    /// block.
    pub out: GpuTensor<R, F>,
}
