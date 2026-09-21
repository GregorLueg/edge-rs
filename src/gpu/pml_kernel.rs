//! NEBULA's penalised maximum likelihood inner solver, one gene per thread.
//!
//! A port of [`crate::sc::pml`]'s Newton loop to CubeCL. The CPU module is the
//! reference for every formula; this file changes where the work happens, what
//! precision it happens in, and — because of that precision — how three of the
//! quantities are arranged.
//!
//! ### Mapping
//!
//! One thread owns one gene and runs the entire Newton loop, including the
//! backtracking, without touching any other thread's memory. The loop-carried
//! state is per gene, so no reduction is shared and no summation is
//! reassociated between threads. The design, the log offsets and the subject
//! boundaries are gene-independent, so all threads in a plane read the same
//! cell at the same step and the traffic is served once from cache.
//!
//! ### Precision: the part better summation cannot fix
//!
//! wgpu exposes no `f64`, so the device arithmetic is `f32`. The obvious
//! defence is compensated summation, and on this backend it **does not work**.
//! Kahan and full double-single accumulation both recover the rounding error of
//! an addition through a `two_sum`, and under exact algebra that error is
//! identically zero, so a compiler permitted to reassociate folds the whole
//! thing away. Measured on wgpu/Metal: scaling the recovered low part by a
//! thousand changed not one digit of any output, and routing it through a
//! bit-level round-trip to make it opaque did not help either. Compensated
//! summation here is a silent no-op that costs instructions.
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
//!   cell, so it is accumulated by Welford's weighted online update instead:
//!   one pass, and stable for the same reason the two-pass form is.
//!
//! What is left is the ordinary `f32` rounding of one `exp` and one `ln` per
//! cell, which nothing can undo, plus the error growth of a long naive sum. The
//! second is handled by [`SUM_BLOCK`]: summing in blocks is a pure change of
//! association with no algebraic identity for the optimiser to exploit, so
//! unlike compensation it survives.
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
use crate::gpu::GENE_WORKGROUP;

////////////
// Consts //
////////////

/// Magnitude past which a Newton component is treated as unbounded rather than
/// damped. `STEP_CUTOFF` in [`crate::sc::pml`].
const STEP_CUTOFF: f32 = 40.0;

/// Largest gradient component still called a critical point, nebula's `convd`.
const GRADIENT_TOLERANCE: f32 = 0.01;

/// Cells summed into a block accumulator before it is flushed into the total.
///
/// Two-level summation, which turns the `n * eps` error growth of a naive sum
/// into roughly `(n / SUM_BLOCK + SUM_BLOCK) * eps`. At a million cells that is
/// the difference between a few per cent and a few parts in `1e6`. Unlike
/// compensated summation it is only a change of association, so there is no
/// algebraic identity for the shader compiler to fold away; see the module doc.
const SUM_BLOCK: u32 = 256;

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

/// Fits one gene per thread by penalised maximum likelihood.
///
/// Mirrors `optimise` in [`crate::sc::pml`] with the gamma penalty, NEBULA's
/// NBGMM, at Laplace order one. Higher orders are left to the host, which still
/// has the `f64` path for them.
///
/// ### Params
///
/// * `design` - Shared design, row-major `n_cells * nb`
/// * `log_offset` - Shared log offset per cell, length `n_cells`
/// * `subject_start` - Shared subject boundaries, length `k + 1`
/// * `counts` - Every gene's positive counts, concatenated
/// * `cells` - Cell index of each entry of `counts`
/// * `gene_ptr` - Start of each gene's block in `counts`, length `n_genes + 1`
/// * `subject_total` - Count total per subject, `[s * n_genes + g]`
/// * `gene_params` - Three per gene: the gamma prior's `alpha` and `lambda`,
///   then the cell-level size `gamma`
/// * `beta_init` - Starting fixed effects, `[j * n_genes + g]`
/// * `tolerance` - Two elements: nebula's absolute stopping tolerance, then the
///   resolution floor relative to the objective ([`F32_NOISE_SCALE`]). A buffer
///   rather than scalar arguments because a runtime float scalar would need a
///   `ScalarArgSettings` bound this module otherwise has no use for
/// * `subject_scratch` - Per-subject working store,
///   `[(slot * k + s) * n_genes + g]`, [`SUBJECT_SLOTS`] slots
/// * `vwb_scratch` - Cross block of the information, `[(s * nb + j) * n_genes + g]`
/// * `out_beta` - Fitted fixed effects, `[j * n_genes + g]`
/// * `out_information` - Schur complement, `[(i * nb + j) * n_genes + g]`
/// * `out_scalars` - Four per gene: the log-likelihood, the previous one, the
///   log-determinant, and the final improvement
/// * `out_counts` - Two per gene: Newton steps taken, then backtracks used
/// * `n_genes` - Genes in the batch, which is the thread count
/// * `k` - Subjects
/// * `nb` - Design width
/// * `max_iter` - Newton budget
/// * `max_backtrack` - Backtracking budget within one step
/// * `nb_cap` - Comptime capacity of the `n_beta`-sized register arrays
///
/// ### Grid mapping
///
/// * `CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X` -> block of [`GENE_WORKGROUP`] genes
/// * `UNIT_POS_X` -> gene within the block
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn opt_pml_gpu<F: Float + CubeElement>(
    design: &Tensor<F>,
    log_offset: &Tensor<F>,
    subject_start: &Tensor<u32>,
    counts: &Tensor<F>,
    cells: &Tensor<u32>,
    gene_ptr: &Tensor<u32>,
    subject_total: &Tensor<F>,
    request_gene: &Tensor<u32>,
    request_params: &Tensor<F>,
    beta_init: &Tensor<F>,
    tolerance: &Tensor<F>,
    subject_scratch: &mut Tensor<F>,
    vwb_scratch: &mut Tensor<F>,
    out_beta: &mut Tensor<F>,
    out_information: &mut Tensor<F>,
    out_scalars: &mut Tensor<F>,
    out_counts: &mut Tensor<u32>,
    n_genes: u32,
    n_req: u32,
    k: u32,
    nb: u32,
    max_iter: u32,
    max_backtrack: u32,
    #[comptime] nb_cap: u32,
) {
    let q = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) * GENE_WORKGROUP + UNIT_POS_X;
    if q >= n_req {
        terminate!();
    }
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

    let ptr_lo = gene_ptr[gene as usize];
    let ptr_hi = gene_ptr[(gene + 1u32) as usize];

    let mut beta = Array::<F>::new(nb_cap as usize);
    let mut new_beta = Array::<F>::new(nb_cap as usize);
    let mut step_beta = Array::<F>::new(nb_cap as usize);
    let mut damp_beta = Array::<F>::new(nb_cap as usize);
    let mut db = Array::<F>::new(nb_cap as usize);
    let mut db_block = Array::<F>::new(nb_cap as usize);
    let mut a_bar = Array::<F>::new(nb_cap as usize);
    let mut vb2 = Array::<F>::new((nb_cap * nb_cap) as usize);

    let mut j = 0u32;
    while j < nb {
        beta[j as usize] = beta_init[(j * n_req + q) as usize];
        j += 1u32;
    }
    let mut s = 0u32;
    while s < k {
        subject_scratch[((SLOT_LOG_W * k + s) * n_req + q) as usize] = zero;
        s += 1u32;
    }

    let mut ll = evaluate_pass::<F>(
        design,
        log_offset,
        subject_start,
        counts,
        cells,
        subject_total,
        subject_scratch,
        &beta,
        SLOT_LOG_W,
        alpha,
        lambda,
        gamma,
        ptr_lo,
        ptr_hi,
        n_genes,
        n_req,
        k,
        q,
        gene,
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
            db_block[j as usize] = zero;
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

        let mut ptr = ptr_lo;
        s = 0u32;
        while s < k {
            let log_w_s = subject_scratch[((SLOT_LOG_W * k + s) * n_req + q) as usize];
            let w_s = F::exp(log_w_s);
            subject_scratch[((SLOT_W * k + s) * n_req + q) as usize] = w_s;

            let begin = subject_start[s as usize];
            let end = subject_start[(s + 1u32) as usize];
            // -- One pass: the residual gradient, and the weighted
            //    within-subject covariance by Welford's online update. The
            //    centred form needs the centre, and computing the centre first
            //    would mean a second pass over the block and a second `exp` per
            //    cell; the online update carries the centre along instead and
            //    is cancellation-free for the same reason the two-pass form is.
            let mut resid = zero;
            let mut resid_block = zero;

            let mut p_sum = zero;
            j = 0u32;
            while j < nb {
                a_bar[j as usize] = zero;
                j += 1u32;
            }
            let mut r = begin;
            let mut since_flush = 0u32;
            while r < end {
                let mut eta = log_offset[r as usize];
                j = 0u32;
                while j < nb {
                    eta += design[(r * nb + j) as usize] * beta[j as usize];
                    j += 1u32;
                }
                let extb = F::exp(eta + log_w_s);

                let mut y = zero;
                if ptr < ptr_hi {
                    if cells[ptr as usize] == r {
                        y = counts[ptr as usize];
                        ptr += 1u32;
                    }
                }

                // `phi` of the CPU's `gradient_only`, then of its `curvature`.
                let phi_g = (gamma + y) / (one + gamma / extb);
                let phi_c = phi_g / (extb + gamma);

                // Residual form: the difference is taken per cell, not between
                // two sums of order `1e4`. See the module doc.
                let d = y - phi_g;
                resid_block += d;
                j = 0u32;
                while j < nb {
                    db_block[j as usize] += design[(r * nb + j) as usize] * d;
                    j += 1u32;
                }

                // Welford, weighted: the deltas are taken against the running
                // centre, then the centre moves.
                let p_next = p_sum + phi_c;
                if p_next > zero {
                    let scale = gamma * phi_c * p_sum / p_next;
                    let mut a = 0u32;
                    while a < nb {
                        let da = design[(r * nb + a) as usize] - a_bar[a as usize];
                        let mut b = a;
                        while b < nb {
                            let db_delta = design[(r * nb + b) as usize] - a_bar[b as usize];
                            vb2[(a * nb + b) as usize] += scale * da * db_delta;
                            b += 1u32;
                        }
                        a += 1u32;
                    }
                    let step_frac = phi_c / p_next;
                    j = 0u32;
                    while j < nb {
                        let centre = a_bar[j as usize];
                        a_bar[j as usize] =
                            centre + step_frac * (design[(r * nb + j) as usize] - centre);
                        j += 1u32;
                    }
                }
                p_sum = p_next;

                since_flush += 1u32;
                if since_flush == SUM_BLOCK {
                    resid += resid_block;
                    resid_block = zero;
                    j = 0u32;
                    while j < nb {
                        db[j as usize] += db_block[j as usize];
                        db_block[j as usize] = zero;
                        j += 1u32;
                    }
                    since_flush = 0u32;
                }
                r += 1u32;
            }
            resid += resid_block;
            j = 0u32;
            while j < nb {
                db[j as usize] += db_block[j as usize];
                db_block[j as usize] = zero;
                j += 1u32;
            }

            let dw_s = resid + (alpha - lambda * w_s);
            subject_scratch[((SLOT_DW * k + s) * n_req + q) as usize] = dw_s;
            let vw_s = gamma * p_sum + lambda * w_s;
            subject_scratch[((SLOT_VW * k + s) * n_req + q) as usize] = vw_s;

            j = 0u32;
            while j < nb {
                vwb_scratch[((s * nb + j) * n_req + q) as usize] =
                    gamma * a_bar[j as usize] * p_sum;
                j += 1u32;
            }

            let shrink = lambda * w_s / vw_s;
            // What the centring leaves over, which the prior's share of the
            // subject curvature keeps from cancelling.
            let mut a = 0u32;
            while a < nb {
                let mut b = a;
                while b < nb {
                    vb2[(a * nb + b) as usize] +=
                        gamma * a_bar[a as usize] * a_bar[b as usize] * p_sum * shrink;
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

            // The solve destroys its matrix. That is safe here: this sweep is
            // not the final one, so `vb2` is rebuilt before it is read out.
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
                subject_total,
                subject_scratch,
                &new_beta,
                SLOT_NEW_LOG_W,
                alpha,
                lambda,
                gamma,
                ptr_lo,
                ptr_hi,
                n_genes,
                n_req,
                k,
                q,
                gene,
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
                        subject_total,
                        subject_scratch,
                        &new_beta,
                        SLOT_NEW_LOG_W,
                        alpha,
                        lambda,
                        gamma,
                        ptr_lo,
                        ptr_hi,
                        n_genes,
                        n_req,
                        k,
                        q,
                        gene,
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

    j = 0u32;
    while j < nb {
        out_beta[(j * n_req + q) as usize] = beta[j as usize];
        j += 1u32;
    }
    let mut idx = 0u32;
    while idx < nb * nb {
        out_information[(idx * n_req + q) as usize] = vb2[idx as usize];
        idx += 1u32;
    }
    out_scalars[q as usize] = ll;
    out_scalars[(n_req + q) as usize] = ll_prev;
    out_scalars[(2u32 * n_req + q) as usize] = log_det;
    out_scalars[(3u32 * n_req + q) as usize] = likdif;
    out_counts[q as usize] = step;
    out_counts[(n_req + q) as usize] = backtracks;
}

/////////////////
// Cell passes //
/////////////////

/// The penalised log-likelihood at `(beta, log_w)`.
///
/// Accumulates in the four stages the CPU's `Workspace::evaluate` uses and in
/// the same per-stage order: the linear term over the positive counts, the
/// subject term, `-gamma` times the sum of `log(extb + gamma)` over all cells,
/// and the count-weighted sum of the same logs over the positive counts. Each
/// runs through a block accumulator; see [`SUM_BLOCK`].
///
/// ### Params
///
/// * `design` - Shared design, row-major
/// * `log_offset` - Shared log offset per cell
/// * `subject_start` - Shared subject boundaries
/// * `counts` - Concatenated positive counts
/// * `cells` - Cell index of each count
/// * `subject_total` - Count total per subject, gene-minor
/// * `subject_scratch` - Per-subject store holding the random effects
/// * `beta` - Fixed effects to evaluate at
/// * `slot_log_w` - Which scratch slot holds the random effects
/// * `alpha` - Shape of the gamma prior
/// * `lambda` - Rate of the gamma prior
/// * `gamma` - Cell-level negative binomial size
/// * `ptr_lo` - Start of this gene's block in `counts`
/// * `ptr_hi` - End of this gene's block in `counts`
/// * `n_genes` - Genes resident on the device, the stride of `subject_total`
/// * `n_req` - Requests in the launch, the stride of `subject_scratch`
/// * `k` - Subjects
/// * `q` - This thread's request
/// * `gene` - The gene the request is for
/// * `nb` - Design width
///
/// ### Returns
///
/// The penalised log-likelihood.
#[cube]
#[allow(clippy::too_many_arguments)]
fn evaluate_pass<F: Float>(
    design: &Tensor<F>,
    log_offset: &Tensor<F>,
    subject_start: &Tensor<u32>,
    counts: &Tensor<F>,
    cells: &Tensor<u32>,
    subject_total: &Tensor<F>,
    subject_scratch: &Tensor<F>,
    beta: &Array<F>,
    slot_log_w: u32,
    alpha: F,
    lambda: F,
    gamma: F,
    ptr_lo: u32,
    ptr_hi: u32,
    n_genes: u32,
    n_req: u32,
    k: u32,
    q: u32,
    gene: u32,
    nb: u32,
) -> F {
    let zero = F::new(0.0_f32);

    // Stage one: `sum y_i (offset_i + x_i beta)` over the positive counts, with
    // no random effect. The CPU adds the random effect to the linear predictor
    // only after this term.
    let mut acc = zero;
    let mut block = zero;
    let mut since_flush = 0u32;
    let mut p = ptr_lo;
    while p < ptr_hi {
        let c = cells[p as usize];
        let mut eta = log_offset[c as usize];
        let mut j = 0u32;
        while j < nb {
            eta += design[(c * nb + j) as usize] * beta[j as usize];
            j += 1u32;
        }
        block += eta * counts[p as usize];
        since_flush += 1u32;
        if since_flush == SUM_BLOCK {
            acc += block;
            block = zero;
            since_flush = 0u32;
        }
        p += 1u32;
    }
    acc += block;

    // Stage two: the random effect against each subject's count total.
    let mut s = 0u32;
    while s < k {
        let log_w_s = subject_scratch[((slot_log_w * k + s) * n_req + q) as usize];
        acc += log_w_s * subject_total[(s * n_genes + gene) as usize];
        s += 1u32;
    }

    // Stages three and four share the pass over cells: the unweighted sum of
    // `log(extb + gamma)` and the count-weighted one, kept apart so each keeps
    // the CPU's own order.
    let mut phil = zero;
    let mut phil_block = zero;
    let mut weighted = zero;
    let mut weighted_block = zero;
    let mut sum_log_w = zero;
    let mut sum_w = zero;

    let mut ptr = ptr_lo;
    since_flush = 0u32;
    s = 0u32;
    while s < k {
        let log_w_s = subject_scratch[((slot_log_w * k + s) * n_req + q) as usize];
        sum_log_w += log_w_s;
        sum_w += F::exp(log_w_s);

        let begin = subject_start[s as usize];
        let end = subject_start[(s + 1u32) as usize];
        let mut r = begin;
        while r < end {
            let mut eta = log_offset[r as usize];
            let mut j = 0u32;
            while j < nb {
                eta += design[(r * nb + j) as usize] * beta[j as usize];
                j += 1u32;
            }
            let value = F::ln(F::exp(eta + log_w_s) + gamma);
            phil_block += value;

            if ptr < ptr_hi {
                if cells[ptr as usize] == r {
                    weighted_block += counts[ptr as usize] * value;
                    ptr += 1u32;
                }
            }

            since_flush += 1u32;
            if since_flush == SUM_BLOCK {
                phil += phil_block;
                weighted += weighted_block;
                phil_block = zero;
                weighted_block = zero;
                since_flush = 0u32;
            }
            r += 1u32;
        }
        s += 1u32;
    }
    phil += phil_block;
    weighted += weighted_block;

    acc -= gamma * phil;
    acc -= weighted;
    acc + (alpha * sum_log_w - lambda * sum_w)
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
    let blocks = (n_req as u32).div_ceil(GENE_WORKGROUP);
    let (gx, gy) = grid_2d(blocks, &limits).map_err(|e| EdgeErrors::Gpu(e.to_string()))?;
    let count = checked_cube_count("opt_pml_gpu", gx, gy, 1, &limits)
        .map_err(|e| EdgeErrors::Gpu(e.to_string()))?;

    macro_rules! dispatch {
        ($cap:expr) => {
            unsafe {
                opt_pml_gpu::launch_unchecked::<F, R>(
                    client,
                    count,
                    CubeDim::new_1d(GENE_WORKGROUP),
                    tensors.design.clone().into_tensor_arg(),
                    tensors.log_offset.clone().into_tensor_arg(),
                    tensors.subject_start.clone().into_tensor_arg(),
                    tensors.counts.clone().into_tensor_arg(),
                    tensors.cells.clone().into_tensor_arg(),
                    tensors.gene_ptr.clone().into_tensor_arg(),
                    tensors.subject_total.clone().into_tensor_arg(),
                    tensors.request_gene.clone().into_tensor_arg(),
                    tensors.request_params.clone().into_tensor_arg(),
                    tensors.beta_init.clone().into_tensor_arg(),
                    tensors.tolerance.clone().into_tensor_arg(),
                    tensors.subject_scratch.clone().into_tensor_arg(),
                    tensors.vwb_scratch.clone().into_tensor_arg(),
                    tensors.out_beta.clone().into_tensor_arg(),
                    tensors.out_information.clone().into_tensor_arg(),
                    tensors.out_scalars.clone().into_tensor_arg(),
                    tensors.out_counts.clone().into_tensor_arg(),
                    n_genes as u32,
                    n_req as u32,
                    k as u32,
                    nb as u32,
                    max_iter,
                    max_backtrack,
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
    /// Start of each gene's block in `counts`, length `n_genes + 1`.
    pub gene_ptr: GpuTensor<R, u32>,
    /// Count total per subject, `[s * n_genes + gene]`.
    pub subject_total: GpuTensor<R, F>,
    /// The gene each request is for.
    pub request_gene: GpuTensor<R, u32>,
    /// `alpha`, `lambda` and `gamma` per request, `[i * n_req + q]`.
    pub request_params: GpuTensor<R, F>,
    /// Starting fixed effects, `[j * n_req + q]`.
    pub beta_init: GpuTensor<R, F>,
    /// nebula's absolute stopping tolerance, then the relative resolution
    /// floor. Two elements.
    pub tolerance: GpuTensor<R, F>,
    /// Per-subject working store, `[(slot * k + s) * n_req + q]`.
    pub subject_scratch: GpuTensor<R, F>,
    /// Cross block of the information, `[(s * nb + j) * n_req + q]`.
    pub vwb_scratch: GpuTensor<R, F>,
    /// Fitted fixed effects, `[j * n_req + q]`.
    pub out_beta: GpuTensor<R, F>,
    /// Schur complement, `[(i * nb + j) * n_req + q]`.
    pub out_information: GpuTensor<R, F>,
    /// Log-likelihood, previous log-likelihood, log-determinant and final
    /// improvement, `[i * n_req + q]`.
    pub out_scalars: GpuTensor<R, F>,
    /// Newton steps taken and backtracks used, `[i * n_req + q]`.
    pub out_counts: GpuTensor<R, u32>,
}
