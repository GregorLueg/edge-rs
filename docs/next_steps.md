# Next steps: the CPU side of GPU NEBULA

Hand-over for the agent taking the CPU work. Written 2026-09-22 at the end of the
GPU optimisation pass on `feat-gpu` (commits `bd9d5a5..6223bc1`).

## Where things stand

Forced-HL GPU NEBULA runs 2.35x to 4.1x faster than the 10-thread CPU on an M1
Max across six shapes. The measured table is in the module doc of
`src/gpu/stage_two.rs`. The run is now bound by CPU work on every shape except
wide designs (nb = 8). Read this before touching anything:

```sh
NEBULA_BENCH_GENES=500 NEBULA_BENCH_ONLY=hl,gpu_hl EDGE_RS_GPU_TIMING=1 \
  cargo bench --features gpu --bench nebula_bench
```

The `EDGE_RS_GPU_TIMING` report prints, per run: stage one, upload, gather,
solve (with staging / blocked on the device / scatter), the `f64` finish, tell,
stage three, the request count per round, and histograms of Newton steps on the
device and in the finish. Bench knobs: `NEBULA_BENCH_CELLS`, `_COEF`,
`_SUBJECTS`, `_INTERCEPT_SHIFT` (density), `_IMBALANCE` (largest subject over
smallest). The bench prints checksums and the drift against the CPU fit; the
CPU checksums must not move unless the change is meant to touch the CPU path.

Stage split at 4000 genes, 20000 cells, nb = 3, 46 per cent dense (seconds):

| total | stage one | host finish | blocked on device | stage three |
|---|---|---|---|---|
| 76 | 24 | 46 | 4 | 2 |

The two CPU items are 92 per cent of the run. Nothing below carries a
predicted saving; measure the small case first, report, then scale.

## Lever 1: the `f64` finish (`newton_finish`, `src/sc/pml.rs`)

Called once per device fit, in parallel over requests, from
`finish_at_argmax` in `src/gpu/stage_two.rs`. Two fused sweeps over every cell
of the gene: sweep one takes the `f64` gradient at the device's point (the
Hessian comes from the device on the first step), sweep two evaluates the
objective and the log-determinant at the stepped point. Over 99 per cent of
fits settle in one step.

What has been done: no per-cell workspace; logarithms taken on running
products of eight; the second sweep updates the first sweep's stored `exp` by a
fifth-order series instead of recomputing it; the loops over design columns are
width-specialised through a const generic.

What has been tried and failed on aarch64: the crate's `exp_in_place_simd` in a
blocked form of sweep one. 5.3 s to 7.9 s on S1, because Apple's scalar libm
`exp` beats `wide`'s polynomial there. Unmeasured on x86, where the picture may
invert; if you test it, gate on `detect_simd_level` and keep the scalar path.

Untried:

- Multiple accumulators. Every sum in both sweeps is a single dependency chain
  (`sum_log`, `resid`, `weight`, `linear`, `weighted_log`, and the `first` /
  `second` moment arrays). The `cpu-optimisation` skill covers the pattern.
- The dense sweep's `1.0 / t` divide per cell. Two of them in sweep one. A
  reciprocal approximation is not acceptable here (the value feeds the
  objective at `1e-7`), but the divide could be shared between the two sweeps
  where the series path is taken.
- Cells with `y = 0` and cells with `y > 0` are walked as one dense pass plus
  one sparse pass, so positive cells are visited twice. Merging the two into one
  pass with a pointer into the sparse run is what `opt_pml` does; measured 2.4x
  *slower* on the device because of scattered loads, unmeasured on the host.

Gate: `tests/e2e_nebula_gpu.rs` and the `gpu` test in `tests/e2e_nebula.rs`
with `EDGE_RS_TOL_REPORT=1`; the recorded worst needs are in the doc on
`GPU_TOLS`. Any change that moves the CPU checksum in the bench is a bug, since
`newton_finish` is only reachable from the GPU path.

## Lever 2: stage one (`plan_gene` -> `minimise_marginal`, `src/sc/nebula.rs`)

L-BFGS-B on the marginal likelihood, `ptmg_value_and_gradient` in
`src/sc/ptmg.rs`, tens to hundreds of evaluations per gene. Shared with the CPU
path, so it is held to the `1e-6` parity gate against nebula 1.5.8 in
`tests/e2e_nebula.rs`, and the bench's CPU checksums must not move.

Nothing has been done here. Start by timing one `ptmg` evaluation against the
count of evaluations per gene (`NEBULA_BENCH_ONLY=ptmg` gives the former; add a
counter for the latter) to know whether the cost is per evaluation or the
optimiser's evaluation count. The L-BFGS-B tolerances (`STAGE_ONE_FTOL` and
friends at the top of `nebula.rs`) are set tighter than nebula's on purpose;
the doc on `STAGE_ONE_FTOL` says why, and loosening them is a parity question
for Gregor, not a performance knob.

Inside `ptmg.rs`, the evaluation is a scan over cells and a sum over subjects
with several `exp` and `ln` per cell. The same single-accumulator observation as
lever 1 applies. Moving stage one to the device is out of scope: it has an
exact gradient the device's `f32` cannot reproduce to the parity gate.

## Lever 3: rayon and thread contention

The finish runs under `par_iter().map_init(..)` over requests while the device
runs the other cohort. `ResidentBatch::collect` blocks one thread on
`block_on`. Not measured: whether that blocked thread is a rayon worker (it is
called from the main thread, so it should not be), and whether the finish's
per-request allocation of `db`, `vw`, `vwb`, `vb`, `first`, `second`, `factor`
and the two step vectors is visible. They are `O(k * nb)`, so probably not, but
`map_init` already provides the per-thread slot to hoist them into.

## Not on the list

- Stage three is 2 of 76 s.
- The device. Wide designs are bound by launch latency (one fit's serial walk
  over the cells), and the fix there is more lanes per fit, which is GPU work.
- Compensated summation on the device (folded by fast-math), searching on
  device values (drives `sigma^2` onto its bound), warm-starting from the
  previous fit's optimum (24 to 39 per cent more evaluations). All in the
  module docs of `src/gpu/`.

## Rules

- One lever per commit, before/after numbers on at least S1 (20000 cells,
  `NEBULA_BENCH_INTERCEPT_SHIFT=-2.5`) and S2 (100000 cells, shift `-3.2`) in
  the commit's report. Never two benches at once.
- `cargo clippy --all-targets --features gpu -- -D warnings` and the same
  without the feature; `cargo test --release --features gpu`.
- `cargo fmt --check` fails on seven files that predate this work. Leave them.
- Work on the branch, never push, no version bumps, no README or NEWS edits.
  Report the branch from `git rev-parse --abbrev-ref HEAD`, the commit count,
  and the merge line.
