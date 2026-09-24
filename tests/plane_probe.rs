//! Probes how the adapter lays out planes, with the launch shape the NEBULA
//! kernel uses. Diagnostic for the macos-15-intel runner, which reports planes
//! of 4 to 64 lanes and returns wrong NEBULA fits.
//!
//! ```text
//! cargo test --release --features gpu-tests --test plane_probe -- --nocapture
//! ```

#![cfg(feature = "gpu-tests")]

mod common;

use std::collections::BTreeMap;

use cubecl::prelude::*;
use cubecl::wgpu::WgpuRuntime;
use cubecl_utils_rs::prelude::*;

const FIELDS: u32 = 9;
const UNSET: u32 = u32::MAX;
/// Loop bound for the strided pass, prime so no plane width divides it.
const SPAN: u32 = 37;

#[cube(launch_unchecked)]
fn probe(out: &mut Tensor<u32>, fsum: &mut Tensor<f32>, n_req: u32) {
    let q = CUBE_POS_X * (CUBE_DIM_X / PLANE_DIM) + PLANE_POS;
    if q >= n_req {
        terminate!();
    }
    let u = CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X;
    let base = (u * FIELDS) as usize;
    out[base] = PLANE_DIM;
    out[base + 1] = PLANE_POS;
    out[base + 2] = UNIT_POS_PLANE;
    out[base + 3] = UNIT_POS_X;
    out[base + 4] = plane_sum(1u32);
    out[base + 5] = plane_sum(UNIT_POS_X);
    out[base + 6] = plane_broadcast(UNIT_POS_X, 0u32);
    out[base + 7] = q;

    // Same shape as the NEBULA passes over cells: a strided loop whose trip
    // count differs by lane, then a plane reduction. Sums to 0 + 1 + .. + 36.
    let mut acc = 0.0f32;
    let mut i = UNIT_POS_PLANE;
    while i < SPAN {
        acc += f32::cast_from(i);
        i += PLANE_DIM;
    }
    fsum[u as usize] = plane_sum(acc);
    out[base + 8] = plane_sum(1u32);
}

#[cube(launch_unchecked)]
fn raw(out: &mut Tensor<u32>) {
    let base = (UNIT_POS_X * 5u32) as usize;
    out[base] = PLANE_DIM;
    out[base + 1] = PLANE_POS;
    out[base + 2] = UNIT_POS_PLANE;
    out[base + 3] = plane_sum(1u32);
    out[base + 4] = plane_broadcast(UNIT_POS_X, 0u32);
}

/// Dumps the plane builtins of one cube, before any early exit can hide them.
fn dump_raw(client: &ComputeClient<WgpuRuntime>, width: u32) {
    let n = (width * 5) as usize;
    let out = GpuTensor::<WgpuRuntime, u32>::from_slice(&vec![UNSET; n], vec![n], client).unwrap();
    unsafe {
        raw::launch_unchecked::<WgpuRuntime>(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(width),
            out.into_tensor_arg(),
        );
    }
    let out = out.read(client).unwrap();
    println!(
        "  raw builtins, width {width} (unit: PLANE_DIM PLANE_POS UNIT_POS_PLANE plane_sum(1) broadcast(lane 0)):"
    );
    for u in 0..width as usize {
        let r = &out[u * 5..u * 5 + 5];
        if u < 4 || u % 16 == 0 || u + 1 == width as usize {
            println!("    {u:>3}: {r:?}");
        }
    }
}

/// One launch; returns the number of invariant violations and prints a summary.
fn run(client: &ComputeClient<WgpuRuntime>, width: u32, cubes: u32, n_req: u32) -> usize {
    let n_units = (width * cubes) as usize;
    let out = GpuTensor::<WgpuRuntime, u32>::from_slice(
        &vec![UNSET; n_units * FIELDS as usize],
        vec![n_units * FIELDS as usize],
        client,
    )
    .unwrap();
    let fsum =
        GpuTensor::<WgpuRuntime, f32>::from_slice(&vec![f32::NAN; n_units], vec![n_units], client)
            .unwrap();
    unsafe {
        probe::launch_unchecked::<WgpuRuntime>(
            client,
            CubeCount::Static(cubes, 1, 1),
            CubeDim::new_1d(width),
            out.into_tensor_arg(),
            fsum.into_tensor_arg(),
            n_req,
        );
    }
    let out = out.read(client).unwrap();
    let fsum = fsum.read(client).unwrap();

    let want_fsum: f32 = (0..SPAN).map(|i| i as f32).sum();
    let mut bad = 0usize;
    let mut dims = BTreeMap::<u32, usize>::new();
    let mut reqs = BTreeMap::<u32, Vec<usize>>::new();
    let mut skipped = 0usize;

    for c in 0..cubes as usize {
        // Group this cube's live units by the plane id the hardware reports.
        let mut planes = BTreeMap::<u32, Vec<usize>>::new();
        for x in 0..width as usize {
            let u = c * width as usize + x;
            let row = &out[u * FIELDS as usize..(u + 1) * FIELDS as usize];
            if row[0] == UNSET {
                skipped += 1;
                continue;
            }
            *dims.entry(row[0]).or_default() += 1;
            planes.entry(row[1]).or_default().push(u);
            reqs.entry(row[7]).or_default().push(u);
        }
        for (pos, units) in &planes {
            let rows: Vec<&[u32]> = units
                .iter()
                .map(|&u| &out[u * FIELDS as usize..(u + 1) * FIELDS as usize])
                .collect();
            let dim = rows[0][0];
            let mut lanes: Vec<u32> = rows.iter().map(|r| r[2]).collect();
            lanes.sort_unstable();
            let unit_sum: u32 = rows.iter().map(|r| r[3]).sum();
            let lane0 = rows.iter().find(|r| r[2] == 0).map(|r| r[3]);
            let mut problems = Vec::new();
            if rows.iter().any(|r| r[0] != dim) {
                problems.push("PLANE_DIM differs within the plane".to_string());
            }
            if units.len() as u32 != dim {
                problems.push(format!(
                    "{} units carry this PLANE_POS, PLANE_DIM says {dim}",
                    units.len()
                ));
            }
            if lanes != (0..dim).collect::<Vec<_>>() {
                problems.push(format!("UNIT_POS_PLANE is not 0..{dim}: {lanes:?}"));
            }
            if rows.iter().any(|r| r[4] != dim || r[8] != dim) {
                let s: Vec<_> = rows.iter().map(|r| (r[4], r[8])).collect();
                problems.push(format!("plane_sum(1) != PLANE_DIM: {s:?}"));
            }
            if rows.iter().any(|r| r[5] != unit_sum) {
                problems
                    .push("plane_sum(UNIT_POS_X) != sum over units sharing PLANE_POS".to_string());
            }
            if rows.iter().any(|r| Some(r[6]) != lane0) {
                problems.push("plane_broadcast from lane 0 disagrees".to_string());
            }
            let f: Vec<f32> = units.iter().map(|&u| fsum[u]).collect();
            if f.iter().any(|v| v.to_bits() != want_fsum.to_bits()) {
                problems.push(format!(
                    "strided plane_sum {:?}, want {want_fsum}",
                    dedup(&f)
                ));
            }
            if !problems.is_empty() {
                bad += 1;
                if bad <= 5 {
                    let xs: Vec<u32> = rows.iter().map(|r| r[3]).collect();
                    println!("    cube {c} plane {pos} units {xs:?}");
                    for p in problems {
                        println!("      {p}");
                    }
                }
            }
        }
    }
    let shared: Vec<_> = reqs
        .iter()
        .filter(|(_, u)| {
            let first = u[0] / width as usize;
            u.iter().any(|&x| x / width as usize != first)
        })
        .map(|(q, _)| *q)
        .collect();
    if !shared.is_empty() {
        bad += 1;
        println!("    requests spread over several cubes: {shared:?}");
    }
    let covered = reqs.len() as u32;
    println!(
        "  width {width:>3} cubes {cubes} n_req {n_req:>4}: PLANE_DIM counts {dims:?}, {covered} requests live (want {n_req}), {skipped} units exited, {bad} bad planes"
    );
    if covered != n_req {
        bad += 1;
        let missing: Vec<u32> = (0..n_req).filter(|q| !reqs.contains_key(q)).collect();
        println!("    requests never run: {missing:?}");
    }
    bad
}

fn dedup(v: &[f32]) -> Vec<f32> {
    let mut out: Vec<f32> = Vec::new();
    for &x in v {
        if !out.iter().any(|y| y.to_bits() == x.to_bits()) {
            out.push(x);
        }
    }
    out
}

#[test]
fn plane_layout_probe() {
    let client = common::gpu_client();
    let limits = GpuLimits::from_client(&client);
    println!(
        "\nplanes {} to {}, max units per cube {}",
        limits.plane_size_min, limits.plane_size_max, limits.max_units_per_cube
    );

    let mut total = 0usize;
    for width in [128, 64, 32] {
        dump_raw(&client, width);
    }

    // The NEBULA launch: a cube of two widest planes, one cube per two requests.
    let width = 2 * limits.plane_size_max;
    for n_req in [1u32, 3, 5, 64, 200] {
        total += run(&client, width, n_req.div_ceil(2), n_req);
    }
    assert_eq!(
        total, 0,
        "plane layout violates what the NEBULA kernel assumes"
    );
}
