//! The CubeCL NEBULA path. Feature-gated on `gpu`.
//!
//! Everything is `f32` on the device (wgpu exposes no `f64`), so this path has
//! its own measured tolerance instead of the `1e-6` CPU parity gate. See
//! [`crate::gpu::pml_kernel`] for where the precision goes.
//!
//! ### Mapping
//!
//! One plane is one fit; see [`crate::gpu::pml_kernel`]. Planes share no state
//! and need no barrier.
//!
//! The design, log offsets and subject boundaries are identical for every
//! gene. They upload once and every plane reads the same cell at the same
//! step, so the traffic is a cache broadcast, not one copy per fit.

pub mod nebula_gpu;
pub mod pml_kernel;
pub mod stage_two;
