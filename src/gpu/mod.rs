//! The CubeCL NEBULA path. Feature-gated on `gpu`, CPU-only build otherwise.
//!
//! Everything here is `f32` on the device, because wgpu exposes no `f64` on any
//! backend. That is a deliberate departure from the crate's numeric policy and
//! the reason this path carries its own, looser, measured tolerance rather than
//! the `1e-6` parity gate the CPU path is held to. See the module doc of
//! [`crate::gpu::pml_kernel`] for where the precision goes and what is done
//! about it.
//!
//! ### Mapping
//!
//! One thread is one gene, and the whole sequential optimiser runs inside that
//! thread. The loop-carried state of NEBULA's Newton and simplex loops is
//! per gene, so genes are the parallel axis and nothing is reassociated
//! relative to the CPU order. Threads share no state, read no other thread's
//! memory and need no barrier.
//!
//! The design, the log offsets and the subject boundaries are the same buffers
//! for every gene, so they upload once and every thread in a plane reads the
//! same cell at the same step: a cache broadcast rather than one copy of the
//! traffic per gene. That is what makes the mapping affordable.

pub mod nebula_gpu;
pub mod pml_kernel;

////////////
// Consts //
////////////

/// Workgroup width for the per-gene kernels.
///
/// One thread is one gene and the threads share nothing, so the width is purely
/// an occupancy knob. Sixty-four is two Apple Silicon planes; narrowing to one
/// costs more in hidden memory latency than the idle lanes save.
pub const GENE_WORKGROUP: u32 = 64;

/// Largest design width the kernels are compiled for.
///
/// The per-thread `beta`-sized arrays are registers, so their capacity has to
/// be a compile-time constant. NEBULA designs are an intercept plus a handful
/// of covariates; eight is already generous and the `nb * nb` blocks grow
/// quadratically in it.
pub const MAX_BETA: u32 = 8;
