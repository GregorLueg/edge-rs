//! The numerical support layer: everything edgePython reaches into `scipy` for.
//!
//! Special functions, distribution tails, optimisers, interpolation and
//! quadrature. Per the crate numeric policy these all work in `f64` regardless
//! of the `EdgeFloat` the caller's data is held in: they sit inside likelihood
//! evaluation, where the arithmetic is a difference of large logs and `f32`
//! loses parity with edgeR long before it saves anything.

pub mod dist;
pub mod gamma;
pub mod interpolate;
pub mod lbfgsb;
pub mod optimise;
pub mod quadrature;
pub mod stats;
