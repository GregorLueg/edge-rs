//! The numerical support layer: everything edgePython reaches into `scipy` for.
//!
//! Special functions, distribution tails, optimisers and interpolation. All
//! `f64`, whatever the `EdgeFloat` of the caller's data.

pub mod bobyqa;
pub mod dist;
pub mod gamma;
pub mod interpolate;
pub mod lbfgsb;
pub mod optimise;
pub mod stats;
