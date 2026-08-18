//! The types most modules want. Prefer `use crate::prelude::*;` over deep
//! imports in new code.

pub use crate::errors::EdgeErrors;
pub use crate::utils::recycled::{Recycled, RecycledRow};
pub use crate::utils::simd::EdgeSimd;
pub use crate::utils::sparse::{CompressedSparse, SparseFormat};
pub use crate::utils::traits::EdgeFloat;
