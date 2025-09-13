//! Optimized GPU management and memory allocation

pub mod adaptive_memory;

#[cfg(feature = "cudarc")]
mod cuda_impl;
#[cfg(not(feature = "cudarc"))]
mod fallback;

#[cfg(feature = "cudarc")]
pub use cuda_impl::*;
#[cfg(not(feature = "cudarc"))]
pub use fallback::*;
