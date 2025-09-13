//! Optimized GPU management and memory allocation

pub mod adaptive_memory;
pub mod hardware_detection;
pub mod memory_pool;

#[cfg(feature = "cudarc")]
mod cuda_impl;
#[cfg(not(feature = "cudarc"))]
mod fallback;

#[cfg(feature = "cudarc")]
pub use cuda_impl::*;
#[cfg(not(feature = "cudarc"))]
pub use fallback::*;
