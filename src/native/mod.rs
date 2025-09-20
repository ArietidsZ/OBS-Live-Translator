//! Native FFI module for high-performance C++ components

pub mod engine_ffi;
pub mod simd_ffi;

pub use engine_ffi::{
    OptimizedEngine,
    OptimizedEngineBuilder,
};

pub use simd_ffi::{
    SimdAudioProcessor,
    OnnxEngine,
    WhisperOnnx,
};