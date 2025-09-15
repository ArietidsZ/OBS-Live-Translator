//! AI Models and Hardware Acceleration
//!
//! Contains implementations for:
//! - Google USM/Chirp speech recognition
//! - Meta MMS multilingual speech models
//! - Hardware acceleration for modern GPUs
//! - Native Rust inference engine using Burn ML framework

pub mod engine;
pub mod usm_chirp;
pub mod mms_multilingual;
pub mod nvidia_acceleration;
pub mod amd_optimization;
pub mod inference;

pub use engine::*;
pub use usm_chirp::*;
pub use mms_multilingual::*;
pub use nvidia_acceleration::*;
pub use amd_optimization::*;
pub use inference::*;