//! AI Models and Hardware Acceleration
//!
//! Contains implementations for:
//! - Google USM/Chirp speech recognition
//! - Meta MMS multilingual speech models
//! - Hardware acceleration for modern GPUs
//! - Native Rust inference engine using Burn ML framework

pub mod engine;
#[path = "usm-chirp.rs"]
pub mod usm_chirp;
#[path = "mms-multilingual.rs"]
pub mod mms_multilingual;
#[path = "nvidia-acceleration.rs"]
pub mod nvidia_acceleration;
#[path = "amd-optimization.rs"]
pub mod amd_optimization;
pub mod inference;

pub use engine::*;
pub use usm_chirp::*;
pub use mms_multilingual::*;
pub use nvidia_acceleration::*;
pub use amd_optimization::*;
pub use inference::*;