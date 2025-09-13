//! Language bindings for Node.js integration

#[cfg(feature = "napi")]
pub mod node;

#[cfg(feature = "web")]
pub mod wasm;

// Re-export for convenience
#[cfg(feature = "napi")]
pub use node::*;

#[cfg(feature = "web")]
pub use wasm::*;