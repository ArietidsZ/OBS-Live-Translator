//! High-performance parallel ASR inference engine

#[cfg(all(feature = "candle-core", feature = "candle-nn", feature = "candle-transformers"))]
mod candle_impl;
#[cfg(not(all(feature = "candle-core", feature = "candle-nn", feature = "candle-transformers")))]
mod fallback;

#[cfg(all(feature = "candle-core", feature = "candle-nn", feature = "candle-transformers"))]
pub use candle_impl::*;
#[cfg(not(all(feature = "candle-core", feature = "candle-nn", feature = "candle-transformers")))]
pub use fallback::*;
