//! Real-time streaming and WebSocket server

pub mod websocket;
pub mod session;
pub mod optimized_websocket;
pub mod enhanced_session;
pub mod realtime_pipeline;
pub mod protocol_optimizer;
pub mod opus_codec;

pub use websocket::{WebSocketHandler, WebSocketMessage};
pub use session::{StreamingSession, SessionManager};
pub use optimized_websocket::{OptimizedWebSocketHandler, BinaryMessageType, OpusEncoder, AdaptiveJitterBuffer};
pub use enhanced_session::{EnhancedSessionManager, EnhancedStreamingSession, SessionManagerConfig};
pub use realtime_pipeline::{RealtimeStreamingPipeline, PipelineConfig};
pub use protocol_optimizer::{ProtocolOptimizer, FrameOptimizationConfig, CompressionCapabilities, OptimizedBinaryFrame, BinaryFrameType};
pub use opus_codec::{OpusEncoder as StreamingOpusEncoder, OpusDecoder, G722Codec, AudioCodecManager, OpusCodecConfig, G722CodecConfig, AudioCodec};

/// Streaming configuration
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    pub buffer_size: usize,
    pub max_latency_ms: u32,
    pub reconnect_attempts: u32,
    pub heartbeat_interval_ms: u32,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            buffer_size: 8192,
            max_latency_ms: 500,
            reconnect_attempts: 3,
            heartbeat_interval_ms: 30000,
        }
    }
}