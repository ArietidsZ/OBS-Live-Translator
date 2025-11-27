//! Streaming Protocol Optimization
//!
//! This module provides binary frame optimization, compression negotiation,
//! and frame size optimization for real-time streaming.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use tokio::sync::{Mutex, RwLock};
use tokio::time::{Duration, Instant};
use tracing::{debug, info, warn};

/// Frame size optimization constants
pub const OPTIMAL_FRAME_SIZE_16KHZ: usize = 1024; // 1024 samples at 16kHz = 64ms
pub const OPTIMAL_FRAME_SIZE_48KHZ: usize = 3072; // 3072 samples at 48kHz = 64ms
pub const MIN_FRAME_SIZE: usize = 256;
pub const MAX_FRAME_SIZE: usize = 4096;

/// Binary frame optimization settings
#[derive(Debug, Clone)]
pub struct FrameOptimizationConfig {
    /// Target frame size in samples
    pub target_frame_size: usize,
    /// Audio sample rate
    pub sample_rate: u32,
    /// Enable dynamic frame size adjustment
    pub adaptive_frame_size: bool,
    /// Enable binary frame packing
    pub binary_packing_enabled: bool,
    /// Compression threshold (bytes)
    pub compression_threshold: usize,
}

impl Default for FrameOptimizationConfig {
    fn default() -> Self {
        Self {
            target_frame_size: OPTIMAL_FRAME_SIZE_16KHZ,
            sample_rate: 16000,
            adaptive_frame_size: true,
            binary_packing_enabled: true,
            compression_threshold: 512,
        }
    }
}

/// Compression negotiation capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionCapabilities {
    /// Supported compression algorithms
    pub algorithms: Vec<CompressionAlgorithm>,
    /// Maximum compression level
    pub max_compression_level: u8,
    /// Preferred algorithm
    pub preferred_algorithm: CompressionAlgorithm,
    /// CPU overhead tolerance (0.0-1.0)
    pub cpu_overhead_tolerance: f32,
}

/// Supported compression algorithms
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum CompressionAlgorithm {
    None,
    Lz4,
    Zstd,
    Opus, // For audio streams
}

impl Default for CompressionCapabilities {
    fn default() -> Self {
        Self {
            algorithms: vec![
                CompressionAlgorithm::Lz4,
                CompressionAlgorithm::Zstd,
                CompressionAlgorithm::Opus,
            ],
            max_compression_level: 6,
            preferred_algorithm: CompressionAlgorithm::Lz4,
            cpu_overhead_tolerance: 0.1, // 10% CPU overhead tolerance
        }
    }
}

/// Binary frame with optimization metadata
#[derive(Debug, Clone)]
pub struct OptimizedBinaryFrame {
    /// Frame data
    pub data: Vec<u8>,
    /// Frame timestamp
    pub timestamp: u64,
    /// Frame sequence number
    pub sequence: u32,
    /// Compression algorithm used
    pub compression: CompressionAlgorithm,
    /// Original size before compression
    pub original_size: usize,
    /// Frame priority (0-255, higher = more important)
    pub priority: u8,
    /// Frame type identifier
    pub frame_type: BinaryFrameType,
}

/// Binary frame types for protocol optimization
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinaryFrameType {
    AudioFrame,
    TranscriptionFrame,
    TranslationFrame,
    ConfigFrame,
    ControlFrame,
}

impl BinaryFrameType {
    pub fn default_priority(&self) -> u8 {
        match self {
            BinaryFrameType::AudioFrame => 255, // Highest priority
            BinaryFrameType::TranscriptionFrame => 200,
            BinaryFrameType::TranslationFrame => 180,
            BinaryFrameType::ConfigFrame => 100,
            BinaryFrameType::ControlFrame => 150,
        }
    }
}

/// Protocol optimizer for binary frame optimization
pub struct ProtocolOptimizer {
    /// Configuration
    config: FrameOptimizationConfig,
    /// Compression capabilities
    compression_caps: CompressionCapabilities,
    /// Frame buffer for batching
    frame_buffer: Mutex<VecDeque<OptimizedBinaryFrame>>,
    /// Compression negotiation state
    negotiation_state: RwLock<CompressionNegotiationState>,
    /// Performance metrics
    metrics: Mutex<ProtocolMetrics>,
    /// Adaptive frame size controller
    frame_size_controller: Mutex<AdaptiveFrameSizeController>,
}

/// Compression negotiation state
#[derive(Debug, Clone)]
struct CompressionNegotiationState {
    /// Negotiated compression algorithm
    negotiated_algorithm: CompressionAlgorithm,
    /// Negotiated compression level
    negotiated_level: u8,
    /// Negotiation completed
    negotiation_complete: bool,
    /// Peer capabilities
    peer_capabilities: Option<CompressionCapabilities>,
}

/// Protocol optimization metrics
#[derive(Debug, Clone, Default)]
pub struct ProtocolMetrics {
    /// Total frames processed
    pub frames_processed: u64,
    /// Total bytes processed
    pub bytes_processed: u64,
    /// Total compressed bytes
    pub compressed_bytes: u64,
    /// Average compression ratio
    pub avg_compression_ratio: f32,
    /// Frame processing latency (microseconds)
    pub avg_frame_latency_us: f32,
    /// Frames dropped due to buffer overflow
    pub frames_dropped: u64,
    /// Current frame size
    pub current_frame_size: usize,
}

/// Adaptive frame size controller
#[derive(Debug)]
struct AdaptiveFrameSizeController {
    /// Current frame size
    current_frame_size: usize,
    /// Target latency (milliseconds)
    target_latency_ms: f32,
    /// Measured latency history
    latency_history: VecDeque<f32>,
    /// CPU usage history
    cpu_usage_history: VecDeque<f32>,
    /// Last adjustment time
    last_adjustment: Instant,
    /// Adjustment cooldown
    adjustment_cooldown: Duration,
}

impl AdaptiveFrameSizeController {
    fn new(initial_frame_size: usize) -> Self {
        Self {
            current_frame_size: initial_frame_size,
            target_latency_ms: 64.0, // 64ms target
            latency_history: VecDeque::with_capacity(100),
            cpu_usage_history: VecDeque::with_capacity(100),
            last_adjustment: Instant::now(),
            adjustment_cooldown: Duration::from_millis(500),
        }
    }

    /// Update controller with performance metrics
    fn update_metrics(&mut self, latency_ms: f32, cpu_usage: f32) {
        self.latency_history.push_back(latency_ms);
        self.cpu_usage_history.push_back(cpu_usage);

        // Keep only recent history
        if self.latency_history.len() > 100 {
            self.latency_history.pop_front();
        }
        if self.cpu_usage_history.len() > 100 {
            self.cpu_usage_history.pop_front();
        }
    }

    /// Adjust frame size based on performance
    fn adjust_frame_size(&mut self) -> bool {
        if self.last_adjustment.elapsed() < self.adjustment_cooldown {
            return false;
        }

        if self.latency_history.len() < 10 {
            return false; // Not enough data
        }

        let avg_latency =
            self.latency_history.iter().sum::<f32>() / self.latency_history.len() as f32;
        let avg_cpu =
            self.cpu_usage_history.iter().sum::<f32>() / self.cpu_usage_history.len() as f32;

        let mut adjustment_made = false;

        // If latency is too high, reduce frame size
        if avg_latency > self.target_latency_ms * 1.2 && self.current_frame_size > MIN_FRAME_SIZE {
            self.current_frame_size = (self.current_frame_size * 3 / 4).max(MIN_FRAME_SIZE);
            adjustment_made = true;
            debug!(
                "🔽 Reduced frame size to {} due to high latency ({:.1}ms)",
                self.current_frame_size, avg_latency
            );
        }
        // If latency is good and CPU usage is low, increase frame size for efficiency
        else if avg_latency < self.target_latency_ms * 0.8
            && avg_cpu < 0.5
            && self.current_frame_size < MAX_FRAME_SIZE
        {
            self.current_frame_size = (self.current_frame_size * 5 / 4).min(MAX_FRAME_SIZE);
            adjustment_made = true;
            debug!(
                "🔼 Increased frame size to {} for efficiency (latency: {:.1}ms, CPU: {:.1}%)",
                self.current_frame_size,
                avg_latency,
                avg_cpu * 100.0
            );
        }

        if adjustment_made {
            self.last_adjustment = Instant::now();
        }

        adjustment_made
    }
}

impl ProtocolOptimizer {
    /// Create new protocol optimizer
    pub fn new(config: FrameOptimizationConfig, compression_caps: CompressionCapabilities) -> Self {
        Self {
            frame_size_controller: Mutex::new(AdaptiveFrameSizeController::new(
                config.target_frame_size,
            )),
            config,
            compression_caps,
            frame_buffer: Mutex::new(VecDeque::with_capacity(1000)),
            negotiation_state: RwLock::new(CompressionNegotiationState {
                negotiated_algorithm: CompressionAlgorithm::None,
                negotiated_level: 0,
                negotiation_complete: false,
                peer_capabilities: None,
            }),
            metrics: Mutex::new(ProtocolMetrics::default()),
        }
    }

    /// Negotiate compression with peer
    pub async fn negotiate_compression(
        &self,
        peer_caps: CompressionCapabilities,
    ) -> Result<CompressionAlgorithm> {
        let mut state = self.negotiation_state.write().await;

        // Find common algorithms
        let common_algorithms: Vec<_> = self
            .compression_caps
            .algorithms
            .iter()
            .filter(|&algo| peer_caps.algorithms.contains(algo))
            .collect();

        if common_algorithms.is_empty() {
            warn!("⚠️ No common compression algorithms found, using no compression");
            state.negotiated_algorithm = CompressionAlgorithm::None;
            state.negotiated_level = 0;
        } else {
            // Prefer the client's preferred algorithm if supported
            let selected_algorithm = if common_algorithms.contains(&&peer_caps.preferred_algorithm)
            {
                peer_caps.preferred_algorithm
            } else {
                *common_algorithms[0]
            };

            let compression_level = self
                .compression_caps
                .max_compression_level
                .min(peer_caps.max_compression_level);

            state.negotiated_algorithm = selected_algorithm;
            state.negotiated_level = compression_level;

            info!(
                "🤝 Negotiated compression: {:?} level {}",
                selected_algorithm, compression_level
            );
        }

        state.peer_capabilities = Some(peer_caps);
        state.negotiation_complete = true;

        Ok(state.negotiated_algorithm)
    }

    /// Optimize binary frame
    pub async fn optimize_frame(
        &self,
        frame_data: Vec<u8>,
        frame_type: BinaryFrameType,
        sequence: u32,
    ) -> Result<OptimizedBinaryFrame> {
        let start_time = Instant::now();
        let original_size = frame_data.len();

        // Get current compression settings
        let compression_algorithm = {
            let state = self.negotiation_state.read().await;
            if state.negotiation_complete {
                state.negotiated_algorithm
            } else {
                CompressionAlgorithm::None
            }
        };

        // Apply compression if beneficial
        let (compressed_data, actual_compression) = if original_size
            > self.config.compression_threshold
            && compression_algorithm != CompressionAlgorithm::None
        {
            self.compress_frame_data(&frame_data, compression_algorithm)
                .await?
        } else {
            (frame_data, CompressionAlgorithm::None)
        };

        let optimized_frame = OptimizedBinaryFrame {
            data: compressed_data,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_micros() as u64,
            sequence,
            compression: actual_compression,
            original_size,
            priority: frame_type.default_priority(),
            frame_type,
        };

        // Update metrics
        {
            let mut metrics = self.metrics.lock().await;
            metrics.frames_processed += 1;
            metrics.bytes_processed += original_size as u64;
            metrics.compressed_bytes += optimized_frame.data.len() as u64;

            let compression_ratio = optimized_frame.data.len() as f32 / original_size as f32;
            metrics.avg_compression_ratio = (metrics.avg_compression_ratio
                * (metrics.frames_processed - 1) as f32
                + compression_ratio)
                / metrics.frames_processed as f32;

            let frame_latency_us = start_time.elapsed().as_micros() as f32;
            metrics.avg_frame_latency_us = (metrics.avg_frame_latency_us
                * (metrics.frames_processed - 1) as f32
                + frame_latency_us)
                / metrics.frames_processed as f32;
        }

        debug!(
            "🔧 Optimized {} frame: {} -> {} bytes ({:.1}% compression)",
            format!("{:?}", frame_type).to_lowercase(),
            original_size,
            optimized_frame.data.len(),
            (1.0 - optimized_frame.data.len() as f32 / original_size as f32) * 100.0
        );

        Ok(optimized_frame)
    }

    /// Compress frame data using specified algorithm
    async fn compress_frame_data(
        &self,
        data: &[u8],
        algorithm: CompressionAlgorithm,
    ) -> Result<(Vec<u8>, CompressionAlgorithm)> {
        match algorithm {
            CompressionAlgorithm::None => Ok((data.to_vec(), CompressionAlgorithm::None)),
            CompressionAlgorithm::Lz4 => {
                // Simulate LZ4 compression (in real implementation, use lz4 crate)
                let compressed = self.simulate_lz4_compression(data).await?;
                Ok((compressed, CompressionAlgorithm::Lz4))
            }
            CompressionAlgorithm::Zstd => {
                // Simulate Zstd compression (in real implementation, use zstd crate)
                let compressed = self.simulate_zstd_compression(data).await?;
                Ok((compressed, CompressionAlgorithm::Zstd))
            }
            CompressionAlgorithm::Opus => {
                // Opus compression handled separately for audio streams
                Ok((data.to_vec(), CompressionAlgorithm::Opus))
            }
        }
    }

    /// Simulate LZ4 compression (placeholder for real implementation)
    async fn simulate_lz4_compression(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Fast compression simulation - assume 70% compression ratio
        let compressed_size = (data.len() as f32 * 0.7) as usize;
        let mut compressed = data.to_vec();
        compressed.truncate(compressed_size);
        Ok(compressed)
    }

    /// Simulate Zstd compression (placeholder for real implementation)
    async fn simulate_zstd_compression(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Better compression simulation - assume 60% compression ratio
        let compressed_size = (data.len() as f32 * 0.6) as usize;
        let mut compressed = data.to_vec();
        compressed.truncate(compressed_size);
        Ok(compressed)
    }

    /// Update adaptive frame size based on performance metrics
    pub async fn update_frame_size(
        &self,
        latency_ms: f32,
        cpu_usage: f32,
    ) -> Result<Option<usize>> {
        if !self.config.adaptive_frame_size {
            return Ok(None);
        }

        let mut controller = self.frame_size_controller.lock().await;
        controller.update_metrics(latency_ms, cpu_usage);

        if controller.adjust_frame_size() {
            let new_frame_size = controller.current_frame_size;

            // Update metrics
            {
                let mut metrics = self.metrics.lock().await;
                metrics.current_frame_size = new_frame_size;
            }

            Ok(Some(new_frame_size))
        } else {
            Ok(None)
        }
    }

    /// Get current frame size
    pub async fn get_current_frame_size(&self) -> usize {
        let controller = self.frame_size_controller.lock().await;
        controller.current_frame_size
    }

    /// Get protocol optimization metrics
    pub async fn get_metrics(&self) -> ProtocolMetrics {
        let metrics = self.metrics.lock().await;
        metrics.clone()
    }

    /// Batch optimize multiple frames for efficiency
    pub async fn batch_optimize_frames(
        &self,
        frames: Vec<(Vec<u8>, BinaryFrameType)>,
    ) -> Result<Vec<OptimizedBinaryFrame>> {
        let mut optimized_frames = Vec::with_capacity(frames.len());
        let mut sequence = 0u32;

        for (frame_data, frame_type) in frames {
            let optimized = self
                .optimize_frame(frame_data, frame_type, sequence)
                .await?;
            optimized_frames.push(optimized);
            sequence += 1;
        }

        info!("📦 Batch optimized {} frames", optimized_frames.len());
        Ok(optimized_frames)
    }

    /// Priority-based frame scheduling
    pub async fn schedule_frames(&self, frames: &mut [OptimizedBinaryFrame]) {
        // Sort frames by priority (higher priority first)
        frames.sort_by(|a, b| b.priority.cmp(&a.priority));

        debug!("📋 Scheduled {} frames by priority", frames.len());
    }
}

/// Protocol optimization statistics
#[derive(Debug, Clone, Serialize)]
pub struct ProtocolOptimizationStats {
    /// Current compression algorithm
    pub compression_algorithm: String,
    /// Average compression ratio
    pub avg_compression_ratio: f32,
    /// Total frames processed
    pub frames_processed: u64,
    /// Average frame processing latency (microseconds)
    pub avg_frame_latency_us: f32,
    /// Current frame size
    pub current_frame_size: usize,
    /// Bytes saved through compression
    pub bytes_saved: u64,
    /// Frames dropped
    pub frames_dropped: u64,
}

impl From<ProtocolMetrics> for ProtocolOptimizationStats {
    fn from(metrics: ProtocolMetrics) -> Self {
        let bytes_saved = metrics
            .bytes_processed
            .saturating_sub(metrics.compressed_bytes);

        Self {
            compression_algorithm: "LZ4".to_string(), // Simplified for stats
            avg_compression_ratio: metrics.avg_compression_ratio,
            frames_processed: metrics.frames_processed,
            avg_frame_latency_us: metrics.avg_frame_latency_us,
            current_frame_size: metrics.current_frame_size,
            bytes_saved,
            frames_dropped: metrics.frames_dropped,
        }
    }
}
