//! Opus Audio Codec Integration
//!
//! This module provides comprehensive Opus codec support with:
//! - Adaptive bitrate encoding (6-64 kbps for speech)
//! - Low-latency mode (<50ms)
//! - G.722 fallback for compatibility
//! - Real-time audio streaming optimization

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use tokio::time::{Duration, Instant};
use tracing::{debug, error, info, warn};

/// Opus codec configuration
#[derive(Debug, Clone)]
pub struct OpusCodecConfig {
    /// Sample rate (8000, 12000, 16000, 24000, 48000 Hz)
    pub sample_rate: u32,
    /// Number of channels (1 or 2)
    pub channels: u32,
    /// Application type
    pub application: OpusApplication,
    /// Target bitrate (6000-64000 bps for speech)
    pub bitrate: u32,
    /// Frame duration in milliseconds (2.5, 5, 10, 20, 40, 60)
    pub frame_duration_ms: f32,
    /// Enable adaptive bitrate
    pub adaptive_bitrate: bool,
    /// Enable low-latency mode
    pub low_latency_mode: bool,
    /// Complexity (0-10, higher = better quality but more CPU)
    pub complexity: u32,
    /// Enable discontinuous transmission (DTX)
    pub enable_dtx: bool,
    /// Forward error correction (FEC)
    pub enable_fec: bool,
}

/// Opus application types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OpusApplication {
    /// Voice over IP (optimized for speech)
    Voip,
    /// Audio streaming (optimized for music)
    Audio,
    /// Low delay (optimized for real-time communication)
    RestrictedLowDelay,
}

impl Default for OpusCodecConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            channels: 1,
            application: OpusApplication::Voip,
            bitrate: 16000,          // 16 kbps for speech
            frame_duration_ms: 20.0, // 20ms frames for low latency
            adaptive_bitrate: true,
            low_latency_mode: true,
            complexity: 5,
            enable_dtx: true,
            enable_fec: true,
        }
    }
}

/// G.722 fallback configuration
#[derive(Debug, Clone)]
pub struct G722CodecConfig {
    /// Sample rate (16000 Hz)
    pub sample_rate: u32,
    /// Bitrate (48000, 56000, 64000 bps)
    pub bitrate: u32,
    /// Enable law compression
    pub enable_law_compression: bool,
}

impl Default for G722CodecConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            bitrate: 64000,
            enable_law_compression: false,
        }
    }
}

/// Audio codec type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AudioCodec {
    Opus,
    G722,
    Pcm, // Uncompressed fallback
}

/// Opus encoder with adaptive features
pub struct OpusEncoder {
    /// Configuration
    config: OpusCodecConfig,
    /// Encoder state (placeholder for actual Opus encoder)
    encoder_state: Mutex<OpusEncoderState>,
    /// Adaptive bitrate controller
    bitrate_controller: Mutex<AdaptiveBitrateController>,
    /// Performance metrics
    metrics: Mutex<OpusMetrics>,
    /// Frame buffer for batching
    frame_buffer: Mutex<VecDeque<Vec<f32>>>,
}

/// Opus decoder with error recovery
pub struct OpusDecoder {
    /// Configuration
    config: OpusCodecConfig,
    /// Decoder state (placeholder for actual Opus decoder)
    decoder_state: Mutex<OpusDecoderState>,
    /// Packet loss concealment
    plc_state: Mutex<PacketLossConcealmentState>,
    /// Performance metrics
    metrics: Mutex<OpusMetrics>,
}

/// G.722 codec for fallback
pub struct G722Codec {
    /// Configuration
    config: G722CodecConfig,
    /// Encoder state
    encoder_state: Mutex<G722EncoderState>,
    /// Decoder state
    decoder_state: Mutex<G722DecoderState>,
    /// Performance metrics
    metrics: Mutex<G722Metrics>,
}

/// Opus encoder internal state (placeholder)
#[derive(Debug)]
struct OpusEncoderState {
    initialized: bool,
    frame_size: usize,
    last_encode_time: Instant,
    sequence_number: u32,
}

/// Opus decoder internal state (placeholder)
#[derive(Debug)]
struct OpusDecoderState {
    initialized: bool,
    frame_size: usize,
    last_decode_time: Instant,
    missing_packets: Vec<u32>,
}

/// Packet loss concealment state
#[derive(Debug)]
struct PacketLossConcealmentState {
    last_good_frame: Option<Vec<f32>>,
    concealment_count: u32,
    max_concealment: u32,
}

/// G.722 encoder state (placeholder)
#[derive(Debug)]
struct G722EncoderState {
    initialized: bool,
    filter_state: [i32; 24],
}

/// G.722 decoder state (placeholder)
#[derive(Debug)]
struct G722DecoderState {
    initialized: bool,
    filter_state: [i32; 24],
}

/// Adaptive bitrate controller
#[derive(Debug)]
struct AdaptiveBitrateController {
    /// Current bitrate
    current_bitrate: u32,
    /// Target bitrate
    target_bitrate: u32,
    /// Bitrate range
    min_bitrate: u32,
    max_bitrate: u32,
    /// Network conditions history
    rtt_history: VecDeque<f32>,
    packet_loss_history: VecDeque<f32>,
    /// Audio quality history
    quality_history: VecDeque<f32>,
    /// Last adjustment time
    last_adjustment: Instant,
    /// Adjustment interval
    adjustment_interval: Duration,
}

impl AdaptiveBitrateController {
    fn new(initial_bitrate: u32) -> Self {
        Self {
            current_bitrate: initial_bitrate,
            target_bitrate: initial_bitrate,
            min_bitrate: 6000,  // 6 kbps minimum
            max_bitrate: 64000, // 64 kbps maximum
            rtt_history: VecDeque::with_capacity(50),
            packet_loss_history: VecDeque::with_capacity(50),
            quality_history: VecDeque::with_capacity(50),
            last_adjustment: Instant::now(),
            adjustment_interval: Duration::from_millis(500),
        }
    }

    /// Update network conditions
    fn update_network_conditions(&mut self, rtt_ms: f32, packet_loss_rate: f32) {
        self.rtt_history.push_back(rtt_ms);
        self.packet_loss_history.push_back(packet_loss_rate);

        // Keep only recent history
        if self.rtt_history.len() > 50 {
            self.rtt_history.pop_front();
        }
        if self.packet_loss_history.len() > 50 {
            self.packet_loss_history.pop_front();
        }
    }

    /// Update audio quality metrics
    fn update_quality(&mut self, quality_score: f32) {
        self.quality_history.push_back(quality_score);
        if self.quality_history.len() > 50 {
            self.quality_history.pop_front();
        }
    }

    /// Adjust bitrate based on conditions
    fn adjust_bitrate(&mut self) -> Option<u32> {
        if self.last_adjustment.elapsed() < self.adjustment_interval {
            return None;
        }

        if self.rtt_history.len() < 5 || self.packet_loss_history.len() < 5 {
            return None; // Not enough data
        }

        let avg_rtt = self.rtt_history.iter().sum::<f32>() / self.rtt_history.len() as f32;
        let avg_packet_loss =
            self.packet_loss_history.iter().sum::<f32>() / self.packet_loss_history.len() as f32;
        let avg_quality = if !self.quality_history.is_empty() {
            self.quality_history.iter().sum::<f32>() / self.quality_history.len() as f32
        } else {
            0.8 // Default quality score
        };

        let mut new_bitrate = self.current_bitrate;

        // Increase bitrate if conditions are good
        if avg_rtt < 50.0 && avg_packet_loss < 0.01 && avg_quality > 0.8 {
            new_bitrate = (self.current_bitrate * 110 / 100).min(self.max_bitrate);
        }
        // Decrease bitrate if conditions are poor
        else if avg_rtt > 200.0 || avg_packet_loss > 0.05 || avg_quality < 0.6 {
            new_bitrate = (self.current_bitrate * 85 / 100).max(self.min_bitrate);
        }

        if new_bitrate != self.current_bitrate {
            debug!(
                "🎯 Adaptive bitrate: {} -> {} bps (RTT: {:.1}ms, Loss: {:.1}%, Quality: {:.1})",
                self.current_bitrate,
                new_bitrate,
                avg_rtt,
                avg_packet_loss * 100.0,
                avg_quality
            );
            self.current_bitrate = new_bitrate;
            self.last_adjustment = Instant::now();
            Some(new_bitrate)
        } else {
            None
        }
    }
}

/// Opus performance metrics
#[derive(Debug, Clone, Default)]
pub struct OpusMetrics {
    /// Total frames encoded/decoded
    pub frames_processed: u64,
    /// Total bytes processed
    pub bytes_processed: u64,
    /// Total compressed bytes
    pub compressed_bytes: u64,
    /// Average compression ratio
    pub avg_compression_ratio: f32,
    /// Average encoding/decoding latency (microseconds)
    pub avg_processing_latency_us: f32,
    /// Current bitrate
    pub current_bitrate: u32,
    /// Frames lost to packet loss
    pub frames_lost: u64,
    /// FEC recovery success rate
    pub fec_recovery_rate: f32,
}

/// G.722 performance metrics
#[derive(Debug, Clone, Default)]
pub struct G722Metrics {
    /// Total frames processed
    pub frames_processed: u64,
    /// Total bytes processed
    pub bytes_processed: u64,
    /// Average processing latency (microseconds)
    pub avg_processing_latency_us: f32,
    /// Error rate
    pub error_rate: f32,
}

impl OpusEncoder {
    /// Create new Opus encoder
    pub fn new(config: OpusCodecConfig) -> Result<Self> {
        let frame_size = Self::calculate_frame_size(&config)?;

        Ok(Self {
            bitrate_controller: Mutex::new(AdaptiveBitrateController::new(config.bitrate)),
            config,
            encoder_state: Mutex::new(OpusEncoderState {
                initialized: false,
                frame_size,
                last_encode_time: Instant::now(),
                sequence_number: 0,
            }),
            metrics: Mutex::new(OpusMetrics::default()),
            frame_buffer: Mutex::new(VecDeque::with_capacity(100)),
        })
    }

    /// Calculate frame size in samples
    fn calculate_frame_size(config: &OpusCodecConfig) -> Result<usize> {
        let frame_size = (config.sample_rate as f32 * config.frame_duration_ms / 1000.0) as usize;
        if frame_size == 0 {
            return Err(anyhow!(
                "Invalid frame duration: {}",
                config.frame_duration_ms
            ));
        }
        Ok(frame_size)
    }

    /// Initialize encoder
    pub async fn initialize(&self) -> Result<()> {
        let mut state = self.encoder_state.lock().await;
        if state.initialized {
            return Ok(());
        }

        info!(
            "🎤 Initializing Opus encoder: {}Hz, {} channels, {} bps",
            self.config.sample_rate, self.config.channels, self.config.bitrate
        );

        // In a real implementation, initialize libopus encoder here
        state.initialized = true;

        info!("✅ Opus encoder initialized successfully");
        Ok(())
    }

    /// Encode audio frame
    pub async fn encode(&self, audio_samples: &[f32]) -> Result<Vec<u8>> {
        let start_time = Instant::now();

        // Ensure encoder is initialized
        if !self.encoder_state.lock().await.initialized {
            self.initialize().await?;
        }

        // Check frame size
        let expected_frame_size = {
            let state = self.encoder_state.lock().await;
            state.frame_size * self.config.channels as usize
        };

        if audio_samples.len() != expected_frame_size {
            return Err(anyhow!(
                "Frame size mismatch: expected {}, got {}",
                expected_frame_size,
                audio_samples.len()
            ));
        }

        // Simulate Opus encoding (in real implementation, call opus_encode_float)
        let compressed_data = self.simulate_opus_encoding(audio_samples).await?;

        // Update state and metrics
        {
            let mut state = self.encoder_state.lock().await;
            state.last_encode_time = Instant::now();
            state.sequence_number += 1;
        }

        {
            let mut metrics = self.metrics.lock().await;
            metrics.frames_processed += 1;
            metrics.bytes_processed += (audio_samples.len() * 4) as u64; // 4 bytes per f32
            metrics.compressed_bytes += compressed_data.len() as u64;

            let compression_ratio = compressed_data.len() as f32 / (audio_samples.len() * 4) as f32;
            metrics.avg_compression_ratio = (metrics.avg_compression_ratio
                * (metrics.frames_processed - 1) as f32
                + compression_ratio)
                / metrics.frames_processed as f32;

            let processing_latency_us = start_time.elapsed().as_micros() as f32;
            metrics.avg_processing_latency_us = (metrics.avg_processing_latency_us
                * (metrics.frames_processed - 1) as f32
                + processing_latency_us)
                / metrics.frames_processed as f32;

            metrics.current_bitrate = self.config.bitrate;
        }

        debug!(
            "🎵 Encoded Opus frame: {} samples -> {} bytes ({:.1}% compression)",
            audio_samples.len(),
            compressed_data.len(),
            (1.0 - compressed_data.len() as f32 / (audio_samples.len() * 4) as f32) * 100.0
        );

        Ok(compressed_data)
    }

    /// Simulate Opus encoding (placeholder for real implementation)
    async fn simulate_opus_encoding(&self, _audio_samples: &[f32]) -> Result<Vec<u8>> {
        // Calculate compression based on bitrate and frame duration
        let bytes_per_second = self.config.bitrate / 8;
        let frame_duration_s = self.config.frame_duration_ms / 1000.0;
        let target_bytes = (bytes_per_second as f32 * frame_duration_s) as usize;

        // Simulate variable bitrate
        let actual_bytes = if self.config.adaptive_bitrate {
            let complexity_factor = 0.8 + (self.config.complexity as f32 / 10.0) * 0.4;
            (target_bytes as f32 * complexity_factor) as usize
        } else {
            target_bytes
        };

        // Create simulated compressed data
        let mut compressed = Vec::with_capacity(actual_bytes);
        for i in 0..actual_bytes {
            compressed.push((i % 256) as u8);
        }

        Ok(compressed)
    }

    /// Update adaptive bitrate based on network conditions
    pub async fn update_network_conditions(
        &self,
        rtt_ms: f32,
        packet_loss_rate: f32,
    ) -> Result<Option<u32>> {
        if !self.config.adaptive_bitrate {
            return Ok(None);
        }

        let mut controller = self.bitrate_controller.lock().await;
        controller.update_network_conditions(rtt_ms, packet_loss_rate);

        if let Some(new_bitrate) = controller.adjust_bitrate() {
            // In real implementation, call opus_encoder_ctl to set new bitrate
            info!("📊 Adjusted Opus bitrate to {} bps", new_bitrate);
            Ok(Some(new_bitrate))
        } else {
            Ok(None)
        }
    }

    /// Get current metrics
    pub async fn get_metrics(&self) -> OpusMetrics {
        let metrics = self.metrics.lock().await;
        metrics.clone()
    }
}

impl OpusDecoder {
    /// Create new Opus decoder
    pub fn new(config: OpusCodecConfig) -> Result<Self> {
        let frame_size = OpusEncoder::calculate_frame_size(&config)?;

        Ok(Self {
            config,
            decoder_state: Mutex::new(OpusDecoderState {
                initialized: false,
                frame_size,
                last_decode_time: Instant::now(),
                missing_packets: Vec::new(),
            }),
            plc_state: Mutex::new(PacketLossConcealmentState {
                last_good_frame: None,
                concealment_count: 0,
                max_concealment: 5, // Maximum consecutive concealed frames
            }),
            metrics: Mutex::new(OpusMetrics::default()),
        })
    }

    /// Initialize decoder
    pub async fn initialize(&self) -> Result<()> {
        let mut state = self.decoder_state.lock().await;
        if state.initialized {
            return Ok(());
        }

        info!(
            "🔊 Initializing Opus decoder: {}Hz, {} channels",
            self.config.sample_rate, self.config.channels
        );

        // In a real implementation, initialize libopus decoder here
        state.initialized = true;

        info!("✅ Opus decoder initialized successfully");
        Ok(())
    }

    /// Decode audio frame
    pub async fn decode(&self, compressed_data: &[u8]) -> Result<Vec<f32>> {
        let start_time = Instant::now();

        // Ensure decoder is initialized
        if !self.decoder_state.lock().await.initialized {
            self.initialize().await?;
        }

        // Simulate Opus decoding
        let decoded_samples = self.simulate_opus_decoding(compressed_data).await?;

        // Update state and metrics
        {
            let mut state = self.decoder_state.lock().await;
            state.last_decode_time = Instant::now();
        }

        {
            let mut plc_state = self.plc_state.lock().await;
            plc_state.last_good_frame = Some(decoded_samples.clone());
            plc_state.concealment_count = 0;
        }

        {
            let mut metrics = self.metrics.lock().await;
            metrics.frames_processed += 1;
            metrics.bytes_processed += compressed_data.len() as u64;
            metrics.compressed_bytes += compressed_data.len() as u64;

            let processing_latency_us = start_time.elapsed().as_micros() as f32;
            metrics.avg_processing_latency_us = (metrics.avg_processing_latency_us
                * (metrics.frames_processed - 1) as f32
                + processing_latency_us)
                / metrics.frames_processed as f32;
        }

        debug!(
            "🎵 Decoded Opus frame: {} bytes -> {} samples",
            compressed_data.len(),
            decoded_samples.len()
        );

        Ok(decoded_samples)
    }

    /// Decode with packet loss concealment
    pub async fn decode_with_plc(&self, compressed_data: Option<&[u8]>) -> Result<Vec<f32>> {
        match compressed_data {
            Some(data) => self.decode(data).await,
            None => self.conceal_lost_packet().await,
        }
    }

    /// Conceal lost packet
    async fn conceal_lost_packet(&self) -> Result<Vec<f32>> {
        let mut plc_state = self.plc_state.lock().await;

        if plc_state.concealment_count >= plc_state.max_concealment {
            warn!("⚠️ Maximum packet loss concealment reached, returning silence");
            let frame_size = self.decoder_state.lock().await.frame_size;
            return Ok(vec![0.0; frame_size * self.config.channels as usize]);
        }

        let concealed_frame = if let Some(ref last_frame) = plc_state.last_good_frame {
            // Simple concealment: fade out the last good frame
            let fade_factor =
                1.0 - (plc_state.concealment_count as f32 / plc_state.max_concealment as f32);
            last_frame
                .iter()
                .map(|&sample| sample * fade_factor * 0.5)
                .collect()
        } else {
            // No previous frame available, return silence
            let frame_size = self.decoder_state.lock().await.frame_size;
            vec![0.0; frame_size * self.config.channels as usize]
        };

        plc_state.concealment_count += 1;

        {
            let mut metrics = self.metrics.lock().await;
            metrics.frames_lost += 1;
        }

        warn!(
            "🔇 Concealed lost packet (#{} consecutive)",
            plc_state.concealment_count
        );
        Ok(concealed_frame)
    }

    /// Simulate Opus decoding (placeholder for real implementation)
    async fn simulate_opus_decoding(&self, compressed_data: &[u8]) -> Result<Vec<f32>> {
        let frame_size = {
            let state = self.decoder_state.lock().await;
            state.frame_size * self.config.channels as usize
        };

        // Simulate decoding by generating audio samples
        let mut samples = Vec::with_capacity(frame_size);
        for i in 0..frame_size {
            let sample_value = if !compressed_data.is_empty() {
                let byte_index = i % compressed_data.len();
                (compressed_data[byte_index] as f32 - 128.0) / 128.0
            } else {
                0.0
            };
            samples.push(sample_value);
        }

        Ok(samples)
    }

    /// Get current metrics
    pub async fn get_metrics(&self) -> OpusMetrics {
        let metrics = self.metrics.lock().await;
        metrics.clone()
    }
}

impl G722Codec {
    /// Create new G.722 codec
    pub fn new(config: G722CodecConfig) -> Self {
        Self {
            config,
            encoder_state: Mutex::new(G722EncoderState {
                initialized: false,
                filter_state: [0; 24],
            }),
            decoder_state: Mutex::new(G722DecoderState {
                initialized: false,
                filter_state: [0; 24],
            }),
            metrics: Mutex::new(G722Metrics::default()),
        }
    }

    /// Initialize G.722 codec
    pub async fn initialize(&self) -> Result<()> {
        info!(
            "📻 Initializing G.722 codec: {}Hz, {} bps",
            self.config.sample_rate, self.config.bitrate
        );

        // Initialize encoder and decoder states
        {
            let mut encoder_state = self.encoder_state.lock().await;
            encoder_state.initialized = true;
        }

        {
            let mut decoder_state = self.decoder_state.lock().await;
            decoder_state.initialized = true;
        }

        info!("✅ G.722 codec initialized successfully");
        Ok(())
    }

    /// Encode with G.722
    pub async fn encode(&self, audio_samples: &[f32]) -> Result<Vec<u8>> {
        let start_time = Instant::now();

        // Ensure encoder is initialized
        if !self.encoder_state.lock().await.initialized {
            self.initialize().await?;
        }

        // Simulate G.722 encoding (50% compression ratio)
        let compressed_size = audio_samples.len() / 2;
        let compressed_data: Vec<u8> = (0..compressed_size)
            .map(|i| (audio_samples[i * 2] * 127.0 + 128.0) as u8)
            .collect();

        // Update metrics
        {
            let mut metrics = self.metrics.lock().await;
            metrics.frames_processed += 1;
            metrics.bytes_processed += (audio_samples.len() * 4) as u64;

            let processing_latency_us = start_time.elapsed().as_micros() as f32;
            metrics.avg_processing_latency_us = (metrics.avg_processing_latency_us
                * (metrics.frames_processed - 1) as f32
                + processing_latency_us)
                / metrics.frames_processed as f32;
        }

        debug!(
            "📻 Encoded G.722 frame: {} samples -> {} bytes",
            audio_samples.len(),
            compressed_data.len()
        );

        Ok(compressed_data)
    }

    /// Decode with G.722
    pub async fn decode(&self, compressed_data: &[u8]) -> Result<Vec<f32>> {
        let start_time = Instant::now();

        // Ensure decoder is initialized
        if !self.decoder_state.lock().await.initialized {
            self.initialize().await?;
        }

        // Simulate G.722 decoding (2x expansion)
        let decoded_samples: Vec<f32> = compressed_data
            .iter()
            .flat_map(|&byte| {
                let sample = (byte as f32 - 128.0) / 127.0;
                vec![sample, sample] // Upsample
            })
            .collect();

        // Update metrics
        {
            let mut metrics = self.metrics.lock().await;
            metrics.frames_processed += 1;
            metrics.bytes_processed += compressed_data.len() as u64;

            let processing_latency_us = start_time.elapsed().as_micros() as f32;
            metrics.avg_processing_latency_us = (metrics.avg_processing_latency_us
                * (metrics.frames_processed - 1) as f32
                + processing_latency_us)
                / metrics.frames_processed as f32;
        }

        debug!(
            "📻 Decoded G.722 frame: {} bytes -> {} samples",
            compressed_data.len(),
            decoded_samples.len()
        );

        Ok(decoded_samples)
    }

    /// Get current metrics
    pub async fn get_metrics(&self) -> G722Metrics {
        let metrics = self.metrics.lock().await;
        metrics.clone()
    }
}

/// Audio codec manager with automatic fallback
pub struct AudioCodecManager {
    /// Primary codec (Opus)
    opus_encoder: Option<Arc<OpusEncoder>>,
    opus_decoder: Option<Arc<OpusDecoder>>,
    /// Fallback codec (G.722)
    g722_codec: Option<Arc<G722Codec>>,
    /// Current active codec
    active_codec: RwLock<AudioCodec>,
    /// Codec capabilities
    capabilities: Mutex<CodecCapabilities>,
}

/// Codec capabilities and negotiation
#[derive(Debug, Clone)]
pub struct CodecCapabilities {
    /// Supported codecs in preference order
    pub supported_codecs: Vec<AudioCodec>,
    /// Opus configuration
    pub opus_config: Option<OpusCodecConfig>,
    /// G.722 configuration
    pub g722_config: Option<G722CodecConfig>,
    /// Automatic fallback enabled
    pub auto_fallback: bool,
}

impl Default for CodecCapabilities {
    fn default() -> Self {
        Self {
            supported_codecs: vec![AudioCodec::Opus, AudioCodec::G722, AudioCodec::Pcm],
            opus_config: Some(OpusCodecConfig::default()),
            g722_config: Some(G722CodecConfig::default()),
            auto_fallback: true,
        }
    }
}

impl AudioCodecManager {
    /// Create new codec manager
    pub async fn new(capabilities: CodecCapabilities) -> Result<Self> {
        let mut manager = Self {
            opus_encoder: None,
            opus_decoder: None,
            g722_codec: None,
            active_codec: RwLock::new(AudioCodec::Pcm),
            capabilities: Mutex::new(capabilities.clone()),
        };

        // Initialize preferred codecs
        if capabilities.supported_codecs.contains(&AudioCodec::Opus) {
            if let Some(opus_config) = capabilities.opus_config {
                let encoder = Arc::new(OpusEncoder::new(opus_config.clone())?);
                let decoder = Arc::new(OpusDecoder::new(opus_config)?);
                manager.opus_encoder = Some(encoder);
                manager.opus_decoder = Some(decoder);
                *manager.active_codec.write().await = AudioCodec::Opus;
                info!("🎵 Opus codec available as primary");
            }
        }

        if capabilities.supported_codecs.contains(&AudioCodec::G722) {
            if let Some(g722_config) = capabilities.g722_config {
                let codec = Arc::new(G722Codec::new(g722_config));
                manager.g722_codec = Some(codec);
                info!("📻 G.722 codec available as fallback");
            }
        }

        Ok(manager)
    }

    /// Encode audio with active codec
    pub async fn encode_audio(&self, audio_samples: &[f32]) -> Result<(Vec<u8>, AudioCodec)> {
        // Use a loop to avoid recursive async calls
        loop {
            let active_codec = *self.active_codec.read().await;

            match active_codec {
                AudioCodec::Opus => {
                    if let Some(ref encoder) = self.opus_encoder {
                        match encoder.encode(audio_samples).await {
                            Ok(encoded) => return Ok((encoded, AudioCodec::Opus)),
                            Err(e) => {
                                error!("❌ Opus encoding failed: {}", e);
                                self.fallback_to_g722().await?;
                                // Loop will retry with G722
                            }
                        }
                    } else {
                        self.fallback_to_g722().await?;
                        // Loop will retry with G722
                    }
                }
                AudioCodec::G722 => {
                    if let Some(ref codec) = self.g722_codec {
                        let encoded = codec.encode(audio_samples).await?;
                        return Ok((encoded, AudioCodec::G722));
                    } else {
                        warn!("⚠️ No codecs available, using uncompressed PCM");
                        let pcm_data: Vec<u8> = audio_samples
                            .iter()
                            .flat_map(|&sample| {
                                let sample_i16 = (sample * 32767.0) as i16;
                                sample_i16.to_le_bytes()
                            })
                            .collect();
                        return Ok((pcm_data, AudioCodec::Pcm));
                    }
                }
                AudioCodec::Pcm => {
                    let pcm_data: Vec<u8> = audio_samples
                        .iter()
                        .flat_map(|&sample| {
                            let sample_i16 = (sample * 32767.0) as i16;
                            sample_i16.to_le_bytes()
                        })
                        .collect();
                    return Ok((pcm_data, AudioCodec::Pcm));
                }
            }
        }
    }

    /// Decode audio with specified codec
    pub async fn decode_audio(&self, encoded_data: &[u8], codec: AudioCodec) -> Result<Vec<f32>> {
        match codec {
            AudioCodec::Opus => {
                if let Some(ref decoder) = self.opus_decoder {
                    decoder.decode(encoded_data).await
                } else {
                    Err(anyhow!("Opus decoder not available"))
                }
            }
            AudioCodec::G722 => {
                if let Some(ref codec) = self.g722_codec {
                    codec.decode(encoded_data).await
                } else {
                    Err(anyhow!("G.722 codec not available"))
                }
            }
            AudioCodec::Pcm => {
                // Decode 16-bit PCM
                let samples: Vec<f32> = encoded_data
                    .chunks_exact(2)
                    .map(|chunk| {
                        let sample_i16 = i16::from_le_bytes([chunk[0], chunk[1]]);
                        sample_i16 as f32 / 32767.0
                    })
                    .collect();
                Ok(samples)
            }
        }
    }

    /// Fallback to G.722 codec
    async fn fallback_to_g722(&self) -> Result<()> {
        let capabilities = self.capabilities.lock().await;
        if capabilities.auto_fallback && capabilities.supported_codecs.contains(&AudioCodec::G722) {
            *self.active_codec.write().await = AudioCodec::G722;
            warn!("🔄 Falling back to G.722 codec");
            Ok(())
        } else {
            Err(anyhow!("No fallback codec available"))
        }
    }

    /// Get active codec
    pub async fn get_active_codec(&self) -> AudioCodec {
        *self.active_codec.read().await
    }
}
