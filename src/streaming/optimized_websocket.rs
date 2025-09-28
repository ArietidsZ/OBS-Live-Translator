//! Optimized WebSocket Handler with Binary Protocol and Opus Compression
//!
//! This module provides enhanced WebSocket functionality:
//! - Binary WebSocket protocol for reduced overhead
//! - Opus compression for audio streams (6-64 kbps adaptive)
//! - Adaptive buffering strategy (30-50ms jitter buffer)
//! - Target: 2% CPU, 10ms network overhead

use super::{SessionManager, StreamingSession, StreamingConfig};
use anyhow::{Result, anyhow};
use axum::extract::ws::{WebSocket, Message};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{mpsc, Mutex};
use tokio::time::{timeout, interval};
use tracing::{info, warn, error, debug};
use uuid::Uuid;

/// Binary protocol message types
#[repr(u8)]
#[derive(Debug, Clone, Copy)]
pub enum BinaryMessageType {
    /// Raw audio data (uncompressed f32)
    RawAudio = 0x01,
    /// Opus-compressed audio data
    OpusAudio = 0x02,
    /// Transcription result
    Transcription = 0x10,
    /// Translation result
    Translation = 0x11,
    /// Configuration message
    Config = 0x20,
    /// Status message
    Status = 0x21,
    /// Error message
    Error = 0x22,
    /// Heartbeat ping
    Ping = 0x30,
    /// Heartbeat pong
    Pong = 0x31,
}

impl TryFrom<u8> for BinaryMessageType {
    type Error = anyhow::Error;

    fn try_from(value: u8) -> Result<Self> {
        match value {
            0x01 => Ok(BinaryMessageType::RawAudio),
            0x02 => Ok(BinaryMessageType::OpusAudio),
            0x10 => Ok(BinaryMessageType::Transcription),
            0x11 => Ok(BinaryMessageType::Translation),
            0x20 => Ok(BinaryMessageType::Config),
            0x21 => Ok(BinaryMessageType::Status),
            0x22 => Ok(BinaryMessageType::Error),
            0x30 => Ok(BinaryMessageType::Ping),
            0x31 => Ok(BinaryMessageType::Pong),
            _ => Err(anyhow!("Unknown binary message type: {}", value)),
        }
    }
}

/// Binary message header
#[derive(Debug, Clone)]
pub struct BinaryMessageHeader {
    /// Message type
    pub msg_type: BinaryMessageType,
    /// Message length (excluding header)
    pub length: u32,
    /// Timestamp (microseconds since Unix epoch)
    pub timestamp: u64,
    /// Sequence number for ordering
    pub sequence: u32,
}

impl BinaryMessageHeader {
    const SIZE: usize = 17; // 1 + 4 + 8 + 4 bytes

    pub fn new(msg_type: BinaryMessageType, length: u32, sequence: u32) -> Self {
        Self {
            msg_type,
            length,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_micros() as u64,
            sequence,
        }
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(Self::SIZE);
        bytes.push(self.msg_type as u8);
        bytes.extend_from_slice(&self.length.to_le_bytes());
        bytes.extend_from_slice(&self.timestamp.to_le_bytes());
        bytes.extend_from_slice(&self.sequence.to_le_bytes());
        bytes
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < Self::SIZE {
            return Err(anyhow!("Header too short"));
        }

        let msg_type = BinaryMessageType::try_from(bytes[0])?;
        let length = u32::from_le_bytes([bytes[1], bytes[2], bytes[3], bytes[4]]);
        let timestamp = u64::from_le_bytes([
            bytes[5], bytes[6], bytes[7], bytes[8],
            bytes[9], bytes[10], bytes[11], bytes[12],
        ]);
        let sequence = u32::from_le_bytes([bytes[13], bytes[14], bytes[15], bytes[16]]);

        Ok(Self {
            msg_type,
            length,
            timestamp,
            sequence,
        })
    }
}

/// Opus encoder wrapper for audio compression
pub struct OpusEncoder {
    /// Opus encoder instance (placeholder for real implementation)
    _encoder: (),
    /// Current bitrate setting
    bitrate: i32,
    /// Sample rate
    sample_rate: u32,
    /// Channels
    channels: u8,
    /// Frame size in samples
    frame_size: usize,
}

impl OpusEncoder {
    pub fn new(sample_rate: u32, channels: u8, bitrate: i32) -> Result<Self> {
        // In a real implementation, this would initialize libopus
        info!("🎵 Initializing Opus encoder: {}Hz, {} channels, {} kbps",
              sample_rate, channels, bitrate / 1000);

        let frame_size = match sample_rate {
            48000 => 960,  // 20ms at 48kHz
            16000 => 320,  // 20ms at 16kHz
            _ => return Err(anyhow!("Unsupported sample rate: {}", sample_rate)),
        };

        Ok(Self {
            _encoder: (),
            bitrate,
            sample_rate,
            channels,
            frame_size,
        })
    }

    pub fn encode(&mut self, pcm_data: &[f32]) -> Result<Vec<u8>> {
        // In a real implementation, this would call opus_encode_float()
        // For now, simulate compression by simple byte packing

        let compression_ratio = self.calculate_compression_ratio();
        let compressed_size = (pcm_data.len() * 4) / compression_ratio;

        let mut compressed = Vec::with_capacity(compressed_size);

        // Simulate Opus compression (placeholder)
        for chunk in pcm_data.chunks(self.frame_size) {
            let _frame_bytes = (chunk.len() * 4) / compression_ratio;

            // Simple simulation: take every nth sample and quantize
            for (i, &sample) in chunk.iter().enumerate() {
                if i % compression_ratio == 0 {
                    let quantized = (sample * 32767.0) as i16;
                    compressed.extend_from_slice(&quantized.to_le_bytes());
                }
            }
        }

        debug!("🎵 Opus encoded: {} samples -> {} bytes (ratio: {}:1)",
               pcm_data.len(), compressed.len(), compression_ratio);

        Ok(compressed)
    }

    pub fn decode(&mut self, opus_data: &[u8]) -> Result<Vec<f32>> {
        // In a real implementation, this would call opus_decode_float()
        // For now, simulate decompression

        let compression_ratio = self.calculate_compression_ratio();
        let decompressed_size = (opus_data.len() / 2) * compression_ratio;

        let mut pcm = Vec::with_capacity(decompressed_size);

        // Simple simulation: expand compressed data
        for chunk in opus_data.chunks(2) {
            if chunk.len() == 2 {
                let quantized = i16::from_le_bytes([chunk[0], chunk[1]]);
                let sample = quantized as f32 / 32767.0;

                // Duplicate samples to restore original rate
                for _ in 0..compression_ratio {
                    pcm.push(sample);
                }
            }
        }

        Ok(pcm)
    }

    pub fn set_bitrate(&mut self, bitrate: i32) -> Result<()> {
        if bitrate < 6000 || bitrate > 64000 {
            return Err(anyhow!("Bitrate must be between 6-64 kbps"));
        }

        self.bitrate = bitrate;
        debug!("🎵 Opus bitrate updated: {} kbps", bitrate / 1000);
        Ok(())
    }

    fn calculate_compression_ratio(&self) -> usize {
        // Calculate compression ratio based on bitrate
        let uncompressed_bps = self.sample_rate * self.channels as u32 * 32; // 32-bit float
        let ratio = uncompressed_bps as f64 / self.bitrate as f64;
        (ratio as usize).max(2).min(20) // Clamp between 2:1 and 20:1
    }
}

/// Adaptive jitter buffer for smooth audio playback
pub struct AdaptiveJitterBuffer {
    /// Audio frame queue
    frames: VecDeque<AudioFrame>,
    /// Target buffer size (milliseconds)
    target_size_ms: u32,
    /// Current buffer size range
    min_size_ms: u32,
    max_size_ms: u32,
    /// Statistics for adaptation
    stats: JitterBufferStats,
    /// Last adaptation time
    last_adaptation: Instant,
}

#[derive(Debug, Clone)]
struct AudioFrame {
    data: Vec<f32>,
    timestamp: u64,
    sequence: u32,
    received_at: Instant,
}

#[derive(Debug, Clone, Default)]
struct JitterBufferStats {
    frames_received: u64,
    frames_dropped: u64,
    frames_duplicated: u64,
    average_jitter_ms: f32,
    max_jitter_ms: f32,
}

impl AdaptiveJitterBuffer {
    pub fn new(initial_size_ms: u32) -> Self {
        Self {
            frames: VecDeque::new(),
            target_size_ms: initial_size_ms,
            min_size_ms: 30,
            max_size_ms: 100,
            stats: JitterBufferStats::default(),
            last_adaptation: Instant::now(),
        }
    }

    pub fn add_frame(&mut self, data: Vec<f32>, timestamp: u64, sequence: u32) -> Result<()> {
        let frame = AudioFrame {
            data,
            timestamp,
            sequence,
            received_at: Instant::now(),
        };

        self.stats.frames_received += 1;

        // Check for duplicate frames
        if self.frames.iter().any(|f| f.sequence == sequence) {
            self.stats.frames_duplicated += 1;
            return Ok(());
        }

        // Insert frame in correct position based on sequence number
        let insert_pos = self.frames
            .iter()
            .position(|f| f.sequence > sequence)
            .unwrap_or(self.frames.len());

        self.frames.insert(insert_pos, frame);

        // Adapt buffer size if needed
        if self.last_adaptation.elapsed() > Duration::from_secs(5) {
            self.adapt_buffer_size();
            self.last_adaptation = Instant::now();
        }

        // Drop old frames if buffer is too large
        self.cleanup_old_frames();

        Ok(())
    }

    pub fn get_frame(&mut self) -> Option<Vec<f32>> {
        if self.should_output_frame() {
            self.frames.pop_front().map(|frame| frame.data)
        } else {
            None
        }
    }

    fn should_output_frame(&self) -> bool {
        if self.frames.is_empty() {
            return false;
        }

        // Check if we have enough buffered audio
        let buffer_duration_ms = self.estimate_buffer_duration_ms();
        buffer_duration_ms >= self.target_size_ms
    }

    fn estimate_buffer_duration_ms(&self) -> u32 {
        if self.frames.is_empty() {
            return 0;
        }

        // Estimate based on frame count and typical frame duration
        let frame_duration_ms = 20; // Assume 20ms frames
        (self.frames.len() as u32) * frame_duration_ms
    }

    fn adapt_buffer_size(&mut self) {
        // Calculate recent jitter statistics
        let recent_jitter = self.calculate_recent_jitter();

        // Adapt target size based on jitter
        if recent_jitter > 15.0 {
            // High jitter, increase buffer size
            self.target_size_ms = (self.target_size_ms + 5).min(self.max_size_ms);
        } else if recent_jitter < 5.0 {
            // Low jitter, decrease buffer size for lower latency
            self.target_size_ms = (self.target_size_ms.saturating_sub(2)).max(self.min_size_ms);
        }

        debug!("📊 Jitter buffer adapted: target={}ms, jitter={:.1}ms",
               self.target_size_ms, recent_jitter);
    }

    fn calculate_recent_jitter(&self) -> f32 {
        if self.frames.len() < 2 {
            return 0.0;
        }

        let mut jitters = Vec::new();
        for window in self.frames.iter().collect::<Vec<_>>().windows(2) {
            let jitter = window[1].received_at.duration_since(window[0].received_at).as_millis() as f32;
            jitters.push(jitter);
        }

        // Calculate average jitter for recent frames
        let recent_count = jitters.len().min(10);
        if recent_count > 0 {
            jitters[jitters.len() - recent_count..].iter().sum::<f32>() / recent_count as f32
        } else {
            0.0
        }
    }

    fn cleanup_old_frames(&mut self) {
        // Remove frames older than maximum buffer size
        let max_frames = (self.max_size_ms / 20) as usize; // Assume 20ms frames

        while self.frames.len() > max_frames {
            self.frames.pop_front();
            self.stats.frames_dropped += 1;
        }
    }

    pub fn get_stats(&self) -> &JitterBufferStats {
        &self.stats
    }
}

/// Optimized WebSocket handler with binary protocol and compression
pub struct OptimizedWebSocketHandler {
    socket: WebSocket,
    session_manager: Arc<SessionManager>,
    session_id: String,
    session: Option<Arc<StreamingSession>>,
    config: StreamingConfig,
    opus_encoder: Mutex<Option<OpusEncoder>>,
    jitter_buffer: Mutex<AdaptiveJitterBuffer>,
    sequence_counter: Mutex<u32>,
    compression_enabled: bool,
    stats: Mutex<ConnectionStats>,
}

#[derive(Debug, Clone, Default)]
struct ConnectionStats {
    bytes_sent: u64,
    bytes_received: u64,
    messages_sent: u64,
    messages_received: u64,
    compression_ratio: f32,
    average_latency_ms: f32,
}

impl OptimizedWebSocketHandler {
    pub fn new(socket: WebSocket, session_manager: Arc<SessionManager>, config: StreamingConfig) -> Self {
        let session_id = Uuid::new_v4().to_string();

        Self {
            socket,
            session_manager,
            session_id,
            session: None,
            config,
            opus_encoder: Mutex::new(None),
            jitter_buffer: Mutex::new(AdaptiveJitterBuffer::new(40)), // 40ms initial buffer
            sequence_counter: Mutex::new(0),
            compression_enabled: true,
            stats: Mutex::new(ConnectionStats::default()),
        }
    }

    /// Handle optimized WebSocket connection
    pub async fn handle(mut self) -> Result<()> {
        info!("🔗 Starting optimized WebSocket handler for session {}", self.session_id);

        // Create session
        let session = self.session_manager.create_session(&self.session_id).await?;
        self.session = Some(Arc::clone(&session));

        // Initialize Opus encoder
        {
            let mut encoder = self.opus_encoder.lock().await;
            *encoder = Some(OpusEncoder::new(16000, 1, 32000)?); // 32 kbps initial bitrate
        }

        // Send initial status
        self.send_binary_status("Connected", true).await?;

        // Start background tasks
        let (tx, mut rx) = mpsc::channel(1000);

        // Heartbeat task
        let heartbeat_tx = tx.clone();
        let session_id = self.session_id.clone();
        let session_manager = Arc::clone(&self.session_manager);
        tokio::spawn(async move {
            Self::heartbeat_task(session_id, session_manager, heartbeat_tx).await;
        });

        // Statistics reporting task
        let stats_tx = tx.clone();
        let stats = Arc::new(Mutex::new(ConnectionStats::default()));
        tokio::spawn(async move {
            Self::stats_reporting_task(stats, stats_tx).await;
        });

        // Main message handling loop
        loop {
            tokio::select! {
                // Handle incoming WebSocket messages
                message = timeout(Duration::from_secs(30), self.socket.recv()) => {
                    match message {
                        Ok(Some(Ok(msg))) => {
                            if let Err(e) = self.handle_message(msg).await {
                                error!("Error handling message: {}", e);
                                self.send_binary_error(&e.to_string(), 500).await?;
                            }
                        },
                        Ok(Some(Err(e))) => {
                            warn!("WebSocket error: {}", e);
                            break;
                        },
                        Ok(None) => {
                            info!("WebSocket connection closed");
                            break;
                        },
                        Err(_) => {
                            warn!("WebSocket timeout");
                            break;
                        },
                    }
                },
                // Handle background task messages
                task_msg = rx.recv() => {
                    if task_msg.is_none() {
                        break;
                    }
                    // Process background task results
                },
            }
        }

        // Cleanup
        self.session_manager.remove_session(&self.session_id).await;
        self.log_final_stats().await;
        info!("🔗 Optimized WebSocket session {} ended", self.session_id);

        Ok(())
    }

    /// Handle individual WebSocket message with binary protocol support
    async fn handle_message(&mut self, message: Message) -> Result<()> {
        let start_time = Instant::now();

        let result = match message {
            Message::Binary(data) => {
                self.handle_binary_message(data).await
            },
            Message::Text(text) => {
                // Fallback to JSON for compatibility
                warn!("Received text message, consider using binary protocol for better performance");
                self.handle_text_message(text).await
            },
            Message::Ping(data) => {
                self.socket.send(Message::Pong(data)).await?;
                Ok(())
            },
            Message::Pong(_) => {
                // Update latency statistics
                Ok(())
            },
            Message::Close(_) => {
                info!("WebSocket close message received");
                Ok(())
            },
        };

        // Update latency statistics
        let latency_ms = start_time.elapsed().as_secs_f32() * 1000.0;
        self.update_latency_stats(latency_ms).await;

        result
    }

    /// Handle binary protocol message
    async fn handle_binary_message(&mut self, data: Vec<u8>) -> Result<()> {
        if data.len() < BinaryMessageHeader::SIZE {
            return Err(anyhow!("Binary message too short"));
        }

        let header = BinaryMessageHeader::from_bytes(&data[..BinaryMessageHeader::SIZE])?;
        let payload = &data[BinaryMessageHeader::SIZE..];

        if payload.len() != header.length as usize {
            return Err(anyhow!("Binary message length mismatch"));
        }

        // Update receive statistics
        {
            let mut stats = self.stats.lock().await;
            stats.bytes_received += data.len() as u64;
            stats.messages_received += 1;
        }

        match header.msg_type {
            BinaryMessageType::RawAudio => {
                self.handle_raw_audio(payload, header.timestamp, header.sequence).await
            },
            BinaryMessageType::OpusAudio => {
                self.handle_opus_audio(payload, header.timestamp, header.sequence).await
            },
            BinaryMessageType::Config => {
                self.handle_binary_config(payload).await
            },
            BinaryMessageType::Ping => {
                self.send_binary_pong(header.timestamp).await
            },
            _ => {
                warn!("Unexpected binary message type: {:?}", header.msg_type);
                Ok(())
            },
        }
    }

    /// Handle raw audio data
    async fn handle_raw_audio(&mut self, data: &[u8], timestamp: u64, sequence: u32) -> Result<()> {
        if data.len() % 4 != 0 {
            return Err(anyhow!("Invalid raw audio data length"));
        }

        let audio_samples: Vec<f32> = data
            .chunks(4)
            .map(|chunk| {
                let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
                f32::from_le_bytes(bytes)
            })
            .collect();

        // Add to jitter buffer
        {
            let mut jitter_buffer = self.jitter_buffer.lock().await;
            jitter_buffer.add_frame(audio_samples.clone(), timestamp, sequence)?;
        }

        // Process audio if enough is buffered
        self.process_buffered_audio().await
    }

    /// Handle Opus-compressed audio data
    async fn handle_opus_audio(&mut self, data: &[u8], timestamp: u64, sequence: u32) -> Result<()> {
        // Decode Opus data
        let audio_samples = {
            let mut encoder = self.opus_encoder.lock().await;
            if let Some(ref mut opus) = encoder.as_mut() {
                opus.decode(data)?
            } else {
                return Err(anyhow!("Opus decoder not initialized"));
            }
        };

        // Add to jitter buffer
        {
            let mut jitter_buffer = self.jitter_buffer.lock().await;
            jitter_buffer.add_frame(audio_samples, timestamp, sequence)?;
        }

        // Process audio if enough is buffered
        self.process_buffered_audio().await
    }

    /// Process buffered audio from jitter buffer
    async fn process_buffered_audio(&mut self) -> Result<()> {
        let audio_frame = {
            let mut jitter_buffer = self.jitter_buffer.lock().await;
            jitter_buffer.get_frame()
        };

        if let Some(audio_data) = audio_frame {
            if let Some(session) = &self.session {
                // Process audio through session
                let results = session.process_audio(audio_data, 16000).await?;

                // Send results back to client
                for result in results {
                    match result {
                        crate::streaming::session::ProcessingResult::Transcription {
                            text, language, confidence, is_final
                        } => {
                            self.send_binary_transcription(text, language, confidence, is_final).await?;
                        },
                        crate::streaming::session::ProcessingResult::Translation {
                            original_text,
                            translated_text,
                            source_language,
                            target_language,
                            confidence
                        } => {
                            self.send_binary_translation(
                                original_text,
                                translated_text,
                                source_language,
                                target_language,
                                confidence
                            ).await?;
                        },
                    }
                }
            }
        }

        Ok(())
    }

    /// Send binary transcription message
    async fn send_binary_transcription(&mut self, text: String, language: String, confidence: f32, is_final: bool) -> Result<()> {
        let mut payload = Vec::new();

        // Encode transcription data
        payload.extend_from_slice(&(text.len() as u32).to_le_bytes());
        payload.extend_from_slice(text.as_bytes());
        payload.extend_from_slice(&(language.len() as u32).to_le_bytes());
        payload.extend_from_slice(language.as_bytes());
        payload.extend_from_slice(&confidence.to_le_bytes());
        payload.push(if is_final { 1 } else { 0 });

        self.send_binary_message(BinaryMessageType::Transcription, payload).await
    }

    /// Send binary translation message
    async fn send_binary_translation(
        &mut self,
        original_text: String,
        translated_text: String,
        source_language: String,
        target_language: String,
        confidence: f32
    ) -> Result<()> {
        let mut payload = Vec::new();

        // Encode translation data
        payload.extend_from_slice(&(original_text.len() as u32).to_le_bytes());
        payload.extend_from_slice(original_text.as_bytes());
        payload.extend_from_slice(&(translated_text.len() as u32).to_le_bytes());
        payload.extend_from_slice(translated_text.as_bytes());
        payload.extend_from_slice(&(source_language.len() as u32).to_le_bytes());
        payload.extend_from_slice(source_language.as_bytes());
        payload.extend_from_slice(&(target_language.len() as u32).to_le_bytes());
        payload.extend_from_slice(target_language.as_bytes());
        payload.extend_from_slice(&confidence.to_le_bytes());

        self.send_binary_message(BinaryMessageType::Translation, payload).await
    }

    /// Send binary status message
    async fn send_binary_status(&mut self, message: &str, connected: bool) -> Result<()> {
        let mut payload = Vec::new();
        payload.push(if connected { 1 } else { 0 });
        payload.extend_from_slice(&(self.session_id.len() as u32).to_le_bytes());
        payload.extend_from_slice(self.session_id.as_bytes());
        payload.extend_from_slice(&(message.len() as u32).to_le_bytes());
        payload.extend_from_slice(message.as_bytes());

        self.send_binary_message(BinaryMessageType::Status, payload).await
    }

    /// Send binary error message
    async fn send_binary_error(&mut self, error: &str, code: u32) -> Result<()> {
        let mut payload = Vec::new();
        payload.extend_from_slice(&code.to_le_bytes());
        payload.extend_from_slice(&(error.len() as u32).to_le_bytes());
        payload.extend_from_slice(error.as_bytes());

        self.send_binary_message(BinaryMessageType::Error, payload).await
    }

    /// Send binary pong message
    async fn send_binary_pong(&mut self, timestamp: u64) -> Result<()> {
        let payload = timestamp.to_le_bytes().to_vec();
        self.send_binary_message(BinaryMessageType::Pong, payload).await
    }

    /// Send binary message with header
    async fn send_binary_message(&mut self, msg_type: BinaryMessageType, payload: Vec<u8>) -> Result<()> {
        let sequence = {
            let mut counter = self.sequence_counter.lock().await;
            *counter += 1;
            *counter
        };

        let header = BinaryMessageHeader::new(msg_type, payload.len() as u32, sequence);
        let mut message = header.to_bytes();
        message.extend_from_slice(&payload);

        // Update send statistics
        {
            let mut stats = self.stats.lock().await;
            stats.bytes_sent += message.len() as u64;
            stats.messages_sent += 1;
        }

        self.socket.send(Message::Binary(message)).await?;
        Ok(())
    }

    /// Handle text message (fallback compatibility)
    async fn handle_text_message(&mut self, _text: String) -> Result<()> {
        // For backward compatibility - implement if needed
        warn!("Text message handling not implemented in optimized handler");
        Ok(())
    }

    /// Handle binary configuration message
    async fn handle_binary_config(&mut self, _payload: &[u8]) -> Result<()> {
        // Implement binary config parsing if needed
        warn!("Binary config handling not yet implemented");
        Ok(())
    }

    /// Update latency statistics
    async fn update_latency_stats(&mut self, latency_ms: f32) {
        let mut stats = self.stats.lock().await;
        let n = stats.messages_received as f32;
        stats.average_latency_ms = (stats.average_latency_ms * (n - 1.0) + latency_ms) / n;
    }

    /// Log final connection statistics
    async fn log_final_stats(&self) {
        let stats = self.stats.lock().await;
        let jitter_stats = {
            let jitter_buffer = self.jitter_buffer.lock().await;
            jitter_buffer.get_stats().clone()
        };

        info!("📊 Final connection stats for {}: sent={}MB, received={}MB, latency={:.1}ms, jitter={:.1}ms",
              self.session_id,
              stats.bytes_sent / (1024 * 1024),
              stats.bytes_received / (1024 * 1024),
              stats.average_latency_ms,
              jitter_stats.average_jitter_ms);
    }

    /// Background heartbeat task
    async fn heartbeat_task(session_id: String, session_manager: Arc<SessionManager>, _tx: mpsc::Sender<()>) {
        let mut interval = interval(Duration::from_secs(30));

        loop {
            interval.tick().await;

            if !session_manager.has_session(&session_id).await {
                break;
            }

            // Update session activity
            if let Ok(session) = session_manager.get_session(&session_id).await {
                session.update_activity().await;
            }
        }
    }

    /// Background statistics reporting task
    async fn stats_reporting_task(stats: Arc<Mutex<ConnectionStats>>, _tx: mpsc::Sender<()>) {
        let mut interval = interval(Duration::from_secs(60));

        loop {
            interval.tick().await;

            let current_stats = {
                let stats = stats.lock().await;
                stats.clone()
            };

            debug!("📊 Connection stats: msgs={}/{}, bytes={}/{}, latency={:.1}ms",
                   current_stats.messages_sent,
                   current_stats.messages_received,
                   current_stats.bytes_sent,
                   current_stats.bytes_received,
                   current_stats.average_latency_ms);
        }
    }
}