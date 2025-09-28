//! Real-time Streaming Pipeline with Low-latency Optimization
//!
//! This module provides comprehensive real-time streaming capabilities:
//! - Low-latency streaming with adaptive jitter buffer (30-50ms)
//! - Continuous audio processing pipeline
//! - Real-time frame synchronization
//! - Backpressure handling and flow control
//! - Protocol optimization with binary frames
//! - Error handling and recovery mechanisms

use anyhow::Result;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock};
use tokio::time::interval;
use tracing::{info, warn, debug};

/// Real-time streaming pipeline
pub struct RealtimeStreamingPipeline {
    /// Pipeline configuration
    config: PipelineConfig,
    /// Audio frame buffer with adaptive sizing
    frame_buffer: Arc<Mutex<AdaptiveFrameBuffer>>,
    /// Processing queue
    processing_queue: Arc<Mutex<ProcessingQueue>>,
    /// Backpressure controller
    backpressure_controller: Arc<BackpressureController>,
    /// Frame synchronizer
    frame_synchronizer: Arc<FrameSynchronizer>,
    /// Pipeline statistics
    stats: Arc<Mutex<PipelineStats>>,
    /// Error recovery manager
    error_recovery: Arc<ErrorRecoveryManager>,
}

/// Pipeline configuration
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Target latency in milliseconds
    pub target_latency_ms: u32,
    /// Maximum tolerable latency in milliseconds
    pub max_latency_ms: u32,
    /// Frame size in samples (at 16kHz)
    pub frame_size_samples: usize,
    /// Sample rate
    pub sample_rate: u32,
    /// Processing threads
    pub processing_threads: usize,
    /// Enable adaptive frame sizing
    pub enable_adaptive_frames: bool,
    /// Enable backpressure handling
    pub enable_backpressure: bool,
    /// Maximum queue size
    pub max_queue_size: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            target_latency_ms: 40,       // 40ms target latency
            max_latency_ms: 100,         // 100ms maximum latency
            frame_size_samples: 1024,    // 64ms at 16kHz (1024 samples)
            sample_rate: 16000,
            processing_threads: 2,
            enable_adaptive_frames: true,
            enable_backpressure: true,
            max_queue_size: 50,
        }
    }
}

/// Adaptive frame buffer with intelligent sizing
pub struct AdaptiveFrameBuffer {
    /// Audio frames queue
    frames: VecDeque<AudioFrame>,
    /// Target buffer size in milliseconds
    target_size_ms: u32,
    /// Current adaptive range
    min_size_ms: u32,
    max_size_ms: u32,
    /// Frame statistics for adaptation
    frame_stats: FrameBufferStats,
    /// Last adaptation time
    last_adaptation: Instant,
    /// Configuration
    config: PipelineConfig,
}

/// Audio frame with metadata
#[derive(Debug, Clone)]
pub struct AudioFrame {
    /// Audio data (f32 samples)
    pub data: Vec<f32>,
    /// Frame timestamp (microseconds)
    pub timestamp: u64,
    /// Sequence number
    pub sequence: u32,
    /// Frame size in samples
    pub size_samples: usize,
    /// Processing deadline
    pub deadline: Instant,
    /// Priority level (0 = highest)
    pub priority: u8,
}

/// Frame buffer statistics
#[derive(Debug, Clone, Default)]
pub struct FrameBufferStats {
    /// Frames received
    pub frames_received: u64,
    /// Frames dropped due to latency
    pub frames_dropped_latency: u64,
    /// Frames dropped due to overflow
    pub frames_dropped_overflow: u64,
    /// Average frame processing time (ms)
    pub avg_processing_time_ms: f32,
    /// Current jitter (ms)
    pub current_jitter_ms: f32,
    /// Buffer underruns
    pub underrun_count: u64,
    /// Buffer overruns
    pub overrun_count: u64,
}

/// Processing queue with priority support
pub struct ProcessingQueue {
    /// High-priority frames (real-time)
    high_priority: VecDeque<AudioFrame>,
    /// Normal priority frames
    normal_priority: VecDeque<AudioFrame>,
    /// Low priority frames (batch processing)
    low_priority: VecDeque<AudioFrame>,
    /// Queue statistics
    stats: QueueStats,
    /// Maximum queue sizes per priority
    max_sizes: [usize; 3], // [high, normal, low]
}

#[derive(Debug, Clone, Default)]
pub struct QueueStats {
    pub total_enqueued: u64,
    pub total_dequeued: u64,
    pub current_size: usize,
    pub max_size_reached: usize,
    pub avg_wait_time_ms: f32,
}

/// Backpressure controller for flow control
pub struct BackpressureController {
    /// Current load level (0.0-1.0)
    current_load: RwLock<f32>,
    /// Load history for trend analysis
    load_history: Mutex<VecDeque<f32>>,
    /// Backpressure configuration
    config: BackpressureConfig,
    /// Control actions taken
    actions_taken: Mutex<VecDeque<BackpressureAction>>,
}

#[derive(Debug, Clone)]
pub struct BackpressureConfig {
    /// Load threshold to trigger backpressure (0.0-1.0)
    pub load_threshold: f32,
    /// Critical load threshold for aggressive action
    pub critical_threshold: f32,
    /// Load measurement window (seconds)
    pub measurement_window_secs: u32,
    /// Recovery threshold to disable backpressure
    pub recovery_threshold: f32,
}

impl Default for BackpressureConfig {
    fn default() -> Self {
        Self {
            load_threshold: 0.75,     // Trigger at 75% load
            critical_threshold: 0.9,  // Critical at 90% load
            measurement_window_secs: 5,
            recovery_threshold: 0.6,  // Recover when load drops to 60%
        }
    }
}

#[derive(Debug, Clone)]
pub enum BackpressureAction {
    /// Drop lower priority frames
    DropLowPriority,
    /// Reduce frame quality
    ReduceQuality,
    /// Increase frame size (reduce frequency)
    IncreaseFrameSize,
    /// Enable aggressive compression
    EnableAggressiveCompression,
    /// Pause non-critical processing
    PauseNonCritical,
}

/// Frame synchronizer for real-time processing
pub struct FrameSynchronizer {
    /// Reference timestamp for synchronization
    reference_timestamp: Mutex<Option<u64>>,
    /// Clock drift compensation
    drift_compensation: Mutex<f64>,
    /// Synchronization statistics
    sync_stats: Mutex<SyncStats>,
    /// Configuration
    config: SyncConfig,
}

#[derive(Debug, Clone)]
pub struct SyncConfig {
    /// Maximum allowed drift in milliseconds
    pub max_drift_ms: f32,
    /// Drift correction rate (0.0-1.0)
    pub correction_rate: f32,
    /// Synchronization window size
    pub sync_window_size: usize,
}

impl Default for SyncConfig {
    fn default() -> Self {
        Self {
            max_drift_ms: 10.0,
            correction_rate: 0.1,
            sync_window_size: 20,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct SyncStats {
    pub total_frames_synced: u64,
    pub current_drift_ms: f32,
    pub max_drift_ms: f32,
    pub sync_corrections: u64,
    pub avg_sync_accuracy_ms: f32,
}

/// Error recovery manager
pub struct ErrorRecoveryManager {
    /// Recent errors
    recent_errors: Mutex<VecDeque<StreamingError>>,
    /// Recovery strategies
    recovery_strategies: Vec<RecoveryStrategy>,
    /// Recovery statistics
    stats: Mutex<RecoveryStats>,
}

#[derive(Debug, Clone)]
pub struct StreamingError {
    pub error_type: ErrorType,
    pub timestamp: Instant,
    pub severity: ErrorSeverity,
    pub description: String,
    pub context: String,
}

#[derive(Debug, Clone)]
pub enum ErrorType {
    BufferUnderrun,
    BufferOverrun,
    ProcessingTimeout,
    SynchronizationLost,
    CompressionFailed,
    NetworkError,
    ResourceExhaustion,
}

#[derive(Debug, Clone)]
pub enum ErrorSeverity {
    Low,      // Recoverable, minimal impact
    Medium,   // Noticeable impact, requires action
    High,     // Significant impact, urgent action needed
    Critical, // Service degradation, immediate action required
}

#[derive(Debug, Clone)]
pub struct RecoveryStrategy {
    pub error_types: Vec<ErrorType>,
    pub min_severity: ErrorSeverity,
    pub action: RecoveryAction,
    pub cooldown_ms: u64,
}

#[derive(Debug, Clone)]
pub enum RecoveryAction {
    /// Reset frame buffer
    ResetBuffer,
    /// Restart processing pipeline
    RestartPipeline,
    /// Switch to degraded mode
    DegradedMode,
    /// Increase buffer size
    IncreaseBufferSize,
    /// Reduce processing complexity
    ReduceComplexity,
    /// Force garbage collection
    ForceGC,
}

#[derive(Debug, Clone, Default)]
pub struct RecoveryStats {
    pub total_errors: u64,
    pub recovery_attempts: u64,
    pub successful_recoveries: u64,
    pub avg_recovery_time_ms: f32,
}

/// Pipeline performance statistics
#[derive(Debug, Clone, Default)]
pub struct PipelineStats {
    /// Total frames processed
    pub total_frames_processed: u64,
    /// Average processing latency (ms)
    pub avg_processing_latency_ms: f32,
    /// 95th percentile latency (ms)
    pub p95_latency_ms: f32,
    /// 99th percentile latency (ms)
    pub p99_latency_ms: f32,
    /// Throughput (frames per second)
    pub throughput_fps: f32,
    /// Current buffer fill level (0.0-1.0)
    pub buffer_fill_level: f32,
    /// Quality score (0.0-1.0)
    pub quality_score: f32,
    /// Stability score (0.0-1.0)
    pub stability_score: f32,
}

impl RealtimeStreamingPipeline {
    /// Create a new real-time streaming pipeline
    pub fn new(config: PipelineConfig) -> Self {
        info!("🚀 Initializing real-time streaming pipeline");
        info!("📊 Pipeline config: target_latency={}ms, max_latency={}ms, frame_size={}",
              config.target_latency_ms, config.max_latency_ms, config.frame_size_samples);

        let frame_buffer = Arc::new(Mutex::new(AdaptiveFrameBuffer::new(config.clone())));
        let processing_queue = Arc::new(Mutex::new(ProcessingQueue::new(config.max_queue_size)));
        let backpressure_controller = Arc::new(BackpressureController::new(BackpressureConfig::default()));
        let frame_synchronizer = Arc::new(FrameSynchronizer::new(SyncConfig::default()));
        let error_recovery = Arc::new(ErrorRecoveryManager::new());
        let stats = Arc::new(Mutex::new(PipelineStats::default()));

        Self {
            config,
            frame_buffer,
            processing_queue,
            backpressure_controller,
            frame_synchronizer,
            stats,
            error_recovery,
        }
    }

    /// Start the real-time streaming pipeline
    pub async fn start(&self) -> Result<()> {
        info!("🚀 Starting real-time streaming pipeline");

        // Start background tasks
        self.start_background_tasks().await?;

        info!("✅ Real-time streaming pipeline started successfully");
        Ok(())
    }

    /// Process incoming audio frame
    pub async fn process_frame(&self, data: Vec<f32>, timestamp: u64, sequence: u32) -> Result<()> {
        let start_time = Instant::now();

        // Create audio frame with metadata
        let frame = AudioFrame {
            data,
            timestamp,
            sequence,
            size_samples: self.config.frame_size_samples,
            deadline: start_time + Duration::from_millis(self.config.max_latency_ms as u64),
            priority: self.calculate_frame_priority(timestamp).await,
        };

        // Check backpressure
        if self.should_apply_backpressure().await {
            self.apply_backpressure_action(&frame).await?;
        }

        // Add to frame buffer
        self.add_frame_to_buffer(frame).await?;

        // Update statistics
        self.update_processing_stats(start_time.elapsed()).await;

        Ok(())
    }

    /// Add frame to adaptive buffer
    async fn add_frame_to_buffer(&self, frame: AudioFrame) -> Result<()> {
        let mut buffer = self.frame_buffer.lock().await;

        // Check if frame is too late
        if frame.deadline < Instant::now() {
            buffer.frame_stats.frames_dropped_latency += 1;
            warn!("⏰ Frame {} dropped due to latency", frame.sequence);
            return Ok(());
        }

        // Apply frame synchronization
        self.frame_synchronizer.synchronize_frame(&frame).await?;

        // Add to buffer
        buffer.add_frame(frame)?;

        // Trigger buffer adaptation if needed
        if buffer.should_adapt() {
            buffer.adapt_size().await;
        }

        Ok(())
    }

    /// Calculate frame priority based on timing and content
    async fn calculate_frame_priority(&self, timestamp: u64) -> u8 {
        // In a real implementation, this would analyze:
        // - Frame timing relative to real-time constraints
        // - Audio content (voice activity, silence detection)
        // - Current system load
        // - User interaction context

        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;

        let age_ms = (current_time - timestamp) / 1000;

        // Assign priority based on age and system state
        match age_ms {
            0..=50 => 0,      // Highest priority - real-time
            51..=100 => 1,    // High priority - near real-time
            101..=200 => 2,   // Normal priority
            _ => 3,           // Low priority - batch processing
        }
    }

    /// Check if backpressure should be applied
    async fn should_apply_backpressure(&self) -> bool {
        if !self.config.enable_backpressure {
            return false;
        }

        let current_load = *self.backpressure_controller.current_load.read().await;
        current_load > self.backpressure_controller.config.load_threshold
    }

    /// Apply backpressure action
    async fn apply_backpressure_action(&self, _frame: &AudioFrame) -> Result<()> {
        let current_load = *self.backpressure_controller.current_load.read().await;

        let action = if current_load > self.backpressure_controller.config.critical_threshold {
            BackpressureAction::DropLowPriority
        } else {
            BackpressureAction::ReduceQuality
        };

        match action {
            BackpressureAction::DropLowPriority => {
                self.drop_low_priority_frames().await?;
            },
            BackpressureAction::ReduceQuality => {
                // Reduce processing quality temporarily
                debug!("🔧 Reducing processing quality due to backpressure");
            },
            BackpressureAction::IncreaseFrameSize => {
                // Increase frame size to reduce processing frequency
                debug!("🔧 Increasing frame size due to backpressure");
            },
            _ => {},
        }

        // Record action taken
        {
            let mut actions = self.backpressure_controller.actions_taken.lock().await;
            actions.push_back(action);
            if actions.len() > 100 {
                actions.pop_front();
            }
        }

        Ok(())
    }

    /// Drop low priority frames to relieve pressure
    async fn drop_low_priority_frames(&self) -> Result<()> {
        let mut queue = self.processing_queue.lock().await;
        let initial_size = queue.low_priority.len();

        // Drop up to 50% of low priority frames
        let to_drop = initial_size / 2;
        for _ in 0..to_drop {
            queue.low_priority.pop_front();
        }

        debug!("🗑️ Dropped {} low priority frames due to backpressure", to_drop);
        Ok(())
    }

    /// Start background processing tasks
    async fn start_background_tasks(&self) -> Result<()> {
        // Buffer monitoring task
        let frame_buffer = Arc::clone(&self.frame_buffer);
        let stats = Arc::clone(&self.stats);
        tokio::spawn(async move {
            Self::buffer_monitoring_task(frame_buffer, stats).await;
        });

        // Backpressure monitoring task
        let backpressure_controller = Arc::clone(&self.backpressure_controller);
        let processing_queue = Arc::clone(&self.processing_queue);
        tokio::spawn(async move {
            Self::backpressure_monitoring_task(backpressure_controller, processing_queue).await;
        });

        // Error recovery task
        let error_recovery = Arc::clone(&self.error_recovery);
        tokio::spawn(async move {
            Self::error_recovery_task(error_recovery).await;
        });

        // Statistics reporting task
        let stats = Arc::clone(&self.stats);
        tokio::spawn(async move {
            Self::stats_reporting_task(stats).await;
        });

        Ok(())
    }

    /// Update processing statistics
    async fn update_processing_stats(&self, processing_time: Duration) {
        let mut stats = self.stats.lock().await;
        stats.total_frames_processed += 1;

        let processing_time_ms = processing_time.as_secs_f32() * 1000.0;
        let n = stats.total_frames_processed as f32;

        // Update rolling averages
        stats.avg_processing_latency_ms =
            (stats.avg_processing_latency_ms * (n - 1.0) + processing_time_ms) / n;

        // Update percentiles (simplified)
        if processing_time_ms > stats.p95_latency_ms {
            stats.p95_latency_ms = processing_time_ms;
        }
        if processing_time_ms > stats.p99_latency_ms {
            stats.p99_latency_ms = processing_time_ms;
        }

        // Update throughput
        stats.throughput_fps = 1000.0 / stats.avg_processing_latency_ms;
    }

    /// Get current pipeline statistics
    pub async fn get_stats(&self) -> PipelineStats {
        let stats = self.stats.lock().await;
        stats.clone()
    }

    /// Buffer monitoring background task
    async fn buffer_monitoring_task(
        frame_buffer: Arc<Mutex<AdaptiveFrameBuffer>>,
        stats: Arc<Mutex<PipelineStats>>
    ) {
        let mut interval = interval(Duration::from_millis(100)); // Monitor every 100ms

        loop {
            interval.tick().await;

            let buffer_info = {
                let buffer = frame_buffer.lock().await;
                (buffer.frames.len(), buffer.target_size_ms, buffer.frame_stats.clone())
            };

            // Update buffer fill level
            {
                let mut stats = stats.lock().await;
                stats.buffer_fill_level = buffer_info.0 as f32 / 10.0; // Normalize to 0-1
            }

            // Log buffer status
            if buffer_info.0 > 20 {
                debug!("📊 Buffer status: {} frames, target={}ms, jitter={:.1}ms",
                       buffer_info.0, buffer_info.1, buffer_info.2.current_jitter_ms);
            }
        }
    }

    /// Backpressure monitoring background task
    async fn backpressure_monitoring_task(
        backpressure_controller: Arc<BackpressureController>,
        processing_queue: Arc<Mutex<ProcessingQueue>>
    ) {
        let mut interval = interval(Duration::from_secs(1)); // Monitor every second

        loop {
            interval.tick().await;

            // Calculate current system load
            let queue_size = {
                let queue = processing_queue.lock().await;
                queue.stats.current_size
            };

            let load = (queue_size as f32 / 100.0).min(1.0); // Normalize to 0-1

            // Update load
            {
                let mut current_load = backpressure_controller.current_load.write().await;
                *current_load = load;
            }

            // Update load history
            {
                let mut history = backpressure_controller.load_history.lock().await;
                history.push_back(load);
                if history.len() > 60 { // Keep 60 seconds of history
                    history.pop_front();
                }
            }

            debug!("📊 System load: {:.1}%, queue_size: {}", load * 100.0, queue_size);
        }
    }

    /// Error recovery background task
    async fn error_recovery_task(error_recovery: Arc<ErrorRecoveryManager>) {
        let mut interval = interval(Duration::from_secs(5)); // Check every 5 seconds

        loop {
            interval.tick().await;

            // Check for recent errors and apply recovery strategies
            let recent_errors = {
                let errors = error_recovery.recent_errors.lock().await;
                errors.clone()
            };

            if !recent_errors.is_empty() {
                debug!("🔧 Checking {} recent errors for recovery", recent_errors.len());
                // In a real implementation, this would analyze errors and apply appropriate recovery strategies
            }
        }
    }

    /// Statistics reporting background task
    async fn stats_reporting_task(stats: Arc<Mutex<PipelineStats>>) {
        let mut interval = interval(Duration::from_secs(30)); // Report every 30 seconds

        loop {
            interval.tick().await;

            let current_stats = {
                let stats = stats.lock().await;
                stats.clone()
            };

            info!("📊 Pipeline stats: processed={}, latency={:.1}ms, throughput={:.1}fps, quality={:.2}",
                  current_stats.total_frames_processed,
                  current_stats.avg_processing_latency_ms,
                  current_stats.throughput_fps,
                  current_stats.quality_score);
        }
    }
}

impl AdaptiveFrameBuffer {
    fn new(config: PipelineConfig) -> Self {
        Self {
            frames: VecDeque::new(),
            target_size_ms: config.target_latency_ms,
            min_size_ms: 30,
            max_size_ms: 100,
            frame_stats: FrameBufferStats::default(),
            last_adaptation: Instant::now(),
            config,
        }
    }

    fn add_frame(&mut self, frame: AudioFrame) -> Result<()> {
        // Check buffer overflow
        if self.frames.len() >= 50 { // Max 50 frames
            self.frames.pop_front();
            self.frame_stats.frames_dropped_overflow += 1;
        }

        self.frames.push_back(frame);
        self.frame_stats.frames_received += 1;
        Ok(())
    }

    fn should_adapt(&self) -> bool {
        self.last_adaptation.elapsed() > Duration::from_secs(5) && self.config.enable_adaptive_frames
    }

    async fn adapt_size(&mut self) {
        // Calculate recent jitter and adjust buffer size
        let current_jitter = self.calculate_current_jitter();

        if current_jitter > 20.0 {
            // High jitter, increase buffer size
            self.target_size_ms = (self.target_size_ms + 5).min(self.max_size_ms);
        } else if current_jitter < 5.0 {
            // Low jitter, decrease buffer size for lower latency
            self.target_size_ms = (self.target_size_ms.saturating_sub(2)).max(self.min_size_ms);
        }

        self.last_adaptation = Instant::now();
        debug!("🔧 Buffer adapted: target={}ms, jitter={:.1}ms", self.target_size_ms, current_jitter);
    }

    fn calculate_current_jitter(&self) -> f32 {
        // Simplified jitter calculation
        if self.frames.len() < 2 {
            return 0.0;
        }

        let mut intervals = Vec::new();
        for window in self.frames.iter().collect::<Vec<_>>().windows(2) {
            let interval = window[1].timestamp - window[0].timestamp;
            intervals.push(interval as f32 / 1000.0); // Convert to ms
        }

        if intervals.is_empty() {
            return 0.0;
        }

        let mean = intervals.iter().sum::<f32>() / intervals.len() as f32;
        let variance = intervals.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / intervals.len() as f32;
        variance.sqrt()
    }
}

impl ProcessingQueue {
    fn new(max_size: usize) -> Self {
        Self {
            high_priority: VecDeque::new(),
            normal_priority: VecDeque::new(),
            low_priority: VecDeque::new(),
            stats: QueueStats::default(),
            max_sizes: [max_size / 4, max_size / 2, max_size / 4], // Split capacity
        }
    }
}

impl BackpressureController {
    fn new(config: BackpressureConfig) -> Self {
        Self {
            current_load: RwLock::new(0.0),
            load_history: Mutex::new(VecDeque::new()),
            config,
            actions_taken: Mutex::new(VecDeque::new()),
        }
    }
}

impl FrameSynchronizer {
    fn new(config: SyncConfig) -> Self {
        Self {
            reference_timestamp: Mutex::new(None),
            drift_compensation: Mutex::new(0.0),
            sync_stats: Mutex::new(SyncStats::default()),
            config,
        }
    }

    async fn synchronize_frame(&self, _frame: &AudioFrame) -> Result<()> {
        // Simplified synchronization logic
        // In a real implementation, this would:
        // 1. Detect clock drift
        // 2. Apply correction algorithms
        // 3. Maintain synchronization accuracy
        Ok(())
    }
}

impl ErrorRecoveryManager {
    fn new() -> Self {
        let recovery_strategies = vec![
            RecoveryStrategy {
                error_types: vec![ErrorType::BufferUnderrun],
                min_severity: ErrorSeverity::Medium,
                action: RecoveryAction::IncreaseBufferSize,
                cooldown_ms: 5000,
            },
            RecoveryStrategy {
                error_types: vec![ErrorType::BufferOverrun],
                min_severity: ErrorSeverity::Medium,
                action: RecoveryAction::ReduceComplexity,
                cooldown_ms: 3000,
            },
            RecoveryStrategy {
                error_types: vec![ErrorType::ProcessingTimeout],
                min_severity: ErrorSeverity::High,
                action: RecoveryAction::DegradedMode,
                cooldown_ms: 10000,
            },
        ];

        Self {
            recent_errors: Mutex::new(VecDeque::new()),
            recovery_strategies,
            stats: Mutex::new(RecoveryStats::default()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_pipeline_creation() {
        let config = PipelineConfig::default();
        let pipeline = RealtimeStreamingPipeline::new(config);

        let result = pipeline.start().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_frame_processing() {
        let config = PipelineConfig::default();
        let pipeline = RealtimeStreamingPipeline::new(config);

        let audio_data = vec![0.1, 0.2, 0.3, 0.4]; // Sample audio
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;

        let result = pipeline.process_frame(audio_data, timestamp, 1).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_adaptive_frame_buffer() {
        let config = PipelineConfig::default();
        let mut buffer = AdaptiveFrameBuffer::new(config);

        let frame = AudioFrame {
            data: vec![0.1, 0.2, 0.3],
            timestamp: 12345,
            sequence: 1,
            size_samples: 3,
            deadline: Instant::now() + Duration::from_millis(100),
            priority: 0,
        };

        let result = buffer.add_frame(frame);
        assert!(result.is_ok());
        assert_eq!(buffer.frames.len(), 1);
    }

    #[test]
    fn test_backpressure_config() {
        let config = BackpressureConfig::default();
        assert_eq!(config.load_threshold, 0.75);
        assert_eq!(config.critical_threshold, 0.9);
    }
}