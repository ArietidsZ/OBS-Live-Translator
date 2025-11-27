//! Language detection pipeline with real-time processing
//!
//! This module provides the complete language detection pipeline:
//! - Real-time language switching support
//! - Confidence scoring and validation
//! - Integration with ASR and translation systems
//! - Performance monitoring and optimization

use super::{DetectionStats, LanguageDetection, LanguageDetectionManager, LanguageDetectorConfig};
use crate::profile::Profile;
use anyhow::Result;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{mpsc, Mutex};
use tracing::{error, info, warn};

/// Language detection pipeline configuration
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Profile for detector selection
    pub profile: Profile,
    /// Detection configuration
    pub detector_config: LanguageDetectorConfig,
    /// Real-time processing settings
    pub real_time_config: RealTimeConfig,
    /// Performance monitoring settings
    pub monitoring_config: MonitoringConfig,
}

/// Real-time processing configuration
#[derive(Debug, Clone)]
pub struct RealTimeConfig {
    /// Maximum processing latency (ms)
    pub max_latency_ms: u64,
    /// Buffer size for text chunks
    pub text_buffer_size: usize,
    /// Minimum text length for processing
    pub min_text_length: usize,
    /// Maximum text length for processing
    pub max_text_length: usize,
    /// Enable batch processing
    pub enable_batching: bool,
    /// Batch timeout (ms)
    pub batch_timeout_ms: u64,
}

impl Default for RealTimeConfig {
    fn default() -> Self {
        Self {
            max_latency_ms: 50,
            text_buffer_size: 1000,
            min_text_length: 5,
            max_text_length: 2000,
            enable_batching: false,
            batch_timeout_ms: 10,
        }
    }
}

/// Performance monitoring configuration
#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    /// Enable performance metrics collection
    pub enable_metrics: bool,
    /// Metrics collection interval (seconds)
    pub metrics_interval_s: u64,
    /// Enable latency tracking
    pub track_latency: bool,
    /// Enable accuracy tracking
    pub track_accuracy: bool,
    /// Performance alert thresholds
    pub alert_thresholds: AlertThresholds,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enable_metrics: true,
            metrics_interval_s: 30,
            track_latency: true,
            track_accuracy: true,
            alert_thresholds: AlertThresholds::default(),
        }
    }
}

/// Performance alert thresholds
#[derive(Debug, Clone)]
pub struct AlertThresholds {
    /// Maximum acceptable latency (ms)
    pub max_latency_ms: f32,
    /// Minimum acceptable confidence
    pub min_confidence: f32,
    /// Maximum acceptable error rate
    pub max_error_rate: f32,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            max_latency_ms: 100.0,
            min_confidence: 0.6,
            max_error_rate: 0.05,
        }
    }
}

/// Language detection request
#[derive(Debug, Clone)]
pub struct DetectionRequest {
    /// Unique request ID
    pub id: String,
    /// Text to analyze
    pub text: String,
    /// Optional audio language hint
    pub audio_language_hint: Option<String>,
    /// Enable multimodal detection
    pub use_multimodal: bool,
    /// Request timestamp
    pub timestamp: Instant,
    /// Request priority
    pub priority: RequestPriority,
}

/// Request priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RequestPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Language detection response
#[derive(Debug, Clone)]
pub struct DetectionResponse {
    /// Request ID
    pub request_id: String,
    /// Detection result
    pub result: Result<LanguageDetection, String>,
    /// Processing timestamp
    pub processed_at: Instant,
    /// Total processing time (ms)
    pub total_processing_time_ms: f32,
}

/// Real-time language detection pipeline
pub struct LanguageDetectionPipeline {
    config: PipelineConfig,
    manager: Arc<Mutex<LanguageDetectionManager>>,

    // Processing channels
    request_sender: mpsc::UnboundedSender<DetectionRequest>,
    response_receiver: Arc<Mutex<mpsc::UnboundedReceiver<DetectionResponse>>>,

    // Performance monitoring
    performance_monitor: PerformanceMonitor,

    // Pipeline state
    is_running: Arc<Mutex<bool>>,
    stats: Arc<Mutex<PipelineStats>>,
}

/// Pipeline performance monitor
struct PerformanceMonitor {
    config: MonitoringConfig,
    metrics_history: Vec<PerformanceMetrics>,
    last_metrics_time: Instant,
    alert_callbacks: Vec<Box<dyn Fn(&PerformanceAlert) + Send + Sync>>,
}

/// Performance metrics snapshot
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub timestamp: Instant,
    pub requests_processed: u64,
    pub average_latency_ms: f32,
    pub error_rate: f32,
    pub average_confidence: f32,
    pub throughput_requests_per_second: f32,
    pub language_distribution: std::collections::HashMap<String, u64>,
}

/// Performance alert
#[derive(Debug, Clone)]
pub struct PerformanceAlert {
    pub alert_type: AlertType,
    pub message: String,
    pub metric_value: f32,
    pub threshold: f32,
    pub timestamp: Instant,
}

/// Alert types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AlertType {
    HighLatency,
    LowConfidence,
    HighErrorRate,
    SystemOverload,
}

/// Pipeline statistics
#[derive(Debug, Clone, Default)]
pub struct PipelineStats {
    pub total_requests: u64,
    pub successful_detections: u64,
    pub failed_detections: u64,
    pub average_processing_time_ms: f32,
    pub peak_processing_time_ms: f32,
    pub cache_hit_rate: f32,
    pub uptime_seconds: f64,
    pub start_time: Option<Instant>,
}

impl LanguageDetectionPipeline {
    /// Create a new language detection pipeline
    pub fn new(config: PipelineConfig) -> Result<Self> {
        info!(
            "🚀 Initializing Language Detection Pipeline: {:?} profile",
            config.profile
        );

        // Create detection manager
        let manager =
            LanguageDetectionManager::new(config.profile, config.detector_config.clone())?;
        let manager = Arc::new(Mutex::new(manager));

        // Create processing channels
        let (request_sender, request_receiver) = mpsc::unbounded_channel();
        let (response_sender, response_receiver) = mpsc::unbounded_channel();
        let response_receiver = Arc::new(Mutex::new(response_receiver));

        // Create performance monitor
        let performance_monitor = PerformanceMonitor::new(config.monitoring_config.clone());

        // Create pipeline state
        let is_running = Arc::new(Mutex::new(false));
        let stats = Arc::new(Mutex::new(PipelineStats::default()));

        let pipeline = Self {
            config,
            manager,
            request_sender,
            response_receiver,
            performance_monitor,
            is_running,
            stats,
        };

        // Start processing task
        pipeline.start_processing_task(request_receiver, response_sender)?;

        info!("✅ Language Detection Pipeline initialized successfully");
        Ok(pipeline)
    }

    /// Initialize the pipeline
    pub async fn initialize(&mut self) -> Result<()> {
        info!("🔧 Initializing language detection pipeline components");

        // Initialize detection manager
        {
            let mut manager = self.manager.lock().await;
            manager.initialize(self.config.detector_config.clone())?;
        }

        // Start pipeline
        {
            let mut is_running = self.is_running.lock().await;
            *is_running = true;
        }

        // Initialize stats
        {
            let mut stats = self.stats.lock().await;
            stats.start_time = Some(Instant::now());
        }

        info!("✅ Language detection pipeline initialized");
        Ok(())
    }

    /// Submit a detection request
    pub async fn detect(&self, request: DetectionRequest) -> Result<()> {
        let is_running = *self.is_running.lock().await;
        if !is_running {
            return Err(anyhow::anyhow!("Pipeline is not running"));
        }

        // Validate request
        self.validate_request(&request)?;

        // Submit to processing queue
        self.request_sender
            .send(request)
            .map_err(|e| anyhow::anyhow!("Failed to submit request: {}", e))?;

        Ok(())
    }

    /// Get the next detection response
    pub async fn get_response(&self) -> Option<DetectionResponse> {
        let mut receiver = self.response_receiver.lock().await;
        receiver.recv().await
    }

    /// Detect language synchronously (convenience method)
    pub async fn detect_sync(
        &self,
        text: &str,
        audio_hint: Option<&str>,
    ) -> Result<LanguageDetection> {
        let request = DetectionRequest {
            id: format!(
                "sync-{}",
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ),
            text: text.to_string(),
            audio_language_hint: audio_hint.map(|s| s.to_string()),
            use_multimodal: audio_hint.is_some() && self.config.profile == Profile::High,
            timestamp: Instant::now(),
            priority: RequestPriority::Normal,
        };

        // Submit request
        self.detect(request).await?;

        // Wait for response
        if let Some(response) = self.get_response().await {
            response.result.map_err(|e| anyhow::anyhow!(e))
        } else {
            Err(anyhow::anyhow!("No response received"))
        }
    }

    /// Get pipeline statistics
    pub async fn get_stats(&self) -> PipelineStats {
        let stats = self.stats.lock().await;
        let mut current_stats = stats.clone();

        // Update uptime
        if let Some(start_time) = current_stats.start_time {
            current_stats.uptime_seconds = start_time.elapsed().as_secs_f64();
        }

        current_stats
    }

    /// Get detection manager statistics
    pub async fn get_detection_stats(&self) -> DetectionStats {
        let manager = self.manager.lock().await;
        manager.stats().clone()
    }

    /// Shutdown the pipeline
    pub async fn shutdown(&self) -> Result<()> {
        info!("🔄 Shutting down language detection pipeline");

        let mut is_running = self.is_running.lock().await;
        *is_running = false;

        info!("✅ Language detection pipeline shut down");
        Ok(())
    }

    /// Start the background processing task
    fn start_processing_task(
        &self,
        mut request_receiver: mpsc::UnboundedReceiver<DetectionRequest>,
        response_sender: mpsc::UnboundedSender<DetectionResponse>,
    ) -> Result<()> {
        let manager = Arc::clone(&self.manager);
        let is_running = Arc::clone(&self.is_running);
        let stats = Arc::clone(&self.stats);
        let real_time_config = self.config.real_time_config.clone();

        tokio::spawn(async move {
            info!("🔄 Starting language detection processing task");

            while *is_running.lock().await {
                if let Some(request) = request_receiver.recv().await {
                    let start_time = Instant::now();

                    // Process request
                    let response =
                        Self::process_request(&manager, request, &real_time_config).await;

                    // Update statistics
                    Self::update_stats(&stats, &response, start_time).await;

                    // Send response
                    if let Err(e) = response_sender.send(response) {
                        error!("Failed to send detection response: {}", e);
                        break;
                    }
                } else {
                    // Channel closed, exit
                    break;
                }
            }

            info!("🔄 Language detection processing task stopped");
        });

        Ok(())
    }

    /// Process a single detection request
    async fn process_request(
        manager: &Arc<Mutex<LanguageDetectionManager>>,
        request: DetectionRequest,
        config: &RealTimeConfig,
    ) -> DetectionResponse {
        let start_time = Instant::now();

        // Check latency constraints
        let request_age = start_time.duration_since(request.timestamp).as_millis() as u64;
        if request_age > config.max_latency_ms {
            warn!(
                "Request {} exceeded latency constraint: {}ms > {}ms",
                request.id, request_age, config.max_latency_ms
            );

            return DetectionResponse {
                request_id: request.id,
                result: Err("Request exceeded latency constraint".to_string()),
                processed_at: Instant::now(),
                total_processing_time_ms: request_age as f32,
            };
        }

        // Validate text length
        if request.text.len() < config.min_text_length {
            return DetectionResponse {
                request_id: request.id,
                result: Err("Text too short for reliable detection".to_string()),
                processed_at: Instant::now(),
                total_processing_time_ms: start_time.elapsed().as_secs_f32() * 1000.0,
            };
        }

        if request.text.len() > config.max_text_length {
            warn!(
                "Text truncated for request {}: {} > {} characters",
                request.id,
                request.text.len(),
                config.max_text_length
            );
        }

        // Truncate text if necessary
        let processed_text = if request.text.len() > config.max_text_length {
            request.text.chars().take(config.max_text_length).collect()
        } else {
            request.text
        };

        // Perform detection
        let result = {
            let mut manager_guard = manager.lock().await;
            if request.use_multimodal {
                manager_guard
                    .detect_multimodal(&processed_text, request.audio_language_hint.as_deref())
            } else {
                manager_guard.detect(&processed_text)
            }
        };

        let total_processing_time = start_time.elapsed().as_secs_f32() * 1000.0;

        DetectionResponse {
            request_id: request.id,
            result: result.map_err(|e| e.to_string()),
            processed_at: Instant::now(),
            total_processing_time_ms: total_processing_time,
        }
    }

    /// Update pipeline statistics
    async fn update_stats(
        stats: &Arc<Mutex<PipelineStats>>,
        response: &DetectionResponse,
        _start_time: Instant,
    ) {
        let mut stats_guard = stats.lock().await;

        stats_guard.total_requests += 1;

        match &response.result {
            Ok(_) => stats_guard.successful_detections += 1,
            Err(_) => stats_guard.failed_detections += 1,
        }

        // Update average processing time
        let processing_time = response.total_processing_time_ms;
        let total_requests = stats_guard.total_requests as f32;

        stats_guard.average_processing_time_ms =
            (stats_guard.average_processing_time_ms * (total_requests - 1.0) + processing_time)
                / total_requests;

        // Update peak processing time
        if processing_time > stats_guard.peak_processing_time_ms {
            stats_guard.peak_processing_time_ms = processing_time;
        }
    }

    /// Validate detection request
    fn validate_request(&self, request: &DetectionRequest) -> Result<()> {
        if request.text.is_empty() {
            return Err(anyhow::anyhow!("Empty text in request"));
        }

        if request.text.len() > self.config.real_time_config.max_text_length * 2 {
            return Err(anyhow::anyhow!("Text too long for processing"));
        }

        Ok(())
    }
}

impl PerformanceMonitor {
    fn new(config: MonitoringConfig) -> Self {
        Self {
            config,
            metrics_history: Vec::new(),
            last_metrics_time: Instant::now(),
            alert_callbacks: Vec::new(),
        }
    }

    /// Add performance alert callback
    pub fn add_alert_callback<F>(&mut self, callback: F)
    where
        F: Fn(&PerformanceAlert) + Send + Sync + 'static,
    {
        self.alert_callbacks.push(Box::new(callback));
    }

    /// Collect current performance metrics
    pub fn collect_metrics(&mut self, stats: &PipelineStats) -> PerformanceMetrics {
        let metrics = PerformanceMetrics {
            timestamp: Instant::now(),
            requests_processed: stats.total_requests,
            average_latency_ms: stats.average_processing_time_ms,
            error_rate: if stats.total_requests > 0 {
                stats.failed_detections as f32 / stats.total_requests as f32
            } else {
                0.0
            },
            average_confidence: 0.8, // Would be calculated from actual detections
            throughput_requests_per_second: if stats.uptime_seconds > 0.0 {
                stats.total_requests as f32 / stats.uptime_seconds as f32
            } else {
                0.0
            },
            language_distribution: std::collections::HashMap::new(), // Would be populated from actual data
        };

        self.metrics_history.push(metrics.clone());

        // Limit history size
        if self.metrics_history.len() > 1000 {
            self.metrics_history.remove(0);
        }

        // Check for alerts
        self.check_alerts(&metrics);

        metrics
    }

    /// Check for performance alerts
    fn check_alerts(&self, metrics: &PerformanceMetrics) {
        let thresholds = &self.config.alert_thresholds;

        // Check latency
        if metrics.average_latency_ms > thresholds.max_latency_ms {
            let alert = PerformanceAlert {
                alert_type: AlertType::HighLatency,
                message: format!(
                    "High latency detected: {:.1}ms > {:.1}ms",
                    metrics.average_latency_ms, thresholds.max_latency_ms
                ),
                metric_value: metrics.average_latency_ms,
                threshold: thresholds.max_latency_ms,
                timestamp: Instant::now(),
            };
            self.trigger_alert(&alert);
        }

        // Check error rate
        if metrics.error_rate > thresholds.max_error_rate {
            let alert = PerformanceAlert {
                alert_type: AlertType::HighErrorRate,
                message: format!(
                    "High error rate detected: {:.3} > {:.3}",
                    metrics.error_rate, thresholds.max_error_rate
                ),
                metric_value: metrics.error_rate,
                threshold: thresholds.max_error_rate,
                timestamp: Instant::now(),
            };
            self.trigger_alert(&alert);
        }

        // Check confidence
        if metrics.average_confidence < thresholds.min_confidence {
            let alert = PerformanceAlert {
                alert_type: AlertType::LowConfidence,
                message: format!(
                    "Low confidence detected: {:.3} < {:.3}",
                    metrics.average_confidence, thresholds.min_confidence
                ),
                metric_value: metrics.average_confidence,
                threshold: thresholds.min_confidence,
                timestamp: Instant::now(),
            };
            self.trigger_alert(&alert);
        }
    }

    /// Trigger performance alert
    fn trigger_alert(&self, alert: &PerformanceAlert) {
        warn!("Performance alert: {}", alert.message);

        for callback in &self.alert_callbacks {
            callback(alert);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_pipeline_creation() {
        let config = PipelineConfig {
            profile: Profile::Medium,
            detector_config: LanguageDetectorConfig::default(),
            real_time_config: RealTimeConfig::default(),
            monitoring_config: MonitoringConfig::default(),
        };

        let pipeline = LanguageDetectionPipeline::new(config);
        assert!(pipeline.is_ok());
    }

    #[test]
    fn test_detection_request() {
        let request = DetectionRequest {
            id: "test-123".to_string(),
            text: "Hello world".to_string(),
            audio_language_hint: Some("en".to_string()),
            use_multimodal: true,
            timestamp: Instant::now(),
            priority: RequestPriority::Normal,
        };

        assert_eq!(request.id, "test-123");
        assert_eq!(request.text, "Hello world");
        assert!(request.use_multimodal);
    }

    #[test]
    fn test_request_priority_ordering() {
        assert!(RequestPriority::Critical > RequestPriority::High);
        assert!(RequestPriority::High > RequestPriority::Normal);
        assert!(RequestPriority::Normal > RequestPriority::Low);
    }

    #[test]
    fn test_performance_metrics() {
        let metrics = PerformanceMetrics {
            timestamp: Instant::now(),
            requests_processed: 100,
            average_latency_ms: 15.5,
            error_rate: 0.02,
            average_confidence: 0.87,
            throughput_requests_per_second: 50.0,
            language_distribution: std::collections::HashMap::new(),
        };

        assert_eq!(metrics.requests_processed, 100);
        assert_eq!(metrics.error_rate, 0.02);
        assert!(metrics.average_confidence > 0.8);
    }
}
