// Logging and telemetry initialization
// Part 1.7: Enhanced logging with platform and performance metrics

use anyhow::Result;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::{fmt, prelude::*, EnvFilter, Registry};

use crate::execution_provider::ExecutionProviderConfig;
use crate::platform_detect::Platform;
use crate::profile::ProfileDetector;

/// Logging configuration
#[derive(Debug, Clone)]
pub struct LogConfig {
    /// Log level filter (e.g., "info", "debug", "trace")
    pub level: String,
    /// Enable JSON formatting for structured logs
    pub json_format: bool,
    /// Log to file in addition to stdout
    pub log_file: Option<String>,
    /// Enable performance span tracking
    pub enable_spans: bool,
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            json_format: false,
            log_file: None,
            enable_spans: true,
        }
    }
}

/// Initialize logging and telemetry
pub fn init_logging(config: LogConfig) -> Result<()> {
    // Create base filter from config
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&config.level));

    // Determine span events based on config
    let span_events = if config.enable_spans {
        FmtSpan::NEW | FmtSpan::CLOSE
    } else {
        FmtSpan::NONE
    };

    if config.json_format {
        // JSON structured logging for production
        let fmt_layer = fmt::layer()
            .json()
            .with_span_events(span_events)
            .with_current_span(true)
            .with_thread_ids(true)
            .with_thread_names(true);

        Registry::default().with(filter).with(fmt_layer).init();
    } else {
        // Human-readable logging for development
        let fmt_layer = fmt::layer()
            .with_span_events(span_events)
            .with_thread_ids(false)
            .with_thread_names(false)
            .pretty();

        Registry::default().with(filter).with(fmt_layer).init();
    }

    tracing::info!("OBS Live Translator v5.0 - Logging initialized");
    tracing::info!("Log level: {}", config.level);

    Ok(())
}

/// Log system information on startup
/// Uses Part 1.4, 1.5, and 1.6 for comprehensive system reporting
pub fn log_system_info() -> Result<()> {
    tracing::info!("=== System Information ===");

    // Platform detection (Part 1.4)
    let platform = Platform::detect()?;
    tracing::info!("Platform: {}", platform.description());

    // Execution provider selection (Part 1.5)
    let ep_config = ExecutionProviderConfig::from_platform(platform.clone());
    tracing::info!("Execution Provider: {}", ep_config.provider().name());
    tracing::info!(
        "Recommended Quantization: {}",
        ep_config.recommended_quantization()
    );

    // Profile detection (Part 1.6)
    let profile = ProfileDetector::detect()?;
    tracing::info!("Performance Profile: {:?}", profile);

    // Quantization support details
    tracing::info!("Hardware Capabilities:");
    tracing::info!("  - FP8 support: {}", platform.supports_quantization("fp8"));
    tracing::info!(
        "  - INT4 support: {}",
        platform.supports_quantization("int4")
    );
    tracing::info!(
        "  - INT8 support: {}",
        platform.supports_quantization("int8")
    );
    tracing::info!(
        "  - FP16 support: {}",
        platform.supports_quantization("fp16")
    );

    tracing::info!("=========================");

    Ok(())
}

/// Performance metrics tracking
pub mod metrics {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;

    /// Performance counters for telemetry
    #[derive(Debug, Clone)]
    pub struct PerformanceMetrics {
        pub vad_calls: Arc<AtomicU64>,
        pub asr_calls: Arc<AtomicU64>,
        pub translation_calls: Arc<AtomicU64>,
        pub total_latency_ms: Arc<AtomicU64>,
        pub error_count: Arc<AtomicU64>,
    }

    impl Default for PerformanceMetrics {
        fn default() -> Self {
            Self {
                vad_calls: Arc::new(AtomicU64::new(0)),
                asr_calls: Arc::new(AtomicU64::new(0)),
                translation_calls: Arc::new(AtomicU64::new(0)),
                total_latency_ms: Arc::new(AtomicU64::new(0)),
                error_count: Arc::new(AtomicU64::new(0)),
            }
        }
    }

    impl PerformanceMetrics {
        pub fn new() -> Self {
            Self::default()
        }

        pub fn record_vad(&self) {
            self.vad_calls.fetch_add(1, Ordering::Relaxed);
        }

        pub fn record_asr(&self) {
            self.asr_calls.fetch_add(1, Ordering::Relaxed);
        }

        pub fn record_translation(&self) {
            self.translation_calls.fetch_add(1, Ordering::Relaxed);
        }

        pub fn record_latency(&self, latency_ms: u64) {
            self.total_latency_ms
                .fetch_add(latency_ms, Ordering::Relaxed);
        }

        pub fn record_error(&self) {
            self.error_count.fetch_add(1, Ordering::Relaxed);
        }

        pub fn get_stats(&self) -> MetricsSnapshot {
            MetricsSnapshot {
                vad_calls: self.vad_calls.load(Ordering::Relaxed),
                asr_calls: self.asr_calls.load(Ordering::Relaxed),
                translation_calls: self.translation_calls.load(Ordering::Relaxed),
                total_latency_ms: self.total_latency_ms.load(Ordering::Relaxed),
                error_count: self.error_count.load(Ordering::Relaxed),
            }
        }

        pub fn log_stats(&self) {
            let stats = self.get_stats();
            tracing::info!("=== Performance Metrics ===");
            tracing::info!("VAD calls: {}", stats.vad_calls);
            tracing::info!("ASR calls: {}", stats.asr_calls);
            tracing::info!("Translation calls: {}", stats.translation_calls);
            tracing::info!("Total latency: {}ms", stats.total_latency_ms);
            tracing::info!("Error count: {}", stats.error_count);

            if stats.asr_calls > 0 {
                let avg_latency = stats.total_latency_ms / stats.asr_calls;
                tracing::info!("Average latency per request: {}ms", avg_latency);
            }
            tracing::info!("==========================");
        }
    }

    #[derive(Debug, Clone)]
    pub struct MetricsSnapshot {
        pub vad_calls: u64,
        pub asr_calls: u64,
        pub translation_calls: u64,
        pub total_latency_ms: u64,
        pub error_count: u64,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_config_default() {
        let config = LogConfig::default();
        assert_eq!(config.level, "info");
        assert!(!config.json_format);
        assert!(config.log_file.is_none());
        assert!(config.enable_spans);
    }

    #[test]
    fn test_performance_metrics() {
        let metrics = metrics::PerformanceMetrics::new();

        metrics.record_vad();
        metrics.record_asr();
        metrics.record_translation();
        metrics.record_latency(150);

        let stats = metrics.get_stats();
        assert_eq!(stats.vad_calls, 1);
        assert_eq!(stats.asr_calls, 1);
        assert_eq!(stats.translation_calls, 1);
        assert_eq!(stats.total_latency_ms, 150);
    }
}
