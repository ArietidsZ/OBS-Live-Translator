//! Performance Monitoring System
//!
//! This module provides comprehensive performance monitoring including:
//! - Latency tracking per component
//! - Resource utilization monitoring
//! - Quality metrics (WER, BLEU scores)
//! - Real-time performance dashboards
//! - Adaptive optimization with automatic profile switching
//! - Performance alert system

pub mod metrics;
pub mod dashboard;
pub mod adaptive_optimizer;
pub mod alerts;

use anyhow::Result;
use serde::{Serialize, Deserialize};
use std::time::{Duration, Instant};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error, debug};

pub use metrics::{
    MetricsCollector, ComponentMetrics, PerformanceMetrics,
    ResourceMetrics, QualityMetrics, LatencyMetrics
};
pub use dashboard::{
    PerformanceDashboard, DashboardConfig,
    MetricsVisualization
};
pub use adaptive_optimizer::{
    AdaptiveOptimizer, OptimizationAction, PerformanceTarget,
    AdaptiveConfig
};
pub use alerts::{
    AlertManager, AlertRule, AlertSeverity, AlertEvent,
    AlertChannel
};

/// Global performance monitoring manager
pub struct PerformanceMonitor {
    /// Metrics collector
    metrics_collector: MetricsCollector,
    /// Performance dashboard
    dashboard: PerformanceDashboard,
    /// Adaptive optimizer
    adaptive_optimizer: AdaptiveOptimizer,
    /// Alert manager
    alert_manager: AlertManager,
    /// Monitoring configuration
    config: MonitoringConfig,
    /// Monitoring state
    state: Arc<RwLock<MonitoringState>>,
}

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Enable monitoring
    pub enabled: bool,
    /// Metrics collection interval (milliseconds)
    pub collection_interval_ms: u64,
    /// Dashboard update interval (milliseconds)
    pub dashboard_update_interval_ms: u64,
    /// Enable adaptive optimization
    pub enable_adaptive_optimization: bool,
    /// Enable performance alerts
    pub enable_alerts: bool,
    /// Metrics retention period (hours)
    pub metrics_retention_hours: u32,
    /// Performance targets
    pub performance_targets: PerformanceTargets,
}

/// Performance targets for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTargets {
    /// Maximum end-to-end latency (ms)
    pub max_end_to_end_latency_ms: f32,
    /// Maximum CPU utilization (%)
    pub max_cpu_utilization_percent: f32,
    /// Maximum memory utilization (%)
    pub max_memory_utilization_percent: f32,
    /// Maximum GPU utilization (%) if available
    pub max_gpu_utilization_percent: Option<f32>,
    /// Minimum WER score (%)
    pub min_wer_score_percent: f32,
    /// Minimum BLEU score
    pub min_bleu_score: f32,
}

/// Monitoring state
#[derive(Debug, Clone)]
struct MonitoringState {
    /// Monitoring active
    active: bool,
    /// Start time
    start_time: Instant,
    /// Total samples collected
    total_samples: u64,
    /// Last alert time
    last_alert_time: Option<Instant>,
    /// Current optimization level
    optimization_level: u8,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            collection_interval_ms: 1000, // 1 second
            dashboard_update_interval_ms: 5000, // 5 seconds
            enable_adaptive_optimization: true,
            enable_alerts: true,
            metrics_retention_hours: 24,
            performance_targets: PerformanceTargets {
                max_end_to_end_latency_ms: 500.0,
                max_cpu_utilization_percent: 80.0,
                max_memory_utilization_percent: 85.0,
                max_gpu_utilization_percent: Some(90.0),
                min_wer_score_percent: 85.0,
                min_bleu_score: 0.7,
            },
        }
    }
}

impl PerformanceMonitor {
    /// Create new performance monitor
    pub async fn new(config: MonitoringConfig) -> Result<Self> {
        let metrics_collector = MetricsCollector::new(
            Duration::from_millis(config.collection_interval_ms)
        ).await?;

        let dashboard_config = DashboardConfig {
            update_interval_ms: config.dashboard_update_interval_ms,
            enable_real_time: true,
            max_data_points: 1000,
            chart_types: vec!["line", "gauge", "histogram"].iter().map(|s| s.to_string()).collect(),
            enable_export: true,
            theme: dashboard::DashboardTheme {
                primary_color: "#3498db".to_string(),
                background_color: "#2c3e50".to_string(),
                text_color: "#ecf0f1".to_string(),
                grid_color: "#34495e".to_string(),
            },
        };
        let dashboard = PerformanceDashboard::new(dashboard_config).await?;

        let adaptive_config = AdaptiveConfig {
            enabled: config.enable_adaptive_optimization,
            optimization_interval_s: 30,
            sensitivity: 0.7,
            max_optimization_level: 5,
            min_samples_for_decision: 10,
            optimization_cooldown_s: 60,
        };
        let adaptive_optimizer = AdaptiveOptimizer::new(adaptive_config).await?;

        let alert_manager = AlertManager::new().await?;

        let state = MonitoringState {
            active: false,
            start_time: Instant::now(),
            total_samples: 0,
            last_alert_time: None,
            optimization_level: 0,
        };

        Ok(Self {
            metrics_collector,
            dashboard,
            adaptive_optimizer,
            alert_manager,
            config,
            state: Arc::new(RwLock::new(state)),
        })
    }

    /// Start performance monitoring
    pub async fn start(&self) -> Result<()> {
        info!("🚀 Starting performance monitoring system");

        {
            let mut state = self.state.write().await;
            state.active = true;
            state.start_time = Instant::now();
        }

        // Start metrics collection
        self.metrics_collector.start().await?;

        // Start dashboard
        if self.config.enabled {
            self.dashboard.start().await?;
        }

        // Start adaptive optimizer
        if self.config.enable_adaptive_optimization {
            self.adaptive_optimizer.start().await?;
        }

        // Start alert manager
        if self.config.enable_alerts {
            self.alert_manager.start().await?;
        }

        // Start monitoring loop
        self.start_monitoring_loop().await;

        info!("✅ Performance monitoring system started successfully");
        Ok(())
    }

    /// Stop performance monitoring
    pub async fn stop(&self) -> Result<()> {
        info!("🛑 Stopping performance monitoring system");

        {
            let mut state = self.state.write().await;
            state.active = false;
        }

        // Stop components
        self.metrics_collector.stop().await?;
        self.dashboard.stop().await?;
        self.adaptive_optimizer.stop().await?;
        self.alert_manager.stop().await?;

        info!("✅ Performance monitoring system stopped");
        Ok(())
    }

    /// Start monitoring loop
    async fn start_monitoring_loop(&self) {
        let metrics_collector = self.metrics_collector.clone();
        let dashboard = self.dashboard.clone();
        let adaptive_optimizer = self.adaptive_optimizer.clone();
        let alert_manager = self.alert_manager.clone();
        let config = self.config.clone();
        let state_handle = self.state.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(
                Duration::from_millis(config.collection_interval_ms)
            );

            loop {
                interval.tick().await;

                let active = {
                    let state = state_handle.read().await;
                    state.active
                };

                if !active {
                    break;
                }

                // Collect metrics
                if let Ok(metrics) = metrics_collector.collect_current_metrics().await {
                    // Update sample count
                    {
                        let mut state = state_handle.write().await;
                        state.total_samples += 1;
                    }

                    // Update dashboard
                    if let Err(e) = dashboard.update_metrics(&metrics).await {
                        warn!("Failed to update dashboard: {}", e);
                    }

                    // Check for optimization opportunities
                    if config.enable_adaptive_optimization {
                        if let Ok(action) = adaptive_optimizer.analyze_metrics(&metrics).await {
                            if let Some(action) = action {
                                info!("🔧 Applying optimization: {:?}", action);
                                if let Err(e) = Self::apply_optimization_action(action).await {
                                    error!("Failed to apply optimization: {}", e);
                                }
                            }
                        }
                    }

                    // Check for alerts
                    if config.enable_alerts {
                        if let Err(e) = alert_manager.check_metrics(&metrics).await {
                            warn!("Alert check failed: {}", e);
                        }
                    }
                }
            }

            debug!("Monitoring loop stopped");
        });
    }

    /// Apply optimization action
    async fn apply_optimization_action(action: OptimizationAction) -> Result<()> {
        match action {
            OptimizationAction::ReduceQuality { target_quality } => {
                info!("🔽 Reducing quality to {:.1}%", target_quality * 100.0);
                // In real implementation, update model precision, reduce batch size, etc.
            },
            OptimizationAction::IncreaseResources { cpu_threads, memory_mb } => {
                info!("🔼 Increasing resources: {} threads, {}MB memory", cpu_threads, memory_mb);
                // In real implementation, adjust thread pool, memory allocation
            },
            OptimizationAction::SwitchProfile { target_profile } => {
                info!("🔄 Switching to {:?} profile", target_profile);
                // In real implementation, switch processing profile
            },
            OptimizationAction::ReduceLatency { target_latency_ms } => {
                info!("⚡ Optimizing for {:.1}ms latency", target_latency_ms);
                // In real implementation, reduce frame size, increase processing priority
            },
            OptimizationAction::OptimizeBatching { batch_size, timeout_ms } => {
                info!("📦 Optimizing batch processing: batch size {}, timeout {}ms", batch_size, timeout_ms);
                // In real implementation, adjust batch processing parameters
            },
            OptimizationAction::AdjustFrameProcessing { frame_size, hop_length } => {
                info!("🎞️ Adjusting frame processing: size {}, hop {}", frame_size, hop_length);
                // In real implementation, update audio frame processing parameters
            },
        }
        Ok(())
    }

    /// Get current performance snapshot
    pub async fn get_performance_snapshot(&self) -> Result<PerformanceSnapshot> {
        let current_metrics = self.metrics_collector.collect_current_metrics().await?;
        let dashboard_data = self.dashboard.get_current_data().await?;
        let optimization_status = self.adaptive_optimizer.get_status().await?;
        let alert_status = self.alert_manager.get_status().await?;

        let state = self.state.read().await;
        let uptime = state.start_time.elapsed();

        Ok(PerformanceSnapshot {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            uptime_seconds: uptime.as_secs(),
            current_metrics,
            dashboard_data: DashboardData {
                charts: dashboard_data.charts,
                data_points: dashboard_data.data_points,
            },
            optimization_status: OptimizationStatus {
                active: optimization_status.active,
                level: optimization_status.level,
                last_action: optimization_status.last_action,
            },
            alert_status: AlertStatus {
                active_alerts: alert_status.active_alerts,
                last_alert: alert_status.last_alert,
            },
            total_samples_collected: state.total_samples,
            monitoring_active: state.active,
        })
    }

    /// Get performance summary
    pub async fn get_performance_summary(&self) -> Result<PerformanceSummary> {
        let metrics_summary = self.metrics_collector.get_summary().await?;
        let dashboard_summary = self.dashboard.get_summary().await?;
        let optimization_summary = self.adaptive_optimizer.get_summary().await?;
        let alert_summary = self.alert_manager.get_summary().await?;

        Ok(PerformanceSummary {
            metrics_summary: MetricsSummary {
                avg_latency_ms: metrics_summary.avg_latency_ms,
                avg_cpu_percent: metrics_summary.avg_cpu_percent,
                avg_memory_percent: metrics_summary.avg_memory_percent,
            },
            dashboard_summary: DashboardSummary {
                total_charts: dashboard_summary.total_charts,
                update_frequency_ms: dashboard_summary.update_frequency_ms,
            },
            optimization_summary: OptimizationSummary {
                total_optimizations: optimization_summary.total_optimizations,
                success_rate: optimization_summary.success_rate,
            },
            alert_summary: AlertSummary {
                total_alerts: alert_summary.total_alerts as u32,
                critical_alerts: alert_summary.critical_alerts,
            },
            config: self.config.clone(),
        })
    }

    /// Export performance data
    pub async fn export_performance_data(&self, format: ExportFormat) -> Result<Vec<u8>> {
        let snapshot = self.get_performance_snapshot().await?;

        match format {
            ExportFormat::Json => {
                let json_data = serde_json::to_string_pretty(&snapshot)?;
                Ok(json_data.into_bytes())
            },
            ExportFormat::Csv => {
                // Convert metrics to CSV format
                let csv_data = self.convert_to_csv(&snapshot).await?;
                Ok(csv_data.into_bytes())
            },
            ExportFormat::Prometheus => {
                // Convert to Prometheus metrics format
                let prometheus_data = self.convert_to_prometheus(&snapshot).await?;
                Ok(prometheus_data.into_bytes())
            },
        }
    }

    /// Convert metrics to CSV format
    async fn convert_to_csv(&self, snapshot: &PerformanceSnapshot) -> Result<String> {
        let mut csv_data = String::new();
        csv_data.push_str("timestamp,component,metric,value\n");

        // Add latency metrics
        for (component, latency) in &snapshot.current_metrics.latency.component_latencies {
            csv_data.push_str(&format!("{},{},latency_ms,{:.2}\n",
                                     snapshot.timestamp, component, latency));
        }

        // Add resource metrics
        csv_data.push_str(&format!("{},system,cpu_percent,{:.2}\n",
                                 snapshot.timestamp, snapshot.current_metrics.resources.cpu_utilization_percent));
        csv_data.push_str(&format!("{},system,memory_percent,{:.2}\n",
                                 snapshot.timestamp, snapshot.current_metrics.resources.memory_utilization_percent));

        Ok(csv_data)
    }

    /// Convert metrics to Prometheus format
    async fn convert_to_prometheus(&self, snapshot: &PerformanceSnapshot) -> Result<String> {
        let mut prometheus_data = String::new();

        // Add latency metrics
        for (component, latency) in &snapshot.current_metrics.latency.component_latencies {
            prometheus_data.push_str(&format!(
                "obs_translator_latency_ms{{component=\"{}\"}} {:.2}\n",
                component, latency
            ));
        }

        // Add resource metrics
        prometheus_data.push_str(&format!(
            "obs_translator_cpu_utilization_percent {:.2}\n",
            snapshot.current_metrics.resources.cpu_utilization_percent
        ));
        prometheus_data.push_str(&format!(
            "obs_translator_memory_utilization_percent {:.2}\n",
            snapshot.current_metrics.resources.memory_utilization_percent
        ));

        // Add quality metrics
        prometheus_data.push_str(&format!(
            "obs_translator_wer_score {:.4}\n",
            snapshot.current_metrics.quality.wer_score
        ));
        prometheus_data.push_str(&format!(
            "obs_translator_bleu_score {:.4}\n",
            snapshot.current_metrics.quality.bleu_score
        ));

        Ok(prometheus_data)
    }

    /// Get metrics collector reference
    pub fn metrics_collector(&self) -> &MetricsCollector {
        &self.metrics_collector
    }

    /// Get dashboard reference
    pub fn dashboard(&self) -> &PerformanceDashboard {
        &self.dashboard
    }

    /// Get adaptive optimizer reference
    pub fn adaptive_optimizer(&self) -> &AdaptiveOptimizer {
        &self.adaptive_optimizer
    }

    /// Get alert manager reference
    pub fn alert_manager(&self) -> &AlertManager {
        &self.alert_manager
    }
}

/// Performance snapshot at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSnapshot {
    /// Snapshot timestamp
    pub timestamp: u64,
    /// System uptime (seconds)
    pub uptime_seconds: u64,
    /// Current performance metrics
    pub current_metrics: PerformanceMetrics,
    /// Dashboard data
    pub dashboard_data: DashboardData,
    /// Optimization status
    pub optimization_status: OptimizationStatus,
    /// Alert status
    pub alert_status: AlertStatus,
    /// Total samples collected
    pub total_samples_collected: u64,
    /// Monitoring active
    pub monitoring_active: bool,
}

/// Performance summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSummary {
    /// Metrics summary
    pub metrics_summary: MetricsSummary,
    /// Dashboard summary
    pub dashboard_summary: DashboardSummary,
    /// Optimization summary
    pub optimization_summary: OptimizationSummary,
    /// Alert summary
    pub alert_summary: AlertSummary,
    /// Configuration
    pub config: MonitoringConfig,
}

/// Export formats
#[derive(Debug, Clone)]
pub enum ExportFormat {
    Json,
    Csv,
    Prometheus,
}

// Placeholder types for external dependencies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardData {
    pub charts: Vec<String>,
    pub data_points: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationStatus {
    pub active: bool,
    pub level: u8,
    pub last_action: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertStatus {
    pub active_alerts: u32,
    pub last_alert: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSummary {
    pub avg_latency_ms: f32,
    pub avg_cpu_percent: f32,
    pub avg_memory_percent: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardSummary {
    pub total_charts: u32,
    pub update_frequency_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSummary {
    pub total_optimizations: u32,
    pub success_rate: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertSummary {
    pub total_alerts: u32,
    pub critical_alerts: u32,
}