//! Performance Dashboard System
//!
//! This module provides real-time performance visualization including:
//! - Live metrics dashboards
//! - Performance trend charts
//! - Resource utilization graphs
//! - Quality metrics visualization

use crate::monitoring::metrics::PerformanceMetrics;
use anyhow::Result;
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Mutex};
use tokio::time::interval;
use tracing::{info, debug};

/// Performance dashboard for real-time visualization
#[derive(Clone)]
pub struct PerformanceDashboard {
    /// Dashboard configuration
    config: DashboardConfig,
    /// Dashboard state
    state: Arc<RwLock<DashboardState>>,
    /// Metrics data for visualization
    visualization_data: Arc<Mutex<MetricsVisualization>>,
    /// Chart configurations
    chart_configs: Arc<RwLock<HashMap<String, ChartConfig>>>,
}

/// Dashboard configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    /// Update interval in milliseconds
    pub update_interval_ms: u64,
    /// Enable real-time updates
    pub enable_real_time: bool,
    /// Maximum data points to keep for charts
    pub max_data_points: usize,
    /// Chart types to display
    pub chart_types: Vec<String>,
    /// Enable export functionality
    pub enable_export: bool,
    /// Dashboard theme
    pub theme: DashboardTheme,
}

/// Dashboard theme configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardTheme {
    /// Primary color
    pub primary_color: String,
    /// Background color
    pub background_color: String,
    /// Text color
    pub text_color: String,
    /// Grid color
    pub grid_color: String,
}

/// Dashboard state
#[derive(Debug)]
struct DashboardState {
    /// Dashboard active
    active: bool,
    /// Start time
    start_time: Instant,
    /// Last update time
    last_update: Instant,
    /// Total updates processed
    total_updates: u64,
    /// Current metrics snapshot
    current_snapshot: Option<PerformanceMetrics>,
}

/// Metrics visualization data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsVisualization {
    /// Time series data for charts
    pub time_series: HashMap<String, TimeSeries>,
    /// Current gauge values
    pub gauges: HashMap<String, GaugeData>,
    /// Histogram data
    pub histograms: HashMap<String, HistogramData>,
    /// Status indicators
    pub status_indicators: HashMap<String, StatusIndicator>,
    /// Performance summary
    pub summary: VisualizationSummary,
}

/// Time series data for line charts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeries {
    /// Data points (timestamp, value)
    pub data_points: VecDeque<(u64, f32)>,
    /// Series label
    pub label: String,
    /// Series color
    pub color: String,
    /// Y-axis unit
    pub unit: String,
    /// Min/max values for scaling
    pub min_value: f32,
    pub max_value: f32,
}

/// Gauge data for circular/linear gauges
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaugeData {
    /// Current value
    pub value: f32,
    /// Minimum value
    pub min_value: f32,
    /// Maximum value
    pub max_value: f32,
    /// Warning threshold
    pub warning_threshold: f32,
    /// Critical threshold
    pub critical_threshold: f32,
    /// Unit of measurement
    pub unit: String,
    /// Gauge label
    pub label: String,
}

/// Histogram data for distribution visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramData {
    /// Histogram bins
    pub bins: Vec<HistogramBin>,
    /// Total sample count
    pub total_samples: u64,
    /// Statistics
    pub mean: f32,
    pub median: f32,
    pub std_dev: f32,
}

/// Histogram bin
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramBin {
    /// Bin start value
    pub start: f32,
    /// Bin end value
    pub end: f32,
    /// Count of values in this bin
    pub count: u64,
}

/// Status indicator for system health
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusIndicator {
    /// Current status
    pub status: IndicatorStatus,
    /// Status message
    pub message: String,
    /// Last update timestamp
    pub last_update: u64,
}

/// Indicator status levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IndicatorStatus {
    /// All systems operating normally
    Healthy,
    /// Warning conditions detected
    Warning,
    /// Critical issues detected
    Critical,
    /// System unavailable
    Unavailable,
}

/// Visualization summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualizationSummary {
    /// Overall system health
    pub overall_health: IndicatorStatus,
    /// Performance score (0-100)
    pub performance_score: f32,
    /// Active alerts count
    pub active_alerts: u32,
    /// Uptime in seconds
    pub uptime_seconds: u64,
    /// Key performance indicators
    pub kpis: HashMap<String, f32>,
}

/// Chart configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartConfig {
    /// Chart type
    pub chart_type: ChartType,
    /// Chart title
    pub title: String,
    /// Data source metric
    pub metric_source: String,
    /// Chart position and size
    pub layout: ChartLayout,
    /// Chart styling
    pub style: ChartStyle,
}

/// Chart types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChartType {
    /// Line chart for time series
    LineChart,
    /// Gauge for single values
    Gauge,
    /// Bar chart for comparisons
    BarChart,
    /// Histogram for distributions
    Histogram,
    /// Area chart for filled time series
    AreaChart,
}

/// Chart layout configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartLayout {
    /// Chart width
    pub width: u32,
    /// Chart height
    pub height: u32,
    /// Chart position X
    pub x: u32,
    /// Chart position Y
    pub y: u32,
}

/// Chart styling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartStyle {
    /// Colors for data series
    pub colors: Vec<String>,
    /// Font size
    pub font_size: u32,
    /// Show grid
    pub show_grid: bool,
    /// Show legend
    pub show_legend: bool,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            update_interval_ms: 1000,
            enable_real_time: true,
            max_data_points: 1000,
            chart_types: vec![
                "latency".to_string(),
                "cpu".to_string(),
                "memory".to_string(),
                "quality".to_string(),
            ],
            enable_export: true,
            theme: DashboardTheme {
                primary_color: "#3498db".to_string(),
                background_color: "#2c3e50".to_string(),
                text_color: "#ecf0f1".to_string(),
                grid_color: "#34495e".to_string(),
            },
        }
    }
}

impl Default for MetricsVisualization {
    fn default() -> Self {
        Self {
            time_series: HashMap::new(),
            gauges: HashMap::new(),
            histograms: HashMap::new(),
            status_indicators: HashMap::new(),
            summary: VisualizationSummary {
                overall_health: IndicatorStatus::Healthy,
                performance_score: 100.0,
                active_alerts: 0,
                uptime_seconds: 0,
                kpis: HashMap::new(),
            },
        }
    }
}

impl TimeSeries {
    fn new(label: String, color: String, unit: String) -> Self {
        Self {
            data_points: VecDeque::with_capacity(1000),
            label,
            color,
            unit,
            min_value: f32::MAX,
            max_value: f32::MIN,
        }
    }

    fn add_data_point(&mut self, timestamp: u64, value: f32, max_points: usize) {
        // Update min/max values
        self.min_value = self.min_value.min(value);
        self.max_value = self.max_value.max(value);

        // Add new data point
        self.data_points.push_back((timestamp, value));

        // Remove old data points if we exceed the limit
        while self.data_points.len() > max_points {
            self.data_points.pop_front();
        }
    }

    fn get_latest_value(&self) -> Option<f32> {
        self.data_points.back().map(|(_, value)| *value)
    }
}

impl PerformanceDashboard {
    /// Create new performance dashboard
    pub async fn new(config: DashboardConfig) -> Result<Self> {
        let state = DashboardState {
            active: false,
            start_time: Instant::now(),
            last_update: Instant::now(),
            total_updates: 0,
            current_snapshot: None,
        };

        let mut visualization_data = MetricsVisualization::default();
        Self::initialize_visualization_data(&mut visualization_data, &config).await?;

        let dashboard = Self {
            config: config.clone(),
            state: Arc::new(RwLock::new(state)),
            visualization_data: Arc::new(Mutex::new(visualization_data)),
            chart_configs: Arc::new(RwLock::new(HashMap::new())),
        };

        // Initialize default charts
        dashboard.initialize_default_charts().await?;

        Ok(dashboard)
    }

    /// Initialize visualization data structures
    async fn initialize_visualization_data(
        viz_data: &mut MetricsVisualization,
        config: &DashboardConfig,
    ) -> Result<()> {
        // Initialize time series for key metrics
        viz_data.time_series.insert(
            "latency".to_string(),
            TimeSeries::new(
                "Latency".to_string(),
                config.theme.primary_color.clone(),
                "ms".to_string(),
            ),
        );

        viz_data.time_series.insert(
            "cpu".to_string(),
            TimeSeries::new(
                "CPU Usage".to_string(),
                "#e74c3c".to_string(),
                "%".to_string(),
            ),
        );

        viz_data.time_series.insert(
            "memory".to_string(),
            TimeSeries::new(
                "Memory Usage".to_string(),
                "#f39c12".to_string(),
                "%".to_string(),
            ),
        );

        viz_data.time_series.insert(
            "quality".to_string(),
            TimeSeries::new(
                "Quality Score".to_string(),
                "#27ae60".to_string(),
                "score".to_string(),
            ),
        );

        // Initialize gauges
        viz_data.gauges.insert(
            "cpu_gauge".to_string(),
            GaugeData {
                value: 0.0,
                min_value: 0.0,
                max_value: 100.0,
                warning_threshold: 75.0,
                critical_threshold: 90.0,
                unit: "%".to_string(),
                label: "CPU Usage".to_string(),
            },
        );

        viz_data.gauges.insert(
            "memory_gauge".to_string(),
            GaugeData {
                value: 0.0,
                min_value: 0.0,
                max_value: 100.0,
                warning_threshold: 80.0,
                critical_threshold: 95.0,
                unit: "%".to_string(),
                label: "Memory Usage".to_string(),
            },
        );

        // Initialize status indicators
        viz_data.status_indicators.insert(
            "system_health".to_string(),
            StatusIndicator {
                status: IndicatorStatus::Healthy,
                message: "All systems operational".to_string(),
                last_update: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            },
        );

        Ok(())
    }

    /// Initialize default chart configurations
    async fn initialize_default_charts(&self) -> Result<()> {
        let mut charts = self.chart_configs.write().await;

        // Latency chart
        charts.insert(
            "latency_chart".to_string(),
            ChartConfig {
                chart_type: ChartType::LineChart,
                title: "Processing Latency".to_string(),
                metric_source: "latency".to_string(),
                layout: ChartLayout { width: 400, height: 200, x: 0, y: 0 },
                style: ChartStyle {
                    colors: vec![self.config.theme.primary_color.clone()],
                    font_size: 12,
                    show_grid: true,
                    show_legend: true,
                },
            },
        );

        // CPU gauge
        charts.insert(
            "cpu_gauge".to_string(),
            ChartConfig {
                chart_type: ChartType::Gauge,
                title: "CPU Usage".to_string(),
                metric_source: "cpu_gauge".to_string(),
                layout: ChartLayout { width: 200, height: 200, x: 420, y: 0 },
                style: ChartStyle {
                    colors: vec!["#e74c3c".to_string()],
                    font_size: 14,
                    show_grid: false,
                    show_legend: false,
                },
            },
        );

        // Memory gauge
        charts.insert(
            "memory_gauge".to_string(),
            ChartConfig {
                chart_type: ChartType::Gauge,
                title: "Memory Usage".to_string(),
                metric_source: "memory_gauge".to_string(),
                layout: ChartLayout { width: 200, height: 200, x: 640, y: 0 },
                style: ChartStyle {
                    colors: vec!["#f39c12".to_string()],
                    font_size: 14,
                    show_grid: false,
                    show_legend: false,
                },
            },
        );

        Ok(())
    }

    /// Start dashboard updates
    pub async fn start(&self) -> Result<()> {
        {
            let mut state = self.state.write().await;
            state.active = true;
            state.start_time = Instant::now();
        }

        if self.config.enable_real_time {
            self.start_update_loop().await;
        }

        info!("📊 Performance dashboard started");
        Ok(())
    }

    /// Stop dashboard updates
    pub async fn stop(&self) -> Result<()> {
        {
            let mut state = self.state.write().await;
            state.active = false;
        }

        info!("📊 Performance dashboard stopped");
        Ok(())
    }

    /// Start dashboard update loop
    async fn start_update_loop(&self) {
        let state = Arc::clone(&self.state);
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(config.update_interval_ms));

            loop {
                interval.tick().await;

                let active = {
                    let state_guard = state.read().await;
                    state_guard.active
                };

                if !active {
                    break;
                }

                // Dashboard update logic would go here
                // In a real implementation, this would generate dashboard updates
                debug!("Dashboard update tick");
            }

            debug!("Dashboard update loop stopped");
        });
    }

    /// Update dashboard with new metrics
    pub async fn update_metrics(&self, metrics: &PerformanceMetrics) -> Result<()> {
        let timestamp = metrics.timestamp;

        {
            let mut viz_data = self.visualization_data.lock().await;

            // Update time series
            if let Some(latency_series) = viz_data.time_series.get_mut("latency") {
                latency_series.add_data_point(
                    timestamp,
                    metrics.latency.avg_latency_ms,
                    self.config.max_data_points,
                );
            }

            if let Some(cpu_series) = viz_data.time_series.get_mut("cpu") {
                cpu_series.add_data_point(
                    timestamp,
                    metrics.resources.cpu_utilization_percent,
                    self.config.max_data_points,
                );
            }

            if let Some(memory_series) = viz_data.time_series.get_mut("memory") {
                memory_series.add_data_point(
                    timestamp,
                    metrics.resources.memory_utilization_percent,
                    self.config.max_data_points,
                );
            }

            if let Some(quality_series) = viz_data.time_series.get_mut("quality") {
                quality_series.add_data_point(
                    timestamp,
                    metrics.quality.overall_quality_score * 100.0,
                    self.config.max_data_points,
                );
            }

            // Update gauges
            if let Some(cpu_gauge) = viz_data.gauges.get_mut("cpu_gauge") {
                cpu_gauge.value = metrics.resources.cpu_utilization_percent;
            }

            if let Some(memory_gauge) = viz_data.gauges.get_mut("memory_gauge") {
                memory_gauge.value = metrics.resources.memory_utilization_percent;
            }

            // Update summary
            viz_data.summary.performance_score = Self::calculate_performance_score(metrics);
            viz_data.summary.overall_health = Self::determine_health_status(metrics);

            // Update KPIs
            viz_data.summary.kpis.insert("latency".to_string(), metrics.latency.avg_latency_ms);
            viz_data.summary.kpis.insert("cpu".to_string(), metrics.resources.cpu_utilization_percent);
            viz_data.summary.kpis.insert("quality".to_string(), metrics.quality.overall_quality_score);
        }

        // Update state
        {
            let mut state = self.state.write().await;
            state.last_update = Instant::now();
            state.total_updates += 1;
            state.current_snapshot = Some(metrics.clone());
            let _ = state.start_time.elapsed().as_secs();
        }

        Ok(())
    }

    /// Calculate overall performance score
    fn calculate_performance_score(metrics: &PerformanceMetrics) -> f32 {
        let latency_score = (500.0 - metrics.latency.avg_latency_ms).max(0.0) / 500.0 * 100.0;
        let cpu_score = (100.0 - metrics.resources.cpu_utilization_percent).max(0.0);
        let memory_score = (100.0 - metrics.resources.memory_utilization_percent).max(0.0);
        let quality_score = metrics.quality.overall_quality_score * 100.0;

        (latency_score + cpu_score + memory_score + quality_score) / 4.0
    }

    /// Determine overall health status
    fn determine_health_status(metrics: &PerformanceMetrics) -> IndicatorStatus {
        let high_latency = metrics.latency.avg_latency_ms > 1000.0;
        let high_cpu = metrics.resources.cpu_utilization_percent > 90.0;
        let high_memory = metrics.resources.memory_utilization_percent > 95.0;
        let low_quality = metrics.quality.overall_quality_score < 0.5;

        if high_latency || high_cpu || high_memory || low_quality {
            IndicatorStatus::Critical
        } else if metrics.latency.avg_latency_ms > 500.0
                 || metrics.resources.cpu_utilization_percent > 75.0
                 || metrics.resources.memory_utilization_percent > 80.0
                 || metrics.quality.overall_quality_score < 0.7 {
            IndicatorStatus::Warning
        } else {
            IndicatorStatus::Healthy
        }
    }

    /// Get current dashboard data
    pub async fn get_current_data(&self) -> Result<DashboardData> {
        let viz_data = self.visualization_data.lock().await;
        let _state = self.state.read().await;

        Ok(DashboardData {
            charts: self.config.chart_types.clone(),
            data_points: viz_data.time_series.len() as u32,
        })
    }

    /// Get dashboard summary
    pub async fn get_summary(&self) -> Result<DashboardSummary> {
        let _state = self.state.read().await;

        Ok(DashboardSummary {
            total_charts: self.config.chart_types.len() as u32,
            update_frequency_ms: self.config.update_interval_ms,
        })
    }

    /// Export dashboard data
    pub async fn export_data(&self, format: ExportFormat) -> Result<Vec<u8>> {
        let viz_data = self.visualization_data.lock().await;

        match format {
            ExportFormat::Json => {
                let json_data = serde_json::to_string_pretty(&*viz_data)?;
                Ok(json_data.into_bytes())
            },
            ExportFormat::Csv => {
                let csv_data = self.convert_to_csv(&viz_data).await?;
                Ok(csv_data.into_bytes())
            },
        }
    }

    /// Convert visualization data to CSV
    async fn convert_to_csv(&self, viz_data: &MetricsVisualization) -> Result<String> {
        let mut csv_data = String::new();
        csv_data.push_str("timestamp,metric,value\n");

        for (metric_name, time_series) in &viz_data.time_series {
            for (timestamp, value) in &time_series.data_points {
                csv_data.push_str(&format!("{},{},{:.2}\n", timestamp, metric_name, value));
            }
        }

        Ok(csv_data)
    }
}

/// Dashboard data structure for external access
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardData {
    /// Available charts
    pub charts: Vec<String>,
    /// Number of data points
    pub data_points: u32,
}

/// Dashboard summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardSummary {
    /// Total number of charts
    pub total_charts: u32,
    /// Update frequency in milliseconds
    pub update_frequency_ms: u64,
}

/// Export formats for dashboard data
#[derive(Debug, Clone)]
pub enum ExportFormat {
    Json,
    Csv,
}