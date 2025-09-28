//! Resource monitoring and adaptive profile management

use crate::profile::Profile;
use anyhow::Result;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Resource monitoring and profile adaptation
pub struct ResourceMonitor {
    current_profile: Profile,
    metrics: Arc<RwLock<ResourceMetrics>>,
    history: Arc<RwLock<MetricsHistory>>,
    config: MonitorConfig,
}

/// Current resource utilization metrics
#[derive(Debug, Clone)]
pub struct ResourceMetrics {
    pub cpu_usage_percent: f32,
    pub memory_usage_mb: u64,
    pub memory_usage_percent: f32,
    pub gpu_memory_usage_mb: u64,
    pub gpu_memory_usage_percent: f32,
    pub current_latency_ms: f32,
    pub p99_latency_ms: f32,
    pub last_updated: Instant,
}

/// Historical metrics for trend analysis
#[derive(Debug)]
pub struct MetricsHistory {
    cpu_usage: VecDeque<(Instant, f32)>,
    memory_usage: VecDeque<(Instant, f32)>,
    gpu_memory_usage: VecDeque<(Instant, f32)>,
    latency: VecDeque<(Instant, f32)>,
    max_history_size: usize,
}

/// Configuration for resource monitoring
#[derive(Debug, Clone)]
pub struct MonitorConfig {
    pub sampling_interval: Duration,
    pub history_retention: Duration,
    pub upgrade_threshold_duration: Duration,
    pub downgrade_threshold_duration: Duration,
    pub latency_threshold_multiplier: f32,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            sampling_interval: Duration::from_secs(1),
            history_retention: Duration::from_secs(300), // 5 minutes
            upgrade_threshold_duration: Duration::from_secs(30),
            downgrade_threshold_duration: Duration::from_secs(10),
            latency_threshold_multiplier: 1.2,
        }
    }
}

impl ResourceMonitor {
    /// Create new resource monitor for the given profile
    pub fn new(profile: Profile) -> Self {
        Self {
            current_profile: profile,
            metrics: Arc::new(RwLock::new(ResourceMetrics::default())),
            history: Arc::new(RwLock::new(MetricsHistory::new())),
            config: MonitorConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(profile: Profile, config: MonitorConfig) -> Self {
        Self {
            current_profile: profile,
            metrics: Arc::new(RwLock::new(ResourceMetrics::default())),
            history: Arc::new(RwLock::new(MetricsHistory::new())),
            config,
        }
    }

    /// Start monitoring in background
    pub async fn start_monitoring(&self) -> Result<()> {
        let metrics = Arc::clone(&self.metrics);
        let history = Arc::clone(&self.history);
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(config.sampling_interval);

            loop {
                interval.tick().await;

                if let Ok(current_metrics) = Self::collect_system_metrics().await {
                    // Update current metrics
                    {
                        let mut metrics_guard = metrics.write().await;
                        *metrics_guard = current_metrics.clone();
                    }

                    // Update history
                    {
                        let mut history_guard = history.write().await;
                        history_guard.add_sample(current_metrics);
                        history_guard.cleanup_old_samples(config.history_retention);
                    }
                }
            }
        });

        Ok(())
    }

    /// Check if profile should be upgraded
    pub async fn should_upgrade(&self) -> bool {
        let metrics = self.metrics.read().await;
        let constraints = self.current_profile.resource_constraints();

        // Don't upgrade if already at highest profile
        if self.current_profile == Profile::High {
            return false;
        }

        // Check if resource usage is consistently low
        let cpu_headroom = constraints.max_cpu_percent - metrics.cpu_usage_percent;
        let memory_headroom = constraints.max_ram_percent - metrics.memory_usage_percent;

        // Upgrade if we have significant headroom and good performance
        cpu_headroom > 30.0 &&
        memory_headroom > 20.0 &&
        metrics.p99_latency_ms < (constraints.target_latency_ms as f32 * 0.7) &&
        self.has_sustained_performance(true).await
    }

    /// Check if profile should be downgraded
    pub async fn should_downgrade(&self) -> bool {
        let metrics = self.metrics.read().await;
        let constraints = self.current_profile.resource_constraints();

        // Don't downgrade if already at lowest profile
        if self.current_profile == Profile::Low {
            return false;
        }

        // Check if resource usage is too high or latency is poor
        let over_cpu_limit = metrics.cpu_usage_percent > constraints.max_cpu_percent;
        let over_memory_limit = metrics.memory_usage_percent > constraints.max_ram_percent;
        let over_latency_limit = metrics.p99_latency_ms > (constraints.target_latency_ms as f32 * self.config.latency_threshold_multiplier);

        (over_cpu_limit || over_memory_limit || over_latency_limit) &&
        self.has_sustained_performance(false).await
    }

    /// Get current resource metrics
    pub async fn get_current_metrics(&self) -> ResourceMetrics {
        self.metrics.read().await.clone()
    }

    /// Get performance trend analysis
    pub async fn get_performance_trend(&self) -> PerformanceTrend {
        let history = self.history.read().await;
        history.analyze_trends()
    }

    /// Update current profile
    pub fn set_current_profile(&mut self, profile: Profile) {
        self.current_profile = profile;
    }

    /// Check if performance has been sustained for required duration
    async fn has_sustained_performance(&self, is_upgrade_check: bool) -> bool {
        let history = self.history.read().await;
        let threshold_duration = if is_upgrade_check {
            self.config.upgrade_threshold_duration
        } else {
            self.config.downgrade_threshold_duration
        };

        history.has_sustained_condition(threshold_duration, is_upgrade_check, self.current_profile)
    }

    /// Collect current system metrics
    async fn collect_system_metrics() -> Result<ResourceMetrics> {
        let now = Instant::now();

        // Get CPU usage
        let cpu_usage = Self::get_cpu_usage().await?;

        // Get memory usage
        let (memory_mb, memory_percent) = Self::get_memory_usage().await?;

        // Get GPU memory usage
        let (gpu_memory_mb, gpu_memory_percent) = Self::get_gpu_memory_usage().await?;

        // Placeholder for latency - this would be measured by the actual processing pipeline
        let current_latency = 100.0; // ms
        let p99_latency = 120.0; // ms

        Ok(ResourceMetrics {
            cpu_usage_percent: cpu_usage,
            memory_usage_mb: memory_mb,
            memory_usage_percent: memory_percent,
            gpu_memory_usage_mb: gpu_memory_mb,
            gpu_memory_usage_percent: gpu_memory_percent,
            current_latency_ms: current_latency,
            p99_latency_ms: p99_latency,
            last_updated: now,
        })
    }

    /// Get current CPU usage percentage
    async fn get_cpu_usage() -> Result<f32> {
        #[cfg(target_os = "linux")]
        {
            use std::fs;

            // Read /proc/stat for CPU usage
            if let Ok(stat) = fs::read_to_string("/proc/stat") {
                if let Some(cpu_line) = stat.lines().next() {
                    let parts: Vec<u64> = cpu_line
                        .split_whitespace()
                        .skip(1)
                        .map(|s| s.parse().unwrap_or(0))
                        .collect();

                    if parts.len() >= 4 {
                        let idle = parts[3];
                        let total: u64 = parts.iter().sum();
                        let usage = 100.0 - (idle as f32 / total as f32 * 100.0);
                        return Ok(usage);
                    }
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            use std::process::Command;

            if let Ok(output) = Command::new("top")
                .args(&["-l", "1", "-s", "0", "-n", "0"])
                .output()
            {
                if let Ok(output_str) = String::from_utf8(output.stdout) {
                    for line in output_str.lines() {
                        if line.contains("CPU usage:") {
                            // Parse CPU usage from top output
                            // This is a simplified parser
                            if let Some(percent_str) = line.split_whitespace()
                                .find(|s| s.ends_with('%'))
                            {
                                if let Ok(usage) = percent_str.trim_end_matches('%').parse::<f32>() {
                                    return Ok(usage);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Fallback: estimate based on load average
        Ok(50.0) // Placeholder
    }

    /// Get current memory usage
    async fn get_memory_usage() -> Result<(u64, f32)> {
        #[cfg(target_os = "linux")]
        {
            use std::fs;

            if let Ok(meminfo) = fs::read_to_string("/proc/meminfo") {
                let mut total_kb = 0u64;
                let mut available_kb = 0u64;

                for line in meminfo.lines() {
                    if line.starts_with("MemTotal:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            total_kb = kb_str.parse().unwrap_or(0);
                        }
                    } else if line.starts_with("MemAvailable:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            available_kb = kb_str.parse().unwrap_or(0);
                        }
                    }
                }

                if total_kb > 0 {
                    let used_kb = total_kb - available_kb;
                    let used_mb = used_kb / 1024;
                    let used_percent = (used_kb as f32 / total_kb as f32) * 100.0;
                    return Ok((used_mb, used_percent));
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            use std::process::Command;

            if let Ok(output) = Command::new("vm_stat").output() {
                if let Ok(_output_str) = String::from_utf8(output.stdout) {
                    // Parse vm_stat output
                    // This is a simplified implementation
                    return Ok((4096, 60.0)); // Placeholder
                }
            }
        }

        // Fallback
        Ok((2048, 50.0)) // Placeholder
    }

    /// Get current GPU memory usage
    async fn get_gpu_memory_usage() -> Result<(u64, f32)> {
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        {
            use std::process::Command;

            // Try nvidia-smi first
            if let Ok(output) = Command::new("nvidia-smi")
                .args(&["--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"])
                .output()
            {
                if output.status.success() {
                    if let Ok(_output_str) = String::from_utf8(output.stdout) {
                        if let Some(line) = output_str.lines().next() {
                            let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                            if parts.len() >= 2 {
                                let used_mb: u64 = parts[0].parse().unwrap_or(0);
                                let total_mb: u64 = parts[1].parse().unwrap_or(1);
                                let used_percent = (used_mb as f32 / total_mb as f32) * 100.0;
                                return Ok((used_mb, used_percent));
                            }
                        }
                    }
                }
            }
        }

        // No GPU or unable to detect
        Ok((0, 0.0))
    }
}

impl Default for ResourceMetrics {
    fn default() -> Self {
        Self {
            cpu_usage_percent: 0.0,
            memory_usage_mb: 0,
            memory_usage_percent: 0.0,
            gpu_memory_usage_mb: 0,
            gpu_memory_usage_percent: 0.0,
            current_latency_ms: 0.0,
            p99_latency_ms: 0.0,
            last_updated: Instant::now(),
        }
    }
}

impl MetricsHistory {
    fn new() -> Self {
        Self {
            cpu_usage: VecDeque::new(),
            memory_usage: VecDeque::new(),
            gpu_memory_usage: VecDeque::new(),
            latency: VecDeque::new(),
            max_history_size: 1000,
        }
    }

    fn add_sample(&mut self, metrics: ResourceMetrics) {
        let now = metrics.last_updated;

        self.cpu_usage.push_back((now, metrics.cpu_usage_percent));
        self.memory_usage.push_back((now, metrics.memory_usage_percent));
        self.gpu_memory_usage.push_back((now, metrics.gpu_memory_usage_percent));
        self.latency.push_back((now, metrics.p99_latency_ms));

        // Limit history size
        if self.cpu_usage.len() > self.max_history_size {
            self.cpu_usage.pop_front();
            self.memory_usage.pop_front();
            self.gpu_memory_usage.pop_front();
            self.latency.pop_front();
        }
    }

    fn cleanup_old_samples(&mut self, retention: Duration) {
        let cutoff = Instant::now() - retention;

        while let Some(&(timestamp, _)) = self.cpu_usage.front() {
            if timestamp < cutoff {
                self.cpu_usage.pop_front();
                self.memory_usage.pop_front();
                self.gpu_memory_usage.pop_front();
                self.latency.pop_front();
            } else {
                break;
            }
        }
    }

    fn has_sustained_condition(&self, duration: Duration, is_upgrade_check: bool, current_profile: Profile) -> bool {
        let cutoff = Instant::now() - duration;
        let constraints = current_profile.resource_constraints();

        // Check if condition has been sustained for the required duration
        for &(timestamp, cpu_usage) in &self.cpu_usage {
            if timestamp < cutoff {
                continue;
            }

            if is_upgrade_check {
                // For upgrade: check if we have headroom
                if cpu_usage > constraints.max_cpu_percent - 30.0 {
                    return false;
                }
            } else {
                // For downgrade: check if we're over limits
                if cpu_usage <= constraints.max_cpu_percent {
                    return false;
                }
            }
        }

        true
    }

    fn analyze_trends(&self) -> PerformanceTrend {
        let cpu_trend = self.calculate_trend(&self.cpu_usage);
        let memory_trend = self.calculate_trend(&self.memory_usage);
        let latency_trend = self.calculate_trend(&self.latency);

        PerformanceTrend {
            cpu_trend,
            memory_trend,
            latency_trend,
            overall_trend: self.determine_overall_trend(cpu_trend, memory_trend, latency_trend),
        }
    }

    fn calculate_trend(&self, data: &VecDeque<(Instant, f32)>) -> TrendDirection {
        if data.len() < 10 {
            return TrendDirection::Stable;
        }

        let recent_samples = &data.iter().rev().take(10).collect::<Vec<_>>();
        let first_half: f32 = recent_samples.iter().skip(5).map(|(_, v)| *v).sum::<f32>() / 5.0;
        let second_half: f32 = recent_samples.iter().take(5).map(|(_, v)| *v).sum::<f32>() / 5.0;

        let change_percent = ((second_half - first_half) / first_half) * 100.0;

        if change_percent > 10.0 {
            TrendDirection::Increasing
        } else if change_percent < -10.0 {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        }
    }

    fn determine_overall_trend(&self, cpu: TrendDirection, memory: TrendDirection, latency: TrendDirection) -> TrendDirection {
        match (cpu, memory, latency) {
            (TrendDirection::Increasing, _, _) | (_, TrendDirection::Increasing, _) | (_, _, TrendDirection::Increasing) => {
                TrendDirection::Increasing
            }
            (TrendDirection::Decreasing, TrendDirection::Decreasing, _) |
            (TrendDirection::Decreasing, _, TrendDirection::Decreasing) |
            (_, TrendDirection::Decreasing, TrendDirection::Decreasing) => {
                TrendDirection::Decreasing
            }
            _ => TrendDirection::Stable
        }
    }
}

/// Performance trend analysis
#[derive(Debug, Clone)]
pub struct PerformanceTrend {
    pub cpu_trend: TrendDirection,
    pub memory_trend: TrendDirection,
    pub latency_trend: TrendDirection,
    pub overall_trend: TrendDirection,
}

/// Trend direction enumeration
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TrendDirection {
    Increasing,
    Stable,
    Decreasing,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_resource_monitor_creation() {
        let monitor = ResourceMonitor::new(Profile::Medium);
        assert_eq!(monitor.current_profile, Profile::Medium);
    }

    #[tokio::test]
    async fn test_metrics_collection() {
        let metrics = ResourceMonitor::collect_system_metrics().await.unwrap();
        assert!(metrics.cpu_usage_percent >= 0.0);
        assert!(metrics.memory_usage_percent >= 0.0);
    }

    #[test]
    fn test_metrics_history() {
        let mut history = MetricsHistory::new();
        let metrics = ResourceMetrics::default();

        history.add_sample(metrics);
        assert_eq!(history.cpu_usage.len(), 1);
    }
}