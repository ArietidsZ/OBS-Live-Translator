//! Performance Regression Detection

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};

/// Performance regression detector
pub struct RegressionDetector {
    baseline_path: String,
    threshold_percent: f32,
    history: Vec<PerformanceRun>,
}

/// Performance run data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceRun {
    pub timestamp: DateTime<Utc>,
    pub git_commit: String,
    pub metrics: HashMap<String, MetricValue>,
    pub environment: EnvironmentInfo,
}

/// Metric value with statistical info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricValue {
    pub value: f64,
    pub unit: String,
    pub samples: u32,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
}

/// Environment information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentInfo {
    pub os: String,
    pub arch: String,
    pub cpu_model: String,
    pub cpu_cores: u32,
    pub memory_gb: f32,
    pub rust_version: String,
}

/// Regression analysis result
#[derive(Debug)]
pub struct RegressionAnalysis {
    pub has_regression: bool,
    pub regressions: Vec<RegressionDetail>,
    pub improvements: Vec<ImprovementDetail>,
    pub unchanged: Vec<String>,
}

/// Regression detail
#[derive(Debug)]
pub struct RegressionDetail {
    pub metric: String,
    pub baseline: f64,
    pub current: f64,
    pub change_percent: f64,
    pub severity: RegressionSeverity,
}

/// Improvement detail
#[derive(Debug)]
pub struct ImprovementDetail {
    pub metric: String,
    pub baseline: f64,
    pub current: f64,
    pub improvement_percent: f64,
}

/// Regression severity
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum RegressionSeverity {
    Minor,    // < 10% regression
    Moderate, // 10-25% regression
    Major,    // 25-50% regression
    Critical, // > 50% regression
}

impl RegressionDetector {
    pub fn new(baseline_path: String, threshold_percent: f32) -> Self {
        Self {
            baseline_path,
            threshold_percent,
            history: Vec::new(),
        }
    }

    /// Load baseline performance data
    pub fn load_baseline(&mut self) -> Result<PerformanceRun, Box<dyn std::error::Error>> {
        let data = fs::read_to_string(&self.baseline_path)?;
        let baseline: PerformanceRun = serde_json::from_str(&data)?;
        Ok(baseline)
    }

    /// Save current run as new baseline
    pub fn save_baseline(&self, run: &PerformanceRun) -> Result<(), Box<dyn std::error::Error>> {
        let data = serde_json::to_string_pretty(run)?;
        fs::write(&self.baseline_path, data)?;
        Ok(())
    }

    /// Analyze current performance against baseline
    pub fn analyze(&self, current: &PerformanceRun) -> Result<RegressionAnalysis, Box<dyn std::error::Error>> {
        let baseline = self.load_baseline()?;
        
        let mut regressions = Vec::new();
        let mut improvements = Vec::new();
        let mut unchanged = Vec::new();

        for (metric_name, current_metric) in &current.metrics {
            if let Some(baseline_metric) = baseline.metrics.get(metric_name) {
                let change_percent = ((current_metric.value - baseline_metric.value) 
                    / baseline_metric.value) * 100.0;

                if change_percent > self.threshold_percent {
                    // Performance regression (higher is worse)
                    let severity = self.classify_severity(change_percent);
                    regressions.push(RegressionDetail {
                        metric: metric_name.clone(),
                        baseline: baseline_metric.value,
                        current: current_metric.value,
                        change_percent,
                        severity,
                    });
                } else if change_percent < -self.threshold_percent {
                    // Performance improvement (lower is better)
                    improvements.push(ImprovementDetail {
                        metric: metric_name.clone(),
                        baseline: baseline_metric.value,
                        current: current_metric.value,
                        improvement_percent: -change_percent,
                    });
                } else {
                    unchanged.push(metric_name.clone());
                }
            }
        }

        // Sort by severity/impact
        regressions.sort_by(|a, b| b.severity.cmp(&a.severity));
        improvements.sort_by(|a, b| b.improvement_percent.partial_cmp(&a.improvement_percent).unwrap());

        Ok(RegressionAnalysis {
            has_regression: !regressions.is_empty(),
            regressions,
            improvements,
            unchanged,
        })
    }

    /// Classify regression severity
    fn classify_severity(&self, change_percent: f64) -> RegressionSeverity {
        if change_percent > 50.0 {
            RegressionSeverity::Critical
        } else if change_percent > 25.0 {
            RegressionSeverity::Major
        } else if change_percent > 10.0 {
            RegressionSeverity::Moderate
        } else {
            RegressionSeverity::Minor
        }
    }

    /// Track performance over time
    pub fn track_performance(&mut self, run: PerformanceRun) {
        self.history.push(run);
        
        // Keep only last 100 runs
        if self.history.len() > 100 {
            self.history.remove(0);
        }
    }

    /// Detect trends in performance
    pub fn detect_trends(&self, metric: &str, window_size: usize) -> Option<TrendAnalysis> {
        if self.history.len() < window_size {
            return None;
        }

        let mut values = Vec::new();
        for run in self.history.iter().rev().take(window_size) {
            if let Some(metric_value) = run.metrics.get(metric) {
                values.push(metric_value.value);
            }
        }

        if values.len() < window_size {
            return None;
        }

        // Calculate trend using linear regression
        let n = values.len() as f64;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;

        for (i, value) in values.iter().enumerate() {
            let x = i as f64;
            sum_x += x;
            sum_y += value;
            sum_xy += x * value;
            sum_x2 += x * x;
        }

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n;

        // Calculate R-squared
        let mean_y = sum_y / n;
        let mut ss_tot = 0.0;
        let mut ss_res = 0.0;

        for (i, value) in values.iter().enumerate() {
            let predicted = slope * i as f64 + intercept;
            ss_tot += (value - mean_y).powi(2);
            ss_res += (value - predicted).powi(2);
        }

        let r_squared = 1.0 - (ss_res / ss_tot);

        Some(TrendAnalysis {
            metric: metric.to_string(),
            slope,
            intercept,
            r_squared,
            direction: if slope > 0.01 {
                TrendDirection::Degrading
            } else if slope < -0.01 {
                TrendDirection::Improving
            } else {
                TrendDirection::Stable
            },
            window_size,
            latest_value: values[0],
            oldest_value: values[values.len() - 1],
        })
    }

    /// Generate regression report
    pub fn generate_report(&self, analysis: &RegressionAnalysis) -> String {
        let mut report = String::new();
        
        report.push_str("Performance Regression Analysis\n");
        report.push_str("==============================\n\n");

        if analysis.has_regression {
            report.push_str("⚠️  PERFORMANCE REGRESSIONS DETECTED\n\n");
            
            for regression in &analysis.regressions {
                let emoji = match regression.severity {
                    RegressionSeverity::Critical => "🔴",
                    RegressionSeverity::Major => "🟠",
                    RegressionSeverity::Moderate => "🟡",
                    RegressionSeverity::Minor => "🟢",
                };

                report.push_str(&format!(
                    "{} {} ({:?} regression)\n",
                    emoji, regression.metric, regression.severity
                ));
                report.push_str(&format!(
                    "    Baseline: {:.2}, Current: {:.2}, Change: +{:.1}%\n\n",
                    regression.baseline, regression.current, regression.change_percent
                ));
            }
        } else {
            report.push_str("✅ No performance regressions detected\n\n");
        }

        if !analysis.improvements.is_empty() {
            report.push_str("🎆 Performance Improvements:\n\n");
            for improvement in &analysis.improvements {
                report.push_str(&format!(
                    "  ✅ {}: -{:.1}% (was {:.2}, now {:.2})\n",
                    improvement.metric,
                    improvement.improvement_percent,
                    improvement.baseline,
                    improvement.current
                ));
            }
            report.push_str("\n");
        }

        if !analysis.unchanged.is_empty() {
            report.push_str(&format!(
                "📊 Unchanged metrics (within ±{}%): {}\n",
                self.threshold_percent,
                analysis.unchanged.join(", ")
            ));
        }

        report
    }
}

/// Trend analysis result
#[derive(Debug)]
pub struct TrendAnalysis {
    pub metric: String,
    pub slope: f64,
    pub intercept: f64,
    pub r_squared: f64,
    pub direction: TrendDirection,
    pub window_size: usize,
    pub latest_value: f64,
    pub oldest_value: f64,
}

/// Trend direction
#[derive(Debug, PartialEq)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
}

impl TrendAnalysis {
    pub fn print_summary(&self) {
        println!("Trend Analysis: {}", self.metric);
        println!("  Direction: {:?}", self.direction);
        println!("  Slope: {:.4}", self.slope);
        println!("  R²: {:.4}", self.r_squared);
        println!("  Latest: {:.2}", self.latest_value);
        println!("  Oldest: {:.2}", self.oldest_value);
        println!("  Change: {:.1}%", 
                 ((self.latest_value - self.oldest_value) / self.oldest_value * 100.0));
    }
}