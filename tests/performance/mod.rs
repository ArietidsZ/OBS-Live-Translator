//! Performance Validation Suite
//!
//! Comprehensive benchmarking and performance validation framework including:
//! - Real-time processing benchmarks
//! - Memory usage validation
//! - Latency measurement
//! - Performance regression testing

pub mod benchmarks;
pub mod latency_suite;
pub mod memory_validator;
pub mod regression_detector;

use std::time::{Duration, Instant};
use std::collections::HashMap;

/// Performance test result
#[derive(Debug, Clone)]
pub struct PerformanceTestResult {
    pub test_name: String,
    pub duration: Duration,
    pub iterations: u32,
    pub ops_per_second: f64,
    pub memory_usage_mb: f32,
    pub cpu_usage_percent: f32,
    pub passed: bool,
    pub notes: Vec<String>,
}

/// Performance baseline for regression testing
#[derive(Debug, Clone)]
pub struct PerformanceBaseline {
    pub test_name: String,
    pub avg_duration: Duration,
    pub max_duration: Duration,
    pub min_ops_per_second: f64,
    pub max_memory_mb: f32,
    pub max_cpu_percent: f32,
    pub tolerance_percent: f32,
}

/// Performance test harness
pub struct PerformanceTestHarness {
    results: Vec<PerformanceTestResult>,
    baselines: HashMap<String, PerformanceBaseline>,
    start_time: Instant,
}

impl PerformanceTestHarness {
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
            baselines: HashMap::new(),
            start_time: Instant::now(),
        }
    }

    pub fn load_baselines(&mut self, baselines: Vec<PerformanceBaseline>) {
        for baseline in baselines {
            self.baselines.insert(baseline.test_name.clone(), baseline);
        }
    }

    pub fn run_test<F>(&mut self, test_name: &str, iterations: u32, mut test_fn: F) -> PerformanceTestResult
    where
        F: FnMut() -> Result<(), Box<dyn std::error::Error>>,
    {
        let start = Instant::now();
        let mut errors = 0;

        for _ in 0..iterations {
            if let Err(e) = test_fn() {
                errors += 1;
                eprintln!("Test {} error: {}", test_name, e);
            }
        }

        let duration = start.elapsed();
        let ops_per_second = iterations as f64 / duration.as_secs_f64();

        let result = PerformanceTestResult {
            test_name: test_name.to_string(),
            duration,
            iterations,
            ops_per_second,
            memory_usage_mb: self.get_current_memory_mb(),
            cpu_usage_percent: self.get_current_cpu_percent(),
            passed: errors == 0,
            notes: if errors > 0 {
                vec![format!("{} errors occurred", errors)]
            } else {
                vec![]
            },
        };

        self.results.push(result.clone());
        result
    }

    pub fn check_regression(&self, result: &PerformanceTestResult) -> bool {
        if let Some(baseline) = self.baselines.get(&result.test_name) {
            let duration_ok = result.duration <= baseline.max_duration;
            let ops_ok = result.ops_per_second >= baseline.min_ops_per_second;
            let memory_ok = result.memory_usage_mb <= baseline.max_memory_mb;
            let cpu_ok = result.cpu_usage_percent <= baseline.max_cpu_percent;

            duration_ok && ops_ok && memory_ok && cpu_ok
        } else {
            true // No baseline, so no regression
        }
    }

    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str("Performance Test Report\n");
        report.push_str("======================\n\n");

        for result in &self.results {
            report.push_str(&format!("Test: {}\n", result.test_name));
            report.push_str(&format!("  Duration: {:?}\n", result.duration));
            report.push_str(&format!("  Iterations: {}\n", result.iterations));
            report.push_str(&format!("  Ops/sec: {:.2}\n", result.ops_per_second));
            report.push_str(&format!("  Memory: {:.2} MB\n", result.memory_usage_mb));
            report.push_str(&format!("  CPU: {:.1}%\n", result.cpu_usage_percent));
            report.push_str(&format!("  Status: {}\n", if result.passed { "PASS" } else { "FAIL" }));
            
            if let Some(baseline) = self.baselines.get(&result.test_name) {
                let regression = !self.check_regression(result);
                report.push_str(&format!("  Regression: {}\n", if regression { "YES" } else { "NO" }));
            }
            
            if !result.notes.is_empty() {
                report.push_str("  Notes:\n");
                for note in &result.notes {
                    report.push_str(&format!("    - {}\n", note));
                }
            }
            report.push_str("\n");
        }

        report.push_str(&format!("Total runtime: {:?}\n", self.start_time.elapsed()));
        report
    }

    fn get_current_memory_mb(&self) -> f32 {
        // Placeholder - use actual system metrics
        128.0
    }

    fn get_current_cpu_percent(&self) -> f32 {
        // Placeholder - use actual system metrics
        25.0
    }
}