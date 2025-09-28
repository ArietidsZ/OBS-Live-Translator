//! Latency Measurement Suite

use std::time::{Duration, Instant};
use std::collections::HashMap;
use tokio::sync::mpsc;
use async_trait::async_trait;

/// Latency measurement point
#[derive(Debug, Clone)]
pub struct LatencyMeasurement {
    pub component: String,
    pub operation: String,
    pub start_time: Instant,
    pub end_time: Instant,
    pub latency: Duration,
    pub metadata: HashMap<String, String>,
}

/// Latency percentiles
#[derive(Debug, Clone)]
pub struct LatencyPercentiles {
    pub p50: Duration,
    pub p75: Duration,
    pub p90: Duration,
    pub p95: Duration,
    pub p99: Duration,
    pub p999: Duration,
    pub min: Duration,
    pub max: Duration,
    pub mean: Duration,
    pub std_dev: Duration,
}

/// Latency tracker for measuring component latencies
pub struct LatencyTracker {
    measurements: Vec<LatencyMeasurement>,
    active_operations: HashMap<String, Instant>,
    sender: mpsc::UnboundedSender<LatencyMeasurement>,
    receiver: mpsc::UnboundedReceiver<LatencyMeasurement>,
}

impl LatencyTracker {
    pub fn new() -> Self {
        let (sender, receiver) = mpsc::unbounded_channel();
        Self {
            measurements: Vec::new(),
            active_operations: HashMap::new(),
            sender,
            receiver,
        }
    }

    /// Start measuring an operation
    pub fn start_operation(&mut self, component: &str, operation: &str) -> String {
        let key = format!("{}::{}", component, operation);
        self.active_operations.insert(key.clone(), Instant::now());
        key
    }

    /// End measuring an operation
    pub fn end_operation(&mut self, key: &str, metadata: HashMap<String, String>) {
        if let Some(start_time) = self.active_operations.remove(key) {
            let end_time = Instant::now();
            let parts: Vec<&str> = key.split("::").collect();
            
            let measurement = LatencyMeasurement {
                component: parts[0].to_string(),
                operation: parts[1].to_string(),
                start_time,
                end_time,
                latency: end_time - start_time,
                metadata,
            };

            self.measurements.push(measurement.clone());
            let _ = self.sender.send(measurement);
        }
    }

    /// Calculate percentiles for a component
    pub fn calculate_percentiles(&self, component: &str) -> Option<LatencyPercentiles> {
        let mut latencies: Vec<Duration> = self.measurements
            .iter()
            .filter(|m| m.component == component)
            .map(|m| m.latency)
            .collect();

        if latencies.is_empty() {
            return None;
        }

        latencies.sort();
        let n = latencies.len();

        let percentile = |p: f64| -> Duration {
            let index = ((n as f64 - 1.0) * p / 100.0) as usize;
            latencies[index]
        };

        let mean = latencies.iter().sum::<Duration>() / n as u32;
        
        let variance: f64 = latencies
            .iter()
            .map(|&d| {
                let diff = d.as_secs_f64() - mean.as_secs_f64();
                diff * diff
            })
            .sum::<f64>() / n as f64;
        
        let std_dev = Duration::from_secs_f64(variance.sqrt());

        Some(LatencyPercentiles {
            p50: percentile(50.0),
            p75: percentile(75.0),
            p90: percentile(90.0),
            p95: percentile(95.0),
            p99: percentile(99.0),
            p999: percentile(99.9),
            min: latencies[0],
            max: latencies[n - 1],
            mean,
            std_dev,
        })
    }

    /// Get latency breakdown for end-to-end analysis
    pub fn get_latency_breakdown(&self) -> HashMap<String, Duration> {
        let mut breakdown = HashMap::new();
        
        for measurement in &self.measurements {
            let key = format!("{}::{}", measurement.component, measurement.operation);
            breakdown
                .entry(key)
                .and_modify(|e| *e += measurement.latency)
                .or_insert(measurement.latency);
        }
        
        breakdown
    }

    /// Export measurements to CSV
    pub fn export_csv(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(path)?;
        writeln!(file, "timestamp,component,operation,latency_ms,metadata")?;

        for measurement in &self.measurements {
            let metadata_str = measurement.metadata
                .iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect::<Vec<_>>()
                .join(";\u{0020}");

            writeln!(
                file,
                "{:?},{},{},{:.3},{}",
                measurement.start_time.elapsed(),
                measurement.component,
                measurement.operation,
                measurement.latency.as_secs_f64() * 1000.0,
                metadata_str
            )?;
        }

        Ok(())
    }
}

/// End-to-end latency test suite
pub struct E2ELatencyTester {
    tracker: LatencyTracker,
    test_duration: Duration,
    target_latencies: HashMap<String, Duration>,
}

impl E2ELatencyTester {
    pub fn new(test_duration: Duration) -> Self {
        let mut target_latencies = HashMap::new();
        
        // Define target latencies for different operations
        target_latencies.insert("audio_processing".to_string(), Duration::from_millis(10));
        target_latencies.insert("asr_inference".to_string(), Duration::from_millis(50));
        target_latencies.insert("translation".to_string(), Duration::from_millis(30));
        target_latencies.insert("streaming".to_string(), Duration::from_millis(5));
        target_latencies.insert("end_to_end".to_string(), Duration::from_millis(100));

        Self {
            tracker: LatencyTracker::new(),
            test_duration,
            target_latencies,
        }
    }

    /// Run comprehensive latency test
    pub async fn run_test(&mut self) -> LatencyTestReport {
        let start = Instant::now();
        let mut violations = Vec::new();

        while start.elapsed() < self.test_duration {
            // Test audio processing latency
            let key = self.tracker.start_operation("audio_processing", "process_frame");
            self.simulate_audio_processing().await;
            self.tracker.end_operation(&key, HashMap::new());

            // Test ASR latency
            let key = self.tracker.start_operation("asr_inference", "transcribe");
            self.simulate_asr().await;
            self.tracker.end_operation(&key, HashMap::new());

            // Test translation latency
            let key = self.tracker.start_operation("translation", "translate");
            self.simulate_translation().await;
            self.tracker.end_operation(&key, HashMap::new());

            // Test streaming latency
            let key = self.tracker.start_operation("streaming", "send_frame");
            self.simulate_streaming().await;
            self.tracker.end_operation(&key, HashMap::new());

            // Test end-to-end
            let key = self.tracker.start_operation("end_to_end", "full_pipeline");
            self.simulate_end_to_end().await;
            self.tracker.end_operation(&key, HashMap::new());

            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        // Check for violations
        for (component, target) in &self.target_latencies {
            if let Some(percentiles) = self.tracker.calculate_percentiles(component) {
                if percentiles.p95 > *target {
                    violations.push(LatencyViolation {
                        component: component.clone(),
                        target: *target,
                        actual_p95: percentiles.p95,
                        actual_p99: percentiles.p99,
                    });
                }
            }
        }

        LatencyTestReport {
            total_measurements: self.tracker.measurements.len(),
            test_duration: start.elapsed(),
            component_percentiles: self.get_all_percentiles(),
            violations,
            breakdown: self.tracker.get_latency_breakdown(),
        }
    }

    fn get_all_percentiles(&self) -> HashMap<String, LatencyPercentiles> {
        let mut percentiles = HashMap::new();
        
        for component in self.target_latencies.keys() {
            if let Some(p) = self.tracker.calculate_percentiles(component) {
                percentiles.insert(component.clone(), p);
            }
        }
        
        percentiles
    }

    async fn simulate_audio_processing(&self) {
        tokio::time::sleep(Duration::from_millis(8)).await;
    }

    async fn simulate_asr(&self) {
        tokio::time::sleep(Duration::from_millis(45)).await;
    }

    async fn simulate_translation(&self) {
        tokio::time::sleep(Duration::from_millis(25)).await;
    }

    async fn simulate_streaming(&self) {
        tokio::time::sleep(Duration::from_millis(3)).await;
    }

    async fn simulate_end_to_end(&self) {
        tokio::time::sleep(Duration::from_millis(85)).await;
    }
}

/// Latency test report
#[derive(Debug)]
pub struct LatencyTestReport {
    pub total_measurements: usize,
    pub test_duration: Duration,
    pub component_percentiles: HashMap<String, LatencyPercentiles>,
    pub violations: Vec<LatencyViolation>,
    pub breakdown: HashMap<String, Duration>,
}

/// Latency violation
#[derive(Debug)]
pub struct LatencyViolation {
    pub component: String,
    pub target: Duration,
    pub actual_p95: Duration,
    pub actual_p99: Duration,
}

impl LatencyTestReport {
    pub fn print_summary(&self) {
        println!("Latency Test Report");
        println!("==================\n");
        println!("Test Duration: {:?}", self.test_duration);
        println!("Total Measurements: {}\n", self.total_measurements);

        println!("Component Latencies (ms):");
        for (component, percentiles) in &self.component_percentiles {
            println!("  {}:", component);
            println!("    P50:  {:.2}", percentiles.p50.as_secs_f64() * 1000.0);
            println!("    P95:  {:.2}", percentiles.p95.as_secs_f64() * 1000.0);
            println!("    P99:  {:.2}", percentiles.p99.as_secs_f64() * 1000.0);
            println!("    Mean: {:.2}\n", percentiles.mean.as_secs_f64() * 1000.0);
        }

        if !self.violations.is_empty() {
            println!("VIOLATIONS:");
            for violation in &self.violations {
                println!("  {} exceeded target:", violation.component);
                println!("    Target: {:.2}ms", violation.target.as_secs_f64() * 1000.0);
                println!("    P95: {:.2}ms", violation.actual_p95.as_secs_f64() * 1000.0);
                println!("    P99: {:.2}ms\n", violation.actual_p99.as_secs_f64() * 1000.0);
            }
        } else {
            println!("All components within target latencies!");
        }
    }
}