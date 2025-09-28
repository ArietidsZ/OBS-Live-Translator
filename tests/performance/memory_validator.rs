//! Memory Usage Validation

use std::time::{Duration, Instant};
use std::collections::HashMap;
use sysinfo::{System, SystemExt, ProcessExt, Pid};

/// Memory usage snapshot
#[derive(Debug, Clone)]
pub struct MemorySnapshot {
    pub timestamp: Instant,
    pub heap_size_bytes: u64,
    pub resident_set_size: u64,
    pub virtual_memory_size: u64,
    pub available_memory: u64,
    pub total_memory: u64,
    pub memory_percent: f32,
}

/// Memory leak detection
#[derive(Debug)]
pub struct MemoryLeakDetector {
    baseline: Option<MemorySnapshot>,
    snapshots: Vec<MemorySnapshot>,
    growth_threshold_mb: f32,
    measurement_interval: Duration,
    system: System,
    pid: Pid,
}

impl MemoryLeakDetector {
    pub fn new(growth_threshold_mb: f32) -> Self {
        let mut system = System::new_all();
        system.refresh_all();
        let pid = sysinfo::get_current_pid().unwrap();

        Self {
            baseline: None,
            snapshots: Vec::new(),
            growth_threshold_mb,
            measurement_interval: Duration::from_secs(1),
            system,
            pid,
        }
    }

    /// Capture baseline memory usage
    pub fn capture_baseline(&mut self) {
        self.baseline = Some(self.capture_snapshot());
    }

    /// Capture current memory snapshot
    pub fn capture_snapshot(&mut self) -> MemorySnapshot {
        self.system.refresh_processes();
        self.system.refresh_memory();

        let process = self.system.process(self.pid).unwrap();
        
        MemorySnapshot {
            timestamp: Instant::now(),
            heap_size_bytes: process.memory() * 1024, // Convert KB to bytes
            resident_set_size: process.virtual_memory() * 1024,
            virtual_memory_size: process.virtual_memory() * 1024,
            available_memory: self.system.available_memory() * 1024,
            total_memory: self.system.total_memory() * 1024,
            memory_percent: (process.memory() as f32 / self.system.total_memory() as f32) * 100.0,
        }
    }

    /// Monitor memory for potential leaks
    pub async fn monitor_for_duration(&mut self, duration: Duration) -> MemoryLeakReport {
        let start = Instant::now();
        self.capture_baseline();

        while start.elapsed() < duration {
            tokio::time::sleep(self.measurement_interval).await;
            let snapshot = self.capture_snapshot();
            self.snapshots.push(snapshot);
        }

        self.analyze_for_leaks()
    }

    /// Analyze snapshots for memory leaks
    pub fn analyze_for_leaks(&self) -> MemoryLeakReport {
        let baseline = self.baseline.as_ref().expect("No baseline captured");
        
        let mut growth_events = Vec::new();
        let mut max_memory = baseline.heap_size_bytes;
        let mut sustained_growth = false;
        let mut growth_rate_mb_per_min = 0.0;

        if self.snapshots.len() > 10 {
            // Calculate growth rate using linear regression
            let (slope, _intercept) = self.calculate_memory_trend();
            growth_rate_mb_per_min = slope * 60.0 / (1024.0 * 1024.0);

            // Check for sustained growth
            let recent_average = self.snapshots
                .iter()
                .rev()
                .take(10)
                .map(|s| s.heap_size_bytes)
                .sum::<u64>() / 10;
            
            let baseline_mb = baseline.heap_size_bytes as f32 / (1024.0 * 1024.0);
            let recent_mb = recent_average as f32 / (1024.0 * 1024.0);
            
            if recent_mb - baseline_mb > self.growth_threshold_mb {
                sustained_growth = true;
            }
        }

        // Identify growth events
        for window in self.snapshots.windows(2) {
            let growth_mb = (window[1].heap_size_bytes as f32 - window[0].heap_size_bytes as f32) 
                / (1024.0 * 1024.0);
            
            if growth_mb > 1.0 { // Growth of more than 1MB
                growth_events.push(MemoryGrowthEvent {
                    timestamp: window[1].timestamp,
                    growth_mb,
                    total_mb: window[1].heap_size_bytes as f32 / (1024.0 * 1024.0),
                });
            }

            max_memory = max_memory.max(window[1].heap_size_bytes);
        }

        let total_growth_mb = if let Some(last) = self.snapshots.last() {
            (last.heap_size_bytes as f32 - baseline.heap_size_bytes as f32) / (1024.0 * 1024.0)
        } else {
            0.0
        };

        MemoryLeakReport {
            likely_leak: sustained_growth && growth_rate_mb_per_min > 1.0,
            total_growth_mb,
            growth_rate_mb_per_min,
            max_memory_mb: max_memory as f32 / (1024.0 * 1024.0),
            baseline_mb: baseline.heap_size_bytes as f32 / (1024.0 * 1024.0),
            growth_events,
            sustained_growth,
        }
    }

    /// Calculate memory trend using linear regression
    fn calculate_memory_trend(&self) -> (f64, f64) {
        let n = self.snapshots.len() as f64;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;

        let start_time = self.snapshots[0].timestamp;
        
        for (i, snapshot) in self.snapshots.iter().enumerate() {
            let x = (snapshot.timestamp - start_time).as_secs_f64();
            let y = snapshot.heap_size_bytes as f64;
            
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_x2 += x * x;
        }

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n;

        (slope, intercept)
    }
}

/// Memory growth event
#[derive(Debug)]
pub struct MemoryGrowthEvent {
    pub timestamp: Instant,
    pub growth_mb: f32,
    pub total_mb: f32,
}

/// Memory leak analysis report
#[derive(Debug)]
pub struct MemoryLeakReport {
    pub likely_leak: bool,
    pub total_growth_mb: f32,
    pub growth_rate_mb_per_min: f64,
    pub max_memory_mb: f32,
    pub baseline_mb: f32,
    pub growth_events: Vec<MemoryGrowthEvent>,
    pub sustained_growth: bool,
}

impl MemoryLeakReport {
    pub fn print_summary(&self) {
        println!("Memory Leak Analysis Report");
        println!("==========================\n");
        
        println!("Baseline Memory: {:.2} MB", self.baseline_mb);
        println!("Maximum Memory: {:.2} MB", self.max_memory_mb);
        println!("Total Growth: {:.2} MB", self.total_growth_mb);
        println!("Growth Rate: {:.2} MB/min\n", self.growth_rate_mb_per_min);

        if self.likely_leak {
            println!("⚠️  LIKELY MEMORY LEAK DETECTED");
            println!("Sustained growth: {}", self.sustained_growth);
            println!("Growth events: {}\n", self.growth_events.len());
            
            println!("Top Growth Events:");
            for (i, event) in self.growth_events.iter().take(5).enumerate() {
                println!("  {}. Growth: {:.2} MB, Total: {:.2} MB", 
                         i + 1, event.growth_mb, event.total_mb);
            }
        } else {
            println!("✅ No memory leaks detected");
        }
    }
}

/// Memory profiler for component-level analysis
pub struct MemoryProfiler {
    component_usage: HashMap<String, Vec<u64>>,
    start_time: Instant,
}

impl MemoryProfiler {
    pub fn new() -> Self {
        Self {
            component_usage: HashMap::new(),
            start_time: Instant::now(),
        }
    }

    /// Record memory usage for a component
    pub fn record_usage(&mut self, component: &str, bytes: u64) {
        self.component_usage
            .entry(component.to_string())
            .or_insert_with(Vec::new)
            .push(bytes);
    }

    /// Get memory statistics for a component
    pub fn get_component_stats(&self, component: &str) -> Option<MemoryStats> {
        self.component_usage.get(component).map(|usage| {
            let sum: u64 = usage.iter().sum();
            let count = usage.len() as f64;
            let mean = sum as f64 / count;
            
            let variance: f64 = usage
                .iter()
                .map(|&v| {
                    let diff = v as f64 - mean;
                    diff * diff
                })
                .sum::<f64>() / count;

            MemoryStats {
                mean_bytes: mean,
                std_dev_bytes: variance.sqrt(),
                min_bytes: *usage.iter().min().unwrap_or(&0),
                max_bytes: *usage.iter().max().unwrap_or(&0),
                total_allocations: usage.len(),
            }
        })
    }

    /// Generate memory profile report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str("Memory Profile Report\n");
        report.push_str("====================\n\n");

        for (component, usage) in &self.component_usage {
            if let Some(stats) = self.get_component_stats(component) {
                report.push_str(&format!("Component: {}\n", component));
                report.push_str(&format!("  Mean: {:.2} MB\n", stats.mean_bytes / (1024.0 * 1024.0)));
                report.push_str(&format!("  Max:  {:.2} MB\n", stats.max_bytes as f64 / (1024.0 * 1024.0)));
                report.push_str(&format!("  Allocations: {}\n\n", stats.total_allocations));
            }
        }

        report
    }
}

/// Memory statistics
#[derive(Debug)]
pub struct MemoryStats {
    pub mean_bytes: f64,
    pub std_dev_bytes: f64,
    pub min_bytes: u64,
    pub max_bytes: u64,
    pub total_allocations: usize,
}