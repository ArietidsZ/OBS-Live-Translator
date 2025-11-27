//! Performance metrics collection

use std::collections::HashMap;
use std::time::Duration;

/// Metrics collector
pub struct MetricsCollector {
    latencies: HashMap<String, Vec<Duration>>,
    counts: HashMap<String, u64>,
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            latencies: HashMap::new(),
            counts: HashMap::new(),
        }
    }

    /// Record a latency measurement
    pub fn record_latency(&mut self, component: &str, latency: Duration) {
        // Emit to global metrics recorder
        let key = metrics::Key::from_name(component.to_string());
        metrics::with_recorder(|recorder| {
            recorder
                .register_histogram(
                    &key,
                    &metrics::Metadata::new(component, metrics::Level::INFO, None),
                )
                .record(latency.as_secs_f64());
        });

        self.latencies
            .entry(component.to_string())
            .or_default()
            .push(latency);

        // Keep only last 1000 measurements
        let entry = self.latencies.get_mut(component).unwrap();
        if entry.len() > 1000 {
            entry.remove(0);
        }
    }

    /// Increment a counter
    pub fn increment(&mut self, metric: &str) {
        // Emit to global metrics recorder
        let key = metrics::Key::from_name(metric.to_string());
        metrics::with_recorder(|recorder| {
            recorder
                .register_counter(
                    &key,
                    &metrics::Metadata::new(metric, metrics::Level::INFO, None),
                )
                .increment(1);
        });

        *self.counts.entry(metric.to_string()).or_insert(0) += 1;
    }

    /// Get snapshot of current metrics
    pub fn snapshot(&self) -> Metrics {
        let mut latency_stats = HashMap::new();

        for (component, latencies) in &self.latencies {
            if latencies.is_empty() {
                continue;
            }

            let mut sorted = latencies.clone();
            sorted.sort();

            let p50_idx = sorted.len() / 2;
            let p95_idx = (sorted.len() as f64 * 0.95) as usize;
            let p99_idx = (sorted.len() as f64 * 0.99) as usize;

            let sum: Duration = sorted.iter().sum();
            let mean = sum / sorted.len() as u32;

            latency_stats.insert(
                component.clone(),
                LatencyStats {
                    mean,
                    p50: sorted[p50_idx],
                    p95: sorted[p95_idx],
                    p99: sorted[p99_idx],
                },
            );
        }

        Metrics {
            latency_stats,
            counts: self.counts.clone(),
        }
    }
}

/// Snapshot of metrics
#[derive(Debug, Clone)]
pub struct Metrics {
    pub latency_stats: HashMap<String, LatencyStats>,
    pub counts: HashMap<String, u64>,
}

/// Latency statistics
#[derive(Debug, Clone)]
pub struct LatencyStats {
    pub mean: Duration,
    pub p50: Duration,
    pub p95: Duration,
    pub p99: Duration,
}
