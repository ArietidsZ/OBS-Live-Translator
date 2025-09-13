use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use anyhow::Result;

#[derive(Debug, Clone)]
pub struct MetricValue {
    pub value: f64,
    pub timestamp: Instant,
    pub tags: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct Metric {
    pub name: String,
    pub values: Vec<MetricValue>,
    pub metric_type: MetricType,
}

#[derive(Debug, Clone)]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
    Timer,
}

pub struct MetricsCollector {
    metrics: Arc<RwLock<HashMap<String, Metric>>>,
    max_values_per_metric: usize,
}

impl MetricsCollector {
    pub fn new(max_values_per_metric: usize) -> Self {
        Self {
            metrics: Arc::new(RwLock::new(HashMap::new())),
            max_values_per_metric,
        }
    }

    pub async fn record_counter(&self, name: &str, value: f64, tags: HashMap<String, String>) -> Result<()> {
        self.record_metric(name, value, MetricType::Counter, tags).await
    }

    pub async fn record_gauge(&self, name: &str, value: f64, tags: HashMap<String, String>) -> Result<()> {
        self.record_metric(name, value, MetricType::Gauge, tags).await
    }

    pub async fn record_timer(&self, name: &str, duration: Duration, tags: HashMap<String, String>) -> Result<()> {
        self.record_metric(name, duration.as_millis() as f64, MetricType::Timer, tags).await
    }

    async fn record_metric(&self, name: &str, value: f64, metric_type: MetricType, tags: HashMap<String, String>) -> Result<()> {
        let mut metrics = self.metrics.write().await;

        let metric = metrics.entry(name.to_string()).or_insert(Metric {
            name: name.to_string(),
            values: Vec::new(),
            metric_type,
        });

        metric.values.push(MetricValue {
            value,
            timestamp: Instant::now(),
            tags,
        });

        if metric.values.len() > self.max_values_per_metric {
            metric.values.drain(0..100);
        }

        Ok(())
    }

    pub async fn get_metric(&self, name: &str) -> Option<Metric> {
        let metrics = self.metrics.read().await;
        metrics.get(name).cloned()
    }

    pub async fn get_all_metrics(&self) -> HashMap<String, Metric> {
        let metrics = self.metrics.read().await;
        metrics.clone()
    }

    pub async fn clear(&self) {
        let mut metrics = self.metrics.write().await;
        metrics.clear();
    }
}