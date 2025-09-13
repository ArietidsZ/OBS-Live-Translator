use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{Mutex, RwLock};
use anyhow::Result;
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryEvent {
    pub timestamp: SystemTime,
    pub event_type: EventType,
    pub component: String,
    pub data: HashMap<String, serde_json::Value>,
    pub duration_ms: Option<u64>,
    pub correlation_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventType {
    StreamStarted,
    StreamStopped,
    TranscriptionCompleted,
    TranslationCompleted,
    CacheHit,
    CacheMiss,
    ResourceAllocation,
    ResourceRelease,
    Error,
    Warning,
    Performance,
}

pub struct TelemetryCollector {
    events: Arc<RwLock<Vec<TelemetryEvent>>>,
    max_events: usize,
    enabled: Arc<Mutex<bool>>,
}

impl TelemetryCollector {
    pub fn new(max_events: usize) -> Self {
        Self {
            events: Arc::new(RwLock::new(Vec::with_capacity(max_events))),
            max_events,
            enabled: Arc::new(Mutex::new(true)),
        }
    }

    pub async fn record_event(&self, event: TelemetryEvent) -> Result<()> {
        if !*self.enabled.lock().await {
            return Ok(());
        }

        let mut events = self.events.write().await;
        events.push(event);

        if events.len() > self.max_events {
            events.drain(0..100);
        }

        Ok(())
    }

    pub async fn get_events(&self, count: usize) -> Vec<TelemetryEvent> {
        let events = self.events.read().await;
        events.iter().rev().take(count).cloned().collect()
    }

    pub async fn clear(&self) {
        let mut events = self.events.write().await;
        events.clear();
    }

    pub async fn set_enabled(&self, enabled: bool) {
        *self.enabled.lock().await = enabled;
    }
}