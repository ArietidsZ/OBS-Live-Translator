//! High-performance networking layer for OBS integration

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};
use tracing::{debug, error, info, warn};

use crate::{TranslationResult, PerformanceMetrics};

/// WebSocket server for OBS overlay communication
pub struct WebSocketServer {
    port: u16,
    clients: Arc<RwLock<HashMap<String, WebSocketClient>>>,
    translation_sender: broadcast::Sender<OverlayMessage>,
    _translation_receiver: broadcast::Receiver<OverlayMessage>,
}

impl WebSocketServer {
    /// Create new WebSocket server
    pub fn new(port: u16) -> Self {
        let (translation_sender, translation_receiver) = broadcast::channel(1000);
        
        Self { 
            port,
            clients: Arc::new(RwLock::new(HashMap::new())),
            translation_sender,
            _translation_receiver: translation_receiver,
        }
    }

    /// Start WebSocket server
    pub async fn start(&self) -> Result<()> {
        info!("Starting WebSocket server on port {}", self.port);
        
        // In a real implementation, this would use a WebSocket library like tokio-tungstenite
        // For now, we'll simulate the server behavior
        
        // Start background task to handle client connections
        let clients = Arc::clone(&self.clients);
        let translation_rx = self.translation_sender.subscribe();
        
        tokio::spawn(async move {
            Self::handle_client_connections(clients, translation_rx).await;
        });
        
        info!("WebSocket server started successfully on port {}", self.port);
        Ok(())
    }
    
    /// Broadcast translation result to all connected overlay clients
    pub async fn broadcast_translation(&self, result: &TranslationResult) -> Result<()> {
        let message = OverlayMessage::Translation {
            original: result.original_text.clone(),
            translated: result.translated_text.clone(),
            source_lang: result.source_language.clone(),
            target_lang: result.target_language.clone(),
            confidence: result.confidence,
            timestamp: result.timestamp,
        };
        
        if let Err(e) = self.translation_sender.send(message) {
            warn!("Failed to broadcast translation: {}", e);
        }
        
        debug!("Broadcasted translation to {} clients", self.get_client_count().await);
        Ok(())
    }
    
    /// Broadcast performance metrics to connected clients
    pub async fn broadcast_metrics(&self, metrics: &PerformanceMetrics) -> Result<()> {
        let message = OverlayMessage::Metrics {
            audio_latency: metrics.audio_latency_ms,
            asr_latency: metrics.asr_latency_ms,
            translation_latency: metrics.translation_latency_ms,
            total_latency: metrics.total_latency_ms,
            gpu_utilization: metrics.gpu_utilization_percent,
            memory_usage: metrics.memory_usage_mb,
            cache_hit_rate: metrics.cache_hit_rate,
            voice_status: metrics.voice_cloning_status.clone(),
        };
        
        if let Err(e) = self.translation_sender.send(message) {
            warn!("Failed to broadcast metrics: {}", e);
        }
        
        Ok(())
    }
    
    /// Broadcast topic summary for new viewers
    pub async fn broadcast_topic_summary(&self, summary: &str) -> Result<()> {
        let message = OverlayMessage::TopicSummary {
            summary: summary.to_string(),
            timestamp: std::time::SystemTime::now(),
        };
        
        if let Err(e) = self.translation_sender.send(message) {
            warn!("Failed to broadcast topic summary: {}", e);
        }
        
        Ok(())
    }
    
    /// Get number of connected clients
    pub async fn get_client_count(&self) -> usize {
        self.clients.read().await.len()
    }
    
    /// Get connection status for overlay
    pub async fn get_connection_status(&self) -> ConnectionStatus {
        let client_count = self.get_client_count().await;
        
        ConnectionStatus {
            server_running: true,
            connected_clients: client_count,
            port: self.port,
            last_broadcast: std::time::SystemTime::now(),
        }
    }
    
    // Private helper methods
    
    async fn handle_client_connections(
        clients: Arc<RwLock<HashMap<String, WebSocketClient>>>,
        mut translation_rx: broadcast::Receiver<OverlayMessage>,
    ) {
        info!("Started client connection handler");
        
        while let Ok(message) = translation_rx.recv().await {
            let clients_guard = clients.read().await;
            let client_count = clients_guard.len();
            
            if client_count > 0 {
                debug!("Broadcasting message to {} clients", client_count);
                
                // In real implementation, this would send WebSocket messages
                for (client_id, client) in clients_guard.iter() {
                    if let Err(e) = client.send_message(&message).await {
                        error!("Failed to send message to client {}: {}", client_id, e);
                    }
                }
            }
        }
    }
}

/// WebSocket client connection handler
#[derive(Debug, Clone)]
pub struct WebSocketClient {
    pub client_id: String,
    pub connected_at: std::time::SystemTime,
    pub last_activity: std::time::SystemTime,
    pub client_type: ClientType,
}

impl WebSocketClient {
    pub fn new(client_id: String, client_type: ClientType) -> Self {
        let now = std::time::SystemTime::now();
        Self {
            client_id,
            connected_at: now,
            last_activity: now,
            client_type,
        }
    }
    
    pub async fn send_message(&self, message: &OverlayMessage) -> Result<()> {
        // In real implementation, this would send WebSocket message
        debug!("Sending message to client {} ({:?}): {:?}", 
               self.client_id, self.client_type, message);
        Ok(())
    }
}

/// Type of WebSocket client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClientType {
    /// OBS overlay display
    Overlay,
    /// Dashboard/control panel
    Dashboard,
    /// Mobile app
    Mobile,
    /// Third-party integration
    ThirdParty,
}

/// Messages sent to overlay clients
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum OverlayMessage {
    /// New translation result
    Translation {
        original: String,
        translated: String,
        source_lang: String,
        target_lang: String,
        confidence: f32,
        timestamp: std::time::SystemTime,
    },
    /// Performance metrics update
    Metrics {
        audio_latency: f32,
        asr_latency: f32,
        translation_latency: f32,
        total_latency: f32,
        gpu_utilization: f32,
        memory_usage: f32,
        cache_hit_rate: f32,
        voice_status: String,
    },
    /// Topic summary for new viewers
    TopicSummary {
        summary: String,
        timestamp: std::time::SystemTime,
    },
    /// Voice cloning status update
    VoiceStatus {
        status: String,
        profile_ready: bool,
        samples_analyzed: usize,
    },
    /// System status update
    SystemStatus {
        translator_status: String,
        uptime_seconds: u64,
        processed_audio_chunks: u64,
        total_translations: u64,
    },
}

/// Server connection status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionStatus {
    pub server_running: bool,
    pub connected_clients: usize,
    pub port: u16,
    pub last_broadcast: std::time::SystemTime,
}

/// WebSocket server configuration
#[derive(Debug, Clone)]
pub struct WebSocketConfig {
    pub port: u16,
    pub max_clients: usize,
    pub heartbeat_interval_ms: u64,
    pub message_buffer_size: usize,
}

impl Default for WebSocketConfig {
    fn default() -> Self {
        Self {
            port: 8080,
            max_clients: 100,
            heartbeat_interval_ms: 30000, // 30 seconds
            message_buffer_size: 1000,
        }
    }
}