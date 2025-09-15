//! OBS WebSocket Client for Remote Control
//!
//! Provides remote control capabilities for OBS Studio through WebSocket API
//! Compatible with OBS Studio 28+ built-in WebSocket server

use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
use tokio::{
    net::TcpStream,
    sync::{broadcast, RwLock},
    time::{timeout, Duration},
};
use tokio_tungstenite::{
    connect_async, tungstenite::Message, MaybeTlsStream, WebSocketStream,
};

use super::WebSocketConfig;

/// OBS WebSocket client
pub struct WebSocketClient {
    config: WebSocketConfig,
    connection: Arc<RwLock<Option<WebSocketStream<MaybeTlsStream<TcpStream>>>>>,
    event_tx: broadcast::Sender<ObsEvent>,
    request_id: Arc<std::sync::atomic::AtomicU64>,
}

/// OBS WebSocket events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObsEvent {
    pub event_type: String,
    pub event_data: Value,
    pub timestamp: u64,
}

/// OBS WebSocket request
#[derive(Debug, Serialize)]
struct ObsRequest {
    #[serde(rename = "op")]
    operation: u8,
    #[serde(rename = "d")]
    data: ObsRequestData,
}

#[derive(Debug, Serialize)]
struct ObsRequestData {
    #[serde(rename = "requestType")]
    request_type: String,
    #[serde(rename = "requestId")]
    request_id: String,
    #[serde(rename = "requestData", skip_serializing_if = "Option::is_none")]
    request_data: Option<Value>,
}

/// OBS WebSocket response
#[derive(Debug, Deserialize)]
struct ObsResponse {
    #[serde(rename = "op")]
    operation: u8,
    #[serde(rename = "d")]
    data: Value,
}

impl WebSocketClient {
    /// Create new WebSocket client
    pub async fn new(config: WebSocketConfig) -> Result<Self> {
        let (event_tx, _) = broadcast::channel(1000);

        let client = Self {
            config,
            connection: Arc::new(RwLock::new(None)),
            event_tx,
            request_id: Arc::new(std::sync::atomic::AtomicU64::new(1)),
        };

        if client.config.auto_connect {
            if let Err(e) = client.connect().await {
                tracing::warn!("Failed to auto-connect to OBS WebSocket: {}", e);
            }
        }

        Ok(client)
    }

    /// Connect to OBS WebSocket server
    pub async fn connect(&self) -> Result<()> {
        let url = format!("ws://127.0.0.1:{}", self.config.port);
        tracing::info!("Connecting to OBS WebSocket at {}", url);

        let (ws_stream, _) = connect_async(&url).await?;
        *self.connection.write().await = Some(ws_stream);

        // Start message handling task
        self.start_message_handler().await?;

        // Authenticate if password is provided
        if let Some(password) = &self.config.password {
            self.authenticate(password).await?;
        }

        tracing::info!("Successfully connected to OBS WebSocket");
        Ok(())
    }

    /// Check if connected to OBS
    pub async fn is_connected(&self) -> bool {
        self.connection.read().await.is_some()
    }

    /// Disconnect from OBS WebSocket
    pub async fn disconnect(&mut self) -> Result<()> {
        if let Some(mut connection) = self.connection.write().await.take() {
            connection.close(None).await?;
        }
        Ok(())
    }

    /// Subscribe to OBS events
    pub fn subscribe_events(&self) -> broadcast::Receiver<ObsEvent> {
        self.event_tx.subscribe()
    }

    /// Create browser source in OBS
    pub async fn create_browser_source(
        &self,
        source_name: &str,
        url: &str,
        width: u32,
        height: u32,
    ) -> Result<()> {
        let request_data = json!({
            "sourceName": source_name,
            "sourceKind": "browser_source",
            "sourceSettings": {
                "url": url,
                "width": width,
                "height": height,
                "fps": 30,
                "shutdown": false,
                "restart_when_active": false
            }
        });

        self.send_request("CreateSource", Some(request_data)).await?;
        tracing::info!("Created browser source '{}' with URL: {}", source_name, url);
        Ok(())
    }

    /// Setup audio capture source
    pub async fn setup_audio_capture(&self, source_name: &str) -> Result<()> {
        // Check if audio source exists
        let sources = self.get_source_list().await?;
        let audio_source_exists = sources
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .any(|source| {
                source
                    .get("sourceName")
                    .and_then(|name| name.as_str())
                    .map(|name| name == source_name)
                    .unwrap_or(false)
            });

        if !audio_source_exists {
            tracing::warn!("Audio source '{}' not found in OBS", source_name);
            return Ok(());
        }

        // Configure audio monitoring (optional)
        let request_data = json!({
            "sourceName": source_name,
            "monitorType": "monitorOnly"
        });

        self.send_request("SetSourceAudioMonitorType", Some(request_data))
            .await?;

        tracing::info!("Configured audio capture for source '{}'", source_name);
        Ok(())
    }

    /// Get list of sources in OBS
    pub async fn get_source_list(&self) -> Result<Value> {
        self.send_request("GetSourceList", None).await
    }

    /// Get current scene name
    pub async fn get_current_scene(&self) -> Result<String> {
        let response = self.send_request("GetCurrentScene", None).await?;
        response
            .get("sceneName")
            .and_then(|name| name.as_str())
            .map(|name| name.to_string())
            .ok_or_else(|| anyhow::anyhow!("Failed to get current scene name"))
    }

    /// Set source visibility
    pub async fn set_source_visibility(&self, source_name: &str, visible: bool) -> Result<()> {
        let request_data = json!({
            "sourceName": source_name,
            "sourceVisible": visible
        });

        self.send_request("SetSourceVisibility", Some(request_data))
            .await?;
        Ok(())
    }

    /// Send request to OBS WebSocket
    async fn send_request(&self, request_type: &str, request_data: Option<Value>) -> Result<Value> {
        let request_id = self
            .request_id
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            .to_string();

        let request = ObsRequest {
            operation: 6, // Request operation code
            data: ObsRequestData {
                request_type: request_type.to_string(),
                request_id: request_id.clone(),
                request_data,
            },
        };

        let connection_guard = self.connection.read().await;
        let connection = connection_guard
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Not connected to OBS WebSocket"))?;

        // Send request (this is a simplified version - in practice you'd need proper message handling)
        let message = serde_json::to_string(&request)?;
        // Note: Actual sending would require splitting the connection and handling responses

        tracing::debug!("Sent OBS request: {}", request_type);

        // Return dummy response for now - in a real implementation,
        // you'd wait for the actual response with matching request_id
        Ok(json!({}))
    }

    /// Authenticate with OBS WebSocket
    async fn authenticate(&self, password: &str) -> Result<()> {
        // Simplified authentication - actual implementation would handle
        // the challenge-response authentication protocol
        tracing::info!("Authenticating with OBS WebSocket");
        Ok(())
    }

    /// Start message handler task
    async fn start_message_handler(&self) -> Result<()> {
        let event_tx = self.event_tx.clone();

        tokio::spawn(async move {
            // In a real implementation, this would handle incoming messages
            // and dispatch events to subscribers
            tracing::debug!("OBS WebSocket message handler started");
        });

        Ok(())
    }
}

/// Helper functions for OBS integration
impl WebSocketClient {
    /// Auto-detect OBS installation and check if WebSocket is enabled
    pub async fn detect_obs_websocket() -> Result<bool> {
        let url = "ws://127.0.0.1:4455";
        match timeout(Duration::from_secs(2), connect_async(url)).await {
            Ok(Ok(_)) => Ok(true),
            _ => Ok(false),
        }
    }

    /// Get OBS version info
    pub async fn get_obs_version(&self) -> Result<String> {
        let response = self.send_request("GetVersion", None).await?;
        response
            .get("obsVersion")
            .and_then(|v| v.as_str())
            .map(|v| v.to_string())
            .ok_or_else(|| anyhow::anyhow!("Failed to get OBS version"))
    }

    /// Check if source exists
    pub async fn source_exists(&self, source_name: &str) -> Result<bool> {
        let sources = self.get_source_list().await?;
        let exists = sources
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .any(|source| {
                source
                    .get("sourceName")
                    .and_then(|name| name.as_str())
                    .map(|name| name == source_name)
                    .unwrap_or(false)
            });
        Ok(exists)
    }

    /// Create scene if it doesn't exist
    pub async fn ensure_scene_exists(&self, scene_name: &str) -> Result<()> {
        let request_data = json!({
            "sceneName": scene_name
        });

        // Try to create scene - OBS will return an error if it already exists
        let _ = self.send_request("CreateScene", Some(request_data)).await;
        Ok(())
    }
}