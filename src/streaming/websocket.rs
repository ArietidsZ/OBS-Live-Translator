//! WebSocket handler for real-time communication

use super::{SessionManager, StreamingSession};
use anyhow::{Result, anyhow};
use axum::extract::ws::{WebSocket, Message};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::time::{Duration, timeout};
use tracing::{info, warn, error};
use uuid::Uuid;

/// WebSocket message types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum WebSocketMessage {
    /// Client sends audio data
    Audio {
        data: Vec<f32>,
        sample_rate: u32,
        timestamp: u64,
    },
    /// Server sends transcription result
    Transcription {
        text: String,
        language: String,
        confidence: f32,
        is_final: bool,
        timestamp: u64,
    },
    /// Server sends translation result
    Translation {
        original_text: String,
        translated_text: String,
        source_language: String,
        target_language: String,
        confidence: f32,
        timestamp: u64,
    },
    /// Configuration message
    Config {
        source_language: Option<String>,
        target_languages: Vec<String>,
        enable_translation: bool,
    },
    /// Status message
    Status {
        connected: bool,
        session_id: String,
        message: String,
    },
    /// Error message
    Error {
        error: String,
        code: u32,
    },
    /// Heartbeat message
    Ping {
        timestamp: u64,
    },
    /// Heartbeat response
    Pong {
        timestamp: u64,
    },
}

/// WebSocket handler
pub struct WebSocketHandler {
    socket: WebSocket,
    session_manager: Arc<SessionManager>,
    session_id: String,
    session: Option<Arc<StreamingSession>>,
}

impl WebSocketHandler {
    pub fn new(socket: WebSocket, session_manager: Arc<SessionManager>) -> Self {
        let session_id = Uuid::new_v4().to_string();

        Self {
            socket,
            session_manager,
            session_id,
            session: None,
        }
    }

    /// Handle WebSocket connection
    pub async fn handle(mut self) -> Result<()> {
        // Create session
        let session = self.session_manager.create_session(&self.session_id).await?;
        self.session = Some(Arc::clone(&session));

        // Send initial status
        self.send_status("Connected", true).await?;

        // Start heartbeat task
        let session_id = self.session_id.clone();
        let session_manager = Arc::clone(&self.session_manager);
        tokio::spawn(async move {
            Self::heartbeat_task(session_id, session_manager).await;
        });

        // Handle messages
        loop {
            match timeout(Duration::from_secs(30), self.socket.recv()).await {
                Ok(Some(Ok(message))) => {
                    if let Err(e) = self.handle_message(message).await {
                        error!("Error handling message: {}", e);
                        self.send_error(&e.to_string(), 500).await?;
                    }
                },
                Ok(Some(Err(e))) => {
                    warn!("WebSocket error: {}", e);
                    break;
                },
                Ok(None) => {
                    info!("WebSocket connection closed");
                    break;
                },
                Err(_) => {
                    warn!("WebSocket timeout");
                    break;
                },
            }
        }

        // Cleanup
        self.session_manager.remove_session(&self.session_id).await;
        info!("Session {} ended", self.session_id);

        Ok(())
    }

    /// Handle individual WebSocket message
    async fn handle_message(&mut self, message: Message) -> Result<()> {
        match message {
            Message::Text(text) => {
                let ws_message: WebSocketMessage = serde_json::from_str(&text)
                    .map_err(|e| anyhow!("Invalid JSON: {}", e))?;

                self.handle_websocket_message(ws_message).await
            },
            Message::Binary(data) => {
                // Handle binary audio data
                if data.len() % 4 != 0 {
                    return Err(anyhow!("Invalid audio data length"));
                }

                let audio_data: Vec<f32> = data
                    .chunks(4)
                    .map(|chunk| {
                        let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
                        f32::from_le_bytes(bytes)
                    })
                    .collect();

                self.handle_audio_data(audio_data, 16000).await
            },
            Message::Ping(data) => {
                self.socket.send(Message::Pong(data)).await?;
                Ok(())
            },
            Message::Pong(_) => {
                // Pong received, connection is alive
                Ok(())
            },
            Message::Close(_) => {
                info!("WebSocket close message received");
                Ok(())
            },
        }
    }

    /// Handle structured WebSocket message
    async fn handle_websocket_message(&mut self, message: WebSocketMessage) -> Result<()> {
        match message {
            WebSocketMessage::Audio { data, sample_rate, timestamp: _ } => {
                self.handle_audio_data(data, sample_rate).await
            },
            WebSocketMessage::Config {
                source_language,
                target_languages,
                enable_translation
            } => {
                if let Some(session) = &self.session {
                    session.update_config(source_language, target_languages, enable_translation).await?;
                    self.send_status("Configuration updated", true).await
                } else {
                    Err(anyhow!("No active session"))
                }
            },
            WebSocketMessage::Ping { timestamp } => {
                self.send_message(WebSocketMessage::Pong { timestamp }).await
            },
            _ => {
                warn!("Unexpected message type from client");
                Ok(())
            },
        }
    }

    /// Handle audio data
    async fn handle_audio_data(&mut self, data: Vec<f32>, sample_rate: u32) -> Result<()> {
        if let Some(session) = &self.session {
            // Process audio through session
            let results = session.process_audio(data, sample_rate).await?;

            // Send results back to client
            for result in results {
                match result {
                    crate::streaming::session::ProcessingResult::Transcription {
                        text, language, confidence, is_final
                    } => {
                        let msg = WebSocketMessage::Transcription {
                            text,
                            language,
                            confidence,
                            is_final,
                            timestamp: chrono::Utc::now().timestamp_millis() as u64,
                        };
                        self.send_message(msg).await?;
                    },
                    crate::streaming::session::ProcessingResult::Translation {
                        original_text,
                        translated_text,
                        source_language,
                        target_language,
                        confidence
                    } => {
                        let msg = WebSocketMessage::Translation {
                            original_text,
                            translated_text,
                            source_language,
                            target_language,
                            confidence,
                            timestamp: chrono::Utc::now().timestamp_millis() as u64,
                        };
                        self.send_message(msg).await?;
                    },
                }
            }
        }

        Ok(())
    }

    /// Send WebSocket message
    async fn send_message(&mut self, message: WebSocketMessage) -> Result<()> {
        let json = serde_json::to_string(&message)?;
        self.socket.send(Message::Text(json)).await?;
        Ok(())
    }

    /// Send status message
    async fn send_status(&mut self, message: &str, connected: bool) -> Result<()> {
        let status = WebSocketMessage::Status {
            connected,
            session_id: self.session_id.clone(),
            message: message.to_string(),
        };
        self.send_message(status).await
    }

    /// Send error message
    async fn send_error(&mut self, error: &str, code: u32) -> Result<()> {
        let error_msg = WebSocketMessage::Error {
            error: error.to_string(),
            code,
        };
        self.send_message(error_msg).await
    }

    /// Heartbeat task to keep connection alive
    async fn heartbeat_task(session_id: String, session_manager: Arc<SessionManager>) {
        let mut interval = tokio::time::interval(Duration::from_secs(30));

        loop {
            interval.tick().await;

            // Check if session still exists
            if !session_manager.has_session(&session_id).await {
                break;
            }

            // In a real implementation, we would send ping messages here
            // For now, just update the session's last activity time
            if let Ok(session) = session_manager.get_session(&session_id).await {
                session.update_activity().await;
            }
        }
    }
}

