//! Browser Source Integration for OBS
//!
//! Provides a local HTTP server that serves translation overlays
//! optimized for OBS browser sources with real-time updates via WebSocket

use anyhow::Result;
use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    extract::State,
    response::{Html, IntoResponse, Response},
    routing::{get, get_service},
    Router,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::{
    net::TcpListener,
    sync::{broadcast, RwLock},
};
use tower::ServiceBuilder;
use tower_http::{
    cors::{Any, CorsLayer},
    services::ServeDir,
};

use super::BrowserSourceConfig;
use crate::TranslationResult;

/// Browser source server for OBS integration
pub struct BrowserServer {
    config: BrowserSourceConfig,
    translation_tx: broadcast::Sender<TranslationUpdate>,
    server_handle: Option<tokio::task::JoinHandle<()>>,
}

/// Translation update for browser overlay
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationUpdate {
    pub timestamp: u64,
    pub original_text: String,
    pub translated_text: String,
    pub source_language: String,
    pub target_language: String,
    pub confidence: f32,
    pub speaker_id: Option<String>,
    pub is_final: bool,
}

impl From<TranslationResult> for TranslationUpdate {
    fn from(result: TranslationResult) -> Self {
        Self {
            timestamp: result.timestamp.duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            original_text: result.original_text,
            translated_text: result.translated_text,
            source_language: result.source_language,
            target_language: result.target_language,
            confidence: result.confidence,
            speaker_id: None,
            is_final: true,
        }
    }
}

/// Shared server state
#[derive(Clone)]
struct ServerState {
    translation_tx: broadcast::Sender<TranslationUpdate>,
    config: Arc<RwLock<BrowserSourceConfig>>,
}

impl BrowserServer {
    /// Create new browser server
    pub async fn new(config: BrowserSourceConfig) -> Result<Self> {
        let (translation_tx, _) = broadcast::channel(1000);

        Ok(Self {
            config,
            translation_tx,
            server_handle: None,
        })
    }

    /// Start the browser source server
    pub async fn start_translation_overlay(&mut self) -> Result<()> {
        let state = ServerState {
            translation_tx: self.translation_tx.clone(),
            config: Arc::new(RwLock::new(self.config.clone())),
        };

        let app = Router::new()
            .route("/", get(serve_index))
            .route("/overlay", get(serve_overlay))
            .route("/config", get(serve_config))
            .route("/ws", get(websocket_handler))
            .route("/health", get(health_check))
            .nest_service("/static", get_service(ServeDir::new("static")))
            .layer(
                ServiceBuilder::new()
                    .layer(CorsLayer::new().allow_origin(Any))
                    .into_inner(),
            )
            .with_state(state);

        let addr = format!("127.0.0.1:{}", self.config.overlay_port);
        let listener = TcpListener::bind(&addr).await?;

        tracing::info!("Browser source server starting on {}", addr);

        let server_handle = tokio::spawn(async move {
            if let Err(e) = axum::serve(listener, app).await {
                tracing::error!("Browser server error: {}", e);
            }
        });

        self.server_handle = Some(server_handle);
        Ok(())
    }

    /// Send translation update to all connected clients
    pub async fn send_translation(&self, translation: TranslationResult) -> Result<()> {
        let update = TranslationUpdate::from(translation);
        let _ = self.translation_tx.send(update);
        Ok(())
    }

    /// Update server configuration
    pub async fn update_config(&mut self, config: BrowserSourceConfig) -> Result<()> {
        self.config = config;
        Ok(())
    }

    /// Stop the server
    pub async fn shutdown(&mut self) -> Result<()> {
        if let Some(handle) = self.server_handle.take() {
            handle.abort();
        }
        Ok(())
    }
}

/// Serve main index page
async fn serve_index() -> Html<&'static str> {
    Html(include_str!("../../static/index.html"))
}

/// Serve translation overlay page optimized for OBS
async fn serve_overlay(State(state): State<ServerState>) -> Response {
    let config = state.config.read().await;
    let html = generate_overlay_html(&config);
    Html(html).into_response()
}

/// Serve configuration page
async fn serve_config(State(state): State<ServerState>) -> impl IntoResponse {
    let config = state.config.read().await;
    axum::Json(config.clone())
}

/// Health check endpoint
async fn health_check() -> impl IntoResponse {
    axum::Json(serde_json::json!({
        "status": "healthy",
        "service": "obs-live-translator-browser-source",
        "timestamp": chrono::Utc::now().timestamp()
    }))
}

/// WebSocket handler for real-time translation updates
async fn websocket_handler(
    ws: WebSocketUpgrade,
    State(state): State<ServerState>,
) -> Response {
    ws.on_upgrade(|socket| websocket_connection(socket, state))
}

/// Handle WebSocket connection
async fn websocket_connection(socket: WebSocket, state: ServerState) {
    let mut rx = state.translation_tx.subscribe();
    let (mut sender, mut receiver) = socket.split();

    // Send initial connection message
    let welcome = serde_json::json!({
        "type": "connected",
        "message": "OBS Live Translator overlay connected",
        "timestamp": chrono::Utc::now().timestamp()
    });

    if sender.send(Message::Text(welcome.to_string())).await.is_err() {
        return;
    }

    // Handle incoming messages and translation updates
    let send_task = tokio::spawn(async move {
        while let Ok(update) = rx.recv().await {
            if let Ok(json) = serde_json::to_string(&update) {
                if sender.send(Message::Text(json)).await.is_err() {
                    break;
                }
            }
        }
    });

    let recv_task = tokio::spawn(async move {
        while let Some(msg) = receiver.next().await {
            if let Ok(msg) = msg {
                match msg {
                    Message::Text(text) => {
                        // Handle configuration updates from client
                        tracing::debug!("Received WebSocket message: {}", text);
                    }
                    Message::Close(_) => break,
                    _ => {}
                }
            } else {
                break;
            }
        }
    });

    // Wait for either task to complete
    tokio::select! {
        _ = send_task => {},
        _ = recv_task => {},
    }
}

/// Generate optimized HTML overlay for OBS
fn generate_overlay_html(config: &BrowserSourceConfig) -> String {
    let custom_css = config.custom_css.as_deref().unwrap_or("");

    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OBS Live Translation Overlay</title>
    <style>
        body {{
            margin: 0;
            padding: 0;
            width: {width}px;
            height: {height}px;
            background: transparent;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            overflow: hidden;
        }}

        .translation-container {{
            position: absolute;
            bottom: 50px;
            left: 50%;
            transform: translateX(-50%);
            max-width: 90%;
            text-align: center;
        }}

        .translation-box {{
            background: linear-gradient(135deg, rgba(0, 0, 0, 0.8), rgba(0, 0, 0, 0.6));
            backdrop-filter: blur(10px);
            border: 2px solid rgba(255, 255, 255, 0.2);
            border-radius: 12px;
            padding: 16px 24px;
            margin: 8px 0;
            transition: all 0.3s ease;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        }}

        .original-text {{
            color: #e0e0e0;
            font-size: 16px;
            margin-bottom: 8px;
            opacity: 0.9;
        }}

        .translated-text {{
            color: #ffffff;
            font-size: 20px;
            font-weight: 600;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);
        }}

        .language-info {{
            color: #00d4aa;
            font-size: 12px;
            margin-top: 8px;
            font-weight: 500;
        }}

        .confidence-bar {{
            width: 100%;
            height: 3px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 2px;
            margin-top: 8px;
            overflow: hidden;
        }}

        .confidence-fill {{
            height: 100%;
            background: linear-gradient(90deg, #ff6b6b, #ffd93d, #6bcf7f);
            border-radius: 2px;
            transition: width 0.5s ease;
        }}

        .fade-in {{
            animation: fadeIn 0.5s ease-in;
        }}

        .fade-out {{
            animation: fadeOut 0.5s ease-out forwards;
        }}

        @keyframes fadeIn {{
            from {{
                opacity: 0;
                transform: translateX(-50%) translateY(20px);
            }}
            to {{
                opacity: 1;
                transform: translateX(-50%) translateY(0);
            }}
        }}

        @keyframes fadeOut {{
            from {{
                opacity: 1;
                transform: translateX(-50%) translateY(0);
            }}
            to {{
                opacity: 0;
                transform: translateX(-50%) translateY(-20px);
            }}
        }}

        {custom_css}
    </style>
</head>
<body>
    <div class="translation-container" id="translationContainer">
        <!-- Translations will appear here -->
    </div>

    <script>
        const container = document.getElementById('translationContainer');
        let currentTranslation = null;

        // WebSocket connection for real-time updates
        const ws = new WebSocket('ws://localhost:{port}/ws');

        ws.onopen = function() {{
            console.log('Connected to OBS Live Translator');
        }};

        ws.onmessage = function(event) {{
            try {{
                const data = JSON.parse(event.data);
                if (data.type !== 'connected') {{
                    updateTranslation(data);
                }}
            }} catch (error) {{
                console.error('Error parsing WebSocket message:', error);
            }}
        }};

        ws.onerror = function(error) {{
            console.error('WebSocket error:', error);
        }};

        ws.onclose = function() {{
            console.log('Disconnected from OBS Live Translator');
            // Attempt to reconnect after 3 seconds
            setTimeout(function() {{
                window.location.reload();
            }}, 3000);
        }};

        function updateTranslation(translation) {{
            // Clear previous translation with fade out
            if (currentTranslation) {{
                currentTranslation.classList.add('fade-out');
                setTimeout(() => {{
                    if (currentTranslation.parentNode) {{
                        currentTranslation.parentNode.removeChild(currentTranslation);
                    }}
                }}, 500);
            }}

            // Create new translation element
            const translationBox = document.createElement('div');
            translationBox.className = 'translation-box fade-in';

            const originalText = document.createElement('div');
            originalText.className = 'original-text';
            originalText.textContent = translation.original_text;

            const translatedText = document.createElement('div');
            translatedText.className = 'translated-text';
            translatedText.textContent = translation.translated_text;

            const languageInfo = document.createElement('div');
            languageInfo.className = 'language-info';
            languageInfo.textContent = `${{translation.source_language.toUpperCase()}} → ${{translation.target_language.toUpperCase()}}`;

            const confidenceBar = document.createElement('div');
            confidenceBar.className = 'confidence-bar';

            const confidenceFill = document.createElement('div');
            confidenceFill.className = 'confidence-fill';
            confidenceFill.style.width = `${{Math.round(translation.confidence * 100)}}%`;

            confidenceBar.appendChild(confidenceFill);

            translationBox.appendChild(originalText);
            translationBox.appendChild(translatedText);
            translationBox.appendChild(languageInfo);
            translationBox.appendChild(confidenceBar);

            container.appendChild(translationBox);
            currentTranslation = translationBox;

            // Auto-hide after 5 seconds for non-final translations
            if (!translation.is_final) {{
                setTimeout(() => {{
                    if (translationBox.parentNode) {{
                        translationBox.classList.add('fade-out');
                        setTimeout(() => {{
                            if (translationBox.parentNode) {{
                                translationBox.parentNode.removeChild(translationBox);
                            }}
                        }}, 500);
                    }}
                }}, 5000);
            }}
        }}
    </script>
</body>
</html>"#,
        width = config.width,
        height = config.height,
        port = config.overlay_port,
        custom_css = custom_css
    )
}