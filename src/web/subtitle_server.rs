use axum::{
    extract::{WebSocketUpgrade, ws::{WebSocket, Message}},
    response::{Html, IntoResponse},
    routing::{get, get_service},
    Router,
};
use std::{net::SocketAddr, sync::Arc, path::PathBuf};
use tokio::sync::{broadcast, RwLock};
use tower_http::services::ServeDir;
use serde::{Deserialize, Serialize};
use tracing::{info, error, debug};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SubtitleData {
    pub original_text: String,
    pub translated_text: String,
    pub source_lang: String,
    pub target_lang: String,
    pub confidence: f32,
    pub timestamp: u64,
}

#[derive(Clone)]
pub struct SubtitleServer {
    subtitle_tx: broadcast::Sender<SubtitleData>,
    current_subtitle: Arc<RwLock<Option<SubtitleData>>>,
}

impl SubtitleServer {
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(100);
        Self {
            subtitle_tx: tx,
            current_subtitle: Arc::new(RwLock::new(None)),
        }
    }

    pub async fn broadcast_subtitle(&self, subtitle: SubtitleData) {
        // Update current subtitle
        *self.current_subtitle.write().await = Some(subtitle.clone());

        // Broadcast to all connected clients
        if let Err(e) = self.subtitle_tx.send(subtitle) {
            debug!("No clients connected to receive subtitle: {}", e);
        }
    }

    pub async fn start_server(self: Arc<Self>, port: u16) {
        let app = Router::new()
            .route("/", get(serve_overlay))
            .route("/ws/subtitles", get({
                let server = Arc::clone(&self);
                move |ws| handle_websocket(ws, server.clone())
            }))
            .nest_service("/static", get_service(ServeDir::new("web")))
            .layer(tower_http::cors::CorsLayer::permissive());

        let addr = SocketAddr::from(([0, 0, 0, 0], port));
        info!("Subtitle server listening on http://{}", addr);
        info!("OBS Browser Source URL: http://localhost:{}/", port);

        axum::Server::bind(&addr)
            .serve(app.into_make_service())
            .await
            .unwrap();
    }

    pub fn get_sender(&self) -> broadcast::Sender<SubtitleData> {
        self.subtitle_tx.clone()
    }
}

async fn serve_overlay() -> impl IntoResponse {
    let html = include_str!("../../web/subtitle_overlay.html");
    Html(html)
}

async fn handle_websocket(
    ws: WebSocketUpgrade,
    server: Arc<SubtitleServer>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| websocket_handler(socket, server))
}

async fn websocket_handler(mut socket: WebSocket, server: Arc<SubtitleServer>) {
    info!("New WebSocket client connected");

    // Send current subtitle if exists
    if let Some(subtitle) = server.current_subtitle.read().await.clone() {
        let msg = serde_json::to_string(&subtitle).unwrap();
        let _ = socket.send(Message::Text(msg)).await;
    }

    // Subscribe to subtitle updates
    let mut rx = server.subtitle_tx.subscribe();

    // Handle incoming messages and broadcast updates
    loop {
        tokio::select! {
            // Receive subtitle updates
            Ok(subtitle) = rx.recv() => {
                let msg = match serde_json::to_string(&subtitle) {
                    Ok(json) => json,
                    Err(e) => {
                        error!("Failed to serialize subtitle: {}", e);
                        continue;
                    }
                };

                if socket.send(Message::Text(msg)).await.is_err() {
                    break;
                }
            }

            // Handle client messages (if any)
            Some(msg) = socket.recv() => {
                match msg {
                    Ok(Message::Text(txt)) => {
                        debug!("Received message from client: {}", txt);
                        // Handle client commands if needed
                    }
                    Ok(Message::Close(_)) => {
                        info!("Client disconnected");
                        break;
                    }
                    Err(e) => {
                        error!("WebSocket error: {}", e);
                        break;
                    }
                    _ => {}
                }
            }
        }
    }

    info!("WebSocket client disconnected");
}

// Test function to generate sample subtitles
pub async fn generate_test_subtitles(server: Arc<SubtitleServer>) {
    use tokio::time::{sleep, Duration};
    use std::time::SystemTime;

    let test_data = vec![
        ("Bonjour tout le monde", "Hello everyone", "fr", "en", 0.96),
        ("Comment allez-vous aujourd'hui?", "How are you today?", "fr", "en", 0.94),
        ("Bienvenidos a la transmisión en vivo", "Welcome to the live stream", "es", "en", 0.97),
        ("Este es un sistema de traducción increíble", "This is an amazing translation system", "es", "en", 0.95),
        ("Guten Tag, wie geht es Ihnen?", "Good day, how are you?", "de", "en", 0.93),
        ("Das Wetter ist heute schön", "The weather is nice today", "de", "en", 0.98),
        ("こんにちは、皆さん", "Hello, everyone", "ja", "en", 0.92),
        ("今日はライブ配信をしています", "We are live streaming today", "ja", "en", 0.89),
    ];

    loop {
        for (original, translated, source, target, confidence) in &test_data {
            let subtitle = SubtitleData {
                original_text: original.to_string(),
                translated_text: translated.to_string(),
                source_lang: source.to_string(),
                target_lang: target.to_string(),
                confidence: *confidence,
                timestamp: SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            };

            server.broadcast_subtitle(subtitle).await;
            sleep(Duration::from_secs(4)).await;
        }
    }
}