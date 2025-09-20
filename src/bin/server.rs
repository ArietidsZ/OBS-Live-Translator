//! WebSocket server for OBS Live Translator

use anyhow::Result;
use axum::{
    extract::ws::{WebSocket, WebSocketUpgrade},
    response::Response,
    routing::get,
    Router,
};
use obs_live_translator::Translator;
use std::sync::Arc;
use tokio::net::TcpListener;
use tracing::{info, error};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    info!("🚀 Starting OBS Live Translator Server");

    // Create translator with GPU acceleration if available
    let translator = Arc::new(tokio::sync::Mutex::new(
        Translator::new("models/whisper.onnx", true)
            .expect("Failed to initialize translator")
    ));

    // Create router
    let app = Router::new()
        .route("/ws", get(move |ws| websocket_handler(ws, translator.clone())))
        .route("/health", get(|| async { "OK" }));

    // Start server
    let listener = TcpListener::bind("0.0.0.0:8080").await?;
    info!("✅ Server listening on http://0.0.0.0:8080");
    info!("📡 WebSocket endpoint: ws://localhost:8080/ws");

    axum::serve(listener, app).await?;

    Ok(())
}

async fn websocket_handler(
    ws: WebSocketUpgrade,
    translator: Arc<tokio::sync::Mutex<Translator>>,
) -> Response {
    ws.on_upgrade(move |socket| handle_socket(socket, translator))
}

async fn handle_socket(mut socket: WebSocket, translator: Arc<tokio::sync::Mutex<Translator>>) {
    info!("New WebSocket connection");

    // Buffer for batching
    let mut batch_buffer: Vec<Vec<f32>> = Vec::new();
    let batch_size = 8;  // Optimal batch size for throughput
    let mut last_batch_time = tokio::time::Instant::now();
    let batch_timeout = std::time::Duration::from_millis(100);

    while let Some(msg) = socket.recv().await {
        if let Ok(msg) = msg {
            // Handle audio data
            if let Ok(text) = msg.to_text() {
                // Check for batch command
                if text.starts_with("BATCH:") {
                    // Handle batched audio data
                    let batch_data = &text[6..];
                    let batches: Vec<Vec<f32>> = batch_data
                        .split(';')
                        .map(|batch| {
                            batch.split(',')
                                .filter_map(|s| s.parse().ok())
                                .collect()
                        })
                        .filter(|b: &Vec<f32>| !b.is_empty())
                        .collect();

                    if !batches.is_empty() {
                        // Process batch directly
                        let mut translator_guard = translator.lock().await;
                        match translator_guard.process_batch(batches).await {
                            Ok(results) => {
                                let response = serde_json::json!({
                                    "batch_results": results.iter().map(|r| serde_json::json!({
                                        "text": r.original_text,
                                        "translated": r.translated_text,
                                        "confidence": r.confidence,
                                        "latency_ms": r.processing_time_ms,
                                    })).collect::<Vec<_>>(),
                                    "batch_size": results.len(),
                                });

                                if let Ok(json) = serde_json::to_string(&response) {
                                    let _ = socket.send(json.into()).await;
                                }
                            }
                            Err(e) => {
                                error!("Batch processing error: {}", e);
                            }
                        }
                    }
                } else {
                    // Single sample - add to batch buffer
                    let samples: Vec<f32> = text
                        .split(',')
                        .filter_map(|s| s.parse().ok())
                        .collect();

                    if !samples.is_empty() {
                        batch_buffer.push(samples);

                        // Process batch if full or timeout
                        let should_process = batch_buffer.len() >= batch_size ||
                            last_batch_time.elapsed() > batch_timeout;

                        if should_process && !batch_buffer.is_empty() {
                            let batch = std::mem::take(&mut batch_buffer);
                            last_batch_time = tokio::time::Instant::now();

                            // Process batch
                            let mut translator_guard = translator.lock().await;
                            match translator_guard.process_batch(batch).await {
                                Ok(results) => {
                                    for result in results {
                                        let response = serde_json::json!({
                                            "text": result.original_text,
                                            "translated": result.translated_text,
                                            "confidence": result.confidence,
                                            "latency_ms": result.processing_time_ms,
                                        });

                                        if let Ok(json) = serde_json::to_string(&response) {
                                            let _ = socket.send(json.into()).await;
                                        }
                                    }
                                }
                                Err(e) => {
                                    error!("Batch processing error: {}", e);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Process remaining batch
    if !batch_buffer.is_empty() {
        let mut translator_guard = translator.lock().await;
        let _ = translator_guard.process_batch(batch_buffer).await;
    }

    info!("WebSocket connection closed");
}