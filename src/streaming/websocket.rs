//! WebSocket server implementation

use axum::{
    extract::{
        ws::{Message, WebSocket, WebSocketUpgrade},
        State,
    },
    response::Response,
};

use crate::{server::AppState, streaming::StreamMessage};

/// Handle WebSocket upgrade
pub async fn handle_websocket(
    State(state): State<std::sync::Arc<AppState>>,
    ws: WebSocketUpgrade,
) -> Response {
    ws.on_upgrade(move |socket| handle_socket(state, socket))
}

async fn handle_socket(state: std::sync::Arc<AppState>, mut socket: WebSocket) {
    let connections = state.connection_opened();
    tracing::info!(connections, "WebSocket connection established");

    while let Some(message) = socket.recv().await {
        match message {
            Ok(Message::Text(payload)) => {
                match serde_json::from_str::<StreamMessage>(&payload) {
                    Ok(StreamMessage::Ping) => {
                        if let Err(error) = send_message(&mut socket, &StreamMessage::Pong).await {
                            tracing::warn!("Failed to send pong: {}", error);
                        }
                    }
                    Ok(StreamMessage::Config {
                        source_language,
                        target_language,
                        profile,
                    }) => {
                        tracing::info!(
                            "Received config: source={:?} target={} profile={:?}",
                            source_language,
                            target_language,
                            profile
                        );
                    }
                    Ok(StreamMessage::Error { message }) => {
                        tracing::warn!("Client error message: {}", message);
                    }
                    Ok(StreamMessage::Result(_)) => {
                        tracing::info!("Received result message from client");
                    }
                    Ok(StreamMessage::Audio { .. }) => {
                        tracing::info!("Received audio message over text channel");
                    }
                    Ok(StreamMessage::Pong) => {
                        tracing::debug!("Received pong from client");
                    }
                    Err(error) => {
                        tracing::warn!("Invalid stream message: {}", error);
                        let _ = send_message(
                            &mut socket,
                            &StreamMessage::Error {
                                message: format!("Invalid message: {}", error),
                            },
                        )
                        .await;
                    }
                }
            }
            Ok(Message::Binary(payload)) => match parse_audio_samples(&payload) {
                Ok(samples) => {
                    tracing::debug!("Received {} audio samples", samples.len());
                }
                Err(error) => {
                    tracing::warn!("Invalid audio payload: {}", error);
                    let _ = send_message(
                        &mut socket,
                        &StreamMessage::Error {
                            message: format!("Invalid audio payload: {}", error),
                        },
                    )
                    .await;
                }
            },
            Ok(Message::Close(reason)) => {
                tracing::info!("WebSocket closed: {:?}", reason);
                break;
            }
            Ok(Message::Ping(payload)) => {
                tracing::debug!("Received ping: {} bytes", payload.len());
            }
            Ok(Message::Pong(payload)) => {
                tracing::debug!("Received pong: {} bytes", payload.len());
            }
            Err(error) => {
                tracing::warn!("WebSocket error: {}", error);
                break;
            }
        }
    }

    let connections = state.connection_closed();
    tracing::info!(connections, "WebSocket connection closed");
}

async fn send_message(socket: &mut WebSocket, message: &StreamMessage) -> anyhow::Result<()> {
    let payload = serde_json::to_string(message)?;
    socket.send(Message::Text(payload)).await?;
    Ok(())
}

fn parse_audio_samples(payload: &[u8]) -> anyhow::Result<Vec<f32>> {
    if payload.len() % 4 != 0 {
        anyhow::bail!("payload length {} is not multiple of 4", payload.len());
    }

    let mut samples = Vec::with_capacity(payload.len() / 4);
    for chunk in payload.chunks_exact(4) {
        let bytes: [u8; 4] = chunk
            .try_into()
            .expect("chunk size verified at 4 bytes");
        samples.push(f32::from_le_bytes(bytes));
    }
    Ok(samples)
}
