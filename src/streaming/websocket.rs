//! WebSocket server implementation

use axum::{
    extract::ws::{WebSocket, WebSocketUpgrade},
    response::Response,
};

/// Handle WebSocket upgrade
pub async fn handle_websocket(ws: WebSocketUpgrade) -> Response {
    ws.on_upgrade(handle_socket)
}

async fn handle_socket(_socket: WebSocket) {
    // TODO: Implement WebSocket handling
    tracing::info!("WebSocket connection established");
}
