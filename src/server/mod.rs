//! Server setup and shared application state.

use std::sync::{
    atomic::{AtomicBool, AtomicUsize, Ordering},
    Arc,
};

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::get,
    Router,
};

use metrics_exporter_prometheus::PrometheusHandle;

use crate::{streaming, Translator};

/// Shared application state for the HTTP server.
pub struct AppState {
    translator: Arc<Translator>,
    is_ready: AtomicBool,
    connections: AtomicUsize,
}

impl AppState {
    /// Create a new application state with readiness unset.
    pub fn new(translator: Arc<Translator>) -> Self {
        Self {
            translator,
            is_ready: AtomicBool::new(false),
            connections: AtomicUsize::new(0),
        }
    }

    /// Mark the server as ready to accept requests.
    pub fn mark_ready(&self) {
        self.is_ready.store(true, Ordering::SeqCst);
    }

    /// Track a new WebSocket connection and return the total.
    pub fn connection_opened(&self) -> usize {
        self.connections.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Track a closed WebSocket connection and return the remaining total.
    pub fn connection_closed(&self) -> usize {
        let mut current = self.connections.load(Ordering::SeqCst);
        loop {
            if current == 0 {
                return 0;
            }

            let next = current - 1;
            match self.connections.compare_exchange(
                current,
                next,
                Ordering::SeqCst,
                Ordering::SeqCst,
            ) {
                Ok(_) => return next,
                Err(updated) => current = updated,
            }
        }
    }

    /// Access the translator for downstream handlers.
    pub fn translator(&self) -> Arc<Translator> {
        self.translator.clone()
    }
}

/// Build the HTTP router for the service.
pub fn build_router(state: Arc<AppState>, recorder_handle: PrometheusHandle) -> Router {
    Router::new()
        .route(
            "/metrics",
            get(move || {
                let handle = recorder_handle.clone();
                async move { handle.render() }
            }),
        )
        .route("/healthz", get(health))
        .route("/readyz", get(readiness))
        .route("/ws", get(streaming::websocket::handle_websocket))
        .with_state(state)
}

/// Liveness probe endpoint.
async fn health() -> &'static str {
    "OK"
}

/// Readiness probe endpoint.
async fn readiness(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    if state.is_ready.load(Ordering::SeqCst) {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}

/// Wait for a shutdown signal.
pub async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };

    #[cfg(unix)]
    let terminate = async {
        use tokio::signal::unix::{signal, SignalKind};

        let mut sigterm = signal(SignalKind::terminate()).expect("failed to install SIGTERM handler");
        sigterm.recv().await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }

    tracing::info!("Shutdown signal received");
}
