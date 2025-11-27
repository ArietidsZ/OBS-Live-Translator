//! ONNX Runtime inference engine with hardware acceleration

use crate::execution_provider::ExecutionProviderConfig;
use crate::{Error, Result};
use ort::session::Session;
use std::path::PathBuf;

/// Precision mode for model execution
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    FP32,
    FP16,
    BF16,
    INT8,
}

use crate::models::session_cache::SessionCache;
use std::sync::Arc;
use tokio::sync::OnceCell;

/// Inference engine wrapper around ONNX Runtime
/// Stores model path and creates session on demand to avoid lifetime issues
pub struct InferenceEngine {
    model_path: PathBuf,
    precision: Precision,
    provider_config: ExecutionProviderConfig,
    acceleration_config: crate::types::AccelerationConfig,
    cache: Arc<SessionCache>,
    session: OnceCell<Arc<Session>>,
}

impl InferenceEngine {
    /// Create a new inference engine from ONNX model file
    pub fn new<P: AsRef<std::path::Path>>(
        model_path: P,
        precision: Precision,
        provider_config: &ExecutionProviderConfig,
        acceleration_config: &crate::types::AccelerationConfig,
        cache: Arc<SessionCache>,
    ) -> Result<Self> {
        let path = model_path.as_ref().to_path_buf();

        // Verify file exists
        if !path.exists() {
            return Err(Error::ModelLoad(format!(
                "Model file not found: {}",
                path.display()
            )));
        }

        tracing::info!("Registered ONNX model: {:?}", path);

        Ok(Self {
            model_path: path,
            precision,
            provider_config: provider_config.clone(),
            acceleration_config: acceleration_config.clone(),
            cache,
            session: OnceCell::new(),
        })
    }

    /// Get session for inference (lazy load)
    pub async fn get_session(&self) -> Result<Arc<Session>> {
        self.session
            .get_or_try_init(|| async {
                let mut current_config = self.provider_config.clone();

                loop {
                    match self.cache.get_or_load(
                        &self.model_path,
                        &current_config,
                        &self.acceleration_config,
                    ) {
                        Ok(session) => return Ok(session),
                        Err(e) => {
                            tracing::warn!(
                                "Failed to load model {} with provider {}: {}",
                                self.model_path.display(),
                                current_config.provider().name(),
                                e
                            );

                            if let Some(fallback) = current_config.fallback() {
                                tracing::info!(
                                    "Falling back to provider: {}",
                                    fallback.provider().name()
                                );
                                current_config = fallback;
                            } else {
                                // No more fallbacks
                                return Err(e);
                            }
                        }
                    }
                }
            })
            .await.cloned()
    }

    /// Create a new session (bypass cache - legacy/debug use)
    pub fn create_session_uncached(&self) -> Result<Session> {
        let builder = Session::builder()?;

        // Configure execution provider
        let builder = self
            .provider_config
            .configure_session(builder, &self.acceleration_config)
            .map_err(|e| Error::Acceleration(e.to_string()))?;

        // Load model
        let session = builder.commit_from_file(&self.model_path)?;

        Ok(session)
    }

    /// Get model path
    pub fn model_path(&self) -> &std::path::Path {
        &self.model_path
    }

    /// Get the precision mode
    pub fn precision(&self) -> Precision {
        self.precision
    }
}

pub mod optimization;
pub mod quantization;

// Note: graph optimization and model caching are handled internally by ONNX Runtime

/// Retry an async operation with exponential backoff
pub async fn retry<F, Fut, T, E>(operation: F, max_retries: u32) -> std::result::Result<T, E>
where
    F: Fn() -> Fut,
    Fut: std::future::Future<Output = std::result::Result<T, E>>,
    E: std::fmt::Display,
{
    let mut last_err = None;
    for i in 0..=max_retries {
        match operation().await {
            Ok(result) => return Ok(result),
            Err(e) => {
                tracing::warn!(
                    "Operation failed (attempt {}/{}): {}",
                    i + 1,
                    max_retries + 1,
                    e
                );
                last_err = Some(e);
                if i < max_retries {
                    tokio::time::sleep(std::time::Duration::from_millis(100 * (1 << i))).await;
                    // Exponential backoff
                }
            }
        }
    }
    Err(last_err.unwrap())
}
