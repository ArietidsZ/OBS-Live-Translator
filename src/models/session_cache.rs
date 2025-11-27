use crate::execution_provider::ExecutionProviderConfig;
use crate::types::AccelerationConfig;
use crate::{Error, Result};
use ort::session::Session;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// In-memory cache for ONNX Runtime sessions
/// Allows sharing loaded models across multiple engine instances
#[derive(Clone)]
pub struct SessionCache {
    sessions: Arc<Mutex<HashMap<PathBuf, Arc<Session>>>>,
}

impl SessionCache {
    /// Create a new session cache
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Get an existing session or load a new one
    pub fn get_or_load(
        &self,
        model_path: &std::path::Path,
        provider_config: &ExecutionProviderConfig,
        acceleration_config: &AccelerationConfig,
    ) -> Result<Arc<Session>> {
        let path_buf = model_path.to_path_buf();

        // Check cache first
        {
            let cache = self
                .sessions
                .lock()
                .map_err(|_| Error::ResourceExhausted("Cache lock poisoned".to_string()))?;
            if let Some(session) = cache.get(&path_buf) {
                tracing::debug!("Cache hit for model: {:?}", path_buf);
                return Ok(session.clone());
            }
        }

        // Load model if not in cache
        tracing::info!("Loading model into cache: {:?}", path_buf);
        let builder = Session::builder()?;

        // Configure execution provider
        let builder = provider_config
            .configure_session(builder, acceleration_config)
            .map_err(|e| Error::Acceleration(e.to_string()))?;

        // Load model
        let session = builder.commit_from_file(&path_buf)?;
        let session = Arc::new(session);

        // Store in cache
        {
            let mut cache = self
                .sessions
                .lock()
                .map_err(|_| Error::ResourceExhausted("Cache lock poisoned".to_string()))?;
            cache.insert(path_buf, session.clone());
        }

        Ok(session)
    }

    /// Remove a model from cache
    pub fn remove(&self, model_path: &std::path::Path) {
        if let Ok(mut cache) = self.sessions.lock() {
            cache.remove(model_path);
        }
    }

    /// Clear all cached models
    pub fn clear(&self) {
        if let Ok(mut cache) = self.sessions.lock() {
            cache.clear();
        }
    }
}

impl Default for SessionCache {
    fn default() -> Self {
        Self::new()
    }
}
