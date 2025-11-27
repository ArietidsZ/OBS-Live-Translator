use crate::Result;
use async_trait::async_trait;

pub mod cobra;
pub mod silero;
pub mod ten_vad;
// pub mod silero; // Will migrate later

/// Trait for Voice Activity Detection engines
#[async_trait]
pub trait VadEngine: Send + Sync {
    /// Detect speech in audio samples
    /// Returns true if speech is detected
    async fn detect(&self, samples: &[f32]) -> Result<bool>;

    /// Reset internal state (if any)
    fn reset(&mut self);

    /// Set detection threshold
    fn set_threshold(&mut self, threshold: f32);

    /// Get engine name
    fn name(&self) -> &str;
}

use crate::types::{TranslatorConfig, VadType};

use std::sync::Arc;

/// Create a VAD engine based on configuration
pub async fn create_vad_engine(
    config: &TranslatorConfig,
    cache: Arc<crate::models::session_cache::SessionCache>,
) -> Result<Arc<dyn VadEngine>> {
    match config.vad_type {
        VadType::TenVad => {
            let engine = ten_vad::TenVad::new(config, cache).await?;
            Ok(Arc::new(engine))
        }
        VadType::Cobra => {
            let engine = cobra::CobraVad::new(config)?;
            Ok(Arc::new(engine))
        }
        VadType::SileroV5 => {
            let engine = silero::SileroVad::new(config, cache).await?;
            Ok(Arc::new(engine))
        }
        VadType::Silero => {
            // Legacy Silero VAD
            let engine = crate::audio::vad::SileroVAD::new(config, cache).await?;
            Ok(Arc::new(engine))
        }
    }
}
