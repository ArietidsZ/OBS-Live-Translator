//! Translation engines

pub mod claude;
pub mod madlad;
pub mod nllb;

use crate::Result;
use async_trait::async_trait;

/// Translation engine trait
#[async_trait]
pub trait TranslationEngine {
    /// Translate text from source language to target language
    async fn translate(&self, text: &str, source_lang: &str, target_lang: &str) -> Result<String>;

    /// Get model name
    fn model_name(&self) -> &str;

    /// Get supported language count
    fn language_count(&self) -> usize;
}
