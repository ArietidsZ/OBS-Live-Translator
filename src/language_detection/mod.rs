use crate::{
    types::{LanguageDetection, LanguageDetectionStrategy, TranslatorConfig},
    Result,
};
use std::sync::Arc;

pub mod fasttext_detector;
pub mod lingua_detector;
pub mod whatlang_detector;

/// Multi-detector with fusion strategy
pub struct MultiDetector {
    primary: Arc<whatlang_detector::WhatlangDetector>,
    fasttext: Option<Arc<fasttext_detector::FastTextDetector>>,
    lingua: Option<Arc<lingua_detector::LinguaDetector>>,
    confidence_threshold: f32,
    strategy: LanguageDetectionStrategy,
}

impl MultiDetector {
    /// Create a new multi-detector
    pub async fn new(config: &TranslatorConfig) -> Result<Self> {
        tracing::info!("Initializing language detection");

        // Always use Whatlang as primary (fast, lightweight)
        let primary = Arc::new(whatlang_detector::WhatlangDetector::new()?);

        // FastText as fallback for longer text
        let fasttext = match fasttext_detector::FastTextDetector::new(&config.model_path) {
            Ok(detector) => Some(Arc::new(detector)),
            Err(e) => {
                tracing::warn!("FastText detector not available: {}", e);
                None
            }
        };

        // Lingua as high-accuracy fallback (if enabled)
        let lingua = if cfg!(feature = "lingua-fallback") {
            match lingua_detector::LinguaDetector::new() {
                Ok(detector) => Some(Arc::new(detector)),
                Err(e) => {
                    tracing::warn!("Lingua detector failed to initialize: {}", e);
                    None
                }
            }
        } else {
            None
        };

        tracing::info!("Language detection initialized");

        Ok(Self {
            primary,
            fasttext,
            lingua,
            confidence_threshold: 0.7,
            strategy: config.language_detection_strategy,
        })
    }

    /// Detect language from text with optional ASR hint
    pub async fn detect(&self, text: &str, asr_hint: Option<&str>) -> Result<String> {
        // Apply strategy
        match self.strategy {
            LanguageDetectionStrategy::AsrOnly => {
                if let Some(hint) = asr_hint {
                    return Ok(hint.to_string());
                }
                // Fallback if no hint available
            }
            LanguageDetectionStrategy::Hybrid => {
                // Trust ASR hint if available (usually high confidence from model)
                if let Some(hint) = asr_hint {
                    return Ok(hint.to_string());
                }
            }
            LanguageDetectionStrategy::TextOnly => {
                // Ignore hint, proceed to text detection
            }
        }

        // Text-based detection pipeline

        // 1. Try Whatlang (fastest)
        let primary_result = self.primary.detect(text)?;

        // If confidence is high enough, use primary result
        if primary_result.confidence >= self.confidence_threshold {
            return Ok(primary_result.language);
        }

        // 2. Try Lingua (most accurate, but slower) if available
        if let Some(ref lingua) = self.lingua {
            let lingua_result = lingua.detect(text)?;
            // Lingua is very accurate, so we trust it if it returns a result
            if lingua_result.language != "unknown" {
                return Ok(lingua_result.language);
            }
        }

        // 3. Try FastText for more context
        if let Some(ref fasttext) = self.fasttext {
            let ft_result = fasttext.detect(text)?;

            // Use the result with higher confidence
            if ft_result.confidence > primary_result.confidence {
                return Ok(ft_result.language);
            }
        }

        // Fallback to primary result
        Ok(primary_result.language)
    }

    /// Get detailed detection results from all detectors
    pub async fn detect_detailed(&self, text: &str) -> Result<Vec<LanguageDetection>> {
        let mut results = Vec::new();

        // Primary detection
        let primary_result = self.primary.detect(text)?;
        results.push(primary_result);

        // Lingua detection
        if let Some(ref lingua) = self.lingua {
            if let Ok(l_result) = lingua.detect(text) {
                results.push(l_result);
            }
        }

        // FastText detection (if available)
        if let Some(ref fasttext) = self.fasttext {
            let ft_result = fasttext.detect(text)?;
            results.push(ft_result);
        }

        Ok(results)
    }
}
