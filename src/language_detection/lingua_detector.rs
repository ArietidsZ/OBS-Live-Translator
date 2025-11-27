//! Lingua-based language detector (High accuracy, slower)
//!
//! Uses the lingua crate for high-accuracy language detection.
//! This is heavier than Whatlang but more accurate for short text.

use crate::{types::LanguageDetection, Result};

#[cfg(feature = "lingua-fallback")]
use lingua::{Language, LanguageDetector, LanguageDetectorBuilder};

/// Lingua-based language detector
pub struct LinguaDetector {
    #[cfg(feature = "lingua-fallback")]
    detector: LanguageDetector,
}

impl LinguaDetector {
    /// Create a new Lingua detector
    pub fn new() -> Result<Self> {
        tracing::info!("Initializing Lingua language detector");

        #[cfg(feature = "lingua-fallback")]
        {
            // Configure detector with common languages to reduce memory/startup time
            // or use all languages if needed. For now, let's use all spoken languages.
            let detector = LanguageDetectorBuilder::from_all_spoken_languages()
                .with_minimum_relative_distance(0.1)
                .build();

            Ok(Self { detector })
        }

        #[cfg(not(feature = "lingua-fallback"))]
        {
            Err(crate::Error::Config(
                "Lingua feature not enabled".to_string(),
            ))
        }
    }

    /// Detect language from text
    pub fn detect(&self, _text: &str) -> Result<LanguageDetection> {
        #[cfg(feature = "lingua-fallback")]
        {
            let result = self.detector.detect_language_of(text);

            let (language, confidence) = if let Some(lang) = result {
                // Map Lingua Language enum to ISO 639-1 code
                let code = lang.iso_code_639_1().to_string();
                // Lingua doesn't provide a direct confidence score in the simple API,
                // but we can assume high confidence if it returned a result.
                // For more detail we could use compute_language_confidence_values
                (code, 0.9)
            } else {
                ("unknown".to_string(), 0.0)
            };

            Ok(LanguageDetection {
                language,
                confidence: confidence as f32,
                alternatives: Vec::new(),
            })
        }

        #[cfg(not(feature = "lingua-fallback"))]
        {
            Ok(LanguageDetection {
                language: "unknown".to_string(),
                confidence: 0.0,
                alternatives: Vec::new(),
            })
        }
    }

    /// Detect with detailed confidence values
    #[cfg(feature = "lingua-fallback")]
    pub fn detect_detailed(&self, text: &str) -> Result<Vec<(String, f32)>> {
        let results = self.detector.compute_language_confidence_values(text);

        let mapped = results
            .into_iter()
            .map(|(lang, conf)| (lang.iso_code_639_1().to_string(), conf as f32))
            .collect();

        Ok(mapped)
    }
}
