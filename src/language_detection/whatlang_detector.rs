//! Whatlang-based language detector
//!
//! Uses the whatlang crate for fast, trigram-based language detection.
//! Replaces CLD3 due to build issues with the cld3 crate.

use crate::{types::LanguageDetection, Result};
use whatlang::{Detector, Lang};

/// Whatlang-based language detector
pub struct WhatlangDetector {
    detector: Detector,
}

impl WhatlangDetector {
    /// Create a new Whatlang detector
    pub fn new() -> Result<Self> {
        tracing::info!("Initializing Whatlang language detector");
        Ok(Self {
            detector: Detector::new(),
        })
    }

    /// Detect language from text
    pub fn detect(&self, text: &str) -> Result<LanguageDetection> {
        let info = self.detector.detect(text);

        let (language, confidence) = if let Some(info) = info {
            let lang_code = match info.lang() {
                Lang::Eng => "en",
                Lang::Fra => "fr",
                Lang::Spa => "es",
                Lang::Deu => "de",
                Lang::Ita => "it",
                Lang::Rus => "ru",
                Lang::Cmn => "zh", // whatlang uses Cmn for Mandarin
                Lang::Jpn => "ja",
                Lang::Kor => "ko",
                Lang::Ara => "ar",
                Lang::Hin => "hi",
                Lang::Por => "pt",
                Lang::Nld => "nl",
                Lang::Tur => "tr",
                Lang::Pol => "pl",
                Lang::Swe => "sv",
                Lang::Vie => "vi",
                Lang::Ukr => "uk",
                // Fallback to 3-letter code if not explicitly mapped
                l => l.code(),
            };
            (lang_code.to_string(), info.confidence())
        } else {
            ("unknown".to_string(), 0.0)
        };

        Ok(LanguageDetection {
            language,
            confidence: confidence as f32,
            alternatives: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detector_creation() {
        let detector = WhatlangDetector::new().unwrap();
        let result = detector.detect("This is a longer English sentence to ensure correct detection.").unwrap();
        assert_eq!(result.language, "en");

        let result_fr = detector.detect("Bonjour le monde").unwrap();
        assert_eq!(result_fr.language, "fr");
    }
}
