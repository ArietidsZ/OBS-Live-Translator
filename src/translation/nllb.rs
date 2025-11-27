//! NLLB-200 - No Language Left Behind (Meta AI)
//!
//! 202 languages with focus on low-resource languages
//! Part 2.2: Enhanced with INT8 quantization support

use crate::{
    execution_provider::ExecutionProviderConfig,
    inference::{InferenceEngine, Precision},
    platform_detect::Platform,
    translation::TranslationEngine,
    types::TranslatorConfig,
    Error, Result,
};
use async_trait::async_trait;
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;

/// NLLB-200 translation engine
#[allow(dead_code)]
pub struct NLLB200 {
    encoder: Arc<InferenceEngine>,
    decoder: Arc<InferenceEngine>,
    tokenizer: Arc<Tokenizer>,
    precision: Precision,
}

impl NLLB200 {
    /// Create a new NLLB-200 instance
    /// Part 2.2: Now supports INT8 quantization
    pub async fn new(
        config: &TranslatorConfig,
        requested_precision: Precision,
        cache: Arc<crate::models::session_cache::SessionCache>,
    ) -> Result<Self> {
        // Determine best available precision
        let (precision, encoder_path, decoder_path) =
            Self::select_models(config, requested_precision)?;

        let precision_str = match precision {
            Precision::INT8 => "INT8",
            Precision::FP16 => "FP16",
            _ => "FP32",
        };

        tracing::info!("Loading NLLB-200 ({})", precision_str);
        tracing::info!("  Encoder: {}", encoder_path);
        tracing::info!("  Decoder: {}", decoder_path);

        let model_path = &config.model_path;
        let tokenizer_path = format!("{model_path}/nllb-tokenizer.json");

        // Detect platform and configure execution provider
        let platform = Platform::detect().map_err(|e| Error::Acceleration(e.to_string()))?;
        let provider_config = ExecutionProviderConfig::from_platform(platform);
        tracing::info!("Execution Provider: {}", provider_config.summary());

        let encoder = InferenceEngine::new(
            &encoder_path,
            precision,
            &provider_config,
            &config.acceleration,
            cache.clone(),
        )?;
        let decoder = InferenceEngine::new(
            &decoder_path,
            precision,
            &provider_config,
            &config.acceleration,
            cache,
        )?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| Error::Tokenization(format!("Failed to load tokenizer: {e}")))?;

        tracing::info!("NLLB-200 loaded successfully ({})", precision_str);

        Ok(Self {
            encoder: Arc::new(encoder),
            decoder: Arc::new(decoder),
            tokenizer: Arc::new(tokenizer),
            precision,
        })
    }

    /// Select best available models (INT8 preferred, FP16/FP32 fallback)
    /// Part 2.2: Auto-detection with graceful degradation
    fn select_models(
        config: &TranslatorConfig,
        requested: Precision,
    ) -> Result<(Precision, String, String)> {
        let base_path = &config.model_path;

        // Try INT8 first (if requested or as default for Medium profile)
        if matches!(requested, Precision::INT8) || matches!(requested, Precision::FP16) {
            let encoder_int8 = format!("{base_path}/nllb-encoder.onnx");
            let decoder_int8 = format!("{base_path}/nllb-decoder-int8.onnx");

            if Path::new(&encoder_int8).exists() && Path::new(&decoder_int8).exists() {
                tracing::info!("Using INT8 quantized NLLB model (50% memory, 1.5-2x faster)");
                return Ok((Precision::INT8, encoder_int8, decoder_int8));
            }
        }

        // Fallback to FP16
        let encoder_fp16 = format!("{base_path}/nllb-encoder.onnx");
        let decoder_fp16 = format!("{base_path}/nllb-decoder-fp16.onnx");

        if Path::new(&encoder_fp16).exists() && Path::new(&decoder_fp16).exists() {
            tracing::info!("Using FP16 NLLB model (INT8 not available)");
            return Ok((Precision::FP16, encoder_fp16, decoder_fp16));
        }

        // Fallback to FP32
        let encoder_fp32 = format!("{base_path}/nllb-encoder.onnx");
        let decoder_fp32 = format!("{base_path}/nllb-decoder.onnx");

        if Path::new(&encoder_fp32).exists() && Path::new(&decoder_fp32).exists() {
            tracing::info!("Using FP32 NLLB model");
            return Ok((Precision::FP32, encoder_fp32, decoder_fp32));
        }

        Err(Error::ModelLoad(
            "NLLB-200 model not found (no encoder/decoder pair available)".to_string(),
        ))
    }
}

#[async_trait]
impl TranslationEngine for NLLB200 {
    async fn translate(&self, text: &str, source_lang: &str, target_lang: &str) -> Result<String> {
        // TODO: Implement NLLB-200 inference
        Ok(format!(
            "[NLLB translated from {source_lang} to {target_lang}]: {text}"
        ))
    }

    fn model_name(&self) -> &str {
        match self.precision {
            Precision::INT8 => "NLLB-200 (INT8)",
            Precision::FP16 => "NLLB-200 (FP16)",
            _ => "NLLB-200 (FP32)",
        }
    }

    fn language_count(&self) -> usize {
        202
    }
}
