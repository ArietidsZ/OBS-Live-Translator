//! Model management system with automated download, validation, and caching
//!
//! This module implements comprehensive model management:
//! - Profile-aware model selection and downloading
//! - Model validation and integrity checking
//! - Automatic quantization pipeline
//! - Model caching and version management
//! - Cross-platform model deployment

pub mod downloader;
pub mod validator;
pub mod quantizer;
pub mod cache;

use crate::profile::Profile;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tracing::{info, warn, error};

/// Model configuration for different profiles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub profile: Profile,
    pub models: Vec<ModelInfo>,
}

/// Information about a specific model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub version: String,
    pub url: String,
    pub checksum: String,
    pub size_mb: u64,
    pub quantized: bool,
    pub model_type: ModelType,
    pub required_memory_mb: u64,
    pub supported_languages: Vec<String>,
}

/// Type of model for different tasks
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ModelType {
    VAD,        // Voice Activity Detection
    ASR,        // Automatic Speech Recognition
    Translation, // Translation model
    Resampling, // Audio resampling
}

/// Model manager for automated operations
pub struct ModelManager {
    profile: Profile,
    models_dir: PathBuf,
    cache_dir: PathBuf,
    downloader: downloader::ModelDownloader,
    validator: validator::ModelValidator,
    quantizer: quantizer::ModelQuantizer,
    cache: cache::ModelCache,
}

impl ModelManager {
    /// Create a new model manager for the given profile
    pub fn new(profile: Profile) -> Result<Self> {
        info!("🤖 Initializing model manager for profile {:?}", profile);

        let models_dir = PathBuf::from(env!("MODELS_DIR"));
        let cache_dir = PathBuf::from(env!("MODEL_CACHE_DIR"));

        // Create directories if they don't exist
        std::fs::create_dir_all(&models_dir)?;
        std::fs::create_dir_all(&cache_dir)?;

        let downloader = downloader::ModelDownloader::new(&models_dir)?;
        let validator = validator::ModelValidator::new()?;
        let quantizer = quantizer::ModelQuantizer::new(profile)?;
        let cache = cache::ModelCache::new(&cache_dir)?;

        info!("✅ Model manager initialized for profile {:?}", profile);

        Ok(Self {
            profile,
            models_dir,
            cache_dir,
            downloader,
            validator,
            quantizer,
            cache,
        })
    }

    /// Get model configuration for the current profile
    pub fn get_profile_models(&self) -> Result<Vec<ModelInfo>> {
        let models = match self.profile {
            Profile::Low => vec![
                ModelInfo {
                    name: "webrtc_vad".to_string(),
                    version: "1.0.0".to_string(),
                    url: "https://github.com/webrtc/webrtc/raw/main/common_audio/vad/vad_core.c".to_string(),
                    checksum: "sha256:a1b2c3d4e5f6...".to_string(),
                    size_mb: 1,
                    quantized: false,
                    model_type: ModelType::VAD,
                    required_memory_mb: 1,
                    supported_languages: vec!["*".to_string()], // Language agnostic
                },
                ModelInfo {
                    name: "linear_resampler".to_string(),
                    version: "1.0.0".to_string(),
                    url: "built-in".to_string(),
                    checksum: "built-in".to_string(),
                    size_mb: 0,
                    quantized: false,
                    model_type: ModelType::Resampling,
                    required_memory_mb: 2,
                    supported_languages: vec!["*".to_string()],
                },
            ],
            Profile::Medium => vec![
                ModelInfo {
                    name: "ten_vad".to_string(),
                    version: "1.0.0".to_string(),
                    url: "https://huggingface.co/microsoft/DialoGPT-medium/resolve/main/ten_vad.onnx".to_string(),
                    checksum: "sha256:x1y2z3a4b5c6...".to_string(),
                    size_mb: 15,
                    quantized: true,
                    model_type: ModelType::VAD,
                    required_memory_mb: 15,
                    supported_languages: vec!["*".to_string()],
                },
                ModelInfo {
                    name: "whisper_base".to_string(),
                    version: "1.0.0".to_string(),
                    url: "https://huggingface.co/openai/whisper-base/resolve/main/model.onnx".to_string(),
                    checksum: "sha256:m1n2o3p4q5r6...".to_string(),
                    size_mb: 142,
                    quantized: true,
                    model_type: ModelType::ASR,
                    required_memory_mb: 200,
                    supported_languages: vec!["en".to_string(), "es".to_string(), "fr".to_string(), "de".to_string()],
                },
                ModelInfo {
                    name: "nllb_200_600m".to_string(),
                    version: "1.0.0".to_string(),
                    url: "https://huggingface.co/facebook/nllb-200-600M/resolve/main/model.onnx".to_string(),
                    checksum: "sha256:t1u2v3w4x5y6...".to_string(),
                    size_mb: 600,
                    quantized: true,
                    model_type: ModelType::Translation,
                    required_memory_mb: 800,
                    supported_languages: vec![
                        "en".to_string(), "es".to_string(), "fr".to_string(), "de".to_string(),
                        "it".to_string(), "pt".to_string(), "ru".to_string(), "ja".to_string(),
                        "ko".to_string(), "zh".to_string(), "ar".to_string()
                    ],
                },
            ],
            Profile::High => vec![
                ModelInfo {
                    name: "silero_vad".to_string(),
                    version: "4.0.0".to_string(),
                    url: "https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx".to_string(),
                    checksum: "sha256:z1a2b3c4d5e6...".to_string(),
                    size_mb: 50,
                    quantized: false,
                    model_type: ModelType::VAD,
                    required_memory_mb: 50,
                    supported_languages: (0..100).map(|i| format!("lang_{}", i)).collect(),
                },
                ModelInfo {
                    name: "whisper_large_v3".to_string(),
                    version: "3.0.0".to_string(),
                    url: "https://huggingface.co/openai/whisper-large-v3/resolve/main/model.onnx".to_string(),
                    checksum: "sha256:g1h2i3j4k5l6...".to_string(),
                    size_mb: 3100,
                    quantized: false,
                    model_type: ModelType::ASR,
                    required_memory_mb: 4000,
                    supported_languages: (0..100).map(|i| format!("lang_{}", i)).collect(),
                },
                ModelInfo {
                    name: "nllb_200_3_3b".to_string(),
                    version: "1.0.0".to_string(),
                    url: "https://huggingface.co/facebook/nllb-200-3.3B/resolve/main/model.onnx".to_string(),
                    checksum: "sha256:p1q2r3s4t5u6...".to_string(),
                    size_mb: 6600,
                    quantized: false,
                    model_type: ModelType::Translation,
                    required_memory_mb: 8000,
                    supported_languages: (0..200).map(|i| format!("lang_{}", i)).collect(),
                },
            ],
        };

        Ok(models)
    }

    /// Setup all models for the current profile
    pub async fn setup_profile_models(&mut self) -> Result<()> {
        info!("🚀 Setting up models for profile {:?}", self.profile);

        let models = self.get_profile_models()?;
        let mut setup_count = 0;

        for model in &models {
            match self.setup_model(&model).await {
                Ok(_) => {
                    setup_count += 1;
                    info!("✅ Model {} setup complete", model.name);
                }
                Err(e) => {
                    error!("❌ Failed to setup model {}: {}", model.name, e);
                    // For critical models, we should fail the entire setup
                    if model.model_type == ModelType::VAD {
                        return Err(anyhow::anyhow!("Critical VAD model setup failed: {}", e));
                    }
                }
            }
        }

        info!("🎉 Profile setup complete: {}/{} models ready", setup_count, models.len());
        Ok(())
    }

    /// Setup a single model (download, validate, quantize if needed, cache)
    async fn setup_model(&mut self, model: &ModelInfo) -> Result<()> {
        info!("🔧 Setting up model: {} v{}", model.name, model.version);

        // Check if model is already cached and valid
        if let Ok(cached_path) = self.cache.get_model_path(&model.name, &model.version) {
            if self.validator.validate_model(&cached_path, &model.checksum).await? {
                info!("✅ Model {} found in cache and valid", model.name);
                return Ok(());
            } else {
                warn!("⚠️ Cached model {} invalid, re-downloading", model.name);
                self.cache.remove_model(&model.name, &model.version)?;
            }
        }

        // Download the model
        let download_path = if model.url != "built-in" {
            Some(self.downloader.download_model(model).await?)
        } else {
            None
        };

        // Validate the downloaded model
        if let Some(ref path) = download_path {
            if !self.validator.validate_model(path, &model.checksum).await? {
                return Err(anyhow::anyhow!("Model validation failed: {}", model.name));
            }
        }

        // Quantize if needed and profile supports it
        let final_path = if model.quantized && self.should_quantize_for_profile() {
            if let Some(download_path) = download_path {
                self.quantizer.quantize_model(&download_path, model).await?
            } else {
                return Err(anyhow::anyhow!("Cannot quantize built-in model: {}", model.name));
            }
        } else {
            download_path.unwrap_or_else(|| PathBuf::from("built-in"))
        };

        // Cache the final model
        if final_path != PathBuf::from("built-in") {
            self.cache.cache_model(&model.name, &model.version, &final_path)?;
        }

        info!("🎯 Model {} setup completed successfully", model.name);
        Ok(())
    }

    /// Check if quantization should be performed for current profile
    fn should_quantize_for_profile(&self) -> bool {
        match self.profile {
            Profile::Low => false,    // No quantization for low profile (too slow)
            Profile::Medium => true,  // Quantize for balanced performance/memory
            Profile::High => false,   // No quantization for maximum quality
        }
    }

    /// Get path to a specific model
    pub fn get_model_path(&mut self, model_name: &str, version: &str) -> Result<PathBuf> {
        self.cache.get_model_path(model_name, version)
    }

    /// Check model availability for current profile
    pub fn check_model_availability(&mut self) -> Result<ModelAvailability> {
        let models = self.get_profile_models()?;
        let mut availability = ModelAvailability::default();

        for model in models {
            let is_available = match self.cache.get_model_path(&model.name, &model.version) {
                Ok(path) => path.exists(),
                Err(_) => false,
            };

            match model.model_type {
                ModelType::VAD => availability.vad_available = is_available,
                ModelType::ASR => availability.asr_available = is_available,
                ModelType::Translation => availability.translation_available = is_available,
                ModelType::Resampling => availability.resampling_available = true, // Built-in
            }
        }

        Ok(availability)
    }

    /// Clean up unused models and optimize cache
    pub async fn cleanup_models(&mut self) -> Result<()> {
        info!("🧹 Cleaning up unused models");

        // Remove models not needed for current profile
        let current_models: std::collections::HashSet<String> = self
            .get_profile_models()?
            .iter()
            .map(|m| format!("{}_{}", m.name, m.version))
            .collect();

        let removed_count = self.cache.cleanup_unused_models(&current_models)?;

        info!("✅ Cleanup complete: {} unused models removed", removed_count);
        Ok(())
    }
}

/// Model availability status
#[derive(Debug, Default)]
pub struct ModelAvailability {
    pub vad_available: bool,
    pub asr_available: bool,
    pub translation_available: bool,
    pub resampling_available: bool,
}

impl ModelAvailability {
    /// Check if all required models are available
    pub fn all_available(&self) -> bool {
        self.vad_available && self.asr_available && self.translation_available && self.resampling_available
    }

    /// Get missing model types
    pub fn missing_models(&self) -> Vec<ModelType> {
        let mut missing = Vec::new();
        if !self.vad_available { missing.push(ModelType::VAD); }
        if !self.asr_available { missing.push(ModelType::ASR); }
        if !self.translation_available { missing.push(ModelType::Translation); }
        if !self.resampling_available { missing.push(ModelType::Resampling); }
        missing
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_info_creation() {
        let model = ModelInfo {
            name: "test_model".to_string(),
            version: "1.0.0".to_string(),
            url: "https://example.com/model.onnx".to_string(),
            checksum: "sha256:abc123".to_string(),
            size_mb: 100,
            quantized: true,
            model_type: ModelType::VAD,
            required_memory_mb: 150,
            supported_languages: vec!["en".to_string()],
        };

        assert_eq!(model.name, "test_model");
        assert_eq!(model.model_type, ModelType::VAD);
        assert!(model.quantized);
    }

    #[tokio::test]
    async fn test_model_manager_creation() {
        let manager = ModelManager::new(Profile::Low).unwrap();
        assert_eq!(manager.profile, Profile::Low);
    }

    #[test]
    fn test_profile_models() {
        let manager = ModelManager::new(Profile::Medium).unwrap();
        let models = manager.get_profile_models().unwrap();

        // Medium profile should have VAD, ASR, and Translation models
        assert!(models.iter().any(|m| m.model_type == ModelType::VAD));
        assert!(models.iter().any(|m| m.model_type == ModelType::ASR));
        assert!(models.iter().any(|m| m.model_type == ModelType::Translation));
    }

    #[test]
    fn test_model_availability() {
        let mut availability = ModelAvailability::default();
        assert!(!availability.all_available());

        availability.vad_available = true;
        availability.asr_available = true;
        availability.translation_available = true;
        availability.resampling_available = true;

        assert!(availability.all_available());
        assert!(availability.missing_models().is_empty());
    }
}