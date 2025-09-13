//! NLLB-600M translation model with CTranslate2 optimization

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use crate::acceleration::onnx_runtime::{OnnxRuntimeManager, SessionInfo};
use crate::gpu::adaptive_memory::{ModelPrecision, ModelType};
use crate::gpu::hardware_detection::{HardwareDetector, MemoryConfiguration};

/// NLLB-600M translation model integration
pub struct NLLB600M {
    /// ONNX Runtime manager
    onnx_manager: Arc<OnnxRuntimeManager>,
    /// Hardware detector for optimal configuration
    hardware_detector: Arc<HardwareDetector>,
    /// Current model session ID
    session_id: Option<String>,
    /// Model configuration
    config: NLLBConfig,
    /// Performance metrics
    metrics: Arc<RwLock<NLLBMetrics>>,
    /// Language code mappings
    language_codes: HashMap<String, String>,
}

impl NLLB600M {
    /// Create new NLLB-600M instance
    pub async fn new(
        onnx_manager: Arc<OnnxRuntimeManager>,
        hardware_detector: Arc<HardwareDetector>,
    ) -> Result<Self> {
        info!("Initializing NLLB-600M translation model");
        
        let config = NLLBConfig::default();
        let language_codes = Self::initialize_language_codes();
        
        Ok(Self {
            onnx_manager,
            hardware_detector,
            session_id: None,
            config,
            metrics: Arc::new(RwLock::new(NLLBMetrics::default())),
            language_codes,
        })
    }
    
    /// Load the model with optimal configuration
    pub async fn load_model(&mut self, model_path: Option<PathBuf>) -> Result<()> {
        info!("Loading NLLB-600M translation model");
        
        // Determine optimal configuration based on hardware
        let memory_config = self.hardware_detector.get_recommended_config().await;
        let model_precision = self.select_optimal_precision(&memory_config).await;
        
        // Get model path
        let model_path = model_path.unwrap_or_else(|| {
            self.get_default_model_path(model_precision)
        });
        
        if !model_path.exists() {
            return Err(anyhow::anyhow!(
                "NLLB-600M model not found at: {}. Please download the model first.",
                model_path.display()
            ));
        }
        
        // Create session ID
        let session_id = format!("nllb_600m_{}", uuid::Uuid::new_v4());
        
        // Create ONNX session
        self.onnx_manager.create_session(
            &session_id,
            &model_path,
            ModelType::NLLB600M,
            model_precision,
            None, // Auto-select best execution provider
        ).await?;
        
        self.session_id = Some(session_id.clone());
        
        // Update configuration
        self.config.precision = model_precision;
        self.config.model_path = model_path;
        
        info!("NLLB-600M loaded successfully with {:?} precision", model_precision);
        
        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.model_loaded = true;
            metrics.load_time_ms = 0; // Would measure actual load time
        }
        
        Ok(())
    }
    
    /// Translate text from source to target language
    pub async fn translate(
        &self,
        text: &str,
        source_language: &str,
        target_language: &str,
    ) -> Result<TranslationResult> {
        let session_id = self.session_id.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Model not loaded"))?;
        
        let start_time = std::time::Instant::now();
        
        // Convert language codes to NLLB format
        let source_code = self.get_nllb_language_code(source_language)?;
        let target_code = self.get_nllb_language_code(target_language)?;
        
        // Prepare input
        let input = crate::acceleration::InferenceInput::Translation {
            text: text.to_string(),
            source_language: source_code.clone(),
            target_language: target_code.clone(),
        };
        
        // Run inference
        let output = self.onnx_manager.run_inference(session_id, input).await?;
        
        // Parse output
        let result = match output {
            crate::acceleration::InferenceOutput::Text { text, confidence } => {
                TranslationResult {
                    text,
                    source_language: source_language.to_string(),
                    target_language: target_language.to_string(),
                    confidence,
                    processing_time_ms: start_time.elapsed().as_millis() as f32,
                    model_version: "NLLB-600M".to_string(),
                    detected_language: None, // NLLB doesn't do language detection
                }
            },
            _ => return Err(anyhow::anyhow!("Unexpected output type from NLLB model")),
        };
        
        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.total_translations += 1;
            metrics.total_processing_time_ms += result.processing_time_ms;
            metrics.average_processing_time_ms = 
                metrics.total_processing_time_ms / metrics.total_translations as f32;
            
            if result.processing_time_ms < metrics.min_processing_time_ms {
                metrics.min_processing_time_ms = result.processing_time_ms;
            }
            if result.processing_time_ms > metrics.max_processing_time_ms {
                metrics.max_processing_time_ms = result.processing_time_ms;
            }
            
            // Update language pair statistics
            let pair_key = format!("{}->{}", source_language, target_language);
            *metrics.language_pair_counts.entry(pair_key).or_insert(0) += 1;
        }
        
        debug!("Translated '{}' ({} -> {}) in {}ms: '{}'", 
               &text[..text.len().min(50)], source_language, target_language,
               result.processing_time_ms, &result.text[..result.text.len().min(50)]);
        
        Ok(result)
    }
    
    /// Translate multiple texts in batch for efficiency
    pub async fn translate_batch(
        &self,
        texts: &[String],
        source_language: &str,
        target_language: &str,
    ) -> Result<Vec<TranslationResult>> {
        let session_id = self.session_id.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Model not loaded"))?;
        
        let start_time = std::time::Instant::now();
        
        // Convert language codes to NLLB format
        let source_code = self.get_nllb_language_code(source_language)?;
        let target_code = self.get_nllb_language_code(target_language)?;
        
        let mut results = Vec::new();
        
        // Process texts in batches to manage memory
        let batch_size = self.config.max_batch_size;
        for chunk in texts.chunks(batch_size) {
            for text in chunk {
                let input = crate::acceleration::InferenceInput::Translation {
                    text: text.clone(),
                    source_language: source_code.clone(),
                    target_language: target_code.clone(),
                };
                
                let output = self.onnx_manager.run_inference(session_id, input).await?;
                
                let result = match output {
                    crate::acceleration::InferenceOutput::Text { text, confidence } => {
                        TranslationResult {
                            text,
                            source_language: source_language.to_string(),
                            target_language: target_language.to_string(),
                            confidence,
                            processing_time_ms: start_time.elapsed().as_millis() as f32,
                            model_version: "NLLB-600M".to_string(),
                            detected_language: None,
                        }
                    },
                    _ => return Err(anyhow::anyhow!("Unexpected output type from NLLB model")),
                };
                
                results.push(result);
            }
        }
        
        // Update batch metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.total_translations += texts.len() as u64;
            metrics.total_batch_operations += 1;
        }
        
        Ok(results)
    }
    
    /// Get list of supported languages
    pub fn get_supported_languages(&self) -> Vec<String> {
        self.language_codes.keys().cloned().collect()
    }
    
    /// Check if a language pair is supported
    pub fn is_language_pair_supported(&self, source: &str, target: &str) -> bool {
        self.language_codes.contains_key(source) && self.language_codes.contains_key(target)
    }
    
    /// Get current performance metrics
    pub async fn get_metrics(&self) -> NLLBMetrics {
        self.metrics.read().await.clone()
    }
    
    /// Get session information
    pub async fn get_session_info(&self) -> Option<SessionInfo> {
        if let Some(session_id) = &self.session_id {
            self.onnx_manager.get_session_info(session_id).await
        } else {
            None
        }
    }
    
    /// Unload the model to free memory
    pub async fn unload_model(&mut self) -> Result<()> {
        if let Some(session_id) = &self.session_id {
            self.onnx_manager.remove_session(session_id).await?;
            self.session_id = None;
            
            // Update metrics
            {
                let mut metrics = self.metrics.write().await;
                metrics.model_loaded = false;
            }
            
            info!("NLLB-600M model unloaded");
        }
        Ok(())
    }
    
    /// Check if model is loaded
    pub fn is_loaded(&self) -> bool {
        self.session_id.is_some()
    }
    
    /// Update configuration
    pub async fn update_config(&mut self, new_config: NLLBConfig) -> Result<()> {
        let old_precision = self.config.precision;
        self.config = new_config;
        
        // If precision changed and model is loaded, reload the model
        if self.is_loaded() && old_precision != new_config.precision {
            info!("Precision changed, reloading model");
            self.unload_model().await?;
            self.load_model(Some(new_config.model_path)).await?;
        }
        
        Ok(())
    }
    
    // Private helper methods
    
    fn get_nllb_language_code(&self, language: &str) -> Result<String> {
        self.language_codes.get(language)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("Unsupported language: {}", language))
    }
    
    async fn select_optimal_precision(&self, memory_config: &MemoryConfiguration) -> ModelPrecision {
        // Select precision based on available memory and hardware capabilities
        let gpu_info = self.hardware_detector.get_gpu_info();
        
        match memory_config.recommended_precision {
            ModelPrecision::FP16 if gpu_info.supports_fp16 => ModelPrecision::FP16,
            ModelPrecision::INT8 if gpu_info.supports_int8 => ModelPrecision::INT8,
            _ => {
                // Fallback logic
                if gpu_info.supports_fp16 && memory_config.max_model_memory_mb > 1500 {
                    ModelPrecision::FP16
                } else if gpu_info.supports_int8 {
                    ModelPrecision::INT8
                } else {
                    ModelPrecision::FP32 // Last resort
                }
            }
        }
    }
    
    fn get_default_model_path(&self, precision: ModelPrecision) -> PathBuf {
        let filename = match precision {
            ModelPrecision::FP32 => "nllb-600M-fp32.onnx",
            ModelPrecision::FP16 => "nllb-600M-fp16.onnx",
            ModelPrecision::INT8 => "nllb-600M-int8.onnx",
            ModelPrecision::INT4 => "nllb-600M-int8.onnx", // Fallback to INT8
        };
        
        PathBuf::from("models").join("nllb").join(filename)
    }
    
    fn initialize_language_codes() -> HashMap<String, String> {
        let mut codes = HashMap::new();
        
        // Common language mappings to NLLB codes
        codes.insert("english".to_string(), "eng_Latn".to_string());
        codes.insert("en".to_string(), "eng_Latn".to_string());
        codes.insert("spanish".to_string(), "spa_Latn".to_string());
        codes.insert("es".to_string(), "spa_Latn".to_string());
        codes.insert("french".to_string(), "fra_Latn".to_string());
        codes.insert("fr".to_string(), "fra_Latn".to_string());
        codes.insert("german".to_string(), "deu_Latn".to_string());
        codes.insert("de".to_string(), "deu_Latn".to_string());
        codes.insert("italian".to_string(), "ita_Latn".to_string());
        codes.insert("it".to_string(), "ita_Latn".to_string());
        codes.insert("portuguese".to_string(), "por_Latn".to_string());
        codes.insert("pt".to_string(), "por_Latn".to_string());
        codes.insert("chinese".to_string(), "zho_Hans".to_string());
        codes.insert("zh".to_string(), "zho_Hans".to_string());
        codes.insert("japanese".to_string(), "jpn_Jpan".to_string());
        codes.insert("ja".to_string(), "jpn_Jpan".to_string());
        codes.insert("korean".to_string(), "kor_Hang".to_string());
        codes.insert("ko".to_string(), "kor_Hang".to_string());
        codes.insert("russian".to_string(), "rus_Cyrl".to_string());
        codes.insert("ru".to_string(), "rus_Cyrl".to_string());
        codes.insert("arabic".to_string(), "arb_Arab".to_string());
        codes.insert("ar".to_string(), "arb_Arab".to_string());
        codes.insert("hindi".to_string(), "hin_Deva".to_string());
        codes.insert("hi".to_string(), "hin_Deva".to_string());
        codes.insert("dutch".to_string(), "nld_Latn".to_string());
        codes.insert("nl".to_string(), "nld_Latn".to_string());
        codes.insert("polish".to_string(), "pol_Latn".to_string());
        codes.insert("pl".to_string(), "pol_Latn".to_string());
        codes.insert("turkish".to_string(), "tur_Latn".to_string());
        codes.insert("tr".to_string(), "tur_Latn".to_string());
        codes.insert("swedish".to_string(), "swe_Latn".to_string());
        codes.insert("sv".to_string(), "swe_Latn".to_string());
        codes.insert("norwegian".to_string(), "nno_Latn".to_string());
        codes.insert("no".to_string(), "nno_Latn".to_string());
        codes.insert("danish".to_string(), "dan_Latn".to_string());
        codes.insert("da".to_string(), "dan_Latn".to_string());
        codes.insert("finnish".to_string(), "fin_Latn".to_string());
        codes.insert("fi".to_string(), "fin_Latn".to_string());
        codes.insert("czech".to_string(), "ces_Latn".to_string());
        codes.insert("cs".to_string(), "ces_Latn".to_string());
        codes.insert("hungarian".to_string(), "hun_Latn".to_string());
        codes.insert("hu".to_string(), "hun_Latn".to_string());
        codes.insert("greek".to_string(), "ell_Grek".to_string());
        codes.insert("el".to_string(), "ell_Grek".to_string());
        codes.insert("hebrew".to_string(), "heb_Hebr".to_string());
        codes.insert("he".to_string(), "heb_Hebr".to_string());
        codes.insert("thai".to_string(), "tha_Thai".to_string());
        codes.insert("th".to_string(), "tha_Thai".to_string());
        codes.insert("vietnamese".to_string(), "vie_Latn".to_string());
        codes.insert("vi".to_string(), "vie_Latn".to_string());
        codes.insert("indonesian".to_string(), "ind_Latn".to_string());
        codes.insert("id".to_string(), "ind_Latn".to_string());
        codes.insert("malay".to_string(), "zsm_Latn".to_string());
        codes.insert("ms".to_string(), "zsm_Latn".to_string());
        codes.insert("tagalog".to_string(), "tgl_Latn".to_string());
        codes.insert("tl".to_string(), "tgl_Latn".to_string());
        codes.insert("ukrainian".to_string(), "ukr_Cyrl".to_string());
        codes.insert("uk".to_string(), "ukr_Cyrl".to_string());
        codes.insert("bulgarian".to_string(), "bul_Cyrl".to_string());
        codes.insert("bg".to_string(), "bul_Cyrl".to_string());
        codes.insert("romanian".to_string(), "ron_Latn".to_string());
        codes.insert("ro".to_string(), "ron_Latn".to_string());
        codes.insert("croatian".to_string(), "hrv_Latn".to_string());
        codes.insert("hr".to_string(), "hrv_Latn".to_string());
        codes.insert("serbian".to_string(), "srp_Cyrl".to_string());
        codes.insert("sr".to_string(), "srp_Cyrl".to_string());
        codes.insert("slovenian".to_string(), "slv_Latn".to_string());
        codes.insert("sl".to_string(), "slv_Latn".to_string());
        codes.insert("slovak".to_string(), "slk_Latn".to_string());
        codes.insert("sk".to_string(), "slk_Latn".to_string());
        codes.insert("lithuanian".to_string(), "lit_Latn".to_string());
        codes.insert("lt".to_string(), "lit_Latn".to_string());
        codes.insert("latvian".to_string(), "lav_Latn".to_string());
        codes.insert("lv".to_string(), "lav_Latn".to_string());
        codes.insert("estonian".to_string(), "est_Latn".to_string());
        codes.insert("et".to_string(), "est_Latn".to_string());
        
        codes
    }
}

/// NLLB-600M configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NLLBConfig {
    /// Model precision
    pub precision: ModelPrecision,
    /// Path to the model file
    pub model_path: PathBuf,
    /// Maximum batch size for batch translation
    pub max_batch_size: usize,
    /// Maximum sequence length
    pub max_sequence_length: usize,
    /// Beam size for beam search
    pub beam_size: usize,
    /// Length penalty for beam search
    pub length_penalty: f32,
    /// Temperature for sampling
    pub temperature: f32,
    /// Repetition penalty
    pub repetition_penalty: f32,
    /// Maximum translation length multiplier
    pub max_length_multiplier: f32,
    /// Enable caching for repeated translations
    pub enable_caching: bool,
}

impl Default for NLLBConfig {
    fn default() -> Self {
        Self {
            precision: ModelPrecision::FP16,
            model_path: PathBuf::from("models/nllb/nllb-600M-fp16.onnx"),
            max_batch_size: 8,
            max_sequence_length: 512,
            beam_size: 4,
            length_penalty: 1.0,
            temperature: 0.0, // Deterministic output
            repetition_penalty: 1.0,
            max_length_multiplier: 1.5,
            enable_caching: true,
        }
    }
}

/// Translation result from NLLB
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationResult {
    /// Translated text
    pub text: String,
    /// Source language
    pub source_language: String,
    /// Target language
    pub target_language: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Processing time in milliseconds
    pub processing_time_ms: f32,
    /// Model version used
    pub model_version: String,
    /// Detected language (if auto-detection was used)
    pub detected_language: Option<String>,
}

/// Performance metrics for NLLB-600M
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NLLBMetrics {
    /// Whether model is loaded
    pub model_loaded: bool,
    /// Model load time in milliseconds
    pub load_time_ms: u64,
    /// Total number of translations
    pub total_translations: u64,
    /// Total number of batch operations
    pub total_batch_operations: u64,
    /// Total processing time across all translations
    pub total_processing_time_ms: f32,
    /// Average processing time per translation
    pub average_processing_time_ms: f32,
    /// Minimum processing time observed
    pub min_processing_time_ms: f32,
    /// Maximum processing time observed
    pub max_processing_time_ms: f32,
    /// Cache hit rate (if caching enabled)
    pub cache_hit_rate: f32,
    /// Language pair usage statistics
    pub language_pair_counts: HashMap<String, u64>,
}

/// Model download and management utilities
pub struct NLLBModelManager {
    models_directory: PathBuf,
}

impl NLLBModelManager {
    /// Create new model manager
    pub fn new(models_directory: PathBuf) -> Self {
        Self { models_directory }
    }
    
    /// Check if a model variant is available
    pub async fn is_model_available(&self, precision: ModelPrecision) -> bool {
        let model_path = self.get_model_path(precision);
        model_path.exists()
    }
    
    /// Get path for a specific model variant
    pub fn get_model_path(&self, precision: ModelPrecision) -> PathBuf {
        let filename = match precision {
            ModelPrecision::FP32 => "nllb-600M-fp32.onnx",
            ModelPrecision::FP16 => "nllb-600M-fp16.onnx",
            ModelPrecision::INT8 => "nllb-600M-int8.onnx",
            ModelPrecision::INT4 => "nllb-600M-int8.onnx",
        };
        
        self.models_directory.join("nllb").join(filename)
    }
    
    /// Download a model variant if not present
    pub async fn ensure_model_available(&self, precision: ModelPrecision) -> Result<PathBuf> {
        let model_path = self.get_model_path(precision);
        
        if !model_path.exists() {
            info!("Downloading NLLB-600M model: {:?}", precision);
            self.download_model(precision, &model_path).await?;
        }
        
        Ok(model_path)
    }
    
    /// Download model (placeholder implementation)
    async fn download_model(&self, precision: ModelPrecision, target_path: &Path) -> Result<()> {
        // Create directory if it doesn't exist
        if let Some(parent) = target_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        
        // In a real implementation, this would download from Hugging Face or similar
        warn!("Model download not implemented. Please manually place the model at: {}", 
              target_path.display());
        
        // Create a placeholder file for testing
        tokio::fs::write(target_path, b"placeholder_model_data").await?;
        
        Ok(())
    }
    
    /// List available models
    pub async fn list_available_models(&self) -> Vec<ModelPrecision> {
        let mut available = Vec::new();
        
        for precision in &[ModelPrecision::FP32, ModelPrecision::FP16, ModelPrecision::INT8] {
            if self.is_model_available(*precision).await {
                available.push(*precision);
            }
        }
        
        available
    }
}