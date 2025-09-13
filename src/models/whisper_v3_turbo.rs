//! Whisper V3 Turbo integration for high-performance speech recognition

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use crate::acceleration::onnx_runtime::{OnnxRuntimeManager, SessionInfo};
use crate::gpu::adaptive_memory::{ModelPrecision, ModelType};
use crate::gpu::hardware_detection::{HardwareDetector, MemoryConfiguration};

/// Whisper V3 Turbo model integration
pub struct WhisperV3Turbo {
    /// ONNX Runtime manager
    onnx_manager: Arc<OnnxRuntimeManager>,
    /// Hardware detector for optimal configuration
    hardware_detector: Arc<HardwareDetector>,
    /// Current model session ID
    session_id: Option<String>,
    /// Model configuration
    config: WhisperConfig,
    /// Performance metrics
    metrics: Arc<RwLock<WhisperMetrics>>,
}

impl WhisperV3Turbo {
    /// Create new Whisper V3 Turbo instance
    pub async fn new(
        onnx_manager: Arc<OnnxRuntimeManager>,
        hardware_detector: Arc<HardwareDetector>,
    ) -> Result<Self> {
        info!("Initializing Whisper V3 Turbo");
        
        let config = WhisperConfig::default();
        
        Ok(Self {
            onnx_manager,
            hardware_detector,
            session_id: None,
            config,
            metrics: Arc::new(RwLock::new(WhisperMetrics::default())),
        })
    }
    
    /// Load the model with optimal configuration
    pub async fn load_model(&mut self, model_path: Option<PathBuf>) -> Result<()> {
        info!("Loading Whisper V3 Turbo model");
        
        // Determine optimal configuration based on hardware
        let memory_config = self.hardware_detector.get_recommended_config().await;
        let model_precision = self.select_optimal_precision(&memory_config).await;
        
        // Get model path
        let model_path = model_path.unwrap_or_else(|| {
            self.get_default_model_path(model_precision)
        });
        
        if !model_path.exists() {
            return Err(anyhow::anyhow!(
                "Whisper V3 Turbo model not found at: {}. Please download the model first.",
                model_path.display()
            ));
        }
        
        // Create session ID
        let session_id = format!("whisper_v3_turbo_{}", uuid::Uuid::new_v4());
        
        // Create ONNX session
        self.onnx_manager.create_session(
            &session_id,
            &model_path,
            ModelType::WhisperV3Turbo,
            model_precision,
            None, // Auto-select best execution provider
        ).await?;
        
        self.session_id = Some(session_id.clone());
        
        // Update configuration
        self.config.precision = model_precision;
        self.config.model_path = model_path;
        
        info!("Whisper V3 Turbo loaded successfully with {:?} precision", model_precision);
        
        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.model_loaded = true;
            metrics.load_time_ms = 0; // Would measure actual load time
        }
        
        Ok(())
    }
    
    /// Transcribe audio samples
    pub async fn transcribe(
        &self,
        audio_samples: &[f32],
        sample_rate: u32,
        language: Option<&str>,
    ) -> Result<TranscriptionResult> {
        let session_id = self.session_id.as_ref()
            .ok_or_else(|| anyhow::anyhow!("Model not loaded"))?;
        
        let start_time = std::time::Instant::now();
        
        // Prepare input
        let input = crate::acceleration::InferenceInput::Audio {
            samples: audio_samples.to_vec(),
            sample_rate,
        };
        
        // Run inference
        let output = self.onnx_manager.run_inference(session_id, input).await?;
        
        // Parse output
        let result = match output {
            crate::acceleration::InferenceOutput::Text { text, confidence } => {
                TranscriptionResult {
                    text,
                    language: language.unwrap_or("auto").to_string(),
                    confidence,
                    processing_time_ms: start_time.elapsed().as_millis() as f32,
                    segments: vec![], // Would be populated with actual segments
                    words: vec![],    // Would be populated with word-level timestamps
                }
            },
            _ => return Err(anyhow::anyhow!("Unexpected output type from Whisper model")),
        };
        
        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.total_inferences += 1;
            metrics.total_processing_time_ms += result.processing_time_ms;
            metrics.average_processing_time_ms = 
                metrics.total_processing_time_ms / metrics.total_inferences as f32;
            
            if result.processing_time_ms < metrics.min_processing_time_ms {
                metrics.min_processing_time_ms = result.processing_time_ms;
            }
            if result.processing_time_ms > metrics.max_processing_time_ms {
                metrics.max_processing_time_ms = result.processing_time_ms;
            }
        }
        
        debug!("Transcribed {} samples in {}ms: '{}'", 
               audio_samples.len(), result.processing_time_ms, 
               &result.text[..result.text.len().min(50)]);
        
        Ok(result)
    }
    
    /// Transcribe with streaming (for real-time processing)
    pub async fn transcribe_streaming(
        &self,
        audio_chunk: &[f32],
        sample_rate: u32,
        context: Option<&str>,
    ) -> Result<StreamingTranscriptionResult> {
        // For streaming, we use smaller chunks and maintain context
        let result = self.transcribe(audio_chunk, sample_rate, None).await?;
        
        Ok(StreamingTranscriptionResult {
            text: result.text,
            confidence: result.confidence,
            is_final: audio_chunk.len() >= self.config.min_chunk_size,
            processing_time_ms: result.processing_time_ms,
            context_length: context.map(|c| c.len()).unwrap_or(0),
        })
    }
    
    /// Get current performance metrics
    pub async fn get_metrics(&self) -> WhisperMetrics {
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
            
            info!("Whisper V3 Turbo model unloaded");
        }
        Ok(())
    }
    
    /// Check if model is loaded
    pub fn is_loaded(&self) -> bool {
        self.session_id.is_some()
    }
    
    /// Update configuration
    pub async fn update_config(&mut self, new_config: WhisperConfig) -> Result<()> {
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
    
    async fn select_optimal_precision(&self, memory_config: &MemoryConfiguration) -> ModelPrecision {
        // Select precision based on available memory and hardware capabilities
        let gpu_info = self.hardware_detector.get_gpu_info();
        
        match memory_config.recommended_precision {
            ModelPrecision::FP16 if gpu_info.supports_fp16 => ModelPrecision::FP16,
            ModelPrecision::INT8 if gpu_info.supports_int8 => ModelPrecision::INT8,
            _ => {
                // Fallback logic
                if gpu_info.supports_fp16 && memory_config.max_model_memory_mb > 2000 {
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
            ModelPrecision::FP32 => "whisper-large-v3-turbo-fp32.onnx",
            ModelPrecision::FP16 => "whisper-large-v3-turbo-fp16.onnx",
            ModelPrecision::INT8 => "whisper-large-v3-turbo-int8.onnx",
            ModelPrecision::INT4 => "whisper-large-v3-turbo-int8.onnx", // Fallback to INT8
        };
        
        PathBuf::from("models").join("whisper").join(filename)
    }
}

/// Whisper V3 Turbo configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhisperConfig {
    /// Model precision
    pub precision: ModelPrecision,
    /// Path to the model file
    pub model_path: PathBuf,
    /// Minimum chunk size for streaming
    pub min_chunk_size: usize,
    /// Maximum sequence length
    pub max_sequence_length: usize,
    /// Enable timestamps
    pub enable_timestamps: bool,
    /// Enable word-level timestamps
    pub enable_word_timestamps: bool,
    /// Language hint (optional)
    pub language_hint: Option<String>,
    /// Temperature for sampling
    pub temperature: f32,
    /// Beam size for beam search
    pub beam_size: usize,
}

impl Default for WhisperConfig {
    fn default() -> Self {
        Self {
            precision: ModelPrecision::FP16,
            model_path: PathBuf::from("models/whisper/whisper-large-v3-turbo-fp16.onnx"),
            min_chunk_size: 16000, // 1 second at 16kHz
            max_sequence_length: 448, // Whisper default
            enable_timestamps: true,
            enable_word_timestamps: false, // Expensive, disable by default
            language_hint: None,
            temperature: 0.0, // Deterministic output
            beam_size: 1, // Greedy decoding for speed
        }
    }
}

/// Transcription result from Whisper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionResult {
    /// Transcribed text
    pub text: String,
    /// Detected/specified language
    pub language: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Processing time in milliseconds
    pub processing_time_ms: f32,
    /// Segment-level results (for longer audio)
    pub segments: Vec<TranscriptionSegment>,
    /// Word-level results with timestamps
    pub words: Vec<WordTimestamp>,
}

/// Streaming transcription result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingTranscriptionResult {
    /// Transcribed text (partial or complete)
    pub text: String,
    /// Confidence score
    pub confidence: f32,
    /// Whether this is a final result
    pub is_final: bool,
    /// Processing time for this chunk
    pub processing_time_ms: f32,
    /// Length of context used
    pub context_length: usize,
}

/// Segment within a transcription
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionSegment {
    /// Segment text
    pub text: String,
    /// Start time in seconds
    pub start_time: f32,
    /// End time in seconds
    pub end_time: f32,
    /// Confidence score
    pub confidence: f32,
}

/// Word-level timestamp information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordTimestamp {
    /// Word text
    pub word: String,
    /// Start time in seconds
    pub start_time: f32,
    /// End time in seconds
    pub end_time: f32,
    /// Confidence score
    pub confidence: f32,
}

/// Performance metrics for Whisper V3 Turbo
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WhisperMetrics {
    /// Whether model is loaded
    pub model_loaded: bool,
    /// Model load time in milliseconds
    pub load_time_ms: u64,
    /// Total number of inferences
    pub total_inferences: u64,
    /// Total processing time across all inferences
    pub total_processing_time_ms: f32,
    /// Average processing time per inference
    pub average_processing_time_ms: f32,
    /// Minimum processing time observed
    pub min_processing_time_ms: f32,
    /// Maximum processing time observed
    pub max_processing_time_ms: f32,
    /// Real-time factor (processing_time / audio_duration)
    pub real_time_factor: f32,
}

/// Model download and management utilities
pub struct WhisperModelManager {
    models_directory: PathBuf,
}

impl WhisperModelManager {
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
            ModelPrecision::FP32 => "whisper-large-v3-turbo-fp32.onnx",
            ModelPrecision::FP16 => "whisper-large-v3-turbo-fp16.onnx",
            ModelPrecision::INT8 => "whisper-large-v3-turbo-int8.onnx",
            ModelPrecision::INT4 => "whisper-large-v3-turbo-int8.onnx",
        };
        
        self.models_directory.join("whisper").join(filename)
    }
    
    /// Download a model variant if not present
    pub async fn ensure_model_available(&self, precision: ModelPrecision) -> Result<PathBuf> {
        let model_path = self.get_model_path(precision);
        
        if !model_path.exists() {
            info!("Downloading Whisper V3 Turbo model: {:?}", precision);
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