//! Dynamic model switching for adaptive performance optimization

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use crate::acceleration::onnx_runtime::{OnnxRuntimeManager, SessionInfo};
use crate::gpu::adaptive_memory::{
    AdaptiveMemoryManager, ModelPrecision, ModelType, MemoryUtilization,
};
use crate::gpu::hardware_detection::HardwareDetector;
use crate::models::{
    whisper_v3_turbo::{WhisperV3Turbo, WhisperConfig, TranscriptionResult},
    nllb_600m::{NLLB600M, NLLBConfig, TranslationResult},
};

/// Dynamic model switching manager for optimal performance
pub struct ModelSwitcher {
    /// Memory manager for monitoring VRAM usage
    memory_manager: Arc<AdaptiveMemoryManager>,
    /// Hardware detector for system monitoring
    hardware_detector: Arc<HardwareDetector>,
    /// ONNX Runtime manager
    onnx_manager: Arc<OnnxRuntimeManager>,
    /// Current active models
    active_models: Arc<RwLock<ActiveModels>>,
    /// Switching configuration
    config: SwitchingConfig,
    /// Performance metrics
    metrics: Arc<RwLock<SwitchingMetrics>>,
    /// Model switching rules
    switching_rules: Vec<SwitchingRule>,
}

impl ModelSwitcher {
    /// Create new model switcher
    pub async fn new(
        memory_manager: Arc<AdaptiveMemoryManager>,
        hardware_detector: Arc<HardwareDetector>,
        onnx_manager: Arc<OnnxRuntimeManager>,
    ) -> Result<Self> {
        info!("Initializing dynamic model switcher");
        
        let config = SwitchingConfig::default();
        let switching_rules = Self::initialize_switching_rules();
        
        Ok(Self {
            memory_manager,
            hardware_detector,
            onnx_manager,
            active_models: Arc::new(RwLock::new(ActiveModels::default())),
            config,
            metrics: Arc::new(RwLock::new(SwitchingMetrics::default())),
            switching_rules,
        })
    }
    
    /// Start the automatic model switching monitor
    pub async fn start_monitor(&self) -> Result<()> {
        info!("Starting model switching monitor");
        
        let memory_manager = Arc::clone(&self.memory_manager);
        let hardware_detector = Arc::clone(&self.hardware_detector);
        let active_models = Arc::clone(&self.active_models);
        let metrics = Arc::clone(&self.metrics);
        let switching_rules = self.switching_rules.clone();
        let config = self.config.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(config.monitor_interval_seconds));
            
            loop {
                interval.tick().await;
                
                if let Err(e) = Self::evaluate_switching_conditions(
                    &memory_manager,
                    &hardware_detector,
                    &active_models,
                    &metrics,
                    &switching_rules,
                    &config,
                ).await {
                    error!("Error evaluating switching conditions: {}", e);
                }
            }
        });
        
        Ok(())
    }
    
    /// Request speech recognition with automatic model selection
    pub async fn transcribe(
        &self,
        audio_samples: &[f32],
        sample_rate: u32,
        language: Option<&str>,
    ) -> Result<TranscriptionResult> {
        let start_time = Instant::now();
        
        // Get or create appropriate speech recognition model
        let whisper = self.get_or_create_speech_model().await?;
        
        // Perform transcription
        let result = whisper.transcribe(audio_samples, sample_rate, language).await?;
        
        // Update usage metrics
        self.update_model_usage(ModelType::WhisperV3Turbo, start_time.elapsed()).await;
        
        Ok(result)
    }
    
    /// Request translation with automatic model selection
    pub async fn translate(
        &self,
        text: &str,
        source_language: &str,
        target_language: &str,
    ) -> Result<TranslationResult> {
        let start_time = Instant::now();
        
        // Get or create appropriate translation model
        let nllb = self.get_or_create_translation_model().await?;
        
        // Perform translation
        let result = nllb.translate(text, source_language, target_language).await?;
        
        // Update usage metrics
        self.update_model_usage(ModelType::NLLB600M, start_time.elapsed()).await;
        
        Ok(result)
    }
    
    /// Force a model switch for testing or optimization
    pub async fn force_model_switch(
        &self,
        model_type: ModelType,
        target_precision: ModelPrecision,
    ) -> Result<()> {
        info!("Forcing model switch: {:?} to {:?}", model_type, target_precision);
        
        match model_type {
            ModelType::WhisperV3Turbo | ModelType::DistilWhisper => {
                self.switch_speech_model(target_precision).await?;
            },
            ModelType::NLLB600M | ModelType::OpusMT => {
                self.switch_translation_model(target_precision).await?;
            },
            _ => {
                warn!("Model type {:?} not supported for switching", model_type);
            }
        }
        
        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.forced_switches += 1;
        }
        
        Ok(())
    }
    
    /// Get current model status
    pub async fn get_model_status(&self) -> ModelStatus {
        let active_models = self.active_models.read().await;
        let memory_util = self.memory_manager.get_memory_utilization().await;
        let metrics = self.metrics.read().await.clone();
        
        ModelStatus {
            speech_model: active_models.speech_model.as_ref().map(|m| m.clone()),
            translation_model: active_models.translation_model.as_ref().map(|m| m.clone()),
            memory_utilization: memory_util,
            switching_metrics: metrics,
            last_switch_time: active_models.last_switch_time,
        }
    }
    
    /// Unload unused models to free memory
    pub async fn cleanup_unused_models(&self) -> Result<()> {
        let cleanup_threshold = Duration::from_secs(self.config.model_cleanup_threshold_seconds);
        let now = Instant::now();
        
        let mut models_to_unload = Vec::new();
        
        {
            let active_models = self.active_models.read().await;
            
            // Check speech model
            if let Some(ref model_info) = active_models.speech_model {
                if now.duration_since(model_info.last_used) > cleanup_threshold {
                    models_to_unload.push((ModelType::WhisperV3Turbo, model_info.session_id.clone()));
                }
            }
            
            // Check translation model
            if let Some(ref model_info) = active_models.translation_model {
                if now.duration_since(model_info.last_used) > cleanup_threshold {
                    models_to_unload.push((ModelType::NLLB600M, model_info.session_id.clone()));
                }
            }
        }
        
        // Unload identified models
        for (model_type, session_id) in models_to_unload {
            info!("Cleaning up unused model: {:?} ({})", model_type, session_id);
            
            // Remove from ONNX Runtime
            self.onnx_manager.remove_session(&session_id).await?;
            
            // Update active models
            {
                let mut active_models = self.active_models.write().await;
                match model_type {
                    ModelType::WhisperV3Turbo | ModelType::DistilWhisper => {
                        active_models.speech_model = None;
                    },
                    ModelType::NLLB600M | ModelType::OpusMT => {
                        active_models.translation_model = None;
                    },
                    _ => {}
                }
            }
            
            // Update metrics
            {
                let mut metrics = self.metrics.write().await;
                metrics.models_cleaned_up += 1;
            }
        }
        
        Ok(())
    }
    
    // Private implementation methods
    
    async fn get_or_create_speech_model(&self) -> Result<WhisperV3Turbo> {
        let mut active_models = self.active_models.write().await;
        
        if let Some(ref model_info) = active_models.speech_model {
            // Update last used time
            let mut updated_info = model_info.clone();
            updated_info.last_used = Instant::now();
            active_models.speech_model = Some(updated_info);
            
            // Create Whisper instance with existing session
            let mut whisper = WhisperV3Turbo::new(
                Arc::clone(&self.onnx_manager),
                Arc::clone(&self.hardware_detector),
            ).await?;
            
            // Note: In a real implementation, we'd restore the session state
            // For now, we'll assume the model is already loaded
            
            return Ok(whisper);
        }
        
        // Create new model
        let memory_config = self.memory_manager.get_recommended_config().await;
        let precision = self.select_optimal_precision(&memory_config);
        
        let mut whisper = WhisperV3Turbo::new(
            Arc::clone(&self.onnx_manager),
            Arc::clone(&self.hardware_detector),
        ).await?;
        
        whisper.load_model(None).await?;
        
        // Store model info
        let session_info = whisper.get_session_info().await
            .ok_or_else(|| anyhow::anyhow!("Failed to get session info"))?;
        
        active_models.speech_model = Some(ModelInfo {
            model_type: ModelType::WhisperV3Turbo,
            precision,
            session_id: session_info.session_id,
            loaded_at: Instant::now(),
            last_used: Instant::now(),
            usage_count: 0,
        });
        
        Ok(whisper)
    }
    
    async fn get_or_create_translation_model(&self) -> Result<NLLB600M> {
        let mut active_models = self.active_models.write().await;
        
        if let Some(ref model_info) = active_models.translation_model {
            // Update last used time
            let mut updated_info = model_info.clone();
            updated_info.last_used = Instant::now();
            active_models.translation_model = Some(updated_info);
            
            // Create NLLB instance with existing session
            let mut nllb = NLLB600M::new(
                Arc::clone(&self.onnx_manager),
                Arc::clone(&self.hardware_detector),
            ).await?;
            
            return Ok(nllb);
        }
        
        // Create new model
        let memory_config = self.memory_manager.get_recommended_config().await;
        let precision = self.select_optimal_precision(&memory_config);
        
        let mut nllb = NLLB600M::new(
            Arc::clone(&self.onnx_manager),
            Arc::clone(&self.hardware_detector),
        ).await?;
        
        nllb.load_model(None).await?;
        
        // Store model info
        let session_info = nllb.get_session_info().await
            .ok_or_else(|| anyhow::anyhow!("Failed to get session info"))?;
        
        active_models.translation_model = Some(ModelInfo {
            model_type: ModelType::NLLB600M,
            precision,
            session_id: session_info.session_id,
            loaded_at: Instant::now(),
            last_used: Instant::now(),
            usage_count: 0,
        });
        
        Ok(nllb)
    }
    
    async fn switch_speech_model(&self, target_precision: ModelPrecision) -> Result<()> {
        info!("Switching speech model to {:?} precision", target_precision);
        
        // Unload current model if exists
        {
            let mut active_models = self.active_models.write().await;
            if let Some(ref model_info) = active_models.speech_model {
                self.onnx_manager.remove_session(&model_info.session_id).await?;
                active_models.speech_model = None;
            }
        }
        
        // Create new model with target precision
        let mut whisper = WhisperV3Turbo::new(
            Arc::clone(&self.onnx_manager),
            Arc::clone(&self.hardware_detector),
        ).await?;
        
        // Update configuration for target precision
        let mut config = WhisperConfig::default();
        config.precision = target_precision;
        whisper.update_config(config).await?;
        
        // Load with new precision
        whisper.load_model(None).await?;
        
        // Update active models
        {
            let mut active_models = self.active_models.write().await;
            let session_info = whisper.get_session_info().await
                .ok_or_else(|| anyhow::anyhow!("Failed to get session info"))?;
            
            active_models.speech_model = Some(ModelInfo {
                model_type: ModelType::WhisperV3Turbo,
                precision: target_precision,
                session_id: session_info.session_id,
                loaded_at: Instant::now(),
                last_used: Instant::now(),
                usage_count: 0,
            });
            
            active_models.last_switch_time = Some(Instant::now());
        }
        
        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.automatic_switches += 1;
        }
        
        Ok(())
    }
    
    async fn switch_translation_model(&self, target_precision: ModelPrecision) -> Result<()> {
        info!("Switching translation model to {:?} precision", target_precision);
        
        // Unload current model if exists
        {
            let mut active_models = self.active_models.write().await;
            if let Some(ref model_info) = active_models.translation_model {
                self.onnx_manager.remove_session(&model_info.session_id).await?;
                active_models.translation_model = None;
            }
        }
        
        // Create new model with target precision
        let mut nllb = NLLB600M::new(
            Arc::clone(&self.onnx_manager),
            Arc::clone(&self.hardware_detector),
        ).await?;
        
        // Update configuration for target precision
        let mut config = NLLBConfig::default();
        config.precision = target_precision;
        nllb.update_config(config).await?;
        
        // Load with new precision
        nllb.load_model(None).await?;
        
        // Update active models
        {
            let mut active_models = self.active_models.write().await;
            let session_info = nllb.get_session_info().await
                .ok_or_else(|| anyhow::anyhow!("Failed to get session info"))?;
            
            active_models.translation_model = Some(ModelInfo {
                model_type: ModelType::NLLB600M,
                precision: target_precision,
                session_id: session_info.session_id,
                loaded_at: Instant::now(),
                last_used: Instant::now(),
                usage_count: 0,
            });
            
            active_models.last_switch_time = Some(Instant::now());
        }
        
        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.automatic_switches += 1;
        }
        
        Ok(())
    }
    
    async fn update_model_usage(&self, model_type: ModelType, processing_time: Duration) {
        let mut active_models = self.active_models.write().await;
        
        match model_type {
            ModelType::WhisperV3Turbo | ModelType::DistilWhisper => {
                if let Some(ref mut model_info) = active_models.speech_model {
                    model_info.usage_count += 1;
                    model_info.last_used = Instant::now();
                }
            },
            ModelType::NLLB600M | ModelType::OpusMT => {
                if let Some(ref mut model_info) = active_models.translation_model {
                    model_info.usage_count += 1;
                    model_info.last_used = Instant::now();
                }
            },
            _ => {}
        }
        
        // Update global metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.total_inferences += 1;
            metrics.total_processing_time_ms += processing_time.as_millis() as u64;
        }
    }
    
    fn select_optimal_precision(&self, memory_config: &crate::gpu::hardware_detection::MemoryConfiguration) -> ModelPrecision {
        match memory_config.recommended_precision {
            ModelPrecision::FP16 => ModelPrecision::FP16,
            ModelPrecision::INT8 => ModelPrecision::INT8,
            _ => ModelPrecision::FP32,
        }
    }
    
    async fn evaluate_switching_conditions(
        memory_manager: &Arc<AdaptiveMemoryManager>,
        _hardware_detector: &Arc<HardwareDetector>,
        active_models: &Arc<RwLock<ActiveModels>>,
        metrics: &Arc<RwLock<SwitchingMetrics>>,
        switching_rules: &[SwitchingRule],
        _config: &SwitchingConfig,
    ) -> Result<()> {
        let memory_util = memory_manager.get_memory_utilization().await;
        
        // Evaluate each switching rule
        for rule in switching_rules {
            if rule.should_trigger(&memory_util).await {
                info!("Switching rule triggered: {}", rule.name);
                
                // Apply the rule (placeholder implementation)
                // In a real implementation, this would perform the actual model switch
                
                // Update metrics
                {
                    let mut metrics = metrics.write().await;
                    metrics.rule_triggers += 1;
                }
            }
        }
        
        Ok(())
    }
    
    fn initialize_switching_rules() -> Vec<SwitchingRule> {
        vec![
            SwitchingRule {
                name: "High Memory Pressure".to_string(),
                trigger_condition: SwitchingTrigger::MemoryPressure { threshold: 85.0 },
                action: SwitchingAction::ReducePrecision,
                priority: 1,
            },
            SwitchingRule {
                name: "Low Memory Usage".to_string(),
                trigger_condition: SwitchingTrigger::MemoryPressure { threshold: 50.0 },
                action: SwitchingAction::IncreasePrecision,
                priority: 2,
            },
            SwitchingRule {
                name: "Performance Degradation".to_string(),
                trigger_condition: SwitchingTrigger::PerformanceThreshold { min_fps: 10.0 },
                action: SwitchingAction::SwitchToFasterModel,
                priority: 1,
            },
        ]
    }
}

// Supporting types and structures

#[derive(Debug, Clone)]
pub struct ActiveModels {
    pub speech_model: Option<ModelInfo>,
    pub translation_model: Option<ModelInfo>,
    pub last_switch_time: Option<Instant>,
}

impl Default for ActiveModels {
    fn default() -> Self {
        Self {
            speech_model: None,
            translation_model: None,
            last_switch_time: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub model_type: ModelType,
    pub precision: ModelPrecision,
    pub session_id: String,
    pub loaded_at: Instant,
    pub last_used: Instant,
    pub usage_count: u64,
}

#[derive(Debug, Clone)]
pub struct SwitchingConfig {
    /// Monitor interval in seconds
    pub monitor_interval_seconds: u64,
    /// Threshold for cleaning up unused models (seconds)
    pub model_cleanup_threshold_seconds: u64,
    /// Maximum memory utilization before switching
    pub max_memory_utilization: f32,
    /// Minimum performance threshold
    pub min_performance_fps: f32,
    /// Enable automatic switching
    pub enable_auto_switching: bool,
}

impl Default for SwitchingConfig {
    fn default() -> Self {
        Self {
            monitor_interval_seconds: 5,
            model_cleanup_threshold_seconds: 300, // 5 minutes
            max_memory_utilization: 85.0,
            min_performance_fps: 15.0,
            enable_auto_switching: true,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SwitchingMetrics {
    /// Total number of automatic switches
    pub automatic_switches: u64,
    /// Total number of forced switches
    pub forced_switches: u64,
    /// Total number of rule triggers
    pub rule_triggers: u64,
    /// Total number of models cleaned up
    pub models_cleaned_up: u64,
    /// Total inferences processed
    pub total_inferences: u64,
    /// Total processing time
    pub total_processing_time_ms: u64,
    /// Average switching time
    pub average_switch_time_ms: f64,
}

#[derive(Debug, Clone)]
pub struct SwitchingRule {
    pub name: String,
    pub trigger_condition: SwitchingTrigger,
    pub action: SwitchingAction,
    pub priority: u8,
}

impl SwitchingRule {
    async fn should_trigger(&self, memory_util: &MemoryUtilization) -> bool {
        match &self.trigger_condition {
            SwitchingTrigger::MemoryPressure { threshold } => {
                memory_util.utilization_percentage > *threshold
            },
            SwitchingTrigger::PerformanceThreshold { min_fps: _ } => {
                // Placeholder - would check actual performance metrics
                false
            },
        }
    }
}

#[derive(Debug, Clone)]
pub enum SwitchingTrigger {
    MemoryPressure { threshold: f32 },
    PerformanceThreshold { min_fps: f32 },
}

#[derive(Debug, Clone)]
pub enum SwitchingAction {
    ReducePrecision,
    IncreasePrecision,
    SwitchToFasterModel,
    UnloadModel,
}

#[derive(Debug, Clone)]
pub struct ModelStatus {
    pub speech_model: Option<ModelInfo>,
    pub translation_model: Option<ModelInfo>,
    pub memory_utilization: MemoryUtilization,
    pub switching_metrics: SwitchingMetrics,
    pub last_switch_time: Option<Instant>,
}