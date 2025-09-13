//! Model quantization pipeline for optimal performance and memory efficiency

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use crate::gpu::adaptive_memory::{ModelPrecision, ModelType};
use crate::gpu::hardware_detection::HardwareDetector;

/// Quantization pipeline for converting models to different precision levels
pub struct QuantizationPipeline {
    /// Hardware detector for capability assessment
    hardware_detector: Arc<HardwareDetector>,
    /// Quantization configuration
    config: QuantizationConfig,
    /// Supported quantization methods
    supported_methods: Vec<QuantizationMethod>,
    /// Quantization metrics
    metrics: Arc<RwLock<QuantizationMetrics>>,
    /// Cached quantized models
    quantized_cache: Arc<RwLock<HashMap<String, QuantizedModelInfo>>>,
}

impl QuantizationPipeline {
    /// Create new quantization pipeline
    pub async fn new(
        hardware_detector: Arc<HardwareDetector>,
        config: QuantizationConfig,
    ) -> Result<Self> {
        info!("Initializing quantization pipeline");
        
        let supported_methods = Self::detect_supported_methods(&hardware_detector).await;
        
        info!("Supported quantization methods: {:?}", 
              supported_methods.iter().map(|m| &m.name).collect::<Vec<_>>());
        
        Ok(Self {
            hardware_detector,
            config,
            supported_methods,
            metrics: Arc::new(RwLock::new(QuantizationMetrics::default())),
            quantized_cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }
    
    /// Quantize a model to target precision
    pub async fn quantize_model(
        &self,
        source_model_path: &Path,
        target_precision: ModelPrecision,
        model_type: ModelType,
        output_path: &Path,
    ) -> Result<QuantizationResult> {
        let start_time = std::time::Instant::now();
        
        info!("Quantizing {:?} model from {} to {:?} precision", 
              model_type, source_model_path.display(), target_precision);
        
        // Check if already quantized and cached
        let cache_key = self.generate_cache_key(source_model_path, target_precision, model_type);
        if let Some(cached_info) = self.quantized_cache.read().await.get(&cache_key) {
            if cached_info.output_path.exists() {
                info!("Using cached quantized model: {}", cached_info.output_path.display());
                return Ok(QuantizationResult {
                    source_path: source_model_path.to_path_buf(),
                    output_path: cached_info.output_path.clone(),
                    source_precision: cached_info.source_precision,
                    target_precision,
                    model_type,
                    compression_ratio: cached_info.compression_ratio,
                    quantization_time_ms: 0, // Cached, no quantization time
                    quality_metrics: cached_info.quality_metrics.clone(),
                    method_used: cached_info.method_used.clone(),
                });
            }
        }
        
        // Select optimal quantization method
        let method = self.select_quantization_method(target_precision, model_type).await?;
        
        // Prepare quantization parameters
        let params = self.prepare_quantization_parameters(&method, target_precision, model_type).await;
        
        // Perform quantization
        let result = self.execute_quantization(
            source_model_path,
            output_path,
            &method,
            &params,
            model_type,
        ).await?;
        
        // Validate quantized model
        let quality_metrics = self.validate_quantized_model(
            source_model_path,
            &result.output_path,
            model_type,
        ).await?;
        
        let quantization_time = start_time.elapsed();
        
        // Calculate compression ratio
        let source_size = self.get_file_size(source_model_path).await?;
        let target_size = self.get_file_size(&result.output_path).await?;
        let compression_ratio = source_size as f32 / target_size as f32;
        
        let final_result = QuantizationResult {
            source_path: source_model_path.to_path_buf(),
            output_path: result.output_path.clone(),
            source_precision: result.source_precision,
            target_precision,
            model_type,
            compression_ratio,
            quantization_time_ms: quantization_time.as_millis() as u64,
            quality_metrics: quality_metrics.clone(),
            method_used: method.name.clone(),
        };
        
        // Cache the result
        {
            let mut cache = self.quantized_cache.write().await;
            cache.insert(cache_key, QuantizedModelInfo {
                output_path: result.output_path,
                source_precision: result.source_precision,
                target_precision,
                compression_ratio,
                quality_metrics,
                method_used: method.name,
                quantized_at: std::time::SystemTime::now(),
            });
        }
        
        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.total_quantizations += 1;
            metrics.total_quantization_time_ms += quantization_time.as_millis() as u64;
            metrics.total_size_reduction_bytes += (source_size - target_size) as u64;
            metrics.average_compression_ratio = 
                (metrics.average_compression_ratio * (metrics.total_quantizations - 1) as f32 + compression_ratio) 
                / metrics.total_quantizations as f32;
        }
        
        info!("Quantization completed in {}ms, compression ratio: {:.2}x", 
              quantization_time.as_millis(), compression_ratio);
        
        Ok(final_result)
    }
    
    /// Batch quantize multiple models
    pub async fn batch_quantize(
        &self,
        models: Vec<BatchQuantizationRequest>,
    ) -> Result<Vec<QuantizationResult>> {
        info!("Starting batch quantization of {} models", models.len());
        
        let mut results = Vec::new();
        
        for request in models {
            match self.quantize_model(
                &request.source_path,
                request.target_precision,
                request.model_type,
                &request.output_path,
            ).await {
                Ok(result) => {
                    results.push(result);
                    info!("✓ Quantized: {} -> {:?}", 
                          request.source_path.display(), request.target_precision);
                },
                Err(e) => {
                    error!("✗ Failed to quantize {}: {}", request.source_path.display(), e);
                    // Continue with other models
                }
            }
        }
        
        info!("Batch quantization completed: {}/{} successful", results.len(), models.len());
        Ok(results)
    }
    
    /// Auto-quantize model based on hardware capabilities
    pub async fn auto_quantize(
        &self,
        source_model_path: &Path,
        model_type: ModelType,
        output_dir: &Path,
    ) -> Result<Vec<QuantizationResult>> {
        info!("Auto-quantizing {:?} model based on hardware capabilities", model_type);
        
        let gpu_info = self.hardware_detector.get_gpu_info();
        let memory_config = self.hardware_detector.get_recommended_config().await;
        
        // Determine target precisions based on hardware
        let mut target_precisions = Vec::new();
        
        // Always create FP32 as fallback
        target_precisions.push(ModelPrecision::FP32);
        
        // Add FP16 if supported
        if gpu_info.supports_fp16 {
            target_precisions.push(ModelPrecision::FP16);
        }
        
        // Add INT8 for memory-constrained systems
        if gpu_info.supports_int8 || memory_config.max_model_memory_mb < 4000 {
            target_precisions.push(ModelPrecision::INT8);
        }
        
        // Create quantization requests
        let mut requests = Vec::new();
        for precision in target_precisions {
            let filename = format!("{:?}-{:?}.onnx", model_type, precision).to_lowercase();
            let output_path = output_dir.join(filename);
            
            requests.push(BatchQuantizationRequest {
                source_path: source_model_path.to_path_buf(),
                output_path,
                target_precision: precision,
                model_type,
            });
        }
        
        self.batch_quantize(requests).await
    }
    
    /// Get quantization metrics
    pub async fn get_metrics(&self) -> QuantizationMetrics {
        self.metrics.read().await.clone()
    }
    
    /// Clear quantization cache
    pub async fn clear_cache(&self) -> Result<()> {
        let mut cache = self.quantized_cache.write().await;
        cache.clear();
        
        {
            let mut metrics = self.metrics.write().await;
            metrics.cache_clears += 1;
        }
        
        info!("Quantization cache cleared");
        Ok(())
    }
    
    /// Validate quantization quality
    pub async fn validate_quantization_quality(
        &self,
        original_model: &Path,
        quantized_model: &Path,
        model_type: ModelType,
    ) -> Result<QualityMetrics> {
        self.validate_quantized_model(original_model, quantized_model, model_type).await
    }
    
    // Private implementation methods
    
    async fn detect_supported_methods(
        hardware_detector: &Arc<HardwareDetector>,
    ) -> Vec<QuantizationMethod> {
        let gpu_info = hardware_detector.get_gpu_info();
        let mut methods = Vec::new();
        
        // Dynamic quantization (always supported)
        methods.push(QuantizationMethod {
            name: "dynamic".to_string(),
            method_type: QuantizationMethodType::Dynamic,
            supported_precisions: vec![
                ModelPrecision::FP16, 
                ModelPrecision::INT8,
            ],
            hardware_requirements: HardwareRequirements {
                min_compute_capability: 0.0,
                requires_gpu: false,
                memory_overhead_mb: 100,
            },
            quality_impact: QualityImpact::Low,
        });
        
        // Static quantization
        methods.push(QuantizationMethod {
            name: "static".to_string(),
            method_type: QuantizationMethodType::Static,
            supported_precisions: vec![
                ModelPrecision::INT8,
            ],
            hardware_requirements: HardwareRequirements {
                min_compute_capability: 0.0,
                requires_gpu: false,
                memory_overhead_mb: 200,
            },
            quality_impact: QualityImpact::Medium,
        });
        
        // Mixed precision (if GPU supports it)
        if gpu_info.supports_fp16 {
            methods.push(QuantizationMethod {
                name: "mixed_precision".to_string(),
                method_type: QuantizationMethodType::MixedPrecision,
                supported_precisions: vec![
                    ModelPrecision::FP16,
                ],
                hardware_requirements: HardwareRequirements {
                    min_compute_capability: 6.0,
                    requires_gpu: true,
                    memory_overhead_mb: 150,
                },
                quality_impact: QualityImpact::Minimal,
            });
        }
        
        methods
    }
    
    async fn select_quantization_method(
        &self,
        target_precision: ModelPrecision,
        model_type: ModelType,
    ) -> Result<QuantizationMethod> {
        // Find the best method for target precision and model type
        for method in &self.supported_methods {
            if method.supported_precisions.contains(&target_precision) {
                // Check hardware requirements
                let gpu_info = self.hardware_detector.get_gpu_info();
                
                if method.hardware_requirements.requires_gpu && !gpu_info.has_discrete_gpu {
                    continue;
                }
                
                if gpu_info.compute_capability < method.hardware_requirements.min_compute_capability {
                    continue;
                }
                
                // Check if we have enough memory overhead
                let memory_config = self.hardware_detector.get_recommended_config().await;
                if memory_config.max_model_memory_mb < method.hardware_requirements.memory_overhead_mb {
                    continue;
                }
                
                return Ok(method.clone());
            }
        }
        
        Err(anyhow::anyhow!(
            "No suitable quantization method found for {:?} precision and {:?} model",
            target_precision, model_type
        ))
    }
    
    async fn prepare_quantization_parameters(
        &self,
        method: &QuantizationMethod,
        target_precision: ModelPrecision,
        model_type: ModelType,
    ) -> QuantizationParameters {
        let mut params = QuantizationParameters::default();
        
        // Precision-specific parameters
        match target_precision {
            ModelPrecision::FP16 => {
                params.activation_type = "fp16".to_string();
                params.weight_type = "fp16".to_string();
            },
            ModelPrecision::INT8 => {
                params.activation_type = "int8".to_string();
                params.weight_type = "int8".to_string();
                params.calibration_method = "minmax".to_string();
            },
            ModelPrecision::INT4 => {
                params.activation_type = "int8".to_string(); // Activation stays INT8
                params.weight_type = "int4".to_string();
                params.calibration_method = "percentile".to_string();
            },
            _ => {
                // FP32 - no quantization needed
            }
        }
        
        // Model-specific optimizations
        match model_type {
            ModelType::WhisperV3Turbo => {
                params.optimize_for_latency = true;
                params.preserve_accuracy_layers = vec![
                    "encoder.final_layer_norm".to_string(),
                    "decoder.final_layer_norm".to_string(),
                ];
            },
            ModelType::NLLB600M => {
                params.optimize_for_throughput = true;
                params.preserve_accuracy_layers = vec![
                    "encoder.embed_tokens".to_string(),
                    "decoder.embed_tokens".to_string(),
                ];
            },
            _ => {
                // Default parameters
            }
        }
        
        // Method-specific parameters
        match method.method_type {
            QuantizationMethodType::Dynamic => {
                params.use_dynamic_quantization = true;
            },
            QuantizationMethodType::Static => {
                params.use_static_quantization = true;
                params.calibration_dataset_size = self.config.calibration_samples;
            },
            QuantizationMethodType::MixedPrecision => {
                params.use_mixed_precision = true;
                params.fp16_ops_percentage = 0.8; // 80% of ops in FP16
            },
        }
        
        params
    }
    
    async fn execute_quantization(
        &self,
        source_path: &Path,
        output_path: &Path,
        method: &QuantizationMethod,
        params: &QuantizationParameters,
        model_type: ModelType,
    ) -> Result<IntermediateQuantizationResult> {
        // Create output directory if it doesn't exist
        if let Some(parent) = output_path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        
        // In a real implementation, this would use ONNX quantization tools
        // For now, we'll simulate the quantization process
        
        debug!("Executing quantization with method: {} for {:?}", method.name, model_type);
        debug!("Parameters: {:?}", params);
        
        // Simulate quantization by copying the file (placeholder)
        tokio::fs::copy(source_path, output_path).await?;
        
        // Add metadata to indicate this is quantized
        let metadata_path = output_path.with_extension("quantized.json");
        let metadata = QuantizationMetadata {
            source_model: source_path.to_path_buf(),
            target_precision: params.weight_type.clone(),
            method_used: method.name.clone(),
            quantization_params: params.clone(),
            quantized_at: chrono::Utc::now(),
        };
        
        let metadata_json = serde_json::to_string_pretty(&metadata)?;
        tokio::fs::write(metadata_path, metadata_json).await?;
        
        // Determine source precision (placeholder logic)
        let source_precision = if source_path.to_string_lossy().contains("fp16") {
            ModelPrecision::FP16
        } else if source_path.to_string_lossy().contains("int8") {
            ModelPrecision::INT8
        } else {
            ModelPrecision::FP32
        };
        
        Ok(IntermediateQuantizationResult {
            output_path: output_path.to_path_buf(),
            source_precision,
        })
    }
    
    async fn validate_quantized_model(
        &self,
        original_path: &Path,
        quantized_path: &Path,
        model_type: ModelType,
    ) -> Result<QualityMetrics> {
        // In a real implementation, this would run validation tests
        // For now, we'll provide estimated quality metrics
        
        let original_size = self.get_file_size(original_path).await?;
        let quantized_size = self.get_file_size(quantized_path).await?;
        let compression_ratio = original_size as f32 / quantized_size as f32;
        
        // Estimate quality degradation based on compression ratio and model type
        let estimated_accuracy_loss = match model_type {
            ModelType::WhisperV3Turbo => {
                // Speech models are generally robust to quantization
                (compression_ratio - 1.0) * 0.02 // 2% loss per compression ratio unit
            },
            ModelType::NLLB600M => {
                // Translation models are more sensitive
                (compression_ratio - 1.0) * 0.03 // 3% loss per compression ratio unit
            },
            _ => {
                (compression_ratio - 1.0) * 0.025 // Default 2.5% loss
            }
        };
        
        Ok(QualityMetrics {
            accuracy_retention: 1.0 - estimated_accuracy_loss.min(0.15), // Cap at 15% loss
            latency_improvement: compression_ratio * 0.8, // Assume 80% of compression translates to speedup
            memory_reduction: compression_ratio,
            perplexity_increase: estimated_accuracy_loss * 5.0, // Rough estimate
            bleu_score_change: -estimated_accuracy_loss, // Negative change
        })
    }
    
    async fn get_file_size(&self, path: &Path) -> Result<u64> {
        let metadata = tokio::fs::metadata(path).await?;
        Ok(metadata.len())
    }
    
    fn generate_cache_key(
        &self,
        source_path: &Path,
        target_precision: ModelPrecision,
        model_type: ModelType,
    ) -> String {
        format!(
            "{}_{:?}_{:?}",
            source_path.to_string_lossy(),
            target_precision,
            model_type
        )
    }
}

// Supporting types and structures

#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    /// Number of calibration samples for static quantization
    pub calibration_samples: usize,
    /// Enable automatic validation
    pub enable_validation: bool,
    /// Cache quantized models
    pub enable_caching: bool,
    /// Maximum cache size in MB
    pub max_cache_size_mb: usize,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            calibration_samples: 100,
            enable_validation: true,
            enable_caching: true,
            max_cache_size_mb: 2048, // 2GB cache
        }
    }
}

#[derive(Debug, Clone)]
pub struct QuantizationMethod {
    pub name: String,
    pub method_type: QuantizationMethodType,
    pub supported_precisions: Vec<ModelPrecision>,
    pub hardware_requirements: HardwareRequirements,
    pub quality_impact: QualityImpact,
}

#[derive(Debug, Clone)]
pub enum QuantizationMethodType {
    Dynamic,
    Static,
    MixedPrecision,
}

#[derive(Debug, Clone)]
pub struct HardwareRequirements {
    pub min_compute_capability: f32,
    pub requires_gpu: bool,
    pub memory_overhead_mb: usize,
}

#[derive(Debug, Clone)]
pub enum QualityImpact {
    Minimal,   // <1% accuracy loss
    Low,       // 1-3% accuracy loss
    Medium,    // 3-7% accuracy loss
    High,      // >7% accuracy loss
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QuantizationParameters {
    pub activation_type: String,
    pub weight_type: String,
    pub calibration_method: String,
    pub use_dynamic_quantization: bool,
    pub use_static_quantization: bool,
    pub use_mixed_precision: bool,
    pub optimize_for_latency: bool,
    pub optimize_for_throughput: bool,
    pub preserve_accuracy_layers: Vec<String>,
    pub calibration_dataset_size: usize,
    pub fp16_ops_percentage: f32,
}

#[derive(Debug, Clone)]
pub struct BatchQuantizationRequest {
    pub source_path: PathBuf,
    pub output_path: PathBuf,
    pub target_precision: ModelPrecision,
    pub model_type: ModelType,
}

#[derive(Debug, Clone)]
pub struct QuantizationResult {
    pub source_path: PathBuf,
    pub output_path: PathBuf,
    pub source_precision: ModelPrecision,
    pub target_precision: ModelPrecision,
    pub model_type: ModelType,
    pub compression_ratio: f32,
    pub quantization_time_ms: u64,
    pub quality_metrics: QualityMetrics,
    pub method_used: String,
}

#[derive(Debug, Clone)]
struct IntermediateQuantizationResult {
    output_path: PathBuf,
    source_precision: ModelPrecision,
}

#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// Accuracy retention (0.0 to 1.0)
    pub accuracy_retention: f32,
    /// Latency improvement factor
    pub latency_improvement: f32,
    /// Memory reduction factor
    pub memory_reduction: f32,
    /// Perplexity increase
    pub perplexity_increase: f32,
    /// BLEU score change
    pub bleu_score_change: f32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QuantizationMetrics {
    /// Total quantizations performed
    pub total_quantizations: u64,
    /// Total quantization time
    pub total_quantization_time_ms: u64,
    /// Total size reduction achieved
    pub total_size_reduction_bytes: u64,
    /// Average compression ratio
    pub average_compression_ratio: f32,
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// Cache clears
    pub cache_clears: u64,
}

#[derive(Debug, Clone)]
struct QuantizedModelInfo {
    output_path: PathBuf,
    source_precision: ModelPrecision,
    target_precision: ModelPrecision,
    compression_ratio: f32,
    quality_metrics: QualityMetrics,
    method_used: String,
    quantized_at: std::time::SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct QuantizationMetadata {
    source_model: PathBuf,
    target_precision: String,
    method_used: String,
    quantization_params: QuantizationParameters,
    quantized_at: chrono::DateTime<chrono::Utc>,
}