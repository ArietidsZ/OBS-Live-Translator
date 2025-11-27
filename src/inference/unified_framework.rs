//! Unified ML Inference Framework
//!
//! This module provides a comprehensive framework for ML inference across different profiles:
//! - Profile-aware model loading and optimization
//! - Batch processing optimization with dynamic batching
//! - Model warming and caching strategies
//! - Hardware acceleration integration (TensorRT, ONNX Runtime)

use super::{Device, ModelMetadata, SessionConfig, TimingInfo};
use crate::profile::Profile;
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::time::Instant;
use tracing::{debug, info, warn};

/// Unified inference framework that manages multiple inference engines
pub struct UnifiedInferenceFramework {
    /// Current active profile
    profile: Profile,
    /// Engine registry by model type and profile
    engines: HashMap<(ModelType, Profile), Box<dyn InferenceEngine>>,
    /// Model cache manager
    model_cache: ModelCacheManager,
    /// Batch processor for throughput optimization (future feature)
    #[allow(dead_code)]
    batch_processor: DynamicBatchProcessor, // Future batch processing
    #[allow(dead_code)]
    /// Performance monitor
    performance_monitor: PerformanceMonitor,
    /// Hardware capabilities
    hardware_info: HardwareCapabilities,
}

/// Core inference engine trait that all models must implement
pub trait InferenceEngine: Send + Sync {
    /// Initialize the model with the given configuration
    fn initialize(&mut self, config: &InferenceConfig) -> Result<()>;

    /// Run single inference
    fn infer(&mut self, inputs: &InferenceInputs) -> Result<InferenceOutputs>;

    /// Run batch inference (optional, defaults to sequential single inference)
    fn infer_batch(&mut self, batch: &[InferenceInputs]) -> Result<Vec<InferenceOutputs>> {
        batch.iter().map(|input| self.infer(input)).collect()
    }

    /// Get model metadata
    fn metadata(&self) -> &ModelMetadata;

    /// Get performance statistics
    fn stats(&self) -> InferenceStats;

    /// Warm up the model (optional)
    fn warmup(&mut self) -> Result<()> {
        Ok(())
    }

    /// Check if model supports hardware acceleration
    fn supports_acceleration(&self) -> bool {
        false
    }
}

/// Model types supported by the framework
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum ModelType {
    /// Speech-to-text models (Whisper variants)
    ASR,
    /// Neural machine translation models
    Translation,
    /// Language detection models
    LanguageDetection,
    /// Voice activity detection
    VAD,
    /// Custom model type
    Custom(String),
}

/// Enhanced inference configuration
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Basic session configuration
    pub session: SessionConfig,
    /// Target profile for optimization
    pub profile: Profile,
    /// Model warming configuration
    pub warming_config: ModelWarmingConfig,
    /// Batch processing configuration
    pub batch_config: BatchConfig,
    /// Hardware acceleration preferences
    pub acceleration_config: AccelerationConfig,
}

/// Model warming configuration
#[derive(Debug, Clone)]
pub struct ModelWarmingConfig {
    /// Enable model warming on load
    pub enable_warming: bool,
    /// Number of warmup iterations
    pub warmup_iterations: usize,
    /// Warmup input shapes (if None, use default shapes)
    pub warmup_shapes: Option<Vec<Vec<i64>>>,
    /// Maximum warmup time in seconds
    pub max_warmup_time_secs: u64,
}

impl Default for ModelWarmingConfig {
    fn default() -> Self {
        Self {
            enable_warming: true,
            warmup_iterations: 3,
            warmup_shapes: None,
            max_warmup_time_secs: 30,
        }
    }
}

/// Batch processing configuration
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum batch size
    pub max_batch_size: usize,
    /// Batch timeout in milliseconds
    pub batch_timeout_ms: u64,
    /// Enable dynamic batching
    pub enable_dynamic_batching: bool,
    /// Preferred batch size for optimal throughput
    pub optimal_batch_size: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 8,
            batch_timeout_ms: 50,
            enable_dynamic_batching: true,
            optimal_batch_size: 4,
        }
    }
}

/// Hardware acceleration configuration
#[derive(Debug, Clone)]
pub struct AccelerationConfig {
    /// Enable TensorRT optimization (NVIDIA GPUs)
    pub enable_tensorrt: bool,
    /// Enable ONNX Runtime optimization passes
    pub enable_onnx_optimization: bool,
    /// Enable Flash Attention (for transformer models)
    pub enable_flash_attention: bool,
    /// GPU memory limit in MB (0 = unlimited)
    pub gpu_memory_limit_mb: usize,
    /// Precision preference
    pub precision_preference: PrecisionPreference,
}

#[derive(Debug, Clone)]
pub enum PrecisionPreference {
    /// Highest accuracy, may be slower
    FP32,
    /// Balanced accuracy and speed
    FP16,
    /// Fastest, reduced accuracy
    INT8,
    /// Experimental: very fast, significantly reduced accuracy
    INT4,
    /// Auto-select based on hardware capabilities
    Auto,
}

impl Default for AccelerationConfig {
    fn default() -> Self {
        Self {
            enable_tensorrt: true,
            enable_onnx_optimization: true,
            enable_flash_attention: true,
            gpu_memory_limit_mb: 0,
            precision_preference: PrecisionPreference::Auto,
        }
    }
}

/// Generic inference inputs container
#[derive(Debug, Clone)]
pub struct InferenceInputs {
    /// Named input tensors
    pub tensors: HashMap<String, Vec<f32>>,
    /// Input metadata (optional)
    pub metadata: HashMap<String, String>,
}

/// Generic inference outputs container
#[derive(Debug, Clone)]
pub struct InferenceOutputs {
    /// Named output tensors
    pub tensors: HashMap<String, Vec<f32>>,
    /// Output metadata (optional)
    pub metadata: HashMap<String, String>,
    /// Timing information
    pub timing: TimingInfo,
    /// Confidence score (0.0-1.0)
    pub confidence: f32,
}

/// Model cache manager for efficient model loading/unloading
struct ModelCacheManager {
    /// Cached models by profile and type
    cache: HashMap<(ModelType, Profile), CachedModel>,
    /// Cache configuration (for future functionality)
    #[allow(dead_code)]
    config: CacheConfig,
    /// Memory usage tracking
    memory_usage_mb: f64,
}

#[derive(Debug, Clone)]
struct CachedModel {
    /// Last access time (for LRU eviction)
    #[allow(dead_code)]
    last_access: Instant,
    /// Model size in MB (for cache size tracking)
    #[allow(dead_code)]
    size_mb: f64,
    /// Access count (for statistics)
    #[allow(dead_code)]
    access_count: u64,
}

#[derive(Debug, Clone)]
struct CacheConfig {
    /// Maximum cache size in MB (future feature)
    #[allow(dead_code)]
    max_cache_size_mb: f64,
    /// Maximum model age before eviction (future feature)
    #[allow(dead_code)]
    max_age_seconds: u64,
    /// Enable LRU eviction (future feature)
    #[allow(dead_code)]
    enable_lru: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_cache_size_mb: 8192.0, // 8GB default
            max_age_seconds: 3600,     // 1 hour
            enable_lru: true,
        }
    }
}

/// Dynamic batch processor for optimal throughput
struct DynamicBatchProcessor {
    /// Pending requests queue (future batch processing)
    #[allow(dead_code)]
    pending_requests: Vec<PendingRequest>,
    /// Batch configuration
    config: BatchConfig,
    /// Throughput statistics
    throughput_stats: ThroughputStats,
}

struct PendingRequest {
    #[allow(dead_code)]
    inputs: InferenceInputs,
    #[allow(dead_code)]
    timestamp: Instant,
}

#[derive(Debug, Clone, Default)]
struct ThroughputStats {
    #[allow(dead_code)]
    total_requests: u64,
    #[allow(dead_code)]
    total_batches: u64,
    #[allow(dead_code)]
    average_batch_size: f32,
    #[allow(dead_code)]
    average_latency_ms: f32,
    #[allow(dead_code)]
    throughput_per_second: f32,
}

/// Performance monitoring for the inference framework
struct PerformanceMonitor {
    /// Performance metrics by model type
    metrics: HashMap<ModelType, ModelPerformanceMetrics>,
    /// System resource monitoring (future feature)
    #[allow(dead_code)]
    resource_monitor: SystemResourceMonitor,
}

#[derive(Debug, Clone, Default)]
pub struct ModelPerformanceMetrics {
    /// Total inferences run
    total_inferences: u64,
    /// Average latency in milliseconds
    average_latency_ms: f32,
    /// 95th percentile latency
    p95_latency_ms: f32,
    /// Throughput (inferences per second)
    throughput_per_second: f32,
    /// Memory usage in MB
    memory_usage_mb: f64,
    /// GPU utilization (0.0-1.0)
    gpu_utilization: f32,
}

#[derive(Debug, Clone, Default)]
struct SystemResourceMonitor {
    /// CPU usage percentage (future monitoring)
    #[allow(dead_code)]
    cpu_usage: f32,
    /// Available memory in MB (future monitoring)
    #[allow(dead_code)]
    available_memory_mb: f64,
    /// GPU memory usage in MB (future monitoring)
    #[allow(dead_code)]
    gpu_memory_usage_mb: f64,
    /// GPU temperature in Celsius (future monitoring)
    #[allow(dead_code)]
    gpu_temperature_c: f32,
}

/// Hardware capabilities detection
#[derive(Debug, Clone)]
pub struct HardwareCapabilities {
    /// Available compute devices
    devices: Vec<ComputeDevice>,
    /// Supported precision types (future feature)
    #[allow(dead_code)]
    supported_precisions: Vec<PrecisionPreference>,
    /// Available memory per device (future feature)
    #[allow(dead_code)]
    device_memory_mb: HashMap<Device, f64>,
}

#[derive(Debug, Clone)]
struct ComputeDevice {
    device: Device,
    name: String,
    compute_capability: Option<String>,
    memory_mb: f64,
    supports_tensorrt: bool,
    supports_flash_attention: bool,
}

/// Inference statistics
#[derive(Debug, Clone, Default)]
pub struct InferenceStats {
    /// Total inference time in milliseconds
    pub total_time_ms: f32,
    /// Preprocessing time
    pub preprocessing_ms: f32,
    /// Core inference time
    pub inference_ms: f32,
    /// Postprocessing time
    pub postprocessing_ms: f32,
    /// Memory usage in MB
    pub memory_usage_mb: f64,
    /// Batch size used
    pub batch_size: usize,
}

impl UnifiedInferenceFramework {
    /// Create a new unified inference framework
    pub fn new(profile: Profile) -> Result<Self> {
        info!(
            "🚀 Initializing Unified Inference Framework for profile: {:?}",
            profile
        );

        let hardware_info = Self::detect_hardware_capabilities()?;
        debug!(
            "🔧 Detected hardware: {} devices",
            hardware_info.devices.len()
        );

        let model_cache = ModelCacheManager {
            cache: HashMap::new(),
            config: CacheConfig::default(),
            memory_usage_mb: 0.0,
        };

        let batch_processor = DynamicBatchProcessor {
            pending_requests: Vec::new(),
            config: BatchConfig::default(),
            throughput_stats: ThroughputStats::default(),
        };

        let performance_monitor = PerformanceMonitor {
            metrics: HashMap::new(),
            resource_monitor: SystemResourceMonitor::default(),
        };

        Ok(Self {
            profile,
            engines: HashMap::new(),
            model_cache,
            batch_processor,
            performance_monitor,
            hardware_info,
        })
    }

    /// Register an inference engine for a specific model type and profile
    pub fn register_engine(
        &mut self,
        model_type: ModelType,
        profile: Profile,
        engine: Box<dyn InferenceEngine>,
    ) -> Result<()> {
        info!(
            "📝 Registering {} engine for profile {:?}",
            format!("{:?}", model_type),
            profile
        );

        self.engines.insert((model_type, profile), engine);
        Ok(())
    }

    /// Get an inference engine for the current profile and model type
    pub fn get_engine(&mut self, model_type: &ModelType) -> Result<&mut Box<dyn InferenceEngine>> {
        let key = (model_type.clone(), self.profile);
        self.engines.get_mut(&key).ok_or_else(|| {
            anyhow!(
                "No engine registered for {:?} with profile {:?}",
                model_type,
                self.profile
            )
        })
    }

    /// Initialize all registered engines
    pub fn initialize_engines(
        &mut self,
        configs: HashMap<ModelType, InferenceConfig>,
    ) -> Result<()> {
        info!("🔄 Initializing all inference engines...");

        let current_profile = self.profile;
        for ((model_type, profile), engine) in &mut self.engines {
            if *profile == current_profile {
                if let Some(config) = configs.get(model_type) {
                    engine.initialize(config)?;

                    // Perform model warming if enabled
                    if config.warming_config.enable_warming {
                        let warming_config = &config.warming_config;
                        Self::warm_model_static(engine, warming_config)?;
                    }

                    info!(
                        "✅ Initialized {} engine for profile {:?}",
                        format!("{:?}", model_type),
                        profile
                    );
                }
            }
        }

        Ok(())
    }

    /// Warm up a model with dummy inputs
    fn warm_model(
        &self,
        engine: &mut Box<dyn InferenceEngine>,
        config: &ModelWarmingConfig,
    ) -> Result<()> {
        Self::warm_model_static(engine, config)
    }

    /// Static version of warm_model to avoid borrowing issues
    fn warm_model_static(
        engine: &mut Box<dyn InferenceEngine>,
        config: &ModelWarmingConfig,
    ) -> Result<()> {
        debug!(
            "🔥 Warming up model with {} iterations...",
            config.warmup_iterations
        );

        let start_time = Instant::now();
        let metadata = engine.metadata();

        // Create dummy inputs based on model metadata
        let dummy_inputs = Self::create_dummy_inputs_static(metadata, &config.warmup_shapes)?;

        for i in 0..config.warmup_iterations {
            if start_time.elapsed().as_secs() > config.max_warmup_time_secs {
                warn!(
                    "⚠️ Model warmup timeout after {} seconds",
                    config.max_warmup_time_secs
                );
                break;
            }

            debug!("🔥 Warmup iteration {}/{}", i + 1, config.warmup_iterations);
            let _ = engine.infer(&dummy_inputs); // Ignore warmup results
        }

        let warmup_time = start_time.elapsed();
        info!(
            "✅ Model warmup completed in {:.2}s",
            warmup_time.as_secs_f32()
        );

        Ok(())
    }

    /// Create dummy inputs for model warming
    fn create_dummy_inputs(
        &self,
        metadata: &ModelMetadata,
        warmup_shapes: &Option<Vec<Vec<i64>>>,
    ) -> Result<InferenceInputs> {
        Self::create_dummy_inputs_static(metadata, warmup_shapes)
    }

    /// Static version of create_dummy_inputs
    fn create_dummy_inputs_static(
        metadata: &ModelMetadata,
        warmup_shapes: &Option<Vec<Vec<i64>>>,
    ) -> Result<InferenceInputs> {
        let mut tensors = HashMap::new();

        let shapes = warmup_shapes.as_ref().unwrap_or(&metadata.input_shapes);

        for (i, shape) in shapes.iter().enumerate() {
            let input_name = metadata
                .input_names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("input_{}", i));

            let size: usize = shape.iter().map(|&dim| dim.max(1) as usize).product();
            let dummy_data = vec![0.1; size]; // Small positive values

            tensors.insert(input_name, dummy_data);
        }

        Ok(InferenceInputs {
            tensors,
            metadata: HashMap::new(),
        })
    }

    /// Detect hardware capabilities
    fn detect_hardware_capabilities() -> Result<HardwareCapabilities> {
        let mut devices = Vec::new();
        let mut device_memory = HashMap::new();

        // CPU device
        devices.push(ComputeDevice {
            device: Device::CPU,
            name: "CPU".to_string(),
            compute_capability: None,
            memory_mb: 8192.0, // Assume 8GB available for CPU
            supports_tensorrt: false,
            supports_flash_attention: false,
        });
        device_memory.insert(Device::CPU, 8192.0);

        // GPU detection (placeholder - would use actual GPU detection libraries)
        #[cfg(feature = "cuda")]
        {
            devices.push(ComputeDevice {
                device: Device::CUDA(0),
                name: "NVIDIA GPU".to_string(),
                compute_capability: Some("8.6".to_string()),
                memory_mb: 16384.0, // Assume 16GB VRAM
                supports_tensorrt: true,
                supports_flash_attention: true,
            });
            device_memory.insert(Device::CUDA(0), 16384.0);
        }

        #[cfg(target_os = "macos")]
        {
            devices.push(ComputeDevice {
                device: Device::CoreML,
                name: "Apple Neural Engine".to_string(),
                compute_capability: None,
                memory_mb: 32768.0, // Unified memory
                supports_tensorrt: false,
                supports_flash_attention: true,
            });
            device_memory.insert(Device::CoreML, 32768.0);
        }

        let supported_precisions = vec![
            PrecisionPreference::FP32,
            PrecisionPreference::FP16,
            PrecisionPreference::INT8,
            PrecisionPreference::Auto,
        ];

        Ok(HardwareCapabilities {
            devices,
            supported_precisions,
            device_memory_mb: device_memory,
        })
    }

    /// Switch to a different profile
    pub fn switch_profile(&mut self, new_profile: Profile) -> Result<()> {
        if new_profile == self.profile {
            return Ok(());
        }

        info!(
            "🔄 Switching inference framework from {:?} to {:?}",
            self.profile, new_profile
        );

        self.profile = new_profile;

        // Clear model cache to force reloading with new profile optimizations
        self.model_cache.cache.clear();
        self.model_cache.memory_usage_mb = 0.0;

        info!("✅ Profile switch completed");
        Ok(())
    }

    /// Get framework performance statistics
    pub fn get_performance_stats(&self) -> HashMap<ModelType, ModelPerformanceMetrics> {
        self.performance_monitor.metrics.clone()
    }

    /// Get hardware information
    pub fn get_hardware_info(&self) -> &HardwareCapabilities {
        &self.hardware_info
    }

    /// Update performance metrics
    fn update_performance_metrics(&mut self, model_type: &ModelType, stats: &InferenceStats) {
        let metrics = self
            .performance_monitor
            .metrics
            .entry(model_type.clone())
            .or_default();

        metrics.total_inferences += 1;

        // Update rolling averages
        let n = metrics.total_inferences as f32;
        metrics.average_latency_ms =
            (metrics.average_latency_ms * (n - 1.0) + stats.total_time_ms) / n;
        metrics.memory_usage_mb = stats.memory_usage_mb;

        // Update throughput (simplified calculation)
        if stats.total_time_ms > 0.0 {
            let current_throughput = 1000.0 / stats.total_time_ms; // inferences per second
            metrics.throughput_per_second =
                (metrics.throughput_per_second * (n - 1.0) + current_throughput) / n;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_framework_creation() {
        let framework = UnifiedInferenceFramework::new(Profile::Medium);
        assert!(framework.is_ok());
    }

    #[test]
    fn test_hardware_detection() {
        let hardware = UnifiedInferenceFramework::detect_hardware_capabilities();
        assert!(hardware.is_ok());

        let hw = hardware.unwrap();
        assert!(!hw.devices.is_empty());
        assert!(!hw.supported_precisions.is_empty());
    }

    #[test]
    fn test_profile_switching() {
        let mut framework = UnifiedInferenceFramework::new(Profile::Low).unwrap();
        assert_eq!(framework.profile, Profile::Low);

        let result = framework.switch_profile(Profile::High);
        assert!(result.is_ok());
        assert_eq!(framework.profile, Profile::High);
    }

    #[test]
    fn test_model_type_uniqueness() {
        use std::collections::HashSet;

        let types = vec![
            ModelType::ASR,
            ModelType::Translation,
            ModelType::LanguageDetection,
            ModelType::VAD,
            ModelType::Custom("test".to_string()),
        ];

        let set: HashSet<_> = types.into_iter().collect();
        assert_eq!(set.len(), 5);
    }
}
