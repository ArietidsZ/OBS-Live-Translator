//! Adaptive GPU memory management for cross-platform AI model deployment

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Adaptive memory manager that optimizes model deployment based on available VRAM
pub struct AdaptiveMemoryManager {
    /// Total VRAM available in bytes
    total_vram: AtomicU64,
    /// Currently allocated VRAM in bytes
    allocated_vram: AtomicU64,
    /// Reserved VRAM for system/other applications (20% default)
    reserved_vram: AtomicU64,
    /// Current memory configuration
    current_config: Arc<RwLock<MemoryConfiguration>>,
    /// Active model allocations
    model_allocations: Arc<RwLock<HashMap<String, ModelAllocation>>>,
    /// GPU vendor and capabilities
    hardware_info: HardwareInfo,
    /// Performance metrics
    metrics: Arc<RwLock<MemoryMetrics>>,
}

impl AdaptiveMemoryManager {
    /// Create new adaptive memory manager
    pub async fn new() -> Result<Self> {
        let hardware_info = Self::detect_hardware().await?;
        let total_vram = hardware_info.total_vram_mb * 1024 * 1024; // Convert MB to bytes
        let reserved_vram = (total_vram as f64 * 0.2) as u64; // Reserve 20%
        
        info!(
            "Detected GPU: {} with {}MB VRAM ({}MB usable)", 
            hardware_info.gpu_name,
            hardware_info.total_vram_mb,
            (total_vram - reserved_vram) / 1024 / 1024
        );
        
        let config = Self::determine_optimal_configuration(total_vram - reserved_vram).await;
        
        Ok(Self {
            total_vram: AtomicU64::new(total_vram),
            allocated_vram: AtomicU64::new(0),
            reserved_vram: AtomicU64::new(reserved_vram),
            current_config: Arc::new(RwLock::new(config)),
            model_allocations: Arc::new(RwLock::new(HashMap::new())),
            hardware_info,
            metrics: Arc::new(RwLock::new(MemoryMetrics::default())),
        })
    }
    
    /// Get the optimal model configuration for current hardware
    pub async fn get_model_configuration(&self) -> ModelConfiguration {
        let config = self.current_config.read().await;
        config.model_config.clone()
    }
    
    /// Allocate memory for a specific model
    pub async fn allocate_model_memory(
        &self,
        model_id: &str,
        model_type: ModelType,
        precision: ModelPrecision,
    ) -> Result<ModelAllocation> {
        let required_bytes = self.calculate_model_memory_requirements(model_type, precision).await;
        let available_bytes = self.get_available_vram().await;
        
        if required_bytes > available_bytes {
            // Try to free memory or reduce precision
            if let Some(allocation) = self.optimize_memory_allocation(model_id, model_type, required_bytes).await? {
                return Ok(allocation);
            }
            
            return Err(anyhow::anyhow!(
                "Insufficient VRAM: need {}MB, have {}MB", 
                required_bytes / 1024 / 1024,
                available_bytes / 1024 / 1024
            ));
        }
        
        let allocation = ModelAllocation {
            model_id: model_id.to_string(),
            model_type,
            precision,
            allocated_bytes: required_bytes,
            allocated_at: std::time::SystemTime::now(),
            last_used: std::time::SystemTime::now(),
            usage_count: AtomicUsize::new(0),
        };
        
        // Record allocation
        self.allocated_vram.fetch_add(required_bytes, Ordering::SeqCst);
        self.model_allocations.write().await.insert(model_id.to_string(), allocation.clone());
        
        // Update metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.total_allocations += 1;
            metrics.current_allocated_mb = self.allocated_vram.load(Ordering::SeqCst) / 1024 / 1024;
            metrics.peak_allocated_mb = metrics.peak_allocated_mb.max(metrics.current_allocated_mb);
        }
        
        info!(
            "Allocated {}MB VRAM for {} model ({})", 
            required_bytes / 1024 / 1024, 
            model_id,
            format!("{:?}", precision)
        );
        
        Ok(allocation)
    }
    
    /// Deallocate memory for a specific model
    pub async fn deallocate_model_memory(&self, model_id: &str) -> Result<()> {
        let mut allocations = self.model_allocations.write().await;
        
        if let Some(allocation) = allocations.remove(model_id) {
            self.allocated_vram.fetch_sub(allocation.allocated_bytes, Ordering::SeqCst);
            
            // Update metrics
            {
                let mut metrics = self.metrics.write().await;
                metrics.total_deallocations += 1;
                metrics.current_allocated_mb = self.allocated_vram.load(Ordering::SeqCst) / 1024 / 1024;
            }
            
            info!("Deallocated {}MB VRAM from {} model", 
                  allocation.allocated_bytes / 1024 / 1024, model_id);
        }
        
        Ok(())
    }
    
    /// Get current memory utilization
    pub async fn get_memory_utilization(&self) -> MemoryUtilization {
        let total = self.total_vram.load(Ordering::SeqCst);
        let allocated = self.allocated_vram.load(Ordering::SeqCst);
        let reserved = self.reserved_vram.load(Ordering::SeqCst);
        let available = total - allocated - reserved;
        
        MemoryUtilization {
            total_vram_mb: total / 1024 / 1024,
            allocated_vram_mb: allocated / 1024 / 1024,
            available_vram_mb: available / 1024 / 1024,
            reserved_vram_mb: reserved / 1024 / 1024,
            utilization_percentage: (allocated as f64 / (total - reserved) as f64) * 100.0,
            active_models: self.model_allocations.read().await.len(),
        }
    }
    
    /// Optimize memory allocation by potentially reducing precision or swapping models
    async fn optimize_memory_allocation(
        &self,
        model_id: &str,
        model_type: ModelType,
        required_bytes: u64,
    ) -> Result<Option<ModelAllocation>> {
        // Try reducing precision first
        let precisions = [ModelPrecision::INT4, ModelPrecision::INT8, ModelPrecision::FP16];
        
        for precision in precisions.iter() {
            let bytes_needed = self.calculate_model_memory_requirements(model_type, *precision).await;
            if bytes_needed <= self.get_available_vram().await {
                warn!("Reducing precision to {:?} for model {} due to memory constraints", precision, model_id);
                return Ok(Some(ModelAllocation {
                    model_id: model_id.to_string(),
                    model_type,
                    precision: *precision,
                    allocated_bytes: bytes_needed,
                    allocated_at: std::time::SystemTime::now(),
                    last_used: std::time::SystemTime::now(),
                    usage_count: AtomicUsize::new(0),
                }));
            }
        }
        
        // Try freeing least recently used models
        if let Some(lru_model) = self.find_least_recently_used_model().await {
            info!("Freeing LRU model {} to make space for {}", lru_model, model_id);
            self.deallocate_model_memory(&lru_model).await?;
            
            // Retry allocation
            if required_bytes <= self.get_available_vram().await {
                return Ok(Some(ModelAllocation {
                    model_id: model_id.to_string(),
                    model_type,
                    precision: ModelPrecision::FP16, // Default precision
                    allocated_bytes: required_bytes,
                    allocated_at: std::time::SystemTime::now(),
                    last_used: std::time::SystemTime::now(),
                    usage_count: AtomicUsize::new(0),
                }));
            }
        }
        
        Ok(None)
    }
    
    /// Calculate memory requirements for a model
    async fn calculate_model_memory_requirements(
        &self,
        model_type: ModelType,
        precision: ModelPrecision,
    ) -> u64 {
        let base_parameters = match model_type {
            // Voice Recognition Models
            ModelType::WhisperV3Turbo => 1_550_000_000_u64, // 1.55B parameters
            ModelType::DistilWhisper => 244_000_000,        // 244M parameters
            ModelType::CanaryFlash => 1_000_000_000,        // 1B parameters
            
            // Translation Models
            ModelType::NLLB600M => 600_000_000,             // 600M parameters
            ModelType::OpusMT => 77_000_000,                // 77M parameters average
            ModelType::mBART => 610_000_000,                // 610M parameters
            
            // Summarization Models
            ModelType::DistilBART => 406_000_000,           // 406M parameters
            ModelType::PegasusSmall => 568_000_000,         // 568M parameters
            ModelType::T5Small => 60_000_000,               // 60M parameters
        };
        
        let bytes_per_parameter = match precision {
            ModelPrecision::FP32 => 4,
            ModelPrecision::FP16 => 2,
            ModelPrecision::INT8 => 1,
            ModelPrecision::INT4 => 1, // 0.5 bytes but rounded up for simplicity
        };
        
        let model_size = base_parameters * bytes_per_parameter;
        
        // Add overhead for activations, gradients, optimizer states
        let overhead_multiplier = match model_type {
            ModelType::WhisperV3Turbo | ModelType::CanaryFlash => 1.5, // Transformer models need more activation memory
            ModelType::NLLB600M | ModelType::mBART => 1.4,
            _ => 1.2,
        };
        
        (model_size as f64 * overhead_multiplier) as u64
    }
    
    /// Detect hardware information
    async fn detect_hardware() -> Result<HardwareInfo> {
        // This would integrate with actual hardware detection libraries
        // For now, we'll simulate detection
        
        #[cfg(feature = "cuda")]
        if let Ok(device_count) = Self::detect_nvidia_gpu().await {
            if device_count > 0 {
                return Ok(HardwareInfo {
                    gpu_vendor: GpuVendor::NVIDIA,
                    gpu_name: "NVIDIA GPU".to_string(),
                    total_vram_mb: Self::get_nvidia_vram().await.unwrap_or(4096),
                    compute_capability: (7, 5), // Example compute capability
                    supports_fp16: true,
                    supports_int8: true,
                    supports_int4: false, // Most consumer GPUs don't support INT4 natively
                });
            }
        }
        
        #[cfg(feature = "rocm")]
        if Self::detect_amd_gpu().await.is_ok() {
            return Ok(HardwareInfo {
                gpu_vendor: GpuVendor::AMD,
                gpu_name: "AMD GPU".to_string(),
                total_vram_mb: Self::get_amd_vram().await.unwrap_or(8192),
                compute_capability: (0, 0), // AMD uses different versioning
                supports_fp16: true,
                supports_int8: true,
                supports_int4: false,
            });
        }
        
        #[cfg(target_os = "macos")]
        {
            return Ok(HardwareInfo {
                gpu_vendor: GpuVendor::Apple,
                gpu_name: "Apple GPU".to_string(),
                total_vram_mb: Self::get_apple_unified_memory().await.unwrap_or(16384), // Unified memory
                compute_capability: (0, 0),
                supports_fp16: true,
                supports_int8: true,
                supports_int4: false,
            });
        }
        
        // Fallback to Intel integrated graphics
        Ok(HardwareInfo {
            gpu_vendor: GpuVendor::Intel,
            gpu_name: "Intel Integrated Graphics".to_string(),
            total_vram_mb: 2048, // Typical shared memory allocation
            compute_capability: (0, 0),
            supports_fp16: true,
            supports_int8: false,
            supports_int4: false,
        })
    }
    
    /// Determine optimal configuration based on available VRAM
    async fn determine_optimal_configuration(usable_vram_bytes: u64) -> MemoryConfiguration {
        let usable_vram_mb = usable_vram_bytes / 1024 / 1024;
        
        let model_config = match usable_vram_mb {
            mb if mb >= 6144 => ModelConfiguration::HighEnd,    // 6GB+
            mb if mb >= 4096 => ModelConfiguration::MidRange,   // 4-6GB
            mb if mb >= 2048 => ModelConfiguration::LowEnd,     // 2-4GB
            _ => ModelConfiguration::CPUFallback,               // <2GB
        };
        
        MemoryConfiguration {
            total_usable_mb: usable_vram_mb,
            model_config,
            max_concurrent_models: match usable_vram_mb {
                mb if mb >= 6144 => 4,
                mb if mb >= 4096 => 3,
                mb if mb >= 2048 => 2,
                _ => 1,
            },
            preferred_precision: match usable_vram_mb {
                mb if mb >= 6144 => ModelPrecision::FP16,
                mb if mb >= 4096 => ModelPrecision::FP16,
                mb if mb >= 2048 => ModelPrecision::INT8,
                _ => ModelPrecision::INT8,
            },
        }
    }
    
    async fn get_available_vram(&self) -> u64 {
        let total = self.total_vram.load(Ordering::SeqCst);
        let allocated = self.allocated_vram.load(Ordering::SeqCst);
        let reserved = self.reserved_vram.load(Ordering::SeqCst);
        total - allocated - reserved
    }
    
    async fn find_least_recently_used_model(&self) -> Option<String> {
        let allocations = self.model_allocations.read().await;
        
        allocations
            .values()
            .min_by_key(|allocation| allocation.last_used)
            .map(|allocation| allocation.model_id.clone())
    }
    
    // Platform-specific hardware detection methods
    #[cfg(feature = "cuda")]
    async fn detect_nvidia_gpu() -> Result<i32> {
        // This would use nvidia-ml-py or similar
        // Placeholder implementation
        Ok(1)
    }
    
    #[cfg(feature = "cuda")]
    async fn get_nvidia_vram() -> Result<u64> {
        // This would query actual NVIDIA VRAM
        Ok(8192) // 8GB placeholder
    }
    
    #[cfg(feature = "rocm")]
    async fn detect_amd_gpu() -> Result<()> {
        // This would use rocm-smi or similar
        Ok(())
    }
    
    #[cfg(feature = "rocm")]
    async fn get_amd_vram() -> Result<u64> {
        // This would query actual AMD VRAM
        Ok(16384) // 16GB placeholder
    }
    
    #[cfg(target_os = "macos")]
    async fn get_apple_unified_memory() -> Result<u64> {
        // This would query actual unified memory
        Ok(16384) // 16GB placeholder
    }
}

/// Hardware information detected at runtime
#[derive(Debug, Clone)]
pub struct HardwareInfo {
    pub gpu_vendor: GpuVendor,
    pub gpu_name: String,
    pub total_vram_mb: u64,
    pub compute_capability: (i32, i32),
    pub supports_fp16: bool,
    pub supports_int8: bool,
    pub supports_int4: bool,
}

/// GPU vendor types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GpuVendor {
    NVIDIA,
    AMD,
    Intel,
    Apple,
    Unknown,
}

/// Memory configuration tiers
#[derive(Debug, Clone)]
pub struct MemoryConfiguration {
    pub total_usable_mb: u64,
    pub model_config: ModelConfiguration,
    pub max_concurrent_models: usize,
    pub preferred_precision: ModelPrecision,
}

/// Model configuration tiers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelConfiguration {
    HighEnd,      // 6GB+ VRAM
    MidRange,     // 4-6GB VRAM
    LowEnd,       // 2-4GB VRAM
    CPUFallback,  // <2GB VRAM
}

/// Model types supported by the system
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ModelType {
    // Voice Recognition
    WhisperV3Turbo,
    DistilWhisper,
    CanaryFlash,
    
    // Translation
    NLLB600M,
    OpusMT,
    mBART,
    
    // Summarization
    DistilBART,
    PegasusSmall,
    T5Small,
}

/// Model precision levels
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ModelPrecision {
    FP32,
    FP16,
    INT8,
    INT4,
}

/// Model allocation information
#[derive(Debug, Clone)]
pub struct ModelAllocation {
    pub model_id: String,
    pub model_type: ModelType,
    pub precision: ModelPrecision,
    pub allocated_bytes: u64,
    pub allocated_at: std::time::SystemTime,
    pub last_used: std::time::SystemTime,
    pub usage_count: AtomicUsize,
}

/// Current memory utilization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUtilization {
    pub total_vram_mb: u64,
    pub allocated_vram_mb: u64,
    pub available_vram_mb: u64,
    pub reserved_vram_mb: u64,
    pub utilization_percentage: f64,
    pub active_models: usize,
}

/// Memory management metrics
#[derive(Debug, Clone, Default)]
pub struct MemoryMetrics {
    pub total_allocations: u64,
    pub total_deallocations: u64,
    pub current_allocated_mb: u64,
    pub peak_allocated_mb: u64,
    pub allocation_failures: u64,
    pub precision_downgrades: u64,
    pub lru_evictions: u64,
}