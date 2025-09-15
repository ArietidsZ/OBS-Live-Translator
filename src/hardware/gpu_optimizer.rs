//! GPU-specific optimizations for 2020-2025 hardware
//!
//! Adaptive optimizations for NVIDIA RTX 2000-4000, AMD RX 5000-7000, and Intel Arc

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use super::{GpuArchitecture, GpuInfo, OptimizationLevel};

/// GPU optimization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuOptimizationSettings {
    /// Memory allocation strategy
    pub memory_strategy: MemoryStrategy,
    /// Compute precision
    pub precision: ComputePrecision,
    /// Batch size for inference
    pub batch_size: usize,
    /// Number of parallel streams
    pub stream_count: usize,
    /// Enable tensor core operations
    pub use_tensor_cores: bool,
    /// Memory pool size in MB
    pub memory_pool_mb: u32,
    /// Power efficiency mode
    pub power_mode: PowerMode,
}

/// Memory allocation strategy
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MemoryStrategy {
    /// Pre-allocate all memory
    PreAllocate,
    /// Dynamic allocation as needed
    Dynamic,
    /// Conservative with fallback
    Conservative,
    /// Unified memory (for integrated GPUs)
    Unified,
}

/// Compute precision mode
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ComputePrecision {
    /// FP32 full precision
    FP32,
    /// FP16 half precision
    FP16,
    /// FP8 quarter precision (Blackwell/RDNA4)
    FP8,
    /// INT8 quantized
    INT8,
    /// Mixed precision
    Mixed,
}

/// Power efficiency mode
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PowerMode {
    /// Maximum performance
    Performance,
    /// Balanced performance/power
    Balanced,
    /// Power saving mode
    PowerSaver,
}

/// GPU optimizer
pub struct GpuOptimizer {
    gpu_info: GpuInfo,
    settings: GpuOptimizationSettings,
}

impl GpuOptimizer {
    /// Create optimizer for detected GPU
    pub fn new(gpu_info: GpuInfo) -> Self {
        let settings = Self::get_optimal_settings(&gpu_info);
        Self { gpu_info, settings }
    }

    /// Get optimal settings based on GPU architecture
    fn get_optimal_settings(gpu_info: &GpuInfo) -> GpuOptimizationSettings {
        match gpu_info.architecture {
            // NVIDIA RTX 4000 series (Ada Lovelace)
            GpuArchitecture::NvidiaAdaLovelace => GpuOptimizationSettings {
                memory_strategy: MemoryStrategy::PreAllocate,
                precision: if gpu_info.supports_fp8 {
                    ComputePrecision::FP8
                } else {
                    ComputePrecision::FP16
                },
                batch_size: 16,
                stream_count: 4,
                use_tensor_cores: true,
                memory_pool_mb: (gpu_info.vram_mb as f32 * 0.8) as u32,
                power_mode: PowerMode::Performance,
            },

            // NVIDIA RTX 3000 series (Ampere)
            GpuArchitecture::NvidiaAmpere => GpuOptimizationSettings {
                memory_strategy: MemoryStrategy::PreAllocate,
                precision: ComputePrecision::FP16,
                batch_size: 12,
                stream_count: 3,
                use_tensor_cores: true,
                memory_pool_mb: (gpu_info.vram_mb as f32 * 0.75) as u32,
                power_mode: PowerMode::Balanced,
            },

            // NVIDIA RTX 2000 series (Turing)
            GpuArchitecture::NvidiaTuring => GpuOptimizationSettings {
                memory_strategy: MemoryStrategy::Dynamic,
                precision: ComputePrecision::Mixed,
                batch_size: 8,
                stream_count: 2,
                use_tensor_cores: gpu_info.supports_tensor_cores,
                memory_pool_mb: (gpu_info.vram_mb as f32 * 0.7) as u32,
                power_mode: PowerMode::Balanced,
            },

            // AMD RX 7000 series (RDNA3)
            GpuArchitecture::AmdRdna3 => GpuOptimizationSettings {
                memory_strategy: MemoryStrategy::PreAllocate,
                precision: ComputePrecision::FP16,
                batch_size: 12,
                stream_count: 3,
                use_tensor_cores: false,
                memory_pool_mb: (gpu_info.vram_mb as f32 * 0.75) as u32,
                power_mode: PowerMode::Balanced,
            },

            // AMD RX 6000 series (RDNA2)
            GpuArchitecture::AmdRdna2 => GpuOptimizationSettings {
                memory_strategy: MemoryStrategy::Dynamic,
                precision: ComputePrecision::FP16,
                batch_size: 8,
                stream_count: 2,
                use_tensor_cores: false,
                memory_pool_mb: (gpu_info.vram_mb as f32 * 0.7) as u32,
                power_mode: PowerMode::Balanced,
            },

            // AMD RX 5000 series (RDNA1)
            GpuArchitecture::AmdRdna1 => GpuOptimizationSettings {
                memory_strategy: MemoryStrategy::Conservative,
                precision: ComputePrecision::FP32,
                batch_size: 4,
                stream_count: 1,
                use_tensor_cores: false,
                memory_pool_mb: (gpu_info.vram_mb as f32 * 0.6) as u32,
                power_mode: PowerMode::PowerSaver,
            },

            // Intel Arc A-series (Xe-HPG)
            GpuArchitecture::IntelXeHpg => GpuOptimizationSettings {
                memory_strategy: MemoryStrategy::Dynamic,
                precision: ComputePrecision::FP16,
                batch_size: 8,
                stream_count: 2,
                use_tensor_cores: gpu_info.supports_tensor_cores,
                memory_pool_mb: (gpu_info.vram_mb as f32 * 0.7) as u32,
                power_mode: PowerMode::Balanced,
            },

            // Future architectures
            GpuArchitecture::NvidiaBlackwell | GpuArchitecture::AmdRdna4 | GpuArchitecture::IntelXe2 => {
                GpuOptimizationSettings {
                    memory_strategy: MemoryStrategy::PreAllocate,
                    precision: ComputePrecision::FP8,
                    batch_size: 24,
                    stream_count: 6,
                    use_tensor_cores: true,
                    memory_pool_mb: (gpu_info.vram_mb as f32 * 0.85) as u32,
                    power_mode: PowerMode::Performance,
                }
            }

            // Unknown/legacy
            GpuArchitecture::Unknown => GpuOptimizationSettings {
                memory_strategy: MemoryStrategy::Conservative,
                precision: ComputePrecision::FP32,
                batch_size: 2,
                stream_count: 1,
                use_tensor_cores: false,
                memory_pool_mb: 1024, // 1GB default
                power_mode: PowerMode::PowerSaver,
            },
        }
    }

    /// Optimize for specific workload
    pub fn optimize_for_workload(&mut self, workload: WorkloadType) {
        match workload {
            WorkloadType::RealTimeTranslation => {
                // Optimize for low latency
                self.settings.batch_size = self.settings.batch_size.min(4);
                self.settings.stream_count = self.settings.stream_count.max(2);
                self.settings.precision = match self.gpu_info.architecture {
                    GpuArchitecture::NvidiaAdaLovelace | GpuArchitecture::NvidiaBlackwell
                        if self.gpu_info.supports_fp8 => ComputePrecision::FP8,
                    _ if self.gpu_info.supports_fp16 => ComputePrecision::FP16,
                    _ => ComputePrecision::FP32,
                };
            }
            WorkloadType::BatchProcessing => {
                // Optimize for throughput
                self.settings.batch_size = self.settings.batch_size.max(16);
                self.settings.stream_count = 1;
            }
            WorkloadType::PowerEfficient => {
                // Optimize for power efficiency
                self.settings.power_mode = PowerMode::PowerSaver;
                self.settings.batch_size = self.settings.batch_size / 2;
                self.settings.memory_pool_mb = (self.settings.memory_pool_mb as f32 * 0.7) as u32;
            }
        }
    }

    /// Get memory allocation size for model
    pub fn get_model_memory_allocation(&self, model_size_mb: u32) -> Result<u32> {
        let available_memory = self.settings.memory_pool_mb;

        // Check if model fits in available memory
        if model_size_mb > available_memory {
            return Err(anyhow::anyhow!(
                "Model size {}MB exceeds available GPU memory {}MB",
                model_size_mb,
                available_memory
            ));
        }

        // Calculate allocation based on precision
        let precision_multiplier = match self.settings.precision {
            ComputePrecision::FP32 => 1.0,
            ComputePrecision::FP16 => 0.5,
            ComputePrecision::FP8 => 0.25,
            ComputePrecision::INT8 => 0.25,
            ComputePrecision::Mixed => 0.75,
        };

        let allocated_size = (model_size_mb as f32 * precision_multiplier) as u32;
        Ok(allocated_size)
    }

    /// Check if GPU supports specific optimization
    pub fn supports_optimization(&self, optimization: &str) -> bool {
        match optimization {
            "tensor_cores" => self.gpu_info.supports_tensor_cores,
            "fp16" => self.gpu_info.supports_fp16,
            "fp8" => self.gpu_info.supports_fp8,
            "int8" => self.gpu_info.supports_int8,
            "ray_tracing" => self.gpu_info.supports_ray_tracing,
            "multi_stream" => self.settings.stream_count > 1,
            _ => false,
        }
    }

    /// Get optimization recommendations
    pub fn get_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        // VRAM recommendations
        if self.gpu_info.vram_mb < 4096 {
            recommendations.push("Consider upgrading to GPU with at least 4GB VRAM for better performance".to_string());
        }

        // Architecture-specific recommendations
        match self.gpu_info.architecture {
            GpuArchitecture::NvidiaTuring => {
                recommendations.push("Enable DLSS for improved performance in supported applications".to_string());
            }
            GpuArchitecture::NvidiaAmpere | GpuArchitecture::NvidiaAdaLovelace => {
                recommendations.push("Utilize tensor cores for AI workloads with FP16 precision".to_string());
            }
            GpuArchitecture::AmdRdna2 | GpuArchitecture::AmdRdna3 => {
                recommendations.push("Enable Smart Access Memory if supported by CPU".to_string());
            }
            GpuArchitecture::IntelXeHpg => {
                recommendations.push("Update to latest Intel GPU drivers for optimal performance".to_string());
            }
            _ => {}
        }

        // Power recommendations
        if self.settings.power_mode == PowerMode::PowerSaver {
            recommendations.push("Consider Balanced or Performance mode for better results".to_string());
        }

        recommendations
    }
}

/// Workload type for optimization
#[derive(Debug, Clone, Copy)]
pub enum WorkloadType {
    /// Real-time translation workload
    RealTimeTranslation,
    /// Batch processing workload
    BatchProcessing,
    /// Power-efficient workload
    PowerEfficient,
}

/// GPU memory manager
pub struct GpuMemoryManager {
    total_memory: u32,
    allocated_memory: u32,
    memory_pools: Vec<MemoryPool>,
}

/// Memory pool for efficient allocation
#[derive(Debug, Clone)]
struct MemoryPool {
    name: String,
    size_mb: u32,
    used_mb: u32,
}

impl GpuMemoryManager {
    /// Create new memory manager
    pub fn new(total_memory_mb: u32) -> Self {
        Self {
            total_memory: total_memory_mb,
            allocated_memory: 0,
            memory_pools: Vec::new(),
        }
    }

    /// Allocate memory pool
    pub fn allocate_pool(&mut self, name: String, size_mb: u32) -> Result<()> {
        if self.allocated_memory + size_mb > self.total_memory {
            return Err(anyhow::anyhow!(
                "Insufficient GPU memory: requested {}MB, available {}MB",
                size_mb,
                self.total_memory - self.allocated_memory
            ));
        }

        self.memory_pools.push(MemoryPool {
            name,
            size_mb,
            used_mb: 0,
        });
        self.allocated_memory += size_mb;

        Ok(())
    }

    /// Get memory usage statistics
    pub fn get_usage_stats(&self) -> (u32, u32, f32) {
        let used = self.allocated_memory;
        let total = self.total_memory;
        let usage_percent = (used as f32 / total as f32) * 100.0;

        (used, total, usage_percent)
    }
}