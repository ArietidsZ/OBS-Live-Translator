//! Hardware Detection and Optimization Module
//!
//! Adaptive optimization for hardware from 2020-2025 including:
//! - NVIDIA RTX 2000/3000/4000 series
//! - AMD RX 5000/6000/7000 series
//! - Intel Arc A-series
//! - Intel 10th-14th gen CPUs
//! - AMD Ryzen 3000-7000 series

#[path = "gpu-optimizer.rs"]
pub mod gpu_optimizer;
#[path = "cpu-optimizer.rs"]
pub mod cpu_optimizer;
pub mod compatibility;
pub mod fallback;

pub use gpu_optimizer::*;
pub use cpu_optimizer::*;
pub use compatibility::*;
pub use fallback::*;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Hardware generation enumeration (2020-2025)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HardwareGeneration {
    /// 2020 hardware
    Gen2020,
    /// 2021 hardware
    Gen2021,
    /// 2022 hardware
    Gen2022,
    /// 2023 hardware
    Gen2023,
    /// 2024 hardware
    Gen2024,
    /// 2025 hardware
    Gen2025,
    /// Unknown/older hardware
    Legacy,
}

/// GPU vendor and architecture
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuArchitecture {
    /// NVIDIA architectures
    NvidiaTuring,       // RTX 2000 series (2018-2020)
    NvidiaAmpere,       // RTX 3000 series (2020-2022)
    NvidiaAdaLovelace,  // RTX 4000 series (2022-2024)
    NvidiaBlackwell,    // RTX 5000 series (2024-2025)

    /// AMD architectures
    AmdRdna1,           // RX 5000 series (2019-2020)
    AmdRdna2,           // RX 6000 series (2020-2022)
    AmdRdna3,           // RX 7000 series (2022-2024)
    AmdRdna4,           // RX 8000 series (2024-2025)

    /// Intel architectures
    IntelXeHpg,         // Arc Alchemist (2022-2023)
    IntelXe2,           // Arc Battlemage (2024-2025)

    /// Fallback
    Unknown,
}

/// CPU vendor and architecture
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CpuArchitecture {
    /// Intel architectures
    IntelCometLake,     // 10th gen (2020)
    IntelRocketLake,    // 11th gen (2021)
    IntelAlderLake,     // 12th gen (2021-2022)
    IntelRaptorLake,    // 13th gen (2022-2023)
    IntelMeteorLake,    // 14th gen (2023-2024)
    IntelArrowLake,     // 15th gen (2024-2025)

    /// AMD architectures
    AmdZen2,            // Ryzen 3000 (2019-2020)
    AmdZen3,            // Ryzen 5000 (2020-2022)
    AmdZen4,            // Ryzen 7000 (2022-2024)
    AmdZen5,            // Ryzen 9000 (2024-2025)

    /// ARM architectures
    AppleM1,            // M1/M1 Pro/Max (2020-2021)
    AppleM2,            // M2/M2 Pro/Max (2022-2023)
    AppleM3,            // M3/M3 Pro/Max (2023-2024)
    AppleM4,            // M4/M4 Pro/Max (2024-2025)

    /// Fallback
    Unknown,
}

/// SIMD instruction support
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimdSupport {
    pub sse2: bool,
    pub sse3: bool,
    pub ssse3: bool,
    pub sse41: bool,
    pub sse42: bool,
    pub avx: bool,
    pub avx2: bool,
    pub avx512: bool,
    pub fma: bool,
    pub neon: bool,     // ARM NEON
    pub sve: bool,      // ARM SVE
}

/// Hardware capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareCapabilities {
    /// CPU information
    pub cpu: CpuInfo,
    /// GPU information
    pub gpu: Option<GpuInfo>,
    /// Memory information
    pub memory: MemoryInfo,
    /// SIMD support
    pub simd: SimdSupport,
    /// Hardware generation
    pub generation: HardwareGeneration,
}

/// CPU information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuInfo {
    pub vendor: String,
    pub model: String,
    pub architecture: CpuArchitecture,
    pub core_count: usize,
    pub thread_count: usize,
    pub base_frequency_mhz: u32,
    pub boost_frequency_mhz: Option<u32>,
    pub cache_size_kb: u32,
    pub supports_hyperthreading: bool,
}

/// GPU information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub vendor: String,
    pub model: String,
    pub architecture: GpuArchitecture,
    pub vram_mb: u32,
    pub compute_units: u32,
    pub clock_speed_mhz: u32,
    pub memory_bandwidth_gbps: f32,
    pub supports_ray_tracing: bool,
    pub supports_tensor_cores: bool,
    pub supports_fp16: bool,
    pub supports_fp8: bool,
    pub supports_int8: bool,
}

/// Memory information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInfo {
    pub total_ram_mb: u32,
    pub available_ram_mb: u32,
    pub memory_speed_mhz: Option<u32>,
    pub memory_type: Option<String>, // DDR4, DDR5, LPDDR5, etc.
}

/// Hardware detector
pub struct HardwareDetector;

impl HardwareDetector {
    /// Detect system hardware capabilities
    pub async fn detect() -> Result<HardwareCapabilities> {
        let cpu = Self::detect_cpu()?;
        let gpu = Self::detect_gpu().await.ok();
        let memory = Self::detect_memory()?;
        let simd = Self::detect_simd_support();
        let generation = Self::determine_generation(&cpu, &gpu);

        Ok(HardwareCapabilities {
            cpu,
            gpu,
            memory,
            simd,
            generation,
        })
    }

    /// Detect CPU information
    fn detect_cpu() -> Result<CpuInfo> {
        let core_count = num_cpus::get_physical();
        let thread_count = num_cpus::get();

        // Use cpuid for detailed CPU detection
        let cpuid = raw_cpuid::CpuId::new();

        let vendor = cpuid
            .get_vendor_info()
            .map(|v| v.as_str().to_string())
            .unwrap_or_else(|| "Unknown".to_string());

        let model = cpuid
            .get_processor_brand_string()
            .map(|b| b.as_str().to_string())
            .unwrap_or_else(|| "Unknown CPU".to_string());

        let architecture = Self::detect_cpu_architecture(&vendor, &model);

        // Get frequency information
        let (base_freq, boost_freq) = if let Some(freq_info) = cpuid.get_processor_frequency_info() {
            (
                freq_info.processor_base_frequency() as u32,
                Some(freq_info.processor_max_frequency() as u32),
            )
        } else {
            (2000, Some(4000)) // Default values
        };

        Ok(CpuInfo {
            vendor,
            model,
            architecture,
            core_count,
            thread_count,
            base_frequency_mhz: base_freq,
            boost_frequency_mhz: boost_freq,
            cache_size_kb: 8192, // Default 8MB L3 cache
            supports_hyperthreading: thread_count > core_count,
        })
    }

    /// Detect CPU architecture from model string
    fn detect_cpu_architecture(vendor: &str, model: &str) -> CpuArchitecture {
        let model_lower = model.to_lowercase();

        if vendor.contains("Intel") {
            if model_lower.contains("14th gen") || model_lower.contains("meteor") {
                CpuArchitecture::IntelMeteorLake
            } else if model_lower.contains("13th gen") || model_lower.contains("raptor") {
                CpuArchitecture::IntelRaptorLake
            } else if model_lower.contains("12th gen") || model_lower.contains("alder") {
                CpuArchitecture::IntelAlderLake
            } else if model_lower.contains("11th gen") || model_lower.contains("rocket") {
                CpuArchitecture::IntelRocketLake
            } else if model_lower.contains("10th gen") || model_lower.contains("comet") {
                CpuArchitecture::IntelCometLake
            } else {
                CpuArchitecture::Unknown
            }
        } else if vendor.contains("AMD") {
            if model_lower.contains("7000") || model_lower.contains("zen 4") {
                CpuArchitecture::AmdZen4
            } else if model_lower.contains("5000") || model_lower.contains("zen 3") {
                CpuArchitecture::AmdZen3
            } else if model_lower.contains("3000") || model_lower.contains("zen 2") {
                CpuArchitecture::AmdZen2
            } else {
                CpuArchitecture::Unknown
            }
        } else if vendor.contains("Apple") || cfg!(target_os = "macos") {
            if model_lower.contains("m3") {
                CpuArchitecture::AppleM3
            } else if model_lower.contains("m2") {
                CpuArchitecture::AppleM2
            } else if model_lower.contains("m1") {
                CpuArchitecture::AppleM1
            } else {
                CpuArchitecture::Unknown
            }
        } else {
            CpuArchitecture::Unknown
        }
    }

    /// Detect GPU information
    async fn detect_gpu() -> Result<GpuInfo> {
        // This would use platform-specific APIs or libraries
        // For now, return a placeholder

        // Try to detect NVIDIA GPU
        #[cfg(feature = "nvidia-acceleration")]
        if let Ok(info) = Self::detect_nvidia_gpu().await {
            return Ok(info);
        }

        // Try to detect AMD GPU
        #[cfg(feature = "amd-acceleration")]
        if let Ok(info) = Self::detect_amd_gpu().await {
            return Ok(info);
        }

        // Try to detect Intel GPU
        #[cfg(feature = "intel-acceleration")]
        if let Ok(info) = Self::detect_intel_gpu().await {
            return Ok(info);
        }

        Err(anyhow::anyhow!("No compatible GPU detected"))
    }

    /// Detect NVIDIA GPU
    #[cfg(feature = "nvidia-acceleration")]
    async fn detect_nvidia_gpu() -> Result<GpuInfo> {
        // Use NVML or similar API to detect NVIDIA GPU
        // This is a simplified version
        Ok(GpuInfo {
            vendor: "NVIDIA".to_string(),
            model: "RTX 3080".to_string(), // Placeholder
            architecture: GpuArchitecture::NvidiaAmpere,
            vram_mb: 10240,
            compute_units: 68,
            clock_speed_mhz: 1710,
            memory_bandwidth_gbps: 760.0,
            supports_ray_tracing: true,
            supports_tensor_cores: true,
            supports_fp16: true,
            supports_fp8: false,
            supports_int8: true,
        })
    }

    /// Detect AMD GPU
    #[cfg(feature = "amd-acceleration")]
    async fn detect_amd_gpu() -> Result<GpuInfo> {
        // Use ROCm or similar API to detect AMD GPU
        Ok(GpuInfo {
            vendor: "AMD".to_string(),
            model: "RX 6800 XT".to_string(), // Placeholder
            architecture: GpuArchitecture::AmdRdna2,
            vram_mb: 16384,
            compute_units: 72,
            clock_speed_mhz: 2250,
            memory_bandwidth_gbps: 512.0,
            supports_ray_tracing: true,
            supports_tensor_cores: false,
            supports_fp16: true,
            supports_fp8: false,
            supports_int8: true,
        })
    }

    /// Detect Intel GPU
    #[cfg(feature = "intel-acceleration")]
    async fn detect_intel_gpu() -> Result<GpuInfo> {
        // Use Level Zero or similar API to detect Intel GPU
        Ok(GpuInfo {
            vendor: "Intel".to_string(),
            model: "Arc A750".to_string(), // Placeholder
            architecture: GpuArchitecture::IntelXeHpg,
            vram_mb: 8192,
            compute_units: 28,
            clock_speed_mhz: 2400,
            memory_bandwidth_gbps: 512.0,
            supports_ray_tracing: true,
            supports_tensor_cores: true,
            supports_fp16: true,
            supports_fp8: false,
            supports_int8: true,
        })
    }

    /// Detect system memory
    fn detect_memory() -> Result<MemoryInfo> {
        let sys = sysinfo::System::new_all();

        Ok(MemoryInfo {
            total_ram_mb: (sys.total_memory() / 1024 / 1024) as u32,
            available_ram_mb: (sys.available_memory() / 1024 / 1024) as u32,
            memory_speed_mhz: None, // Would require platform-specific detection
            memory_type: None,
        })
    }

    /// Detect SIMD instruction support
    fn detect_simd_support() -> SimdSupport {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            let cpuid = raw_cpuid::CpuId::new();
            let features = cpuid.get_feature_info();

            SimdSupport {
                sse2: features.as_ref().map_or(false, |f| f.has_sse2()),
                sse3: features.as_ref().map_or(false, |f| f.has_sse3()),
                ssse3: features.as_ref().map_or(false, |f| f.has_ssse3()),
                sse41: features.as_ref().map_or(false, |f| f.has_sse41()),
                sse42: features.as_ref().map_or(false, |f| f.has_sse42()),
                avx: features.as_ref().map_or(false, |f| f.has_avx()),
                avx2: cpuid.get_extended_feature_info()
                    .as_ref()
                    .map_or(false, |f| f.has_avx2()),
                avx512: cpuid.get_extended_feature_info()
                    .as_ref()
                    .map_or(false, |f| f.has_avx512f()),
                fma: features.as_ref().map_or(false, |f| f.has_fma()),
                neon: false,
                sve: false,
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            SimdSupport {
                sse2: false,
                sse3: false,
                ssse3: false,
                sse41: false,
                sse42: false,
                avx: false,
                avx2: false,
                avx512: false,
                fma: false,
                neon: true,  // ARM64 always has NEON
                sve: false,  // Would need runtime detection
            }
        }

        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        {
            SimdSupport {
                sse2: false,
                sse3: false,
                ssse3: false,
                sse41: false,
                sse42: false,
                avx: false,
                avx2: false,
                avx512: false,
                fma: false,
                neon: false,
                sve: false,
            }
        }
    }

    /// Determine hardware generation
    fn determine_generation(cpu: &CpuInfo, gpu: &Option<GpuInfo>) -> HardwareGeneration {
        // Check GPU generation first (if available)
        if let Some(gpu_info) = gpu {
            match gpu_info.architecture {
                GpuArchitecture::NvidiaBlackwell | GpuArchitecture::AmdRdna4 | GpuArchitecture::IntelXe2 => {
                    return HardwareGeneration::Gen2025;
                }
                GpuArchitecture::NvidiaAdaLovelace | GpuArchitecture::AmdRdna3 => {
                    return HardwareGeneration::Gen2023;
                }
                GpuArchitecture::NvidiaAmpere | GpuArchitecture::AmdRdna2 | GpuArchitecture::IntelXeHpg => {
                    return HardwareGeneration::Gen2021;
                }
                GpuArchitecture::NvidiaTuring | GpuArchitecture::AmdRdna1 => {
                    return HardwareGeneration::Gen2020;
                }
                _ => {}
            }
        }

        // Fall back to CPU generation
        match cpu.architecture {
            CpuArchitecture::IntelArrowLake | CpuArchitecture::AmdZen5 | CpuArchitecture::AppleM4 => {
                HardwareGeneration::Gen2025
            }
            CpuArchitecture::IntelMeteorLake | CpuArchitecture::AmdZen4 | CpuArchitecture::AppleM3 => {
                HardwareGeneration::Gen2024
            }
            CpuArchitecture::IntelRaptorLake | CpuArchitecture::AppleM2 => {
                HardwareGeneration::Gen2023
            }
            CpuArchitecture::IntelAlderLake | CpuArchitecture::AmdZen3 => {
                HardwareGeneration::Gen2022
            }
            CpuArchitecture::IntelRocketLake | CpuArchitecture::AppleM1 => {
                HardwareGeneration::Gen2021
            }
            CpuArchitecture::IntelCometLake | CpuArchitecture::AmdZen2 => {
                HardwareGeneration::Gen2020
            }
            _ => HardwareGeneration::Legacy,
        }
    }
}

/// Optimization level based on hardware
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel {
    /// Maximum performance for latest hardware
    Aggressive,
    /// Balanced performance for recent hardware
    Balanced,
    /// Conservative for older hardware
    Conservative,
    /// Fallback for legacy hardware
    Compatible,
}

impl HardwareCapabilities {
    /// Get recommended optimization level
    pub fn get_optimization_level(&self) -> OptimizationLevel {
        match self.generation {
            HardwareGeneration::Gen2024 | HardwareGeneration::Gen2025 => {
                OptimizationLevel::Aggressive
            }
            HardwareGeneration::Gen2022 | HardwareGeneration::Gen2023 => {
                OptimizationLevel::Balanced
            }
            HardwareGeneration::Gen2020 | HardwareGeneration::Gen2021 => {
                OptimizationLevel::Conservative
            }
            HardwareGeneration::Legacy => {
                OptimizationLevel::Compatible
            }
        }
    }

    /// Check if hardware supports a specific feature
    pub fn supports_feature(&self, feature: &str) -> bool {
        match feature {
            "avx2" => self.simd.avx2,
            "avx512" => self.simd.avx512,
            "fp16" => self.gpu.as_ref().map_or(false, |g| g.supports_fp16),
            "fp8" => self.gpu.as_ref().map_or(false, |g| g.supports_fp8),
            "tensor_cores" => self.gpu.as_ref().map_or(false, |g| g.supports_tensor_cores),
            "ray_tracing" => self.gpu.as_ref().map_or(false, |g| g.supports_ray_tracing),
            _ => false,
        }
    }
}