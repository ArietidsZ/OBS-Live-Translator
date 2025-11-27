// Platform detection and hardware capability reporting
// Based on Part 6 & 7 research findings

use anyhow::{anyhow, Result};
use sysinfo::System;

/// Detected platform type with hardware details
#[derive(Debug, Clone)]
pub enum Platform {
    NvidiaGpu(NvidiaInfo),
    AppleSilicon(AppleInfo),
    IntelCpu(IntelInfo),
    AmdGpu(AmdInfo),
    GenericCpu(CpuInfo),
}

/// NVIDIA GPU information
#[derive(Debug, Clone)]
pub struct NvidiaInfo {
    pub gpu_model: String,
    pub compute_capability: (u32, u32),
    pub vram_gb: usize,
    pub cuda_version: Option<String>,
    pub supports_fp8: bool,  // Hopper/Blackwell (compute capability >= 8.9)
    pub supports_int4: bool, // Ampere and newer (compute capability >= 8.0)
}

/// Apple Silicon information
#[derive(Debug, Clone)]
pub struct AppleInfo {
    pub chip_model: String, // M4, M4 Pro, M4 Max, etc.
    pub neural_engine_tops: u32,
    pub unified_memory_gb: usize,
    pub memory_bandwidth_gbps: u32,
}

/// Intel CPU/iGPU information
#[derive(Debug, Clone)]
pub struct IntelInfo {
    pub cpu_model: String,
    pub has_npu: bool, // Core Ultra 200V series
    pub npu_tops: Option<u32>,
    pub cores: usize,
    pub threads: usize,
}

/// AMD GPU information
#[derive(Debug, Clone)]
pub struct AmdInfo {
    pub gpu_model: String,
    pub vram_gb: usize,
    pub is_rdna4: bool, // RX 9000 series
    pub supports_fp8: bool,
}

/// Generic CPU information
#[derive(Debug, Clone)]
pub struct CpuInfo {
    pub model: String,
    pub cores: usize,
    pub threads: usize,
    pub total_memory_gb: usize,
}

impl Platform {
    /// Detect the current platform and hardware capabilities
    pub fn detect() -> Result<Self> {
        // Try NVIDIA GPU first (highest priority for inference)
        if let Ok(nvidia_info) = Self::detect_nvidia() {
            return Ok(Platform::NvidiaGpu(nvidia_info));
        }

        // Try Apple Silicon (macOS only)
        #[cfg(target_os = "macos")]
        if let Ok(apple_info) = Self::detect_apple_silicon() {
            return Ok(Platform::AppleSilicon(apple_info));
        }

        // Try AMD GPU (Windows primarily)
        #[cfg(target_os = "windows")]
        if let Ok(amd_info) = Self::detect_amd_gpu() {
            return Ok(Platform::AmdGpu(amd_info));
        }

        // Try Intel with NPU (Core Ultra)
        if let Ok(intel_info) = Self::detect_intel_npu() {
            return Ok(Platform::IntelCpu(intel_info));
        }

        // Fallback to generic CPU
        Ok(Platform::GenericCpu(Self::detect_cpu()))
    }

    /// Detect NVIDIA GPU using nvidia-smi or CUDA runtime
    fn detect_nvidia() -> Result<NvidiaInfo> {
        // Try using nvidia-smi command
        let output = std::process::Command::new("nvidia-smi")
            .arg("--query-gpu=name,compute_cap,memory.total")
            .arg("--format=csv,noheader,nounits")
            .output();

        if let Ok(output) = output {
            if output.status.success() {
                let output_str = String::from_utf8_lossy(&output.stdout);
                let parts: Vec<&str> = output_str.trim().split(',').collect();

                if parts.len() >= 3 {
                    let gpu_model = parts[0].trim().to_string();
                    let vram_mb = parts[2].trim().parse::<f64>().unwrap_or(0.0);
                    let vram_gb = (vram_mb / 1024.0).ceil() as usize;

                    // Parse compute capability (e.g., "8.9" -> (8, 9))
                    let compute_cap_str = parts[1].trim();
                    let compute_cap: Vec<u32> = compute_cap_str
                        .split('.')
                        .filter_map(|s| s.parse().ok())
                        .collect();
                    let compute_capability = if compute_cap.len() >= 2 {
                        (compute_cap[0], compute_cap[1])
                    } else {
                        (0, 0)
                    };

                    // Detect capabilities based on compute capability
                    // Part 7 research: RTX 5090 (Blackwell) = 9.0, supports FP4
                    // Hopper = 8.9-9.0, Ampere = 8.0-8.6
                    let supports_fp8 = compute_capability.0 >= 9
                        || (compute_capability.0 == 8 && compute_capability.1 >= 9);
                    let supports_int4 = compute_capability.0 >= 8;

                    return Ok(NvidiaInfo {
                        gpu_model,
                        compute_capability,
                        vram_gb,
                        cuda_version: None, // Could query with nvcc --version
                        supports_fp8,
                        supports_int4,
                    });
                }
            }
        }

        Err(anyhow!("NVIDIA GPU not detected"))
    }

    /// Detect Apple Silicon (macOS only)
    #[cfg(target_os = "macos")]
    fn detect_apple_silicon() -> Result<AppleInfo> {
        // Use sysctl to get chip information
        let output = std::process::Command::new("sysctl")
            .arg("-n")
            .arg("machdep.cpu.brand_string")
            .output();

        if let Ok(output) = output {
            if output.status.success() {
                let brand_string = String::from_utf8_lossy(&output.stdout);

                // Check if it's Apple Silicon (contains "Apple")
                if brand_string.contains("Apple") {
                    // Detect specific chip model
                    let chip_model = if brand_string.contains("M4") {
                        if brand_string.contains("Max") {
                            "M4 Max".to_string()
                        } else if brand_string.contains("Pro") {
                            "M4 Pro".to_string()
                        } else {
                            "M4".to_string()
                        }
                    } else {
                        "Apple Silicon".to_string()
                    };

                    // Part 7 research: M4 Neural Engine specs
                    let (neural_engine_tops, memory_bandwidth_gbps) = match chip_model.as_str() {
                        "M4 Max" => (38, 546), // M4 Max: 38 TOPS, 546 GB/s
                        "M4 Pro" => (38, 273), // M4 Pro: 38 TOPS, 273 GB/s
                        "M4" => (38, 120),     // M4: 38 TOPS, 120 GB/s
                        _ => (38, 120),        // Default estimate
                    };

                    // Get unified memory size
                    let mut sys = System::new_all();
                    sys.refresh_all();
                    let unified_memory_gb = (sys.total_memory() / (1024 * 1024 * 1024)) as usize;

                    return Ok(AppleInfo {
                        chip_model,
                        neural_engine_tops,
                        unified_memory_gb,
                        memory_bandwidth_gbps,
                    });
                }
            }
        }

        Err(anyhow!("Apple Silicon not detected"))
    }

    /// Detect AMD GPU (primarily Windows)
    #[cfg(target_os = "windows")]
    fn detect_amd_gpu() -> Result<AmdInfo> {
        // Try using DirectX diagnostic tool or AMD specific tools
        // For now, simplified detection

        // This would require Windows API calls or parsing wmic output
        // Placeholder implementation
        Err(anyhow!("AMD GPU detection not yet implemented"))
    }

    /// Detect Intel CPU with NPU (Core Ultra series)
    fn detect_intel_npu() -> Result<IntelInfo> {
        let mut sys = System::new_all();
        sys.refresh_all();

        // Get CPU info - in sysinfo 0.32, use cpus() and get first CPU's brand
        let cpu_brand = if let Some(cpu) = sys.cpus().first() {
            cpu.brand().to_string()
        } else {
            "Unknown CPU".to_string()
        };
        let cores = sys.cpus().len();
        let threads = num_cpus::get();

        // Check if it's Intel Core Ultra with NPU
        // Part 7 research: Core Ultra 200V = 120 TOPS NPU
        let has_npu = cpu_brand.contains("Ultra") && cpu_brand.contains("200V");
        let npu_tops = if has_npu { Some(120) } else { None };

        if has_npu {
            return Ok(IntelInfo {
                cpu_model: cpu_brand,
                has_npu,
                npu_tops,
                cores,
                threads,
            });
        }

        Err(anyhow!("Intel NPU not detected"))
    }

    /// Detect generic CPU capabilities
    fn detect_cpu() -> CpuInfo {
        let mut sys = System::new_all();
        sys.refresh_all();

        let model = if let Some(cpu) = sys.cpus().first() {
            cpu.brand().to_string()
        } else {
            "Unknown CPU".to_string()
        };
        let cores = sys.cpus().len();
        let threads = num_cpus::get();
        let total_memory_gb = (sys.total_memory() / (1024 * 1024 * 1024)) as usize;

        CpuInfo {
            model,
            cores,
            threads,
            total_memory_gb,
        }
    }

    /// Get a human-readable description of the platform
    pub fn description(&self) -> String {
        match self {
            Platform::NvidiaGpu(info) => {
                format!(
                    "NVIDIA {} ({}.{}, {}GB VRAM, FP8: {}, INT4: {})",
                    info.gpu_model,
                    info.compute_capability.0,
                    info.compute_capability.1,
                    info.vram_gb,
                    info.supports_fp8,
                    info.supports_int4
                )
            }
            Platform::AppleSilicon(info) => {
                format!(
                    "{} ({} TOPS Neural Engine, {}GB Unified Memory, {} GB/s bandwidth)",
                    info.chip_model,
                    info.neural_engine_tops,
                    info.unified_memory_gb,
                    info.memory_bandwidth_gbps
                )
            }
            Platform::IntelCpu(info) => {
                let npu_info = if let Some(tops) = info.npu_tops {
                    format!(", {tops} TOPS NPU")
                } else {
                    String::new()
                };
                format!(
                    "{} ({} cores, {} threads{})",
                    info.cpu_model, info.cores, info.threads, npu_info
                )
            }
            Platform::AmdGpu(info) => {
                format!(
                    "AMD {} ({}GB VRAM, RDNA4: {}, FP8: {})",
                    info.gpu_model, info.vram_gb, info.is_rdna4, info.supports_fp8
                )
            }
            Platform::GenericCpu(info) => {
                format!(
                    "{} ({} cores, {} threads, {}GB RAM)",
                    info.model, info.cores, info.threads, info.total_memory_gb
                )
            }
        }
    }

    /// Get recommended execution provider based on platform
    pub fn recommended_execution_provider(&self) -> &'static str {
        match self {
            Platform::NvidiaGpu(_) => "tensorrt", // Part 6: TensorRT for NVIDIA
            Platform::AppleSilicon(_) => "coreml", // Part 6: CoreML for Apple
            Platform::IntelCpu(info) if info.has_npu => "openvino", // Part 6: OpenVINO for Intel NPU
            Platform::AmdGpu(_) => "directml", // Part 6: DirectML for AMD on Windows
            Platform::GenericCpu(_) => "cpu",  // Fallback to CPU
            _ => "cpu",
        }
    }

    /// Check if platform supports specific quantization format
    pub fn supports_quantization(&self, format: &str) -> bool {
        match (self, format) {
            // FP8 support (Part 9 research)
            (Platform::NvidiaGpu(info), "fp8") => info.supports_fp8,
            (Platform::AmdGpu(info), "fp8") => info.supports_fp8,

            // INT4 support
            (Platform::NvidiaGpu(info), "int4") => info.supports_int4,
            (Platform::AppleSilicon(_), "int4") => true, // CoreML supports INT4

            // INT8 support (universal)
            (_, "int8") => true,

            // FP16 support (universal)
            (_, "fp16") => true,

            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_platform_detection() {
        let platform = Platform::detect();
        assert!(platform.is_ok(), "Platform detection should always succeed");

        if let Ok(p) = platform {
            println!("Detected platform: {}", p.description());
            println!("Recommended EP: {}", p.recommended_execution_provider());
        }
    }

    #[test]
    fn test_quantization_support() {
        let platform = Platform::detect().unwrap();

        // INT8 should be universally supported
        assert!(platform.supports_quantization("int8"));

        // FP16 should be universally supported
        assert!(platform.supports_quantization("fp16"));
    }
}
