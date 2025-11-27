//! Performance profile management
//! Enhanced in Part 1.6 to use Platform detection and ExecutionProvider selection

pub use crate::types::Profile;

use crate::execution_provider::ExecutionProviderConfig;
use crate::platform_detect::Platform;
use anyhow::Result;

/// Profile detection based on system resources
/// Enhanced in Part 1.6 to use Part 1.4 (Platform) and Part 1.5 (ExecutionProvider)
pub struct ProfileDetector;

#[allow(dead_code)]
impl ProfileDetector {
    /// Auto-detect the best profile for current hardware
    /// Enhanced to use comprehensive platform detection
    pub fn detect() -> Result<Profile> {
        // Use Part 1.4 platform detection
        let platform = Platform::detect()?;
        let ep_config = ExecutionProviderConfig::from_platform(platform.clone());

        tracing::info!(
            "Platform detection: {}, EP: {}, Recommended Quantization: {}",
            platform.description(),
            ep_config.provider().name(),
            ep_config.recommended_quantization()
        );

        // Determine profile based on platform capabilities
        let profile = Self::select_profile_for_platform(&platform, &ep_config);

        tracing::info!(
            "Selected {} profile for detected hardware",
            match profile {
                Profile::Low => "LOW",
                Profile::Medium => "MEDIUM",
                Profile::High => "HIGH",
            }
        );

        Ok(profile)
    }

    /// Select optimal profile based on platform and execution provider
    fn select_profile_for_platform(
        platform: &Platform,
        _ep_config: &ExecutionProviderConfig,
    ) -> Profile {
        match platform {
            // NVIDIA GPUs
            Platform::NvidiaGpu(info) => {
                // Part 7: RTX 5090 = High, RTX 4090 = High, RTX 3090 = Medium
                if info.vram_gb >= 16 && info.supports_fp8 {
                    // Blackwell/Hopper (RTX 5090, H100) - FP8 support
                    Profile::High
                } else if info.vram_gb >= 12 {
                    // Ampere/Ada (RTX 4090, RTX 3090)
                    Profile::High
                } else if info.vram_gb >= 6 {
                    // Mid-range (RTX 3060, RTX 4060)
                    Profile::Medium
                } else {
                    // Low VRAM GPUs
                    Profile::Low
                }
            }

            // Apple Silicon
            Platform::AppleSilicon(info) => {
                // Part 7.2: M4 Max = High, M4 Pro = Medium, M4 = Medium
                if info.chip_model.contains("Max") && info.unified_memory_gb >= 32 {
                    Profile::High // M4 Max 32GB+
                } else if info.chip_model.contains("Pro") || info.chip_model.contains("Max") || info.unified_memory_gb >= 16 {
                    Profile::Medium // M4 Pro/Max or M4 with 16GB+
                } else {
                    Profile::Low // M4 base
                }
            }

            // Intel with NPU
            Platform::IntelCpu(info) => {
                // Part 7.3: Core Ultra with NPU = Medium, without = Low
                if info.has_npu && info.cores >= 12 {
                    Profile::Medium // Core Ultra 200V with good CPU
                } else {
                    Profile::Low // Regular Intel CPUs
                }
            }

            // AMD GPUs
            Platform::AmdGpu(info) => {
                // Part 7.4: RDNA4 with good VRAM = Medium/High
                if info.vram_gb >= 12 && info.is_rdna4 {
                    Profile::High // High-end RDNA4
                } else if info.vram_gb >= 8 {
                    Profile::Medium // Mid-range AMD
                } else {
                    Profile::Low
                }
            }

            // Generic CPU fallback
            Platform::GenericCpu(info) => {
                // CPU-only: use Low profile
                // Exception: very beefy CPUs (16+ cores, 32GB+ RAM) can do Medium
                if info.cores >= 16 && info.total_memory_gb >= 32 {
                    Profile::Medium // Workstation-class CPUs
                } else {
                    Profile::Low // Default CPU-only
                }
            }
        }
    }

    /// Get detailed profile recommendation with reasoning
    pub fn recommend_with_reasoning() -> Result<(Profile, String)> {
        let platform = Platform::detect()?;
        let ep_config = ExecutionProviderConfig::from_platform(platform.clone());
        let profile = Self::select_profile_for_platform(&platform, &ep_config);

        let reasoning = format!(
            "Platform: {}\nExecution Provider: {}\nQuantization: {}\n\nRecommended Profile: {:?}\n\nReasoning: {}",
            platform.description(),
            ep_config.provider().name(),
            ep_config.recommended_quantization(),
            profile,
            Self::get_profile_reasoning(&platform, profile)
        );

        Ok((profile, reasoning))
    }

    fn get_profile_reasoning(platform: &Platform, profile: Profile) -> String {
        match (platform, profile) {
            (Platform::NvidiaGpu(info), Profile::High) => {
                format!("GPU has {}GB VRAM and supports advanced features (FP8: {}, INT4: {}). Can handle high-quality models with low latency.",
                    info.vram_gb, info.supports_fp8, info.supports_int4)
            }
            (Platform::NvidiaGpu(info), Profile::Medium) => {
                format!(
                    "GPU has {}GB VRAM, sufficient for quantized models with good performance.",
                    info.vram_gb
                )
            }
            (Platform::AppleSilicon(info), Profile::High) => {
                format!("{} with {}GB unified memory and {}GB/s bandwidth. Neural Engine provides excellent acceleration.",
                    info.chip_model, info.unified_memory_gb, info.memory_bandwidth_gbps)
            }
            (Platform::AppleSilicon(info), Profile::Medium) => {
                format!(
                    "{} with {}GB unified memory. Good performance with quantized models.",
                    info.chip_model, info.unified_memory_gb
                )
            }
            (Platform::IntelCpu(info), Profile::Medium) if info.has_npu => {
                format!(
                    "Intel {} with {} TOPS NPU. OpenVINO can leverage NPU acceleration.",
                    info.cpu_model,
                    info.npu_tops.unwrap_or(0)
                )
            }
            (Platform::GenericCpu(_), Profile::Low) => {
                "CPU-only system. Using heavily quantized models for acceptable performance."
                    .to_string()
            }
            _ => "Based on detected hardware capabilities.".to_string(),
        }
    }

    /// Legacy method (kept for backward compatibility)
    fn get_total_memory() -> Result<u64> {
        use sysinfo::System;
        let mut sys = System::new_all();
        sys.refresh_all();
        Ok(sys.total_memory())
    }

    /// Legacy method (kept for backward compatibility)  
    fn has_gpu() -> bool {
        // Use platform detection now
        if let Ok(platform) = Platform::detect() {
            matches!(
                platform,
                Platform::NvidiaGpu(_) | Platform::AppleSilicon(_) | Platform::AmdGpu(_)
            )
        } else {
            false
        }
    }

    /// Legacy method (kept for backward compatibility)
    fn get_gpu_memory() -> Result<u64> {
        // Use platform detection now
        let platform = Platform::detect()?;

        let memory_bytes = match platform {
            Platform::NvidiaGpu(info) => (info.vram_gb as u64) * 1024 * 1024 * 1024,
            Platform::AppleSilicon(info) => {
                (info.unified_memory_gb as u64) * 1024 * 1024 * 1024 / 2
            }
            Platform::AmdGpu(info) => (info.vram_gb as u64) * 1024 * 1024 * 1024,
            _ => 0,
        };

        Ok(memory_bytes)
    }
}

/// Profile-specific resource limits
#[derive(Debug, Clone)]
pub struct ProfileLimits {
    pub max_memory_mb: usize,
    pub max_vram_mb: usize,
    pub target_latency_ms: u64,
    pub max_batch_size: usize,
}

impl ProfileLimits {
    pub fn for_profile(profile: Profile) -> Self {
        match profile {
            Profile::Low => Self {
                max_memory_mb: 1200,
                max_vram_mb: 0,
                target_latency_ms: 500,
                max_batch_size: 1,
            },
            Profile::Medium => Self {
                max_memory_mb: 800,
                max_vram_mb: 3200,
                target_latency_ms: 300,
                max_batch_size: 2,
            },
            Profile::High => Self {
                max_memory_mb: 1500,
                max_vram_mb: 7000,
                target_latency_ms: 150,
                max_batch_size: 4,
            },
        }
    }
}
