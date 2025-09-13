use std::process::Command;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub vendor: String,
    pub name: String,
    pub gpu_type: GpuType,
    pub tflops: f32,
    pub memory_mb: u32,
    pub recommended_profile: String,
    pub expected_latency_ms: (u32, u32),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum GpuType {
    Discrete,
    Integrated,
    Apple,
    Unknown,
}

pub struct GpuDetector;

impl GpuDetector {
    /// Detect GPU and return information about capabilities
    pub fn detect() -> GpuInfo {
        // Try platform-specific detection
        #[cfg(target_os = "macos")]
        {
            if let Some(info) = Self::detect_apple_silicon() {
                return info;
            }
        }

        #[cfg(target_os = "linux")]
        {
            if let Some(info) = Self::detect_linux_gpu() {
                return info;
            }
        }

        #[cfg(target_os = "windows")]
        {
            if let Some(info) = Self::detect_windows_gpu() {
                return info;
            }
        }

        // Fallback to unknown
        GpuInfo {
            vendor: "Unknown".to_string(),
            name: "Unknown GPU".to_string(),
            gpu_type: GpuType::Unknown,
            tflops: 0.0,
            memory_mb: 0,
            recommended_profile: "cpu_only".to_string(),
            expected_latency_ms: (1000, 2000),
        }
    }

    #[cfg(target_os = "macos")]
    fn detect_apple_silicon() -> Option<GpuInfo> {
        // Check if running on Apple Silicon
        let output = Command::new("sysctl")
            .args(&["-n", "machdep.cpu.brand_string"])
            .output()
            .ok()?;

        let cpu_info = String::from_utf8_lossy(&output.stdout);

        if !cpu_info.contains("Apple") {
            return None;
        }

        // Detect specific Apple Silicon model
        let (name, tflops, profile) = if cpu_info.contains("M3") {
            if cpu_info.contains("Max") {
                ("Apple M3 Max GPU", 14.2, "mps")
            } else if cpu_info.contains("Pro") {
                ("Apple M3 Pro GPU", 7.0, "mps")
            } else {
                ("Apple M3 GPU", 4.5, "mps")
            }
        } else if cpu_info.contains("M2") {
            if cpu_info.contains("Ultra") {
                ("Apple M2 Ultra GPU", 27.2, "mps")
            } else if cpu_info.contains("Max") {
                ("Apple M2 Max GPU", 13.6, "mps")
            } else if cpu_info.contains("Pro") {
                ("Apple M2 Pro GPU", 6.8, "mps")
            } else {
                ("Apple M2 GPU", 3.6, "mps")
            }
        } else if cpu_info.contains("M1") {
            if cpu_info.contains("Ultra") {
                ("Apple M1 Ultra GPU", 21.0, "mps")
            } else if cpu_info.contains("Max") {
                ("Apple M1 Max GPU", 10.4, "mps")
            } else if cpu_info.contains("Pro") {
                ("Apple M1 Pro GPU", 5.2, "mps")
            } else {
                ("Apple M1 GPU", 2.6, "mps")
            }
        } else {
            ("Apple Silicon GPU", 3.0, "mps")
        };

        // Estimate latency based on performance
        let expected_latency = match tflops {
            t if t >= 10.0 => (50, 80),
            t if t >= 7.0 => (70, 100),
            t if t >= 5.0 => (80, 120),
            t if t >= 3.5 => (100, 150),
            _ => (120, 180),
        };

        // Get system memory for estimation
        let memory_mb = Self::get_system_memory_mb() / 3; // Conservative estimate

        Some(GpuInfo {
            vendor: "Apple".to_string(),
            name: name.to_string(),
            gpu_type: GpuType::Apple,
            tflops,
            memory_mb,
            recommended_profile: profile.to_string(),
            expected_latency_ms: expected_latency,
        })
    }

    #[cfg(target_os = "linux")]
    fn detect_linux_gpu() -> Option<GpuInfo> {
        // Try lspci first
        let output = Command::new("lspci")
            .output()
            .ok()?;

        let lspci_output = String::from_utf8_lossy(&output.stdout);

        for line in lspci_output.lines() {
            if line.contains("VGA") || line.contains("Display") || line.contains("3D") {
                let line_lower = line.to_lowercase();

                // Check for Intel integrated GPU
                if line_lower.contains("intel") {
                    return Some(Self::parse_intel_gpu(&line_lower));
                }

                // Check for AMD integrated GPU
                if line_lower.contains("amd") && line_lower.contains("radeon") {
                    // Check if it's an APU model
                    if line_lower.contains("780m") || line_lower.contains("760m") ||
                       line_lower.contains("680m") || line_lower.contains("660m") {
                        return Some(Self::parse_amd_apu(&line_lower));
                    }
                }

                // Check for NVIDIA discrete GPU
                if line_lower.contains("nvidia") {
                    // Try nvidia-smi for more details
                    if let Some(info) = Self::detect_nvidia_gpu() {
                        return Some(info);
                    }
                }
            }
        }

        None
    }

    fn parse_intel_gpu(line: &str) -> GpuInfo {
        let (name, tflops, memory_mb, profile) = if line.contains("arc") {
            ("Intel Arc Graphics", 4.6, 4096, "igpu_intel")
        } else if line.contains("iris xe") {
            if line.contains("96eu") {
                ("Intel Iris Xe 96EU", 2.4, 3072, "igpu_intel")
            } else {
                ("Intel Iris Xe 80EU", 2.0, 2048, "igpu_intel")
            }
        } else if line.contains("uhd") && (line.contains("770") || line.contains("750")) {
            ("Intel UHD 770", 1.5, 2048, "igpu_intel")
        } else {
            ("Intel HD Graphics", 0.8, 1024, "cpu_only")
        };

        let expected_latency = match tflops {
            t if t >= 4.0 => (220, 300),
            t if t >= 2.0 => (400, 550),
            t if t >= 1.0 => (600, 800),
            _ => (800, 1200),
        };

        GpuInfo {
            vendor: "Intel".to_string(),
            name: name.to_string(),
            gpu_type: GpuType::Integrated,
            tflops,
            memory_mb,
            recommended_profile: profile.to_string(),
            expected_latency_ms: expected_latency,
        }
    }

    fn parse_amd_apu(line: &str) -> GpuInfo {
        let (name, tflops, memory_mb, profile) = if line.contains("780m") {
            ("AMD Radeon 780M", 8.9, 4096, "igpu_amd")
        } else if line.contains("760m") {
            ("AMD Radeon 760M", 4.3, 3072, "igpu_amd")
        } else if line.contains("680m") {
            ("AMD Radeon 680M", 3.4, 3072, "igpu_amd")
        } else if line.contains("660m") {
            ("AMD Radeon 660M", 1.8, 2048, "igpu_amd")
        } else {
            ("AMD Radeon Graphics", 2.0, 2048, "igpu_amd")
        };

        let expected_latency = match tflops {
            t if t >= 8.0 => (180, 250),
            t if t >= 4.0 => (280, 350),
            t if t >= 3.0 => (350, 450),
            _ => (500, 650),
        };

        GpuInfo {
            vendor: "AMD".to_string(),
            name: name.to_string(),
            gpu_type: GpuType::Integrated,
            tflops,
            memory_mb,
            recommended_profile: profile.to_string(),
            expected_latency_ms: expected_latency,
        }
    }

    fn detect_nvidia_gpu() -> Option<GpuInfo> {
        // Try nvidia-smi for detailed info
        let output = Command::new("nvidia-smi")
            .args(&["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
            .output()
            .ok()?;

        let info = String::from_utf8_lossy(&output.stdout);
        let parts: Vec<&str> = info.trim().split(',').collect();

        if parts.len() >= 2 {
            let name = parts[0].trim();
            let memory_mb: u32 = parts[1].trim().parse().unwrap_or(0);

            // Estimate TFLOPS based on model name
            let (tflops, profile, latency) = Self::estimate_nvidia_performance(name);

            return Some(GpuInfo {
                vendor: "NVIDIA".to_string(),
                name: name.to_string(),
                gpu_type: GpuType::Discrete,
                tflops,
                memory_mb,
                recommended_profile: profile.to_string(),
                expected_latency_ms: latency,
            });
        }

        None
    }

    fn estimate_nvidia_performance(model: &str) -> (f32, &str, (u32, u32)) {
        // RTX 40 series
        if model.contains("4090") {
            (82.6, "vram_24gb_ultimate", (45, 60))
        } else if model.contains("4080") {
            (48.7, "vram_24gb_ultimate", (50, 65))
        } else if model.contains("4070") && model.contains("Ti") {
            (40.1, "vram_6gb_plus", (55, 70))
        } else if model.contains("4070") {
            (29.1, "vram_6gb_plus", (70, 85))
        } else if model.contains("4060") && model.contains("Ti") {
            (22.0, "vram_6gb_plus", (90, 110))
        } else if model.contains("4060") {
            (15.0, "vram_6gb_plus", (110, 140))
        }
        // RTX 30 series
        else if model.contains("3090") {
            (35.6, "vram_24gb_ultimate", (60, 75))
        } else if model.contains("3080") {
            (29.8, "vram_6gb_plus", (70, 90))
        } else if model.contains("3070") {
            (20.3, "vram_6gb_plus", (85, 105))
        } else if model.contains("3060") {
            (13.0, "vram_6gb_plus", (120, 150))
        }
        // Default for unknown NVIDIA GPU
        else {
            (10.0, "vram_4gb", (150, 200))
        }
    }

    #[cfg(target_os = "windows")]
    fn detect_windows_gpu() -> Option<GpuInfo> {
        // On Windows, try WMI or DirectX diagnostics
        // This is a simplified version - full implementation would use Windows APIs

        // Try to detect through DirectML or DXGI
        None // Placeholder - would need Windows-specific implementation
    }

    fn get_system_memory_mb() -> u32 {
        // Try to get system memory
        #[cfg(target_os = "macos")]
        {
            if let Ok(output) = Command::new("sysctl")
                .args(&["-n", "hw.memsize"])
                .output()
            {
                let mem_bytes = String::from_utf8_lossy(&output.stdout)
                    .trim()
                    .parse::<u64>()
                    .unwrap_or(0);
                return (mem_bytes / (1024 * 1024)) as u32;
            }
        }

        #[cfg(target_os = "linux")]
        {
            if let Ok(output) = Command::new("free")
                .args(&["-m"])
                .output()
            {
                let mem_info = String::from_utf8_lossy(&output.stdout);
                for line in mem_info.lines() {
                    if line.starts_with("Mem:") {
                        let parts: Vec<&str> = line.split_whitespace().collect();
                        if parts.len() > 1 {
                            return parts[1].parse().unwrap_or(8192);
                        }
                    }
                }
            }
        }

        // Default fallback
        8192
    }

    /// Get recommended configuration based on detected GPU
    pub fn get_recommended_config(info: &GpuInfo) -> String {
        format!(
            r#"# Auto-detected GPU Configuration
# GPU: {} ({})
# Performance: {} TFLOPS
# Expected Latency: {}-{}ms

[gpu]
vendor = "{}"
model = "{}"
memory_mb = {}
gpu_type = "{:?}"

[profile]
config_file = "config/profiles/{}.toml"

[performance]
expected_latency_ms = [{}, {}]
realtime_capable = {}
"#,
            info.name,
            info.vendor,
            info.tflops,
            info.expected_latency_ms.0,
            info.expected_latency_ms.1,
            info.vendor,
            info.name,
            info.memory_mb,
            info.gpu_type,
            info.recommended_profile,
            info.expected_latency_ms.0,
            info.expected_latency_ms.1,
            info.expected_latency_ms.1 < 500
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_detection() {
        let info = GpuDetector::detect();
        println!("Detected GPU: {:?}", info);

        assert!(!info.vendor.is_empty());
        assert!(!info.name.is_empty());
    }

    #[test]
    fn test_config_generation() {
        let info = GpuDetector::detect();
        let config = GpuDetector::get_recommended_config(&info);

        println!("Recommended config:\n{}", config);

        assert!(config.contains(&info.vendor));
        assert!(config.contains(&info.recommended_profile));
    }
}