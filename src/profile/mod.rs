//! Dynamic Profile System for OBS Live Translator
//!
//! Provides automatic hardware detection and optimal profile selection
//! for CPU-only, GPU-accelerated, and high-performance configurations.

pub mod components;
pub mod monitor;
pub mod manager;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Performance profiles for different hardware configurations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Profile {
    /// Low profile: 4GB RAM, 2 cores, CPU-only processing
    Low,
    /// Medium profile: 4GB RAM, 2GB VRAM, 2 cores, balanced GPU acceleration
    Medium,
    /// High profile: 8GB RAM, 8GB VRAM, 6 cores, maximum performance
    High,
}

impl Profile {
    /// Get profile name as string
    pub fn as_str(&self) -> &'static str {
        match self {
            Profile::Low => "low",
            Profile::Medium => "medium",
            Profile::High => "high",
        }
    }

    /// Upgrade to next profile if possible
    pub fn upgrade(&self) -> Profile {
        match self {
            Profile::Low => Profile::Medium,
            Profile::Medium => Profile::High,
            Profile::High => Profile::High,
        }
    }

    /// Downgrade to previous profile if possible
    pub fn downgrade(&self) -> Profile {
        match self {
            Profile::Low => Profile::Low,
            Profile::Medium => Profile::Low,
            Profile::High => Profile::Medium,
        }
    }

    /// Get resource constraints for this profile
    pub fn resource_constraints(&self) -> ResourceConstraints {
        match self {
            Profile::Low => ResourceConstraints {
                max_ram_mb: 4096,
                max_vram_mb: 0,
                cpu_cores: 2,
                target_latency_ms: 500,
                max_cpu_percent: 90.0,
                max_ram_percent: 85.0,
                max_vram_percent: 0.0,
            },
            Profile::Medium => ResourceConstraints {
                max_ram_mb: 4096,
                max_vram_mb: 2048,
                cpu_cores: 2,
                target_latency_ms: 400,
                max_cpu_percent: 80.0,
                max_ram_percent: 85.0,
                max_vram_percent: 90.0,
            },
            Profile::High => ResourceConstraints {
                max_ram_mb: 8192,
                max_vram_mb: 8192,
                cpu_cores: 6,
                target_latency_ms: 300,
                max_cpu_percent: 75.0,
                max_ram_percent: 80.0,
                max_vram_percent: 85.0,
            },
        }
    }
}

/// Resource constraints for a profile
#[derive(Debug, Clone)]
pub struct ResourceConstraints {
    pub max_ram_mb: u64,
    pub max_vram_mb: u64,
    pub cpu_cores: u32,
    pub target_latency_ms: u64,
    pub max_cpu_percent: f32,
    pub max_ram_percent: f32,
    pub max_vram_percent: f32,
}

/// Hardware capabilities detected from the system
#[derive(Debug, Clone)]
pub struct HardwareInfo {
    pub cpu_cores: u32,
    pub cpu_threads: u32,
    pub cpu_base_freq_mhz: u32,
    pub cpu_model: String,
    pub total_ram_mb: u64,
    pub available_ram_mb: u64,
    pub gpu_info: Vec<GpuInfo>,
    pub simd_support: SimdSupport,
}

/// GPU information
#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub name: String,
    pub vendor: GpuVendor,
    pub total_vram_mb: u64,
    pub available_vram_mb: u64,
    pub compute_capability: Option<String>,
    pub driver_version: Option<String>,
}

/// GPU vendor enumeration
#[derive(Debug, Clone, PartialEq)]
pub enum GpuVendor {
    Nvidia,
    Amd,
    Intel,
    Apple,
    Unknown,
}

/// SIMD instruction set support
#[derive(Debug, Clone)]
pub struct SimdSupport {
    pub sse2: bool,
    pub avx: bool,
    pub avx2: bool,
    pub avx512: bool,
    pub neon: bool,
}

/// Hardware detection and profile selection
pub struct ProfileDetector;

impl ProfileDetector {
    /// Detect optimal profile based on available hardware
    pub fn detect_optimal_profile() -> Result<Profile> {
        let hardware = Self::scan_hardware()?;
        Ok(Self::determine_optimal_profile(&hardware))
    }

    /// Scan system hardware capabilities
    pub fn scan_hardware() -> Result<HardwareInfo> {
        let cpu_info = Self::detect_cpu_info()?;
        let memory_info = Self::detect_memory_info()?;
        let gpu_info = Self::detect_gpu_info()?;
        let simd_support = Self::detect_simd_support();

        Ok(HardwareInfo {
            cpu_cores: cpu_info.0,
            cpu_threads: cpu_info.1,
            cpu_base_freq_mhz: cpu_info.2,
            cpu_model: cpu_info.3,
            total_ram_mb: memory_info.0,
            available_ram_mb: memory_info.1,
            gpu_info,
            simd_support,
        })
    }

    /// Determine optimal profile from hardware capabilities
    pub fn determine_optimal_profile(hardware: &HardwareInfo) -> Profile {
        // Check for high profile requirements
        if hardware.cpu_cores >= 6
            && hardware.total_ram_mb >= 8192
            && hardware.gpu_info.iter().any(|gpu| gpu.total_vram_mb >= 8192)
        {
            return Profile::High;
        }

        // Check for medium profile requirements
        if hardware.cpu_cores >= 2
            && hardware.total_ram_mb >= 4096
            && hardware.gpu_info.iter().any(|gpu| gpu.total_vram_mb >= 2048)
        {
            return Profile::Medium;
        }

        // Default to low profile
        Profile::Low
    }

    /// Validate if current hardware can support the given profile
    pub fn validate_profile_support(profile: Profile, hardware: &HardwareInfo) -> bool {
        let constraints = profile.resource_constraints();

        // Check CPU cores
        if hardware.cpu_cores < constraints.cpu_cores {
            return false;
        }

        // Check RAM
        if hardware.total_ram_mb < constraints.max_ram_mb {
            return false;
        }

        // Check VRAM for GPU profiles
        if constraints.max_vram_mb > 0 {
            if !hardware.gpu_info.iter().any(|gpu| gpu.total_vram_mb >= constraints.max_vram_mb) {
                return false;
            }
        }

        true
    }

    /// Detect CPU information
    fn detect_cpu_info() -> Result<(u32, u32, u32, String)> {
        let cores = num_cpus::get_physical() as u32;
        let threads = num_cpus::get() as u32;

        // Try to get CPU model and frequency
        let (freq, model) = Self::get_cpu_details();

        Ok((cores, threads, freq, model))
    }

    /// Detect memory information
    fn detect_memory_info() -> Result<(u64, u64)> {
        #[cfg(target_os = "windows")]
        {
            use windows::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

            let mut mem_status = MEMORYSTATUSEX {
                dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32,
                ..Default::default()
            };

            unsafe {
                if GlobalMemoryStatusEx(&mut mem_status).is_ok() {
                    let total_mb = mem_status.ullTotalPhys / 1024 / 1024;
                    let available_mb = mem_status.ullAvailPhys / 1024 / 1024;
                    return Ok((total_mb, available_mb));
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            use std::process::Command;

            // Get total memory
            if let Ok(output) = Command::new("sysctl")
                .args(&["-n", "hw.memsize"])
                .output()
            {
                if let Ok(mem_str) = String::from_utf8(output.stdout) {
                    if let Ok(total_bytes) = mem_str.trim().parse::<u64>() {
                        let total_mb = total_bytes / 1024 / 1024;

                        // Estimate available memory (rough approximation)
                        let available_mb = total_mb * 80 / 100; // Assume 80% available
                        return Ok((total_mb, available_mb));
                    }
                }
            }
        }

        #[cfg(target_os = "linux")]
        {
            use std::fs;

            if let Ok(meminfo) = fs::read_to_string("/proc/meminfo") {
                let mut total_kb = 0u64;
                let mut available_kb = 0u64;

                for line in meminfo.lines() {
                    if line.starts_with("MemTotal:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            total_kb = kb_str.parse().unwrap_or(0);
                        }
                    } else if line.starts_with("MemAvailable:") {
                        if let Some(kb_str) = line.split_whitespace().nth(1) {
                            available_kb = kb_str.parse().unwrap_or(0);
                        }
                    }
                }

                if total_kb > 0 {
                    return Ok((total_kb / 1024, available_kb / 1024));
                }
            }
        }

        // Fallback: estimate based on number of cores
        let estimated_mb = num_cpus::get() as u64 * 2048; // 2GB per core estimate
        Ok((estimated_mb, estimated_mb * 80 / 100))
    }

    /// Detect GPU information
    fn detect_gpu_info() -> Result<Vec<GpuInfo>> {
        let mut gpus = Vec::new();

        // Try NVIDIA first
        if let Ok(nvidia_gpus) = Self::detect_nvidia_gpus() {
            gpus.extend(nvidia_gpus);
        }

        // Try AMD
        if let Ok(amd_gpus) = Self::detect_amd_gpus() {
            gpus.extend(amd_gpus);
        }

        // Try Intel/Apple
        if let Ok(other_gpus) = Self::detect_other_gpus() {
            gpus.extend(other_gpus);
        }

        Ok(gpus)
    }

    /// Detect NVIDIA GPUs
    fn detect_nvidia_gpus() -> Result<Vec<GpuInfo>> {
        use std::process::Command;

        let output = Command::new("nvidia-smi")
            .args(&["--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader,nounits"])
            .output();

        if let Ok(output) = output {
            if output.status.success() {
                let output_str = String::from_utf8_lossy(&output.stdout);
                let mut gpus = Vec::new();

                for line in output_str.lines() {
                    let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                    if parts.len() >= 4 {
                        let total_mb = parts[1].parse().unwrap_or(0);
                        let free_mb = parts[2].parse().unwrap_or(0);

                        gpus.push(GpuInfo {
                            name: parts[0].to_string(),
                            vendor: GpuVendor::Nvidia,
                            total_vram_mb: total_mb,
                            available_vram_mb: free_mb,
                            compute_capability: None, // Could be detected separately
                            driver_version: Some(parts[3].to_string()),
                        });
                    }
                }

                return Ok(gpus);
            }
        }

        Ok(Vec::new())
    }

    /// Detect AMD GPUs
    fn detect_amd_gpus() -> Result<Vec<GpuInfo>> {
        // AMD GPU detection would go here
        // For now, return empty vector
        Ok(Vec::new())
    }

    /// Detect other GPUs (Intel, Apple)
    fn detect_other_gpus() -> Result<Vec<GpuInfo>> {
        #[cfg(target_os = "macos")]
        {
            // Check for Apple Silicon
            use std::process::Command;

            if let Ok(output) = Command::new("sysctl")
                .args(&["-n", "machdep.cpu.brand_string"])
                .output()
            {
                if let Ok(cpu_str) = String::from_utf8(output.stdout) {
                    if cpu_str.contains("Apple") {
                        // Apple Silicon has unified memory
                        if let Ok((total_mb, available_mb)) = Self::detect_memory_info() {
                            return Ok(vec![GpuInfo {
                                name: "Apple Silicon GPU".to_string(),
                                vendor: GpuVendor::Apple,
                                total_vram_mb: total_mb / 2, // Assume half of unified memory for GPU
                                available_vram_mb: available_mb / 2,
                                compute_capability: Some("Apple Neural Engine".to_string()),
                                driver_version: None,
                            }]);
                        }
                    }
                }
            }
        }

        Ok(Vec::new())
    }

    /// Detect SIMD instruction set support
    fn detect_simd_support() -> SimdSupport {
        SimdSupport {
            sse2: is_x86_feature_detected!("sse2"),
            avx: is_x86_feature_detected!("avx"),
            avx2: is_x86_feature_detected!("avx2"),
            avx512: is_x86_feature_detected!("avx512f"),
            neon: cfg!(target_arch = "aarch64"),
        }
    }

    /// Get detailed CPU information
    fn get_cpu_details() -> (u32, String) {
        #[cfg(target_os = "windows")]
        {
            use std::process::Command;

            if let Ok(output) = Command::new("wmic")
                .args(&["cpu", "get", "Name,MaxClockSpeed", "/format:csv"])
                .output()
            {
                if let Ok(output_str) = String::from_utf8(output.stdout) {
                    for line in output_str.lines().skip(1) {
                        let parts: Vec<&str> = line.split(',').collect();
                        if parts.len() >= 3 && !parts[1].trim().is_empty() {
                            let freq = parts[1].trim().parse().unwrap_or(3000);
                            let model = parts[2].trim().to_string();
                            return (freq, model);
                        }
                    }
                }
            }
        }

        #[cfg(target_os = "macos")]
        {
            use std::process::Command;

            let mut freq = 3000u32;
            let mut model = "Unknown CPU".to_string();

            // Get CPU model
            if let Ok(output) = Command::new("sysctl")
                .args(&["-n", "machdep.cpu.brand_string"])
                .output()
            {
                if let Ok(model_str) = String::from_utf8(output.stdout) {
                    model = model_str.trim().to_string();
                }
            }

            // Get CPU frequency (rough estimate for Apple Silicon)
            if model.contains("Apple") {
                freq = if model.contains("M1") { 3200 }
                      else if model.contains("M2") { 3500 }
                      else if model.contains("M3") { 4000 }
                      else if model.contains("M4") { 4300 }
                      else { 3000 };
            }

            return (freq, model);
        }

        #[cfg(all(target_os = "linux", not(target_os = "macos")))]
        {
            use std::fs;

            if let Ok(cpuinfo) = fs::read_to_string("/proc/cpuinfo") {
                let mut freq = 3000u32;
                let mut model = "Unknown CPU".to_string();

                for line in cpuinfo.lines() {
                    if line.starts_with("model name") {
                        if let Some(name) = line.split(':').nth(1) {
                            model = name.trim().to_string();
                        }
                    } else if line.starts_with("cpu MHz") {
                        if let Some(freq_str) = line.split(':').nth(1) {
                            if let Ok(freq_f) = freq_str.trim().parse::<f32>() {
                                freq = freq_f as u32;
                            }
                        }
                    }
                }

                return (freq, model);
            }
        }

        (3000, "Unknown CPU".to_string())
    }
}

/// Performance benchmarking for profile validation
pub struct ProfileBenchmark;

impl ProfileBenchmark {
    /// Run a quick benchmark to validate profile performance
    pub fn validate_profile_performance(profile: Profile) -> Result<BenchmarkResult> {
        let start = Instant::now();

        // CPU benchmark: Simple mathematical operations
        let cpu_score = Self::benchmark_cpu();

        // Memory benchmark: Allocation and access patterns
        let memory_score = Self::benchmark_memory();

        // GPU benchmark (if applicable)
        let gpu_score = if profile != Profile::Low {
            Self::benchmark_gpu().unwrap_or(0.0)
        } else {
            0.0
        };

        let total_time = start.elapsed();

        Ok(BenchmarkResult {
            profile,
            cpu_score,
            memory_score,
            gpu_score,
            total_time,
            can_meet_targets: Self::evaluate_performance(profile, cpu_score, memory_score, gpu_score),
        })
    }

    /// Benchmark CPU performance
    fn benchmark_cpu() -> f32 {
        let start = Instant::now();
        let mut _result = 0.0f32;

        // Simple computational workload
        for i in 0..1_000_000 {
            _result += (i as f32).sin().cos();
        }

        let elapsed = start.elapsed().as_secs_f32();
        1000.0 / elapsed // Higher is better
    }

    /// Benchmark memory performance
    fn benchmark_memory() -> f32 {
        let start = Instant::now();

        // Allocate and access memory patterns similar to audio processing
        let mut buffers: Vec<Vec<f32>> = Vec::new();

        for _ in 0..100 {
            let mut buffer = vec![0.0f32; 16384]; // 16K samples

            // Simulate audio processing patterns
            for i in 0..buffer.len() {
                buffer[i] = (i as f32 * 0.001).sin();
            }

            buffers.push(buffer);
        }

        // Access patterns
        let mut _sum = 0.0f32;
        for buffer in &buffers {
            _sum += buffer.iter().sum::<f32>();
        }

        let elapsed = start.elapsed().as_secs_f32();
        1000.0 / elapsed // Higher is better
    }

    /// Benchmark GPU performance (placeholder)
    fn benchmark_gpu() -> Result<f32> {
        // GPU benchmark would require actual GPU libraries
        // For now, return a placeholder score
        Ok(100.0)
    }

    /// Evaluate if performance meets profile targets
    fn evaluate_performance(profile: Profile, cpu_score: f32, memory_score: f32, gpu_score: f32) -> bool {
        let min_scores = match profile {
            Profile::Low => (50.0, 30.0, 0.0),
            Profile::Medium => (80.0, 50.0, 50.0),
            Profile::High => (120.0, 80.0, 100.0),
        };

        cpu_score >= min_scores.0 &&
        memory_score >= min_scores.1 &&
        (profile == Profile::Low || gpu_score >= min_scores.2)
    }
}

/// Benchmark results
#[derive(Debug)]
pub struct BenchmarkResult {
    pub profile: Profile,
    pub cpu_score: f32,
    pub memory_score: f32,
    pub gpu_score: f32,
    pub total_time: Duration,
    pub can_meet_targets: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profile_upgrade_downgrade() {
        assert_eq!(Profile::Low.upgrade(), Profile::Medium);
        assert_eq!(Profile::Medium.upgrade(), Profile::High);
        assert_eq!(Profile::High.upgrade(), Profile::High);

        assert_eq!(Profile::Low.downgrade(), Profile::Low);
        assert_eq!(Profile::Medium.downgrade(), Profile::Low);
        assert_eq!(Profile::High.downgrade(), Profile::Medium);
    }

    #[test]
    fn test_hardware_detection() {
        let hardware = ProfileDetector::scan_hardware().unwrap();

        assert!(hardware.cpu_cores > 0);
        assert!(hardware.cpu_threads >= hardware.cpu_cores);
        assert!(hardware.total_ram_mb > 0);

        println!("Detected hardware: {:#?}", hardware);
    }

    #[test]
    fn test_profile_detection() {
        let profile = ProfileDetector::detect_optimal_profile().unwrap();
        println!("Optimal profile: {:?}", profile);

        // Should detect a valid profile
        assert!(matches!(profile, Profile::Low | Profile::Medium | Profile::High));
    }

    #[test]
    fn test_benchmark() {
        let result = ProfileBenchmark::validate_profile_performance(Profile::Low).unwrap();
        println!("Benchmark result: {:#?}", result);

        assert!(result.cpu_score > 0.0);
        assert!(result.memory_score > 0.0);
    }
}