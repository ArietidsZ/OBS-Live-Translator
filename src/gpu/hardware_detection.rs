//! Hardware detection and VRAM monitoring for cross-platform GPU acceleration

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tracing::{debug, error, info, warn};

use super::adaptive_memory::GpuVendor;

/// Hardware detection and monitoring system
pub struct HardwareDetector {
    /// Detected GPU information
    gpu_info: GpuInfo,
    /// Real-time VRAM monitoring
    vram_monitor: Arc<VramMonitor>,
    /// System capabilities
    capabilities: SystemCapabilities,
}

impl HardwareDetector {
    /// Initialize hardware detection
    pub async fn new() -> Result<Self> {
        info!("Starting hardware detection");
        
        let gpu_info = Self::detect_gpu_hardware().await?;
        let vram_monitor = Arc::new(VramMonitor::new(&gpu_info).await?);
        let capabilities = Self::detect_system_capabilities().await?;
        
        info!("Hardware detection completed:");
        info!("  GPU: {} ({})", gpu_info.name, gpu_info.vendor);
        info!("  VRAM: {}MB total, {}MB available", 
              gpu_info.total_vram_mb, vram_monitor.get_available_vram_mb().await);
        info!("  Compute: {:?}", capabilities.compute_capabilities);
        
        Ok(Self {
            gpu_info,
            vram_monitor,
            capabilities,
        })
    }
    
    /// Get GPU information
    pub fn get_gpu_info(&self) -> &GpuInfo {
        &self.gpu_info
    }
    
    /// Get current VRAM usage
    pub async fn get_vram_usage(&self) -> VramUsage {
        self.vram_monitor.get_usage().await
    }
    
    /// Get system capabilities
    pub fn get_capabilities(&self) -> &SystemCapabilities {
        &self.capabilities
    }
    
    /// Check if a specific amount of VRAM is available
    pub async fn is_vram_available(&self, required_mb: u64) -> bool {
        self.vram_monitor.get_available_vram_mb().await >= required_mb
    }
    
    /// Get recommended memory configuration
    pub async fn get_recommended_config(&self) -> MemoryConfiguration {
        let available_vram = self.vram_monitor.get_available_vram_mb().await;
        
        match available_vram {
            mb if mb >= 6144 => MemoryConfiguration {
                tier: MemoryTier::HighEnd,
                max_model_memory_mb: 5400,
                concurrent_models: 4,
                recommended_precision: ModelPrecision::FP16,
                batch_size: 4,
            },
            mb if mb >= 4096 => MemoryConfiguration {
                tier: MemoryTier::MidRange,
                max_model_memory_mb: 3000,
                concurrent_models: 2,
                recommended_precision: ModelPrecision::INT8,
                batch_size: 2,
            },
            mb if mb >= 2048 => MemoryConfiguration {
                tier: MemoryTier::LowEnd,
                max_model_memory_mb: 1000,
                concurrent_models: 1,
                recommended_precision: ModelPrecision::INT8,
                batch_size: 1,
            },
            _ => MemoryConfiguration {
                tier: MemoryTier::CPUFallback,
                max_model_memory_mb: 0,
                concurrent_models: 1,
                recommended_precision: ModelPrecision::INT8,
                batch_size: 1,
            },
        }
    }
    
    // Private implementation methods
    
    async fn detect_gpu_hardware() -> Result<GpuInfo> {
        // Try NVIDIA first
        if let Ok(nvidia_info) = Self::detect_nvidia_gpu().await {
            return Ok(nvidia_info);
        }
        
        // Try AMD
        if let Ok(amd_info) = Self::detect_amd_gpu().await {
            return Ok(amd_info);
        }
        
        // Try Intel
        if let Ok(intel_info) = Self::detect_intel_gpu().await {
            return Ok(intel_info);
        }
        
        // Try Apple (macOS)
        #[cfg(target_os = "macos")]
        if let Ok(apple_info) = Self::detect_apple_gpu().await {
            return Ok(apple_info);
        }
        
        // Fallback to integrated graphics
        warn!("No discrete GPU detected, using integrated graphics fallback");
        Ok(GpuInfo {
            vendor: GpuVendor::Unknown,
            name: "Integrated Graphics".to_string(),
            total_vram_mb: 2048, // Conservative estimate
            compute_capability: ComputeCapability::Basic,
            driver_version: "Unknown".to_string(),
            pci_id: "Unknown".to_string(),
            supports_fp16: false,
            supports_int8: true,
            max_memory_bandwidth_gbps: 50.0,
        })
    }
    
    async fn detect_nvidia_gpu() -> Result<GpuInfo> {
        #[cfg(feature = "cuda")]
        {
            // Try nvidia-smi first
            if let Ok(output) = Command::new("nvidia-smi")
                .args(&["--query-gpu=name,memory.total,driver_version,pci.bus_id", "--format=csv,noheader,nounits"])
                .output() {
                
                if output.status.success() {
                    let output_str = String::from_utf8_lossy(&output.stdout);
                    if let Some(line) = output_str.lines().next() {
                        let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                        if parts.len() >= 4 {
                            let total_vram = parts[1].parse::<u64>().unwrap_or(4096);
                            
                            return Ok(GpuInfo {
                                vendor: GpuVendor::NVIDIA,
                                name: parts[0].to_string(),
                                total_vram_mb: total_vram,
                                compute_capability: Self::detect_nvidia_compute_capability().await,
                                driver_version: parts[2].to_string(),
                                pci_id: parts[3].to_string(),
                                supports_fp16: true,
                                supports_int8: true,
                                max_memory_bandwidth_gbps: Self::estimate_nvidia_bandwidth(&parts[0]).await,
                            });
                        }
                    }
                }
            }
            
            // Try CUDA runtime API as fallback
            // This would use actual CUDA libraries in production
            info!("CUDA detection via runtime API (placeholder)");
        }
        
        Err(anyhow::anyhow!("NVIDIA GPU not detected"))
    }
    
    async fn detect_amd_gpu() -> Result<GpuInfo> {
        #[cfg(feature = "rocm")]
        {
            // Try rocm-smi
            if let Ok(output) = Command::new("rocm-smi")
                .args(&["--showmeminfo", "vram", "--csv"])
                .output() {
                
                if output.status.success() {
                    let output_str = String::from_utf8_lossy(&output.stdout);
                    for line in output_str.lines().skip(1) { // Skip header
                        if let Some(vram_str) = line.split(',').nth(1) {
                            if let Ok(vram_mb) = vram_str.trim().parse::<u64>() {
                                // Get GPU name from different command
                                let name = Self::get_amd_gpu_name().await.unwrap_or_else(|| "AMD GPU".to_string());
                                
                                return Ok(GpuInfo {
                                    vendor: GpuVendor::AMD,
                                    name,
                                    total_vram_mb: vram_mb,
                                    compute_capability: ComputeCapability::ROCm,
                                    driver_version: "ROCm".to_string(),
                                    pci_id: "Unknown".to_string(),
                                    supports_fp16: true,
                                    supports_int8: true,
                                    max_memory_bandwidth_gbps: 500.0, // Conservative estimate
                                });
                            }
                        }
                    }
                }
            }
        }
        
        Err(anyhow::anyhow!("AMD GPU not detected"))
    }
    
    async fn detect_intel_gpu() -> Result<GpuInfo> {
        // Try Intel GPU detection via system info
        #[cfg(target_os = "windows")]
        {
            if let Ok(output) = Command::new("wmic")
                .args(&["path", "win32_VideoController", "get", "Name,AdapterRAM", "/format:csv"])
                .output() {
                
                let output_str = String::from_utf8_lossy(&output.stdout);
                for line in output_str.lines().skip(1) {
                    if line.contains("Intel") {
                        let parts: Vec<&str> = line.split(',').collect();
                        if parts.len() >= 3 {
                            let vram_bytes = parts[1].parse::<u64>().unwrap_or(2147483648); // 2GB default
                            let vram_mb = vram_bytes / (1024 * 1024);
                            
                            return Ok(GpuInfo {
                                vendor: GpuVendor::Intel,
                                name: parts[2].to_string(),
                                total_vram_mb: vram_mb,
                                compute_capability: ComputeCapability::Basic,
                                driver_version: "Unknown".to_string(),
                                pci_id: "Unknown".to_string(),
                                supports_fp16: true,
                                supports_int8: false,
                                max_memory_bandwidth_gbps: 100.0,
                            });
                        }
                    }
                }
            }
        }
        
        #[cfg(target_os = "linux")]
        {
            // Try lspci for Intel GPU detection
            if let Ok(output) = Command::new("lspci").args(&["-nn"]).output() {
                let output_str = String::from_utf8_lossy(&output.stdout);
                for line in output_str.lines() {
                    if line.contains("Intel") && (line.contains("VGA") || line.contains("Display")) {
                        return Ok(GpuInfo {
                            vendor: GpuVendor::Intel,
                            name: "Intel Integrated Graphics".to_string(),
                            total_vram_mb: 2048, // Shared memory
                            compute_capability: ComputeCapability::Basic,
                            driver_version: "Unknown".to_string(),
                            pci_id: line.split_whitespace().next().unwrap_or("Unknown").to_string(),
                            supports_fp16: true,
                            supports_int8: false,
                            max_memory_bandwidth_gbps: 50.0,
                        });
                    }
                }
            }
        }
        
        Err(anyhow::anyhow!("Intel GPU not detected"))
    }
    
    #[cfg(target_os = "macos")]
    async fn detect_apple_gpu() -> Result<GpuInfo> {
        // Use system_profiler to get GPU info
        if let Ok(output) = Command::new("system_profiler")
            .args(&["SPDisplaysDataType", "-json"])
            .output() {
            
            let output_str = String::from_utf8_lossy(&output.stdout);
            // Parse JSON to extract GPU information
            // This is a simplified version - would use proper JSON parsing in production
            
            if output_str.contains("Apple") {
                // Estimate VRAM based on system memory (unified memory architecture)
                let system_memory = Self::get_system_memory_gb().await.unwrap_or(16);
                let estimated_gpu_memory = (system_memory * 1024) / 2; // Use half of system memory
                
                return Ok(GpuInfo {
                    vendor: GpuVendor::Apple,
                    name: "Apple Silicon GPU".to_string(),
                    total_vram_mb: estimated_gpu_memory,
                    compute_capability: ComputeCapability::Metal,
                    driver_version: "Metal".to_string(),
                    pci_id: "Apple".to_string(),
                    supports_fp16: true,
                    supports_int8: true,
                    max_memory_bandwidth_gbps: 400.0, // M1/M2 typical bandwidth
                });
            }
        }
        
        Err(anyhow::anyhow!("Apple GPU not detected"))
    }
    
    async fn detect_system_capabilities() -> Result<SystemCapabilities> {
        let cpu_cores = num_cpus::get();
        let cpu_threads = num_cpus::get(); // For simplicity, assume 1:1 core:thread ratio
        let total_ram_gb = Self::get_system_memory_gb().await.unwrap_or(16);
        
        // Detect CPU features
        let supports_avx2 = Self::cpu_supports_avx2().await;
        let supports_avx512 = Self::cpu_supports_avx512().await;
        
        Ok(SystemCapabilities {
            cpu_cores,
            cpu_threads,
            total_ram_gb,
            supports_avx2,
            supports_avx512,
            compute_capabilities: vec![ComputeCapability::CPU],
        })
    }
    
    async fn detect_nvidia_compute_capability() -> ComputeCapability {
        // This would query actual CUDA device properties
        // For now, assume modern GPU with compute capability 7.5+
        ComputeCapability::CUDA { major: 7, minor: 5 }
    }
    
    async fn estimate_nvidia_bandwidth(gpu_name: &str) -> f64 {
        // Rough estimates based on GPU name
        match gpu_name.to_lowercase() {
            name if name.contains("4090") => 1008.0,
            name if name.contains("4080") => 716.8,
            name if name.contains("4070") => 504.2,
            name if name.contains("3090") => 936.2,
            name if name.contains("3080") => 760.3,
            name if name.contains("3070") => 448.0,
            _ => 400.0, // Conservative default
        }
    }
    
    #[cfg(feature = "rocm")]
    async fn get_amd_gpu_name() -> Option<String> {
        if let Ok(output) = Command::new("rocm-smi").args(&["--showproductname"]).output() {
            if output.status.success() {
                let output_str = String::from_utf8_lossy(&output.stdout);
                if let Some(line) = output_str.lines().find(|l| !l.starts_with("=") && !l.trim().is_empty()) {
                    return Some(line.trim().to_string());
                }
            }
        }
        None
    }
    
    async fn get_system_memory_gb() -> Option<u64> {
        #[cfg(target_os = "linux")]
        {
            if let Ok(output) = Command::new("free").args(&["-g"]).output() {
                let output_str = String::from_utf8_lossy(&output.stdout);
                if let Some(line) = output_str.lines().nth(1) {
                    if let Some(mem_str) = line.split_whitespace().nth(1) {
                        return mem_str.parse().ok();
                    }
                }
            }
        }
        
        #[cfg(target_os = "macos")]
        {
            if let Ok(output) = Command::new("sysctl").args(&["-n", "hw.memsize"]).output() {
                let output_str = String::from_utf8_lossy(&output.stdout);
                if let Ok(bytes) = output_str.trim().parse::<u64>() {
                    return Some(bytes / (1024 * 1024 * 1024));
                }
            }
        }
        
        #[cfg(target_os = "windows")]
        {
            if let Ok(output) = Command::new("wmic")
                .args(&["computersystem", "get", "TotalPhysicalMemory", "/value"])
                .output() {
                let output_str = String::from_utf8_lossy(&output.stdout);
                for line in output_str.lines() {
                    if line.starts_with("TotalPhysicalMemory=") {
                        if let Some(mem_str) = line.split('=').nth(1) {
                            if let Ok(bytes) = mem_str.parse::<u64>() {
                                return Some(bytes / (1024 * 1024 * 1024));
                            }
                        }
                    }
                }
            }
        }
        
        None
    }
    
    async fn cpu_supports_avx2() -> bool {
        // This would use proper CPU feature detection
        // For now, assume modern CPUs support AVX2
        true
    }
    
    async fn cpu_supports_avx512() -> bool {
        // AVX-512 is less common, be conservative
        false
    }
}

/// VRAM monitoring system
pub struct VramMonitor {
    gpu_vendor: GpuVendor,
    total_vram_mb: Arc<AtomicU64>,
    allocated_vram_mb: Arc<AtomicU64>,
    monitoring_enabled: bool,
}

impl VramMonitor {
    async fn new(gpu_info: &GpuInfo) -> Result<Self> {
        Ok(Self {
            gpu_vendor: gpu_info.vendor.clone(),
            total_vram_mb: Arc::new(AtomicU64::new(gpu_info.total_vram_mb)),
            allocated_vram_mb: Arc::new(AtomicU64::new(0)),
            monitoring_enabled: true,
        })
    }
    
    async fn get_usage(&self) -> VramUsage {
        let total = self.total_vram_mb.load(Ordering::Relaxed);
        let allocated = self.allocated_vram_mb.load(Ordering::Relaxed);
        let used = self.query_actual_usage().await.unwrap_or(allocated);
        let available = total.saturating_sub(used);
        
        VramUsage {
            total_mb: total,
            used_mb: used,
            available_mb: available,
            utilization_percent: (used as f64 / total as f64) * 100.0,
        }
    }
    
    async fn get_available_vram_mb(&self) -> u64 {
        self.get_usage().await.available_mb
    }
    
    pub fn record_allocation(&self, size_mb: u64) {
        self.allocated_vram_mb.fetch_add(size_mb, Ordering::Relaxed);
    }
    
    pub fn record_deallocation(&self, size_mb: u64) {
        self.allocated_vram_mb.fetch_sub(size_mb, Ordering::Relaxed);
    }
    
    async fn query_actual_usage(&self) -> Option<u64> {
        match self.gpu_vendor {
            GpuVendor::NVIDIA => self.query_nvidia_usage().await,
            GpuVendor::AMD => self.query_amd_usage().await,
            _ => None,
        }
    }
    
    async fn query_nvidia_usage(&self) -> Option<u64> {
        if let Ok(output) = Command::new("nvidia-smi")
            .args(&["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
            .output() {
            
            if output.status.success() {
                let output_str = String::from_utf8_lossy(&output.stdout);
                if let Some(line) = output_str.lines().next() {
                    return line.trim().parse().ok();
                }
            }
        }
        None
    }
    
    async fn query_amd_usage(&self) -> Option<u64> {
        // AMD GPU memory usage query would go here
        // ROCm tools don't have as straightforward memory usage query
        None
    }
}

// Supporting structures

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    pub vendor: GpuVendor,
    pub name: String,
    pub total_vram_mb: u64,
    pub compute_capability: ComputeCapability,
    pub driver_version: String,
    pub pci_id: String,
    pub supports_fp16: bool,
    pub supports_int8: bool,
    pub max_memory_bandwidth_gbps: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComputeCapability {
    CPU,
    Basic,
    CUDA { major: i32, minor: i32 },
    ROCm,
    OpenVINO,
    Metal,
    DirectML,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemCapabilities {
    pub cpu_cores: usize,
    pub cpu_threads: usize,
    pub total_ram_gb: u64,
    pub supports_avx2: bool,
    pub supports_avx512: bool,
    pub compute_capabilities: Vec<ComputeCapability>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VramUsage {
    pub total_mb: u64,
    pub used_mb: u64,
    pub available_mb: u64,
    pub utilization_percent: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfiguration {
    pub tier: MemoryTier,
    pub max_model_memory_mb: u64,
    pub concurrent_models: usize,
    pub recommended_precision: ModelPrecision,
    pub batch_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryTier {
    HighEnd,      // 6GB+
    MidRange,     // 4-6GB
    LowEnd,       // 2-4GB
    CPUFallback,  // <2GB
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ModelPrecision {
    FP32,
    FP16,
    INT8,
    INT4,
}