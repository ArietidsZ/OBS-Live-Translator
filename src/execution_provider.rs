// Execution provider selection and configuration
// Based on Part 6 research (Inference Frameworks & Hardware Acceleration)

use crate::platform_detect::Platform;
use anyhow::{anyhow, Result};

/// Execution provider types supported by ONNX Runtime
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionProvider {
    /// NVIDIA TensorRT (Windows/Linux, NVIDIA GPUs)
    TensorRT,
    /// Apple CoreML (macOS, Apple Silicon)
    CoreML,
    /// Intel OpenVINO (Windows/Linux, Intel CPUs/iGPUs/NPUs)
    OpenVINO,
    /// DirectML (Windows, generic GPUs including AMD)
    DirectML,
    /// CUDA (NVIDIA GPUs, fallback if TensorRT unavailable)
    Cuda,
    /// CPU (universal fallback)
    Cpu,
}

impl ExecutionProvider {
    /// Select optimal execution provider based on detected platform
    /// Uses Part 1.4 platform detection and Part 6 EP recommendations
    pub fn select_optimal(platform: &Platform) -> Self {
        match platform.recommended_execution_provider() {
            "tensorrt" => ExecutionProvider::TensorRT,
            "coreml" => ExecutionProvider::CoreML,
            "openvino" => ExecutionProvider::OpenVINO,
            "directml" => ExecutionProvider::DirectML,
            "cuda" => ExecutionProvider::Cuda,
            _ => ExecutionProvider::Cpu,
        }
    }

    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            ExecutionProvider::TensorRT => "TensorRT",
            ExecutionProvider::CoreML => "CoreML",
            ExecutionProvider::OpenVINO => "OpenVINO",
            ExecutionProvider::DirectML => "DirectML",
            ExecutionProvider::Cuda => "CUDA",
            ExecutionProvider::Cpu => "CPU",
        }
    }

    /// Check if this EP is available on current platform
    pub fn is_available(&self) -> bool {
        match self {
            ExecutionProvider::TensorRT => Self::check_tensorrt_available(),
            ExecutionProvider::CoreML => cfg!(target_os = "macos"),
            ExecutionProvider::OpenVINO => Self::check_openvino_available(),
            ExecutionProvider::DirectML => cfg!(target_os = "windows"),
            ExecutionProvider::Cuda => Self::check_cuda_available(),
            ExecutionProvider::Cpu => true, // Always available
        }
    }

    /// Configure ONNX Runtime session options for this EP
    pub fn configure_session(
        &self,
        builder: ort::session::builder::SessionBuilder,
    ) -> Result<ort::session::builder::SessionBuilder> {
        if !self.is_available() {
            return Err(anyhow!(
                "{} execution provider not available on this platform",
                self.name()
            ));
        }

        match self {
            ExecutionProvider::TensorRT => self.configure_tensorrt(builder),
            ExecutionProvider::CoreML => self.configure_coreml(builder),
            ExecutionProvider::OpenVINO => self.configure_openvino(builder),
            ExecutionProvider::DirectML => self.configure_directml(builder),
            ExecutionProvider::Cuda => self.configure_cuda(builder),
            ExecutionProvider::Cpu => Ok(builder), // Default
        }
    }

    // Platform availability checks

    fn check_tensorrt_available() -> bool {
        // Check if TensorRT is available
        // This would typically check for TensorRT library presence
        // For now, simplified: check if nvidia-smi works (NVIDIA GPU present)
        std::process::Command::new("nvidia-smi")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    fn check_openvino_available() -> bool {
        // OpenVINO is available on Windows/Linux with Intel CPUs
        cfg!(any(target_os = "windows", target_os = "linux"))
    }

    fn check_cuda_available() -> bool {
        // Same as TensorRT check for now
        Self::check_tensorrt_available()
    }

    // EP-specific configuration methods
    // Based on Part 6 research recommendations

    fn configure_tensorrt(
        &self,
        builder: ort::session::builder::SessionBuilder,
    ) -> Result<ort::session::builder::SessionBuilder> {
        // Part 6.2: TensorRT configuration for NVIDIA GPUs
        #[cfg(feature = "tensorrt")]
        {
            use ort::execution_providers::TensorRTExecutionProvider;

            // Configure TensorRT with optimal settings for real-time inference
            let options = TensorRTExecutionProvider::default()
                .with_device_id(0)
                .with_fp16(true) // Enable FP16 precision (Volta+)
                .with_int8(true) // Enable INT8 precision (if model supports it)
                .with_engine_cache(true)
                .with_engine_cache_path("./trt_cache") // Cache compiled engines
                .with_max_workspace_size(2 * 1024 * 1024 * 1024); // 2GB workspace

            let builder = builder.with_execution_providers([options.into()])?;
            tracing::info!("Configured TensorRT EP: FP16+INT8 enabled, Cache: ./trt_cache");
            Ok(builder)
        }
        #[cfg(not(feature = "tensorrt"))]
        {
            tracing::warn!("TensorRT feature not enabled in build");
            Ok(builder)
        }
    }

    fn configure_coreml(
        &self,
        builder: ort::session::builder::SessionBuilder,
    ) -> Result<ort::session::builder::SessionBuilder> {
        // Part 6.3: CoreML configuration for Apple Silicon
        #[cfg(feature = "coreml")]
        {
            use ort::execution_providers::CoreMLExecutionProvider;

            let options = CoreMLExecutionProvider::default()
                .with_use_cpu_only(false) // Use Neural Engine + GPU
                .with_enable_on_subgraph(true)
                .with_only_enable_device_with_ane(true); // Prefer Neural Engine

            let builder = builder.with_execution_providers([options.into()])?;
            tracing::info!("Configured CoreML EP: Neural Engine + GPU enabled");
            Ok(builder)
        }
        #[cfg(not(feature = "coreml"))]
        {
            tracing::warn!("CoreML feature not enabled in build");
            Ok(builder)
        }
    }

    fn configure_openvino(
        &self,
        builder: ort::session::builder::SessionBuilder,
    ) -> Result<ort::session::builder::SessionBuilder> {
        // Part 6.4: OpenVINO configuration for Intel CPUs/NPUs
        #[cfg(feature = "openvino")]
        {
            use ort::execution_providers::OpenVINOExecutionProvider;

            let options = OpenVINOExecutionProvider::default()
                .with_precision("FP16")
                .with_opencl_throttling(false)
                .with_cache_dir("./openvino_cache")
                .with_device_type("CPU");

            let builder = builder.with_execution_providers([options.into()])?;
            tracing::info!("Configured OpenVINO EP: Device=CPU, Precision=FP16");
            Ok(builder)
        }
        #[cfg(not(feature = "openvino"))]
        {
            tracing::warn!("OpenVINO feature not enabled in build");
            Ok(builder)
        }
    }

    fn configure_directml(
        &self,
        builder: ort::session::builder::SessionBuilder,
    ) -> Result<ort::session::builder::SessionBuilder> {
        // Part 2.7: DirectML configuration for Windows (AMD/Intel/NVIDIA)
        #[cfg(feature = "directml")]
        {
            use ort::execution_providers::DirectMLExecutionProvider;

            let options = DirectMLExecutionProvider::default().with_device_id(0); // Default to first GPU

            let builder = builder.with_execution_providers([options.into()])?;
            tracing::info!("Configured DirectML EP: Device ID=0");
            Ok(builder)
        }
        #[cfg(not(feature = "directml"))]
        {
            tracing::warn!("DirectML feature not enabled in build");
            Ok(builder)
        }
    }

    fn configure_cuda(
        &self,
        builder: ort::session::builder::SessionBuilder,
    ) -> Result<ort::session::builder::SessionBuilder> {
        // CUDA fallback configuration
        #[cfg(feature = "cuda")]
        {
            tracing::info!("Configuring CUDA EP (placeholder)");
        }
        Ok(builder)
    }

    /// Get recommended quantization format for this EP
    /// Based on Part 9 research findings
    pub fn recommended_quantization(&self, platform: &Platform) -> &'static str {
        match self {
            ExecutionProvider::TensorRT => {
                // Part 9: Check if FP8/INT4 supported by hardware
                if platform.supports_quantization("fp8") {
                    "fp8" // RTX 5090, H100 (Blackwell/Hopper)
                } else if platform.supports_quantization("int4") {
                    "int4" // Ampere and newer
                } else {
                    "int8" // Universal fallback
                }
            }
            ExecutionProvider::CoreML => {
                // Part 9: CoreML works well with INT4/INT8
                if platform.supports_quantization("int4") {
                    "int4"
                } else {
                    "int8"
                }
            }
            ExecutionProvider::OpenVINO => {
                // Part 9: OpenVINO optimized for INT8
                "int8"
            }
            ExecutionProvider::DirectML => {
                // Part 9: DirectML with FP8 for RDNA4, else FP16
                if platform.supports_quantization("fp8") {
                    "fp8"
                } else {
                    "fp16"
                }
            }
            ExecutionProvider::Cuda | ExecutionProvider::Cpu => {
                "int8" // Safe default
            }
        }
    }
}

/// Execution provider configuration builder
#[derive(Debug, Clone)]
pub struct ExecutionProviderConfig {
    pub provider: ExecutionProvider,
    pub platform: Platform,
}

impl ExecutionProviderConfig {
    /// Create new configuration with explicit provider and platform
    pub fn new(provider: ExecutionProvider, platform: Platform) -> Self {
        Self { provider, platform }
    }

    /// Create new configuration from detected platform
    pub fn from_platform(platform: Platform) -> Self {
        let provider = ExecutionProvider::select_optimal(&platform);
        Self { provider, platform }
    }

    /// Create with explicit provider selection
    pub fn with_provider(platform: Platform, provider: ExecutionProvider) -> Result<Self> {
        if !provider.is_available() {
            return Err(anyhow!(
                "{} is not available on this platform",
                provider.name()
            ));
        }
        Ok(Self { provider, platform })
    }

    /// Get the selected execution provider
    pub fn provider(&self) -> ExecutionProvider {
        self.provider
    }

    /// Configure session builder
    pub fn configure_session(
        &self,
        mut builder: ort::session::builder::SessionBuilder,
        config: &crate::types::AccelerationConfig,
    ) -> Result<ort::session::builder::SessionBuilder> {
        // Apply graph optimizations
        if config.flash_attention {
            builder = builder
                .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?;
            tracing::info!("Graph Optimization: Level 3 (Flash Attention enabled)");
        } else {
            builder = builder
                .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level1)?;
            tracing::info!("Graph Optimization: Level 1 (Basic)");
        }

        self.provider.configure_session(builder)
    }

    /// Get recommended quantization format
    pub fn recommended_quantization(&self) -> &'static str {
        self.provider.recommended_quantization(&self.platform)
    }

    /// Get fallback configuration if this one fails
    pub fn fallback(&self) -> Option<Self> {
        let fallback_provider = match self.provider {
            ExecutionProvider::TensorRT => Some(ExecutionProvider::Cuda),
            ExecutionProvider::Cuda => Some(ExecutionProvider::Cpu),
            ExecutionProvider::CoreML => Some(ExecutionProvider::Cpu),
            ExecutionProvider::OpenVINO => Some(ExecutionProvider::Cpu),
            ExecutionProvider::DirectML => Some(ExecutionProvider::Cpu),
            ExecutionProvider::Cpu => None,
        };

        fallback_provider.map(|p| Self {
            provider: p,
            platform: self.platform.clone(),
        })
    }

    /// Get summary for logging
    pub fn summary(&self) -> String {
        format!(
            "EP: {}, Platform: {}, Recommended Quantization: {}",
            self.provider.name(),
            self.platform.description(),
            self.recommended_quantization()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ep_selection() {
        let platform = Platform::detect().unwrap();
        let ep = ExecutionProvider::select_optimal(&platform);

        println!("Platform: {}", platform.description());
        println!("Selected EP: {}", ep.name());
        println!("EP Available: {}", ep.is_available());
    }

    #[test]
    fn test_ep_config() {
        let platform = Platform::detect().unwrap();
        let config = ExecutionProviderConfig::from_platform(platform);

        println!("Config Summary: {}", config.summary());

        // Just verify we can call the method (requires ort::session::SessionBuilder which is hard to mock here)
        // So we'll skip the actual builder call in this unit test
    }

    #[test]
    fn test_cpu_always_available() {
        assert!(ExecutionProvider::Cpu.is_available());
    }

    #[test]
    fn test_quantization_recommendation() {
        let platform = Platform::detect().unwrap();
        let config = ExecutionProviderConfig::from_platform(platform);
        let quant = config.recommended_quantization();

        println!("Recommended quantization: {quant}");
        assert!(["fp8", "int4", "int8", "fp16"].contains(&quant));
    }
}
