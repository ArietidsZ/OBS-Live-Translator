// Enhanced configuration loader
// Part 1.8: Comprehensive TOML configuration with auto-detection

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::execution_provider::ExecutionProvider;
use crate::logging::LogConfig;
use crate::platform_detect::Platform;
use crate::profile::ProfileDetector;
use crate::types::Profile;
use crate::types::VadType;

/// Main application configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AppConfig {
    /// Performance profile ("auto", "low", "medium", "high")
    #[serde(default = "default_profile")]
    pub profile: String,

    /// Execution provider override (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub execution_provider: Option<String>,

    /// Source language (ISO code or "auto")
    #[serde(default = "default_source_lang")]
    pub source_language: String,

    /// Target language (ISO code)
    #[serde(default = "default_target_lang")]
    pub target_language: String,

    /// Audio configuration
    #[serde(default)]
    pub audio: AudioConfig,

    /// Model paths
    #[serde(default)]
    pub models: ModelsConfig,

    /// Logging configuration
    #[serde(default)]
    pub logging: LoggingConfig,

    /// Hardware acceleration settings
    #[serde(default)]
    pub acceleration: AccelerationConfig,

    /// Streaming configuration
    #[serde(default)]
    pub streaming: StreamingConfig,

    /// Performance tuning
    #[serde(default)]
    pub performance: PerformanceConfig,

    /// Advanced options
    #[serde(default)]
    pub advanced: AdvancedConfig,

    /// Metrics configuration
    #[serde(default)]
    pub metrics: MetricsConfig,

    /// Debug options
    #[serde(default)]
    pub debug: DebugConfig,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            profile: "auto".to_string(),
            execution_provider: None,
            source_language: "auto".to_string(),
            target_language: "en".to_string(),
            audio: AudioConfig::default(),
            models: ModelsConfig::default(),
            logging: LoggingConfig::default(),
            acceleration: AccelerationConfig::default(),
            streaming: StreamingConfig::default(),
            performance: PerformanceConfig::default(),
            advanced: AdvancedConfig::default(),
            metrics: MetricsConfig::default(),
            debug: DebugConfig::default(),
        }
    }
}

impl AppConfig {
    /// Load configuration from TOML file
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let config: AppConfig = toml::from_str(&contents)?;
        Ok(config)
    }

    /// Save configuration to TOML file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let contents = toml::to_string_pretty(self)?;
        std::fs::write(path, contents)?;
        Ok(())
    }

    /// Resolve profile (handle "auto" detection)
    pub fn resolve_profile(&self) -> Result<Profile> {
        if self.profile == "auto" {
            ProfileDetector::detect()
        } else {
            match self.profile.to_lowercase().as_str() {
                "low" => Ok(Profile::Low),
                "medium" => Ok(Profile::Medium),
                "high" => Ok(Profile::High),
                _ => ProfileDetector::detect(), // Fallback to auto
            }
        }
    }

    /// Resolve execution provider (handle "auto" detection)
    pub fn resolve_execution_provider(&self) -> Result<ExecutionProvider> {
        if let Some(ref ep_str) = self.execution_provider {
            if ep_str == "auto" {
                let platform = Platform::detect()?;
                Ok(ExecutionProvider::select_optimal(&platform))
            } else {
                match ep_str.to_lowercase().as_str() {
                    "tensorrt" => Ok(ExecutionProvider::TensorRT),
                    "coreml" => Ok(ExecutionProvider::CoreML),
                    "openvino" => Ok(ExecutionProvider::OpenVINO),
                    "directml" => Ok(ExecutionProvider::DirectML),
                    "cuda" => Ok(ExecutionProvider::Cuda),
                    "cpu" => Ok(ExecutionProvider::Cpu),
                    _ => {
                        // Invalid, fall back to auto
                        let platform = Platform::detect()?;
                        Ok(ExecutionProvider::select_optimal(&platform))
                    }
                }
            }
        } else {
            // No override, use auto-detection
            let platform = Platform::detect()?;
            Ok(ExecutionProvider::select_optimal(&platform))
        }
    }

    /// Convert logging config to LogConfig
    pub fn to_log_config(&self) -> LogConfig {
        LogConfig {
            level: self.logging.level.clone(),
            json_format: self.logging.json_format,
            log_file: if self.logging.log_file.is_empty() {
                None
            } else {
                Some(self.logging.log_file.clone())
            },
            enable_spans: self.logging.enable_spans,
        }
    }
}

// Default value functions
fn default_profile() -> String {
    "auto".to_string()
}
fn default_source_lang() -> String {
    "auto".to_string()
}
fn default_target_lang() -> String {
    "en".to_string()
}

// Configuration subsections

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AudioConfig {
    #[serde(default = "default_sample_rate")]
    pub sample_rate: u32,
    #[serde(default = "default_chunk_size")]
    pub chunk_size: usize,
    #[serde(default = "default_channels")]
    pub channels: u32,
    #[serde(default)]
    pub vad_type: VadType,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            chunk_size: 512,
            channels: 1,
            vad_type: VadType::default(),
        }
    }
}

fn default_sample_rate() -> u32 {
    16000
}
fn default_chunk_size() -> usize {
    512
}
fn default_channels() -> u32 {
    1
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelsConfig {
    #[serde(default = "default_models_base_path")]
    pub base_path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parakeet_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub canary_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nllb_encoder_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nllb_decoder_path: Option<String>,
}

impl Default for ModelsConfig {
    fn default() -> Self {
        Self {
            base_path: "./models".to_string(),
            parakeet_path: None,
            canary_path: None,
            nllb_encoder_path: None,
            nllb_decoder_path: None,
        }
    }
}

fn default_models_base_path() -> String {
    "./models".to_string()
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LoggingConfig {
    #[serde(default = "default_log_level")]
    pub level: String,
    #[serde(default)]
    pub json_format: bool,
    #[serde(default)]
    pub log_file: String,
    #[serde(default = "default_true")]
    pub enable_spans: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            json_format: false,
            log_file: String::new(),
            enable_spans: true,
        }
    }
}

fn default_log_level() -> String {
    "info".to_string()
}
fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[derive(Default)]
pub struct AccelerationConfig {
    #[serde(default)]
    pub tensorrt: TensorRTConfig,
    #[serde(default)]
    pub coreml: CoreMLConfig,
    #[serde(default)]
    pub openvino: OpenVINOConfig,
    #[serde(default)]
    pub directml: DirectMLConfig,
}


#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TensorRTConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub device_id: u32,
    #[serde(default = "default_true")]
    pub fp16: bool,
    #[serde(default = "default_true")]
    pub int8: bool,
    #[serde(default = "default_tensorrt_cache")]
    pub engine_cache_dir: String,
    #[serde(default = "default_workspace_size")]
    pub workspace_size_mb: u32,
}

impl Default for TensorRTConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            device_id: 0,
            fp16: true,
            int8: true,
            engine_cache_dir: "./cache/tensorrt".to_string(),
            workspace_size_mb: 2048,
        }
    }
}

fn default_tensorrt_cache() -> String {
    "./cache/tensorrt".to_string()
}
fn default_workspace_size() -> u32 {
    2048
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CoreMLConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub use_cpu_only: bool,
    #[serde(default = "default_true")]
    pub enable_on_subgraph: bool,
    #[serde(default = "default_true")]
    pub only_enable_device_with_ane: bool,
}

impl Default for CoreMLConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            use_cpu_only: false,
            enable_on_subgraph: true,
            only_enable_device_with_ane: true,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenVINOConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_device_auto")]
    pub device_type: String,
    #[serde(default = "default_precision_fp16")]
    pub precision: String,
    #[serde(default = "default_openvino_cache")]
    pub cache_dir: String,
}

impl Default for OpenVINOConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            device_type: "AUTO".to_string(),
            precision: "FP16".to_string(),
            cache_dir: "./cache/openvino".to_string(),
        }
    }
}

fn default_device_auto() -> String {
    "AUTO".to_string()
}
fn default_precision_fp16() -> String {
    "FP16".to_string()
}
fn default_openvino_cache() -> String {
    "./cache/openvino".to_string()
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DirectMLConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub device_id: u32,
}

impl Default for DirectMLConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            device_id: 0,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct StreamingConfig {
    #[serde(default = "default_websocket_port")]
    pub websocket_port: u16,
    #[serde(default = "default_websocket_host")]
    pub websocket_host: String,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            websocket_port: 8080,
            websocket_host: "0.0.0.0".to_string(),
        }
    }
}

fn default_websocket_port() -> u16 {
    8080
}
fn default_websocket_host() -> String {
    "0.0.0.0".to_string()
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PerformanceConfig {
    #[serde(default = "default_batch_size")]
    pub max_batch_size: usize,
    #[serde(default = "default_batch_timeout")]
    pub batch_timeout_ms: u64,
    #[serde(default)]
    pub max_memory_mb: usize,
    #[serde(default)]
    pub max_vram_mb: usize,
    #[serde(default = "default_vad_latency")]
    pub target_vad_latency_ms: u64,
    #[serde(default = "default_asr_latency")]
    pub target_asr_latency_ms: u64,
    #[serde(default = "default_translation_latency")]
    pub target_translation_latency_ms: u64,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 4,
            batch_timeout_ms: 50,
            max_memory_mb: 8192,
            max_vram_mb: 8192,
            target_vad_latency_ms: 10,
            target_asr_latency_ms: 150,
            target_translation_latency_ms: 100,
        }
    }
}

fn default_batch_size() -> usize {
    4
}
fn default_batch_timeout() -> u64 {
    50
}
fn default_vad_latency() -> u64 {
    10
}
fn default_asr_latency() -> u64 {
    150
}
fn default_translation_latency() -> u64 {
    100
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AdvancedConfig {
    #[serde(default = "default_auto")]
    pub quantization: String,
    #[serde(default = "default_true")]
    pub enable_model_cache: bool,
    #[serde(default = "default_cache_dir")]
    pub cache_dir: String,
    #[serde(default = "default_retries")]
    pub max_retries: u32,
    #[serde(default = "default_retry_delay")]
    pub retry_delay_ms: u64,
}

impl Default for AdvancedConfig {
    fn default() -> Self {
        Self {
            quantization: "auto".to_string(),
            enable_model_cache: true,
            cache_dir: "./cache/models".to_string(),
            max_retries: 3,
            retry_delay_ms: 100,
        }
    }
}

fn default_auto() -> String {
    "auto".to_string()
}
fn default_cache_dir() -> String {
    "./cache/models".to_string()
}
fn default_retries() -> u32 {
    3
}
fn default_retry_delay() -> u64 {
    100
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MetricsConfig {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_export_interval")]
    pub export_interval_seconds: u64,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            export_interval_seconds: 60,
        }
    }
}

fn default_export_interval() -> u64 {
    60
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[derive(Default)]
pub struct DebugConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub dump_audio: bool,
    #[serde(default)]
    pub dump_models: bool,
    #[serde(default)]
    pub verbose_logging: bool,
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = AppConfig::default();
        assert_eq!(config.profile, "auto");
        assert_eq!(config.source_language, "auto");
        assert_eq!(config.target_language, "en");
        assert_eq!(config.audio.sample_rate, 16000);
    }

    #[test]
    fn test_profile_resolution() {
        let mut config = AppConfig::default();

        // Test explicit profile
        config.profile = "low".to_string();
        let profile = config.resolve_profile().unwrap();
        assert!(matches!(profile, Profile::Low));

        config.profile = "medium".to_string();
        let profile = config.resolve_profile().unwrap();
        assert!(matches!(profile, Profile::Medium));

        config.profile = "high".to_string();
        let profile = config.resolve_profile().unwrap();
        assert!(matches!(profile, Profile::High));
    }
}
