use std::path::Path;
use std::env;
use anyhow::{Result, anyhow};
use serde::{Serialize, Deserialize};
use tokio::fs;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub server: ServerConfig,
    pub streaming: StreamingConfig,
    pub cache: CacheConfig,
    pub resources: ResourceConfig,
    pub monitoring: MonitoringConfig,
    pub models: ModelConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub monitoring_port: u16,
    pub max_connections: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfig {
    pub max_concurrent_streams: usize,
    pub default_priority: String,
    pub buffer_size: usize,
    pub max_latency_ms: u32,
    pub enable_vad: bool,
    pub enable_noise_reduction: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    pub max_size_mb: usize,
    pub eviction_policy: String,
    pub enable_compression: bool,
    pub enable_warming: bool,
    pub warming_threshold: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceConfig {
    pub max_cpu_percent: f32,
    pub max_memory_mb: usize,
    pub max_gpu_memory_mb: usize,
    pub max_threads: usize,
    pub enable_dynamic_scaling: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    pub enable_metrics: bool,
    pub enable_tracing: bool,
    pub metrics_interval_seconds: u64,
    pub log_level: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub whisper_model: String,
    pub translation_model: String,
    pub enable_quantization: bool,
    pub precision: String,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            server: ServerConfig {
                host: "0.0.0.0".to_string(),
                port: 8080,
                monitoring_port: 8081,
                max_connections: 1000,
            },
            streaming: StreamingConfig {
                max_concurrent_streams: 20,
                default_priority: "normal".to_string(),
                buffer_size: 1024,
                max_latency_ms: 100,
                enable_vad: true,
                enable_noise_reduction: true,
            },
            cache: CacheConfig {
                max_size_mb: 2048,
                eviction_policy: "adaptive".to_string(),
                enable_compression: true,
                enable_warming: true,
                warming_threshold: 0.7,
            },
            resources: ResourceConfig {
                max_cpu_percent: 80.0,
                max_memory_mb: 8192,
                max_gpu_memory_mb: 4096,
                max_threads: 16,
                enable_dynamic_scaling: true,
            },
            monitoring: MonitoringConfig {
                enable_metrics: true,
                enable_tracing: true,
                metrics_interval_seconds: 60,
                log_level: "info".to_string(),
            },
            models: ModelConfig {
                whisper_model: "whisper-v3-turbo".to_string(),
                translation_model: "nllb-600m".to_string(),
                enable_quantization: true,
                precision: "fp16".to_string(),
            },
        }
    }
}

pub struct ConfigManager {
    config: AppConfig,
    config_path: Option<String>,
}

impl ConfigManager {
    pub async fn load() -> Result<Self> {
        let config_path = env::var("CONFIG_PATH")
            .unwrap_or_else(|_| "config/production.toml".to_string());

        let config = if Path::new(&config_path).exists() {
            Self::load_from_file(&config_path).await?
        } else {
            Self::load_from_env()?
        };

        Ok(Self {
            config,
            config_path: Some(config_path),
        })
    }

    async fn load_from_file(path: &str) -> Result<AppConfig> {
        let content = fs::read_to_string(path).await?;
        let config: AppConfig = toml::from_str(&content)?;
        Ok(config)
    }

    fn load_from_env() -> Result<AppConfig> {
        let mut config = AppConfig::default();

        if let Ok(host) = env::var("SERVER_HOST") {
            config.server.host = host;
        }
        if let Ok(port) = env::var("SERVER_PORT") {
            config.server.port = port.parse()?;
        }
        if let Ok(max_streams) = env::var("MAX_CONCURRENT_STREAMS") {
            config.streaming.max_concurrent_streams = max_streams.parse()?;
        }
        if let Ok(cache_size) = env::var("CACHE_SIZE_MB") {
            config.cache.max_size_mb = cache_size.parse()?;
        }

        Ok(config)
    }

    pub fn get(&self) -> &AppConfig {
        &self.config
    }

    pub async fn reload(&mut self) -> Result<()> {
        if let Some(path) = &self.config_path {
            self.config = Self::load_from_file(path).await?;
        }
        Ok(())
    }
}