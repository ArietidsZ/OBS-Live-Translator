//! Profile-Aware Configuration System
//!
//! This module provides advanced configuration management with:
//! - Profile inheritance (Low/Medium/High)
//! - TOML-based configuration files
//! - Runtime configuration updates with hot-reloading
//! - Configuration validation and migration
//! - Configuration versioning and backup

use crate::profile::Profile;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
// HashMap import removed as unused
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::{watch, RwLock};
use tokio::time::{interval, Duration};
use tracing::{debug, error, info, warn};

/// Configuration version for migration support
pub const CONFIG_VERSION: u32 = 1;

/// Profile-aware configuration with inheritance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileConfig {
    /// Configuration version
    pub version: u32,
    /// Configuration metadata
    pub metadata: ConfigMetadata,
    /// Base configuration (shared by all profiles)
    pub base: BaseConfiguration,
    /// Profile-specific overrides
    pub profiles: ProfileOverrides,
}

/// Configuration metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigMetadata {
    /// Configuration creation timestamp
    pub created_at: u64,
    /// Last modification timestamp
    pub modified_at: u64,
    /// Configuration author/source
    pub author: String,
    /// Configuration description
    pub description: String,
    /// Profile that generated this config
    pub generated_for_profile: Option<Profile>,
}

/// Base configuration shared by all profiles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaseConfiguration {
    /// Application settings
    pub app: AppSettings,
    /// Server settings
    pub server: ServerSettings,
    /// OBS integration settings
    pub obs: ObsSettings,
    /// Logging configuration
    pub logging: LoggingConfiguration,
}

/// Profile-specific configuration overrides
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileOverrides {
    /// Low profile configuration
    pub low: Option<ProfileSpecificConfig>,
    /// Medium profile configuration
    pub medium: Option<ProfileSpecificConfig>,
    /// High profile configuration
    pub high: Option<ProfileSpecificConfig>,
}

/// Profile-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileSpecificConfig {
    /// Audio processing configuration
    pub audio: Option<AudioConfiguration>,
    /// Model configuration
    pub models: Option<ModelConfiguration>,
    /// Performance settings
    pub performance: Option<PerformanceConfiguration>,
    /// Memory settings
    pub memory: Option<MemoryConfiguration>,
    /// Streaming settings
    pub streaming: Option<StreamingConfiguration>,
}

/// Application settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppSettings {
    /// Application name
    pub name: String,
    /// Application version
    pub version: String,
    /// Enable debug mode
    pub debug_mode: bool,
    /// Telemetry settings
    pub telemetry: TelemetrySettings,
}

/// Telemetry settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetrySettings {
    /// Enable performance metrics collection
    pub enable_metrics: bool,
    /// Enable error reporting
    pub enable_error_reporting: bool,
    /// Metrics collection interval (seconds)
    pub metrics_interval_s: u64,
}

/// Audio processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConfiguration {
    /// Sample rate (Hz)
    pub sample_rate: u32,
    /// Number of channels
    pub channels: u32,
    /// Frame size (milliseconds)
    pub frame_size_ms: f32,
    /// VAD configuration
    pub vad: VadConfiguration,
    /// Resampling configuration
    pub resampling: ResamplingConfiguration,
    /// Feature extraction configuration
    pub features: FeatureConfiguration,
}

/// VAD configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VadConfiguration {
    /// VAD engine type ("webrtc", "ten", "silero")
    pub engine: String,
    /// Voice activity threshold (0.0-1.0)
    pub threshold: f32,
    /// Minimum voice duration (ms)
    pub min_voice_duration_ms: u32,
    /// Minimum silence duration (ms)
    pub min_silence_duration_ms: u32,
}

/// Resampling configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResamplingConfiguration {
    /// Resampling algorithm ("linear", "cubic", "soxr")
    pub algorithm: String,
    /// Quality level (0-10)
    pub quality: u8,
    /// Enable SIMD optimization
    pub enable_simd: bool,
}

/// Feature extraction configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureConfiguration {
    /// Feature extractor type ("rustfft", "ipp", "enhanced")
    pub extractor: String,
    /// Number of mel bins
    pub n_mels: u32,
    /// Number of FFT points
    pub n_fft: u32,
    /// Hop length
    pub hop_length: u32,
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfiguration {
    /// ASR model settings
    pub asr: AsrModelConfig,
    /// Language detection model settings
    pub language_detection: LanguageDetectionConfig,
    /// Translation model settings
    pub translation: TranslationModelConfig,
}

/// ASR model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsrModelConfig {
    /// Model type ("whisper_tiny", "whisper_small", "parakeet")
    pub model_type: String,
    /// Model path
    pub model_path: String,
    /// Batch size
    pub batch_size: u32,
    /// Enable quantization
    pub enable_quantization: bool,
    /// Quantization precision ("int8", "fp16", "fp32")
    pub quantization_precision: String,
}

/// Language detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageDetectionConfig {
    /// Detection engine ("fasttext", "fusion")
    pub engine: String,
    /// Confidence threshold
    pub confidence_threshold: f32,
    /// Enable multimodal fusion
    pub enable_multimodal: bool,
}

/// Translation model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationModelConfig {
    /// Model type ("nllb", "m2m", "marian")
    pub model_type: String,
    /// Model path
    pub model_path: String,
    /// Source language (auto-detect if None)
    pub source_language: Option<String>,
    /// Target languages
    pub target_languages: Vec<String>,
    /// Enable caching
    pub enable_caching: bool,
    /// Cache size (MB)
    pub cache_size_mb: u32,
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfiguration {
    /// Number of worker threads
    pub worker_threads: u32,
    /// CPU affinity mask
    pub cpu_affinity: Option<Vec<u32>>,
    /// Thread priority ("normal", "high", "realtime")
    pub thread_priority: String,
    /// Enable performance monitoring
    pub enable_monitoring: bool,
    /// Performance target latency (ms)
    pub target_latency_ms: f32,
}

/// Memory configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfiguration {
    /// Global allocator ("system", "mimalloc", "tlsf")
    pub global_allocator: String,
    /// Audio buffer pool size (MB)
    pub audio_buffer_pool_mb: u32,
    /// Model cache size (MB)
    pub model_cache_mb: u32,
    /// Enable memory monitoring
    pub enable_monitoring: bool,
    /// Memory usage warning threshold (%)
    pub warning_threshold_percent: f32,
}

/// Streaming configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamingConfiguration {
    /// WebSocket buffer size (bytes)
    pub websocket_buffer_size: u32,
    /// Maximum connections
    pub max_connections: u32,
    /// Enable compression
    pub enable_compression: bool,
    /// Compression algorithm ("lz4", "zstd", "none")
    pub compression_algorithm: String,
    /// Opus codec settings
    pub opus: OpusCodecSettings,
}

/// Opus codec settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpusCodecSettings {
    /// Bitrate (6000-64000 bps)
    pub bitrate: u32,
    /// Frame duration (ms)
    pub frame_duration_ms: f32,
    /// Enable adaptive bitrate
    pub adaptive_bitrate: bool,
    /// Enable DTX (Discontinuous Transmission)
    pub enable_dtx: bool,
}

/// Server settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerSettings {
    /// Server host
    pub host: String,
    /// Server port
    pub port: u16,
    /// Enable HTTPS
    pub enable_https: bool,
    /// SSL certificate path
    pub ssl_cert_path: Option<String>,
    /// SSL private key path
    pub ssl_key_path: Option<String>,
    /// CORS settings
    pub cors: CorsSettings,
}

/// CORS settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorsSettings {
    /// Enable CORS
    pub enabled: bool,
    /// Allowed origins
    pub allowed_origins: Vec<String>,
    /// Allowed methods
    pub allowed_methods: Vec<String>,
    /// Allowed headers
    pub allowed_headers: Vec<String>,
}

/// OBS integration settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObsSettings {
    /// Enable OBS plugin
    pub enable_plugin: bool,
    /// OBS WebSocket port
    pub websocket_port: u16,
    /// OBS WebSocket password
    pub websocket_password: Option<String>,
    /// Auto-connect on startup
    pub auto_connect: bool,
    /// Default scene name
    pub scene_name: String,
    /// Default source name
    pub source_name: String,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfiguration {
    /// Log level ("trace", "debug", "info", "warn", "error")
    pub level: String,
    /// Log format ("json", "pretty", "compact")
    pub format: String,
    /// Enable file logging
    pub enable_file_logging: bool,
    /// Log file path
    pub log_file_path: Option<String>,
    /// Log file rotation
    pub rotation: LogRotationSettings,
}

/// Log rotation settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogRotationSettings {
    /// Enable rotation
    pub enabled: bool,
    /// Maximum file size (MB)
    pub max_file_size_mb: u32,
    /// Maximum number of files
    pub max_files: u32,
}

/// Configuration change event
#[derive(Debug, Clone)]
pub enum ConfigChangeEvent {
    /// Configuration was reloaded from file
    Reloaded(ProfileConfig),
    /// Profile was changed
    ProfileChanged(Profile),
    /// Configuration was validated and applied
    Applied(ProfileConfig),
    /// Configuration validation failed
    ValidationFailed(String),
}

/// Configuration manager with hot-reloading and versioning
pub struct ProfileConfigManager {
    /// Current configuration
    config: RwLock<ProfileConfig>,
    /// Current active profile
    active_profile: RwLock<Profile>,
    /// Configuration file path
    config_path: PathBuf,
    /// Configuration change notifier
    change_notifier: watch::Sender<ConfigChangeEvent>,
    /// Configuration change receiver
    change_receiver: watch::Receiver<ConfigChangeEvent>,
    /// Configuration backup directory
    backup_dir: PathBuf,
    /// Enable hot-reloading
    hot_reload_enabled: bool,
}

impl ProfileConfigManager {
    /// Create new configuration manager
    pub async fn new(config_path: PathBuf, profile: Profile) -> Result<Self> {
        let backup_dir = config_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("backups");

        // Create backup directory
        fs::create_dir_all(&backup_dir)?;

        let config = if config_path.exists() {
            Self::load_config_from_file(&config_path).await?
        } else {
            Self::create_default_config(profile).await?
        };

        let (change_notifier, change_receiver) =
            watch::channel(ConfigChangeEvent::Applied(config.clone()));

        let manager = Self {
            config: RwLock::new(config),
            active_profile: RwLock::new(profile),
            config_path,
            change_notifier,
            change_receiver,
            backup_dir,
            hot_reload_enabled: true,
        };

        // Start hot-reload monitoring if enabled
        if manager.hot_reload_enabled {
            manager.start_hot_reload_monitoring().await;
        }

        info!(
            "🔧 ProfileConfigManager initialized for {:?} profile",
            profile
        );
        Ok(manager)
    }

    /// Create default configuration for profile
    async fn create_default_config(profile: Profile) -> Result<ProfileConfig> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let metadata = ConfigMetadata {
            created_at: now,
            modified_at: now,
            author: "obs-live-translator".to_string(),
            description: format!("Default configuration for {:?} profile", profile),
            generated_for_profile: Some(profile),
        };

        let base = BaseConfiguration {
            app: AppSettings {
                name: "OBS Live Translator".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                debug_mode: cfg!(debug_assertions),
                telemetry: TelemetrySettings {
                    enable_metrics: true,
                    enable_error_reporting: false,
                    metrics_interval_s: 60,
                },
            },
            server: ServerSettings {
                host: "127.0.0.1".to_string(),
                port: 8080,
                enable_https: false,
                ssl_cert_path: None,
                ssl_key_path: None,
                cors: CorsSettings {
                    enabled: true,
                    allowed_origins: vec!["*".to_string()],
                    allowed_methods: vec![
                        "GET".to_string(),
                        "POST".to_string(),
                        "OPTIONS".to_string(),
                    ],
                    allowed_headers: vec!["*".to_string()],
                },
            },
            obs: ObsSettings {
                enable_plugin: false,
                websocket_port: 4455,
                websocket_password: None,
                auto_connect: false,
                scene_name: "Main Scene".to_string(),
                source_name: "Live Translation".to_string(),
            },
            logging: LoggingConfiguration {
                level: if cfg!(debug_assertions) {
                    "debug".to_string()
                } else {
                    "info".to_string()
                },
                format: "pretty".to_string(),
                enable_file_logging: true,
                log_file_path: Some("logs/obs-live-translator.log".to_string()),
                rotation: LogRotationSettings {
                    enabled: true,
                    max_file_size_mb: 10,
                    max_files: 5,
                },
            },
        };

        let profiles = Self::create_profile_overrides(profile).await?;

        Ok(ProfileConfig {
            version: CONFIG_VERSION,
            metadata,
            base,
            profiles,
        })
    }

    /// Create profile-specific overrides
    async fn create_profile_overrides(_profile: Profile) -> Result<ProfileOverrides> {
        let mut overrides = ProfileOverrides {
            low: None,
            medium: None,
            high: None,
        };

        // Create configuration for each profile
        for p in [Profile::Low, Profile::Medium, Profile::High] {
            let config = Self::create_profile_specific_config(p).await?;
            match p {
                Profile::Low => overrides.low = Some(config),
                Profile::Medium => overrides.medium = Some(config),
                Profile::High => overrides.high = Some(config),
            }
        }

        Ok(overrides)
    }

    /// Create profile-specific configuration
    async fn create_profile_specific_config(profile: Profile) -> Result<ProfileSpecificConfig> {
        let (sample_rate, frame_size_ms, vad_engine, resampling_algo, extractor) = match profile {
            Profile::Low => (16000, 30.0, "webrtc", "linear", "rustfft"),
            Profile::Medium => (16000, 20.0, "ten", "cubic", "enhanced"),
            Profile::High => (48000, 10.0, "silero", "soxr", "ipp"),
        };

        let (asr_model, batch_size, quantization) = match profile {
            Profile::Low => ("whisper_tiny", 1, "int8"),
            Profile::Medium => ("whisper_small", 2, "fp16"),
            Profile::High => ("parakeet", 4, "fp32"),
        };

        let (worker_threads, memory_mb, bitrate) = match profile {
            Profile::Low => (2, 512, 16000),
            Profile::Medium => (4, 2048, 32000),
            Profile::High => (8, 8192, 64000),
        };

        Ok(ProfileSpecificConfig {
            audio: Some(AudioConfiguration {
                sample_rate,
                channels: 1,
                frame_size_ms,
                vad: VadConfiguration {
                    engine: vad_engine.to_string(),
                    threshold: 0.5,
                    min_voice_duration_ms: 100,
                    min_silence_duration_ms: 200,
                },
                resampling: ResamplingConfiguration {
                    algorithm: resampling_algo.to_string(),
                    quality: match profile {
                        Profile::Low => 3,
                        Profile::Medium => 6,
                        Profile::High => 10,
                    },
                    enable_simd: profile != Profile::Low,
                },
                features: FeatureConfiguration {
                    extractor: extractor.to_string(),
                    n_mels: 80,
                    n_fft: 1024,
                    hop_length: 256,
                },
            }),
            models: Some(ModelConfiguration {
                asr: AsrModelConfig {
                    model_type: asr_model.to_string(),
                    model_path: format!("models/{}.onnx", asr_model),
                    batch_size,
                    enable_quantization: profile != Profile::High,
                    quantization_precision: quantization.to_string(),
                },
                language_detection: LanguageDetectionConfig {
                    engine: if profile == Profile::High {
                        "fusion"
                    } else {
                        "fasttext"
                    }
                    .to_string(),
                    confidence_threshold: 0.7,
                    enable_multimodal: profile == Profile::High,
                },
                translation: TranslationModelConfig {
                    model_type: match profile {
                        Profile::Low => "marian",
                        Profile::Medium => "m2m",
                        Profile::High => "nllb",
                    }
                    .to_string(),
                    model_path: "models/translation.onnx".to_string(),
                    source_language: None, // Auto-detect
                    target_languages: vec!["en".to_string(), "es".to_string(), "fr".to_string()],
                    enable_caching: true,
                    cache_size_mb: memory_mb / 4,
                },
            }),
            performance: Some(PerformanceConfiguration {
                worker_threads,
                cpu_affinity: None,
                thread_priority: match profile {
                    Profile::Low => "normal",
                    Profile::Medium => "high",
                    Profile::High => "realtime",
                }
                .to_string(),
                enable_monitoring: true,
                target_latency_ms: match profile {
                    Profile::Low => 500.0,
                    Profile::Medium => 200.0,
                    Profile::High => 100.0,
                },
            }),
            memory: Some(MemoryConfiguration {
                global_allocator: match profile {
                    Profile::Low => "system",
                    Profile::Medium => "mimalloc",
                    Profile::High => "tlsf",
                }
                .to_string(),
                audio_buffer_pool_mb: memory_mb / 8,
                model_cache_mb: memory_mb / 2,
                enable_monitoring: true,
                warning_threshold_percent: 80.0,
            }),
            streaming: Some(StreamingConfiguration {
                websocket_buffer_size: match profile {
                    Profile::Low => 4096,
                    Profile::Medium => 8192,
                    Profile::High => 16384,
                },
                max_connections: match profile {
                    Profile::Low => 10,
                    Profile::Medium => 50,
                    Profile::High => 200,
                },
                enable_compression: profile != Profile::Low,
                compression_algorithm: if profile == Profile::Low {
                    "none"
                } else {
                    "lz4"
                }
                .to_string(),
                opus: OpusCodecSettings {
                    bitrate,
                    frame_duration_ms: frame_size_ms,
                    adaptive_bitrate: profile != Profile::Low,
                    enable_dtx: true,
                },
            }),
        })
    }

    /// Load configuration from file
    async fn load_config_from_file(path: &Path) -> Result<ProfileConfig> {
        let content = fs::read_to_string(path)?;
        let mut config: ProfileConfig = toml::from_str(&content)?;

        // Migrate configuration if needed
        if config.version < CONFIG_VERSION {
            config = Self::migrate_config(config).await?;
        }

        Self::validate_config(&config).await?;
        Ok(config)
    }

    /// Migrate configuration to current version
    async fn migrate_config(mut config: ProfileConfig) -> Result<ProfileConfig> {
        warn!(
            "🔄 Migrating configuration from version {} to {}",
            config.version, CONFIG_VERSION
        );

        // Perform migration based on version differences
        while config.version < CONFIG_VERSION {
            config = match config.version {
                // Add migration logic for future versions
                _ => {
                    config.version = CONFIG_VERSION;
                    config
                }
            };
        }

        config.metadata.modified_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        info!("✅ Configuration migrated successfully");
        Ok(config)
    }

    /// Validate configuration
    async fn validate_config(config: &ProfileConfig) -> Result<()> {
        if config.version > CONFIG_VERSION {
            return Err(anyhow!(
                "Configuration version {} is newer than supported version {}",
                config.version,
                CONFIG_VERSION
            ));
        }

        // Validate base configuration
        if config.base.server.port == 0 {
            return Err(anyhow!("Server port must be greater than 0"));
        }

        if config.base.app.name.is_empty() {
            return Err(anyhow!("Application name cannot be empty"));
        }

        // Validate profile-specific configurations
        for (profile_name, profile_config) in [
            ("low", &config.profiles.low),
            ("medium", &config.profiles.medium),
            ("high", &config.profiles.high),
        ] {
            if let Some(profile_config) = profile_config {
                Self::validate_profile_config(profile_name, profile_config).await?;
            }
        }

        Ok(())
    }

    /// Validate profile-specific configuration
    async fn validate_profile_config(
        profile_name: &str,
        config: &ProfileSpecificConfig,
    ) -> Result<()> {
        if let Some(ref audio) = config.audio {
            if audio.sample_rate == 0 {
                return Err(anyhow!(
                    "Sample rate must be greater than 0 for {} profile",
                    profile_name
                ));
            }
            if audio.channels == 0 {
                return Err(anyhow!(
                    "Channels must be greater than 0 for {} profile",
                    profile_name
                ));
            }
        }

        if let Some(ref models) = config.models {
            if models.asr.model_path.is_empty() {
                return Err(anyhow!(
                    "ASR model path cannot be empty for {} profile",
                    profile_name
                ));
            }
            if models.asr.batch_size == 0 {
                return Err(anyhow!(
                    "ASR batch size must be greater than 0 for {} profile",
                    profile_name
                ));
            }
        }

        if let Some(ref performance) = config.performance {
            if performance.worker_threads == 0 {
                return Err(anyhow!(
                    "Worker threads must be greater than 0 for {} profile",
                    profile_name
                ));
            }
        }

        Ok(())
    }

    /// Save configuration to file
    pub async fn save_config(&self) -> Result<()> {
        let config = self.config.read().await;

        // Create backup before saving
        self.create_backup(&config).await?;

        let content = toml::to_string_pretty(&*config)?;

        // Create parent directory if it doesn't exist
        if let Some(parent) = self.config_path.parent() {
            fs::create_dir_all(parent)?;
        }

        fs::write(&self.config_path, content)?;

        info!("💾 Configuration saved to {:?}", self.config_path);
        Ok(())
    }

    /// Create configuration backup
    async fn create_backup(&self, config: &ProfileConfig) -> Result<()> {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();

        let backup_filename = format!("config_backup_{}.toml", timestamp);
        let backup_path = self.backup_dir.join(backup_filename);

        let content = toml::to_string_pretty(config)?;
        fs::write(&backup_path, content)?;

        debug!("📋 Configuration backup created: {:?}", backup_path);

        // Clean old backups (keep last 10)
        self.cleanup_old_backups().await?;
        Ok(())
    }

    /// Clean up old backup files
    async fn cleanup_old_backups(&self) -> Result<()> {
        let mut backups = Vec::new();

        if let Ok(entries) = fs::read_dir(&self.backup_dir) {
            for entry in entries.filter_map(Result::ok) {
                if let Some(name) = entry.file_name().to_str() {
                    if name.starts_with("config_backup_") && name.ends_with(".toml") {
                        if let Ok(metadata) = entry.metadata() {
                            if let Ok(modified) = metadata.modified() {
                                backups.push((entry.path(), modified));
                            }
                        }
                    }
                }
            }
        }

        // Sort by modification time (newest first)
        backups.sort_by(|a, b| b.1.cmp(&a.1));

        // Remove old backups (keep latest 10)
        for (path, _) in backups.into_iter().skip(10) {
            if let Err(e) = fs::remove_file(&path) {
                warn!("Failed to remove old backup {:?}: {}", path, e);
            }
        }

        Ok(())
    }

    /// Start hot-reload monitoring
    async fn start_hot_reload_monitoring(&self) {
        let config_path = self.config_path.clone();
        let change_notifier = self.change_notifier.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(1));
            let mut last_modified = SystemTime::UNIX_EPOCH;

            loop {
                interval.tick().await;

                if let Ok(metadata) = fs::metadata(&config_path) {
                    if let Ok(modified) = metadata.modified() {
                        if modified > last_modified {
                            last_modified = modified;

                            match Self::load_config_from_file(&config_path).await {
                                Ok(new_config) => {
                                    info!("🔄 Configuration hot-reloaded from file");
                                    let _ = change_notifier
                                        .send(ConfigChangeEvent::Reloaded(new_config));
                                }
                                Err(e) => {
                                    error!("❌ Failed to hot-reload configuration: {}", e);
                                    let _ = change_notifier
                                        .send(ConfigChangeEvent::ValidationFailed(e.to_string()));
                                }
                            }
                        }
                    }
                }
            }
        });
    }

    /// Get configuration for active profile
    pub async fn get_active_config(&self) -> Result<ProfileSpecificConfig> {
        let config = self.config.read().await;
        let profile = *self.active_profile.read().await;

        let profile_config = match profile {
            Profile::Low => &config.profiles.low,
            Profile::Medium => &config.profiles.medium,
            Profile::High => &config.profiles.high,
        };

        profile_config
            .clone()
            .ok_or_else(|| anyhow!("No configuration found for {:?} profile", profile))
    }

    /// Switch to different profile
    pub async fn switch_profile(&self, new_profile: Profile) -> Result<()> {
        {
            let mut profile = self.active_profile.write().await;
            *profile = new_profile;
        }

        let _active_config = self.get_active_config().await?;

        info!("🔄 Switched to {:?} profile", new_profile);
        let _ = self
            .change_notifier
            .send(ConfigChangeEvent::ProfileChanged(new_profile));

        Ok(())
    }

    /// Update configuration
    pub async fn update_config<F>(&self, updater: F) -> Result<()>
    where
        F: FnOnce(&mut ProfileConfig),
    {
        {
            let mut config = self.config.write().await;
            updater(&mut config);

            config.metadata.modified_at = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs();
        }

        self.save_config().await?;

        let config = self.config.read().await;
        let _ = self
            .change_notifier
            .send(ConfigChangeEvent::Applied(config.clone()));

        Ok(())
    }

    /// Get configuration change receiver
    pub fn get_change_receiver(&self) -> watch::Receiver<ConfigChangeEvent> {
        self.change_receiver.clone()
    }

    /// Get base configuration
    pub async fn get_base_config(&self) -> BaseConfiguration {
        let config = self.config.read().await;
        config.base.clone()
    }

    /// Get current profile
    pub async fn get_current_profile(&self) -> Profile {
        *self.active_profile.read().await
    }

    /// Enable/disable hot-reload
    pub async fn set_hot_reload(&mut self, enabled: bool) {
        self.hot_reload_enabled = enabled;
        if enabled && !self.hot_reload_enabled {
            self.start_hot_reload_monitoring().await;
        }
    }
}
