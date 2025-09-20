//! Configuration management for OBS Live Translator

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::fs;

/// Main application configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub audio: AudioConfiguration,
    pub model: ModelConfiguration,
    pub server: ServerConfiguration,
    pub obs: ObsConfiguration,
    pub translation: TranslationConfiguration,
}

/// Audio processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioConfiguration {
    pub sample_rate: u32,
    pub channels: u16,
    pub frame_size_ms: f32,
    pub hop_length_ms: f32,
    pub enable_vad: bool,
    pub vad_sensitivity: f32,
    pub noise_reduction: bool,
    pub auto_gain_control: bool,
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfiguration {
    pub whisper_model_path: String,
    pub translation_model_path: Option<String>,
    pub device: String, // "cpu", "cuda", "coreml", "directml"
    pub batch_size: usize,
    pub max_concurrent_requests: usize,
    pub cache_size: usize,
}

/// Server configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfiguration {
    pub host: String,
    pub port: u16,
    pub max_connections: usize,
    pub websocket_buffer_size: usize,
    pub enable_cors: bool,
    pub cors_origins: Vec<String>,
}

/// OBS integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObsConfiguration {
    pub enable_plugin: bool,
    pub websocket_port: u16,
    pub password: Option<String>,
    pub auto_connect: bool,
    pub scene_name: String,
    pub source_name: String,
}

/// Translation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationConfiguration {
    pub source_language: Option<String>,
    pub target_languages: Vec<String>,
    pub auto_detect_language: bool,
    pub confidence_threshold: f32,
    pub max_text_length: usize,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            audio: AudioConfiguration::default(),
            model: ModelConfiguration::default(),
            server: ServerConfiguration::default(),
            obs: ObsConfiguration::default(),
            translation: TranslationConfiguration::default(),
        }
    }
}

impl Default for AudioConfiguration {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            channels: 1,
            frame_size_ms: 30.0,
            hop_length_ms: 10.0,
            enable_vad: true,
            vad_sensitivity: 0.5,
            noise_reduction: true,
            auto_gain_control: false,
        }
    }
}

impl Default for ModelConfiguration {
    fn default() -> Self {
        Self {
            whisper_model_path: "models/whisper-base.onnx".to_string(),
            translation_model_path: None,
            device: "cpu".to_string(),
            batch_size: 1,
            max_concurrent_requests: 4,
            cache_size: 100,
        }
    }
}

impl Default for ServerConfiguration {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            max_connections: 100,
            websocket_buffer_size: 8192,
            enable_cors: true,
            cors_origins: vec!["*".to_string()],
        }
    }
}

impl Default for ObsConfiguration {
    fn default() -> Self {
        Self {
            enable_plugin: false,
            websocket_port: 4455,
            password: None,
            auto_connect: false,
            scene_name: "Main Scene".to_string(),
            source_name: "Live Translation".to_string(),
        }
    }
}

impl Default for TranslationConfiguration {
    fn default() -> Self {
        Self {
            source_language: None, // Auto-detect
            target_languages: vec!["en".to_string()],
            auto_detect_language: true,
            confidence_threshold: 0.7,
            max_text_length: 1000,
        }
    }
}

/// Configuration manager
pub struct ConfigManager {
    config: AppConfig,
    config_path: PathBuf,
}

impl ConfigManager {
    /// Create new configuration manager
    pub fn new() -> Result<Self> {
        let config_path = Self::default_config_path()?;
        let config = if config_path.exists() {
            Self::load_from_file(&config_path)?
        } else {
            AppConfig::default()
        };

        Ok(Self {
            config,
            config_path,
        })
    }

    /// Create with custom config path
    pub fn with_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let config_path = path.as_ref().to_path_buf();
        let config = if config_path.exists() {
            Self::load_from_file(&config_path)?
        } else {
            AppConfig::default()
        };

        Ok(Self {
            config,
            config_path,
        })
    }

    /// Load configuration from file
    fn load_from_file<P: AsRef<Path>>(path: P) -> Result<AppConfig> {
        let content = fs::read_to_string(path.as_ref())
            .map_err(|e| anyhow!("Failed to read config file: {}", e))?;

        let config: AppConfig = toml::from_str(&content)
            .map_err(|e| anyhow!("Failed to parse config file: {}", e))?;

        Ok(config)
    }

    /// Save configuration to file
    pub fn save(&self) -> Result<()> {
        let content = toml::to_string_pretty(&self.config)
            .map_err(|e| anyhow!("Failed to serialize config: {}", e))?;

        // Create parent directory if it doesn't exist
        if let Some(parent) = self.config_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| anyhow!("Failed to create config directory: {}", e))?;
        }

        fs::write(&self.config_path, content)
            .map_err(|e| anyhow!("Failed to write config file: {}", e))?;

        Ok(())
    }

    /// Get configuration reference
    pub fn config(&self) -> &AppConfig {
        &self.config
    }

    /// Get mutable configuration reference
    pub fn config_mut(&mut self) -> &mut AppConfig {
        &mut self.config
    }

    /// Update configuration
    pub fn update<F>(&mut self, updater: F) -> Result<()>
    where
        F: FnOnce(&mut AppConfig),
    {
        updater(&mut self.config);
        self.save()
    }

    /// Get default configuration path
    fn default_config_path() -> Result<PathBuf> {
        let config_dir = if cfg!(target_os = "windows") {
            std::env::var("APPDATA")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("."))
        } else if cfg!(target_os = "macos") {
            std::env::var("HOME")
                .map(|home| PathBuf::from(home).join("Library/Application Support"))
                .unwrap_or_else(|_| PathBuf::from("."))
        } else {
            std::env::var("XDG_CONFIG_HOME")
                .map(PathBuf::from)
                .or_else(|_| {
                    std::env::var("HOME")
                        .map(|home| PathBuf::from(home).join(".config"))
                })
                .unwrap_or_else(|_| PathBuf::from("."))
        };

        Ok(config_dir.join("obs-live-translator").join("config.toml"))
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        // Validate audio configuration
        if self.config.audio.sample_rate == 0 {
            return Err(anyhow!("Audio sample rate must be greater than 0"));
        }

        if self.config.audio.channels == 0 {
            return Err(anyhow!("Audio channels must be greater than 0"));
        }

        // Validate model configuration
        if self.config.model.whisper_model_path.is_empty() {
            return Err(anyhow!("Whisper model path cannot be empty"));
        }

        if self.config.model.batch_size == 0 {
            return Err(anyhow!("Batch size must be greater than 0"));
        }

        // Validate server configuration
        if self.config.server.port == 0 {
            return Err(anyhow!("Server port must be greater than 0"));
        }

        // Validate translation configuration
        if self.config.translation.target_languages.is_empty() {
            return Err(anyhow!("At least one target language must be specified"));
        }

        Ok(())
    }

    /// Reset to default configuration
    pub fn reset_to_default(&mut self) -> Result<()> {
        self.config = AppConfig::default();
        self.save()
    }

    /// Get configuration file path
    pub fn config_path(&self) -> &Path {
        &self.config_path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_default_config() {
        let config = AppConfig::default();
        assert_eq!(config.audio.sample_rate, 16000);
        assert_eq!(config.server.port, 8080);
        assert!(config.audio.enable_vad);
    }

    #[test]
    fn test_config_serialization() {
        let config = AppConfig::default();
        let toml_str = toml::to_string(&config).unwrap();
        let parsed: AppConfig = toml::from_str(&toml_str).unwrap();

        assert_eq!(config.audio.sample_rate, parsed.audio.sample_rate);
        assert_eq!(config.server.port, parsed.server.port);
    }

    #[test]
    fn test_config_validation() {
        let config_manager = ConfigManager::new().unwrap();
        assert!(config_manager.validate().is_ok());

        // Test invalid configuration
        let mut invalid_config = AppConfig::default();
        invalid_config.audio.sample_rate = 0;

        let manager = ConfigManager {
            config: invalid_config,
            config_path: PathBuf::from("/tmp/test_config.toml"),
        };

        assert!(manager.validate().is_err());
    }

    #[test]
    fn test_config_save_load() {
        let temp_path = env::temp_dir().join("test_obs_config.toml");

        // Create and save config
        {
            let mut manager = ConfigManager::with_path(&temp_path).unwrap();
            manager.config_mut().server.port = 9090;
            manager.save().unwrap();
        }

        // Load and verify
        {
            let manager = ConfigManager::with_path(&temp_path).unwrap();
            assert_eq!(manager.config().server.port, 9090);
        }

        // Cleanup
        let _ = std::fs::remove_file(&temp_path);
    }
}