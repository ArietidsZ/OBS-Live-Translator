//! Configuration management
//! Enhanced in Part 1.8 with comprehensive TOML configuration

pub mod app_config; // Part 1.8: Comprehensive TOML configuration with auto-detection

pub use app_config::AppConfig;

use crate::{types::TranslatorConfig, Result};
use std::path::Path;

/// Load configuration from file
pub fn load_config<P: AsRef<Path>>(path: P) -> Result<TranslatorConfig> {
    let contents = std::fs::read_to_string(path)?;
    let config: TranslatorConfig = toml::from_str(&contents)?;
    Ok(config)
}

/// Save configuration to file
pub fn save_config<P: AsRef<Path>>(config: &TranslatorConfig, path: P) -> Result<()> {
    let contents = toml::to_string_pretty(config)?;
    std::fs::write(path, contents)?;
    Ok(())
}

/// Default configuration file paths
pub mod paths {
    use std::path::PathBuf;

    pub fn default_config_path() -> PathBuf {
        PathBuf::from("./config.toml")
    }

    pub fn default_model_path() -> PathBuf {
        PathBuf::from("./models")
    }
}
