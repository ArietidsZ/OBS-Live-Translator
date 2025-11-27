//! ONNX Runtime integration for machine learning inference  

use anyhow::{Context, Result};
use ort::session::Session;
use std::path::Path;
use tracing::info;

use crate::profile::Profile;

/// ONNX model wrapper
pub struct OnnxModel {
    session: Session,
    profile: Profile,
    model_name: String,
}

/// Configuration for ONNX Runtime session
#[derive(Debug, Clone)]
pub struct OnnxConfig {
    pub profile: Profile,
    pub num_threads: usize,
}

impl OnnxModel {
    /// Load an ONNX model
    pub fn load<P: AsRef<Path>>(
        model_path: P,
        config: OnnxConfig,
    ) -> Result<Self> {
        let model_path = model_path.as_ref();
        let model_name = model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        info!(
            "🔧 Loading ONNX model: {} for profile {:?}",
            model_name, config.profile
        );

        // Build session
        let session = Session::builder()?
            .with_intra_threads(config.num_threads)?
            .commit_from_file(model_path)
            .with_context(|| format!("Failed to load model: {}", model_name))?;

        Ok(Self {
            session,
            profile: config.profile,
            model_name,
        })
    }

    /// Get the ONNX session reference
    pub fn session(&self) -> &Session {
        &self.session
    }

    /// Get model name
    pub fn model_name(&self) -> &str {
        &self.model_name
    }
}

impl Default for OnnxConfig {
    fn default() -> Self {
        Self {
            profile: Profile::Medium,
            num_threads: num_cpus::get(),
        }
    }
}

impl OnnxConfig {
    /// Create configuration for a specific profile
    pub fn for_profile(profile: Profile) -> Self {
        let num_threads = match profile {
            Profile::Low => 2,
            Profile::Medium => 4,
            Profile::High => num_cpus::get(),
        };

        Self {
            profile,
            num_threads,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onnx_config_default() {
        let config = OnnxConfig::default();
        assert_eq!(config.profile, Profile::Medium);
    }

    #[test]
    fn test_onnx_config_for_profile() {
        let low_config = OnnxConfig::for_profile(Profile::Low);
        assert_eq!(low_config.num_threads, 2);

        let high_config = OnnxConfig::for_profile(Profile::High);
        assert!(high_config.num_threads > 2);
    }
}
