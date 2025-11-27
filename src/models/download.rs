//! Model download utilities

use crate::{Error, Result};
use sha2::{Digest, Sha256};
use std::path::Path;
use tokio::io::AsyncWriteExt;

/// Model file definition
pub struct ModelFile {
    pub name: String,
    pub url: String,
    pub sha256: String,
}

/// Download a model file with verification
pub async fn download_model(model: &ModelFile, destination: &Path) -> Result<()> {
    tracing::info!("Downloading model: {}", model.name);
    tracing::info!("URL: {}", model.url);

    // Download file
    let response = reqwest::get(&model.url).await?;

    if !response.status().is_success() {
        return Err(Error::ModelLoad(format!(
            "Failed to download {}: HTTP {}",
            model.name,
            response.status()
        )));
    }

    let bytes = response.bytes().await?;

    // Verify SHA256
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    let hash = format!("{:x}", hasher.finalize());

    if hash != model.sha256 {
        return Err(Error::ModelLoad(format!(
            "SHA256 mismatch for {}: expected {}, got {}",
            model.name, model.sha256, hash
        )));
    }

    // Write to file
    let mut file = tokio::fs::File::create(destination).await?;
    file.write_all(&bytes).await?;

    tracing::info!("Successfully downloaded and verified: {}", model.name);
    Ok(())
}

/// Get model definitions for a specific profile
pub fn get_profile_models(profile: crate::Profile) -> Vec<ModelFile> {
    let mut models = vec![
        // Silero VAD (shared across all profiles)
        ModelFile {
            name: "Silero VAD".to_string(),
            url: "https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx"
                .to_string(),
            sha256: "placeholder_sha256".to_string(), // TODO: Add real SHA256
        },
    ];

    match profile {
        crate::Profile::Low => {
            models.extend(vec![
                ModelFile {
                    name: "Distil-Whisper Large v3 Encoder (INT8)".to_string(),
                    url: "https://huggingface.co/models/distil-whisper-large-v3".to_string(),
                    sha256: "placeholder".to_string(),
                },
                ModelFile {
                    name: "MADLAD-400 INT8".to_string(),
                    url: "https://huggingface.co/google/madlad400".to_string(),
                    sha256: "placeholder".to_string(),
                },
            ]);
        }
        crate::Profile::Medium => {
            models.extend(vec![
                ModelFile {
                    name: "Parakeet TDT 0.6B FP16".to_string(),
                    url: "https://huggingface.co/nvidia/parakeet-tdt".to_string(),
                    sha256: "placeholder".to_string(),
                },
                ModelFile {
                    name: "NLLB-200 FP16".to_string(),
                    url: "https://huggingface.co/facebook/nllb-200".to_string(),
                    sha256: "placeholder".to_string(),
                },
            ]);
        }
        crate::Profile::High => {
            models.extend(vec![
                ModelFile {
                    name: "Canary Qwen 2.5B BF16".to_string(),
                    url: "https://huggingface.co/nvidia/canary-qwen".to_string(),
                    sha256: "placeholder".to_string(),
                },
                ModelFile {
                    name: "MADLAD-400 BF16".to_string(),
                    url: "https://huggingface.co/google/madlad400".to_string(),
                    sha256: "placeholder".to_string(),
                },
            ]);
        }
    }

    models
}
