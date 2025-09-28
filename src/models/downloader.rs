//! Model downloader with resume support and progress tracking

use crate::models::ModelInfo;
use anyhow::Result;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{info, warn, error};

/// Model downloader with resume support and concurrent downloads
pub struct ModelDownloader {
    models_dir: PathBuf,
    download_progress: Arc<Mutex<DownloadProgress>>,
    client: reqwest::Client,
}

/// Download progress tracking
#[derive(Debug, Default)]
pub struct DownloadProgress {
    pub total_bytes: u64,
    pub downloaded_bytes: u64,
    pub current_file: String,
    pub download_speed_mbps: f64,
    pub eta_seconds: u64,
}

impl ModelDownloader {
    /// Create a new model downloader
    pub fn new(models_dir: &Path) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(3600)) // 1 hour timeout
            .build()?;

        Ok(Self {
            models_dir: models_dir.to_path_buf(),
            download_progress: Arc::new(Mutex::new(DownloadProgress::default())),
            client,
        })
    }

    /// Download a model with resume support
    pub async fn download_model(&mut self, model: &ModelInfo) -> Result<PathBuf> {
        info!("📥 Starting download: {} v{} ({} MB)", model.name, model.version, model.size_mb);

        let file_name = format!("{}_{}.onnx", model.name, model.version);
        let file_path = self.models_dir.join(&file_name);

        // Check if file already exists and is complete
        if file_path.exists() {
            let file_size = std::fs::metadata(&file_path)?.len();
            let expected_size = model.size_mb * 1024 * 1024;

            if file_size == expected_size {
                info!("✅ Model already downloaded: {}", model.name);
                return Ok(file_path);
            } else {
                warn!("⚠️ Partial download detected, resuming: {}", model.name);
            }
        }

        // Update progress tracking
        {
            let mut progress = self.download_progress.lock().await;
            progress.current_file = model.name.clone();
            progress.total_bytes = model.size_mb * 1024 * 1024;
            progress.downloaded_bytes = 0;
        }

        // Start download with resume support
        self.download_with_resume(&model.url, &file_path, model.size_mb).await?;

        info!("🎉 Download completed: {} ({} MB)", model.name, model.size_mb);
        Ok(file_path)
    }

    /// Download with resume support and progress tracking
    async fn download_with_resume(&self, url: &str, file_path: &Path, expected_size_mb: u64) -> Result<()> {
        let mut downloaded = 0u64;
        let expected_size = expected_size_mb * 1024 * 1024;

        // Check if partial file exists
        if file_path.exists() {
            downloaded = std::fs::metadata(file_path)?.len();
            info!("📂 Resuming download from {} MB", downloaded / 1024 / 1024);
        }

        // Create request with range header for resume
        let mut request = self.client.get(url);
        if downloaded > 0 {
            request = request.header("Range", format!("bytes={}-", downloaded));
        }

        let response = request.send().await?;

        if !response.status().is_success() && response.status() != reqwest::StatusCode::PARTIAL_CONTENT {
            return Err(anyhow::anyhow!("Download failed with status: {}", response.status()));
        }

        // Create or open file for writing
        let mut file = if downloaded > 0 {
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(file_path)?
        } else {
            std::fs::File::create(file_path)?
        };

        // Download with progress tracking
        let mut stream = response.bytes_stream();
        let start_time = std::time::Instant::now();

        use futures_util::StreamExt;
        use std::io::Write;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            file.write_all(&chunk)?;
            downloaded += chunk.len() as u64;

            // Update progress every 1MB
            if downloaded % (1024 * 1024) == 0 || downloaded >= expected_size {
                self.update_progress(downloaded, expected_size, start_time).await;
            }
        }

        file.flush()?;

        // Verify download size
        if downloaded != expected_size {
            warn!("⚠️ Download size mismatch: got {} bytes, expected {} bytes", downloaded, expected_size);
        }

        Ok(())
    }

    /// Update download progress
    async fn update_progress(&self, downloaded: u64, total: u64, start_time: std::time::Instant) {
        let elapsed = start_time.elapsed().as_secs_f64();
        let speed_mbps = if elapsed > 0.0 {
            (downloaded as f64 / 1024.0 / 1024.0) / elapsed
        } else {
            0.0
        };

        let remaining_bytes = total.saturating_sub(downloaded);
        let eta_seconds = if speed_mbps > 0.0 {
            (remaining_bytes as f64 / 1024.0 / 1024.0) / speed_mbps
        } else {
            0.0
        } as u64;

        let mut progress = self.download_progress.lock().await;
        progress.downloaded_bytes = downloaded;
        progress.total_bytes = total;
        progress.download_speed_mbps = speed_mbps;
        progress.eta_seconds = eta_seconds;

        let percent = (downloaded as f64 / total as f64) * 100.0;
        info!("📊 Download progress: {:.1}% ({:.1} MB/s, ETA: {}s)",
              percent, speed_mbps, eta_seconds);
    }

    /// Get current download progress
    pub async fn get_progress(&self) -> DownloadProgress {
        self.download_progress.lock().await.clone()
    }

    /// Download multiple models concurrently
    pub async fn download_models_concurrent(&mut self, models: &[ModelInfo], max_concurrent: usize) -> Result<Vec<PathBuf>> {
        info!("🚀 Starting concurrent download of {} models", models.len());

        let semaphore = Arc::new(tokio::sync::Semaphore::new(max_concurrent));
        let mut handles = Vec::new();

        for model in models {
            let model = model.clone();
            let semaphore = Arc::clone(&semaphore);
            let models_dir = self.models_dir.clone();
            let client = self.client.clone();

            let handle = tokio::spawn(async move {
                let _permit = semaphore.acquire().await.unwrap();

                let mut downloader = ModelDownloader {
                    models_dir,
                    download_progress: Arc::new(Mutex::new(DownloadProgress::default())),
                    client,
                };

                downloader.download_model(&model).await
            });

            handles.push(handle);
        }

        // Wait for all downloads to complete
        let mut results = Vec::new();
        for handle in handles {
            match handle.await? {
                Ok(path) => results.push(path),
                Err(e) => {
                    error!("❌ Download failed: {}", e);
                    return Err(e);
                }
            }
        }

        info!("🎉 All {} models downloaded successfully", results.len());
        Ok(results)
    }

    /// Verify download URL accessibility
    pub async fn verify_url(&self, url: &str) -> Result<bool> {
        let response = self.client.head(url).send().await?;
        Ok(response.status().is_success())
    }

    /// Get remote file size without downloading
    pub async fn get_remote_file_size(&self, url: &str) -> Result<u64> {
        let response = self.client.head(url).send().await?;

        if let Some(content_length) = response.headers().get("content-length") {
            let size_str = content_length.to_str()?;
            Ok(size_str.parse()?)
        } else {
            Err(anyhow::anyhow!("Content-Length header not found"))
        }
    }

    /// Cleanup partial downloads
    pub fn cleanup_partial_downloads(&self) -> Result<usize> {
        let mut cleaned = 0;

        for entry in std::fs::read_dir(&self.models_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() {
                if let Some(extension) = path.extension() {
                    if extension == "partial" || extension == "tmp" {
                        std::fs::remove_file(&path)?;
                        cleaned += 1;
                        info!("🗑️ Cleaned partial download: {:?}", path);
                    }
                }
            }
        }

        if cleaned > 0 {
            info!("✅ Cleaned {} partial downloads", cleaned);
        }

        Ok(cleaned)
    }
}

impl Clone for DownloadProgress {
    fn clone(&self) -> Self {
        Self {
            total_bytes: self.total_bytes,
            downloaded_bytes: self.downloaded_bytes,
            current_file: self.current_file.clone(),
            download_speed_mbps: self.download_speed_mbps,
            eta_seconds: self.eta_seconds,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_downloader_creation() {
        let temp_dir = TempDir::new().unwrap();
        let downloader = ModelDownloader::new(temp_dir.path()).unwrap();
        assert_eq!(downloader.models_dir, temp_dir.path());
    }

    #[tokio::test]
    async fn test_progress_tracking() {
        let temp_dir = TempDir::new().unwrap();
        let downloader = ModelDownloader::new(temp_dir.path()).unwrap();

        let progress = downloader.get_progress().await;
        assert_eq!(progress.downloaded_bytes, 0);
        assert_eq!(progress.total_bytes, 0);
    }

    #[test]
    fn test_cleanup_partial_downloads() {
        let temp_dir = TempDir::new().unwrap();

        // Create some partial files
        std::fs::write(temp_dir.path().join("model.onnx.partial"), b"partial").unwrap();
        std::fs::write(temp_dir.path().join("model.onnx.tmp"), b"tmp").unwrap();
        std::fs::write(temp_dir.path().join("complete.onnx"), b"complete").unwrap();

        let downloader = ModelDownloader::new(temp_dir.path()).unwrap();
        let cleaned = downloader.cleanup_partial_downloads().unwrap();

        assert_eq!(cleaned, 2);
        assert!(!temp_dir.path().join("model.onnx.partial").exists());
        assert!(!temp_dir.path().join("model.onnx.tmp").exists());
        assert!(temp_dir.path().join("complete.onnx").exists());
    }
}