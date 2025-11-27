//! Model caching and version management

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tracing::{info, warn};

/// Model cache manager for efficient model storage and retrieval
pub struct ModelCache {
    cache_dir: PathBuf,
    metadata_file: PathBuf,
    cache_metadata: CacheMetadata,
}

/// Cache metadata for tracking cached models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheMetadata {
    pub models: HashMap<String, CachedModelInfo>,
    pub cache_size_mb: u64,
    pub last_cleanup: Option<chrono::DateTime<chrono::Utc>>,
    pub version: String,
}

/// Information about a cached model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedModelInfo {
    pub name: String,
    pub version: String,
    pub file_path: PathBuf,
    pub size_mb: u64,
    pub cached_at: chrono::DateTime<chrono::Utc>,
    pub last_accessed: chrono::DateTime<chrono::Utc>,
    pub access_count: u64,
    pub checksum: String,
    pub model_type: String,
}

/// Cache statistics
#[derive(Debug)]
pub struct CacheStats {
    pub total_models: usize,
    pub total_size_mb: u64,
    pub most_accessed_model: Option<String>,
    pub least_accessed_model: Option<String>,
    pub cache_hit_ratio: f32,
    pub average_model_size_mb: f32,
}

impl ModelCache {
    /// Create a new model cache
    pub fn new(cache_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(cache_dir)?;

        let metadata_file = cache_dir.join("cache_metadata.json");
        let cache_metadata = if metadata_file.exists() {
            Self::load_metadata(&metadata_file)?
        } else {
            CacheMetadata::default()
        };

        info!(
            "💾 Model cache initialized: {} models cached",
            cache_metadata.models.len()
        );

        Ok(Self {
            cache_dir: cache_dir.to_path_buf(),
            metadata_file,
            cache_metadata,
        })
    }

    /// Cache a model file
    pub fn cache_model(
        &mut self,
        name: &str,
        version: &str,
        source_path: &Path,
    ) -> Result<PathBuf> {
        let model_key = format!("{}_{}", name, version);
        let cached_filename = format!("{}.onnx", model_key);
        let cached_path = self.cache_dir.join(&cached_filename);

        // Copy model to cache
        std::fs::copy(source_path, &cached_path)?;

        // Calculate file size and checksum
        let metadata = std::fs::metadata(&cached_path)?;
        let size_mb = metadata.len() / 1024 / 1024;

        // Create cache entry
        let cached_info = CachedModelInfo {
            name: name.to_string(),
            version: version.to_string(),
            file_path: cached_path.clone(),
            size_mb,
            cached_at: chrono::Utc::now(),
            last_accessed: chrono::Utc::now(),
            access_count: 0,
            checksum: "calculated_checksum".to_string(), // Would be calculated in real implementation
            model_type: "onnx".to_string(),
        };

        // Update cache metadata
        self.cache_metadata
            .models
            .insert(model_key.clone(), cached_info);
        self.cache_metadata.cache_size_mb += size_mb;
        self.save_metadata()?;

        info!("💾 Model cached: {} v{} ({} MB)", name, version, size_mb);
        Ok(cached_path)
    }

    /// Get cached model path
    pub fn get_model_path(&mut self, name: &str, version: &str) -> Result<PathBuf> {
        let model_key = format!("{}_{}", name, version);

        if let Some(cached_info) = self.cache_metadata.models.get_mut(&model_key) {
            // Update access statistics
            cached_info.last_accessed = chrono::Utc::now();
            cached_info.access_count += 1;

            // Verify file still exists
            if cached_info.file_path.exists() {
                let file_path = cached_info.file_path.clone();
                self.save_metadata()?;
                return Ok(file_path);
            } else {
                warn!("⚠️ Cached model file missing: {:?}", cached_info.file_path);
                let should_remove = true;
                let _ = cached_info; // Release the mutable borrow
                if should_remove {
                    self.cache_metadata.models.remove(&model_key);
                    self.save_metadata()?;
                }
            }
        }

        Err(anyhow::anyhow!(
            "Model not found in cache: {} v{}",
            name,
            version
        ))
    }

    /// Check if model is cached
    pub fn is_cached(&self, name: &str, version: &str) -> bool {
        let model_key = format!("{}_{}", name, version);
        if let Some(cached_info) = self.cache_metadata.models.get(&model_key) {
            cached_info.file_path.exists()
        } else {
            false
        }
    }

    /// Remove a model from cache
    pub fn remove_model(&mut self, name: &str, version: &str) -> Result<()> {
        let model_key = format!("{}_{}", name, version);

        if let Some(cached_info) = self.cache_metadata.models.remove(&model_key) {
            // Remove file
            if cached_info.file_path.exists() {
                std::fs::remove_file(&cached_info.file_path)?;
            }

            // Update cache size
            self.cache_metadata.cache_size_mb = self
                .cache_metadata
                .cache_size_mb
                .saturating_sub(cached_info.size_mb);
            self.save_metadata()?;

            info!("🗑️ Model removed from cache: {} v{}", name, version);
        }

        Ok(())
    }

    /// Cleanup unused models based on current model set
    pub fn cleanup_unused_models(
        &mut self,
        current_models: &std::collections::HashSet<String>,
    ) -> Result<usize> {
        let mut removed_count = 0;
        let mut models_to_remove = Vec::new();

        // Find models not in current set
        for (model_key, _) in &self.cache_metadata.models {
            if !current_models.contains(model_key) {
                models_to_remove.push(model_key.clone());
            }
        }

        // Remove unused models
        for model_key in models_to_remove {
            if let Some(cached_info) = self.cache_metadata.models.remove(&model_key) {
                if cached_info.file_path.exists() {
                    std::fs::remove_file(&cached_info.file_path)?;
                }
                self.cache_metadata.cache_size_mb = self
                    .cache_metadata
                    .cache_size_mb
                    .saturating_sub(cached_info.size_mb);
                removed_count += 1;
                info!("🗑️ Removed unused model: {}", model_key);
            }
        }

        if removed_count > 0 {
            self.cache_metadata.last_cleanup = Some(chrono::Utc::now());
            self.save_metadata()?;
        }

        Ok(removed_count)
    }

    /// Cleanup based on LRU (Least Recently Used) policy
    pub fn cleanup_lru(&mut self, max_size_mb: u64) -> Result<usize> {
        if self.cache_metadata.cache_size_mb <= max_size_mb {
            return Ok(0);
        }

        info!(
            "🧹 Starting LRU cleanup: {} MB -> {} MB target",
            self.cache_metadata.cache_size_mb, max_size_mb
        );

        // Sort models by last accessed time (oldest first)
        let mut models_by_access: Vec<_> = self.cache_metadata.models.iter().collect();
        models_by_access.sort_by_key(|(_, info)| info.last_accessed);

        let mut removed_count = 0;
        let mut current_size = self.cache_metadata.cache_size_mb;

        for (model_key, cached_info) in models_by_access {
            if current_size <= max_size_mb {
                break;
            }

            // Remove the model
            if cached_info.file_path.exists() {
                std::fs::remove_file(&cached_info.file_path)?;
            }

            current_size = current_size.saturating_sub(cached_info.size_mb);
            removed_count += 1;

            info!(
                "🗑️ LRU removed: {} (last accessed: {})",
                model_key,
                cached_info.last_accessed.format("%Y-%m-%d %H:%M")
            );
        }

        // Remove from metadata (collect keys to remove first to avoid borrow checker issues)
        let keys_to_remove: Vec<String> = self
            .cache_metadata
            .models
            .iter()
            .filter_map(|(key, info)| {
                if !info.file_path.exists() {
                    Some(key.clone())
                } else {
                    None
                }
            })
            .collect();

        for key in keys_to_remove {
            self.cache_metadata.models.remove(&key);
        }

        self.cache_metadata.cache_size_mb = current_size;
        self.cache_metadata.last_cleanup = Some(chrono::Utc::now());
        self.save_metadata()?;

        info!("✅ LRU cleanup complete: {} models removed", removed_count);
        Ok(removed_count)
    }

    /// Get cache statistics
    pub fn get_stats(&self) -> CacheStats {
        let total_models = self.cache_metadata.models.len();
        let total_size_mb = self.cache_metadata.cache_size_mb;

        let (most_accessed, least_accessed) = if !self.cache_metadata.models.is_empty() {
            let mut models_by_access: Vec<_> = self.cache_metadata.models.iter().collect();
            models_by_access.sort_by_key(|(_, info)| info.access_count);

            let least = models_by_access.first().map(|(key, _)| (*key).clone());
            let most = models_by_access.last().map(|(key, _)| (*key).clone());

            (most, least)
        } else {
            (None, None)
        };

        let average_model_size_mb = if total_models > 0 {
            total_size_mb as f32 / total_models as f32
        } else {
            0.0
        };

        // Cache hit ratio would be calculated based on access patterns
        // For now, we'll use a simplified calculation
        let total_accesses: u64 = self
            .cache_metadata
            .models
            .values()
            .map(|info| info.access_count)
            .sum();

        let cache_hit_ratio = if total_accesses > 0 {
            // Simplified: assume each cached model represents successful hits
            total_models as f32 / (total_accesses + total_models as u64) as f32
        } else {
            0.0
        };

        CacheStats {
            total_models,
            total_size_mb,
            most_accessed_model: most_accessed,
            least_accessed_model: least_accessed,
            cache_hit_ratio,
            average_model_size_mb,
        }
    }

    /// Validate cache integrity
    pub fn validate_cache(&mut self) -> Result<CacheValidationResult> {
        let mut result = CacheValidationResult::default();
        let mut models_to_remove = Vec::new();

        for (model_key, cached_info) in &self.cache_metadata.models {
            result.total_checked += 1;

            // Check if file exists
            if !cached_info.file_path.exists() {
                result.missing_files += 1;
                models_to_remove.push(model_key.clone());
                continue;
            }

            // Check file size
            match std::fs::metadata(&cached_info.file_path) {
                Ok(metadata) => {
                    let actual_size_mb = metadata.len() / 1024 / 1024;
                    if actual_size_mb != cached_info.size_mb {
                        result.size_mismatches += 1;
                        warn!(
                            "⚠️ Size mismatch for {}: expected {} MB, got {} MB",
                            model_key, cached_info.size_mb, actual_size_mb
                        );
                    } else {
                        result.valid_models += 1;
                    }
                }
                Err(_) => {
                    result.missing_files += 1;
                    models_to_remove.push(model_key.clone());
                }
            }
        }

        // Remove invalid entries
        for model_key in models_to_remove {
            self.cache_metadata.models.remove(&model_key);
        }

        if result.missing_files > 0 || result.size_mismatches > 0 {
            self.save_metadata()?;
        }

        result.integrity_score = if result.total_checked > 0 {
            result.valid_models as f32 / result.total_checked as f32
        } else {
            1.0
        };

        Ok(result)
    }

    /// Load cache metadata from file
    fn load_metadata(metadata_file: &Path) -> Result<CacheMetadata> {
        let content = std::fs::read_to_string(metadata_file)?;
        let metadata: CacheMetadata = serde_json::from_str(&content)?;
        Ok(metadata)
    }

    /// Save cache metadata to file
    fn save_metadata(&self) -> Result<()> {
        let content = serde_json::to_string_pretty(&self.cache_metadata)?;
        std::fs::write(&self.metadata_file, content)?;
        Ok(())
    }

    /// Export cache metadata for backup
    pub fn export_metadata(&self, export_path: &Path) -> Result<()> {
        let content = serde_json::to_string_pretty(&self.cache_metadata)?;
        std::fs::write(export_path, content)?;
        info!("📤 Cache metadata exported to: {:?}", export_path);
        Ok(())
    }

    /// Import cache metadata from backup
    pub fn import_metadata(&mut self, import_path: &Path) -> Result<()> {
        let imported_metadata = Self::load_metadata(import_path)?;

        // Validate imported models exist
        let mut valid_models = HashMap::new();
        for (key, info) in imported_metadata.models {
            if info.file_path.exists() {
                valid_models.insert(key, info);
            }
        }

        self.cache_metadata.models = valid_models;
        self.recalculate_cache_size();
        self.save_metadata()?;

        info!("📥 Cache metadata imported from: {:?}", import_path);
        Ok(())
    }

    /// Recalculate total cache size
    fn recalculate_cache_size(&mut self) {
        let total_size: u64 = self
            .cache_metadata
            .models
            .values()
            .map(|info| info.size_mb)
            .sum();
        self.cache_metadata.cache_size_mb = total_size;
    }
}

impl Default for CacheMetadata {
    fn default() -> Self {
        Self {
            models: HashMap::new(),
            cache_size_mb: 0,
            last_cleanup: None,
            version: "1.0.0".to_string(),
        }
    }
}

/// Cache validation result
#[derive(Debug, Default)]
pub struct CacheValidationResult {
    pub total_checked: usize,
    pub valid_models: usize,
    pub missing_files: usize,
    pub size_mismatches: usize,
    pub integrity_score: f32,
}

impl CacheValidationResult {
    /// Check if cache is healthy
    pub fn is_healthy(&self) -> bool {
        self.integrity_score >= 0.95 && self.missing_files == 0
    }

    /// Get validation summary
    pub fn summary(&self) -> String {
        format!(
            "Cache validation: {}/{} valid ({:.1}% integrity), {} missing, {} size mismatches",
            self.valid_models,
            self.total_checked,
            self.integrity_score * 100.0,
            self.missing_files,
            self.size_mismatches
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_cache_creation() {
        let temp_dir = TempDir::new().unwrap();
        let cache = ModelCache::new(temp_dir.path()).unwrap();
        assert_eq!(cache.cache_metadata.models.len(), 0);
    }

    #[test]
    fn test_cache_model() {
        let temp_dir = TempDir::new().unwrap();
        let mut cache = ModelCache::new(temp_dir.path()).unwrap();

        // Create a test model file
        let test_model = temp_dir.path().join("test_model.onnx");
        std::fs::write(&test_model, b"test model content").unwrap();

        // Cache the model
        let cached_path = cache
            .cache_model("test_model", "1.0.0", &test_model)
            .unwrap();
        assert!(cached_path.exists());
        assert!(cache.is_cached("test_model", "1.0.0"));
    }

    #[test]
    fn test_cache_removal() {
        let temp_dir = TempDir::new().unwrap();
        let mut cache = ModelCache::new(temp_dir.path()).unwrap();

        // Create and cache a test model
        let test_model = temp_dir.path().join("test_model.onnx");
        std::fs::write(&test_model, b"test model content").unwrap();
        cache
            .cache_model("test_model", "1.0.0", &test_model)
            .unwrap();

        // Remove the model
        cache.remove_model("test_model", "1.0.0").unwrap();
        assert!(!cache.is_cached("test_model", "1.0.0"));
    }

    #[test]
    fn test_cache_stats() {
        let temp_dir = TempDir::new().unwrap();
        let cache = ModelCache::new(temp_dir.path()).unwrap();

        let stats = cache.get_stats();
        assert_eq!(stats.total_models, 0);
        assert_eq!(stats.total_size_mb, 0);
    }

    #[test]
    fn test_cache_validation() {
        let temp_dir = TempDir::new().unwrap();
        let mut cache = ModelCache::new(temp_dir.path()).unwrap();

        let validation_result = cache.validate_cache().unwrap();
        assert!(validation_result.is_healthy());
        assert_eq!(validation_result.total_checked, 0);
    }
}
