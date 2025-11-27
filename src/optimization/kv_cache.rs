use crate::types::KvCacheConfig;
use ndarray::{ArrayD, Axis};
use std::collections::HashMap;

/// Manager for Key-Value caches in Transformer models
pub struct KvCacheManager {
    config: KvCacheConfig,
    caches: HashMap<String, ArrayD<f32>>,
    current_length: usize,
}

impl KvCacheManager {
    /// Create a new KV cache manager
    pub fn new(config: KvCacheConfig) -> Self {
        Self {
            config,
            caches: HashMap::new(),
            current_length: 0,
        }
    }

    /// Reset the cache (e.g., for new sentence)
    pub fn reset(&mut self) {
        self.caches.clear();
        self.current_length = 0;
    }

    /// Update cache with new key/value tensors
    ///
    /// Expected shape: [batch_size, num_heads, seq_len, head_dim]
    pub fn update(&mut self, layer_name: &str, key: ArrayD<f32>, value: ArrayD<f32>) {
        if !self.config.enabled {
            return;
        }

        // Store or append key
        self.append_to_cache(&format!("{layer_name}_key"), key);

        // Store or append value
        self.append_to_cache(&format!("{layer_name}_value"), value);
    }

    /// Get current cache for a layer
    pub fn get(&self, layer_name: &str) -> Option<(&ArrayD<f32>, &ArrayD<f32>)> {
        if !self.config.enabled {
            return None;
        }

        let key = self.caches.get(&format!("{layer_name}_key"))?;
        let value = self.caches.get(&format!("{layer_name}_value"))?;

        Some((key, value))
    }

    /// Append tensor to existing cache along sequence dimension (axis 2)
    fn append_to_cache(&mut self, name: &str, new_data: ArrayD<f32>) {
        if let Some(existing) = self.caches.get_mut(name) {
            // Concatenate along sequence dimension (usually axis 2 for [B, H, S, D])
            // Note: This is a simplified implementation. In production, we would pre-allocate
            // and use slicing to avoid allocation on every step.
            tracing::debug!(
                "Concatenating shapes: existing={:?}, new={:?}",
                existing.shape(),
                new_data.shape()
            );
            match ndarray::concatenate(Axis(2), &[existing.view(), new_data.view()]) {
                Ok(concatenated) => {
                    *existing = concatenated;

                    // Check for eviction (sliding window)
                    let seq_len = existing.shape()[2];
                    if seq_len > self.config.max_cache_size {
                        // Truncate from beginning
                        let start = seq_len - self.config.max_cache_size;
                        let truncated = existing
                            .slice(ndarray::s![.., .., start.., ..])
                            .to_owned()
                            .into_dyn();
                        *existing = truncated;
                    }
                }
                Err(e) => {
                    tracing::error!("Failed to concatenate KV cache tensors: {}", e);
                }
            }
        } else {
            self.caches.insert(name.to_string(), new_data);
        }
    }

    /// Increment sequence length counter
    pub fn increment_length(&mut self, len: usize) {
        self.current_length += len;
    }

    /// Get current sequence length
    pub fn current_length(&self) -> usize {
        self.current_length
    }
}
