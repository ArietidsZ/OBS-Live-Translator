//! Intelligent caching system for translation optimization

use anyhow::Result;
use dashmap::DashMap;
use std::sync::Arc;

use crate::{TranslatorConfig, TranslationResult};

/// High-performance intelligent cache for translations
pub struct IntelligentCache {
    translation_cache: Arc<DashMap<u64, TranslationResult>>,
    cache_hits: std::sync::atomic::AtomicU64,
    cache_misses: std::sync::atomic::AtomicU64,
    max_entries: usize,
}

impl IntelligentCache {
    /// Create new intelligent cache
    pub fn new(config: &TranslatorConfig) -> Result<Self> {
        Ok(Self {
            translation_cache: Arc::new(DashMap::new()),
            cache_hits: std::sync::atomic::AtomicU64::new(0),
            cache_misses: std::sync::atomic::AtomicU64::new(0),
            max_entries: config.batch_size * 1000, // Scale with batch size
        })
    }

    /// Get cached translation if available
    pub async fn get_translation(&self, fingerprint: &u64) -> Option<TranslationResult> {
        if let Some(result) = self.translation_cache.get(fingerprint) {
            self.cache_hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Some(result.clone())
        } else {
            self.cache_misses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            None
        }
    }

    /// Cache translation result
    pub async fn cache_translation(&self, fingerprint: &u64, result: &TranslationResult) {
        // Evict old entries if cache is full
        if self.translation_cache.len() >= self.max_entries {
            // Simple eviction: remove some random entries
            let keys_to_remove: Vec<u64> = self.translation_cache
                .iter()
                .take(self.max_entries / 10) // Remove 10%
                .map(|entry| *entry.key())
                .collect();
            
            for key in keys_to_remove {
                self.translation_cache.remove(&key);
            }
        }
        
        self.translation_cache.insert(*fingerprint, result.clone());
    }

    /// Get cache hit rate
    pub async fn get_hit_rate(&self) -> f32 {
        let hits = self.cache_hits.load(std::sync::atomic::Ordering::Relaxed);
        let misses = self.cache_misses.load(std::sync::atomic::Ordering::Relaxed);
        
        if hits + misses == 0 {
            0.0
        } else {
            hits as f32 / (hits + misses) as f32
        }
    }
}