//! Translation caching system
//!
//! This module provides intelligent caching for translation results:
//! - Sequence caching for repeated phrases
//! - Context-aware caching
//! - Quality-based cache eviction
//! - Performance optimization

use super::{LanguagePair, TranslationResult};
use std::collections::HashMap;
use std::time::Instant;
use tracing::{debug, info};

/// Translation cache with advanced features
pub struct TranslationCache {
    /// Main translation cache
    translations: HashMap<String, CachedTranslation>,
    /// Sequence cache for common phrases
    sequences: HashMap<String, SequenceCache>,
    /// Context-aware cache
    contexts: HashMap<String, ContextCache>,
    /// Cache configuration
    config: CacheConfig,
    /// Cache statistics
    stats: CacheStats,
}

/// Cached translation entry
#[derive(Debug, Clone)]
pub struct CachedTranslation {
    pub result: TranslationResult,
    pub timestamp: Instant,
    pub access_count: u32,
    pub quality_score: f32,
    pub ttl_seconds: u64,
}

/// Sequence cache for repeated phrases
#[derive(Debug, Clone)]
pub struct SequenceCache {
    pub translations: HashMap<String, String>,
    pub confidence_scores: HashMap<String, f32>,
    pub last_used: Instant,
}

/// Context-aware cache
#[derive(Debug, Clone)]
pub struct ContextCache {
    pub context_hash: u64,
    pub translations: HashMap<String, TranslationResult>,
    pub last_updated: Instant,
}

/// Cache configuration
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum cache size
    pub max_entries: usize,
    /// Default TTL in seconds
    pub default_ttl_seconds: u64,
    /// High-quality translation TTL
    pub high_quality_ttl_seconds: u64,
    /// Enable sequence caching
    pub enable_sequence_cache: bool,
    /// Enable context caching
    pub enable_context_cache: bool,
    /// Quality threshold for caching
    pub min_quality_threshold: f32,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 10000,
            default_ttl_seconds: 3600,       // 1 hour
            high_quality_ttl_seconds: 86400, // 24 hours
            enable_sequence_cache: true,
            enable_context_cache: true,
            min_quality_threshold: 0.7,
        }
    }
}

/// Cache performance statistics
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    pub total_lookups: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub evictions: u64,
    pub memory_usage_mb: f64,
}

impl TranslationCache {
    /// Create a new translation cache
    pub fn new(config: CacheConfig) -> Self {
        info!(
            "🗄️ Initializing translation cache: max_entries={}, ttl={}s",
            config.max_entries, config.default_ttl_seconds
        );

        Self {
            translations: HashMap::new(),
            sequences: HashMap::new(),
            contexts: HashMap::new(),
            config,
            stats: CacheStats::default(),
        }
    }

    /// Get cached translation
    pub fn get(&mut self, key: &str) -> Option<TranslationResult> {
        self.stats.total_lookups += 1;

        if let Some(cached) = self.translations.get_mut(key) {
            // Check if entry is still valid
            if cached.timestamp.elapsed().as_secs() < cached.ttl_seconds {
                cached.access_count += 1;
                self.stats.cache_hits += 1;

                let mut result = cached.result.clone();
                result.processing_time_ms = 0.5; // Cache hit is very fast

                debug!(
                    "Cache hit: {} (accessed {} times)",
                    key, cached.access_count
                );
                return Some(result);
            } else {
                // Entry expired, remove it
                self.translations.remove(key);
                self.stats.evictions += 1;
            }
        }

        self.stats.cache_misses += 1;
        None
    }

    /// Cache a translation result
    pub fn put(&mut self, key: String, result: TranslationResult) {
        // Only cache high-quality translations
        if result.confidence < self.config.min_quality_threshold {
            return;
        }

        // Determine TTL based on quality
        let ttl_seconds = if result.confidence >= 0.9 {
            self.config.high_quality_ttl_seconds
        } else {
            self.config.default_ttl_seconds
        };

        let cached = CachedTranslation {
            result: result.clone(),
            timestamp: Instant::now(),
            access_count: 0,
            quality_score: result.confidence,
            ttl_seconds,
        };

        self.translations.insert(key, cached);

        // Cache management
        if self.translations.len() > self.config.max_entries {
            self.evict_least_valuable();
        }

        // Update sequence cache if enabled
        if self.config.enable_sequence_cache {
            self.update_sequence_cache(&result);
        }
    }

    /// Get cached sequence translation
    pub fn get_sequence(&self, language_pair: &LanguagePair, sequence: &str) -> Option<String> {
        if !self.config.enable_sequence_cache {
            return None;
        }

        let key = format!("{}:{}", language_pair.source, language_pair.target);
        if let Some(seq_cache) = self.sequences.get(&key) {
            seq_cache.translations.get(sequence).cloned()
        } else {
            None
        }
    }

    /// Cache a sequence translation
    pub fn put_sequence(
        &mut self,
        language_pair: &LanguagePair,
        source: String,
        target: String,
        confidence: f32,
    ) {
        if !self.config.enable_sequence_cache || confidence < self.config.min_quality_threshold {
            return;
        }

        let key = format!("{}:{}", language_pair.source, language_pair.target);

        let seq_cache = self.sequences.entry(key).or_insert_with(|| SequenceCache {
            translations: HashMap::new(),
            confidence_scores: HashMap::new(),
            last_used: Instant::now(),
        });

        seq_cache.translations.insert(source.clone(), target);
        seq_cache.confidence_scores.insert(source, confidence);
        seq_cache.last_used = Instant::now();
    }

    /// Get context-aware cache
    pub fn get_context(&self, context_hash: u64, text: &str) -> Option<TranslationResult> {
        if !self.config.enable_context_cache {
            return None;
        }

        let key = context_hash.to_string();
        if let Some(ctx_cache) = self.contexts.get(&key) {
            ctx_cache.translations.get(text).cloned()
        } else {
            None
        }
    }

    /// Cache with context
    pub fn put_context(&mut self, context_hash: u64, text: String, result: TranslationResult) {
        if !self.config.enable_context_cache
            || result.confidence < self.config.min_quality_threshold
        {
            return;
        }

        let key = context_hash.to_string();

        let ctx_cache = self.contexts.entry(key).or_insert_with(|| ContextCache {
            context_hash,
            translations: HashMap::new(),
            last_updated: Instant::now(),
        });

        ctx_cache.translations.insert(text, result);
        ctx_cache.last_updated = Instant::now();
    }

    /// Update sequence cache from translation result
    fn update_sequence_cache(&mut self, result: &TranslationResult) {
        // Extract common phrases from the translation
        let source_phrases = self.extract_phrases(&result.translated_text); // Note: would use source text in real implementation
        let target_phrases = self.extract_phrases(&result.translated_text);

        let language_pair = LanguagePair::new(&result.source_language, &result.target_language);

        // Cache phrase-level translations
        for (source_phrase, target_phrase) in source_phrases.iter().zip(target_phrases.iter()) {
            if source_phrase.len() > 5 && target_phrase.len() > 5 {
                // Only cache substantial phrases
                self.put_sequence(
                    &language_pair,
                    source_phrase.clone(),
                    target_phrase.clone(),
                    result.confidence,
                );
            }
        }
    }

    /// Extract phrases from text
    fn extract_phrases(&self, text: &str) -> Vec<String> {
        // Simple phrase extraction (in real implementation, would use NLP techniques)
        let mut phrases = Vec::new();

        // Extract by punctuation
        for sentence in text.split(&['.', '!', '?'][..]) {
            let sentence = sentence.trim();
            if sentence.len() > 10 {
                phrases.push(sentence.to_string());
            }
        }

        // Extract by common phrase patterns
        for chunk in text.split(&[',', ';'][..]) {
            let chunk = chunk.trim();
            if chunk.len() > 5 && chunk.len() < 50 {
                phrases.push(chunk.to_string());
            }
        }

        phrases
    }

    /// Evict least valuable cache entries
    fn evict_least_valuable(&mut self) {
        let entries: Vec<_> = self
            .translations
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        // Sort by value score (quality * access_count / age)
        let mut sorted_entries = entries;
        sorted_entries.sort_by(|a, b| {
            let score_a = self.calculate_value_score(&a.1);
            let score_b = self.calculate_value_score(&b.1);
            score_a
                .partial_cmp(&score_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Remove least valuable 10% of entries
        let mut to_remove = (sorted_entries.len() / 10).max(1);
        to_remove = to_remove.min(sorted_entries.len());

        for (key, _) in sorted_entries.iter().take(to_remove) {
            self.translations.remove(key);
            self.stats.evictions += 1;
        }

        debug!("Evicted {} cache entries", to_remove);
    }

    /// Calculate value score for cache entry
    fn calculate_value_score(&self, cached: &CachedTranslation) -> f32 {
        let age_seconds = cached.timestamp.elapsed().as_secs() as f32;
        let age_factor = 1.0 / (1.0 + age_seconds / 3600.0); // Decay over hours

        cached.quality_score * (cached.access_count as f32 + 1.0).log2() * age_factor
    }

    /// Get cache statistics
    pub fn stats(&self) -> &CacheStats {
        &self.stats
    }

    /// Get cache hit rate
    pub fn hit_rate(&self) -> f32 {
        if self.stats.total_lookups > 0 {
            self.stats.cache_hits as f32 / self.stats.total_lookups as f32
        } else {
            0.0
        }
    }

    /// Clear all caches
    pub fn clear(&mut self) {
        self.translations.clear();
        self.sequences.clear();
        self.contexts.clear();
        self.stats = CacheStats::default();
        info!("🗄️ Translation cache cleared");
    }

    /// Perform cache maintenance
    pub fn maintenance(&mut self) {
        let start_time = Instant::now();
        let initial_size = self.translations.len();

        // Remove expired entries
        let now = Instant::now();
        self.translations.retain(|_, cached| {
            now.duration_since(cached.timestamp).as_secs() < cached.ttl_seconds
        });

        // Clean up sequence caches
        self.sequences.retain(|_, seq_cache| {
            now.duration_since(seq_cache.last_used).as_secs() < 7200 // 2 hours
        });

        // Clean up context caches
        self.contexts.retain(|_, ctx_cache| {
            now.duration_since(ctx_cache.last_updated).as_secs() < 3600 // 1 hour
        });

        let final_size = self.translations.len();
        let removed = initial_size - final_size;

        if removed > 0 {
            debug!(
                "Cache maintenance: removed {} expired entries in {:.2}ms",
                removed,
                start_time.elapsed().as_secs_f32() * 1000.0
            );
            self.stats.evictions += removed as u64;
        }

        // Update memory usage estimate
        self.update_memory_usage_estimate();
    }

    /// Update memory usage estimate
    fn update_memory_usage_estimate(&mut self) {
        // Rough estimate of memory usage
        let translation_memory = self.translations.len() * 1024; // ~1KB per entry
        let sequence_memory = self.sequences.len() * 512; // ~0.5KB per sequence cache
        let context_memory = self.contexts.len() * 2048; // ~2KB per context cache

        self.stats.memory_usage_mb =
            (translation_memory + sequence_memory + context_memory) as f64 / 1024.0 / 1024.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::translation::TranslationMetrics;

    fn create_test_result(text: &str, confidence: f32) -> TranslationResult {
        TranslationResult {
            translated_text: text.to_string(),
            source_language: "en".to_string(),
            target_language: "es".to_string(),
            confidence,
            word_alignments: Vec::new(),
            processing_time_ms: 100.0,
            model_name: "test-model".to_string(),
            metrics: TranslationMetrics::default(),
        }
    }

    #[test]
    fn test_cache_basic_operations() {
        let mut cache = TranslationCache::new(CacheConfig::default());

        let result = create_test_result("hola mundo", 0.9);
        let key = "en:es:hello world".to_string();

        // Test cache miss
        assert!(cache.get(&key).is_none());
        assert_eq!(cache.stats().cache_misses, 1);

        // Test cache put and hit
        cache.put(key.clone(), result.clone());
        let cached_result = cache.get(&key);
        assert!(cached_result.is_some());
        assert_eq!(cached_result.unwrap().translated_text, "hola mundo");
        assert_eq!(cache.stats().cache_hits, 1);
    }

    #[test]
    fn test_quality_threshold() {
        let mut cache = TranslationCache::new(CacheConfig::default());

        let low_quality = create_test_result("bad translation", 0.3);
        let high_quality = create_test_result("good translation", 0.9);

        cache.put("key1".to_string(), low_quality);
        cache.put("key2".to_string(), high_quality);

        // Low quality should not be cached
        assert!(cache.get("key1").is_none());
        // High quality should be cached
        assert!(cache.get("key2").is_some());
    }

    #[test]
    fn test_sequence_cache() {
        let mut cache = TranslationCache::new(CacheConfig::default());
        let pair = LanguagePair::new("en", "es");

        cache.put_sequence(&pair, "hello".to_string(), "hola".to_string(), 0.9);

        let cached = cache.get_sequence(&pair, "hello");
        assert_eq!(cached, Some("hola".to_string()));
    }

    #[test]
    fn test_cache_eviction() {
        let mut config = CacheConfig::default();
        config.max_entries = 2; // Very small cache for testing

        let mut cache = TranslationCache::new(config);

        // Fill cache beyond capacity
        cache.put("key1".to_string(), create_test_result("text1", 0.8));
        cache.put("key2".to_string(), create_test_result("text2", 0.9));
        cache.put("key3".to_string(), create_test_result("text3", 0.7));

        // Should have evicted some entries
        assert!(cache.stats().evictions > 0);
    }

    #[test]
    fn test_hit_rate_calculation() {
        let mut cache = TranslationCache::new(CacheConfig::default());

        // Start with 0 hit rate
        assert_eq!(cache.hit_rate(), 0.0);

        let result = create_test_result("test", 0.9);
        cache.put("test_key".to_string(), result);

        // Miss first, then hit
        cache.get("nonexistent"); // Miss
        cache.get("test_key"); // Hit

        assert!(cache.hit_rate() > 0.0 && cache.hit_rate() < 1.0);
    }

    #[test]
    fn test_phrase_extraction() {
        let cache = TranslationCache::new(CacheConfig::default());

        let text = "Hello world! How are you today? I am fine, thank you.";
        let phrases = cache.extract_phrases(text);

        assert!(!phrases.is_empty());
        assert!(phrases.iter().any(|p| p.contains("Hello world")));
    }
}
