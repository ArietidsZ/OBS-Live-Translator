//! Context-aware conversation tracking

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime};
use tracing::{debug, info};

use crate::TranslatorConfig;
use super::{ConversationContext, ConversationTone};

/// Context-aware conversation tracker
pub struct ContextTracker {
    /// Recent conversation context
    recent_context: VecDeque<ContextEntry>,
    /// Key terms and their translations
    term_dictionary: HashMap<String, TermInfo>,
    /// Current conversation state
    current_state: ConversationState,
    /// Configuration
    config: ContextTrackerConfig,
}

impl ContextTracker {
    pub async fn new(config: &TranslatorConfig) -> Result<Self> {
        info!("Initializing context-aware conversation tracker");
        
        Ok(Self {
            recent_context: VecDeque::with_capacity(100),
            term_dictionary: HashMap::new(),
            current_state: ConversationState::default(),
            config: ContextTrackerConfig::default(),
        })
    }

    /// Update conversation context with new translation
    pub async fn update_context(&mut self, original: &str, translated: &str) -> Result<()> {
        let timestamp = SystemTime::now();
        
        // Extract key terms from the conversation
        self.extract_and_update_terms(original, translated).await?;
        
        // Analyze conversation tone
        let tone = self.analyze_conversation_tone(original).await;
        
        // Calculate complexity level
        let complexity = self.calculate_complexity(original, translated).await;
        
        // Create context entry
        let entry = ContextEntry {
            original_text: original.to_string(),
            translated_text: translated.to_string(),
            timestamp,
            detected_tone: tone.clone(),
            complexity_score: complexity,
            key_terms: self.extract_key_terms_from_text(original).await,
        };
        
        // Add to recent context
        self.recent_context.push_back(entry);
        
        // Clean old entries (keep last 10 minutes)
        let cutoff = timestamp - Duration::from_secs(600);
        while let Some(front) = self.recent_context.front() {
            if front.timestamp < cutoff {
                self.recent_context.pop_front();
            } else {
                break;
            }
        }
        
        // Update current state
        self.current_state.last_update = timestamp;
        self.current_state.current_tone = tone;
        self.current_state.average_complexity = self.calculate_average_complexity().await;
        
        debug!("Updated context with {} recent entries", self.recent_context.len());
        Ok(())
    }

    /// Get current conversation context
    pub async fn get_current_context(&self) -> ConversationContext {
        ConversationContext {
            current_topics: self.get_current_topics().await,
            key_terms: self.get_active_key_terms().await,
            tone: self.current_state.current_tone.clone(),
            complexity_level: self.current_state.average_complexity,
        }
    }

    /// Reset context tracker
    pub async fn reset(&mut self) -> Result<()> {
        self.recent_context.clear();
        self.term_dictionary.clear();
        self.current_state = ConversationState::default();
        info!("Context tracker reset");
        Ok(())
    }

    /// Get translation suggestions based on context
    pub async fn get_translation_suggestions(&self, text: &str) -> Vec<String> {
        let mut suggestions = Vec::new();
        
        // Check for known terms that might need special translation
        for term in self.extract_key_terms_from_text(text).await {
            if let Some(term_info) = self.term_dictionary.get(&term) {
                if term_info.confidence_score > 0.7 {
                    suggestions.push(format!(
                        "Consider using '{}' for '{}' (used {} times in this conversation)",
                        term_info.preferred_translation,
                        term,
                        term_info.usage_count
                    ));
                }
            }
        }
        
        // Suggest tone adjustments
        match self.current_state.current_tone {
            ConversationTone::Technical => {
                if text.to_lowercase().contains("explain") {
                    suggestions.push("Consider using more technical terminology for accuracy".to_string());
                }
            },
            ConversationTone::Casual => {
                if text.split_whitespace().count() > 20 {
                    suggestions.push("Consider breaking this into shorter, more casual phrases".to_string());
                }
            },
            _ => {}
        }
        
        suggestions
    }

    // Private helper methods

    async fn extract_and_update_terms(&mut self, original: &str, translated: &str) -> Result<()> {
        // Simple term extraction (in production, use NLP)
        let original_words: Vec<&str> = original.split_whitespace()
            .filter(|w| w.len() > 3)
            .collect();
        let translated_words: Vec<&str> = translated.split_whitespace()
            .filter(|w| w.len() > 3)
            .collect();
        
        // Look for potential technical terms or proper nouns
        for word in original_words {
            if self.is_important_term(word) {
                let entry = self.term_dictionary.entry(word.to_string()).or_insert_with(|| {
                    TermInfo {
                        original_term: word.to_string(),
                        preferred_translation: self.find_corresponding_translation(word, &translated_words).unwrap_or_else(|| word.to_string()),
                        usage_count: 0,
                        confidence_score: 0.0,
                        first_seen: SystemTime::now(),
                        last_seen: SystemTime::now(),
                    }
                });
                
                entry.usage_count += 1;
                entry.last_seen = SystemTime::now();
                entry.confidence_score = (entry.usage_count as f32 * 0.1).min(1.0);
            }
        }
        
        Ok(())
    }

    fn is_important_term(&self, word: &str) -> bool {
        // Check if word is likely a technical term or proper noun
        word.chars().next().map_or(false, |c| c.is_uppercase()) ||
        word.chars().any(|c| c.is_numeric()) ||
        word.len() > 8 // Long words often technical
    }

    fn find_corresponding_translation(&self, original: &str, translated_words: &[&str]) -> Option<String> {
        // Simple heuristic: find word with similar characteristics
        for &translated in translated_words {
            if translated.len() > 3 && 
               (translated.chars().next().map_or(false, |c| c.is_uppercase()) == 
                original.chars().next().map_or(false, |c| c.is_uppercase())) {
                return Some(translated.to_string());
            }
        }
        None
    }

    async fn analyze_conversation_tone(&self, text: &str) -> ConversationTone {
        let text_lower = text.to_lowercase();
        
        // Technical indicators
        let technical_words = ["function", "variable", "algorithm", "implementation", "configure", "parameter"];
        if technical_words.iter().any(|&word| text_lower.contains(word)) {
            return ConversationTone::Technical;
        }
        
        // Gaming indicators
        let gaming_words = ["game", "level", "boss", "weapon", "skill", "quest"];
        if gaming_words.iter().any(|&word| text_lower.contains(word)) {
            return ConversationTone::Gaming;
        }
        
        // Educational indicators
        let educational_words = ["learn", "understand", "explain", "concept", "theory", "example"];
        if educational_words.iter().any(|&word| text_lower.contains(word)) {
            return ConversationTone::Educational;
        }
        
        // Casual indicators
        let casual_words = ["like", "awesome", "cool", "hey", "wow", "lol"];
        if casual_words.iter().any(|&word| text_lower.contains(word)) || text.contains("!") {
            return ConversationTone::Casual;
        }
        
        ConversationTone::Neutral
    }

    async fn calculate_complexity(&self, original: &str, translated: &str) -> f32 {
        let mut complexity = 0.0;
        
        // Sentence length factor
        let word_count = original.split_whitespace().count();
        complexity += (word_count as f32 / 10.0).min(2.0);
        
        // Vocabulary complexity
        let avg_word_length: f32 = original.split_whitespace()
            .map(|w| w.len() as f32)
            .sum::<f32>() / word_count.max(1) as f32;
        complexity += (avg_word_length - 4.0).max(0.0) * 0.2;
        
        // Translation length difference (might indicate complex concepts)
        let length_diff = (translated.len() as f32 - original.len() as f32).abs() / original.len().max(1) as f32;
        complexity += length_diff;
        
        complexity.min(5.0)
    }

    async fn calculate_average_complexity(&self) -> f32 {
        if self.recent_context.is_empty() {
            return 0.0;
        }
        
        let sum: f32 = self.recent_context.iter()
            .map(|entry| entry.complexity_score)
            .sum();
        
        sum / self.recent_context.len() as f32
    }

    async fn get_current_topics(&self) -> Vec<String> {
        let mut topic_counts: HashMap<String, usize> = HashMap::new();
        
        // Extract topics from recent context
        for entry in self.recent_context.iter().rev().take(10) {
            for topic in &entry.key_terms {
                *topic_counts.entry(topic.clone()).or_insert(0) += 1;
            }
        }
        
        // Return top 5 topics
        let mut topics: Vec<(String, usize)> = topic_counts.into_iter().collect();
        topics.sort_by(|a, b| b.1.cmp(&a.1));
        
        topics.into_iter()
            .take(5)
            .map(|(topic, _)| topic)
            .collect()
    }

    async fn get_active_key_terms(&self) -> HashMap<String, String> {
        let mut active_terms = HashMap::new();
        
        // Get terms used in last 5 minutes
        let cutoff = SystemTime::now() - Duration::from_secs(300);
        for (term, info) in &self.term_dictionary {
            if info.last_seen >= cutoff && info.confidence_score > 0.5 {
                active_terms.insert(term.clone(), info.preferred_translation.clone());
            }
        }
        
        active_terms
    }

    async fn extract_key_terms_from_text(&self, text: &str) -> Vec<String> {
        text.split_whitespace()
            .filter(|word| word.len() > 4)
            .filter(|word| self.is_important_term(word))
            .map(|word| word.to_string())
            .collect()
    }
}

/// Configuration for context tracking
#[derive(Debug, Clone)]
pub struct ContextTrackerConfig {
    pub max_context_entries: usize,
    pub term_confidence_threshold: f32,
    pub complexity_smoothing_factor: f32,
}

impl Default for ContextTrackerConfig {
    fn default() -> Self {
        Self {
            max_context_entries: 100,
            term_confidence_threshold: 0.5,
            complexity_smoothing_factor: 0.8,
        }
    }
}

/// A single context entry from conversation
#[derive(Debug, Clone)]
pub struct ContextEntry {
    pub original_text: String,
    pub translated_text: String,
    pub timestamp: SystemTime,
    pub detected_tone: ConversationTone,
    pub complexity_score: f32,
    pub key_terms: Vec<String>,
}

/// Information about a tracked term
#[derive(Debug, Clone)]
pub struct TermInfo {
    pub original_term: String,
    pub preferred_translation: String,
    pub usage_count: usize,
    pub confidence_score: f32,
    pub first_seen: SystemTime,
    pub last_seen: SystemTime,
}

/// Current conversation state
#[derive(Debug, Clone, Default)]
pub struct ConversationState {
    pub last_update: SystemTime,
    pub current_tone: ConversationTone,
    pub average_complexity: f32,
    pub active_topics: Vec<String>,
}