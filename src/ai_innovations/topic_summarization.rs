//! Real-time topic summarization for new viewers

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::TranslatorConfig;

/// Real-time topic summarization engine for new viewers
pub struct TopicSummarizer {
    /// Recent conversation history (last 5 minutes)
    conversation_history: VecDeque<ConversationSegment>,
    /// Detected topics and their importance scores
    topic_tracker: HashMap<String, TopicInfo>,
    /// Current session context
    session_context: SessionContext,
    /// Configuration
    config: TopicSummarizerConfig,
}

impl TopicSummarizer {
    pub async fn new(config: &TranslatorConfig) -> Result<Self> {
        info!("Initializing real-time topic summarizer");
        
        Ok(Self {
            conversation_history: VecDeque::with_capacity(300), // ~5 minutes at 1 entry/second
            topic_tracker: HashMap::new(),
            session_context: SessionContext::default(),
            config: TopicSummarizerConfig::default(),
        })
    }

    /// Add new conversation segment and update topic tracking
    pub async fn add_conversation_segment(
        &mut self,
        original_text: &str,
        translated_text: &str,
        timestamp: SystemTime,
    ) -> Result<()> {
        let segment = ConversationSegment {
            original_text: original_text.to_string(),
            translated_text: translated_text.to_string(),
            timestamp,
            detected_topics: self.extract_topics(original_text).await?,
            importance_score: self.calculate_importance_score(original_text).await,
        };

        // Update topic tracker
        for topic in &segment.detected_topics {
            let entry = self.topic_tracker.entry(topic.clone()).or_insert_with(|| TopicInfo {
                name: topic.clone(),
                first_mention: timestamp,
                last_mention: timestamp,
                mention_count: 0,
                importance_score: 0.0,
                context_snippets: Vec::new(),
            });
            
            entry.last_mention = timestamp;
            entry.mention_count += 1;
            entry.importance_score += segment.importance_score;
            
            // Keep top 3 context snippets
            entry.context_snippets.push(original_text.to_string());
            if entry.context_snippets.len() > 3 {
                entry.context_snippets.remove(0);
            }
        }

        // Add to conversation history
        self.conversation_history.push_back(segment);
        
        // Clean old entries (older than 5 minutes)
        let cutoff_time = timestamp - Duration::from_secs(300);
        while let Some(front) = self.conversation_history.front() {
            if front.timestamp < cutoff_time {
                self.conversation_history.pop_front();
            } else {
                break;
            }
        }

        debug!("Added conversation segment, total segments: {}", self.conversation_history.len());
        Ok(())
    }

    /// Generate real-time summary for newcomers joining the stream
    pub async fn generate_summary_for_newcomers(&self) -> Result<String> {
        if self.conversation_history.is_empty() {
            return Ok("Stream just started! Join the conversation.".to_string());
        }

        let mut summary_parts = Vec::new();

        // Get top topics from recent conversation
        let top_topics = self.get_top_topics(3).await;
        
        if !top_topics.is_empty() {
            let topics_list: Vec<String> = top_topics.iter()
                .map(|t| t.name.clone())
                .collect();
            
            summary_parts.push(format!(
                "📋 Currently discussing: {}",
                topics_list.join(", ")
            ));
        }

        // Get recent context (last 2-3 important segments)
        let recent_context = self.get_recent_important_context(3).await;
        if !recent_context.is_empty() {
            summary_parts.push("💬 Recent highlights:".to_string());
            for (i, segment) in recent_context.iter().enumerate() {
                // Get a short excerpt from the segment
                let excerpt = self.create_excerpt(&segment.translated_text, 80);
                summary_parts.push(format!("  {}. {}", i + 1, excerpt));
            }
        }

        // Add session info if available
        if let Some(session_info) = self.get_session_info().await {
            summary_parts.push(session_info);
        }

        if summary_parts.is_empty() {
            Ok("🎥 Live stream in progress - jump in anytime!".to_string())
        } else {
            Ok(summary_parts.join("\n"))
        }
    }

    /// Detect if there's been a major topic change
    pub async fn detect_topic_change(&self) -> Option<TopicChangeEvent> {
        // Look at last 30 seconds vs previous 2 minutes
        let now = SystemTime::now();
        let recent_cutoff = now - Duration::from_secs(30);
        let previous_cutoff = now - Duration::from_secs(150); // 2.5 minutes ago
        
        let recent_topics: Vec<String> = self.conversation_history
            .iter()
            .filter(|seg| seg.timestamp >= recent_cutoff)
            .flat_map(|seg| seg.detected_topics.iter())
            .cloned()
            .collect();
        
        let previous_topics: Vec<String> = self.conversation_history
            .iter()
            .filter(|seg| seg.timestamp >= previous_cutoff && seg.timestamp < recent_cutoff)
            .flat_map(|seg| seg.detected_topics.iter())
            .cloned()
            .collect();
        
        // Calculate topic overlap
        let recent_set: std::collections::HashSet<_> = recent_topics.iter().collect();
        let previous_set: std::collections::HashSet<_> = previous_topics.iter().collect();
        
        let overlap = recent_set.intersection(&previous_set).count();
        let total_recent = recent_set.len();
        
        if total_recent > 0 {
            let overlap_ratio = overlap as f32 / total_recent as f32;
            
            // If less than 30% overlap, it's likely a topic change
            if overlap_ratio < 0.3 && total_recent >= 2 {
                return Some(TopicChangeEvent {
                    previous_topics: previous_topics.into_iter().take(3).collect(),
                    new_topics: recent_topics.into_iter().take(3).collect(),
                    confidence: 1.0 - overlap_ratio,
                    timestamp: now,
                });
            }
        }
        
        None
    }

    /// Reset for new stream session
    pub async fn reset(&mut self) -> Result<()> {
        self.conversation_history.clear();
        self.topic_tracker.clear();
        self.session_context = SessionContext::default();
        info!("Topic summarizer reset for new session");
        Ok(())
    }

    // Private helper methods

    async fn extract_topics(&self, text: &str) -> Result<Vec<String>> {
        let mut topics = Vec::new();
        let text_lower = text.to_lowercase();
        
        // Simple keyword-based topic extraction (in production, use NLP models)
        let topic_keywords = [
            ("gaming", vec!["game", "play", "level", "boss", "enemy", "weapon", "strategy"]),
            ("programming", vec!["code", "function", "variable", "algorithm", "debug", "compile"]),
            ("tutorial", vec!["how to", "step", "first", "next", "tutorial", "guide", "learn"]),
            ("review", vec!["review", "opinion", "rating", "recommend", "like", "dislike"]),
            ("technology", vec!["tech", "software", "hardware", "computer", "device", "app"]),
            ("entertainment", vec!["movie", "show", "music", "artist", "episode", "season"]),
            ("news", vec!["news", "breaking", "update", "report", "announced", "happened"]),
            ("education", vec!["explain", "concept", "theory", "understand", "knowledge", "study"]),
        ];
        
        for (topic, keywords) in topic_keywords.iter() {
            let matches = keywords.iter()
                .filter(|&&keyword| text_lower.contains(keyword))
                .count();
            
            if matches >= 2 || (matches >= 1 && keywords.iter().any(|&kw| kw.len() > 6 && text_lower.contains(kw))) {
                topics.push(topic.to_string());
            }
        }
        
        // Extract proper nouns as potential topics
        let words: Vec<&str> = text.split_whitespace().collect();
        for word in words {
            if word.len() > 3 && word.chars().next().map_or(false, |c| c.is_uppercase()) {
                // Simple heuristic for proper nouns
                if !word.chars().any(|c| c.is_lowercase()) {
                    continue; // Skip all-caps words
                }
                topics.push(word.to_string());
            }
        }
        
        // Limit to top 5 topics to avoid noise
        topics.truncate(5);
        Ok(topics)
    }

    async fn calculate_importance_score(&self, text: &str) -> f32 {
        let mut score = 0.0;
        
        // Length factor (longer statements often more important)
        score += (text.len() as f32 / 100.0).min(2.0);
        
        // Question words boost importance
        let text_lower = text.to_lowercase();
        if text_lower.contains("what") || text_lower.contains("how") || 
           text_lower.contains("why") || text_lower.contains("where") {
            score += 1.5;
        }
        
        // Exclamation marks suggest emphasis
        score += text.matches('!').count() as f32 * 0.5;
        
        // Key phrases that suggest important information
        let important_phrases = [
            "important", "key", "main point", "remember", "note that",
            "first", "finally", "in conclusion", "to summarize"
        ];
        
        for phrase in important_phrases.iter() {
            if text_lower.contains(phrase) {
                score += 1.0;
                break;
            }
        }
        
        score.min(5.0) // Cap at 5.0
    }

    async fn get_top_topics(&self, limit: usize) -> Vec<&TopicInfo> {
        let mut topics: Vec<&TopicInfo> = self.topic_tracker.values().collect();
        topics.sort_by(|a, b| {
            // Sort by recency-weighted importance
            let score_a = a.importance_score * self.recency_weight(a.last_mention);
            let score_b = b.importance_score * self.recency_weight(b.last_mention);
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        topics.into_iter().take(limit).collect()
    }

    async fn get_recent_important_context(&self, limit: usize) -> Vec<&ConversationSegment> {
        let mut segments: Vec<&ConversationSegment> = self.conversation_history
            .iter()
            .filter(|seg| seg.importance_score >= 1.0) // Only important segments
            .collect();
        
        // Sort by importance and recency
        segments.sort_by(|a, b| {
            let score_a = a.importance_score * self.recency_weight(a.timestamp);
            let score_b = b.importance_score * self.recency_weight(b.timestamp);
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        segments.into_iter().take(limit).collect()
    }

    fn recency_weight(&self, timestamp: SystemTime) -> f32 {
        let elapsed = SystemTime::now()
            .duration_since(timestamp)
            .unwrap_or_default()
            .as_secs() as f32;
        
        // Exponential decay: newer = higher weight
        (-elapsed / 120.0).exp() // Half-life of 2 minutes
    }

    fn create_excerpt(&self, text: &str, max_length: usize) -> String {
        if text.len() <= max_length {
            text.to_string()
        } else {
            format!("{}...", &text[..max_length.saturating_sub(3)])
        }
    }

    async fn get_session_info(&self) -> Option<String> {
        // Could include stream duration, viewer count changes, etc.
        let total_segments = self.conversation_history.len();
        if total_segments > 10 {
            Some(format!("📊 Active discussion with {} recent exchanges", total_segments))
        } else {
            None
        }
    }
}

/// Configuration for topic summarization
#[derive(Debug, Clone)]
pub struct TopicSummarizerConfig {
    pub max_history_duration: Duration,
    pub min_importance_threshold: f32,
    pub max_topics_tracked: usize,
}

impl Default for TopicSummarizerConfig {
    fn default() -> Self {
        Self {
            max_history_duration: Duration::from_secs(300), // 5 minutes
            min_importance_threshold: 0.5,
            max_topics_tracked: 20,
        }
    }
}

/// A segment of conversation with metadata
#[derive(Debug, Clone)]
pub struct ConversationSegment {
    pub original_text: String,
    pub translated_text: String,
    pub timestamp: SystemTime,
    pub detected_topics: Vec<String>,
    pub importance_score: f32,
}

/// Information about a tracked topic
#[derive(Debug, Clone)]
pub struct TopicInfo {
    pub name: String,
    pub first_mention: SystemTime,
    pub last_mention: SystemTime,
    pub mention_count: usize,
    pub importance_score: f32,
    pub context_snippets: Vec<String>,
}

/// Topic change detection event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicChangeEvent {
    pub previous_topics: Vec<String>,
    pub new_topics: Vec<String>,
    pub confidence: f32,
    pub timestamp: SystemTime,
}

/// Session context for the current stream
#[derive(Debug, Clone, Default)]
pub struct SessionContext {
    pub start_time: Option<SystemTime>,
    pub estimated_viewer_count: u32,
    pub stream_category: Option<String>,
    pub language_stats: HashMap<String, u32>,
}