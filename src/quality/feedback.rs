//! User Feedback System

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

/// User feedback entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserFeedback {
    pub timestamp: DateTime<Utc>,
    pub feedback_type: FeedbackType,
    pub rating: Option<u8>, // 1-5 scale
    pub text: Option<String>,
    pub source_text: Option<String>,
    pub translated_text: Option<String>,
    pub source_lang: Option<String>,
    pub target_lang: Option<String>,
    pub session_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeedbackType {
    TranslationQuality,
    AudioQuality,
    Latency,
    UserInterface,
    GeneralComment,
    BugReport,
}

/// Feedback summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeedbackSummary {
    pub total_feedback: usize,
    pub average_rating: f32,
    pub feedback_by_type: std::collections::HashMap<String, usize>,
    pub common_issues: Vec<String>,
    pub positive_feedback_ratio: f32,
}

/// Feedback handler
pub struct FeedbackHandler {
    config: super::QualityConfig,
    feedback_store: Arc<RwLock<Vec<UserFeedback>>>,
}

impl FeedbackHandler {
    pub async fn new(config: super::QualityConfig) -> Result<Self> {
        Ok(Self {
            config,
            feedback_store: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Submit user feedback
    pub async fn submit(&self, feedback: UserFeedback) -> Result<()> {
        let mut store = self.feedback_store.write().await;
        store.push(feedback.clone());

        // Keep only last 1000 feedback entries
        if store.len() > 1000 {
            store.remove(0);
        }

        // Process feedback for insights
        self.process_feedback(&feedback).await?;

        Ok(())
    }

    /// Process feedback for insights
    async fn process_feedback(&self, feedback: &UserFeedback) -> Result<()> {
        // Log low ratings for investigation
        if let Some(rating) = feedback.rating {
            if rating <= 2 {
                tracing::warn!(
                    "Low rating ({}/5) received for {:?}: {:?}",
                    rating,
                    feedback.feedback_type,
                    feedback.text
                );
            }
        }

        // Track quality issues
        if matches!(feedback.feedback_type, FeedbackType::TranslationQuality) {
            if let (Some(source), Some(translated)) =
                (&feedback.source_text, &feedback.translated_text)
            {
                tracing::debug!(
                    "Translation feedback - Source: '{}', Translation: '{}', Rating: {:?}",
                    source,
                    translated,
                    feedback.rating
                );
            }
        }

        Ok(())
    }

    /// Get feedback summary
    pub async fn get_summary(&self) -> Result<FeedbackSummary> {
        let store = self.feedback_store.read().await;

        let total_feedback = store.len();

        // Calculate average rating
        let ratings: Vec<u8> = store.iter().filter_map(|f| f.rating).collect();

        let average_rating = if !ratings.is_empty() {
            ratings.iter().map(|&r| r as f32).sum::<f32>() / ratings.len() as f32
        } else {
            0.0
        };

        // Count feedback by type
        let mut feedback_by_type = std::collections::HashMap::new();
        for feedback in store.iter() {
            let type_str = format!("{:?}", feedback.feedback_type);
            *feedback_by_type.entry(type_str).or_insert(0) += 1;
        }

        // Identify common issues
        let mut issue_counts = std::collections::HashMap::new();
        for feedback in store.iter() {
            if let Some(text) = &feedback.text {
                // Simple keyword extraction
                let keywords = vec![
                    "slow",
                    "wrong",
                    "error",
                    "bad",
                    "incorrect",
                    "noise",
                    "quiet",
                ];
                for keyword in keywords {
                    if text.to_lowercase().contains(keyword) {
                        *issue_counts.entry(keyword.to_string()).or_insert(0) += 1;
                    }
                }
            }
        }

        let mut common_issues: Vec<String> = issue_counts
            .into_iter()
            .filter(|(_, count)| *count >= 3)
            .map(|(issue, _)| issue)
            .collect();
        common_issues.sort();

        // Calculate positive feedback ratio
        let positive_count = ratings.iter().filter(|&&r| r >= 4).count();
        let positive_feedback_ratio = if !ratings.is_empty() {
            positive_count as f32 / ratings.len() as f32
        } else {
            0.0
        };

        Ok(FeedbackSummary {
            total_feedback,
            average_rating,
            feedback_by_type,
            common_issues,
            positive_feedback_ratio,
        })
    }

    /// Get recent feedback
    pub async fn get_recent(&self, count: usize) -> Vec<UserFeedback> {
        let store = self.feedback_store.read().await;
        store.iter().rev().take(count).cloned().collect()
    }

    /// Export feedback for analysis
    pub async fn export_feedback(&self) -> Result<String> {
        let store = self.feedback_store.read().await;
        serde_json::to_string_pretty(&*store).map_err(Into::into)
    }
}
