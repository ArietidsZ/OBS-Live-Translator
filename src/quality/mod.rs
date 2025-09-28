//! Quality Assurance Module
//!
//! Provides real-time quality monitoring and validation for translation and audio processing

pub mod translation_qa;
pub mod audio_qa;
pub mod metrics;
pub mod feedback;

use std::sync::Arc;
use tokio::sync::RwLock;
use anyhow::Result;

/// Quality assurance configuration
#[derive(Debug, Clone)]
pub struct QualityConfig {
    /// Enable real-time quality monitoring
    pub enable_monitoring: bool,
    /// Quality threshold for alerts
    pub quality_threshold: f32,
    /// Enable user feedback collection
    pub enable_feedback: bool,
    /// Quality check interval in milliseconds
    pub check_interval_ms: u64,
}

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            enable_monitoring: true,
            quality_threshold: 0.7,
            enable_feedback: true,
            check_interval_ms: 1000,
        }
    }
}

/// Quality assurance manager
pub struct QualityAssuranceManager {
    config: Arc<RwLock<QualityConfig>>,
    translation_qa: Arc<translation_qa::TranslationQA>,
    audio_qa: Arc<audio_qa::AudioQA>,
    metrics_collector: Arc<metrics::QualityMetricsCollector>,
    feedback_handler: Arc<feedback::FeedbackHandler>,
}

impl QualityAssuranceManager {
    pub async fn new(config: QualityConfig) -> Result<Self> {
        let translation_qa = Arc::new(translation_qa::TranslationQA::new(config.clone()).await?);
        let audio_qa = Arc::new(audio_qa::AudioQA::new(config.clone()).await?);
        let metrics_collector = Arc::new(metrics::QualityMetricsCollector::new());
        let feedback_handler = Arc::new(feedback::FeedbackHandler::new(config.clone()).await?);

        Ok(Self {
            config: Arc::new(RwLock::new(config)),
            translation_qa,
            audio_qa,
            metrics_collector,
            feedback_handler,
        })
    }

    /// Check translation quality
    pub async fn check_translation_quality(
        &self,
        source_text: &str,
        translated_text: &str,
        source_lang: &str,
        target_lang: &str,
    ) -> Result<translation_qa::TranslationQualityResult> {
        let result = self.translation_qa.evaluate(
            source_text,
            translated_text,
            source_lang,
            target_lang,
        ).await?;

        // Record metrics
        self.metrics_collector.record_translation_quality(&result).await;

        // Check threshold
        let config = self.config.read().await;
        if result.overall_score < config.quality_threshold {
            tracing::warn!(
                "Translation quality below threshold: {:.2} < {:.2}",
                result.overall_score,
                config.quality_threshold
            );
        }

        Ok(result)
    }

    /// Check audio quality
    pub async fn check_audio_quality(
        &self,
        audio_data: &[f32],
        sample_rate: u32,
    ) -> Result<audio_qa::AudioQualityResult> {
        let result = self.audio_qa.analyze(audio_data, sample_rate).await?;

        // Record metrics
        self.metrics_collector.record_audio_quality(&result).await;

        // Check for issues
        if result.has_issues() {
            tracing::warn!("Audio quality issues detected: {:?}", result.issues);
        }

        Ok(result)
    }

    /// Submit user feedback
    pub async fn submit_feedback(
        &self,
        feedback: feedback::UserFeedback,
    ) -> Result<()> {
        self.feedback_handler.submit(feedback).await?;
        Ok(())
    }

    /// Get quality report
    pub async fn get_quality_report(&self) -> Result<QualityReport> {
        let translation_metrics = self.metrics_collector.get_translation_metrics().await;
        let audio_metrics = self.metrics_collector.get_audio_metrics().await;
        let feedback_summary = self.feedback_handler.get_summary().await?;

        Ok(QualityReport {
            translation_metrics,
            audio_metrics,
            feedback_summary,
            timestamp: std::time::SystemTime::now(),
        })
    }
}

/// Comprehensive quality report
#[derive(Debug, Clone)]
pub struct QualityReport {
    pub translation_metrics: metrics::TranslationMetrics,
    pub audio_metrics: metrics::AudioMetrics,
    pub feedback_summary: feedback::FeedbackSummary,
    pub timestamp: std::time::SystemTime,
}