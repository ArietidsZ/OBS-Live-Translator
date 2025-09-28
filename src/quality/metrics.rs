//! Quality Metrics Collection

use std::sync::Arc;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};

/// Translation quality metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TranslationMetrics {
    pub total_translations: u64,
    pub average_bleu_score: f32,
    pub average_confidence: f32,
    pub average_fluency: f32,
    pub average_adequacy: f32,
    pub quality_degradations: u64,
    pub critical_issues: u64,
}

/// Audio quality metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AudioMetrics {
    pub total_samples: u64,
    pub average_snr_db: f32,
    pub clipping_incidents: u64,
    pub dropout_incidents: u64,
    pub average_quality_score: f32,
}

/// Quality metrics collector
pub struct QualityMetricsCollector {
    translation_metrics: Arc<RwLock<TranslationMetrics>>,
    audio_metrics: Arc<RwLock<AudioMetrics>>,
}

impl QualityMetricsCollector {
    pub fn new() -> Self {
        Self {
            translation_metrics: Arc::new(RwLock::new(TranslationMetrics::default())),
            audio_metrics: Arc::new(RwLock::new(AudioMetrics::default())),
        }
    }

    pub async fn record_translation_quality(&self, result: &super::translation_qa::TranslationQualityResult) {
        let mut metrics = self.translation_metrics.write().await;
        metrics.total_translations += 1;
        
        // Update running averages
        let n = metrics.total_translations as f32;
        metrics.average_confidence = (metrics.average_confidence * (n - 1.0) + result.confidence_score) / n;
        metrics.average_fluency = (metrics.average_fluency * (n - 1.0) + result.fluency_score) / n;
        metrics.average_adequacy = (metrics.average_adequacy * (n - 1.0) + result.adequacy_score) / n;
        
        if let Some(bleu) = result.bleu_score {
            metrics.average_bleu_score = (metrics.average_bleu_score * (n - 1.0) + bleu) / n;
        }
        
        // Count issues
        for issue in &result.issues {
            match issue.severity {
                super::translation_qa::IssueSeverity::Critical => metrics.critical_issues += 1,
                _ => {}
            }
        }
        
        if matches!(result.trend, super::translation_qa::QualityTrend::Degrading) {
            metrics.quality_degradations += 1;
        }
    }

    pub async fn record_audio_quality(&self, result: &super::audio_qa::AudioQualityResult) {
        let mut metrics = self.audio_metrics.write().await;
        metrics.total_samples += 1;
        
        // Update running averages
        let n = metrics.total_samples as f32;
        metrics.average_snr_db = (metrics.average_snr_db * (n - 1.0) + result.snr_db) / n;
        metrics.average_quality_score = (metrics.average_quality_score * (n - 1.0) + result.quality_score) / n;
        
        // Count incidents
        if result.distortion.has_clipping {
            metrics.clipping_incidents += 1;
        }
        
        for issue in &result.issues {
            if matches!(issue.issue_type, super::audio_qa::AudioIssueType::Dropout) {
                metrics.dropout_incidents += 1;
            }
        }
    }

    pub async fn get_translation_metrics(&self) -> TranslationMetrics {
        self.translation_metrics.read().await.clone()
    }

    pub async fn get_audio_metrics(&self) -> AudioMetrics {
        self.audio_metrics.read().await.clone()
    }
}