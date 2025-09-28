//! Translation Quality Assurance
//!
//! Implements real-time quality metrics for translation including BLEU score,
//! confidence scoring, context-aware validation, and multi-reference evaluation

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Translation quality result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranslationQualityResult {
    /// Overall quality score (0.0 to 1.0)
    pub overall_score: f32,
    /// BLEU score if reference available
    pub bleu_score: Option<f32>,
    /// Translation confidence
    pub confidence_score: f32,
    /// Context coherence score
    pub context_score: f32,
    /// Fluency score
    pub fluency_score: f32,
    /// Adequacy score (semantic preservation)
    pub adequacy_score: f32,
    /// Detected issues
    pub issues: Vec<QualityIssue>,
    /// Quality trend (improving/stable/degrading)
    pub trend: QualityTrend,
}

/// Quality issue detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityIssue {
    pub issue_type: IssueType,
    pub severity: IssueSeverity,
    pub description: String,
    pub position: Option<(usize, usize)>, // Start and end position in text
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IssueType {
    MissingTranslation,
    InconsistentTerminology,
    GrammaticalError,
    ContextMismatch,
    UntranslatedText,
    LowConfidence,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IssueSeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QualityTrend {
    Improving,
    Stable,
    Degrading,
}

/// Translation quality assurance engine
pub struct TranslationQA {
    config: super::QualityConfig,
    history: Arc<RwLock<Vec<TranslationQualityResult>>>,
    reference_translations: Arc<RwLock<HashMap<String, Vec<String>>>>,
    terminology_database: Arc<RwLock<HashMap<String, HashMap<String, String>>>>,
    context_buffer: Arc<RwLock<Vec<String>>>,
}

impl TranslationQA {
    pub async fn new(config: super::QualityConfig) -> Result<Self> {
        Ok(Self {
            config,
            history: Arc::new(RwLock::new(Vec::new())),
            reference_translations: Arc::new(RwLock::new(HashMap::new())),
            terminology_database: Arc::new(RwLock::new(HashMap::new())),
            context_buffer: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Evaluate translation quality
    pub async fn evaluate(
        &self,
        source_text: &str,
        translated_text: &str,
        source_lang: &str,
        target_lang: &str,
    ) -> Result<TranslationQualityResult> {
        let mut issues = Vec::new();

        // Calculate BLEU score if reference available
        let bleu_score = self.calculate_bleu_score(source_text, translated_text, target_lang).await;

        // Calculate confidence score
        let confidence_score = self.calculate_confidence(translated_text, source_text).await;

        // Check context coherence
        let context_score = self.check_context_coherence(translated_text).await;

        // Calculate fluency
        let fluency_score = self.calculate_fluency(translated_text, target_lang).await;

        // Calculate adequacy (semantic preservation)
        let adequacy_score = self.calculate_adequacy(source_text, translated_text).await;

        // Detect specific issues
        self.detect_issues(
            source_text,
            translated_text,
            source_lang,
            target_lang,
            &mut issues,
        ).await;

        // Calculate overall score
        let overall_score = self.calculate_overall_score(
            bleu_score,
            confidence_score,
            context_score,
            fluency_score,
            adequacy_score,
        );

        // Determine trend
        let trend = self.determine_trend(overall_score).await;

        let result = TranslationQualityResult {
            overall_score,
            bleu_score,
            confidence_score,
            context_score,
            fluency_score,
            adequacy_score,
            issues,
            trend,
        };

        // Update history
        self.update_history(result.clone()).await;

        Ok(result)
    }

    /// Calculate BLEU score
    async fn calculate_bleu_score(
        &self,
        source_text: &str,
        translated_text: &str,
        target_lang: &str,
    ) -> Option<f32> {
        let references = self.reference_translations.read().await;
        let key = format!("{}:{}", source_text, target_lang);
        
        if let Some(refs) = references.get(&key) {
            let bleu = self.compute_bleu(translated_text, refs);
            Some(bleu)
        } else {
            None
        }
    }

    /// Compute BLEU score
    fn compute_bleu(&self, candidate: &str, references: &[String]) -> f32 {
        // Simplified BLEU-4 implementation
        let candidate_tokens: Vec<&str> = candidate.split_whitespace().collect();
        let mut max_precision_sum = 0.0;
        
        for n in 1..=4 {
            let precision = self.calculate_ngram_precision(&candidate_tokens, references, n);
            max_precision_sum += precision.ln();
        }
        
        let brevity_penalty = self.calculate_brevity_penalty(&candidate_tokens, references);
        let bleu = brevity_penalty * (max_precision_sum / 4.0).exp();
        
        bleu.min(1.0).max(0.0)
    }

    /// Calculate n-gram precision
    fn calculate_ngram_precision(
        &self,
        candidate: &[&str],
        references: &[String],
        n: usize,
    ) -> f32 {
        if candidate.len() < n {
            return 0.0;
        }

        let candidate_ngrams = self.get_ngrams(candidate, n);
        let mut matches = 0;
        let total = candidate_ngrams.len();

        for ngram in &candidate_ngrams {
            for reference in references {
                let ref_tokens: Vec<&str> = reference.split_whitespace().collect();
                let ref_ngrams = self.get_ngrams(&ref_tokens, n);
                if ref_ngrams.contains(ngram) {
                    matches += 1;
                    break;
                }
            }
        }

        if total > 0 {
            matches as f32 / total as f32
        } else {
            0.0
        }
    }

    /// Get n-grams from tokens
    fn get_ngrams(&self, tokens: &[&str], n: usize) -> Vec<Vec<String>> {
        let mut ngrams = Vec::new();
        for i in 0..=tokens.len().saturating_sub(n) {
            let ngram: Vec<String> = tokens[i..i + n].iter().map(|s| s.to_string()).collect();
            ngrams.push(ngram);
        }
        ngrams
    }

    /// Calculate brevity penalty
    fn calculate_brevity_penalty(&self, candidate: &[&str], references: &[String]) -> f32 {
        let c = candidate.len() as f32;
        let r = references
            .iter()
            .map(|ref_text| ref_text.split_whitespace().count())
            .min()
            .unwrap_or(0) as f32;

        if c > r {
            1.0
        } else if r > 0.0 {
            (1.0 - r / c).exp()
        } else {
            0.0
        }
    }

    /// Calculate translation confidence
    async fn calculate_confidence(&self, translated_text: &str, source_text: &str) -> f32 {
        // Simplified confidence calculation
        let length_ratio = translated_text.len() as f32 / source_text.len().max(1) as f32;
        let length_confidence = 1.0 - (length_ratio - 1.0).abs().min(1.0);
        
        // Check for repeated words (low confidence indicator)
        let words: Vec<&str> = translated_text.split_whitespace().collect();
        let unique_words: std::collections::HashSet<_> = words.iter().collect();
        let repetition_ratio = unique_words.len() as f32 / words.len().max(1) as f32;
        
        (length_confidence + repetition_ratio) / 2.0
    }

    /// Check context coherence
    async fn check_context_coherence(&self, translated_text: &str) -> f32 {
        let context = self.context_buffer.read().await;
        if context.is_empty() {
            return 1.0; // No context to compare
        }

        // Simple coherence check based on vocabulary overlap
        let current_words: std::collections::HashSet<_> = 
            translated_text.split_whitespace().collect();
        
        let mut coherence_scores = Vec::new();
        for previous in context.iter().take(3) {
            let prev_words: std::collections::HashSet<_> = 
                previous.split_whitespace().collect();
            
            let intersection = current_words.intersection(&prev_words).count();
            let union = current_words.union(&prev_words).count();
            
            if union > 0 {
                coherence_scores.push(intersection as f32 / union as f32);
            }
        }

        if coherence_scores.is_empty() {
            1.0
        } else {
            coherence_scores.iter().sum::<f32>() / coherence_scores.len() as f32
        }
    }

    /// Calculate fluency score
    async fn calculate_fluency(&self, translated_text: &str, _target_lang: &str) -> f32 {
        // Simplified fluency calculation
        // In production, use language model perplexity
        
        let words: Vec<&str> = translated_text.split_whitespace().collect();
        if words.is_empty() {
            return 0.0;
        }

        // Check for basic fluency indicators
        let mut score = 1.0;
        
        // Penalize very short or very long sentences
        let word_count = words.len();
        if word_count < 3 {
            score *= 0.7;
        } else if word_count > 50 {
            score *= 0.8;
        }
        
        // Check for proper capitalization
        if let Some(first_char) = translated_text.chars().next() {
            if !first_char.is_uppercase() && first_char.is_alphabetic() {
                score *= 0.9;
            }
        }
        
        // Check for ending punctuation
        if let Some(last_char) = translated_text.chars().last() {
            if !".,!?;:".contains(last_char) {
                score *= 0.9;
            }
        }
        
        score
    }

    /// Calculate adequacy score
    async fn calculate_adequacy(&self, source_text: &str, translated_text: &str) -> f32 {
        // Simplified adequacy calculation
        // In production, use semantic similarity models
        
        let source_words: std::collections::HashSet<_> = 
            source_text.split_whitespace().collect();
        let translated_words: std::collections::HashSet<_> = 
            translated_text.split_whitespace().collect();
        
        // Basic length preservation check
        let length_ratio = translated_words.len() as f32 / source_words.len().max(1) as f32;
        let length_score = 1.0 - (length_ratio - 1.0).abs().min(1.0) * 0.5;
        
        // Check for numbers preservation
        let source_numbers = self.extract_numbers(source_text);
        let translated_numbers = self.extract_numbers(translated_text);
        let numbers_preserved = source_numbers.intersection(&translated_numbers).count();
        let number_score = if !source_numbers.is_empty() {
            numbers_preserved as f32 / source_numbers.len() as f32
        } else {
            1.0
        };
        
        (length_score + number_score) / 2.0
    }

    /// Extract numbers from text
    fn extract_numbers(&self, text: &str) -> std::collections::HashSet<String> {
        text.split_whitespace()
            .filter_map(|word| {
                if word.chars().any(|c| c.is_numeric()) {
                    Some(word.to_string())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Detect quality issues
    async fn detect_issues(
        &self,
        source_text: &str,
        translated_text: &str,
        _source_lang: &str,
        _target_lang: &str,
        issues: &mut Vec<QualityIssue>,
    ) {
        // Check for missing translation
        if translated_text.is_empty() && !source_text.is_empty() {
            issues.push(QualityIssue {
                issue_type: IssueType::MissingTranslation,
                severity: IssueSeverity::Critical,
                description: "Translation is empty".to_string(),
                position: None,
            });
        }
        
        // Check for untranslated text (same as source)
        if source_text == translated_text && source_text.len() > 10 {
            issues.push(QualityIssue {
                issue_type: IssueType::UntranslatedText,
                severity: IssueSeverity::High,
                description: "Text appears to be untranslated".to_string(),
                position: None,
            });
        }
        
        // Check terminology consistency
        let terminology = self.terminology_database.read().await;
        if let Some(terms) = terminology.get("_source_lang") {
            for (source_term, expected_translation) in terms {
                if source_text.contains(source_term) && !translated_text.contains(expected_translation) {
                    issues.push(QualityIssue {
                        issue_type: IssueType::InconsistentTerminology,
                        severity: IssueSeverity::Medium,
                        description: format!("Term '{}' not translated as expected", source_term),
                        position: None,
                    });
                }
            }
        }
    }

    /// Calculate overall score
    fn calculate_overall_score(
        &self,
        bleu_score: Option<f32>,
        confidence_score: f32,
        context_score: f32,
        fluency_score: f32,
        adequacy_score: f32,
    ) -> f32 {
        let mut scores = vec![confidence_score, context_score, fluency_score, adequacy_score];
        let mut weights = vec![0.2, 0.2, 0.3, 0.3];
        
        if let Some(bleu) = bleu_score {
            scores.push(bleu);
            weights.push(0.5);
            // Renormalize weights
            let sum: f32 = weights.iter().sum();
            weights = weights.iter().map(|w| w / sum).collect();
        }
        
        scores.iter()
            .zip(weights.iter())
            .map(|(s, w)| s * w)
            .sum()
    }

    /// Determine quality trend
    async fn determine_trend(&self, current_score: f32) -> QualityTrend {
        let history = self.history.read().await;
        if history.len() < 5 {
            return QualityTrend::Stable;
        }
        
        let recent_scores: Vec<f32> = history
            .iter()
            .rev()
            .take(5)
            .map(|r| r.overall_score)
            .collect();
        
        let avg_recent = recent_scores.iter().sum::<f32>() / recent_scores.len() as f32;
        
        if current_score > avg_recent * 1.1 {
            QualityTrend::Improving
        } else if current_score < avg_recent * 0.9 {
            QualityTrend::Degrading
        } else {
            QualityTrend::Stable
        }
    }

    /// Update history
    async fn update_history(&self, result: TranslationQualityResult) {
        let mut history = self.history.write().await;
        history.push(result);
        
        // Keep only last 100 results
        if history.len() > 100 {
            history.remove(0);
        }
    }

    /// Add reference translation for BLEU scoring
    pub async fn add_reference_translation(
        &self,
        source_text: &str,
        reference: &str,
        target_lang: &str,
    ) {
        let mut references = self.reference_translations.write().await;
        let key = format!("{}:{}", source_text, target_lang);
        references.entry(key)
            .or_insert_with(Vec::new)
            .push(reference.to_string());
    }

    /// Update terminology database
    pub async fn update_terminology(
        &self,
        source_lang: &str,
        terms: HashMap<String, String>,
    ) {
        let mut terminology = self.terminology_database.write().await;
        terminology.insert(source_lang.to_string(), terms);
    }

    /// Update context buffer
    pub async fn update_context(&self, text: &str) {
        let mut context = self.context_buffer.write().await;
        context.push(text.to_string());
        
        // Keep only last 10 context items
        if context.len() > 10 {
            context.remove(0);
        }
    }
}