use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Advanced AI-powered emotion and tone analysis for real-time translation
/// This gives us a competitive edge in the hackathon

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Emotion {
    Neutral,
    Happy,
    Sad,
    Angry,
    Surprised,
    Fear,
    Disgust,
    Excited,
    Confused,
    Professional,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Tone {
    Casual,
    Formal,
    Enthusiastic,
    Serious,
    Humorous,
    Sarcastic,
    Educational,
    Dramatic,
    Aggressive,
    Compassionate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionAnalysis {
    pub primary_emotion: Emotion,
    pub emotion_confidence: f32,
    pub secondary_emotion: Option<Emotion>,
    pub tone: Tone,
    pub tone_confidence: f32,
    pub sentiment_score: f32, // -1.0 (negative) to 1.0 (positive)
    pub energy_level: f32,    // 0.0 (low) to 1.0 (high)
    pub keywords: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioFeatures {
    pub pitch_mean: f32,
    pub pitch_variance: f32,
    pub energy_mean: f32,
    pub speaking_rate: f32,
    pub pause_ratio: f32,
    pub voice_quality: VoiceQuality,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VoiceQuality {
    Clear,
    Breathy,
    Creaky,
    Tense,
    Whisper,
    Shout,
}

pub struct EmotionAnalyzer {
    /// Emotion lexicon for text-based analysis
    emotion_lexicon: HashMap<String, Vec<(Emotion, f32)>>,
    /// Acoustic patterns for audio-based analysis
    acoustic_patterns: HashMap<Emotion, AudioFeatures>,
    /// Context window for temporal smoothing
    context_window: Vec<EmotionAnalysis>,
    window_size: usize,
}

impl EmotionAnalyzer {
    pub fn new() -> Self {
        Self {
            emotion_lexicon: Self::build_emotion_lexicon(),
            acoustic_patterns: Self::build_acoustic_patterns(),
            context_window: Vec::new(),
            window_size: 5,
        }
    }

    /// Analyze emotion from both text and audio features
    pub fn analyze(
        &mut self,
        text: &str,
        audio_features: Option<AudioFeatures>,
        language: &str,
    ) -> EmotionAnalysis {
        // Text-based emotion detection
        let text_emotion = self.analyze_text_emotion(text, language);

        // Audio-based emotion detection (if available)
        let audio_emotion = audio_features
            .as_ref()
            .map(|features| self.analyze_audio_emotion(features));

        // Combine text and audio analysis
        let combined = self.combine_analyses(text_emotion, audio_emotion);

        // Apply temporal smoothing
        let smoothed = self.apply_temporal_smoothing(combined);

        // Update context window
        self.update_context_window(smoothed.clone());

        smoothed
    }

    fn analyze_text_emotion(&self, text: &str, language: &str) -> EmotionAnalysis {
        let text_lower = text.to_lowercase();
        let mut emotion_scores: HashMap<Emotion, f32> = HashMap::new();
        let mut tone_scores: HashMap<Tone, f32> = HashMap::new();

        // Simple keyword-based analysis (would use ML model in production)
        let keywords = self.extract_emotional_keywords(&text_lower, language);

        // Check for emotion indicators
        for keyword in &keywords {
            if let Some(emotions) = self.emotion_lexicon.get(keyword) {
                for (emotion, score) in emotions {
                    *emotion_scores.entry(emotion.clone()).or_insert(0.0) += score;
                }
            }
        }

        // Analyze tone based on punctuation and structure
        let tone = self.analyze_tone(&text_lower);

        // Sentiment analysis
        let sentiment = self.calculate_sentiment(&text_lower, &keywords);

        // Find primary emotion
        let primary_emotion = emotion_scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(e, _)| e.clone())
            .unwrap_or(Emotion::Neutral);

        let emotion_confidence = emotion_scores
            .get(&primary_emotion)
            .copied()
            .unwrap_or(0.5)
            .min(1.0);

        // Find secondary emotion
        let secondary_emotion = emotion_scores
            .iter()
            .filter(|(e, _)| **e != primary_emotion)
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(e, _)| e.clone());

        EmotionAnalysis {
            primary_emotion,
            emotion_confidence,
            secondary_emotion,
            tone,
            tone_confidence: 0.8,
            sentiment_score: sentiment,
            energy_level: self.calculate_energy_level(text),
            keywords: keywords.into_iter().take(3).collect(),
        }
    }

    fn analyze_audio_emotion(&self, features: &AudioFeatures) -> EmotionAnalysis {
        // Compare audio features with known patterns
        let mut best_match = Emotion::Neutral;
        let mut best_score = 0.0;

        for (emotion, pattern) in &self.acoustic_patterns {
            let score = self.calculate_audio_similarity(features, pattern);
            if score > best_score {
                best_score = score;
                best_match = emotion.clone();
            }
        }

        // Determine tone from voice quality and speaking rate
        let tone = match features.voice_quality {
            VoiceQuality::Whisper => Tone::Serious,
            VoiceQuality::Shout => Tone::Aggressive,
            VoiceQuality::Tense => Tone::Serious,
            _ => {
                if features.speaking_rate > 1.5 {
                    Tone::Enthusiastic
                } else if features.speaking_rate < 0.8 {
                    Tone::Casual
                } else {
                    Tone::Formal
                }
            }
        };

        EmotionAnalysis {
            primary_emotion: best_match,
            emotion_confidence: best_score,
            secondary_emotion: None,
            tone,
            tone_confidence: 0.7,
            sentiment_score: 0.0,
            energy_level: features.energy_mean,
            keywords: vec![],
        }
    }

    fn calculate_audio_similarity(&self, features: &AudioFeatures, pattern: &AudioFeatures) -> f32 {
        // Simple Euclidean distance (normalized)
        let pitch_diff = (features.pitch_mean - pattern.pitch_mean).abs() / 100.0;
        let energy_diff = (features.energy_mean - pattern.energy_mean).abs();
        let rate_diff = (features.speaking_rate - pattern.speaking_rate).abs();

        1.0 - (pitch_diff + energy_diff + rate_diff) / 3.0
    }

    fn combine_analyses(
        &self,
        text: EmotionAnalysis,
        audio: Option<EmotionAnalysis>,
    ) -> EmotionAnalysis {
        match audio {
            Some(audio_analysis) => {
                // Weight: 60% text, 40% audio
                let text_weight = 0.6;
                let audio_weight = 0.4;

                // If emotions match, increase confidence
                let (emotion, confidence) = if text.primary_emotion == audio_analysis.primary_emotion {
                    (
                        text.primary_emotion,
                        (text.emotion_confidence * text_weight + audio_analysis.emotion_confidence * audio_weight) * 1.2,
                    )
                } else {
                    // Use the one with higher confidence
                    if text.emotion_confidence > audio_analysis.emotion_confidence {
                        (text.primary_emotion, text.emotion_confidence)
                    } else {
                        (audio_analysis.primary_emotion, audio_analysis.emotion_confidence)
                    }
                };

                EmotionAnalysis {
                    primary_emotion: emotion,
                    emotion_confidence: confidence.min(1.0),
                    secondary_emotion: text.secondary_emotion.or(Some(audio_analysis.primary_emotion)),
                    tone: text.tone, // Prefer text-based tone
                    tone_confidence: (text.tone_confidence + audio_analysis.tone_confidence) / 2.0,
                    sentiment_score: text.sentiment_score,
                    energy_level: (text.energy_level * text_weight + audio_analysis.energy_level * audio_weight),
                    keywords: text.keywords,
                }
            }
            None => text,
        }
    }

    fn apply_temporal_smoothing(&self, current: EmotionAnalysis) -> EmotionAnalysis {
        if self.context_window.is_empty() {
            return current;
        }

        // Average confidence over window
        let avg_confidence = (self.context_window
            .iter()
            .map(|a| a.emotion_confidence)
            .sum::<f32>()
            + current.emotion_confidence)
            / (self.context_window.len() + 1) as f32;

        // Check for emotion stability
        let stable_emotion = self.context_window
            .iter()
            .filter(|a| a.primary_emotion == current.primary_emotion)
            .count() > self.context_window.len() / 2;

        EmotionAnalysis {
            emotion_confidence: if stable_emotion {
                avg_confidence * 1.1 // Boost confidence for stable emotions
            } else {
                avg_confidence * 0.9
            }.min(1.0),
            ..current
        }
    }

    fn update_context_window(&mut self, analysis: EmotionAnalysis) {
        self.context_window.push(analysis);
        if self.context_window.len() > self.window_size {
            self.context_window.remove(0);
        }
    }

    fn build_emotion_lexicon() -> HashMap<String, Vec<(Emotion, f32)>> {
        let mut lexicon = HashMap::new();

        // English emotional keywords
        lexicon.insert("happy".to_string(), vec![(Emotion::Happy, 0.9)]);
        lexicon.insert("excited".to_string(), vec![(Emotion::Excited, 0.9)]);
        lexicon.insert("sad".to_string(), vec![(Emotion::Sad, 0.9)]);
        lexicon.insert("angry".to_string(), vec![(Emotion::Angry, 0.9)]);
        lexicon.insert("love".to_string(), vec![(Emotion::Happy, 0.8)]);
        lexicon.insert("hate".to_string(), vec![(Emotion::Angry, 0.8)]);
        lexicon.insert("amazing".to_string(), vec![(Emotion::Excited, 0.7), (Emotion::Happy, 0.5)]);
        lexicon.insert("terrible".to_string(), vec![(Emotion::Sad, 0.7), (Emotion::Angry, 0.5)]);
        lexicon.insert("worried".to_string(), vec![(Emotion::Fear, 0.7)]);
        lexicon.insert("confused".to_string(), vec![(Emotion::Confused, 0.8)]);

        // Chinese emotional keywords
        lexicon.insert("开心".to_string(), vec![(Emotion::Happy, 0.9)]);
        lexicon.insert("高兴".to_string(), vec![(Emotion::Happy, 0.8)]);
        lexicon.insert("难过".to_string(), vec![(Emotion::Sad, 0.9)]);
        lexicon.insert("生气".to_string(), vec![(Emotion::Angry, 0.9)]);
        lexicon.insert("惊讶".to_string(), vec![(Emotion::Surprised, 0.9)]);
        lexicon.insert("害怕".to_string(), vec![(Emotion::Fear, 0.8)]);
        lexicon.insert("激动".to_string(), vec![(Emotion::Excited, 0.9)]);

        // Japanese emotional keywords
        lexicon.insert("嬉しい".to_string(), vec![(Emotion::Happy, 0.9)]);
        lexicon.insert("悲しい".to_string(), vec![(Emotion::Sad, 0.9)]);
        lexicon.insert("怒る".to_string(), vec![(Emotion::Angry, 0.8)]);
        lexicon.insert("驚く".to_string(), vec![(Emotion::Surprised, 0.9)]);

        lexicon
    }

    fn build_acoustic_patterns() -> HashMap<Emotion, AudioFeatures> {
        let mut patterns = HashMap::new();

        // Happy: Higher pitch, higher energy, faster rate
        patterns.insert(Emotion::Happy, AudioFeatures {
            pitch_mean: 250.0,
            pitch_variance: 50.0,
            energy_mean: 0.8,
            speaking_rate: 1.2,
            pause_ratio: 0.1,
            voice_quality: VoiceQuality::Clear,
        });

        // Sad: Lower pitch, lower energy, slower rate
        patterns.insert(Emotion::Sad, AudioFeatures {
            pitch_mean: 180.0,
            pitch_variance: 20.0,
            energy_mean: 0.3,
            speaking_rate: 0.8,
            pause_ratio: 0.3,
            voice_quality: VoiceQuality::Breathy,
        });

        // Angry: Variable pitch, high energy, fast rate
        patterns.insert(Emotion::Angry, AudioFeatures {
            pitch_mean: 220.0,
            pitch_variance: 80.0,
            energy_mean: 0.9,
            speaking_rate: 1.3,
            pause_ratio: 0.05,
            voice_quality: VoiceQuality::Tense,
        });

        // Excited: Very high pitch, very high energy, very fast rate
        patterns.insert(Emotion::Excited, AudioFeatures {
            pitch_mean: 280.0,
            pitch_variance: 70.0,
            energy_mean: 0.95,
            speaking_rate: 1.5,
            pause_ratio: 0.05,
            voice_quality: VoiceQuality::Clear,
        });

        patterns
    }

    fn extract_emotional_keywords(&self, text: &str, _language: &str) -> Vec<String> {
        // Simple word extraction (would use NLP in production)
        text.split_whitespace()
            .filter(|word| self.emotion_lexicon.contains_key(*word))
            .map(|s| s.to_string())
            .collect()
    }

    fn analyze_tone(&self, text: &str) -> Tone {
        // Simple heuristics for tone detection
        if text.contains('!') && text.matches('!').count() > 1 {
            Tone::Enthusiastic
        } else if text.contains('?') && text.contains("how") || text.contains("what") || text.contains("why") {
            Tone::Educational
        } else if text.contains("haha") || text.contains("lol") || text.contains("😄") {
            Tone::Humorous
        } else if text.len() < 20 {
            Tone::Casual
        } else {
            Tone::Formal
        }
    }

    fn calculate_sentiment(&self, text: &str, keywords: &[String]) -> f32 {
        // Simple sentiment calculation
        let positive_words = ["good", "great", "excellent", "love", "happy", "amazing"];
        let negative_words = ["bad", "terrible", "hate", "sad", "angry", "awful"];

        let mut score = 0.0;
        for word in positive_words.iter() {
            if text.contains(word) {
                score += 0.2;
            }
        }
        for word in negative_words.iter() {
            if text.contains(word) {
                score -= 0.2;
            }
        }

        score.max(-1.0).min(1.0)
    }

    fn calculate_energy_level(&self, text: &str) -> f32 {
        // Energy based on exclamation marks and capitals
        let exclamation_count = text.matches('!').count() as f32;
        let capital_ratio = text.chars().filter(|c| c.is_uppercase()).count() as f32
            / text.len().max(1) as f32;

        ((exclamation_count * 0.2) + (capital_ratio * 2.0)).min(1.0)
    }
}

/// Visual effects generator based on emotion
pub struct EmotionVisualizer {
    particle_configs: HashMap<Emotion, ParticleConfig>,
    color_schemes: HashMap<Emotion, ColorScheme>,
}

#[derive(Debug, Clone)]
pub struct ParticleConfig {
    pub count: u32,
    pub speed: f32,
    pub size: f32,
    pub spread: f32,
    pub lifetime: f32,
}

#[derive(Debug, Clone)]
pub struct ColorScheme {
    pub primary: String,
    pub secondary: String,
    pub glow: String,
}

impl EmotionVisualizer {
    pub fn new() -> Self {
        let mut particle_configs = HashMap::new();
        let mut color_schemes = HashMap::new();

        // Happy: Yellow particles, upward movement
        particle_configs.insert(Emotion::Happy, ParticleConfig {
            count: 30,
            speed: 2.0,
            size: 3.0,
            spread: 1.5,
            lifetime: 2.0,
        });
        color_schemes.insert(Emotion::Happy, ColorScheme {
            primary: "#FFD700".to_string(),
            secondary: "#FFA500".to_string(),
            glow: "#FFEB3B".to_string(),
        });

        // Excited: Red/orange particles, fast movement
        particle_configs.insert(Emotion::Excited, ParticleConfig {
            count: 50,
            speed: 3.0,
            size: 2.5,
            spread: 2.0,
            lifetime: 1.5,
        });
        color_schemes.insert(Emotion::Excited, ColorScheme {
            primary: "#FF6B6B".to_string(),
            secondary: "#FF9F43".to_string(),
            glow: "#FF5722".to_string(),
        });

        // Sad: Blue particles, downward movement
        particle_configs.insert(Emotion::Sad, ParticleConfig {
            count: 20,
            speed: 0.5,
            size: 2.0,
            spread: 0.8,
            lifetime: 3.0,
        });
        color_schemes.insert(Emotion::Sad, ColorScheme {
            primary: "#4A90E2".to_string(),
            secondary: "#5DADE2".to_string(),
            glow: "#2196F3".to_string(),
        });

        Self {
            particle_configs,
            color_schemes,
        }
    }

    pub fn get_visual_config(&self, emotion: &Emotion) -> (ParticleConfig, ColorScheme) {
        (
            self.particle_configs.get(emotion).cloned().unwrap_or(ParticleConfig {
                count: 10,
                speed: 1.0,
                size: 2.0,
                spread: 1.0,
                lifetime: 2.0,
            }),
            self.color_schemes.get(emotion).cloned().unwrap_or(ColorScheme {
                primary: "#FFFFFF".to_string(),
                secondary: "#CCCCCC".to_string(),
                glow: "#FFFFFF".to_string(),
            })
        )
    }
}