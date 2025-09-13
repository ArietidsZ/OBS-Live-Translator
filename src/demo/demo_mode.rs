use std::sync::Arc;
use tokio::sync::{RwLock, broadcast};
use serde::{Deserialize, Serialize};
use rand::Rng;

/// Demo mode for impressive presentations without real audio input
/// Simulates realistic multilingual conversations with synthetic voices

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemoScript {
    pub id: String,
    pub name: String,
    pub description: String,
    pub segments: Vec<DemoSegment>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemoSegment {
    pub timestamp_ms: u64,
    pub speaker: String,
    pub language: String,
    pub original_text: String,
    pub emotion: Emotion,
    pub tone: Tone,
    pub speed: f32, // 0.5 = slow, 1.0 = normal, 1.5 = fast
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Emotion {
    Neutral,
    Happy,
    Excited,
    Sad,
    Angry,
    Surprised,
    Confused,
    Professional,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Tone {
    Casual,
    Formal,
    Enthusiastic,
    Serious,
    Humorous,
    Educational,
    Dramatic,
}

pub struct DemoMode {
    scripts: Vec<DemoScript>,
    current_script: Option<DemoScript>,
    subtitle_tx: broadcast::Sender<SubtitleUpdate>,
    is_running: Arc<RwLock<bool>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SubtitleUpdate {
    pub timestamp: u64,
    pub speaker: String,
    pub language: String,
    pub original: String,
    pub translation: String,
    pub confidence: f32,
    pub emotion: Emotion,
    pub tone: Tone,
    pub visual_effects: VisualEffects,
}

#[derive(Debug, Clone, Serialize)]
pub struct VisualEffects {
    pub color_theme: String,
    pub animation: String,
    pub emphasis_words: Vec<String>,
    pub particle_effect: bool,
}

impl DemoMode {
    pub fn new() -> Self {
        let (subtitle_tx, _) = broadcast::channel(100);

        Self {
            scripts: Self::create_demo_scripts(),
            current_script: None,
            subtitle_tx,
            is_running: Arc::new(RwLock::new(false)),
        }
    }

    fn create_demo_scripts() -> Vec<DemoScript> {
        vec![
            // Tech Conference Demo
            DemoScript {
                id: "tech_conference".to_string(),
                name: "Tech Conference - Multilingual AI Discussion".to_string(),
                description: "Simulates a tech conference with speakers in different languages discussing AI".to_string(),
                segments: vec![
                    DemoSegment {
                        timestamp_ms: 0,
                        speaker: "Dr. Wang".to_string(),
                        language: "zh-CN".to_string(),
                        original_text: "欢迎大家参加今天的会议。".to_string(),
                        emotion: Emotion::Professional,
                        tone: Tone::Formal,
                        speed: 1.0,
                    },
                    DemoSegment {
                        timestamp_ms: 5000,
                        speaker: "Prof. Smith".to_string(),
                        language: "en-US".to_string(),
                        original_text: "Thank you Dr. Wang. The breakthrough in transformer architecture has revolutionized natural language processing.".to_string(),
                        emotion: Emotion::Excited,
                        tone: Tone::Educational,
                        speed: 1.1,
                    },
                    DemoSegment {
                        timestamp_ms: 10000,
                        speaker: "Marie Dubois".to_string(),
                        language: "fr-FR".to_string(),
                        original_text: "C'est fascinant! Les applications en temps réel sont maintenant possibles avec une latence minimale.".to_string(),
                        emotion: Emotion::Enthusiastic,
                        tone: Tone::Enthusiastic,
                        speed: 1.2,
                    },
                    DemoSegment {
                        timestamp_ms: 15000,
                        speaker: "Tanaka-san".to_string(),
                        language: "ja-JP".to_string(),
                        original_text: "私たちの研究結果を共有します。".to_string(),
                        emotion: Emotion::Professional,
                        tone: Tone::Serious,
                        speed: 0.9,
                    },
                ],
            },

            // Gaming Stream Demo
            DemoScript {
                id: "gaming_stream".to_string(),
                name: "Gaming Stream - International Team Raid".to_string(),
                description: "Simulates an exciting gaming stream with international players".to_string(),
                segments: vec![
                    DemoSegment {
                        timestamp_ms: 0,
                        speaker: "xXDragonKillerXx".to_string(),
                        language: "en-US".to_string(),
                        original_text: "Alright team, boss is at 50% health! Focus fire on the adds!".to_string(),
                        emotion: Emotion::Excited,
                        tone: Tone::Casual,
                        speed: 1.5,
                    },
                    DemoSegment {
                        timestamp_ms: 3000,
                        speaker: "SakuraWarrior".to_string(),
                        language: "ja-JP".to_string(),
                        original_text: "了解！ヒーラー、準備はいいですか？大技が来ます！".to_string(),
                        emotion: Emotion::Excited,
                        tone: Tone::Dramatic,
                        speed: 1.4,
                    },
                    DemoSegment {
                        timestamp_ms: 6000,
                        speaker: "ElMago".to_string(),
                        language: "es-ES".to_string(),
                        original_text: "¡Cuidado! ¡Fase de enrage en 10 segundos! ¡Usen todos los cooldowns!".to_string(),
                        emotion: Emotion::Excited,
                        tone: Tone::Dramatic,
                        speed: 1.6,
                    },
                    DemoSegment {
                        timestamp_ms: 9000,
                        speaker: "김프로".to_string(),
                        language: "ko-KR".to_string(),
                        original_text: "잘했어요! 우리가 해냈습니다! 서버 최초 클리어!".to_string(),
                        emotion: Emotion::Happy,
                        tone: Tone::Enthusiastic,
                        speed: 1.3,
                    },
                ],
            },

            // Educational Lecture Demo
            DemoScript {
                id: "education".to_string(),
                name: "Online Education - Quantum Physics Lecture".to_string(),
                description: "International online classroom discussing quantum mechanics".to_string(),
                segments: vec![
                    DemoSegment {
                        timestamp_ms: 0,
                        speaker: "Prof. Mueller".to_string(),
                        language: "de-DE".to_string(),
                        original_text: "Willkommen zur heutigen Vorlesung über Quantenverschränkung. Dies ist eines der faszinierendsten Phänomene der Quantenmechanik.".to_string(),
                        emotion: Emotion::Professional,
                        tone: Tone::Educational,
                        speed: 0.9,
                    },
                    DemoSegment {
                        timestamp_ms: 7000,
                        speaker: "Student_Alice".to_string(),
                        language: "en-GB".to_string(),
                        original_text: "Professor, could you explain how quantum entanglement relates to quantum computing applications?".to_string(),
                        emotion: Emotion::Confused,
                        tone: Tone::Formal,
                        speed: 1.0,
                    },
                    DemoSegment {
                        timestamp_ms: 12000,
                        speaker: "李同学".to_string(),
                        language: "zh-CN".to_string(),
                        original_text: "教授，量子纠缠是否违反了相对论的光速限制？这个问题一直困扰着我。".to_string(),
                        emotion: Emotion::Confused,
                        tone: Tone::Serious,
                        speed: 1.1,
                    },
                ],
            },

            // Business Meeting Demo
            DemoScript {
                id: "business".to_string(),
                name: "Global Business Meeting - Product Launch".to_string(),
                description: "International team discussing new product launch strategy".to_string(),
                segments: vec![
                    DemoSegment {
                        timestamp_ms: 0,
                        speaker: "CEO Johnson".to_string(),
                        language: "en-US".to_string(),
                        original_text: "Good morning everyone. Today we're finalizing our global launch strategy for the AI translator.".to_string(),
                        emotion: Emotion::Professional,
                        tone: Tone::Formal,
                        speed: 1.0,
                    },
                    DemoSegment {
                        timestamp_ms: 5000,
                        speaker: "市場部長".to_string(),
                        language: "ja-JP".to_string(),
                        original_text: "アジア市場の現状について説明します。".to_string(),
                        emotion: Emotion::Serious,
                        tone: Tone::Formal,
                        speed: 0.95,
                    },
                    DemoSegment {
                        timestamp_ms: 11000,
                        speaker: "Director López".to_string(),
                        language: "es-ES".to_string(),
                        original_text: "En Europa, la privacidad de datos es crítica. Necesitamos procesamiento local sin enviar datos a la nube.".to_string(),
                        emotion: Emotion::Serious,
                        tone: Tone::Professional,
                        speed: 1.0,
                    },
                ],
            },
        ]
    }

    pub async fn start_demo(&mut self, script_id: &str) -> Result<(), String> {
        // Find the script
        let script = self.scripts.iter()
            .find(|s| s.id == script_id)
            .ok_or_else(|| format!("Script '{}' not found", script_id))?
            .clone();

        // Set as current script
        self.current_script = Some(script.clone());
        *self.is_running.write().await = true;

        // Start playing the script
        let subtitle_tx = self.subtitle_tx.clone();
        let is_running = self.is_running.clone();

        tokio::spawn(async move {
            let start_time = std::time::Instant::now();

            for segment in script.segments {
                // Check if still running
                if !*is_running.read().await {
                    break;
                }

                // Wait until the segment timestamp
                let elapsed = start_time.elapsed().as_millis() as u64;
                if segment.timestamp_ms > elapsed {
                    tokio::time::sleep(
                        std::time::Duration::from_millis(segment.timestamp_ms - elapsed)
                    ).await;
                }

                // Translate the text (simulated)
                let translation = Self::simulate_translation(&segment);

                // Create visual effects based on emotion and tone
                let visual_effects = Self::generate_visual_effects(&segment);

                // Send subtitle update
                let update = SubtitleUpdate {
                    timestamp: segment.timestamp_ms,
                    speaker: segment.speaker.clone(),
                    language: segment.language.clone(),
                    original: segment.original_text.clone(),
                    translation,
                    confidence: 0.95 + rand::thread_rng().gen::<f32>() * 0.05,
                    emotion: segment.emotion.clone(),
                    tone: segment.tone.clone(),
                    visual_effects,
                };

                let _ = subtitle_tx.send(update);

                // Simulate word-by-word appearance for realism
                Self::simulate_typing_effect(&segment, &subtitle_tx).await;
            }

            *is_running.write().await = false;
        });

        Ok(())
    }

    fn simulate_translation(segment: &DemoSegment) -> String {
        // Simulate translation to English (or Chinese if source is English)
        match segment.language.as_str() {
            "zh-CN" => match segment.speaker.as_str() {
                "Dr. Wang" => "Welcome everyone to today's meeting.".to_string(),
                "李同学" => "Professor, does quantum entanglement violate the speed of light limit in relativity? This question has been puzzling me.".to_string(),
                _ => "Translation in progress...".to_string(),
            },
            "ja-JP" => match segment.speaker.as_str() {
                "SakuraWarrior" => "Got it! Healers, are you ready? Big attack incoming!".to_string(),
                "市場部長" => "I will explain the current situation in the Asian market.".to_string(),
                "Tanaka-san" => "I will share our research results.".to_string(),
                _ => "Translation in progress...".to_string(),
            },
            "fr-FR" => "It's fascinating! Real-time applications are now possible with minimal latency.".to_string(),
            "es-ES" => match segment.speaker.as_str() {
                "ElMago" => "Watch out! Enrage phase in 10 seconds! Use all cooldowns!".to_string(),
                "Director López" => "In Europe, data privacy is critical. We need local processing without sending data to the cloud.".to_string(),
                _ => "Translation in progress...".to_string(),
            },
            "ko-KR" => "Well done! We did it! Server first clear!".to_string(),
            "de-DE" => "Welcome to today's lecture on quantum entanglement. This is one of the most fascinating phenomena in quantum mechanics.".to_string(),
            _ => segment.original_text.clone(), // Return original if already in target language
        }
    }

    fn generate_visual_effects(segment: &DemoSegment) -> VisualEffects {
        let (color_theme, particle_effect) = match segment.emotion {
            Emotion::Excited | Emotion::Happy => ("#FFD700", true), // Gold with particles
            Emotion::Professional => ("#4A90E2", false), // Professional blue
            Emotion::Confused => ("#FFA500", false), // Orange
            Emotion::Serious => ("#2C3E50", false), // Dark blue-gray
            _ => ("#FFFFFF", false), // Default white
        };

        let animation = match segment.tone {
            Tone::Dramatic => "pulse",
            Tone::Enthusiastic => "bounce",
            Tone::Casual => "slide",
            _ => "fade",
        }.to_string();

        // Extract important words (simplified)
        let words: Vec<String> = segment.original_text
            .split_whitespace()
            .take(2)
            .map(|s| s.to_string())
            .collect();

        VisualEffects {
            color_theme: color_theme.to_string(),
            animation,
            emphasis_words: words,
            particle_effect,
        }
    }

    async fn simulate_typing_effect(
        segment: &DemoSegment,
        subtitle_tx: &broadcast::Sender<SubtitleUpdate>,
    ) {
        // Simulate gradual text appearance
        let words: Vec<&str> = segment.original_text.split_whitespace().collect();
        let delay_ms = (1000.0 / segment.speed / words.len() as f32) as u64;

        for (i, _word) in words.iter().enumerate() {
            tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;

            // Could send partial updates here for word-by-word effect
            // This would create a more realistic typing appearance
        }
    }

    pub async fn stop_demo(&mut self) {
        *self.is_running.write().await = false;
        self.current_script = None;
    }

    pub fn get_available_scripts(&self) -> Vec<(String, String, String)> {
        self.scripts.iter()
            .map(|s| (s.id.clone(), s.name.clone(), s.description.clone()))
            .collect()
    }

    pub fn subscribe(&self) -> broadcast::Receiver<SubtitleUpdate> {
        self.subtitle_tx.subscribe()
    }
}

/// Special effects for hackathon demo
pub struct HackathonEffects {
    pub enable_ai_avatars: bool,
    pub enable_voice_synthesis: bool,
    pub enable_3d_subtitles: bool,
    pub enable_sentiment_particles: bool,
}

impl HackathonEffects {
    pub fn impressive_demo() -> Self {
        Self {
            enable_ai_avatars: true,
            enable_voice_synthesis: true,
            enable_3d_subtitles: true,
            enable_sentiment_particles: true,
        }
    }
}