//! Audio Quality Assurance
//!
//! Implements audio quality metrics including SNR monitoring, distortion detection,
//! frequency response validation, and latency measurement

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Audio quality result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioQualityResult {
    /// Signal-to-noise ratio in dB
    pub snr_db: f32,
    /// Peak level in dB
    pub peak_level_db: f32,
    /// RMS level in dB
    pub rms_level_db: f32,
    /// Frequency response metrics
    pub frequency_response: FrequencyResponse,
    /// Audio distortion metrics
    pub distortion: DistortionMetrics,
    /// Latency measurement
    pub latency_ms: f32,
    /// Detected audio issues
    pub issues: Vec<AudioIssue>,
    /// Overall quality score (0.0 to 1.0)
    pub quality_score: f32,
}

impl AudioQualityResult {
    pub fn has_issues(&self) -> bool {
        !self.issues.is_empty()
    }
}

/// Frequency response analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyResponse {
    /// Low frequency energy (< 250 Hz)
    pub low_freq_energy: f32,
    /// Mid frequency energy (250 Hz - 4 kHz)
    pub mid_freq_energy: f32,
    /// High frequency energy (> 4 kHz)
    pub high_freq_energy: f32,
    /// Spectral centroid in Hz
    pub spectral_centroid: f32,
    /// Spectral rolloff frequency
    pub spectral_rolloff: f32,
}

/// Distortion metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistortionMetrics {
    /// Total harmonic distortion percentage
    pub thd_percent: f32,
    /// Clipping detected
    pub has_clipping: bool,
    /// Number of clipped samples
    pub clipped_samples: usize,
    /// DC offset
    pub dc_offset: f32,
}

/// Audio quality issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioIssue {
    pub issue_type: AudioIssueType,
    pub severity: IssueSeverity,
    pub description: String,
    pub timestamp_ms: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AudioIssueType {
    LowSNR,
    Clipping,
    Silence,
    HighNoise,
    FrequencyImbalance,
    DCOffset,
    Distortion,
    Dropout,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IssueSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Audio quality assurance engine
pub struct AudioQA {
    config: super::QualityConfig,
    history: Arc<RwLock<Vec<AudioQualityResult>>>,
    noise_profile: Arc<RwLock<Option<NoiseProfile>>>,
}

/// Noise profile for SNR calculation
#[derive(Debug, Clone)]
struct NoiseProfile {
    noise_floor: f32,
    noise_spectrum: Vec<f32>,
}

impl AudioQA {
    pub async fn new(config: super::QualityConfig) -> Result<Self> {
        Ok(Self {
            config,
            history: Arc::new(RwLock::new(Vec::new())),
            noise_profile: Arc::new(RwLock::new(None)),
        })
    }

    /// Analyze audio quality
    pub async fn analyze(
        &self,
        audio_data: &[f32],
        sample_rate: u32,
    ) -> Result<AudioQualityResult> {
        let mut issues = Vec::new();

        // Calculate basic metrics
        let (peak_level_db, rms_level_db) = self.calculate_levels(audio_data);

        // Calculate SNR
        let snr_db = self.calculate_snr(audio_data).await;

        // Analyze frequency response
        let frequency_response = self.analyze_frequency_response(audio_data, sample_rate);

        // Detect distortion
        let distortion = self.detect_distortion(audio_data);

        // Measure latency (simplified - in production use timestamps)
        let latency_ms = self.measure_latency();

        // Detect issues
        self.detect_audio_issues(audio_data, snr_db, peak_level_db, &distortion, &mut issues);

        // Calculate overall quality score
        let quality_score = self.calculate_quality_score(
            snr_db,
            peak_level_db,
            &frequency_response,
            &distortion,
            &issues,
        );

        let result = AudioQualityResult {
            snr_db,
            peak_level_db,
            rms_level_db,
            frequency_response,
            distortion,
            latency_ms,
            issues,
            quality_score,
        };

        // Update history
        self.update_history(result.clone()).await;

        Ok(result)
    }

    /// Calculate peak and RMS levels
    fn calculate_levels(&self, audio_data: &[f32]) -> (f32, f32) {
        if audio_data.is_empty() {
            return (-100.0, -100.0);
        }

        let peak = audio_data
            .iter()
            .map(|s| s.abs())
            .fold(0.0f32, |a, b| a.max(b));

        let sum_squares: f32 = audio_data.iter().map(|s| s * s).sum();
        let rms = (sum_squares / audio_data.len() as f32).sqrt();

        let peak_db = 20.0 * peak.max(1e-10).log10();
        let rms_db = 20.0 * rms.max(1e-10).log10();

        (peak_db, rms_db)
    }

    /// Calculate signal-to-noise ratio
    async fn calculate_snr(&self, audio_data: &[f32]) -> f32 {
        // Get or estimate noise profile
        let noise_profile = self.noise_profile.read().await;

        let noise_floor = if let Some(profile) = noise_profile.as_ref() {
            profile.noise_floor
        } else {
            // Estimate noise floor from quiet sections
            self.estimate_noise_floor(audio_data)
        };

        // Calculate signal power
        let signal_power: f32 =
            audio_data.iter().map(|s| s * s).sum::<f32>() / audio_data.len() as f32;

        // Calculate SNR
        if noise_floor > 0.0 && signal_power > noise_floor {
            10.0 * (signal_power / noise_floor).log10()
        } else {
            60.0 // Default good SNR
        }
    }

    /// Estimate noise floor from audio
    fn estimate_noise_floor(&self, audio_data: &[f32]) -> f32 {
        // Find quietest 10% of samples
        let mut sorted_abs: Vec<f32> = audio_data.iter().map(|s| s.abs()).collect();
        sorted_abs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let percentile_10 = sorted_abs.len() / 10;
        let quiet_samples = &sorted_abs[..percentile_10];

        let noise_power: f32 =
            quiet_samples.iter().map(|s| s * s).sum::<f32>() / quiet_samples.len().max(1) as f32;

        noise_power
    }

    /// Analyze frequency response
    fn analyze_frequency_response(
        &self,
        audio_data: &[f32],
        sample_rate: u32,
    ) -> FrequencyResponse {
        // Simplified frequency analysis using zero-crossing rate and energy
        let mut low_energy = 0.0;
        let mut mid_energy = 0.0;
        let mut high_energy = 0.0;

        // Apply simple filters (in production, use proper FFT)
        let window_size = 512;
        let windows = audio_data.chunks(window_size);

        for window in windows {
            if window.len() < window_size {
                continue;
            }

            // Estimate frequency content using zero-crossing rate
            let zcr = self.calculate_zero_crossing_rate(window);
            let energy: f32 = window.iter().map(|s| s * s).sum();

            // Rough frequency classification based on ZCR
            let estimated_freq = zcr * sample_rate as f32 / 2.0;

            if estimated_freq < 250.0 {
                low_energy += energy;
            } else if estimated_freq < 4000.0 {
                mid_energy += energy;
            } else {
                high_energy += energy;
            }
        }

        let total_energy = low_energy + mid_energy + high_energy;

        if total_energy > 0.0 {
            low_energy /= total_energy;
            mid_energy /= total_energy;
            high_energy /= total_energy;
        }

        // Calculate spectral centroid (simplified)
        let spectral_centroid = 250.0 * low_energy + 2000.0 * mid_energy + 6000.0 * high_energy;

        // Calculate spectral rolloff (simplified)
        let spectral_rolloff = if high_energy > 0.1 {
            8000.0
        } else if mid_energy > 0.3 {
            4000.0
        } else {
            2000.0
        };

        FrequencyResponse {
            low_freq_energy: low_energy,
            mid_freq_energy: mid_energy,
            high_freq_energy: high_energy,
            spectral_centroid,
            spectral_rolloff,
        }
    }

    /// Calculate zero-crossing rate
    fn calculate_zero_crossing_rate(&self, window: &[f32]) -> f32 {
        let mut crossings = 0;

        for i in 1..window.len() {
            if (window[i - 1] >= 0.0 && window[i] < 0.0)
                || (window[i - 1] < 0.0 && window[i] >= 0.0)
            {
                crossings += 1;
            }
        }

        crossings as f32 / window.len() as f32
    }

    /// Detect audio distortion
    fn detect_distortion(&self, audio_data: &[f32]) -> DistortionMetrics {
        let mut clipped_samples = 0;
        let dc_offset = audio_data.iter().sum::<f32>() / audio_data.len().max(1) as f32;

        // Detect clipping
        for sample in audio_data {
            if sample.abs() >= 0.99 {
                clipped_samples += 1;
            }
        }

        let has_clipping = clipped_samples > audio_data.len() / 1000; // More than 0.1% clipped

        // Estimate THD (simplified)
        let thd_percent = if has_clipping {
            10.0 // High distortion if clipping
        } else {
            // Estimate based on peak-to-average ratio
            let peak = audio_data
                .iter()
                .map(|s| s.abs())
                .fold(0.0f32, |a, b| a.max(b));
            let avg = audio_data.iter().map(|s| s.abs()).sum::<f32>() / audio_data.len() as f32;

            if avg > 0.0 {
                ((peak / avg - 3.0).max(0.0) * 2.0).min(5.0)
            } else {
                0.0
            }
        };

        DistortionMetrics {
            thd_percent,
            has_clipping,
            clipped_samples,
            dc_offset: dc_offset.abs(),
        }
    }

    /// Measure latency
    fn measure_latency(&self) -> f32 {
        // In production, measure actual processing latency
        // For now, return a simulated value
        10.0 // 10ms
    }

    /// Detect audio issues
    fn detect_audio_issues(
        &self,
        audio_data: &[f32],
        snr_db: f32,
        peak_level_db: f32,
        distortion: &DistortionMetrics,
        issues: &mut Vec<AudioIssue>,
    ) {
        // Check SNR
        if snr_db < 20.0 {
            issues.push(AudioIssue {
                issue_type: AudioIssueType::LowSNR,
                severity: if snr_db < 10.0 {
                    IssueSeverity::High
                } else {
                    IssueSeverity::Medium
                },
                description: format!("Low signal-to-noise ratio: {:.1} dB", snr_db),
                timestamp_ms: None,
            });
        }

        // Check for clipping
        if distortion.has_clipping {
            issues.push(AudioIssue {
                issue_type: AudioIssueType::Clipping,
                severity: IssueSeverity::High,
                description: format!(
                    "Audio clipping detected: {} samples",
                    distortion.clipped_samples
                ),
                timestamp_ms: None,
            });
        }

        // Check for silence
        if peak_level_db < -60.0 {
            issues.push(AudioIssue {
                issue_type: AudioIssueType::Silence,
                severity: IssueSeverity::Medium,
                description: "Audio is too quiet or silent".to_string(),
                timestamp_ms: None,
            });
        }

        // Check DC offset
        if distortion.dc_offset > 0.1 {
            issues.push(AudioIssue {
                issue_type: AudioIssueType::DCOffset,
                severity: IssueSeverity::Low,
                description: format!("DC offset detected: {:.3}", distortion.dc_offset),
                timestamp_ms: None,
            });
        }

        // Check for distortion
        if distortion.thd_percent > 5.0 {
            issues.push(AudioIssue {
                issue_type: AudioIssueType::Distortion,
                severity: if distortion.thd_percent > 10.0 {
                    IssueSeverity::High
                } else {
                    IssueSeverity::Medium
                },
                description: format!("High distortion: {:.1}% THD", distortion.thd_percent),
                timestamp_ms: None,
            });
        }

        // Check for dropouts
        let silence_threshold = 0.001;
        let mut silence_count = 0;
        let mut in_silence = false;

        for sample in audio_data {
            if sample.abs() < silence_threshold {
                if !in_silence {
                    silence_count += 1;
                    in_silence = true;
                }
            } else {
                in_silence = false;
            }
        }

        if silence_count > 10 {
            issues.push(AudioIssue {
                issue_type: AudioIssueType::Dropout,
                severity: IssueSeverity::Medium,
                description: format!("Audio dropouts detected: {silence_count} occurrences"),
                timestamp_ms: None,
            });
        }
    }

    /// Calculate overall quality score
    fn calculate_quality_score(
        &self,
        snr_db: f32,
        peak_level_db: f32,
        frequency_response: &FrequencyResponse,
        distortion: &DistortionMetrics,
        issues: &[AudioIssue],
    ) -> f32 {
        let mut score = 1.0;

        // SNR contribution (0 to 0.3)
        let snr_score = ((snr_db - 10.0) / 40.0).clamp(0.0, 1.0) * 0.3;
        score = f32::min(score, snr_score + 0.7);

        // Level contribution (0 to 0.2)
        let level_score = if peak_level_db > -60.0 && peak_level_db < -3.0 {
            0.2
        } else {
            0.1
        };
        score = f32::min(score, level_score + 0.8);

        // Frequency balance contribution (0 to 0.2)
        let freq_balance = 1.0
            - (frequency_response.low_freq_energy - 0.3).abs()
            - (frequency_response.mid_freq_energy - 0.5).abs()
            - (frequency_response.high_freq_energy - 0.2).abs();
        let freq_score = freq_balance.max(0.0) * 0.2;
        score = f32::min(score, freq_score + 0.8);

        // Distortion penalty
        if distortion.has_clipping {
            score *= 0.7;
        }
        score *= (1.0 - distortion.thd_percent / 20.0).max(0.5);

        // Issue penalty
        for issue in issues {
            match issue.severity {
                IssueSeverity::Critical => score *= 0.5,
                IssueSeverity::High => score *= 0.7,
                IssueSeverity::Medium => score *= 0.85,
                IssueSeverity::Low => score *= 0.95,
            }
        }

        score.clamp(0.0, 1.0)
    }

    /// Update history
    async fn update_history(&self, result: AudioQualityResult) {
        let mut history = self.history.write().await;
        history.push(result);

        // Keep only last 100 results
        if history.len() > 100 {
            history.remove(0);
        }
    }

    /// Update noise profile for better SNR calculation
    pub async fn update_noise_profile(&self, noise_audio: &[f32]) {
        let noise_floor = self.estimate_noise_floor(noise_audio);

        // In production, calculate actual noise spectrum using FFT
        let noise_spectrum = vec![noise_floor; 256];

        let profile = NoiseProfile {
            noise_floor,
            noise_spectrum,
        };

        *self.noise_profile.write().await = Some(profile);
    }
}
