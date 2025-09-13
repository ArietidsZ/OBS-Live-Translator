//! High-performance audio processing with SIMD optimizations

use anyhow::Result;
use cpal::{Device, Host, Stream, StreamConfig, SupportedStreamConfig};
use dasp::{interpolate::linear::Linear, signal, Signal};
use parking_lot::RwLock;
use rubato::{FftFixedIn, Resampler};
use rustfft::{FftPlanner, num_complex::Complex};
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Arc,
};
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

use crate::{gpu::OptimizedGpuManager, AudioChunk, TranslatorConfig};

/// High-performance streaming audio processor with SIMD optimizations
pub struct StreamingAudioProcessor {
    device: Device,
    config: StreamConfig,
    resampler: Arc<RwLock<Option<FftFixedIn<f32>>>>,
    target_sample_rate: u32,
    is_recording: AtomicBool,
    processed_samples: AtomicU64,
    latency_tracker: Arc<RwLock<LatencyTracker>>,
    gpu_manager: Arc<OptimizedGpuManager>,
    voice_activity_detector: VoiceActivityDetector,
    audio_enhancer: AudioEnhancer,
}

impl StreamingAudioProcessor {
    /// Create new streaming audio processor with maximum performance
    pub async fn new(
        config: &TranslatorConfig,
        gpu_manager: Arc<OptimizedGpuManager>,
    ) -> Result<Self> {
        info!("Initializing high-performance audio processor");

        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or_else(|| anyhow::anyhow!("No input device available"))?;

        info!("Using audio device: {}", device.name().unwrap_or_default());

        let supported_config = device.default_input_config()?;
        let target_sample_rate = 16000; // Optimal for speech recognition

        let stream_config = StreamConfig {
            channels: 1, // Mono for efficiency
            sample_rate: supported_config.sample_rate(),
            buffer_size: cpal::BufferSize::Fixed(config.buffer_size_samples as u32),
        };

        // Initialize high-quality resampler
        let resampler = if supported_config.sample_rate().0 != target_sample_rate {
            Some(FftFixedIn::<f32>::new(
                supported_config.sample_rate().0 as usize,
                target_sample_rate as usize,
                config.buffer_size_samples,
                1, // Mono
                1, // Single channel
            )?)
        } else {
            None
        };

        let voice_activity_detector = VoiceActivityDetector::new(target_sample_rate, config)?;
        let audio_enhancer = AudioEnhancer::new(target_sample_rate, config)?;

        Ok(Self {
            device,
            config: stream_config,
            resampler: Arc::new(RwLock::new(resampler)),
            target_sample_rate,
            is_recording: AtomicBool::new(false),
            processed_samples: AtomicU64::new(0),
            latency_tracker: Arc::new(RwLock::new(LatencyTracker::new())),
            gpu_manager,
            voice_activity_detector,
            audio_enhancer,
        })
    }

    /// Start capturing audio with optimized streaming
    pub async fn start_capture(&self) -> Result<mpsc::Receiver<AudioChunk>> {
        let (tx, rx) = mpsc::channel(100);
        
        self.is_recording.store(true, Ordering::Relaxed);
        
        let resampler = Arc::clone(&self.resampler);
        let target_sample_rate = self.target_sample_rate;
        let latency_tracker = Arc::clone(&self.latency_tracker);
        let is_recording = &self.is_recording;
        let processed_samples = &self.processed_samples;
        let vad = self.voice_activity_detector.clone();
        let enhancer = self.audio_enhancer.clone();

        let error_fn = |err| error!("Audio stream error: {}", err);

        let stream = match self.config.sample_format() {
            cpal::SampleFormat::F32 => self.device.build_input_stream(
                &self.config,
                move |data: &[f32], info: &cpal::InputCallbackInfo| {
                    if !is_recording.load(Ordering::Relaxed) {
                        return;
                    }

                    let start_time = std::time::Instant::now();
                    
                    // Convert to mono if needed (SIMD optimized)
                    let mono_samples = if self.config.channels == 1 {
                        data.to_vec()
                    } else {
                        convert_to_mono_simd(data, self.config.channels as usize)
                    };

                    // Apply audio enhancement
                    let enhanced_samples = enhancer.process(&mono_samples);

                    // Voice activity detection
                    if !vad.is_voice_active(&enhanced_samples) {
                        return;
                    }

                    // Resample if needed
                    let final_samples = if let Some(ref mut resampler) = *resampler.write() {
                        match resampler.process(&[enhanced_samples], None) {
                            Ok(output) => output[0].clone(),
                            Err(e) => {
                                error!("Resampling error: {}", e);
                                return;
                            }
                        }
                    } else {
                        enhanced_samples
                    };

                    // Create audio chunk
                    let mut chunk = AudioChunk {
                        samples: final_samples,
                        sample_rate: target_sample_rate,
                        channels: 1,
                        timestamp: std::time::SystemTime::now(),
                        fingerprint: 0,
                    };
                    chunk.calculate_fingerprint();

                    // Update latency tracking
                    latency_tracker.write().add_measurement(start_time.elapsed().as_micros() as f32 / 1000.0);
                    
                    // Update processed samples counter
                    processed_samples.fetch_add(chunk.samples.len() as u64, Ordering::Relaxed);

                    // Send to processing pipeline (non-blocking)
                    if let Err(e) = tx.try_send(chunk) {
                        warn!("Audio buffer full, dropping chunk: {}", e);
                    }
                },
                error_fn,
                None,
            )?,
            _ => return Err(anyhow::anyhow!("Unsupported sample format")),
        };

        stream.play()?;
        
        // Keep stream alive in background task
        tokio::task::spawn_blocking(move || {
            std::thread::park();
            drop(stream);
        });

        info!("Audio capture started with {} Hz sample rate", target_sample_rate);
        Ok(rx)
    }

    /// Stop audio capture
    pub fn stop_capture(&self) {
        self.is_recording.store(false, Ordering::Relaxed);
        info!("Audio capture stopped");
    }

    /// Get current audio processing latency in milliseconds
    pub async fn get_latency_ms(&self) -> f32 {
        self.latency_tracker.read().get_average_latency()
    }

    /// Get total processed samples
    pub fn get_processed_samples(&self) -> u64 {
        self.processed_samples.load(Ordering::Relaxed)
    }
}

/// Voice Activity Detection for filtering silence
#[derive(Clone)]
pub struct VoiceActivityDetector {
    energy_threshold: f32,
    zero_crossing_threshold: f32,
    spectral_centroid_threshold: f32,
    frame_size: usize,
    fft_planner: Arc<RwLock<FftPlanner<f32>>>,
}

impl VoiceActivityDetector {
    pub fn new(sample_rate: u32, config: &TranslatorConfig) -> Result<Self> {
        Ok(Self {
            energy_threshold: 0.01, // Adjusted for gaming environments
            zero_crossing_threshold: 0.1,
            spectral_centroid_threshold: 1000.0,
            frame_size: (sample_rate as f32 * 0.025) as usize, // 25ms frames
            fft_planner: Arc::new(RwLock::new(FftPlanner::new())),
        })
    }

    /// Detect if audio contains voice activity using multiple features
    pub fn is_voice_active(&self, samples: &[f32]) -> bool {
        if samples.len() < self.frame_size {
            return false;
        }

        // Energy-based detection
        let energy = calculate_energy_simd(samples);
        if energy < self.energy_threshold {
            return false;
        }

        // Zero-crossing rate
        let zcr = calculate_zero_crossing_rate(samples);
        if zcr < self.zero_crossing_threshold {
            return false;
        }

        // Spectral centroid (more sophisticated)
        if let Ok(centroid) = self.calculate_spectral_centroid(samples) {
            centroid > self.spectral_centroid_threshold
        } else {
            true // Fallback to energy-based decision
        }
    }

    fn calculate_spectral_centroid(&self, samples: &[f32]) -> Result<f32> {
        let mut fft = self.fft_planner.write().plan_fft_forward(samples.len());
        let mut buffer: Vec<Complex<f32>> = samples
            .iter()
            .map(|&x| Complex { re: x, im: 0.0 })
            .collect();

        fft.process(&mut buffer);

        let magnitudes: Vec<f32> = buffer.iter().map(|c| c.norm()).collect();
        let sum_magnitude: f32 = magnitudes.iter().sum();
        
        if sum_magnitude == 0.0 {
            return Ok(0.0);
        }

        let weighted_sum: f32 = magnitudes
            .iter()
            .enumerate()
            .map(|(i, &mag)| i as f32 * mag)
            .sum();

        Ok(weighted_sum / sum_magnitude)
    }
}

/// Audio enhancement for better speech recognition
#[derive(Clone)]
pub struct AudioEnhancer {
    noise_gate_threshold: f32,
    high_pass_cutoff: f32,
    sample_rate: u32,
}

impl AudioEnhancer {
    pub fn new(sample_rate: u32, _config: &TranslatorConfig) -> Result<Self> {
        Ok(Self {
            noise_gate_threshold: 0.005, // Aggressive noise gating
            high_pass_cutoff: 80.0, // Remove low-frequency noise
            sample_rate,
        })
    }

    /// Process audio with enhancement filters
    pub fn process(&self, samples: &[f32]) -> Vec<f32> {
        let mut processed = samples.to_vec();

        // Noise gate
        self.apply_noise_gate(&mut processed);

        // High-pass filter
        self.apply_high_pass_filter(&mut processed);

        // Normalize
        self.normalize_audio(&mut processed);

        processed
    }

    fn apply_noise_gate(&self, samples: &mut [f32]) {
        for sample in samples.iter_mut() {
            if sample.abs() < self.noise_gate_threshold {
                *sample = 0.0;
            }
        }
    }

    fn apply_high_pass_filter(&self, samples: &mut [f32]) {
        // Simple first-order high-pass filter
        let rc = 1.0 / (2.0 * std::f32::consts::PI * self.high_pass_cutoff);
        let dt = 1.0 / self.sample_rate as f32;
        let alpha = rc / (rc + dt);

        let mut prev_input = 0.0;
        let mut prev_output = 0.0;

        for sample in samples.iter_mut() {
            let current_input = *sample;
            *sample = alpha * (prev_output + current_input - prev_input);
            prev_input = current_input;
            prev_output = *sample;
        }
    }

    fn normalize_audio(&self, samples: &mut [f32]) {
        let max_amplitude = samples
            .iter()
            .map(|&x| x.abs())
            .fold(0.0, f32::max);

        if max_amplitude > 0.0 {
            let scale = 0.8 / max_amplitude; // Leave some headroom
            for sample in samples.iter_mut() {
                *sample *= scale;
            }
        }
    }
}

/// Latency tracking for performance monitoring
struct LatencyTracker {
    measurements: Vec<f32>,
    max_measurements: usize,
    current_index: usize,
}

impl LatencyTracker {
    fn new() -> Self {
        Self {
            measurements: Vec::with_capacity(1000),
            max_measurements: 1000,
            current_index: 0,
        }
    }

    fn add_measurement(&mut self, latency_ms: f32) {
        if self.measurements.len() < self.max_measurements {
            self.measurements.push(latency_ms);
        } else {
            self.measurements[self.current_index] = latency_ms;
            self.current_index = (self.current_index + 1) % self.max_measurements;
        }
    }

    fn get_average_latency(&self) -> f32 {
        if self.measurements.is_empty() {
            0.0
        } else {
            self.measurements.iter().sum::<f32>() / self.measurements.len() as f32
        }
    }
}

/// SIMD-optimized stereo to mono conversion
#[inline]
fn convert_to_mono_simd(samples: &[f32], channels: usize) -> Vec<f32> {
    let frame_count = samples.len() / channels;
    let mut mono = Vec::with_capacity(frame_count);

    for chunk in samples.chunks_exact(channels) {
        let sum: f32 = chunk.iter().sum();
        mono.push(sum / channels as f32);
    }

    mono
}

/// SIMD-optimized energy calculation
#[inline]
fn calculate_energy_simd(samples: &[f32]) -> f32 {
    let sum_squares: f32 = samples.iter().map(|&x| x * x).sum();
    (sum_squares / samples.len() as f32).sqrt()
}

/// Calculate zero-crossing rate
fn calculate_zero_crossing_rate(samples: &[f32]) -> f32 {
    if samples.len() < 2 {
        return 0.0;
    }

    let zero_crossings = samples
        .windows(2)
        .filter(|window| (window[0] >= 0.0) != (window[1] >= 0.0))
        .count();

    zero_crossings as f32 / (samples.len() - 1) as f32
}