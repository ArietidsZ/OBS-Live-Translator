//! Real-time audio processing with windowing and pre-emphasis

use super::{AudioConfig, AudioBuffer, Sample};
use anyhow::{Result, anyhow};
use std::collections::VecDeque;

/// Real-time audio processor with buffering and preprocessing
pub struct AudioProcessor {
    config: AudioConfig,
    buffer: VecDeque<Sample>,
    window: Vec<f32>,
    pre_emphasis_state: f32,
}

impl AudioProcessor {
    pub fn new(config: AudioConfig) -> Self {
        let window = Self::create_hann_window(config.frame_size);

        Self {
            config,
            buffer: VecDeque::new(),
            window,
            pre_emphasis_state: 0.0,
        }
    }

    /// Process incoming audio samples
    pub fn process(&mut self, input: &[Sample]) -> Result<Vec<AudioBuffer>> {
        let mut frames = Vec::new();

        // Add new samples to buffer
        for &sample in input {
            self.buffer.push_back(sample);
        }

        // Extract frames when we have enough samples
        while self.buffer.len() >= self.config.frame_size {
            let frame_data: Vec<Sample> = self.buffer
                .drain(..self.config.hop_length)
                .take(self.config.frame_size)
                .collect();

            let processed_frame = self.preprocess_frame(&frame_data)?;

            frames.push(AudioBuffer {
                data: processed_frame,
                sample_rate: self.config.sample_rate,
                channels: 1,
                timestamp: std::time::Instant::now(),
            });
        }

        Ok(frames)
    }

    /// Preprocess a single frame with windowing and pre-emphasis
    fn preprocess_frame(&mut self, frame: &[Sample]) -> Result<Vec<Sample>> {
        if frame.len() != self.config.frame_size {
            return Err(anyhow!("Frame size mismatch: expected {}, got {}",
                               self.config.frame_size, frame.len()));
        }

        let mut processed = Vec::with_capacity(frame.len());

        // Apply pre-emphasis filter: y[n] = x[n] - α * x[n-1]
        let alpha = 0.97f32;
        for (i, &sample) in frame.iter().enumerate() {
            let emphasized = if i == 0 {
                sample - alpha * self.pre_emphasis_state
            } else {
                sample - alpha * frame[i - 1]
            };
            processed.push(emphasized);
        }

        // Update pre-emphasis state
        if !frame.is_empty() {
            self.pre_emphasis_state = frame[frame.len() - 1];
        }

        // Apply Hann window
        for (i, sample) in processed.iter_mut().enumerate() {
            *sample *= self.window[i];
        }

        Ok(processed)
    }

    /// Create Hann window function
    fn create_hann_window(size: usize) -> Vec<f32> {
        (0..size)
            .map(|i| {
                let phase = 2.0 * std::f32::consts::PI * i as f32 / (size - 1) as f32;
                0.5 * (1.0 - phase.cos())
            })
            .collect()
    }

    /// Get the number of buffered samples
    pub fn buffered_samples(&self) -> usize {
        self.buffer.len()
    }

    /// Clear internal buffers
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.pre_emphasis_state = 0.0;
    }

    /// Check if enough samples are available for processing
    pub fn can_process(&self) -> bool {
        self.buffer.len() >= self.config.frame_size
    }

    /// Get configuration
    pub fn config(&self) -> &AudioConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_processor_creation() {
        let config = AudioConfig::default();
        let processor = AudioProcessor::new(config);
        assert_eq!(processor.buffered_samples(), 0);
    }

    #[test]
    fn test_frame_processing() {
        let config = AudioConfig::default();
        let mut processor = AudioProcessor::new(config);

        // Generate test signal
        let samples: Vec<f32> = (0..1000)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin())
            .collect();

        let frames = processor.process(&samples).unwrap();
        assert!(!frames.is_empty());
    }

    #[test]
    fn test_windowing() {
        let window = AudioProcessor::create_hann_window(512);
        assert_eq!(window.len(), 512);
        assert!((window[0] - 0.0).abs() < 1e-6);
        assert!((window[256] - 1.0).abs() < 1e-6);
    }
}