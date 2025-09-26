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
    frame_buffer: Vec<Sample>,
    processed_buffer: Vec<Sample>,
}

impl AudioProcessor {
    pub fn new(config: AudioConfig) -> Self {
        let window = Self::create_hann_window(config.frame_size);
        let buffer_capacity = config.frame_size * 4;
        let mut buffer = VecDeque::with_capacity(buffer_capacity);
        buffer.reserve(buffer_capacity);

        let frame_size = config.frame_size;

        Self {
            config,
            buffer,
            window,
            pre_emphasis_state: 0.0,
            frame_buffer: Vec::with_capacity(frame_size),
            processed_buffer: Vec::with_capacity(frame_size),
        }
    }

    /// Process incoming audio samples
    pub fn process(&mut self, input: &[Sample]) -> Result<Vec<AudioBuffer>> {
        let max_frames = (self.buffer.len() + input.len()) / self.config.hop_length;
        let mut frames = Vec::with_capacity(max_frames);

        // Add new samples to buffer
        self.buffer.extend(input.iter());

        // Extract frames when we have enough samples
        while self.buffer.len() >= self.config.frame_size {
            self.frame_buffer.clear();
            for _ in 0..self.config.frame_size {
                if let Some(sample) = self.buffer.pop_front() {
                    self.frame_buffer.push(sample);
                }
            }

            // Put back samples for overlap (frame_size - hop_length)
            for &sample in self.frame_buffer.iter().skip(self.config.hop_length) {
                self.buffer.push_front(sample);
            }

            let processed_frame = self.preprocess_frame_internal()?;

            frames.push(AudioBuffer {
                data: processed_frame,
                sample_rate: self.config.sample_rate,
                channels: 1,
                timestamp: std::time::Instant::now(),
            });
        }

        Ok(frames)
    }

    /// Preprocess internal frame buffer
    fn preprocess_frame_internal(&mut self) -> Result<Vec<Sample>> {
        self.preprocess_frame(&self.frame_buffer.clone())
    }

    /// Preprocess a single frame with windowing and pre-emphasis
    fn preprocess_frame(&mut self, frame: &[Sample]) -> Result<Vec<Sample>> {
        if frame.len() != self.config.frame_size {
            return Err(anyhow!("Frame size mismatch: expected {}, got {}",
                               self.config.frame_size, frame.len()));
        }

        self.processed_buffer.clear();
        self.processed_buffer.reserve(frame.len());

        // Apply pre-emphasis filter: y[n] = x[n] - α * x[n-1]
        const ALPHA: f32 = 0.97;
        let mut prev_sample = self.pre_emphasis_state;

        for &sample in frame.iter() {
            let emphasized = sample - ALPHA * prev_sample;
            self.processed_buffer.push(emphasized);
            prev_sample = sample;
        }

        // Update pre-emphasis state
        self.pre_emphasis_state = prev_sample;

        // Apply Hann window in-place
        for (i, sample) in self.processed_buffer.iter_mut().enumerate() {
            *sample *= self.window[i];
        }

        Ok(self.processed_buffer.clone())
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

