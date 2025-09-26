//! Audio resampling for format conversion

use super::Sample;
use anyhow::Result;

/// Audio resampler for converting between sample rates
pub struct Resampler {
    source_rate: u32,
    target_rate: u32,
    ratio: f64,
    buffer: Vec<Sample>,
}

impl Resampler {
    pub fn new(source_rate: u32, target_rate: u32) -> Self {
        Self {
            source_rate,
            target_rate,
            ratio: target_rate as f64 / source_rate as f64,
            buffer: Vec::new(),
        }
    }

    /// Resample audio using linear interpolation
    pub fn resample(&mut self, input: &[Sample]) -> Result<Vec<Sample>> {
        if self.source_rate == self.target_rate {
            return Ok(input.to_vec());
        }

        // Add input to buffer
        self.buffer.extend_from_slice(input);

        let output_len = (self.buffer.len() as f64 * self.ratio) as usize;
        let mut output = Vec::with_capacity(output_len);

        for i in 0..output_len {
            let source_index = i as f64 / self.ratio;
            let index_floor = source_index.floor() as usize;
            let index_ceil = (index_floor + 1).min(self.buffer.len() - 1);
            let fraction = source_index - index_floor as f64;

            if index_floor < self.buffer.len() {
                let sample = if index_floor == index_ceil {
                    self.buffer[index_floor]
                } else {
                    // Linear interpolation
                    let a = self.buffer[index_floor];
                    let b = self.buffer[index_ceil];
                    a + (b - a) * fraction as f32
                };
                output.push(sample);
            }
        }

        // Keep some buffer for next iteration
        let consumed = (output.len() as f64 / self.ratio) as usize;
        if consumed < self.buffer.len() {
            self.buffer.drain(..consumed);
        } else {
            self.buffer.clear();
        }

        Ok(output)
    }

    /// Convert stereo to mono by averaging channels
    pub fn stereo_to_mono(stereo: &[Sample]) -> Vec<Sample> {
        stereo
            .chunks(2)
            .map(|chunk| {
                if chunk.len() == 2 {
                    (chunk[0] + chunk[1]) * 0.5
                } else {
                    chunk[0]
                }
            })
            .collect()
    }

    /// Convert mono to stereo by duplicating channel
    pub fn mono_to_stereo(mono: &[Sample]) -> Vec<Sample> {
        let mut stereo = Vec::with_capacity(mono.len() * 2);
        for &sample in mono {
            stereo.push(sample);
            stereo.push(sample);
        }
        stereo
    }

    /// Normalize audio to prevent clipping
    pub fn normalize(audio: &mut [Sample]) {
        let max_amplitude = audio.iter().map(|&x| x.abs()).fold(0.0, f32::max);
        if max_amplitude > 1e-8 {
            let scale = 0.95 / max_amplitude; // Leave some headroom
            for sample in audio.iter_mut() {
                *sample *= scale;
            }
        }
    }

    /// Apply DC offset removal (high-pass filter)
    pub fn remove_dc_offset(audio: &mut [Sample]) {
        if audio.is_empty() {
            return;
        }

        // Simple high-pass filter: y[n] = x[n] - x[n-1] + 0.99 * y[n-1]
        let mut prev_input = 0.0;
        let mut prev_output = 0.0;

        for sample in audio.iter_mut() {
            let output = *sample - prev_input + 0.99 * prev_output;
            prev_input = *sample;
            prev_output = output;
            *sample = output;
        }
    }

    /// Get resampling ratio
    pub fn ratio(&self) -> f64 {
        self.ratio
    }

    /// Clear internal buffers
    pub fn reset(&mut self) {
        self.buffer.clear();
    }
}

