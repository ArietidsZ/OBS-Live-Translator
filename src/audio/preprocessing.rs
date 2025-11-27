//! Audio preprocessing utilities

/// Audio preprocessor
pub struct AudioPreprocessor;

impl Default for AudioPreprocessor {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioPreprocessor {
    pub fn new() -> Self {
        Self
    }

    /// Normalize audio samples to [-1.0, 1.0]
    pub fn normalize(&self, samples: &mut [f32]) {
        let max_abs = samples.iter().map(|&s| s.abs()).fold(0.0f32, f32::max);

        if max_abs > 0.0 {
            let scale = 1.0 / max_abs;
            for sample in samples.iter_mut() {
                *sample *= scale;
            }
        }
    }
}
