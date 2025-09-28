//! Unit Tests for OBS Live Translator
//!
//! This module provides comprehensive unit testing for all components:
//! - Audio processing pipeline tests
//! - ML inference accuracy tests
//! - Streaming protocol tests
//! - Configuration system tests

pub mod audio_tests;
pub mod inference_tests;
pub mod streaming_tests;
pub mod config_tests;
pub mod profile_tests;
pub mod monitoring_tests;

#[cfg(test)]
mod test_utils {
    use std::path::PathBuf;
    use std::time::Duration;
    use tokio::time::timeout;

    /// Test timeout duration
    pub const TEST_TIMEOUT: Duration = Duration::from_secs(30);

    /// Get test data directory
    pub fn test_data_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("data")
    }

    /// Get test audio file path
    pub fn test_audio_file() -> PathBuf {
        test_data_dir().join("test_audio.wav")
    }

    /// Assert approximately equal for floating point comparisons
    pub fn assert_approx_eq(a: f32, b: f32, epsilon: f32) {
        assert!(
            (a - b).abs() < epsilon,
            "Values not approximately equal: {} != {} (epsilon: {})",
            a, b, epsilon
        );
    }

    /// Run async test with timeout
    pub async fn run_with_timeout<F, T>(test_fn: F) -> T
    where
        F: std::future::Future<Output = T>,
    {
        timeout(TEST_TIMEOUT, test_fn)
            .await
            .expect("Test timed out")
    }

    /// Generate test audio samples
    pub fn generate_test_audio(duration_ms: u32, sample_rate: u32) -> Vec<f32> {
        let num_samples = (sample_rate * duration_ms / 1000) as usize;
        let mut samples = Vec::with_capacity(num_samples);

        // Generate a simple sine wave
        let frequency = 440.0; // A4 note
        for i in 0..num_samples {
            let t = i as f32 / sample_rate as f32;
            let sample = (2.0 * std::f32::consts::PI * frequency * t).sin();
            samples.push(sample);
        }

        samples
    }
}