//! Whisper-specific mel spectrogram processing

use anyhow::Result;

/// Whisper mel spectrogram configuration
pub struct WhisperMelConfig {
    pub sample_rate: u32,
    pub n_fft: usize,
    pub hop_length: usize,
    pub n_mels: usize,
    pub chunk_length: usize, // 30 seconds for Whisper
}

impl Default for WhisperMelConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            n_fft: 400,
            hop_length: 160,
            n_mels: 80,
            chunk_length: 480000, // 30 seconds * 16000 Hz
        }
    }
}

/// Convert audio to Whisper-compatible mel spectrogram
pub fn audio_to_whisper_mel(audio: &[f32], config: &WhisperMelConfig) -> Result<Vec<f32>> {
    // Whisper expects:
    // - 16kHz sample rate
    // - 80 mel bins
    // - 25ms frame size (400 samples)
    // - 10ms hop length (160 samples)
    // - Up to 30 seconds of audio

    // Pad or truncate to 30 seconds
    let mut padded_audio = audio.to_vec();
    if padded_audio.len() < config.chunk_length {
        // Pad with zeros
        padded_audio.resize(config.chunk_length, 0.0);
    } else if padded_audio.len() > config.chunk_length {
        // Truncate
        padded_audio.truncate(config.chunk_length);
    }

    // Calculate number of frames
    let n_frames = 1 + (padded_audio.len() - config.n_fft) / config.hop_length;

    // Create mel spectrogram (simplified - real implementation would use FFT and mel filterbank)
    // For Whisper, we need shape: [80, 3000] (80 mel bins, 3000 time frames for 30 seconds)
    let expected_frames = 3000;
    let mut mel_spectrogram = vec![0.0f32; config.n_mels * expected_frames];

    // Fill with actual computed values (simplified for now)
    for frame_idx in 0..n_frames.min(expected_frames) {
        let start = frame_idx * config.hop_length;
        let end = (start + config.n_fft).min(padded_audio.len());

        if start < padded_audio.len() {
            // Compute energy for this frame (simplified)
            let frame_energy = padded_audio[start..end]
                .iter()
                .map(|x| x * x)
                .sum::<f32>()
                .sqrt();

            // Distribute energy across mel bins (simplified)
            for mel_idx in 0..config.n_mels {
                let idx = mel_idx * expected_frames + frame_idx;
                if idx < mel_spectrogram.len() {
                    mel_spectrogram[idx] = frame_energy * 0.01; // Scale down
                }
            }
        }
    }

    Ok(mel_spectrogram)
}

/// Prepare mel spectrogram for ONNX input
pub fn prepare_mel_for_onnx(mel: &[f32]) -> Vec<f32> {
    // Ensure correct shape [1, 80, 3000] for batch size 1
    // The mel should already be in [80, 3000] format

    if mel.len() != 80 * 3000 {
        // Pad or truncate as needed
        let mut formatted = vec![0.0f32; 80 * 3000];
        let copy_len = mel.len().min(formatted.len());
        formatted[..copy_len].copy_from_slice(&mel[..copy_len]);
        formatted
    } else {
        mel.to_vec()
    }
}
