//! High-quality audio resampling using rubato

use crate::{Error, Result};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};

/// Resample audio from source to target sample rate
pub fn resample(
    samples: &[f32],
    source_rate: u32,
    target_rate: u32,
    channels: u16,
) -> Result<Vec<f32>> {
    if source_rate == target_rate {
        return Ok(samples.to_vec());
    }

    // Use high-quality windowed sinc interpolation
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    // Calculate chunk size for optimal processing
    let chunk_size = 1024;
    let resample_ratio = target_rate as f64 / source_rate as f64;
    let _output_chunk_size = (chunk_size as f64 * resample_ratio).ceil() as usize;

    let mut resampler = SincFixedIn::<f32>::new(
        resample_ratio,
        2.0, // Maximum relative deviation from target rate
        params,
        chunk_size,
        channels as usize,
    )
    .map_err(|e| Error::Audio(format!("Failed to create resampler: {e}")))?;

    // Deinterleave samples if multi-channel
    let deinterleaved = if channels == 1 {
        vec![samples.to_vec()]
    } else {
        deinterleave(samples, channels as usize)
    };

    // Resample each channel
    let mut resampled_channels = Vec::new();
    for channel in deinterleaved {
        let mut output = Vec::new();
        let mut input_frames = channel.chunks(chunk_size).peekable();

        while let Some(chunk) = input_frames.next() {
            let chunk_vec = vec![chunk.to_vec()];
            let is_last = input_frames.peek().is_none();

            let result = if is_last {
                // Process last chunk with partial output
                resampler
                    .process_partial(Some(&chunk_vec), None)
                    .map_err(|e| Error::Audio(format!("Resampling failed: {e}")))?
            } else {
                resampler
                    .process(&chunk_vec, None)
                    .map_err(|e| Error::Audio(format!("Resampling failed: {e}")))?
            };

            if let Some(output_frames) = result.first() {
                output.extend_from_slice(output_frames);
            }
        }

        resampled_channels.push(output);
    }

    // Reinterleave if multi-channel
    let resampled = if channels == 1 {
        resampled_channels.into_iter().next().unwrap()
    } else {
        interleave(&resampled_channels)
    };

    Ok(resampled)
}

/// Deinterleave multi-channel audio
fn deinterleave(samples: &[f32], channels: usize) -> Vec<Vec<f32>> {
    let frames = samples.len() / channels;
    let mut output = vec![Vec::with_capacity(frames); channels];

    for (i, &sample) in samples.iter().enumerate() {
        output[i % channels].push(sample);
    }

    output
}

/// Interleave multi-channel audio
fn interleave(channels: &[Vec<f32>]) -> Vec<f32> {
    let num_channels = channels.len();
    let frames = channels[0].len();
    let mut output = Vec::with_capacity(frames * num_channels);

    for frame_idx in 0..frames {
        for channel in channels {
            output.push(channel[frame_idx]);
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deinterleave_interleave() {
        let stereo = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let deinterleaved = deinterleave(&stereo, 2);

        assert_eq!(deinterleaved[0], vec![1.0, 3.0, 5.0]);
        assert_eq!(deinterleaved[1], vec![2.0, 4.0, 6.0]);

        let reinterleaved = interleave(&deinterleaved);
        assert_eq!(reinterleaved, stereo);
    }

    #[test]
    fn test_no_resampling_needed() {
        let samples = vec![1.0, 2.0, 3.0, 4.0];
        let result = resample(&samples, 16000, 16000, 1).unwrap();
        assert_eq!(result, samples);
    }
}
