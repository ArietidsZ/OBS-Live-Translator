//! FFI bindings for SIMD-optimized audio processing and ONNX inference

use std::os::raw::{c_char, c_float, c_void};
use std::ffi::CString;

// SIMD Audio Processing FFI
extern "C" {
    fn simd_audio_create(frame_size: u32, n_mels: u32) -> *mut c_void;
    fn simd_audio_destroy(processor: *mut c_void);
    fn simd_audio_process_frame(processor: *mut c_void, data: *mut c_float, mel_output: *mut c_float);
    fn simd_compute_energy(data: *const c_float, size: u32) -> c_float;
    fn simd_compute_zero_crossings(data: *const c_float, size: u32) -> u32;
}

// ONNX Inference Engine FFI
extern "C" {
    fn onnx_engine_create() -> *mut c_void;
    fn onnx_engine_destroy(engine: *mut c_void);
    fn onnx_engine_initialize(engine: *mut c_void, model_path: *const c_char, device_type: *const c_char) -> i32;
    fn onnx_engine_run(engine: *mut c_void, input: *const c_float, input_size: u32,
                      output: *mut c_float, output_size: *mut u32) -> i32;
    fn onnx_engine_run_batch(engine: *mut c_void, batch_data: *const c_float,
                            batch_size: u32, sample_size: u32,
                            output_data: *mut c_float, output_sizes: *mut u32) -> i32;
    fn onnx_engine_get_latency_ms(engine: *mut c_void) -> c_float;

    // Whisper-specific
    fn whisper_onnx_create() -> *mut c_void;
    fn whisper_onnx_destroy(model: *mut c_void);
    fn whisper_onnx_initialize(model: *mut c_void, encoder_path: *const c_char,
                              decoder_path: *const c_char, device: *const c_char) -> i32;
    fn whisper_onnx_transcribe(model: *mut c_void, mel_spectrogram: *const c_float,
                              mel_size: u32, tokens: *mut i32, token_count: *mut u32) -> i32;
}

/// SIMD-optimized audio processor
pub struct SimdAudioProcessor {
    ptr: *mut c_void,
    frame_size: usize,
    n_mels: usize,
}

impl SimdAudioProcessor {
    pub fn new(frame_size: usize, n_mels: usize) -> Self {
        unsafe {
            let ptr = simd_audio_create(frame_size as u32, n_mels as u32);
            Self { ptr, frame_size, n_mels }
        }
    }

    pub fn process_frame(&self, data: &mut [f32]) -> Vec<f32> {
        let mut mel_output = vec![0.0f32; self.n_mels];
        unsafe {
            simd_audio_process_frame(self.ptr, data.as_mut_ptr(), mel_output.as_mut_ptr());
        }
        mel_output
    }

    pub fn compute_energy(data: &[f32]) -> f32 {
        unsafe {
            simd_compute_energy(data.as_ptr(), data.len() as u32)
        }
    }

    pub fn compute_zero_crossings(data: &[f32]) -> u32 {
        unsafe {
            simd_compute_zero_crossings(data.as_ptr(), data.len() as u32)
        }
    }
}

impl Drop for SimdAudioProcessor {
    fn drop(&mut self) {
        unsafe {
            simd_audio_destroy(self.ptr);
        }
    }
}

unsafe impl Send for SimdAudioProcessor {}
unsafe impl Sync for SimdAudioProcessor {}

/// High-performance ONNX inference engine
pub struct OnnxEngine {
    ptr: *mut c_void,
}

impl OnnxEngine {
    pub fn new() -> Self {
        unsafe {
            Self { ptr: onnx_engine_create() }
        }
    }

    pub fn initialize(&self, model_path: &str, device: &str) -> Result<(), String> {
        let c_path = CString::new(model_path).map_err(|e| e.to_string())?;
        let c_device = CString::new(device).map_err(|e| e.to_string())?;

        unsafe {
            if onnx_engine_initialize(self.ptr, c_path.as_ptr(), c_device.as_ptr()) == 1 {
                Ok(())
            } else {
                Err("Failed to initialize ONNX engine".to_string())
            }
        }
    }

    pub fn run(&self, input: &[f32]) -> Result<Vec<f32>, String> {
        let mut output = vec![0.0f32; 65536]; // Max output size
        let mut output_size = output.len() as u32;

        unsafe {
            if onnx_engine_run(self.ptr, input.as_ptr(), input.len() as u32,
                             output.as_mut_ptr(), &mut output_size) == 1 {
                output.truncate(output_size as usize);
                Ok(output)
            } else {
                Err("ONNX inference failed".to_string())
            }
        }
    }

    pub fn run_batch(&self, batch: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, String> {
        if batch.is_empty() {
            return Ok(Vec::new());
        }

        let sample_size = batch[0].len();
        let batch_size = batch.len();

        // Flatten batch data
        let mut batch_data = Vec::with_capacity(batch_size * sample_size);
        for sample in batch {
            batch_data.extend_from_slice(sample);
        }

        let mut output_data = vec![0.0f32; batch_size * 65536];
        let mut output_sizes = vec![0u32; batch_size];

        unsafe {
            if onnx_engine_run_batch(self.ptr, batch_data.as_ptr(),
                                    batch_size as u32, sample_size as u32,
                                    output_data.as_mut_ptr(), output_sizes.as_mut_ptr()) == 1 {
                // Unpack batch outputs
                let mut results = Vec::with_capacity(batch_size);
                let mut offset = 0;

                for &size in &output_sizes {
                    let end = offset + size as usize;
                    results.push(output_data[offset..end].to_vec());
                    offset = end;
                }

                Ok(results)
            } else {
                Err("Batch inference failed".to_string())
            }
        }
    }

    pub fn get_latency_ms(&self) -> f32 {
        unsafe {
            onnx_engine_get_latency_ms(self.ptr)
        }
    }
}

impl Drop for OnnxEngine {
    fn drop(&mut self) {
        unsafe {
            onnx_engine_destroy(self.ptr);
        }
    }
}

unsafe impl Send for OnnxEngine {}
unsafe impl Sync for OnnxEngine {}

/// Whisper ONNX model for speech recognition
pub struct WhisperOnnx {
    ptr: *mut c_void,
}

impl WhisperOnnx {
    pub fn new() -> Self {
        unsafe {
            Self { ptr: whisper_onnx_create() }
        }
    }

    pub fn initialize(&self, encoder_path: &str, decoder_path: &str, device: &str) -> Result<(), String> {
        let c_encoder = CString::new(encoder_path).map_err(|e| e.to_string())?;
        let c_decoder = CString::new(decoder_path).map_err(|e| e.to_string())?;
        let c_device = CString::new(device).map_err(|e| e.to_string())?;

        unsafe {
            if whisper_onnx_initialize(self.ptr, c_encoder.as_ptr(),
                                      c_decoder.as_ptr(), c_device.as_ptr()) == 1 {
                Ok(())
            } else {
                Err("Failed to initialize Whisper model".to_string())
            }
        }
    }

    pub fn transcribe(&self, mel_spectrogram: &[f32]) -> Result<Vec<i32>, String> {
        let mut tokens = vec![0i32; 448]; // Max context length
        let mut token_count = tokens.len() as u32;

        unsafe {
            if whisper_onnx_transcribe(self.ptr, mel_spectrogram.as_ptr(),
                                      mel_spectrogram.len() as u32,
                                      tokens.as_mut_ptr(), &mut token_count) == 1 {
                tokens.truncate(token_count as usize);
                Ok(tokens)
            } else {
                Err("Transcription failed".to_string())
            }
        }
    }
}

impl Drop for WhisperOnnx {
    fn drop(&mut self) {
        unsafe {
            whisper_onnx_destroy(self.ptr);
        }
    }
}

unsafe impl Send for WhisperOnnx {}
unsafe impl Sync for WhisperOnnx {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_energy_computation() {
        let data = vec![0.5, -0.3, 0.8, -0.2, 0.1];
        let energy = SimdAudioProcessor::compute_energy(&data);
        assert!(energy > 0.0);
    }

    #[test]
    fn test_simd_zero_crossings() {
        let data = vec![0.5, -0.3, 0.8, -0.2, 0.1];
        let crossings = SimdAudioProcessor::compute_zero_crossings(&data);
        assert_eq!(crossings, 3);
    }
}