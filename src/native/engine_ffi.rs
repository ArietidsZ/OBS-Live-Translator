//! FFI bindings for optimized speech recognition engine

use std::ffi::{CString, CStr, c_char, c_float, c_int, c_void};
use std::ptr;

// Optimized Engine FFI
#[link(name = "engine_stub")]
extern "C" {
    // Core engine management
    fn engine_create(model_path: *const c_char, device: *const c_char) -> *mut c_void;
    fn engine_destroy(engine: *mut c_void);

    // Single transcription
    fn engine_transcribe(
        engine: *mut c_void,
        audio: *const c_float,
        audio_size: c_int,
        result_buffer: *mut c_char,
        buffer_size: c_int,
        confidence: *mut c_float,
    ) -> c_int;

    // Batch transcription for maximum throughput
    fn engine_transcribe_batch(
        engine: *mut c_void,
        audio_batch: *const *const c_float,
        sizes: *const c_int,
        batch_size: c_int,
        results: *mut *mut c_char,
        confidences: *mut c_float,
    ) -> c_int;

    // Streaming with VAD
    fn engine_stream_start(engine: *mut c_void) -> c_int;
    fn engine_stream_feed(
        engine: *mut c_void,
        audio: *const c_float,
        size: c_int,
    ) -> c_int;
    fn engine_stream_get_result(
        engine: *mut c_void,
        result_buffer: *mut c_char,
        buffer_size: c_int,
        is_final: *mut bool,
    ) -> c_int;
    fn engine_stream_stop(engine: *mut c_void) -> c_int;

    // Configuration
    fn engine_set_language(engine: *mut c_void, language: *const c_char) -> c_int;
    fn engine_enable_vad(engine: *mut c_void, enable: bool) -> c_int;
    fn engine_set_beam_size(engine: *mut c_void, beam_size: c_int) -> c_int;
    fn engine_enable_word_timestamps(engine: *mut c_void, enable: bool) -> c_int;

    // Performance metrics
    fn engine_get_latency_ms(engine: *mut c_void) -> c_float;
    fn engine_get_throughput_fps(engine: *mut c_void) -> c_float;
    fn engine_get_memory_mb(engine: *mut c_void) -> c_float;
}

/// Optimized ASR Engine for speech recognition
///
/// Features:
/// - Faster-Whisper with CTranslate2 optimization
/// - Batched inference for improved throughput
/// - MLX support on Apple Silicon
/// - TensorRT optimization on NVIDIA
/// - Voice Activity Detection
/// - Streaming with partial results
pub struct OptimizedEngine {
    handle: *mut c_void,
    device: String,
    stream_active: bool,
}

unsafe impl Send for OptimizedEngine {}
unsafe impl Sync for OptimizedEngine {}

impl OptimizedEngine {
    /// Create a new Optimized engine with specified model and device
    ///
    /// Devices:
    /// - "cpu": Optimized CPU inference with INT8 quantization
    /// - "cuda": NVIDIA GPU with TensorRT optimization
    /// - "mlx": Apple Silicon with MLX framework
    /// - "auto": Automatically select best available device
    pub fn new(model_path: &str, device: &str) -> Result<Self, String> {
        let c_path = CString::new(model_path)
            .map_err(|e| format!("Invalid model path: {}", e))?;
        let c_device = CString::new(device)
            .map_err(|e| format!("Invalid device: {}", e))?;

        unsafe {
            let handle = engine_create(c_path.as_ptr(), c_device.as_ptr());
            if handle.is_null() {
                return Err(format!(
                    "Failed to create Optimized engine with model: {} on device: {}",
                    model_path, device
                ));
            }

            Ok(Self {
                handle,
                device: device.to_string(),
                stream_active: false,
            })
        }
    }

    /// Transcribe audio with optimized performance
    pub fn transcribe(&self, audio: &[f32]) -> Result<(String, f32), String> {
        let mut result_buffer = vec![0u8; 4096];
        let mut confidence = 0.0f32;

        unsafe {
            let status = engine_transcribe(
                self.handle,
                audio.as_ptr(),
                audio.len() as c_int,
                result_buffer.as_mut_ptr() as *mut c_char,
                4096,
                &mut confidence,
            );

            if status != 0 {
                return Err(format!("Transcription failed with code: {}", status));
            }

            let c_str = CStr::from_ptr(result_buffer.as_ptr() as *const c_char);
            let text = c_str.to_string_lossy().into_owned();

            Ok((text, confidence))
        }
    }

    /// Batch transcription for improved throughput
    pub fn transcribe_batch(&self, audio_batch: &[Vec<f32>]) -> Result<Vec<(String, f32)>, String> {
        if audio_batch.is_empty() {
            return Ok(Vec::new());
        }

        // Prepare batch data
        let ptrs: Vec<*const c_float> = audio_batch.iter()
            .map(|audio| audio.as_ptr())
            .collect();
        let sizes: Vec<c_int> = audio_batch.iter()
            .map(|audio| audio.len() as c_int)
            .collect();

        // Allocate result buffers
        let batch_size = audio_batch.len();
        let mut result_ptrs = vec![ptr::null_mut::<c_char>(); batch_size];
        let mut confidences = vec![0.0f32; batch_size];

        // Pre-allocate result strings
        let mut result_buffers: Vec<Vec<u8>> = (0..batch_size)
            .map(|_| vec![0u8; 4096])
            .collect();

        for (i, buffer) in result_buffers.iter_mut().enumerate() {
            result_ptrs[i] = buffer.as_mut_ptr() as *mut c_char;
        }

        unsafe {
            let status = engine_transcribe_batch(
                self.handle,
                ptrs.as_ptr(),
                sizes.as_ptr(),
                batch_size as c_int,
                result_ptrs.as_mut_ptr(),
                confidences.as_mut_ptr(),
            );

            if status != 0 {
                return Err(format!("Batch transcription failed with code: {}", status));
            }

            // Convert results
            let mut results = Vec::new();
            for (i, result_ptr) in result_ptrs.iter().enumerate() {
                if !result_ptr.is_null() {
                    let c_str = CStr::from_ptr(*result_ptr);
                    let text = c_str.to_string_lossy().into_owned();
                    results.push((text, confidences[i]));
                } else {
                    results.push((String::new(), 0.0));
                }
            }

            Ok(results)
        }
    }

    /// Start streaming transcription with VAD
    pub fn stream_start(&mut self) -> Result<(), String> {
        if self.stream_active {
            return Err("Stream already active".to_string());
        }

        unsafe {
            let status = engine_stream_start(self.handle);
            if status != 0 {
                return Err(format!("Failed to start stream: {}", status));
            }
        }

        self.stream_active = true;
        Ok(())
    }

    /// Feed audio to streaming session
    pub fn stream_feed(&mut self, audio: &[f32]) -> Result<(), String> {
        if !self.stream_active {
            return Err("Stream not active".to_string());
        }

        unsafe {
            let status = engine_stream_feed(
                self.handle,
                audio.as_ptr(),
                audio.len() as c_int,
            );

            if status != 0 {
                return Err(format!("Failed to feed stream: {}", status));
            }
        }

        Ok(())
    }

    /// Get streaming result (partial or final)
    pub fn stream_get_result(&self) -> Result<(String, bool), String> {
        if !self.stream_active {
            return Err("Stream not active".to_string());
        }

        let mut result_buffer = vec![0u8; 4096];
        let mut is_final = false;

        unsafe {
            let status = engine_stream_get_result(
                self.handle,
                result_buffer.as_mut_ptr() as *mut c_char,
                4096,
                &mut is_final,
            );

            if status != 0 {
                return Err(format!("Failed to get stream result: {}", status));
            }

            let c_str = CStr::from_ptr(result_buffer.as_ptr() as *const c_char);
            let text = c_str.to_string_lossy().into_owned();

            Ok((text, is_final))
        }
    }

    /// Stop streaming session
    pub fn stream_stop(&mut self) -> Result<(), String> {
        if !self.stream_active {
            return Ok(());
        }

        unsafe {
            let status = engine_stream_stop(self.handle);
            if status != 0 {
                return Err(format!("Failed to stop stream: {}", status));
            }
        }

        self.stream_active = false;
        Ok(())
    }

    /// Set language for transcription
    pub fn set_language(&self, language: &str) -> Result<(), String> {
        let c_lang = CString::new(language)
            .map_err(|e| format!("Invalid language: {}", e))?;

        unsafe {
            let status = engine_set_language(self.handle, c_lang.as_ptr());
            if status != 0 {
                return Err(format!("Failed to set language: {}", language));
            }
        }

        Ok(())
    }

    /// Enable/disable Voice Activity Detection
    pub fn enable_vad(&self, enable: bool) -> Result<(), String> {
        unsafe {
            let status = engine_enable_vad(self.handle, enable);
            if status != 0 {
                return Err("Failed to configure VAD".to_string());
            }
        }

        Ok(())
    }

    /// Set beam size for beam search decoding
    pub fn set_beam_size(&self, beam_size: i32) -> Result<(), String> {
        unsafe {
            let status = engine_set_beam_size(self.handle, beam_size as c_int);
            if status != 0 {
                return Err(format!("Failed to set beam size: {}", beam_size));
            }
        }

        Ok(())
    }

    /// Enable word-level timestamps
    pub fn enable_word_timestamps(&self, enable: bool) -> Result<(), String> {
        unsafe {
            let status = engine_enable_word_timestamps(self.handle, enable);
            if status != 0 {
                return Err("Failed to configure word timestamps".to_string());
            }
        }

        Ok(())
    }

    /// Get current latency in milliseconds
    pub fn get_latency_ms(&self) -> f32 {
        unsafe { engine_get_latency_ms(self.handle) }
    }

    /// Get throughput in frames per second
    pub fn get_throughput_fps(&self) -> f32 {
        unsafe { engine_get_throughput_fps(self.handle) }
    }

    /// Get memory usage in megabytes
    pub fn get_memory_mb(&self) -> f32 {
        unsafe { engine_get_memory_mb(self.handle) }
    }

    /// Get device being used
    pub fn get_device(&self) -> &str {
        &self.device
    }
}

impl Drop for OptimizedEngine {
    fn drop(&mut self) {
        // Stop stream if active
        if self.stream_active {
            let _ = self.stream_stop();
        }

        unsafe {
            engine_destroy(self.handle);
        }
    }
}

/// Builder for configuring Optimized engine
pub struct OptimizedEngineBuilder {
    model_path: String,
    device: String,
    language: Option<String>,
    beam_size: Option<i32>,
    enable_vad: bool,
    enable_word_timestamps: bool,
}

impl OptimizedEngineBuilder {
    pub fn new(model_path: impl Into<String>) -> Self {
        Self {
            model_path: model_path.into(),
            device: "auto".to_string(),
            language: None,
            beam_size: None,
            enable_vad: false,
            enable_word_timestamps: false,
        }
    }

    pub fn device(mut self, device: impl Into<String>) -> Self {
        self.device = device.into();
        self
    }

    pub fn language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    pub fn beam_size(mut self, beam_size: i32) -> Self {
        self.beam_size = Some(beam_size);
        self
    }

    pub fn enable_vad(mut self, enable: bool) -> Self {
        self.enable_vad = enable;
        self
    }

    pub fn enable_word_timestamps(mut self, enable: bool) -> Self {
        self.enable_word_timestamps = enable;
        self
    }

    pub fn build(self) -> Result<OptimizedEngine, String> {
        let engine = OptimizedEngine::new(&self.model_path, &self.device)?;

        // Apply configuration
        if let Some(language) = self.language {
            engine.set_language(&language)?;
        }

        if let Some(beam_size) = self.beam_size {
            engine.set_beam_size(beam_size)?;
        }

        engine.enable_vad(self.enable_vad)?;
        engine.enable_word_timestamps(self.enable_word_timestamps)?;

        Ok(engine)
    }
}

