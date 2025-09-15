//! NVIDIA Canary-1B Flash ASR Model Integration
//!
//! Ultra-high-performance streaming ASR with >1000 RTFx performance
//! Supports English, German, French, Spanish with multilingual capabilities

use anyhow::Result;
use ort::{Environment, ExecutionProvider, GraphOptimizationLevel, Session, SessionBuilder};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, atomic::{AtomicU64, Ordering}};
use std::time::{Duration, Instant};
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, error, info, warn, instrument};

use crate::gpu::adaptive_memory::{ModelPrecision, ModelType};
use crate::gpu::hardware_detection::HardwareDetector;
use crate::streaming::advanced_cache::{CacheKey, ContentFingerprint};
use crate::acceleration::inline_asm_optimizations::timing;

/// NVIDIA Canary-1B Flash ASR model with streaming capabilities
pub struct CanaryFlashModel {
    /// ONNX Runtime session
    session: Arc<Session>,
    /// Model configuration
    config: CanaryConfig,
    /// Hardware detector for optimization
    hardware_detector: Arc<HardwareDetector>,
    /// Performance metrics
    metrics: Arc<CanaryMetrics>,
    /// Input preprocessing cache
    feature_cache: Arc<RwLock<HashMap<CacheKey, CanaryFeatures>>>,
    /// Streaming state for continuous recognition
    streaming_state: Arc<Mutex<StreamingState>>,
    /// Language detection confidence tracker
    language_tracker: Arc<RwLock<LanguageTracker>>,
}

impl CanaryFlashModel {
    /// Create new Canary Flash model instance
    pub async fn new(
        model_path: &Path,
        config: CanaryConfig,
        hardware_detector: Arc<HardwareDetector>,
    ) -> Result<Self> {
        info!("Initializing NVIDIA Canary-1B Flash ASR model");

        let start_time = Instant::now();

        // Configure ONNX Runtime for maximum performance
        let environment = Environment::builder()
            .with_name("canary_flash")
            .with_log_level(ort::LoggingLevel::Warning)
            .build()?;

        // Select optimal execution provider based on hardware
        let execution_providers = Self::select_execution_providers(&hardware_detector).await;

        let mut session_builder = SessionBuilder::new(&environment)?
            .with_optimization_level(GraphOptimizationLevel::All)
            .with_intra_threads(config.intra_op_threads)
            .with_inter_threads(config.inter_op_threads);

        // Add execution providers in priority order
        for provider in execution_providers {
            session_builder = session_builder.with_execution_providers([provider])?;
        }

        let session = Arc::new(session_builder.with_model_from_file(model_path)?);

        // Verify model inputs/outputs
        Self::validate_model_signature(&session)?;

        let model = Self {
            session,
            config,
            hardware_detector,
            metrics: Arc::new(CanaryMetrics::new()),
            feature_cache: Arc::new(RwLock::new(HashMap::new())),
            streaming_state: Arc::new(Mutex::new(StreamingState::new())),
            language_tracker: Arc::new(RwLock::new(LanguageTracker::new())),
        };

        let initialization_time = start_time.elapsed();
        info!("Canary Flash model initialized in {}ms", initialization_time.as_millis());

        // Warm up model with dummy input
        model.warmup().await?;

        Ok(model)
    }

    /// Perform streaming ASR on audio chunk
    #[instrument(skip(self, audio_data))]
    pub async fn transcribe_streaming(
        &self,
        audio_data: &[f32],
        stream_id: u64,
        language_hint: Option<&str>,
    ) -> Result<CanaryStreamingResult> {
        let start_time = unsafe { timing::rdtsc() };

        // Extract features with caching
        let features = self.extract_features_cached(audio_data, stream_id).await?;

        // Get streaming state
        let mut streaming_state = self.streaming_state.lock().await;

        // Update context with new features
        streaming_state.update_context(&features, stream_id);

        // Prepare model inputs
        let input_tensors = self.prepare_model_inputs(&features, &streaming_state, language_hint).await?;

        // Run inference with performance monitoring
        let inference_start = unsafe { timing::rdtsc() };
        let outputs = self.session.run(input_tensors)?;
        let inference_time = unsafe { timing::rdtsc() - inference_start };

        // Process model outputs
        let result = self.process_streaming_outputs(
            &outputs,
            &mut streaming_state,
            stream_id,
            language_hint,
        ).await?;

        // Update performance metrics
        let total_time = unsafe { timing::rdtsc() - start_time };
        self.metrics.record_inference(inference_time, total_time, audio_data.len()).await;

        // Update language detection
        self.update_language_detection(&result, stream_id).await;

        debug!("Canary Flash transcription completed: {} chars in {}μs",
               result.text.len(), total_time / 1000);

        Ok(result)
    }

    /// Batch transcription for multiple audio chunks
    pub async fn transcribe_batch(
        &self,
        audio_batch: Vec<&[f32]>,
        stream_ids: Vec<u64>,
        language_hints: Vec<Option<&str>>,
    ) -> Result<Vec<CanaryStreamingResult>> {
        info!("Processing batch of {} audio chunks", audio_batch.len());

        let start_time = Instant::now();

        // Extract features for all chunks in parallel
        let feature_futures: Vec<_> = audio_batch
            .iter()
            .zip(&stream_ids)
            .map(|(audio, &stream_id)| {
                self.extract_features_cached(audio, stream_id)
            })
            .collect();

        let features_batch = futures::future::try_join_all(feature_futures).await?;

        // Prepare batch inputs
        let batch_size = audio_batch.len();
        let input_tensors = self.prepare_batch_inputs(&features_batch, &language_hints).await?;

        // Run batch inference
        let inference_start = Instant::now();
        let outputs = self.session.run(input_tensors)?;
        let inference_time = inference_start.elapsed();

        // Process batch outputs
        let results = self.process_batch_outputs(
            &outputs,
            &stream_ids,
            &language_hints,
            batch_size,
        ).await?;

        let total_time = start_time.elapsed();

        info!("Batch transcription completed: {} chunks in {}ms (avg: {}ms/chunk)",
              batch_size, total_time.as_millis(), total_time.as_millis() / batch_size as u128);

        // Update batch metrics
        self.metrics.record_batch_inference(
            inference_time,
            total_time,
            batch_size,
            audio_batch.iter().map(|a| a.len()).sum(),
        ).await;

        Ok(results)
    }

    /// Get real-time performance metrics
    pub async fn get_metrics(&self) -> CanaryPerformanceMetrics {
        self.metrics.get_current_metrics().await
    }

    /// Reset streaming state for a specific stream
    pub async fn reset_stream(&self, stream_id: u64) {
        let mut state = self.streaming_state.lock().await;
        state.reset_stream(stream_id);

        let mut tracker = self.language_tracker.write().await;
        tracker.reset_stream(stream_id);

        debug!("Reset streaming state for stream {}", stream_id);
    }

    /// Get detected language for stream
    pub async fn get_detected_language(&self, stream_id: u64) -> Option<DetectedLanguage> {
        let tracker = self.language_tracker.read().await;
        tracker.get_language(stream_id)
    }

    // Private implementation methods

    async fn select_execution_providers(
        hardware_detector: &Arc<HardwareDetector>,
    ) -> Vec<ExecutionProvider> {
        let mut providers = Vec::new();
        let gpu_info = hardware_detector.get_gpu_info();

        // NVIDIA CUDA + TensorRT (highest priority)
        if gpu_info.vendor.contains("NVIDIA") {
            if gpu_info.supports_tensorrt {
                providers.push(ExecutionProvider::TensorRT(Default::default()));
            }
            providers.push(ExecutionProvider::CUDA(Default::default()));
        }

        // AMD ROCm
        if gpu_info.vendor.contains("AMD") {
            providers.push(ExecutionProvider::ROCm(Default::default()));
        }

        // Intel OpenVINO
        if gpu_info.vendor.contains("Intel") {
            providers.push(ExecutionProvider::OpenVINO(Default::default()));
        }

        // Apple CoreML/Metal
        #[cfg(target_os = "macos")]
        if gpu_info.vendor.contains("Apple") {
            providers.push(ExecutionProvider::CoreML(Default::default()));
        }

        // CPU fallback
        providers.push(ExecutionProvider::CPU(Default::default()));

        info!("Selected execution providers: {:?}",
              providers.iter().map(|p| format!("{:?}", p)).collect::<Vec<_>>());

        providers
    }

    fn validate_model_signature(session: &Session) -> Result<()> {
        let inputs = session.inputs.len();
        let outputs = session.outputs.len();

        info!("Canary Flash model signature: {} inputs, {} outputs", inputs, outputs);

        // Validate expected inputs for Canary model
        if inputs < 1 {
            return Err(anyhow::anyhow!("Invalid Canary model: insufficient inputs"));
        }

        if outputs < 1 {
            return Err(anyhow::anyhow!("Invalid Canary model: insufficient outputs"));
        }

        // Log input/output details
        for (i, input) in session.inputs.iter().enumerate() {
            debug!("Input {}: {} {:?}", i, input.name, input.input_type);
        }

        for (i, output) in session.outputs.iter().enumerate() {
            debug!("Output {}: {} {:?}", i, output.name, output.output_type);
        }

        Ok(())
    }

    async fn warmup(&self) -> Result<()> {
        info!("Warming up Canary Flash model");

        // Create dummy audio (1 second at 16kHz)
        let dummy_audio = vec![0.0f32; 16000];

        // Perform warmup inference
        let _result = self.transcribe_streaming(&dummy_audio, 0, Some("en")).await?;

        info!("Model warmup completed successfully");
        Ok(())
    }

    async fn extract_features_cached(
        &self,
        audio_data: &[f32],
        stream_id: u64,
    ) -> Result<CanaryFeatures> {
        // Generate cache key based on audio fingerprint
        let fingerprint = ContentFingerprint::from_audio(audio_data);
        let cache_key = CacheKey::new(format!("canary_features_{}_{}", stream_id, fingerprint.hash));

        // Check cache first
        {
            let cache = self.feature_cache.read().await;
            if let Some(cached_features) = cache.get(&cache_key) {
                self.metrics.record_cache_hit().await;
                return Ok(cached_features.clone());
            }
        }

        // Extract features
        let features = self.extract_features(audio_data).await?;

        // Cache the result
        {
            let mut cache = self.feature_cache.write().await;
            cache.insert(cache_key, features.clone());
        }

        self.metrics.record_cache_miss().await;
        Ok(features)
    }

    async fn extract_features(&self, audio_data: &[f32]) -> Result<CanaryFeatures> {
        // Canary Flash uses Mel-scale features similar to Whisper
        let n_fft = 400;
        let hop_length = 160;
        let n_mels = 80;

        // Pad or truncate audio to expected length
        let target_length = 16000; // 1 second at 16kHz
        let mut processed_audio = if audio_data.len() >= target_length {
            audio_data[..target_length].to_vec()
        } else {
            let mut padded = audio_data.to_vec();
            padded.resize(target_length, 0.0);
            padded
        };

        // Apply pre-emphasis filter
        Self::apply_preemphasis(&mut processed_audio);

        // Extract Mel spectrogram features
        let mel_features = Self::extract_mel_spectrogram(
            &processed_audio,
            16000,
            n_fft,
            hop_length,
            n_mels,
        )?;

        Ok(CanaryFeatures {
            mel_spectrogram: mel_features,
            sequence_length: processed_audio.len(),
            sample_rate: 16000,
        })
    }

    fn apply_preemphasis(audio: &mut [f32]) {
        let alpha = 0.97;
        for i in (1..audio.len()).rev() {
            audio[i] = audio[i] - alpha * audio[i - 1];
        }
    }

    fn extract_mel_spectrogram(
        audio: &[f32],
        sample_rate: usize,
        n_fft: usize,
        hop_length: usize,
        n_mels: usize,
    ) -> Result<Vec<Vec<f32>>> {
        // Simplified Mel spectrogram extraction
        // In production, this would use a full FFT library like RustFFT

        let n_frames = (audio.len() - n_fft) / hop_length + 1;
        let mut mel_spectrogram = vec![vec![0.0; n_mels]; n_frames];

        // Apply Hann window and compute magnitude spectrogram
        let window: Vec<f32> = (0..n_fft)
            .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (n_fft - 1) as f32).cos()))
            .collect();

        for frame in 0..n_frames {
            let start = frame * hop_length;

            // Extract windowed frame
            let mut windowed_frame = vec![0.0; n_fft];
            for i in 0..n_fft {
                if start + i < audio.len() {
                    windowed_frame[i] = audio[start + i] * window[i];
                }
            }

            // Simplified magnitude spectrum (would use FFT in production)
            for mel_bin in 0..n_mels {
                let mut magnitude = 0.0;
                let bin_start = mel_bin * n_fft / (2 * n_mels);
                let bin_end = (mel_bin + 1) * n_fft / (2 * n_mels);

                for bin in bin_start..bin_end.min(windowed_frame.len()) {
                    magnitude += windowed_frame[bin].abs();
                }

                mel_spectrogram[frame][mel_bin] = magnitude.max(1e-10).ln();
            }
        }

        Ok(mel_spectrogram)
    }

    async fn prepare_model_inputs(
        &self,
        features: &CanaryFeatures,
        streaming_state: &StreamingState,
        language_hint: Option<&str>,
    ) -> Result<HashMap<String, ort::Value>> {
        let mut inputs = HashMap::new();

        // Prepare input tensors based on Canary model requirements
        let mel_tensor = self.create_mel_tensor(&features.mel_spectrogram)?;
        inputs.insert("input_features".to_string(), mel_tensor);

        // Add language tokens if specified
        if let Some(lang) = language_hint {
            let lang_token = self.get_language_token(lang);
            let lang_tensor = self.create_language_tensor(lang_token)?;
            inputs.insert("language".to_string(), lang_tensor);
        }

        // Add streaming context if available
        if let Some(context) = streaming_state.get_context() {
            let context_tensor = self.create_context_tensor(context)?;
            inputs.insert("context".to_string(), context_tensor);
        }

        Ok(inputs)
    }

    fn create_mel_tensor(&self, mel_spectrogram: &[Vec<f32>]) -> Result<ort::Value> {
        let n_frames = mel_spectrogram.len();
        let n_mels = mel_spectrogram[0].len();

        // Flatten the mel spectrogram for tensor creation
        let flattened: Vec<f32> = mel_spectrogram.iter().flatten().cloned().collect();

        // Create tensor with shape [1, n_mels, n_frames] (batch, features, time)
        let tensor = ort::Value::from_array(
            ([1, n_mels, n_frames], flattened.as_slice())
        )?;

        Ok(tensor)
    }

    fn create_language_tensor(&self, language_token: i64) -> Result<ort::Value> {
        let tensor = ort::Value::from_array(([1], [language_token].as_slice()))?;
        Ok(tensor)
    }

    fn create_context_tensor(&self, context: &[i64]) -> Result<ort::Value> {
        let tensor = ort::Value::from_array(([1, context.len()], context))?;
        Ok(tensor)
    }

    fn get_language_token(&self, language: &str) -> i64 {
        // Canary Flash language tokens
        match language {
            "en" => 1,  // English
            "de" => 2,  // German
            "fr" => 3,  // French
            "es" => 4,  // Spanish
            _ => 1,     // Default to English
        }
    }

    async fn prepare_batch_inputs(
        &self,
        features_batch: &[CanaryFeatures],
        language_hints: &[Option<&str>],
    ) -> Result<HashMap<String, ort::Value>> {
        let batch_size = features_batch.len();
        let mut inputs = HashMap::new();

        // Prepare batch mel spectrogram tensor
        let max_frames = features_batch.iter()
            .map(|f| f.mel_spectrogram.len())
            .max()
            .unwrap_or(0);
        let n_mels = features_batch[0].mel_spectrogram[0].len();

        let mut batch_mel = vec![0.0f32; batch_size * n_mels * max_frames];

        for (batch_idx, features) in features_batch.iter().enumerate() {
            for (frame_idx, frame) in features.mel_spectrogram.iter().enumerate() {
                for (mel_idx, &value) in frame.iter().enumerate() {
                    let tensor_idx = batch_idx * (n_mels * max_frames) +
                                   mel_idx * max_frames + frame_idx;
                    batch_mel[tensor_idx] = value;
                }
            }
        }

        let mel_tensor = ort::Value::from_array(
            ([batch_size, n_mels, max_frames], batch_mel.as_slice())
        )?;
        inputs.insert("input_features".to_string(), mel_tensor);

        // Prepare batch language tokens
        let language_tokens: Vec<i64> = language_hints.iter()
            .map(|hint| hint.map(|lang| self.get_language_token(lang)).unwrap_or(1))
            .collect();

        let lang_tensor = ort::Value::from_array(
            ([batch_size], language_tokens.as_slice())
        )?;
        inputs.insert("language".to_string(), lang_tensor);

        Ok(inputs)
    }

    async fn process_streaming_outputs(
        &self,
        outputs: &HashMap<String, ort::Value>,
        streaming_state: &mut StreamingState,
        stream_id: u64,
        language_hint: Option<&str>,
    ) -> Result<CanaryStreamingResult> {
        // Extract logits from model output
        let logits_output = outputs.get("logits")
            .ok_or_else(|| anyhow::anyhow!("Missing logits output"))?;

        let logits = logits_output.try_extract_tensor::<f32>()?;

        // Convert logits to tokens using greedy decoding
        let tokens = self.decode_tokens(&logits.view())?;

        // Convert tokens to text
        let text = self.decode_text(&tokens)?;

        // Extract confidence scores
        let confidence = self.calculate_confidence(&logits.view());

        // Update streaming state
        streaming_state.add_tokens(stream_id, &tokens);

        Ok(CanaryStreamingResult {
            text,
            tokens,
            confidence,
            language: language_hint.unwrap_or("en").to_string(),
            is_final: false, // Streaming results are incremental
            word_timestamps: Vec::new(), // Would be extracted from alignment output
        })
    }

    async fn process_batch_outputs(
        &self,
        outputs: &HashMap<String, ort::Value>,
        stream_ids: &[u64],
        language_hints: &[Option<&str>],
        batch_size: usize,
    ) -> Result<Vec<CanaryStreamingResult>> {
        let logits_output = outputs.get("logits")
            .ok_or_else(|| anyhow::anyhow!("Missing logits output"))?;

        let logits = logits_output.try_extract_tensor::<f32>()?;
        let logits_view = logits.view();

        let mut results = Vec::with_capacity(batch_size);

        for batch_idx in 0..batch_size {
            // Extract logits for this batch item
            let item_logits = logits_view.slice(s![batch_idx, .., ..]);

            // Decode tokens and text
            let tokens = self.decode_tokens(&item_logits)?;
            let text = self.decode_text(&tokens)?;
            let confidence = self.calculate_confidence(&item_logits);

            results.push(CanaryStreamingResult {
                text,
                tokens,
                confidence,
                language: language_hints[batch_idx].unwrap_or("en").to_string(),
                is_final: true, // Batch results are final
                word_timestamps: Vec::new(),
            });
        }

        Ok(results)
    }

    fn decode_tokens(&self, logits: &ndarray::ArrayView3<f32>) -> Result<Vec<i64>> {
        // Greedy decoding: select token with highest probability at each step
        let mut tokens = Vec::new();

        for time_step in 0..logits.shape()[1] {
            let step_logits = logits.slice(s![0, time_step, ..]);

            // Find token with maximum logit value
            let max_token = step_logits.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as i64)
                .unwrap_or(0);

            tokens.push(max_token);
        }

        Ok(tokens)
    }

    fn decode_text(&self, tokens: &[i64]) -> Result<String> {
        // Convert tokens to text using Canary tokenizer
        // In production, this would use the actual tokenizer
        let text = tokens.iter()
            .filter_map(|&token| {
                // Simple token-to-char mapping for demonstration
                if token > 0 && token < 128 {
                    Some(token as u8 as char)
                } else {
                    None
                }
            })
            .collect::<String>()
            .trim()
            .to_string();

        Ok(text)
    }

    fn calculate_confidence(&self, logits: &ndarray::ArrayView3<f32>) -> f32 {
        // Calculate average confidence across all time steps
        let mut total_confidence = 0.0;
        let mut count = 0;

        for time_step in 0..logits.shape()[1] {
            let step_logits = logits.slice(s![0, time_step, ..]);

            // Apply softmax and get max probability
            let max_logit = step_logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exp_sum: f32 = step_logits.iter().map(|&x| (x - max_logit).exp()).sum();
            let max_prob = (max_logit - max_logit).exp() / exp_sum;

            total_confidence += max_prob;
            count += 1;
        }

        if count > 0 {
            total_confidence / count as f32
        } else {
            0.0
        }
    }

    async fn update_language_detection(
        &self,
        result: &CanaryStreamingResult,
        stream_id: u64,
    ) {
        let mut tracker = self.language_tracker.write().await;
        tracker.update(stream_id, &result.language, result.confidence);
    }
}

// Supporting data structures

#[derive(Debug, Clone)]
pub struct CanaryConfig {
    pub model_precision: ModelPrecision,
    pub intra_op_threads: usize,
    pub inter_op_threads: usize,
    pub enable_streaming: bool,
    pub context_window: usize,
    pub beam_size: usize,
    pub temperature: f32,
}

impl Default for CanaryConfig {
    fn default() -> Self {
        Self {
            model_precision: ModelPrecision::FP16,
            intra_op_threads: 4,
            inter_op_threads: 2,
            enable_streaming: true,
            context_window: 512,
            beam_size: 1, // Greedy decoding for speed
            temperature: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
struct CanaryFeatures {
    mel_spectrogram: Vec<Vec<f32>>,
    sequence_length: usize,
    sample_rate: usize,
}

#[derive(Debug, Clone)]
pub struct CanaryStreamingResult {
    pub text: String,
    pub tokens: Vec<i64>,
    pub confidence: f32,
    pub language: String,
    pub is_final: bool,
    pub word_timestamps: Vec<WordTimestamp>,
}

#[derive(Debug, Clone)]
pub struct WordTimestamp {
    pub word: String,
    pub start_time: f32,
    pub end_time: f32,
    pub confidence: f32,
}

struct StreamingState {
    contexts: HashMap<u64, StreamContext>,
}

impl StreamingState {
    fn new() -> Self {
        Self {
            contexts: HashMap::new(),
        }
    }

    fn update_context(&mut self, features: &CanaryFeatures, stream_id: u64) {
        let context = self.contexts.entry(stream_id).or_insert_with(StreamContext::new);
        context.add_features(features);
    }

    fn add_tokens(&mut self, stream_id: u64, tokens: &[i64]) {
        if let Some(context) = self.contexts.get_mut(&stream_id) {
            context.add_tokens(tokens);
        }
    }

    fn get_context(&self) -> Option<&[i64]> {
        // Return most recent context tokens
        self.contexts.values()
            .next()
            .map(|ctx| ctx.get_tokens())
    }

    fn reset_stream(&mut self, stream_id: u64) {
        self.contexts.remove(&stream_id);
    }
}

struct StreamContext {
    token_history: Vec<i64>,
    max_history: usize,
}

impl StreamContext {
    fn new() -> Self {
        Self {
            token_history: Vec::new(),
            max_history: 256,
        }
    }

    fn add_features(&mut self, _features: &CanaryFeatures) {
        // Update context based on features
    }

    fn add_tokens(&mut self, tokens: &[i64]) {
        self.token_history.extend_from_slice(tokens);

        // Keep only recent history
        if self.token_history.len() > self.max_history {
            self.token_history.drain(0..self.token_history.len() - self.max_history);
        }
    }

    fn get_tokens(&self) -> &[i64] {
        &self.token_history
    }
}

struct LanguageTracker {
    streams: HashMap<u64, DetectedLanguage>,
}

impl LanguageTracker {
    fn new() -> Self {
        Self {
            streams: HashMap::new(),
        }
    }

    fn update(&mut self, stream_id: u64, language: &str, confidence: f32) {
        let detected = self.streams.entry(stream_id).or_insert_with(|| DetectedLanguage {
            language: language.to_string(),
            confidence: 0.0,
            sample_count: 0,
        });

        // Update with exponential moving average
        detected.confidence = 0.9 * detected.confidence + 0.1 * confidence;
        detected.sample_count += 1;

        // Update language if confidence is high enough
        if confidence > 0.8 && confidence > detected.confidence {
            detected.language = language.to_string();
        }
    }

    fn get_language(&self, stream_id: u64) -> Option<DetectedLanguage> {
        self.streams.get(&stream_id).cloned()
    }

    fn reset_stream(&mut self, stream_id: u64) {
        self.streams.remove(&stream_id);
    }
}

#[derive(Debug, Clone)]
pub struct DetectedLanguage {
    pub language: String,
    pub confidence: f32,
    pub sample_count: usize,
}

struct CanaryMetrics {
    inference_count: AtomicU64,
    total_inference_time: AtomicU64,
    total_processing_time: AtomicU64,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
    batch_count: AtomicU64,
}

impl CanaryMetrics {
    fn new() -> Self {
        Self {
            inference_count: AtomicU64::new(0),
            total_inference_time: AtomicU64::new(0),
            total_processing_time: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            batch_count: AtomicU64::new(0),
        }
    }

    async fn record_inference(&self, inference_time: u64, total_time: u64, audio_samples: usize) {
        self.inference_count.fetch_add(1, Ordering::Relaxed);
        self.total_inference_time.fetch_add(inference_time, Ordering::Relaxed);
        self.total_processing_time.fetch_add(total_time, Ordering::Relaxed);

        // Calculate and log RTF (Real-Time Factor)
        let audio_duration_us = (audio_samples as u64 * 1_000_000) / 16000; // Assume 16kHz
        let rtf = if audio_duration_us > 0 {
            total_time as f64 / audio_duration_us as f64
        } else {
            0.0
        };

        if rtf > 0.0 {
            debug!("Canary RTF: {:.4} (processing time: {}μs, audio: {}μs)",
                   rtf, total_time, audio_duration_us);
        }
    }

    async fn record_batch_inference(&self, inference_time: Duration, total_time: Duration, batch_size: usize, total_samples: usize) {
        self.batch_count.fetch_add(1, Ordering::Relaxed);
        self.inference_count.fetch_add(batch_size as u64, Ordering::Relaxed);
        self.total_inference_time.fetch_add(inference_time.as_nanos() as u64, Ordering::Relaxed);
        self.total_processing_time.fetch_add(total_time.as_nanos() as u64, Ordering::Relaxed);

        info!("Batch processing: {} items in {}ms ({}ms/item)",
              batch_size, total_time.as_millis(), total_time.as_millis() / batch_size as u128);
    }

    async fn record_cache_hit(&self) {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    async fn record_cache_miss(&self) {
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
    }

    async fn get_current_metrics(&self) -> CanaryPerformanceMetrics {
        let inference_count = self.inference_count.load(Ordering::Relaxed);
        let total_inference_time = self.total_inference_time.load(Ordering::Relaxed);
        let total_processing_time = self.total_processing_time.load(Ordering::Relaxed);
        let cache_hits = self.cache_hits.load(Ordering::Relaxed);
        let cache_misses = self.cache_misses.load(Ordering::Relaxed);

        CanaryPerformanceMetrics {
            total_inferences: inference_count,
            average_inference_time_us: if inference_count > 0 {
                total_inference_time / inference_count
            } else {
                0
            },
            average_processing_time_us: if inference_count > 0 {
                total_processing_time / inference_count
            } else {
                0
            },
            cache_hit_rate: if cache_hits + cache_misses > 0 {
                cache_hits as f32 / (cache_hits + cache_misses) as f32
            } else {
                0.0
            },
            throughput_rtf: if total_processing_time > 0 {
                // Estimate based on 16kHz audio
                (inference_count * 16000) as f64 / (total_processing_time / 1_000_000) as f64
            } else {
                0.0
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct CanaryPerformanceMetrics {
    pub total_inferences: u64,
    pub average_inference_time_us: u64,
    pub average_processing_time_us: u64,
    pub cache_hit_rate: f32,
    pub throughput_rtf: f64,
}

use ndarray::{s, ArrayView3};