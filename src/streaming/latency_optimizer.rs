use std::sync::{Arc, RwLock};
use std::sync::atomic::{AtomicU64, AtomicF64, Ordering};
use std::collections::{VecDeque, HashMap};
use std::time::{Instant, Duration};
use tokio::time::timeout;
use crossbeam_queue::SegQueue;
use dashmap::DashMap;
use metrics::{histogram, gauge, counter};
use serde::{Serialize, Deserialize};

use crate::streaming::unified_pipeline::{AudioChunk, TranslationResult, OptimizationLevel};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyOptimizerConfig {
    pub target_latency_ms: u32,
    pub max_latency_ms: u32,
    pub adaptation_rate: f32,
    pub prediction_window_ms: u32,
    pub quality_threshold: f32,
    pub enable_predictive_scheduling: bool,
    pub enable_dynamic_batching: bool,
    pub enable_speculative_execution: bool,
}

#[derive(Debug, Clone)]
pub struct LatencyProfile {
    pub p50_ms: f32,
    pub p95_ms: f32,
    pub p99_ms: f32,
    pub avg_ms: f32,
    pub jitter_ms: f32,
    pub throughput_chunks_per_sec: f32,
}

#[derive(Debug, Clone)]
pub struct ProcessingStageMetrics {
    pub asr_latency: LatencyProfile,
    pub translation_latency: LatencyProfile,
    pub cultural_adaptation_latency: LatencyProfile,
    pub synthesis_latency: LatencyProfile,
    pub total_pipeline_latency: LatencyProfile,
}

#[derive(Debug, Clone)]
pub struct AdaptiveQualitySettings {
    pub optimization_level: OptimizationLevel,
    pub model_precision: ModelPrecision,
    pub batch_size: usize,
    pub chunk_size_ms: u32,
    pub lookahead_ms: u32,
    pub enable_speculative_decoding: bool,
    pub cpu_threads: usize,
    pub gpu_streams: usize,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ModelPrecision {
    FP32,      // Full precision
    FP16,      // Half precision
    FP8,       // 8-bit precision (TensorRT)
    INT8,      // 8-bit integer
    INT4,      // 4-bit integer (experimental)
}

pub struct LatencyOptimizer {
    config: LatencyOptimizerConfig,

    // Real-time latency tracking
    latency_history: Arc<RwLock<VecDeque<(Instant, u32)>>>,
    stage_metrics: Arc<RwLock<ProcessingStageMetrics>>,

    // Adaptive quality management
    quality_settings: Arc<RwLock<AdaptiveQualitySettings>>,
    quality_history: Arc<RwLock<VecDeque<f32>>>,

    // Predictive scheduling
    processing_queue: Arc<SegQueue<ScheduledTask>>,
    prediction_model: Arc<RwLock<LatencyPredictionModel>>,

    // Dynamic batching
    batch_optimizer: Arc<RwLock<BatchOptimizer>>,

    // Performance counters
    total_optimizations: AtomicU64,
    latency_violations: AtomicU64,
    quality_adjustments: AtomicU64,
    prediction_accuracy: AtomicF64,
}

#[derive(Debug, Clone)]
struct ScheduledTask {
    chunk: AudioChunk,
    priority: TaskPriority,
    deadline: Instant,
    estimated_latency_ms: u32,
    retry_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum TaskPriority {
    Critical = 0,    // < 50ms deadline
    High = 1,        // < 100ms deadline
    Normal = 2,      // < 200ms deadline
    Low = 3,         // > 200ms deadline
}

#[derive(Debug)]
struct LatencyPredictionModel {
    // Moving averages for different processing stages
    asr_ma: MovingAverage,
    translation_ma: MovingAverage,
    cultural_ma: MovingAverage,
    synthesis_ma: MovingAverage,

    // Contextual predictors
    language_latency_map: HashMap<String, f32>,
    chunk_size_latency_map: HashMap<u32, f32>,
    load_factor_predictor: LoadFactorPredictor,

    // Model confidence
    prediction_errors: VecDeque<f32>,
    confidence_score: f32,
}

#[derive(Debug)]
struct MovingAverage {
    window_size: usize,
    values: VecDeque<f32>,
    sum: f32,
}

#[derive(Debug)]
struct LoadFactorPredictor {
    cpu_usage_history: VecDeque<f32>,
    gpu_usage_history: VecDeque<f32>,
    memory_usage_history: VecDeque<f32>,
    concurrent_streams: usize,
    predicted_load: f32,
}

#[derive(Debug)]
struct BatchOptimizer {
    target_batch_size: usize,
    min_batch_size: usize,
    max_batch_size: usize,
    batch_latency_map: HashMap<usize, f32>,
    current_batch: Vec<AudioChunk>,
    batch_deadline: Option<Instant>,
    adaptive_batching_enabled: bool,
}

impl Default for LatencyOptimizerConfig {
    fn default() -> Self {
        Self {
            target_latency_ms: 150,
            max_latency_ms: 300,
            adaptation_rate: 0.1,
            prediction_window_ms: 1000,
            quality_threshold: 0.85,
            enable_predictive_scheduling: true,
            enable_dynamic_batching: true,
            enable_speculative_execution: true,
        }
    }
}

impl Default for AdaptiveQualitySettings {
    fn default() -> Self {
        Self {
            optimization_level: OptimizationLevel::Balanced,
            model_precision: ModelPrecision::FP16,
            batch_size: 4,
            chunk_size_ms: 100,
            lookahead_ms: 200,
            enable_speculative_decoding: true,
            cpu_threads: num_cpus::get(),
            gpu_streams: 4,
        }
    }
}

impl LatencyOptimizer {
    pub fn new(config: LatencyOptimizerConfig) -> Self {
        Self {
            config: config.clone(),
            latency_history: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
            stage_metrics: Arc::new(RwLock::new(ProcessingStageMetrics::default())),
            quality_settings: Arc::new(RwLock::new(AdaptiveQualitySettings::default())),
            quality_history: Arc::new(RwLock::new(VecDeque::with_capacity(100))),
            processing_queue: Arc::new(SegQueue::new()),
            prediction_model: Arc::new(RwLock::new(LatencyPredictionModel::new())),
            batch_optimizer: Arc::new(RwLock::new(BatchOptimizer::new())),
            total_optimizations: AtomicU64::new(0),
            latency_violations: AtomicU64::new(0),
            quality_adjustments: AtomicU64::new(0),
            prediction_accuracy: AtomicF64::new(0.0),
        }
    }

    pub async fn optimize_chunk_processing(&self, chunk: AudioChunk) -> Result<OptimizedProcessingPlan, Box<dyn std::error::Error + Send + Sync>> {
        let start_time = Instant::now();

        // Predict processing latency
        let predicted_latency = self.predict_processing_latency(&chunk).await?;

        // Determine task priority based on deadline
        let priority = self.calculate_task_priority(&chunk, predicted_latency);

        // Adapt quality settings if needed
        if predicted_latency > self.config.target_latency_ms {
            self.adapt_quality_settings(predicted_latency).await?;
        }

        // Create optimized processing plan
        let plan = OptimizedProcessingPlan {
            chunk: chunk.clone(),
            priority,
            estimated_latency_ms: predicted_latency,
            quality_settings: self.quality_settings.read().unwrap().clone(),
            use_speculative_execution: self.should_use_speculative_execution(&chunk),
            batch_with_others: self.should_batch_processing(&chunk),
            processing_hints: self.generate_processing_hints(&chunk),
        };

        // Schedule the task
        if self.config.enable_predictive_scheduling {
            self.schedule_task(ScheduledTask {
                chunk,
                priority,
                deadline: start_time + Duration::from_millis(self.config.target_latency_ms as u64),
                estimated_latency_ms: predicted_latency,
                retry_count: 0,
            });
        }

        self.total_optimizations.fetch_add(1, Ordering::Relaxed);

        // Record optimization metrics
        histogram!("latency_optimization_time", start_time.elapsed());
        gauge!("predicted_latency_ms", predicted_latency as f64);

        Ok(plan)
    }

    async fn predict_processing_latency(&self, chunk: &AudioChunk) -> Result<u32, Box<dyn std::error::Error + Send + Sync>> {
        let prediction_model = self.prediction_model.read().unwrap();

        // Base predictions from historical data
        let base_asr_latency = prediction_model.asr_ma.average();
        let base_translation_latency = prediction_model.translation_ma.average();
        let base_cultural_latency = prediction_model.cultural_ma.average();
        let base_synthesis_latency = prediction_model.synthesis_ma.average();

        // Contextual adjustments
        let language_factor = chunk.language_hint
            .as_ref()
            .and_then(|lang| prediction_model.language_latency_map.get(lang))
            .unwrap_or(&1.0);

        let chunk_size_factor = prediction_model.chunk_size_latency_map
            .get(&chunk.data.len() as u32)
            .unwrap_or(&1.0);

        let load_factor = prediction_model.load_factor_predictor.predicted_load;

        // Combine predictions
        let total_predicted_latency = (
            base_asr_latency +
            base_translation_latency +
            base_cultural_latency +
            base_synthesis_latency
        ) * language_factor * chunk_size_factor * load_factor;

        Ok(total_predicted_latency as u32)
    }

    fn calculate_task_priority(&self, chunk: &AudioChunk, predicted_latency: u32) -> TaskPriority {
        let time_since_capture = chunk.timestamp.elapsed().as_millis() as u32;
        let total_expected_latency = time_since_capture + predicted_latency;

        match total_expected_latency {
            0..=50 => TaskPriority::Critical,
            51..=100 => TaskPriority::High,
            101..=200 => TaskPriority::Normal,
            _ => TaskPriority::Low,
        }
    }

    async fn adapt_quality_settings(&self, predicted_latency: u32) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut settings = self.quality_settings.write().unwrap();

        let latency_ratio = predicted_latency as f32 / self.config.target_latency_ms as f32;

        if latency_ratio > 1.5 {
            // Aggressive optimization for high latency
            settings.optimization_level = OptimizationLevel::Speed;
            settings.model_precision = ModelPrecision::FP8;
            settings.batch_size = settings.batch_size.max(1).min(settings.batch_size / 2);
            settings.chunk_size_ms = settings.chunk_size_ms.max(50);
            settings.enable_speculative_decoding = true;
            settings.gpu_streams = settings.gpu_streams.min(8);

        } else if latency_ratio > 1.2 {
            // Moderate optimization
            settings.optimization_level = OptimizationLevel::Balanced;
            settings.model_precision = ModelPrecision::FP16;
            settings.enable_speculative_decoding = true;

        } else if latency_ratio < 0.8 {
            // Can afford higher quality
            settings.optimization_level = OptimizationLevel::Accuracy;
            settings.model_precision = ModelPrecision::FP32;
            settings.batch_size = settings.batch_size.min(16);
            settings.enable_speculative_decoding = false;
        }

        self.quality_adjustments.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    fn should_use_speculative_execution(&self, chunk: &AudioChunk) -> bool {
        if !self.config.enable_speculative_execution {
            return false;
        }

        // Use speculative execution for high-priority, low-latency chunks
        let time_since_capture = chunk.timestamp.elapsed().as_millis() as u32;
        time_since_capture < 50 || chunk.sequence % 10 == 0 // Every 10th chunk for consistency
    }

    fn should_batch_processing(&self, chunk: &AudioChunk) -> bool {
        if !self.config.enable_dynamic_batching {
            return false;
        }

        let batch_optimizer = self.batch_optimizer.read().unwrap();
        let current_batch_size = batch_optimizer.current_batch.len();

        // Batch if we have room and similar chunks
        current_batch_size < batch_optimizer.max_batch_size &&
        chunk.language_hint.is_some() &&
        self.has_similar_pending_chunks(chunk)
    }

    fn has_similar_pending_chunks(&self, chunk: &AudioChunk) -> bool {
        let batch_optimizer = self.batch_optimizer.read().unwrap();

        batch_optimizer.current_batch.iter().any(|existing| {
            existing.language_hint == chunk.language_hint &&
            existing.cultural_context == chunk.cultural_context
        })
    }

    fn generate_processing_hints(&self, chunk: &AudioChunk) -> ProcessingHints {
        ProcessingHints {
            prefer_gpu: chunk.data.len() > 3200, // > 200ms of audio
            use_cache: chunk.language_hint.is_some(),
            parallel_stages: true,
            memory_pool: true,
            optimize_for_throughput: chunk.sequence % 100 < 10, // 10% of chunks
        }
    }

    fn schedule_task(&self, task: ScheduledTask) {
        self.processing_queue.push(task);
    }

    pub fn get_next_scheduled_task(&self) -> Option<ScheduledTask> {
        self.processing_queue.pop()
    }

    pub async fn record_processing_result(&self, result: &TranslationResult, stage_latencies: StageLatencies) {
        // Update latency history
        {
            let mut history = self.latency_history.write().unwrap();
            history.push_back((Instant::now(), result.latency_ms));
            if history.len() > 1000 {
                history.pop_front();
            }
        }

        // Update stage metrics
        {
            let mut metrics = self.stage_metrics.write().unwrap();
            metrics.update_with_latencies(&stage_latencies);
        }

        // Update prediction model
        {
            let mut model = self.prediction_model.write().unwrap();
            model.update_with_result(result, &stage_latencies);
        }

        // Check for latency violations
        if result.latency_ms > self.config.max_latency_ms {
            self.latency_violations.fetch_add(1, Ordering::Relaxed);
            counter!("latency_violations", 1);
        }

        // Update quality tracking
        {
            let mut quality_history = self.quality_history.write().unwrap();
            quality_history.push_back(result.confidence);
            if quality_history.len() > 100 {
                quality_history.pop_front();
            }
        }

        // Record metrics
        histogram!("total_processing_latency", Duration::from_millis(result.latency_ms as u64));
        histogram!("asr_stage_latency", Duration::from_millis(stage_latencies.asr_ms as u64));
        histogram!("translation_stage_latency", Duration::from_millis(stage_latencies.translation_ms as u64));
        histogram!("cultural_stage_latency", Duration::from_millis(stage_latencies.cultural_ms as u64));
        histogram!("synthesis_stage_latency", Duration::from_millis(stage_latencies.synthesis_ms as u64));
        gauge!("processing_confidence", result.confidence as f64);
    }

    pub fn get_latency_profile(&self) -> LatencyProfile {
        let history = self.latency_history.read().unwrap();
        let recent_latencies: Vec<u32> = history.iter()
            .filter(|(timestamp, _)| timestamp.elapsed() < Duration::from_secs(60))
            .map(|(_, latency)| *latency)
            .collect();

        if recent_latencies.is_empty() {
            return LatencyProfile::default();
        }

        let mut sorted_latencies = recent_latencies.clone();
        sorted_latencies.sort_unstable();

        let len = sorted_latencies.len();
        let p50 = sorted_latencies[len * 50 / 100] as f32;
        let p95 = sorted_latencies[len * 95 / 100] as f32;
        let p99 = sorted_latencies[len * 99 / 100] as f32;
        let avg = recent_latencies.iter().sum::<u32>() as f32 / len as f32;

        // Calculate jitter (standard deviation)
        let variance: f32 = recent_latencies.iter()
            .map(|&x| {
                let diff = x as f32 - avg;
                diff * diff
            })
            .sum::<f32>() / len as f32;
        let jitter = variance.sqrt();

        LatencyProfile {
            p50_ms: p50,
            p95_ms: p95,
            p99_ms: p99,
            avg_ms: avg,
            jitter_ms: jitter,
            throughput_chunks_per_sec: len as f32, // Approximate
        }
    }

    pub fn get_optimization_stats(&self) -> OptimizationStats {
        let prediction_accuracy = self.prediction_accuracy.load(Ordering::Relaxed);

        OptimizationStats {
            total_optimizations: self.total_optimizations.load(Ordering::Relaxed),
            latency_violations: self.latency_violations.load(Ordering::Relaxed),
            quality_adjustments: self.quality_adjustments.load(Ordering::Relaxed),
            prediction_accuracy,
            current_quality_settings: self.quality_settings.read().unwrap().clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct OptimizedProcessingPlan {
    pub chunk: AudioChunk,
    pub priority: TaskPriority,
    pub estimated_latency_ms: u32,
    pub quality_settings: AdaptiveQualitySettings,
    pub use_speculative_execution: bool,
    pub batch_with_others: bool,
    pub processing_hints: ProcessingHints,
}

#[derive(Debug, Clone)]
pub struct ProcessingHints {
    pub prefer_gpu: bool,
    pub use_cache: bool,
    pub parallel_stages: bool,
    pub memory_pool: bool,
    pub optimize_for_throughput: bool,
}

#[derive(Debug, Clone)]
pub struct StageLatencies {
    pub asr_ms: u32,
    pub translation_ms: u32,
    pub cultural_ms: u32,
    pub synthesis_ms: u32,
}

#[derive(Debug, Clone)]
pub struct OptimizationStats {
    pub total_optimizations: u64,
    pub latency_violations: u64,
    pub quality_adjustments: u64,
    pub prediction_accuracy: f64,
    pub current_quality_settings: AdaptiveQualitySettings,
}

impl Default for LatencyProfile {
    fn default() -> Self {
        Self {
            p50_ms: 0.0,
            p95_ms: 0.0,
            p99_ms: 0.0,
            avg_ms: 0.0,
            jitter_ms: 0.0,
            throughput_chunks_per_sec: 0.0,
        }
    }
}

impl Default for ProcessingStageMetrics {
    fn default() -> Self {
        Self {
            asr_latency: LatencyProfile::default(),
            translation_latency: LatencyProfile::default(),
            cultural_adaptation_latency: LatencyProfile::default(),
            synthesis_latency: LatencyProfile::default(),
            total_pipeline_latency: LatencyProfile::default(),
        }
    }
}

impl ProcessingStageMetrics {
    fn update_with_latencies(&mut self, latencies: &StageLatencies) {
        // This would update the stage metrics with new latency data
        // Implementation would involve updating moving averages and percentiles
    }
}

impl LatencyPredictionModel {
    fn new() -> Self {
        Self {
            asr_ma: MovingAverage::new(50),
            translation_ma: MovingAverage::new(50),
            cultural_ma: MovingAverage::new(50),
            synthesis_ma: MovingAverage::new(50),
            language_latency_map: HashMap::new(),
            chunk_size_latency_map: HashMap::new(),
            load_factor_predictor: LoadFactorPredictor::new(),
            prediction_errors: VecDeque::with_capacity(100),
            confidence_score: 0.5,
        }
    }

    fn update_with_result(&mut self, result: &TranslationResult, stage_latencies: &StageLatencies) {
        self.asr_ma.add(stage_latencies.asr_ms as f32);
        self.translation_ma.add(stage_latencies.translation_ms as f32);
        self.cultural_ma.add(stage_latencies.cultural_ms as f32);
        self.synthesis_ma.add(stage_latencies.synthesis_ms as f32);

        // Update contextual predictors
        if let Some(lang) = result.original_text.chars().next().map(|_| "detected_language".to_string()) {
            self.language_latency_map.insert(lang, result.latency_ms as f32);
        }

        // Update prediction accuracy
        let predicted_total = self.asr_ma.average() + self.translation_ma.average() +
                            self.cultural_ma.average() + self.synthesis_ma.average();
        let actual_total = result.latency_ms as f32;
        let error = (predicted_total - actual_total).abs() / actual_total;

        self.prediction_errors.push_back(error);
        if self.prediction_errors.len() > 100 {
            self.prediction_errors.pop_front();
        }

        self.confidence_score = 1.0 - (self.prediction_errors.iter().sum::<f32>() / self.prediction_errors.len() as f32);
    }
}

impl MovingAverage {
    fn new(window_size: usize) -> Self {
        Self {
            window_size,
            values: VecDeque::with_capacity(window_size),
            sum: 0.0,
        }
    }

    fn add(&mut self, value: f32) {
        if self.values.len() >= self.window_size {
            if let Some(old_value) = self.values.pop_front() {
                self.sum -= old_value;
            }
        }
        self.values.push_back(value);
        self.sum += value;
    }

    fn average(&self) -> f32 {
        if self.values.is_empty() {
            0.0
        } else {
            self.sum / self.values.len() as f32
        }
    }
}

impl LoadFactorPredictor {
    fn new() -> Self {
        Self {
            cpu_usage_history: VecDeque::with_capacity(60),
            gpu_usage_history: VecDeque::with_capacity(60),
            memory_usage_history: VecDeque::with_capacity(60),
            concurrent_streams: 0,
            predicted_load: 1.0,
        }
    }
}

impl BatchOptimizer {
    fn new() -> Self {
        Self {
            target_batch_size: 4,
            min_batch_size: 1,
            max_batch_size: 16,
            batch_latency_map: HashMap::new(),
            current_batch: Vec::new(),
            batch_deadline: None,
            adaptive_batching_enabled: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latency_optimizer_creation() {
        let config = LatencyOptimizerConfig::default();
        let optimizer = LatencyOptimizer::new(config);

        assert_eq!(optimizer.total_optimizations.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_moving_average() {
        let mut ma = MovingAverage::new(3);

        ma.add(10.0);
        assert_eq!(ma.average(), 10.0);

        ma.add(20.0);
        assert_eq!(ma.average(), 15.0);

        ma.add(30.0);
        assert_eq!(ma.average(), 20.0);

        ma.add(40.0); // Should evict 10.0
        assert_eq!(ma.average(), 30.0);
    }

    #[tokio::test]
    async fn test_optimization_plan_generation() {
        let config = LatencyOptimizerConfig::default();
        let optimizer = LatencyOptimizer::new(config);

        let chunk = AudioChunk {
            data: vec![0.0; 1600],
            sample_rate: 16000,
            timestamp: Instant::now(),
            stream_id: 1,
            sequence: 1,
            language_hint: Some("en".to_string()),
            cultural_context: Some("business".to_string()),
        };

        let plan = optimizer.optimize_chunk_processing(chunk).await;
        assert!(plan.is_ok());
    }
}