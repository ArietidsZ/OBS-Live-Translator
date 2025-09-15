use std::sync::{Arc, RwLock};
use std::sync::atomic::{AtomicU64, AtomicF64, Ordering};
use std::collections::{HashMap, VecDeque};
use std::time::{Instant, Duration};
use tokio::sync::RwLock as AsyncRwLock;
use dashmap::DashMap;
use metrics::{histogram, gauge, counter};
use serde::{Serialize, Deserialize};

use crate::streaming::{AudioChunk, OptimizationLevel};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveSelectionConfig {
    pub performance_weight: f32,      // 0.0-1.0
    pub accuracy_weight: f32,         // 0.0-1.0
    pub efficiency_weight: f32,       // 0.0-1.0
    pub latency_weight: f32,          // 0.0-1.0
    pub context_weight: f32,          // 0.0-1.0
    pub adaptation_rate: f32,         // Learning rate
    pub min_samples_for_switch: usize,
    pub confidence_threshold: f32,
    pub enable_ensemble_voting: bool,
    pub enable_a_b_testing: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelClass {
    ASR,
    Translation,
    TTS,
    CulturalAdaptation,
    VoiceCloning,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelVariant {
    // ASR Models
    CanaryFlash,
    WhisperLarge,
    WhisperTurbo,
    SpeechT5,
    Wav2Vec2,

    // Translation Models
    SeamlessM4T,
    NLLB200,
    mBART,
    MarianMT,
    M2M100,

    // TTS Models
    VITS,
    TorToise,
    Bark,
    YourTTS,
    FastSpeech2,

    // Cultural Adaptation
    CulturalLLM,
    ContextualLLM,
    BiasAwareLLM,

    // Voice Cloning
    RealTimeVC,
    FreeVC,
    YourTTSVC,
}

#[derive(Debug, Clone)]
pub struct ModelCapabilities {
    pub supported_languages: Vec<String>,
    pub max_throughput_rtf: f32,
    pub min_latency_ms: u32,
    pub accuracy_score: f32,
    pub memory_requirement_mb: u32,
    pub gpu_memory_mb: u32,
    pub cpu_cores_required: u8,
    pub quality_tiers: Vec<QualityTier>,
    pub cultural_awareness: f32,
    pub voice_preservation: f32,
    pub real_time_capable: bool,
    pub streaming_compatible: bool,
}

#[derive(Debug, Clone)]
pub struct QualityTier {
    pub name: String,
    pub accuracy: f32,
    pub latency_ms: u32,
    pub resource_multiplier: f32,
}

#[derive(Debug, Clone)]
pub struct ModelPerformanceMetrics {
    pub avg_latency_ms: f32,
    pub p95_latency_ms: f32,
    pub accuracy_score: f32,
    pub throughput_chunks_per_sec: f32,
    pub error_rate: f32,
    pub gpu_utilization: f32,
    pub memory_usage_mb: f32,
    pub energy_efficiency: f32,
    pub cultural_adaptation_score: f32,
    pub voice_quality_score: f32,
    pub total_processed: u64,
    pub last_updated: Instant,
}

#[derive(Debug, Clone)]
pub struct ContextualRequirements {
    pub languages: Vec<String>,
    pub cultural_context: Option<String>,
    pub audio_quality: AudioQuality,
    pub target_latency_ms: u32,
    pub accuracy_priority: f32,      // 0.0-1.0
    pub efficiency_priority: f32,    // 0.0-1.0
    pub cultural_sensitivity: f32,   // 0.0-1.0
    pub voice_preservation: f32,     // 0.0-1.0
    pub real_time_required: bool,
    pub batch_processing: bool,
    pub resource_constraints: ResourceConstraints,
}

#[derive(Debug, Clone)]
pub enum AudioQuality {
    Low,      // < 16kHz, compressed
    Standard, // 16kHz, uncompressed
    High,     // 48kHz+, studio quality
    Broadcast, // Professional broadcast quality
}

#[derive(Debug, Clone)]
pub struct ResourceConstraints {
    pub max_gpu_memory_mb: u32,
    pub max_cpu_cores: u8,
    pub max_system_memory_mb: u32,
    pub power_limited: bool,
    pub network_bandwidth_mbps: Option<u32>,
}

pub struct AdaptiveModelSelector {
    config: AdaptiveSelectionConfig,

    // Model registry and capabilities
    model_registry: DashMap<(ModelClass, ModelVariant), ModelCapabilities>,
    performance_history: DashMap<(ModelClass, ModelVariant), ModelPerformanceMetrics>,

    // Current selections
    active_models: Arc<AsyncRwLock<HashMap<ModelClass, ModelVariant>>>,

    // Decision engine
    decision_weights: Arc<RwLock<DecisionWeights>>,
    context_analyzer: Arc<ContextAnalyzer>,

    // A/B testing framework
    ab_tests: Arc<RwLock<HashMap<ModelClass, ABTest>>>,

    // Ensemble voting
    ensemble_tracker: Arc<RwLock<EnsembleTracker>>,

    // Performance tracking
    selection_history: Arc<RwLock<VecDeque<SelectionEvent>>>,
    adaptation_metrics: Arc<AdaptationMetrics>,

    // Real-time optimizer
    optimizer: Arc<RwLock<RealTimeOptimizer>>,
}

#[derive(Debug, Clone)]
struct DecisionWeights {
    performance: f32,
    accuracy: f32,
    efficiency: f32,
    latency: f32,
    context: f32,
    adaptation_momentum: f32,
}

#[derive(Debug)]
struct ContextAnalyzer {
    language_patterns: DashMap<String, LanguagePattern>,
    cultural_patterns: DashMap<String, CulturalPattern>,
    temporal_patterns: VecDeque<(Instant, ContextualRequirements)>,
    prediction_model: ContextPredictionModel,
}

#[derive(Debug, Clone)]
struct LanguagePattern {
    frequency: f32,
    complexity_score: f32,
    optimal_models: Vec<ModelVariant>,
    performance_multipliers: HashMap<ModelVariant, f32>,
}

#[derive(Debug, Clone)]
struct CulturalPattern {
    sensitivity_required: f32,
    adaptation_complexity: f32,
    preferred_models: Vec<ModelVariant>,
    success_rates: HashMap<ModelVariant, f32>,
}

#[derive(Debug)]
struct ContextPredictionModel {
    language_predictor: LanguagePredictor,
    workload_predictor: WorkloadPredictor,
    resource_predictor: ResourcePredictor,
    confidence_scores: HashMap<String, f32>,
}

#[derive(Debug)]
struct ABTest {
    model_a: ModelVariant,
    model_b: ModelVariant,
    traffic_split: f32, // 0.0-1.0, percentage to model_a
    start_time: Instant,
    metrics_a: ModelPerformanceMetrics,
    metrics_b: ModelPerformanceMetrics,
    statistical_significance: f32,
    winner: Option<ModelVariant>,
}

#[derive(Debug)]
struct EnsembleTracker {
    voting_strategies: HashMap<ModelClass, VotingStrategy>,
    ensemble_performance: HashMap<ModelClass, f32>,
    confidence_weights: HashMap<ModelVariant, f32>,
}

#[derive(Debug, Clone)]
enum VotingStrategy {
    Majority,
    WeightedByConfidence,
    WeightedByPerformance,
    AdaptiveWeighting,
}

#[derive(Debug, Clone)]
struct SelectionEvent {
    timestamp: Instant,
    model_class: ModelClass,
    old_variant: Option<ModelVariant>,
    new_variant: ModelVariant,
    reason: SelectionReason,
    context: ContextualRequirements,
    confidence: f32,
}

#[derive(Debug, Clone)]
enum SelectionReason {
    PerformanceImprovement,
    LatencyOptimization,
    AccuracyBoost,
    ResourceConstraints,
    ContextualFit,
    ABTestResult,
    EnsembleVoting,
    UserFeedback,
}

#[derive(Debug)]
struct AdaptationMetrics {
    total_selections: AtomicU64,
    successful_adaptations: AtomicU64,
    failed_adaptations: AtomicU64,
    avg_improvement: AtomicF64,
    adaptation_latency: AtomicU64,
    context_prediction_accuracy: AtomicF64,
}

#[derive(Debug)]
struct RealTimeOptimizer {
    current_workload: WorkloadProfile,
    resource_availability: ResourceProfile,
    optimization_target: OptimizationTarget,
    dynamic_thresholds: DynamicThresholds,
}

#[derive(Debug, Clone)]
struct WorkloadProfile {
    concurrent_streams: usize,
    avg_chunk_size: u32,
    processing_rate: f32,
    complexity_score: f32,
    language_diversity: f32,
}

#[derive(Debug, Clone)]
struct ResourceProfile {
    cpu_utilization: f32,
    gpu_utilization: f32,
    memory_usage: f32,
    network_bandwidth: f32,
    thermal_state: ThermalState,
}

#[derive(Debug, Clone)]
enum ThermalState {
    Cool,
    Warm,
    Hot,
    Critical,
}

#[derive(Debug, Clone)]
enum OptimizationTarget {
    MinimizeLatency,
    MaximizeAccuracy,
    MaximizeThroughput,
    MinimizeResourceUsage,
    BalancedOptimization,
}

#[derive(Debug, Clone)]
struct DynamicThresholds {
    latency_threshold_ms: f32,
    accuracy_threshold: f32,
    resource_threshold: f32,
    adaptation_sensitivity: f32,
}

impl Default for AdaptiveSelectionConfig {
    fn default() -> Self {
        Self {
            performance_weight: 0.3,
            accuracy_weight: 0.3,
            efficiency_weight: 0.2,
            latency_weight: 0.15,
            context_weight: 0.05,
            adaptation_rate: 0.1,
            min_samples_for_switch: 10,
            confidence_threshold: 0.8,
            enable_ensemble_voting: true,
            enable_a_b_testing: true,
        }
    }
}

impl AdaptiveModelSelector {
    pub async fn new(config: AdaptiveSelectionConfig) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let mut selector = Self {
            config: config.clone(),
            model_registry: DashMap::new(),
            performance_history: DashMap::new(),
            active_models: Arc::new(AsyncRwLock::new(HashMap::new())),
            decision_weights: Arc::new(RwLock::new(DecisionWeights::from_config(&config))),
            context_analyzer: Arc::new(ContextAnalyzer::new()),
            ab_tests: Arc::new(RwLock::new(HashMap::new())),
            ensemble_tracker: Arc::new(RwLock::new(EnsembleTracker::new())),
            selection_history: Arc::new(RwLock::new(VecDeque::with_capacity(1000))),
            adaptation_metrics: Arc::new(AdaptationMetrics::default()),
            optimizer: Arc::new(RwLock::new(RealTimeOptimizer::new())),
        };

        // Initialize model registry
        selector.initialize_model_registry().await?;

        // Set default model selections
        selector.initialize_default_selections().await?;

        // Start background optimization
        selector.start_background_optimization().await;

        Ok(selector)
    }

    async fn initialize_model_registry(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Register ASR models
        self.register_model(
            ModelClass::ASR,
            ModelVariant::CanaryFlash,
            ModelCapabilities {
                supported_languages: vec!["en", "es", "fr", "de", "it", "pt", "pl", "uk", "bg", "hr"].iter().map(|s| s.to_string()).collect(),
                max_throughput_rtf: 1000.0,
                min_latency_ms: 5,
                accuracy_score: 0.98,
                memory_requirement_mb: 2048,
                gpu_memory_mb: 4096,
                cpu_cores_required: 4,
                quality_tiers: vec![
                    QualityTier { name: "Ultra".to_string(), accuracy: 0.98, latency_ms: 5, resource_multiplier: 1.0 },
                    QualityTier { name: "High".to_string(), accuracy: 0.96, latency_ms: 3, resource_multiplier: 0.7 },
                    QualityTier { name: "Fast".to_string(), accuracy: 0.93, latency_ms: 1, resource_multiplier: 0.4 },
                ],
                cultural_awareness: 0.7,
                voice_preservation: 0.9,
                real_time_capable: true,
                streaming_compatible: true,
            }
        );

        self.register_model(
            ModelClass::ASR,
            ModelVariant::WhisperLarge,
            ModelCapabilities {
                supported_languages: vec!["en", "es", "fr", "de", "it", "pt", "pl", "uk", "bg", "hr", "cs", "da", "et", "fi", "hu", "lv", "lt", "nl", "ro", "sk", "sl", "sv"].iter().map(|s| s.to_string()).collect(),
                max_throughput_rtf: 2.0,
                min_latency_ms: 200,
                accuracy_score: 0.96,
                memory_requirement_mb: 4096,
                gpu_memory_mb: 8192,
                cpu_cores_required: 8,
                quality_tiers: vec![
                    QualityTier { name: "Large".to_string(), accuracy: 0.96, latency_ms: 200, resource_multiplier: 1.0 },
                    QualityTier { name: "Medium".to_string(), accuracy: 0.94, latency_ms: 150, resource_multiplier: 0.6 },
                    QualityTier { name: "Small".to_string(), accuracy: 0.91, latency_ms: 100, resource_multiplier: 0.3 },
                ],
                cultural_awareness: 0.8,
                voice_preservation: 0.6,
                real_time_capable: false,
                streaming_compatible: true,
            }
        );

        // Register Translation models
        self.register_model(
            ModelClass::Translation,
            ModelVariant::SeamlessM4T,
            ModelCapabilities {
                supported_languages: vec!["en", "es", "fr", "de", "it", "pt", "pl", "uk", "bg", "hr", "cs", "da", "et", "fi", "hu", "lv", "lt", "nl", "ro", "sk", "sl", "sv", "ar", "zh", "ja", "ko", "hi", "th"].iter().map(|s| s.to_string()).collect(),
                max_throughput_rtf: 5.0,
                min_latency_ms: 50,
                accuracy_score: 0.94,
                memory_requirement_mb: 6144,
                gpu_memory_mb: 12288,
                cpu_cores_required: 8,
                quality_tiers: vec![
                    QualityTier { name: "Large".to_string(), accuracy: 0.94, latency_ms: 50, resource_multiplier: 1.0 },
                    QualityTier { name: "Medium".to_string(), accuracy: 0.91, latency_ms: 35, resource_multiplier: 0.6 },
                ],
                cultural_awareness: 0.9,
                voice_preservation: 0.95,
                real_time_capable: true,
                streaming_compatible: true,
            }
        );

        // Register more models...
        self.register_cultural_models().await?;
        self.register_tts_models().await?;

        Ok(())
    }

    async fn register_cultural_models(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.register_model(
            ModelClass::CulturalAdaptation,
            ModelVariant::CulturalLLM,
            ModelCapabilities {
                supported_languages: vec!["en", "es", "fr", "de", "it", "pt", "zh", "ja", "ar", "hi"].iter().map(|s| s.to_string()).collect(),
                max_throughput_rtf: 10.0,
                min_latency_ms: 100,
                accuracy_score: 0.92,
                memory_requirement_mb: 8192,
                gpu_memory_mb: 16384,
                cpu_cores_required: 12,
                quality_tiers: vec![
                    QualityTier { name: "Expert".to_string(), accuracy: 0.92, latency_ms: 100, resource_multiplier: 1.0 },
                    QualityTier { name: "Standard".to_string(), accuracy: 0.88, latency_ms: 60, resource_multiplier: 0.5 },
                ],
                cultural_awareness: 0.98,
                voice_preservation: 0.3,
                real_time_capable: true,
                streaming_compatible: true,
            }
        );

        Ok(())
    }

    async fn register_tts_models(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.register_model(
            ModelClass::TTS,
            ModelVariant::VITS,
            ModelCapabilities {
                supported_languages: vec!["en", "es", "fr", "de", "it", "pt", "ja", "zh"].iter().map(|s| s.to_string()).collect(),
                max_throughput_rtf: 20.0,
                min_latency_ms: 30,
                accuracy_score: 0.89,
                memory_requirement_mb: 2048,
                gpu_memory_mb: 4096,
                cpu_cores_required: 4,
                quality_tiers: vec![
                    QualityTier { name: "High".to_string(), accuracy: 0.89, latency_ms: 30, resource_multiplier: 1.0 },
                    QualityTier { name: "Fast".to_string(), accuracy: 0.85, latency_ms: 15, resource_multiplier: 0.6 },
                ],
                cultural_awareness: 0.6,
                voice_preservation: 0.85,
                real_time_capable: true,
                streaming_compatible: true,
            }
        );

        Ok(())
    }

    fn register_model(&self, class: ModelClass, variant: ModelVariant, capabilities: ModelCapabilities) {
        self.model_registry.insert((class, variant), capabilities);

        // Initialize performance metrics
        let initial_metrics = ModelPerformanceMetrics {
            avg_latency_ms: capabilities.min_latency_ms as f32,
            p95_latency_ms: capabilities.min_latency_ms as f32 * 1.5,
            accuracy_score: capabilities.accuracy_score,
            throughput_chunks_per_sec: capabilities.max_throughput_rtf,
            error_rate: 1.0 - capabilities.accuracy_score,
            gpu_utilization: 0.5,
            memory_usage_mb: capabilities.memory_requirement_mb as f32,
            energy_efficiency: 0.8,
            cultural_adaptation_score: capabilities.cultural_awareness,
            voice_quality_score: capabilities.voice_preservation,
            total_processed: 0,
            last_updated: Instant::now(),
        };

        self.performance_history.insert((class, variant), initial_metrics);
    }

    async fn initialize_default_selections(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut active = self.active_models.write().await;

        active.insert(ModelClass::ASR, ModelVariant::CanaryFlash);
        active.insert(ModelClass::Translation, ModelVariant::SeamlessM4T);
        active.insert(ModelClass::CulturalAdaptation, ModelVariant::CulturalLLM);
        active.insert(ModelClass::TTS, ModelVariant::VITS);

        Ok(())
    }

    pub async fn select_optimal_model(&self,
        class: ModelClass,
        context: &ContextualRequirements
    ) -> Result<ModelVariant, Box<dyn std::error::Error + Send + Sync>> {
        let start_time = Instant::now();

        // Update context analysis
        self.context_analyzer.analyze_context(context).await;

        // Get candidate models
        let candidates = self.get_candidate_models(class, context);

        if candidates.is_empty() {
            return Err("No suitable models found for the given context".into());
        }

        // Score each candidate
        let mut scored_candidates = Vec::new();
        for candidate in candidates {
            let score = self.calculate_model_score(class, candidate, context).await?;
            scored_candidates.push((candidate, score));
        }

        // Sort by score (highest first)
        scored_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let selected_model = scored_candidates[0].0;
        let confidence = scored_candidates[0].1;

        // Check if we should switch models
        let current_model = {
            let active = self.active_models.read().await;
            active.get(&class).copied()
        };

        let should_switch = self.should_switch_model(
            class,
            current_model,
            selected_model,
            confidence
        ).await?;

        if should_switch {
            // Update active model
            {
                let mut active = self.active_models.write().await;
                let old_model = active.insert(class, selected_model);

                // Record selection event
                self.record_selection_event(SelectionEvent {
                    timestamp: Instant::now(),
                    model_class: class,
                    old_variant: old_model,
                    new_variant: selected_model,
                    reason: SelectionReason::PerformanceImprovement,
                    context: context.clone(),
                    confidence,
                });
            }

            self.adaptation_metrics.successful_adaptations.fetch_add(1, Ordering::Relaxed);
        }

        // Record metrics
        let selection_time = start_time.elapsed();
        histogram!("model_selection_time", selection_time);
        gauge!("selection_confidence", confidence as f64);
        counter!("model_selections", 1, "class" => format!("{:?}", class));

        Ok(selected_model)
    }

    fn get_candidate_models(&self, class: ModelClass, context: &ContextualRequirements) -> Vec<ModelVariant> {
        self.model_registry
            .iter()
            .filter_map(|entry| {
                let ((model_class, variant), capabilities) = entry.pair();

                if *model_class != class {
                    return None;
                }

                // Check basic compatibility
                if context.real_time_required && !capabilities.real_time_capable {
                    return None;
                }

                if context.resource_constraints.max_gpu_memory_mb < capabilities.gpu_memory_mb {
                    return None;
                }

                if context.resource_constraints.max_cpu_cores < capabilities.cpu_cores_required {
                    return None;
                }

                // Check language support
                let language_supported = context.languages.iter().any(|lang| {
                    capabilities.supported_languages.contains(lang)
                });

                if !language_supported {
                    return None;
                }

                Some(*variant)
            })
            .collect()
    }

    async fn calculate_model_score(&self,
        class: ModelClass,
        variant: ModelVariant,
        context: &ContextualRequirements
    ) -> Result<f32, Box<dyn std::error::Error + Send + Sync>> {
        let capabilities = self.model_registry
            .get(&(class, variant))
            .ok_or("Model not found in registry")?;

        let performance = self.performance_history
            .get(&(class, variant))
            .map(|entry| entry.value().clone())
            .unwrap_or_else(|| ModelPerformanceMetrics::default());

        let weights = self.decision_weights.read().unwrap().clone();

        // Performance score (0.0-1.0)
        let performance_score = self.calculate_performance_score(&performance, context);

        // Accuracy score (0.0-1.0)
        let accuracy_score = performance.accuracy_score.min(1.0);

        // Efficiency score (0.0-1.0)
        let efficiency_score = self.calculate_efficiency_score(&capabilities, &performance, context);

        // Latency score (0.0-1.0)
        let latency_score = self.calculate_latency_score(&performance, context);

        // Context score (0.0-1.0)
        let context_score = self.calculate_context_score(&capabilities, context);

        // Combine scores
        let total_score =
            performance_score * weights.performance +
            accuracy_score * weights.accuracy +
            efficiency_score * weights.efficiency +
            latency_score * weights.latency +
            context_score * weights.context;

        Ok(total_score)
    }

    fn calculate_performance_score(&self, metrics: &ModelPerformanceMetrics, _context: &ContextualRequirements) -> f32 {
        // Combine various performance metrics
        let throughput_score = (metrics.throughput_chunks_per_sec / 100.0).min(1.0);
        let utilization_score = metrics.gpu_utilization.min(1.0);
        let reliability_score = 1.0 - metrics.error_rate;

        (throughput_score + utilization_score + reliability_score) / 3.0
    }

    fn calculate_efficiency_score(&self, capabilities: &ModelCapabilities, metrics: &ModelPerformanceMetrics, context: &ContextualRequirements) -> f32 {
        let memory_efficiency = 1.0 - (metrics.memory_usage_mb / context.resource_constraints.max_system_memory_mb as f32).min(1.0);
        let gpu_efficiency = 1.0 - (capabilities.gpu_memory_mb as f32 / context.resource_constraints.max_gpu_memory_mb as f32).min(1.0);
        let energy_efficiency = metrics.energy_efficiency;

        (memory_efficiency + gpu_efficiency + energy_efficiency) / 3.0
    }

    fn calculate_latency_score(&self, metrics: &ModelPerformanceMetrics, context: &ContextualRequirements) -> f32 {
        let target_latency = context.target_latency_ms as f32;
        let actual_latency = metrics.avg_latency_ms;

        if actual_latency <= target_latency {
            1.0
        } else {
            (target_latency / actual_latency).min(1.0)
        }
    }

    fn calculate_context_score(&self, capabilities: &ModelCapabilities, context: &ContextualRequirements) -> f32 {
        let mut score = 0.0;
        let mut factors = 0;

        // Cultural awareness
        if context.cultural_sensitivity > 0.0 {
            score += capabilities.cultural_awareness * context.cultural_sensitivity;
            factors += 1;
        }

        // Voice preservation
        if context.voice_preservation > 0.0 {
            score += capabilities.voice_preservation * context.voice_preservation;
            factors += 1;
        }

        // Language support quality
        let language_score = context.languages.iter()
            .map(|lang| {
                if capabilities.supported_languages.contains(lang) {
                    1.0
                } else {
                    0.0
                }
            })
            .sum::<f32>() / context.languages.len() as f32;

        score += language_score;
        factors += 1;

        if factors > 0 {
            score / factors as f32
        } else {
            0.5
        }
    }

    async fn should_switch_model(&self,
        class: ModelClass,
        current: Option<ModelVariant>,
        candidate: ModelVariant,
        confidence: f32
    ) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
        if current.is_none() {
            return Ok(true);
        }

        let current = current.unwrap();
        if current == candidate {
            return Ok(false);
        }

        // Check confidence threshold
        if confidence < self.config.confidence_threshold {
            return Ok(false);
        }

        // Check minimum samples
        let current_metrics = self.performance_history
            .get(&(class, current))
            .map(|entry| entry.value().clone());

        let candidate_metrics = self.performance_history
            .get(&(class, candidate))
            .map(|entry| entry.value().clone());

        if let (Some(current_perf), Some(candidate_perf)) = (current_metrics, candidate_metrics) {
            if current_perf.total_processed < self.config.min_samples_for_switch as u64 ||
               candidate_perf.total_processed < self.config.min_samples_for_switch as u64 {
                return Ok(false);
            }

            // Check for significant improvement
            let improvement = self.calculate_improvement(&current_perf, &candidate_perf);
            return Ok(improvement > 0.05); // 5% improvement threshold
        }

        Ok(true)
    }

    fn calculate_improvement(&self, current: &ModelPerformanceMetrics, candidate: &ModelPerformanceMetrics) -> f32 {
        let latency_improvement = (current.avg_latency_ms - candidate.avg_latency_ms) / current.avg_latency_ms;
        let accuracy_improvement = candidate.accuracy_score - current.accuracy_score;
        let throughput_improvement = (candidate.throughput_chunks_per_sec - current.throughput_chunks_per_sec) / current.throughput_chunks_per_sec;

        (latency_improvement + accuracy_improvement + throughput_improvement) / 3.0
    }

    fn record_selection_event(&self, event: SelectionEvent) {
        let mut history = self.selection_history.write().unwrap();
        history.push_back(event);

        if history.len() > 1000 {
            history.pop_front();
        }

        self.adaptation_metrics.total_selections.fetch_add(1, Ordering::Relaxed);
    }

    pub async fn update_model_performance(&self,
        class: ModelClass,
        variant: ModelVariant,
        metrics: ModelPerformanceMetrics
    ) {
        self.performance_history.insert((class, variant), metrics);

        // Update adaptation metrics
        gauge!("model_accuracy", metrics.accuracy_score as f64, "class" => format!("{:?}", class), "variant" => format!("{:?}", variant));
        gauge!("model_latency", metrics.avg_latency_ms as f64, "class" => format!("{:?}", class), "variant" => format!("{:?}", variant));
        gauge!("model_throughput", metrics.throughput_chunks_per_sec as f64, "class" => format!("{:?}", class), "variant" => format!("{:?}", variant));
    }

    async fn start_background_optimization(&self) {
        let optimizer = self.optimizer.clone();
        let adaptation_metrics = self.adaptation_metrics.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));

            loop {
                interval.tick().await;

                let mut opt = optimizer.write().unwrap();
                opt.update_optimization_targets();

                // Emit adaptation metrics
                gauge!("total_selections", adaptation_metrics.total_selections.load(Ordering::Relaxed) as f64);
                gauge!("successful_adaptations", adaptation_metrics.successful_adaptations.load(Ordering::Relaxed) as f64);
                gauge!("adaptation_success_rate",
                    adaptation_metrics.successful_adaptations.load(Ordering::Relaxed) as f64 /
                    adaptation_metrics.total_selections.load(Ordering::Relaxed).max(1) as f64
                );
            }
        });
    }

    pub async fn get_current_models(&self) -> HashMap<ModelClass, ModelVariant> {
        self.active_models.read().await.clone()
    }

    pub fn get_adaptation_metrics(&self) -> AdaptationMetrics {
        AdaptationMetrics {
            total_selections: AtomicU64::new(self.adaptation_metrics.total_selections.load(Ordering::Relaxed)),
            successful_adaptations: AtomicU64::new(self.adaptation_metrics.successful_adaptations.load(Ordering::Relaxed)),
            failed_adaptations: AtomicU64::new(self.adaptation_metrics.failed_adaptations.load(Ordering::Relaxed)),
            avg_improvement: AtomicF64::new(self.adaptation_metrics.avg_improvement.load(Ordering::Relaxed)),
            adaptation_latency: AtomicU64::new(self.adaptation_metrics.adaptation_latency.load(Ordering::Relaxed)),
            context_prediction_accuracy: AtomicF64::new(self.adaptation_metrics.context_prediction_accuracy.load(Ordering::Relaxed)),
        }
    }
}

// Helper trait implementations and structs

impl DecisionWeights {
    fn from_config(config: &AdaptiveSelectionConfig) -> Self {
        Self {
            performance: config.performance_weight,
            accuracy: config.accuracy_weight,
            efficiency: config.efficiency_weight,
            latency: config.latency_weight,
            context: config.context_weight,
            adaptation_momentum: 0.1,
        }
    }
}

impl ContextAnalyzer {
    fn new() -> Self {
        Self {
            language_patterns: DashMap::new(),
            cultural_patterns: DashMap::new(),
            temporal_patterns: VecDeque::with_capacity(1000),
            prediction_model: ContextPredictionModel::new(),
        }
    }

    async fn analyze_context(&self, context: &ContextualRequirements) {
        // Update language patterns
        for language in &context.languages {
            self.language_patterns.entry(language.clone()).or_insert_with(|| {
                LanguagePattern {
                    frequency: 0.0,
                    complexity_score: 0.5,
                    optimal_models: vec![],
                    performance_multipliers: HashMap::new(),
                }
            }).frequency += 1.0;
        }

        // Update cultural patterns
        if let Some(cultural_context) = &context.cultural_context {
            self.cultural_patterns.entry(cultural_context.clone()).or_insert_with(|| {
                CulturalPattern {
                    sensitivity_required: context.cultural_sensitivity,
                    adaptation_complexity: 0.5,
                    preferred_models: vec![],
                    success_rates: HashMap::new(),
                }
            });
        }

        // Store temporal pattern
        let mut patterns = match std::mem::replace(&mut *std::ptr::null_mut(), VecDeque::new()) {
            mut patterns => {
                patterns.push_back((Instant::now(), context.clone()));
                if patterns.len() > 1000 {
                    patterns.pop_front();
                }
                patterns
            }
        };
        // This is a simplified version - in practice, we'd need proper synchronization
    }
}

impl ContextPredictionModel {
    fn new() -> Self {
        Self {
            language_predictor: LanguagePredictor::new(),
            workload_predictor: WorkloadPredictor::new(),
            resource_predictor: ResourcePredictor::new(),
            confidence_scores: HashMap::new(),
        }
    }
}

impl EnsembleTracker {
    fn new() -> Self {
        Self {
            voting_strategies: HashMap::new(),
            ensemble_performance: HashMap::new(),
            confidence_weights: HashMap::new(),
        }
    }
}

impl Default for AdaptationMetrics {
    fn default() -> Self {
        Self {
            total_selections: AtomicU64::new(0),
            successful_adaptations: AtomicU64::new(0),
            failed_adaptations: AtomicU64::new(0),
            avg_improvement: AtomicF64::new(0.0),
            adaptation_latency: AtomicU64::new(0),
            context_prediction_accuracy: AtomicF64::new(0.0),
        }
    }
}

impl Default for ModelPerformanceMetrics {
    fn default() -> Self {
        Self {
            avg_latency_ms: 100.0,
            p95_latency_ms: 150.0,
            accuracy_score: 0.85,
            throughput_chunks_per_sec: 10.0,
            error_rate: 0.05,
            gpu_utilization: 0.5,
            memory_usage_mb: 1024.0,
            energy_efficiency: 0.8,
            cultural_adaptation_score: 0.7,
            voice_quality_score: 0.8,
            total_processed: 0,
            last_updated: Instant::now(),
        }
    }
}

impl RealTimeOptimizer {
    fn new() -> Self {
        Self {
            current_workload: WorkloadProfile::default(),
            resource_availability: ResourceProfile::default(),
            optimization_target: OptimizationTarget::BalancedOptimization,
            dynamic_thresholds: DynamicThresholds::default(),
        }
    }

    fn update_optimization_targets(&mut self) {
        // This would analyze current performance and adjust optimization targets
        // Implementation would involve complex heuristics and machine learning
    }
}

impl Default for WorkloadProfile {
    fn default() -> Self {
        Self {
            concurrent_streams: 1,
            avg_chunk_size: 1600,
            processing_rate: 10.0,
            complexity_score: 0.5,
            language_diversity: 0.3,
        }
    }
}

impl Default for ResourceProfile {
    fn default() -> Self {
        Self {
            cpu_utilization: 0.5,
            gpu_utilization: 0.5,
            memory_usage: 0.5,
            network_bandwidth: 0.5,
            thermal_state: ThermalState::Cool,
        }
    }
}

impl Default for DynamicThresholds {
    fn default() -> Self {
        Self {
            latency_threshold_ms: 150.0,
            accuracy_threshold: 0.85,
            resource_threshold: 0.8,
            adaptation_sensitivity: 0.1,
        }
    }
}

// Placeholder structs for compilation
#[derive(Debug)]
struct LanguagePredictor;
impl LanguagePredictor {
    fn new() -> Self { Self }
}

#[derive(Debug)]
struct WorkloadPredictor;
impl WorkloadPredictor {
    fn new() -> Self { Self }
}

#[derive(Debug)]
struct ResourcePredictor;
impl ResourcePredictor {
    fn new() -> Self { Self }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_model_selector_creation() {
        let config = AdaptiveSelectionConfig::default();
        let selector = AdaptiveModelSelector::new(config).await;
        assert!(selector.is_ok());
    }

    #[tokio::test]
    async fn test_model_selection() {
        let config = AdaptiveSelectionConfig::default();
        let selector = AdaptiveModelSelector::new(config).await.unwrap();

        let context = ContextualRequirements {
            languages: vec!["en".to_string()],
            cultural_context: Some("business".to_string()),
            audio_quality: AudioQuality::Standard,
            target_latency_ms: 100,
            accuracy_priority: 0.8,
            efficiency_priority: 0.6,
            cultural_sensitivity: 0.7,
            voice_preservation: 0.8,
            real_time_required: true,
            batch_processing: false,
            resource_constraints: ResourceConstraints {
                max_gpu_memory_mb: 8192,
                max_cpu_cores: 8,
                max_system_memory_mb: 16384,
                power_limited: false,
                network_bandwidth_mbps: Some(1000),
            },
        };

        let result = selector.select_optimal_model(ModelClass::ASR, &context).await;
        assert!(result.is_ok());
    }

    #[test]
    fn test_model_registry() {
        let selector_config = AdaptiveSelectionConfig::default();
        let registry = DashMap::new();

        // Test model registration logic
        let capabilities = ModelCapabilities {
            supported_languages: vec!["en".to_string()],
            max_throughput_rtf: 10.0,
            min_latency_ms: 50,
            accuracy_score: 0.9,
            memory_requirement_mb: 1024,
            gpu_memory_mb: 2048,
            cpu_cores_required: 4,
            quality_tiers: vec![],
            cultural_awareness: 0.8,
            voice_preservation: 0.7,
            real_time_capable: true,
            streaming_compatible: true,
        };

        registry.insert((ModelClass::ASR, ModelVariant::CanaryFlash), capabilities);
        assert_eq!(registry.len(), 1);
    }
}