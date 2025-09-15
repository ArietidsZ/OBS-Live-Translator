pub mod multi_stream;
pub mod advanced_cache;
pub mod resource_manager;
pub mod pipeline;
pub mod cache_warmer;
pub mod optimized_pipeline;
pub mod unified_pipeline;
pub mod latency_optimizer;

pub use multi_stream::{
    MultiStreamProcessor, StreamConfig, StreamPriority, AudioFrame, ProcessedSegment, StreamStats
};

pub use advanced_cache::{
    AdvancedCache, CacheKey, ModelType, CacheEntry, CacheMetadata, CacheStats, EvictionPolicy
};

pub use resource_manager::{
    ResourceManager, ResourceAllocation, ResourceLimits, SystemStats
};

pub use pipeline::{
    ConcurrentPipeline, PipelineConfig, PipelineMetricsReport
};

pub use cache_warmer::{
    CacheWarmer, WarmingPattern, PredictionModel, WarmingStats
};

pub use optimized_pipeline::{
    OptimizedStreamingPipeline, StreamResult, LatencyBreakdown,
    PipelineConfig as OptimizedPipelineConfig, QuantizationLevel
};

pub use unified_pipeline::{
    UnifiedStreamingPipeline, StreamingConfig, OptimizationLevel, AudioChunk,
    TranslationResult, VoiceCharacteristics, StreamMetrics
};

pub use latency_optimizer::{
    LatencyOptimizer, LatencyOptimizerConfig, LatencyProfile, ProcessingStageMetrics,
    AdaptiveQualitySettings, ModelPrecision, OptimizedProcessingPlan, ProcessingHints,
    StageLatencies, OptimizationStats
};