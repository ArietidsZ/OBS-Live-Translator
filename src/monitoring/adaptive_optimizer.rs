//! Adaptive Performance Optimizer
//!
//! This module provides intelligent performance optimization including:
//! - Automatic profile switching based on performance metrics
//! - Quality degradation under resource constraints
//! - Resource-aware processing adjustments
//! - Performance alert integration

use crate::profile::Profile;
use crate::monitoring::metrics::PerformanceMetrics;
use anyhow::Result;
use serde::{Serialize, Deserialize};
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Mutex};
use tokio::time::interval;
use tracing::{info, debug};

/// Adaptive optimizer for performance management
#[derive(Clone)]
pub struct AdaptiveOptimizer {
    /// Configuration
    config: AdaptiveConfig,
    /// Optimizer state
    state: Arc<RwLock<OptimizerState>>,
    /// Performance history for decision making
    performance_history: Arc<Mutex<VecDeque<PerformanceMetrics>>>,
    /// Current performance targets
    targets: Arc<RwLock<PerformanceTarget>>,
    /// Optimization actions history
    actions_history: Arc<Mutex<Vec<OptimizationAction>>>,
}

/// Adaptive optimizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveConfig {
    /// Enable adaptive optimization
    pub enabled: bool,
    /// Optimization evaluation interval (seconds)
    pub optimization_interval_s: u64,
    /// Sensitivity to performance changes (0.0-1.0)
    pub sensitivity: f32,
    /// Maximum optimization level (0-10)
    pub max_optimization_level: u8,
    /// Minimum samples before optimization decisions
    pub min_samples_for_decision: u32,
    /// Cooldown period between optimizations (seconds)
    pub optimization_cooldown_s: u64,
}

/// Performance targets for optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceTarget {
    /// Target end-to-end latency (ms)
    pub target_latency_ms: f32,
    /// Maximum acceptable latency (ms)
    pub max_latency_ms: f32,
    /// Target CPU utilization (%)
    pub target_cpu_percent: f32,
    /// Maximum acceptable CPU utilization (%)
    pub max_cpu_percent: f32,
    /// Target memory utilization (%)
    pub target_memory_percent: f32,
    /// Maximum acceptable memory utilization (%)
    pub max_memory_percent: f32,
    /// Minimum acceptable quality score
    pub min_quality_score: f32,
    /// Target quality score
    pub target_quality_score: f32,
}

/// Optimization actions that can be taken
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationAction {
    /// Reduce processing quality to save resources
    ReduceQuality {
        /// Target quality level (0.0-1.0)
        target_quality: f32,
    },
    /// Increase resource allocation
    IncreaseResources {
        /// Additional CPU threads
        cpu_threads: u32,
        /// Additional memory in MB
        memory_mb: u32,
    },
    /// Switch to different performance profile
    SwitchProfile {
        /// Target profile
        target_profile: Profile,
    },
    /// Optimize for reduced latency
    ReduceLatency {
        /// Target latency in ms
        target_latency_ms: f32,
    },
    /// Optimize batch processing
    OptimizeBatching {
        /// New batch size
        batch_size: u32,
        /// Batch timeout in ms
        timeout_ms: u32,
    },
    /// Adjust frame processing parameters
    AdjustFrameProcessing {
        /// Frame size in samples
        frame_size: u32,
        /// Hop length in samples
        hop_length: u32,
    },
}

/// Optimizer internal state
#[derive(Debug)]
struct OptimizerState {
    /// Optimizer active
    active: bool,
    /// Current optimization level (0-10)
    optimization_level: u8,
    /// Current profile
    current_profile: Profile,
    /// Last optimization time
    last_optimization: Option<Instant>,
    /// Performance trend analysis
    trend_analyzer: TrendAnalyzer,
    /// Optimization effectiveness tracking
    effectiveness_tracker: EffectivenessTracker,
}

/// Performance trend analyzer
#[derive(Debug)]
struct TrendAnalyzer {
    /// Recent latency values
    latency_samples: VecDeque<f32>,
    /// Recent CPU values
    cpu_samples: VecDeque<f32>,
    /// Recent memory values
    memory_samples: VecDeque<f32>,
    /// Recent quality values
    quality_samples: VecDeque<f32>,
    /// Sample window size
    window_size: usize,
}

/// Optimization effectiveness tracker
#[derive(Debug)]
struct EffectivenessTracker {
    /// Optimization attempts
    optimization_attempts: u32,
    /// Successful optimizations
    successful_optimizations: u32,
    /// Performance improvements achieved
    improvements: Vec<PerformanceImprovement>,
}

/// Performance improvement record
#[derive(Debug, Clone)]
struct PerformanceImprovement {
    /// Action taken
    action: OptimizationAction,
    /// Performance before optimization
    before_metrics: PerformanceSnapshot,
    /// Performance after optimization
    after_metrics: PerformanceSnapshot,
    /// Improvement timestamp
    timestamp: Instant,
}

/// Simplified performance snapshot for tracking
#[derive(Debug, Clone)]
struct PerformanceSnapshot {
    /// Latency (ms)
    latency_ms: f32,
    /// CPU utilization (%)
    cpu_percent: f32,
    /// Memory utilization (%)
    memory_percent: f32,
    /// Quality score
    quality_score: f32,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            optimization_interval_s: 30,
            sensitivity: 0.7,
            max_optimization_level: 5,
            min_samples_for_decision: 10,
            optimization_cooldown_s: 60,
        }
    }
}

impl Default for PerformanceTarget {
    fn default() -> Self {
        Self {
            target_latency_ms: 200.0,
            max_latency_ms: 500.0,
            target_cpu_percent: 60.0,
            max_cpu_percent: 85.0,
            target_memory_percent: 70.0,
            max_memory_percent: 90.0,
            min_quality_score: 0.75,
            target_quality_score: 0.90,
        }
    }
}

impl TrendAnalyzer {
    fn new(window_size: usize) -> Self {
        Self {
            latency_samples: VecDeque::with_capacity(window_size),
            cpu_samples: VecDeque::with_capacity(window_size),
            memory_samples: VecDeque::with_capacity(window_size),
            quality_samples: VecDeque::with_capacity(window_size),
            window_size,
        }
    }

    fn add_sample(&mut self, metrics: &PerformanceMetrics) {
        self.add_latency_sample(metrics.latency.avg_latency_ms);
        self.add_cpu_sample(metrics.resources.cpu_utilization_percent);
        self.add_memory_sample(metrics.resources.memory_utilization_percent);
        self.add_quality_sample(metrics.quality.overall_quality_score);
    }

    fn add_latency_sample(&mut self, latency: f32) {
        if self.latency_samples.len() >= self.window_size {
            self.latency_samples.pop_front();
        }
        self.latency_samples.push_back(latency);
    }

    fn add_cpu_sample(&mut self, cpu: f32) {
        if self.cpu_samples.len() >= self.window_size {
            self.cpu_samples.pop_front();
        }
        self.cpu_samples.push_back(cpu);
    }

    fn add_memory_sample(&mut self, memory: f32) {
        if self.memory_samples.len() >= self.window_size {
            self.memory_samples.pop_front();
        }
        self.memory_samples.push_back(memory);
    }

    fn add_quality_sample(&mut self, quality: f32) {
        if self.quality_samples.len() >= self.window_size {
            self.quality_samples.pop_front();
        }
        self.quality_samples.push_back(quality);
    }

    fn get_latency_trend(&self) -> TrendDirection {
        self.calculate_trend(&self.latency_samples)
    }

    fn get_cpu_trend(&self) -> TrendDirection {
        self.calculate_trend(&self.cpu_samples)
    }

    fn get_memory_trend(&self) -> TrendDirection {
        self.calculate_trend(&self.memory_samples)
    }

    fn get_quality_trend(&self) -> TrendDirection {
        self.calculate_trend(&self.quality_samples)
    }

    fn calculate_trend(&self, samples: &VecDeque<f32>) -> TrendDirection {
        if samples.len() < 3 {
            return TrendDirection::Stable;
        }

        let recent_avg = samples.iter().rev().take(3).sum::<f32>() / 3.0;
        let older_avg = samples.iter().take(3).sum::<f32>() / 3.0;

        let change_percent = (recent_avg - older_avg) / older_avg;

        if change_percent > 0.1 {
            TrendDirection::Increasing
        } else if change_percent < -0.1 {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        }
    }
}

/// Trend direction for performance metrics
#[derive(Debug, Clone, PartialEq)]
enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
}

impl EffectivenessTracker {
    fn new() -> Self {
        Self {
            optimization_attempts: 0,
            successful_optimizations: 0,
            improvements: Vec::new(),
        }
    }

    fn record_optimization_attempt(&mut self, action: OptimizationAction, before: PerformanceSnapshot) {
        self.optimization_attempts += 1;
        // Store the action and before metrics for later evaluation
    }

    fn record_optimization_result(&mut self, action: OptimizationAction,
                                  before: PerformanceSnapshot, after: PerformanceSnapshot) {
        let improvement = PerformanceImprovement {
            action,
            before_metrics: before,
            after_metrics: after,
            timestamp: Instant::now(),
        };

        // Check if optimization was successful
        let was_successful = self.evaluate_improvement(&improvement);
        if was_successful {
            self.successful_optimizations += 1;
        }

        self.improvements.push(improvement);

        // Keep only recent improvements
        if self.improvements.len() > 100 {
            self.improvements.remove(0);
        }
    }

    fn evaluate_improvement(&self, improvement: &PerformanceImprovement) -> bool {
        let before = &improvement.before_metrics;
        let after = &improvement.after_metrics;

        // Consider optimization successful if it improved target metrics
        let latency_improved = after.latency_ms < before.latency_ms;
        let cpu_improved = after.cpu_percent < before.cpu_percent;
        let quality_maintained = after.quality_score >= before.quality_score * 0.95; // Allow 5% quality loss

        // Different criteria based on optimization type
        match improvement.action {
            OptimizationAction::ReduceLatency { .. } => latency_improved && quality_maintained,
            OptimizationAction::ReduceQuality { .. } => (cpu_improved || latency_improved),
            OptimizationAction::IncreaseResources { .. } => latency_improved || quality_maintained,
            _ => latency_improved || cpu_improved,
        }
    }

    fn get_success_rate(&self) -> f32 {
        if self.optimization_attempts == 0 {
            1.0
        } else {
            self.successful_optimizations as f32 / self.optimization_attempts as f32
        }
    }
}

impl AdaptiveOptimizer {
    /// Create new adaptive optimizer
    pub async fn new(config: AdaptiveConfig) -> Result<Self> {
        let state = OptimizerState {
            active: false,
            optimization_level: 0,
            current_profile: Profile::Medium,
            last_optimization: None,
            trend_analyzer: TrendAnalyzer::new(50),
            effectiveness_tracker: EffectivenessTracker::new(),
        };

        Ok(Self {
            config,
            state: Arc::new(RwLock::new(state)),
            performance_history: Arc::new(Mutex::new(VecDeque::with_capacity(1000))),
            targets: Arc::new(RwLock::new(PerformanceTarget::default())),
            actions_history: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Start adaptive optimization
    pub async fn start(&self) -> Result<()> {
        if !self.config.enabled {
            info!("🔧 Adaptive optimizer disabled in configuration");
            return Ok(());
        }

        {
            let mut state = self.state.write().await;
            state.active = true;
        }

        // Start optimization loop
        self.start_optimization_loop().await;

        info!("🔧 Adaptive optimizer started");
        Ok(())
    }

    /// Stop adaptive optimization
    pub async fn stop(&self) -> Result<()> {
        {
            let mut state = self.state.write().await;
            state.active = false;
        }

        info!("🔧 Adaptive optimizer stopped");
        Ok(())
    }

    /// Start optimization loop
    async fn start_optimization_loop(&self) {
        let state = Arc::clone(&self.state);
        let performance_history = Arc::clone(&self.performance_history);
        let targets = Arc::clone(&self.targets);
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(config.optimization_interval_s));

            loop {
                interval.tick().await;

                let active = {
                    let state_guard = state.read().await;
                    state_guard.active
                };

                if !active {
                    break;
                }

                // Analyze performance and determine if optimization is needed
                if let Ok(action) = Self::evaluate_optimization_need(
                    &state, &performance_history, &targets, &config
                ).await {
                    if let Some(action) = action {
                        info!("🔧 Adaptive optimizer recommending action: {:?}", action);
                        // In real implementation, this would trigger the actual optimization
                    }
                }
            }

            debug!("Adaptive optimization loop stopped");
        });
    }

    /// Evaluate if optimization is needed
    async fn evaluate_optimization_need(
        state: &Arc<RwLock<OptimizerState>>,
        performance_history: &Arc<Mutex<VecDeque<PerformanceMetrics>>>,
        targets: &Arc<RwLock<PerformanceTarget>>,
        config: &AdaptiveConfig,
    ) -> Result<Option<OptimizationAction>> {
        let history = performance_history.lock().await;

        if history.len() < config.min_samples_for_decision as usize {
            return Ok(None);
        }

        let targets_guard = targets.read().await;
        let mut state_guard = state.write().await;

        // Check cooldown period
        if let Some(last_opt) = state_guard.last_optimization {
            if last_opt.elapsed().as_secs() < config.optimization_cooldown_s {
                return Ok(None);
            }
        }

        // Get recent performance metrics
        let recent_metrics = history.back().unwrap();

        // Update trend analyzer
        state_guard.trend_analyzer.add_sample(recent_metrics);

        // Analyze current performance against targets
        let latency_violation = recent_metrics.latency.avg_latency_ms > targets_guard.max_latency_ms;
        let cpu_violation = recent_metrics.resources.cpu_utilization_percent > targets_guard.max_cpu_percent;
        let memory_violation = recent_metrics.resources.memory_utilization_percent > targets_guard.max_memory_percent;
        let quality_violation = recent_metrics.quality.overall_quality_score < targets_guard.min_quality_score;

        // Determine appropriate optimization action
        let action = if latency_violation {
            // High latency - try to reduce it
            if recent_metrics.quality.overall_quality_score > targets_guard.target_quality_score {
                Some(OptimizationAction::ReduceQuality {
                    target_quality: targets_guard.target_quality_score * 0.9,
                })
            } else {
                Some(OptimizationAction::ReduceLatency {
                    target_latency_ms: targets_guard.target_latency_ms,
                })
            }
        } else if cpu_violation || memory_violation {
            // High resource usage
            if state_guard.current_profile != Profile::Low {
                Some(OptimizationAction::SwitchProfile {
                    target_profile: state_guard.current_profile.downgrade(),
                })
            } else {
                Some(OptimizationAction::ReduceQuality {
                    target_quality: recent_metrics.quality.overall_quality_score * 0.85,
                })
            }
        } else if quality_violation {
            // Low quality - try to improve it
            if recent_metrics.resources.cpu_utilization_percent < targets_guard.target_cpu_percent * 0.8 {
                Some(OptimizationAction::IncreaseResources {
                    cpu_threads: 1,
                    memory_mb: 256,
                })
            } else if state_guard.current_profile != Profile::High {
                Some(OptimizationAction::SwitchProfile {
                    target_profile: state_guard.current_profile.upgrade(),
                })
            } else {
                None
            }
        } else {
            None
        };

        if action.is_some() {
            state_guard.last_optimization = Some(Instant::now());
        }

        Ok(action)
    }

    /// Analyze metrics and suggest optimization
    pub async fn analyze_metrics(&self, metrics: &PerformanceMetrics) -> Result<Option<OptimizationAction>> {
        // Add metrics to history
        {
            let mut history = self.performance_history.lock().await;
            history.push_back(metrics.clone());

            // Keep only recent history
            if history.len() > 1000 {
                history.pop_front();
            }
        }

        // Check if optimization is needed
        Self::evaluate_optimization_need(
            &self.state, &self.performance_history, &self.targets, &self.config
        ).await
    }

    /// Set performance targets
    pub async fn set_targets(&self, targets: PerformanceTarget) {
        let mut targets_guard = self.targets.write().await;
        *targets_guard = targets;
    }

    /// Get current optimization status
    pub async fn get_status(&self) -> Result<OptimizationStatus> {
        let state = self.state.read().await;
        let actions = self.actions_history.lock().await;

        Ok(OptimizationStatus {
            active: state.active,
            level: state.optimization_level,
            last_action: actions.last().map(|a| format!("{:?}", a)),
        })
    }

    /// Get optimization summary
    pub async fn get_summary(&self) -> Result<OptimizationSummary> {
        let state = self.state.read().await;
        let actions = self.actions_history.lock().await;

        Ok(OptimizationSummary {
            total_optimizations: state.effectiveness_tracker.optimization_attempts,
            success_rate: state.effectiveness_tracker.get_success_rate(),
        })
    }

    /// Record optimization action
    pub async fn record_action(&self, action: OptimizationAction) {
        let mut actions = self.actions_history.lock().await;
        actions.push(action);

        // Keep only recent actions
        if actions.len() > 100 {
            actions.remove(0);
        }
    }
}

/// Optimization status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationStatus {
    /// Optimization active
    pub active: bool,
    /// Current optimization level
    pub level: u8,
    /// Last optimization action
    pub last_action: Option<String>,
}

/// Optimization summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSummary {
    /// Total optimization attempts
    pub total_optimizations: u32,
    /// Success rate (0.0-1.0)
    pub success_rate: f32,
}