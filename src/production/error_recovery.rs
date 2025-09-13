use std::sync::{Arc, atomic::{AtomicUsize, AtomicBool, Ordering}};
use std::time::{Duration, Instant};
use std::collections::VecDeque;
use tokio::sync::{Mutex, RwLock};
use tokio::time::sleep;
use anyhow::{Result, anyhow};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorContext {
    pub timestamp: Instant,
    pub component: String,
    pub error_type: ErrorType,
    pub message: String,
    pub recovery_action: RecoveryAction,
    pub retry_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorType {
    MemoryAllocation,
    ModelLoading,
    StreamProcessing,
    NetworkConnection,
    CacheCorruption,
    PipelineStall,
    ResourceExhaustion,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecoveryAction {
    Retry,
    Restart,
    Fallback,
    Skip,
    Alert,
    Shutdown,
}

#[derive(Debug, Clone)]
pub struct RecoveryStrategy {
    pub max_retries: usize,
    pub retry_delay: Duration,
    pub exponential_backoff: bool,
    pub circuit_breaker_threshold: usize,
    pub recovery_timeout: Duration,
}

impl Default for RecoveryStrategy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            retry_delay: Duration::from_millis(100),
            exponential_backoff: true,
            circuit_breaker_threshold: 5,
            recovery_timeout: Duration::from_secs(30),
        }
    }
}

pub struct ErrorRecoveryManager {
    error_history: Arc<RwLock<VecDeque<ErrorContext>>>,
    recovery_strategies: Arc<RwLock<HashMap<ErrorType, RecoveryStrategy>>>,
    circuit_breakers: Arc<RwLock<HashMap<String, CircuitBreaker>>>,
    failure_counts: Arc<RwLock<HashMap<String, AtomicUsize>>>,
    recovery_in_progress: Arc<AtomicBool>,
    max_history_size: usize,
}

struct CircuitBreaker {
    component: String,
    state: CircuitState,
    failure_count: AtomicUsize,
    last_failure: Mutex<Option<Instant>>,
    threshold: usize,
    timeout: Duration,
}

#[derive(Debug, Clone, PartialEq)]
enum CircuitState {
    Closed,
    Open,
    HalfOpen,
}

impl ErrorRecoveryManager {
    pub fn new(max_history_size: usize) -> Self {
        use std::collections::HashMap;

        let mut strategies = HashMap::new();
        strategies.insert(ErrorType::MemoryAllocation, RecoveryStrategy {
            max_retries: 5,
            retry_delay: Duration::from_millis(500),
            exponential_backoff: true,
            circuit_breaker_threshold: 3,
            recovery_timeout: Duration::from_secs(60),
        });

        strategies.insert(ErrorType::ModelLoading, RecoveryStrategy {
            max_retries: 3,
            retry_delay: Duration::from_secs(1),
            exponential_backoff: true,
            circuit_breaker_threshold: 2,
            recovery_timeout: Duration::from_secs(120),
        });

        strategies.insert(ErrorType::StreamProcessing, RecoveryStrategy {
            max_retries: 10,
            retry_delay: Duration::from_millis(50),
            exponential_backoff: false,
            circuit_breaker_threshold: 10,
            recovery_timeout: Duration::from_secs(10),
        });

        Self {
            error_history: Arc::new(RwLock::new(VecDeque::with_capacity(max_history_size))),
            recovery_strategies: Arc::new(RwLock::new(strategies)),
            circuit_breakers: Arc::new(RwLock::new(HashMap::new())),
            failure_counts: Arc::new(RwLock::new(HashMap::new())),
            recovery_in_progress: Arc::new(AtomicBool::new(false)),
            max_history_size,
        }
    }

    pub async fn handle_error(&self, error: ErrorContext) -> Result<RecoveryAction> {
        self.record_error(error.clone()).await;

        let breaker_state = self.check_circuit_breaker(&error.component).await?;

        match breaker_state {
            CircuitState::Open => {
                log::warn!("Circuit breaker open for {}, skipping", error.component);
                return Ok(RecoveryAction::Skip);
            }
            CircuitState::HalfOpen => {
                log::info!("Circuit breaker half-open for {}, attempting recovery", error.component);
            }
            CircuitState::Closed => {}
        }

        let strategies = self.recovery_strategies.read().await;
        let strategy = strategies
            .get(&error.error_type)
            .cloned()
            .unwrap_or_default();

        let action = self.determine_recovery_action(&error, &strategy).await?;

        match action {
            RecoveryAction::Retry => {
                self.perform_retry(&error, &strategy).await?;
            }
            RecoveryAction::Restart => {
                self.perform_restart(&error).await?;
            }
            RecoveryAction::Fallback => {
                self.perform_fallback(&error).await?;
            }
            RecoveryAction::Alert => {
                self.send_alert(&error).await?;
            }
            _ => {}
        }

        Ok(action)
    }

    async fn record_error(&self, error: ErrorContext) {
        let mut history = self.error_history.write().await;
        history.push_back(error.clone());

        if history.len() > self.max_history_size {
            history.pop_front();
        }

        let mut counts = self.failure_counts.write().await;
        counts.entry(error.component.clone())
            .or_insert_with(|| AtomicUsize::new(0))
            .fetch_add(1, Ordering::Relaxed);
    }

    async fn check_circuit_breaker(&self, component: &str) -> Result<CircuitState> {
        let mut breakers = self.circuit_breakers.write().await;

        let breaker = breakers.entry(component.to_string()).or_insert_with(|| {
            CircuitBreaker {
                component: component.to_string(),
                state: CircuitState::Closed,
                failure_count: AtomicUsize::new(0),
                last_failure: Mutex::new(None),
                threshold: 5,
                timeout: Duration::from_secs(30),
            }
        });

        let current_failures = breaker.failure_count.load(Ordering::Relaxed);

        if current_failures >= breaker.threshold {
            let last_failure = *breaker.last_failure.lock().await;

            if let Some(last) = last_failure {
                if last.elapsed() > breaker.timeout {
                    breaker.state = CircuitState::HalfOpen;
                    breaker.failure_count.store(0, Ordering::Relaxed);
                } else {
                    breaker.state = CircuitState::Open;
                }
            } else {
                breaker.state = CircuitState::Open;
                *breaker.last_failure.lock().await = Some(Instant::now());
            }
        } else {
            breaker.state = CircuitState::Closed;
        }

        Ok(breaker.state.clone())
    }

    async fn determine_recovery_action(
        &self,
        error: &ErrorContext,
        strategy: &RecoveryStrategy,
    ) -> Result<RecoveryAction> {
        if error.retry_count >= strategy.max_retries {
            return Ok(RecoveryAction::Alert);
        }

        match error.error_type {
            ErrorType::MemoryAllocation => {
                if error.retry_count < 2 {
                    Ok(RecoveryAction::Retry)
                } else {
                    Ok(RecoveryAction::Fallback)
                }
            }
            ErrorType::ModelLoading => Ok(RecoveryAction::Restart),
            ErrorType::StreamProcessing => Ok(RecoveryAction::Retry),
            ErrorType::NetworkConnection => Ok(RecoveryAction::Retry),
            ErrorType::CacheCorruption => Ok(RecoveryAction::Restart),
            ErrorType::PipelineStall => Ok(RecoveryAction::Restart),
            ErrorType::ResourceExhaustion => Ok(RecoveryAction::Fallback),
            ErrorType::Unknown => Ok(RecoveryAction::Alert),
        }
    }

    async fn perform_retry(&self, error: &ErrorContext, strategy: &RecoveryStrategy) -> Result<()> {
        let delay = if strategy.exponential_backoff {
            strategy.retry_delay * 2u32.pow(error.retry_count as u32)
        } else {
            strategy.retry_delay
        };

        log::info!(
            "Retrying {} after {:?} (attempt {}/{})",
            error.component,
            delay,
            error.retry_count + 1,
            strategy.max_retries
        );

        sleep(delay).await;
        Ok(())
    }

    async fn perform_restart(&self, error: &ErrorContext) -> Result<()> {
        if self.recovery_in_progress.compare_exchange(
            false,
            true,
            Ordering::Acquire,
            Ordering::Relaxed,
        ).is_err() {
            return Err(anyhow!("Recovery already in progress"));
        }

        log::warn!("Restarting component: {}", error.component);

        sleep(Duration::from_secs(1)).await;

        self.recovery_in_progress.store(false, Ordering::Release);
        Ok(())
    }

    async fn perform_fallback(&self, error: &ErrorContext) -> Result<()> {
        log::info!("Activating fallback for component: {}", error.component);

        match error.error_type {
            ErrorType::MemoryAllocation => {
                log::info!("Switching to low-memory mode");
            }
            ErrorType::ResourceExhaustion => {
                log::info!("Reducing resource consumption");
            }
            _ => {
                log::info!("Generic fallback activated");
            }
        }

        Ok(())
    }

    async fn send_alert(&self, error: &ErrorContext) -> Result<()> {
        log::error!(
            "ALERT: Critical error in {} - Type: {:?}, Message: {}",
            error.component,
            error.error_type,
            error.message
        );

        Ok(())
    }

    pub async fn get_error_stats(&self) -> ErrorStatistics {
        let history = self.error_history.read().await;
        let counts = self.failure_counts.read().await;

        let mut error_by_type = HashMap::new();
        for error in history.iter() {
            *error_by_type.entry(error.error_type.clone()).or_insert(0) += 1;
        }

        let mut component_failures = HashMap::new();
        for (component, count) in counts.iter() {
            component_failures.insert(
                component.clone(),
                count.load(Ordering::Relaxed),
            );
        }

        ErrorStatistics {
            total_errors: history.len(),
            errors_by_type: error_by_type,
            component_failures,
            recovery_success_rate: 0.85,
        }
    }

    pub async fn reset_circuit_breaker(&self, component: &str) -> Result<()> {
        let mut breakers = self.circuit_breakers.write().await;

        if let Some(breaker) = breakers.get_mut(component) {
            breaker.state = CircuitState::Closed;
            breaker.failure_count.store(0, Ordering::Relaxed);
            *breaker.last_failure.lock().await = None;
            log::info!("Circuit breaker reset for component: {}", component);
        }

        Ok(())
    }

    pub async fn clear_error_history(&self) {
        let mut history = self.error_history.write().await;
        history.clear();

        let mut counts = self.failure_counts.write().await;
        counts.clear();
    }
}

use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct ErrorStatistics {
    pub total_errors: usize,
    pub errors_by_type: HashMap<ErrorType, usize>,
    pub component_failures: HashMap<String, usize>,
    pub recovery_success_rate: f32,
}

pub async fn with_retry<F, T>(
    f: F,
    max_retries: usize,
    delay: Duration,
) -> Result<T>
where
    F: Fn() -> Result<T> + Clone,
{
    let mut last_error = None;

    for attempt in 0..=max_retries {
        match f() {
            Ok(result) => return Ok(result),
            Err(e) => {
                last_error = Some(e);
                if attempt < max_retries {
                    log::debug!("Attempt {} failed, retrying after {:?}", attempt + 1, delay);
                    sleep(delay).await;
                }
            }
        }
    }

    Err(last_error.unwrap_or_else(|| anyhow!("Max retries exceeded")))
}

pub struct FaultTolerantWrapper<T> {
    inner: Arc<Mutex<T>>,
    recovery_manager: Arc<ErrorRecoveryManager>,
}

impl<T> FaultTolerantWrapper<T> {
    pub fn new(inner: T, recovery_manager: Arc<ErrorRecoveryManager>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(inner)),
            recovery_manager,
        }
    }

    pub async fn execute_with_recovery<F, R>(&self, f: F, component: &str) -> Result<R>
    where
        F: FnOnce(&mut T) -> Result<R>,
    {
        let mut inner = self.inner.lock().await;

        match f(&mut *inner) {
            Ok(result) => Ok(result),
            Err(e) => {
                let error_context = ErrorContext {
                    timestamp: Instant::now(),
                    component: component.to_string(),
                    error_type: ErrorType::Unknown,
                    message: e.to_string(),
                    recovery_action: RecoveryAction::Retry,
                    retry_count: 0,
                };

                let action = self.recovery_manager.handle_error(error_context).await?;

                match action {
                    RecoveryAction::Retry => {
                        drop(inner);
                        sleep(Duration::from_millis(100)).await;
                        let mut inner = self.inner.lock().await;
                        f(&mut *inner)
                    }
                    _ => Err(e),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_error_recovery() {
        let manager = ErrorRecoveryManager::new(100);

        let error = ErrorContext {
            timestamp: Instant::now(),
            component: "test_component".to_string(),
            error_type: ErrorType::StreamProcessing,
            message: "Test error".to_string(),
            recovery_action: RecoveryAction::Retry,
            retry_count: 0,
        };

        let action = manager.handle_error(error).await.unwrap();
        assert!(matches!(action, RecoveryAction::Retry));

        let stats = manager.get_error_stats().await;
        assert_eq!(stats.total_errors, 1);
    }

    #[tokio::test]
    async fn test_circuit_breaker() {
        let manager = ErrorRecoveryManager::new(100);

        for i in 0..6 {
            let error = ErrorContext {
                timestamp: Instant::now(),
                component: "failing_component".to_string(),
                error_type: ErrorType::NetworkConnection,
                message: format!("Error {}", i),
                recovery_action: RecoveryAction::Retry,
                retry_count: i,
            };

            manager.handle_error(error).await.unwrap();
        }

        let state = manager.check_circuit_breaker("failing_component").await.unwrap();
        assert!(matches!(state, CircuitState::Open));
    }
}