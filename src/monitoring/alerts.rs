//! Performance Alert System
//!
//! This module provides comprehensive alerting including:
//! - Configurable alert rules and thresholds
//! - Multiple alert severity levels
//! - Alert channels (log, webhook, email)
//! - Alert suppression and rate limiting

use crate::monitoring::metrics::PerformanceMetrics;
use anyhow::Result;
use serde::{Serialize, Deserialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Mutex};
use tokio::time::interval;
use tracing::{info, warn, error, debug};

/// Alert manager for performance monitoring
#[derive(Clone)]
pub struct AlertManager {
    /// Alert rules
    alert_rules: Arc<RwLock<HashMap<String, AlertRule>>>,
    /// Active alerts
    active_alerts: Arc<RwLock<HashMap<String, ActiveAlert>>>,
    /// Alert history
    alert_history: Arc<Mutex<VecDeque<AlertEvent>>>,
    /// Alert channels
    alert_channels: Arc<RwLock<Vec<AlertChannel>>>,
    /// Manager state
    state: Arc<RwLock<AlertManagerState>>,
    /// Rate limiting
    rate_limiter: Arc<Mutex<RateLimiter>>,
}

/// Alert rule configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    /// Rule ID
    pub id: String,
    /// Rule name
    pub name: String,
    /// Rule description
    pub description: String,
    /// Metric to monitor
    pub metric: String,
    /// Comparison operator
    pub operator: ComparisonOperator,
    /// Threshold value
    pub threshold: f32,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Evaluation window (seconds)
    pub evaluation_window_s: u64,
    /// Number of violations required to trigger
    pub violation_count_threshold: u32,
    /// Rule enabled
    pub enabled: bool,
    /// Alert channels to notify
    pub channels: Vec<String>,
    /// Suppression settings
    pub suppression: SuppressionConfig,
}

/// Comparison operators for alert rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonOperator {
    /// Greater than threshold
    GreaterThan,
    /// Less than threshold
    LessThan,
    /// Equal to threshold
    Equal,
    /// Greater than or equal to threshold
    GreaterThanOrEqual,
    /// Less than or equal to threshold
    LessThanOrEqual,
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertSeverity {
    /// Informational alerts
    Info,
    /// Warning alerts
    Warning,
    /// Critical alerts requiring immediate attention
    Critical,
    /// Emergency alerts for system failure
    Emergency,
}

/// Alert suppression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuppressionConfig {
    /// Enable suppression
    pub enabled: bool,
    /// Suppression duration (seconds)
    pub duration_s: u64,
    /// Maximum alerts per hour
    pub max_alerts_per_hour: u32,
}

/// Active alert information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveAlert {
    /// Alert ID
    pub id: String,
    /// Rule that triggered the alert
    pub rule_id: String,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert message
    pub message: String,
    /// Alert start time
    #[serde(skip, default = "Instant::now")]
    pub start_time: Instant,
    /// Current value that triggered the alert
    pub current_value: f32,
    /// Threshold that was exceeded
    pub threshold: f32,
    /// Number of consecutive violations
    pub violation_count: u32,
    /// Last update time
    #[serde(skip, default = "Instant::now")]
    pub last_update: Instant,
}

/// Alert event for history tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertEvent {
    /// Event ID
    pub id: String,
    /// Event type
    pub event_type: AlertEventType,
    /// Rule ID
    pub rule_id: String,
    /// Event timestamp
    pub timestamp: u64,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Event message
    pub message: String,
    /// Event metadata
    pub metadata: HashMap<String, String>,
}

/// Alert event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertEventType {
    /// Alert triggered
    Triggered,
    /// Alert resolved
    Resolved,
    /// Alert suppressed
    Suppressed,
    /// Alert escalated
    Escalated,
}

/// Alert notification channels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertChannel {
    /// Log-based alerts
    Log {
        /// Log level
        level: String,
    },
    /// Webhook notifications
    Webhook {
        /// Webhook URL
        url: String,
        /// HTTP headers
        headers: HashMap<String, String>,
        /// Timeout in seconds
        timeout_s: u64,
    },
    /// Email notifications
    Email {
        /// SMTP server
        smtp_server: String,
        /// Sender email
        from_email: String,
        /// Recipient emails
        to_emails: Vec<String>,
        /// Email subject template
        subject_template: String,
    },
    /// Slack notifications
    Slack {
        /// Slack webhook URL
        webhook_url: String,
        /// Slack channel
        channel: String,
        /// Bot username
        username: String,
    },
}

/// Alert manager state
#[derive(Debug)]
struct AlertManagerState {
    /// Manager active
    active: bool,
    /// Start time
    start_time: Instant,
    /// Total alerts processed
    total_alerts: u64,
    /// Last evaluation time
    last_evaluation: Instant,
    /// Evaluation interval
    evaluation_interval: Duration,
}

/// Rate limiter for alert suppression
#[derive(Debug)]
struct RateLimiter {
    /// Alert counts per rule (timestamp, count)
    rule_counts: HashMap<String, VecDeque<Instant>>,
    /// Rate limit window
    window_duration: Duration,
}

impl Default for SuppressionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            duration_s: 300, // 5 minutes
            max_alerts_per_hour: 10,
        }
    }
}

impl RateLimiter {
    fn new() -> Self {
        Self {
            rule_counts: HashMap::new(),
            window_duration: Duration::from_secs(3600), // 1 hour
        }
    }

    fn should_allow_alert(&mut self, rule_id: &str, max_per_hour: u32) -> bool {
        let now = Instant::now();
        let rule_counts = self.rule_counts.entry(rule_id.to_string())
            .or_insert_with(VecDeque::new);

        // Remove old entries outside the window
        while let Some(&front_time) = rule_counts.front() {
            if now.duration_since(front_time) > self.window_duration {
                rule_counts.pop_front();
            } else {
                break;
            }
        }

        // Check if we can add another alert
        if rule_counts.len() < max_per_hour as usize {
            rule_counts.push_back(now);
            true
        } else {
            false
        }
    }
}

impl AlertManager {
    /// Create new alert manager
    pub async fn new() -> Result<Self> {
        let state = AlertManagerState {
            active: false,
            start_time: Instant::now(),
            total_alerts: 0,
            last_evaluation: Instant::now(),
            evaluation_interval: Duration::from_secs(10),
        };

        let alert_manager = Self {
            alert_rules: Arc::new(RwLock::new(HashMap::new())),
            active_alerts: Arc::new(RwLock::new(HashMap::new())),
            alert_history: Arc::new(Mutex::new(VecDeque::with_capacity(10000))),
            alert_channels: Arc::new(RwLock::new(Vec::new())),
            state: Arc::new(RwLock::new(state)),
            rate_limiter: Arc::new(Mutex::new(RateLimiter::new())),
        };

        // Initialize default alert rules
        alert_manager.initialize_default_rules().await?;

        // Initialize default channels
        alert_manager.initialize_default_channels().await?;

        Ok(alert_manager)
    }

    /// Initialize default alert rules
    async fn initialize_default_rules(&self) -> Result<()> {
        let mut rules = self.alert_rules.write().await;

        // High latency alert
        rules.insert(
            "high_latency".to_string(),
            AlertRule {
                id: "high_latency".to_string(),
                name: "High Processing Latency".to_string(),
                description: "Processing latency exceeds acceptable threshold".to_string(),
                metric: "latency.avg_latency_ms".to_string(),
                operator: ComparisonOperator::GreaterThan,
                threshold: 500.0,
                severity: AlertSeverity::Warning,
                evaluation_window_s: 60,
                violation_count_threshold: 3,
                enabled: true,
                channels: vec!["log".to_string()],
                suppression: SuppressionConfig::default(),
            },
        );

        // Critical latency alert
        rules.insert(
            "critical_latency".to_string(),
            AlertRule {
                id: "critical_latency".to_string(),
                name: "Critical Processing Latency".to_string(),
                description: "Processing latency is critically high".to_string(),
                metric: "latency.avg_latency_ms".to_string(),
                operator: ComparisonOperator::GreaterThan,
                threshold: 1000.0,
                severity: AlertSeverity::Critical,
                evaluation_window_s: 30,
                violation_count_threshold: 2,
                enabled: true,
                channels: vec!["log".to_string()],
                suppression: SuppressionConfig {
                    enabled: true,
                    duration_s: 600, // 10 minutes
                    max_alerts_per_hour: 5,
                },
            },
        );

        // High CPU usage alert
        rules.insert(
            "high_cpu".to_string(),
            AlertRule {
                id: "high_cpu".to_string(),
                name: "High CPU Usage".to_string(),
                description: "CPU utilization is above normal thresholds".to_string(),
                metric: "resources.cpu_utilization_percent".to_string(),
                operator: ComparisonOperator::GreaterThan,
                threshold: 85.0,
                severity: AlertSeverity::Warning,
                evaluation_window_s: 120,
                violation_count_threshold: 5,
                enabled: true,
                channels: vec!["log".to_string()],
                suppression: SuppressionConfig::default(),
            },
        );

        // Low quality alert
        rules.insert(
            "low_quality".to_string(),
            AlertRule {
                id: "low_quality".to_string(),
                name: "Low Translation Quality".to_string(),
                description: "Translation quality has degraded below acceptable levels".to_string(),
                metric: "quality.overall_quality_score".to_string(),
                operator: ComparisonOperator::LessThan,
                threshold: 0.7,
                severity: AlertSeverity::Warning,
                evaluation_window_s: 300,
                violation_count_threshold: 3,
                enabled: true,
                channels: vec!["log".to_string()],
                suppression: SuppressionConfig::default(),
            },
        );

        Ok(())
    }

    /// Initialize default alert channels
    async fn initialize_default_channels(&self) -> Result<()> {
        let mut channels = self.alert_channels.write().await;

        // Default log channel
        channels.push(AlertChannel::Log {
            level: "warn".to_string(),
        });

        Ok(())
    }

    /// Start alert manager
    pub async fn start(&self) -> Result<()> {
        {
            let mut state = self.state.write().await;
            state.active = true;
            state.start_time = Instant::now();
        }

        // Start alert evaluation loop
        self.start_evaluation_loop().await;

        info!("🚨 Alert manager started");
        Ok(())
    }

    /// Stop alert manager
    pub async fn stop(&self) -> Result<()> {
        {
            let mut state = self.state.write().await;
            state.active = false;
        }

        info!("🚨 Alert manager stopped");
        Ok(())
    }

    /// Start alert evaluation loop
    async fn start_evaluation_loop(&self) {
        let state = Arc::clone(&self.state);

        tokio::spawn(async move {
            let mut interval = {
                let state_guard = state.read().await;
                interval(state_guard.evaluation_interval)
            };

            loop {
                interval.tick().await;

                let active = {
                    let state_guard = state.read().await;
                    state_guard.active
                };

                if !active {
                    break;
                }

                // Alert evaluation logic would go here
                debug!("Alert evaluation tick");
            }

            debug!("Alert evaluation loop stopped");
        });
    }

    /// Check metrics against alert rules
    pub async fn check_metrics(&self, metrics: &PerformanceMetrics) -> Result<()> {
        let rules = self.alert_rules.read().await;
        let mut rate_limiter = self.rate_limiter.lock().await;

        for (rule_id, rule) in rules.iter() {
            if !rule.enabled {
                continue;
            }

            // Extract metric value
            let metric_value = self.extract_metric_value(metrics, &rule.metric)?;

            // Evaluate rule condition
            let condition_met = self.evaluate_condition(metric_value, &rule.operator, rule.threshold);

            if condition_met {
                // Check rate limiting
                if rule.suppression.enabled {
                    if !rate_limiter.should_allow_alert(rule_id, rule.suppression.max_alerts_per_hour) {
                        debug!("Alert {} suppressed due to rate limiting", rule_id);
                        continue;
                    }
                }

                // Check if alert is already active
                let mut active_alerts = self.active_alerts.write().await;

                if let Some(active_alert) = active_alerts.get_mut(rule_id) {
                    // Update existing alert
                    active_alert.violation_count += 1;
                    active_alert.current_value = metric_value;
                    active_alert.last_update = Instant::now();

                    // Check if we should escalate
                    if active_alert.violation_count >= rule.violation_count_threshold &&
                       active_alert.severity < AlertSeverity::Critical {
                        self.escalate_alert(active_alert, rule).await?;
                    }
                } else {
                    // Create new alert
                    let alert = ActiveAlert {
                        id: format!("alert_{}", uuid::Uuid::new_v4()),
                        rule_id: rule_id.clone(),
                        severity: rule.severity.clone(),
                        message: self.generate_alert_message(rule, metric_value),
                        start_time: Instant::now(),
                        current_value: metric_value,
                        threshold: rule.threshold,
                        violation_count: 1,
                        last_update: Instant::now(),
                    };

                    // Only trigger if we meet the violation threshold
                    if alert.violation_count >= rule.violation_count_threshold {
                        self.trigger_alert(&alert, rule).await?;
                        active_alerts.insert(rule_id.clone(), alert);
                    }
                }
            } else {
                // Condition not met - resolve alert if active
                let mut active_alerts = self.active_alerts.write().await;
                if let Some(alert) = active_alerts.remove(rule_id) {
                    self.resolve_alert(&alert, rule).await?;
                }
            }
        }

        Ok(())
    }

    /// Extract metric value from performance metrics
    fn extract_metric_value(&self, metrics: &PerformanceMetrics, metric_path: &str) -> Result<f32> {
        match metric_path {
            "latency.avg_latency_ms" => Ok(metrics.latency.avg_latency_ms),
            "latency.max_latency_ms" => Ok(metrics.latency.max_latency_ms),
            "resources.cpu_utilization_percent" => Ok(metrics.resources.cpu_utilization_percent),
            "resources.memory_utilization_percent" => Ok(metrics.resources.memory_utilization_percent),
            "quality.overall_quality_score" => Ok(metrics.quality.overall_quality_score),
            "quality.wer_score" => Ok(metrics.quality.wer_score),
            "quality.bleu_score" => Ok(metrics.quality.bleu_score),
            _ => Err(anyhow::anyhow!("Unknown metric path: {}", metric_path)),
        }
    }

    /// Evaluate alert condition
    fn evaluate_condition(&self, value: f32, operator: &ComparisonOperator, threshold: f32) -> bool {
        match operator {
            ComparisonOperator::GreaterThan => value > threshold,
            ComparisonOperator::LessThan => value < threshold,
            ComparisonOperator::Equal => (value - threshold).abs() < f32::EPSILON,
            ComparisonOperator::GreaterThanOrEqual => value >= threshold,
            ComparisonOperator::LessThanOrEqual => value <= threshold,
        }
    }

    /// Generate alert message
    fn generate_alert_message(&self, rule: &AlertRule, current_value: f32) -> String {
        format!(
            "{}: {} ({:.2}) {} threshold ({:.2})",
            rule.name,
            rule.metric,
            current_value,
            match rule.operator {
                ComparisonOperator::GreaterThan => "exceeds",
                ComparisonOperator::LessThan => "below",
                ComparisonOperator::GreaterThanOrEqual => "at or above",
                ComparisonOperator::LessThanOrEqual => "at or below",
                ComparisonOperator::Equal => "equals",
            },
            rule.threshold
        )
    }

    /// Trigger alert
    async fn trigger_alert(&self, alert: &ActiveAlert, rule: &AlertRule) -> Result<()> {
        let alert_event = AlertEvent {
            id: format!("event_{}", uuid::Uuid::new_v4()),
            event_type: AlertEventType::Triggered,
            rule_id: rule.id.clone(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
            severity: alert.severity.clone(),
            message: alert.message.clone(),
            metadata: HashMap::new(),
        };

        // Add to history
        {
            let mut history = self.alert_history.lock().await;
            history.push_back(alert_event.clone());

            // Keep only recent events
            if history.len() > 10000 {
                history.pop_front();
            }
        }

        // Send notifications
        self.send_alert_notifications(&alert_event, rule).await?;

        // Update state
        {
            let mut state = self.state.write().await;
            state.total_alerts += 1;
        }

        warn!("🚨 Alert triggered: {}", alert.message);
        Ok(())
    }

    /// Resolve alert
    async fn resolve_alert(&self, alert: &ActiveAlert, rule: &AlertRule) -> Result<()> {
        let alert_event = AlertEvent {
            id: format!("event_{}", uuid::Uuid::new_v4()),
            event_type: AlertEventType::Resolved,
            rule_id: rule.id.clone(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
            severity: alert.severity.clone(),
            message: format!("Resolved: {}", alert.message),
            metadata: HashMap::new(),
        };

        // Add to history
        {
            let mut history = self.alert_history.lock().await;
            history.push_back(alert_event);
        }

        info!("✅ Alert resolved: {}", alert.message);
        Ok(())
    }

    /// Escalate alert
    async fn escalate_alert(&self, alert: &mut ActiveAlert, rule: &AlertRule) -> Result<()> {
        alert.severity = AlertSeverity::Critical;

        let alert_event = AlertEvent {
            id: format!("event_{}", uuid::Uuid::new_v4()),
            event_type: AlertEventType::Escalated,
            rule_id: rule.id.clone(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_secs(),
            severity: alert.severity.clone(),
            message: format!("Escalated: {}", alert.message),
            metadata: HashMap::new(),
        };

        // Add to history
        {
            let mut history = self.alert_history.lock().await;
            history.push_back(alert_event.clone());
        }

        // Send escalation notifications
        self.send_alert_notifications(&alert_event, rule).await?;

        error!("🚨🚨 Alert escalated: {}", alert.message);
        Ok(())
    }

    /// Send alert notifications through configured channels
    async fn send_alert_notifications(&self, event: &AlertEvent, _rule: &AlertRule) -> Result<()> {
        let channels = self.alert_channels.read().await;

        for channel in channels.iter() {
            match channel {
                AlertChannel::Log { level } => {
                    match level.as_str() {
                        "error" => error!("Alert: {}", event.message),
                        "warn" => warn!("Alert: {}", event.message),
                        "info" => info!("Alert: {}", event.message),
                        _ => debug!("Alert: {}", event.message),
                    }
                },
                AlertChannel::Webhook { url, headers: _, timeout_s: _ } => {
                    debug!("Would send webhook notification to: {}", url);
                    // In real implementation, send HTTP POST to webhook
                },
                AlertChannel::Email { to_emails, .. } => {
                    debug!("Would send email notification to: {:?}", to_emails);
                    // In real implementation, send email via SMTP
                },
                AlertChannel::Slack { channel, .. } => {
                    debug!("Would send Slack notification to: {}", channel);
                    // In real implementation, send Slack message
                },
            }
        }

        Ok(())
    }

    /// Add alert rule
    pub async fn add_rule(&self, rule: AlertRule) -> Result<()> {
        let mut rules = self.alert_rules.write().await;
        rules.insert(rule.id.clone(), rule);
        Ok(())
    }

    /// Remove alert rule
    pub async fn remove_rule(&self, rule_id: &str) -> Result<()> {
        let mut rules = self.alert_rules.write().await;
        rules.remove(rule_id);

        // Also remove any active alerts for this rule
        let mut active_alerts = self.active_alerts.write().await;
        active_alerts.remove(rule_id);

        Ok(())
    }

    /// Add alert channel
    pub async fn add_channel(&self, channel: AlertChannel) -> Result<()> {
        let mut channels = self.alert_channels.write().await;
        channels.push(channel);
        Ok(())
    }

    /// Get alert status
    pub async fn get_status(&self) -> Result<AlertStatus> {
        let active_alerts = self.active_alerts.read().await;
        let _critical_count = active_alerts.values()
            .filter(|alert| alert.severity >= AlertSeverity::Critical)
            .count() as u32;

        Ok(AlertStatus {
            active_alerts: active_alerts.len() as u32,
            last_alert: active_alerts.values()
                .max_by_key(|alert| alert.start_time)
                .map(|alert| alert.message.clone()),
        })
    }

    /// Get alert summary
    pub async fn get_summary(&self) -> Result<AlertSummary> {
        let state = self.state.read().await;
        let active_alerts = self.active_alerts.read().await;
        let critical_count = active_alerts.values()
            .filter(|alert| alert.severity >= AlertSeverity::Critical)
            .count() as u32;

        Ok(AlertSummary {
            total_alerts: state.total_alerts,
            critical_alerts: critical_count,
        })
    }
}

/// Alert status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertStatus {
    /// Number of active alerts
    pub active_alerts: u32,
    /// Last alert message
    pub last_alert: Option<String>,
}

/// Alert summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertSummary {
    /// Total alerts processed
    pub total_alerts: u64,
    /// Critical alerts
    pub critical_alerts: u32,
}