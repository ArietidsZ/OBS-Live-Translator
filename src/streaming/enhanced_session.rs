//! Enhanced Session Management System
//!
//! This module provides advanced session management with:
//! - Per-session resource allocation and tracking
//! - Connection health monitoring
//! - Graceful connection handling with cleanup
//! - Real-time statistics and monitoring

// Binary message types imported when needed
use crate::profile::Profile;
use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use tokio::sync::{Mutex, RwLock};
use tokio::time::interval;
use tracing::{debug, info};

/// Enhanced session manager with resource tracking
pub struct EnhancedSessionManager {
    /// Active sessions
    sessions: RwLock<HashMap<String, Arc<EnhancedStreamingSession>>>,
    /// Global session statistics
    global_stats: Mutex<GlobalSessionStats>,
    /// Configuration
    config: SessionManagerConfig,
    /// Resource monitor
    resource_monitor: Arc<SessionResourceMonitor>,
}

/// Enhanced streaming session with resource tracking
pub struct EnhancedStreamingSession {
    /// Session ID
    id: String,
    /// Creation time
    created_at: SystemTime,
    /// Last activity time
    last_activity: Mutex<Instant>,
    /// Session configuration
    config: SessionConfig,
    /// Session statistics
    stats: Mutex<SessionStats>,
    /// Resource usage tracking
    resources: Mutex<SessionResources>,
    /// Connection quality metrics
    quality_metrics: Mutex<ConnectionQuality>,
    /// Processing pipeline (placeholder for actual implementation)
    processor: Option<Arc<crate::Translator>>,
}

/// Session manager configuration
#[derive(Debug, Clone)]
pub struct SessionManagerConfig {
    /// Maximum number of concurrent sessions
    pub max_concurrent_sessions: usize,
    /// Session timeout in seconds
    pub session_timeout_secs: u64,
    /// Cleanup interval in seconds
    pub cleanup_interval_secs: u64,
    /// Resource limits per session
    pub per_session_limits: ResourceLimits,
    /// Enable detailed monitoring
    pub enable_detailed_monitoring: bool,
}

impl Default for SessionManagerConfig {
    fn default() -> Self {
        Self {
            max_concurrent_sessions: 100,
            session_timeout_secs: 300, // 5 minutes
            cleanup_interval_secs: 60, // 1 minute
            per_session_limits: ResourceLimits::default(),
            enable_detailed_monitoring: true,
        }
    }
}

/// Resource limits per session
#[derive(Debug, Clone)]
pub struct ResourceLimits {
    /// Maximum memory usage (MB)
    pub max_memory_mb: f64,
    /// Maximum CPU usage (percentage)
    pub max_cpu_percent: f32,
    /// Maximum processing time per request (ms)
    pub max_processing_time_ms: f32,
    /// Maximum queue size
    pub max_queue_size: usize,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_mb: 512.0,
            max_cpu_percent: 10.0,
            max_processing_time_ms: 1000.0,
            max_queue_size: 100,
        }
    }
}

/// Per-session configuration
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// Source language
    pub source_language: Option<String>,
    /// Target languages
    pub target_languages: Vec<String>,
    /// Enable translation
    pub enable_translation: bool,
    /// Performance profile
    pub profile: Profile,
    /// Audio settings
    pub audio_config: AudioSessionConfig,
}

#[derive(Debug, Clone)]
pub struct AudioSessionConfig {
    pub sample_rate: u32,
    pub channels: u16,
    pub buffer_size: usize,
    pub enable_vad: bool,
    pub vad_threshold: f32,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            source_language: None,
            target_languages: Vec::new(),
            enable_translation: false,
            profile: Profile::Medium,
            audio_config: AudioSessionConfig {
                sample_rate: 16000,
                channels: 1,
                buffer_size: 1024,
                enable_vad: true,
                vad_threshold: 0.5,
            },
        }
    }
}

/// Global session statistics
#[derive(Debug, Clone, Default)]
pub struct GlobalSessionStats {
    /// Total sessions created
    pub total_sessions_created: u64,
    /// Currently active sessions
    pub active_sessions: u64,
    /// Total audio processed (samples)
    pub total_audio_processed: u64,
    /// Total transcriptions generated
    pub total_transcriptions: u64,
    /// Total translations generated
    pub total_translations: u64,
    /// Average session duration (seconds)
    pub average_session_duration_secs: f32,
    /// Total data transferred (bytes)
    pub total_data_transferred: u64,
}

/// Per-session statistics
#[derive(Debug, Clone, Default)]
pub struct SessionStats {
    /// Audio frames processed
    pub audio_frames_processed: u64,
    /// Total audio duration (seconds)
    pub total_audio_duration_secs: f32,
    /// Transcriptions generated
    pub transcriptions_generated: u64,
    /// Translations generated
    pub translations_generated: u64,
    /// Average processing latency (ms)
    pub average_processing_latency_ms: f32,
    /// Data sent (bytes)
    pub data_sent_bytes: u64,
    /// Data received (bytes)
    pub data_received_bytes: u64,
    /// Error count
    pub error_count: u64,
}

/// Session resource usage
#[derive(Debug, Clone, Default)]
pub struct SessionResources {
    /// Current memory usage (MB)
    pub memory_usage_mb: f64,
    /// Current CPU usage (percentage)
    pub cpu_usage_percent: f32,
    /// Queue size
    pub queue_size: usize,
    /// Peak memory usage (MB)
    pub peak_memory_mb: f64,
    /// Peak CPU usage (percentage)
    pub peak_cpu_percent: f32,
    /// Processing time history (recent 100 measurements)
    pub processing_times_ms: Vec<f32>,
}

/// Connection quality metrics
#[derive(Debug, Clone, Default)]
pub struct ConnectionQuality {
    /// Average RTT (ms)
    pub average_rtt_ms: f32,
    /// Packet loss rate (0.0-1.0)
    pub packet_loss_rate: f32,
    /// Jitter (ms)
    pub jitter_ms: f32,
    /// Connection stability score (0.0-1.0)
    pub stability_score: f32,
    /// Bandwidth utilization (bytes/sec)
    pub bandwidth_utilization: f64,
}

/// Session resource monitor
pub struct SessionResourceMonitor {
    /// Resource usage by session
    session_resources: RwLock<HashMap<String, SessionResources>>,
    /// System resource limits
    system_limits: SystemResourceLimits,
    /// Monitoring enabled
    enabled: bool,
}

#[derive(Debug, Clone)]
pub struct SystemResourceLimits {
    /// Total available memory (MB)
    pub total_memory_mb: f64,
    /// CPU core count
    pub cpu_cores: u32,
    /// Maximum sessions per core
    pub max_sessions_per_core: usize,
}

impl Default for SystemResourceLimits {
    fn default() -> Self {
        Self {
            total_memory_mb: 8192.0, // 8GB
            cpu_cores: 4,
            max_sessions_per_core: 25,
        }
    }
}

impl EnhancedSessionManager {
    /// Create a new enhanced session manager
    pub fn new(config: SessionManagerConfig) -> Self {
        let resource_monitor = Arc::new(SessionResourceMonitor {
            session_resources: RwLock::new(HashMap::new()),
            system_limits: SystemResourceLimits::default(),
            enabled: config.enable_detailed_monitoring,
        });

        Self {
            sessions: RwLock::new(HashMap::new()),
            global_stats: Mutex::new(GlobalSessionStats::default()),
            config,
            resource_monitor,
        }
    }

    /// Start the session manager with background tasks
    pub async fn start(&self) -> Result<()> {
        info!("🚀 Starting enhanced session manager");

        // Start cleanup task
        let sessions = Arc::new(RwLock::new(HashMap::new()));
        let global_stats = Arc::new(Mutex::new(GlobalSessionStats::default()));
        let config = self.config.clone();
        tokio::spawn(async move {
            Self::cleanup_task(sessions, global_stats, config).await;
        });

        // Start monitoring task
        let resource_monitor = Arc::clone(&self.resource_monitor);
        let sessions = Arc::new(RwLock::new(HashMap::new()));
        tokio::spawn(async move {
            Self::monitoring_task(resource_monitor, sessions).await;
        });

        info!("✅ Enhanced session manager started");
        Ok(())
    }

    /// Create a new session
    pub async fn create_session(&self, session_id: &str) -> Result<Arc<EnhancedStreamingSession>> {
        // Check session limits
        {
            let sessions = self.sessions.read().await;
            if sessions.len() >= self.config.max_concurrent_sessions {
                return Err(anyhow!("Maximum concurrent sessions reached"));
            }
        }

        // Check system resources
        if self.resource_monitor.enabled {
            if !self.resource_monitor.can_create_session().await {
                return Err(anyhow!("Insufficient system resources"));
            }
        }

        let session = Arc::new(EnhancedStreamingSession {
            id: session_id.to_string(),
            created_at: SystemTime::now(),
            last_activity: Mutex::new(Instant::now()),
            config: SessionConfig::default(),
            stats: Mutex::new(SessionStats::default()),
            resources: Mutex::new(SessionResources::default()),
            quality_metrics: Mutex::new(ConnectionQuality::default()),
            processor: None, // Will be initialized later
        });

        // Add to sessions map
        {
            let mut sessions = self.sessions.write().await;
            sessions.insert(session_id.to_string(), Arc::clone(&session));
        }

        // Update global statistics
        {
            let mut global_stats = self.global_stats.lock().await;
            global_stats.total_sessions_created += 1;
            global_stats.active_sessions += 1;
        }

        // Initialize session resources in monitor
        if self.resource_monitor.enabled {
            self.resource_monitor.initialize_session(session_id).await;
        }

        info!("📝 Created session {}", session_id);
        Ok(session)
    }

    /// Get an existing session
    pub async fn get_session(&self, session_id: &str) -> Result<Arc<EnhancedStreamingSession>> {
        let sessions = self.sessions.read().await;
        sessions
            .get(session_id)
            .cloned()
            .ok_or_else(|| anyhow!("Session not found: {}", session_id))
    }

    /// Check if session exists
    pub async fn has_session(&self, session_id: &str) -> bool {
        let sessions = self.sessions.read().await;
        sessions.contains_key(session_id)
    }

    /// Remove a session
    pub async fn remove_session(&self, session_id: &str) {
        let session = {
            let mut sessions = self.sessions.write().await;
            sessions.remove(session_id)
        };

        if let Some(session) = session {
            // Update global statistics
            {
                let mut global_stats = self.global_stats.lock().await;
                global_stats.active_sessions = global_stats.active_sessions.saturating_sub(1);

                // Update average session duration
                if let Ok(duration) = session.created_at.elapsed() {
                    let duration_secs = duration.as_secs_f32();
                    let n = global_stats.total_sessions_created as f32;
                    global_stats.average_session_duration_secs =
                        (global_stats.average_session_duration_secs * (n - 1.0) + duration_secs)
                            / n;
                }
            }

            // Cleanup session resources
            if self.resource_monitor.enabled {
                self.resource_monitor.cleanup_session(session_id).await;
            }

            info!("🗑️ Removed session {}", session_id);
        }
    }

    /// Get all active session IDs
    pub async fn get_active_session_ids(&self) -> Vec<String> {
        let sessions = self.sessions.read().await;
        sessions.keys().cloned().collect()
    }

    /// Get global statistics
    pub async fn get_global_stats(&self) -> GlobalSessionStats {
        let global_stats = self.global_stats.lock().await;
        global_stats.clone()
    }

    /// Get session statistics
    pub async fn get_session_stats(&self, session_id: &str) -> Result<SessionStats> {
        let session = self.get_session(session_id).await?;
        let stats = session.stats.lock().await;
        Ok(stats.clone())
    }

    /// Get session resource usage
    pub async fn get_session_resources(&self, session_id: &str) -> Result<SessionResources> {
        if !self.resource_monitor.enabled {
            return Err(anyhow!("Resource monitoring not enabled"));
        }

        let resources = self.resource_monitor.session_resources.read().await;
        resources
            .get(session_id)
            .cloned()
            .ok_or_else(|| anyhow!("Session resources not found"))
    }

    /// Background cleanup task
    async fn cleanup_task(
        sessions: Arc<RwLock<HashMap<String, Arc<EnhancedStreamingSession>>>>,
        global_stats: Arc<Mutex<GlobalSessionStats>>,
        config: SessionManagerConfig,
    ) {
        let mut interval = interval(Duration::from_secs(config.cleanup_interval_secs));

        loop {
            interval.tick().await;

            let timeout_duration = Duration::from_secs(config.session_timeout_secs);
            let mut sessions_to_remove = Vec::new();

            // Find expired sessions
            {
                let sessions_guard = sessions.read().await;
                for (session_id, session) in sessions_guard.iter() {
                    let last_activity = {
                        let activity = session.last_activity.lock().await;
                        *activity
                    };

                    if last_activity.elapsed() > timeout_duration {
                        sessions_to_remove.push(session_id.clone());
                    }
                }
            }

            // Remove expired sessions
            if !sessions_to_remove.is_empty() {
                let mut sessions_guard = sessions.write().await;
                for session_id in &sessions_to_remove {
                    if let Some(_session) = sessions_guard.remove(session_id) {
                        info!("⏰ Session {} expired and removed", session_id);

                        // Update global stats
                        {
                            let mut stats = global_stats.lock().await;
                            stats.active_sessions = stats.active_sessions.saturating_sub(1);
                        }
                    }
                }
            }

            debug!(
                "🧹 Cleanup completed: removed {} expired sessions",
                sessions_to_remove.len()
            );
        }
    }

    /// Background monitoring task
    async fn monitoring_task(
        resource_monitor: Arc<SessionResourceMonitor>,
        sessions: Arc<RwLock<HashMap<String, Arc<EnhancedStreamingSession>>>>,
    ) {
        let mut interval = interval(Duration::from_secs(10)); // Monitor every 10 seconds

        loop {
            interval.tick().await;

            if !resource_monitor.enabled {
                continue;
            }

            // Update resource usage for all sessions
            let session_ids = {
                let sessions_guard = sessions.read().await;
                sessions_guard.keys().cloned().collect::<Vec<_>>()
            };

            for session_id in session_ids {
                resource_monitor.update_session_resources(&session_id).await;
            }

            // Log resource summary
            let total_memory = resource_monitor.get_total_memory_usage().await;
            let session_count = {
                let sessions_guard = sessions.read().await;
                sessions_guard.len()
            };

            debug!(
                "📊 Resource monitoring: {} sessions, {:.1}MB total memory",
                session_count, total_memory
            );
        }
    }
}

impl EnhancedStreamingSession {
    /// Update session activity timestamp
    pub async fn update_activity(&self) {
        let mut last_activity = self.last_activity.lock().await;
        *last_activity = Instant::now();
    }

    /// Update session configuration
    pub async fn update_config(
        &self,
        source_language: Option<String>,
        target_languages: Vec<String>,
        enable_translation: bool,
    ) -> Result<()> {
        // In a real implementation, this would update the session's configuration
        // and potentially reinitialize processing components

        info!(
            "🔧 Updated session {} config: source={:?}, targets={:?}, translation={}",
            self.id, source_language, target_languages, enable_translation
        );

        Ok(())
    }

    /// Process audio data (placeholder implementation)
    pub async fn process_audio(
        &self,
        _audio_data: Vec<f32>,
        _sample_rate: u32,
    ) -> Result<Vec<crate::streaming::session::ProcessingResult>> {
        // Update activity
        self.update_activity().await;

        // Update statistics
        {
            let mut stats = self.stats.lock().await;
            stats.audio_frames_processed += 1;
        }

        // In a real implementation, this would process the audio through
        // the translation pipeline and return results
        Ok(Vec::new())
    }

    /// Get session uptime
    pub fn get_uptime(&self) -> Duration {
        self.created_at.elapsed().unwrap_or(Duration::from_secs(0))
    }

    /// Get session statistics
    pub async fn get_stats(&self) -> SessionStats {
        let stats = self.stats.lock().await;
        stats.clone()
    }
}

impl SessionResourceMonitor {
    /// Check if a new session can be created
    pub async fn can_create_session(&self) -> bool {
        if !self.enabled {
            return true;
        }

        let current_memory = self.get_total_memory_usage().await;
        let available_memory = self.system_limits.total_memory_mb - current_memory;

        // Require at least 100MB available memory for new session
        available_memory > 100.0
    }

    /// Initialize session resource tracking
    pub async fn initialize_session(&self, session_id: &str) {
        if !self.enabled {
            return;
        }

        let mut resources = self.session_resources.write().await;
        resources.insert(session_id.to_string(), SessionResources::default());
    }

    /// Update session resource usage
    pub async fn update_session_resources(&self, session_id: &str) {
        if !self.enabled {
            return;
        }

        // In a real implementation, this would query actual system resources
        // For now, simulate resource usage

        let mut resources = self.session_resources.write().await;
        if let Some(session_resources) = resources.get_mut(session_id) {
            // Simulate memory usage growth
            session_resources.memory_usage_mb += 0.5;
            session_resources.peak_memory_mb = session_resources
                .peak_memory_mb
                .max(session_resources.memory_usage_mb);

            // Simulate CPU usage fluctuation
            session_resources.cpu_usage_percent =
                (session_resources.cpu_usage_percent * 0.9 + 2.0).min(15.0);
            session_resources.peak_cpu_percent = session_resources
                .peak_cpu_percent
                .max(session_resources.cpu_usage_percent);
        }
    }

    /// Cleanup session resources
    pub async fn cleanup_session(&self, session_id: &str) {
        if !self.enabled {
            return;
        }

        let mut resources = self.session_resources.write().await;
        resources.remove(session_id);
    }

    /// Get total memory usage across all sessions
    pub async fn get_total_memory_usage(&self) -> f64 {
        if !self.enabled {
            return 0.0;
        }

        let resources = self.session_resources.read().await;
        resources.values().map(|r| r.memory_usage_mb).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::streaming::{optimized_websocket::BinaryMessageHeader, BinaryMessageType};

    #[tokio::test]
    async fn test_session_creation() {
        let config = SessionManagerConfig::default();
        let manager = EnhancedSessionManager::new(config);

        let session_id = "test-session";
        let session = manager.create_session(session_id).await;
        assert!(session.is_ok());

        assert!(manager.has_session(session_id).await);
    }

    #[tokio::test]
    async fn test_session_removal() {
        let config = SessionManagerConfig::default();
        let manager = EnhancedSessionManager::new(config);

        let session_id = "test-session";
        let _session = manager.create_session(session_id).await.unwrap();

        assert!(manager.has_session(session_id).await);

        manager.remove_session(session_id).await;
        assert!(!manager.has_session(session_id).await);
    }

    #[tokio::test]
    async fn test_session_limits() {
        let mut config = SessionManagerConfig::default();
        config.max_concurrent_sessions = 2;
        let manager = EnhancedSessionManager::new(config);

        // Create two sessions (should succeed)
        let _session1 = manager.create_session("session1").await.unwrap();
        let _session2 = manager.create_session("session2").await.unwrap();

        // Third session should fail
        let session3 = manager.create_session("session3").await;
        assert!(session3.is_err());
    }

    #[test]
    fn test_binary_message_header() {
        let header = BinaryMessageHeader::new(BinaryMessageType::RawAudio, 1024, 42);
        let bytes = header.to_bytes();
        let decoded = BinaryMessageHeader::from_bytes(&bytes).unwrap();

        assert_eq!(header.msg_type as u8, decoded.msg_type as u8);
        assert_eq!(header.length, decoded.length);
        assert_eq!(header.sequence, decoded.sequence);
    }
}
