//! Main profile manager orchestrating dynamic profile switching

use crate::profile::components::ComponentRegistry;
use crate::profile::monitor::ResourceMonitor;
use crate::profile::{Profile, ProfileBenchmark, ProfileDetector};
use anyhow::Result;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Main profile manager for OBS Live Translator
pub struct ProfileManager {
    current_profile: Profile,
    component_registry: Arc<RwLock<ComponentRegistry>>,
    resource_monitor: Arc<RwLock<ResourceMonitor>>,
    model_manager: Arc<RwLock<ModelManager>>,
    config: ProfileManagerConfig,
    switching_in_progress: Arc<RwLock<bool>>,
}

/// Configuration for profile manager
#[derive(Debug, Clone)]
pub struct ProfileManagerConfig {
    pub auto_switching_enabled: bool,
    pub switch_check_interval: Duration,
    pub benchmark_on_switch: bool,
    pub fallback_on_failure: bool,
}

impl Default for ProfileManagerConfig {
    fn default() -> Self {
        Self {
            auto_switching_enabled: true,
            switch_check_interval: Duration::from_secs(5),
            benchmark_on_switch: true,
            fallback_on_failure: true,
        }
    }
}

/// Model management for profile switching
pub struct ModelManager {
    current_profile: Profile,
    download_progress: Arc<RwLock<DownloadProgress>>,
}

/// Download progress tracking
#[derive(Debug, Clone)]
pub struct DownloadProgress {
    pub current_model: Option<String>,
    pub progress_percent: f32,
    pub total_size_mb: u64,
    pub downloaded_mb: u64,
    pub download_speed_mbps: f32,
    pub eta_seconds: Option<u64>,
}

/// Resource requirements for a profile
#[derive(Debug, Clone)]
pub struct ResourceRequirements {
    pub memory_mb: u64,
    pub gpu_memory_mb: u64,
    pub cpu_cores: u32,
}

impl ProfileManager {
    /// Initialize with optimal profile detection
    pub async fn initialize_optimal() -> Result<Self> {
        info!("🚀 Initializing OBS Live Translator with optimal profile detection");

        // Detect optimal profile
        let optimal_profile = ProfileDetector::detect_optimal_profile()?;
        info!("✅ Detected optimal profile: {:?}", optimal_profile);

        Self::initialize_for_profile(optimal_profile).await
    }

    /// Initialize for specific profile
    pub async fn initialize_for_profile(profile: Profile) -> Result<Self> {
        info!("Initializing ProfileManager for profile: {:?}", profile);

        // Create component registry
        let component_registry = Arc::new(RwLock::new(
            ComponentRegistry::new_for_profile(profile).await?,
        ));

        // Create resource monitor
        let resource_monitor = Arc::new(RwLock::new(ResourceMonitor::new(profile)));

        // Create model manager
        let model_manager = Arc::new(RwLock::new(ModelManager::new(profile).await?));

        let manager = Self {
            current_profile: profile,
            component_registry,
            resource_monitor,
            model_manager,
            config: ProfileManagerConfig::default(),
            switching_in_progress: Arc::new(RwLock::new(false)),
        };

        // Start resource monitoring
        manager.start_monitoring().await?;

        // Start adaptive management
        if manager.config.auto_switching_enabled {
            manager.start_adaptive_management().await?;
        }

        Ok(manager)
    }

    /// Get current profile
    pub fn current_profile(&self) -> Profile {
        self.current_profile
    }

    /// Manual profile switch with comprehensive validation and fallback
    pub async fn switch_profile(&mut self, target_profile: Profile) -> Result<()> {
        if target_profile == self.current_profile {
            info!("Already using profile {:?}", target_profile);
            return Ok(());
        }

        info!(
            "🔄 Switching from {:?} to {:?}",
            self.current_profile, target_profile
        );

        // Validate the target profile before attempting switch
        let validation_result = self.validate_profile_feasibility(target_profile).await;
        if let Err(e) = validation_result {
            warn!("❌ Profile validation failed: {}", e);

            if self.config.fallback_on_failure {
                return self.attempt_fallback_sequence(target_profile).await;
            } else {
                return Err(e);
            }
        }

        // Set switching flag
        {
            let mut switching = self.switching_in_progress.write().await;
            *switching = true;
        }

        let original_profile = self.current_profile;
        let result = self.perform_profile_switch(target_profile).await;

        // Clear switching flag
        {
            let mut switching = self.switching_in_progress.write().await;
            *switching = false;
        }

        match result {
            Ok(()) => {
                self.current_profile = target_profile;
                info!("✅ Successfully switched to profile {:?}", target_profile);

                // Verify stability after switch
                if let Err(stability_error) = self.verify_profile_stability(target_profile).await {
                    warn!("⚠️ Profile stability check failed: {}", stability_error);

                    if self.config.fallback_on_failure {
                        return self.emergency_fallback(original_profile).await;
                    }
                }

                Ok(())
            }
            Err(e) => {
                error!("❌ Failed to switch to profile {:?}: {}", target_profile, e);

                if self.config.fallback_on_failure {
                    self.emergency_fallback(original_profile).await
                } else {
                    Err(e)
                }
            }
        }
    }

    /// Validate if a profile is feasible on current hardware
    async fn validate_profile_feasibility(&self, target_profile: Profile) -> Result<()> {
        info!("🔍 Validating feasibility of profile {:?}", target_profile);

        if std::env::var("OBS_SKIP_PROFILE_VALIDATION").is_ok() {
            debug!("Skipping profile validation checks via OBS_SKIP_PROFILE_VALIDATION");
            return Ok(());
        }

        // 1. Hardware validation
        let hardware = ProfileDetector::scan_hardware()?;
        if !ProfileDetector::validate_profile_support(target_profile, &hardware) {
            return Err(anyhow::anyhow!(
                "Hardware insufficient for profile {:?}",
                target_profile
            ));
        }

        // 2. Resource availability check
        let current_metrics = self.get_resource_metrics().await?;
        let resource_requirements = self.get_profile_resource_requirements(target_profile);

        if current_metrics.memory_usage_mb + resource_requirements.memory_mb > hardware.total_ram_mb
        {
            return Err(anyhow::anyhow!(
                "Insufficient memory for profile {:?}: need {}MB, available {}MB",
                target_profile,
                resource_requirements.memory_mb,
                hardware.total_ram_mb - current_metrics.memory_usage_mb
            ));
        }

        if target_profile != Profile::Low {
            let gpu_memory_mb = hardware
                .gpu_info
                .first()
                .map(|gpu| gpu.total_vram_mb)
                .unwrap_or(0);
            if gpu_memory_mb < resource_requirements.gpu_memory_mb {
                return Err(anyhow::anyhow!(
                    "Insufficient GPU memory for profile {:?}: need {}MB, available {}MB",
                    target_profile,
                    resource_requirements.gpu_memory_mb,
                    gpu_memory_mb
                ));
            }
        }

        // 3. Model availability check
        let model_paths = self.get_required_model_paths(target_profile);
        for model_path in model_paths {
            if !std::path::Path::new(&model_path).exists() {
                return Err(anyhow::anyhow!(
                    "Required model missing for profile {:?}: {}",
                    target_profile,
                    model_path
                ));
            }
        }

        info!("✅ Profile {:?} validation passed", target_profile);
        Ok(())
    }

    /// Attempt fallback sequence when primary profile switch fails
    async fn attempt_fallback_sequence(&mut self, failed_profile: Profile) -> Result<()> {
        warn!(
            "🔄 Attempting fallback sequence for failed profile {:?}",
            failed_profile
        );

        let fallback_profiles = self.get_fallback_sequence(failed_profile);

        for fallback_profile in fallback_profiles {
            info!("🔄 Trying fallback to profile {:?}", fallback_profile);

            if let Ok(()) = self.validate_profile_feasibility(fallback_profile).await {
                match self.perform_profile_switch(fallback_profile).await {
                    Ok(()) => {
                        self.current_profile = fallback_profile;
                        warn!(
                            "✅ Fallback successful: switched to profile {:?}",
                            fallback_profile
                        );
                        return Ok(());
                    }
                    Err(e) => {
                        warn!("❌ Fallback to {:?} failed: {}", fallback_profile, e);
                        continue;
                    }
                }
            }
        }

        Err(anyhow::anyhow!(
            "All fallback profiles failed for target {:?}",
            failed_profile
        ))
    }

    /// Emergency fallback to restore stable state
    async fn emergency_fallback(&mut self, original_profile: Profile) -> Result<()> {
        warn!(
            "🚨 Emergency fallback to restore profile {:?}",
            original_profile
        );

        // Try to restore original profile
        if let Ok(()) = self.perform_profile_switch(original_profile).await {
            self.current_profile = original_profile;
            warn!(
                "✅ Emergency fallback successful: restored profile {:?}",
                original_profile
            );
            return Ok(());
        }

        // If that fails, try Low profile as last resort
        warn!("🚨 Original profile restoration failed, attempting Low profile");
        match self.perform_profile_switch(Profile::Low).await {
            Ok(()) => {
                self.current_profile = Profile::Low;
                warn!("✅ Emergency fallback to Low profile successful");
                Ok(())
            }
            Err(e) => {
                error!("💥 Complete system fallback failure: {}", e);
                Err(anyhow::anyhow!("Emergency fallback failed: {}", e))
            }
        }
    }

    /// Verify profile stability after switching
    async fn verify_profile_stability(&self, profile: Profile) -> Result<()> {
        info!("🔍 Verifying stability of profile {:?}", profile);

        if std::env::var("OBS_SKIP_PROFILE_VALIDATION").is_ok() {
            debug!("Skipping profile stability verification via OBS_SKIP_PROFILE_VALIDATION");
            return Ok(());
        }

        // Wait for system to stabilize
        tokio::time::sleep(std::time::Duration::from_millis(1000)).await;

        // Check resource usage
        let metrics = self.get_resource_metrics().await?;
        let requirements = self.get_profile_resource_requirements(profile);

        if metrics.cpu_usage_percent > 95.0 {
            return Err(anyhow::anyhow!(
                "CPU usage too high: {:.1}%",
                metrics.cpu_usage_percent
            ));
        }

        if metrics.memory_usage_mb > (requirements.memory_mb as f64 * 1.2) as u64 {
            return Err(anyhow::anyhow!(
                "Memory usage exceeds expected: {}MB > {}MB",
                metrics.memory_usage_mb,
                requirements.memory_mb
            ));
        }

        // Test component functionality
        if let Err(e) = self.test_component_functionality().await {
            return Err(anyhow::anyhow!(
                "Component functionality test failed: {}",
                e
            ));
        }

        info!("✅ Profile {:?} stability verified", profile);
        Ok(())
    }

    /// Test basic functionality of all components
    async fn test_component_functionality(&self) -> Result<()> {
        let registry = self.component_registry.read().await;

        // Test VAD
        {
            let vad_arc = registry.vad();
            let mut vad = vad_arc.write().await;
            let test_audio = vec![0.1f32; 1024];
            vad.detect(&test_audio)?;
        }

        // Test feature extractor
        {
            let extractor_arc = registry.feature_extractor();
            let mut extractor = extractor_arc.write().await;
            let test_audio = vec![0.1f32; 16000];
            extractor.extract_mel_spectrogram(&test_audio)?;
        }

        Ok(())
    }

    /// Get fallback sequence for a given profile
    fn get_fallback_sequence(&self, failed_profile: Profile) -> Vec<Profile> {
        match failed_profile {
            Profile::High => vec![Profile::Medium, Profile::Low],
            Profile::Medium => vec![Profile::Low],
            Profile::Low => vec![], // No fallback for Low profile
        }
    }

    /// Get resource requirements for a profile
    fn get_profile_resource_requirements(&self, profile: Profile) -> ResourceRequirements {
        match profile {
            Profile::Low => ResourceRequirements {
                memory_mb: 1200,
                gpu_memory_mb: 0,
                cpu_cores: 2,
            },
            Profile::Medium => ResourceRequirements {
                memory_mb: 3500,
                gpu_memory_mb: 2000,
                cpu_cores: 2,
            },
            Profile::High => ResourceRequirements {
                memory_mb: 8000,
                gpu_memory_mb: 8000,
                cpu_cores: 6,
            },
        }
    }

    /// Get required model paths for a profile
    fn get_required_model_paths(&self, profile: Profile) -> Vec<String> {
        match profile {
            Profile::Low => vec![
                "models/webrtc-vad.bin".to_string(),
                "models/whisper-tiny-int8.onnx".to_string(),
                "models/marian-core-pairs-int8.onnx".to_string(),
            ],
            Profile::Medium => vec![
                "models/ten-vad.onnx".to_string(),
                "models/whisper-small-fp16.onnx".to_string(),
                "models/m2m-100-418m-fp16.onnx".to_string(),
            ],
            Profile::High => vec![
                "models/silero-vad-fp16.onnx".to_string(),
                "models/parakeet-tdt-0.6b-fp16.onnx".to_string(),
                "models/nllb-200-1.3b-int8.onnx".to_string(),
            ],
        }
    }

    /// Start resource monitoring
    async fn start_monitoring(&self) -> Result<()> {
        let monitor = self.resource_monitor.read().await;
        monitor.start_monitoring().await?;
        info!("📊 Started resource monitoring");
        Ok(())
    }

    /// Start adaptive profile management
    async fn start_adaptive_management(&self) -> Result<()> {
        let resource_monitor = Arc::clone(&self.resource_monitor);
        let switching_flag = Arc::clone(&self.switching_in_progress);
        let check_interval = self.config.switch_check_interval;

        // Clone data needed for the spawned task
        let current_profile = self.current_profile;

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(check_interval);
            let mut current_profile = current_profile;

            loop {
                interval.tick().await;

                // Skip if switching is in progress
                {
                    let switching = switching_flag.read().await;
                    if *switching {
                        continue;
                    }
                }

                // Check if profile should change
                let monitor = resource_monitor.read().await;

                if monitor.should_upgrade().await {
                    let target_profile = current_profile.upgrade();
                    if target_profile != current_profile {
                        info!("💡 Recommending upgrade to {:?}", target_profile);
                        // Note: In a real implementation, this would send a message
                        // to the main manager to perform the switch
                        current_profile = target_profile;
                    }
                } else if monitor.should_downgrade().await {
                    let target_profile = current_profile.downgrade();
                    if target_profile != current_profile {
                        warn!("⚠️ Recommending downgrade to {:?}", target_profile);
                        current_profile = target_profile;
                    }
                }
            }
        });

        info!("🤖 Started adaptive profile management");
        Ok(())
    }

    /// Perform the actual profile switch
    async fn perform_profile_switch(&self, target_profile: Profile) -> Result<()> {
        info!("🚀 Starting profile switch to {:?}", target_profile);

        // 1. Validate target profile can be supported
        let hardware = ProfileDetector::scan_hardware()?;
        if !ProfileDetector::validate_profile_support(target_profile, &hardware) {
            return Err(anyhow::anyhow!(
                "Hardware does not support profile {:?}",
                target_profile
            ));
        }
        info!(
            "✅ Hardware validation passed for profile {:?}",
            target_profile
        );

        // 2. Benchmark if enabled
        if self.config.benchmark_on_switch {
            info!("🔍 Running benchmark for profile {:?}", target_profile);
            let benchmark_result = ProfileBenchmark::validate_profile_performance(target_profile)?;
            if !benchmark_result.can_meet_targets {
                warn!(
                    "⚠️ Benchmark indicates profile {:?} may not meet performance targets",
                    target_profile
                );
                if self.config.fallback_on_failure {
                    return Err(anyhow::anyhow!(
                        "Profile {:?} failed performance validation",
                        target_profile
                    ));
                }
            } else {
                info!(
                    "✅ Benchmark validation passed for profile {:?}",
                    target_profile
                );
            }
        }

        // 3. Download/prepare models if needed (with rollback capability)
        let model_preparation_start = std::time::Instant::now();
        {
            let mut model_manager = self.model_manager.write().await;
            model_manager
                .prepare_for_profile(target_profile)
                .await
                .map_err(|e| {
                    anyhow::anyhow!(
                        "Model preparation failed for profile {:?}: {}",
                        target_profile,
                        e
                    )
                })?;
        }
        info!(
            "✅ Model preparation completed in {:.2}s",
            model_preparation_start.elapsed().as_secs_f32()
        );

        // 4. Switch components gracefully (with timeout and rollback)
        let component_switch_start = std::time::Instant::now();
        let switch_result = {
            let mut registry = self.component_registry.write().await;

            // Set a timeout for component switching
            let switch_future = registry.switch_profile(target_profile);
            match tokio::time::timeout(std::time::Duration::from_secs(30), switch_future).await {
                Ok(result) => result,
                Err(_) => {
                    error!("⏰ Component switch timed out after 30 seconds");
                    return Err(anyhow::anyhow!(
                        "Component switch timeout for profile {:?}",
                        target_profile
                    ));
                }
            }
        };

        match switch_result {
            Ok(()) => {
                info!(
                    "✅ Component switch completed in {:.2}s",
                    component_switch_start.elapsed().as_secs_f32()
                );
            }
            Err(e) => {
                error!("❌ Component switch failed: {}", e);
                // Attempt to restore previous state would happen here in production
                return Err(anyhow::anyhow!(
                    "Component switch failed for profile {:?}: {}",
                    target_profile,
                    e
                ));
            }
        }

        // 5. Update resource monitor
        {
            let mut monitor = self.resource_monitor.write().await;
            monitor.set_current_profile(target_profile);
        }
        info!(
            "✅ Resource monitor updated for profile {:?}",
            target_profile
        );

        // 6. Verify system stability after switch
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        let metrics = self.get_resource_metrics().await?;
        if metrics.cpu_usage_percent > 95.0 || metrics.memory_usage_percent > 95.0 {
            warn!(
                "⚠️ High resource usage detected after profile switch: CPU {:.1}%, Memory {:.1}%",
                metrics.cpu_usage_percent, metrics.memory_usage_percent
            );
        }

        info!(
            "🎉 Profile switch to {:?} completed successfully",
            target_profile
        );
        Ok(())
    }

    /// Get current resource metrics
    pub async fn get_resource_metrics(&self) -> Result<crate::profile::monitor::ResourceMetrics> {
        let monitor = self.resource_monitor.read().await;
        Ok(monitor.get_current_metrics().await)
    }

    /// Get performance trend analysis
    pub async fn get_performance_trend(&self) -> Result<crate::profile::monitor::PerformanceTrend> {
        let monitor = self.resource_monitor.read().await;
        Ok(monitor.get_performance_trend().await)
    }

    /// Get component registry for processing
    pub fn get_component_registry(&self) -> Arc<RwLock<ComponentRegistry>> {
        Arc::clone(&self.component_registry)
    }

    /// Check if switching is in progress
    pub async fn is_switching(&self) -> bool {
        *self.switching_in_progress.read().await
    }

    /// Force profile validation and potential switch
    pub async fn validate_and_optimize(&mut self) -> Result<()> {
        let monitor = self.resource_monitor.read().await;

        if monitor.should_upgrade().await {
            let target = self.current_profile.upgrade();
            drop(monitor); // Release read lock before switching
            self.switch_profile(target).await?;
        } else if monitor.should_downgrade().await {
            let target = self.current_profile.downgrade();
            drop(monitor); // Release read lock before switching
            self.switch_profile(target).await?;
        }

        Ok(())
    }
}

impl ModelManager {
    /// Create new model manager for profile
    pub async fn new(profile: Profile) -> Result<Self> {
        Ok(Self {
            current_profile: profile,
            download_progress: Arc::new(RwLock::new(DownloadProgress::default())),
        })
    }

    /// Prepare models for the given profile
    pub async fn prepare_for_profile(&mut self, profile: Profile) -> Result<()> {
        if profile == self.current_profile {
            return Ok(());
        }

        info!("📦 Preparing models for profile {:?}", profile);

        // Start download progress tracking
        {
            let mut progress = self.download_progress.write().await;
            *progress = DownloadProgress {
                current_model: Some(format!("profile-{}", profile.as_str())),
                progress_percent: 0.0,
                total_size_mb: self.get_profile_model_size(profile),
                downloaded_mb: 0,
                download_speed_mbps: 0.0,
                eta_seconds: None,
            };
        }

        // Simulate model preparation (in real implementation, this would download/load models)
        self.download_models_for_profile(profile).await?;
        self.optimize_models_for_hardware(profile).await?;
        self.validate_models(profile).await?;

        self.current_profile = profile;
        info!("✅ Models prepared for profile {:?}", profile);
        Ok(())
    }

    /// Get model size for profile
    fn get_profile_model_size(&self, profile: Profile) -> u64 {
        match profile {
            Profile::Low => 1200,    // 1.2GB
            Profile::Medium => 3500, // 3.5GB
            Profile::High => 8000,   // 8GB
        }
    }

    /// Download models for profile
    async fn download_models_for_profile(&self, profile: Profile) -> Result<()> {
        let model_list = self.get_required_models(profile);
        let total_models = model_list.len();

        for (i, model_name) in model_list.iter().enumerate() {
            info!("⬇️ Downloading model: {}", model_name);

            // Update progress
            {
                let mut progress = self.download_progress.write().await;
                progress.current_model = Some(model_name.clone());
                progress.progress_percent = (i as f32 / total_models as f32) * 100.0;
            }

            // Simulate download
            self.download_single_model(model_name).await?;
        }

        // Mark download complete
        {
            let mut progress = self.download_progress.write().await;
            progress.progress_percent = 100.0;
            progress.current_model = None;
        }

        Ok(())
    }

    /// Get required models for profile
    fn get_required_models(&self, profile: Profile) -> Vec<String> {
        match profile {
            Profile::Low => vec![
                "webrtc-vad.bin".to_string(),
                "whisper-tiny-int8.onnx".to_string(),
                "marian-core-pairs-int8.onnx".to_string(),
                "fasttext-lite.bin".to_string(),
            ],
            Profile::Medium => vec![
                "ten-vad.onnx".to_string(),
                "whisper-small-fp16.onnx".to_string(),
                "m2m-100-418m-fp16.onnx".to_string(),
                "fasttext-full.bin".to_string(),
            ],
            Profile::High => vec![
                "silero-vad-fp16.onnx".to_string(),
                "parakeet-tdt-0.6b-fp16.onnx".to_string(),
                "nllb-200-1.3b-int8.onnx".to_string(),
                "fasttext-full.bin".to_string(),
            ],
        }
    }

    /// Download single model (placeholder)
    async fn download_single_model(&self, model_name: &str) -> Result<()> {
        // Simulate download time
        tokio::time::sleep(Duration::from_millis(100)).await;
        info!("✅ Downloaded: {}", model_name);
        Ok(())
    }

    /// Optimize models for current hardware
    async fn optimize_models_for_hardware(&self, _profile: Profile) -> Result<()> {
        info!("⚙️ Optimizing models for hardware");
        // Placeholder for model optimization (quantization, TensorRT, etc.)
        tokio::time::sleep(Duration::from_millis(50)).await;
        Ok(())
    }

    /// Validate models work correctly
    async fn validate_models(&self, _profile: Profile) -> Result<()> {
        info!("✓ Validating models");
        // Placeholder for model validation
        tokio::time::sleep(Duration::from_millis(50)).await;
        Ok(())
    }

    /// Get download progress
    pub async fn get_download_progress(&self) -> DownloadProgress {
        self.download_progress.read().await.clone()
    }
}

impl Default for DownloadProgress {
    fn default() -> Self {
        Self {
            current_model: None,
            progress_percent: 0.0,
            total_size_mb: 0,
            downloaded_mb: 0,
            download_speed_mbps: 0.0,
            eta_seconds: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_profile_manager_initialization() {
        std::env::set_var("OBS_SKIP_PROFILE_VALIDATION", "1");
        let manager = ProfileManager::initialize_for_profile(Profile::Low)
            .await
            .unwrap();
        assert_eq!(manager.current_profile(), Profile::Low);
    }

    #[tokio::test]
    async fn test_model_manager() {
        std::env::set_var("OBS_SKIP_PROFILE_VALIDATION", "1");
        let mut model_manager = ModelManager::new(Profile::Low).await.unwrap();
        assert_eq!(model_manager.current_profile, Profile::Low);

        model_manager
            .prepare_for_profile(Profile::Medium)
            .await
            .unwrap();
        assert_eq!(model_manager.current_profile, Profile::Medium);
    }

    #[tokio::test]
    async fn test_profile_switching() {
        std::env::set_var("OBS_SKIP_PROFILE_VALIDATION", "1");
        let mut manager = ProfileManager::initialize_for_profile(Profile::Low)
            .await
            .unwrap();

        // Test switch to medium
        manager.switch_profile(Profile::Medium).await.unwrap();
        assert_eq!(manager.current_profile(), Profile::Medium);

        // Test switch to high
        manager.switch_profile(Profile::High).await.unwrap();
        assert_eq!(manager.current_profile(), Profile::High);

        // Test downgrade
        manager.switch_profile(Profile::Low).await.unwrap();
        assert_eq!(manager.current_profile(), Profile::Low);
    }
}
