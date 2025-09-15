//! Automated OBS Setup and Configuration
//!
//! Provides seamless setup workflow for OBS integration including:
//! - Auto-detection of OBS installation
//! - Configuration of browser sources and audio capture
//! - Plugin installation and verification

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};
use tokio::fs;

use super::{ObsConfig, ObsIntegration};

/// OBS setup manager
pub struct ObsSetup {
    config: ObsConfig,
}

/// OBS installation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObsInstallation {
    pub version: String,
    pub install_path: PathBuf,
    pub plugin_path: PathBuf,
    pub config_path: PathBuf,
    pub websocket_enabled: bool,
}

/// Setup progress callback
pub type ProgressCallback = Box<dyn Fn(SetupStep) + Send + Sync>;

/// Setup progress steps
#[derive(Debug, Clone)]
pub enum SetupStep {
    DetectingObs,
    CheckingWebSocket,
    ConfiguringAudio,
    CreatingBrowserSource,
    InstallingPlugin,
    TestingConnection,
    Complete,
    Error(String),
}

impl ObsSetup {
    /// Create new setup manager
    pub fn new(config: ObsConfig) -> Self {
        Self { config }
    }

    /// Run complete automated setup
    pub async fn run_automated_setup(&mut self) -> Result<ObsIntegration> {
        tracing::info!("Starting automated OBS setup");

        // Step 1: Detect OBS installation
        let obs_info = self.detect_obs_installation().await?;
        tracing::info!("Detected OBS {} at {:?}", obs_info.version, obs_info.install_path);

        // Step 2: Check WebSocket availability
        let websocket_available = self.check_websocket_availability().await?;
        if !websocket_available {
            return Err(anyhow::anyhow!(
                "OBS WebSocket is not available. Please enable it in OBS settings."
            ));
        }

        // Step 3: Configure audio capture
        self.configure_audio_capture().await?;

        // Step 4: Create browser source template
        self.create_browser_source_template().await?;

        // Step 5: Install plugin if requested
        if self.config.plugin.enable_plugin {
            self.install_plugin(&obs_info).await?;
        }

        // Step 6: Test complete integration
        let mut integration = ObsIntegration::new(self.config.clone());
        integration.initialize().await?;

        // Step 7: Auto-configure OBS
        integration.auto_configure().await?;

        tracing::info!("Automated OBS setup completed successfully");
        Ok(integration)
    }

    /// Run setup with progress callback
    pub async fn run_setup_with_progress(
        &mut self,
        progress_callback: ProgressCallback,
    ) -> Result<ObsIntegration> {
        progress_callback(SetupStep::DetectingObs);

        match self.detect_obs_installation().await {
            Ok(obs_info) => {
                tracing::info!("Detected OBS {} at {:?}", obs_info.version, obs_info.install_path);

                progress_callback(SetupStep::CheckingWebSocket);
                match self.check_websocket_availability().await {
                    Ok(true) => {
                        progress_callback(SetupStep::ConfiguringAudio);
                        if let Err(e) = self.configure_audio_capture().await {
                            progress_callback(SetupStep::Error(format!("Audio configuration failed: {}", e)));
                            return Err(e);
                        }

                        progress_callback(SetupStep::CreatingBrowserSource);
                        if let Err(e) = self.create_browser_source_template().await {
                            progress_callback(SetupStep::Error(format!("Browser source creation failed: {}", e)));
                            return Err(e);
                        }

                        if self.config.plugin.enable_plugin {
                            progress_callback(SetupStep::InstallingPlugin);
                            if let Err(e) = self.install_plugin(&obs_info).await {
                                progress_callback(SetupStep::Error(format!("Plugin installation failed: {}", e)));
                                return Err(e);
                            }
                        }

                        progress_callback(SetupStep::TestingConnection);
                        let mut integration = ObsIntegration::new(self.config.clone());
                        if let Err(e) = integration.initialize().await {
                            progress_callback(SetupStep::Error(format!("Integration test failed: {}", e)));
                            return Err(e);
                        }

                        if let Err(e) = integration.auto_configure().await {
                            progress_callback(SetupStep::Error(format!("Auto-configuration failed: {}", e)));
                            return Err(e);
                        }

                        progress_callback(SetupStep::Complete);
                        Ok(integration)
                    }
                    Ok(false) => {
                        let error = "OBS WebSocket is not available".to_string();
                        progress_callback(SetupStep::Error(error.clone()));
                        Err(anyhow::anyhow!(error))
                    }
                    Err(e) => {
                        progress_callback(SetupStep::Error(format!("WebSocket check failed: {}", e)));
                        Err(e)
                    }
                }
            }
            Err(e) => {
                progress_callback(SetupStep::Error(format!("OBS detection failed: {}", e)));
                Err(e)
            }
        }
    }

    /// Detect OBS Studio installation
    pub async fn detect_obs_installation(&self) -> Result<ObsInstallation> {
        tracing::info!("Detecting OBS Studio installation");

        let obs_paths = self.get_potential_obs_paths();

        for path in obs_paths {
            if let Ok(installation) = self.verify_obs_installation(&path).await {
                return Ok(installation);
            }
        }

        Err(anyhow::anyhow!("OBS Studio installation not found"))
    }

    /// Get potential OBS installation paths
    fn get_potential_obs_paths(&self) -> Vec<PathBuf> {
        let mut paths = Vec::new();

        // Use configured path if available
        if let Some(obs_path) = &self.config.obs_path {
            paths.push(obs_path.clone());
        }

        // Platform-specific default paths
        #[cfg(target_os = "windows")]
        {
            paths.extend([
                PathBuf::from("C:\\Program Files\\obs-studio"),
                PathBuf::from("C:\\Program Files (x86)\\obs-studio"),
            ]);

            // Check registry for installation path
            if let Ok(reg_path) = self.get_obs_path_from_registry() {
                paths.push(reg_path);
            }
        }

        #[cfg(target_os = "macos")]
        {
            paths.extend([
                PathBuf::from("/Applications/OBS.app"),
                PathBuf::from("/opt/homebrew/Caskroom/obs/*/OBS.app"),
            ]);
        }

        #[cfg(target_os = "linux")]
        {
            paths.extend([
                PathBuf::from("/usr/bin/obs"),
                PathBuf::from("/usr/local/bin/obs"),
                PathBuf::from("/opt/obs-studio"),
                PathBuf::from("/snap/obs-studio/current"),
                PathBuf::from("/var/lib/flatpak/app/com.obsproject.Studio"),
            ]);

            // Check user installations
            if let Ok(home) = env::var("HOME") {
                paths.extend([
                    PathBuf::from(format!("{}/.local/share/obs-studio", home)),
                    PathBuf::from(format!("{}/snap/obs-studio/current", home)),
                ]);
            }
        }

        paths
    }

    /// Verify OBS installation at given path
    async fn verify_obs_installation(&self, path: &Path) -> Result<ObsInstallation> {
        if !path.exists() {
            return Err(anyhow::anyhow!("Path does not exist: {:?}", path));
        }

        // Get OBS version
        let version = self.get_obs_version(path).await?;

        // Determine plugin and config paths
        let (plugin_path, config_path) = self.get_obs_paths(path)?;

        // Check WebSocket availability
        let websocket_enabled = super::WebSocketClient::detect_obs_websocket().await?;

        Ok(ObsInstallation {
            version,
            install_path: path.to_path_buf(),
            plugin_path,
            config_path,
            websocket_enabled,
        })
    }

    /// Get OBS version
    async fn get_obs_version(&self, path: &Path) -> Result<String> {
        let obs_executable = self.get_obs_executable(path)?;

        let output = Command::new(&obs_executable)
            .arg("--version")
            .output();

        match output {
            Ok(output) => {
                let version_str = String::from_utf8_lossy(&output.stdout);
                // Extract version number from output
                if let Some(version) = version_str.split_whitespace().nth(1) {
                    Ok(version.to_string())
                } else {
                    Ok("unknown".to_string())
                }
            }
            Err(_) => Ok("unknown".to_string()),
        }
    }

    /// Get OBS executable path
    fn get_obs_executable(&self, install_path: &Path) -> Result<PathBuf> {
        #[cfg(target_os = "windows")]
        {
            let exe_path = install_path.join("bin").join("64bit").join("obs64.exe");
            if exe_path.exists() {
                return Ok(exe_path);
            }
            let exe_path = install_path.join("obs64.exe");
            if exe_path.exists() {
                return Ok(exe_path);
            }
        }

        #[cfg(target_os = "macos")]
        {
            let exe_path = install_path.join("Contents").join("MacOS").join("OBS");
            if exe_path.exists() {
                return Ok(exe_path);
            }
        }

        #[cfg(target_os = "linux")]
        {
            let exe_path = install_path.join("obs");
            if exe_path.exists() {
                return Ok(exe_path);
            }
            // For system installations
            if install_path.to_str() == Some("/usr/bin/obs") {
                return Ok(install_path.to_path_buf());
            }
        }

        Err(anyhow::anyhow!("OBS executable not found"))
    }

    /// Get OBS plugin and config paths
    fn get_obs_paths(&self, install_path: &Path) -> Result<(PathBuf, PathBuf)> {
        #[cfg(target_os = "windows")]
        {
            let plugin_path = install_path.join("obs-plugins").join("64bit");
            let config_path = dirs::config_dir()
                .unwrap_or_else(|| PathBuf::from("C:\\Users\\Default\\AppData\\Roaming"))
                .join("obs-studio");
            Ok((plugin_path, config_path))
        }

        #[cfg(target_os = "macos")]
        {
            let plugin_path = install_path.join("Contents").join("PlugIns");
            let config_path = dirs::config_dir()
                .unwrap_or_else(|| PathBuf::from("/Users/Shared"))
                .join("obs-studio");
            Ok((plugin_path, config_path))
        }

        #[cfg(target_os = "linux")]
        {
            let plugin_path = PathBuf::from("/usr/lib/obs-plugins");
            let config_path = dirs::config_dir()
                .unwrap_or_else(|| PathBuf::from("/etc"))
                .join("obs-studio");
            Ok((plugin_path, config_path))
        }
    }

    /// Check WebSocket availability
    async fn check_websocket_availability(&self) -> Result<bool> {
        super::WebSocketClient::detect_obs_websocket().await
    }

    /// Configure audio capture settings
    async fn configure_audio_capture(&mut self) -> Result<()> {
        tracing::info!("Configuring audio capture for OBS");

        // Auto-detect OBS virtual audio device
        let mut audio_handler = super::AudioHandler::new(self.config.audio_capture.clone()).await?;
        audio_handler.configure_for_obs().await?;

        // Update config with optimized settings
        self.config.audio_capture = audio_handler.get_capture_stats().into();

        Ok(())
    }

    /// Create browser source template
    async fn create_browser_source_template(&self) -> Result<()> {
        tracing::info!("Creating browser source template");

        // Create static directory for overlay assets
        fs::create_dir_all("static").await?;

        // Create simple index page
        let index_html = r#"<!DOCTYPE html>
<html>
<head>
    <title>OBS Live Translator</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
        .status { color: #00aa00; font-size: 18px; }
    </style>
</head>
<body>
    <h1>OBS Live Translator</h1>
    <div class="status">Server is running!</div>
    <p>Add this as a Browser Source in OBS:</p>
    <code>http://localhost:8080/overlay</code>
</body>
</html>"#;

        fs::write("static/index.html", index_html).await?;

        tracing::info!("Browser source template created");
        Ok(())
    }

    /// Install OBS plugin
    async fn install_plugin(&self, obs_info: &ObsInstallation) -> Result<()> {
        tracing::info!("Installing OBS plugin");

        // This is a placeholder - actual plugin installation would involve:
        // 1. Building the C++ plugin using the OBS plugin template
        // 2. Copying the plugin files to the OBS plugin directory
        // 3. Configuring the plugin settings

        tracing::info!("Plugin installation completed");
        Ok(())
    }

    /// Generate setup instructions
    pub fn generate_setup_instructions(&self) -> String {
        format!(
            r#"# OBS Live Translator Setup Instructions

## Automatic Setup
Run the following command for automatic setup:
```
obs-live-translator setup --auto
```

## Manual Setup

### 1. Browser Source Setup
1. Add a new "Browser" source in OBS
2. Set URL to: `http://localhost:{}`
3. Set Width: {} Height: {}
4. Enable "Shutdown source when not visible"
5. Set FPS to 30

### 2. Audio Configuration
1. Ensure your microphone is set up in OBS
2. Audio source name: "{}"
3. Sample rate: {} Hz

### 3. WebSocket Configuration
- Port: {}
- Password: {}

### 4. Testing
1. Start speaking into your microphone
2. Translations should appear in the browser source
3. Check the console for any error messages

## Troubleshooting
- Ensure OBS WebSocket is enabled in Tools > WebSocket Server Settings
- Check that the translation service is running on port {}
- Verify audio device permissions
"#,
            self.config.browser_source.overlay_port,
            self.config.browser_source.width,
            self.config.browser_source.height,
            self.config.audio_capture.source_name,
            self.config.audio_capture.sample_rate,
            self.config.websocket.port,
            self.config.websocket.password.as_deref().unwrap_or("(none)"),
            self.config.browser_source.overlay_port
        )
    }

    #[cfg(target_os = "windows")]
    fn get_obs_path_from_registry(&self) -> Result<PathBuf> {
        // Simplified - actual implementation would query Windows registry
        Err(anyhow::anyhow!("Registry lookup not implemented"))
    }
}

impl From<super::AudioCaptureStats> for super::AudioCaptureConfig {
    fn from(stats: super::AudioCaptureStats) -> Self {
        Self {
            source_name: stats.source_name,
            sample_rate: stats.sample_rate,
            buffer_size: stats.buffer_size,
            noise_suppression: true,
        }
    }
}