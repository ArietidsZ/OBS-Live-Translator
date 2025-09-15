//! OBS Studio Integration Module
//!
//! Provides seamless integration with OBS Studio through multiple methods:
//! - Native C++ plugin for audio capture and processing
//! - Browser source overlay for translation display
//! - WebSocket API for remote control and automation
//! - Auto-configuration and setup utilities

pub mod plugin;
#[path = "browser-source.rs"]
pub mod browser_source;
pub mod websocket;
#[path = "audio-capture.rs"]
pub mod audio_capture;
pub mod setup;

pub use plugin::*;
pub use browser_source::*;
pub use websocket::*;
pub use audio_capture::*;
pub use setup::*;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// OBS integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObsConfig {
    /// OBS installation path
    pub obs_path: Option<PathBuf>,
    /// WebSocket server configuration
    pub websocket: WebSocketConfig,
    /// Browser source configuration
    pub browser_source: BrowserSourceConfig,
    /// Audio capture settings
    pub audio_capture: AudioCaptureConfig,
    /// Plugin installation settings
    pub plugin: PluginConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSocketConfig {
    /// WebSocket server port (default: 4455)
    pub port: u16,
    /// Authentication password (optional)
    pub password: Option<String>,
    /// Enable automatic connection
    pub auto_connect: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrowserSourceConfig {
    /// Local server port for overlay
    pub overlay_port: u16,
    /// Overlay dimensions
    pub width: u32,
    pub height: u32,
    /// Enable hardware acceleration
    pub hardware_acceleration: bool,
    /// Custom CSS styling
    pub custom_css: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioCaptureConfig {
    /// Audio source name in OBS
    pub source_name: String,
    /// Sample rate (Hz)
    pub sample_rate: u32,
    /// Buffer size for processing
    pub buffer_size: usize,
    /// Enable noise suppression
    pub noise_suppression: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginConfig {
    /// Enable native plugin installation
    pub enable_plugin: bool,
    /// Plugin installation path
    pub plugin_path: Option<PathBuf>,
    /// Auto-detect OBS installation
    pub auto_detect: bool,
}

impl Default for ObsConfig {
    fn default() -> Self {
        Self {
            obs_path: None,
            websocket: WebSocketConfig {
                port: 4455,
                password: None,
                auto_connect: true,
            },
            browser_source: BrowserSourceConfig {
                overlay_port: 8080,
                width: 1920,
                height: 1080,
                hardware_acceleration: true,
                custom_css: None,
            },
            audio_capture: AudioCaptureConfig {
                source_name: "Microphone".to_string(),
                sample_rate: 48000,
                buffer_size: 1024,
                noise_suppression: true,
            },
            plugin: PluginConfig {
                enable_plugin: true,
                plugin_path: None,
                auto_detect: true,
            },
        }
    }
}

/// OBS integration manager
pub struct ObsIntegration {
    config: ObsConfig,
    websocket_client: Option<WebSocketClient>,
    browser_server: Option<BrowserServer>,
    audio_handler: Option<AudioHandler>,
}

impl ObsIntegration {
    /// Create new OBS integration instance
    pub fn new(config: ObsConfig) -> Self {
        Self {
            config,
            websocket_client: None,
            browser_server: None,
            audio_handler: None,
        }
    }

    /// Initialize all integration components
    pub async fn initialize(&mut self) -> Result<()> {
        tracing::info!("Initializing OBS integration");

        // Start browser source server
        if self.browser_server.is_none() {
            let server = BrowserServer::new(self.config.browser_source.clone()).await?;
            self.browser_server = Some(server);
        }

        // Connect to OBS WebSocket
        if self.config.websocket.auto_connect && self.websocket_client.is_none() {
            let client = WebSocketClient::new(self.config.websocket.clone()).await?;
            self.websocket_client = Some(client);
        }

        // Initialize audio capture
        if self.audio_handler.is_none() {
            let handler = AudioHandler::new(self.config.audio_capture.clone()).await?;
            self.audio_handler = Some(handler);
        }

        Ok(())
    }

    /// Get browser source URL for OBS
    pub fn get_browser_source_url(&self) -> String {
        format!("http://localhost:{}/overlay", self.config.browser_source.overlay_port)
    }

    /// Check if OBS is running and accessible
    pub async fn check_obs_connection(&self) -> bool {
        if let Some(client) = &self.websocket_client {
            client.is_connected().await
        } else {
            false
        }
    }

    /// Auto-configure OBS with optimal settings
    pub async fn auto_configure(&mut self) -> Result<()> {
        tracing::info!("Auto-configuring OBS integration");

        if let Some(client) = &mut self.websocket_client {
            // Create browser source if it doesn't exist
            client.create_browser_source(
                "Live Translation Overlay",
                &self.get_browser_source_url(),
                self.config.browser_source.width,
                self.config.browser_source.height,
            ).await?;

            // Configure audio capture
            client.setup_audio_capture(&self.config.audio_capture.source_name).await?;

            tracing::info!("OBS auto-configuration completed");
        }

        Ok(())
    }

    /// Start translation overlay
    pub async fn start_overlay(&mut self) -> Result<()> {
        if let Some(server) = &mut self.browser_server {
            server.start_translation_overlay().await?;
        }
        Ok(())
    }

    /// Stop all integration components
    pub async fn shutdown(&mut self) -> Result<()> {
        tracing::info!("Shutting down OBS integration");

        if let Some(mut server) = self.browser_server.take() {
            server.shutdown().await?;
        }

        if let Some(mut client) = self.websocket_client.take() {
            client.disconnect().await?;
        }

        if let Some(mut handler) = self.audio_handler.take() {
            handler.stop().await?;
        }

        Ok(())
    }
}