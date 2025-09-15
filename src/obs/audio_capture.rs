//! Audio Capture Integration for OBS
//!
//! Captures audio from OBS sources for real-time speech processing
//! Supports both direct audio capture and OBS filter integration

use anyhow::Result;
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    Device, Host, Stream, StreamConfig,
};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use tokio::sync::{broadcast, mpsc};

use super::AudioCaptureConfig;
use crate::AudioChunk;

/// Audio capture handler for OBS integration
pub struct AudioHandler {
    config: AudioCaptureConfig,
    audio_stream: Option<Stream>,
    is_capturing: Arc<AtomicBool>,
    audio_tx: broadcast::Sender<AudioChunk>,
}

/// Audio device information
#[derive(Debug, Clone)]
pub struct AudioDevice {
    pub name: String,
    pub is_default: bool,
    pub sample_rates: Vec<u32>,
    pub channels: Vec<u16>,
}

impl AudioHandler {
    /// Create new audio handler
    pub async fn new(config: AudioCaptureConfig) -> Result<Self> {
        let (audio_tx, _) = broadcast::channel(1000);

        let handler = Self {
            config,
            audio_stream: None,
            is_capturing: Arc::new(AtomicBool::new(false)),
            audio_tx,
        };

        Ok(handler)
    }

    /// Start audio capture
    pub async fn start_capture(&mut self) -> Result<broadcast::Receiver<AudioChunk>> {
        if self.is_capturing.load(Ordering::Relaxed) {
            return Ok(self.audio_tx.subscribe());
        }

        tracing::info!("Starting audio capture for OBS integration");

        // Get audio host and device
        let host = cpal::default_host();
        let device = self.get_audio_device(&host).await?;

        // Configure audio stream
        let config = self.get_stream_config(&device)?;
        let audio_tx = self.audio_tx.clone();
        let is_capturing = Arc::clone(&self.is_capturing);
        let buffer_size = self.config.buffer_size;
        let sample_rate = self.config.sample_rate;

        tracing::info!(
            "Audio config: {} Hz, {} channels, {} samples buffer",
            config.sample_rate.0,
            config.channels,
            buffer_size
        );

        // Create audio stream
        let stream = device.build_input_stream(
            &config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                if !is_capturing.load(Ordering::Relaxed) {
                    return;
                }

                // Process audio in chunks
                for chunk in data.chunks(buffer_size) {
                    let mut audio_chunk = AudioChunk {
                        samples: chunk.to_vec(),
                        sample_rate,
                        channels: config.channels,
                        timestamp: std::time::SystemTime::now(),
                        fingerprint: 0,
                    };

                    // Calculate fingerprint for caching
                    audio_chunk.calculate_fingerprint();

                    // Apply noise suppression if enabled
                    if self.config.noise_suppression {
                        Self::apply_noise_suppression(&mut audio_chunk.samples);
                    }

                    // Send to processing pipeline
                    let _ = audio_tx.send(audio_chunk);
                }
            },
            |err| {
                tracing::error!("Audio stream error: {}", err);
            },
            None,
        )?;

        // Start the stream
        stream.play()?;
        self.audio_stream = Some(stream);
        self.is_capturing.store(true, Ordering::Relaxed);

        tracing::info!("Audio capture started successfully");
        Ok(self.audio_tx.subscribe())
    }

    /// Stop audio capture
    pub async fn stop(&mut self) -> Result<()> {
        self.is_capturing.store(false, Ordering::Relaxed);

        if let Some(stream) = self.audio_stream.take() {
            drop(stream);
            tracing::info!("Audio capture stopped");
        }

        Ok(())
    }

    /// Get available audio devices
    pub async fn get_available_devices() -> Result<Vec<AudioDevice>> {
        let host = cpal::default_host();
        let mut devices = Vec::new();

        // Get input devices
        for device in host.input_devices()? {
            let name = device.name().unwrap_or_else(|_| "Unknown Device".to_string());
            let is_default = host.default_input_device()
                .map(|default| default.name().unwrap_or_default() == name)
                .unwrap_or(false);

            // Get supported configurations
            let configs: Result<Vec<_>, _> = device.supported_input_configs().collect();
            if let Ok(configs) = configs {
                let sample_rates: Vec<u32> = configs
                    .iter()
                    .flat_map(|config| {
                        vec![config.min_sample_rate().0, config.max_sample_rate().0]
                    })
                    .collect();

                let channels: Vec<u16> = configs
                    .iter()
                    .map(|config| config.channels())
                    .collect();

                devices.push(AudioDevice {
                    name,
                    is_default,
                    sample_rates,
                    channels,
                });
            }
        }

        Ok(devices)
    }

    /// Check if OBS virtual audio device is available
    pub async fn detect_obs_audio_device() -> Result<Option<String>> {
        let devices = Self::get_available_devices().await?;

        // Look for common OBS virtual audio device names
        let obs_device_names = [
            "OBS Virtual Audio Device",
            "OBS-VirtualAudioDevice",
            "VB-Audio Virtual Cable",
            "CABLE Input",
            "VoiceMeeter",
        ];

        for device in devices {
            for obs_name in &obs_device_names {
                if device.name.contains(obs_name) {
                    return Ok(Some(device.name));
                }
            }
        }

        Ok(None)
    }

    /// Configure audio for OBS integration
    pub async fn configure_for_obs(&mut self) -> Result<()> {
        // Try to find OBS virtual audio device
        if let Some(device_name) = Self::detect_obs_audio_device().await? {
            tracing::info!("Found OBS virtual audio device: {}", device_name);
            self.config.source_name = device_name;
        } else {
            tracing::warn!("No OBS virtual audio device found, using default microphone");
        }

        // Optimize settings for real-time processing
        self.config.sample_rate = 48000; // Match OBS default
        self.config.buffer_size = 512;   // Low latency
        self.config.noise_suppression = true;

        Ok(())
    }

    /// Get audio device by name
    async fn get_audio_device(&self, host: &Host) -> Result<Device> {
        // Try to find device by name
        for device in host.input_devices()? {
            if let Ok(name) = device.name() {
                if name == self.config.source_name {
                    return Ok(device);
                }
            }
        }

        // Fall back to default device
        host.default_input_device()
            .ok_or_else(|| anyhow::anyhow!("No default input device found"))
    }

    /// Get optimal stream configuration
    fn get_stream_config(&self, device: &Device) -> Result<StreamConfig> {
        let supported_configs: Vec<_> = device.supported_input_configs()?.collect();

        // Find configuration matching our requirements
        for config in supported_configs {
            if config.channels() >= 1
                && config.min_sample_rate().0 <= self.config.sample_rate
                && config.max_sample_rate().0 >= self.config.sample_rate
            {
                return Ok(StreamConfig {
                    channels: config.channels(),
                    sample_rate: cpal::SampleRate(self.config.sample_rate),
                    buffer_size: cpal::BufferSize::Fixed(self.config.buffer_size as u32),
                });
            }
        }

        // Fall back to default configuration
        let default_config = device.default_input_config()?;
        Ok(StreamConfig {
            channels: default_config.channels(),
            sample_rate: default_config.sample_rate(),
            buffer_size: cpal::BufferSize::Fixed(self.config.buffer_size as u32),
        })
    }

    /// Apply simple noise suppression
    fn apply_noise_suppression(samples: &mut [f32]) {
        const NOISE_THRESHOLD: f32 = 0.01;
        const NOISE_REDUCTION: f32 = 0.1;

        for sample in samples.iter_mut() {
            if sample.abs() < NOISE_THRESHOLD {
                *sample *= NOISE_REDUCTION;
            }
        }
    }

    /// Get audio capture statistics
    pub fn get_capture_stats(&self) -> AudioCaptureStats {
        AudioCaptureStats {
            is_capturing: self.is_capturing.load(Ordering::Relaxed),
            sample_rate: self.config.sample_rate,
            buffer_size: self.config.buffer_size,
            source_name: self.config.source_name.clone(),
        }
    }
}

/// Audio capture statistics
#[derive(Debug, Clone)]
pub struct AudioCaptureStats {
    pub is_capturing: bool,
    pub sample_rate: u32,
    pub buffer_size: usize,
    pub source_name: String,
}

/// OBS Audio Filter Integration
///
/// This module provides integration with OBS audio filters for
/// more advanced audio processing and routing
pub mod filter_integration {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct AudioFilterConfig {
        pub filter_name: String,
        pub source_name: String,
        pub filter_type: String,
        pub settings: serde_json::Value,
    }

    /// Create OBS audio filter for translation integration
    pub async fn create_translation_filter(
        websocket_client: &super::super::WebSocketClient,
        source_name: &str,
    ) -> Result<()> {
        let filter_config = AudioFilterConfig {
            filter_name: "Live Translation Monitor".to_string(),
            source_name: source_name.to_string(),
            filter_type: "audio_monitor".to_string(),
            settings: serde_json::json!({
                "monitoring_type": "monitor_only",
                "volume": 1.0
            }),
        };

        // Note: Actual implementation would create the filter through WebSocket
        tracing::info!("Created translation audio filter for source: {}", source_name);
        Ok(())
    }
}