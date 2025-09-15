//! Native OBS Plugin Integration
//!
//! Provides infrastructure for building and integrating a native C++ OBS plugin
//! for optimal audio capture and processing performance

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::{
    path::{Path, PathBuf},
    process::Command,
};
use tokio::fs;

/// OBS plugin manager
pub struct PluginManager {
    plugin_name: String,
    plugin_path: Option<PathBuf>,
    obs_path: Option<PathBuf>,
}

/// Plugin build configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginBuildConfig {
    pub plugin_name: String,
    pub version: String,
    pub author: String,
    pub description: String,
    pub obs_version_requirement: String,
    pub enable_audio_capture: bool,
    pub enable_websocket_integration: bool,
}

impl Default for PluginBuildConfig {
    fn default() -> Self {
        Self {
            plugin_name: "obs-live-translator".to_string(),
            version: "1.0.0".to_string(),
            author: "OBS Live Translator Team".to_string(),
            description: "Real-time speech translation plugin for OBS Studio".to_string(),
            obs_version_requirement: "28.0.0".to_string(),
            enable_audio_capture: true,
            enable_websocket_integration: true,
        }
    }
}

impl PluginManager {
    /// Create new plugin manager
    pub fn new(plugin_name: String) -> Self {
        Self {
            plugin_name,
            plugin_path: None,
            obs_path: None,
        }
    }

    /// Generate OBS plugin template
    pub async fn generate_plugin_template(&self, config: &PluginBuildConfig) -> Result<()> {
        let plugin_dir = format!("obs-plugin-{}", config.plugin_name);
        fs::create_dir_all(&plugin_dir).await?;

        // Generate CMakeLists.txt
        self.generate_cmake_file(&plugin_dir, config).await?;

        // Generate main plugin file
        self.generate_main_plugin_file(&plugin_dir, config).await?;

        // Generate audio capture module
        if config.enable_audio_capture {
            self.generate_audio_capture_module(&plugin_dir, config).await?;
        }

        // Generate buildspec.json for GitHub Actions
        self.generate_buildspec(&plugin_dir, config).await?;

        // Generate GitHub Actions workflow
        self.generate_github_actions(&plugin_dir, config).await?;

        tracing::info!("OBS plugin template generated in: {}", plugin_dir);
        Ok(())
    }

    /// Generate CMakeLists.txt
    async fn generate_cmake_file(&self, plugin_dir: &str, config: &PluginBuildConfig) -> Result<()> {
        let cmake_content = format!(
            r#"cmake_minimum_required(VERSION 3.16...3.25)

project({plugin_name} VERSION {version})

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OBS Studio development package
find_package(libobs REQUIRED)
find_package(obs-frontend-api REQUIRED)

# Plugin source files
set(PLUGIN_SOURCES
    src/plugin-main.cpp
    src/audio-capture.cpp
    src/translation-client.cpp
)

# Create plugin library
add_library(${{CMAKE_PROJECT_NAME}} MODULE ${{PLUGIN_SOURCES}})

# Link OBS libraries
target_link_libraries(${{CMAKE_PROJECT_NAME}}
    OBS::libobs
    OBS::obs-frontend-api
)

# Platform-specific settings
if(WIN32)
    target_compile_definitions(${{CMAKE_PROJECT_NAME}} PRIVATE WIN32_LEAN_AND_MEAN)
endif()

if(APPLE)
    set_target_properties(${{CMAKE_PROJECT_NAME}} PROPERTIES
        BUNDLE TRUE
        BUNDLE_EXTENSION "so"
    )
endif()

# Install plugin
if(WIN32)
    install(TARGETS ${{CMAKE_PROJECT_NAME}}
        DESTINATION obs-plugins/64bit)
else()
    install(TARGETS ${{CMAKE_PROJECT_NAME}}
        DESTINATION lib/obs-plugins)
endif()

# Install data files
install(DIRECTORY data/
    DESTINATION data/obs-plugins/${{CMAKE_PROJECT_NAME}})
"#,
            plugin_name = config.plugin_name,
            version = config.version
        );

        fs::write(format!("{}/CMakeLists.txt", plugin_dir), cmake_content).await?;
        Ok(())
    }

    /// Generate main plugin file
    async fn generate_main_plugin_file(&self, plugin_dir: &str, config: &PluginBuildConfig) -> Result<()> {
        let main_content = format!(
            r#"#include <obs-module.h>
#include <obs-frontend-api.h>
#include <util/config-file.h>
#include <thread>
#include <memory>
#include "audio-capture.h"
#include "translation-client.h"

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE("{plugin_name}", "en-US")

static std::unique_ptr<AudioCapture> audio_capture;
static std::unique_ptr<TranslationClient> translation_client;

const char *obs_module_name(void)
{{
    return "{plugin_name}";
}}

const char *obs_module_description(void)
{{
    return "{description}";
}}

bool obs_module_load(void)
{{
    blog(LOG_INFO, "[{plugin_name}] Plugin loaded (version {version})");

    // Initialize translation client
    translation_client = std::make_unique<TranslationClient>("http://localhost:8080");

    // Initialize audio capture
    audio_capture = std::make_unique<AudioCapture>();
    audio_capture->set_translation_client(translation_client.get());

    // Register audio capture source
    obs_source_info audio_source_info = {{}};
    audio_source_info.id = "{plugin_name}_audio_capture";
    audio_source_info.type = OBS_SOURCE_TYPE_INPUT;
    audio_source_info.output_flags = OBS_SOURCE_AUDIO;
    audio_source_info.get_name = [](void*) -> const char* {{
        return obs_module_text("LiveTranslationAudioCapture");
    }};
    audio_source_info.create = [](obs_data_t *settings, obs_source_t *source) -> void* {{
        return audio_capture->create_source(settings, source);
    }};
    audio_source_info.destroy = [](void *data) {{
        audio_capture->destroy_source(data);
    }};
    audio_source_info.activate = [](void *data) {{
        audio_capture->activate_source(data);
    }};
    audio_source_info.deactivate = [](void *data) {{
        audio_capture->deactivate_source(data);
    }};

    obs_register_source(&audio_source_info);

    return true;
}}

void obs_module_unload(void)
{{
    blog(LOG_INFO, "[{plugin_name}] Plugin unloaded");

    audio_capture.reset();
    translation_client.reset();
}}

void obs_module_post_load(void)
{{
    // Setup frontend event callbacks
    obs_frontend_add_event_callback([](enum obs_frontend_event event, void *private_data) {{
        switch (event) {{
        case OBS_FRONTEND_EVENT_STREAMING_STARTED:
            audio_capture->start_capture();
            break;
        case OBS_FRONTEND_EVENT_STREAMING_STOPPED:
            audio_capture->stop_capture();
            break;
        case OBS_FRONTEND_EVENT_RECORDING_STARTED:
            audio_capture->start_capture();
            break;
        case OBS_FRONTEND_EVENT_RECORDING_STOPPED:
            audio_capture->stop_capture();
            break;
        default:
            break;
        }}
    }}, nullptr);
}}
"#,
            plugin_name = config.plugin_name,
            version = config.version,
            description = config.description
        );

        fs::create_dir_all(format!("{}/src", plugin_dir)).await?;
        fs::write(format!("{}/src/plugin-main.cpp", plugin_dir), main_content).await?;
        Ok(())
    }

    /// Generate audio capture module
    async fn generate_audio_capture_module(&self, plugin_dir: &str, config: &PluginBuildConfig) -> Result<()> {
        let header_content = r#"#pragma once

#include <obs.h>
#include <memory>
#include <vector>
#include <thread>
#include <atomic>

class TranslationClient;

class AudioCapture {
public:
    AudioCapture();
    ~AudioCapture();

    void set_translation_client(TranslationClient* client);

    // OBS source interface
    void* create_source(obs_data_t* settings, obs_source_t* source);
    void destroy_source(void* data);
    void activate_source(void* data);
    void deactivate_source(void* data);

    // Capture control
    void start_capture();
    void stop_capture();

private:
    struct AudioData;
    std::unique_ptr<AudioData> data_;

    TranslationClient* translation_client_;
    std::atomic<bool> is_capturing_;

    void process_audio(obs_source_t* source, const struct audio_data* audio);
    static void audio_callback(void* param, obs_source_t* source, const struct audio_data* audio, bool muted);
};
"#;

        let impl_content = r#"#include "audio-capture.h"
#include "translation-client.h"
#include <util/circlebuf.h>
#include <util/threading.h>

struct AudioCapture::AudioData {
    obs_source_t* source;
    circlebuf audio_buffer;
    pthread_mutex_t mutex;
    std::vector<float> samples;
    uint32_t sample_rate;
    uint8_t channels;
    bool active;
};

AudioCapture::AudioCapture()
    : translation_client_(nullptr)
    , is_capturing_(false)
    , data_(std::make_unique<AudioData>())
{
    pthread_mutex_init(&data_->mutex, nullptr);
    circlebuf_init(&data_->audio_buffer);
    data_->active = false;
    data_->sample_rate = 48000;
    data_->channels = 2;
}

AudioCapture::~AudioCapture() {
    stop_capture();
    circlebuf_free(&data_->audio_buffer);
    pthread_mutex_destroy(&data_->mutex);
}

void AudioCapture::set_translation_client(TranslationClient* client) {
    translation_client_ = client;
}

void* AudioCapture::create_source(obs_data_t* settings, obs_source_t* source) {
    UNUSED_PARAMETER(settings);

    data_->source = source;

    // Add audio capture callback
    obs_source_add_audio_capture_callback(source, audio_callback, this);

    return data_.get();
}

void AudioCapture::destroy_source(void* data) {
    if (data_->source) {
        obs_source_remove_audio_capture_callback(data_->source, audio_callback, this);
    }
}

void AudioCapture::activate_source(void* data) {
    UNUSED_PARAMETER(data);
    data_->active = true;
    start_capture();
}

void AudioCapture::deactivate_source(void* data) {
    UNUSED_PARAMETER(data);
    data_->active = false;
    stop_capture();
}

void AudioCapture::start_capture() {
    if (!is_capturing_.load()) {
        is_capturing_.store(true);
        blog(LOG_INFO, "[audio-capture] Started audio capture");
    }
}

void AudioCapture::stop_capture() {
    if (is_capturing_.load()) {
        is_capturing_.store(false);
        blog(LOG_INFO, "[audio-capture] Stopped audio capture");
    }
}

void AudioCapture::audio_callback(void* param, obs_source_t* source, const struct audio_data* audio, bool muted) {
    UNUSED_PARAMETER(source);

    if (muted) return;

    AudioCapture* capture = static_cast<AudioCapture*>(param);
    capture->process_audio(source, audio);
}

void AudioCapture::process_audio(obs_source_t* source, const struct audio_data* audio) {
    if (!is_capturing_.load() || !data_->active || !translation_client_) {
        return;
    }

    pthread_mutex_lock(&data_->mutex);

    // Convert audio data to float samples
    const size_t frames = audio->frames;
    const size_t channels = audio_convert_info_get_channels(&audio->format);

    data_->samples.clear();
    data_->samples.reserve(frames * channels);

    // Simple conversion for demonstration - actual implementation would handle different formats
    if (audio->format.format == AUDIO_FORMAT_FLOAT) {
        const float* samples = (const float*)audio->data[0];
        for (size_t i = 0; i < frames * channels; ++i) {
            data_->samples.push_back(samples[i]);
        }
    }

    pthread_mutex_unlock(&data_->mutex);

    // Send to translation service
    if (!data_->samples.empty()) {
        translation_client_->send_audio(data_->samples, data_->sample_rate, channels);
    }
}
"#;

        fs::write(format!("{}/src/audio-capture.h", plugin_dir), header_content).await?;
        fs::write(format!("{}/src/audio-capture.cpp", plugin_dir), impl_content).await?;

        // Generate translation client stub
        let client_header = r#"#pragma once

#include <vector>
#include <string>

class TranslationClient {
public:
    TranslationClient(const std::string& server_url);
    ~TranslationClient();

    void send_audio(const std::vector<float>& samples, uint32_t sample_rate, uint8_t channels);

private:
    std::string server_url_;
};
"#;

        let client_impl = r#"#include "translation-client.h"
#include <obs.h>

TranslationClient::TranslationClient(const std::string& server_url)
    : server_url_(server_url) {
    blog(LOG_INFO, "[translation-client] Initialized with server: %s", server_url.c_str());
}

TranslationClient::~TranslationClient() {
    blog(LOG_INFO, "[translation-client] Destroyed");
}

void TranslationClient::send_audio(const std::vector<float>& samples, uint32_t sample_rate, uint8_t channels) {
    // TODO: Implement HTTP/WebSocket client to send audio data to Rust translation service
    // This would typically involve:
    // 1. Encoding audio samples (possibly to base64)
    // 2. Sending HTTP POST request to translation service
    // 3. Handling the translation response

    blog(LOG_DEBUG, "[translation-client] Received audio: %zu samples at %u Hz",
         samples.size(), sample_rate);
}
"#;

        fs::write(format!("{}/src/translation-client.h", plugin_dir), client_header).await?;
        fs::write(format!("{}/src/translation-client.cpp", plugin_dir), client_impl).await?;

        Ok(())
    }

    /// Generate buildspec.json for GitHub Actions
    async fn generate_buildspec(&self, plugin_dir: &str, config: &PluginBuildConfig) -> Result<()> {
        let buildspec = serde_json::json!({
            "name": config.plugin_name,
            "version": config.version,
            "author": config.author,
            "description": config.description,
            "website": "https://github.com/your-username/obs-live-translator"
        });

        let buildspec_content = serde_json::to_string_pretty(&buildspec)?;
        fs::write(format!("{}/buildspec.json", plugin_dir), buildspec_content).await?;
        Ok(())
    }

    /// Generate GitHub Actions workflow
    async fn generate_github_actions(&self, plugin_dir: &str, config: &PluginBuildConfig) -> Result<()> {
        let workflow_content = format!(
            r#"name: Plugin Build

on:
  push:
    tags:
      - '*'
  pull_request:
    branches: [main]

env:
  PLUGIN_NAME: {plugin_name}

jobs:
  clang_check:
    name: 01 - Code Format Check
    runs-on: ubuntu-22.04
    steps:
      - name: Checkout
        uses: actions/checkout@v3
        with:
          submodules: recursive

      - name: Check Format
        uses: jidicula/clang-format-action@v4.11.0
        with:
          clang-format-version: '13'

  build:
    strategy:
      matrix:
        include:
          - os: windows-2022
            obs-deps: windows-x64
          - os: macos-13
            obs-deps: macos-x86_64
          - os: ubuntu-22.04
            obs-deps: linux-x86_64

    name: 02 - Build (${{{{ matrix.obs-deps }}}})
    runs-on: ${{{{ matrix.os }}}}
    needs: [clang_check]

    steps:
      - name: Checkout
        uses: actions/checkout@v3
        with:
          submodules: recursive

      - name: Setup Environment
        id: setup
        run: |
          echo "obs-deps=${{{{ matrix.obs-deps }}}}" >> $GITHUB_OUTPUT

      - name: Build Plugin
        uses: obsproject/obs-plugintemplate/.github/actions/build-plugin@main
        with:
          target: ${{{{ steps.setup.outputs.obs-deps }}}}

      - name: Package Plugin
        if: startsWith(github.ref, 'refs/tags/')
        uses: obsproject/obs-plugintemplate/.github/actions/package-plugin@main
        with:
          target: ${{{{ steps.setup.outputs.obs-deps }}}}

      - name: Upload Build Artifact
        if: startsWith(github.ref, 'refs/tags/')
        uses: actions/upload-artifact@v3
        with:
          name: ${{{{ env.PLUGIN_NAME }}}}-${{{{ steps.setup.outputs.obs-deps }}}}
          path: release/*.zip
"#,
            plugin_name = config.plugin_name
        );

        fs::create_dir_all(format!("{}/.github/workflows", plugin_dir)).await?;
        fs::write(
            format!("{}/.github/workflows/main.yml", plugin_dir),
            workflow_content,
        ).await?;
        Ok(())
    }

    /// Build plugin using CMake
    pub async fn build_plugin(&self, plugin_dir: &Path, build_type: &str) -> Result<()> {
        let build_dir = plugin_dir.join("build");
        fs::create_dir_all(&build_dir).await?;

        // Configure with CMake
        let mut cmake_config = Command::new("cmake");
        cmake_config
            .arg("..")
            .arg(format!("-DCMAKE_BUILD_TYPE={}", build_type))
            .current_dir(&build_dir);

        #[cfg(target_os = "windows")]
        cmake_config.arg("-G").arg("Visual Studio 17 2022").arg("-A").arg("x64");

        let output = cmake_config.output()?;
        if !output.status.success() {
            return Err(anyhow::anyhow!(
                "CMake configuration failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }

        // Build with CMake
        let build_output = Command::new("cmake")
            .arg("--build")
            .arg(".")
            .arg("--config")
            .arg(build_type)
            .current_dir(&build_dir)
            .output()?;

        if !build_output.status.success() {
            return Err(anyhow::anyhow!(
                "Plugin build failed: {}",
                String::from_utf8_lossy(&build_output.stderr)
            ));
        }

        tracing::info!("Plugin built successfully");
        Ok(())
    }

    /// Install plugin to OBS
    pub async fn install_plugin(&self, plugin_dir: &Path, obs_plugin_dir: &Path) -> Result<()> {
        let build_dir = plugin_dir.join("build");

        // Find built plugin file
        let plugin_extensions = if cfg!(target_os = "windows") {
            vec!["dll"]
        } else if cfg!(target_os = "macos") {
            vec!["so", "dylib"]
        } else {
            vec!["so"]
        };

        for ext in plugin_extensions {
            let plugin_file = format!("{}.{}", self.plugin_name, ext);
            let source_path = build_dir.join(&plugin_file);

            if source_path.exists() {
                let dest_path = obs_plugin_dir.join(&plugin_file);
                fs::copy(&source_path, &dest_path).await?;
                tracing::info!("Plugin installed to: {:?}", dest_path);
                return Ok(());
            }
        }

        Err(anyhow::anyhow!("Built plugin file not found"))
    }
}