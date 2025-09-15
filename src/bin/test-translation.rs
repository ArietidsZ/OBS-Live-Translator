//! Test harness for OBS Live Translator with real-time subtitle display

use anyhow::Result;
use obs_live_translator_core::streaming::OptimizedStreamingPipeline;
use std::sync::Arc;
use tokio::sync::broadcast;
use tracing::{info, error};

mod web {
    pub mod subtitle_server;
}

use web::subtitle_server::{SubtitleServer, SubtitleData};

/// Test configuration
struct TestConfig {
    /// Source language for input audio
    source_lang: String,
    /// Target language for translation
    target_lang: String,
    /// Path to test audio files
    audio_path: String,
    /// WebSocket server port
    ws_port: u16,
    /// Enable test mode with synthetic data
    test_mode: bool,
}

impl Default for TestConfig {
    fn default() -> Self {
        Self {
            source_lang: "auto".to_string(),  // Auto-detect source language
            target_lang: "en".to_string(),
            audio_path: "./test_data".to_string(),
            ws_port: 8080,
            test_mode: true,
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    info!("🚀 OBS Live Translator - Test Mode with Subtitle Display");
    info!("=====================================================");

    let config = TestConfig::default();

    // Create subtitle server
    let subtitle_server = Arc::new(SubtitleServer::new());

    // Start WebSocket server in background
    let server_handle = {
        let server = Arc::clone(&subtitle_server);
        tokio::spawn(async move {
            server.start_server(config.ws_port).await;
        })
    };

    info!("📺 Subtitle overlay available at:");
    info!("   http://localhost:{}/", config.ws_port);
    info!("");
    info!("🎬 OBS Setup Instructions:");
    info!("   1. Add a Browser Source in OBS");
    info!("   2. Set URL to: http://localhost:{}/", config.ws_port);
    info!("   3. Set Width: 1920, Height: 1080");
    info!("   4. Check 'Shutdown source when not visible'");
    info!("   5. Check 'Refresh browser when scene becomes active'");
    info!("");

    // Wait a moment for server to start
    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

    if config.test_mode {
        info!("🧪 Running in test mode with synthetic data");
        info!("   Subtitles will cycle through multiple languages");
        info!("");

        // Run test subtitle generation
        let test_handle = {
            let server = Arc::clone(&subtitle_server);
            tokio::spawn(async move {
                web::subtitle_server::generate_test_subtitles(server).await;
            })
        };

        // Also simulate processing pipeline
        simulate_translation_pipeline(Arc::clone(&subtitle_server)).await?;

        test_handle.await?;
    } else {
        info!("🎤 Starting live translation pipeline");
        info!("   Source language: {}", config.source_lang);
        info!("   Target language: {}", config.target_lang);
        info!("");

        // Initialize real translation pipeline
        run_live_translation(
            Arc::clone(&subtitle_server),
            &config.source_lang,
            &config.target_lang,
        ).await?;
    }

    server_handle.await?;
    Ok(())
}

/// Simulate translation pipeline with test data
async fn simulate_translation_pipeline(subtitle_server: Arc<SubtitleServer>) -> Result<()> {
    use tokio::time::{sleep, Duration};
    use std::time::SystemTime;

    info!("Starting simulated translation pipeline...");

    // Simulate different streaming scenarios
    let scenarios = vec![
        // Scenario 1: Spanish news broadcast
        vec![
            ("Buenos días y bienvenidos a las noticias", "Good morning and welcome to the news", "es", 0.97),
            ("El clima de hoy será soleado con temperaturas agradables", "Today's weather will be sunny with pleasant temperatures", "es", 0.95),
            ("En las noticias principales de hoy", "In today's main news", "es", 0.96),
        ],
        // Scenario 2: French cooking show
        vec![
            ("Aujourd'hui, nous allons préparer un plat délicieux", "Today, we will prepare a delicious dish", "fr", 0.94),
            ("Commencez par couper les légumes en petits morceaux", "Start by cutting the vegetables into small pieces", "fr", 0.93),
            ("Ajoutez une pincée de sel et de poivre", "Add a pinch of salt and pepper", "fr", 0.95),
        ],
        // Scenario 3: Japanese gaming stream
        vec![
            ("このゲームは本当に面白いですね", "This game is really interesting", "ja", 0.91),
            ("次のレベルに進みましょう", "Let's move on to the next level", "ja", 0.89),
            ("皆さん、コメントありがとうございます", "Thank you everyone for your comments", "ja", 0.92),
        ],
        // Scenario 4: German tech review
        vec![
            ("Willkommen zu unserem Technologie-Review", "Welcome to our technology review", "de", 0.96),
            ("Dieses neue Smartphone hat beeindruckende Funktionen", "This new smartphone has impressive features", "de", 0.94),
            ("Die Kameraqualität ist ausgezeichnet", "The camera quality is excellent", "de", 0.95),
        ],
        // Scenario 5: Korean music show
        vec![
            ("안녕하세요, 오늘의 음악 프로그램입니다", "Hello, this is today's music program", "ko", 0.90),
            ("다음 곡은 최신 히트곡입니다", "The next song is the latest hit", "ko", 0.88),
            ("함께 즐겨주세요", "Please enjoy together", "ko", 0.91),
        ],
    ];

    for (scenario_idx, scenario) in scenarios.iter().enumerate() {
        info!("📝 Scenario {}: Processing {} subtitles", scenario_idx + 1, scenario.len());

        for (original, translated, lang, confidence) in scenario {
            // Simulate processing delay
            sleep(Duration::from_millis(500)).await;

            let subtitle = SubtitleData {
                original_text: original.to_string(),
                translated_text: translated.to_string(),
                source_lang: lang.to_string(),
                target_lang: "en".to_string(),
                confidence: *confidence,
                timestamp: SystemTime::now()
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
            };

            // Broadcast subtitle
            subtitle_server.broadcast_subtitle(subtitle).await;

            // Simulate realistic timing between subtitles
            sleep(Duration::from_secs(3)).await;
        }

        // Pause between scenarios
        info!("   ✅ Scenario {} complete", scenario_idx + 1);
        sleep(Duration::from_secs(2)).await;
    }

    info!("✨ All test scenarios completed. Looping...");

    // Keep running
    loop {
        sleep(Duration::from_secs(10)).await;
        info!("📊 System running... Check http://localhost:8080 for subtitle overlay");
    }
}

/// Run live translation with actual audio input
async fn run_live_translation(
    subtitle_server: Arc<SubtitleServer>,
    source_lang: &str,
    target_lang: &str,
) -> Result<()> {
    use obs_live_translator_core::gpu::AdaptiveMemoryManager;
    use obs_live_translator_core::acceleration::ONNXAccelerator;

    info!("Initializing live translation pipeline...");

    // Detect available VRAM
    let vram_mb = detect_vram().await.unwrap_or(4096);
    info!("Detected VRAM: {}MB", vram_mb);

    // Initialize components
    let memory_manager = Arc::new(AdaptiveMemoryManager::new(vram_mb));
    let accelerator = Arc::new(ONNXAccelerator::new().await?);

    // Create optimized pipeline
    let pipeline = Arc::new(
        OptimizedStreamingPipeline::new(
            memory_manager,
            accelerator,
            vram_mb,
        ).await?
    );

    info!("✅ Pipeline initialized successfully");
    info!("🎤 Waiting for audio input...");

    // In a real implementation, this would connect to OBS audio input
    // For now, we'll simulate with a message
    info!("");
    info!("ℹ️  To use with real audio:");
    info!("   1. Configure OBS audio capture");
    info!("   2. Route audio through virtual cable");
    info!("   3. Pipeline will auto-detect and translate");

    // Keep the server running
    loop {
        tokio::time::sleep(tokio::time::Duration::from_secs(60)).await;
        info!("💚 System healthy - Subtitle server running on http://localhost:8080");
    }
}

/// Detect available VRAM
async fn detect_vram() -> Option<usize> {
    // Try to detect NVIDIA GPU
    #[cfg(not(target_os = "macos"))]
    {
        if let Ok(output) = tokio::process::Command::new("nvidia-smi")
            .arg("--query-gpu=memory.total")
            .arg("--format=csv,noheader,nounits")
            .output()
            .await
        {
            if let Ok(vram_str) = String::from_utf8(output.stdout) {
                if let Ok(vram) = vram_str.trim().parse::<usize>() {
                    return Some(vram);
                }
            }
        }
    }

    // Apple Silicon detection
    #[cfg(target_os = "macos")]
    {
        // Estimate based on total system memory
        if let Ok(output) = tokio::process::Command::new("sysctl")
            .arg("-n")
            .arg("hw.memsize")
            .output()
            .await
        {
            if let Ok(mem_str) = String::from_utf8(output.stdout) {
                if let Ok(total_mem) = mem_str.trim().parse::<usize>() {
                    // Use 40% of total memory for unified memory architecture
                    return Some((total_mem / 1024 / 1024) * 40 / 100);
                }
            }
        }
    }

    None
}