//! OBS Setup Utility
//!
//! Automated setup tool for OBS Studio integration

use anyhow::Result;
use clap::{Parser, Subcommand};
use obs_live_translator::obs::{ObsConfig, ObsSetup, SetupStep};
use std::io::{self, Write};
use tokio::time::{sleep, Duration};

#[derive(Parser)]
#[command(name = "obs-setup")]
#[command(about = "OBS Live Translator Setup Utility")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run automated setup
    Auto {
        /// Skip confirmation prompts
        #[arg(short, long)]
        yes: bool,
    },
    /// Run interactive setup with progress
    Interactive,
    /// Check OBS installation and compatibility
    Check,
    /// Generate setup instructions
    Instructions,
    /// Test current configuration
    Test,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("obs_setup=info,obs_live_translator=info")
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Auto { yes } => run_auto_setup(yes).await,
        Commands::Interactive => run_interactive_setup().await,
        Commands::Check => run_compatibility_check().await,
        Commands::Instructions => generate_instructions().await,
        Commands::Test => test_configuration().await,
    }
}

async fn run_auto_setup(skip_confirmation: bool) -> Result<()> {
    println!("🌍 OBS Live Translator - Automated Setup");
    println!("==========================================\n");

    if !skip_confirmation {
        print!("This will configure OBS Studio for real-time translation. Continue? (y/N): ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        if !input.trim().to_lowercase().starts_with('y') {
            println!("Setup cancelled.");
            return Ok(());
        }
    }

    let config = ObsConfig::default();
    let mut setup = ObsSetup::new(config);

    println!("Starting automated setup...\n");

    match setup.run_automated_setup().await {
        Ok(integration) => {
            println!("✅ Setup completed successfully!");
            println!("\n🎥 Next steps:");
            println!("1. Open OBS Studio");
            println!("2. Add Browser Source with URL: {}", integration.get_browser_source_url());
            println!("3. Start streaming/recording to begin translation");
            println!("\n🔧 Configuration saved. You can modify settings in the web interface at:");
            println!("   http://localhost:8080");
        }
        Err(e) => {
            eprintln!("❌ Setup failed: {}", e);
            eprintln!("\n🔧 Try running 'obs-setup check' to diagnose issues");
            std::process::exit(1);
        }
    }

    Ok(())
}

async fn run_interactive_setup() -> Result<()> {
    println!("🌍 OBS Live Translator - Interactive Setup");
    println!("===========================================\n");

    let config = ObsConfig::default();
    let mut setup = ObsSetup::new(config);

    let progress_callback = Box::new(|step: SetupStep| {
        match step {
            SetupStep::DetectingObs => {
                print!("🔍 Detecting OBS Studio installation... ");
                io::stdout().flush().unwrap();
            }
            SetupStep::CheckingWebSocket => {
                println!("✅");
                print!("🔌 Checking WebSocket availability... ");
                io::stdout().flush().unwrap();
            }
            SetupStep::ConfiguringAudio => {
                println!("✅");
                print!("🎤 Configuring audio capture... ");
                io::stdout().flush().unwrap();
            }
            SetupStep::CreatingBrowserSource => {
                println!("✅");
                print!("🌐 Creating browser source template... ");
                io::stdout().flush().unwrap();
            }
            SetupStep::InstallingPlugin => {
                println!("✅");
                print!("🔧 Installing OBS plugin... ");
                io::stdout().flush().unwrap();
            }
            SetupStep::TestingConnection => {
                println!("✅");
                print!("🧪 Testing integration... ");
                io::stdout().flush().unwrap();
            }
            SetupStep::Complete => {
                println!("✅");
                println!("\n🎉 Setup completed successfully!");
            }
            SetupStep::Error(msg) => {
                println!("❌");
                eprintln!("Error: {}", msg);
            }
        }
    });

    match setup.run_setup_with_progress(progress_callback).await {
        Ok(integration) => {
            println!("\n📋 Setup Summary:");
            println!("├─ Browser Source URL: {}", integration.get_browser_source_url());
            println!("├─ Web Interface: http://localhost:8080");
            println!("├─ WebSocket Port: 4455");
            println!("└─ Status: Ready for streaming");

            println!("\n🎥 OBS Configuration:");
            println!("1. Open OBS Studio");
            println!("2. Add 'Browser' source");
            println!("3. Set URL to: {}", integration.get_browser_source_url());
            println!("4. Set dimensions: 1920x1080");
            println!("5. Enable hardware acceleration");
            println!("6. Start streaming to see live translations!");
        }
        Err(e) => {
            eprintln!("\n❌ Setup failed: {}", e);
            std::process::exit(1);
        }
    }

    Ok(())
}

async fn run_compatibility_check() -> Result<()> {
    println!("🔍 OBS Live Translator - Compatibility Check");
    println!("=============================================\n");

    let config = ObsConfig::default();
    let setup = ObsSetup::new(config);

    // Check OBS installation
    print!("Checking OBS Studio installation... ");
    match setup.detect_obs_installation().await {
        Ok(obs_info) => {
            println!("✅");
            println!("  ├─ Version: {}", obs_info.version);
            println!("  ├─ Path: {:?}", obs_info.install_path);
            println!("  └─ WebSocket: {}", if obs_info.websocket_enabled { "✅ Enabled" } else { "❌ Disabled" });
        }
        Err(e) => {
            println!("❌");
            println!("  └─ Error: {}", e);
        }
    }

    // Check WebSocket connectivity
    print!("Testing WebSocket connection... ");
    match obs_live_translator::obs::WebSocketClient::detect_obs_websocket().await {
        Ok(true) => println!("✅ Available"),
        Ok(false) => println!("❌ Not available"),
        Err(e) => println!("❌ Error: {}", e),
    }

    // Check audio devices
    print!("Scanning audio devices... ");
    match obs_live_translator::obs::AudioHandler::get_available_devices().await {
        Ok(devices) => {
            println!("✅ {} devices found", devices.len());
            for device in devices.iter().take(3) {
                println!("  ├─ {}{}", device.name, if device.is_default { " (default)" } else { "" });
            }
            if devices.len() > 3 {
                println!("  └─ ... and {} more", devices.len() - 3);
            }
        }
        Err(e) => {
            println!("❌ Error: {}", e);
        }
    }

    // Check OBS virtual audio
    print!("Checking OBS virtual audio... ");
    match obs_live_translator::obs::AudioHandler::detect_obs_audio_device().await {
        Ok(Some(device)) => println!("✅ Found: {}", device),
        Ok(None) => println!("⚠️  Not found (will use default microphone)"),
        Err(e) => println!("❌ Error: {}", e),
    }

    // Check system requirements
    println!("\n💻 System Information:");
    println!("  ├─ OS: {}", std::env::consts::OS);
    println!("  ├─ Architecture: {}", std::env::consts::ARCH);
    println!("  └─ CPU cores: {}", num_cpus::get());

    println!("\n✅ Compatibility check completed");
    Ok(())
}

async fn generate_instructions() -> Result<()> {
    println!("📋 OBS Live Translator - Setup Instructions");
    println!("============================================\n");

    let config = ObsConfig::default();
    let setup = ObsSetup::new(config);

    let instructions = setup.generate_setup_instructions();
    println!("{}", instructions);

    Ok(())
}

async fn test_configuration() -> Result<()> {
    println!("🧪 OBS Live Translator - Configuration Test");
    println!("============================================\n");

    let config = ObsConfig::default();
    let mut integration = obs_live_translator::obs::ObsIntegration::new(config);

    print!("Initializing translation service... ");
    io::stdout().flush()?;

    match integration.initialize().await {
        Ok(()) => {
            println!("✅");

            print!("Testing OBS connection... ");
            io::stdout().flush()?;

            let connected = integration.check_obs_connection().await;
            if connected {
                println!("✅");
            } else {
                println!("⚠️  OBS not connected (this is normal if OBS isn't running)");
            }

            print!("Starting overlay server... ");
            io::stdout().flush()?;

            match integration.start_overlay().await {
                Ok(()) => {
                    println!("✅");
                    println!("\n🌐 Test complete! Access the overlay at:");
                    println!("   {}", integration.get_browser_source_url());
                    println!("\n⏱️  Running for 10 seconds...");

                    sleep(Duration::from_secs(10)).await;

                    integration.shutdown().await?;
                    println!("🛑 Test completed successfully");
                }
                Err(e) => {
                    println!("❌ Error: {}", e);
                }
            }
        }
        Err(e) => {
            println!("❌ Error: {}", e);
            return Err(e);
        }
    }

    Ok(())
}