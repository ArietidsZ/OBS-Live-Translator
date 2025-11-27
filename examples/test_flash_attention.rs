use anyhow::Result;
use obs_live_translator::execution_provider::ExecutionProviderConfig;
use obs_live_translator::platform_detect::Platform;
use obs_live_translator::types::AccelerationConfig;
use ort::session::builder::SessionBuilder;

fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("Testing Flash Attention Configuration...");

    // Create a configuration with Flash Attention enabled
    let mut acceleration_config = AccelerationConfig::default();
    acceleration_config.flash_attention = true;

    // Create a session builder
    let builder = SessionBuilder::new()?;

    // Create execution provider config
    let platform = Platform::detect()?;
    let provider_config = ExecutionProviderConfig::from_platform(platform);

    // Configure the session
    println!("Configuring session with Flash Attention...");
    let _builder = provider_config.configure_session(builder, &acceleration_config)?;

    println!(
        "Session configured successfully with Flash Attention (GraphOptimizationLevel::Level3)!"
    );

    // Test with Flash Attention disabled
    acceleration_config.flash_attention = false;
    let builder = SessionBuilder::new()?;

    println!("Configuring session without Flash Attention...");
    let _builder = provider_config.configure_session(builder, &acceleration_config)?;

    println!(
        "Session configured successfully without Flash Attention (GraphOptimizationLevel::Level1)!"
    );

    Ok(())
}
