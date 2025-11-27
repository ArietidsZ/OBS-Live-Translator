use obs_live_translator::execution_provider::ExecutionProviderConfig;
use obs_live_translator::platform_detect::Platform;

fn main() {
    match Platform::detect() {
        Ok(platform) => {
            println!("Detected Platform Result: {platform:?}");
            println!("Description: {}", platform.description());

            let config = ExecutionProviderConfig::from_platform(platform.clone());
            println!("Execution Provider Summary: {}", config.summary());

            match platform {
                Platform::AppleSilicon(_) => {
                    if config.summary().contains("CoreML") {
                        println!("✅ CoreML selected for Apple Silicon - Part 2.5 Success");
                    } else {
                        println!("⚠️ CoreML NOT selected for Apple Silicon (Check feature flags)");
                    }
                }
                Platform::NvidiaGpu(_) => {
                    println!("✅ NVIDIA GPU detected - Proceeding with Part 2.3/2.4")
                }
                _ => println!("ℹ️ Other platform detected"),
            }
        }
        Err(e) => {
            println!("❌ Platform detection failed: {e}");
        }
    }
}
