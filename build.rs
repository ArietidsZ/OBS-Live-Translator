use std::env;
use std::path::Path;
use std::process::Command;
use cc::Build;

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();

    // Detect active profile features
    let profile_low = env::var("CARGO_FEATURE_PROFILE_LOW").is_ok();
    let profile_medium = env::var("CARGO_FEATURE_PROFILE_MEDIUM").is_ok();
    let profile_high = env::var("CARGO_FEATURE_PROFILE_HIGH").is_ok();
    let has_simd = env::var("CARGO_FEATURE_SIMD").is_ok();

    // Default to all profiles if none specified
    let enable_all = !profile_low && !profile_medium && !profile_high;

    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_PROFILE_LOW");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_PROFILE_MEDIUM");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_PROFILE_HIGH");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_SIMD");

    // Detect hardware capabilities for build optimization
    let cpu_features = detect_cpu_features();
    let has_ipp = detect_intel_ipp(&target_os);
    let onnx_config = detect_onnx_runtime(&target_os, &target_arch);

    println!("cargo:rustc-cfg=cpu_features=\"{}\"", cpu_features.join(","));
    if has_ipp {
        println!("cargo:rustc-cfg=has_intel_ipp");
    }

    println!("cargo:rerun-if-changed=src/native/audio_simd.cpp");
    println!("cargo:rerun-if-changed=src/native/onnx_inference.cpp");
    println!("cargo:rerun-if-changed=src/native/gpu_kernels_unified.cu");

    // Build components based on active profiles
    build_profile_components(&target_os, &target_arch, profile_low || enable_all,
                           profile_medium || enable_all, profile_high || enable_all,
                           has_simd, &cpu_features, has_ipp, &onnx_config);

    // Setup model management if any profile is enabled
    if profile_low || profile_medium || profile_high || enable_all {
        setup_model_management(&target_os);
    }

    // Platform-specific linking
    setup_platform_linking(&target_os);
}

/// Detect CPU features for build optimization
fn detect_cpu_features() -> Vec<String> {
    let mut features = Vec::new();

    // Try to detect CPU features through various methods
    if cfg!(target_arch = "x86_64") {
        // Use std::arch for x86_64 feature detection
        if std::is_x86_feature_detected!("sse2") { features.push("sse2".to_string()); }
        if std::is_x86_feature_detected!("avx") { features.push("avx".to_string()); }
        if std::is_x86_feature_detected!("avx2") { features.push("avx2".to_string()); }
        if std::is_x86_feature_detected!("fma") { features.push("fma".to_string()); }
    } else if cfg!(target_arch = "aarch64") {
        // ARM64 has NEON by default
        features.push("neon".to_string());
    }

    features
}

/// Detect Intel IPP availability
fn detect_intel_ipp(target_os: &str) -> bool {
    // Check for Intel IPP installation
    let ipp_paths = match target_os {
        "windows" => vec![
            "C:\\Program Files (x86)\\Intel\\oneAPI\\ipp\\latest\\include",
            "C:\\Intel\\compilers_and_libraries\\windows\\ipp\\include",
        ],
        "macos" => vec![
            "/opt/intel/oneapi/ipp/latest/include",
            "/usr/local/include/ipp",
        ],
        "linux" => vec![
            "/opt/intel/oneapi/ipp/latest/include",
            "/usr/include/ipp",
            "/usr/local/include/ipp",
        ],
        _ => vec![],
    };

    for path in ipp_paths {
        if Path::new(path).exists() {
            println!("cargo:rustc-link-search={}/lib", path.replace("/include", ""));
            println!("cargo:rustc-link-lib=ippi");
            println!("cargo:rustc-link-lib=ipps");
            println!("cargo:rustc-link-lib=ippcore");
            return true;
        }
    }

    false
}

/// ONNX Runtime configuration
#[derive(Debug)]
struct OnnxConfig {
    available: bool,
    include_path: String,
    lib_path: String,
    provider: String, // CPU, CUDA, CoreML, DirectML
}

/// Detect ONNX Runtime availability and configuration
fn detect_onnx_runtime(target_os: &str, _target_arch: &str) -> OnnxConfig {
    let mut config = OnnxConfig {
        available: false,
        include_path: String::new(),
        lib_path: String::new(),
        provider: "CPU".to_string(),
    };

    let potential_paths = match target_os {
        "macos" => vec![
            ("/opt/homebrew/include/onnxruntime", "/opt/homebrew/lib"),
            ("/usr/local/include/onnxruntime", "/usr/local/lib"),
        ],
        "linux" => vec![
            ("/usr/include/onnxruntime", "/usr/lib"),
            ("/usr/local/include/onnxruntime", "/usr/local/lib"),
        ],
        "windows" => vec![
            ("C:\\Program Files\\onnxruntime\\include", "C:\\Program Files\\onnxruntime\\lib"),
        ],
        _ => vec![],
    };

    for (inc_path, lib_path) in potential_paths {
        if Path::new(inc_path).exists() && Path::new(lib_path).exists() {
            config.available = true;
            config.include_path = inc_path.to_string();
            config.lib_path = lib_path.to_string();

            // Detect available providers
            if target_os == "macos" {
                config.provider = "CoreML".to_string();
            } else if target_os == "windows" {
                config.provider = "DirectML".to_string();
            } else {
                // Check for CUDA availability on Linux
                if Path::new("/usr/local/cuda").exists() {
                    config.provider = "CUDA".to_string();
                }
            }
            break;
        }
    }

    if config.available {
        println!("cargo:rustc-link-search={}", config.lib_path);
        println!("cargo:rustc-link-lib=onnxruntime");
    }

    config
}

/// Build components based on active profiles
fn build_profile_components(
    target_os: &str,
    target_arch: &str,
    profile_low: bool,
    profile_medium: bool,
    profile_high: bool,
    has_simd: bool,
    cpu_features: &[String],
    has_ipp: bool,
    onnx_config: &OnnxConfig,
) {
    // Build profile-specific components
    if profile_low {
        build_low_profile_components(target_os, target_arch, cpu_features);
    }

    if profile_medium {
        build_medium_profile_components(target_os, target_arch, cpu_features, onnx_config);
    }

    if profile_high {
        build_high_profile_components(target_os, target_arch, cpu_features, has_ipp, onnx_config);
    }

    // Build SIMD components if enabled and supported
    if has_simd && !cpu_features.is_empty() {
        build_simd_components(target_os, target_arch, cpu_features);
    }
}


/// Build components for low profile (CPU-only, basic optimizations)
fn build_low_profile_components(_target_os: &str, _target_arch: &str, cpu_features: &[String]) {
    println!("cargo:rustc-cfg=profile_low_enabled");

    // Only build if source files exist
    if Path::new("src/native/vad_webrtc.cpp").exists() {
        let mut build = Build::new();
        build
            .cpp(true)
            .file("src/native/vad_webrtc.cpp")
            .flag_if_supported("-std=c++17")
            .flag_if_supported("-O2"); // More conservative optimization for compatibility

        // Basic SIMD if available
        if cpu_features.contains(&"sse2".to_string()) {
            build.flag_if_supported("-msse2");
        }

        build.compile("low_profile");
    } else {
        println!("cargo:warning=Low profile C++ components not found, skipping compilation");
    }
}

/// Build components for medium profile (GPU support, enhanced features)
fn build_medium_profile_components(
    _target_os: &str,
    _target_arch: &str,
    cpu_features: &[String],
    onnx_config: &OnnxConfig,
) {
    println!("cargo:rustc-cfg=profile_medium_enabled");

    if onnx_config.available && Path::new("src/native/onnx_inference.cpp").exists() {
        let mut build = Build::new();
        build
            .cpp(true)
            .file("src/native/onnx_inference.cpp")
            .include(&onnx_config.include_path)
            .flag_if_supported("-std=c++17")
            .flag_if_supported("-O3");

        // Enhanced CPU optimizations
        if cpu_features.contains(&"avx2".to_string()) {
            build.flag_if_supported("-mavx2");
        }

        if cpu_features.contains(&"fma".to_string()) {
            build.flag_if_supported("-mfma");
        }

        build.compile("medium_profile");
    } else {
        println!("cargo:warning=Medium profile C++ components not found or ONNX unavailable, skipping compilation");
    }
}

/// Build components for high profile (maximum performance, Intel IPP)
fn build_high_profile_components(
    target_os: &str,
    target_arch: &str,
    cpu_features: &[String],
    has_ipp: bool,
    onnx_config: &OnnxConfig,
) {
    println!("cargo:rustc-cfg=profile_high_enabled");

    if !Path::new("src/native/high_perf_audio.cpp").exists() {
        println!("cargo:warning=High profile C++ components not found, skipping compilation");
        return;
    }

    let mut build = Build::new();
    build
        .cpp(true)
        .file("src/native/high_perf_audio.cpp")
        .flag_if_supported("-std=c++17")
        .flag_if_supported("-O3")
        .flag_if_supported("-ffast-math")
        .flag_if_supported("-funroll-loops");

    // Maximum CPU optimizations
    if target_arch == "x86_64" {
        if cpu_features.contains(&"avx2".to_string()) {
            build.flag_if_supported("-mavx2");
        }
        if cpu_features.contains(&"fma".to_string()) {
            build.flag_if_supported("-mfma");
        }
        // Enable all supported x86_64 features
        build.flag_if_supported("-march=native");
    } else if target_arch == "aarch64" {
        build.flag_if_supported("-march=armv8-a+fp+simd+crypto");
    }

    // Intel IPP integration for maximum performance
    if has_ipp {
        build.define("USE_INTEL_IPP", "1");
        if target_os == "linux" {
            build.flag_if_supported("-fopenmp");
        }
    }

    // GPU acceleration setup
    if onnx_config.available {
        build.include(&onnx_config.include_path);
        build.define("USE_ONNX_RUNTIME", "1");
        match onnx_config.provider.as_str() {
            "CUDA" => { build.define("USE_CUDA", "1"); },
            "CoreML" => { build.define("USE_COREML", "1"); },
            "DirectML" => { build.define("USE_DIRECTML", "1"); },
            _ => {}
        }
    }

    build.compile("high_profile");
}

/// Build SIMD-optimized components
fn build_simd_components(target_os: &str, target_arch: &str, cpu_features: &[String]) {
    if !Path::new("src/native/audio_simd.cpp").exists() {
        println!("cargo:warning=SIMD C++ components not found, skipping compilation");
        return;
    }

    let mut build = Build::new();
    build
        .cpp(true)
        .file("src/native/audio_simd.cpp")
        .flag_if_supported("-std=c++17")
        .flag_if_supported("-O3")
        .flag_if_supported("-ffast-math");

    // Architecture-specific SIMD optimizations
    if target_arch == "x86_64" {
        if cpu_features.contains(&"avx2".to_string()) {
            build.flag_if_supported("-mavx2");
        }
        if cpu_features.contains(&"fma".to_string()) {
            build.flag_if_supported("-mfma");
        }
    } else if target_arch == "aarch64" {
        build.flag_if_supported("-march=armv8-a+fp+simd");
    }

    // Platform-specific optimizations
    if target_os == "linux" {
        build.flag_if_supported("-fopenmp");
    }

    build.compile("audio_simd");
}

/// Setup model management and download integration
fn setup_model_management(target_os: &str) {
    // Create models directory if it doesn't exist
    std::fs::create_dir_all("models").unwrap_or_else(|e| {
        println!("cargo:warning=Failed to create models directory: {}", e);
    });

    // Set up model download script
    let script_name = if target_os == "windows" {
        "download_models.bat"
    } else {
        "download_models.sh"
    };

    let script_path = format!("scripts/{}", script_name);
    if Path::new(&script_path).exists() {
        println!("cargo:rerun-if-changed={}", script_path);

        // Make script executable on Unix systems
        if target_os != "windows" {
            let _ = Command::new("chmod")
                .arg("+x")
                .arg(&script_path)
                .output();
        }
    }

    // Set environment variables for model paths
    println!("cargo:rustc-env=MODELS_DIR=models");
    println!("cargo:rustc-env=MODEL_CACHE_DIR=models/cache");
}

/// Setup platform-specific linking
fn setup_platform_linking(target_os: &str) {
    match target_os {
        "windows" => {
            println!("cargo:rustc-link-lib=ole32");
            println!("cargo:rustc-link-lib=kernel32");
        }
        "linux" => {
            println!("cargo:rustc-link-lib=gomp");  // OpenMP
            println!("cargo:rustc-link-lib=pthread");
        }
        "macos" => {
            println!("cargo:rustc-link-lib=framework=Accelerate");
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=MetalKit");
        }
        _ => {}
    }

    // Link C++ standard library
    let cpp_stdlib = match target_os {
        "macos" => "c++",
        "windows" => return, // MSVC doesn't need explicit linking
        _ => "stdc++",
    };

    println!("cargo:rustc-link-lib={}", cpp_stdlib);
}