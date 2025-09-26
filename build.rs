use std::env;
use cc::Build;

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    #[allow(unused_variables)]
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();

    println!("cargo:rerun-if-changed=src/native/engine_stub.cpp");
    println!("cargo:rerun-if-changed=src/native/audio_simd.cpp");
    println!("cargo:rerun-if-changed=src/native/onnx_inference.cpp");

    // Build engine stub always (it's required for basic FFI)
    let mut engine_build = Build::new();
    engine_build
        .cpp(true)
        .file("src/native/engine_stub.cpp")
        .flag_if_supported("-std=c++17")
        .flag_if_supported("-O3")
        .flag_if_supported("-march=native")
        .flag_if_supported("-ffast-math");

    // Engine stub doesn't need ONNX Runtime - it's just basic FFI stubs

    // Platform-specific optimizations
    if target_os == "linux" {
        engine_build
            .flag_if_supported("-fopenmp")
            .flag_if_supported("-mavx512f");
    }

    engine_build.compile("engine_stub");

    // Only build SIMD components if feature is enabled
    #[cfg(feature = "simd")]
    {
        // Build SIMD audio processor
        let mut audio_build = Build::new();
        audio_build
            .cpp(true)
            .file("src/native/audio_simd.cpp")
            .flag_if_supported("-std=c++17")
            .flag_if_supported("-O3")
            .flag_if_supported("-march=native")
            .flag_if_supported("-ffast-math");

        // Add architecture-specific SIMD flags
        if target_arch == "x86_64" {
            audio_build
                .flag_if_supported("-mavx2")
                .flag_if_supported("-mfma");
        } else if target_arch == "aarch64" {
            audio_build.flag_if_supported("-march=armv8-a+fp+simd");
        }

        audio_build.compile("audio_simd");

        // Build ONNX Runtime inference engine
        let mut onnx_build = Build::new();
        onnx_build
            .cpp(true)
            .file("src/native/onnx_inference.cpp")
            .flag_if_supported("-std=c++17")
            .flag_if_supported("-O3")
            .flag_if_supported("-march=native");

        // Add ONNX Runtime paths and handle linking centrally
        if target_os == "macos" {
            if target_arch == "aarch64" {
                // For Apple Silicon, using x86_64 ONNX Runtime via Rosetta 2
                onnx_build.include("/usr/local/include/onnxruntime");
                onnx_build.include("/usr/local/include");
                println!("cargo:rustc-link-search=/usr/local/lib");
                println!("cargo:rustc-link-lib=onnxruntime");
            } else {
                onnx_build.include("/usr/local/include/onnxruntime");
                onnx_build.include("/usr/local/include");
                println!("cargo:rustc-link-search=/usr/local/lib");
                println!("cargo:rustc-link-lib=onnxruntime");
            }
        } else {
            onnx_build.include("/usr/include/onnxruntime");
            println!("cargo:rustc-link-search=/usr/lib");
            println!("cargo:rustc-link-lib=onnxruntime");
        }

        onnx_build.compile("onnx_inference");
    }

    // Platform-specific linking
    match target_os.as_str() {
        "windows" => {
            println!("cargo:rustc-link-lib=ole32");
        }
        "linux" => {
            println!("cargo:rustc-link-lib=gomp");  // OpenMP
        }
        _ => {}
    }

    // Link C++ standard library
    let cpp_stdlib = if target_os == "macos" {
        "c++"
    } else if target_os == "windows" {
        ""  // MSVC doesn't need explicit linking
    } else {
        "stdc++"
    };

    if !cpp_stdlib.is_empty() {
        println!("cargo:rustc-link-lib={}", cpp_stdlib);
    }
}