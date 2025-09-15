//! Hardware compatibility matrix for 2020-2025 hardware
//!
//! Defines compatibility and performance expectations for various hardware combinations

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::{GpuArchitecture, CpuArchitecture, HardwareGeneration, OptimizationLevel};

/// Hardware compatibility rating
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompatibilityRating {
    /// Fully optimized for this hardware
    Optimal,
    /// Good performance with minor limitations
    Good,
    /// Functional but not optimized
    Compatible,
    /// Limited functionality
    Limited,
    /// Not supported
    Unsupported,
}

/// Performance expectations for hardware
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceExpectations {
    /// Expected latency in milliseconds
    pub latency_ms: u32,
    /// Maximum supported languages
    pub max_languages: usize,
    /// Expected ASR accuracy
    pub accuracy_percent: f32,
    /// Maximum concurrent streams
    pub max_streams: usize,
    /// Recommended batch size
    pub batch_size: usize,
    /// Power consumption estimate (watts)
    pub power_consumption: u32,
}

/// Hardware compatibility matrix
pub struct CompatibilityMatrix {
    cpu_compatibility: HashMap<CpuArchitecture, CompatibilityInfo>,
    gpu_compatibility: HashMap<GpuArchitecture, CompatibilityInfo>,
    generation_performance: HashMap<HardwareGeneration, PerformanceExpectations>,
}

/// Compatibility information
#[derive(Debug, Clone)]
struct CompatibilityInfo {
    rating: CompatibilityRating,
    min_driver_version: Option<String>,
    known_issues: Vec<String>,
    optimizations: Vec<String>,
}

impl CompatibilityMatrix {
    /// Create compatibility matrix
    pub fn new() -> Self {
        let mut matrix = Self {
            cpu_compatibility: HashMap::new(),
            gpu_compatibility: HashMap::new(),
            generation_performance: HashMap::new(),
        };

        matrix.initialize_cpu_compatibility();
        matrix.initialize_gpu_compatibility();
        matrix.initialize_performance_expectations();

        matrix
    }

    /// Initialize CPU compatibility data
    fn initialize_cpu_compatibility(&mut self) {
        // Intel 14th gen (Meteor Lake) - 2024
        self.cpu_compatibility.insert(
            CpuArchitecture::IntelMeteorLake,
            CompatibilityInfo {
                rating: CompatibilityRating::Optimal,
                min_driver_version: None,
                known_issues: vec![],
                optimizations: vec![
                    "P-core/E-core optimization".to_string(),
                    "Intel Thread Director support".to_string(),
                    "AVX2 acceleration".to_string(),
                ],
            },
        );

        // Intel 13th gen (Raptor Lake) - 2023
        self.cpu_compatibility.insert(
            CpuArchitecture::IntelRaptorLake,
            CompatibilityInfo {
                rating: CompatibilityRating::Optimal,
                min_driver_version: None,
                known_issues: vec![],
                optimizations: vec![
                    "P-core/E-core optimization".to_string(),
                    "AVX2 acceleration".to_string(),
                ],
            },
        );

        // Intel 12th gen (Alder Lake) - 2021-2022
        self.cpu_compatibility.insert(
            CpuArchitecture::IntelAlderLake,
            CompatibilityInfo {
                rating: CompatibilityRating::Good,
                min_driver_version: None,
                known_issues: vec![
                    "No AVX-512 support".to_string(),
                ],
                optimizations: vec![
                    "P-core/E-core scheduling".to_string(),
                    "AVX2 acceleration".to_string(),
                ],
            },
        );

        // Intel 11th gen (Rocket Lake) - 2021
        self.cpu_compatibility.insert(
            CpuArchitecture::IntelRocketLake,
            CompatibilityInfo {
                rating: CompatibilityRating::Good,
                min_driver_version: None,
                known_issues: vec![
                    "Higher power consumption with AVX-512".to_string(),
                ],
                optimizations: vec![
                    "AVX-512 acceleration".to_string(),
                    "Improved IPC".to_string(),
                ],
            },
        );

        // Intel 10th gen (Comet Lake) - 2020
        self.cpu_compatibility.insert(
            CpuArchitecture::IntelCometLake,
            CompatibilityInfo {
                rating: CompatibilityRating::Compatible,
                min_driver_version: None,
                known_issues: vec![],
                optimizations: vec![
                    "AVX2 acceleration".to_string(),
                ],
            },
        );

        // AMD Zen 4 (Ryzen 7000) - 2022-2024
        self.cpu_compatibility.insert(
            CpuArchitecture::AmdZen4,
            CompatibilityInfo {
                rating: CompatibilityRating::Optimal,
                min_driver_version: None,
                known_issues: vec![],
                optimizations: vec![
                    "AVX-512 support".to_string(),
                    "Improved branch prediction".to_string(),
                    "Doubled L2 cache".to_string(),
                ],
            },
        );

        // AMD Zen 3 (Ryzen 5000) - 2020-2022
        self.cpu_compatibility.insert(
            CpuArchitecture::AmdZen3,
            CompatibilityInfo {
                rating: CompatibilityRating::Good,
                min_driver_version: None,
                known_issues: vec![],
                optimizations: vec![
                    "Unified CCX design".to_string(),
                    "AVX2 acceleration".to_string(),
                ],
            },
        );

        // AMD Zen 2 (Ryzen 3000) - 2019-2020
        self.cpu_compatibility.insert(
            CpuArchitecture::AmdZen2,
            CompatibilityInfo {
                rating: CompatibilityRating::Compatible,
                min_driver_version: None,
                known_issues: vec![
                    "Higher inter-CCX latency".to_string(),
                ],
                optimizations: vec![
                    "AVX2 acceleration".to_string(),
                ],
            },
        );

        // Apple Silicon
        for arch in [CpuArchitecture::AppleM1, CpuArchitecture::AppleM2,
                     CpuArchitecture::AppleM3, CpuArchitecture::AppleM4] {
            self.cpu_compatibility.insert(
                arch,
                CompatibilityInfo {
                    rating: CompatibilityRating::Optimal,
                    min_driver_version: None,
                    known_issues: vec![],
                    optimizations: vec![
                        "Neural Engine acceleration".to_string(),
                        "Unified memory architecture".to_string(),
                        "NEON SIMD".to_string(),
                    ],
                },
            );
        }
    }

    /// Initialize GPU compatibility data
    fn initialize_gpu_compatibility(&mut self) {
        // NVIDIA RTX 4000 series (Ada Lovelace) - 2022-2024
        self.gpu_compatibility.insert(
            GpuArchitecture::NvidiaAdaLovelace,
            CompatibilityInfo {
                rating: CompatibilityRating::Optimal,
                min_driver_version: Some("525.60".to_string()),
                known_issues: vec![],
                optimizations: vec![
                    "3rd gen RT cores".to_string(),
                    "4th gen Tensor cores".to_string(),
                    "AV1 encoding".to_string(),
                    "FP8 support".to_string(),
                ],
            },
        );

        // NVIDIA RTX 3000 series (Ampere) - 2020-2022
        self.gpu_compatibility.insert(
            GpuArchitecture::NvidiaAmpere,
            CompatibilityInfo {
                rating: CompatibilityRating::Optimal,
                min_driver_version: Some("456.71".to_string()),
                known_issues: vec![],
                optimizations: vec![
                    "2nd gen RT cores".to_string(),
                    "3rd gen Tensor cores".to_string(),
                    "FP16 acceleration".to_string(),
                ],
            },
        );

        // NVIDIA RTX 2000 series (Turing) - 2018-2020
        self.gpu_compatibility.insert(
            GpuArchitecture::NvidiaTuring,
            CompatibilityInfo {
                rating: CompatibilityRating::Good,
                min_driver_version: Some("411.31".to_string()),
                known_issues: vec![
                    "Lower tensor core performance".to_string(),
                ],
                optimizations: vec![
                    "1st gen RT cores".to_string(),
                    "1st gen Tensor cores".to_string(),
                ],
            },
        );

        // AMD RX 7000 series (RDNA3) - 2022-2024
        self.gpu_compatibility.insert(
            GpuArchitecture::AmdRdna3,
            CompatibilityInfo {
                rating: CompatibilityRating::Good,
                min_driver_version: Some("22.40".to_string()),
                known_issues: vec![
                    "Limited ray tracing performance".to_string(),
                ],
                optimizations: vec![
                    "Chiplet architecture".to_string(),
                    "AI accelerators".to_string(),
                    "FP16 support".to_string(),
                ],
            },
        );

        // AMD RX 6000 series (RDNA2) - 2020-2022
        self.gpu_compatibility.insert(
            GpuArchitecture::AmdRdna2,
            CompatibilityInfo {
                rating: CompatibilityRating::Good,
                min_driver_version: Some("20.45".to_string()),
                known_issues: vec![],
                optimizations: vec![
                    "Infinity Cache".to_string(),
                    "Smart Access Memory".to_string(),
                ],
            },
        );

        // AMD RX 5000 series (RDNA1) - 2019-2020
        self.gpu_compatibility.insert(
            GpuArchitecture::AmdRdna1,
            CompatibilityInfo {
                rating: CompatibilityRating::Compatible,
                min_driver_version: Some("19.30".to_string()),
                known_issues: vec![
                    "No hardware ray tracing".to_string(),
                ],
                optimizations: vec![
                    "Improved compute units".to_string(),
                ],
            },
        );

        // Intel Arc A-series (Xe-HPG) - 2022-2023
        self.gpu_compatibility.insert(
            GpuArchitecture::IntelXeHpg,
            CompatibilityInfo {
                rating: CompatibilityRating::Compatible,
                min_driver_version: Some("31.0.101.3430".to_string()),
                known_issues: vec![
                    "Driver maturity issues".to_string(),
                    "Variable performance in some workloads".to_string(),
                ],
                optimizations: vec![
                    "XMX engines".to_string(),
                    "AV1 encoding".to_string(),
                ],
            },
        );
    }

    /// Initialize performance expectations by generation
    fn initialize_performance_expectations(&mut self) {
        // 2025 hardware
        self.generation_performance.insert(
            HardwareGeneration::Gen2025,
            PerformanceExpectations {
                latency_ms: 30,
                max_languages: 1107,
                accuracy_percent: 98.5,
                max_streams: 16,
                batch_size: 24,
                power_consumption: 250,
            },
        );

        // 2024 hardware
        self.generation_performance.insert(
            HardwareGeneration::Gen2024,
            PerformanceExpectations {
                latency_ms: 40,
                max_languages: 1107,
                accuracy_percent: 97.5,
                max_streams: 12,
                batch_size: 16,
                power_consumption: 220,
            },
        );

        // 2023 hardware
        self.generation_performance.insert(
            HardwareGeneration::Gen2023,
            PerformanceExpectations {
                latency_ms: 50,
                max_languages: 800,
                accuracy_percent: 96.5,
                max_streams: 10,
                batch_size: 12,
                power_consumption: 200,
            },
        );

        // 2022 hardware
        self.generation_performance.insert(
            HardwareGeneration::Gen2022,
            PerformanceExpectations {
                latency_ms: 60,
                max_languages: 500,
                accuracy_percent: 95.5,
                max_streams: 8,
                batch_size: 8,
                power_consumption: 180,
            },
        );

        // 2021 hardware
        self.generation_performance.insert(
            HardwareGeneration::Gen2021,
            PerformanceExpectations {
                latency_ms: 75,
                max_languages: 300,
                accuracy_percent: 94.0,
                max_streams: 6,
                batch_size: 6,
                power_consumption: 160,
            },
        );

        // 2020 hardware
        self.generation_performance.insert(
            HardwareGeneration::Gen2020,
            PerformanceExpectations {
                latency_ms: 100,
                max_languages: 100,
                accuracy_percent: 92.0,
                max_streams: 4,
                batch_size: 4,
                power_consumption: 150,
            },
        );

        // Legacy hardware
        self.generation_performance.insert(
            HardwareGeneration::Legacy,
            PerformanceExpectations {
                latency_ms: 150,
                max_languages: 50,
                accuracy_percent: 88.0,
                max_streams: 2,
                batch_size: 2,
                power_consumption: 120,
            },
        );
    }

    /// Get CPU compatibility rating
    pub fn get_cpu_compatibility(&self, arch: &CpuArchitecture) -> CompatibilityRating {
        self.cpu_compatibility
            .get(arch)
            .map(|info| info.rating)
            .unwrap_or(CompatibilityRating::Limited)
    }

    /// Get GPU compatibility rating
    pub fn get_gpu_compatibility(&self, arch: &GpuArchitecture) -> CompatibilityRating {
        self.gpu_compatibility
            .get(arch)
            .map(|info| info.rating)
            .unwrap_or(CompatibilityRating::Limited)
    }

    /// Get performance expectations for hardware generation
    pub fn get_performance_expectations(&self, gen: &HardwareGeneration) -> Option<&PerformanceExpectations> {
        self.generation_performance.get(gen)
    }

    /// Get known issues for CPU
    pub fn get_cpu_issues(&self, arch: &CpuArchitecture) -> Vec<String> {
        self.cpu_compatibility
            .get(arch)
            .map(|info| info.known_issues.clone())
            .unwrap_or_default()
    }

    /// Get known issues for GPU
    pub fn get_gpu_issues(&self, arch: &GpuArchitecture) -> Vec<String> {
        self.gpu_compatibility
            .get(arch)
            .map(|info| info.known_issues.clone())
            .unwrap_or_default()
    }

    /// Get minimum driver version for GPU
    pub fn get_min_driver_version(&self, arch: &GpuArchitecture) -> Option<String> {
        self.gpu_compatibility
            .get(arch)
            .and_then(|info| info.min_driver_version.clone())
    }
}