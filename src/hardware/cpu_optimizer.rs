//! CPU-specific optimizations for 2020-2025 hardware
//!
//! Adaptive optimizations for Intel 10th-14th gen and AMD Ryzen 3000-7000 series

use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::{CpuArchitecture, CpuInfo, SimdSupport, OptimizationLevel};

/// CPU optimization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuOptimizationSettings {
    /// Number of worker threads
    pub thread_count: usize,
    /// SIMD instruction set to use
    pub simd_level: SimdLevel,
    /// Thread affinity strategy
    pub affinity: ThreadAffinity,
    /// Prefetch strategy
    pub prefetch: PrefetchStrategy,
    /// Cache optimization
    pub cache_strategy: CacheStrategy,
    /// Power/performance preference
    pub power_preference: PowerPreference,
}

/// SIMD instruction level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SimdLevel {
    /// No SIMD (scalar operations)
    None,
    /// SSE2 (baseline x86_64)
    SSE2,
    /// SSE4.2
    SSE42,
    /// AVX2 (most common modern)
    AVX2,
    /// AVX-512 (Intel 11th gen, AMD Zen 4+)
    AVX512,
    /// ARM NEON
    NEON,
}

/// Thread affinity strategy
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ThreadAffinity {
    /// Let OS handle thread scheduling
    None,
    /// Pin threads to physical cores
    PhysicalCores,
    /// Pin threads to performance cores (Intel 12th gen+)
    PerformanceCores,
    /// Pin threads to efficiency cores
    EfficiencyCores,
    /// NUMA-aware pinning
    NumaAware,
}

/// Prefetch strategy for memory access
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PrefetchStrategy {
    /// No prefetching
    None,
    /// Conservative prefetch
    Conservative,
    /// Aggressive prefetch
    Aggressive,
    /// Adaptive based on access patterns
    Adaptive,
}

/// Cache optimization strategy
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CacheStrategy {
    /// Default caching
    Default,
    /// Optimize for L1/L2 cache
    LocalCache,
    /// Optimize for L3 cache
    SharedCache,
    /// Streaming (bypass cache)
    Streaming,
}

/// Power/performance preference
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PowerPreference {
    /// Maximum performance
    Performance,
    /// Balanced
    Balanced,
    /// Power efficiency
    PowerEfficient,
}

/// CPU optimizer
pub struct CpuOptimizer {
    cpu_info: CpuInfo,
    simd_support: SimdSupport,
    settings: CpuOptimizationSettings,
}

impl CpuOptimizer {
    /// Create optimizer for detected CPU
    pub fn new(cpu_info: CpuInfo, simd_support: SimdSupport) -> Self {
        let settings = Self::get_optimal_settings(&cpu_info, &simd_support);
        Self {
            cpu_info,
            simd_support,
            settings,
        }
    }

    /// Get optimal settings based on CPU architecture
    fn get_optimal_settings(cpu_info: &CpuInfo, simd_support: &SimdSupport) -> CpuOptimizationSettings {
        // Determine SIMD level
        let simd_level = if simd_support.avx512 {
            // Only use AVX-512 on architectures with good implementation
            match cpu_info.architecture {
                CpuArchitecture::IntelRocketLake | // Good AVX-512
                CpuArchitecture::AmdZen4 | // AMD's first AVX-512, efficient
                CpuArchitecture::AmdZen5 => SimdLevel::AVX512,
                _ => SimdLevel::AVX2, // Avoid AVX-512 on other architectures
            }
        } else if simd_support.avx2 {
            SimdLevel::AVX2
        } else if simd_support.sse42 {
            SimdLevel::SSE42
        } else if simd_support.sse2 {
            SimdLevel::SSE2
        } else if simd_support.neon {
            SimdLevel::NEON
        } else {
            SimdLevel::None
        };

        // Architecture-specific optimizations
        match cpu_info.architecture {
            // Intel 14th gen (Meteor Lake)
            CpuArchitecture::IntelMeteorLake => CpuOptimizationSettings {
                thread_count: cpu_info.thread_count * 3 / 4, // Use 75% of threads
                simd_level,
                affinity: ThreadAffinity::PerformanceCores,
                prefetch: PrefetchStrategy::Aggressive,
                cache_strategy: CacheStrategy::LocalCache,
                power_preference: PowerPreference::Performance,
            },

            // Intel 13th gen (Raptor Lake)
            CpuArchitecture::IntelRaptorLake => CpuOptimizationSettings {
                thread_count: cpu_info.thread_count * 3 / 4,
                simd_level,
                affinity: ThreadAffinity::PerformanceCores,
                prefetch: PrefetchStrategy::Aggressive,
                cache_strategy: CacheStrategy::LocalCache,
                power_preference: PowerPreference::Balanced,
            },

            // Intel 12th gen (Alder Lake) - P-cores and E-cores
            CpuArchitecture::IntelAlderLake => CpuOptimizationSettings {
                thread_count: cpu_info.core_count, // Use P-cores primarily
                simd_level,
                affinity: ThreadAffinity::PerformanceCores,
                prefetch: PrefetchStrategy::Adaptive,
                cache_strategy: CacheStrategy::SharedCache,
                power_preference: PowerPreference::Balanced,
            },

            // Intel 11th gen (Rocket Lake) - Good AVX-512
            CpuArchitecture::IntelRocketLake => CpuOptimizationSettings {
                thread_count: cpu_info.thread_count,
                simd_level: SimdLevel::AVX512, // Good AVX-512 implementation
                affinity: ThreadAffinity::PhysicalCores,
                prefetch: PrefetchStrategy::Aggressive,
                cache_strategy: CacheStrategy::LocalCache,
                power_preference: PowerPreference::Performance,
            },

            // Intel 10th gen (Comet Lake)
            CpuArchitecture::IntelCometLake => CpuOptimizationSettings {
                thread_count: cpu_info.thread_count,
                simd_level,
                affinity: ThreadAffinity::PhysicalCores,
                prefetch: PrefetchStrategy::Conservative,
                cache_strategy: CacheStrategy::Default,
                power_preference: PowerPreference::Balanced,
            },

            // AMD Zen 4 (Ryzen 7000) - AVX-512 support
            CpuArchitecture::AmdZen4 => CpuOptimizationSettings {
                thread_count: cpu_info.thread_count,
                simd_level: SimdLevel::AVX512, // Efficient AVX-512
                affinity: ThreadAffinity::NumaAware,
                prefetch: PrefetchStrategy::Adaptive,
                cache_strategy: CacheStrategy::SharedCache,
                power_preference: PowerPreference::Performance,
            },

            // AMD Zen 3 (Ryzen 5000)
            CpuArchitecture::AmdZen3 => CpuOptimizationSettings {
                thread_count: cpu_info.thread_count,
                simd_level,
                affinity: ThreadAffinity::NumaAware,
                prefetch: PrefetchStrategy::Adaptive,
                cache_strategy: CacheStrategy::SharedCache,
                power_preference: PowerPreference::Balanced,
            },

            // AMD Zen 2 (Ryzen 3000)
            CpuArchitecture::AmdZen2 => CpuOptimizationSettings {
                thread_count: cpu_info.thread_count * 7 / 8, // Leave some headroom
                simd_level,
                affinity: ThreadAffinity::NumaAware,
                prefetch: PrefetchStrategy::Conservative,
                cache_strategy: CacheStrategy::SharedCache,
                power_preference: PowerPreference::Balanced,
            },

            // Apple Silicon
            CpuArchitecture::AppleM1 | CpuArchitecture::AppleM2 |
            CpuArchitecture::AppleM3 | CpuArchitecture::AppleM4 => CpuOptimizationSettings {
                thread_count: cpu_info.core_count, // Use performance cores
                simd_level: SimdLevel::NEON,
                affinity: ThreadAffinity::PerformanceCores,
                prefetch: PrefetchStrategy::Adaptive,
                cache_strategy: CacheStrategy::LocalCache,
                power_preference: PowerPreference::Balanced,
            },

            // Unknown/Legacy
            _ => CpuOptimizationSettings {
                thread_count: cpu_info.thread_count / 2, // Conservative
                simd_level: SimdLevel::SSE2,
                affinity: ThreadAffinity::None,
                prefetch: PrefetchStrategy::None,
                cache_strategy: CacheStrategy::Default,
                power_preference: PowerPreference::PowerEfficient,
            },
        }
    }

    /// Optimize for specific workload
    pub fn optimize_for_workload(&mut self, workload: WorkloadType) {
        match workload {
            WorkloadType::RealTimeAudio => {
                // Optimize for low latency
                self.settings.thread_count = self.settings.thread_count.min(4);
                self.settings.prefetch = PrefetchStrategy::Aggressive;
                self.settings.cache_strategy = CacheStrategy::LocalCache;
            }
            WorkloadType::ParallelProcessing => {
                // Optimize for throughput
                self.settings.thread_count = self.cpu_info.thread_count;
                self.settings.affinity = ThreadAffinity::PhysicalCores;
            }
            WorkloadType::PowerEfficient => {
                // Optimize for power efficiency
                self.settings.power_preference = PowerPreference::PowerEfficient;
                self.settings.thread_count = self.cpu_info.core_count / 2;
                if matches!(self.cpu_info.architecture,
                    CpuArchitecture::IntelAlderLake |
                    CpuArchitecture::IntelRaptorLake |
                    CpuArchitecture::IntelMeteorLake) {
                    self.settings.affinity = ThreadAffinity::EfficiencyCores;
                }
            }
        }
    }

    /// Get SIMD optimization code
    pub fn get_simd_code(&self) -> &'static str {
        match self.settings.simd_level {
            SimdLevel::AVX512 => "avx512",
            SimdLevel::AVX2 => "avx2",
            SimdLevel::SSE42 => "sse4.2",
            SimdLevel::SSE2 => "sse2",
            SimdLevel::NEON => "neon",
            SimdLevel::None => "generic",
        }
    }

    /// Check if CPU supports specific optimization
    pub fn supports_optimization(&self, optimization: &str) -> bool {
        match optimization {
            "avx2" => self.simd_support.avx2,
            "avx512" => self.simd_support.avx512,
            "fma" => self.simd_support.fma,
            "hyperthreading" => self.cpu_info.supports_hyperthreading,
            "p_cores" => matches!(self.cpu_info.architecture,
                CpuArchitecture::IntelAlderLake |
                CpuArchitecture::IntelRaptorLake |
                CpuArchitecture::IntelMeteorLake),
            _ => false,
        }
    }

    /// Get optimization recommendations
    pub fn get_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Thread count recommendations
        if self.cpu_info.thread_count < 8 {
            recommendations.push("Consider upgrading to CPU with at least 8 threads for better performance".to_string());
        }

        // Architecture-specific recommendations
        match self.cpu_info.architecture {
            CpuArchitecture::IntelAlderLake | CpuArchitecture::IntelRaptorLake => {
                recommendations.push("Use Thread Director for optimal P-core/E-core scheduling".to_string());
            }
            CpuArchitecture::AmdZen4 => {
                recommendations.push("Enable AVX-512 optimizations for improved performance".to_string());
            }
            CpuArchitecture::AmdZen2 | CpuArchitecture::AmdZen3 => {
                recommendations.push("Consider memory overclocking for improved Infinity Fabric performance".to_string());
            }
            _ => {}
        }

        // SIMD recommendations
        if !self.simd_support.avx2 {
            recommendations.push("CPU lacks AVX2 support - performance may be limited".to_string());
        }

        recommendations
    }
}

/// Workload type for CPU optimization
#[derive(Debug, Clone, Copy)]
pub enum WorkloadType {
    /// Real-time audio processing
    RealTimeAudio,
    /// Parallel batch processing
    ParallelProcessing,
    /// Power-efficient processing
    PowerEfficient,
}

/// Thread pool manager with CPU affinity
pub struct ThreadPoolManager {
    thread_count: usize,
    affinity: ThreadAffinity,
}

impl ThreadPoolManager {
    /// Create new thread pool manager
    pub fn new(settings: &CpuOptimizationSettings) -> Self {
        Self {
            thread_count: settings.thread_count,
            affinity: settings.affinity,
        }
    }

    /// Create optimized thread pool
    pub fn create_thread_pool(&self) -> Result<rayon::ThreadPool> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.thread_count)
            .thread_name(|i| format!("translator-worker-{}", i))
            .build()?;

        // Note: Actual CPU affinity would require platform-specific code
        // using libraries like core_affinity or hwloc

        Ok(pool)
    }

    /// Set thread affinity (platform-specific)
    #[cfg(target_os = "linux")]
    pub fn set_thread_affinity(&self, thread_id: usize, core_id: usize) -> Result<()> {
        // Linux-specific CPU affinity using libc
        Ok(())
    }

    #[cfg(target_os = "windows")]
    pub fn set_thread_affinity(&self, thread_id: usize, core_id: usize) -> Result<()> {
        // Windows-specific CPU affinity using Windows API
        Ok(())
    }

    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    pub fn set_thread_affinity(&self, _thread_id: usize, _core_id: usize) -> Result<()> {
        // Not supported on other platforms
        Ok(())
    }
}