//! Integration Tests for OBS Live Translator
//!
//! This module provides end-to-end integration testing including:
//! - End-to-end latency testing
//! - Resource constraint validation
//! - Profile switching testing
//! - Multi-language testing

pub mod e2e_tests;
pub mod performance_tests;
pub mod stress_tests;

#[cfg(test)]
mod integration_test_utils {
    use std::time::{Duration, Instant};
    use std::path::PathBuf;
    use tokio::time::timeout;

    /// Integration test timeout
    pub const INTEGRATION_TEST_TIMEOUT: Duration = Duration::from_secs(60);

    /// Create test environment
    pub async fn setup_test_environment() -> TestEnvironment {
        TestEnvironment {
            temp_dir: tempfile::tempdir().unwrap(),
            start_time: Instant::now(),
            test_data_dir: PathBuf::from("tests/data"),
        }
    }

    /// Test environment container
    pub struct TestEnvironment {
        pub temp_dir: tempfile::TempDir,
        pub start_time: Instant,
        pub test_data_dir: PathBuf,
    }

    impl TestEnvironment {
        pub fn get_temp_path(&self, filename: &str) -> PathBuf {
            self.temp_dir.path().join(filename)
        }

        pub fn elapsed(&self) -> Duration {
            self.start_time.elapsed()
        }
    }

    /// Measure latency of async operation
    pub async fn measure_latency<F, T>(operation: F) -> (T, Duration)
    where
        F: std::future::Future<Output = T>,
    {
        let start = Instant::now();
        let result = operation.await;
        (result, start.elapsed())
    }

    /// Run test with resource monitoring
    pub async fn run_with_monitoring<F, T>(test_fn: F) -> (T, ResourceMetrics)
    where
        F: std::future::Future<Output = T>,
    {
        let start_memory = get_current_memory_usage();
        let start_cpu = get_current_cpu_usage();

        let result = test_fn.await;

        let end_memory = get_current_memory_usage();
        let end_cpu = get_current_cpu_usage();

        let metrics = ResourceMetrics {
            memory_delta_mb: (end_memory - start_memory) as f32 / 1024.0 / 1024.0,
            peak_memory_mb: end_memory as f32 / 1024.0 / 1024.0,
            avg_cpu_percent: (start_cpu + end_cpu) / 2.0,
        };

        (result, metrics)
    }

    #[derive(Debug)]
    pub struct ResourceMetrics {
        pub memory_delta_mb: f32,
        pub peak_memory_mb: f32,
        pub avg_cpu_percent: f32,
    }

    fn get_current_memory_usage() -> usize {
        // Placeholder - in real implementation use system metrics
        1024 * 1024 * 100 // 100MB
    }

    fn get_current_cpu_usage() -> f32 {
        // Placeholder - in real implementation use system metrics
        25.0
    }
}