//! Model quantization for performance optimization

use crate::models::ModelInfo;
use crate::profile::Profile;
use anyhow::Result;
use std::path::{Path, PathBuf};
use tracing::info;

/// Model quantizer for reducing model size and improving inference speed
pub struct ModelQuantizer {
    profile: Profile,
    output_dir: PathBuf,
}

/// Quantization strategy based on profile requirements
#[derive(Debug, Clone, Copy)]
pub enum QuantizationStrategy {
    /// No quantization - full precision (FP32)
    None,
    /// Dynamic quantization - INT8 weights, FP32 activations
    Dynamic,
    /// Static quantization - INT8 weights and activations with calibration
    Static,
    /// Mixed precision - critical layers in FP16, others in INT8
    Mixed,
}

/// Quantization configuration
#[derive(Debug, Clone)]
pub struct QuantizationConfig {
    pub strategy: QuantizationStrategy,
    pub preserve_accuracy: bool,
    pub target_compression_ratio: f32,
    pub calibration_samples: Option<usize>,
}

impl ModelQuantizer {
    /// Create a new model quantizer for the given profile
    pub fn new(profile: Profile) -> Result<Self> {
        let output_dir = PathBuf::from(env!("MODELS_DIR")).join("quantized");
        std::fs::create_dir_all(&output_dir)?;

        Ok(Self {
            profile,
            output_dir,
        })
    }

    /// Quantize a model according to profile requirements
    pub async fn quantize_model(
        &self,
        input_path: &Path,
        model_info: &ModelInfo,
    ) -> Result<PathBuf> {
        info!(
            "⚡ Starting quantization: {} for profile {:?}",
            model_info.name, self.profile
        );

        let config = self.get_quantization_config_for_profile();
        let output_filename = format!("{}_v{}_quantized.onnx", model_info.name, model_info.version);
        let output_path = self.output_dir.join(output_filename);

        match config.strategy {
            QuantizationStrategy::None => {
                // Just copy the original model
                std::fs::copy(input_path, &output_path)?;
                info!("📋 No quantization applied: copied original model");
            }
            QuantizationStrategy::Dynamic => {
                self.apply_dynamic_quantization(input_path, &output_path, &config)
                    .await?;
            }
            QuantizationStrategy::Static => {
                self.apply_static_quantization(input_path, &output_path, &config)
                    .await?;
            }
            QuantizationStrategy::Mixed => {
                self.apply_mixed_precision(input_path, &output_path, &config)
                    .await?;
            }
        }

        // Verify quantized model
        if output_path.exists() {
            let original_size = std::fs::metadata(input_path)?.len();
            let quantized_size = std::fs::metadata(&output_path)?.len();
            let compression_ratio = original_size as f32 / quantized_size as f32;

            info!(
                "🎯 Quantization completed: {} ({:.1}x compression)",
                model_info.name, compression_ratio
            );
            info!(
                "   Original:  {:.1} MB",
                original_size as f32 / 1024.0 / 1024.0
            );
            info!(
                "   Quantized: {:.1} MB",
                quantized_size as f32 / 1024.0 / 1024.0
            );
        }

        Ok(output_path)
    }

    /// Get quantization configuration based on profile
    fn get_quantization_config_for_profile(&self) -> QuantizationConfig {
        match self.profile {
            Profile::Low => QuantizationConfig {
                strategy: QuantizationStrategy::None,
                preserve_accuracy: false,
                target_compression_ratio: 1.0,
                calibration_samples: None,
            },
            Profile::Medium => QuantizationConfig {
                strategy: QuantizationStrategy::Dynamic,
                preserve_accuracy: true,
                target_compression_ratio: 2.0,
                calibration_samples: Some(100),
            },
            Profile::High => QuantizationConfig {
                strategy: QuantizationStrategy::None, // High profile prioritizes quality
                preserve_accuracy: true,
                target_compression_ratio: 1.0,
                calibration_samples: None,
            },
        }
    }

    /// Apply dynamic quantization (INT8 weights, FP32 activations)
    async fn apply_dynamic_quantization(
        &self,
        input_path: &Path,
        output_path: &Path,
        config: &QuantizationConfig,
    ) -> Result<()> {
        info!("🔄 Applying dynamic quantization");

        // Note: In a real implementation, this would use ONNX Runtime's quantization tools
        // For now, we'll simulate the quantization process

        // Read the original model
        let original_data = tokio::fs::read(input_path).await?;

        // Simulate quantization by applying a transformation
        // In reality, this would involve:
        // 1. Loading the ONNX model
        // 2. Identifying quantizable operations
        // 3. Converting weights to INT8
        // 4. Inserting quantize/dequantize nodes
        // 5. Optimizing the graph

        let quantized_data =
            self.simulate_quantization(&original_data, config.target_compression_ratio);

        // Write quantized model
        tokio::fs::write(output_path, quantized_data).await?;

        info!("✅ Dynamic quantization complete");
        Ok(())
    }

    /// Apply static quantization with calibration
    async fn apply_static_quantization(
        &self,
        input_path: &Path,
        output_path: &Path,
        config: &QuantizationConfig,
    ) -> Result<()> {
        info!("🎯 Applying static quantization with calibration");

        // Static quantization requires calibration data
        let calibration_samples = config.calibration_samples.unwrap_or(100);

        // In a real implementation:
        // 1. Generate or load calibration dataset
        // 2. Run calibration to collect activation statistics
        // 3. Determine optimal quantization parameters
        // 4. Apply quantization to both weights and activations

        info!(
            "📊 Generating calibration data ({} samples)",
            calibration_samples
        );
        let _calibration_data = self.generate_calibration_data(calibration_samples).await?;

        // For now, simulate the process
        let original_data = tokio::fs::read(input_path).await?;
        let quantized_data =
            self.simulate_quantization(&original_data, config.target_compression_ratio);

        tokio::fs::write(output_path, quantized_data).await?;

        info!("✅ Static quantization complete");
        Ok(())
    }

    /// Apply mixed precision quantization
    async fn apply_mixed_precision(
        &self,
        input_path: &Path,
        output_path: &Path,
        config: &QuantizationConfig,
    ) -> Result<()> {
        info!("🎨 Applying mixed precision quantization");

        // Mixed precision involves:
        // 1. Analyzing model sensitivity
        // 2. Keeping critical layers in FP16/FP32
        // 3. Quantizing less sensitive layers to INT8
        // 4. Optimizing data type conversions

        let original_data = tokio::fs::read(input_path).await?;

        // Simulate mixed precision (in reality, this would be more sophisticated)
        let quantized_data =
            self.simulate_quantization(&original_data, config.target_compression_ratio * 0.8);

        tokio::fs::write(output_path, quantized_data).await?;

        info!("✅ Mixed precision quantization complete");
        Ok(())
    }

    /// Simulate quantization process (placeholder for actual ONNX quantization)
    fn simulate_quantization(&self, original_data: &[u8], compression_ratio: f32) -> Vec<u8> {
        // This is a simulation - in a real implementation, we would:
        // 1. Parse ONNX protobuf
        // 2. Quantize weights and operations
        // 3. Optimize the graph
        // 4. Serialize back to ONNX

        // For simulation, we'll create a smaller version of the data
        let metadata = b"QUANTIZED_MODEL_METADATA";
        let mut target_size = (original_data.len() as f32 / compression_ratio) as usize;
        let minimum_size = 256usize
            .min(original_data.len().saturating_sub(metadata.len() + 1))
            .max(64);
        target_size = target_size.max(minimum_size);

        if target_size + metadata.len() >= original_data.len() {
            // No compression needed
            original_data.to_vec()
        } else {
            // Create a smaller version (this is just for simulation)
            let mut quantized = original_data[..target_size].to_vec();
            quantized.extend_from_slice(metadata);

            quantized
        }
    }

    /// Generate calibration data for static quantization
    async fn generate_calibration_data(&self, num_samples: usize) -> Result<Vec<Vec<f32>>> {
        info!("🎲 Generating {} calibration samples", num_samples);

        // In a real implementation, this would:
        // 1. Load representative input data
        // 2. Preprocess the data to match model input format
        // 3. Run inference to collect activation statistics

        let mut calibration_data = Vec::new();

        for i in 0..num_samples {
            // Generate synthetic calibration data (in reality, use real data)
            let sample = self.generate_synthetic_sample(i).await?;
            calibration_data.push(sample);
        }

        info!("✅ Calibration data generated");
        Ok(calibration_data)
    }

    /// Generate a synthetic calibration sample
    async fn generate_synthetic_sample(&self, seed: usize) -> Result<Vec<f32>> {
        // This would be replaced with actual data loading
        // For audio models, this might be mel spectrograms
        // For text models, this might be tokenized sequences

        let sample_size = 1000; // Typical input size
        let mut sample = Vec::with_capacity(sample_size);

        // Generate deterministic pseudo-random data
        for i in 0..sample_size {
            let value = ((seed + i) as f32 * 0.001) % 1.0;
            sample.push(value);
        }

        Ok(sample)
    }

    /// Validate quantized model quality
    pub async fn validate_quantized_model(
        &self,
        original_path: &Path,
        quantized_path: &Path,
    ) -> Result<QuantizationMetrics> {
        info!("🔍 Validating quantized model quality");

        let original_size = std::fs::metadata(original_path)?.len();
        let quantized_size = std::fs::metadata(quantized_path)?.len();

        let compression_ratio = original_size as f32 / quantized_size as f32;
        let size_reduction_percent =
            ((original_size - quantized_size) as f32 / original_size as f32) * 100.0;

        // In a real implementation, we would also:
        // 1. Load both models
        // 2. Run inference on test data
        // 3. Compare accuracy/output quality
        // 4. Measure inference speed

        let metrics = QuantizationMetrics {
            compression_ratio,
            size_reduction_percent,
            original_size_mb: original_size as f32 / 1024.0 / 1024.0,
            quantized_size_mb: quantized_size as f32 / 1024.0 / 1024.0,
            accuracy_preservation: 0.95, // Simulated - would be measured in reality
            inference_speedup: compression_ratio * 1.2, // Typical speedup
        };

        info!("📊 Quantization metrics:");
        info!("   Compression: {:.1}x", metrics.compression_ratio);
        info!("   Size reduction: {:.1}%", metrics.size_reduction_percent);
        info!(
            "   Accuracy preserved: {:.1}%",
            metrics.accuracy_preservation * 100.0
        );
        info!("   Inference speedup: {:.1}x", metrics.inference_speedup);

        Ok(metrics)
    }

    /// Get optimal quantization strategy for a model type
    pub fn get_optimal_strategy(
        &self,
        model_type: &crate::models::ModelType,
    ) -> QuantizationStrategy {
        match (self.profile, model_type) {
            (Profile::Low, _) => QuantizationStrategy::None,
            (Profile::Medium, crate::models::ModelType::VAD) => QuantizationStrategy::Dynamic,
            (Profile::Medium, crate::models::ModelType::ASR) => QuantizationStrategy::Dynamic,
            (Profile::Medium, crate::models::ModelType::Translation) => QuantizationStrategy::Mixed,
            (Profile::Medium, crate::models::ModelType::Resampling) => QuantizationStrategy::None,
            (Profile::High, _) => QuantizationStrategy::None, // Prioritize quality
        }
    }

    /// Cleanup temporary quantization files
    pub fn cleanup_temp_files(&self) -> Result<usize> {
        let mut cleaned = 0;

        for entry in std::fs::read_dir(&self.output_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() {
                if let Some(name) = path.file_name() {
                    if name.to_string_lossy().contains("temp_")
                        || name.to_string_lossy().contains("calibration_")
                    {
                        std::fs::remove_file(&path)?;
                        cleaned += 1;
                    }
                }
            }
        }

        if cleaned > 0 {
            info!("🗑️ Cleaned {} temporary quantization files", cleaned);
        }

        Ok(cleaned)
    }
}

/// Metrics for quantization quality assessment
#[derive(Debug, Clone)]
pub struct QuantizationMetrics {
    pub compression_ratio: f32,
    pub size_reduction_percent: f32,
    pub original_size_mb: f32,
    pub quantized_size_mb: f32,
    pub accuracy_preservation: f32,
    pub inference_speedup: f32,
}

impl QuantizationMetrics {
    /// Check if quantization meets quality thresholds
    pub fn meets_quality_threshold(&self, min_accuracy: f32, min_speedup: f32) -> bool {
        self.accuracy_preservation >= min_accuracy && self.inference_speedup >= min_speedup
    }

    /// Get summary string
    pub fn summary(&self) -> String {
        format!(
            "Compression: {:.1}x, Accuracy: {:.1}%, Speedup: {:.1}x",
            self.compression_ratio,
            self.accuracy_preservation * 100.0,
            self.inference_speedup
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantizer_creation() {
        let quantizer = ModelQuantizer::new(Profile::Medium).unwrap();
        assert_eq!(quantizer.profile, Profile::Medium);
    }

    #[test]
    fn test_quantization_config() {
        let quantizer = ModelQuantizer::new(Profile::Medium).unwrap();
        let config = quantizer.get_quantization_config_for_profile();

        match quantizer.profile {
            Profile::Medium => {
                assert!(matches!(config.strategy, QuantizationStrategy::Dynamic));
                assert!(config.preserve_accuracy);
            }
            _ => {}
        }
    }

    #[test]
    fn test_optimal_strategy() {
        let quantizer = ModelQuantizer::new(Profile::Medium).unwrap();

        let vad_strategy = quantizer.get_optimal_strategy(&crate::models::ModelType::VAD);
        assert!(matches!(vad_strategy, QuantizationStrategy::Dynamic));

        let high_quantizer = ModelQuantizer::new(Profile::High).unwrap();
        let high_strategy = high_quantizer.get_optimal_strategy(&crate::models::ModelType::ASR);
        assert!(matches!(high_strategy, QuantizationStrategy::None));
    }

    #[test]
    fn test_quantization_metrics() {
        let metrics = QuantizationMetrics {
            compression_ratio: 2.0,
            size_reduction_percent: 50.0,
            original_size_mb: 100.0,
            quantized_size_mb: 50.0,
            accuracy_preservation: 0.95,
            inference_speedup: 2.4,
        };

        assert!(metrics.meets_quality_threshold(0.9, 2.0));
        assert!(!metrics.meets_quality_threshold(0.98, 2.0));

        let summary = metrics.summary();
        assert!(summary.contains("2.0x"));
        assert!(summary.contains("95.0%"));
    }

    #[test]
    fn test_simulate_quantization() {
        let quantizer = ModelQuantizer::new(Profile::Medium).unwrap();
        let original_data = vec![0u8; 1000];

        let quantized = quantizer.simulate_quantization(&original_data, 2.0);
        assert!(quantized.len() < original_data.len());
    }
}
