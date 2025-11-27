//! Model validation and integrity checking

use anyhow::Result;
use std::path::Path;
use tracing::{error, info, warn};

/// Model validator for integrity checking and format validation
pub struct ModelValidator {
    // Future: Could add ONNX runtime for model loading validation
}

impl ModelValidator {
    /// Create a new model validator
    pub fn new() -> Result<Self> {
        Ok(Self {})
    }

    /// Validate model integrity using checksum
    pub async fn validate_model(&self, model_path: &Path, expected_checksum: &str) -> Result<bool> {
        if !model_path.exists() {
            return Ok(false);
        }

        info!("🔍 Validating model: {:?}", model_path);

        // For built-in models, skip checksum validation
        if expected_checksum == "built-in" {
            return Ok(true);
        }

        // Calculate file checksum
        let calculated_checksum = self.calculate_file_checksum(model_path).await?;

        // Extract hash from expected format (e.g., "sha256:abc123...")
        let expected_hash = if expected_checksum.starts_with("sha256:") {
            &expected_checksum[7..]
        } else {
            expected_checksum
        };

        let is_valid = calculated_checksum == expected_hash;

        if is_valid {
            info!("✅ Model validation successful: {:?}", model_path);
        } else {
            error!("❌ Model validation failed: {:?}", model_path);
            error!("   Expected: {}", expected_hash);
            error!("   Got:      {}", calculated_checksum);
        }

        Ok(is_valid)
    }

    /// Calculate SHA256 checksum of a file
    async fn calculate_file_checksum(&self, file_path: &Path) -> Result<String> {
        use sha2::{Digest, Sha256};
        use tokio::io::AsyncReadExt;

        let mut file = tokio::fs::File::open(file_path).await?;
        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; 8192]; // 8KB buffer

        loop {
            let bytes_read = file.read(&mut buffer).await?;
            if bytes_read == 0 {
                break;
            }
            hasher.update(&buffer[..bytes_read]);
        }

        let result = hasher.finalize();
        Ok(format!("{:x}", result))
    }

    /// Validate ONNX model format (basic checks)
    pub async fn validate_onnx_format(&self, model_path: &Path) -> Result<bool> {
        if !model_path.exists() {
            return Ok(false);
        }

        // Check file extension
        if let Some(extension) = model_path.extension() {
            if extension != "onnx" {
                warn!(
                    "⚠️ Model file does not have .onnx extension: {:?}",
                    model_path
                );
                return Ok(false);
            }
        }

        // Basic file size check (ONNX models should be at least a few KB)
        let metadata = tokio::fs::metadata(model_path).await?;
        if metadata.len() < 1024 {
            warn!("⚠️ Model file too small to be valid ONNX: {:?}", model_path);
            return Ok(false);
        }

        // Check ONNX magic bytes (ONNX files start with specific protobuf magic)
        let mut file = tokio::fs::File::open(model_path).await?;
        let mut magic_bytes = [0u8; 8];

        use tokio::io::AsyncReadExt;
        file.read_exact(&mut magic_bytes).await?;

        // ONNX files are protobuf format, should start with reasonable protobuf bytes
        // This is a basic check - a more thorough validation would parse the protobuf
        let is_valid_format = magic_bytes[0] != 0 && magic_bytes.iter().any(|&b| b != 0);

        if is_valid_format {
            info!("✅ ONNX format validation successful: {:?}", model_path);
        } else {
            error!("❌ ONNX format validation failed: {:?}", model_path);
        }

        Ok(is_valid_format)
    }

    /// Validate model size constraints
    pub async fn validate_model_size(
        &self,
        model_path: &Path,
        expected_size_mb: u64,
        tolerance_percent: f64,
    ) -> Result<bool> {
        if !model_path.exists() {
            return Ok(false);
        }

        let metadata = tokio::fs::metadata(model_path).await?;
        let actual_size_mb = metadata.len() / 1024 / 1024;
        let expected_size_mb = expected_size_mb;

        let size_diff_percent = ((actual_size_mb as f64 - expected_size_mb as f64).abs()
            / expected_size_mb as f64)
            * 100.0;

        let is_valid = size_diff_percent <= tolerance_percent;

        if is_valid {
            info!(
                "✅ Model size validation successful: {:?} ({} MB)",
                model_path, actual_size_mb
            );
        } else {
            warn!("⚠️ Model size validation failed: {:?}", model_path);
            warn!(
                "   Expected: {} MB (±{:.1}%)",
                expected_size_mb, tolerance_percent
            );
            warn!(
                "   Actual:   {} MB ({:.1}% difference)",
                actual_size_mb, size_diff_percent
            );
        }

        Ok(is_valid)
    }

    /// Perform comprehensive model validation
    pub async fn validate_comprehensive(
        &self,
        model_path: &Path,
        expected_checksum: &str,
        expected_size_mb: u64,
    ) -> Result<ValidationResult> {
        let mut result = ValidationResult::default();

        // File existence check
        result.file_exists = model_path.exists();
        if !result.file_exists {
            error!("❌ Model file does not exist: {:?}", model_path);
            return Ok(result);
        }

        // Checksum validation
        match self.validate_model(model_path, expected_checksum).await {
            Ok(valid) => result.checksum_valid = valid,
            Err(e) => {
                error!("❌ Checksum validation error: {}", e);
                result.checksum_valid = false;
            }
        }

        // Format validation
        match self.validate_onnx_format(model_path).await {
            Ok(valid) => result.format_valid = valid,
            Err(e) => {
                error!("❌ Format validation error: {}", e);
                result.format_valid = false;
            }
        }

        // Size validation (10% tolerance)
        match self
            .validate_model_size(model_path, expected_size_mb, 10.0)
            .await
        {
            Ok(valid) => result.size_valid = valid,
            Err(e) => {
                error!("❌ Size validation error: {}", e);
                result.size_valid = false;
            }
        }

        result.overall_valid =
            result.file_exists && result.checksum_valid && result.format_valid && result.size_valid;

        if result.overall_valid {
            info!("🎉 Comprehensive validation successful: {:?}", model_path);
        } else {
            error!("❌ Comprehensive validation failed: {:?}", model_path);
        }

        Ok(result)
    }

    /// Quick validation for cached models
    pub async fn quick_validate(&self, model_path: &Path) -> Result<bool> {
        if !model_path.exists() {
            return Ok(false);
        }

        // Quick checks: file exists, has content, basic format
        let metadata = tokio::fs::metadata(model_path).await?;
        if metadata.len() < 1024 {
            return Ok(false);
        }

        // Check if it's an ONNX file
        if let Some(extension) = model_path.extension() {
            return Ok(extension == "onnx" || extension == "bin");
        }

        Ok(true)
    }

    /// Validate model dependencies and requirements
    pub async fn validate_model_requirements(
        &self,
        model_path: &Path,
        required_memory_mb: u64,
    ) -> Result<bool> {
        // Check available system memory
        let available_memory_mb = self.get_available_memory_mb()?;

        if available_memory_mb < required_memory_mb {
            warn!("⚠️ Insufficient memory for model: {:?}", model_path);
            warn!("   Required: {} MB", required_memory_mb);
            warn!("   Available: {} MB", available_memory_mb);
            return Ok(false);
        }

        info!(
            "✅ Memory requirements satisfied for model: {:?}",
            model_path
        );
        Ok(true)
    }

    /// Get available system memory in MB
    fn get_available_memory_mb(&self) -> Result<u64> {
        // This is a simplified implementation
        // In a real system, you'd check actual available memory
        #[cfg(target_os = "macos")]
        {
            // On macOS, we could use sysctl to get memory info
            // For now, return a conservative estimate
            Ok(8192) // 8GB
        }

        #[cfg(target_os = "linux")]
        {
            // On Linux, we could read /proc/meminfo
            // For now, return a conservative estimate
            Ok(8192) // 8GB
        }

        #[cfg(target_os = "windows")]
        {
            // On Windows, we could use GlobalMemoryStatus
            // For now, return a conservative estimate
            Ok(8192) // 8GB
        }

        #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
        {
            Ok(4096) // 4GB conservative estimate
        }
    }
}

/// Comprehensive validation result
#[derive(Debug, Default)]
pub struct ValidationResult {
    pub file_exists: bool,
    pub checksum_valid: bool,
    pub format_valid: bool,
    pub size_valid: bool,
    pub overall_valid: bool,
}

impl ValidationResult {
    /// Get validation summary
    pub fn summary(&self) -> String {
        let mut issues = Vec::new();

        if !self.file_exists {
            issues.push("file missing");
        }
        if !self.checksum_valid {
            issues.push("checksum mismatch");
        }
        if !self.format_valid {
            issues.push("invalid format");
        }
        if !self.size_valid {
            issues.push("size mismatch");
        }

        if issues.is_empty() {
            "All validations passed".to_string()
        } else {
            format!("Issues: {}", issues.join(", "))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_validator_creation() {
        let validator = ModelValidator::new().unwrap();
        // Just ensure it can be created
        let _ = validator;
    }

    #[tokio::test]
    async fn test_file_checksum() {
        let validator = ModelValidator::new().unwrap();

        // Create a temporary file with known content
        let mut temp_file = NamedTempFile::new().unwrap();
        write!(temp_file, "test content").unwrap();

        let checksum = validator
            .calculate_file_checksum(temp_file.path())
            .await
            .unwrap();
        assert!(!checksum.is_empty());
        assert_eq!(checksum.len(), 64); // SHA256 is 64 hex characters
    }

    #[tokio::test]
    async fn test_model_size_validation() {
        let validator = ModelValidator::new().unwrap();

        // Create a temporary file with known size
        let mut temp_file = NamedTempFile::new().unwrap();
        let content = vec![0u8; 1024 * 1024]; // 1MB
        temp_file.write_all(&content).unwrap();

        // Should pass with correct size and reasonable tolerance
        let is_valid = validator
            .validate_model_size(temp_file.path(), 1, 10.0)
            .await
            .unwrap();
        assert!(is_valid);

        // Should fail with very different size
        let is_valid = validator
            .validate_model_size(temp_file.path(), 100, 1.0)
            .await
            .unwrap();
        assert!(!is_valid);
    }

    #[tokio::test]
    async fn test_quick_validate() {
        let validator = ModelValidator::new().unwrap();

        // Non-existent file should fail
        let result = validator
            .quick_validate(Path::new("/nonexistent/model.onnx"))
            .await
            .unwrap();
        assert!(!result);

        // Create a valid-looking ONNX file
        let mut temp_file = NamedTempFile::new().unwrap();
        let content = vec![0u8; 2048]; // 2KB
        temp_file.write_all(&content).unwrap();

        // Rename to have .onnx extension
        let onnx_path = temp_file.path().with_extension("onnx");
        std::fs::copy(temp_file.path(), &onnx_path).unwrap();

        let result = validator.quick_validate(&onnx_path).await.unwrap();
        assert!(result);

        // Cleanup
        std::fs::remove_file(&onnx_path).ok();
    }

    #[test]
    fn test_validation_result() {
        let mut result = ValidationResult::default();
        assert!(!result.overall_valid);

        result.file_exists = true;
        result.checksum_valid = true;
        result.format_valid = true;
        result.size_valid = true;
        result.overall_valid = true;

        assert_eq!(result.summary(), "All validations passed");

        result.checksum_valid = false;
        assert!(result.summary().contains("checksum mismatch"));
    }
}
