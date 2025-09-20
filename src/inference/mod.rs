//! Model inference engine with ONNX Runtime support

pub mod engine;
pub mod whisper;
pub mod translation;
pub mod batch;

pub use engine::{InferenceEngine, InferenceConfig, InferenceResult};
pub use whisper::WhisperModel;
pub use translation::TranslationModel;
pub use batch::BatchProcessor;


/// Supported model types
#[derive(Debug, Clone, PartialEq)]
pub enum ModelType {
    Whisper,
    Translation,
    Custom(String),
}

/// Inference session configuration
#[derive(Debug, Clone)]
pub struct SessionConfig {
    pub model_path: String,
    pub model_type: ModelType,
    pub device: Device,
    pub batch_size: usize,
    pub threads: Option<usize>,
    pub optimization_level: OptimizationLevel,
}

/// Execution device
#[derive(Debug, Clone, PartialEq)]
pub enum Device {
    CPU,
    CUDA(u32),   // GPU ID
    CoreML,      // Apple Silicon
    DirectML,    // Windows
}

/// Model optimization level
#[derive(Debug, Clone, PartialEq)]
pub enum OptimizationLevel {
    None,
    Basic,
    Extended,
    All,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            model_type: ModelType::Whisper,
            device: Device::CPU,
            batch_size: 1,
            threads: None,
            optimization_level: OptimizationLevel::All,
        }
    }
}

/// Model metadata
#[derive(Debug, Clone)]
pub struct ModelMetadata {
    pub name: String,
    pub version: String,
    pub description: String,
    pub input_shapes: Vec<Vec<i64>>,
    pub output_shapes: Vec<Vec<i64>>,
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
}

/// Inference timing information
#[derive(Debug, Clone)]
pub struct TimingInfo {
    pub preprocessing_ms: f32,
    pub inference_ms: f32,
    pub postprocessing_ms: f32,
    pub total_ms: f32,
}

impl TimingInfo {
    pub fn new() -> Self {
        Self {
            preprocessing_ms: 0.0,
            inference_ms: 0.0,
            postprocessing_ms: 0.0,
            total_ms: 0.0,
        }
    }

    pub fn add_preprocessing(&mut self, duration: f32) {
        self.preprocessing_ms += duration;
        self.update_total();
    }

    pub fn add_inference(&mut self, duration: f32) {
        self.inference_ms += duration;
        self.update_total();
    }

    pub fn add_postprocessing(&mut self, duration: f32) {
        self.postprocessing_ms += duration;
        self.update_total();
    }

    fn update_total(&mut self) {
        self.total_ms = self.preprocessing_ms + self.inference_ms + self.postprocessing_ms;
    }
}