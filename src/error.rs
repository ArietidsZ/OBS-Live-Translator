//! Error types for the translator

use thiserror::Error;

/// Main error type
#[derive(Error, Debug)]
pub enum Error {
    #[error("Audio processing error: {0}")]
    Audio(String),

    #[error("VAD error: {0}")]
    VAD(String),

    #[error("ASR error: {0}")]
    ASR(String),

    #[error("Language detection error: {0}")]
    LanguageDetection(String),

    #[error("Translation error: {0}")]
    Translation(String),

    #[error("Model loading error: {0}")]
    ModelLoad(String),

    #[error("ONNX Runtime error: {0}")]
    ONNXRuntime(#[from] ort::Error),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Tokenization error: {0}")]
    Tokenization(String),

    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("TOML deserialization error: {0}")]
    TomlDe(#[from] toml::de::Error),

    #[error("TOML serialization error: {0}")]
    TomlSer(#[from] toml::ser::Error),

    #[error("Hardware acceleration error: {0}")]
    Acceleration(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),

    #[error("Not supported: {0}")]
    NotSupported(String),
}

/// Result type alias
pub type Result<T> = std::result::Result<T, Error>;
