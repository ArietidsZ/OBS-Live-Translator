//! Hardware acceleration detection and setup

pub mod detection;

/// Initialize hardware acceleration for the platform
pub fn initialize() -> crate::Result<()> {
    tracing::info!("Initializing hardware acceleration");

    #[cfg(feature = "cuda")]
    {
        tracing::info!("CUDA support enabled");
    }

    #[cfg(feature = "tensorrt")]
    {
        tracing::info!("TensorRT support enabled");
    }

    #[cfg(feature = "coreml")]
    {
        tracing::info!("CoreML support enabled");
    }

    #[cfg(feature = "openvino")]
    {
        tracing::info!("OpenVINO support enabled");
    }

    #[cfg(feature = "directml")]
    {
        tracing::info!("DirectML support enabled");
    }

    Ok(())
}
