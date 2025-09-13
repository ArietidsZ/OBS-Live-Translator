//! AI model integrations for speech recognition, translation and summarization

pub mod whisper_v3_turbo;
pub mod nllb_600m;
pub mod model_switcher;

pub use whisper_v3_turbo::*;
pub use nllb_600m::*;
pub use model_switcher::*;