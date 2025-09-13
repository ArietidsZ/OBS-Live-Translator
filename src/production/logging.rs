use tracing::{info, warn, error, debug, Level};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};
use std::io;

#[derive(Debug, Clone)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

pub struct ProductionLogger;

impl ProductionLogger {
    pub fn init(level: LogLevel) {
        let level = match level {
            LogLevel::Trace => Level::TRACE,
            LogLevel::Debug => Level::DEBUG,
            LogLevel::Info => Level::INFO,
            LogLevel::Warn => Level::WARN,
            LogLevel::Error => Level::ERROR,
        };

        let env_filter = EnvFilter::new(format!("obs_live_translator={}", level));

        tracing_subscriber::registry()
            .with(env_filter)
            .with(tracing_subscriber::fmt::layer())
            .init();
    }
}

pub fn setup_tracing() {
    ProductionLogger::init(LogLevel::Info);
}