//! Claude API - Premium translation using Anthropic's Claude 3.5 Sonnet
//!
//! #1 in WMT24 competition, best contextual accuracy

use crate::{translation::TranslationEngine, types::TranslatorConfig, Error, Result};
use async_trait::async_trait;
use serde_json::json;

/// Claude API translation engine
pub struct ClaudeAPI {
    client: reqwest::Client,
    api_key: String,
    model: String,
}

impl ClaudeAPI {
    /// Create a new Claude API instance
    pub async fn new(config: &TranslatorConfig) -> Result<Self> {
        tracing::info!("Initializing Claude API for translation");

        let api_key = config
            .claude_api_key
            .as_ref()
            .ok_or_else(|| Error::Config("Claude API key not provided".to_string()))?
            .clone();

        let client = reqwest::Client::new();

        Ok(Self {
            client,
            api_key,
            model: "claude-3-5-sonnet-20241022".to_string(),
        })
    }
}

#[async_trait]
impl TranslationEngine for ClaudeAPI {
    async fn translate(&self, text: &str, source_lang: &str, target_lang: &str) -> Result<String> {
        let prompt = format!(
            "Translate the following text from {source_lang} to {target_lang}. Provide only the translation, no explanations:\n\n{text}"
        );

        let request_body = json!({
            "model": self.model,
            "max_tokens": 1024,
            "messages": [{
                "role": "user",
                "content": prompt
            }]
        });

        let response = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(Error::Network)?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(Error::Translation(format!(
                "Claude API error {status}: {error_text}"
            )));
        }

        let response_json: serde_json::Value = response.json().await?;

        let translation = response_json["content"][0]["text"]
            .as_str()
            .ok_or_else(|| Error::Translation("Invalid response from Claude API".to_string()))?
            .to_string();

        Ok(translation)
    }

    fn model_name(&self) -> &str {
        "Claude 3.5 Sonnet (API)"
    }

    fn language_count(&self) -> usize {
        100 // Claude supports 100+ languages
    }
}
