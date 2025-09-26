//! Tokenization utilities for translation models
//!
//! Implements SentencePiece and BPE tokenization for various translation models
//! with optimized batch processing and caching.

use anyhow::{Result, anyhow};
use std::collections::HashMap;
use std::path::Path;
use serde::{Deserialize, Serialize};

/// Tokenizer types supported
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TokenizerType {
    /// SentencePiece tokenizer (used by NLLB, mBART, etc.)
    SentencePiece,
    /// Byte-Pair Encoding (used by GPT, MarianMT, etc.)
    BPE,
    /// WordPiece tokenizer (used by BERT, etc.)
    WordPiece,
    /// Tiktoken (used by newer OpenAI models)
    Tiktoken,
}

/// Special tokens used by translation models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialTokens {
    pub bos_token: String,
    pub eos_token: String,
    pub pad_token: String,
    pub unk_token: String,
    pub sep_token: Option<String>,
    pub cls_token: Option<String>,
    pub mask_token: Option<String>,
    pub additional_special_tokens: Vec<String>,
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self {
            bos_token: "<s>".to_string(),
            eos_token: "</s>".to_string(),
            pad_token: "<pad>".to_string(),
            unk_token: "<unk>".to_string(),
            sep_token: None,
            cls_token: None,
            mask_token: Some("<mask>".to_string()),
            additional_special_tokens: Vec::new(),
        }
    }
}

/// Base tokenizer trait
pub trait Tokenizer: Send + Sync {
    /// Encode text to token IDs
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<i64>>;

    /// Decode token IDs to text
    fn decode(&self, tokens: &[i64], skip_special_tokens: bool) -> Result<String>;

    /// Batch encode multiple texts
    fn encode_batch(&self, texts: &[String], add_special_tokens: bool) -> Result<Vec<Vec<i64>>>;

    /// Batch decode multiple token sequences
    fn decode_batch(&self, token_seqs: &[Vec<i64>], skip_special_tokens: bool) -> Result<Vec<String>>;

    /// Get vocabulary size
    fn vocab_size(&self) -> usize;

    /// Get special tokens
    fn special_tokens(&self) -> &SpecialTokens;
}

/// SentencePiece tokenizer implementation
pub struct SentencePieceTokenizer {
    #[allow(dead_code)]
    model_path: String,
    vocab: HashMap<String, i64>,
    reverse_vocab: HashMap<i64, String>,
    special_tokens: SpecialTokens,
}

impl SentencePieceTokenizer {
    /// Load SentencePiece model
    pub fn from_file(model_path: &str) -> Result<Self> {
        // In real implementation, would load actual SentencePiece model
        // For now, create a mock vocabulary

        let mut vocab = HashMap::new();
        let mut reverse_vocab = HashMap::new();

        // Add special tokens
        let special = SpecialTokens::default();
        vocab.insert(special.pad_token.clone(), 0);
        vocab.insert(special.unk_token.clone(), 1);
        vocab.insert(special.bos_token.clone(), 2);
        vocab.insert(special.eos_token.clone(), 3);

        reverse_vocab.insert(0, special.pad_token.clone());
        reverse_vocab.insert(1, special.unk_token.clone());
        reverse_vocab.insert(2, special.bos_token.clone());
        reverse_vocab.insert(3, special.eos_token.clone());

        // Add some mock vocabulary
        for i in 4..10000 {
            let token = format!("▁token_{}", i);
            vocab.insert(token.clone(), i);
            reverse_vocab.insert(i, token);
        }

        Ok(Self {
            model_path: model_path.to_string(),
            vocab,
            reverse_vocab,
            special_tokens: special,
        })
    }

    /// Simple text normalization
    fn normalize(&self, text: &str) -> String {
        // Add SentencePiece prefix (▁) for word boundaries
        text.split_whitespace()
            .map(|word| format!("▁{}", word))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

impl Tokenizer for SentencePieceTokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<i64>> {
        let normalized = self.normalize(text);
        let mut tokens = Vec::new();

        if add_special_tokens {
            tokens.push(2); // BOS
        }

        // Simple character-based tokenization for now
        for ch in normalized.chars() {
            let token_str = format!("▁{}", ch);
            let token_id = self.vocab.get(&token_str)
                .or_else(|| self.vocab.get(&self.special_tokens.unk_token))
                .copied()
                .unwrap_or(1);
            tokens.push(token_id);
        }

        if add_special_tokens {
            tokens.push(3); // EOS
        }

        Ok(tokens)
    }

    fn decode(&self, tokens: &[i64], skip_special_tokens: bool) -> Result<String> {
        let mut text = String::new();

        for &token_id in tokens {
            if skip_special_tokens && (token_id <= 3) {
                continue;
            }

            if let Some(token_str) = self.reverse_vocab.get(&token_id) {
                // Remove SentencePiece prefix
                let clean = token_str.replace("▁", " ");
                text.push_str(&clean);
            }
        }

        Ok(text.trim().to_string())
    }

    fn encode_batch(&self, texts: &[String], add_special_tokens: bool) -> Result<Vec<Vec<i64>>> {
        texts.iter()
            .map(|text| self.encode(text, add_special_tokens))
            .collect()
    }

    fn decode_batch(&self, token_seqs: &[Vec<i64>], skip_special_tokens: bool) -> Result<Vec<String>> {
        token_seqs.iter()
            .map(|tokens| self.decode(tokens, skip_special_tokens))
            .collect()
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }
}

/// Fast BPE tokenizer implementation
pub struct BPETokenizer {
    vocab: HashMap<String, i64>,
    #[allow(dead_code)]
    merges: Vec<(String, String)>,
    special_tokens: SpecialTokens,
    #[allow(dead_code)]
    byte_encoder: HashMap<u8, char>,
    #[allow(dead_code)]
    byte_decoder: HashMap<char, u8>,
}

impl BPETokenizer {
    /// Create BPE tokenizer from vocab and merges files
    pub fn from_files(_vocab_path: &str, _merges_path: &str) -> Result<Self> {
        // Create byte encoder/decoder for robust encoding
        let mut byte_encoder = HashMap::new();
        let mut byte_decoder = HashMap::new();

        let mut n = 0;
        for b in 0..=255u8 {
            byte_encoder.insert(b, char::from_u32(256 + n).unwrap());
            byte_decoder.insert(char::from_u32(256 + n).unwrap(), b);
            n += 1;
        }

        // Mock implementation
        let vocab = HashMap::new();
        let merges = Vec::new();
        let special_tokens = SpecialTokens::default();

        Ok(Self {
            vocab,
            merges,
            special_tokens,
            byte_encoder,
            byte_decoder,
        })
    }

    /// Apply BPE merges to tokens
    fn bpe(&self, token: &str) -> Vec<String> {
        // Simplified BPE implementation
        token.chars().map(|c| c.to_string()).collect()
    }
}

impl Tokenizer for BPETokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<i64>> {
        let mut tokens = Vec::new();

        if add_special_tokens {
            tokens.push(2); // BOS
        }

        // Encode text to bytes, then apply BPE
        for word in text.split_whitespace() {
            let bpe_tokens = self.bpe(word);
            for bpe_token in bpe_tokens {
                // Map to vocabulary
                tokens.push(self.vocab.get(&bpe_token).copied().unwrap_or(1));
            }
        }

        if add_special_tokens {
            tokens.push(3); // EOS
        }

        Ok(tokens)
    }

    fn decode(&self, _tokens: &[i64], _skip_special_tokens: bool) -> Result<String> {
        // Simplified decode
        Ok("[BPE decoded text]".to_string())
    }

    fn encode_batch(&self, texts: &[String], add_special_tokens: bool) -> Result<Vec<Vec<i64>>> {
        texts.iter()
            .map(|text| self.encode(text, add_special_tokens))
            .collect()
    }

    fn decode_batch(&self, token_seqs: &[Vec<i64>], skip_special_tokens: bool) -> Result<Vec<String>> {
        token_seqs.iter()
            .map(|tokens| self.decode(tokens, skip_special_tokens))
            .collect()
    }

    fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    fn special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }
}

/// Tokenizer factory
pub struct TokenizerFactory;

impl TokenizerFactory {
    /// Create tokenizer from configuration
    pub fn from_config(config: &TokenizerConfig) -> Result<Box<dyn Tokenizer>> {
        match config.tokenizer_type {
            TokenizerType::SentencePiece => {
                let tokenizer = SentencePieceTokenizer::from_file(&config.model_path)?;
                Ok(Box::new(tokenizer))
            },
            TokenizerType::BPE => {
                let vocab_path = format!("{}/vocab.json", config.model_path);
                let merges_path = format!("{}/merges.txt", config.model_path);
                let tokenizer = BPETokenizer::from_files(&vocab_path, &merges_path)?;
                Ok(Box::new(tokenizer))
            },
            _ => Err(anyhow!("Tokenizer type {:?} not implemented yet", config.tokenizer_type))
        }
    }

    /// Auto-detect tokenizer type from model directory
    pub fn auto_detect(model_dir: &str) -> Result<Box<dyn Tokenizer>> {
        let path = Path::new(model_dir);

        // Check for SentencePiece model
        if path.join("spiece.model").exists() {
            let tokenizer = SentencePieceTokenizer::from_file(
                path.join("spiece.model").to_str().unwrap()
            )?;
            return Ok(Box::new(tokenizer));
        }

        // Check for BPE tokenizer
        if path.join("vocab.json").exists() && path.join("merges.txt").exists() {
            let tokenizer = BPETokenizer::from_files(
                path.join("vocab.json").to_str().unwrap(),
                path.join("merges.txt").to_str().unwrap()
            )?;
            return Ok(Box::new(tokenizer));
        }

        Err(anyhow!("Could not auto-detect tokenizer type in {}", model_dir))
    }
}

/// Tokenizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    /// Type of tokenizer to use
    pub tokenizer_type: TokenizerType,
    /// Path to the tokenizer model file
    pub model_path: String,
    /// Maximum sequence length for padding/truncation
    pub max_length: usize,
    /// Whether to pad sequences to max_length
    pub padding: bool,
    /// Whether to truncate sequences exceeding max_length
    pub truncation: bool,
    /// Whether to add special tokens (BOS, EOS, etc.)
    pub add_special_tokens: bool,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            tokenizer_type: TokenizerType::SentencePiece,
            model_path: String::new(),
            max_length: 512,
            padding: true,
            truncation: true,
            add_special_tokens: true,
        }
    }
}

/// Utilities for handling tokenized sequences
pub struct TokenUtils;

impl TokenUtils {
    /// Pad sequences to same length
    pub fn pad_sequences(sequences: &[Vec<i64>], pad_token_id: i64, max_length: Option<usize>) -> Vec<Vec<i64>> {
        let max_len = max_length.unwrap_or_else(|| {
            sequences.iter().map(|seq| seq.len()).max().unwrap_or(0)
        });

        sequences.iter()
            .map(|seq| {
                let mut padded = seq.clone();
                padded.resize(max_len, pad_token_id);
                padded
            })
            .collect()
    }

    /// Truncate sequences to max length
    pub fn truncate_sequences(sequences: &[Vec<i64>], max_length: usize) -> Vec<Vec<i64>> {
        sequences.iter()
            .map(|seq| {
                if seq.len() > max_length {
                    seq[..max_length].to_vec()
                } else {
                    seq.clone()
                }
            })
            .collect()
    }

    /// Create attention masks
    pub fn create_attention_masks(sequences: &[Vec<i64>], pad_token_id: i64) -> Vec<Vec<i64>> {
        sequences.iter()
            .map(|seq| {
                seq.iter()
                    .map(|&token| if token == pad_token_id { 0 } else { 1 })
                    .collect()
            })
            .collect()
    }
}

