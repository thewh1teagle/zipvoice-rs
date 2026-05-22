use std::collections::HashMap;

use super::{Result, ZipVoiceError};

pub struct Tokenizer {
    token_to_id: HashMap<char, i64>,
}

impl Tokenizer {
    pub fn from_tokens_txt(tokens_txt: &str) -> Result<Self> {
        let mut token_to_id = HashMap::new();
        for line in tokens_txt.lines() {
            if line.is_empty() {
                continue;
            }
            let (token, id) = line
                .split_once('\t')
                .ok_or_else(|| ZipVoiceError::Tokenizer(format!("bad token line: {line}")))?;
            let mut chars = token.chars();
            let Some(ch) = chars.next() else {
                continue;
            };
            if chars.next().is_some() {
                continue;
            }
            let id = id
                .parse::<i64>()
                .map_err(|_| ZipVoiceError::Tokenizer(format!("bad token id: {line}")))?;
            token_to_id.insert(ch, id);
        }
        Ok(Self { token_to_id })
    }

    pub fn encode_chars(&self, text: &str) -> Vec<i64> {
        text.trim()
            .chars()
            .filter_map(|ch| self.token_to_id.get(&ch).copied())
            .collect()
    }
}
