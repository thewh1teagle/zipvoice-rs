use std::collections::HashMap;
use std::path::Path;

use llama_rs_sys as ffi;
use serde::Deserialize;

use crate::gguf::{GgmlWeights, GgufModel};

use super::{Result, ZipVoiceError};

#[derive(Debug, Deserialize)]
pub struct ModelJson {
    pub model: ModelConfig,
    pub feature: FeatureConfig,
}

#[derive(Debug, Deserialize)]
pub struct FeatureConfig {
    pub sampling_rate: u32,
    #[serde(rename = "type")]
    pub feature_type: String,
}

#[derive(Debug, Deserialize)]
pub struct ModelConfig {
    pub fm_decoder_downsampling_factor: Vec<u32>,
    pub fm_decoder_num_layers: Vec<u32>,
    pub fm_decoder_cnn_module_kernel: Vec<u32>,
    pub fm_decoder_feedforward_dim: u32,
    pub fm_decoder_num_heads: u32,
    pub fm_decoder_dim: u32,
    pub text_encoder_num_layers: u32,
    pub text_encoder_feedforward_dim: u32,
    pub text_encoder_cnn_module_kernel: u32,
    pub text_encoder_num_heads: u32,
    pub text_encoder_dim: u32,
    pub query_head_dim: u32,
    pub value_head_dim: u32,
    pub pos_head_dim: u32,
    pub pos_dim: u32,
    pub time_embed_dim: u32,
    pub text_embed_dim: u32,
    pub feat_dim: u32,
}

pub struct ZipVoiceModel {
    gguf: GgufModel,
    weights: GgmlWeights,
    config: ModelJson,
    tokens_txt: String,
    tensor_name_map: HashMap<String, String>,
    vocab_size: u32,
    pad_id: u32,
}

impl ZipVoiceModel {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let gguf = GgufModel::open(path)?;
        let arch = required_string(&gguf, "zipvoice.arch")?;
        if arch != "zipvoice" {
            return Err(ZipVoiceError::MissingMetadata(format!(
                "zipvoice.arch=zipvoice, got {arch}"
            )));
        }
        let config: ModelJson =
            serde_json::from_str(&required_string(&gguf, "zipvoice.model_json")?)?;
        let tokens_txt = required_string(&gguf, "zipvoice.tokens_txt")?;
        let tensor_name_map =
            serde_json::from_str(&required_string(&gguf, "zipvoice.tensor_name_map")?)?;
        let vocab_size = required_u32(&gguf, "zipvoice.vocab_size")?;
        let pad_id = required_u32(&gguf, "zipvoice.pad_id")?;
        let weights = GgmlWeights::load_all(&gguf)?;
        Ok(Self {
            gguf,
            weights,
            config,
            tokens_txt,
            tensor_name_map,
            vocab_size,
            pad_id,
        })
    }

    pub fn config(&self) -> &ModelJson {
        &self.config
    }

    pub fn tokens_txt(&self) -> &str {
        &self.tokens_txt
    }

    pub fn tensor_count(&self) -> i64 {
        self.gguf.tensor_count()
    }

    pub fn tensor_short_name(&self, full_name: &str) -> Option<&str> {
        self.tensor_name_map
            .iter()
            .find_map(|(short, full)| (full == full_name).then_some(short.as_str()))
    }

    pub fn tensor_f32(&self, short_name: &str) -> Result<Vec<f32>> {
        Ok(self.gguf.tensor_f32_by_name(short_name)?)
    }

    pub fn tensor_f32_full(&self, full_name: &str) -> Result<Vec<f32>> {
        let short_name = self
            .tensor_short_name(full_name)
            .ok_or_else(|| ZipVoiceError::MissingMetadata(format!("tensor {full_name}")))?;
        self.tensor_f32(short_name)
    }

    pub fn weight_tensor_full(&self, full_name: &str) -> Result<*mut ffi::ggml_tensor> {
        let short_name = self
            .tensor_short_name(full_name)
            .ok_or_else(|| ZipVoiceError::MissingMetadata(format!("tensor {full_name}")))?;
        Ok(self.weights.tensor(short_name)?)
    }

    pub fn backend(&self) -> ffi::ggml_backend_t {
        self.weights.backend()
    }

    pub fn alloc_graph(&self, graph: *mut ffi::ggml_cgraph) -> Result<()> {
        Ok(self.weights.alloc_graph(graph)?)
    }

    pub fn compute_graph(&self, graph: *mut ffi::ggml_cgraph) -> Result<()> {
        Ok(self.weights.compute_graph(graph)?)
    }

    pub fn reset_scheduler(&self) {
        self.weights.reset_scheduler();
    }

    pub fn backend_name(&self) -> String {
        self.weights.backend_name()
    }

    pub fn vocab_size(&self) -> u32 {
        self.vocab_size
    }

    pub fn pad_id(&self) -> u32 {
        self.pad_id
    }

    pub fn text_embed_dim(&self) -> usize {
        self.config.model.text_embed_dim as usize
    }

    pub fn feat_dim(&self) -> usize {
        self.config.model.feat_dim as usize
    }

    pub fn fm_decoder_dim(&self) -> usize {
        self.config.model.fm_decoder_dim as usize
    }
}

fn required_string(gguf: &GgufModel, key: &str) -> Result<String> {
    gguf.get_string(key)?
        .ok_or_else(|| ZipVoiceError::MissingMetadata(key.into()))
}

fn required_u32(gguf: &GgufModel, key: &str) -> Result<u32> {
    gguf.get_u32(key)?
        .ok_or_else(|| ZipVoiceError::MissingMetadata(key.into()))
}
