pub mod audio;
pub mod flow_matching;
pub mod model;
pub mod text_encoder;
pub mod tokenizer;

use std::path::Path;

use thiserror::Error;

use crate::gguf::{GgufError, set_ggml_verbose};
use crate::vocos::{Vocos, VocosError};

use self::audio::{FEAT_SCALE, postprocess_generated_audio, prepare_prompt_audio};

pub use model::ZipVoiceModel;

const DEFAULT_NUM_STEPS: usize = 8;
const DEFAULT_T_SHIFT: f32 = 0.5;
const DEFAULT_GUIDANCE_SCALE: f32 = 1.0;
const DEFAULT_SEED: u64 = 42;

#[derive(Debug, Clone, Copy)]
pub struct CreateOptions {
    pub speed: f32,
    pub num_steps: usize,
    pub t_shift: f32,
    pub guidance_scale: f32,
    pub seed: u64,
    pub verbose: bool,
}

impl Default for CreateOptions {
    fn default() -> Self {
        Self {
            speed: 1.0,
            num_steps: DEFAULT_NUM_STEPS,
            t_shift: DEFAULT_T_SHIFT,
            guidance_scale: DEFAULT_GUIDANCE_SCALE,
            seed: DEFAULT_SEED,
            verbose: false,
        }
    }
}

#[derive(Debug, Error)]
pub enum ZipVoiceError {
    #[error("gguf error: {0}")]
    Gguf(#[from] GgufError),
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("missing metadata key: {0}")]
    MissingMetadata(String),
    #[error("tokenizer error: {0}")]
    Tokenizer(String),
    #[error("ggml error: {0}")]
    Ggml(String),
    #[error("vocos error: {0}")]
    Vocos(#[from] VocosError),
    #[error("vocos model is not loaded")]
    MissingVocos,
}

pub type Result<T> = std::result::Result<T, ZipVoiceError>;

pub struct ZipVoice {
    model: ZipVoiceModel,
    tokenizer: tokenizer::Tokenizer,
    vocos: Option<Vocos>,
}

impl ZipVoice {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let model = ZipVoiceModel::load(path)?;
        let tokenizer = tokenizer::Tokenizer::from_tokens_txt(model.tokens_txt())?;
        Ok(Self {
            model,
            tokenizer,
            vocos: None,
        })
    }

    pub fn load_with_vocos(path: impl AsRef<Path>, vocos_path: impl AsRef<Path>) -> Result<Self> {
        let mut zipvoice = Self::load(path)?;
        zipvoice.vocos = Some(Vocos::load(vocos_path)?);
        Ok(zipvoice)
    }

    pub fn tokenize_phonemes(&self, phonemes: &str) -> Vec<i64> {
        self.tokenizer.encode_chars(phonemes)
    }

    pub fn token_embeddings(&self, token_ids: &[i64]) -> Result<Vec<f32>> {
        text_encoder::token_embeddings(&self.model, token_ids)
    }

    pub fn text_input_projection(&self, token_ids: &[i64]) -> Result<Vec<f32>> {
        text_encoder::input_projection(&self.model, token_ids)
    }

    pub fn text_first_feed_forward(&self, token_ids: &[i64]) -> Result<Vec<f32>> {
        text_encoder::first_feed_forward(&self.model, token_ids)
    }

    pub fn text_first_conv_module(&self, token_ids: &[i64]) -> Result<Vec<f32>> {
        text_encoder::first_conv_module(&self.model, token_ids)
    }

    pub fn text_first_layer_no_attention(&self, token_ids: &[i64]) -> Result<Vec<f32>> {
        text_encoder::first_layer_no_attention(&self.model, token_ids)
    }

    pub fn text_first_layer_no_attention_norm(&self, token_ids: &[i64]) -> Result<Vec<f32>> {
        text_encoder::first_layer_no_attention_norm(&self.model, token_ids)
    }

    pub fn text_first_layer_no_attention_out_projection(
        &self,
        token_ids: &[i64],
    ) -> Result<Vec<f32>> {
        text_encoder::first_layer_no_attention_out_projection(&self.model, token_ids)
    }

    pub fn text_condition_preview(
        &self,
        prompt_phonemes: &str,
        target_phonemes: &str,
        prompt_feature_frames: usize,
        speed: f32,
    ) -> Result<Vec<f32>> {
        let prompt_tokens = self.tokenize_phonemes(prompt_phonemes);
        let target_tokens = self.tokenize_phonemes(target_phonemes);
        text_encoder::text_condition_preview(
            &self.model,
            &prompt_tokens,
            &target_tokens,
            prompt_feature_frames,
            speed,
        )
    }

    pub fn flow_input_projection(
        &self,
        x: &[f32],
        text_condition: &[f32],
        speech_condition: &[f32],
        frames: usize,
    ) -> Result<Vec<f32>> {
        flow_matching::input_projection(&self.model, x, text_condition, speech_condition, frames)
    }

    pub fn flow_velocity_preview(
        &self,
        t: f32,
        x: &[f32],
        text_condition: &[f32],
        speech_condition: &[f32],
        frames: usize,
    ) -> Result<Vec<f32>> {
        flow_matching::velocity_preview(&self.model, t, x, text_condition, speech_condition, frames)
    }

    pub fn flow_guided_velocity_preview(
        &self,
        t: f32,
        x: &[f32],
        text_condition: &[f32],
        speech_condition: &[f32],
        frames: usize,
        guidance_scale: f32,
    ) -> Result<Vec<f32>> {
        flow_matching::guided_velocity_preview(
            &self.model,
            t,
            x,
            text_condition,
            speech_condition,
            frames,
            guidance_scale,
        )
    }

    pub fn flow_sample_preview(
        &self,
        text_condition: &[f32],
        speech_condition: &[f32],
        frames: usize,
        num_steps: usize,
        t_shift: f32,
        guidance_scale: f32,
        seed: u64,
    ) -> Result<Vec<f32>> {
        flow_matching::sample_preview(
            &self.model,
            text_condition,
            speech_condition,
            frames,
            num_steps,
            t_shift,
            guidance_scale,
            seed,
        )
    }

    pub fn create(
        &self,
        ref_wav: impl AsRef<Path>,
        ref_phonemes: &str,
        target_phonemes: &str,
    ) -> Result<(Vec<f32>, u32)> {
        self.create_with_options(
            ref_wav,
            ref_phonemes,
            target_phonemes,
            CreateOptions::default(),
        )
    }

    pub fn create_with_options(
        &self,
        ref_wav: impl AsRef<Path>,
        ref_phonemes: &str,
        target_phonemes: &str,
        options: CreateOptions,
    ) -> Result<(Vec<f32>, u32)> {
        let vocos = self.vocos.as_ref().ok_or(ZipVoiceError::MissingVocos)?;
        self.create_with_vocos_options(vocos, ref_wav, ref_phonemes, target_phonemes, options)
    }

    pub fn create_with_vocos(
        &self,
        vocos: &Vocos,
        ref_wav: impl AsRef<Path>,
        ref_phonemes: &str,
        target_phonemes: &str,
    ) -> Result<(Vec<f32>, u32)> {
        self.create_with_vocos_options(
            vocos,
            ref_wav,
            ref_phonemes,
            target_phonemes,
            CreateOptions::default(),
        )
    }

    pub fn create_with_vocos_options(
        &self,
        vocos: &Vocos,
        ref_wav: impl AsRef<Path>,
        ref_phonemes: &str,
        target_phonemes: &str,
        options: CreateOptions,
    ) -> Result<(Vec<f32>, u32)> {
        let _verbose_guard = GgmlVerboseGuard::new(options.verbose);
        let prompt_samples = Vocos::load_wav_mono_24khz(ref_wav)?;
        let prompt_audio = prepare_prompt_audio(&prompt_samples);
        let prompt_features = vocos.encode_samples_24khz(&prompt_audio.samples)?;
        let prompt_frames = prompt_features.len() / self.model.feat_dim();

        let text_condition = self.text_condition_preview(
            ref_phonemes,
            target_phonemes,
            prompt_frames,
            options.speed,
        )?;
        let plan = self.plan(ref_phonemes, target_phonemes, prompt_frames, options.speed);

        let scaled_prompt = prompt_features
            .iter()
            .map(|value| value * FEAT_SCALE)
            .collect::<Vec<_>>();
        let mut speech_condition = vec![0.0_f32; plan.total_frames * self.model.feat_dim()];
        speech_condition[..scaled_prompt.len()].copy_from_slice(&scaled_prompt);

        let sampled_features = self.flow_sample_preview(
            &text_condition,
            &speech_condition,
            plan.total_frames,
            options.num_steps,
            options.t_shift,
            options.guidance_scale,
            options.seed,
        )?;
        let generated_mel = sampled_features[prompt_features.len()..]
            .iter()
            .map(|value| value / FEAT_SCALE)
            .collect::<Vec<_>>();
        let wav = vocos.decode_mel_samples_24khz(&generated_mel)?;

        Ok((
            postprocess_generated_audio(wav, prompt_audio.original_rms),
            self.model.config().feature.sampling_rate,
        ))
    }

    pub fn plan(
        &self,
        prompt_phonemes: &str,
        target_phonemes: &str,
        prompt_feature_frames: usize,
        speed: f32,
    ) -> text_encoder::TextPlan {
        let prompt_tokens = self.tokenize_phonemes(prompt_phonemes);
        let target_tokens = self.tokenize_phonemes(target_phonemes);
        text_encoder::plan_text_condition(
            prompt_tokens.len(),
            target_tokens.len(),
            prompt_feature_frames,
            speed,
        )
    }

    pub fn model(&self) -> &ZipVoiceModel {
        &self.model
    }
}

struct GgmlVerboseGuard;

impl GgmlVerboseGuard {
    fn new(verbose: bool) -> Self {
        set_ggml_verbose(verbose);
        Self
    }
}

impl Drop for GgmlVerboseGuard {
    fn drop(&mut self) {
        set_ggml_verbose(false);
    }
}
