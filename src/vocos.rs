use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::ptr;
use std::sync::OnceLock;

use byteorder::{LittleEndian, ReadBytesExt};
use llama_rs_sys as ffi;
use num_complex::Complex32;
use rubato::{FftFixedInOut, Resampler};
use rustfft::FftPlanner;
use thiserror::Error;

use crate::gguf::{GgmlWeights, GgufError, GgufModel};

const SAMPLE_RATE: usize = 24_000;
const N_FFT: usize = 1024;
const HOP: usize = 256;
const N_MELS: usize = 100;
const DIM: usize = 512;
const HIDDEN: usize = 1536;
const LAYERS: usize = 8;

#[derive(Debug, Error)]
pub enum VocosError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("wav error: {0}")]
    Wav(#[from] hound::Error),
    #[error("resample error: {0}")]
    Resample(String),
    #[error("invalid model file: {0}")]
    InvalidModel(String),
    #[error("ggml error: {0}")]
    Ggml(String),
    #[error("gguf error: {0}")]
    Gguf(#[from] GgufError),
    #[error("missing tensor {0}")]
    MissingTensor(String),
}

type Result<T> = std::result::Result<T, VocosError>;

#[derive(Clone)]
struct Tensor {
    data: Vec<f32>,
}

pub struct Vocos {
    tensors: HashMap<String, Tensor>,
    weights: Option<GgmlWeights>,
}

impl Vocos {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if path.extension().is_some_and(|ext| ext == "gguf") {
            return Self::load_gguf(path);
        }
        Self::load_vbin(path)
    }

    fn load_gguf(path: &Path) -> Result<Self> {
        let model = GgufModel::open(path)?;
        let arch = model.get_string("vocos.arch")?;
        if arch.as_deref() != Some("vocos") {
            return Err(VocosError::InvalidModel(format!(
                "expected vocos.arch=vocos, got {arch:?}"
            )));
        }
        let mut tensors = HashMap::with_capacity(model.tensor_count() as usize);
        for info in model.tensors() {
            let info = info?;
            let data = model.tensor_f32_by_name(&info.name)?;
            tensors.insert(info.name, Tensor { data });
        }
        let weights = GgmlWeights::load_all(&model)?;
        Ok(Self {
            tensors,
            weights: Some(weights),
        })
    }

    fn load_vbin(path: &Path) -> Result<Self> {
        let mut reader = BufReader::new(File::open(path)?);
        let mut magic = [0_u8; 8];
        reader.read_exact(&mut magic)?;
        if &magic != b"VOCOSRS1" {
            return Err(VocosError::InvalidModel("bad magic".into()));
        }
        let count = reader.read_u32::<LittleEndian>()? as usize;
        let mut tensors = HashMap::with_capacity(count);
        for _ in 0..count {
            let name_len = reader.read_u16::<LittleEndian>()? as usize;
            let mut name = vec![0_u8; name_len];
            reader.read_exact(&mut name)?;
            let name = String::from_utf8(name)
                .map_err(|_| VocosError::InvalidModel("non-utf8 tensor name".into()))?;
            let ndim = reader.read_u8()? as usize;
            for _ in 0..ndim {
                let _ = reader.read_u32::<LittleEndian>()?;
            }
            let len = reader.read_u64::<LittleEndian>()? as usize;
            let mut data = vec![0.0_f32; len];
            for x in &mut data {
                *x = reader.read_f32::<LittleEndian>()?;
            }
            tensors.insert(name, Tensor { data });
        }
        Ok(Self {
            tensors,
            weights: None,
        })
    }

    pub fn encode(&self, input_wav: impl AsRef<Path>) -> Result<Vec<f32>> {
        let audio = load_wav_mono_24khz(input_wav)?;
        self.encode_samples_24khz(&audio)
    }

    pub fn encode_wav(&self, input_wav: impl AsRef<Path>) -> Result<Vec<f32>> {
        self.encode(input_wav)
    }

    pub fn encode_samples_24khz(&self, audio: &[f32]) -> Result<Vec<f32>> {
        self.log_mel(audio)
    }

    pub fn load_wav_mono_24khz(input_wav: impl AsRef<Path>) -> Result<Vec<f32>> {
        load_wav_mono_24khz(input_wav)
    }

    pub fn decode_mel_samples_24khz(&self, mel: &[f32]) -> Result<Vec<f32>> {
        self.decode_mel(mel)
    }

    pub fn decode(&self, mel: &[f32]) -> Result<Vec<f32>> {
        self.decode_mel_samples_24khz(mel)
    }

    fn tensor(&self, name: &str) -> Result<&[f32]> {
        self.tensors
            .get(name)
            .map(|t| t.data.as_slice())
            .ok_or_else(|| VocosError::MissingTensor(name.into()))
    }

    fn log_mel(&self, audio: &[f32]) -> Result<Vec<f32>> {
        let window = self.tensor("feature_extractor.mel_spec.spectrogram.window")?;
        let mel_filters = librosa_htk_mel_filters();
        let pad = N_FFT / 2;
        let padded = reflect_pad(audio, pad);
        let frames = (padded.len().saturating_sub(N_FFT)) / HOP + 1;
        let bins = N_FFT / 2 + 1;
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);
        let mut scratch = vec![Complex32::new(0.0, 0.0); N_FFT];
        let mut out = vec![0.0_f32; frames * N_MELS];

        for t in 0..frames {
            let start = t * HOP;
            for i in 0..N_FFT {
                scratch[i] = Complex32::new(padded[start + i] * window[i], 0.0);
            }
            fft.process(&mut scratch);
            let mut mag = [0.0_f32; N_FFT / 2 + 1];
            for b in 0..bins {
                mag[b] = scratch[b].norm();
            }
            for m in 0..N_MELS {
                let mut sum = 0.0;
                for b in 0..bins {
                    sum += mag[b] * mel_filters[m * bins + b];
                }
                out[t * N_MELS + m] = sum.max(1e-7).ln();
            }
        }
        Ok(out)
    }

    fn decode_mel(&self, mel: &[f32]) -> Result<Vec<f32>> {
        let frames = mel.len() / N_MELS;
        let spec = self.decode_mel_ggml(mel, frames)?;
        Ok(istft_from_head(
            &spec,
            frames,
            self.tensor("head.istft.window")?,
        ))
    }

    fn decode_mel_ggml(&self, mel: &[f32], frames: usize) -> Result<Vec<f32>> {
        if let Some(weights) = &self.weights {
            return self.decode_mel_ggml_weights(mel, frames, weights);
        }
        self.decode_mel_ggml_cpu_weights(mel, frames)
    }

    fn decode_mel_ggml_weights(
        &self,
        mel: &[f32],
        frames: usize,
        weights: &GgmlWeights,
    ) -> Result<Vec<f32>> {
        if mel.len() != frames * N_MELS {
            return Err(VocosError::Ggml("mel tensor has invalid shape".into()));
        }
        let mut graph = VocosGraph::new_with_backend(frames, weights.backend(), false)?;
        let mut h = graph.input_2d(N_MELS, frames);
        let inputs = vec![(h, mel)];

        h = graph.conv1d_tensor(
            h,
            N_MELS,
            DIM,
            7,
            false,
            weights.tensor("backbone.embed.weight")?,
            weights.tensor("backbone.embed.bias")?,
        );
        h = graph.layer_norm_affine_tensor(
            h,
            weights.tensor("backbone.norm.weight")?,
            weights.tensor("backbone.norm.bias")?,
        );

        for layer in 0..LAYERS {
            let p = format!("backbone.convnext.{layer}");
            let residual = h;
            h = graph.conv1d_tensor(
                h,
                DIM,
                DIM,
                7,
                true,
                weights.tensor(&format!("{p}.dwconv.weight"))?,
                weights.tensor(&format!("{p}.dwconv.bias"))?,
            );
            h = graph.layer_norm_affine_tensor(
                h,
                weights.tensor(&format!("{p}.norm.weight"))?,
                weights.tensor(&format!("{p}.norm.bias"))?,
            );
            h = graph.linear_tensor(
                h,
                weights.tensor(&format!("{p}.pwconv1.weight"))?,
                weights.tensor(&format!("{p}.pwconv1.bias"))?,
                true,
            );
            h = graph.linear_tensor(
                h,
                weights.tensor(&format!("{p}.pwconv2.weight"))?,
                weights.tensor(&format!("{p}.pwconv2.bias"))?,
                false,
            );
            h = graph.residual_gamma_tensor(residual, h, weights.tensor(&format!("{p}.gamma"))?);
        }

        h = graph.layer_norm_affine_tensor(
            h,
            weights.tensor("backbone.final_layer_norm.weight")?,
            weights.tensor("backbone.final_layer_norm.bias")?,
        );
        h = graph.linear_tensor(
            h,
            weights.tensor("head.out.weight")?,
            weights.tensor("head.out.bias")?,
            false,
        );
        graph.compute(h, &inputs, frames * (N_FFT + 2))
    }

    fn decode_mel_ggml_cpu_weights(&self, mel: &[f32], frames: usize) -> Result<Vec<f32>> {
        if mel.len() != frames * N_MELS {
            return Err(VocosError::Ggml("mel tensor has invalid shape".into()));
        }
        let mut graph = VocosGraph::new(frames)?;
        let mut h = graph.input_2d(N_MELS, frames);
        let mut inputs = vec![(h, mel)];

        h = graph.conv1d(
            h,
            N_MELS,
            DIM,
            7,
            false,
            self.tensor("backbone.embed.weight")?,
            self.tensor("backbone.embed.bias")?,
            &mut inputs,
        );
        h = graph.layer_norm_affine(
            h,
            DIM,
            self.tensor("backbone.norm.weight")?,
            self.tensor("backbone.norm.bias")?,
            &mut inputs,
        );

        for layer in 0..LAYERS {
            let p = format!("backbone.convnext.{layer}");
            let residual = h;
            h = graph.conv1d(
                h,
                DIM,
                DIM,
                7,
                true,
                self.tensor(&format!("{p}.dwconv.weight"))?,
                self.tensor(&format!("{p}.dwconv.bias"))?,
                &mut inputs,
            );
            h = graph.layer_norm_affine(
                h,
                DIM,
                self.tensor(&format!("{p}.norm.weight"))?,
                self.tensor(&format!("{p}.norm.bias"))?,
                &mut inputs,
            );
            h = graph.linear(
                h,
                DIM,
                HIDDEN,
                self.tensor(&format!("{p}.pwconv1.weight"))?,
                self.tensor(&format!("{p}.pwconv1.bias"))?,
                true,
                &mut inputs,
            );
            h = graph.linear(
                h,
                HIDDEN,
                DIM,
                self.tensor(&format!("{p}.pwconv2.weight"))?,
                self.tensor(&format!("{p}.pwconv2.bias"))?,
                false,
                &mut inputs,
            );
            h = graph.residual_gamma(
                residual,
                h,
                DIM,
                self.tensor(&format!("{p}.gamma"))?,
                &mut inputs,
            );
        }

        h = graph.layer_norm_affine(
            h,
            DIM,
            self.tensor("backbone.final_layer_norm.weight")?,
            self.tensor("backbone.final_layer_norm.bias")?,
            &mut inputs,
        );
        h = graph.linear(
            h,
            DIM,
            N_FFT + 2,
            self.tensor("head.out.weight")?,
            self.tensor("head.out.bias")?,
            false,
            &mut inputs,
        );
        graph.compute(h, &inputs, frames * (N_FFT + 2))
    }
}

struct VocosGraph {
    frames: usize,
    ctx: *mut ffi::ggml_context,
    backend: ffi::ggml_backend_t,
    buffer: ffi::ggml_backend_buffer_t,
    owns_backend: bool,
}

impl VocosGraph {
    fn new(frames: usize) -> Result<Self> {
        unsafe {
            ffi::ggml_backend_load_all();
            let backend = ffi::ggml_backend_init_best();
            if backend.is_null() {
                return Err(VocosError::Ggml("failed to initialize GGML backend".into()));
            }
            Self::new_with_backend(frames, backend, true)
        }
    }

    fn new_with_backend(
        frames: usize,
        backend: ffi::ggml_backend_t,
        owns_backend: bool,
    ) -> Result<Self> {
        unsafe {
            let params = ffi::ggml_init_params {
                mem_size: ffi::ggml_tensor_overhead() * 4096
                    + ffi::ggml_graph_overhead_custom(4096, false),
                mem_buffer: ptr::null_mut(),
                no_alloc: true,
            };
            let ctx = ffi::ggml_init(params);
            if ctx.is_null() {
                if owns_backend {
                    ffi::ggml_backend_free(backend);
                }
                return Err(VocosError::Ggml("failed to initialize GGML context".into()));
            }

            Ok(Self {
                frames,
                ctx,
                backend,
                buffer: ptr::null_mut(),
                owns_backend,
            })
        }
    }

    fn input_2d(&mut self, channels: usize, frames: usize) -> *mut ffi::ggml_tensor {
        unsafe {
            ffi::ggml_new_tensor_2d(
                self.ctx,
                ffi::ggml_type_GGML_TYPE_F32,
                channels as i64,
                frames as i64,
            )
        }
    }

    fn weight_1d<'a>(
        &mut self,
        len: usize,
        data: &'a [f32],
        inputs: &mut Vec<(*mut ffi::ggml_tensor, &'a [f32])>,
    ) -> *mut ffi::ggml_tensor {
        unsafe {
            let tensor =
                ffi::ggml_new_tensor_1d(self.ctx, ffi::ggml_type_GGML_TYPE_F32, len as i64);
            inputs.push((tensor, data));
            tensor
        }
    }

    fn weight_2d<'a>(
        &mut self,
        ne0: usize,
        ne1: usize,
        data: &'a [f32],
        inputs: &mut Vec<(*mut ffi::ggml_tensor, &'a [f32])>,
    ) -> *mut ffi::ggml_tensor {
        unsafe {
            let tensor = ffi::ggml_new_tensor_2d(
                self.ctx,
                ffi::ggml_type_GGML_TYPE_F32,
                ne0 as i64,
                ne1 as i64,
            );
            inputs.push((tensor, data));
            tensor
        }
    }

    fn weight_3d<'a>(
        &mut self,
        ne0: usize,
        ne1: usize,
        ne2: usize,
        data: &'a [f32],
        inputs: &mut Vec<(*mut ffi::ggml_tensor, &'a [f32])>,
    ) -> *mut ffi::ggml_tensor {
        unsafe {
            let tensor = ffi::ggml_new_tensor_3d(
                self.ctx,
                ffi::ggml_type_GGML_TYPE_F32,
                ne0 as i64,
                ne1 as i64,
                ne2 as i64,
            );
            inputs.push((tensor, data));
            tensor
        }
    }

    fn conv1d<'a>(
        &mut self,
        input: *mut ffi::ggml_tensor,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        depthwise: bool,
        weight: &'a [f32],
        bias: &'a [f32],
        inputs: &mut Vec<(*mut ffi::ggml_tensor, &'a [f32])>,
    ) -> *mut ffi::ggml_tensor {
        unsafe {
            let conv_input = self.to_conv_layout(input, in_ch);
            let weight = if depthwise {
                self.weight_3d(kernel, 1, out_ch, weight, inputs)
            } else {
                self.weight_3d(kernel, in_ch, out_ch, weight, inputs)
            };
            let bias = self.weight_2d(1, out_ch, bias, inputs);
            let conv = if depthwise {
                ffi::ggml_conv_1d_dw(self.ctx, weight, conv_input, 1, (kernel / 2) as i32, 1)
            } else {
                ffi::ggml_conv_1d(self.ctx, weight, conv_input, 1, (kernel / 2) as i32, 1)
            };
            let conv = ffi::ggml_add(self.ctx, conv, ffi::ggml_repeat(self.ctx, bias, conv));
            self.to_channel_layout(conv, out_ch)
        }
    }

    fn conv1d_tensor(
        &mut self,
        input: *mut ffi::ggml_tensor,
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        depthwise: bool,
        weight: *mut ffi::ggml_tensor,
        bias: *mut ffi::ggml_tensor,
    ) -> *mut ffi::ggml_tensor {
        unsafe {
            let conv_input = self.to_conv_layout(input, in_ch);
            let bias = if depthwise || out_ch != 1 {
                ffi::ggml_reshape_2d(self.ctx, bias, 1, out_ch as i64)
            } else {
                bias
            };
            let conv = if depthwise {
                ffi::ggml_conv_1d_dw(self.ctx, weight, conv_input, 1, (kernel / 2) as i32, 1)
            } else {
                ffi::ggml_conv_1d(self.ctx, weight, conv_input, 1, (kernel / 2) as i32, 1)
            };
            let conv = ffi::ggml_add(self.ctx, conv, ffi::ggml_repeat(self.ctx, bias, conv));
            self.to_channel_layout(conv, out_ch)
        }
    }

    fn linear<'a>(
        &mut self,
        input: *mut ffi::ggml_tensor,
        in_dim: usize,
        out_dim: usize,
        weight: &'a [f32],
        bias: &'a [f32],
        gelu: bool,
        inputs: &mut Vec<(*mut ffi::ggml_tensor, &'a [f32])>,
    ) -> *mut ffi::ggml_tensor {
        unsafe {
            let weight = self.weight_2d(in_dim, out_dim, weight, inputs);
            let bias = self.weight_1d(out_dim, bias, inputs);
            let mut out = ffi::ggml_add(self.ctx, ffi::ggml_mul_mat(self.ctx, weight, input), bias);
            if gelu {
                out = ffi::ggml_gelu(self.ctx, out);
            }
            out
        }
    }

    fn linear_tensor(
        &mut self,
        input: *mut ffi::ggml_tensor,
        weight: *mut ffi::ggml_tensor,
        bias: *mut ffi::ggml_tensor,
        gelu: bool,
    ) -> *mut ffi::ggml_tensor {
        unsafe {
            let mut out = ffi::ggml_add(self.ctx, ffi::ggml_mul_mat(self.ctx, weight, input), bias);
            if gelu {
                out = ffi::ggml_gelu(self.ctx, out);
            }
            out
        }
    }

    fn layer_norm_affine<'a>(
        &mut self,
        input: *mut ffi::ggml_tensor,
        channels: usize,
        weight: &'a [f32],
        bias: &'a [f32],
        inputs: &mut Vec<(*mut ffi::ggml_tensor, &'a [f32])>,
    ) -> *mut ffi::ggml_tensor {
        unsafe {
            let weight = self.weight_1d(channels, weight, inputs);
            let bias = self.weight_1d(channels, bias, inputs);
            ffi::ggml_add(
                self.ctx,
                ffi::ggml_mul(
                    self.ctx,
                    ffi::ggml_norm(self.ctx, input, 1e-6),
                    ffi::ggml_repeat(self.ctx, weight, input),
                ),
                ffi::ggml_repeat(self.ctx, bias, input),
            )
        }
    }

    fn layer_norm_affine_tensor(
        &mut self,
        input: *mut ffi::ggml_tensor,
        weight: *mut ffi::ggml_tensor,
        bias: *mut ffi::ggml_tensor,
    ) -> *mut ffi::ggml_tensor {
        unsafe {
            ffi::ggml_add(
                self.ctx,
                ffi::ggml_mul(
                    self.ctx,
                    ffi::ggml_norm(self.ctx, input, 1e-6),
                    ffi::ggml_repeat(self.ctx, weight, input),
                ),
                ffi::ggml_repeat(self.ctx, bias, input),
            )
        }
    }

    fn residual_gamma<'a>(
        &mut self,
        residual: *mut ffi::ggml_tensor,
        input: *mut ffi::ggml_tensor,
        channels: usize,
        gamma: &'a [f32],
        inputs: &mut Vec<(*mut ffi::ggml_tensor, &'a [f32])>,
    ) -> *mut ffi::ggml_tensor {
        unsafe {
            let gamma = self.weight_1d(channels, gamma, inputs);
            ffi::ggml_add(
                self.ctx,
                residual,
                ffi::ggml_mul(self.ctx, input, ffi::ggml_repeat(self.ctx, gamma, input)),
            )
        }
    }

    fn residual_gamma_tensor(
        &mut self,
        residual: *mut ffi::ggml_tensor,
        input: *mut ffi::ggml_tensor,
        gamma: *mut ffi::ggml_tensor,
    ) -> *mut ffi::ggml_tensor {
        unsafe {
            ffi::ggml_add(
                self.ctx,
                residual,
                ffi::ggml_mul(self.ctx, input, ffi::ggml_repeat(self.ctx, gamma, input)),
            )
        }
    }

    fn compute(
        &mut self,
        output: *mut ffi::ggml_tensor,
        inputs: &[(*mut ffi::ggml_tensor, &[f32])],
        output_len: usize,
    ) -> Result<Vec<f32>> {
        unsafe {
            let graph = ffi::ggml_new_graph_custom(self.ctx, 4096, false);
            if graph.is_null() {
                return Err(VocosError::Ggml("failed to create GGML graph".into()));
            }
            ffi::ggml_build_forward_expand(graph, output);

            self.buffer = ffi::ggml_backend_alloc_ctx_tensors(self.ctx, self.backend);
            if self.buffer.is_null() {
                return Err(VocosError::Ggml("failed to allocate GGML tensors".into()));
            }

            for &(tensor, data) in inputs {
                ffi::ggml_backend_tensor_set(
                    tensor,
                    data.as_ptr().cast(),
                    0,
                    std::mem::size_of_val(data),
                );
            }

            let status = ffi::ggml_backend_graph_compute(self.backend, graph);
            if status != ffi::ggml_status_GGML_STATUS_SUCCESS {
                return Err(VocosError::Ggml(format!(
                    "GGML decoder graph failed with status={status}"
                )));
            }

            let mut out = vec![0.0_f32; output_len];
            ffi::ggml_backend_tensor_get(
                output,
                out.as_mut_ptr().cast(),
                0,
                std::mem::size_of_val(out.as_slice()),
            );
            Ok(out)
        }
    }

    fn to_conv_layout(
        &mut self,
        input: *mut ffi::ggml_tensor,
        channels: usize,
    ) -> *mut ffi::ggml_tensor {
        unsafe {
            let transposed = ffi::ggml_permute(self.ctx, input, 1, 0, 2, 3);
            ffi::ggml_cont_3d(self.ctx, transposed, self.frames as i64, channels as i64, 1)
        }
    }

    fn to_channel_layout(
        &mut self,
        input: *mut ffi::ggml_tensor,
        channels: usize,
    ) -> *mut ffi::ggml_tensor {
        unsafe {
            let transposed = ffi::ggml_permute(self.ctx, input, 1, 0, 2, 3);
            ffi::ggml_cont_2d(self.ctx, transposed, channels as i64, self.frames as i64)
        }
    }
}

impl Drop for VocosGraph {
    fn drop(&mut self) {
        unsafe {
            if !self.buffer.is_null() {
                ffi::ggml_backend_buffer_free(self.buffer);
            }
            if !self.ctx.is_null() {
                ffi::ggml_free(self.ctx);
            }
            if self.owns_backend && !self.backend.is_null() {
                ffi::ggml_backend_free(self.backend);
            }
        }
    }
}

fn load_wav_mono_24khz(path: impl AsRef<Path>) -> Result<Vec<f32>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let channels = spec.channels as usize;
    let mut samples = Vec::new();
    match spec.sample_format {
        hound::SampleFormat::Int => {
            let max = (1_i64 << (spec.bits_per_sample - 1)) as f32;
            let mut frame = Vec::with_capacity(channels);
            for s in reader.samples::<i32>() {
                frame.push(s? as f32 / max);
                if frame.len() == channels {
                    samples.push(frame.iter().sum::<f32>() / channels as f32);
                    frame.clear();
                }
            }
        }
        hound::SampleFormat::Float => {
            let mut frame = Vec::with_capacity(channels);
            for s in reader.samples::<f32>() {
                frame.push(s?);
                if frame.len() == channels {
                    samples.push(frame.iter().sum::<f32>() / channels as f32);
                    frame.clear();
                }
            }
        }
    }
    if spec.sample_rate as usize == SAMPLE_RATE {
        return Ok(samples);
    }
    resample_mono(&samples, spec.sample_rate as usize, SAMPLE_RATE)
}

fn resample_mono(input: &[f32], from: usize, to: usize) -> Result<Vec<f32>> {
    let chunk = 1024;
    let mut resampler = FftFixedInOut::<f32>::new(from, to, chunk, 1)
        .map_err(|e| VocosError::Resample(e.to_string()))?;
    let in_frames = resampler.input_frames_next();
    let mut output = Vec::new();
    let mut pos = 0;
    while pos < input.len() {
        let end = (pos + in_frames).min(input.len());
        let mut block = input[pos..end].to_vec();
        if block.len() < in_frames {
            block.resize(in_frames, 0.0);
        }
        let out = resampler
            .process(&[block], None)
            .map_err(|e| VocosError::Resample(e.to_string()))?;
        output.extend_from_slice(&out[0]);
        pos += in_frames;
    }
    let expected = input.len() * to / from;
    output.truncate(expected);
    Ok(output)
}

fn reflect_pad(input: &[f32], pad: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(input.len() + 2 * pad);
    if input.is_empty() {
        out.resize(2 * pad, 0.0);
        return out;
    }
    for i in (1..=pad).rev() {
        out.push(input[i.min(input.len() - 1)]);
    }
    out.extend_from_slice(input);
    let last = input.len() - 1;
    for i in 1..=pad {
        out.push(input[last.saturating_sub(i)]);
    }
    out
}

fn librosa_htk_mel_filters() -> &'static [f32] {
    static FILTERS: OnceLock<Vec<f32>> = OnceLock::new();
    FILTERS.get_or_init(build_librosa_htk_mel_filters)
}

fn build_librosa_htk_mel_filters() -> Vec<f32> {
    let bins = N_FFT / 2 + 1;
    let min_mel = hz_to_mel_htk(0.0);
    let max_mel = hz_to_mel_htk(SAMPLE_RATE as f32 / 2.0);
    let mut mel_points = [0.0_f32; N_MELS + 2];
    for (i, point) in mel_points.iter_mut().enumerate() {
        let ratio = i as f32 / (N_MELS + 1) as f32;
        *point = mel_to_hz_htk(min_mel + ratio * (max_mel - min_mel));
    }

    let mut filters = vec![0.0_f32; N_MELS * bins];
    for m in 0..N_MELS {
        let lower_hz = mel_points[m];
        let center_hz = mel_points[m + 1];
        let upper_hz = mel_points[m + 2];
        let lower_width = (center_hz - lower_hz).max(f32::EPSILON);
        let upper_width = (upper_hz - center_hz).max(f32::EPSILON);
        for b in 0..bins {
            let hz = b as f32 * SAMPLE_RATE as f32 / N_FFT as f32;
            let lower = (hz - lower_hz) / lower_width;
            let upper = (upper_hz - hz) / upper_width;
            filters[m * bins + b] = lower.min(upper).max(0.0);
        }
    }
    filters
}

fn hz_to_mel_htk(hz: f32) -> f32 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

fn mel_to_hz_htk(mel: f32) -> f32 {
    700.0 * (10.0_f32.powf(mel / 2595.0) - 1.0)
}

fn istft_from_head(spec: &[f32], frames: usize, window: &[f32]) -> Vec<f32> {
    let bins = N_FFT / 2 + 1;
    let full_len = (frames - 1) * HOP + N_FFT;
    let mut out = vec![0.0_f32; full_len];
    let mut envelope = vec![0.0_f32; full_len];
    let mut planner = FftPlanner::<f32>::new();
    let ifft = planner.plan_fft_inverse(N_FFT);
    let mut buf = vec![Complex32::new(0.0, 0.0); N_FFT];

    for t in 0..frames {
        let row = &spec[t * (N_FFT + 2)..(t + 1) * (N_FFT + 2)];
        let (mag_logits, phase) = row.split_at(bins);
        for b in 0..bins {
            let mag = mag_logits[b].exp().min(1e2);
            buf[b] = Complex32::new(mag * phase[b].cos(), mag * phase[b].sin());
        }
        for b in bins..N_FFT {
            buf[b] = buf[N_FFT - b].conj();
        }
        ifft.process(&mut buf);
        let start = t * HOP;
        for i in 0..N_FFT {
            let w = window[i];
            out[start + i] += (buf[i].re / N_FFT as f32) * w;
            envelope[start + i] += w * w;
        }
    }
    for (y, e) in out.iter_mut().zip(envelope) {
        if e > 1e-11 {
            *y /= e;
        }
    }
    let trim = N_FFT / 2;
    if out.len() > 2 * trim {
        out[trim..out.len() - trim].to_vec()
    } else {
        out
    }
}
