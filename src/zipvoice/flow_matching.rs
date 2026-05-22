use std::ptr;

use llama_rs_sys as ffi;

use super::{Result, ZipVoiceError, model::ZipVoiceModel};

const FLOW_GRAPH_NODES: usize = 65_536;

pub fn shifted_timesteps(num_steps: usize, t_shift: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(num_steps + 1);
    for step in 0..=num_steps {
        let t = step as f32 / num_steps.max(1) as f32;
        out.push(t_shift * t / (1.0 + (t_shift - 1.0) * t));
    }
    out
}

pub fn input_projection(
    model: &ZipVoiceModel,
    x: &[f32],
    text_condition: &[f32],
    speech_condition: &[f32],
    frames: usize,
) -> Result<Vec<f32>> {
    let feat_dim = model.feat_dim();
    let expected = frames * feat_dim;
    if x.len() != expected || text_condition.len() != expected || speech_condition.len() != expected
    {
        return Err(ZipVoiceError::Ggml(format!(
            "bad flow input shape: expected {expected} values per input"
        )));
    }
    unsafe {
        let params = ffi::ggml_init_params {
            mem_size: ffi::ggml_tensor_overhead() * 64 + ffi::ggml_graph_overhead_custom(32, false),
            mem_buffer: ptr::null_mut(),
            no_alloc: true,
        };
        let ctx = ffi::ggml_init(params);
        if ctx.is_null() {
            return Err(ZipVoiceError::Ggml(
                "failed to initialize flow input GGML context".into(),
            ));
        }
        let result =
            input_projection_graph(ctx, model, x, text_condition, speech_condition, frames);
        ffi::ggml_free(ctx);
        result
    }
}

pub fn velocity_preview(
    model: &ZipVoiceModel,
    t: f32,
    x: &[f32],
    text_condition: &[f32],
    speech_condition: &[f32],
    frames: usize,
) -> Result<Vec<f32>> {
    let feat_dim = model.feat_dim();
    let expected = frames * feat_dim;
    if x.len() != expected || text_condition.len() != expected || speech_condition.len() != expected
    {
        return Err(ZipVoiceError::Ggml(format!(
            "bad flow input shape: expected {expected} values per input"
        )));
    }
    let time_embedding = timestep_embedding(t, model.config().model.time_embed_dim as usize);
    unsafe {
        let params = ffi::ggml_init_params {
            mem_size: ffi::ggml_tensor_overhead() * FLOW_GRAPH_NODES
                + ffi::ggml_graph_overhead_custom(FLOW_GRAPH_NODES, false),
            mem_buffer: ptr::null_mut(),
            no_alloc: true,
        };
        let ctx = ffi::ggml_init(params);
        if ctx.is_null() {
            return Err(ZipVoiceError::Ggml(
                "failed to initialize flow velocity GGML context".into(),
            ));
        }
        let result = velocity_preview_graph(
            ctx,
            model,
            &time_embedding,
            x,
            text_condition,
            speech_condition,
            frames,
        );
        ffi::ggml_free(ctx);
        result
    }
}

pub fn sample_preview(
    model: &ZipVoiceModel,
    text_condition: &[f32],
    speech_condition: &[f32],
    frames: usize,
    num_steps: usize,
    t_shift: f32,
    guidance_scale: f32,
    seed: u64,
) -> Result<Vec<f32>> {
    let feat_dim = model.feat_dim();
    let expected = frames * feat_dim;
    if text_condition.len() != expected || speech_condition.len() != expected {
        return Err(ZipVoiceError::Ggml(format!(
            "bad flow condition shape: expected {expected} values per condition"
        )));
    }

    let mut rng = XorShift64::new(seed);
    let mut x = (0..expected)
        .map(|_| rng.next_gaussian())
        .collect::<Vec<_>>();
    let timesteps = shifted_timesteps(num_steps, t_shift);

    if guidance_scale == 0.0 {
        for step in 0..num_steps {
            let t0 = timesteps[step];
            let dt = timesteps[step + 1] - t0;
            let velocity =
                velocity_preview(model, t0, &x, text_condition, speech_condition, frames)?;
            for (x, v) in x.iter_mut().zip(velocity) {
                *x += v * dt;
            }
        }
    } else {
        let mut runner = None;
        let mut runner_uses_zero_speech = false;
        for step in 0..num_steps {
            let t0 = timesteps[step];
            let dt = timesteps[step + 1] - t0;
            let uses_zero_speech = t0 > 0.5;
            if runner.is_none() || runner_uses_zero_speech != uses_zero_speech {
                drop(runner.take());
                let scale = if uses_zero_speech {
                    guidance_scale
                } else {
                    guidance_scale * 2.0
                };
                runner = Some(GuidedVelocityGraph::new(
                    model,
                    text_condition,
                    speech_condition,
                    frames,
                    uses_zero_speech,
                    scale,
                )?);
                runner_uses_zero_speech = uses_zero_speech;
            }
            let velocity = runner
                .as_mut()
                .expect("guided velocity runner was initialized")
                .compute(t0, &x)?;
            for (x, v) in x.iter_mut().zip(velocity) {
                *x += v * dt;
            }
        }
    }

    Ok(x)
}

pub fn guided_velocity_preview(
    model: &ZipVoiceModel,
    t: f32,
    x: &[f32],
    text_condition: &[f32],
    speech_condition: &[f32],
    frames: usize,
    guidance_scale: f32,
) -> Result<Vec<f32>> {
    if guidance_scale == 0.0 {
        return velocity_preview(model, t, x, text_condition, speech_condition, frames);
    }

    let feat_dim = model.feat_dim();
    let expected = frames * feat_dim;
    if x.len() != expected || text_condition.len() != expected || speech_condition.len() != expected
    {
        return Err(ZipVoiceError::Ggml(format!(
            "bad guided flow input shape: expected {expected} values per input"
        )));
    }
    let time_embedding = timestep_embedding(t, model.config().model.time_embed_dim as usize);
    unsafe {
        let params = ffi::ggml_init_params {
            mem_size: ffi::ggml_tensor_overhead() * FLOW_GRAPH_NODES
                + ffi::ggml_graph_overhead_custom(FLOW_GRAPH_NODES, false),
            mem_buffer: ptr::null_mut(),
            no_alloc: true,
        };
        let ctx = ffi::ggml_init(params);
        if ctx.is_null() {
            return Err(ZipVoiceError::Ggml(
                "failed to initialize guided flow GGML context".into(),
            ));
        }
        let result = guided_velocity_preview_graph(
            ctx,
            model,
            &time_embedding,
            x,
            text_condition,
            speech_condition,
            frames,
            t,
            guidance_scale,
        );
        ffi::ggml_free(ctx);
        result
    }
}

unsafe fn input_projection_graph(
    ctx: *mut ffi::ggml_context,
    model: &ZipVoiceModel,
    x: &[f32],
    text_condition: &[f32],
    speech_condition: &[f32],
    frames: usize,
) -> Result<Vec<f32>> {
    unsafe {
        let feat_dim = model.feat_dim();
        let x_t = input_tensor(ctx, feat_dim, frames);
        let text_t = input_tensor(ctx, feat_dim, frames);
        let speech_t = input_tensor(ctx, feat_dim, frames);
        if x_t.is_null() || text_t.is_null() || speech_t.is_null() {
            return Err(ZipVoiceError::Ggml(
                "failed to create flow input tensors".into(),
            ));
        }
        let xt = ffi::ggml_concat(ctx, x_t, text_t, 0);
        let xt = ffi::ggml_concat(ctx, xt, speech_t, 0);
        let weight = model.weight_tensor_full("fm_decoder.in_proj.weight")?;
        let bias = model.weight_tensor_full("fm_decoder.in_proj.bias")?;
        let out = ffi::ggml_add(ctx, ffi::ggml_mul_mat(ctx, weight, xt), bias);
        let graph = ffi::ggml_new_graph_custom(ctx, 32, false);
        if graph.is_null() {
            return Err(ZipVoiceError::Ggml(
                "failed to create flow input graph".into(),
            ));
        }
        ffi::ggml_build_forward_expand(graph, out);
        model.alloc_graph(graph)?;
        set_tensor(x_t, x);
        set_tensor(text_t, text_condition);
        set_tensor(speech_t, speech_condition);
        if let Err(err) = model.compute_graph(graph) {
            model.reset_scheduler();
            return Err(err);
        }
        let mut out_data = vec![0.0_f32; frames * model.fm_decoder_dim()];
        ffi::ggml_backend_tensor_get(
            out,
            out_data.as_mut_ptr().cast(),
            0,
            std::mem::size_of_val(out_data.as_slice()),
        );
        model.reset_scheduler();
        Ok(out_data)
    }
}

unsafe fn velocity_preview_graph(
    ctx: *mut ffi::ggml_context,
    model: &ZipVoiceModel,
    time_embedding: &[f32],
    x: &[f32],
    text_condition: &[f32],
    speech_condition: &[f32],
    frames: usize,
) -> Result<Vec<f32>> {
    unsafe {
        let feat_dim = model.feat_dim();
        let x_t = input_tensor(ctx, feat_dim, frames);
        let text_t = input_tensor(ctx, feat_dim, frames);
        let speech_t = input_tensor(ctx, feat_dim, frames);
        let time_t = ffi::ggml_new_tensor_2d(
            ctx,
            ffi::ggml_type_GGML_TYPE_F32,
            time_embedding.len() as i64,
            1,
        );
        ffi::ggml_set_input(time_t);
        if x_t.is_null() || text_t.is_null() || speech_t.is_null() || time_t.is_null() {
            return Err(ZipVoiceError::Ggml(
                "failed to create flow velocity input tensors".into(),
            ));
        }

        let mut scalars = Vec::new();
        let mut aux_inputs = Vec::new();
        let out = velocity_output_tensor(
            ctx,
            model,
            x_t,
            text_t,
            speech_t,
            time_t,
            frames,
            &mut scalars,
            &mut aux_inputs,
        )?;

        let graph = ffi::ggml_new_graph_custom(ctx, FLOW_GRAPH_NODES, false);
        if graph.is_null() {
            return Err(ZipVoiceError::Ggml(
                "failed to create flow velocity graph".into(),
            ));
        }
        ffi::ggml_build_forward_expand(graph, out);
        model.alloc_graph(graph)?;
        set_tensor(x_t, x);
        set_tensor(text_t, text_condition);
        set_tensor(speech_t, speech_condition);
        set_tensor(time_t, time_embedding);
        for (tensor, value) in scalars {
            ffi::ggml_backend_tensor_set(
                tensor,
                (&value as *const f32).cast(),
                0,
                std::mem::size_of::<f32>(),
            );
        }
        for (tensor, data) in aux_inputs {
            ffi::ggml_backend_tensor_set(
                tensor,
                data.as_ptr().cast(),
                0,
                std::mem::size_of_val(data.as_slice()),
            );
        }
        if let Err(err) = model.compute_graph(graph) {
            model.reset_scheduler();
            return Err(err);
        }
        let mut out_data = vec![0.0_f32; frames * model.feat_dim()];
        ffi::ggml_backend_tensor_get(
            out,
            out_data.as_mut_ptr().cast(),
            0,
            std::mem::size_of_val(out_data.as_slice()),
        );
        model.reset_scheduler();
        Ok(out_data)
    }
}

unsafe fn guided_velocity_preview_graph(
    ctx: *mut ffi::ggml_context,
    model: &ZipVoiceModel,
    time_embedding: &[f32],
    x: &[f32],
    text_condition: &[f32],
    speech_condition: &[f32],
    frames: usize,
    t: f32,
    guidance_scale: f32,
) -> Result<Vec<f32>> {
    unsafe {
        let feat_dim = model.feat_dim();
        let x_t = input_tensor(ctx, feat_dim, frames);
        let text_t = input_tensor(ctx, feat_dim, frames);
        let speech_t = input_tensor(ctx, feat_dim, frames);
        let zero_text_t = input_tensor(ctx, feat_dim, frames);
        let zero_speech_t = input_tensor(ctx, feat_dim, frames);
        let time_t = ffi::ggml_new_tensor_2d(
            ctx,
            ffi::ggml_type_GGML_TYPE_F32,
            time_embedding.len() as i64,
            1,
        );
        ffi::ggml_set_input(time_t);
        if x_t.is_null()
            || text_t.is_null()
            || speech_t.is_null()
            || zero_text_t.is_null()
            || zero_speech_t.is_null()
            || time_t.is_null()
        {
            return Err(ZipVoiceError::Ggml(
                "failed to create guided flow input tensors".into(),
            ));
        }

        let mut scalars = Vec::new();
        let mut aux_inputs = Vec::new();
        let cond = velocity_output_tensor(
            ctx,
            model,
            x_t,
            text_t,
            speech_t,
            time_t,
            frames,
            &mut scalars,
            &mut aux_inputs,
        )?;
        let uncond_speech_t = if t > 0.5 { zero_speech_t } else { speech_t };
        let uncond = velocity_output_tensor(
            ctx,
            model,
            x_t,
            zero_text_t,
            uncond_speech_t,
            time_t,
            frames,
            &mut scalars,
            &mut aux_inputs,
        )?;
        let scale = if t > 0.5 {
            guidance_scale
        } else {
            guidance_scale * 2.0
        };
        let out = ffi::ggml_sub(
            ctx,
            ffi::ggml_scale(ctx, cond, 1.0 + scale),
            ffi::ggml_scale(ctx, uncond, scale),
        );

        let graph = ffi::ggml_new_graph_custom(ctx, FLOW_GRAPH_NODES, false);
        if graph.is_null() {
            return Err(ZipVoiceError::Ggml(
                "failed to create guided flow graph".into(),
            ));
        }
        ffi::ggml_build_forward_expand(graph, out);
        model.alloc_graph(graph)?;
        set_tensor(x_t, x);
        set_tensor(text_t, text_condition);
        set_tensor(speech_t, speech_condition);
        let zeros = vec![0.0_f32; x.len()];
        set_tensor(zero_text_t, &zeros);
        if t > 0.5 {
            set_tensor(zero_speech_t, &zeros);
        }
        set_tensor(time_t, time_embedding);
        for (tensor, value) in scalars {
            ffi::ggml_backend_tensor_set(
                tensor,
                (&value as *const f32).cast(),
                0,
                std::mem::size_of::<f32>(),
            );
        }
        for (tensor, data) in aux_inputs {
            ffi::ggml_backend_tensor_set(
                tensor,
                data.as_ptr().cast(),
                0,
                std::mem::size_of_val(data.as_slice()),
            );
        }
        if let Err(err) = model.compute_graph(graph) {
            model.reset_scheduler();
            return Err(err);
        }
        let mut out_data = vec![0.0_f32; frames * model.feat_dim()];
        ffi::ggml_backend_tensor_get(
            out,
            out_data.as_mut_ptr().cast(),
            0,
            std::mem::size_of_val(out_data.as_slice()),
        );
        model.reset_scheduler();
        Ok(out_data)
    }
}

struct GuidedVelocityGraph<'a> {
    model: &'a ZipVoiceModel,
    ctx: *mut ffi::ggml_context,
    graph: *mut ffi::ggml_cgraph,
    x_t: *mut ffi::ggml_tensor,
    time_t: *mut ffi::ggml_tensor,
    out: *mut ffi::ggml_tensor,
    frames: usize,
}

impl<'a> GuidedVelocityGraph<'a> {
    fn new(
        model: &'a ZipVoiceModel,
        text_condition: &'a [f32],
        speech_condition: &'a [f32],
        frames: usize,
        uses_zero_speech: bool,
        scale: f32,
    ) -> Result<Self> {
        unsafe {
            let params = ffi::ggml_init_params {
                mem_size: ffi::ggml_tensor_overhead() * FLOW_GRAPH_NODES
                    + ffi::ggml_graph_overhead_custom(FLOW_GRAPH_NODES, false),
                mem_buffer: ptr::null_mut(),
                no_alloc: true,
            };
            let ctx = ffi::ggml_init(params);
            if ctx.is_null() {
                return Err(ZipVoiceError::Ggml(
                    "failed to initialize reusable guided flow context".into(),
                ));
            }
            let result = Self::build(
                model,
                ctx,
                text_condition,
                speech_condition,
                frames,
                uses_zero_speech,
                scale,
            );
            if result.is_err() {
                ffi::ggml_free(ctx);
            }
            result
        }
    }

    unsafe fn build(
        model: &'a ZipVoiceModel,
        ctx: *mut ffi::ggml_context,
        text_condition: &'a [f32],
        speech_condition: &'a [f32],
        frames: usize,
        uses_zero_speech: bool,
        scale: f32,
    ) -> Result<Self> {
        unsafe {
            let feat_dim = model.feat_dim();
            let x_t = input_tensor(ctx, feat_dim, frames);
            let text_t = input_tensor(ctx, feat_dim, frames);
            let speech_t = input_tensor(ctx, feat_dim, frames);
            let zero_text_t = input_tensor(ctx, feat_dim, frames);
            let zero_speech_t = if uses_zero_speech {
                Some(input_tensor(ctx, feat_dim, frames))
            } else {
                None
            };
            let time_t = ffi::ggml_new_tensor_2d(
                ctx,
                ffi::ggml_type_GGML_TYPE_F32,
                model.config().model.time_embed_dim as i64,
                1,
            );
            ffi::ggml_set_input(time_t);
            if x_t.is_null()
                || text_t.is_null()
                || speech_t.is_null()
                || zero_text_t.is_null()
                || zero_speech_t.is_some_and(|tensor| tensor.is_null())
                || time_t.is_null()
            {
                return Err(ZipVoiceError::Ggml(
                    "failed to create reusable guided flow inputs".into(),
                ));
            }

            let mut scalars = Vec::new();
            let mut aux_inputs = Vec::new();
            let cond = velocity_output_tensor(
                ctx,
                model,
                x_t,
                text_t,
                speech_t,
                time_t,
                frames,
                &mut scalars,
                &mut aux_inputs,
            )?;
            let uncond_speech_t = zero_speech_t.unwrap_or(speech_t);
            let uncond = velocity_output_tensor(
                ctx,
                model,
                x_t,
                zero_text_t,
                uncond_speech_t,
                time_t,
                frames,
                &mut scalars,
                &mut aux_inputs,
            )?;
            let out = ffi::ggml_sub(
                ctx,
                ffi::ggml_scale(ctx, cond, 1.0 + scale),
                ffi::ggml_scale(ctx, uncond, scale),
            );
            let graph = ffi::ggml_new_graph_custom(ctx, FLOW_GRAPH_NODES, false);
            if graph.is_null() {
                return Err(ZipVoiceError::Ggml(
                    "failed to create reusable guided flow graph".into(),
                ));
            }
            ffi::ggml_build_forward_expand(graph, out);
            model.alloc_graph(graph)?;

            let zeros = vec![0.0_f32; frames * feat_dim];
            set_tensor(text_t, text_condition);
            set_tensor(speech_t, speech_condition);
            set_tensor(zero_text_t, &zeros);
            if let Some(zero_speech_t) = zero_speech_t {
                set_tensor(zero_speech_t, &zeros);
            }
            for &(tensor, value) in &scalars {
                ffi::ggml_backend_tensor_set(
                    tensor,
                    (&value as *const f32).cast(),
                    0,
                    std::mem::size_of::<f32>(),
                );
            }
            for (tensor, data) in &aux_inputs {
                ffi::ggml_backend_tensor_set(
                    *tensor,
                    data.as_ptr().cast(),
                    0,
                    std::mem::size_of_val(data.as_slice()),
                );
            }

            Ok(Self {
                model,
                ctx,
                graph,
                x_t,
                time_t,
                out,
                frames,
            })
        }
    }

    fn compute(&mut self, t: f32, x: &[f32]) -> Result<Vec<f32>> {
        let time_embedding =
            timestep_embedding(t, self.model.config().model.time_embed_dim as usize);
        unsafe {
            set_tensor(self.x_t, x);
            set_tensor(self.time_t, &time_embedding);
            if let Err(err) = self.model.compute_graph(self.graph) {
                return Err(err);
            }
            let mut out_data = vec![0.0_f32; self.frames * self.model.feat_dim()];
            ffi::ggml_backend_tensor_get(
                self.out,
                out_data.as_mut_ptr().cast(),
                0,
                std::mem::size_of_val(out_data.as_slice()),
            );
            Ok(out_data)
        }
    }
}

impl Drop for GuidedVelocityGraph<'_> {
    fn drop(&mut self) {
        self.model.reset_scheduler();
        unsafe {
            if !self.ctx.is_null() {
                ffi::ggml_free(self.ctx);
            }
        }
    }
}

unsafe fn velocity_output_tensor(
    ctx: *mut ffi::ggml_context,
    model: &ZipVoiceModel,
    x_t: *mut ffi::ggml_tensor,
    text_t: *mut ffi::ggml_tensor,
    speech_t: *mut ffi::ggml_tensor,
    time_t: *mut ffi::ggml_tensor,
    frames: usize,
    scalars: &mut Vec<(*mut ffi::ggml_tensor, f32)>,
    aux_inputs: &mut Vec<(*mut ffi::ggml_tensor, Vec<f32>)>,
) -> Result<*mut ffi::ggml_tensor> {
    unsafe {
        let xt = ffi::ggml_concat(ctx, x_t, text_t, 0);
        let xt = ffi::ggml_concat(ctx, xt, speech_t, 0);
        let mut src = linear_full(ctx, model, xt, "fm_decoder.in_proj")?;
        let global_time = flow_time_embedding(ctx, model, time_t, scalars)?;
        src = flow_encoder_stack(
            ctx,
            model,
            src,
            frames,
            global_time,
            0,
            "fm_decoder.encoders.0",
            scalars,
            aux_inputs,
        )?;
        src = flow_downsampled_encoder_stack(
            ctx,
            model,
            src,
            frames,
            global_time,
            1,
            "fm_decoder.encoders.1",
            scalars,
            aux_inputs,
        )?;
        src = flow_downsampled_encoder_stack(
            ctx,
            model,
            src,
            frames,
            global_time,
            2,
            "fm_decoder.encoders.2",
            scalars,
            aux_inputs,
        )?;
        src = flow_downsampled_encoder_stack(
            ctx,
            model,
            src,
            frames,
            global_time,
            3,
            "fm_decoder.encoders.3",
            scalars,
            aux_inputs,
        )?;
        src = flow_encoder_stack(
            ctx,
            model,
            src,
            frames,
            global_time,
            4,
            "fm_decoder.encoders.4",
            scalars,
            aux_inputs,
        )?;
        linear_full(ctx, model, src, "fm_decoder.out_proj")
    }
}

unsafe fn flow_time_embedding(
    ctx: *mut ffi::ggml_context,
    model: &ZipVoiceModel,
    time_t: *mut ffi::ggml_tensor,
    scalars: &mut Vec<(*mut ffi::ggml_tensor, f32)>,
) -> Result<*mut ffi::ggml_tensor> {
    unsafe {
        let mut time = linear_full(ctx, model, time_t, "fm_decoder.time_embed.0")?;
        time = swoosh_r(ctx, time, scalars);
        time = linear_full(ctx, model, time, "fm_decoder.time_embed.2")?;
        Ok(swoosh_r(ctx, time, scalars))
    }
}

unsafe fn flow_encoder_stack(
    ctx: *mut ffi::ggml_context,
    model: &ZipVoiceModel,
    mut src: *mut ffi::ggml_tensor,
    frames: usize,
    global_time: *mut ffi::ggml_tensor,
    stack_idx: usize,
    stack_prefix: &str,
    scalars: &mut Vec<(*mut ffi::ggml_tensor, f32)>,
    aux_inputs: &mut Vec<(*mut ffi::ggml_tensor, Vec<f32>)>,
) -> Result<*mut ffi::ggml_tensor> {
    unsafe {
        let stack_time = linear_full(
            ctx,
            model,
            global_time,
            &format!("{stack_prefix}.time_emb.1"),
        )?;
        let layers = model.config().model.fm_decoder_num_layers[stack_idx] as usize;
        let kernel = model.config().model.fm_decoder_cnn_module_kernel[stack_idx] as usize;
        for layer in 0..layers {
            src = flow_layer_non_attention(
                ctx,
                model,
                src,
                frames,
                stack_time,
                kernel,
                &format!("{stack_prefix}.layers.{layer}"),
                scalars,
                aux_inputs,
            )?;
        }
        Ok(src)
    }
}

unsafe fn flow_downsampled_encoder_stack(
    ctx: *mut ffi::ggml_context,
    model: &ZipVoiceModel,
    src: *mut ffi::ggml_tensor,
    frames: usize,
    global_time: *mut ffi::ggml_tensor,
    stack_idx: usize,
    stack_prefix: &str,
    scalars: &mut Vec<(*mut ffi::ggml_tensor, f32)>,
    aux_inputs: &mut Vec<(*mut ffi::ggml_tensor, Vec<f32>)>,
) -> Result<*mut ffi::ggml_tensor> {
    unsafe {
        let ds = model.config().model.fm_decoder_downsampling_factor[stack_idx] as usize;
        let src_down = simple_downsample_tensor(
            ctx,
            model,
            src,
            frames,
            model.fm_decoder_dim(),
            ds,
            &format!("{stack_prefix}.downsample"),
            scalars,
        )?;
        let down_frames = frames.div_ceil(ds);
        let stack_time = linear_full(
            ctx,
            model,
            global_time,
            &format!("{stack_prefix}.encoder.time_emb.1"),
        )?;
        let layers = model.config().model.fm_decoder_num_layers[stack_idx] as usize;
        let kernel = model.config().model.fm_decoder_cnn_module_kernel[stack_idx] as usize;
        let mut h = src_down;
        for layer in 0..layers {
            h = flow_layer_non_attention(
                ctx,
                model,
                h,
                down_frames,
                stack_time,
                kernel,
                &format!("{stack_prefix}.encoder.layers.{layer}"),
                scalars,
                aux_inputs,
            )?;
        }
        h = simple_upsample_tensor(ctx, h, down_frames, model.fm_decoder_dim(), ds, frames);
        bypass_tensor(
            ctx,
            model,
            src,
            h,
            &format!("{stack_prefix}.out_combiner.bypass_scale"),
        )
    }
}

unsafe fn flow_layer_non_attention(
    ctx: *mut ffi::ggml_context,
    model: &ZipVoiceModel,
    mut src: *mut ffi::ggml_tensor,
    frames: usize,
    time: *mut ffi::ggml_tensor,
    kernel: usize,
    layer_prefix: &str,
    scalars: &mut Vec<(*mut ffi::ggml_tensor, f32)>,
    aux_inputs: &mut Vec<(*mut ffi::ggml_tensor, Vec<f32>)>,
) -> Result<*mut ffi::ggml_tensor> {
    unsafe {
        let src_orig = src;
        let attn_weights =
            attention_weights_by_head(ctx, model, src_orig, frames, layer_prefix, aux_inputs)?;
        src = ffi::ggml_add(ctx, src, ffi::ggml_repeat(ctx, time, src));
        src = feed_forward_tensor(
            ctx,
            model,
            src,
            &format!("{layer_prefix}.feed_forward1"),
            scalars,
        )?;
        let na = nonlin_attention_tensor(ctx, model, src, frames, layer_prefix, attn_weights[0])?;
        src = ffi::ggml_add(ctx, src, na);
        let attn = self_attention_tensor(
            ctx,
            model,
            src,
            frames,
            layer_prefix,
            "self_attn1",
            &attn_weights,
        )?;
        src = ffi::ggml_add(ctx, src, attn);
        src = ffi::ggml_add(ctx, src, ffi::ggml_repeat(ctx, time, src));
        src = conv_module_tensor(
            ctx,
            model,
            src,
            frames,
            model.fm_decoder_dim(),
            kernel,
            &format!("{layer_prefix}.conv_module1"),
            scalars,
        )?;
        src = feed_forward_tensor(
            ctx,
            model,
            src,
            &format!("{layer_prefix}.feed_forward2"),
            scalars,
        )?;
        src = bypass_tensor(
            ctx,
            model,
            src_orig,
            src,
            &format!("{layer_prefix}.bypass_mid.bypass_scale"),
        )?;
        let attn = self_attention_tensor(
            ctx,
            model,
            src,
            frames,
            layer_prefix,
            "self_attn2",
            &attn_weights,
        )?;
        src = ffi::ggml_add(ctx, src, attn);
        src = ffi::ggml_add(ctx, src, ffi::ggml_repeat(ctx, time, src));
        src = conv_module_tensor(
            ctx,
            model,
            src,
            frames,
            model.fm_decoder_dim(),
            kernel,
            &format!("{layer_prefix}.conv_module2"),
            scalars,
        )?;
        src = feed_forward_tensor(
            ctx,
            model,
            src,
            &format!("{layer_prefix}.feed_forward3"),
            scalars,
        )?;
        src = bias_norm_tensor(ctx, model, src, &format!("{layer_prefix}.norm"), scalars)?;
        bypass_tensor(
            ctx,
            model,
            src_orig,
            src,
            &format!("{layer_prefix}.bypass.bypass_scale"),
        )
    }
}

unsafe fn nonlin_attention_tensor(
    ctx: *mut ffi::ggml_context,
    model: &ZipVoiceModel,
    src: *mut ffi::ggml_tensor,
    frames: usize,
    layer_prefix: &str,
    weights: *mut ffi::ggml_tensor,
) -> Result<*mut ffi::ggml_tensor> {
    unsafe {
        let channels = model.fm_decoder_dim();
        let hidden = 3 * channels / 4;
        let projected = linear_full(
            ctx,
            model,
            src,
            &format!("{layer_prefix}.nonlin_attention.in_proj"),
        )?;
        let stride = 3 * hidden * std::mem::size_of::<f32>();
        let s = ffi::ggml_view_2d(ctx, projected, hidden as i64, frames as i64, stride, 0);
        let x = ffi::ggml_view_2d(
            ctx,
            projected,
            hidden as i64,
            frames as i64,
            stride,
            hidden * std::mem::size_of::<f32>(),
        );
        let y = ffi::ggml_view_2d(
            ctx,
            projected,
            hidden as i64,
            frames as i64,
            stride,
            2 * hidden * std::mem::size_of::<f32>(),
        );
        let x = ffi::ggml_mul(ctx, x, ffi::ggml_tanh(ctx, s));
        let x_t = ffi::ggml_cont_2d(
            ctx,
            ffi::ggml_permute(ctx, x, 1, 0, 2, 3),
            frames as i64,
            hidden as i64,
        );
        let x_t = ffi::ggml_mul_mat(ctx, weights, x_t);
        let x = ffi::ggml_cont_2d(
            ctx,
            ffi::ggml_permute(ctx, x_t, 1, 0, 2, 3),
            hidden as i64,
            frames as i64,
        );
        let x = ffi::ggml_mul(ctx, x, y);
        linear_full(
            ctx,
            model,
            x,
            &format!("{layer_prefix}.nonlin_attention.out_proj"),
        )
    }
}

unsafe fn self_attention_tensor(
    ctx: *mut ffi::ggml_context,
    model: &ZipVoiceModel,
    src: *mut ffi::ggml_tensor,
    frames: usize,
    layer_prefix: &str,
    attention_module: &str,
    attn_weights: &[*mut ffi::ggml_tensor],
) -> Result<*mut ffi::ggml_tensor> {
    unsafe {
        let channels = model.fm_decoder_dim();
        let num_heads = model.config().model.fm_decoder_num_heads as usize;
        let value_head_dim = model.config().model.value_head_dim as usize;
        let value_dim = num_heads * value_head_dim;

        let value_w = model
            .weight_tensor_full(&format!("{layer_prefix}.{attention_module}.in_proj.weight"))?;
        let value_b =
            model.weight_tensor_full(&format!("{layer_prefix}.{attention_module}.in_proj.bias"))?;
        let values = ffi::ggml_add(ctx, ffi::ggml_mul_mat(ctx, value_w, src), value_b);

        let mut heads: *mut ffi::ggml_tensor = ptr::null_mut();
        let value_stride = value_dim * std::mem::size_of::<f32>();
        for head in 0..num_heads {
            let v_offset = head * value_head_dim * std::mem::size_of::<f32>();
            let v = ffi::ggml_view_2d(
                ctx,
                values,
                value_head_dim as i64,
                frames as i64,
                value_stride,
                v_offset,
            );
            let v_t = ffi::ggml_cont_2d(
                ctx,
                ffi::ggml_permute(ctx, v, 1, 0, 2, 3),
                frames as i64,
                value_head_dim as i64,
            );
            let head_t = ffi::ggml_mul_mat(ctx, attn_weights[head], v_t);
            let head_out = ffi::ggml_cont_2d(
                ctx,
                ffi::ggml_permute(ctx, head_t, 1, 0, 2, 3),
                value_head_dim as i64,
                frames as i64,
            );
            heads = if heads.is_null() {
                head_out
            } else {
                ffi::ggml_concat(ctx, heads, head_out, 0)
            };
        }

        let out_w = model.weight_tensor_full(&format!(
            "{layer_prefix}.{attention_module}.out_proj.weight"
        ))?;
        let out_b = model
            .weight_tensor_full(&format!("{layer_prefix}.{attention_module}.out_proj.bias"))?;
        let out = ffi::ggml_add(ctx, ffi::ggml_mul_mat(ctx, out_w, heads), out_b);
        if channels == model.fm_decoder_dim() {
            Ok(out)
        } else {
            Err(ZipVoiceError::Ggml(
                "flow attention channel mismatch".into(),
            ))
        }
    }
}

unsafe fn attention_weights_by_head(
    ctx: *mut ffi::ggml_context,
    model: &ZipVoiceModel,
    src: *mut ffi::ggml_tensor,
    frames: usize,
    layer_prefix: &str,
    aux_inputs: &mut Vec<(*mut ffi::ggml_tensor, Vec<f32>)>,
) -> Result<Vec<*mut ffi::ggml_tensor>> {
    unsafe {
        let num_heads = model.config().model.fm_decoder_num_heads as usize;
        let query_head_dim = model.config().model.query_head_dim as usize;
        let pos_head_dim = model.config().model.pos_head_dim as usize;
        let pos_dim = model.config().model.pos_dim as usize;
        let query_dim = num_heads * query_head_dim;
        let saw_w = model
            .weight_tensor_full(&format!("{layer_prefix}.self_attn_weights.in_proj.weight"))?;
        let saw_b =
            model.weight_tensor_full(&format!("{layer_prefix}.self_attn_weights.in_proj.bias"))?;
        let qkp = ffi::ggml_add(ctx, ffi::ggml_mul_mat(ctx, saw_w, src), saw_b);
        let qkp_stride = (2 * query_dim + num_heads * model.config().model.pos_head_dim as usize)
            * std::mem::size_of::<f32>();
        let seq_len2 = 2 * frames - 1;
        let pos_t = ffi::ggml_new_tensor_2d(
            ctx,
            ffi::ggml_type_GGML_TYPE_F32,
            pos_dim as i64,
            seq_len2 as i64,
        );
        ffi::ggml_set_input(pos_t);
        aux_inputs.push((pos_t, compact_relative_positional_encoding(frames, pos_dim)));
        let pos_w = model.weight_tensor_full(&format!(
            "{layer_prefix}.self_attn_weights.linear_pos.weight"
        ))?;
        let pos = ffi::ggml_mul_mat(ctx, pos_w, pos_t);
        let pos_stride = num_heads * pos_head_dim * std::mem::size_of::<f32>();
        let mut weights = Vec::with_capacity(num_heads);
        for head in 0..num_heads {
            let q_offset = head * query_head_dim * std::mem::size_of::<f32>();
            let k_offset = (query_dim + head * query_head_dim) * std::mem::size_of::<f32>();
            let p_offset = (2 * query_dim + head * pos_head_dim) * std::mem::size_of::<f32>();
            let q = ffi::ggml_view_2d(
                ctx,
                qkp,
                query_head_dim as i64,
                frames as i64,
                qkp_stride,
                q_offset,
            );
            let k = ffi::ggml_view_2d(
                ctx,
                qkp,
                query_head_dim as i64,
                frames as i64,
                qkp_stride,
                k_offset,
            );
            let p = ffi::ggml_view_2d(
                ctx,
                qkp,
                pos_head_dim as i64,
                frames as i64,
                qkp_stride,
                p_offset,
            );
            let pos_head = ffi::ggml_view_2d(
                ctx,
                pos,
                pos_head_dim as i64,
                seq_len2 as i64,
                pos_stride,
                head * pos_head_dim * std::mem::size_of::<f32>(),
            );
            let pos_scores = ffi::ggml_mul_mat(ctx, pos_head, p);
            let pos_scores = ffi::ggml_view_2d(
                ctx,
                pos_scores,
                frames as i64,
                frames as i64,
                (2 * frames - 2) * std::mem::size_of::<f32>(),
                (frames - 1) * std::mem::size_of::<f32>(),
            );
            let scores = ffi::ggml_add(ctx, ffi::ggml_mul_mat(ctx, k, q), pos_scores);
            weights.push(ffi::ggml_soft_max(ctx, scores));
        }
        Ok(weights)
    }
}

fn compact_relative_positional_encoding(frames: usize, pos_dim: usize) -> Vec<f32> {
    let seq_len2 = 2 * frames - 1;
    let half = pos_dim / 2;
    let compression_length = (pos_dim as f32).sqrt();
    let length_scale = pos_dim as f32 / std::f32::consts::TAU;
    let mut out = vec![0.0_f32; pos_dim * seq_len2];
    for rel_idx in 0..seq_len2 {
        let offset = rel_idx as isize - (frames as isize - 1);
        let x = offset as f32;
        let sign = x.signum();
        let compressed = compression_length
            * sign
            * ((x.abs() + compression_length).ln() - compression_length.ln());
        let x_atan = (compressed / length_scale).atan();
        for idx in 0..half {
            let freq = (idx + 1) as f32;
            out[rel_idx * pos_dim + 2 * idx] = (x_atan * freq).cos();
            out[rel_idx * pos_dim + 2 * idx + 1] = (x_atan * freq).sin();
        }
        out[rel_idx * pos_dim + pos_dim - 1] = 1.0;
    }
    out
}

unsafe fn simple_downsample_tensor(
    ctx: *mut ffi::ggml_context,
    model: &ZipVoiceModel,
    src: *mut ffi::ggml_tensor,
    frames: usize,
    channels: usize,
    downsample: usize,
    module: &str,
    scalars: &mut Vec<(*mut ffi::ggml_tensor, f32)>,
) -> Result<*mut ffi::ggml_tensor> {
    unsafe {
        let bias = model.tensor_f32_full(&format!("{module}.bias"))?;
        if bias.len() != downsample {
            return Err(ZipVoiceError::Ggml(format!(
                "bad downsample bias length for {module}: got {}, expected {downsample}",
                bias.len()
            )));
        }
        let weights = softmax(&bias);
        let down_frames = frames.div_ceil(downsample);
        let padded_frames = down_frames * downsample;
        let src = if padded_frames == frames {
            src
        } else {
            let last = ffi::ggml_view_2d(
                ctx,
                src,
                channels as i64,
                1,
                channels * std::mem::size_of::<f32>(),
                (frames - 1) * channels * std::mem::size_of::<f32>(),
            );
            let pad_target = ffi::ggml_new_tensor_2d(
                ctx,
                ffi::ggml_type_GGML_TYPE_F32,
                channels as i64,
                (padded_frames - frames) as i64,
            );
            let pad = ffi::ggml_repeat(ctx, last, pad_target);
            ffi::ggml_concat(ctx, src, pad, 1)
        };
        let column_bytes = channels * std::mem::size_of::<f32>();
        let row_stride = (channels * downsample * std::mem::size_of::<f32>()) as usize;
        let mut out: *mut ffi::ggml_tensor = ptr::null_mut();
        for (offset, &weight) in weights.iter().enumerate() {
            let view = ffi::ggml_view_2d(
                ctx,
                src,
                channels as i64,
                down_frames as i64,
                row_stride,
                offset * column_bytes,
            );
            let weighted = ffi::ggml_mul(
                ctx,
                view,
                ffi::ggml_repeat(ctx, scalar(ctx, weight, scalars), view),
            );
            out = if out.is_null() {
                weighted
            } else {
                ffi::ggml_add(ctx, out, weighted)
            };
        }
        Ok(out)
    }
}

unsafe fn simple_upsample_tensor(
    ctx: *mut ffi::ggml_context,
    src: *mut ffi::ggml_tensor,
    frames: usize,
    channels: usize,
    upsample: usize,
    trim_frames: usize,
) -> *mut ffi::ggml_tensor {
    unsafe {
        let src = ffi::ggml_reshape_3d(ctx, src, channels as i64, 1, frames as i64);
        let repeated =
            ffi::ggml_repeat_4d(ctx, src, channels as i64, upsample as i64, frames as i64, 1);
        let repeated =
            ffi::ggml_cont_2d(ctx, repeated, channels as i64, (frames * upsample) as i64);
        if trim_frames == frames * upsample {
            repeated
        } else {
            ffi::ggml_cont_2d(
                ctx,
                ffi::ggml_view_2d(
                    ctx,
                    repeated,
                    channels as i64,
                    trim_frames as i64,
                    (channels * std::mem::size_of::<f32>()) as usize,
                    0,
                ),
                channels as i64,
                trim_frames as i64,
            )
        }
    }
}

unsafe fn input_tensor(
    ctx: *mut ffi::ggml_context,
    feat_dim: usize,
    frames: usize,
) -> *mut ffi::ggml_tensor {
    unsafe {
        let tensor = ffi::ggml_new_tensor_2d(
            ctx,
            ffi::ggml_type_GGML_TYPE_F32,
            feat_dim as i64,
            frames as i64,
        );
        ffi::ggml_set_input(tensor);
        tensor
    }
}

unsafe fn set_tensor(tensor: *mut ffi::ggml_tensor, data: &[f32]) {
    unsafe {
        ffi::ggml_backend_tensor_set(tensor, data.as_ptr().cast(), 0, std::mem::size_of_val(data));
    }
}

fn timestep_embedding(timestep: f32, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let mut out = Vec::with_capacity(dim);
    for idx in 0..half {
        let freq = (-10000.0_f32.ln() * idx as f32 / half as f32).exp();
        let arg = timestep * freq;
        out.push(arg.cos());
    }
    for idx in 0..half {
        let freq = (-10000.0_f32.ln() * idx as f32 / half as f32).exp();
        let arg = timestep * freq;
        out.push(arg.sin());
    }
    if dim % 2 == 1 {
        out.push(0.0);
    }
    out
}

fn softmax(values: &[f32]) -> Vec<f32> {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut out = values
        .iter()
        .map(|value| (value - max).exp())
        .collect::<Vec<_>>();
    let sum = out.iter().sum::<f32>().max(1e-20);
    for value in &mut out {
        *value /= sum;
    }
    out
}

struct XorShift64 {
    state: u64,
    spare: Option<f32>,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.max(1),
            spare: None,
        }
    }

    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        (x >> 32) as u32
    }

    fn next_f32_open01(&mut self) -> f32 {
        ((self.next_u32() as f32) + 0.5) / ((u32::MAX as f32) + 1.0)
    }

    fn next_gaussian(&mut self) -> f32 {
        if let Some(spare) = self.spare.take() {
            return spare;
        }
        let u1 = self.next_f32_open01().max(1e-12);
        let u2 = self.next_f32_open01();
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = std::f32::consts::TAU * u2;
        let z0 = radius * theta.cos();
        let z1 = radius * theta.sin();
        self.spare = Some(z1);
        z0
    }
}

unsafe fn linear_full(
    ctx: *mut ffi::ggml_context,
    model: &ZipVoiceModel,
    input: *mut ffi::ggml_tensor,
    module: &str,
) -> Result<*mut ffi::ggml_tensor> {
    unsafe {
        let weight = model.weight_tensor_full(&format!("{module}.weight"))?;
        let bias = model.weight_tensor_full(&format!("{module}.bias"))?;
        Ok(ffi::ggml_add(
            ctx,
            ffi::ggml_mul_mat(ctx, weight, input),
            bias,
        ))
    }
}

unsafe fn feed_forward_tensor(
    ctx: *mut ffi::ggml_context,
    model: &ZipVoiceModel,
    src: *mut ffi::ggml_tensor,
    module: &str,
    scalars: &mut Vec<(*mut ffi::ggml_tensor, f32)>,
) -> Result<*mut ffi::ggml_tensor> {
    unsafe {
        let hidden = linear_full(ctx, model, src, &format!("{module}.in_proj"))?;
        let hidden = swoosh_l(ctx, hidden, scalars);
        let ff = linear_full(ctx, model, hidden, &format!("{module}.out_proj"))?;
        Ok(ffi::ggml_add(ctx, src, ff))
    }
}

unsafe fn conv_module_tensor(
    ctx: *mut ffi::ggml_context,
    model: &ZipVoiceModel,
    src: *mut ffi::ggml_tensor,
    frames: usize,
    channels: usize,
    kernel: usize,
    module: &str,
    scalars: &mut Vec<(*mut ffi::ggml_tensor, f32)>,
) -> Result<*mut ffi::ggml_tensor> {
    unsafe {
        let in_w = model.weight_tensor_full(&format!("{module}.in_proj.weight"))?;
        let in_b = model.weight_tensor_full(&format!("{module}.in_proj.bias"))?;
        let dw_w = model.weight_tensor_full(&format!("{module}.depthwise_conv.weight"))?;
        let dw_b = model.weight_tensor_full(&format!("{module}.depthwise_conv.bias"))?;
        let out_w = model.weight_tensor_full(&format!("{module}.out_proj.weight"))?;
        let out_b = model.weight_tensor_full(&format!("{module}.out_proj.bias"))?;

        let h = ffi::ggml_add(ctx, ffi::ggml_mul_mat(ctx, in_w, src), in_b);
        let stride = channels * 2 * std::mem::size_of::<f32>();
        let x = ffi::ggml_view_2d(ctx, h, channels as i64, frames as i64, stride, 0);
        let gate = ffi::ggml_view_2d(
            ctx,
            h,
            channels as i64,
            frames as i64,
            stride,
            channels * std::mem::size_of::<f32>(),
        );
        let h = ffi::ggml_mul(ctx, x, ffi::ggml_sigmoid(ctx, gate));
        let h = conv1d_depthwise(ctx, h, frames, channels, kernel, dw_w, dw_b);
        let h = swoosh_r(ctx, h, scalars);
        let h = ffi::ggml_add(ctx, ffi::ggml_mul_mat(ctx, out_w, h), out_b);
        Ok(ffi::ggml_add(ctx, src, h))
    }
}

unsafe fn bypass_tensor(
    ctx: *mut ffi::ggml_context,
    model: &ZipVoiceModel,
    src_orig: *mut ffi::ggml_tensor,
    src: *mut ffi::ggml_tensor,
    scale_name: &str,
) -> Result<*mut ffi::ggml_tensor> {
    unsafe {
        let scale = model.weight_tensor_full(scale_name)?;
        let delta = ffi::ggml_sub(ctx, src, src_orig);
        Ok(ffi::ggml_add(
            ctx,
            src_orig,
            ffi::ggml_mul(ctx, delta, ffi::ggml_repeat(ctx, scale, delta)),
        ))
    }
}

unsafe fn bias_norm_tensor(
    ctx: *mut ffi::ggml_context,
    model: &ZipVoiceModel,
    src: *mut ffi::ggml_tensor,
    module: &str,
    scalars: &mut Vec<(*mut ffi::ggml_tensor, f32)>,
) -> Result<*mut ffi::ggml_tensor> {
    unsafe {
        let bias = model.weight_tensor_full(&format!("{module}.bias"))?;
        let log_scale = model.tensor_f32_full(&format!("{module}.log_scale"))?;
        let scale = log_scale.first().copied().unwrap_or(0.0).exp();
        let centered = ffi::ggml_sub(ctx, src, ffi::ggml_repeat(ctx, bias, src));
        let power = ffi::ggml_mean(ctx, ffi::ggml_sqr(ctx, centered));
        let scale = ffi::ggml_repeat(ctx, scalar(ctx, scale, scalars), power);
        let inv_rms = ffi::ggml_div(ctx, scale, ffi::ggml_sqrt(ctx, power));
        Ok(ffi::ggml_mul(ctx, src, ffi::ggml_repeat(ctx, inv_rms, src)))
    }
}

unsafe fn conv1d_depthwise(
    ctx: *mut ffi::ggml_context,
    input: *mut ffi::ggml_tensor,
    frames: usize,
    channels: usize,
    kernel: usize,
    weight: *mut ffi::ggml_tensor,
    bias: *mut ffi::ggml_tensor,
) -> *mut ffi::ggml_tensor {
    unsafe {
        let conv_input = to_conv_layout(ctx, input, frames, channels);
        let conv = ffi::ggml_conv_1d_dw(ctx, weight, conv_input, 1, (kernel / 2) as i32, 1);
        let bias = ffi::ggml_reshape_2d(ctx, bias, 1, channels as i64);
        let conv = ffi::ggml_add(ctx, conv, ffi::ggml_repeat(ctx, bias, conv));
        to_channel_layout(ctx, conv, frames, channels)
    }
}

unsafe fn to_conv_layout(
    ctx: *mut ffi::ggml_context,
    input: *mut ffi::ggml_tensor,
    frames: usize,
    channels: usize,
) -> *mut ffi::ggml_tensor {
    unsafe {
        let transposed = ffi::ggml_permute(ctx, input, 1, 0, 2, 3);
        ffi::ggml_cont_3d(ctx, transposed, frames as i64, channels as i64, 1)
    }
}

unsafe fn to_channel_layout(
    ctx: *mut ffi::ggml_context,
    input: *mut ffi::ggml_tensor,
    frames: usize,
    channels: usize,
) -> *mut ffi::ggml_tensor {
    unsafe {
        let transposed = ffi::ggml_permute(ctx, input, 1, 0, 2, 3);
        ffi::ggml_cont_2d(ctx, transposed, channels as i64, frames as i64)
    }
}

unsafe fn scalar(
    ctx: *mut ffi::ggml_context,
    value: f32,
    scalars: &mut Vec<(*mut ffi::ggml_tensor, f32)>,
) -> *mut ffi::ggml_tensor {
    unsafe {
        let tensor = ffi::ggml_new_tensor_1d(ctx, ffi::ggml_type_GGML_TYPE_F32, 1);
        ffi::ggml_set_input(tensor);
        scalars.push((tensor, value));
        tensor
    }
}

unsafe fn add_scalar(
    ctx: *mut ffi::ggml_context,
    input: *mut ffi::ggml_tensor,
    value: f32,
    scalars: &mut Vec<(*mut ffi::ggml_tensor, f32)>,
) -> *mut ffi::ggml_tensor {
    unsafe {
        let value = scalar(ctx, value, scalars);
        ffi::ggml_add(ctx, input, ffi::ggml_repeat(ctx, value, input))
    }
}

unsafe fn swoosh_l(
    ctx: *mut ffi::ggml_context,
    input: *mut ffi::ggml_tensor,
    scalars: &mut Vec<(*mut ffi::ggml_tensor, f32)>,
) -> *mut ffi::ggml_tensor {
    unsafe {
        let shifted = add_scalar(ctx, input, -4.0, scalars);
        let softplus = ffi::ggml_softplus(ctx, shifted);
        let scaled = ffi::ggml_scale(ctx, input, 0.08);
        let out = ffi::ggml_sub(ctx, softplus, scaled);
        add_scalar(ctx, out, -0.035, scalars)
    }
}

unsafe fn swoosh_r(
    ctx: *mut ffi::ggml_context,
    input: *mut ffi::ggml_tensor,
    scalars: &mut Vec<(*mut ffi::ggml_tensor, f32)>,
) -> *mut ffi::ggml_tensor {
    unsafe {
        let shifted = add_scalar(ctx, input, -1.0, scalars);
        let softplus = ffi::ggml_softplus(ctx, shifted);
        let scaled = ffi::ggml_scale(ctx, input, 0.08);
        let out = ffi::ggml_sub(ctx, softplus, scaled);
        add_scalar(ctx, out, -0.313261687, scalars)
    }
}
