use std::ptr;

use llama_rs_sys as ffi;

use super::{Result, ZipVoiceError, model::ZipVoiceModel};

#[derive(Debug, Clone, Copy)]
pub struct TextPlan {
    pub prompt_tokens: usize,
    pub target_tokens: usize,
    pub total_tokens: usize,
    pub prompt_frames: usize,
    pub predicted_frames: usize,
    pub total_frames: usize,
}

pub fn token_embeddings(model: &ZipVoiceModel, tokens: &[i64]) -> Result<Vec<f32>> {
    run_token_graph(model, tokens, TextGraphOutput::Embeddings)
}

pub fn input_projection(model: &ZipVoiceModel, tokens: &[i64]) -> Result<Vec<f32>> {
    run_token_graph(model, tokens, TextGraphOutput::InputProjection)
}

pub fn first_feed_forward(model: &ZipVoiceModel, tokens: &[i64]) -> Result<Vec<f32>> {
    run_token_graph(model, tokens, TextGraphOutput::FirstFeedForward)
}

pub fn first_conv_module(model: &ZipVoiceModel, tokens: &[i64]) -> Result<Vec<f32>> {
    run_token_graph(model, tokens, TextGraphOutput::FirstConvModule)
}

pub fn first_layer_no_attention(model: &ZipVoiceModel, tokens: &[i64]) -> Result<Vec<f32>> {
    run_token_graph(model, tokens, TextGraphOutput::FirstLayerNoAttention)
}

pub fn first_layer_no_attention_norm(model: &ZipVoiceModel, tokens: &[i64]) -> Result<Vec<f32>> {
    run_token_graph(model, tokens, TextGraphOutput::FirstLayerNoAttentionNorm)
}

pub fn first_layer_no_attention_out_projection(
    model: &ZipVoiceModel,
    tokens: &[i64],
) -> Result<Vec<f32>> {
    run_token_graph(
        model,
        tokens,
        TextGraphOutput::FirstLayerNoAttentionOutProjection,
    )
}

pub fn all_layers_no_attention_out_projection(
    model: &ZipVoiceModel,
    tokens: &[i64],
) -> Result<Vec<f32>> {
    run_token_graph(
        model,
        tokens,
        TextGraphOutput::AllLayersNoAttentionOutProjection,
    )
}

fn all_layers_no_attention_out_projection_masked(
    model: &ZipVoiceModel,
    tokens: &[i64],
    valid_tokens: usize,
) -> Result<Vec<f32>> {
    run_token_graph_with_valid_tokens(
        model,
        tokens,
        TextGraphOutput::AllLayersNoAttentionOutProjection,
        valid_tokens,
    )
}

pub fn text_condition_preview(
    model: &ZipVoiceModel,
    prompt_tokens: &[i64],
    target_tokens: &[i64],
    prompt_frames: usize,
    speed: f32,
) -> Result<Vec<f32>> {
    let mut tokens = Vec::with_capacity(prompt_tokens.len() + target_tokens.len() + 1);
    tokens.extend_from_slice(prompt_tokens);
    tokens.extend_from_slice(target_tokens);
    tokens.push(model.pad_id() as i64);

    let plan = plan_text_condition(
        prompt_tokens.len(),
        target_tokens.len(),
        prompt_frames,
        speed,
    );
    let token_features =
        all_layers_no_attention_out_projection_masked(model, &tokens, plan.total_tokens)?;
    let feat_dim = model.feat_dim();
    let token_dur = (plan.total_frames / plan.total_tokens.max(1)).max(1);
    let mut condition = Vec::with_capacity(plan.total_frames * feat_dim);

    for token_idx in 0..plan.total_tokens {
        let start = token_idx * feat_dim;
        let end = start + feat_dim;
        for _ in 0..token_dur {
            if condition.len() >= plan.total_frames * feat_dim {
                break;
            }
            condition.extend_from_slice(&token_features[start..end]);
        }
    }

    let pad_start = plan.total_tokens * feat_dim;
    let pad_end = pad_start + feat_dim;
    while condition.len() < plan.total_frames * feat_dim {
        condition.extend_from_slice(&token_features[pad_start..pad_end]);
    }
    condition.truncate(plan.total_frames * feat_dim);
    Ok(condition)
}

fn run_token_graph(
    model: &ZipVoiceModel,
    tokens: &[i64],
    output_kind: TextGraphOutput,
) -> Result<Vec<f32>> {
    run_token_graph_with_valid_tokens(model, tokens, output_kind, tokens.len())
}

fn run_token_graph_with_valid_tokens(
    model: &ZipVoiceModel,
    tokens: &[i64],
    output_kind: TextGraphOutput,
    valid_tokens: usize,
) -> Result<Vec<f32>> {
    let token_ids = tokens
        .iter()
        .map(|&token| token.max(0) as i32)
        .collect::<Vec<_>>();
    unsafe {
        let params = ffi::ggml_init_params {
            mem_size: ffi::ggml_tensor_overhead() * 8192
                + ffi::ggml_graph_overhead_custom(8192, false),
            mem_buffer: ptr::null_mut(),
            no_alloc: true,
        };
        let ctx = ffi::ggml_init(params);
        if ctx.is_null() {
            return Err(ZipVoiceError::Ggml(
                "failed to initialize text GGML context".into(),
            ));
        }
        let result = token_graph(ctx, model, &token_ids, output_kind, valid_tokens);
        ffi::ggml_free(ctx);
        result
    }
}

#[derive(Clone, Copy)]
enum TextGraphOutput {
    Embeddings,
    InputProjection,
    FirstFeedForward,
    FirstConvModule,
    FirstLayerNoAttention,
    FirstLayerNoAttentionNorm,
    FirstLayerNoAttentionOutProjection,
    AllLayersNoAttentionOutProjection,
}

unsafe fn token_graph(
    ctx: *mut ffi::ggml_context,
    model: &ZipVoiceModel,
    token_ids: &[i32],
    output_kind: TextGraphOutput,
    valid_tokens: usize,
) -> Result<Vec<f32>> {
    unsafe {
        let valid_tokens = valid_tokens.min(token_ids.len());
        let ids =
            ffi::ggml_new_tensor_1d(ctx, ffi::ggml_type_GGML_TYPE_I32, token_ids.len() as i64);
        if ids.is_null() {
            return Err(ZipVoiceError::Ggml(
                "failed to create token id tensor".into(),
            ));
        }
        ffi::ggml_set_input(ids);
        let embed = model.weight_tensor_full("embed.weight")?;
        let embeddings = ffi::ggml_get_rows(ctx, embed, ids);
        let mut scalars = Vec::new();
        let mut aux_inputs = Vec::new();
        let out = match output_kind {
            TextGraphOutput::Embeddings => embeddings,
            TextGraphOutput::InputProjection => {
                let weight = model.weight_tensor_full("text_encoder.in_proj.weight")?;
                let bias = model.weight_tensor_full("text_encoder.in_proj.bias")?;
                ffi::ggml_add(ctx, ffi::ggml_mul_mat(ctx, weight, embeddings), bias)
            }
            TextGraphOutput::FirstFeedForward => {
                let src = text_input_projection(ctx, model, embeddings)?;
                feed_forward_tensor(
                    ctx,
                    model,
                    src,
                    "text_encoder.encoders.0.layers.0.feed_forward1",
                    &mut scalars,
                )?
            }
            TextGraphOutput::FirstConvModule => {
                let src = text_input_projection(ctx, model, embeddings)?;
                let src = feed_forward_tensor(
                    ctx,
                    model,
                    src,
                    "text_encoder.encoders.0.layers.0.feed_forward1",
                    &mut scalars,
                )?;
                conv_module_tensor(
                    ctx,
                    model,
                    src,
                    token_ids.len(),
                    "text_encoder.encoders.0.layers.0.conv_module1",
                    &mut scalars,
                )?
            }
            TextGraphOutput::FirstLayerNoAttention => {
                let src_orig = text_input_projection(ctx, model, embeddings)?;
                first_layer_no_attention_tensor(
                    ctx,
                    model,
                    src_orig,
                    token_ids.len(),
                    valid_tokens,
                    &mut scalars,
                    &mut aux_inputs,
                )?
            }
            TextGraphOutput::FirstLayerNoAttentionNorm => {
                let src_orig = text_input_projection(ctx, model, embeddings)?;
                first_layer_no_attention_tensor(
                    ctx,
                    model,
                    src_orig,
                    token_ids.len(),
                    valid_tokens,
                    &mut scalars,
                    &mut aux_inputs,
                )?
            }
            TextGraphOutput::FirstLayerNoAttentionOutProjection => {
                let src_orig = text_input_projection(ctx, model, embeddings)?;
                let src = first_layer_no_attention_tensor(
                    ctx,
                    model,
                    src_orig,
                    token_ids.len(),
                    valid_tokens,
                    &mut scalars,
                    &mut aux_inputs,
                )?;
                let out_w = model.weight_tensor_full("text_encoder.out_proj.weight")?;
                let out_b = model.weight_tensor_full("text_encoder.out_proj.bias")?;
                ffi::ggml_add(ctx, ffi::ggml_mul_mat(ctx, out_w, src), out_b)
            }
            TextGraphOutput::AllLayersNoAttentionOutProjection => {
                let src = text_input_projection(ctx, model, embeddings)?;
                text_encoder_no_attention_tensor(
                    ctx,
                    model,
                    src,
                    token_ids.len(),
                    valid_tokens,
                    &mut scalars,
                    &mut aux_inputs,
                )?
            }
        };
        let graph = ffi::ggml_new_graph_custom(ctx, 8192, false);
        if graph.is_null() {
            return Err(ZipVoiceError::Ggml("failed to create text graph".into()));
        }
        ffi::ggml_build_forward_expand(graph, out);
        model.alloc_graph(graph)?;
        ffi::ggml_backend_tensor_set(
            ids,
            token_ids.as_ptr().cast(),
            0,
            std::mem::size_of_val(token_ids),
        );
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
        let out_dim = match output_kind {
            TextGraphOutput::Embeddings | TextGraphOutput::InputProjection => {
                model.text_embed_dim()
            }
            TextGraphOutput::FirstFeedForward
            | TextGraphOutput::FirstConvModule
            | TextGraphOutput::FirstLayerNoAttention
            | TextGraphOutput::FirstLayerNoAttentionNorm => {
                model.config().model.text_encoder_dim as usize
            }
            TextGraphOutput::FirstLayerNoAttentionOutProjection
            | TextGraphOutput::AllLayersNoAttentionOutProjection => model.feat_dim(),
        };
        let mut out_data = vec![0.0_f32; token_ids.len() * out_dim];
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

unsafe fn text_encoder_no_attention_tensor(
    ctx: *mut ffi::ggml_context,
    model: &ZipVoiceModel,
    mut src: *mut ffi::ggml_tensor,
    frames: usize,
    valid_frames: usize,
    scalars: &mut Vec<(*mut ffi::ggml_tensor, f32)>,
    aux_inputs: &mut Vec<(*mut ffi::ggml_tensor, Vec<f32>)>,
) -> Result<*mut ffi::ggml_tensor> {
    unsafe {
        for layer in 0..model.config().model.text_encoder_num_layers as usize {
            src = text_layer_no_attention_tensor(
                ctx,
                model,
                src,
                frames,
                valid_frames,
                &format!("text_encoder.encoders.0.layers.{layer}"),
                scalars,
                aux_inputs,
            )?;
        }
        let out_w = model.weight_tensor_full("text_encoder.out_proj.weight")?;
        let out_b = model.weight_tensor_full("text_encoder.out_proj.bias")?;
        Ok(ffi::ggml_add(
            ctx,
            ffi::ggml_mul_mat(ctx, out_w, src),
            out_b,
        ))
    }
}

unsafe fn first_layer_no_attention_tensor(
    ctx: *mut ffi::ggml_context,
    model: &ZipVoiceModel,
    src_orig: *mut ffi::ggml_tensor,
    frames: usize,
    valid_frames: usize,
    scalars: &mut Vec<(*mut ffi::ggml_tensor, f32)>,
    aux_inputs: &mut Vec<(*mut ffi::ggml_tensor, Vec<f32>)>,
) -> Result<*mut ffi::ggml_tensor> {
    unsafe {
        text_layer_no_attention_tensor(
            ctx,
            model,
            src_orig,
            frames,
            valid_frames,
            "text_encoder.encoders.0.layers.0",
            scalars,
            aux_inputs,
        )
    }
}

unsafe fn text_layer_no_attention_tensor(
    ctx: *mut ffi::ggml_context,
    model: &ZipVoiceModel,
    src_orig: *mut ffi::ggml_tensor,
    frames: usize,
    valid_frames: usize,
    layer_prefix: &str,
    scalars: &mut Vec<(*mut ffi::ggml_tensor, f32)>,
    aux_inputs: &mut Vec<(*mut ffi::ggml_tensor, Vec<f32>)>,
) -> Result<*mut ffi::ggml_tensor> {
    unsafe {
        let attn_weights = attention_weights_by_head(
            ctx,
            model,
            src_orig,
            frames,
            valid_frames,
            layer_prefix,
            aux_inputs,
        )?;
        let src = feed_forward_tensor(
            ctx,
            model,
            src_orig,
            &format!("{layer_prefix}.feed_forward1"),
            scalars,
        )?;
        let na = nonlin_attention_tensor(ctx, model, src, frames, layer_prefix, attn_weights[0])?;
        let src = ffi::ggml_add(ctx, src, na);
        let attn = self_attention_tensor(
            ctx,
            model,
            src,
            frames,
            layer_prefix,
            "self_attn1",
            &attn_weights,
        )?;
        let src = ffi::ggml_add(ctx, src, attn);
        let src = conv_module_tensor(
            ctx,
            model,
            src,
            frames,
            &format!("{layer_prefix}.conv_module1"),
            scalars,
        )?;
        let src = feed_forward_tensor(
            ctx,
            model,
            src,
            &format!("{layer_prefix}.feed_forward2"),
            scalars,
        )?;
        let src = bypass_tensor(
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
        let src = ffi::ggml_add(ctx, src, attn);
        let src = conv_module_tensor(
            ctx,
            model,
            src,
            frames,
            &format!("{layer_prefix}.conv_module2"),
            scalars,
        )?;
        let src = feed_forward_tensor(
            ctx,
            model,
            src,
            &format!("{layer_prefix}.feed_forward3"),
            scalars,
        )?;
        let src = bias_norm_tensor(ctx, model, src, &format!("{layer_prefix}.norm"), scalars)?;
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
        let channels = model.config().model.text_encoder_dim as usize;
        let hidden = 3 * channels / 4;
        let projected = {
            let weight = model
                .weight_tensor_full(&format!("{layer_prefix}.nonlin_attention.in_proj.weight"))?;
            let bias = model
                .weight_tensor_full(&format!("{layer_prefix}.nonlin_attention.in_proj.bias"))?;
            ffi::ggml_add(ctx, ffi::ggml_mul_mat(ctx, weight, src), bias)
        };
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
        let weight = model
            .weight_tensor_full(&format!("{layer_prefix}.nonlin_attention.out_proj.weight"))?;
        let bias =
            model.weight_tensor_full(&format!("{layer_prefix}.nonlin_attention.out_proj.bias"))?;
        Ok(ffi::ggml_add(ctx, ffi::ggml_mul_mat(ctx, weight, x), bias))
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
        let channels = model.config().model.text_encoder_dim as usize;
        let num_heads = model.config().model.text_encoder_num_heads as usize;
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
        if channels == model.config().model.text_encoder_dim as usize {
            Ok(out)
        } else {
            Err(ZipVoiceError::Ggml(
                "text attention channel mismatch".into(),
            ))
        }
    }
}

unsafe fn attention_weights_by_head(
    ctx: *mut ffi::ggml_context,
    model: &ZipVoiceModel,
    src: *mut ffi::ggml_tensor,
    frames: usize,
    valid_frames: usize,
    layer_prefix: &str,
    aux_inputs: &mut Vec<(*mut ffi::ggml_tensor, Vec<f32>)>,
) -> Result<Vec<*mut ffi::ggml_tensor>> {
    unsafe {
        let num_heads = model.config().model.text_encoder_num_heads as usize;
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
            let mut scores = ffi::ggml_add(ctx, ffi::ggml_mul_mat(ctx, k, q), pos_scores);
            if valid_frames < frames {
                let mask_t = ffi::ggml_new_tensor_2d(
                    ctx,
                    ffi::ggml_type_GGML_TYPE_F32,
                    frames as i64,
                    frames as i64,
                );
                ffi::ggml_set_input(mask_t);
                aux_inputs.push((mask_t, attention_padding_mask(frames, valid_frames)));
                scores = ffi::ggml_add(ctx, scores, mask_t);
            }
            weights.push(ffi::ggml_soft_max(ctx, scores));
        }
        Ok(weights)
    }
}

fn attention_padding_mask(frames: usize, valid_frames: usize) -> Vec<f32> {
    let mut mask = vec![0.0_f32; frames * frames];
    for query in 0..frames {
        for key in valid_frames..frames {
            mask[query * frames + key] = -1.0e9;
        }
    }
    mask
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

unsafe fn text_input_projection(
    ctx: *mut ffi::ggml_context,
    model: &ZipVoiceModel,
    embeddings: *mut ffi::ggml_tensor,
) -> Result<*mut ffi::ggml_tensor> {
    unsafe {
        let in_w = model.weight_tensor_full("text_encoder.in_proj.weight")?;
        let in_b = model.weight_tensor_full("text_encoder.in_proj.bias")?;
        Ok(ffi::ggml_add(
            ctx,
            ffi::ggml_mul_mat(ctx, in_w, embeddings),
            in_b,
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
        let ff_in_w = model.weight_tensor_full(&format!("{module}.in_proj.weight"))?;
        let ff_in_b = model.weight_tensor_full(&format!("{module}.in_proj.bias"))?;
        let ff_out_w = model.weight_tensor_full(&format!("{module}.out_proj.weight"))?;
        let ff_out_b = model.weight_tensor_full(&format!("{module}.out_proj.bias"))?;
        let hidden = ffi::ggml_add(ctx, ffi::ggml_mul_mat(ctx, ff_in_w, src), ff_in_b);
        let hidden = swoosh_l(ctx, hidden, scalars);
        let ff = ffi::ggml_add(ctx, ffi::ggml_mul_mat(ctx, ff_out_w, hidden), ff_out_b);
        Ok(ffi::ggml_add(ctx, src, ff))
    }
}

unsafe fn conv_module_tensor(
    ctx: *mut ffi::ggml_context,
    model: &ZipVoiceModel,
    src: *mut ffi::ggml_tensor,
    frames: usize,
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
        let channels = model.config().model.text_encoder_dim as usize;
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
        let h = conv1d_depthwise(ctx, h, frames, channels, 9, dw_w, dw_b);
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

pub fn token_embeddings_cpu(model: &ZipVoiceModel, tokens: &[i64]) -> Result<Vec<f32>> {
    let embed = model.tensor_f32_full("embed.weight")?;
    let dim = model.text_embed_dim();
    let vocab = model.vocab_size() as usize;
    let mut out = vec![0.0_f32; tokens.len() * dim];
    for (pos, &token) in tokens.iter().enumerate() {
        let token = token.max(0) as usize;
        if token >= vocab {
            continue;
        }
        let src = token * dim;
        let dst = pos * dim;
        out[dst..dst + dim].copy_from_slice(&embed[src..src + dim]);
    }
    Ok(out)
}

pub fn plan_text_condition(
    prompt_tokens: usize,
    target_tokens: usize,
    prompt_frames: usize,
    speed: f32,
) -> TextPlan {
    let speed = speed.max(1e-6);
    let predicted_frames = ((prompt_frames as f32 / prompt_tokens.max(1) as f32)
        * target_tokens as f32
        / speed)
        .ceil() as usize;
    TextPlan {
        prompt_tokens,
        target_tokens,
        total_tokens: prompt_tokens + target_tokens,
        prompt_frames,
        predicted_frames,
        total_frames: prompt_frames + predicted_frames,
    }
}
