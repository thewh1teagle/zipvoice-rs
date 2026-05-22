pub const SAMPLE_RATE: usize = 24_000;
pub const N_FFT: usize = 1024;
pub const HOP_LENGTH: usize = 256;
pub const N_MELS: usize = 100;
pub const FEAT_SCALE: f32 = 0.1;
pub const TARGET_RMS: f32 = 0.1;

pub struct PromptAudio {
    pub samples: Vec<f32>,
    pub original_rms: f32,
}

pub fn prepare_prompt_audio(samples: &[f32]) -> PromptAudio {
    let trimmed = remove_silence(samples, false, 200.0);
    let original_rms = rms(&trimmed);
    let samples = rms_norm(trimmed, TARGET_RMS, original_rms);
    PromptAudio {
        samples,
        original_rms,
    }
}

pub fn postprocess_generated_audio(samples: Vec<f32>, prompt_rms: f32) -> Vec<f32> {
    let scaled = if prompt_rms < TARGET_RMS {
        let scale = prompt_rms / TARGET_RMS;
        samples.into_iter().map(|sample| sample * scale).collect()
    } else {
        samples
    };
    remove_silence(&scaled, true, 0.0)
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let mean_square =
        samples.iter().map(|sample| sample * sample).sum::<f32>() / samples.len() as f32;
    mean_square.sqrt()
}

fn rms_norm(mut samples: Vec<f32>, target_rms: f32, current_rms: f32) -> Vec<f32> {
    if current_rms > 0.0 && current_rms < target_rms {
        let scale = target_rms / current_rms;
        for sample in &mut samples {
            *sample *= scale;
        }
    }
    samples
}

fn remove_silence(samples: &[f32], only_edge: bool, trail_sil_ms: f32) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }

    let mut audio = if only_edge {
        samples.to_vec()
    } else {
        remove_long_silence(samples)
    };

    let (start, end) = trim_interval(&audio);
    audio = audio[start..end].to_vec();

    if trail_sil_ms > 0.0 {
        let trail_samples = (trail_sil_ms * SAMPLE_RATE as f32 / 1000.0) as usize;
        audio.resize(audio.len() + trail_samples, 0.0);
    }

    audio
}

fn remove_long_silence(samples: &[f32]) -> Vec<f32> {
    let intervals = split_intervals(samples);
    if intervals.is_empty() {
        return samples.to_vec();
    }

    let min_silence = SAMPLE_RATE;
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (start, end) in intervals {
        if let Some((_, prev_end)) = merged.last_mut() {
            if start.saturating_sub(*prev_end) < min_silence {
                *prev_end = end;
                continue;
            }
        }
        merged.push((start, end));
    }

    let keep = SAMPLE_RATE;
    let mut out = Vec::new();
    let mut prev_end = 0;
    for (start, end) in merged {
        let segment_start = start.saturating_sub(keep).max(prev_end);
        let segment_end = (end + keep).min(samples.len());
        out.extend_from_slice(&samples[segment_start..segment_end]);
        prev_end = segment_end;
    }
    out
}

fn split_intervals(samples: &[f32]) -> Vec<(usize, usize)> {
    let frame_length = 2048;
    let hop_length = 512;
    let threshold = amplitude_threshold();
    let mut intervals = Vec::new();
    let mut active_start = None;
    let mut frame_start = 0;

    while frame_start < samples.len() {
        let frame_end = (frame_start + frame_length).min(samples.len());
        let active = samples[frame_start..frame_end]
            .iter()
            .any(|sample| sample.abs() > threshold);
        if active && active_start.is_none() {
            active_start = Some(frame_start);
        } else if !active {
            if let Some(start) = active_start.take() {
                intervals.push((start, frame_start));
            }
        }
        frame_start += hop_length;
    }

    if let Some(start) = active_start {
        intervals.push((start, samples.len()));
    }

    intervals
}

fn trim_interval(samples: &[f32]) -> (usize, usize) {
    if samples.is_empty() {
        return (0, 0);
    }

    let frame_length = 512;
    let hop_length = 128;
    let threshold = amplitude_threshold();
    let keep_edge = (0.1 * SAMPLE_RATE as f32) as usize;
    let mut first = None;
    let mut last = None;
    let pad = frame_length / 2;
    let mut padded = vec![0.0_f32; pad];
    padded.extend_from_slice(samples);
    padded.resize(samples.len() + 2 * pad, 0.0);

    let mut frame_start = 0;
    while frame_start + frame_length <= padded.len() {
        let frame = &padded[frame_start..frame_start + frame_length];
        let rms =
            (frame.iter().map(|sample| sample * sample).sum::<f32>() / frame_length as f32).sqrt();
        let active = rms > threshold;
        if active {
            first.get_or_insert(frame_start);
            last = Some(frame_start + hop_length);
        }
        frame_start += hop_length;
    }

    match (first, last) {
        (Some(start), Some(end)) => (
            start.saturating_sub(keep_edge),
            (end + keep_edge).min(samples.len()),
        ),
        _ => (0, samples.len()),
    }
}

fn amplitude_threshold() -> f32 {
    10.0_f32.powf(-50.0 / 20.0)
}
