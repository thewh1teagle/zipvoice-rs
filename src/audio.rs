use std::path::Path;

use thiserror::Error;

#[derive(Debug, Error)]
pub enum AudioError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("wav error: {0}")]
    Wav(#[from] hound::Error),
}

pub type Result<T> = std::result::Result<T, AudioError>;

pub fn write_wav_mono_16bit(
    path: impl AsRef<Path>,
    samples: &[f32],
    sample_rate: u32,
) -> Result<()> {
    if let Some(parent) = path.as_ref().parent() {
        std::fs::create_dir_all(parent)?;
    }
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &sample in samples {
        writer.write_sample((sample.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16)?;
    }
    writer.finalize()?;
    Ok(())
}

pub fn write_wav_24khz(path: impl AsRef<Path>, samples: &[f32]) -> Result<()> {
    write_wav_mono_16bit(path, samples, 24_000)
}
