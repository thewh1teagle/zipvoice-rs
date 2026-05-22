/*
Prepare assets and models:
    mkdir -p assets models/zipvoice-en models/vocos
    wget https://github.com/thewh1teagle/zipvoice-rs/releases/download/models-v0.1.0/whisper.wav -O assets/whisper.wav
    wget https://github.com/thewh1teagle/zipvoice-rs/releases/download/models-v0.1.0/zipvoice-en-q8_0.gguf -O models/zipvoice-en/zipvoice-en-q8_0.gguf
    wget https://github.com/thewh1teagle/zipvoice-rs/releases/download/models-v0.1.0/vocos-mel-24khz-q8_0.gguf -O models/vocos/vocos-mel-24khz-q8_0.gguf

Run:
    cargo run --release --example basic_espeak --features phonemize-espeak
*/

use std::path::Path;

use zipvoice_rs::{ZipVoice, write_wav_mono_16bit};

const ZIPVOICE_MODEL: &str = "models/zipvoice-en/zipvoice-en-q8_0.gguf";
const VOCOS_MODEL: &str = "models/vocos/vocos-mel-24khz-q8_0.gguf";
const REF_WAV: &str = "assets/whisper.wav";
const OUTPUT: &str = "output/basic_espeak_generated.wav";

const REF_TEXT: &str = "Real change begins when your hope becomes stronger than your excuses.";
const TARGET_TEXT: &str = "The morning train arrived beside the old stone bridge.";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ref_phonemes = text_to_ipa("en-us", REF_TEXT)?;
    let target_phonemes = text_to_ipa("en", TARGET_TEXT)?;
    let zipvoice = ZipVoice::load_with_vocos(ZIPVOICE_MODEL, VOCOS_MODEL)?;
    let (samples, sample_rate) =
        zipvoice.create(REF_WAV, ref_phonemes.as_str(), target_phonemes.as_str())?;

    std::fs::create_dir_all("output")?;
    write_wav_mono_16bit(OUTPUT, &samples, sample_rate)?;

    println!("ref_phonemes={ref_phonemes}");
    println!("target_phonemes={target_phonemes}");
    println!("wrote={OUTPUT}");
    println!("duration={:.3}s", samples.len() as f32 / sample_rate as f32);
    Ok(())
}

fn text_to_ipa(lang: &str, text: &str) -> espeak_ng::Result<String> {
    let data_dir = Path::new("target/espeak-ng-data");
    std::fs::create_dir_all(data_dir)?;
    espeak_ng::install_bundled_language(data_dir, "en")?;

    let engine = espeak_ng::EspeakNg::with_data_dir(lang, data_dir)?;
    engine.text_to_phonemes(text)
}
