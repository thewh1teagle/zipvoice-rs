/*
Prepare assets and models:
    mkdir -p assets models/zipvoice-heb models/vocos
    wget https://github.com/thewh1teagle/phonikud-chatterbox/releases/download/asset-files-v1/female1.wav -O assets/female1.wav
    wget https://github.com/thewh1teagle/zipvoice-rs/releases/download/models-v0.1.0/zipvoice-heb-q8_0.gguf -O models/zipvoice-heb/zipvoice-heb-q8_0.gguf
    wget https://github.com/thewh1teagle/zipvoice-rs/releases/download/models-v0.1.0/vocos-mel-24khz-q8_0.gguf -O models/vocos/vocos-mel-24khz-q8_0.gguf

Run:
    cargo run --release --example basic_espeak --features phonemize-espeak
*/

use std::path::Path;

use zipvoice_rs::{ZipVoice, write_wav_mono_16bit};

const ZIPVOICE_MODEL: &str = "models/zipvoice-heb/zipvoice-heb-q8_0.gguf";
const VOCOS_MODEL: &str = "models/vocos/vocos-mel-24khz-q8_0.gguf";
const REF_WAV: &str = "assets/female1.wav";
const OUTPUT: &str = "output/basic_espeak_generated.wav";

const REF_PHONEMES: &str = "halˈaχti lamakˈolet liknˈot lˈeχem veχalˈav, ubadˈeʁeχ paɡˈaʃti χavˈeʁ jaʃˈan ʃelˈo ʁaʔˈiti haʁbˈe zmˈan.";
const TARGET_TEXT: &str = "The morning train arrived beside the old stone bridge.";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let target_phonemes = text_to_ipa("en", TARGET_TEXT)?;
    let zipvoice = ZipVoice::load_with_vocos(ZIPVOICE_MODEL, VOCOS_MODEL)?;
    let (samples, sample_rate) =
        zipvoice.create(REF_WAV, REF_PHONEMES, target_phonemes.as_str())?;

    std::fs::create_dir_all("output")?;
    write_wav_mono_16bit(OUTPUT, &samples, sample_rate)?;

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
