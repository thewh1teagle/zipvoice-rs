/*
Prepare assets and models:
    mkdir -p assets models/renikud models/zipvoice-heb models/vocos
    wget https://github.com/thewh1teagle/phonikud-chatterbox/releases/download/asset-files-v1/female1.wav -O assets/female1.wav
    wget https://huggingface.co/thewh1teagle/renikud/resolve/main/model.onnx -O models/renikud/model.onnx
    wget https://github.com/thewh1teagle/zipvoice-rs/releases/download/models-v0.1.0/zipvoice-heb-q8_0.gguf -O models/zipvoice-heb/zipvoice-heb-q8_0.gguf
    wget https://github.com/thewh1teagle/zipvoice-rs/releases/download/models-v0.1.0/vocos-mel-24khz-q8_0.gguf -O models/vocos/vocos-mel-24khz-q8_0.gguf

Run:
    cargo run --release --example basic_hebrew_text --features phonemize-hebrew
*/

use std::{path::Path, sync::LazyLock};

use regex::Regex;
use renikud_rs::G2P;
use zipvoice_rs::{CreateOptions, ZipVoice, write_wav_mono_16bit};

const ZIPVOICE_MODEL: &str = "models/zipvoice-heb/zipvoice-heb-q8_0.gguf";
const VOCOS_MODEL: &str = "models/vocos/vocos-mel-24khz-q8_0.gguf";
const RENIKUD_MODEL: &str = "models/renikud/model.onnx";
const REF_WAV: &str = "assets/female1.wav";
const OUTPUT: &str = "output/basic_hebrew_text_generated.wav";
const SPEED: f32 = 1.25;

const REF_PHONEMES: &str = "halˈaχti lamakˈolet liknˈot lˈeχem veχalˈav, ubadˈeʁeχ paɡˈaʃti χavˈeʁ jaʃˈan ʃelˈo ʁaʔˈiti haʁbˈe zmˈan.";
const TARGET_TEXT: &str = "בבוקר הכנתי קפה ופתחתי את GitHub כדי לבדוק את הפרויקט החדש.";

static LATIN_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[A-Za-z]+").unwrap());

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let english = english_phonemizer()?;
    let mut hebrew = G2P::new(RENIKUD_MODEL)?;
    let target_phonemes = text_to_phonemes(TARGET_TEXT, &mut hebrew, &english)?;

    let zipvoice = ZipVoice::load_with_vocos(ZIPVOICE_MODEL, VOCOS_MODEL)?;
    let options = CreateOptions {
        speed: SPEED,
        ..CreateOptions::default()
    };
    let (samples, sample_rate) =
        zipvoice.create_with_options(REF_WAV, REF_PHONEMES, target_phonemes.as_str(), options)?;

    std::fs::create_dir_all("output")?;
    write_wav_mono_16bit(OUTPUT, &samples, sample_rate)?;

    println!("target_text={TARGET_TEXT}");
    println!("target_phonemes={target_phonemes}");
    println!("wrote={OUTPUT}");
    println!("duration={:.3}s", samples.len() as f32 / sample_rate as f32);
    Ok(())
}

fn text_to_phonemes(
    text: &str,
    hebrew: &mut G2P,
    english: &espeak_ng::EspeakNg,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut result = String::new();
    let mut last = 0;

    for span in LATIN_RE.find_iter(text) {
        push_hebrew(&mut result, &text[last..span.start()], hebrew)?;
        result.push_str(english.text_to_phonemes(span.as_str())?.as_str());
        last = span.end();
    }

    push_hebrew(&mut result, &text[last..], hebrew)?;
    Ok(result)
}

fn push_hebrew(
    output: &mut String,
    text: &str,
    hebrew: &mut G2P,
) -> Result<(), Box<dyn std::error::Error>> {
    if !text.is_empty() {
        output.push_str(hebrew.phonemize(text)?.as_str());
    }
    Ok(())
}

fn english_phonemizer() -> espeak_ng::Result<espeak_ng::EspeakNg> {
    let data_dir = Path::new("target/espeak-ng-data");
    std::fs::create_dir_all(data_dir)?;
    espeak_ng::install_bundled_language(data_dir, "en")?;
    espeak_ng::EspeakNg::with_data_dir("en-us", data_dir)
}
