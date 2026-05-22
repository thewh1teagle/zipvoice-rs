/*
Prepare assets and models:
    mkdir -p assets models/zipvoice-heb models/vocos
    wget https://github.com/thewh1teagle/phonikud-chatterbox/releases/download/asset-files-v1/female1.wav -O assets/female1.wav
    wget https://github.com/thewh1teagle/zipvoice-rs/releases/download/models-v0.1.0/zipvoice-heb-q8_0.gguf -O models/zipvoice-heb/zipvoice-heb-q8_0.gguf
    wget https://github.com/thewh1teagle/zipvoice-rs/releases/download/models-v0.1.0/vocos-mel-24khz-q8_0.gguf -O models/vocos/vocos-mel-24khz-q8_0.gguf

Run:
    cargo run --release --example basic
*/

use zipvoice_rs::{ZipVoice, write_wav_mono_16bit};

const ZIPVOICE_MODEL: &str = "models/zipvoice-heb/zipvoice-heb-q8_0.gguf";
const VOCOS_MODEL: &str = "models/vocos/vocos-mel-24khz-q8_0.gguf";
const REF_WAV: &str = "assets/female1.wav";
const OUTPUT: &str = "output/basic_generated_preview.wav";

const REF_PHONEMES: &str = "halˈaχti lamakˈolet liknˈot lˈeχem veχalˈav, ubadˈeʁeχ paɡˈaʃti χavˈeʁ jaʃˈan ʃelˈo ʁaʔˈiti haʁbˈe zmˈan.";
const TARGET_PHONEMES: &str = "halˈaχti lamakˈolet liknˈot lˈeχem veχalˈav, ubadˈeʁeχ paɡˈaʃti χavˈeʁ jaʃˈan ʃelˈo ʁaʔˈiti haʁbˈe zmˈan.";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let zipvoice = ZipVoice::load_with_vocos(ZIPVOICE_MODEL, VOCOS_MODEL)?;
    let (samples, sample_rate) = zipvoice.create(REF_WAV, REF_PHONEMES, TARGET_PHONEMES)?;

    std::fs::create_dir_all("output")?;
    write_wav_mono_16bit(OUTPUT, &samples, sample_rate)?;

    println!("wrote={OUTPUT}");
    println!("duration={:.3}s", samples.len() as f32 / sample_rate as f32);
    Ok(())
}
