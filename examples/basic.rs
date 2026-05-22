/*
Prepare assets and models:
    mkdir -p assets models/zipvoice-heb models/vocos
    wget https://github.com/thewh1teagle/phonikud-chatterbox/releases/download/asset-files-v1/female1.wav -O assets/female1.wav
    wget https://huggingface.co/thewh1teagle/zipvoice-heb/resolve/main/checkpoint-36600.pt?download=true -O models/zipvoice-heb/checkpoint-36600.pt
    wget https://huggingface.co/k2-fsa/ZipVoice/resolve/main/zipvoice/tokens.txt?download=true -O models/zipvoice-heb/tokens.txt
    wget https://huggingface.co/k2-fsa/ZipVoice/resolve/main/zipvoice/model.json?download=true -O models/zipvoice-heb/model.json
    uv run hf download charactr/vocos-mel-24khz config.yaml pytorch_model.bin --local-dir models/vocos
    uv run python tools/convert_vocos.py
    uv run python tools/convert_zipvoice.py

Run:
    cargo run --release --example basic
*/

use zipvoice_rs::{ZipVoice, write_wav_mono_16bit};

const ZIPVOICE_MODEL: &str = "models/zipvoice-heb/zipvoice-heb-f32.gguf";
const VOCOS_MODEL: &str = "models/vocos/vocos-mel-24khz.gguf";
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
