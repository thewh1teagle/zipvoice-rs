/*
Prepare assets and model:
    mkdir -p assets models/vocos
    wget https://github.com/thewh1teagle/phonikud-chatterbox/releases/download/asset-files-v1/female1.wav -O assets/female1.wav
    uv run hf download charactr/vocos-mel-24khz config.yaml pytorch_model.bin --local-dir models/vocos
    uv run python tools/convert_vocos.py

Run:
    cargo run --release --example reconstruct_vocos
*/

use zipvoice_rs::{Vocos, write_wav_24khz};

const MODEL: &str = "models/vocos/vocos-mel-24khz.gguf";
const INPUT: &str = "assets/female1.wav";
const OUTPUT: &str = "output/reconstructed.wav";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    std::fs::create_dir_all("output")?;

    let vocos = Vocos::load(MODEL)?;
    let mel = vocos.encode(INPUT)?;
    let wav = vocos.decode(&mel)?;
    write_wav_24khz(OUTPUT, &wav)?;

    println!("wrote={OUTPUT}");
    Ok(())
}
