# zipvoice-rs

Generate English and Hebrew speech locally in Rust with ZipVoice GGUF models.

Rust GGML/GGUF inference for ZipVoice and Vocos.

## Features

- 🗣️ English and Hebrew text-to-speech generation with ZipVoice.
- 🧬 English phonemization with eSpeak NG.
- 🧬 Optional Hebrew G2P with Renikud, plus eSpeak NG for embedded English words.
- 🔊 Vocos mel encoder/decoder in GGML for 24 kHz waveform output.
- ⚡ GGML backend inference with quiet logs by default.
- 📦 Q8 GGUF models for a smaller local runtime footprint.
- 🦀 Pure Rust examples that write standard mono WAV files.

Original models:

- [ZipVoice](https://github.com/k2-fsa/ZipVoice)
- [ZipVoice English](https://huggingface.co/k2-fsa/ZipVoice)
- [ZipVoice Hebrew](https://huggingface.co/thewh1teagle/zipvoice-heb)
- [Vocos mel 24 kHz](https://huggingface.co/charactr/vocos-mel-24khz)
- [Renikud](https://huggingface.co/thewh1teagle/renikud)

See [BUILDING.md](BUILDING.md) for model downloads and GGUF conversion.

Quick model setup:

```bash
mkdir -p assets models/renikud models/zipvoice-en models/zipvoice-heb models/vocos
wget https://github.com/thewh1teagle/phonikud-chatterbox/releases/download/asset-files-v1/female1.wav -O assets/female1.wav
wget https://github.com/thewh1teagle/zipvoice-rs/releases/download/models-v0.1.0/whisper.wav -O assets/whisper.wav
wget https://github.com/thewh1teagle/zipvoice-rs/releases/download/models-v0.1.0/zipvoice-en-q8_0.gguf -O models/zipvoice-en/zipvoice-en-q8_0.gguf
wget https://github.com/thewh1teagle/zipvoice-rs/releases/download/models-v0.1.0/zipvoice-heb-q8_0.gguf -O models/zipvoice-heb/zipvoice-heb-q8_0.gguf
wget https://github.com/thewh1teagle/zipvoice-rs/releases/download/models-v0.1.0/vocos-mel-24khz-q8_0.gguf -O models/vocos/vocos-mel-24khz-q8_0.gguf
wget https://huggingface.co/thewh1teagle/renikud/resolve/main/model.onnx -O models/renikud/model.onnx
```

## Examples

Hebrew text with Renikud and eSpeak NG:

```bash
cargo run --release --example basic_hebrew_text --features phonemize-hebrew
```

Hardcoded phonemes:

```bash
cargo run --release --example basic
```

English text phonemized with eSpeak NG:

```bash
cargo run --release --example basic_espeak --features phonemize-espeak
```

Vocos encode/decode reconstruction:

```bash
cargo run --release --example reconstruct_vocos
```

Generated WAV files are written to `output/`.

## Model Sizes

The q8 English example uses about 140 MB of model files:

- `zipvoice-en-q8_0.gguf`: 126 MB
- `vocos-mel-24khz-q8_0.gguf`: 14 MB

The q8 Hebrew text example uses about 160 MB of model files:

- `zipvoice-heb-q8_0.gguf`: 126 MB
- `vocos-mel-24khz-q8_0.gguf`: 14 MB
- `renikud/model.onnx`: 20 MB
