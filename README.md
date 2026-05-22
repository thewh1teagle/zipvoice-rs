# zipvoice-rs

Generate Hebrew speech locally in Rust with ZipVoice GGUF models.

Rust GGML/GGUF inference for ZipVoice and Vocos.

## Features

- 🗣️ Hebrew text-to-speech generation with ZipVoice.
- 🧬 Optional Hebrew G2P with Renikud, plus eSpeak NG for embedded English words.
- 🔊 Vocos mel encoder/decoder in GGML for 24 kHz waveform output.
- ⚡ GGML backend inference with quiet logs by default.
- 📦 Q8 GGUF models for a smaller local runtime footprint.
- 🦀 Pure Rust examples that write standard mono WAV files.

Original models:

- [ZipVoice](https://github.com/k2-fsa/ZipVoice)
- [zipvoice-heb](https://huggingface.co/thewh1teagle/zipvoice-heb)
- [Vocos mel 24 kHz](https://huggingface.co/charactr/vocos-mel-24khz)
- [Renikud](https://huggingface.co/thewh1teagle/renikud)

See [BUILDING.md](BUILDING.md) for model downloads and GGUF conversion.

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

The q8 Hebrew example uses about 160 MB of model files:

- `zipvoice-heb-q8_0.gguf`: 126 MB
- `vocos-mel-24khz-q8_0.gguf`: 14 MB
- `renikud/model.onnx`: 20 MB
