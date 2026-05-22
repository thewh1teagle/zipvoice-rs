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

## Examples

- [basic.rs](examples/basic.rs): hardcoded Hebrew phonemes.
- [basic_espeak.rs](examples/basic_espeak.rs): English text with eSpeak NG phonemization.
- [basic_hebrew_text.rs](examples/basic_hebrew_text.rs): Hebrew text with Renikud and eSpeak NG for embedded English.
- [reconstruct_vocos.rs](examples/reconstruct_vocos.rs): Vocos encode/decode reconstruction.

## Model Sizes

The q8 English example uses about 140 MB of model files:

- `zipvoice-en-q8_0.gguf`: 126 MB
- `vocos-mel-24khz-q8_0.gguf`: 14 MB

The q8 Hebrew text example uses about 160 MB of model files:

- `zipvoice-heb-q8_0.gguf`: 126 MB
- `vocos-mel-24khz-q8_0.gguf`: 14 MB
- `renikud/model.onnx`: 20 MB
