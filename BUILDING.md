# Building

Download the PyTorch/ONNX assets and convert ZipVoice and Vocos to GGUF.

## Requirements

- Rust toolchain
- `wget`
- `uv`

The conversion commands use transient Python environments through `uvx`, so the
project does not need a checked-in Python virtualenv.

## Download Assets

```bash
mkdir -p assets models/renikud models/zipvoice-heb models/vocos

wget https://github.com/thewh1teagle/phonikud-chatterbox/releases/download/asset-files-v1/female1.wav \
  -O assets/female1.wav
wget "https://huggingface.co/thewh1teagle/zipvoice-heb/resolve/main/checkpoint-36600.pt?download=true" \
  -O models/zipvoice-heb/checkpoint-36600.pt
wget "https://huggingface.co/k2-fsa/ZipVoice/resolve/main/zipvoice/tokens.txt?download=true" \
  -O models/zipvoice-heb/tokens.txt
wget "https://huggingface.co/k2-fsa/ZipVoice/resolve/main/zipvoice/model.json?download=true" \
  -O models/zipvoice-heb/model.json
wget https://huggingface.co/thewh1teagle/renikud/resolve/main/model.onnx \
  -O models/renikud/model.onnx

uv run hf download charactr/vocos-mel-24khz config.yaml pytorch_model.bin \
  --local-dir models/vocos
```

## Clone Reference Repos

The ZipVoice converter imports the original Python model definition.

```bash
mkdir -p plans
git clone https://github.com/k2-fsa/ZipVoice plans/ZipVoice
git clone https://github.com/gemelo-ai/vocos plans/vocos
git clone https://github.com/thewh1teagle/renikud plans/renikud
```

If a repo already exists, update it instead:

```bash
git -C plans/ZipVoice pull --ff-only
git -C plans/vocos pull --ff-only
git -C plans/renikud pull --ff-only
```

## Convert Vocos

Full precision:

```bash
uvx --from torch --with gguf --with numpy python tools/convert_vocos.py
```

Q8:

```bash
uvx --from torch --with gguf --with numpy python tools/convert_vocos.py \
  --out-type q8_0 \
  --output models/vocos/vocos-mel-24khz-q8_0.gguf
```

The Vocos q8 file is mostly quantized. A few convolution/filter tensors are kept
as f16 because their shapes are not valid for Q8_0 blocks.

## Convert ZipVoice

Full precision:

```bash
uvx --from torch --with gguf --with numpy python tools/convert_zipvoice.py
```

F16:

```bash
uvx --from torch --with gguf --with numpy python tools/convert_zipvoice.py \
  --out-type f16 \
  --output models/zipvoice-heb/zipvoice-heb-f16.gguf
```

Q8:

```bash
uvx --from torch --with gguf --with numpy python tools/convert_zipvoice.py \
  --out-type q8_0 \
  --output models/zipvoice-heb/zipvoice-heb-q8_0.gguf
```

## Run Examples

```bash
cargo run --release --example basic
cargo run --release --example basic_espeak --features phonemize-espeak
cargo run --release --example basic_hebrew_text --features phonemize-hebrew
cargo run --release --example reconstruct_vocos
```

Generated WAV files are written to `output/`.

## Expected Model Files

```text
models/renikud/model.onnx
models/vocos/vocos-mel-24khz.gguf
models/vocos/vocos-mel-24khz-q8_0.gguf
models/zipvoice-heb/zipvoice-heb-f32.gguf
models/zipvoice-heb/zipvoice-heb-f16.gguf
models/zipvoice-heb/zipvoice-heb-q8_0.gguf
```
