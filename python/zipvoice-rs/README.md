# zipvoice-rs

Python bindings for ZipVoice GGUF inference.

The published wheels bundle the native `zipvoice-capi` dynamic library for the
supported platforms:

- macOS Apple Silicon
- Linux x86_64
- Linux aarch64
- Windows x86_64

## Install

```bash
uv pip install zipvoice-rs
```

## Use

```python
from zipvoice_rs import ZipVoice

with ZipVoice("zipvoice-en-q8_0.gguf", "vocos-mel-24khz-q8_0.gguf") as model:
    model.generate_wav(
        "prompt.wav",
        "ɹˈiəl tʃˈeɪndʒ bɪɡˈɪnz wˈɛn jɔːɹ hˈoʊp bɪkˈʌmz stɹˈɔŋɡɚ ðæn jɔːɹ ɛkskjˈuːsᵻz.",
        "ðə mˈɔːɹnɪŋ tɹˈeɪn ɚˈaɪvd bɪsˈaɪd ði ˈoʊld stˈoʊn bɹˈɪdʒ.",
        "out.wav",
    )
```

See `examples/` for English, Hebrew, and mixed Hebrew/English phonemization
examples.

For local development and native-library packaging notes, see `BUILDING.md`.
