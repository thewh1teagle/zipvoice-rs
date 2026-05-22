# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "renikud-onnx @ git+https://github.com/thewh1teagle/renikud.git#subdirectory=renikud-onnx",
#   "numpy>=1.26.0",
#   "phonemizer-fork>=3.3.2",
#   "espeakng-loader>=0.1.9",
# ]
# ///
"""
Prepare assets and models:
    mkdir -p assets models/renikud models/zipvoice-heb models/vocos
    wget https://github.com/thewh1teagle/phonikud-chatterbox/releases/download/asset-files-v1/female1.wav -O assets/female1.wav
    wget https://huggingface.co/thewh1teagle/renikud/resolve/main/model.onnx -O models/renikud/model.onnx
    wget https://github.com/thewh1teagle/zipvoice-rs/releases/download/models-v0.1.0/zipvoice-heb-q8_0.gguf -O models/zipvoice-heb/zipvoice-heb-q8_0.gguf
    wget https://github.com/thewh1teagle/zipvoice-rs/releases/download/models-v0.1.0/vocos-mel-24khz-q8_0.gguf -O models/vocos/vocos-mel-24khz-q8_0.gguf

Run from python/zipvoice:
    uv run --with-editable . examples/basic_hebrew_phonemize.py
"""

from __future__ import annotations

import re
from pathlib import Path

import espeakng_loader
from phonemizer import phonemize as phonemize_en
from phonemizer.backend.espeak.wrapper import EspeakWrapper

from renikud_onnx import G2P
from zipvoice import ZipVoice


ROOT = Path(__file__).resolve().parents[3]
ZIPVOICE_MODEL = ROOT / "models/zipvoice-heb/zipvoice-heb-q8_0.gguf"
VOCOS_MODEL = ROOT / "models/vocos/vocos-mel-24khz-q8_0.gguf"
RENIKUD_MODEL = ROOT / "models/renikud/model.onnx"
REF_WAV = ROOT / "assets/female1.wav"
OUTPUT = ROOT / "output/python-basic-hebrew-phonemize.wav"

REF_PHONEMES = "halˈaχti lamakˈolet liknˈot lˈeχem veχalˈav, ubadˈeʁeχ paɡˈaʃti χavˈeʁ jaʃˈan ʃelˈo ʁaʔˈiti haʁbˈe zmˈan."
TARGET_TEXT = "בבוקר הכנתי קפה ופתחתי את GitHub כדי לבדוק את הפרויקט החדש."
LATIN_WORD_RE = re.compile(r"[A-Za-z]+")


def to_phonemes(text: str, g2p: G2P) -> str:
    def replace_latin(match: re.Match[str]) -> str:
        return phonemize_en(
            match.group(0),
            backend="espeak",
            language="en-us",
            strip=True,
            with_stress=True,
        ).strip()

    return g2p.phonemize(LATIN_WORD_RE.sub(replace_latin, text))


def main() -> None:
    EspeakWrapper.set_library(espeakng_loader.get_library_path())
    EspeakWrapper.set_data_path(espeakng_loader.get_data_path())

    g2p = G2P(RENIKUD_MODEL)
    target_phonemes = to_phonemes(TARGET_TEXT, g2p)

    with ZipVoice(ZIPVOICE_MODEL, VOCOS_MODEL) as model:
        output = model.generate_wav(
            REF_WAV,
            REF_PHONEMES,
            target_phonemes,
            OUTPUT,
            speed=1.25,
        )

    print(f"target_phonemes={target_phonemes}")
    print(output)


if __name__ == "__main__":
    main()
