"""
Setup:
    uv pip install zipvoice-rs
    mkdir -p assets models/zipvoice-heb models/vocos
    wget https://github.com/thewh1teagle/phonikud-chatterbox/releases/download/asset-files-v1/female1.wav -O assets/female1.wav
    wget https://github.com/thewh1teagle/zipvoice-rs/releases/download/models-v0.1.0/zipvoice-heb-q8_0.gguf -O models/zipvoice-heb/zipvoice-heb-q8_0.gguf
    wget https://github.com/thewh1teagle/zipvoice-rs/releases/download/models-v0.1.0/vocos-mel-24khz-q8_0.gguf -O models/vocos/vocos-mel-24khz-q8_0.gguf

Run from the repository root:
    python python/zipvoice-rs/examples/basic_hebrew.py
"""

from pathlib import Path

from zipvoice_rs import ZipVoice


ROOT = Path(__file__).resolve().parents[3]
ZIPVOICE_MODEL = ROOT / "models/zipvoice-heb/zipvoice-heb-q8_0.gguf"
VOCOS_MODEL = ROOT / "models/vocos/vocos-mel-24khz-q8_0.gguf"
REF_WAV = ROOT / "assets/female1.wav"
OUTPUT = ROOT / "output/python-basic-hebrew.wav"
PHONEMES = "halˈaχti lamakˈolet liknˈot lˈeχem veχalˈav, ubadˈeʁeχ paɡˈaʃti χavˈeʁ jaʃˈan ʃelˈo ʁaʔˈiti haʁbˈe zmˈan."


def main() -> None:
    with ZipVoice(ZIPVOICE_MODEL, VOCOS_MODEL) as model:
        output = model.generate_wav(
            REF_WAV,
            PHONEMES,
            PHONEMES,
            OUTPUT,
            speed=1.25,
        )
    print(output)


if __name__ == "__main__":
    main()
