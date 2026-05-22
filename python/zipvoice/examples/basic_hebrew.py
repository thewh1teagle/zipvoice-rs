from pathlib import Path

from zipvoice import ZipVoice


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
