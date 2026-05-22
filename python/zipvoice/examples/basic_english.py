from pathlib import Path

from zipvoice import ZipVoice
from zipvoice.models import asset_path, ensure_asset, ensure_model


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "output/python-basic-english.wav"
REF_PHONEMES = "ɹˈiəl tʃˈeɪndʒ bɪɡˈɪnz wˈɛn jɔːɹ hˈoʊp bɪkˈʌmz stɹˈɔŋɡɚ ðæn jɔːɹ ɛkskjˈuːsᵻz."
TARGET_PHONEMES = "ðə mˈɔːɹnɪŋ tɹˈeɪn ɚˈaɪvd bɪsˈaɪd ði ˈoʊld stˈoʊn bɹˈɪdʒ."


def main() -> None:
    zipvoice, vocos = ensure_model("english", ROOT)
    ensure_asset("whisper", ROOT)
    with ZipVoice(zipvoice, vocos) as model:
        output = model.generate_wav(
            asset_path("whisper", ROOT),
            REF_PHONEMES,
            TARGET_PHONEMES,
            OUTPUT,
        )
    print(output)


if __name__ == "__main__":
    main()
