from pathlib import Path

from zipvoice_rs import ZipVoice


ROOT = Path(__file__).resolve().parents[3]
ZIPVOICE_MODEL = ROOT / "models/zipvoice-en/zipvoice-en-q8_0.gguf"
VOCOS_MODEL = ROOT / "models/vocos/vocos-mel-24khz-q8_0.gguf"
REF_WAV = ROOT / "assets/whisper.wav"
OUTPUT = ROOT / "output/python-basic-english.wav"
REF_PHONEMES = "ɹˈiəl tʃˈeɪndʒ bɪɡˈɪnz wˈɛn jɔːɹ hˈoʊp bɪkˈʌmz stɹˈɔŋɡɚ ðæn jɔːɹ ɛkskjˈuːsᵻz."
TARGET_PHONEMES = "ðə mˈɔːɹnɪŋ tɹˈeɪn ɚˈaɪvd bɪsˈaɪd ði ˈoʊld stˈoʊn bɹˈɪdʒ."


def main() -> None:
    with ZipVoice(ZIPVOICE_MODEL, VOCOS_MODEL) as model:
        output = model.generate_wav(
            REF_WAV,
            REF_PHONEMES,
            TARGET_PHONEMES,
            OUTPUT,
        )
    print(output)


if __name__ == "__main__":
    main()
