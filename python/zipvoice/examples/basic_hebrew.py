from pathlib import Path

from zipvoice import ZipVoice
from zipvoice.models import asset_path, ensure_asset, ensure_model


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "output/python-basic-hebrew.wav"
PHONEMES = "halˈaχti lamakˈolet liknˈot lˈeχem veχalˈav, ubadˈeʁeχ paɡˈaʃti χavˈeʁ jaʃˈan ʃelˈo ʁaʔˈiti haʁbˈe zmˈan."


def main() -> None:
    zipvoice, vocos = ensure_model("hebrew", ROOT)
    ensure_asset("female1", ROOT)
    with ZipVoice(zipvoice, vocos) as model:
        output = model.generate_wav(
            asset_path("female1", ROOT),
            PHONEMES,
            PHONEMES,
            OUTPUT,
            speed=1.25,
        )
    print(output)


if __name__ == "__main__":
    main()
