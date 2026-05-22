from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx


RELEASE_BASE = "https://github.com/thewh1teagle/zipvoice-rs/releases/download/models-v0.1.0"
ProgressCallback = Callable[[float | None, str], None]


@dataclass(frozen=True)
class AssetSpec:
    key: str
    label: str
    filename: str
    path: str
    source_url: str | None = None

    @property
    def url(self) -> str:
        if self.source_url is not None:
            return self.source_url
        return f"{RELEASE_BASE}/{self.filename}"


ASSETS = {
    "vocos": AssetSpec(
        key="vocos",
        label="Vocos Q8",
        filename="vocos-mel-24khz-q8_0.gguf",
        path="models/vocos/vocos-mel-24khz-q8_0.gguf",
    ),
    "english": AssetSpec(
        key="english",
        label="ZipVoice English Q8",
        filename="zipvoice-en-q8_0.gguf",
        path="models/zipvoice-en/zipvoice-en-q8_0.gguf",
    ),
    "hebrew": AssetSpec(
        key="hebrew",
        label="ZipVoice Hebrew Q8",
        filename="zipvoice-heb-q8_0.gguf",
        path="models/zipvoice-heb/zipvoice-heb-q8_0.gguf",
    ),
    "whisper": AssetSpec(
        key="whisper",
        label="English prompt",
        filename="whisper.wav",
        path="assets/whisper.wav",
    ),
    "female1": AssetSpec(
        key="female1",
        label="Hebrew prompt",
        filename="female1.wav",
        path="assets/female1.wav",
        source_url="https://github.com/thewh1teagle/phonikud-chatterbox/releases/download/asset-files-v1/female1.wav",
    ),
}


def asset_path(asset: str, root: str | Path = ".") -> Path:
    return Path(root) / ASSETS[asset].path


def model_paths(model: str, root: str | Path = ".") -> tuple[Path, Path]:
    if model not in {"english", "hebrew"}:
        raise KeyError(f"unknown ZipVoice model: {model}")
    return asset_path(model, root), asset_path("vocos", root)


def ensure_asset(
    asset: str,
    root: str | Path = ".",
    progress: ProgressCallback | None = None,
) -> AssetSpec:
    spec = ASSETS[asset]
    path = asset_path(asset, root)
    if not path.exists():
        _download(spec, path, progress)
    elif progress is not None:
        progress(1.0, f"{spec.label} ready")
    return spec


def ensure_model(
    model: str,
    root: str | Path = ".",
    progress: ProgressCallback | None = None,
) -> tuple[Path, Path]:
    ensure_asset(model, root, progress)
    ensure_asset("vocos", root, progress)
    return model_paths(model, root)


def _download(spec: AssetSpec, path: Path, progress: ProgressCallback | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with httpx.stream("GET", spec.url, follow_redirects=True, timeout=None) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", "0") or "0")
        downloaded = 0
        with path.open("wb") as file:
            for chunk in response.iter_bytes():
                if chunk:
                    file.write(chunk)
                    downloaded += len(chunk)
                    if progress is not None:
                        progress(
                            downloaded / total if total else None,
                            _download_status(spec.filename, downloaded, total),
                        )
    if progress is not None:
        progress(1.0, f"{spec.label} ready")


def _download_status(filename: str, downloaded: int, total: int) -> str:
    downloaded_mb = downloaded / 1024 / 1024
    if total:
        total_mb = total / 1024 / 1024
        return f"Downloading {filename}: {downloaded_mb:.1f}/{total_mb:.1f} MB"
    return f"Downloading {filename}: {downloaded_mb:.1f} MB"
