# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "httpx>=0.28.0",
# ]
# ///
"""
Build platform wheels that bundle zipvoice-capi release libraries.

Usage:
    uv run scripts/build_wheels.py
    uv run scripts/build_wheels.py --c-api-tag c-api-v0.1.1 --out dist
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from dataclasses import dataclass
from pathlib import Path

import httpx


OWNER = "thewh1teagle"
REPO = "zipvoice-rs"
ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "zipvoice_rs"
NATIVE = PACKAGE / "native"


@dataclass(frozen=True)
class Target:
    name: str
    asset: str
    lib_name: str
    wheel_tag: str


TARGETS = [
    Target(
        name="macos-aarch64",
        asset="zipvoice-capi-macos-aarch64.tar.gz",
        lib_name="libzipvoice_capi.dylib",
        wheel_tag="py3-none-macosx_11_0_arm64",
    ),
    Target(
        name="linux-x86_64",
        asset="zipvoice-capi-linux-x86_64.tar.gz",
        lib_name="libzipvoice_capi.so",
        wheel_tag="py3-none-manylinux_2_28_x86_64",
    ),
    Target(
        name="linux-aarch64",
        asset="zipvoice-capi-linux-aarch64.tar.gz",
        lib_name="libzipvoice_capi.so",
        wheel_tag="py3-none-manylinux_2_28_aarch64",
    ),
    Target(
        name="windows-x86_64",
        asset="zipvoice-capi-windows-x86_64.zip",
        lib_name="zipvoice_capi.dll",
        wheel_tag="py3-none-win_amd64",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=f"{OWNER}/{REPO}")
    parser.add_argument("--c-api-tag", default="c-api-v0.1.1")
    parser.add_argument("--out", type=Path, default=ROOT / "dist")
    parser.add_argument("--keep-downloads", action="store_true")
    args = parser.parse_args()

    version = package_version()
    dist_name = "zipvoice_rs"
    out_dir = args.out.resolve()
    downloads = ROOT / ".wheel-downloads"
    out_dir.mkdir(parents=True, exist_ok=True)
    downloads.mkdir(parents=True, exist_ok=True)

    release_assets = get_release_assets(args.repo, args.c_api_tag)
    built: list[Path] = []

    try:
        for target in TARGETS:
            asset_path = downloads / target.asset
            download_asset(release_assets, target.asset, asset_path)
            with tempfile.TemporaryDirectory(prefix=f"zipvoice-{target.name}-") as td:
                native_lib = extract_native_lib(asset_path, target.lib_name, Path(td))
                prepare_native_dir()
                shutil.copy2(native_lib, NATIVE / target.lib_name)
                pure_wheel = build_pure_wheel(Path(td) / "wheel")
                platform_wheel = out_dir / f"{dist_name}-{version}-{target.wheel_tag}.whl"
                rewrite_wheel_tag(pure_wheel, platform_wheel, target.wheel_tag)
                assert_wheel_contains(platform_wheel, target.lib_name, target.wheel_tag)
                built.append(platform_wheel)
                print(f"built {platform_wheel}")
    finally:
        cleanup_native_dir()
        if not args.keep_downloads:
            shutil.rmtree(downloads, ignore_errors=True)

    print("\nBuilt wheels:")
    for wheel in built:
        print(f"  {wheel}")


def package_version() -> str:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    return pyproject["project"]["version"]


def get_release_assets(repo: str, tag: str) -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json"}
    if token := os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
    response = httpx.get(url, headers=headers, follow_redirects=True, timeout=30)
    response.raise_for_status()
    release = response.json()
    return {asset["name"]: asset["browser_download_url"] for asset in release["assets"]}


def download_asset(assets: dict[str, str], name: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    if name not in assets:
        available = "\n".join(sorted(assets))
        raise RuntimeError(f"asset {name!r} not found. Available assets:\n{available}")
    print(f"downloading {name}")
    with httpx.stream("GET", assets[name], follow_redirects=True, timeout=120) as response:
        response.raise_for_status()
        with path.open("wb") as file:
            for chunk in response.iter_bytes():
                file.write(chunk)


def extract_native_lib(asset: Path, lib_name: str, out_dir: Path) -> Path:
    unpack_dir = out_dir / "unpacked"
    unpack_dir.mkdir()
    shutil.unpack_archive(str(asset), str(unpack_dir))
    matches = list(unpack_dir.rglob(lib_name))
    if not matches:
        raise RuntimeError(f"{lib_name} not found in {asset}")
    return matches[0]


def prepare_native_dir() -> None:
    shutil.rmtree(NATIVE, ignore_errors=True)
    NATIVE.mkdir(parents=True, exist_ok=True)


def cleanup_native_dir() -> None:
    shutil.rmtree(NATIVE, ignore_errors=True)


def build_pure_wheel(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(out_dir)],
        cwd=ROOT,
        check=True,
    )
    wheels = sorted(out_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected one wheel in {out_dir}, found {wheels}")
    return wheels[0]


def rewrite_wheel_tag(source: Path, dest: Path, tag: str) -> None:
    with tempfile.TemporaryDirectory(prefix="zipvoice-wheel-") as td:
        wheel_dir = Path(td) / "wheel"
        with zipfile.ZipFile(source) as zf:
            zf.extractall(wheel_dir)

        dist_info = next(wheel_dir.glob("*.dist-info"))
        wheel_file = dist_info / "WHEEL"
        lines = wheel_file.read_text().splitlines()
        lines = [
            "Root-Is-Purelib: false" if line.startswith("Root-Is-Purelib: ") else line
            for line in lines
            if not line.startswith("Tag: ")
        ]
        lines.append(f"Tag: {tag}")
        wheel_file.write_text("\n".join(lines) + "\n")

        record_file = dist_info / "RECORD"
        rows = []
        for path in sorted(p for p in wheel_dir.rglob("*") if p.is_file()):
            rel = path.relative_to(wheel_dir).as_posix()
            if path == record_file:
                rows.append([rel, "", ""])
                continue
            data = path.read_bytes()
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
            rows.append([rel, f"sha256={digest.decode()}", str(len(data))])

        with record_file.open("w", newline="") as file:
            csv.writer(file).writerows(rows)

        dest.unlink(missing_ok=True)
        with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(p for p in wheel_dir.rglob("*") if p.is_file()):
                zf.write(path, path.relative_to(wheel_dir).as_posix())


def assert_wheel_contains(wheel: Path, lib_name: str, tag: str) -> None:
    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()
        wheel_metadata = zf.read(next(n for n in names if n.endswith(".dist-info/WHEEL"))).decode()
    native_path = f"zipvoice_rs/native/{lib_name}"
    if native_path not in names:
        raise RuntimeError(f"{wheel} does not contain {native_path}")
    if f"Tag: {tag}" not in wheel_metadata:
        raise RuntimeError(f"{wheel} does not contain wheel tag {tag}")
    if "Root-Is-Purelib: false" not in wheel_metadata:
        raise RuntimeError(f"{wheel} is not marked as a platform wheel")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
