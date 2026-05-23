# BUILDING

This package is a thin `ctypes` wrapper around the `zipvoice-capi` dynamic
library from the Rust workspace.

Published wheels are expected to bundle the matching native library, so users on
the supported platforms can install with:

```bash
uv pip install zipvoice-rs
```

No Rust toolchain or `ZIPVOICE_CAPI_LIB` override should be needed for normal
wheel users.

The bundled C API should be built with the optimized native backend for each
platform: Vulkan on Linux and Windows, and Metal plus Accelerate on macOS.

## Supported Wheel Targets

- `py3-none-macosx_11_0_arm64`
- `py3-none-manylinux_2_28_x86_64`
- `py3-none-manylinux_2_28_aarch64`
- `py3-none-win_amd64`

The native artifacts should come from the C API release workflow:

- `zipvoice-capi-macos-aarch64.tar.gz`
- `zipvoice-capi-linux-x86_64.tar.gz`
- `zipvoice-capi-linux-aarch64.tar.gz`
- `zipvoice-capi-windows-x86_64.zip`

Each wheel should place the platform library under:

```text
zipvoice_rs/native/
```

with the expected filename for the platform:

- macOS: `libzipvoice_capi.dylib`
- Linux: `libzipvoice_capi.so`
- Windows: `zipvoice_capi.dll`

## Local Development

Build the C API from the repository root:

```bash
cargo build --release -p zipvoice-capi
```

Then run examples from this directory:

```bash
uv run python examples/basic_english.py
uv run python examples/basic_hebrew.py
uv run --with-editable . examples/basic_hebrew_phonemize.py
```

The wrapper searches parent directories for local `target/debug` and
`target/release` builds. If the shared library is somewhere else, set:

```bash
export ZIPVOICE_CAPI_LIB=/path/to/libzipvoice_capi.dylib
```

On Windows:

```powershell
$env:ZIPVOICE_CAPI_LIB = "C:\path\to\zipvoice_capi.dll"
```

## Build Release Wheels Locally

After the C API release exists, build all platform wheels from this directory:

```bash
uv run scripts/build_wheels.py --c-api-tag c-api-v0.1.1
```

The script downloads the C API release assets, bundles the matching native
library into each wheel, and writes platform-tagged wheels to `dist/`.

Publish manually with:

```bash
uv publish dist/*
```

## Models

The Python package does not bundle ZipVoice, Vocos, prompt WAVs, or phonemizers.
Download model assets separately and pass paths to `ZipVoice`.

The C API has no phonemization or Renikud dependency. Text-to-phoneme conversion
belongs in Python examples or downstream applications.
