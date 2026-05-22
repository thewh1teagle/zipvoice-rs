# zipvoice

Python `ctypes` bindings for the local `zipvoice-capi` shared library.

Build the C API first:

```bash
cargo build --release -p zipvoice-capi
```

No library path export is needed when running from this repository. The package
searches parent directories for `target/debug` and `target/release` builds.

If the shared library is somewhere else, set:

```bash
export ZIPVOICE_CAPI_LIB=/path/to/libzipvoice_capi.dylib
```

Run an example from this directory:

```bash
uv run python examples/basic_english.py
uv run python examples/basic_hebrew.py
```
