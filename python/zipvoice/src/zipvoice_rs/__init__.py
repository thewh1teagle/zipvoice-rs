from __future__ import annotations

import ctypes
import os
import platform
from pathlib import Path


def _library_name() -> str:
    system = platform.system()
    if system == "Darwin":
        return "libzipvoice_capi.dylib"
    if system == "Windows":
        return "zipvoice_capi.dll"
    return "libzipvoice_capi.so"


def _candidate_library_paths() -> list[Path]:
    paths = []
    env_path = os.environ.get("ZIPVOICE_CAPI_LIB")
    if env_path:
        paths.append(Path(env_path))

    here = Path(__file__).resolve()
    for parent in here.parents:
        paths.append(parent / "native" / _library_name())
        paths.append(parent / "target" / "debug" / _library_name())
        paths.append(parent / "target" / "release" / _library_name())
    return paths


def _load_library() -> ctypes.CDLL:
    for path in _candidate_library_paths():
        if path.exists():
            return ctypes.CDLL(str(path))
    searched = "\n".join(str(path) for path in _candidate_library_paths())
    raise RuntimeError(
        "Could not find ZipVoice C API library. "
        "Build it with `cargo build --release -p zipvoice-capi` or set "
        f"ZIPVOICE_CAPI_LIB.\nSearched:\n{searched}"
    )


_lib = _load_library()
_lib.zipvoice_last_error.restype = ctypes.c_char_p
_lib.zipvoice_model_load.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
_lib.zipvoice_model_load.restype = ctypes.c_void_p
_lib.zipvoice_model_free.argtypes = [ctypes.c_void_p]
_lib.zipvoice_model_free.restype = None
_lib.zipvoice_generate_wav.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_float,
    ctypes.c_size_t,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_uint64,
    ctypes.c_bool,
    ctypes.c_char_p,
]
_lib.zipvoice_generate_wav.restype = ctypes.c_int


def _bytes_path(path: str | os.PathLike[str]) -> bytes:
    return os.fsencode(Path(path))


def _check_status(status: int) -> None:
    if status == 0:
        return
    message = _lib.zipvoice_last_error()
    raise RuntimeError(message.decode("utf-8") if message else "ZipVoice C API error")


class ZipVoice:
    def __init__(
        self,
        zipvoice: str | os.PathLike[str],
        vocos: str | os.PathLike[str],
    ) -> None:
        handle = _lib.zipvoice_model_load(_bytes_path(zipvoice), _bytes_path(vocos))
        if not handle:
            _check_status(-1)
        self._handle = ctypes.c_void_p(handle)

    def close(self) -> None:
        if getattr(self, "_handle", None):
            _lib.zipvoice_model_free(self._handle)
            self._handle = None

    def generate_wav(
        self,
        ref_wav: str | os.PathLike[str],
        ref_phonemes: str,
        target_phonemes: str,
        output: str | os.PathLike[str],
        *,
        speed: float = 1.0,
        num_steps: int = 8,
        t_shift: float = 0.5,
        guidance_scale: float = 1.0,
        seed: int = 42,
        verbose: bool = False,
    ) -> Path:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        status = _lib.zipvoice_generate_wav(
            self._handle,
            _bytes_path(ref_wav),
            ref_phonemes.encode("utf-8"),
            target_phonemes.encode("utf-8"),
            speed,
            num_steps,
            t_shift,
            guidance_scale,
            seed,
            verbose,
            _bytes_path(output),
        )
        _check_status(status)
        return output

    def __enter__(self) -> ZipVoice:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()


__all__ = ["ZipVoice"]
