#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import gguf
import numpy as np
import torch


def tensor_to_np(tensor: torch.Tensor, out_type: str) -> tuple[np.ndarray, gguf.GGMLQuantizationType]:
    tensor = tensor.detach().cpu().contiguous().float()
    if out_type == "f32" or tensor.ndim <= 1:
        return tensor.numpy().astype(np.float32), gguf.GGMLQuantizationType.F32
    if out_type == "f16":
        return tensor.numpy().astype(np.float16), gguf.GGMLQuantizationType.F16
    quant_type = {"q8_0": gguf.GGMLQuantizationType.Q8_0}[out_type]
    try:
        return gguf.quants.quantize(tensor.numpy().astype(np.float32), quant_type), quant_type
    except Exception as exc:
        print(f"warning: quantization failed, keeping tensor as f16: {exc}")
        return tensor.numpy().astype(np.float16), gguf.GGMLQuantizationType.F16


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert charactr/vocos-mel-24khz to GGUF")
    parser.add_argument("--checkpoint", default="models/vocos/pytorch_model.bin")
    parser.add_argument("--config", default="models/vocos/config.yaml")
    parser.add_argument("--output", default="models/vocos/vocos-mel-24khz.gguf")
    parser.add_argument("--out-type", choices=["f32", "f16", "q8_0"], default="f32")
    args = parser.parse_args()

    state = torch.load(args.checkpoint, map_location="cpu")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    writer = gguf.GGUFWriter(path=None, arch="vocos")
    writer.add_name("Vocos mel 24kHz")
    writer.add_type(gguf.GGUFType.MODEL)
    writer.add_file_type(
        {
            "f32": gguf.LlamaFileType.ALL_F32,
            "f16": gguf.LlamaFileType.MOSTLY_F16,
            "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
        }[args.out_type]
    )
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    writer.add_string("vocos.arch", "vocos")
    writer.add_string("vocos.model_id", "charactr/vocos-mel-24khz")
    writer.add_uint32("vocos.sample_rate", 24000)
    writer.add_uint32("vocos.n_fft", 1024)
    writer.add_uint32("vocos.hop_length", 256)
    writer.add_uint32("vocos.num_mels", 100)
    writer.add_uint32("vocos.backbone_dim", 512)
    writer.add_uint32("vocos.hidden_dim", 1536)
    writer.add_uint32("vocos.layers", 8)
    config = Path(args.config)
    if config.is_file():
        writer.add_string("vocos.config_yaml", config.read_text())

    for name, tensor in state.items():
        data, dtype = tensor_to_np(tensor, args.out_type)
        writer.add_tensor(name, data, raw_dtype=dtype)

    writer.write_header_to_file(path=out)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    print(f"wrote {out} ({out.stat().st_size / 1024 / 1024:.1f} MiB)")


if __name__ == "__main__":
    main()
