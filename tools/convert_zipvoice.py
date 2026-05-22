#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import gguf
import numpy as np
import torch

GGML_MAX_NAME = 63

NAME_REPLACEMENTS = (
    ("fm_decoder.", "fm."),
    ("text_encoder.", "te."),
    ("encoders.", "enc."),
    ("encoder.", "enc."),
    ("layers.", "ly."),
    ("self_attn_weights.", "saw."),
    ("self_attn1.", "sa1."),
    ("self_attn2.", "sa2."),
    ("feed_forward", "ff"),
    ("nonlin_attention.", "na."),
    ("conv_module", "conv"),
    ("depthwise_conv.", "dw."),
    ("downsample.", "down."),
    ("time_embed.", "time."),
    ("in_proj.", "in."),
    ("out_proj.", "out."),
    ("linear_pos.", "pos."),
    ("bypass_mid.", "bpm."),
    ("bypass.", "bp."),
    ("whiten.", "wht."),
    ("balancer", "bal"),
    ("norm.", "n."),
    ("weight", "w"),
    ("bias", "b"),
)


def short_name(name: str) -> str:
    out = name
    for old, new in NAME_REPLACEMENTS:
        out = out.replace(old, new)
    if len(out) <= GGML_MAX_NAME:
        return out
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:10]
    keep = GGML_MAX_NAME - len(digest) - 1
    return f"{out[:keep]}.{digest}"


def parse_tokens(path: Path) -> tuple[int, int, str]:
    token2id = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        token, raw_id = line.rstrip("\n").split("\t")
        token2id[token] = int(raw_id)
    return len(token2id), token2id["_"], path.read_text(encoding="utf-8")


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
        print(f"warning: quantization failed for {tuple(tensor.shape)}, keeping as f16: {exc}")
        return tensor.numpy().astype(np.float16), gguf.GGMLQuantizationType.F16


def load_zipvoice_model(checkpoint: Path, model_json: Path, tokens: Path, state_key: str):
    sys.path.insert(0, str(Path("plans/ZipVoice").resolve()))
    from zipvoice.models.zipvoice import ZipVoice
    from zipvoice.utils.scaling_converter import convert_scaled_to_non_scaled

    vocab_size, pad_id, _ = parse_tokens(tokens)
    config = json.loads(model_json.read_text(encoding="utf-8"))
    model = ZipVoice(**config["model"], vocab_size=vocab_size, pad_id=pad_id)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt[state_key], strict=True)
    model.eval()
    convert_scaled_to_non_scaled(model, inplace=True, is_onnx=True)
    return model, config, ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ZipVoice checkpoint to GGUF")
    parser.add_argument("--checkpoint", default="models/zipvoice-heb/checkpoint-36600.pt")
    parser.add_argument("--model-json", default="models/zipvoice-heb/model.json")
    parser.add_argument("--tokens", default="models/zipvoice-heb/tokens.txt")
    parser.add_argument("--output", default="models/zipvoice-heb/zipvoice-heb-f32.gguf")
    parser.add_argument("--state-key", choices=["model", "model_avg"], default="model")
    parser.add_argument("--out-type", choices=["f32", "f16", "q8_0"], default="f32")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    model_json = Path(args.model_json)
    tokens = Path(args.tokens)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    model, config, ckpt = load_zipvoice_model(checkpoint, model_json, tokens, args.state_key)
    vocab_size, pad_id, token_text = parse_tokens(tokens)
    model_cfg = config["model"]

    writer = gguf.GGUFWriter(path=None, arch="zipvoice")
    writer.add_name("ZipVoice Hebrew")
    writer.add_type(gguf.GGUFType.MODEL)
    writer.add_file_type(
        {
            "f32": gguf.LlamaFileType.ALL_F32,
            "f16": gguf.LlamaFileType.MOSTLY_F16,
            "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
        }[args.out_type]
    )
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    writer.add_string("zipvoice.arch", "zipvoice")
    writer.add_string("zipvoice.model_id", "thewh1teagle/zipvoice-heb")
    writer.add_string("zipvoice.state_key", args.state_key)
    writer.add_string("zipvoice.tokens_txt", token_text)
    writer.add_string("zipvoice.model_json", json.dumps(config, sort_keys=True))
    writer.add_uint32("zipvoice.sample_rate", int(config["feature"]["sampling_rate"]))
    writer.add_string("zipvoice.feature_type", config["feature"]["type"])
    writer.add_uint32("zipvoice.vocab_size", vocab_size)
    writer.add_uint32("zipvoice.pad_id", pad_id)
    writer.add_uint32("zipvoice.feat_dim", int(model_cfg["feat_dim"]))
    writer.add_uint32("zipvoice.text_embed_dim", int(model_cfg["text_embed_dim"]))
    writer.add_uint32("zipvoice.text_encoder_dim", int(model_cfg["text_encoder_dim"]))
    writer.add_uint32("zipvoice.fm_decoder_dim", int(model_cfg["fm_decoder_dim"]))
    writer.add_uint32("zipvoice.time_embed_dim", int(model_cfg["time_embed_dim"]))
    writer.add_uint32("zipvoice.text_encoder_num_layers", int(model_cfg["text_encoder_num_layers"]))
    writer.add_array("zipvoice.fm_decoder_downsampling_factor", model_cfg["fm_decoder_downsampling_factor"])
    writer.add_array("zipvoice.fm_decoder_num_layers", model_cfg["fm_decoder_num_layers"])
    writer.add_array("zipvoice.fm_decoder_cnn_module_kernel", model_cfg["fm_decoder_cnn_module_kernel"])
    if "batch_idx_train" in ckpt:
        writer.add_uint32("zipvoice.checkpoint_batch_idx", int(ckpt["batch_idx_train"]))

    name_map = {}
    used = set()
    for full_name, tensor in model.state_dict().items():
        if tensor.numel() == 0:
            continue
        name = short_name(full_name)
        if name in used:
            raise RuntimeError(f"short tensor name collision: {name}")
        used.add(name)
        name_map[name] = full_name
        data, dtype = tensor_to_np(tensor, args.out_type)
        writer.add_tensor(name, data, raw_dtype=dtype)

    writer.add_string("zipvoice.tensor_name_map", json.dumps(name_map, sort_keys=True))
    writer.write_header_to_file(path=output)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    print(f"wrote {output} ({output.stat().st_size / 1024 / 1024:.1f} MiB), tensors={len(name_map)}")


if __name__ == "__main__":
    main()
