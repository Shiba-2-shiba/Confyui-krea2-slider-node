"""Strict, tensor-at-a-time loading; the full BF16 DiT never visits the GPU."""
import hashlib
import json
import struct
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

from .model import SingleStreamDiT, krea2_config
from .quantization import FrozenLinear, quantize_convrot, validate_quant_spec


def read_header(path):
    path = Path(path)
    with path.open("rb") as f:
        size_bytes = f.read(8)
        if len(size_bytes) != 8:
            raise ValueError("Truncated safetensors file")
        size = struct.unpack("<Q", size_bytes)[0]
        if size > 100_000_000 or size + 8 > path.stat().st_size:
            raise ValueError("Invalid safetensors header size")
        raw = f.read(size)
    header = json.loads(raw)
    return header, {"path": str(path.resolve()), "size_bytes": path.stat().st_size,
                    "mtime_ns": path.stat().st_mtime_ns, "header_sha256": hashlib.sha256(raw).hexdigest()}


def _key_map(header):
    keys = {}
    for key in header:
        if key == "__metadata__":
            continue
        canonical = key
        for prefix in ("model.diffusion_model.", "diffusion_model."):
            if canonical.startswith(prefix):
                canonical = canonical[len(prefix):]
                break
        if canonical in keys:
            raise ValueError(f"Duplicate checkpoint key: {canonical}")
        keys[canonical] = key
    return keys


def _replace(model, name, value):
    parent_name, _, child_name = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    setattr(parent, child_name, value)


def load_training_model(path, *, device="cpu", compute_dtype=torch.bfloat16,
                        quantization="convrot_int8", blocks_to_swap=0, config=None, progress=None):
    header, identity = read_header(path)
    with safe_open(str(path), framework="pt", device="cpu") as source:
        return _load_training_source(source, header, identity, device=device, compute_dtype=compute_dtype,
                                     quantization=quantization, blocks_to_swap=blocks_to_swap, config=config, progress=progress)


def _load_training_source(source, header, identity, *, device="cpu", compute_dtype=torch.bfloat16,
                        quantization="convrot_int8", blocks_to_swap=0, config=None, progress=None):
    if quantization not in ("convrot_int8", "bf16_reference"):
        raise ValueError("Supported quantization modes: convrot_int8, bf16_reference")
    config = config or krea2_config()
    if not 0 <= blocks_to_swap <= config.layers:
        raise ValueError("blocks_to_swap must be between zero and the number of blocks")
    device = torch.device(device)
    metadata = header.get("__metadata__", {})
    if any("turbo" in str(metadata.get(key, "")).lower() for key in ("model_variant", "variant", "model_type")):
        raise ValueError("Train on Krea2 RAW, not a Turbo checkpoint")
    keys = _key_map(header)
    with torch.device("meta"):
        model = SingleStreamDiT(config)
    expected = dict(model.named_parameters())
    linears = {name: module for name, module in model.named_modules() if isinstance(module, nn.Linear)}
    permitted = set(expected)
    for name in linears:
        permitted.update((name + ".comfy_quant", name + ".weight_scale"))
    extra = set(keys) - permitted
    missing = set(expected) - set(keys)
    if missing or extra:
        raise ValueError(f"Checkpoint keys mismatch: missing={sorted(missing)[:5]}, unexpected={sorted(extra)[:5]}")
    for name, parameter in expected.items():
        item = header[keys[name]]
        if item["shape"] != list(parameter.shape):
            raise ValueError(f"Checkpoint shape mismatch for {name}: {item['shape']} != {list(parameter.shape)}")
        if item["dtype"] not in ("BF16", "F16", "F32", "I8"):
            raise ValueError(f"Unsupported dtype {item['dtype']} for {name}; use BF16 or ConvRot INT8 RAW")
    offloaded = set(range(config.layers - blocks_to_swap, config.layers))
    used = set()
    quantized = 0
    for index, (name, module) in enumerate(linears.items()):
        if progress:
            progress(index, len(linears), f"Loading {name}")
        cpu_block = name.startswith("blocks.") and int(name.split(".")[1]) in offloaded
        destination = torch.device("cpu") if cpu_block else device
        weight = source.get_tensor(keys[name + ".weight"])
        bias_key = name + ".bias"
        bias = source.get_tensor(keys[bias_key]).to(device=device, dtype=compute_dtype) if module.bias is not None else None
        scale, group_size = None, 0
        spec_key, scale_key = name + ".comfy_quant", name + ".weight_scale"
        if weight.dtype == torch.int8:
            if spec_key not in keys or scale_key not in keys:
                raise ValueError(f"Missing ConvRot metadata/scale for {name}")
            if quantization != "convrot_int8":
                raise ValueError("INT8 checkpoint requires convrot_int8 mode")
            spec = json.loads(bytes(source.get_tensor(keys[spec_key]).tolist()))
            group_size = validate_quant_spec(spec, module.in_features)
            scale = source.get_tensor(keys[scale_key])
        else:
            if spec_key in keys or scale_key in keys:
                raise ValueError(f"Quantization metadata attached to non-INT8 weight: {name}")
            if quantization == "convrot_int8" and name.startswith("blocks."):
                group_size = next((size for size in (64, 16, 4) if module.in_features % size == 0), 0)
                if group_size:
                    weight, scale = quantize_convrot(weight, group_size)
        if scale is not None:
            quantized += 1
            weight = weight.to(destination)
            scale = scale.to(destination)
        else:
            weight = weight.to(device=destination, dtype=compute_dtype)
        _replace(model, name, FrozenLinear(weight, bias, scale=scale, group_size=group_size))
        used.add(name + ".weight")
        if module.bias is not None:
            used.add(bias_key)
    for name, parameter in expected.items():
        if name in used:
            continue
        value = source.get_tensor(keys[name])
        if not value.is_floating_point():
            raise ValueError(f"Non-Linear parameter must remain floating point: {name}")
        # Preserve the checkpoint's FP32 norm/modulation precision.
        dtype = torch.float32 if value.dtype == torch.float32 else compute_dtype
        _replace(model, name, nn.Parameter(value.to(device=device, dtype=dtype), requires_grad=False))
    model.requires_grad_(False)
    model.gradient_checkpointing = True
    storage = {"cpu": 0, "device": 0}
    for tensor in list(model.parameters()) + list(model.buffers()):
        storage["cpu" if tensor.device.type == "cpu" else "device"] += tensor.numel() * tensor.element_size()
    return model, {"source": identity, "source_metadata": metadata, "quantization": quantization,
                   "quantized_linears": quantized, "offloaded_blocks": sorted(offloaded),
                   "storage_bytes": storage, "offload_mode": "immutable_cpu_linear_staging",
                   "compute_dtype": str(compute_dtype), "linear_backend": "pytorch_eager"}
