"""Frozen weight storage with explicit input gradients and bounded staging.

ConvRot's regular Hadamard convention follows Comfy Kitchen / Musubi Tuner.
See THIRD_PARTY_NOTICES.md. No inference-only or vendor-specific kernels are used.
"""
from functools import lru_cache

import torch
from torch import nn
from torch.nn import functional as F


def validate_quant_spec(spec: dict, in_features: int) -> int:
    size = spec.get("convrot_groupsize")
    if spec.get("format") != "int8_tensorwise" or spec.get("convrot") is not True:
        raise ValueError("Only ComfyUI ConvRot INT8 checkpoints are supported")
    if type(size) is not int or size < 4 or size & (size - 1) or (size.bit_length() - 1) % 2:
        raise ValueError("ConvRot group size must be a power of four")
    if in_features % size or spec.get("per_row", True) is not True:
        raise ValueError("ConvRot requires per-row scales and a group size dividing in_features")
    return size


@lru_cache(maxsize=32)
def hadamard(size: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    validate_quant_spec({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": size}, size)
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], device=device, dtype=dtype)
    h = h4
    while h.shape[0] < size:
        h = torch.kron(h, h4)
    return h / size**0.5


def rotate(x: torch.Tensor, size: int) -> torch.Tensor:
    h = hadamard(size, str(x.device), x.dtype)
    return (x.reshape(-1, x.shape[-1] // size, size) @ h).reshape(x.shape)


@torch.no_grad()
def quantize_convrot(weight: torch.Tensor, group_size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    validate_quant_spec({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": group_size}, weight.shape[1])
    codes = torch.empty_like(weight, dtype=torch.int8)
    scales = torch.empty(weight.shape[0], 1, dtype=torch.float32, device=weight.device)
    # Avoid holding a full FP32 copy of a large BF16 Linear during conversion.
    for start in range(0, weight.shape[0], 256):
        rows = rotate(weight[start:start + 256].float(), group_size)
        if not torch.isfinite(rows).all():
            raise ValueError("Cannot quantize non-finite weights")
        scale = (rows.abs().amax(dim=1, keepdim=True) / 127).clamp_min(1e-30)
        codes[start:start + 256] = (rows / scale).round_().clamp_(-127, 127).to(torch.int8)
        scales[start:start + 256] = scale
    return codes, scales


def _materialize(weight, scale, device, dtype):
    w = weight.to(device=device, dtype=dtype)
    if scale.numel():
        w = w * scale.to(device=device, dtype=dtype)
    return w


class _FrozenLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, scale, bias, group_size):
        # Save the immutable storage, which may remain on CPU. Never save BF16
        # dequantized weights or staged GPU copies for offloaded layers.
        ctx.save_for_backward(weight, scale)
        ctx.group_size = group_size
        w = _materialize(weight, scale, x.device, x.dtype)
        inputs = rotate(x, group_size) if group_size else x
        return F.linear(inputs, w, bias.to(x) if bias is not None else None)

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.needs_input_grad[0]:
            return None, None, None, None, None
        weight, scale = ctx.saved_tensors
        w = _materialize(weight, scale, grad_output.device, grad_output.dtype)
        grad_x = grad_output @ w
        if ctx.group_size:
            grad_x = rotate(grad_x, ctx.group_size)
        return grad_x, None, None, None, None


class FrozenLinear(nn.Module):
    """Frozen BF16/FP32 or ConvRot INT8 weights, optionally resident on CPU."""

    def __init__(self, weight, bias=None, *, scale=None, group_size=0):
        super().__init__()
        if weight.ndim != 2:
            raise ValueError("Linear weight must be a matrix")
        self.out_features, self.in_features = weight.shape
        if weight.dtype == torch.int8:
            validate_quant_spec({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": group_size}, self.in_features)
            if scale is None or scale.numel() != self.out_features:
                raise ValueError("INT8 weight requires one scale per output row")
            scale = scale.reshape(-1, 1).float()
            if not torch.isfinite(scale).all() or not (scale > 0).all():
                raise ValueError("Quantization scales must be finite and positive")
        elif not weight.is_floating_point() or group_size or scale is not None:
            raise ValueError("Unsupported frozen Linear format")
        self.group_size = group_size
        self.register_buffer("weight", weight.detach())
        self.register_buffer("scale", scale.detach() if scale is not None else torch.empty(0, device=weight.device))
        self.register_buffer("bias", bias.detach() if bias is not None else None)

    def forward(self, x):
        if torch.is_autocast_enabled(x.device.type):
            x = x.to(torch.get_autocast_dtype(x.device.type))
        return _FrozenLinearFn.apply(x, self.weight, self.scale, self.bias, self.group_size)
