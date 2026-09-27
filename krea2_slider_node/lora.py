from contextlib import contextmanager
import math

import torch
from torch import nn
from torch.nn import functional as F

from .quantization import FrozenLinear


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha, device):
        super().__init__()
        self.base = base
        self.rank, self.alpha, self.multiplier = rank, float(alpha), 1.0
        self.lora_down = nn.Linear(base.in_features, rank, bias=False, device=device, dtype=torch.float32)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False, device=device, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x):
        result = self.base(x)
        if self.multiplier == 0:
            return result
        # Disable autocast only for the small adapter branch. Parameters and
        # optimizer moments stay FP32 on CUDA and ROCm alike.
        with torch.autocast(device_type=x.device.type, enabled=False):
            delta = F.linear(F.linear(x.float(), self.lora_down.weight), self.lora_up.weight)
        return result + delta.to(result.dtype) * (self.multiplier * self.alpha / self.rank)


def inject_lora(model, rank=8, alpha=8, target="attention", device=None):
    if rank < 1 or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("LoRA rank and alpha must be positive")
    if target not in ("attention", "all"):
        raise ValueError("LoRA target must be attention or all")
    if any(isinstance(module, LoRALinear) for module in model.modules()):
        raise ValueError("LoRA has already been injected")
    model.requires_grad_(False)
    injected = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, (FrozenLinear, nn.Linear)):
            continue
        if target == "attention" and not (name.startswith("blocks.") and ".attn." in name):
            continue
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        layer = LoRALinear(module, rank, alpha, device or module.weight.device)
        setattr(parent, child, layer)
        injected.append(name)
    if not injected:
        raise ValueError("No LoRA target layers matched")
    return injected


def lora_parameters(model):
    return [p for m in model.modules() if isinstance(m, LoRALinear)
            for p in (m.lora_down.weight, m.lora_up.weight)]


@contextmanager
def lora_multiplier(model, multiplier):
    modules = [(m, m.multiplier) for m in model.modules() if isinstance(m, LoRALinear)]
    try:
        for module, _ in modules:
            module.multiplier = float(multiplier)
        yield
    finally:
        for module, value in modules:
            module.multiplier = value


def _prefix(name):
    return "lora_unet_" + name.replace(".", "_")


def lora_state_dict(model):
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            prefix = _prefix(name)
            state[prefix + ".lora_down.weight"] = module.lora_down.weight.detach().cpu().contiguous().clone()
            state[prefix + ".lora_up.weight"] = module.lora_up.weight.detach().cpu().contiguous().clone()
            state[prefix + ".alpha"] = torch.tensor(module.alpha, dtype=torch.float32)
    if not state:
        raise ValueError("No LoRA tensors to save")
    return state


@torch.no_grad()
def load_lora_state_dict(model, state):
    expected = set(lora_state_dict(model))
    if set(state) != expected:
        raise ValueError("LoRA keys do not match the injected network")
    for name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        prefix = _prefix(name)
        for suffix, parameter in (("lora_down.weight", module.lora_down.weight), ("lora_up.weight", module.lora_up.weight)):
            tensor = state[prefix + "." + suffix]
            if tensor.shape != parameter.shape or not torch.isfinite(tensor).all():
                raise ValueError(f"Invalid LoRA tensor: {prefix}.{suffix}")
            parameter.copy_(tensor)
        alpha = float(state[prefix + ".alpha"])
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("Invalid LoRA alpha")
        module.alpha = alpha
