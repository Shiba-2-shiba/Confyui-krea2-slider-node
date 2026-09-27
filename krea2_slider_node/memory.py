"""Measurements and a scoped allocator ceiling (not a physical VRAM guarantee)."""
import math
import platform

import torch


def probe_training_operators(device, dtype):
    device = torch.device(device)
    with torch.inference_mode(False), torch.enable_grad():
        rng = torch.Generator(device=device).manual_seed(729)
        q = torch.randn(1, 48, 16, 128, device=device, dtype=dtype, generator=rng, requires_grad=True)
        k = torch.randn(1, 12, 16, 128, device=device, dtype=dtype, generator=rng, requires_grad=True)
        v = torch.randn(1, 12, 16, 128, device=device, dtype=dtype, generator=rng, requires_grad=True)
        output = torch.nn.functional.scaled_dot_product_attention(q, k, v, enable_gqa=True)
        output.float().square().mean().backward()
        x = torch.randn(16, 64, device=device, dtype=dtype, generator=rng, requires_grad=True)
        weight = torch.randn(64, 16, device=device, dtype=dtype, generator=rng)
        (x @ weight).float().square().mean().backward()
        finite = all(bool(torch.isfinite(t.grad).all()) for t in (q, k, v, x))
        if not finite:
            raise RuntimeError(f"{dtype} GEMM/GQA backward is non-finite on {device}")
    return {"dtype": str(dtype), "device": str(device), "finite_gradients": finite}


def environment_report():
    devices = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        devices.append({"name": props.name, "total_gib": props.total_memory / 2**30})
    return {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
            "cuda": torch.version.cuda, "hip": torch.version.hip, "devices": devices,
            "backend": "rocm" if torch.version.hip else "cuda" if torch.version.cuda else "cpu"}


def memory_snapshot(device):
    device = torch.device(device)
    result = {}
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        result.update(allocated_gib=torch.cuda.memory_allocated(device) / 2**30,
                      reserved_gib=torch.cuda.memory_reserved(device) / 2**30,
                      peak_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved(device) / 2**30,
                      device_free_gib=free / 2**30, device_total_gib=total / 2**30)
    try:
        import psutil
        result["process_rss_gib"] = psutil.Process().memory_info().rss / 2**30
    except ImportError:
        pass
    return result


class MemoryBudget:
    def __init__(self, device, gib=14.0):
        if not math.isfinite(gib) or gib <= 0:
            raise ValueError("Memory budget must be finite and positive")
        self.device, self.gib = torch.device(device), float(gib)
        self.old_fraction = None
        self.samples = []

    def __enter__(self):
        if self.device.type == "cuda":
            if self.device.index is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
            total = torch.cuda.get_device_properties(self.device).total_memory
            free, _ = torch.cuda.mem_get_info(self.device)
            allocated = torch.cuda.memory_allocated(self.device)
            # Leave at least 1 GiB for display/driver and allocations outside torch.
            limit = min(self.gib * 2**30, max(0, free + allocated - 2**30))
            if limit < 2**30 or allocated > limit:
                raise RuntimeError("Insufficient free VRAM; unload inference models before training")
            self.old_fraction = torch.cuda.get_per_process_memory_fraction(self.device)
            torch.cuda.set_per_process_memory_fraction(min(self.old_fraction, limit / total), self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        return self

    def sample(self, stage):
        result = {"stage": stage, **memory_snapshot(self.device)}
        self.samples.append(result)
        if result.get("reserved_gib", 0) > self.gib + 0.01:
            raise RuntimeError(f"VRAM budget exceeded during {stage}: {result['reserved_gib']:.2f} GiB")
        return result

    def __exit__(self, *exc):
        if self.old_fraction is not None:
            torch.cuda.set_per_process_memory_fraction(self.old_fraction, self.device)
