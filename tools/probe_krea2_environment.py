"""Read-only environment plus small real GEMM/GQA backward capability probe."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from krea2_slider_node.memory import environment_report, memory_snapshot


def main():
    report = environment_report()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report["operators"] = {}
    for dtype in (torch.bfloat16, torch.float16):
        try:
            q = torch.randn(1, 4, 32, 16, device=device, dtype=dtype, requires_grad=True)
            k = torch.randn(1, 2, 32, 16, device=device, dtype=dtype, requires_grad=True)
            v = torch.randn_like(k, requires_grad=True)
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, enable_gqa=True)
            y.float().square().mean().backward()
            a = torch.randn(32, 64, device=device, dtype=dtype, requires_grad=True)
            (a @ torch.randn(64, 16, device=device, dtype=dtype)).float().square().mean().backward()
            report["operators"][str(dtype)] = {"finite": bool(torch.isfinite(q.grad).all() and torch.isfinite(a.grad).all())}
        except (RuntimeError, NotImplementedError) as error:
            report["operators"][str(dtype)] = {"error": str(error)}
    report["memory"] = memory_snapshot(device)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
