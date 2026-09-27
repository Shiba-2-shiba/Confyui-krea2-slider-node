"""Real-checkpoint memory/gradient probe using SYNTHETIC text, not quality evidence."""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from krea2_slider_node.lora import inject_lora, lora_parameters, lora_state_dict
from krea2_slider_node.memory import MemoryBudget, environment_report
from krea2_slider_node.model import predict_velocity
from krea2_slider_node.model_io import load_training_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--output", default="test-results/real-step.json")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--blocks-to-swap", type=int, default=16)
    parser.add_argument("--budget-gib", type=float, default=14)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--text-tokens", type=int, default=16)
    args = parser.parse_args()
    report = {"environment": environment_report(), "settings": vars(args), "synthetic_conditioning": True,
              "status": "running", "steps": []}
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.manual_seed(42)
    model = optimizer = None
    started = time.perf_counter()
    try:
        with MemoryBudget(device, args.budget_gib) as budget, torch.inference_mode(False):
            model, report["loading"] = load_training_model(args.model, device=device,
                blocks_to_swap=args.blocks_to_swap, progress=lambda i, n, msg: print(msg, flush=True) if i % 24 == 0 else None)
            report["loading_seconds"] = time.perf_counter() - started
            report["memory"] = budget.samples
            print(budget.sample("loaded"), flush=True)
            report["targets"] = inject_lora(model, rank=args.rank, alpha=args.rank, device=device)
            parameters = lora_parameters(model)
            optimizer = torch.optim.AdamW(parameters, lr=1e-4, foreach=False)
            x = torch.randn(1, 16, args.resolution // 8, args.resolution // 8, device=device, dtype=torch.bfloat16)
            text = torch.randn(1, args.text_tokens, 12, 2560, device=device, dtype=torch.bfloat16)
            model.train()
            for step in range(args.steps):
                tick = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = predict_velocity(model, x, torch.tensor([0.5], device=device), text)
                    loss = out.float().square().mean()
                budget.sample("forward")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
                optimizer.step()
                torch.cuda.synchronize()
                entry = {"step": step + 1, "loss": float(loss.detach()), "grad_norm": float(norm), "seconds": time.perf_counter() - tick,
                         **budget.sample("optimizer")}
                if not torch.isfinite(loss) or norm <= 0:
                    raise RuntimeError("Invalid loss or missing LoRA gradient")
                report["steps"].append(entry)
                print(json.dumps(entry), flush=True)
                output.write_text(json.dumps(report, indent=2), encoding="utf-8")
            state = lora_state_dict(model)
            report["export_tensor_count"] = len(state)
            budget.sample("export")
            report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = repr(error)
        raise
    finally:
        report["total_seconds"] = time.perf_counter() - started
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
