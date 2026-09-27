"""Real prompt encoder + real model Slider smoke; results do not prove visual quality."""
import argparse
import gc
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comfy_root")
    parser.add_argument("model")
    parser.add_argument("clip")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--trajectory-steps", type=int, default=2)
    parser.add_argument("--blocks-to-swap", type=int, default=16)
    parser.add_argument("--budget-gib", type=float, default=14)
    parser.add_argument("--output", default="test-results/slider-acceptance")
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0, str(Path(args.comfy_root).resolve()))
    sys.argv = [sys.argv[0]]
    import torch
    import comfy.sd
    import comfy.model_management as mm
    from krea2_slider_node.config import ModelConfig, TrainConfig
    from krea2_slider_node.conditioning import encode_prompt_records
    from krea2_slider_node.job import run_training_job
    from krea2_slider_node.memory import MemoryBudget
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    def progress(i, n, info):
        if isinstance(info, dict) or i % 24 == 0:
            print(json.dumps({"current": i, "total": n, "info": info}), flush=True)
    with torch.inference_mode(False), MemoryBudget("cuda", args.budget_gib) as encoder_budget:
        clip = comfy.sd.load_clip([args.clip], clip_type=comfy.sd.CLIPType.KREA2)
        records = encode_prompt_records(clip, [{"target": "a portrait of a person",
            "positive": "a portrait of a smiling person", "negative": "a portrait of a person with a neutral expression"}])
        encoder_budget.sample("encoded")
        (out / "encoder-memory.json").write_text(json.dumps(encoder_budget.samples, indent=2), encoding="utf-8")
        print("Text encoded: " + str({k: tuple(v.features.shape) for k, v in records[0].conditions.items()}), flush=True)
        mm.unload_all_models()
        del clip
    gc.collect(); torch.cuda.empty_cache()
    result = run_training_job(ModelConfig(args.model, blocks_to_swap=args.blocks_to_swap, memory_budget_gib=args.budget_gib), records,
        TrainConfig(steps=args.steps, width=args.resolution, height=args.resolution, trajectory_steps=args.trajectory_steps),
        out, "smile", device="cuda", progress=progress)
    (out / "result.json").write_text(json.dumps({"lora": result[0], "report": result[1]}, indent=2), encoding="utf-8")
    print("Saved " + result[0], flush=True)


if __name__ == "__main__":
    main()
