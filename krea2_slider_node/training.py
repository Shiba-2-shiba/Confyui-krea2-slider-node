"""Single-direction Slider training, with optional bidirectional comparison."""
from contextlib import nullcontext
from dataclasses import asdict
import math
import time

import torch
from torch.nn import functional as F

from .lora import inject_lora, lora_multiplier, lora_parameters, lora_state_dict
from .memory import memory_snapshot
from .model import predict_velocity
from .slider_loss import slider_teachers


def raw_schedule(width, height, steps):
    tokens = (width // 16) * (height // 16)
    mu = 0.5 + (tokens - 256) * (1.15 - 0.5) / (6400 - 256)
    grid = torch.linspace(1, 0, steps + 1)
    shift = math.exp(mu)
    return shift * grid / (1 + (shift - 1) * grid)


def train_steps(model, records, request, *, device, compute_dtype, progress=None, cancel=None, budget=None):
    request.validate()
    if not records:
        raise ValueError("At least one encoded prompt record is required")
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    amp = lambda: torch.autocast(device.type, dtype=compute_dtype) if compute_dtype != torch.float32 else nullcontext()
    # Keep ComfyUI's global RNG unchanged. Noise and timestep selection below use
    # local generators; only LoRA initialization needs the global generator.
    with torch.random.fork_rng(devices=[device.index] if device.type == "cuda" else []):
        torch.manual_seed(request.seed)
        targets = inject_lora(model, request.rank, request.alpha, request.target, device)
    parameters = lora_parameters(model)
    optimizer = torch.optim.AdamW(parameters, lr=request.learning_rate, foreach=False)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and compute_dtype == torch.float16)
    model.gradient_checkpointing = request.gradient_checkpointing
    model.train()
    previous_check = model.interrupt_check
    model.interrupt_check = cancel
    schedule = raw_schedule(request.width, request.height, request.trajectory_steps)
    directions = (1,) if request.training_direction == "single" else (1, -1)
    report = {"settings": asdict(request), "targets": targets, "trainable_parameters": sum(p.numel() for p in parameters),
              "prediction_type": "raw_flow_velocity", "teacher": "same_quantized_RAW_lora_disabled",
              "student_directions": list(directions), "loss_reduction": "mean_per_direction",
              "prompts": [record.prompts for record in records], "steps": []}
    started = time.perf_counter()
    try:
        with torch.inference_mode(False):
            for step in range(request.steps):
                tick = time.perf_counter()
                if cancel:
                    cancel()
                record = records[step % len(records)]
                conditions = {role: record.conditions[role] for role in ("target", "positive", "negative")}
                if request.teacher_norm_reference == "neutral":
                    conditions["neutral"] = record.conditions.get("neutral", record.conditions["target"])
                # The prompt encoder shares a TextCondition for identical text.
                # Preserve this sharing during transfer and teacher prediction.
                condition_keys = {role: id(cond.features) for role, cond in conditions.items()}
                feature_cache = {}
                features = {}
                for role, cond in conditions.items():
                    key = condition_keys[role]
                    if key not in feature_cache:
                        feature_cache[key] = cond.features.to(device=device, dtype=compute_dtype)
                    features[role] = feature_cache[key]
                seed = (request.seed + (step if request.vary_seed else 0)) % (2**63)
                generator = torch.Generator(device=device).manual_seed(seed)
                selection = torch.Generator().manual_seed(seed)
                index = int(torch.randint(1, request.trajectory_steps, (1,), generator=selection))
                x = torch.randn(1, model.config.channels, request.height // 8, request.width // 8,
                                generator=generator, device=device, dtype=compute_dtype)
                with torch.no_grad(), lora_multiplier(model, 0), amp():
                    for k in range(index):
                        t = schedule[k].reshape(1).to(device)
                        velocity = predict_velocity(model, x, t, features["target"])
                        x = x + (schedule[k + 1] - schedule[k]).to(x) * velocity
                    x = x.detach()
                    t = schedule[index].reshape(1).to(device)
                    teacher_predictions = {}
                    def teacher_prediction(role):
                        key = condition_keys[role]
                        if key not in teacher_predictions:
                            teacher_predictions[key] = predict_velocity(model, x, t, features[role]).detach()
                        return teacher_predictions[key]
                    base = teacher_prediction("target")
                    positive = teacher_prediction("positive")
                    negative = teacher_prediction("negative")
                    reference = None
                    if request.teacher_norm_reference == "positive":
                        reference = positive
                    elif request.teacher_norm_reference == "neutral":
                        reference = teacher_prediction("neutral")
                    teachers = slider_teachers(base, positive, negative, request.eta * request.teacher_guidance_scale,
                                               reference, directions=directions)
                    teacher_evaluations = len(teacher_predictions)
                    del reference
                del base, positive, negative, velocity, teacher_predictions
                optimizer.zero_grad(set_to_none=True)
                total_loss = 0.0
                for sign, teacher in zip(directions, teachers):
                    if cancel:
                        cancel()
                    # Keep the multiplier unchanged throughout checkpoint backward.
                    with lora_multiplier(model, sign), torch.enable_grad(), amp():
                        prediction = predict_velocity(model, x, t, features["target"])
                        loss = F.mse_loss(prediction.float(), teacher.float()) / len(directions)
                        if not loss.requires_grad or not torch.isfinite(loss):
                            raise RuntimeError("Non-finite loss or missing adapter gradient path")
                        scaler.scale(loss).backward()
                        total_loss += float(loss.detach())
                    del loss, prediction
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, request.max_grad_norm, error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                memory = budget.sample(f"step_{step + 1}") if budget else memory_snapshot(device)
                entry = {"step": step + 1, "loss": total_loss, "grad_norm": float(grad_norm),
                         "teacher_evaluations": teacher_evaluations, "student_backward_passes": len(directions),
                         "seconds": time.perf_counter() - tick, "seed": seed, "timestep_index": index, **memory}
                report["steps"].append(entry)
                if progress:
                    progress(step + 1, request.steps, entry)
                del teachers, features, feature_cache, x
            if not any(entry["grad_norm"] > 0 for entry in report["steps"]):
                raise RuntimeError("No nonzero LoRA gradients; check that the two concepts differ")
            report["total_seconds"] = time.perf_counter() - started
            return lora_state_dict(model), report
    finally:
        optimizer.zero_grad(set_to_none=True)
        model.interrupt_check = previous_check
