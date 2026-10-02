"""One owned model per training job; no shared ComfyUI MODEL is patched."""
from dataclasses import asdict
import gc
import threading

import torch

from .lora_io import save_adapter, validate_output_name
from .memory import MemoryBudget, environment_report, probe_training_operators
from .model_io import load_training_model
from .training import train_steps

_job_lock = threading.Lock()


def run_training_job(model_config, records, request, directory, name, *, device, progress=None, cancel=None,
                     native_model=None, report_context=None):
    model_config.validate()
    request.validate()
    validate_output_name(name)
    if not _job_lock.acquire(blocking=False):
        raise RuntimeError("A Krea2 Slider training job is already running")
    model = None
    result = None
    failure = None
    dtype = torch.bfloat16 if model_config.compute_dtype == "bf16" else torch.float16
    try:
        with torch.inference_mode(False), MemoryBudget(device, model_config.memory_budget_gib) as budget:
            if cancel:
                cancel()
            operator_report = probe_training_operators(device, dtype)
            def loading_progress(i, total, message):
                if cancel:
                    cancel()
                if progress:
                    progress(i, total, message)
            loader = load_training_model
            source = model_config.path
            if native_model is not None:
                from .native_model import load_native_training_model
                loader, source = load_native_training_model, native_model
            model, loading = loader(source, device=device, compute_dtype=dtype,
                quantization=model_config.quantization, blocks_to_swap=model_config.blocks_to_swap, progress=loading_progress)
            budget.sample("loaded")
            state, report = train_steps(model, records, request, device=device, compute_dtype=dtype,
                progress=progress, cancel=cancel, budget=budget)
            report.update(model_settings=asdict(model_config), loading=loading, environment=environment_report(), operators=operator_report,
                          memory_samples=budget.samples, status="training_completed",
                          validation_scope="Training/export only; visual Slider quality requires separate evaluation.")
            if report_context:
                report["input_context"] = report_context
            if cancel:
                cancel()
            budget.sample("export")
            result = (*save_adapter(state, report, directory, name), report)
    except Exception as error:
        # Do not retain a traceback containing a complete GPU model in a ComfyUI
        # error/cache. Re-raise a fresh exception only after leaving this handler.
        failure = (type(error), str(error))
    finally:
        model = None
        try:
            gc.collect()
            if torch.device(device).type == "cuda":
                torch.cuda.empty_cache()
        finally:
            _job_lock.release()
    if failure:
        error_type, message = failure
        if error_type.__name__ == "InterruptProcessingException":
            raise error_type()
        raise RuntimeError(message)
    return result
