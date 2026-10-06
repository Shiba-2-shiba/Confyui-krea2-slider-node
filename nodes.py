import gc
import logging
from pathlib import Path

import torch
import folder_paths
import comfy.model_management as mm
import comfy.utils
from comfy_api.latest import ComfyExtension, io

from .krea2_slider_node.conditioning import encode_prompt_records
from .krea2_slider_node.config import ModelConfig, TrainConfig
from .krea2_slider_node.job import run_training_job
from .krea2_slider_node.lora_io import validate_output_name
from .krea2_slider_node.memory import MemoryBudget
from .krea2_slider_node.native_model import validate_native_model
from .krea2_slider_node.prompt_files import list_prompt_files, load_prompt_file, prompt_file_fingerprint
from .nodes_native_hooks import Krea2NativeLoRAHooksFix
from .nodes_region_masks import Krea2RegionMasks
from .nodes_diagnostics import Krea2ConditioningDebug

LOGGER = logging.getLogger(__name__)
PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _device():
    device = mm.get_torch_device()
    if device.type != "cuda" or not (torch.version.cuda or torch.version.hip):
        raise RuntimeError("Krea2 Slider training requires a CUDA or ROCm PyTorch GPU backend")
    return device


def _release_inference_models():
    mm.unload_all_models()
    gc.collect()
    mm.soft_empty_cache()


def _progress_callback(bar):
    def update(current, total, info):
        mm.throw_exception_if_processing_interrupted()
        bar.update_absolute(current, total)
        if isinstance(info, dict):
            LOGGER.info("Krea2 Slider %s/%s loss=%.6g grad=%.5g reserved=%.2fGiB %.2fs",
                        current, total, info["loss"], info["grad_norm"], info.get("reserved_gib", 0), info["seconds"])
    return update


class Krea2SliderTrainLoRA(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id="Krea2SliderTrainLoRA", display_name="Krea2 Slider Train LoRA",
            category="training/krea2 slider", description="Connect a native Krea2 RAW MODEL and a Krea2 CLIP, select a prompt YAML, and train a Concept Slider LoRA.",
            search_aliases=["train krea2 lora", "concept slider"], is_experimental=True, is_output_node=True, not_idempotent=True,
            inputs=[io.Model.Input("model"), io.Clip.Input("clip"),
                    io.Combo.Input("prompt_yaml", options=list_prompt_files(PROMPTS_DIR)),
                    io.Combo.Input("model_variant", options=["raw"], tooltip="Use an unpatched RAW diffusion model; Turbo is for inference."),
                    io.Combo.Input("quantization", options=["convrot_int8", "bf16_reference"]),
                    io.Int.Input("blocks_to_swap", default=16, min=0, max=28),
                    io.Float.Input("memory_budget_gib", default=14.0, min=1.0, max=14.0, step=0.5),
                    io.Combo.Input("compute_dtype", options=["bf16", "fp16"]),
                    io.Int.Input("steps", default=100, min=1, max=100000),
                    io.Int.Input("rank", default=8, min=1, max=128),
                    io.Float.Input("alpha", default=8.0, min=0.1, max=128.0, step=1.0),
                    io.Combo.Input("target", options=["attention", "all"]),
                    io.Float.Input("learning_rate", default=0.0001, min=0.0000001, max=0.1, step=0.00001),
                    io.Int.Input("width", default=512, min=256, max=1024, step=16),
                    io.Int.Input("height", default=512, min=256, max=1024, step=16),
                    io.Int.Input("trajectory_steps", default=8, min=2, max=64),
                    io.Float.Input("eta", default=1.0, min=0.01, max=10.0, step=0.1),
                    io.Float.Input("teacher_guidance_scale", default=1.0, min=0.01, max=10.0, step=0.1, advanced=True),
                    io.Combo.Input("teacher_norm_reference", options=["none", "positive", "neutral"], advanced=True),
                    io.Int.Input("seed", default=42, min=0, max=2**63 - 1, control_after_generate=True),
                    io.Boolean.Input("vary_seed", default=True),
                    io.Boolean.Input("gradient_checkpointing", default=True),
                    io.String.Input("output_name", default="krea2_slider"),
                    io.Combo.Input("training_direction", options=["single", "bidirectional"], default="single",
                                   optional=True, advanced=True,
                                   tooltip="single learns only the positive YAML direction at LoRA +1. bidirectional also trains an inverse target at -1.")],
            outputs=[io.String.Output("lora_path"), io.String.Output("report_path")])

    @classmethod
    def fingerprint_inputs(cls, prompt_yaml, **kwargs):
        return prompt_file_fingerprint(PROMPTS_DIR, prompt_yaml)

    @classmethod
    def execute(cls, model, clip, prompt_yaml, model_variant, quantization, blocks_to_swap, memory_budget_gib,
                compute_dtype, steps, rank, alpha, target, learning_rate, width, height,
                trajectory_steps, eta, seed, vary_seed, gradient_checkpointing, output_name,
                teacher_guidance_scale=1.0, teacher_norm_reference="none", training_direction="single"):
        validate_native_model(model)
        validate_output_name(output_name)
        specifications = load_prompt_file(PROMPTS_DIR, prompt_yaml)
        prompt_hash = prompt_file_fingerprint(PROMPTS_DIR, prompt_yaml)
        model_config = ModelConfig("", model_variant, quantization, blocks_to_swap, memory_budget_gib, compute_dtype)
        model_config.validate()
        request = TrainConfig(steps=steps, rank=rank, alpha=alpha, target=target, learning_rate=learning_rate,
            width=width, height=height, trajectory_steps=trajectory_steps, eta=eta, seed=seed,
            vary_seed=vary_seed, gradient_checkpointing=gradient_checkpointing,
            teacher_guidance_scale=teacher_guidance_scale, teacher_norm_reference=teacher_norm_reference,
            training_direction=training_direction)
        request.validate()
        device = _device()
        _release_inference_models()
        try:
            with torch.inference_mode(False), MemoryBudget(device, memory_budget_gib) as encoder_budget:
                records = encode_prompt_records(clip, specifications, cancel=mm.throw_exception_if_processing_interrupted,
                    progress=_progress_callback(comfy.utils.ProgressBar(len(specifications))))
                encoder_budget.sample("encoded_prompts")
        finally:
            _release_inference_models()
        # Register a separate output subfolder, so the saved adapter is selectable
        # in the standard LoRA Loader after its filename list is refreshed.
        directory = Path(folder_paths.get_output_directory()) / "krea2_slider_loras"
        folder_paths.add_model_folder_path("loras", str(directory))
        result = run_training_job(model_config, records, request, directory, output_name, device=device,
            native_model=model, progress=_progress_callback(comfy.utils.ProgressBar(steps)),
            cancel=mm.throw_exception_if_processing_interrupted,
            report_context={"prompt_yaml": prompt_yaml, "prompt_sha256": prompt_hash,
                            "encoder_memory": encoder_budget.samples, "source": "native_model_and_clip_inputs"})
        return io.NodeOutput(result[0], result[1], ui={"text": [f"LoRA: {result[0]}", f"Report: {result[1]}"]})


class Krea2SliderExtension(ComfyExtension):
    async def get_node_list(self):
        folder_paths.add_model_folder_path("loras", str(Path(folder_paths.get_output_directory()) / "krea2_slider_loras"))
        return [Krea2SliderTrainLoRA, Krea2NativeLoRAHooksFix, Krea2RegionMasks,
                Krea2ConditioningDebug]
