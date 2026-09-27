"""Read the connected native MODEL; never substitute a guessed checkpoint path."""
import torch

from .model_io import _load_training_source


def validate_native_model(patcher):
    config = getattr(getattr(getattr(patcher, "model", None), "model_config", None), "unet_config", {})
    if config.get("image_model") != "krea2" or not callable(getattr(patcher, "model_state_dict", None)):
        raise ValueError("Connect a Krea2 RAW MODEL from the native Load Diffusion Model node")
    for name in ("patches", "object_patches", "weight_wrapper_patches", "forced_hooks", "hook_patches"):
        if getattr(patcher, name, None):
            raise ValueError("Connect the RAW loader directly: LoRA/weight/forward patches on MODEL are not supported")
    if any(value for value in getattr(patcher, "model_options", {}).values()):
        raise ValueError("Custom MODEL forward options are not supported; connect the RAW loader directly")


class _NativeTensorReader:
    def __init__(self, state):
        self.state = state

    def get_tensor(self, key):
        # ComfyUI executes loaders inside inference_mode. Even CPU INT8 buffers
        # must become normal, independently owned tensors before saving them for
        # backward. Copy only the requested tensor; never materialize a GPU DiT.
        return self.state[key].detach().to(device="cpu", copy=True)


def load_native_training_model(patcher, *, device="cpu", compute_dtype=torch.bfloat16,
                               quantization="convrot_int8", blocks_to_swap=0, config=None, progress=None):
    validate_native_model(patcher)
    with torch.inference_mode(False):
        state = patcher.model_state_dict(filter_prefix="diffusion_model.")
        if not state:
            raise ValueError("The connected Krea2 MODEL has no diffusion weights")
        dtype_names = {torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16",
                       torch.int8: "I8", torch.uint8: "U8"}
        header = {}
        for key, tensor in state.items():
            if not isinstance(tensor, torch.Tensor) or tensor.device.type == "meta":
                raise ValueError(f"Native MODEL tensor is not materialized: {key}")
            header[key] = {"shape": list(tensor.shape), "dtype": dtype_names.get(tensor.dtype, str(tensor.dtype))}
        identity = {"type": "native_comfyui_model", "architecture": "krea2",
                    "patcher_class": type(patcher).__name__, "tensor_count": len(state)}
        cached = getattr(patcher, "cached_patcher_init", None)
        if isinstance(cached, tuple) and len(cached) == 2 and isinstance(cached[1], tuple) and cached[1]:
            # Diagnostic provenance only: this path is never opened by this adapter.
            identity["native_loader_source"] = str(cached[1][0])
        return _load_training_source(_NativeTensorReader(state), header, identity, device=device,
            compute_dtype=compute_dtype, quantization=quantization, blocks_to_swap=blocks_to_swap,
            config=config, progress=progress)
