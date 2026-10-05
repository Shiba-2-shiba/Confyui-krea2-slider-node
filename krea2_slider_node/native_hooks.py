"""Instance-scoped compatibility repair for native Krea2 weight Hooks.

ComfyUI remains responsible for Hook registration, schedules, and regional
conditioning. This adapter only repairs weight access, replacement and restore.
"""
from types import MethodType

import torch


_CLONE_CALLBACK_KEY = "krea2_slider.native_hooks_fix"


def _identity(weight, **kwargs):
    return weight


def _get_key_patches(self, filter_prefix=None):
    from comfy.model_patcher import get_key_weight

    # Quantized state_dicts contain serialized scales/metadata which are not
    # module attributes. Enumerate real parameters/buffers instead.
    with self.use_ejected():
        tensors = dict(self.model.named_parameters())
        tensors.update(self.model.named_buffers())
        result = {}
        for key in tensors:
            if filter_prefix is not None and not key.startswith(filter_prefix):
                continue
            weight, _, convert = get_key_weight(self.model, key)
            if key in self.backup:
                weight = self.backup[key].weight
            elif key in self.hook_backup:
                weight = self.hook_backup[key][0]
            result[key] = [(weight, convert or _identity)] + self.patches.get(key, [])
        return result


def _patch_hook_weight(self, hooks, combined_patches, key, original_weights, memory_counter):
    import comfy.float
    import comfy.lora
    import comfy.model_management
    import comfy.utils
    from comfy.model_patcher import get_key_weight

    if key not in combined_patches:
        return
    weight, setter, convert = get_key_weight(self.model, key)
    if key not in self.hook_backup:
        # Keep packed storage AND quantization parameters. Never back up a
        # dequantized approximation or accumulate copies for every HookGroup.
        self.hook_backup[key] = (weight.to(device=self.offload_device, copy=True), weight.device)
    temporary = comfy.model_management.cast_to_device(weight, weight.device, torch.float32, copy=True)
    if convert is not None:
        temporary = convert(temporary, inplace=True)
    output = comfy.lora.calculate_weight(combined_patches[key], temporary, key, original_weights=original_weights)
    if setter is not None:
        # Ask the native operator to prepare a COMPLETE replacement, including
        # recalculated scales and ConvRot parameters, without mutating storage.
        output = setter(output, inplace_update=False, return_weight=True, seed=comfy.utils.string_to_seed(key))
        if not isinstance(output, torch.Tensor):
            raise RuntimeError(f"Native weight setter did not return a replacement tensor: {key}")
        comfy.utils.set_attr_param(self.model, key, output)
    else:
        output = comfy.float.stochastic_rounding(output, weight.dtype, seed=comfy.utils.string_to_seed(key))
        comfy.utils.copy_to_param(self.model, key, output)
    original_weights.pop(key, None)


def _unpatch_hooks(self, whitelist_keys_set=None):
    import comfy.utils
    from comfy.model_patcher import get_key_weight

    with self.use_ejected():
        # An empty whitelist means restore NO keys, rather than all of them.
        keys = list(self.hook_backup)
        if whitelist_keys_set is not None:
            keys = [key for key in keys if key in whitelist_keys_set]
        for key in keys:
            original, device = self.hook_backup[key]
            _, setter, _ = get_key_weight(self.model, key)
            restored = original.to(device=device)
            if setter is not None:
                # Restoring the packed tensor directly is exact; calling its
                # setter again would unnecessarily requantize the backup.
                comfy.utils.set_attr_param(self.model, key, restored)
            else:
                comfy.utils.copy_to_param(self.model, key, restored)
            del self.hook_backup[key]
        if not self.hook_backup:
            self.current_hooks = None


def _patch_hooks(self, hooks):
    import comfy.hooks
    from comfy.model_patcher import ModelPatcher

    # Native MaxSpeed caches can contain partially built groups and unpacked
    # float weights. Use native MinVram and never publish such a cache.
    self.hook_mode = comfy.hooks.EnumHookMode.MinVram
    self.cached_hook_patches.clear()
    try:
        return ModelPatcher.patch_hooks(self, hooks)
    except BaseException:
        self.unpatch_hooks()
        self.current_hooks = None
        raise


def _install(patcher):
    import comfy.hooks
    from comfy.patcher_extension import CallbacksMP

    # DynamicVRAM routes conditioned Hooks to a native non-dynamic delegate.
    # Keep that routing and install the repair on the delegate via ON_CLONE.
    if not patcher.is_dynamic():
        patcher.get_key_patches = MethodType(_get_key_patches, patcher)
        patcher.patch_hook_weight_to_device = MethodType(_patch_hook_weight, patcher)
        patcher.unpatch_hooks = MethodType(_unpatch_hooks, patcher)
        patcher.patch_hooks = MethodType(_patch_hooks, patcher)
        patcher.hook_mode = comfy.hooks.EnumHookMode.MinVram
        patcher.cached_hook_patches.clear()
    patcher.remove_callbacks_with_key(CallbacksMP.ON_CLONE, _CLONE_CALLBACK_KEY)
    patcher.add_callback_with_key(CallbacksMP.ON_CLONE, _CLONE_CALLBACK_KEY, _on_clone)


def _on_clone(source, clone):
    _install(clone)


def repair_native_hook_model(model):
    """Return a repaired MODEL clone; never alter ComfyUI classes globally."""
    from comfy.model_patcher import ModelPatcher

    config = getattr(getattr(getattr(model, "model", None), "model_config", None), "unet_config", {})
    if not isinstance(model, ModelPatcher) or config.get("image_model") != "krea2":
        raise ValueError("Connect a native Krea2 MODEL to Krea2 Native LoRA Hooks Fix")
    if model.hook_backup or model.current_hooks is not None:
        raise ValueError("Cannot repair a MODEL while weight Hooks are active")
    repaired = model.clone()
    _install(repaired)
    return repaired
