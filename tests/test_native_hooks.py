"""Real native patchers/quantized operators; run with tools/validate_native_hooks.py."""
import importlib.util
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import torch

HAS_COMFY = importlib.util.find_spec("comfy") is not None
if HAS_COMFY:
    import comfy.hooks
    import comfy.model_patcher
    import comfy.ops
    import comfy.samplers
    from comfy.quant_ops import QuantizedTensor


@unittest.skipUnless(HAS_COMFY, "Run tools/validate_native_hooks.py with the ComfyUI Python environment")
class NativeHookTests(unittest.TestCase):
    def make_model(self, quant="int8_tensorwise", layers=1):
        ops = comfy.ops.mixed_precision_ops(compute_dtype=torch.float32)
        model = torch.nn.Module()
        model.model_config = SimpleNamespace(unet_config={"image_model": "krea2"})
        model.diffusion_model = torch.nn.ModuleList()
        for _ in range(layers):
            layer = ops.Linear(8, 8, bias=False, device="cpu")
            codes = torch.arange(64).reshape(8, 8).float() / 32 - 1
            state = {"weight": codes}
            if quant != "plain":
                spec = {"format": "int8_tensorwise" if quant == "convrot" else quant}
                if quant == "convrot":
                    spec.update(convrot=True, convrot_groupsize=4)
                dtype = torch.int8 if spec["format"] == "int8_tensorwise" else torch.float8_e4m3fn
                state = {"weight": (codes * 16).to(dtype), "weight_scale": torch.tensor(0.0625),
                         "comfy_quant": torch.tensor(list(json.dumps(spec).encode()), dtype=torch.uint8)}
            layer.load_state_dict(state)
            model.diffusion_model.append(layer)
        patcher = comfy.model_patcher.ModelPatcher(model, torch.device("cpu"), torch.device("cpu"))
        return patcher

    def repaired(self, model):
        if os.environ.get("KREA2_HOOK_TEST_BASELINE") == "1":
            return model.clone()
        from krea2_slider_node.native_hooks import repair_native_hook_model
        return repair_native_hook_model(model)

    def hook(self, model, strength=1.0, layers=1):
        hook = comfy.hooks.WeightHook(strength_model=strength, strength_clip=0)
        hook.need_weight_init = False
        hook.weights = {f"diffusion_model.{i}.weight": ("diff", (torch.full((8, 8), 0.25),))
                        for i in range(layers)}
        group = comfy.hooks.HookGroup()
        group.add(hook)
        model.add_hook_patches(hook, hook.weights, strength_patch=strength)
        return group

    @staticmethod
    def weight(layer):
        return layer.weight.dequantize() if isinstance(layer.weight, QuantizedTensor) else layer.weight.detach()

    def test_hook_switches_restore_exact_storage_and_scales(self):
        for quant in ("plain", "int8_tensorwise", "convrot", "float8_e4m3fn"):
            with self.subTest(quant=quant):
                model = self.repaired(self.make_model(quant))
                layer = model.model.diffusion_model[0]
                original = {k: v.clone() for k, v in layer.state_dict().items()}
                base = self.weight(layer).clone()
                positive, negative = self.hook(model), self.hook(model, -1)
                for hooks, sign in ((positive, 1), (negative, -1), (positive, 1)):
                    model.patch_hooks(hooks)
                    torch.testing.assert_close(self.weight(layer), base + 0.25 * sign, atol=0.14, rtol=0.04)
                model.patch_hooks(None)
                for key, value in original.items():
                    torch.testing.assert_close(layer.state_dict()[key].float(), value.float(), rtol=0, atol=0)
                self.assertFalse(model.hook_backup)
                self.assertFalse(model.cached_hook_patches)

    def test_direct_setter_path_accepts_quantized_replacement(self):
        model = self.repaired(self.make_model())
        model.hook_mode = comfy.hooks.EnumHookMode.MinVram
        hooks = self.hook(model)
        key = "diffusion_model.0.weight"
        weight, _, convert = comfy.model_patcher.get_key_weight(model.model, key)
        model.patch_hook_weight_to_device(hooks, model.get_combined_hook_patches(hooks), key,
                                         {key: [(weight, convert)]}, None)
        self.assertIsInstance(model.model.diffusion_model[0].weight, QuantizedTensor)
        model.unpatch_hooks()

    def test_clone_retains_fix_without_patching_global_class_or_input(self):
        source = self.make_model()
        native_method = comfy.model_patcher.ModelPatcher.patch_hooks
        model = self.repaired(source)
        clone = model.clone()
        self.assertIsNot(model, source)
        self.assertNotIn("patch_hooks", source.__dict__)
        self.assertIs(comfy.model_patcher.ModelPatcher.patch_hooks, native_method)
        clone.patch_hooks(self.hook(clone))
        clone.cleanup()
        self.assertFalse(clone.hook_backup)

    def test_dynamic_delegate_inherits_fix_via_native_clone_callback(self):
        class DynamicFixture(comfy.model_patcher.ModelPatcher):
            def is_dynamic(self):
                return True

        original = self.make_model("convrot")
        dynamic = DynamicFixture(original.model, torch.device("cpu"), torch.device("cpu"))
        dynamic.cached_patcher_init = (lambda disable_dynamic=False: self.make_model("convrot"), ())
        repaired = self.repaired(dynamic)
        self.assertTrue(repaired.is_dynamic())
        self.assertNotIn("patch_hook_weight_to_device", repaired.__dict__)
        delegate = repaired.clone(disable_dynamic=True)
        self.assertFalse(delegate.is_dynamic())
        delegate.patch_hooks(self.hook(delegate))
        delegate.cleanup()
        self.assertFalse(delegate.hook_backup)

    def test_real_lora_registration_and_keyframe_strengths(self):
        model = self.repaired(self.make_model("convrot"))
        key = "diffusion_model.0.weight"
        hook = comfy.hooks.WeightHook(strength_model=1.0, strength_clip=0)
        hook.need_weight_init = False
        hook.weights = {key: comfy.lora.load_lora({"test.lora_down.weight": torch.full((2, 8), 0.5),
                                                  "test.lora_up.weight": torch.full((8, 2), 0.25)},
                                                 {"test": key})[key]}
        group = comfy.hooks.HookGroup()
        group.add(hook)
        registered = model.register_all_hook_patches(
            group, comfy.hooks.create_target_dict(comfy.hooks.EnumWeightTarget.Model))
        self.assertEqual(len(registered), 1)
        layer = model.model.diffusion_model[0]
        base = self.weight(layer).clone()
        for strength in (0.0, 1.0, -1.0, 0.5):
            hook.hook_keyframe = comfy.hooks.HookKeyframeGroup()
            hook.hook_keyframe.add(comfy.hooks.HookKeyframe(strength=strength))
            model.patch_hooks(group)
            torch.testing.assert_close(self.weight(layer), base + strength * 0.25, atol=0.025, rtol=0.02)
        model.cleanup()
        torch.testing.assert_close(self.weight(layer), base, rtol=0, atol=0)

    def test_whitelist_restore_only_selected_weights(self):
        model = self.repaired(self.make_model(layers=2))
        before = [self.weight(layer).clone() for layer in model.model.diffusion_model]
        model.patch_hooks(self.hook(model, layers=2))
        model.unpatch_hooks(set())
        self.assertEqual(len(model.hook_backup), 2)
        model.unpatch_hooks({"diffusion_model.0.weight"})
        torch.testing.assert_close(self.weight(model.model.diffusion_model[0]), before[0], rtol=0, atol=0)
        self.assertEqual(set(model.hook_backup), {"diffusion_model.1.weight"})
        model.unpatch_hooks()
        torch.testing.assert_close(self.weight(model.model.diffusion_model[1]), before[1], rtol=0, atol=0)

    def test_rejects_other_models_and_active_hooks(self):
        if os.environ.get("KREA2_HOOK_TEST_BASELINE") == "1":
            self.skipTest("Repair-specific input validation")
        from krea2_slider_node.native_hooks import repair_native_hook_model
        wrong = self.make_model()
        wrong.model.model_config.unet_config["image_model"] = "other"
        with self.assertRaisesRegex(ValueError, "Krea2"):
            repair_native_hook_model(wrong)
        active = self.repaired(self.make_model())
        active.patch_hooks(self.hook(active))
        try:
            with self.assertRaisesRegex(ValueError, "active"):
                repair_native_hook_model(active)
        finally:
            active.cleanup()

    def test_inference_mode_hook_replacement_and_restore(self):
        with torch.inference_mode():
            model = self.repaired(self.make_model("convrot"))
            original = self.weight(model.model.diffusion_model[0]).clone()
            model.patch_hooks(self.hook(model))
            self.assertGreater(float((self.weight(model.model.diffusion_model[0]) - original).abs().mean()), 0.15)
            model.cleanup()
            torch.testing.assert_close(self.weight(model.model.diffusion_model[0]), original, rtol=0, atol=0)

    def test_downstream_maxspeed_cannot_enable_partial_weight_cache(self):
        model = self.repaired(self.make_model())
        hooks = self.hook(model)
        model.hook_mode = comfy.hooks.EnumHookMode.MaxSpeed
        model.cached_hook_patches[hooks] = {"diffusion_model.0.weight": (torch.zeros(8, 8), torch.device("cpu"))}
        base = self.weight(model.model.diffusion_model[0]).clone()
        model.patch_hooks(hooks)
        torch.testing.assert_close(self.weight(model.model.diffusion_model[0]), base + 0.25, atol=0.02, rtol=0.02)
        self.assertEqual(model.hook_mode, comfy.hooks.EnumHookMode.MinVram)
        self.assertFalse(model.cached_hook_patches)
        model.cleanup()

    def test_failure_on_second_layer_restores_first_layer(self):
        model = self.repaired(self.make_model(layers=2))
        original = self.weight(model.model.diffusion_model[0]).clone()
        hooks = self.hook(model, layers=2)
        native_calculate = comfy.lora.calculate_weight

        def fail_second(patches, weight, key, **kwargs):
            if key.endswith("1.weight"):
                raise RuntimeError("deliberate second layer failure")
            return native_calculate(patches, weight, key, **kwargs)

        with patch.object(comfy.lora, "calculate_weight", fail_second):
            with self.assertRaisesRegex(RuntimeError, "deliberate second layer failure"):
                model.patch_hooks(hooks)
        torch.testing.assert_close(self.weight(model.model.diffusion_model[0]), original, rtol=0, atol=0)
        self.assertFalse(model.hook_backup)
        self.assertIsNone(model.current_hooks)
        self.assertFalse(model.cached_hook_patches)

    def test_native_sampler_blends_hooked_region_and_unhooked_default(self):
        model = self.repaired(self.make_model())
        base_model = model.model
        base_model.current_patcher = model
        base_model.memory_required = lambda shape, cond_shapes=None: 0
        def apply_model(x, timestep, **kwargs):
            return torch.ones_like(x) * self.weight(base_model.diffusion_model[0]).mean()

        base_model.apply_model = apply_model
        mask = torch.zeros(1, 4, 4)
        mask[:, :, :2] = 1
        hooks = self.hook(model)
        region = {"model_conds": {}, "uuid": uuid.uuid4(), "mask": mask, "mask_strength": 1.0, "hooks": hooks}
        default = {"model_conds": {}, "uuid": uuid.uuid4(), "default": True}
        x = torch.zeros(1, 1, 4, 4)
        output, = comfy.samplers.calc_cond_batch(base_model, [[region, default]], x, torch.tensor([0.5]), {})
        model.patch_hooks(None)
        baseline = self.weight(base_model.diffusion_model[0]).mean()
        torch.testing.assert_close(output[..., 2:], torch.full((1, 1, 4, 2), baseline), rtol=0, atol=0)
        self.assertGreater(float(output[..., :2].mean() - baseline), 0.15)


if __name__ == "__main__":
    unittest.main()
