"""Real Krea2 CPU integration; run tools/validate_regional_attention.py.

These tests never substitute a local model for ComfyUI SingleStreamDiT.
"""
import importlib.util
from types import SimpleNamespace
import unittest

import torch

from krea2_slider_node.regional_runtime import (
    CURRENT_REGIONAL_CALL, WRAPPER_KEY, RegionSpec, RegionalAttentionWrapper,
    apply_regional_attention, compose_regional_conditioning,
)

HAS_COMFY = importlib.util.find_spec('comfy') is not None
if HAS_COMFY:
    from comfy.ldm.krea2.model import SingleStreamDiT
    from comfy.model_patcher import ModelPatcher
    from comfy.patcher_extension import WrappersMP, merge_nested_dicts
    from comfy.ldm.modules.attention import attention_pytorch


@unittest.skipUnless(HAS_COMFY, 'Use the ComfyUI Python environment and the dedicated validator')
class RegionalComfyTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(71)
        self.model = SingleStreamDiT(features=64, tdim=16, txtdim=32, heads=2, kvheads=2,
            multiplier=1, layers=3, patch=2, channels=4, txtlayers=3, txtheads=2, txtkvheads=2,
            dtype=torch.float32, device='cpu', operations=torch.nn).eval()
        with torch.no_grad():
            for parameter in self.model.parameters():
                parameter.normal_(0, 0.09)
        self.x = torch.randn(1, 4, 8, 8)
        self.ts = torch.tensor([0.7])
        self.first = torch.zeros(1, 8, 8); self.first[:, 4:, :4] = 1
        self.second = torch.zeros(1, 8, 8); self.second[:, :, 4:] = 1
        self.base = self.condition()
        self.background = self.condition()
        self.regions = (RegionSpec(self.condition(), self.first), RegionSpec(self.condition(), self.second))

    def condition(self):
        return [[torch.randn(1, 2, 96), {}]]

    def composed(self, isolation='strict', regions=None):
        output, bundle = compose_regional_conditioning(self.base, self.background, regions or self.regions,
                                                       feature_dim=96, isolation=isolation)
        return output[0][0], RegionalAttentionWrapper(bundle)

    def forward(self, x, context, wrapper=None, flags=(0,)):
        options = {'cond_or_uncond': list(flags)}
        if wrapper:
            options['wrappers'] = {WrappersMP.DIFFUSION_MODEL: {WRAPPER_KEY: [wrapper]}}
        with torch.no_grad():
            return self.model(x, self.ts.expand(x.shape[0]), context, transformer_options=options)

    def test_multistep_subject_perturbation_has_no_outside_effect(self):
        context, wrapper = self.composed()
        changed = context.clone(); changed[:, 2:4] += torch.randn_like(changed[:, 2:4]) * 2
        x1, x2 = self.x.clone(), self.x.clone()
        outside = ~self.first.bool().expand_as(self.x)
        for _ in range(3):
            y1 = self.forward(x1, context, wrapper)
            y2 = self.forward(x2, changed, wrapper)
            self.assertLess(float((y1 - y2)[outside].abs().max()), 1e-5)
            self.assertGreater(float((y1 - y2)[~outside].abs().max()), 1e-7)
            x1 = x1 - 0.1 * y1; x2 = x2 - 0.1 * y2
        self.assertEqual(wrapper.forward_count, 6)
        self.assertIsNone(CURRENT_REGIONAL_CALL.get())

    def test_one_region_unrestricted_matches_original(self):
        region = (RegionSpec(self.condition(), torch.ones(1,8,8)),)
        context, wrapper = self.composed('_unrestricted', region)
        torch.testing.assert_close(self.forward(self.x, context), self.forward(self.x, context, wrapper),
                                   atol=1e-6, rtol=1e-5)

    def test_static_5d_and_4d_match_and_odd_canvas_runs(self):
        context, wrapper = self.composed()
        torch.testing.assert_close(self.forward(self.x, context, wrapper),
                                   self.forward(self.x.unsqueeze(2), context, wrapper).squeeze(2))
        odd = torch.randn(1,4,9,7)
        self.assertEqual(self.forward(odd, context, wrapper).shape, odd.shape)

    def test_mixed_cfg_and_same_length_negative(self):
        context, wrapper = self.composed()
        original = self.forward(self.x, context)
        negative = self.forward(self.x, context, wrapper, flags=(1,))
        torch.testing.assert_close(original, negative)
        mixed = self.forward(self.x.repeat(2,1,1,1), context.repeat(2,1,1), wrapper, flags=(0,1))
        torch.testing.assert_close(mixed[1:], original)
        torch.testing.assert_close(mixed[:1], self.forward(self.x, context, wrapper))

    def test_clone_and_new_run_have_independent_state(self):
        native = torch.nn.Module()
        native.model_config = SimpleNamespace(unet_config={'image_model':'krea2'})
        native.diffusion_model = self.model
        patcher = ModelPatcher(native, torch.device('cpu'), torch.device('cpu'))
        applied, _ = apply_regional_attention(patcher, self.base, self.background, self.regions)
        copied = applied.clone()
        a = applied.get_wrappers(WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY)[0]
        b = copied.get_wrappers(WrappersMP.DIFFUSION_MODEL, WRAPPER_KEY)[0]
        self.assertIsNot(a, b)
        self.assertFalse(patcher.get_all_wrappers(WrappersMP.DIFFUSION_MODEL))
        context, _ = self.composed()
        self.forward(self.x, context, a)
        self.assertEqual((a.forward_count, b.forward_count), (1,0))
        old_run = a.run_id
        applied.pre_run()
        self.assertNotEqual(a.run_id, old_run)
        self.assertEqual(a.forward_count, 0)
        self.assertFalse(a.cache)
        self.assertEqual(b.forward_count, 0)

    def test_exception_does_not_poison_next_forward(self):
        context, wrapper = self.composed()
        def bypass(func, *args, **kwargs):
            return args[0]
        options = {'optimized_attention_override': bypass,
                   'wrappers': {WrappersMP.DIFFUSION_MODEL: {WRAPPER_KEY: [wrapper]}}}
        with self.assertRaisesRegex(RuntimeError, 'bypass'):
            self.model(self.x, self.ts, context, transformer_options=options)
        self.assertIsNone(CURRENT_REGIONAL_CALL.get())
        self.assertFalse(wrapper.cache)
        self.assertTrue(torch.isfinite(self.forward(self.x, context, wrapper)).all())

    def test_native_attention_backend_replacement_keeps_isolation(self):
        native = torch.nn.Module()
        native.model_config = SimpleNamespace(unet_config={'image_model':'krea2'})
        native.diffusion_model = self.model
        patcher = ModelPatcher(native, torch.device('cpu'), torch.device('cpu'))
        patcher.set_model_optimized_attention(attention_pytorch)
        applied, output = apply_regional_attention(patcher, self.base, self.background, self.regions)
        context = output[0][0]
        changed = context.clone(); changed[:,2:4] += 2
        # Sampling normally injects patcher.wrappers into transformer_options.
        options = merge_nested_dicts(applied.model_options['transformer_options'],
                                    {'wrappers': applied.wrappers})
        with torch.no_grad():
            before = self.model(self.x, self.ts, context, transformer_options=options)
            after = self.model(self.x, self.ts, changed, transformer_options=options)
        outside = ~self.first.bool().expand_as(self.x)
        self.assertLess(float((before-after)[outside].abs().max()), 1e-5)
        self.assertGreater(float((before-after)[~outside].abs().max()), 1e-7)

    def test_native_setter_with_unknown_mask_discarding_backend_fails(self):
        native = torch.nn.Module()
        native.model_config = SimpleNamespace(unet_config={'image_model':'krea2'})
        native.diffusion_model = self.model
        patcher = ModelPatcher(native, torch.device('cpu'), torch.device('cpu'))
        # It still performs attention, but intentionally ignores its supplied mask.
        def unmasked(q,k,v,heads,mask=None,**kw):
            return attention_pytorch(q,k,v,heads,mask=None,**kw)
        patcher.set_model_optimized_attention(unmasked)
        applied, output = apply_regional_attention(patcher, self.base, self.background, self.regions)
        options = merge_nested_dicts(applied.model_options['transformer_options'],
                                    {'wrappers': applied.wrappers})
        with self.assertRaisesRegex(RuntimeError,'bypass'):
            self.model(self.x,self.ts,output[0][0],transformer_options=options)
        self.assertIsNone(CURRENT_REGIONAL_CALL.get())


if __name__ == '__main__':
    unittest.main()
