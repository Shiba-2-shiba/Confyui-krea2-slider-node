import importlib
import pathlib
import types
import unittest

import torch


def tiny_config(api):
    return api.SingleMMDiTConfig(features=64, tdim=16, txtdim=32, heads=4, kvheads=2,
                                multiplier=2, layers=2, patch=2, channels=4, txtheads=2,
                                txtkvheads=2, txtlayers=3)


class ModelTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec("krea2_slider_node.model"))
        return importlib.import_module("krea2_slider_node.model")

    def test_velocity_matches_official_reference_with_mask(self):
        api = self.api()
        source = pathlib.Path(__file__).resolve().parents[1] / "参考/krea-2/mmdit.py"
        if not source.exists():
            self.skipTest("Optional local upstream reference is absent")
        import sys
        ref = types.ModuleType("krea2_reference_oracle")
        sys.modules[ref.__name__] = ref
        code = source.read_text(encoding="utf-8").replace("    @torch.compile(fullgraph=True)\n", "")
        code = code.replace("with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):", "with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):")
        exec(compile(code, str(source), "exec"), ref.__dict__)
        torch.manual_seed(21)
        config = tiny_config(api)
        model = api.SingleStreamDiT(config)
        oracle = ref.SingleStreamDiT(ref.SingleMMDiTConfig(**vars(config)))
        oracle.load_state_dict(model.state_dict(), strict=True)
        img = torch.randn(1, 4, 16)
        text = torch.randn(1, 3, 3, 32)
        t = torch.tensor([0.7])
        positions = torch.zeros(1, 7, 3)
        mask = torch.tensor([[True, True, False, True, True, True, True]])
        torch.testing.assert_close(model(img, text, t, positions, mask), oracle(img, text, t, positions, mask), rtol=1e-4, atol=1e-5)

    def test_checkpoint_matches_gradients(self):
        api = self.api()
        torch.manual_seed(5)
        model = api.SingleStreamDiT(tiny_config(api)).train()
        text = torch.randn(1, 3, 3, 32)
        x = torch.randn(1, 4, 4, 4, requires_grad=True)
        baseline = api.predict_velocity(model, x, torch.tensor([0.5]), text)
        baseline.square().mean().backward()
        expected = x.grad.clone()
        x.grad = None
        model.zero_grad(set_to_none=True)
        model.gradient_checkpointing = True
        api.predict_velocity(model, x, torch.tensor([0.5]), text).square().mean().backward()
        torch.testing.assert_close(x.grad, expected)

    def test_official_architecture_is_exact_on_meta(self):
        api = self.api()
        with torch.device("meta"):
            model = api.SingleStreamDiT(api.krea2_config())
        self.assertEqual(sum(p.numel() for p in model.parameters()), 12820073036)


if __name__ == "__main__":
    unittest.main()
