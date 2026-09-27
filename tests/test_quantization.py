import importlib
import unittest

import torch
import torch.nn.functional as F


class QuantizationTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec("krea2_slider_node.quantization"))
        return importlib.import_module("krea2_slider_node.quantization")

    def test_convrot_forward_and_input_gradient_match_independent_matrix(self):
        q = self.api()
        codes = torch.tensor([[1, 2, -3, 4], [4, -2, 1, 3]], dtype=torch.int8)
        scales = torch.tensor([[0.25], [0.5]])
        h = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]]) / 2
        weight = (codes.float() * scales) @ h
        layer = q.FrozenLinear(codes, scale=scales, group_size=4)
        x = torch.tensor([[0.5, 1., -2., 3.]], requires_grad=True)
        reference = x.detach().clone().requires_grad_(True)
        y = layer(x)
        expected = F.linear(reference, weight)
        torch.testing.assert_close(y, expected)
        y.square().sum().backward()
        expected.square().sum().backward()
        torch.testing.assert_close(x.grad, reference.grad)
        self.assertFalse(any(p.requires_grad for p in layer.parameters()))

    def test_quantizer_preserves_zero_rows_and_finite_weights(self):
        q = self.api()
        weights = torch.tensor([[0., 0., 0., 0.], [1., 2., 3., 4.]])
        codes, scale = q.quantize_convrot(weights, group_size=4)
        layer = q.FrozenLinear(codes, scale=scale, group_size=4)
        torch.testing.assert_close(layer(torch.eye(4)).T, weights, atol=0.04, rtol=0.02)
        self.assertEqual(codes.dtype, torch.int8)

    def test_checkpoint_keeps_input_gradient_and_saves_no_dequantized_weight(self):
        q = self.api()
        from torch.utils.checkpoint import checkpoint
        codes = torch.ones(8, 4, dtype=torch.int8)
        layer = q.FrozenLinear(codes, scale=torch.ones(8, 1), group_size=4)
        x = torch.randn(3, 4, requires_grad=True)
        saved = []
        with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
            layer(x).sum().backward()
        self.assertFalse(any(t.shape == (8, 4) and t.is_floating_point() for t in saved))
        x.grad = None
        checkpoint(layer, x, use_reentrant=False).sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_rejects_invalid_quantization_metadata(self):
        q = self.api()
        for spec in [{}, {"format": "nvfp4"}, {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 8}]:
            with self.assertRaises(ValueError):
                q.validate_quant_spec(spec, in_features=64)


if __name__ == "__main__":
    unittest.main()
