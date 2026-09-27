import importlib
import json
import pathlib
import tempfile
import unittest

import torch
from safetensors.torch import save_file, load_file

from krea2_slider_node.model import SingleStreamDiT, predict_velocity
from test_model import tiny_config
from krea2_slider_node import model as model_api


class LoadingTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec("krea2_slider_node.model_io"))
        return importlib.import_module("krea2_slider_node.model_io")

    def test_streamed_bf16_source_load_preserves_predictions(self):
        api = self.api()
        config = tiny_config(model_api)
        torch.manual_seed(10)
        source = SingleStreamDiT(config).eval()
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "model.safetensors"
            save_file(source.state_dict(), path)
            loaded, report = api.load_training_model(path, config=config, quantization="bf16_reference", compute_dtype=torch.float32, blocks_to_swap=1)
            x, text, t = torch.randn(1, 4, 4, 4), torch.randn(1, 3, 3, 32), torch.tensor([0.4])
            torch.testing.assert_close(predict_velocity(source, x, t, text), predict_velocity(loaded, x, t, text))
            self.assertFalse(any(p.requires_grad for p in loaded.parameters()))
            self.assertEqual(report["offloaded_blocks"], [1])

    def test_prequantized_comfy_metadata_is_loaded_and_validated(self):
        api = self.api()
        from krea2_slider_node.quantization import quantize_convrot
        config = tiny_config(model_api)
        state = SingleStreamDiT(config).state_dict()
        name = "blocks.0.attn.wq"
        state[name + ".weight"], state[name + ".weight_scale"] = quantize_convrot(state[name + ".weight"], 4)
        spec = dict(format="int8_tensorwise", convrot=True, convrot_groupsize=4, per_row=True)
        state[name + ".comfy_quant"] = torch.tensor(list(json.dumps(spec).encode()), dtype=torch.uint8)
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "model.safetensors"
            save_file(state, path)
            loaded, report = api.load_training_model(path, config=config, compute_dtype=torch.float32)
            self.assertEqual(loaded.blocks[0].attn.wq.weight.dtype, torch.int8)
            self.assertGreater(report["quantized_linears"], 0)
            state.pop(name + ".weight_scale")
            invalid_path = pathlib.Path(d) / "invalid.safetensors"
            save_file(state, invalid_path)
            with self.assertRaisesRegex(ValueError, "scale"):
                api.load_training_model(invalid_path, config=config)

    def test_shape_mismatch_and_turbo_metadata_fail_before_training(self):
        api = self.api()
        config = tiny_config(model_api)
        state = SingleStreamDiT(config).state_dict()
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "wrong.safetensors"
            save_file(state, path, metadata={"model_variant": "turbo"})
            with self.assertRaisesRegex(ValueError, "RAW"):
                api.load_training_model(path, config=config)
            state["first.weight"] = torch.ones(3, 4)
            save_file(state, path)
            with self.assertRaisesRegex(ValueError, "shape"):
                api.load_training_model(path, config=config)


class LoRATests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec("krea2_slider_node.lora"))
        return importlib.import_module("krea2_slider_node.lora")

    def test_only_lora_updates_and_export_roundtrips_both_signs(self):
        api = self.api()
        from krea2_slider_node.quantization import FrozenLinear
        model = torch.nn.Sequential(FrozenLinear(torch.randn(4, 4)))
        base = model[0].weight.clone()
        adapters = api.inject_lora(model, rank=2, alpha=2, target="all")
        x = torch.randn(3, 4)
        expected = model(x).detach()
        opt = torch.optim.AdamW(api.lora_parameters(model), lr=0.01)
        model(x).square().sum().backward(); opt.step()
        self.assertTrue(torch.equal(model[0].base.weight, base))
        with api.lora_multiplier(model, 0):
            torch.testing.assert_close(model(x), expected)
        with api.lora_multiplier(model, -1):
            negative = model(x).detach()
        state = api.lora_state_dict(model)
        fresh = torch.nn.Sequential(FrozenLinear(base.clone()))
        api.inject_lora(fresh, rank=2, alpha=2, target="all")
        api.load_lora_state_dict(fresh, state)
        with api.lora_multiplier(fresh, -1):
            torch.testing.assert_close(fresh(x), negative)
        self.assertEqual(len(adapters), 1)
        self.assertIn("lora_unet_0.lora_down.weight", state)


if __name__ == "__main__":
    unittest.main()
