import importlib
import json
from types import SimpleNamespace
import unittest

import torch

from krea2_slider_node import model as model_api
from krea2_slider_node.model import SingleStreamDiT, predict_velocity
from krea2_slider_node.lora import inject_lora, lora_parameters
from krea2_slider_node.quantization import quantize_convrot
from test_model import tiny_config


class NativePatcherFixture:
    """The external ModelPatcher boundary; real tensors and loader are exercised."""
    def __init__(self, state):
        self.state = {"diffusion_model." + k: v for k, v in state.items()}
        self.model = SimpleNamespace(model_config=SimpleNamespace(unet_config={"image_model": "krea2"}))
        self.patches, self.object_patches, self.weight_wrapper_patches = {}, {}, {}
        self.model_options = {"transformer_options": {}}
        self.forced_hooks = None

    def model_state_dict(self, filter_prefix=None):
        return {k: v for k, v in self.state.items() if filter_prefix is None or k.startswith(filter_prefix)}


class NativeInputTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec("krea2_slider_node.native_model"))
        return importlib.import_module("krea2_slider_node.native_model")

    def test_connected_model_weights_are_used_without_loading_a_filename(self):
        api = self.api()
        config = tiny_config(model_api)
        torch.manual_seed(112)
        source = SingleStreamDiT(config)
        patcher = NativePatcherFixture(source.state_dict())
        loaded, report = api.load_native_training_model(patcher, config=config, compute_dtype=torch.float32,
                                                        quantization="bf16_reference", blocks_to_swap=1)
        x, t, text = torch.randn(1, 4, 4, 4), torch.tensor([0.5]), torch.randn(1, 3, 3, 32)
        torch.testing.assert_close(predict_velocity(loaded, x, t, text), predict_velocity(source, x, t, text))
        original = {k: v.clone() for k, v in patcher.state.items()}
        inject_lora(loaded, rank=2, alpha=2)
        optimizer = torch.optim.AdamW(lora_parameters(loaded))
        predict_velocity(loaded, x, t, text).square().mean().backward()
        optimizer.step()
        for name, value in patcher.state.items():
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)
        self.assertEqual(report["source"]["type"], "native_comfyui_model")

    def test_native_quantized_inference_tensors_can_backpropagate(self):
        api = self.api()
        config = tiny_config(model_api)
        with torch.inference_mode():
            state = SingleStreamDiT(config).state_dict()
            prefix = "blocks.0.attn.wq"
            state[prefix + ".weight"], state[prefix + ".weight_scale"] = quantize_convrot(state[prefix + ".weight"], 4)
            spec = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 4}
            state[prefix + ".comfy_quant"] = torch.tensor(list(json.dumps(spec).encode()), dtype=torch.uint8)
        patcher = NativePatcherFixture(state)
        loaded, _ = api.load_native_training_model(patcher, config=config, compute_dtype=torch.float32, blocks_to_swap=2)
        self.assertEqual(loaded.blocks[0].attn.wq.weight.dtype, torch.int8)
        self.assertFalse(any(t.is_inference() for t in loaded.buffers()))
        inject_lora(loaded, rank=2, alpha=2)
        predict_velocity(loaded, torch.randn(1, 4, 4, 4), torch.tensor([0.5]), torch.randn(1, 3, 3, 32)).square().mean().backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in lora_parameters(loaded)), 0)

    def test_modified_or_wrong_architecture_model_is_not_silently_ignored(self):
        api = self.api()
        patcher = NativePatcherFixture({})
        patcher.patches = {"diffusion_model.first.weight": [("some_lora",)]}
        with self.assertRaisesRegex(ValueError, "patch|LoRA"):
            api.load_native_training_model(patcher)
        patcher.patches = {}
        patcher.model.model_config.unet_config["image_model"] = "anima"
        with self.assertRaisesRegex(ValueError, "Krea2"):
            api.load_native_training_model(patcher)
