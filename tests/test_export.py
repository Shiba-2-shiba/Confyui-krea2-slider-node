import importlib
import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors import safe_open


class ExportTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec("krea2_slider_node.lora_io"))
        return importlib.import_module("krea2_slider_node.lora_io")

    def test_output_is_unique_and_roundtrips_tensor_and_report(self):
        api = self.api()
        state = {"lora_unet_blocks_0_attn_wq.lora_up.weight": torch.ones(4, 2),
                 "lora_unet_blocks_0_attn_wq.lora_down.weight": torch.ones(2, 4),
                 "lora_unet_blocks_0_attn_wq.alpha": torch.tensor(2.)}
        with tempfile.TemporaryDirectory() as d:
            first, report_path = api.save_adapter(state, {"settings": {"rank": 2, "alpha": 2}}, d, "smile")
            second, _ = api.save_adapter(state, {"settings": {"rank": 2, "alpha": 2}}, d, "smile")
            self.assertNotEqual(first, second)
            with safe_open(first, framework="pt") as f:
                self.assertEqual(f.metadata()["base_model_architecture"], "krea2")
                torch.testing.assert_close(f.get_tensor("lora_unet_blocks_0_attn_wq.alpha"), torch.tensor(2.))
            self.assertEqual(json.loads(Path(report_path).read_text())["settings"]["rank"], 2)
            self.assertFalse(list(Path(d).glob("*.tmp*")))

    def test_rejects_path_traversal_and_nonfinite_export(self):
        api = self.api()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                api.save_adapter({"weight": torch.ones(1)}, {}, d, "../escape")
            with self.assertRaises(ValueError):
                api.save_adapter({"weight": torch.tensor([float("nan")])}, {}, d, "slider")
            self.assertFalse(list(Path(d).iterdir()))
