from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from krea2_slider_node.memory import MemoryBudget


class MemoryTests(unittest.TestCase):
    def test_operator_preflight_works_inside_inference_context_without_changing_rng(self):
        import krea2_slider_node.memory as memory
        self.assertTrue(hasattr(memory, "probe_training_operators"))
        rng = torch.get_rng_state().clone()
        with torch.inference_mode():
            result = memory.probe_training_operators("cpu", torch.float32)
        self.assertTrue(result["finite_gradients"])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))

    def test_unindexed_cuda_budget_restores_previous_allocator_limit(self):
        state = {"fraction": 0.8}
        def get_fraction(device):
            if isinstance(device, torch.device) and device.index is None:
                raise ValueError("CUDA allocator API requires an explicit index")
            return state["fraction"]
        def set_fraction(fraction, device):
            state["fraction"] = fraction
        with ExitStack() as stack:
            for name, value in {
                "current_device": lambda: 0,
                "get_device_properties": lambda _: SimpleNamespace(total_memory=32 * 2**30),
                "mem_get_info": lambda _: (30 * 2**30, 32 * 2**30),
                "memory_allocated": lambda _: 0,
                "get_per_process_memory_fraction": get_fraction,
                "set_per_process_memory_fraction": set_fraction,
                "reset_peak_memory_stats": lambda _: None,
            }.items():
                stack.enter_context(patch.object(torch.cuda, name, value))
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                with MemoryBudget("cuda", 14):
                    self.assertLessEqual(state["fraction"] * 32, 14)
                    raise RuntimeError("cancelled")
            self.assertEqual(state["fraction"], 0.8)
