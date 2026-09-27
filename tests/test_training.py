import importlib
from dataclasses import fields
import unittest

import torch

from krea2_slider_node import model as model_api
from krea2_slider_node.model import SingleStreamDiT
from test_model import tiny_config


class TrainingTests(unittest.TestCase):
    def api(self, name):
        self.assertIsNotNone(importlib.util.find_spec("krea2_slider_node." + name))
        return importlib.import_module("krea2_slider_node." + name)

    def test_teacher_sign_is_explicit_and_detached(self):
        api = self.api("slider_loss")
        base = torch.tensor([2.], requires_grad=True)
        plus, minus = api.slider_teachers(base, torch.tensor([4.]), torch.tensor([1.]), 0.5)
        torch.testing.assert_close(plus, torch.tensor([3.5]))
        torch.testing.assert_close(minus, torch.tensor([0.5]))
        self.assertFalse(plus.requires_grad)

    def test_default_training_uses_only_plus_one_and_reuses_identical_teacher_conditions(self):
        self._check_directional_training("single", [1])

    def test_bidirectional_training_remains_an_explicit_two_backward_option(self):
        self._check_directional_training("bidirectional", [1, -1])

    def _check_directional_training(self, mode, expected_signs):
        api = self.api("training")
        cond = self.api("conditioning")
        config_api = self.api("config")
        from krea2_slider_node.lora import LoRALinear
        torch.manual_seed(23)
        model = SingleStreamDiT(tiny_config(model_api))
        baseline = cond.TextCondition(torch.randn(1, 3, 3, 32))
        record = cond.PromptRecord({"target": baseline, "negative": baseline,
                                   "positive": cond.TextCondition(torch.randn(1, 3, 3, 32))}, {})
        kwargs = dict(steps=1, rank=2, alpha=2, width=32, height=32, trajectory_steps=2)
        if mode != "single":
            self.assertIn("training_direction", {f.name for f in fields(config_api.TrainConfig)})
            kwargs["training_direction"] = mode
        forwards, backwards, grad_handles, predictions = [], [], [], []
        def observe(module, _args):
            adapter = next(m for m in module.modules() if isinstance(m, LoRALinear))
            forwards.append((torch.is_grad_enabled(), adapter.multiplier))
            if not grad_handles:
                grad_handles.append(adapter.lora_up.weight.register_hook(lambda grad: backwards.append(grad.clone())))
        handle = model.register_forward_pre_hook(observe)
        output_handle = model.register_forward_hook(
            lambda _module, _args, output: predictions.append((torch.is_grad_enabled(), output.detach().clone())))
        try:
            _, report = api.train_steps(model, [record], config_api.TrainConfig(**kwargs), device="cpu", compute_dtype=torch.float32)
        finally:
            handle.remove()
            output_handle.remove()
            for h in grad_handles:
                h.remove()
        self.assertEqual([sign for grad, sign in forwards if grad], expected_signs)
        self.assertEqual(len(backwards), len(expected_signs))
        # One trajectory step and two unique teacher predictions (target == negative).
        self.assertEqual(sum(not grad for grad, _ in forwards), 3)
        self.assertEqual(report["student_directions"], expected_signs)
        self.assertEqual(report["settings"]["training_direction"], mode)
        self.assertGreater(report["steps"][0]["grad_norm"], 0)
        if mode == "single":
            # eta=1, no normalization, compare=target => teacher is the actual
            # positive prediction. Verify full MSE, not the old half-weight loss.
            positive = [value for grad, value in predictions if not grad][-1]
            student = [value for grad, value in predictions if grad][0]
            error = (student - positive).square()
            expected = error.sum() / error.numel()
            torch.testing.assert_close(torch.tensor(report["steps"][0]["loss"]), expected, rtol=1e-5, atol=1e-8)

    def test_single_teacher_does_not_create_an_inverse_target(self):
        api = self.api("slider_loss")
        self.assertIn("directions", __import__("inspect").signature(api.slider_teachers).parameters)
        teachers = api.slider_teachers(torch.tensor([2.]), torch.tensor([4.]), torch.tensor([1.]),
                                      0.5, directions=(1,))
        self.assertEqual(len(teachers), 1)
        torch.testing.assert_close(teachers[0], torch.tensor([3.5]))

    def test_optional_teacher_normalization_preserves_reference_norm(self):
        api = self.api("slider_loss")
        self.assertIn("normalize_to", __import__("inspect").signature(api.slider_teachers).parameters)
        plus, minus = api.slider_teachers(torch.tensor([[3., 4.]]), torch.tensor([[2., 1.]]),
                                          torch.tensor([[0., 0.]]), 0.5, normalize_to=torch.tensor([[0., 2.]]))
        torch.testing.assert_close(plus.norm(dim=1), torch.tensor([2.]))
        torch.testing.assert_close(minus.norm(dim=1), torch.tensor([2.]))

    def test_conditioning_unpacks_layer_axis_and_removes_masked_tokens(self):
        api = self.api("conditioning")
        packed = torch.arange(3 * 12 * 2560, dtype=torch.float32).reshape(1, 3, 12 * 2560)
        cond = api.from_comfy_conditioning(packed, torch.tensor([[1, 0, 1]]))
        self.assertEqual(tuple(cond.features.shape), (1, 2, 12, 2560))
        torch.testing.assert_close(cond.features[:, 1].float(), packed[:, 2].reshape(1, 12, 2560).to(cond.features.dtype).float())
        with self.assertRaises(ValueError):
            api.from_comfy_conditioning(torch.ones(1, 2, 4096))

    def test_prompt_records_reject_ambiguous_or_empty_inputs(self):
        api = self.api("conditioning")
        for value in ['[]', '[{"target":"person"}]', '{"target":"person"}']:
            with self.assertRaises(ValueError):
                api.parse_prompt_records(value)
        records = api.parse_prompt_records('[{"target":"person","positive":"smiling person","negative":"neutral person"}]')
        self.assertEqual(records[0]["negative"], "neutral person")

    def test_tiny_slider_updates_both_directions_and_preserves_base(self):
        api = self.api("training")
        cond = self.api("conditioning")
        config_api = self.api("config")
        torch.manual_seed(42)
        model = SingleStreamDiT(tiny_config(model_api))
        original = model.first.weight.detach().clone()
        record = cond.PromptRecord({role: cond.TextCondition(torch.randn(1, 3, 3, 32)) for role in ("target", "positive", "negative")}, {})
        request = config_api.TrainConfig(steps=4, rank=2, alpha=2, learning_rate=0.001, width=32, height=32,
                                        training_direction="bidirectional",
                                        trajectory_steps=2, vary_seed=False)
        state, report = api.train_steps(model, [record], request, device="cpu", compute_dtype=torch.float32)
        self.assertEqual(len(report["steps"]), 4)
        self.assertTrue(torch.equal(original, model.first.weight))
        self.assertTrue(any(v.count_nonzero() for k, v in state.items() if k.endswith("lora_up.weight")))
        self.assertTrue(all(s["grad_norm"] > 0 for s in report["steps"]))
        self.assertLess(report["steps"][-1]["loss"], report["steps"][0]["loss"])

    def test_cancellation_does_not_leave_adapter_multiplier_changed(self):
        api = self.api("training")
        cond = self.api("conditioning")
        from krea2_slider_node.lora import LoRALinear
        config_api = self.api("config")
        model = SingleStreamDiT(tiny_config(model_api))
        record = cond.PromptRecord({r: cond.TextCondition(torch.zeros(1, 3, 3, 32)) for r in ("target", "positive", "negative")}, {})
        def cancel():
            raise RuntimeError("cancelled")
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            api.train_steps(model, [record], config_api.TrainConfig(steps=2, width=32, height=32),
                            device="cpu", compute_dtype=torch.float32, cancel=cancel)
        self.assertTrue(all(m.multiplier == 1 for m in model.modules() if isinstance(m, LoRALinear)))
        self.assertIsNone(model.interrupt_check)
