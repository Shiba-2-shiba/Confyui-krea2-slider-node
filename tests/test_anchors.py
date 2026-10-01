"""CPU regressions for optional, additive frozen-base anchor preservation."""
import ast
import copy
import unittest
from dataclasses import fields
from pathlib import Path
from unittest.mock import patch

import torch
from test_model import tiny_config

from krea2_slider_node import model as model_api
from krea2_slider_node import training
from krea2_slider_node.conditioning import (
    PromptRecord,
    TextCondition,
    encode_prompt_records,
    validate_prompt_records,
)
from krea2_slider_node.config import TrainConfig
from krea2_slider_node.lora import LoRALinear, inject_lora
from krea2_slider_node.model import SingleStreamDiT


class RecordingClip:
    """Minimal external encoder boundary; caching/conversion are real code."""
    def __init__(self):
        self.calls = []

    def tokenize(self, text):
        return text

    def encode_from_tokens(self, tokens, return_dict):
        self.calls.append(tokens)
        return {"cond": torch.ones(1, 2, 30720)}


class AnchorInputTests(unittest.TestCase):
    def test_optional_anchor_validation(self):
        record = {"target": "person", "positive": "smiling person", "negative": "person"}
        self.assertEqual(validate_prompt_records([record]), [record])
        anchored = dict(record, anchor="a fully clothed man")
        self.assertEqual(validate_prompt_records([anchored]), [anchored])
        for value in (None, 1, True, [], {}, "", " \n\t", "a" * 8193):
            with self.subTest(anchor=repr(value)[:40]), self.assertRaises(ValueError):
                validate_prompt_records([dict(record, anchor=value)])

    def test_strength_default_and_validation(self):
        self.assertIn("anchor_strength", {field.name for field in fields(TrainConfig)})
        self.assertEqual(TrainConfig().anchor_strength, 1.0)
        for value in (0, 0.25, 1.0, 100.0):
            TrainConfig(anchor_strength=value).validate()
        for value in (-1, float("nan"), float("inf"), -float("inf"), None, "1", True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "anchor_strength"):
                TrainConfig(anchor_strength=value).validate()

    def test_encoder_caches_optional_anchor_on_cpu_and_can_skip_it(self):
        specs = [{"target": "base", "positive": "plus", "negative": "base", "anchor": "keep"},
                 {"target": "base", "positive": "plus", "negative": "base", "anchor": "keep"},
                 {"target": "base", "positive": "plus", "negative": "base"}]
        clip = RecordingClip()
        encoded = encode_prompt_records(clip, specs)
        self.assertEqual(clip.calls, ["base", "plus", "keep"])
        self.assertIs(encoded[0].conditions["anchor"], encoded[1].conditions["anchor"])
        self.assertNotIn("anchor", encoded[2].conditions)
        self.assertEqual(encoded[0].conditions["anchor"].features.device.type, "cpu")
        self.assertFalse(encoded[0].conditions["anchor"].features.requires_grad)
        clip = RecordingClip()
        skipped = encode_prompt_records(clip, specs, include_anchors=False)
        self.assertEqual(clip.calls, ["base", "plus"])
        self.assertTrue(all("anchor" not in record.conditions for record in skipped))

    def test_ui_appends_optional_anchor_without_reordering_existing_inputs(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / "nodes.py").read_text())
        schema = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "define_schema")
        inputs = next(keyword.value for node in ast.walk(schema) if isinstance(node, ast.Call)
                      for keyword in node.keywords if keyword.arg == "inputs")
        old = ["model", "clip", "prompt_yaml", "model_variant", "quantization", "blocks_to_swap",
               "memory_budget_gib", "compute_dtype", "steps", "rank", "alpha", "target", "learning_rate",
               "width", "height", "trajectory_steps", "eta", "teacher_guidance_scale", "teacher_norm_reference",
               "seed", "vary_seed", "gradient_checkpointing", "output_name", "training_direction"]
        self.assertEqual([node.args[0].value for node in inputs.elts], old + ["anchor_strength"])
        options = {kw.arg: ast.literal_eval(kw.value) for kw in inputs.elts[-1].keywords}
        self.assertEqual(options["default"], 1.0)
        self.assertEqual(options["min"], 0.0)
        self.assertTrue(options["optional"])
        self.assertTrue(options["advanced"])
        execute = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "execute")
        self.assertEqual(execute.args.args[-1].arg, "anchor_strength")
        self.assertEqual(ast.literal_eval(execute.args.defaults[-1]), 1.0)


class AnchorTrainingTests(unittest.TestCase):
    def fixture(self, anchor=True):
        torch.manual_seed(81)
        model = SingleStreamDiT(tiny_config(model_api))
        conditions = {role: TextCondition(torch.randn(1, 3, 3, 32))
                      for role in ("target", "positive", "negative", "anchor")}
        if not anchor:
            conditions.pop("anchor")
        return model, PromptRecord(conditions, {})

    def request(self, **kwargs):
        return TrainConfig(**({"steps": 1, "rank": 2, "alpha": 2, "width": 32, "height": 32,
                                   "trajectory_steps": 2, "max_grad_norm": 1000} | kwargs))

    def run_training(self, model, records, request, **kwargs):
        return training.train_steps(model, records, request, device="cpu", compute_dtype=torch.float32, **kwargs)

    @staticmethod
    def nonzero_injection(*args, **kwargs):
        targets = inject_lora(*args, **kwargs)
        with torch.no_grad():
            for module in args[0].modules():
                if isinstance(module, LoRALinear):
                    module.lora_up.weight.fill_(0.05)
        return targets

    def test_anchor_teacher_is_detached_and_signed_students_add_undiluted_loss(self):
        for mode, directions in (("single", [1]), ("bidirectional", [1, -1])):
            with self.subTest(mode=mode):
                self._check_anchor_loss(mode, directions)

    def _check_anchor_loss(self, mode, directions):
        model, record = self.fixture()
        original = {name: value.clone() for name, value in model.state_dict().items()}
        calls, gradients, adapter_gradients, handles = [], [], [], []
        current_role = [None]
        real_predict = training.predict_velocity

        def observe(model, x, t, context):
            adapter = next(m for m in model.modules() if isinstance(m, LoRALinear))
            role = next(role for role, cond in record.conditions.items() if cond.features is context)
            current_role[0] = role
            if not handles:
                handles.append(adapter.lora_up.weight.register_hook(
                    lambda grad: adapter_gradients.append((current_role[0], adapter.multiplier, grad.clone()))))
            output = real_predict(model, x, t, context)
            call = {"role": role, "grad": torch.is_grad_enabled(), "sign": adapter.multiplier,
                        "x": x.detach().clone(), "t": t.clone(), "output": output.detach().clone(),
                        "requires_grad": output.requires_grad}
            calls.append(call)
            if output.requires_grad:
                output.register_hook(lambda grad: gradients.append((role, adapter.multiplier, grad.clone())))
            return output

        with patch.object(training, "inject_lora", side_effect=self.nonzero_injection), \
             patch.object(training, "predict_velocity", side_effect=observe):
            _, report = self.run_training(model, [record], self.request(training_direction=mode, anchor_strength=2.5))
        for handle in handles:
            handle.remove()
        entry = report["steps"][0]
        anchor_teacher = next(c for c in calls if c["role"] == "anchor" and not c["grad"])
        self.assertFalse(anchor_teacher["requires_grad"])
        self.assertEqual(anchor_teacher["sign"], 0)
        students = [c for c in calls if c["grad"]]
        self.assertEqual([(c["role"], c["sign"]) for c in students],
                         [(role, sign) for sign in directions for role in ("target", "anchor")])
        anchor_losses, slider_losses = [], []
        base = next(c["output"] for c in calls[1:] if c["role"] == "target" and not c["grad"])
        positive = next(c["output"] for c in calls if c["role"] == "positive" and not c["grad"])
        negative = next(c["output"] for c in calls if c["role"] == "negative" and not c["grad"])
        for call in students:
            torch.testing.assert_close(call["x"], anchor_teacher["x"], rtol=0, atol=0)
            torch.testing.assert_close(call["t"], anchor_teacher["t"], rtol=0, atol=0)
            target = anchor_teacher["output"] if call["role"] == "anchor" else base + call["sign"] * (positive - negative)
            loss = float((call["output"] - target).square().mean()) / len(directions)
            (anchor_losses if call["role"] == "anchor" else slider_losses).append(loss)
        self.assertAlmostEqual(entry["slider_loss"], sum(slider_losses), places=6)
        self.assertAlmostEqual(entry["anchor_loss"], sum(anchor_losses), places=6)
        self.assertAlmostEqual(entry["weighted_anchor_loss"], 2.5 * sum(anchor_losses), places=6)
        self.assertAlmostEqual(entry["total_loss"], entry["slider_loss"] + entry["weighted_anchor_loss"], places=6)
        self.assertEqual(entry["loss"], entry["total_loss"])
        self.assertGreater(entry["anchor_loss"], 0)
        self.assertEqual(entry["anchor_backward_passes"], len(directions))
        self.assertEqual(entry["student_backward_passes"], 2 * len(directions))
        self.assertEqual(entry["anchor_teacher_evaluations"], 1)
        self.assertEqual(entry["teacher_evaluations"], 4)
        self.assertEqual(report["effective_anchor_records"], 1)
        for role, sign, grad in gradients:
            self.assertIn(sign, directions)
            self.assertGreater(float(grad.abs().sum()), 0, role)
        self.assertEqual([(role, sign) for role, sign, _ in adapter_gradients],
                         [(role, sign) for sign in directions for role in ("target", "anchor")])
        for role, _, grad in adapter_gradients:
            self.assertGreater(float(grad.abs().sum()), 0, role)
        for name, value in model.named_parameters():
            if ".lora_" not in name:
                self.assertFalse(value.requires_grad)
                self.assertIsNone(value.grad)
                torch.testing.assert_close(value, original[name.replace(".base.", ".")], rtol=0, atol=0)

    def test_zero_strength_and_missing_anchor_have_identical_legacy_state_and_rng(self):
        for mode in ("single", "bidirectional"):
            model, record = self.fixture()
            without = PromptRecord({k: v for k, v in record.conditions.items() if k != "anchor"}, {})
            rng = torch.random.get_rng_state().clone()
            baseline, baseline_report = self.run_training(copy.deepcopy(model), [without], self.request(steps=2, training_direction=mode))
            torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
            disabled, disabled_report = self.run_training(model, [record], self.request(steps=2, training_direction=mode, anchor_strength=0))
            torch.testing.assert_close(torch.random.get_rng_state(), rng, rtol=0, atol=0)
            for key in baseline:
                torch.testing.assert_close(disabled[key], baseline[key], rtol=0, atol=0)
            for base, zero in zip(baseline_report["steps"], disabled_report["steps"]):
                self.assertEqual(base["loss"], zero["loss"])
                self.assertEqual(base["grad_norm"], zero["grad_norm"])
                self.assertEqual(zero["anchor_backward_passes"], 0)
                self.assertEqual(zero["anchor_teacher_evaluations"], 0)
                self.assertEqual(zero["weighted_anchor_loss"], 0)

    def test_initial_zero_adapter_has_zero_anchor_loss_then_nonzero_preservation(self):
        model, record = self.fixture()
        _, report = self.run_training(model, [record], self.request(steps=2, vary_seed=False))
        self.assertEqual(report["steps"][0]["anchor_loss"], 0)
        self.assertGreater(report["steps"][1]["anchor_loss"], 0)

    def test_checkpointed_anchor_updates_match_eager_for_both_signs(self):
        model, record = self.fixture()
        with patch.object(training, "inject_lora", side_effect=self.nonzero_injection):
            eager, eager_report = self.run_training(copy.deepcopy(model), [record], self.request(training_direction="bidirectional", gradient_checkpointing=False))
            checkpointed, checkpoint_report = self.run_training(model, [record], self.request(training_direction="bidirectional", gradient_checkpointing=True))
        self.assertGreater(eager_report["steps"][0]["anchor_loss"], 0)
        self.assertGreater(checkpoint_report["steps"][0]["anchor_loss"], 0)
        for key in eager:
            torch.testing.assert_close(checkpointed[key], eager[key], rtol=1e-5, atol=1e-7)
        self.assertTrue(all(m.multiplier == 1 for m in model.modules() if isinstance(m, LoRALinear)))

    def test_mixed_records_only_apply_anchor_where_present(self):
        model, record = self.fixture()
        without = PromptRecord({k: v for k, v in record.conditions.items() if k != "anchor"}, {})
        _, report = self.run_training(model, [record, without], self.request(steps=2))
        self.assertEqual([step["anchor_active"] for step in report["steps"]], [True, False])
        self.assertEqual([step["anchor_backward_passes"] for step in report["steps"]], [1, 0])
        self.assertEqual(report["anchor_records"], 1)

    def test_anchor_teacher_reuses_matching_cached_condition(self):
        model, record = self.fixture()
        record.conditions["anchor"] = record.conditions["target"]
        _, report = self.run_training(model, [record], self.request())
        self.assertEqual(report["steps"][0]["anchor_teacher_evaluations"], 0)
        self.assertEqual(report["steps"][0]["teacher_evaluations"], 3)

    def test_cancellation_during_anchor_checkpoint_backward_restores_multiplier_and_callback(self):
        model, record = self.fixture()
        phase, block_calls = [False], [0]
        previous_check = lambda: None
        model.interrupt_check = previous_check
        def observe(_model, args):
            if torch.is_grad_enabled() and args[1] is record.conditions["anchor"].features:
                phase[0] = True
        def interrupt(_block, _args):
            if phase[0]:
                block_calls[0] += 1
                if block_calls[0] == 2:
                    raise RuntimeError("cancelled in anchor checkpoint recomputation")
        handles = [model.register_forward_pre_hook(observe), model.blocks[0].register_forward_pre_hook(interrupt)]
        try:
            with self.assertRaisesRegex(RuntimeError, "cancelled in anchor checkpoint"):
                self.run_training(model, [record], self.request(training_direction="bidirectional"), cancel=lambda: None)
        finally:
            for handle in handles:
                handle.remove()
        self.assertTrue(phase[0])
        self.assertTrue(all(m.multiplier == 1 for m in model.modules() if isinstance(m, LoRALinear)))
        self.assertIs(model.interrupt_check, previous_check)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
