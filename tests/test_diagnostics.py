"""Metadata-only diagnostics for regional conditioning and native Hooks."""
import json
import logging
import sys
import types
from contextlib import nullcontext
from types import SimpleNamespace

import torch

from krea2_slider_node.diagnostics import (describe_hooks, diagnostic_fingerprint,
                                           object_token, summarize_conditioning)


def test_conditioning_summary_shows_hooked_mask_and_unhooked_default():
    left = torch.zeros(1, 8, 8)
    left[:, 2:, :3] = 1
    right = torch.zeros(1, 8, 8)
    right[:, :, 4:] = 1
    hook = SimpleNamespace(hook_type=SimpleNamespace(value="weight"), _strength_model=2.0,
                           hook_ref=object())
    group = SimpleNamespace(hooks=[hook])
    positive = [
        [torch.zeros(1), {"mask": left, "mask_strength": 1.0, "hooks": group,
                          "set_area_to_bounds": False}],
        [torch.zeros(1), {"mask": right, "mask_strength": 1.0}],
        [torch.zeros(1), {"default": True}],
    ]
    negative = [[torch.zeros(1), {"default": True}]]

    result = summarize_conditioning(positive, negative, label="small-mask")

    assert result["event"] == "conditioning"
    assert result["label"] == "small-mask"
    assert result["positive"]["entries"][0]["mask"]["bounds_xyxy"] == [0, 2, 3, 8]
    assert result["positive"]["entries"][0]["mask"]["coverage"] == 0.28125
    assert result["positive"]["entries"][0]["hooks"]["count"] == 1
    assert result["positive"]["entries"][0]["hooks"]["items"][0]["strength_model"] == 2.0
    assert result["positive"]["entries"][1]["hooks"]["present"] is False
    assert result["positive"]["entries"][2]["default"] is True
    assert result["positive"]["uncovered_fraction"] == 0.21875
    assert result["negative"]["entries"][0]["default"] is True
    json.dumps(result)
    assert positive[0][1]["mask"] is left
    assert positive[0][1]["hooks"] is group


def test_empty_mask_and_hook_group_are_reported_without_tensor_content():
    mask = torch.zeros(1, 4, 4)
    group = SimpleNamespace(hooks=[])

    result = summarize_conditioning([[torch.ones(2), {"mask": mask, "hooks": group}]], [],
                                    label="empty")

    assert result["positive"]["entries"][0]["mask"]["bounds_xyxy"] is None
    assert result["positive"]["entries"][0]["mask"]["coverage"] == 0.0
    assert result["positive"]["entries"][0]["hooks"]["present"] is True
    assert result["positive"]["entries"][0]["hooks"]["count"] == 0
    assert "tensor" not in json.dumps(result).lower()
    assert describe_hooks(None) == {"present": False, "group_id": None, "count": 0,
                                    "items": []}


def test_object_tokens_are_opaque_and_fingerprint_is_stable_when_disabled(monkeypatch):
    item = object()
    token = object_token(item)
    assert token == object_token(item)
    assert token.startswith("obj_") and len(token) == 16
    assert token != hex(id(item))
    assert object_token(None) is None
    assert diagnostic_fingerprint(False) == diagnostic_fingerprint(False) == "disabled"
    values = iter((100, 101))
    monkeypatch.setattr("krea2_slider_node.diagnostics.time.time_ns", lambda: next(values))
    assert diagnostic_fingerprint(True) != diagnostic_fingerprint(True)


def test_mismatched_mask_sizes_do_not_report_a_false_uncovered_fraction():
    positive = [[torch.zeros(1), {"mask": torch.ones(1, 4, 4)}],
                [torch.zeros(1), {"mask": torch.ones(1, 8, 8)}],
                [torch.zeros(1), {"mask": torch.ones(1, 8, 8)}]]

    result = summarize_conditioning(positive, [], label="mismatched")

    assert result["positive"]["uncovered_fraction"] is None


def test_debug_hook_transition_reports_restore_without_weight_data(monkeypatch, caplog):
    from krea2_slider_node.native_hooks import _patch_hooks

    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    hooks_module = types.ModuleType("comfy.hooks")
    hooks_module.EnumHookMode = SimpleNamespace(MinVram="min_vram")
    model_patcher = types.ModuleType("comfy.model_patcher")

    class FakePatcher:
        def __init__(self):
            self.model = object()
            self.current_hooks = None
            self.hook_backup = {}
            self.cached_hook_patches = {"stale": 1}
            self._krea2_hook_debug_logging = True

        def patch_hooks(self, group):
            self.current_hooks = group
            self.hook_backup = {"private-weight-name": object()} if group else {}
            return "patched"

        def unpatch_hooks(self):
            self.hook_backup.clear()
            self.current_hooks = None

    model_patcher.ModelPatcher = FakePatcher
    comfy.hooks = hooks_module
    comfy.model_patcher = model_patcher
    for name, module in (("comfy", comfy), ("comfy.hooks", hooks_module),
                         ("comfy.model_patcher", model_patcher)):
        monkeypatch.setitem(sys.modules, name, module)

    patcher = FakePatcher()
    group = SimpleNamespace(hooks=[])
    with caplog.at_level(logging.WARNING, logger="krea2_slider_node.diagnostics"):
        assert _patch_hooks(patcher, group) == "patched"
        assert _patch_hooks(patcher, None) == "patched"

    events = [json.loads(record.message.split(" ", 1)[1]) for record in caplog.records
              if record.message.startswith("[Krea2HookDebug] ")]
    assert [event["event"] for event in events] == ["hook_switch_start", "hook_switch_end",
                                                    "hook_switch_start", "hook_switch_end"]
    assert events[-1]["requested"]["present"] is False
    assert events[-1]["backup_after"] == 0
    assert "private-weight-name" not in json.dumps(events)


def test_debug_restore_reports_exact_backup_clear(monkeypatch, caplog):
    from krea2_slider_node.native_hooks import _unpatch_hooks

    comfy = types.ModuleType("comfy")
    comfy.__path__ = []
    utils = types.ModuleType("comfy.utils")
    utils.copy_to_param = lambda model, key, value: model.weight.data.copy_(value)
    model_patcher = types.ModuleType("comfy.model_patcher")
    model_patcher.get_key_weight = lambda model, key: (model.weight, None, None)
    comfy.utils = utils
    comfy.model_patcher = model_patcher
    for name, module in (("comfy", comfy), ("comfy.utils", utils),
                         ("comfy.model_patcher", model_patcher)):
        monkeypatch.setitem(sys.modules, name, module)

    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.tensor([2.0]))
    patcher = SimpleNamespace(model=model, hook_backup={"private-weight-name":
                              (torch.tensor([1.0]), torch.device("cpu"))},
                              current_hooks=object(), _krea2_hook_debug_logging=True,
                              use_ejected=nullcontext)
    with caplog.at_level(logging.WARNING, logger="krea2_slider_node.diagnostics"):
        _unpatch_hooks(patcher)

    assert model.weight.item() == 1.0
    assert patcher.hook_backup == {}
    assert patcher.current_hooks is None
    event = json.loads(caplog.records[-1].message.split(" ", 1)[1])
    assert event["event"] == "hook_restore"
    assert event["restored_count"] == 1 and event["backup_after"] == 0
    assert event["current_hooks_cleared"] is True
    assert "private-weight-name" not in caplog.records[-1].message
