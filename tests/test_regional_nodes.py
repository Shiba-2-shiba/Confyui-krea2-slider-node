"""V3 node and workflow contracts for prompt-only Regional Attention."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import types

import pytest
import torch


WORKFLOW = Path("workflows") / "krea2_two_person_attention.json"


class _Input:
    def __init__(self, id, **kwargs):
        self.id = id
        self.optional = kwargs.get("optional", False)
        self.default = kwargs.get("default")
        self.options = kwargs.get("options")
        self.Parent = types.SimpleNamespace(io_type=self.__class__.io_type)


class _Output:
    def __init__(self, id=None, display_name=None, **_kwargs):
        self.id = id
        self.display_name = display_name
        self.Parent = types.SimpleNamespace(io_type=self.__class__.io_type)


def _type(io_type):
    input_type = type("Input", (_Input,), {"io_type": io_type})
    output_type = type("Output", (_Output,), {"io_type": io_type})
    return type(io_type.title(), (), {"Input": input_type, "Output": output_type})


class _Schema:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def validate(self):
        assert self.node_id and self.inputs is not None and self.outputs is not None


class _NodeOutput:
    def __init__(self, *args, **_kwargs):
        self.result = args


class _ComfyNode:
    pass


def _fake_io():
    io = types.SimpleNamespace(
        ComfyNode=_ComfyNode,
        Schema=_Schema,
        NodeOutput=_NodeOutput,
        Conditioning=_type("CONDITIONING"),
        Mask=_type("MASK"),
        Model=_type("MODEL"),
        Combo=_type("COMBO"),
        Boolean=_type("BOOLEAN"),
    )
    io.Custom = _type
    return io


@pytest.fixture
def node_module(monkeypatch):
    fake_latest = types.ModuleType("comfy_api.latest")
    fake_latest.io = _fake_io()
    fake_api = types.ModuleType("comfy_api")
    fake_api.latest = fake_latest
    monkeypatch.setitem(sys.modules, "comfy_api", fake_api)
    monkeypatch.setitem(sys.modules, "comfy_api.latest", fake_latest)
    package_name = "krea2_regional_node_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(Path.cwd())]
    monkeypatch.setitem(sys.modules, package_name, package)
    module_name = f"{package_name}.nodes_regional_attention"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, Path("nodes_regional_attention.py"))
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_phase_a_v3_schemas_expose_only_prompt_regions_and_strict_apply(node_module):
    region = node_module.Krea2RegionalPromptRegion.define_schema()
    apply = node_module.Krea2ApplyRegionalAttention.define_schema()
    region.validate()
    apply.validate()

    assert region.node_id == "Krea2RegionalPromptRegion"
    assert [item.id for item in region.inputs] == ["conditioning", "mask", "prev_regions"]
    assert region.inputs[-1].optional
    assert [item.Parent.io_type for item in region.outputs] == ["KREA2_SLIDER_REGIONS"]

    assert apply.node_id == "Krea2ApplyRegionalAttention"
    assert [item.id for item in apply.inputs] == [
        "model", "base", "background", "regions", "isolation", "debug_logging"
    ]
    isolation = apply.inputs[4]
    assert isolation.options == ["strict"] and isolation.default == "strict"
    assert [item.Parent.io_type for item in apply.outputs] == ["MODEL", "CONDITIONING"]


def test_region_node_builds_immutable_chain_without_loras(node_module):
    conditioning_a = [[torch.zeros(1, 2, 8), {}]]
    conditioning_b = [[torch.ones(1, 2, 8), {}]]
    mask_a = torch.ones(1, 8, 8)
    mask_b = torch.eye(8).unsqueeze(0)

    first = node_module.Krea2RegionalPromptRegion.execute(conditioning_a, mask_a).result[0]
    second = node_module.Krea2RegionalPromptRegion.execute(
        conditioning_b, mask_b, prev_regions=first
    ).result[0]

    assert isinstance(first, tuple) and isinstance(second, tuple)
    assert len(first) == 1 and len(second) == 2
    assert first[0].conditioning is conditioning_a and first[0].mask is mask_a
    assert first[0].loras == ()
    assert second[:1] == first
    assert second[1].conditioning is conditioning_b and second[1].mask is mask_b


def test_apply_node_delegates_to_runtime_and_returns_both_outputs(node_module, monkeypatch):
    expected_model = object()
    expected_conditioning = object()
    captured = {}

    def fake_apply(**kwargs):
        captured.update(kwargs)
        return expected_model, expected_conditioning

    monkeypatch.setattr(node_module, "apply_regional_attention", fake_apply)
    model, base, background = object(), object(), object()
    regions = (object(),)
    result = node_module.Krea2ApplyRegionalAttention.execute(
        model, base, background, regions, isolation="strict", debug_logging=True
    ).result

    assert result == (expected_model, expected_conditioning)
    assert captured == {
        "model": model,
        "base": base,
        "background": background,
        "regions": regions,
        "isolation": "strict",
        "debug_logging": True,
    }


def _workflow():
    return json.loads(WORKFLOW.read_text(encoding="utf-8"))


def test_prompt_only_workflow_has_valid_links_and_expected_regional_path():
    workflow = _workflow()
    nodes = {node["id"]: node for node in workflow["nodes"]}
    links = {link[0]: link for link in workflow["links"]}
    assert len(nodes) == len(workflow["nodes"])
    assert len(links) == len(workflow["links"])

    for link_id, source, source_slot, target, target_slot, link_type in workflow["links"]:
        assert nodes[source]["outputs"][source_slot]["type"] == link_type
        assert link_id in nodes[source]["outputs"][source_slot]["links"]
        assert nodes[target]["inputs"][target_slot]["type"] == link_type
        assert nodes[target]["inputs"][target_slot]["link"] == link_id

    types_present = [node["type"] for node in workflow["nodes"]]
    assert types_present.count("CLIPTextEncode") == 4
    assert types_present.count("Krea2RegionalPromptRegion") == 2
    assert types_present.count("Krea2ApplyRegionalAttention") == 1
    assert types_present.count("Krea2RegionMasks") == 1
    assert types_present.count("LoraLoaderModelOnly") == 1
    assert "CreateHookLoraModelOnly" not in types_present
    assert "PairConditioningSetProperties" not in types_present
    assert "PairConditioningSetDefaultCombine" not in types_present
    assert "Krea2PairRegionArea" not in types_present

    apply_node = next(node for node in workflow["nodes"] if node["type"] == "Krea2ApplyRegionalAttention")
    assert apply_node["widgets_values"] == ["strict", True]
    sampler = next(node for node in workflow["nodes"] if node["type"] == "KSampler")
    assert sampler["widgets_values"] == [42, "fixed", 8, 1, "euler", "simple", 1]
    masks = next(node for node in workflow["nodes"] if node["type"] == "Krea2RegionMasks")
    assert [item["name"] for item in masks["inputs"]] == ["width", "height"]
    assert [item["name"] for item in apply_node["inputs"]] == ["model", "base", "background", "regions"]
    rectangles = json.loads(masks["widgets_values"][2])
    assert rectangles == [
        {"x": 0, "y": 0.367556, "w": 0.460948, "h": 0.632444},
        {"x": 0.5, "y": 0, "w": 0.5, "h": 1},
    ]


def test_workflow_prompts_keep_people_out_of_base_and_background():
    workflow = _workflow()
    encodes = [node for node in workflow["nodes"] if node["type"] == "CLIPTextEncode"]
    prompts = {node["title"]: node["widgets_values"][0] for node in encodes}
    assert set(prompts) == {"Base (shared style)", "Background", "Adult woman region", "Adult man region"}

    for title in ("Base (shared style)", "Background"):
        lowered = prompts[title].lower()
        assert all(word not in lowered for word in ("woman", "man", "person", "people", "female", "male"))
    assert "adult woman" in prompts["Adult woman region"].lower()
    assert "adult man" in prompts["Adult man region"].lower()
    assert "left half" not in prompts["Adult woman region"].lower()
    assert "right half" not in prompts["Adult man region"].lower()
