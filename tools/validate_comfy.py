"""Validate against an actual ComfyUI checkout, without starting its server."""
import argparse
import asyncio
import importlib.util
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comfy_root")
    parser.add_argument("--lora", help="Also validate a real exported adapter against the full native Krea2 key map")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(Path(args.comfy_root).resolve()))
    sys.argv = [sys.argv[0], "--cpu"]
    import comfy.options
    comfy.options.enable_args_parsing()
    assert (root / "__init__.py").exists(), "ComfyUI extension entrypoint is missing"
    spec = importlib.util.spec_from_file_location("krea2_extension_validation", root / "__init__.py", submodule_search_locations=[str(root)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = package
    spec.loader.exec_module(package)
    extension = asyncio.run(package.comfy_entrypoint())
    nodes = asyncio.run(extension.get_node_list())
    schemas = []
    for node in nodes:
        schema = node.GET_SCHEMA()
        schema.validate()
        schemas.append({"id": schema.node_id, "inputs": [item.id for item in schema.inputs], "output_node": schema.is_output_node})
    nodes_by_id = {node.GET_SCHEMA().node_id: node for node in nodes}
    assert len(nodes) == 7 and set(nodes_by_id) == {
        "Krea2SliderTrainLoRA", "Krea2NativeLoRAHooksFix", "Krea2RegionMasks",
        "Krea2ConditioningDebug", "Krea2PairRegionArea", "Krea2RegionalPromptRegion",
        "Krea2ApplyRegionalAttention",
    }, "Expose training, inference, regional attention, regional area, and diagnostics"
    inputs = {item.id: item for item in nodes_by_id["Krea2SliderTrainLoRA"].GET_SCHEMA().inputs}
    assert {"model", "clip", "prompt_yaml"} <= inputs.keys(), "Native MODEL/CLIP sockets and YAML selection are required"
    assert inputs["model"].Parent.io_type == "MODEL"
    assert inputs["clip"].Parent.io_type == "CLIP"
    assert inputs["training_direction"].default == "single"
    assert inputs["training_direction"].optional
    assert inputs["training_direction"].options == ["single", "bidirectional"]
    assert inputs["prompt_yaml"].options == ["aging_slider_fullbody.yaml", "breast_size_slider.yaml",
                                           "breast_size_slider_v2.yaml", "deaging_slider_fullbody.yaml"]
    hook_schema = nodes_by_id["Krea2NativeLoRAHooksFix"].GET_SCHEMA()
    assert [item.Parent.io_type for item in hook_schema.inputs] == ["MODEL", "BOOLEAN"]
    assert hook_schema.inputs[1].optional and hook_schema.inputs[1].default is False
    assert [item.Parent.io_type for item in hook_schema.outputs] == ["MODEL"]
    debug_schema = nodes_by_id["Krea2ConditioningDebug"].GET_SCHEMA()
    assert [item.Parent.io_type for item in debug_schema.inputs[:2]] == ["CONDITIONING", "CONDITIONING"]
    assert [item.Parent.io_type for item in debug_schema.outputs] == ["CONDITIONING", "CONDITIONING"]
    area_schema = nodes_by_id["Krea2PairRegionArea"].GET_SCHEMA()
    assert [item.Parent.io_type for item in area_schema.inputs] == ["CONDITIONING", "CONDITIONING", "MASK", "LATENT"]
    assert [item.Parent.io_type for item in area_schema.outputs] == ["CONDITIONING", "CONDITIONING"]
    prompt_region_schema = nodes_by_id["Krea2RegionalPromptRegion"].GET_SCHEMA()
    assert [item.Parent.io_type for item in prompt_region_schema.inputs] == [
        "CONDITIONING", "MASK", "KREA2_SLIDER_REGIONS"
    ]
    assert prompt_region_schema.inputs[-1].optional
    assert [item.Parent.io_type for item in prompt_region_schema.outputs] == ["KREA2_SLIDER_REGIONS"]
    regional_apply_schema = nodes_by_id["Krea2ApplyRegionalAttention"].GET_SCHEMA()
    assert [item.Parent.io_type for item in regional_apply_schema.inputs] == [
        "MODEL", "CONDITIONING", "CONDITIONING", "KREA2_SLIDER_REGIONS", "COMBO", "BOOLEAN"
    ]
    assert regional_apply_schema.inputs[4].options == ["strict"]
    assert regional_apply_schema.inputs[4].default == "strict"
    assert [item.Parent.io_type for item in regional_apply_schema.outputs] == ["MODEL", "CONDITIONING"]
    region_node = nodes_by_id["Krea2RegionMasks"]
    assert [item.Parent.io_type for item in region_node.GET_SCHEMA().outputs] == ["MASK", "MASK", "INT", "INT"]
    region_output = region_node.execute(16, 8).result
    assert region_output[0].shape == region_output[1].shape == (1, 8, 16)
    assert region_output[2:] == (16, 8)
    assert bool(((region_output[0] + region_output[1]) == 1).all())
    assert bool((region_output[0][:, :, :8] == 1).all())
    assert bool((region_output[0][:, :, 8:] == 0).all())
    assert (root / package.WEB_DIRECTORY / "region_masks.js").is_file()
    import torch
    import uuid
    import comfy.samplers
    from krea2_slider_node.region_area import latent_area_from_mask

    diagnostic_mask = torch.zeros(1, 1024, 1024)
    diagnostic_mask[:, 376:, :472] = 1
    diagnostic_latent = {"samples": torch.zeros(1, 16, 1, 128, 128)}
    area = latent_area_from_mask(diagnostic_mask, diagnostic_latent)
    assert area == (81, 59, 47, 0)
    conditions = [{"model_conds": {}, "uuid": uuid.uuid4(), "mask": diagnostic_mask,
                   "area": area, "set_area_to_bounds": False}]
    comfy.samplers.resolve_areas_and_cond_masks_multidim(
        conditions, diagnostic_latent["samples"].shape[2:], torch.device("cpu"))
    assert conditions[0]["mask"].shape == (1, 1, 128, 128)
    region = comfy.samplers.get_area_and_mult(
        conditions[0], diagnostic_latent["samples"], torch.tensor([0.5]))
    assert tuple(region.area) == (1, 81, 59, 0, 47, 0)
    assert region.input_x.shape == region.mult.shape == (1, 16, 1, 81, 59)

    import comfy.lora
    from krea2_slider_node.lora import inject_lora, lora_state_dict
    from krea2_slider_node.quantization import FrozenLinear
    model = torch.nn.Sequential(FrozenLinear(torch.zeros(4, 4)))
    inject_lora(model, rank=2, alpha=2, target="all")
    with torch.no_grad():
        model[0].lora_down.weight.fill_(0.25)
        model[0].lora_up.weight.fill_(0.5)
    state = lora_state_dict(model)
    key = "diffusion_model.0.weight"
    patches = comfy.lora.load_lora(state, {"lora_unet_0": key})
    assert len(patches) == 1
    for strength in (-1., 0., 1.):
        weight = comfy.lora.calculate_weight([(strength, patches[key], 1.0, None, None)], torch.zeros(4, 4), key)
        torch.testing.assert_close(weight, torch.full((4, 4), 0.25 * strength))
    report = {"status": "passed", "schemas": schemas, "comfy_lora_strengths": [-1, 0, 1],
              "region_area_5d": list(region.area)}
    if args.lora:
        from types import SimpleNamespace
        from safetensors.torch import load_file
        from comfy.ldm.krea2.model import SingleStreamDiT
        with torch.device("meta"):
            native = torch.nn.Module()
            native.diffusion_model = SingleStreamDiT(operations=torch.nn)
        native.model_config = SimpleNamespace(unet_config={})
        mapping = comfy.lora.model_lora_keys_unet(native, {})
        real_state = load_file(args.lora)
        expected = {key[:-len(".lora_down.weight")] for key in real_state if key.endswith(".lora_down.weight")}
        assert expected and expected <= mapping.keys(), "Export contains unknown native Krea2 keys"
        native_state = native.state_dict()
        for prefix in expected:
            output_features, input_features = native_state[mapping[prefix]].shape
            assert real_state[prefix + ".lora_down.weight"].shape[1] == input_features
            assert real_state[prefix + ".lora_up.weight"].shape[0] == output_features
        patches = comfy.lora.load_lora(real_state, mapping)
        assert len(patches) == len(expected)
        report["real_export"] = {"path": args.lora, "matched_layers": len(patches), "tensor_count": len(real_state)}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
