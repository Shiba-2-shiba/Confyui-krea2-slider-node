"""Prompt-only Regional Attention nodes for Krea2 Gate A evaluation."""
from comfy_api.latest import io

from .krea2_slider_node.regional_runtime import RegionSpec, apply_regional_attention


RegionalPromptRegions = io.Custom("KREA2_SLIDER_REGIONS")


class Krea2RegionalPromptRegion(io.ComfyNode):
    """Append one conditioning and its mask to an immutable region chain."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Krea2RegionalPromptRegion",
            display_name="Krea2 Regional Prompt Region",
            category="model/krea2 slider",
            description=("Assign one plain CLIP Text Encode conditioning to one MASK. "
                         "Chain regions in priority order; overlaps belong to the first region."),
            search_aliases=["krea2 regional prompt", "krea2 prompt mask"],
            is_experimental=True,
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.Mask.Input("mask"),
                RegionalPromptRegions.Input("prev_regions", optional=True),
            ],
            outputs=[RegionalPromptRegions.Output("regions", display_name="regions")],
        )

    @classmethod
    def execute(cls, conditioning, mask, prev_regions=None):
        previous = () if prev_regions is None else prev_regions
        if not isinstance(previous, tuple) or not all(isinstance(item, RegionSpec) for item in previous):
            raise ValueError("prev_regions must come from Krea2 Regional Prompt Region")
        if len(previous) >= 4:
            raise ValueError("Krea2 Regional Attention supports at most four prompt regions")
        regions = previous + (RegionSpec(conditioning=conditioning, mask=mask, loras=()),)
        return io.NodeOutput(regions)


class Krea2ApplyRegionalAttention(io.ComfyNode):
    """Clone a Krea2 model and apply strict text/image attention routing."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Krea2ApplyRegionalAttention",
            display_name="Krea2 Apply Regional Attention",
            category="model/krea2 slider",
            description=("Apply prompt-only Regional Attention to a native Krea2 MODEL. "
                         "Use plain CLIP Text Encode inputs and do not combine with native LoRA Hooks or area nodes."),
            search_aliases=["krea2 regional attention", "krea2 attention masks"],
            is_experimental=True,
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("base"),
                io.Conditioning.Input("background"),
                RegionalPromptRegions.Input("regions"),
                io.Combo.Input("isolation", options=["strict"], default="strict"),
                io.Boolean.Input("debug_logging", default=False),
            ],
            outputs=[io.Model.Output("model"), io.Conditioning.Output("conditioning")],
        )

    @classmethod
    def execute(cls, model, base, background, regions, isolation="strict", debug_logging=False):
        regional_model, conditioning = apply_regional_attention(
            model=model,
            base=base,
            background=background,
            regions=regions,
            isolation=isolation,
            debug_logging=debug_logging,
        )
        return io.NodeOutput(regional_model, conditioning)
