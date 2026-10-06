"""Spatial area metadata for masked conditioning on Krea2's 3D latent."""
from comfy_api.latest import io

from .krea2_slider_node.region_area import apply_area_to_conditioning_pair


class Krea2PairRegionArea(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Krea2PairRegionArea",
            display_name="Krea2 Pair Region Area",
            category="model/krea2 slider",
            description=("Derive a spatial sampling area from a MASK for positive and negative "
                         "conditioning. Use after Cond Pair Set Props with set_cond_area=default "
                         "to avoid ComfyUI's 2D-only mask bounds path on Krea2 latents."),
            search_aliases=["krea2 regional prompt", "krea2 mask bounds fix"],
            is_experimental=True,
            inputs=[io.Conditioning.Input("positive"), io.Conditioning.Input("negative"),
                    io.Mask.Input("mask"), io.Latent.Input("latent")],
            outputs=[io.Conditioning.Output("positive"), io.Conditioning.Output("negative")],
        )

    @classmethod
    def execute(cls, positive, negative, mask, latent):
        result = apply_area_to_conditioning_pair(positive, negative, mask, latent)
        return io.NodeOutput(*result)
