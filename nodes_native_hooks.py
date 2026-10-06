"""Inference compatibility node for native masked LoRA Hooks."""
from comfy_api.latest import io

from .krea2_slider_node.native_hooks import repair_native_hook_model


class Krea2NativeLoRAHooksFix(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Krea2NativeLoRAHooksFix",
            display_name="Krea2 Native LoRA Hooks Fix",
            category="model/krea2 slider",
            description=("Repair native Krea2 LoRA Hooks for mixed-precision weights. "
                         "Connect MODEL output to the sampler; use native masked conditioning to select the region. "
                         "Uses uncached Hooks to preserve quantization metadata."),
            search_aliases=["krea2 regional lora", "masked slider", "quantized hooks fix"],
            is_experimental=True,
            inputs=[io.Model.Input("model"),
                    io.Boolean.Input("debug_logging", default=False, advanced=True, optional=True)],
            outputs=[io.Model.Output("model")],
        )

    @classmethod
    def execute(cls, model, debug_logging=False):
        return io.NodeOutput(repair_native_hook_model(model, debug_logging=debug_logging))
