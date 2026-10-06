"""Opt-in pass-through node for inspecting regional conditioning metadata."""
from comfy_api.latest import io

from .krea2_slider_node.diagnostics import (LOGGER, LOG_PREFIX, diagnostic_fingerprint,
                                            emit_debug, summarize_conditioning)


class Krea2ConditioningDebug(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Krea2ConditioningDebug",
            display_name="Krea2 Conditioning Debug",
            category="model/krea2 slider",
            description=("Log mask bounds, default areas, and Hook metadata before sampling. "
                         "Passes CONDITIONING through unchanged; no prompt text or tensors are logged."),
            search_aliases=["krea2 hook debug", "regional conditioning log"],
            is_experimental=True,
            inputs=[io.Conditioning.Input("positive"), io.Conditioning.Input("negative"),
                    io.String.Input("label", default="region comparison"),
                    io.Boolean.Input("enabled", default=True)],
            outputs=[io.Conditioning.Output("positive"), io.Conditioning.Output("negative")],
        )

    @classmethod
    def fingerprint_inputs(cls, enabled=True, **kwargs):
        return diagnostic_fingerprint(enabled)

    @classmethod
    def execute(cls, positive, negative, label="region comparison", enabled=True):
        if enabled:
            try:
                emit_debug(summarize_conditioning(positive, negative, label=label))
            except Exception:
                LOGGER.exception("%s conditioning diagnostics failed (label=%s)", LOG_PREFIX, label)
                raise
        return io.NodeOutput(positive, negative)
