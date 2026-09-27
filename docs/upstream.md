# Upstream provenance

- Krea2 official model: `krea-ai/krea-2`, `db3984fbc6e13b34c0064990fc2d95ac64d00058`, Apache-2.0. Local source: `参考/krea-2/mmdit.py`.
- Quantization and training guidance: `kohya-ss/musubi-tuner`, `4e7c7149249e7715e9168920feb4c420423abba7`. Research snapshot and SHA256 manifest: `.omx/research/krea2-20260927/`.
- Existing Slider behavior: local `Comfyui-anima-slider-node`, `fe3dc33bb0f19878106bc43e96b32066019ebfa3`. The new trainer independently implements the flow-velocity objective and scoped LoRA multiplier.
- ComfyUI integration was validated against local `15ef24d1c0333a3eba56c5cd153d8db65363ff8f` (reported version 0.37.0).

The new model retains official canonical parameter names and text-first sequence order. It removes forced compilation/cuDNN and unnecessary padded queries, selects portable SDPA, and adds non-reentrant checkpointing. A local upstream oracle test compares predictions with masked text.

The offloader does not copy upstream mutable ring-buffer code. Frozen storage is immutable; custom input backward materializes the same source weight transiently. This avoids moving trainable LoRA parameters or invalidating saved tensor versions.
