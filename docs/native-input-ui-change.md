# Native MODEL / CLIP and YAML selector

User-requested correction: connect the native diffusion MODEL and CLIP directly to the training node and choose a YAML from `prompts/`.

Implementation contract:

- The training node accepts MODEL, CLIP and a YAML dropdown. The examples use native UNETLoader and CLIPLoader(type=krea2).
- Native MODEL weights are the training source. Do not infer a filename from the graph and silently reload an unrelated checkpoint.
- Read the connected unpatched Krea2 state on CPU and feed the existing strict loader tensor by tensor; preserve INT8 metadata and normalize inference tensors before autograd. The connected model is never trained or modified.
- Existing weight/forward patches are explicitly rejected initially instead of silently ignored. Direct native RAW loaders are the supported input contract.
- The native model stays under ComfyUI memory management; release inference residency before encoding and before training. Keep the existing 14GiB budget and dedicated frozen-weight/LoRA backend.
- Read YAML with `safe_load`, validate the same target/positive/negative/neutral schema, restrict dropdown selections to packaged files, and invalidate cached selections when contents change.
- Update the provided UI/API workflow and README. The previous split three-custom-node workflow is replaced by one training node with two native loader nodes.

Verification: regression tests for actual source weights and immutability, quantized/inference tensors, unsupported patches, YAML parsing/traversal/fingerprint; real ComfyUI schema and API execution with native MODEL/CLIP and a packaged YAML.
