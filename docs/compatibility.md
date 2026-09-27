# Compatibility contract

| Layer | Supported contract |
|---|---|
| GPU API | PyTorch CUDA namespace; detect ROCm with `torch.version.hip` and NVIDIA with `torch.version.cuda` |
| Compute | BF16 by default, FP16 with GradScaler option; differentiable PyTorch SDPA/GQA |
| Storage | Native Krea2 BF16/FP16/FP32 or ComfyUI ConvRot INT8 with per-row scale and metadata |
| Architecture | 28 blocks, width 6144, Qwen3-VL 12 × 2560 features, 16-channel latents, 2×2 packing |
| LoRA | FP32 adapter parameters, standard `lora_unet_*` down/up/alpha tensors in unrotated coordinates |
| Offloading | Immutable CPU frozen weights for selected blocks, per-Linear synchronous staging; adapters remain on GPU |
| ComfyUI | One V3 training node, native MODEL + CLIP inputs, YAML dropdown; standard CLIP Loader type `krea2` |

The native MODEL input must be an unpatched Krea2 RAW loader output. Its state is copied one tensor at a time on CPU into the dedicated training model, including conversion of inference-mode tensors into normal autograd-compatible storage. LoRA/forward/object patches are rejected explicitly instead of silently using different weights. The filename reload path remains available to standalone CLI probes only.

FP8/NF4/GGUF inputs are rejected rather than treated as INT8. A `cuda` device string alone does not imply NVIDIA. Triton, FlashAttention and bitsandbytes are not runtime requirements.

The current GPU evidence covers Windows 11 / R9700 / torch 2.12.0+rocm7.14.0. CUDA/NVIDIA and Linux remain unverified. A 14GiB PyTorch ceiling on a 32GB card is not equivalent to a physical 16GB qualification.

Reference files are never imported at runtime. Installing this package does not require the `参考` or `.omx/research` directories. The optional upstream parity test is skipped when the local official reference is absent; the remaining model/gradient tests still run.
