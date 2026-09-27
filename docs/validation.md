# Validation record

Date: 2026-09-27. This record distinguishes code checks, memory probes and visual quality.

## Environment

Windows 11, AMD Radeon AI PRO R9700 (31.859GiB reported VRAM), torch 2.12.0+rocm7.14.0, ComfyUI 0.37.0. No NVIDIA or physical 16GB GPU was available. Installed dependencies were reused.

Real model: `krea2CatTower_v20Raw_int8.safetensors`, a local community RAW-named ConvRot INT8 model with no variant metadata. Reports identify its path, size, timestamp and safetensors header hash. This is not an official RAW quality qualification.

## Completed evidence

- Core suite: 25 tests passed, including reference-model parity, meta parameter count, ConvRot input gradients, checkpoint gradients, load validation, CPU-to-GPU staging parity, LoRA roundtrip, cancellation, job lock release, invalid export handling and operator preflight. `compileall` passed. Lint/typecheck tools were not installed and the repository had no existing lint/typecheck setup; no clean linter/typechecker result is claimed.
- Environment probe: BF16 and FP16 GEMM/GQA backward produced finite gradients.
- Standard ComfyUI V3 schemas validated; standard LoRA parser/calculation reproduced exact adapter deltas at strengths -1, 0, +1.
- Real-model synthetic-condition probe: 100 steps, 512px, rank8 attention, 16 CPU blocks, 14GiB allocator ceiling. Peak allocated 7.541GiB, peak reserved 7.803GiB. Evidence: `test-results/real-step-100.json`.
- Real Qwen3-VL prompts + real model: 3 Slider steps, 512px, rank8, 16 CPU blocks, 2-step teacher trajectory. Peak training reserved 7.828GiB; encoder reserved peak 8.613GiB in a separate phase. Adapter and report saved under `test-results/slider-acceptance/`.
- Full real-prompt 100-step run with 4 CPU blocks completed under the same 14GiB ceiling: peak allocated 12.395GiB, peak reserved 12.707GiB, 1095.25 seconds of training, finite nonzero gradients throughout. It used a 2-step trajectory for a bounded acceptance run, rather than the UI's 8-step default. Evidence: `test-results/slider-acceptance-100/result.json` and its referenced report.
- The real exported adapter matched all 140 target layers / 420 tensors in native ComfyUI's full Krea2 model key map. Standard patch calculation was also checked at -1/0/+1. Evidence: `test-results/comfy-real-export.log`.
- Actual isolated ComfyUI API workflow completed successfully: prompt `fc3c908e-aaee-44e3-877f-465072e7b818`, CLIP Loader → Encode Prompts → Train LoRA, and saved adapter/report. It used 1 step / 2-step trajectory / 16 streamed blocks. Evidence: `test-results/api-smoke-final.log` and `test-results/api-smoke-final.json`.
- Live cancellation during the training node produced `execution_interrupted`. Two subsequent identical one-step jobs completed successfully with cached CPU conditioning. Post-job GPU memory was recorded for each attempt in `test-results/runtime-lifecycle.json`; the dedicated trainer released its model and the job lock. This also exercised the final operator-preflight implementation inside ComfyUI's inference execution context.
- After cancellation and both retries, reported device free memory and torch reserved/free bytes exactly matched the pre-test values (torch reserved 163,577,856 bytes). The dedicated validation server was stopped after its queue became empty.

## Limits

These tests prove implementation behavior and measured memory on the listed environment. They do not establish NVIDIA support, physical 16GB qualification, 1024px qualification, concept disentanglement, or RAW-to-Turbo visual quality. No claim of those results is made.

The 100-step synthetic run trains against a deterministic synthetic objective and is labeled accordingly. It is not a Concept Slider quality test. The real-prompt smoke is also too short to judge visual effect.

The training workflow passed live API preflight and execution with the above bounded overrides. The preview graph is a selection template: its LoRA combo must be filled after training, and Turbo settings require an installed Turbo model.

The code-reviewer subagent was unavailable because its configured model could not be used by this account. Author review, regression tests, real model tests and live ComfyUI verification were performed; independent review remains a validation gap.

## Native-input UI correction

The split Model Config / Encode Prompts UI above describes the initial version. The current extension exposes one `Krea2SliderTrainLoRA` node with native `MODEL`, native `CLIP`, and a `prompt_yaml` combo. The example workflow uses UNETLoader + CLIPLoader + this training node.

- Full core suite: **31 passed**. New cases cover connected-model weight parity and immutability, quantized inference tensors, explicit rejection of unsupported MODEL patches, YAML blocks/anchors, filename containment, and content fingerprints. Syntax compilation passed.
- Actual ComfyUI schema reports `model: MODEL`, `clip: CLIP`, and exactly the three bundled YAML choices. Evidence: `test-results/native-ui-object-info.json`, `test-results/native-ui-schema.log`.
- Live API prompt `b5031694-c675-4b8d-8674-d6d8bb845ac4` completed native loading → six aging prompt records → one training step → adapter/report save. The smoke used 512px, rank8, 16 offloaded blocks, a 2-step trajectory and the existing 14GiB budget.
- The source was the connected `ModelPatcherDynamic`, with 878 exported tensors and 224 ConvRot INT8 Linears. No checkpoint filename was reopened by the native adapter. The YAML content hash matched the selected on-disk file.
- Training peak reserved memory: **8.139GiB**, finite nonzero LoRA gradient. Native dynamic encoder allocations can bypass PyTorch's allocator counters, so its small reported torch peak is not a physical VRAM measurement.
- Evidence: `test-results/native-ui-smoke.log`, `test-results/native-ui-smoke-api.json`, `test-results/native-ui-history.json`, and the saved report under `test-results/server-native/output/krea2_slider_loras/`.

The original long-run evidence remains valid for the original file-source backend. The new native-input backend has the above bounded smoke and regression coverage; it is not a new 100-step or physical16GB qualification.

## Single-direction default correction

The default now follows the reference MageFlow `+1`-only student objective. Bidirectional training is explicit opt-in. An omitted API field uses single, and the bundled workflow selects single. Identical condition tensors share teacher predictions.

- Full core suite: **34 tests**, including observed one-versus-two backward passes, absence of an inverse student in default mode, full single-direction MSE, and two unique teacher predictions for target=negative.
- Native ComfyUI schemas validate the optional direction combo and its single default.
- Live API runs: three steps each for single and bidirectional on the same aging YAML, 512px, rank/alpha8/8, trajectory_steps8, 16 offloaded blocks. Both completed with finite gradients and saved adapters. Their settings match except for training direction.
- Excluding the first step, single averaged 38.07s/step versus 51.62s/step bidirectional, a 26.2% reduction in this short check. Both peaked at 8.125GiB torch reserved. Evidence: `test-results/direction-comparison.json`.
- The original 100-step results remain historical bidirectional tests. New single-direction image quality and longer-run behavior have not been qualified by this short timing test.
