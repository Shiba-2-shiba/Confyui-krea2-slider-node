# Implementation ledger — .omx/plans/krea2-concept-slider-16gb.md

2026-09-27: User approved implementation of the written plan.

- P0 environment/contracts: environment probe passed BF16/FP16 GEMM and GQA backward on R9700. No dependencies installed.
- P1 model/conditioning: complete. Model forward matches official reference; checkpoint gradient and exact architecture tests pass. Real Krea2 CLIP features validated (10/11/14 tokens, 12 × 2560).
- P2 quantization/LoRA: CPU gradients, strict loading, sign/zero multiplier and export roundtrip pass.
- P3 memory-bound real-model probe: complete on R9700. Synthetic-condition 100 steps passed at 512px, rank8, 16 streamed blocks, 14GiB ceiling; peak reserved 7.803GiB. Physical16GB qualification remains unavailable.
- P4 slider/training: complete. Sequential positive/negative losses, RAW trajectory, optional guidance/norm reference, cancellation and deterministic local RNG. Real-prompt 100 steps completed with 4 streamed blocks: peak reserved 12.707GiB, finite gradients, adapter saved.
- P5 ComfyUI/export: complete. Current UI is one V3 training node with native MODEL/CLIP and a YAML selector (replacing the initial three custom nodes). Atomic unique export, standard LoRA search path and actual API execution verified. Full exported adapter matches native ComfyUI: 140 layers / 420 tensors.
- P6 integration/acceptance/docs: R9700 integration complete: real-prompt100-step run, native full-adapter matching, live API execution, active cancellation and two successful retries. NVIDIA, physical16GB, Linux, and RAW→Turbo visual-quality qualification remain open.
- P7 optional acceleration: deferred until measurements justify it.

Pre-flight: P1 model predicts packed raw velocity; P2 preserves canonical Linear names; P3 moves frozen tensors only; P4 keeps the LoRA multiplier fixed through each backward; P5 exports unrotated adapter weights. These contracts are consistent.

Ruling: Work directly in the requested empty implementation directory; it is not a Git repository, so worktree/commit helpers are inapplicable. Preserve the existing references and plan. Cost if wrong: files must be moved into a later Git checkout.

Ruling: Use block-selected synchronous per-Linear staging from immutable CPU weights instead of mutable GPU ring-buffer swaps initially. Backward re-reads the same CPU weight and never saves a full dequantized model. This fulfills the memory/offload contract with fewer lifetime hazards. Cost: additional PCIe transfers and slower steps; measure before adding prefetch.

Ruling: Current local INT8 file is a community RAW-named model without model metadata. It can validate architecture/memory with explicitly recorded identity, but is not evidence of official RAW quality. Do not rename it or assume RAW from its filename in the node.

Tests: availability RED→GREEN; quantization 4 RED→GREEN; model/reference 3 RED→GREEN; loader/LoRA 4 RED→GREEN; allocator API regression RED→GREEN. Combined suite: 13 passed.

Ruling: The immutable staging implementation supports all blocks on CPU, without the upstream ring-buffer requirement to retain two blocks. Expose 0..28 rather than the upstream 0..26 cap. Cost: all-block staging can be slower.

Subsequent tests: Slider/conditioning/cancellation RED→GREEN; atomic export RED→GREEN; early output validation regression RED→GREEN; teacher normalization RED→GREEN; operator preflight in inference context RED→GREEN. Final core suite so far: 25 passed, including an actual GPU staging-gradient parity test. `compileall` passed.

Final review: native code-reviewer could not launch (the configured gpt-5.4 model is unavailable for this account). A separate author review pass identified late output validation; fixed with `test_invalid_output_name_is_rejected_before_checkpoint_loading` RED→GREEN. Added a real operator preflight before loading large weights. No independent agent review is claimed.

Ruling: Keep the preview graph as a selection template with an empty LoRA combo; a future trained filename cannot be hard-coded portably. Cost: the user must select the saved adapter before preview. Training graph/API model and encoder values are verified against the live catalog.

Ruling: FP8/NF4 and fast vendor kernels remain optional follow-up work, as planned; the shipped backend is portable eager ConvRot INT8. Cost: lower throughput than a validated specialized kernel.

Runtime note: the isolated ComfyUI server initially stalled in an unrelated Numba cache creation under the protected ComfyUI checkout. A dedicated writable Numba cache and approved test-server launch resolved it. The runtime helper's search command also encountered a null alias in the upstream catalog; direct per-node `/object_info` and its preflight/run commands work. No ComfyUI or installed-skill sources were modified.

Final: no deferred minor code findings were recorded. Optional acceleration and unsupported-format work are planned follow-ups, not implemented features. No independent review, NVIDIA qualification, physical16GB qualification, or visual-quality result is claimed.

Native UI correction: implemented the user's three requested changes. The connected MODEL state is independently copied tensor by tensor; inference-mode INT8 tensors become normal CPU buffers before training. Patched or non-Krea2 models fail explicitly. Native CLIP encoding and safe YAML selection run inside the same training node. No upstream ComfyUI/MageFlow files changed.

Native UI tests: six new regression tests RED→GREEN; full suite 31 passed. Actual schema check RED→GREEN. Live native-loader + aging YAML workflow succeeded through LoRA save, peak training reserved 8.139GiB. Provided graph/API and prompt documentation updated; YAML reformatting preserved all prompt text exactly.

Direction correction: after comparing the user's MageFlow reference, changed the training default to `single` (+1 student/full MSE) and retained `bidirectional` only as an explicit optional comparison. The initial inverse objective was our added constraint, not implied by the reference enhance-only YAML. Aging text is unchanged; its comparison remains the adult baseline, not the separate toddler/deaging prompt.

Direction verification: new regressions RED→GREEN for the default one-backward behavior, explicit two-backward mode, single teacher construction, and teacher reuse; additional assertion checks full MSE. Core suite 34 tests. Both modes completed three real API steps with matching settings apart from direction; mean steps 2–3 were 38.07s single and 51.62s bidirectional. Image-quality improvement remains untested. Same-condition teacher sharing reduces redundant computation without changing its prediction target.
