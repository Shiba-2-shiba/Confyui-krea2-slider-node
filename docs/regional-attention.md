# Krea2 Regional Attention

## Status and goal

This feature is being implemented in stages. The first executable stage is prompt-only regional attention: one KSampler run routes the base, background, woman, and man text segments to owned image tokens. Regional Slider LoRA support is a later stage and must not be treated as implemented or validated until prompt-only Gate A passes on the user's machine.

The target repository began this work at commit `df1d2f3f18f7b2114b27914b6c227b2597da488b`. The reference repository is `../ComfyUI-Krea2-Regional` at commit `307081f2b954d9f5e683dadafdb53ca971ebdbfd`. Both worktrees were clean when Task 1 provenance was inspected, except that the target already contained the untracked implementation plan at `.omx/plans/2026-10-06-krea2-regional-attention.md`.

## Why attention routing is needed

The previous mask and area workflows apply separate conditioning predictions to spatial regions. They do not prevent a prompt's subject concept from affecting the shared denoising context. In the recorded small-mask comparison, a second woman remained outside the requested woman region even when both regional Slider LoRAs were set to zero. Changing only the uncovered-background wording also failed in an earlier user-run comparison. This makes prompt routing the first behavior to test before adding regional LoRA.

Regional attention assigns every image token to one region or to the background complement. Text/image attention edges are then allowed only by that ownership policy. The initial `strict` policy is intentionally stronger than the reference implementation: base text does not read image tokens, each subject region communicates with its own text and image tokens plus base text, and background tokens communicate within the background plus base text. This is designed to prevent multi-layer information relays through shared image or text segments.

An attention permission matrix proves routing behavior, but it does not guarantee image quality or exact pixel bounds. Krea2 patchification turns a pixel mask into a coarser token boundary. Gate A therefore evaluates both subject count and placement tolerance.

## Reference implementation and adaptation boundary

The design is based on these parts of `ComfyUI-Krea2-Regional/krea2_regional.py`:

- `_latent_mask` for reducing an input mask to the effective Krea2 image-token grid.
- `_build_allow` for constructing joint text/image and text-fusion attention permissions.
- `_diffusion_wrapper` and `WrappersMP.DIFFUSION_MODEL` integration for installing an attention override on a cloned ComfyUI model.
- `Krea2RegionalPrompt` and `Krea2ApplyRegional.apply` for representing regions, concatenating conditioning segments, and attaching run data to the model clone.
- `_RegionalLoRAMixin`, `_patched_linear`, `_inject_lora`, and `_build_token_masks` as the later regional LoRA design reference. These functions are documented here for provenance; their presence in the reference does not mean regional LoRA is available in the current prompt-only phase.

The implementation uses project-specific node IDs and data types. It adds explicit background ownership, strict input validation, `ContextVar`-scoped runtime state, cond/uncond identification, existing-attention-mask composition, and fail-closed handling for unsupported attention formats. The existing two-rectangle `Krea2 Region Masks` editor is reused. The reference Builder, its JavaScript and server routes, captioning, Detailer, adaptive masks, region lock, and LoKr support are outside the initial scope.

The reference MIT license is retained verbatim in `licenses/Krea2-Regional-MIT.txt`. The source's `Copyright (c) 2026 YOUR_NAME` text is intentionally unchanged.

## Reproduction evidence

Machine-readable evidence is stored in `test-results/regional-attention/evidence.json`. Only files that were present during capture are listed as file evidence. At capture time, the available comparison was:

| Setting | `regional_area_00001` | `regional_area_00002` |
| --- | --- | --- |
| Resolution | 1024 x 1024 | 1024 x 1024 |
| Seed / sampler | 42 / Euler simple, 8 steps, CFG 1 | same |
| Woman mask | `x=0, y=0.367556, w=0.460948, h=0.632444` | same |
| Man mask | `x=0.5, y=0, w=0.5, h=1` | same |
| Breast Slider | 3 | 0 |
| Deaging Slider | 3 | 0 |
| Global darkbrush | 0.8 | 0.8 |
| User-observed result | extra woman outside the small woman region | extra woman outside the small woman region |

The prompts, hashes, and complete shared generation settings are in the JSON evidence. The embedded debug label says `lora_off` in both files even though image `00001` records Slider strengths of 3; the numeric node inputs are treated as authoritative.

`ログ.txt` is overwritten by later ComfyUI runs. The current file captured in the evidence predates both available PNGs and contains the earlier failing `mask_bounds_lora_off` run (`too many values to unpack`), so it is not evidence for the successful Regional Area images. Earlier conversation analysis reported eight Hook applications with 140 backed-up weights and zero remaining backups after restore. That is retained only as a historical observation because the corresponding earlier log file is no longer present.

## Prompt-only Gate A

Gate A is run by the user on the real ComfyUI, INT8 model, and RTX A4000 environment after the prompt-only workflow is delivered. All regional LoRAs remain disconnected or at zero. The small mask and left-half mask are each generated with seeds `42`, `123`, `777`, `2026`, and `31415`, for ten images total.

Gate A passes only when all ten images contain one woman and one man, show no extra woman face, head, body, or boundary double image outside the woman region, and keep placement within one effective image-token boundary of the requested mask. Each PNG must retain its API metadata and be paired with the matching run ID log. CPU routing tests and a successfully queued workflow do not by themselves pass Gate A.

If Gate A fails, regional LoRA implementation pauses while the routing log, background owner assignment, and base-prompt contribution are isolated. Slider strength, prompt wording, and training changes are not used to mask a prompt-routing failure.

## Delivered prompt-only path

The two new V3 nodes are `Krea2RegionalPromptRegion` and `Krea2ApplyRegionalAttention`. A region chain uses the project-specific `KREA2_SLIDER_REGIONS` socket. The Apply node clones the model and returns one concatenated CONDITIONING. `strict` is the only exposed isolation mode in this stage. The reserved regional LoRA field must be empty; no Regional Slider node is exposed yet.

Every text input must be one plain CLIP Text Encode entry, using the Krea2 encoder's feature width. Padding identified by a binary `attention_mask` is removed before concatenation. Hooks, area/mask metadata, reference latents, custom token patches, video, empty regions, and incompatible attention overrides stop with an explanation. Global standard model LoRA such as darkbrush may be applied before Apply. Subject prompts are centered within the selected rectangle; the global base prompt does not name subjects.

The runtime masks both text-sequence refinement and joint text/image attention. Encoder-tap attention inside each text token remains unchanged. Both masked stages must be reached or the run fails. Existing boolean restrictions are intersected, and additive masks retain their biases. Negative rows are selected using `cond_or_uncond`, including negatives with the same token length. Each model clone has its own wrapper and cache; `ON_PRE_RUN` creates a new run ID and clears counts/cache. Exceptions restore the parent `ContextVar` and clear the cache.

## Validation and diagnostics

Run the regular CPU regression suite from this repository with `python -B -m pytest -q`. The real Krea2 integration suite is separate:

```powershell
# Use the Python interpreter belonging to your ComfyUI installation.
python -B tools/validate_regional_attention.py "C:\path\to\ComfyUI"
python -B tools/validate_comfy.py "C:\path\to\ComfyUI"
```

The first command requires eight actual native `SingleStreamDiT` tests with no skips: three-step perturbation, unrestricted equivalence, 4D/5D and odd dimensions, CFG, clone/run independence, exception recovery, native attention replacement, and rejection of an unknown replacement that discards masks. It reports the ComfyUI commit and fails on missing dependencies or uncollected/skipped tests. This stage does not validate regional INT8 LoRA. The second command validates V3 registration and the existing node contracts.

Local validation against the user's core commit `7c8fbc698b3c3dce0f525f6b5044b93d9395c5c9` cannot import the model because the local Python environment lacks `comfy_aimdo.storage`; its installed `comfy_kitchen` also lacks the newer ConvRot operator. No dependencies were installed to bypass this limitation. Pure CPU routing tests do not replace this integration result or real generation Gate A.

With `debug_logging=true`, events use the existing `[Krea2HookDebug]` prefix and distinct `regional_*` names. `regional_run_start` includes the run ID, source revisions, runtime-file SHA-256 (including uncommitted changes), torch version, and zero regional LoRA count. `regional_geometry` includes latent/token dimensions, segments, pixel-mask hashes, owner counts and token bounding boxes (exclusive upper bounds), mask size, and base-to-image allowed-edge count (zero in strict). `regional_forward` records the masked forward count, CFG flags, joint/text/tap attention calls, and CUDA peak counters. Negative-only forwards use the unchanged model path and do not increment the masked forward counter. Peak counters are process-wide; this node does not reset other GPU users' counters. Save the complete run log with each PNG, including any `regional_error` event. Prompts and weights are not printed.

Owner bounds are token coordinates, not original pixels. For a standard patch-2 model and 128×128 latent, the token grid is 64×64. Decode and patch boundaries can affect neighboring pixels, so mathematical attention isolation is not a promise of identical outside-mask pixels or of artifact-free subject placement.
