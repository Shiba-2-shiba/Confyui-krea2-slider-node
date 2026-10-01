# Anchor preservation

An optional per-record `anchor` prompt asks the adapter to preserve the base model's prediction for a context that should not follow the slider. The bundled gender-specific presets use a fully clothed adult of the opposite gender in the matching scene. This is a regularizer, not a hard gender gate, and it does **not guarantee zero spillover**.

## Input and compatibility

- A record still requires `target`, `positive`, and `negative`; `neutral` remains optional and defaults to `target`.
- `anchor` is an optional, nonempty string, subject to the same prompt length limit. It is a preservation condition, not a second target or a negative generation prompt. The field is unrelated to YAML's `&name`/`*name` alias syntax.
- The advanced, optional node input `anchor_strength` defaults to `1.0`. It must be finite and nonnegative. Its default is a starting point, not a tuned optimum.
- A record without `anchor`, or a run with `anchor_strength=0`, uses the legacy slider-only path. Disabled anchors add no text encoding, anchor model forwards, or backwards. Existing YAMLs and workflows can omit the new inputs.
- Each enabled anchor is encoded through the same CLIP/cache path as the other roles. Identical text shares conditioning. Only the current record's needed conditions are staged to the training device.

## Objective

The original slider teachers, target/positive/negative/neutral text, teacher normalization, `eta`, and direction selection are unchanged. Let `D` be `{+1}` for `single` and `{+1, -1}` for `bidirectional`. Let `x_t` and `t` be the same training latent and timestep used by the slider student, and `c_anchor` the record's anchor condition:

```text
anchor_teacher = stop_gradient(model(x_t, t, c_anchor; LoRA = 0))
anchor_loss = mean over d in D of
    MSE(model(x_t, t, c_anchor; LoRA = d), anchor_teacher)

weighted_anchor_loss = anchor_strength * anchor_loss
total_loss = slider_loss + weighted_anchor_loss
```

The anchor teacher is evaluated with LoRA disabled under `no_grad`, detached, and reused across directions. The anchor student uses active, trainable LoRA at the same signed multiplier as that direction's slider student. It does not use a positive/negative delta or teacher normalization.

The losses are additive. Adding an anchor does not divide the slider loss by the number of slider-plus-anchor terms and does not dilute the original slider objective. In bidirectional mode, each loss is averaged over its two directions; in single mode each is a full MSE. Gradients accumulate into the same LoRA parameters before one optimizer update. Frozen base parameters remain frozen. Student graphs are processed separately to avoid retaining both slider and anchor graphs at once.

An active anchor adds one student forward/backward per trained direction. Its base prediction adds at most one unique teacher evaluation, with reuse possible when conditioning is identical to an existing teacher condition. Expect additional runtime; do not assume the old timing or memory measurements apply unchanged.

## Bundled presets

All six presets contain six training records. Original filenames and all four original slider-role texts are preserved.

| Preset | Positive direction | Preservation anchor |
|---|---|---|
| `aging_slider_fullbody.yaml` | Adult woman's facial, neck and hand skin ages | Matching adult man |
| `aging_slider_fullbody_male.yaml` | Adult man's facial, neck and hand skin ages | Matching adult woman |
| `deaging_slider_fullbody.yaml` | Adult woman becomes a fully clothed toddler girl | Matching adult man |
| `deaging_slider_fullbody_male.yaml` | Adult man becomes a fully clothed toddler boy | Matching adult woman |
| `breast_size_slider_v2.yaml` | Larger breasts on a fully clothed adult woman, contrasted with small breasts | Matching fully clothed adult man with an ordinary male chest |
| `breast_size_slider.yaml` | Original moderate-to-larger clothed adult female bust | None; unchanged legacy preset |

The male aging/deaging variants change gender terms and pronouns only. Clothing, hair, framing, background and concept wording are intentionally held fixed, including the existing skirts and blouses. Anchors use the opposite-gender adult baseline rather than the aged or toddler positive. Deaging intentionally retains the adult-to-toddler face, head-to-body and limb-proportion change; both genders remain fully clothed in a nonsexual composition. No child prompts are used in either breast preset.

For breast v2, `target = neutral` describes medium breasts, `positive` describes large breasts and `negative` describes small breasts. **Do not replace its negative with target or anchor.** The legacy breast v1 remains byte-for-byte unchanged and has no anchor field.

## Reports

The run settings include `anchor_strength`. `anchor_records` counts records with encoded anchor conditioning; `effective_anchor_records` is that count when strength is positive, otherwise zero. The node skips anchor encoding when strength is zero, so both counts can be zero even if the source YAML contains anchor text. These are run-time conditioning counts, not a source-file inventory. Per-step history includes:

- `slider_loss`: the unchanged direction-averaged slider objective
- `anchor_loss`: the unweighted direction-averaged preservation loss, zero when inactive
- `weighted_anchor_loss`: `anchor_strength * anchor_loss`
- `total_loss` and the compatible `loss` alias: the sum of the slider and weighted anchor losses
- `anchor_active`: whether this step used an anchor
- `slider_teacher_evaluations`, `anchor_teacher_evaluations`, and `teacher_evaluations`: actual unique teacher prediction counts; shared conditioning can avoid an extra anchor teacher call
- `slider_backward_passes`, `anchor_backward_passes`, and `student_backward_passes`: slider, anchor and total student backward counts

Use separate losses and counts to confirm that preservation actually ran. A small anchor loss is not proof that generated images are unchanged.

## Required image evaluation

CPU tests can check input validation, conditioning reuse, gradients, loss accounting, disabled paths and report structure. They cannot establish visual gender isolation or the effect of a real Krea2 adapter. The anchor presets have not established image quality merely by passing those tests. Earlier GPU acceptance and timing reports predate anchors.

Before treating a trained anchor adapter as validated:

1. Train on the actual RAW model and GPU, recording the checkpoint, precision/quantization, seed, prompts, direction, `anchor_strength`, settings, losses and peak memory. Compare otherwise matched runs with anchor strength `0` and `1`.
2. For each adapter, generate both male and female subjects at LoRA strengths **`-1 / 0 / +1`**, holding seed, prompt, sampler and generation settings fixed within each triplet. Include the six training contexts and held-out clothing/framing contexts. Use more than one seed.
3. Check the intended concept on the target gender and unwanted age, body-shape, face, clothing and background changes on the preservation gender. For deaging, retain fully clothed, nonsexual prompts and explicitly check the intended adult-to-toddler transformation.
4. Record RAW and Turbo evaluation separately, with their respective inference settings. An effect on RAW is not evidence of the same effect on Turbo.
5. Report both successful and failed comparisons. In `single` mode the `-1` inference result is an extrapolation with no separately trained inverse objective; judge it independently. If preservation is insufficient, adjust strength and evaluate the tradeoff against slider effect rather than claiming a hard guarantee.

See [training direction](training-direction.md) for the unchanged slider objective and [prompt usage](../prompts/README.md) for preset selection.

## Design reference

The preservation design was compared with ai-toolkit's active [ConceptSliderTrainer](https://github.com/ostris/ai-toolkit/blob/main/extensions_built_in/concept_slider/ConceptSliderTrainer.py), inspected on 2026-10-01. This link tracks `main`, not a pinned revision. This node implements its own Krea2 flow-velocity objective; the reference is not evidence of equivalent model behavior or image quality.
