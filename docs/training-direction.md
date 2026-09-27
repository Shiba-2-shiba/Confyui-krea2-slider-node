# Training direction

The default is now `training_direction=single`, following the reference MageFlow node's single `+1` student objective and the user's experience with age sliders. The initial Krea2 implementation always added a `-1` student objective; that was an additional design choice, not a requirement of the reference YAMLs.

For one latent/timestep, let `base`, `positive`, and `compare` be frozen-model predictions for the target, positive, and negative text:

```text
d = positive - compare
single:
    teacher_plus = stop_gradient(base + eta_eff * d)
    loss = MSE(student(+1), teacher_plus)

bidirectional (explicit opt-in):
    teacher_minus = stop_gradient(base - eta_eff * d)
    loss = 0.5 * MSE(student(+1), teacher_plus)
         + 0.5 * MSE(student(-1), teacher_minus)
```

Optional teacher normalization is applied after forming each selected teacher. Both modes perform one optimizer step. Single-direction loss uses the full MSE, not half of it.

The two directions require respectively one or two student forward/backward passes. Trajectory rollout, teacher predictions, and the optimizer update are shared work, so bidirectional does not imply exactly twice the total runtime. Identical conditioning tensors now share device copies and teacher predictions. The supplied YAMLs have `target = negative = neutral`, so their ordinary teacher stage needs two unique predictions rather than three (also two for neutral normalization).

The aging YAML describes a fully clothed adult baseline and the same person with aged facial/neck/hand skin. It does not contain a toddler comparison and does not import the separate deaging YAML. In the previous bidirectional mode, the inverse target was a mathematical extrapolation `2 * baseline - aging` at eta=1 without normalization. Single mode no longer constructs or trains that inverse target.

MageFlow evidence: `C:/ComfyUI/custom_nodes/Comfyui-mageflow-slider-node/mageflow_slider/trainer.py`, teacher prediction caching at lines 99–108 and student multiplier +1 at lines 111–114. MageFlow uses denoised predictions; this Krea2 trainer uses raw flow velocity. Matching the training direction does not make all other training behavior or image quality identical.

Tests observe actual root-model forwards and adapter gradient hooks: default single uses only multiplier +1 and one backward; explicit bidirectional uses +1/-1 and two backwards. Shared target/negative features produce one trajectory forward plus two teacher forwards in the fixed two-step test. Prompt text, rank/alpha, learning rate and source model handling are unchanged.

The old 100-step acceptance reports describe the earlier bidirectional implementation. Image-quality improvement from this correction is not yet established. Negative inference strengths remain possible with a single-direction adapter, but the reverse effect has no separately trained target.

## Bounded timing check

Measured through the actual ComfyUI API on the R9700, using the native RAW/CLIP inputs, `aging_slider_fullbody.yaml`, 512px, rank/alpha 8/8, attention targets, 16 CPU blocks, checkpointing, trajectory_steps=8 and seeds 42–44. Both modes use the same teacher-reuse optimization. Each run contains three steps; the comparison below excludes its first step and averages steps 2–3. These are short-run measurements, not a general throughput guarantee.

| Mode | Student backwards/step | Mean seconds/step after first | Peak torch reserved |
|---|---:|---:|---:|
| single | 1 | 38.07 | 8.125GiB |
| bidirectional | 2 | 51.62 | 8.125GiB |

For this comparison, bidirectional took 1.36× as long per step; single reduced time by 26.2%. Both completed and saved adapters, each used two unique teacher evaluations per step, and both returned to the same post-job memory counters. This validates direction selection, gradient execution, saving and measured runtime; it does not compare image quality.

Evidence: `test-results/direction-comparison.json`, the two referenced training reports, and `test-results/direction-{single,bidirectional}-api.json`.
