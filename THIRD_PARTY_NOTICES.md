# Third-party sources

`krea2_slider_node/model.py` is adapted from the Krea AI Krea2 official inference code, commit `db3984fbc6e13b34c0064990fc2d95ac64d00058`, licensed under Apache-2.0. The license is retained in `licenses/Krea2-Apache-2.0.txt`. Modifications remove forced cuDNN/compilation, add portable SDPA, checkpointing and a latent interface. Model weights have their own license and are not distributed here.

ConvRot's regular Hadamard convention and frozen input-gradient formulation follow Musubi Tuner `4e7c7149249e7715e9168920feb4c420423abba7` and its Apache-2.0 Comfy Kitchen-derived kernels. Upstream notices: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. Copyright (c) 2025 Comfy Org. This implementation uses eager PyTorch operations; it does not include the upstream Triton kernels. Apache-2.0 license text is retained in `licenses/Krea2-Apache-2.0.txt`.

The Slider objective is based on the Concept Sliders method (Gandikota et al., ECCV 2024) and the local Anima Slider behavior, adapted to Krea2 flow velocity. No model weights or third-party datasets are bundled.
