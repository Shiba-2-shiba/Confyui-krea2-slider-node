from dataclasses import dataclass
import math
from numbers import Real


@dataclass(frozen=True)
class ModelConfig:
    path: str
    source_kind: str = "raw"
    quantization: str = "convrot_int8"
    blocks_to_swap: int = 16
    memory_budget_gib: float = 14.0
    compute_dtype: str = "bf16"

    def validate(self):
        if self.source_kind != "raw":
            raise ValueError("Krea2 Slider training requires a RAW base model")
        if self.quantization not in ("convrot_int8", "bf16_reference"):
            raise ValueError("Unsupported quantization")
        if not 0 <= self.blocks_to_swap <= 28:
            raise ValueError("blocks_to_swap must be in 0..28")
        if not math.isfinite(self.memory_budget_gib) or self.memory_budget_gib < 1:
            raise ValueError("VRAM budget must be finite and at least 1 GiB")
        if self.compute_dtype not in ("bf16", "fp16"):
            raise ValueError("Compute dtype must be bf16 or fp16")


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 100
    rank: int = 8
    alpha: float = 8.0
    target: str = "attention"
    learning_rate: float = 1e-4
    width: int = 512
    height: int = 512
    trajectory_steps: int = 8
    eta: float = 1.0
    teacher_guidance_scale: float = 1.0
    teacher_norm_reference: str = "none"
    seed: int = 42
    vary_seed: bool = True
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0
    training_direction: str = "single"
    anchor_strength: float = 1.0

    def validate(self):
        if self.steps < 1 or not 1 <= self.rank <= 128 or self.trajectory_steps < 2:
            raise ValueError("Positive steps, rank 1..128 and at least two trajectory steps are required")
        if self.target not in ("attention", "all"):
            raise ValueError("Unknown LoRA target")
        if self.training_direction not in ("single", "bidirectional"):
            raise ValueError("Training direction must be single or bidirectional")
        if any(v < 16 or v % 16 for v in (self.width, self.height)):
            raise ValueError("Width and height must be positive multiples of 16")
        if not 0 <= self.seed < 2**63:
            raise ValueError("Seed must be in 0..2**63-1")
        if self.teacher_norm_reference not in ("none", "positive", "neutral"):
            raise ValueError("Unknown teacher norm reference")
        if (isinstance(self.anchor_strength, bool) or not isinstance(self.anchor_strength, Real)
                or not math.isfinite(self.anchor_strength) or self.anchor_strength < 0):
            raise ValueError("anchor_strength must be finite and nonnegative")
        for name in ("alpha", "learning_rate", "eta", "max_grad_norm", "teacher_guidance_scale"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
