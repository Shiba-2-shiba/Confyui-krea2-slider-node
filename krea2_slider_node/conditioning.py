from dataclasses import dataclass
import json

import torch


@dataclass(frozen=True)
class TextCondition:
    features: torch.Tensor


@dataclass(frozen=True)
class PromptRecord:
    conditions: dict[str, TextCondition]
    prompts: dict[str, str]


def parse_prompt_records(text):
    try:
        records = json.loads(text)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("Prompt records must be a JSON list") from error
    return validate_prompt_records(records)


def validate_prompt_records(records):
    if not isinstance(records, list) or not 1 <= len(records) <= 64:
        raise ValueError("Provide 1..64 prompt records in a JSON list")
    for record in records:
        if not isinstance(record, dict) or not all(isinstance(record.get(role), str) for role in ("target", "positive", "negative")):
            raise ValueError("Every record requires target, positive and negative strings")
        if set(record) - {"target", "positive", "negative", "neutral"}:
            raise ValueError("Unknown prompt record field")
        if record["positive"] == record["negative"]:
            raise ValueError("Positive and negative concepts must differ")
        if any(not isinstance(value, str) or len(value) > 8192 for value in record.values()):
            raise ValueError("Each prompt must be text of at most 8192 characters")
    return records


def from_comfy_conditioning(packed, attention_mask=None):
    if packed.ndim != 3 or packed.shape[0] != 1 or packed.shape[-1] != 12 * 2560:
        raise ValueError("Use a Krea2 Qwen3-VL-4B CLIP: expected (1, tokens, 30720) conditioning")
    if not torch.isfinite(packed).all():
        raise ValueError("Text conditioning contains NaN or infinity")
    if attention_mask is None:
        valid = torch.ones(packed.shape[1], dtype=torch.bool, device=packed.device)
    else:
        if tuple(attention_mask.shape) != tuple(packed.shape[:2]):
            raise ValueError("Attention mask shape does not match conditioning")
        valid = attention_mask[0].to(device=packed.device, dtype=torch.bool)
    if not valid.any():
        raise ValueError("Text conditioning has no valid tokens")
    # Remove padding before attention; each microbatch is one prompt. This lets
    # SDPA use an unmasked efficient kernel instead of a quadratic padded mask.
    with torch.inference_mode(False):
        features = packed[:, valid].detach().to(device="cpu", dtype=torch.bfloat16).clone()
        features = features.reshape(1, -1, 12, 2560)
    if features.shape[1] > 512:
        raise ValueError("Prompt exceeds the 512-token training limit; shorten it")
    return TextCondition(features)


def encode_prompt_records(clip, specifications, cancel=None, progress=None):
    cache = {}
    output = []
    with torch.inference_mode(False), torch.no_grad():
        for index, spec in enumerate(specifications):
            conditions = {}
            for role in ("target", "positive", "negative", "neutral"):
                if cancel:
                    cancel()
                prompt = spec.get(role, spec["target"])
                if prompt not in cache:
                    tokens = clip.tokenize(prompt)
                    encoded = clip.encode_from_tokens(tokens, return_dict=True)
                    cache[prompt] = from_comfy_conditioning(encoded["cond"], encoded.get("attention_mask"))
                conditions[role] = cache[prompt]
            output.append(PromptRecord(conditions, dict(spec)))
            if progress:
                progress(index + 1, len(specifications), "Encoding Slider prompts")
    return output
