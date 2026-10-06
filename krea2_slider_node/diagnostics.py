"""Small, metadata-only summaries for regional Hook troubleshooting."""
import hashlib
import json
import logging
import secrets
import threading
import time

import torch


LOGGER = logging.getLogger(__name__)
LOG_PREFIX = "[Krea2HookDebug]"
_TOKEN_KEY = secrets.token_bytes(16)


def object_token(value):
    """Correlate objects within this process without exposing raw addresses."""
    if value is None:
        return None
    digest = hashlib.blake2b(str(id(value)).encode("ascii"), key=_TOKEN_KEY,
                             digest_size=6).hexdigest()
    return "obj_" + digest


def diagnostic_fingerprint(enabled):
    """Force a fresh diagnostic only when logging is enabled."""
    return "enabled:" + str(time.time_ns()) if enabled else "disabled"


def describe_hooks(group):
    if group is None:
        return {"present": False, "group_id": None, "count": 0, "items": []}
    hooks = getattr(group, "hooks", ())
    items = []
    for hook in hooks:
        kind = getattr(hook, "hook_type", None)
        raw_strength = getattr(hook, "_strength_model", None)
        items.append({
            "type": getattr(kind, "value", type(hook).__name__),
            "strength_model": float(raw_strength) if raw_strength is not None else None,
            "hook_ref_id": object_token(getattr(hook, "hook_ref", None)),
        })
    return {"present": True, "group_id": object_token(group), "count": len(items), "items": items}


def _mask_summary(mask):
    if not isinstance(mask, torch.Tensor) or mask.ndim < 2:
        return None
    height, width = mask.shape[-2:]
    active = mask.detach().reshape(-1, height, width).to(device="cpu") > 0
    ys = torch.where(active.any(dim=(0, 2)))[0]
    xs = torch.where(active.any(dim=(0, 1)))[0]
    bounds = None
    if len(xs) and len(ys):
        bounds = [int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1]
    return {"shape": list(mask.shape), "bounds_xyxy": bounds,
            "coverage": round(float(active.float().mean()), 6)}


def _side_summary(conditioning):
    entries = []
    union = None
    incompatible_masks = False
    for index, item in enumerate(conditioning):
        metadata = item[1] if isinstance(item, (list, tuple)) and len(item) > 1 and isinstance(item[1], dict) else {}
        mask = metadata.get("mask")
        summary = _mask_summary(mask)
        if summary is not None:
            height, width = mask.shape[-2:]
            active = mask.detach().reshape(-1, height, width).to(device="cpu").gt(0).any(dim=0)
            if not incompatible_masks:
                if union is None:
                    union = active
                elif union.shape == active.shape:
                    union |= active
                else:
                    union = None
                    incompatible_masks = True
        entries.append({
            "index": index,
            "default": bool(metadata.get("default", False)),
            "mask": summary,
            "mask_strength": metadata.get("mask_strength"),
            "set_area_to_bounds": bool(metadata.get("set_area_to_bounds", False)),
            "hooks": describe_hooks(metadata.get("hooks")),
        })
    uncovered = None if union is None or incompatible_masks else round(1.0 - float(union.float().mean()), 6)
    return {"count": len(entries), "entries": entries, "uncovered_fraction": uncovered}


def summarize_conditioning(positive, negative, label=""):
    """Inspect mask and Hook metadata without reading prompt embeddings or weights."""
    return {"event": "conditioning", "label": label,
            "positive": _side_summary(positive), "negative": _side_summary(negative)}


def emit_debug(event):
    """Write one searchable JSON line to the ComfyUI process log."""
    event = {"thread_id": threading.get_ident(), **event}
    LOGGER.warning("%s %s", LOG_PREFIX, json.dumps(event, ensure_ascii=False, separators=(",", ":")))
