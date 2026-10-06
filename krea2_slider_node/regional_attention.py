"""Directed regional attention; adapted from januspluto/Krea2-Regional (MIT).

The base text and background do not relay subject information in strict mode.
See licenses/Krea2-Regional-MIT.txt and THIRD_PARTY_NOTICES.md.
"""
import torch
from torch.nn import functional as F


def build_region_owners(masks, token_hw):
    """Map each image token to one region; the last owner is background."""
    if not 1 <= len(masks) <= 4:
        raise ValueError('Regional attention requires 1–4 regions')
    h, w = token_hw
    if h < 1 or w < 1:
        raise ValueError('Token grid dimensions must be positive')
    owners = torch.full((h * w,), len(masks), dtype=torch.int64)
    pixel_hw = None
    for index, mask in enumerate(masks):
        if not isinstance(mask, torch.Tensor) or mask.ndim not in (2, 3):
            raise ValueError('Regional MASK must have shape (H,W) or (1,H,W)')
        if mask.ndim == 3 and mask.shape[0] != 1:
            raise ValueError('Regional masks must contain a single shared batch')
        if min(mask.shape[-2:]) < 1 or not torch.isfinite(mask).all():
            raise ValueError('Regional masks must be nonempty and finite')
        if mask.min() < 0 or mask.max() > 1:
            raise ValueError('Regional mask values must be in [0,1]')
        if pixel_hw is None:
            pixel_hw = mask.shape[-2:]
        elif mask.shape[-2:] != pixel_hw:
            raise ValueError('Regional masks must use the same canvas dimensions')
        reduced = F.interpolate(mask.detach().cpu().float().reshape(1, 1, *pixel_hw),
                                size=(h, w), mode='area').flatten()
        selected = reduced >= 0.5
        selected &= owners == len(masks)
        if not selected.any():
            raise ValueError(f'Region {index + 1} is empty, too small, or fully covered by an earlier region')
        owners[selected] = index
    return owners


def build_attention_masks(segments, owners, isolation='strict'):
    """Build query→key permissions for base, subjects, background, then images."""
    if len(segments) < 3:
        raise ValueError('Base, region, and background text segments are required')
    offset = 0
    for start, end in segments:
        if start != offset or end <= start:
            raise ValueError('Prompt segments must be contiguous and nonempty')
        offset = end
    if owners.ndim != 1 or owners.numel() == 0:
        raise ValueError('Image token owners must be a nonempty vector')
    if owners.min() < 0 or owners.max() > len(segments) - 2:
        raise ValueError('Token owner does not match the prompt segments')
    if isolation not in ('strict', '_unrestricted'):
        raise ValueError('Only strict regional attention is implemented in phase A')
    text_len = offset
    total = text_len + owners.numel()
    allow = torch.zeros((total, total), dtype=torch.bool, device=owners.device)
    if isolation == '_unrestricted':  # CPU integration equivalence check; never exposed as a node mode
        allow.fill_(True)
        return allow, allow[:text_len, :text_len].clone()
    base_start, base_end = segments[0]
    allow[base_start:base_end, base_start:base_end] = True
    for index, (start, end) in enumerate(segments[1:]):
        image_rows = torch.where(owners == index)[0] + text_len
        allow[start:end, base_start:base_end] = True
        allow[start:end, start:end] = True
        allow[start:end, image_rows] = True
        allow[image_rows, base_start:base_end] = True
        allow[image_rows, start:end] = True
        allow[image_rows[:, None], image_rows[None, :]] = True
    return allow, allow[:text_len, :text_len].clone()
