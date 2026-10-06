"""Derive a spatial area from a mask without ComfyUI's 2D-only AABB path."""
import torch


def _latent_span(pixel_start, pixel_end, pixel_length, latent_length):
    start = pixel_start * latent_length // pixel_length
    end = (pixel_end * latent_length + pixel_length - 1) // pixel_length
    size = min(latent_length, max(8, end - start))
    start = min(start, latent_length - size)
    return start, size


def latent_area_from_mask(mask, latent):
    """Return (height, width, y, x) in latent units for 4D or 5D latents."""
    if not isinstance(mask, torch.Tensor) or mask.ndim != 3:
        raise ValueError("Region area requires a batched MASK with shape (B, H, W)")
    samples = latent.get("samples") if isinstance(latent, dict) else None
    if not isinstance(samples, torch.Tensor) or samples.ndim not in (4, 5):
        raise ValueError("Region area requires a 4D or 5D LATENT")
    pixel_h, pixel_w = mask.shape[-2:]
    latent_h, latent_w = samples.shape[-2:]
    if min(pixel_h, pixel_w, latent_h, latent_w) < 1:
        raise ValueError("Region area requires positive mask and latent dimensions")

    active = mask.detach().reshape(-1, pixel_h, pixel_w).to(device="cpu").gt(0).any(dim=0)
    ys = torch.where(active.any(dim=1))[0]
    xs = torch.where(active.any(dim=0))[0]
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Region area MASK contains no nonzero pixels")

    y, height = _latent_span(int(ys[0]), int(ys[-1]) + 1, pixel_h, latent_h)
    x, width = _latent_span(int(xs[0]), int(xs[-1]) + 1, pixel_w, latent_w)
    return height, width, y, x


def apply_area_to_conditioning_pair(positive, negative, mask, latent):
    """Copy only conditioning metadata; keep prompt tensors, masks, and Hooks."""
    area = latent_area_from_mask(mask, latent)
    reference_mask = mask.detach().to(device="cpu")

    def add_area(conditioning):
        result = []
        for tensor, metadata in conditioning:
            attached_mask = metadata.get("mask")
            if (not isinstance(attached_mask, torch.Tensor) or
                    attached_mask.shape != mask.shape or
                    not torch.equal(attached_mask.detach().to(device="cpu"), reference_mask)):
                raise ValueError("Region area and CONDITIONING must use the same MASK")
            values = metadata.copy()
            values["area"] = area
            values["set_area_to_bounds"] = False
            result.append([tensor, values])
        return result

    return add_area(positive), add_area(negative)
