"""Two editable rectangular masks, adapted from D2 Create Masks (MIT).

Upstream: da2el-ai/D2-nodes-ComfyUI, 0d967672ef72fcc92fe8d915fd4f20c448134aa3.
Copyright (c) 2023 Shingo.T. See licenses/D2-MIT.txt.
"""
import json
import math

import torch


MIN_REGION_SIZE = 0.01
MAX_DIMENSION = 8192
DEFAULT_REGIONS = '[{"x":0,"y":0,"w":0.5,"h":1},{"x":0.5,"y":0,"w":0.5,"h":1}]'


def parse_regions(regions_json):
    """Clamp two normalized rectangles; reject corrupt saved targets."""
    try:
        parsed = json.loads(regions_json)
    except (TypeError, ValueError) as exc:
        raise ValueError('Region rectangles must be valid JSON') from exc
    if parsed == []:
        parsed = json.loads(DEFAULT_REGIONS)
    if not isinstance(parsed, list) or len(parsed) != 2:
        raise ValueError('Region rectangles must contain exactly two objects')
    regions = []
    for rect in parsed:
        if not isinstance(rect, dict):
            raise ValueError('Each region rectangle must have numeric x, y, w, h')
        values = [rect.get(key) for key in ('x', 'y', 'w', 'h')]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            raise ValueError('Each region rectangle must have finite numeric x, y, w, h')
        x, y, w, h = values
        w, h = max(MIN_REGION_SIZE, min(1.0, w)), max(MIN_REGION_SIZE, min(1.0, h))
        regions.append({'x': max(0.0, min(1.0 - w, x)), 'y': max(0.0, min(1.0 - h, y)), 'w': w, 'h': h})
    return regions


def _axis_to_pixels(start, size, length):
    # D2's half-open raster bounds retain at least one pixel for tiny rectangles.
    p0 = min(max(round(start * length), 0), length)
    p1 = min(max(round((start + size) * length), 0), length)
    if p1 <= p0:
        p0 = min(p0, length - 1)
        p1 = p0 + 1
    return p0, p1


def create_region_masks(width, height, regions_json=DEFAULT_REGIONS):
    for name, value in (('width', width), ('height', height)):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_DIMENSION:
            raise ValueError(f'{name} must be an integer between 1 and {MAX_DIMENSION}')
    masks = []
    for rect in parse_regions(regions_json):
        x0, x1 = _axis_to_pixels(rect['x'], rect['w'], width)
        y0, y1 = _axis_to_pixels(rect['y'], rect['h'], height)
        mask = torch.zeros((1, height, width), dtype=torch.float32)
        mask[:, y0:y1, x0:x1] = 1.0
        masks.append(mask)
    return tuple(masks)
