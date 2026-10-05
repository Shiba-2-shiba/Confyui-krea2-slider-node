// Adapted from D2 Create Masks, commit 0d967672ef72fcc92fe8d915fd4f20c448134aa3.
// Copyright (c) 2023 Shingo.T. MIT license: licenses/D2-MIT.txt.
export const DEFAULT_REGIONS = '[{"x":0,"y":0,"w":0.5,"h":1},{"x":0.5,"y":0,"w":0.5,"h":1}]';
const MIN_SIZE = .01;
const corners = [[0, 0], [1, 0], [0, 1], [1, 1]];
const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

export function parseRegions(value) {
    let parsed;
    try { parsed = JSON.parse(value); } catch { throw new Error('Region rectangles must be valid JSON'); }
    if (Array.isArray(parsed) && parsed.length === 0) parsed = JSON.parse(DEFAULT_REGIONS);
    if (!Array.isArray(parsed) || parsed.length !== 2) throw new Error('Exactly two region rectangles are required');
    return parsed.map((rect) => {
        if (!rect || ['x', 'y', 'w', 'h'].some((key) => typeof rect[key] !== 'number' || !Number.isFinite(rect[key]))) {
            throw new Error('Each region rectangle needs finite numeric x, y, w, h');
        }
        const w = clamp(rect.w, MIN_SIZE, 1), h = clamp(rect.h, MIN_SIZE, 1);
        return { x: clamp(rect.x, 0, 1 - w), y: clamp(rect.y, 0, 1 - h), w, h };
    });
}

export function serializeRegions(regions) {
    return JSON.stringify(regions.map((rect) => Object.fromEntries(
        Object.entries(rect).map(([key, value]) => [key, Number(value.toFixed(6))]),
    )));
}

export function moveRegion(rect, dx, dy) {
    return { x: clamp(rect.x + dx, 0, 1 - rect.w), y: clamp(rect.y + dy, 0, 1 - rect.h), w: rect.w, h: rect.h };
}

function resizeAxis(fixed, moving) {
    moving = clamp(moving, 0, 1);
    let start = Math.min(fixed, moving), size = Math.abs(moving - fixed);
    if (size < MIN_SIZE) {
        size = MIN_SIZE;
        start = clamp(moving >= fixed ? fixed : fixed - size, 0, 1 - size);
    }
    return { start, size };
}

export function resizeRegion(rect, corner, dx, dy) {
    const [right, bottom] = corners[corner];
    const x = resizeAxis(rect.x + (right ? 0 : rect.w), rect.x + (right ? rect.w : 0) + dx);
    const y = resizeAxis(rect.y + (bottom ? 0 : rect.h), rect.y + (bottom ? rect.h : 0) + dy);
    return { x: x.start, y: y.start, w: x.size, h: y.size };
}

export function hitTest(regions, active, x, y, radiusX, radiusY) {
    const rect = regions[active];
    for (let corner = 0; corner < corners.length; corner++) {
        const [right, bottom] = corners[corner];
        const dx = (x - rect.x - right * rect.w) / radiusX;
        const dy = (y - rect.y - bottom * rect.h) / radiusY;
        if (dx * dx + dy * dy <= 1) return { index: active, corner };
    }
    for (const index of [active, 1 - active]) {
        const r = regions[index];
        if (x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h) return { index, corner: -1 };
    }
    return null;
}
