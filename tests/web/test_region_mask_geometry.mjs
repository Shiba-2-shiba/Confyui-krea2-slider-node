import assert from 'node:assert/strict';
import { test } from 'node:test';

async function api() {
    const module = await import('../../web/region_mask_geometry.js').catch(() => null);
    assert.ok(module, 'Region mask editor geometry is missing');
    return module;
}

test('default and saved pairs survive serialization without losing the edited positions', async () => {
    const { parseRegions, serializeRegions } = await api();
    assert.deepEqual(parseRegions('[]'), [
        { x: 0, y: 0, w: .5, h: 1 }, { x: .5, y: 0, w: .5, h: 1 },
    ]);
    const edited = [{ x: .1, y: .2, w: .3, h: .4 }, { x: .6, y: .1, w: .3, h: .8 }];
    assert.deepEqual(parseRegions(serializeRegions(edited)), edited);
    assert.throws(() => parseRegions('{'), /JSON|region/i);
    assert.throws(() => parseRegions('[null,null]'), /rectangle|region/i);
    assert.throws(() => parseRegions('[{"x":null,"y":0,"w":1,"h":1},{}]'), /rectangle|region/i);
});

test('dragging beyond the canvas retains the rectangle size', async () => {
    const { moveRegion } = await api();
    assert.deepEqual(moveRegion({ x: .1, y: .2, w: .3, h: .4 }, 2, -1),
        { x: .7, y: 0, w: .3, h: .4 });
});

test('resizing across the opposite corner flips the rectangle and keeps it inside', async () => {
    const { resizeRegion } = await api();
    assert.deepEqual(resizeRegion({ x: .25, y: .25, w: .25, h: .25 }, 0, .5, .5),
        { x: .5, y: .5, w: .25, h: .25 });
    const collapsed = resizeRegion({ x: 0, y: 0, w: .5, h: 1 }, 3, -.5, -1);
    assert.equal(collapsed.w, .01);
    assert.equal(collapsed.h, .01);
});

test('corner handles take priority and selected overlapping rectangle is picked first', async () => {
    const { hitTest } = await api();
    const rectangles = [{ x: .1, y: .1, w: .5, h: .5 }, { x: .2, y: .2, w: .5, h: .5 }];
    assert.deepEqual(hitTest(rectangles, 0, .1, .1, .02, .02), { index: 0, corner: 0 });
    assert.deepEqual(hitTest(rectangles, 1, .4, .4, .02, .02), { index: 1, corner: -1 });
    assert.equal(hitTest(rectangles, 0, .9, .9, .02, .02), null);
});
