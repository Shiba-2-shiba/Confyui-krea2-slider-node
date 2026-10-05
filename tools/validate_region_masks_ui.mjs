// Browser integration check without starting ComfyUI or loading a diffusion model.
// Usage: node tools/validate_region_masks_ui.mjs <path-to-@playwright/test>
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile, mkdir } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const { chromium } = createRequire(import.meta.url)(process.argv[2] || '@playwright/test');
const server = createServer(async (request, response) => {
    try {
        const path = new URL(request.url, 'http://localhost').pathname;
        if (path === '/') {
            response.setHeader('Content-Type', 'text/html');
            response.end(`<!doctype html><html><body style="background:#252525;color:white">
            <main id="fixture" style="width:380px"></main><script type="module">
            import { bindRegionMaskEditor } from '/web/region_mask_editor.js';
            window.bindRegionMaskEditor = bindRegionMaskEditor;
            window.createNode = (saved) => {
                const node = {
                    comfyClass: 'Krea2RegionMasks', size: [380, 400],
                    widgets: [{name:'width',value:1024}, {name:'height',value:1024},
                              {name:'regions',value:saved || '[]'}],
                    setDirtyCanvas() {},
                    addDOMWidget(name, type, element, options) {
                        document.getElementById('fixture').append(element);
                        const widget = {name,type,element,options}; this.widgets.push(widget); return widget;
                    },
                    onRemoved() { this.removed = true; },
                };
                // ComfyUI DOM widgets invoke their callback when value is assigned.
                const geometry = node.widgets.find((widget) => widget.name === 'regions');
                let geometryValue = geometry.value;
                Object.defineProperty(geometry, 'value', {
                    get() { return geometryValue; },
                    set(value) { geometryValue = value; geometry.callback?.(value); },
                });
                bindRegionMaskEditor(node);
                return node;
            };
            window.node = createNode(); window.ready = true;
            </script></body></html>`);
        } else if (/^\/web\/[a-z_]+\.js$/.test(path)) {
            response.setHeader('Content-Type', 'text/javascript');
            response.end(await readFile(resolve(root, '.' + path)));
        } else { response.writeHead(404); response.end(); }
    } catch { response.writeHead(404); response.end(); }
});
await new Promise((accept) => server.listen(0, '127.0.0.1', accept));
let browser;
try {
    browser = await chromium.launch({ channel: 'chrome', headless: true });
    const page = await browser.newPage();
    const errors = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await page.goto(`http://127.0.0.1:${server.address().port}`);
    await page.waitForFunction(() => window.ready, null, { timeout: 5000 });
    const canvas = page.locator('canvas');
    assert.equal(await canvas.count(), 1);
    await page.evaluate(() => { bindRegionMaskEditor(node); bindRegionMaskEditor(node); });
    assert.equal(await canvas.count(), 1, 'Rebinding on graph load must not duplicate the editor');
    const saved = '[{"x":0.1,"y":0.2,"w":0.3,"h":0.4},{"x":0.6,"y":0.1,"w":0.3,"h":0.8}]';
    await page.evaluate((value) => {
        const widget = node.widgets.find((w) => w.name === 'regions');
        widget.value = value; bindRegionMaskEditor(node);
    }, saved);
    const rect = await canvas.boundingBox();
    const point = (x, y) => ({
        x: rect.x + rect.width * (16 + x * 480) / 512,
        y: rect.y + rect.height * (16 + y * 480) / 512,
    });
    const start = point(.25, .4), end = point(.35, .5);
    await page.mouse.move(start.x, start.y); await page.mouse.down();
    await page.mouse.move(end.x, end.y, { steps: 5 }); await page.mouse.up();
    let pair = await page.evaluate(() => JSON.parse(node.widgets.find((w) => w.name === 'regions').value));
    assert.ok(Math.abs(pair[0].x - .2) < .005 && Math.abs(pair[0].y - .3) < .005, 'Dragging must update saved geometry');
    assert.equal(pair[0].w, .3); assert.equal(pair[0].h, .4);
    const corner = point(.5, .7), grown = point(.55, .8);
    await page.mouse.move(corner.x, corner.y); await page.mouse.down();
    await page.mouse.move(grown.x, grown.y, { steps: 5 }); await page.mouse.up();
    pair = await page.evaluate(() => JSON.parse(node.widgets.find((w) => w.name === 'regions').value));
    assert.ok(Math.abs(pair[0].w - .35) < .005 && Math.abs(pair[0].h - .5) < .005, 'Corner drag must resize the saved mask');
    const serialized = await page.evaluate(() => node.widgets.find((w) => w.name === 'regions').value);
    await page.evaluate((value) => {
        node.onRemoved(); node.widgets.find((w) => w.name === 'region_editor').element.remove();
        window.node = createNode(value);
    }, serialized);
    assert.equal(await canvas.count(), 1);
    assert.equal(await page.evaluate(() => node.widgets.find((w) => w.name === 'regions').value), serialized);
    // A restored non-square resolution keeps saved geometry and changes canvas aspect.
    await page.evaluate(() => {
        node.widgets.find((w) => w.name === 'height').value = 512; bindRegionMaskEditor(node);
    });
    const wide = await canvas.boundingBox();
    assert.ok(wide.width / wide.height > 1.7);
    assert.equal(await page.evaluate(() => node.widgets.find((w) => w.name === 'regions').value), serialized);
    // Editing is disabled when an external socket owns the JSON input.
    await page.evaluate(() => { node.inputs = [{name:'regions',link:1}]; bindRegionMaskEditor(node); });
    assert.equal(await page.getByRole('button', { name: 'Reset left / right' }).isDisabled(), true);
    await page.evaluate(() => { node.inputs = []; bindRegionMaskEditor(node); });
    await page.getByRole('button', { name: 'Reset left / right' }).click();
    pair = await page.evaluate(() => JSON.parse(node.widgets.find((w) => w.name === 'regions').value));
    assert.deepEqual(pair, [{x:0,y:0,w:.5,h:1},{x:.5,y:0,w:.5,h:1}]);
    // Invalid saved JSON must remain visible and must not be replaced silently.
    await page.evaluate(() => {
        node.widgets.find((w) => w.name === 'regions').value = '{'; bindRegionMaskEditor(node);
    });
    assert.match(await page.locator('[role="status"]').textContent(), /JSON/i);
    assert.equal(await page.evaluate(() => node.widgets.find((w) => w.name === 'regions').value), '{');
    await page.getByRole('button', { name: 'Reset left / right' }).click();
    await mkdir(resolve(root, 'test-results/region-masks'), { recursive: true });
    await page.screenshot({ path: resolve(root, 'test-results/region-masks/editor.png') });
    assert.deepEqual(errors, []);
    console.log(JSON.stringify({status:'passed', checks:['DOM binding','drag','corner resize','save/reload',
        'resolution change','linked input','invalid JSON','reset'], screenshot:'test-results/region-masks/editor.png'}));
} finally {
    await browser?.close();
    await new Promise((accept) => server.close(accept));
}
