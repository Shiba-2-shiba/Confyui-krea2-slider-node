import { DEFAULT_REGIONS, parseRegions, serializeRegions, moveRegion, resizeRegion, hitTest } from './region_mask_geometry.js';

const bindings = new WeakMap();
const colors = ['#5bc5f4', '#ffa85c'];
const margin = 16;

export function bindRegionMaskEditor(node) {
    if (node.comfyClass !== 'Krea2RegionMasks' || typeof node.addDOMWidget !== 'function') return;
    let state = bindings.get(node);
    if (state) { state.refresh(); return; }
    const geometry = node.widgets?.find((widget) => widget.name === 'regions');
    if (!geometry) return;
    const element = document.createElement('div');
    element.style.cssText = 'display:flex;flex-direction:column;gap:6px;padding:6px;box-sizing:border-box;width:100%;height:100%;font:12px sans-serif;color:#eee;background:#252525;';
    const toolbar = document.createElement('div');
    toolbar.style.cssText = 'display:flex;gap:6px;flex-wrap:wrap;flex-shrink:0;';
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.style.cssText = 'display:block;width:100%;min-height:0;flex:1;object-fit:contain;touch-action:none;cursor:crosshair;';
    canvas.setAttribute('aria-label', 'Region masks: drag a rectangle to move, drag a corner to resize');
    const status = document.createElement('div');
    status.setAttribute('role', 'status');
    status.style.cssText = 'min-height:28px;flex-shrink:0;overflow-wrap:anywhere;';
    element.append(toolbar, canvas, status);
    state = { active: 0, regions: null, drag: null, disabled: false, saving: false, refresh };
    bindings.set(node, state);
    const buttons = [];
    const button = (text, action) => {
        const control = document.createElement('button');
        control.type = 'button'; control.textContent = text;
        control.style.cssText = 'padding:4px 7px;color:#eee;background:#383838;border:1px solid #777;border-radius:4px;cursor:pointer;';
        control.addEventListener('click', (event) => { event.stopPropagation(); action(); });
        toolbar.append(control); buttons.push(control);
        return control;
    };
    const selectors = [0, 1].map((index) => button(`Region ${index + 1}`, () => {
        state.active = index; draw();
    }));
    button('Reset left / right', () => {
        finishDrag();
        state.regions = parseRegions(DEFAULT_REGIONS); state.active = 0;
        save(); refresh();
    });

    function dimensions() {
        const number = (name) => Number(node.widgets.find((widget) => widget.name === name)?.value) || 1024;
        return { width: number('width'), height: number('height') };
    }

    function isLinked() {
        return geometry.type === 'converted-widget' || node.inputs?.some((input) => input.name === 'regions' && input.link != null);
    }

    function save() {
        // DOMWidget's value setter invokes its callback synchronously. An editor
        // update must not refresh and release pointer capture mid-drag.
        state.saving = true;
        try { geometry.value = serializeRegions(state.regions); }
        finally { state.saving = false; }
        node.setDirtyCanvas?.(true, true);
    }

    function refresh() {
        // A graph-load hook runs after standard widget values and connections are restored.
        finishDrag();
        state.disabled = Boolean(isLinked());
        buttons.forEach((control) => { control.disabled = state.disabled; });
        const { width, height } = dimensions();
        canvas.height = Math.round(Math.max(80, Math.min(1440, 480 * height / width))) + margin * 2;
        try {
            state.regions = parseRegions(geometry.value);
            status.textContent = state.disabled ? 'Regions are supplied by an external input.' : 'Drag to move · drag corners to resize';
        } catch (error) {
            state.regions = null; status.textContent = error.message + ' — use Reset left / right to recover.';
        }
        draw();
        node.setDirtyCanvas?.(true, true);
    }

    function draw() {
        const ctx = canvas.getContext('2d');
        const width = canvas.width - margin * 2, height = canvas.height - margin * 2;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#151515'; ctx.fillRect(margin, margin, width, height);
        ctx.strokeStyle = '#414141'; ctx.lineWidth = 1;
        for (const fraction of [.25, .5, .75]) {
            ctx.beginPath(); ctx.moveTo(margin + width * fraction, margin); ctx.lineTo(margin + width * fraction, margin + height);
            ctx.moveTo(margin, margin + height * fraction); ctx.lineTo(margin + width, margin + height * fraction); ctx.stroke();
        }
        selectors.forEach((control, index) => {
            control.setAttribute('aria-pressed', String(index === state.active));
            control.style.borderColor = index === state.active ? colors[index] : '#777';
        });
        if (!state.regions) return;
        for (const index of [1 - state.active, state.active]) {
            const r = state.regions[index];
            const x = margin + r.x * width, y = margin + r.y * height, w = r.w * width, h = r.h * height;
            ctx.globalAlpha = .25; ctx.fillStyle = colors[index]; ctx.fillRect(x, y, w, h); ctx.globalAlpha = 1;
            ctx.strokeStyle = colors[index]; ctx.lineWidth = index === state.active ? 3 : 1; ctx.strokeRect(x, y, w, h);
            ctx.fillStyle = '#fff'; ctx.font = 'bold 16px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            ctx.fillText(String(index + 1), x + w / 2, y + h / 2);
            if (index === state.active) {
                for (const [cx, cy] of [[x, y], [x + w, y], [x, y + h], [x + w, y + h]]) {
                    ctx.beginPath(); ctx.arc(cx, cy, 7, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
                }
            }
        }
    }

    function pointer(event) {
        const bounds = canvas.getBoundingClientRect();
        // object-fit:contain can letterbox the DOM canvas; use its rendered content bounds.
        const scale = Math.min(bounds.width / canvas.width, bounds.height / canvas.height);
        const left = bounds.left + (bounds.width - canvas.width * scale) / 2;
        const top = bounds.top + (bounds.height - canvas.height * scale) / 2;
        return { x: ((event.clientX - left) / scale - margin) / (canvas.width - margin * 2),
            y: ((event.clientY - top) / scale - margin) / (canvas.height - margin * 2),
            rx: 12 / (canvas.width - margin * 2), ry: 12 / (canvas.height - margin * 2) };
    }

    function finishDrag() {
        if (!state.drag) return;
        const id = state.drag.id; state.drag = null;
        if (canvas.hasPointerCapture(id)) canvas.releasePointerCapture(id);
        node.graph?.afterChange?.();
    }

    canvas.addEventListener('pointerdown', (event) => {
        if (event.button !== 0 || isLinked() || !state.regions) return;
        const point = pointer(event);
        const hit = hitTest(state.regions, state.active, point.x, point.y, point.rx, point.ry);
        if (!hit) return;
        event.preventDefault(); event.stopPropagation();
        state.active = hit.index;
        state.drag = { ...hit, origin: { ...state.regions[hit.index] }, point, id: event.pointerId };
        node.graph?.beforeChange?.();
        canvas.setPointerCapture(event.pointerId); draw();
    });
    canvas.addEventListener('pointermove', (event) => {
        const drag = state.drag;
        if (!drag || drag.id !== event.pointerId) return;
        if (!event.buttons || isLinked()) { finishDrag(); return; }
        event.preventDefault(); event.stopPropagation();
        const point = pointer(event), dx = point.x - drag.point.x, dy = point.y - drag.point.y;
        state.regions[drag.index] = drag.corner < 0 ? moveRegion(drag.origin, dx, dy) : resizeRegion(drag.origin, drag.corner, dx, dy);
        // Save during motion too, so queue/save while dragging sees current geometry.
        save(); draw();
    });
    for (const name of ['pointerup', 'pointercancel', 'lostpointercapture']) {
        canvas.addEventListener(name, (event) => { if (state.drag?.id === event.pointerId) { event.stopPropagation(); finishDrag(); } });
    }
    for (const control of node.widgets.filter((widget) => ['regions', 'width', 'height'].includes(widget.name))) {
        const callback = control.callback;
        control.callback = function (...args) {
            const result = callback?.apply(this, args);
            if (!state.saving) refresh();
            return result;
        };
    }
    const onConnectionsChange = node.onConnectionsChange;
    node.onConnectionsChange = function (...args) { const result = onConnectionsChange?.apply(this, args); refresh(); return result; };
    const onRemoved = node.onRemoved;
    node.onRemoved = function (...args) { finishDrag(); bindings.delete(this); return onRemoved?.apply(this, args); };
    const editor = node.addDOMWidget('region_editor', 'krea2_region_editor', element, {
        serialize: false, hideOnZoom: false,
        getMinHeight: () => 200, getMaxHeight: () => 650,
        getHeight: () => 80 + Math.max(160, Math.min(520, ((node.size?.[0] || 380) - 24) * canvas.height / canvas.width)),
    });
    editor.serialize = false;
    refresh();
}
