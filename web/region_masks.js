import { app } from '../../scripts/app.js';
import { bindRegionMaskEditor } from './region_mask_editor.js';

app.registerExtension({
    name: 'krea2.regionMasks',
    nodeCreated: bindRegionMaskEditor,
    loadedGraphNode: bindRegionMaskEditor,
});
