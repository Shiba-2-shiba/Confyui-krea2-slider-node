"""Image-free regional mask editor for native masked conditioning."""
from comfy_api.latest import io

from .krea2_slider_node.region_masks import DEFAULT_REGIONS, MAX_DIMENSION, create_region_masks


class Krea2RegionMasks(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id='Krea2RegionMasks',
            display_name='Krea2 Region Masks',
            category='model/krea2 slider',
            description=('Create two rectangular masks without a source image. '
                         'Drag regions or their corner handles; connect each MASK to Cond Pair Set Props. '
                         'Regions select locations, not genders.'),
            search_aliases=['two people masks', 'regional masks', 'rectangle mask', '男女 マスク'],
            inputs=[
                io.Int.Input('width', default=1024, min=8, max=MAX_DIMENSION, step=8),
                io.Int.Input('height', default=1024, min=8, max=MAX_DIMENSION, step=8),
                io.String.Input('regions', default=DEFAULT_REGIONS, multiline=True, advanced=True,
                                tooltip='Two normalized rectangles (x,y,w,h). Saved automatically by the editor.'),
            ],
            outputs=[
                io.Mask.Output(display_name='mask_1'),
                io.Mask.Output(display_name='mask_2'),
                io.Int.Output(display_name='width'),
                io.Int.Output(display_name='height'),
            ],
        )

    @classmethod
    def execute(cls, width, height, regions=DEFAULT_REGIONS):
        first, second = create_region_masks(width, height, regions)
        return io.NodeOutput(first, second, width, height)
