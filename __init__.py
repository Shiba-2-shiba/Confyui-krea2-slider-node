"""ComfyUI V3 extension entrypoint; core tests can import without ComfyUI."""

WEB_DIRECTORY = './web'


async def comfy_entrypoint():
    from .nodes import Krea2SliderExtension
    return Krea2SliderExtension()
