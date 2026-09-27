"""ComfyUI V3 extension entrypoint; core tests can import without ComfyUI."""


async def comfy_entrypoint():
    from .nodes import Krea2SliderExtension
    return Krea2SliderExtension()
