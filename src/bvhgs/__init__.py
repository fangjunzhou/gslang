import slangpy as spy
import pathlib
import logging
import os

logger = logging.getLogger(__name__)

SHADER_PATH = pathlib.Path(__file__).parent / "slang"
device = spy.create_device(
    type=(
        spy.DeviceType.vulkan if os.name == "nt" else spy.DeviceType.automatic
    ),
    include_paths=[SHADER_PATH.absolute()],
    enable_print=True,
)
logger.info(f"Slang device created: {device}")

math_module = spy.Module.load_from_file(device, "math.slang")
bounding_box_module = spy.Module.load_from_file(device, "bounding-box.slang")

gaussian_module = spy.Module.load_from_file(
    device, "gaussian.slang", link=[math_module, bounding_box_module]
)

camera_module = spy.Module.load_from_file(
    device, "camera.slang", link=[math_module, gaussian_module]
)
