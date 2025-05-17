import slangpy as spy
import pathlib
import logging

logger = logging.getLogger(__name__)

SHADER_PATH = pathlib.Path(__file__).parent / "slang"
device = spy.create_device(include_paths=[SHADER_PATH.absolute()])
logger.info(f"Slang device created: {device}")

camera_module = spy.Module.load_from_file(device, "camera.slang")
gaussian_module = spy.Module.load_from_file(device, "gaussian.slang")
math_module = spy.Module.load_from_file(device, "math.slang")
