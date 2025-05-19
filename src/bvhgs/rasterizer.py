import slangpy as spy

from bvhgs import device
from bvhgs.camera import Camera
from bvhgs.gaussian import GaussianCloud


class Rasterizer:
    gaussians: GaussianCloud
    camera: Camera

    gaussian_buf: spy.Buffer
    render_target: spy.Texture

    kernel: spy.ComputeKernel

    def __init__(self, gaussians: GaussianCloud, camera: Camera) -> None:
        """Constructor for the Rasterizer class.

        :param gaussians: GaussianBuffer object containing the Gaussian points.
        :param camera: Camera object containing the camera parameters.
        """
        self.gaussians = gaussians
        self.camera = camera

        program = device.load_program(
            "rasterizer.slang", entry_point_names=["render"]
        )
        self.kernel = device.create_compute_kernel(program)

        # Create a render texture for rendering.
        self.render_target = device.create_texture(
            type=spy.TextureType.texture_2d,
            format=spy.Format.rgba32_float,
            width=self.camera.sensor_size.x,
            height=self.camera.sensor_size.y,
            usage=spy.TextureUsage.shader_resource
            | spy.TextureUsage.unordered_access,
        )
        # Create a buffer for the Gaussian points.
        self.gaussian_buf = device.create_buffer(
            element_count=len(gaussians),
            struct_type=self.kernel.reflection.g_gaussians,
            usage=spy.BufferUsage.shader_resource,
        )
        # Store all the gaussian points in the buffer.
        gaussian_cursor = spy.BufferCursor(
            self.kernel.reflection.g_gaussians.type_layout.element_type_layout,
            self.gaussian_buf,
        )
        for i in range(len(gaussians)):
            gaussian_cursor[i].write(gaussians[i])
        gaussian_cursor.apply()

    def render(self) -> None:
        """Render the Gaussian points to the render target."""
        camera_params = {
            "_rotation": self.camera.rotation,
            "_translation": self.camera.translation,
            "_sensorSize": self.camera.sensor_size,
            "_focalLength": self.camera.focal_length,
        }

        # Dispatch the compute kernel
        self.kernel.dispatch(
            thread_count=[
                self.camera.sensor_size.x,
                self.camera.sensor_size.y,
                1,
            ],
            vars={
                "g_camera": camera_params,
                "g_gaussians": self.gaussian_buf,
                "g_output": self.render_target,
            },
        )
