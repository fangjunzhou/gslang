import slangpy as spy

from bvhgs import device
from bvhgs.camera import Camera
from bvhgs.gaussian import GaussianCloud


class Renderer:
    gaussians: GaussianCloud
    camera: Camera

    gaussian_3d: spy.Buffer
    render_target: spy.Texture

    program: spy.ShaderProgram

    def __init__(self, gaussians: GaussianCloud, camera: Camera) -> None:
        """Constructor for the Rasterizer class.

        :param gaussians: GaussianBuffer object containing the Gaussian points.
        :param camera: Camera object containing the camera parameters.
        """
        self.gaussians = gaussians
        self.camera = camera

        self.program = device.load_program(
            "renderer.slang",
            entry_point_names=["projection", "cull", "rasterize"],
        )

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
        self.gaussian_3d = device.create_buffer(
            element_count=len(gaussians),
            struct_type=self.program.reflection.g_gaussian_3d,
            usage=spy.BufferUsage.shader_resource,
        )
        # Store all the gaussian points in the buffer.
        gaussian_cursor = spy.BufferCursor(
            self.program.reflection.g_gaussian_3d.type_layout.element_type_layout,
            self.gaussian_3d,
        )
        for i in range(len(gaussians)):
            gaussian_cursor[i].write(gaussians[i])
        gaussian_cursor.apply()

    def render(self) -> None:
        """Render the Gaussian points to the render target."""
        # TODO: Implement the rendering logic.
        pass
