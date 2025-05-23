import slangpy as spy

from bvhgs import device
from bvhgs.camera import Camera
from bvhgs.gaussian import GaussianCloud
from bvhgs.prefix_sum import prefix_sum


class Renderer:
    gaussians: GaussianCloud
    camera: Camera

    gaussian_3d: spy.Buffer
    render_target: spy.Texture

    program: spy.ShaderProgram
    # Projection kernels.
    ker_proj: spy.ComputeKernel
    ker_cull: spy.ComputeKernel
    # Tile kernels.
    ker_tile: spy.ComputeKernel
    ker_gs_table: spy.ComputeKernel
    # Rasterization kernels.
    ker_duplicate_gs: spy.ComputeKernel
    ker_rasterize: spy.ComputeKernel

    def __init__(self, gaussians: GaussianCloud, camera: Camera) -> None:
        """Constructor for the Rasterizer class.

        :param gaussians: GaussianBuffer object containing the Gaussian points.
        :param camera: Camera object containing the camera parameters.
        """
        self.gaussians = gaussians
        self.camera = camera

        # Load the module.
        renderer_module = device.load_module("renderer.slang")
        self.program = device.link_program(
            modules=[renderer_module],
            entry_points=[],
        )
        # Load kernels.
        self.ker_proj = device.create_compute_kernel(
            device.link_program(
                modules=[renderer_module],
                entry_points=[renderer_module.entry_point("project")],
            )
        )
        self.ker_cull = device.create_compute_kernel(
            device.link_program(
                modules=[renderer_module],
                entry_points=[renderer_module.entry_point("cull")],
            )
        )
        self.ker_tile = device.create_compute_kernel(
            device.link_program(
                modules=[renderer_module],
                entry_points=[renderer_module.entry_point("computeTile")],
            )
        )
        self.ker_gs_table = device.create_compute_kernel(
            device.link_program(
                modules=[renderer_module],
                entry_points=[
                    renderer_module.entry_point("buildGaussianTable")
                ],
            )
        )
        self.ker_duplicate_gs = device.create_compute_kernel(
            device.link_program(
                modules=[renderer_module],
                entry_points=[renderer_module.entry_point("duplicateGaussian")],
            )
        )
        self.ker_rasterize = device.create_compute_kernel(
            device.link_program(
                modules=[renderer_module],
                entry_points=[renderer_module.entry_point("rasterize")],
            )
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
        # Get the camera parameters.
        camera_params = self.camera.to_slang()
        # Project the Gaussian points to screen space.
        gaussian_2d_buf = device.create_buffer(
            element_count=len(self.gaussians),
            struct_type=self.program.reflection.g_gaussian_2d,
            usage=spy.BufferUsage.shader_resource
            | spy.BufferUsage.unordered_access,
        )
        inside_flag_buf = device.create_buffer(
            element_count=len(self.gaussians),
            struct_type=self.program.reflection.g_inside_flag,
            usage=spy.BufferUsage.shader_resource
            | spy.BufferUsage.unordered_access,
        )

        self.ker_proj.dispatch(
            thread_count=[len(self.gaussians), 1, 1],
            vars={
                "g_camera": camera_params,
                "g_gaussian_3d": self.gaussian_3d,
                "g_gaussian_2d": gaussian_2d_buf,
                "g_inside_flag": inside_flag_buf,
            },
        )
        # Cull the Gaussian points.
        inside_offset_buf = prefix_sum(inside_flag_buf)
        inside_offset_cursor = spy.BufferCursor(
            self.program.reflection.g_inside_offset.type_layout.element_type_layout,
            inside_offset_buf,
        )
        # Read the last element of the cull prefix buffer to get the number of culled points.
        num_viewing = int(inside_offset_cursor[len(inside_offset_cursor) - 1].read())  # type: ignore
        # Create a buffer for the culled Gaussian points.
        culled_gaussian_2d_buf = device.create_buffer(
            element_count=num_viewing,
            struct_type=self.program.reflection.g_gaussian_2d,
            usage=spy.BufferUsage.shader_resource
            | spy.BufferUsage.unordered_access,
        )
        self.ker_cull.dispatch(
            thread_count=[num_viewing, 1, 1],
            vars={
                "g_gaussian_2d": gaussian_2d_buf,
                "g_inside_flag": inside_flag_buf,
                "g_inside_offset": inside_offset_buf,
                "g_culled_gaussian_2d": culled_gaussian_2d_buf,
            },
        )
        # TODO: Implement the tile and rasterization kernels.
