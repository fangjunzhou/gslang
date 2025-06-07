import logging
import slangpy as spy
import numpy as np

from gslang import device
from gslang.camera import Camera
from gslang.gaussian import GaussianCloud
from gslang.prefix_sum import prefix_sum
from gslang.radix_sort import numpy_sort, radix_sort
import jax
import jax.numpy as jnp
from PIL import Image
from pathlib import Path

logger = logging.getLogger(__name__)


class Renderer:
    camera: Camera
    num_gaussians: int

    gaussian_3d_buf: spy.Buffer
    gaussian_3d_grad_buf: spy.Buffer
    m_buf: spy.Buffer
    v_buf: spy.Buffer
    render_target: spy.Texture
    depth_target: spy.Texture
    tile_heat_map: np.ndarray

    program: spy.ShaderProgram
    # Projection kernels.
    ker_proj: spy.ComputeKernel
    ker_cull: spy.ComputeKernel
    # Tile kernels.
    ker_tile: spy.ComputeKernel
    ker_gs_table: spy.ComputeKernel
    ker_tile_hist: spy.ComputeKernel
    # Rasterization kernels.
    ker_duplicate_gs: spy.ComputeKernel
    ker_rasterize: spy.ComputeKernel
    image_arr: jnp.ndarray = jnp.array([])

    def __init__(self, gaussians: GaussianCloud, camera: Camera) -> None:
        """Constructor for the Rasterizer class.

        :param gaussians: GaussianBuffer object containing the Gaussian points.
        :param camera: Camera object containing the camera parameters.
        """
        self.num_gaussians = len(gaussians)
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
        self.ker_tile_hist = device.create_compute_kernel(
            device.link_program(
                modules=[renderer_module],
                entry_points=[renderer_module.entry_point("tileHistogram")],
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

        self.ker_bwd_rasterize = device.create_compute_kernel(
            device.link_program(
                [renderer_module], [renderer_module.entry_point("bwdRasterize")]
            )
        )
        self.ker_bwd_duplicate = device.create_compute_kernel(
            device.link_program(
                [renderer_module],
                [renderer_module.entry_point("bwdDuplicateGaussian")],
            )
        )
        self.ker_bwd_cull_proj = device.create_compute_kernel(
            device.link_program(
                [renderer_module],
                [renderer_module.entry_point("bwdCullProjection")],
            )
        )
        self.ker_grad_descent = device.create_compute_kernel(
            device.link_program(
                [renderer_module],
                [renderer_module.entry_point("gradDescentGaussian3D")],
            )
        )

        self.ker_extract_culled_gaussian = device.create_compute_kernel(
            device.link_program(
                [renderer_module],
                [renderer_module.entry_point("extractCulledGaussianGrad")],
            )
        )

        # Densification kernel
        self.ker_mark_duplicated = device.create_compute_kernel(
            device.link_program(
                [renderer_module],
                [renderer_module.entry_point("markDuplicate")],
            )
        )
        self.ker_densify = device.create_compute_kernel(
            device.link_program(
                [renderer_module],
                [renderer_module.entry_point("densify")],
            )
        )

        # opacity control kernel
        self.ker_mark_keep = device.create_compute_kernel(
            device.link_program(
                [renderer_module],
                [renderer_module.entry_point("markKeep")],
            )
        )
        self.ker_remove_gaussian = device.create_compute_kernel(
            device.link_program(
                [renderer_module],
                [renderer_module.entry_point("removeGaussian")],
            )
        )

        self.ker_set_all_opacity = device.create_compute_kernel(
            device.link_program(
                [renderer_module],
                [renderer_module.entry_point("setAllOpacity")],
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
        self.depth_target = device.create_texture(
            type=spy.TextureType.texture_2d,
            format=spy.Format.rgba32_float,
            width=self.camera.sensor_size.x,
            height=self.camera.sensor_size.y,
            usage=spy.TextureUsage.shader_resource
            | spy.TextureUsage.unordered_access,
        )
        self.grad_texture = device.create_texture(
            type=spy.TextureType.texture_2d,
            format=spy.Format.rgba32_float,
            width=camera.sensor_size.x,
            height=camera.sensor_size.y,
            usage=spy.TextureUsage.shader_resource
            | spy.TextureUsage.unordered_access,
        )

        self.num_depth_buf = device.create_texture(
            type=spy.TextureType.texture_2d,
            format=spy.Format.r32_uint,
            width=camera.sensor_size.x,
            height=camera.sensor_size.y,
            usage=spy.TextureUsage.shader_resource
            | spy.TextureUsage.unordered_access,
        )

        self.tile_heat_map = np.zeros((16, 16))
        # Create a buffer for the Gaussian points.
        self.gaussian_3d_buf = device.create_buffer(
            element_count=len(gaussians),
            struct_type=self.program.reflection.g_gaussian_3d,
            usage=spy.BufferUsage.shader_resource
            | spy.BufferUsage.unordered_access,
        )
        # Store all the gaussian points in the buffer.
        gaussian_cursor = spy.BufferCursor(
            self.program.reflection.g_gaussian_3d.type_layout.element_type_layout,
            self.gaussian_3d_buf,
        )
        for i in range(len(gaussians)):
            gaussian_cursor[i].write(gaussians[i])
        gaussian_cursor.apply()

        self.gaussian_3d_grad_buf = device.create_buffer(
            element_count=len(gaussians),
            struct_type=self.program.reflection.d_gaussian_3d,
            usage=spy.BufferUsage.shader_resource
            | spy.BufferUsage.unordered_access,
        )
        self.gaussian_3d_grad_pos_buf = device.create_buffer(
            element_count=len(gaussians),
            struct_type=self.program.reflection.d_gaussian_3d_pos,
            usage=spy.BufferUsage.shader_resource
            | spy.BufferUsage.unordered_access,
        )
        self.gaussian_2d_buf = device.create_buffer(
            element_count=len(gaussians),
            struct_type=self.program.reflection.g_gaussian_2d,
            usage=spy.BufferUsage.shader_resource
            | spy.BufferUsage.unordered_access,
        )
        self.m_buf = device.create_buffer(
            element_count=len(gaussians),
            struct_type=self.program.reflection.g_gaussian_3d,
            usage=spy.BufferUsage.shader_resource
            | spy.BufferUsage.unordered_access,
        )
        self.v_buf = device.create_buffer(
            element_count=len(gaussians),
            struct_type=self.program.reflection.g_gaussian_3d,
            usage=spy.BufferUsage.shader_resource
            | spy.BufferUsage.unordered_access,
        )
        self.m_buf.copy_from_numpy(np.zeros((self.m_buf.size,), dtype=np.uint8))
        self.v_buf.copy_from_numpy(np.zeros((self.v_buf.size,), dtype=np.uint8))

        self.adamw_step = 1

        self.loss_grad = jax.value_and_grad(
            lambda src, dst: self.image_loss(src, dst)
        )

    def set_camera(self, camera: Camera):
        """Set the camera for the renderer.

        :param camera: Camera object containing the camera parameters.
        """
        prev_camera = self.camera
        self.camera = camera
        # If the camera sensor size has changed, recreate the render target.
        if (
            prev_camera.sensor_size.x != camera.sensor_size.x
            or prev_camera.sensor_size.y != camera.sensor_size.y
        ):
            logger.debug(
                "Camera sensor size changed from %s to %s. Recreating render target.",
                (prev_camera.sensor_size.x, prev_camera.sensor_size.y),
                (camera.sensor_size.x, camera.sensor_size.y),
            )
            self.render_target = device.create_texture(
                type=spy.TextureType.texture_2d,
                format=spy.Format.rgba32_float,
                width=camera.sensor_size.x,
                height=camera.sensor_size.y,
                usage=spy.TextureUsage.shader_resource
                | spy.TextureUsage.unordered_access,
            )
            self.depth_target = device.create_texture(
                type=spy.TextureType.texture_2d,
                format=spy.Format.rgba32_float,
                width=camera.sensor_size.x,
                height=camera.sensor_size.y,
                usage=spy.TextureUsage.shader_resource
                | spy.TextureUsage.unordered_access,
            )
            self.grad_texture = device.create_texture(
                type=spy.TextureType.texture_2d,
                format=spy.Format.rgba32_float,
                width=camera.sensor_size.x,
                height=camera.sensor_size.y,
                usage=spy.TextureUsage.shader_resource
                | spy.TextureUsage.unordered_access,
            )

    def sync_gaussians(self, buf_data: np.ndarray, num_gaussians: int):
        """Synchronize the Gaussian points with the given buffer."""
        # Create a buffer for the Gaussian points.
        self.gaussian_3d_buf = device.create_buffer(
            element_count=num_gaussians,
            struct_type=self.program.reflection.g_gaussian_3d,
            usage=spy.BufferUsage.shader_resource
            | spy.BufferUsage.unordered_access,
        )
        self.gaussian_3d_buf.copy_from_numpy(buf_data)
        self.num_gaussians = num_gaussians

    def zero_grad(self):
        self.gaussian_3d_grad_buf.copy_from_numpy(
            np.zeros((self.gaussian_3d_grad_buf.size,), dtype=np.uint8)
        )

    def image_loss(
        self,
        src: jnp.ndarray,
        dst: jnp.ndarray,
    ):
        # L1 loss
        l1_loss = jnp.mean(jnp.abs(dst - src))
        return l1_loss

    def render(
        self,
        gt_image: Image.Image | None = None,
        use_densify: bool = False,
        use_opacity_prune: bool = False,
        use_reset_opacity: bool = False,
        densify_scale: float = 1.0,
        overConstructionShrinkScale: float = 1.6,
        gaussian_opacity_prune_threshold: float = -3,
        gaussian_reset_opacity: float = -4,
        split: bool = True,
    ) -> float:
        """Render the Gaussian points to the render target."""
        # Get the camera parameters.
        camera_params = self.camera.to_slang()
        # Project the Gaussian points to screen space.
        gaussian_2d_buf = device.create_buffer(
            element_count=self.num_gaussians,
            struct_type=self.program.reflection.g_gaussian_2d,
            usage=spy.BufferUsage.shader_resource
            | spy.BufferUsage.unordered_access,
        )
        self.gaussian_2d_buf = gaussian_2d_buf
        inside_flag_buf = device.create_buffer(
            element_count=self.num_gaussians,
            struct_type=self.program.reflection.g_inside_flag,
            usage=spy.BufferUsage.shader_resource
            | spy.BufferUsage.unordered_access,
        )

        self.ker_proj.dispatch(
            thread_count=[self.num_gaussians, 1, 1],
            vars={
                "g_camera": camera_params,
                "g_gaussian_3d": self.gaussian_3d_buf,
                "g_gaussian_2d": gaussian_2d_buf,
                "g_inside_flag": inside_flag_buf,
            },
        )
        # Cull the Gaussian points.
        inside_offset_buf = prefix_sum(inside_flag_buf)
        num_viewing = inside_offset_buf.to_numpy().view(np.uint32)[-1].item()
        if num_viewing == 0:
            logger.debug("No Gaussian points inside the camera frustum.")
            return 0.0
        # Create a buffer for the culled Gaussian points.
        culled_gaussian_2d_buf = device.create_buffer(
            element_count=num_viewing,
            struct_type=self.program.reflection.g_gaussian_2d_culled,
            usage=spy.BufferUsage.shader_resource
            | spy.BufferUsage.unordered_access,
        )
        self.ker_cull.dispatch(
            thread_count=[self.num_gaussians, 1, 1],
            vars={
                "g_gaussian_2d": gaussian_2d_buf,
                "g_inside_flag": inside_flag_buf,
                "g_inside_offset": inside_offset_buf,
                "g_gaussian_2d_culled": culled_gaussian_2d_buf,
            },
        )
        # Create tile buffers.
        num_tile_buf = device.create_buffer(
            element_count=num_viewing,
            struct_type=self.program.reflection.g_num_tiles,
            usage=spy.BufferUsage.shader_resource
            | spy.BufferUsage.unordered_access,
        )
        self.ker_tile.dispatch(
            thread_count=[num_viewing, 1, 1],
            numViewing=num_viewing,
            vars={
                "g_gaussian_2d_culled": culled_gaussian_2d_buf,
                "g_num_tiles": num_tile_buf,
            },
        )
        # Calculate total number of table entries
        num_tile_arr = num_tile_buf.to_numpy().view(np.uint32)
        table_size = np.sum(num_tile_arr).item()

        # Build the Gaussian table.
        gaussian_table_buf = device.create_buffer(
            element_count=table_size,
            struct_type=self.program.reflection.g_gaussian_table,
            usage=spy.BufferUsage.shader_resource
            | spy.BufferUsage.unordered_access,
        )
        num_tile_prefix_buf = prefix_sum(num_tile_buf)
        self.ker_gs_table.dispatch(
            thread_count=[num_viewing, 1, 1],
            numViewing=num_viewing,
            vars={
                "g_gaussian_2d_culled": culled_gaussian_2d_buf,
                "g_num_tiles_prefix": num_tile_prefix_buf,
                "g_gaussian_table": gaussian_table_buf,
            },
        )
        # Sort tiles.
        radix_sort(
            gaussian_table_buf,
            bits_per_pass=4,
            total_bits=40,
            entry_per_thread=128,
        )
        # Create a histogram buffer for the tiles.
        hist_buf = device.create_buffer(
            element_count=16 * 16,
            struct_type=self.program.reflection.g_tile_hist_atomic,
            usage=spy.BufferUsage.shader_resource
            | spy.BufferUsage.unordered_access,
        )
        # Compute the histogram of the Gaussian table.
        self.ker_tile_hist.dispatch(
            thread_count=[table_size, 1, 1],
            numEntries=table_size,
            vars={
                "g_gaussian_table": gaussian_table_buf,
                "g_tile_hist_atomic": hist_buf,
            },
        )
        # Duplicate Gaussian points.
        gaussian_2d_sorted_buf = device.create_buffer(
            element_count=table_size,
            struct_type=self.program.reflection.g_gaussian_2d_sorted,
            usage=spy.BufferUsage.shader_resource
            | spy.BufferUsage.unordered_access,
        )
        self.ker_duplicate_gs.dispatch(
            thread_count=[table_size, 1, 1],
            vars={
                "g_gaussian_table": gaussian_table_buf,
                "g_gaussian_2d_culled": culled_gaussian_2d_buf,
                "g_gaussian_2d_sorted": gaussian_2d_sorted_buf,
            },
        )
        # Calculate table offset.
        hist_arr = hist_buf.to_numpy().view(np.uint32)
        self.tile_heat_map = hist_arr.reshape(16, 16)
        hist_offset = np.zeros_like(hist_arr)
        hist_offset[1:] = np.cumsum(hist_arr)[:-1]
        hist_offset_buf = device.create_buffer(
            element_count=len(hist_offset),
            struct_type=self.program.reflection.g_tile_offs,
            usage=spy.BufferUsage.shader_resource
            | spy.BufferUsage.unordered_access,
        )
        hist_offset_buf.copy_from_numpy(hist_offset)
        # Rasterize the Gaussian points.
        self.ker_rasterize.dispatch(
            thread_count=[
                self.camera.sensor_size.x,
                self.camera.sensor_size.y,
                1,
            ],
            vars={
                "g_camera": camera_params,
                "g_tile_hist": hist_buf,
                "g_tile_offs": hist_offset_buf,
                "g_gaussian_2d_sorted": gaussian_2d_sorted_buf,
                "g_render_target": self.render_target,
                "g_depth_target": self.depth_target,
                "g_num_rendered_gaussians": self.num_depth_buf,
            },
        )

        loss = 0.0

        if gt_image is not None:
            loss = self.backward(
                gt_image,
                inside_flag_buf,
                inside_offset_buf,
                num_viewing,
                table_size,
                gaussian_table_buf,
                hist_buf,
                gaussian_2d_sorted_buf,
                hist_offset_buf,
            )

        if use_densify:
            self.densify(
                inside_flag_buf,
                inside_offset_buf,
                densify_scale=densify_scale,
                overConstructionShrinkScale=overConstructionShrinkScale,
                split=split,
            )

        if use_opacity_prune:
            self.gaussian_removal_by_opacity(
                gaussian_opacity_prune_threshold=gaussian_opacity_prune_threshold
            )

        if use_reset_opacity:
            self.set_all_opacity(gaussian_reset_opacity)

        return loss

    def backward(
        self,
        gt_image: Image.Image,
        inside_flag_buf: spy.Buffer,
        inside_offset_buf: spy.Buffer,
        num_viewing: int,
        table_size: int,
        gaussian_table_buf: spy.Buffer,
        hist_buf: spy.Buffer,
        gaussian_2d_sorted_buf: spy.Buffer,
        hist_offset_buf: spy.Buffer,
    ):
        self.image_arr = jnp.array(gt_image).astype(jnp.float32) / 255.0

        if self.image_arr.size == 0:
            raise ValueError("Ground truth image not set for gradient descent.")
        a_gaussian_2d_sorted_grad_buf = device.create_buffer(
            element_count=table_size,
            struct_type=self.program.reflection.d_a_gaussian_2d_sorted,
            usage=spy.BufferUsage.shader_resource
            | spy.BufferUsage.unordered_access,
        )

        a_gaussian_2d_culled_grad_buf = device.create_buffer(
            element_count=num_viewing,
            struct_type=self.program.reflection.d_a_gaussian_2d_culled,
            usage=spy.BufferUsage.shader_resource
            | spy.BufferUsage.unordered_access,
        )

        gaussian_2d_culled_grad_buf = device.create_buffer(
            element_count=num_viewing,
            struct_type=self.program.reflection.d_gaussian_2d_culled,
            usage=spy.BufferUsage.shader_resource
            | spy.BufferUsage.unordered_access,
        )

        raw_image = jnp.array(self.render_target.to_numpy()[:, :, :3])
        loss, render_target_grad = self.loss_grad(raw_image, self.image_arr)

        logger.debug(f"Loss: {loss}")

        rg_shape = render_target_grad.shape
        render_target_grad = jnp.concatenate(
            (render_target_grad, jnp.zeros((rg_shape[0], rg_shape[1], 1))),
            axis=-1,
        )

        self.grad_texture.copy_from_numpy(render_target_grad)

        self.ker_bwd_rasterize.dispatch(
            thread_count=[
                self.camera.sensor_size.x,
                self.camera.sensor_size.y,
                1,
            ],
            vars={
                "g_camera": self.camera.to_slang(),
                "g_tile_hist": hist_buf,
                "g_tile_offs": hist_offset_buf,
                "g_gaussian_2d_sorted": gaussian_2d_sorted_buf,
                "g_render_target": self.render_target,
                "d_render_target": self.grad_texture,
                "d_a_gaussian_2d_sorted": a_gaussian_2d_sorted_grad_buf,
                "g_num_rendered_gaussians": self.num_depth_buf,
            },
        )

        if logger.getEffectiveLevel() <= logging.DEBUG:
            arr = (
                a_gaussian_2d_sorted_grad_buf.to_numpy()
                .view(np.float32)
                .reshape(table_size, -1)
            )
            logger.debug(
                f"Gaussian 2D Sorted Gradients Max: {np.max(arr, axis=0)}"
            )
            logger.debug(
                f"Gaussian 2D Sorted Gradients Min: {np.min(arr, axis=0)}"
            )

            # bwd duplicate
        self.ker_bwd_duplicate.dispatch(
            thread_count=[table_size, 1, 1],
            vars={
                "g_gaussian_table": gaussian_table_buf,
                "d_a_gaussian_2d_sorted": a_gaussian_2d_sorted_grad_buf,
                "d_a_gaussian_2d_culled": a_gaussian_2d_culled_grad_buf,
            },
        )

        if logger.getEffectiveLevel() <= logging.DEBUG:
            arr = (
                a_gaussian_2d_culled_grad_buf.to_numpy()
                .view(np.float32)
                .reshape(num_viewing, -1)
            )
            logger.debug(
                f"Gaussian 2D Culled Gradients Max: {np.max(arr, axis=0)}"
            )
            logger.debug(
                f"Gaussian 2D Culled Gradients Min: {np.min(arr, axis=0)}"
            )

            # Extract the gradients for the culled Gaussian 2D points.
        self.ker_extract_culled_gaussian.dispatch(
            thread_count=[num_viewing, 1, 1],
            vars={
                "d_a_gaussian_2d_culled": a_gaussian_2d_culled_grad_buf,
                "d_gaussian_2d_culled": gaussian_2d_culled_grad_buf,
            },
        )

        # bwd cull and projection
        self.ker_bwd_cull_proj.dispatch(
            thread_count=[self.num_gaussians, 1, 1],
            vars={
                "g_camera": self.camera.to_slang(),
                "g_gaussian_3d": self.gaussian_3d_buf,
                "g_inside_flag": inside_flag_buf,
                "g_inside_offset": inside_offset_buf,
                "d_gaussian_2d_culled": gaussian_2d_culled_grad_buf,
                "d_gaussian_3d": self.gaussian_3d_grad_buf,
                "d_gaussian_3d_pos": self.gaussian_3d_grad_pos_buf,
            },
        )

        if logger.getEffectiveLevel() <= logging.DEBUG:
            arr = (
                self.gaussian_3d_grad_buf.to_numpy()
                .view(np.float32)
                .reshape(self.num_gaussians, -1, 4)[:, :8, :]
            )
            logger.debug(f"Gaussian 3D Gradients Max: {np.max(arr, axis=0)}")
            logger.debug(f"Gaussian 3D Gradients Min: {np.min(arr, axis=0)}")
            arr = (
                self.gaussian_3d_buf.to_numpy()
                .view(np.float32)
                .reshape(self.num_gaussians, -1, 4)[:, :8, :]
            )
            logger.debug(f"Gaussian 3D Points Max: {np.max(arr, axis=0)}")
            logger.debug(f"Gaussian 3D Points Min: {np.min(arr, axis=0)}")
        return loss

    def densify(
        self,
        inside_flag_buf: spy.Buffer,
        inside_offset_buf: spy.Buffer,
        densify_scale: float = 1.0,
        overConstructionShrinkScale: float = 1.6,
        split: bool = True,
    ):
        avg_scale = self.calculate_avg_3dgs_size()
        """Densify the Gaussian points."""
        duplicate_flag_buf = device.create_buffer(
            element_count=self.num_gaussians,
            struct_type=self.program.reflection.g_duplicate_flag,
            usage=spy.BufferUsage.shader_resource
            | spy.BufferUsage.unordered_access,
        )

        # Mark duplicated Gaussian points.
        self.ker_mark_duplicated.dispatch(
            thread_count=[self.num_gaussians, 1, 1],
            threashold=0.002,
            vars={
                "d_gaussian_3d_pos": self.gaussian_3d_grad_pos_buf,
                "g_inside_flag": inside_flag_buf,
                "g_inside_offset": inside_offset_buf,
                "g_duplicate_flag": duplicate_flag_buf,
            },
        )

        # Prefix sum the duplicate flag to get the number of duplicated points.
        duplicate_flag_prefix_buf = prefix_sum(duplicate_flag_buf)
        # Get the number of new Gaussian points.
        num_new_gaussians = (
            duplicate_flag_prefix_buf.to_numpy().view(np.uint32)[-1].item()
        )

        logger.info(
            f"Number of new Gaussian points to be added: {num_new_gaussians}"
        )
        if num_new_gaussians == 0:
            logger.debug("No new Gaussian points to be added.")
            return

        # Old gaussian buffer.
        old_gaussian_3d_buf = self.gaussian_3d_buf
        # Create a new buffer for the Gaussian points.
        self.gaussian_3d_buf = device.create_buffer(
            element_count=self.num_gaussians + num_new_gaussians,
            struct_type=self.program.reflection.g_gaussian_3d,
            usage=spy.BufferUsage.shader_resource
            | spy.BufferUsage.unordered_access,
        )
        # Densify the Gaussian points.
        self.ker_densify.dispatch(
            thread_count=[self.num_gaussians, 1, 1],
            numSrc=self.num_gaussians,
            underConstructionGradScale=densify_scale,
            overConstructionGradScale=densify_scale,
            overConstructionShrinkScale=overConstructionShrinkScale,
            split=split,
            vars={
                "g_gaussian_3d": self.gaussian_3d_buf,
                "g_gaussian_3d_src": old_gaussian_3d_buf,
                "g_duplicate_flag": duplicate_flag_buf,
                "g_duplicate_prefix": duplicate_flag_prefix_buf,
                "d_gaussian_3d_pos": self.gaussian_3d_grad_pos_buf,
            },
        )
        # Create a new buffer for the Gaussian gradients.
        self.gaussian_3d_grad_buf = device.create_buffer(
            element_count=self.num_gaussians + num_new_gaussians,
            struct_type=self.program.reflection.d_gaussian_3d,
            usage=spy.BufferUsage.shader_resource
            | spy.BufferUsage.unordered_access,
        )
        self.gaussian_3d_grad_pos_buf = device.create_buffer(
            element_count=self.num_gaussians + num_new_gaussians,
            struct_type=self.program.reflection.d_gaussian_3d_pos,
            usage=spy.BufferUsage.shader_resource
            | spy.BufferUsage.unordered_access,
        )
        self.m_buf = device.create_buffer(
            element_count=self.num_gaussians + num_new_gaussians,
            struct_type=self.program.reflection.m_gaussian_3d,
            usage=spy.BufferUsage.shader_resource
            | spy.BufferUsage.unordered_access,
        )
        self.v_buf = device.create_buffer(
            element_count=self.num_gaussians + num_new_gaussians,
            struct_type=self.program.reflection.v_gaussian_3d,
            usage=spy.BufferUsage.shader_resource
            | spy.BufferUsage.unordered_access,
        )
        # Update the number of Gaussian points.
        self.num_gaussians += num_new_gaussians

    def gaussian_removal_by_opacity(
        self, gaussian_opacity_prune_threshold: float
    ):
        keep_flag_buf = device.create_buffer(
            element_count=self.num_gaussians,
            struct_type=self.program.reflection.g_keep_flag,
            usage=spy.BufferUsage.shader_resource
            | spy.BufferUsage.unordered_access,
        )

        self.ker_mark_keep.dispatch(
            thread_count=[self.num_gaussians, 1, 1],
            opacityThreshold=gaussian_opacity_prune_threshold,
            numSrc=self.num_gaussians,
            vars={
                "g_gaussian_3d_src": self.gaussian_3d_buf,
                "g_keep_flag": keep_flag_buf,
            },
        )
        keep_prefix_buf = prefix_sum(keep_flag_buf)

        keep_prefix_np = keep_prefix_buf.to_numpy().view(np.uint32)

        num_keep = int(keep_prefix_np[-1])

        old_gaussian_3d_buf = self.gaussian_3d_buf
        old_gaussian_2d_buf = self.gaussian_2d_buf
        old_d_gaussian_3d_buf = self.gaussian_3d_grad_buf
        old_m_buf = self.m_buf
        old_v_buf = self.v_buf

        new_gaussian_3d_buf = device.create_buffer(
            element_count=num_keep,
            struct_type=self.program.reflection.g_gaussian_3d,
            usage=spy.BufferUsage.shader_resource
            | spy.BufferUsage.unordered_access,
        )
        new_gaussian_2d_buf = device.create_buffer(
            element_count=num_keep,
            struct_type=self.program.reflection.g_gaussian_2d,
            usage=spy.BufferUsage.shader_resource
            | spy.BufferUsage.unordered_access,
        )
        new_d_gaussian_3d_buf = device.create_buffer(
            element_count=num_keep,
            struct_type=self.program.reflection.d_gaussian_3d,
            usage=spy.BufferUsage.shader_resource
            | spy.BufferUsage.unordered_access,
        )
        new_gaussian_3d_pos_buf = device.create_buffer(
            element_count=num_keep,
            struct_type=self.program.reflection.d_gaussian_3d_pos,
            usage=spy.BufferUsage.shader_resource
            | spy.BufferUsage.unordered_access,
        )
        new_m_buf = device.create_buffer(
            element_count=num_keep,
            struct_type=self.program.reflection.m_gaussian_3d,
            usage=spy.BufferUsage.shader_resource
            | spy.BufferUsage.unordered_access,
        )
        new_v_buf = device.create_buffer(
            element_count=num_keep,
            struct_type=self.program.reflection.v_gaussian_3d,
            usage=spy.BufferUsage.shader_resource
            | spy.BufferUsage.unordered_access,
        )
        num_removal = self.num_gaussians - num_keep
        self.ker_remove_gaussian.dispatch(
            thread_count=[self.num_gaussians, 1, 1],
            numSrc=self.num_gaussians,
            vars={
                "g_gaussian_3d_src": old_gaussian_3d_buf,
                "g_gaussian_2d_src": old_gaussian_2d_buf,
                "d_gaussian_3d_src": old_d_gaussian_3d_buf,
                "m_gaussian_3d_src": old_m_buf,
                "v_gaussian_3d_src": old_v_buf,
                "g_keep_flag": keep_flag_buf,
                "g_keep_prefix": keep_prefix_buf,
                # target buffers
                "g_gaussian_3d": new_gaussian_3d_buf,
                "g_gaussian_2d": new_gaussian_2d_buf,
                "d_gaussian_3d": new_d_gaussian_3d_buf,
                "m_gaussian_3d": new_m_buf,
                "v_gaussian_3d": new_v_buf,
            },
        )

        self.gaussian_3d_buf = new_gaussian_3d_buf
        self.gaussian_2d_buf = new_gaussian_2d_buf
        self.gaussian_3d_grad_buf = new_d_gaussian_3d_buf
        self.gaussian_3d_grad_pos_buf = new_gaussian_3d_pos_buf
        self.m_buf = new_m_buf
        self.v_buf = new_v_buf
        self.num_gaussians = num_keep

        logger.info(
            f"Removed {num_removal} Gaussian points by opacity thresholding."
        )

    def set_all_opacity(self, new_opacity: float):
        self.ker_set_all_opacity.dispatch(
            thread_count=[self.num_gaussians, 1, 1],
            numGaussians=self.num_gaussians,
            newOpacity=new_opacity,
            vars={
                "g_gaussian_3d": self.gaussian_3d_buf,
            },
        )

    def optimizer_set_step(self, step: int):
        self.adamw_step = step

    def calculate_avg_3dgs_size(self):
        """Recalculate the average size of the Gaussian points."""
        gaussian_arr = (
            self.gaussian_3d_buf.to_numpy()
            .view(np.float32)
            .reshape(self.num_gaussians, -1)
        )
        if device.info.type == spy.DeviceType.metal:
            scale = np.exp(gaussian_arr[:, 9:12])
        else:
            scale = np.exp(gaussian_arr[:, 8:11])
        # Use matrix norm as the average size.
        avg_size = np.mean(scale)
        logger.info(f"Average Gaussian size: {avg_size}")

        return avg_size

    def step(
        self,
        pos_lr: float = 1e-3,
        rot_lr: float = 1e-3,
        scale_lr: float = 1e-3,
        color_lr: float = 1e-3,
        opacity_lr: float = 1e-3,
        sh_lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        weight_decay: float = 0.01,
    ) -> None:
        """Perform gradient descent on the Gaussian points."""
        if self.image_arr.size == 0:
            raise ValueError("Ground truth image not set for gradient descent.")

        # Ensure the Gaussian 3D buffer is initialized.
        if self.gaussian_3d_grad_buf is None:
            raise ValueError("Gaussian 3D gradient buffer is not initialized.")

        self.ker_grad_descent.dispatch(
            thread_count=[self.num_gaussians, 1, 1],
            pos_lr=pos_lr,
            rot_lr=rot_lr,
            scale_lr=scale_lr,
            color_lr=color_lr,
            opacity_lr=opacity_lr,
            sh_lr=sh_lr,
            beta1=beta1,
            beta2=beta2,
            weightDecay=weight_decay,
            step=self.adamw_step,
            vars={
                "g_gaussian_3d": self.gaussian_3d_buf,
                "d_gaussian_3d": self.gaussian_3d_grad_buf,
                "m_gaussian_3d": self.m_buf,
                "v_gaussian_3d": self.v_buf,
            },
        )

    def to_ply(self, path: Path):
        gaussian_arr = (
            self.gaussian_3d_buf.to_numpy()
            .view(np.float32)
            .reshape(self.num_gaussians, -1)
        )
        gaussians = GaussianCloud()
        # Alignment on metal devices.
        if device.info.type == spy.DeviceType.metal:
            gaussians.positions = gaussian_arr[:, :3]
            gaussians.rotations = gaussian_arr[:, 4:8]
            gaussians.scales = gaussian_arr[:, 8:11]
            gaussians.colors = gaussian_arr[:, 12:15]
            gaussians.opacities = gaussian_arr[:, 16:17]
            gaussians.spherical_harmonics = gaussian_arr[
                :, 18 : 18 + 15 * 4
            ].reshape(self.num_gaussians, 15, 4)[:, :, :3]
        else:
            gaussians.positions = gaussian_arr[:, :3]
            gaussians.rotations = gaussian_arr[:, 3:7]
            gaussians.scales = gaussian_arr[:, 7:10]
            gaussians.colors = gaussian_arr[:, 10:13]
            gaussians.opacities = gaussian_arr[:, 13:14]
            gaussians.spherical_harmonics = gaussian_arr[:, 14:].reshape(
                self.num_gaussians, 15, 3
            )
        gaussians.num_gaussians = self.num_gaussians
        gaussians.save_to_ply(path)
