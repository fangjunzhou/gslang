import logging
import slangpy as spy
import numpy as np

from bvhgs import device
from bvhgs.camera import Camera
from bvhgs.gaussian import GaussianCloud
from bvhgs.prefix_sum import prefix_sum
from bvhgs.radix_sort import numpy_sort, radix_sort, stable_radix_sort
import jax
import jax.numpy as jnp
from PIL import Image

logger = logging.getLogger(__name__)


class Renderer:
    gaussians: GaussianCloud
    camera: Camera

    gaussian_3d: spy.Buffer
    gaussian_3d_grad: spy.Buffer
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
            device.link_program([renderer_module], [renderer_module.entry_point("bwdRasterize")])
        )
        self.ker_bwd_duplicate = device.create_compute_kernel(
            device.link_program([renderer_module], [renderer_module.entry_point("bwdDuplicateGaussian")])
        )
        self.ker_bwd_cull_proj = device.create_compute_kernel(
            device.link_program([renderer_module], [renderer_module.entry_point("bwdCullProjection")])
        )
        self.ker_grad_descent = device.create_compute_kernel(
            device.link_program([renderer_module], [renderer_module.entry_point("gradDescentGaussian3D")])
        )

        self.ker_extract_culled_gaussian = device.create_compute_kernel(
            device.link_program([renderer_module], [renderer_module.entry_point("extractCulledGaussianGrad")])
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
        self.gaussian_3d = device.create_buffer(
            element_count=len(gaussians),
            struct_type=self.program.reflection.g_gaussian_3d,
            usage=spy.BufferUsage.shader_resource
            | spy.BufferUsage.unordered_access,
        )
        # Store all the gaussian points in the buffer.
        gaussian_cursor = spy.BufferCursor(
            self.program.reflection.g_gaussian_3d.type_layout.element_type_layout,
            self.gaussian_3d,
        )
        for i in range(len(gaussians)):
            gaussian_cursor[i].write(gaussians[i])
        gaussian_cursor.apply()
        
    
        self.gaussian_3d_grad_buf = device.create_buffer(
            element_count=len(gaussians),
            struct_type=self.program.reflection.d_gaussian_3d,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
        )
        
        self.loss_grad = jax.value_and_grad(self.image_loss)

        
    def zero_grad(self):
        # gaussian_2d_sorted_grad_buf.copy_from_numpy(
        #     np.zeros((gaussian_2d_sorted_grad_buf.size,), dtype=np.uint8))
        # a_gaussian_2d_sorted_grad_buf.copy_from_numpy(
        #     np.zeros((a_gaussian_2d_sorted_grad_buf.size,), dtype=np.uint8))
        
        # gaussian_2d_culled_grad_buf.copy_from_numpy(
        #     np.zeros((gaussian_2d_culled_grad_buf.size,), dtype=np.uint8))
        # a_guassian_2d_culled_grad_buf.copy_from_numpy(
        #     np.zeros((a_guassian_2d_culled_grad_buf.size,), dtype=np.uint8))
        
        self.gaussian_3d_grad_buf.copy_from_numpy(
            np.zeros((self.gaussian_3d_grad_buf.size,), dtype=np.uint8))
        
    def set_gt_image(self, image: Image.Image) -> None:
        """Set the ground truth image for the renderer.

        :param image: The ground truth image as a PIL Image.
        """
        self.image_arr = jnp.array(image).astype(jnp.float32) / 255.0
        
    def image_loss(self, src: jnp.ndarray, dst: jnp.ndarray):
        return jnp.mean((dst - src)**2)

    def render(self, gt_image: Image.Image | None = None) -> None:
        """Render the Gaussian points to the render target."""
        # Get the camera parameters.
        camera_params = self.camera.to_slang()
        logger.debug(
            "Rendering %d Gaussian points with camera parameters: %s",
            len(self.gaussians),
            camera_params,
        )
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
        inside_arr = inside_flag_buf.to_numpy().view(np.uint32)
        num_viewing = np.sum(inside_arr).item()
        if num_viewing == 0:
            logger.debug("No Gaussian points inside the camera frustum.")
            return
        # Create a buffer for the culled Gaussian points.
        culled_gaussian_2d_buf = device.create_buffer(
            element_count=num_viewing,
            struct_type=self.program.reflection.g_gaussian_2d_culled,
            usage=spy.BufferUsage.shader_resource
            | spy.BufferUsage.unordered_access,
        )
        self.ker_cull.dispatch(
            thread_count=[len(self.gaussians), 1, 1],
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
        numpy_sort(gaussian_table_buf)
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

        if gt_image is not None:
            self.set_gt_image(gt_image)
            
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

            print(f"Loss after gradient descent: {loss}")

            rg_shape = render_target_grad.shape
            render_target_grad = jnp.concatenate((render_target_grad, jnp.zeros((rg_shape[0], rg_shape[1], 1))), axis=-1)
            
            self.grad_texture.copy_from_numpy(render_target_grad)
            
            self.ker_bwd_rasterize.dispatch(
                thread_count=[self.camera.sensor_size.x, self.camera.sensor_size.y, 1],
                vars={
                    "g_camera": self.camera.to_slang(),
                    "g_tile_hist": hist_buf,
                    "g_tile_offs": hist_offset_buf,
                    "g_gaussian_2d_sorted": gaussian_2d_sorted_buf,
                    "g_render_target": self.render_target,
                    "d_render_target": self.grad_texture,
                    "d_a_gaussian_2d_sorted": a_gaussian_2d_sorted_grad_buf,
                    "g_num_rendered_gaussians": self.num_depth_buf,
                }
            )
            
            #bwd duplicate
            self.ker_bwd_duplicate.dispatch(
                thread_count=[table_size, 1, 1],
                vars={
                    "g_gaussian_table": gaussian_table_buf,
                    "d_a_gaussian_2d_sorted": a_gaussian_2d_sorted_grad_buf,
                    "d_a_gaussian_2d_culled": a_gaussian_2d_culled_grad_buf,
                }
            )
            
            # Extract the gradients for the culled Gaussian 2D points.
            self.ker_extract_culled_gaussian.dispatch(
                thread_count=[num_viewing, 1, 1],
                vars={
                    "d_a_gaussian_2d_culled": a_gaussian_2d_culled_grad_buf,
                    "d_gaussian_2d_culled": gaussian_2d_culled_grad_buf,
                }
            )
            

            # bwd cull and projection
            self.ker_bwd_cull_proj.dispatch(
                thread_count=[len(self.gaussians), 1, 1],
                vars={
                    "g_camera": self.camera.to_slang(),
                    "g_gaussian_3d": self.gaussian_3d,
                    "g_inside_flag": inside_flag_buf,
                    "g_inside_offset": inside_offset_buf,
                    "d_gaussian_2d_culled": gaussian_2d_culled_grad_buf,
                    "d_gaussian_3d": self.gaussian_3d_grad_buf,
                }
            )
        
            
            
                    
    def backward(self, lr: float = 5) -> None:
        """Perform gradient descent on the Gaussian points."""
        if self.image_arr.size == 0:
            raise ValueError("Ground truth image not set for gradient descent.")
        
        # Ensure the Gaussian 3D buffer is initialized.
        if self.gaussian_3d_grad_buf is None:
            raise ValueError("Gaussian 3D gradient buffer is not initialized.")
        
        self.ker_grad_descent.dispatch(
            thread_count=[len(self.gaussians), 1, 1],
            lr=lr,
            vars={
                "g_gaussian_3d": self.gaussian_3d,
                "d_gaussian_3d": self.gaussian_3d_grad_buf,
            }
        )