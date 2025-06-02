# %%
import slangpy as spy
from pyglm import glm
import matplotlib.pyplot as plt
import numpy as np
import quaternion
from PIL import Image
import pathlib
import jax
import jax.numpy as jnp


from gslang import device
from gslang.data import SFMDataset
from gslang.camera import Camera
from gslang.gaussian import GaussianCloud
from gslang.renderer import Renderer
from gslang.prefix_sum import prefix_sum
from gslang.radix_sort import radix_sort, numpy_sort

# %%
np.random.seed(0)

# %% [markdown]
# # Load Gaussian Buffer

# %%
gaussians = GaussianCloud()
gaussians.load_from_colmap(
    pathlib.Path("../resources/dataset/tandt_db/tandt/truck/sparse/0/")
)
len(gaussians)

# %% [markdown]
# # Load Module and Shader

# %%
module = device.load_module("renderer.slang")
module

# %%
program = device.link_program([module], [])
program

# %% [markdown]
# # Build Slang Buffer and Camera Parameter

# %%
# Create a buffer for the Gaussian points.
gaussian_buf = device.create_buffer(
    element_count=len(gaussians),
    struct_type=program.reflection.g_gaussian_3d,
    usage=spy.BufferUsage.shader_resource,
)
# Store all the gaussian points in the buffer.
gaussian_cursor = spy.BufferCursor(
    program.reflection.g_gaussian_3d.type_layout.element_type_layout,
    gaussian_buf,
)

for i in range(len(gaussians)):
    gaussian_cursor[i].write(gaussians[i])
gaussian_cursor.apply()

# %%
sfm_dataset = SFMDataset()
sfm_dataset.load_from_colmap(
    colmap_path=pathlib.Path(
        "../resources/dataset/tandt_db/tandt/truck/sparse/0/"
    ),
    image_dir=pathlib.Path("../resources/dataset/tandt_db/tandt/truck/images/"),
)

# %%
camera, image_path = sfm_dataset[0]
print(camera.to_slang())
image = Image.open(image_path)
image

# %% [markdown]
# # Projection

# %% [markdown]
# ## Dispatch Projection Kernel

# %%
# Create a buffer for the Gaussian2D points.
gaussian2d_buf = device.create_buffer(
    element_count=len(gaussians),
    struct_type=program.reflection.g_gaussian_2d,
    usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
)
gaussian2d_buf

# %%
# Create flag buffer for culling.
inside_flag_buf = device.create_buffer(
    element_count=len(gaussians),
    struct_type=program.reflection.g_inside_flag,
    usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
)
inside_flag_buf

# %%
ker_proj = device.create_compute_kernel(
    device.link_program([module], [module.entry_point("project")])
)
ker_proj

# %%
ker_proj.dispatch(
    thread_count=[len(gaussians), 1, 1],
    vars={
        "g_camera": camera.to_slang(),
        "g_gaussian_3d": gaussian_buf,
        "g_gaussian_2d": gaussian2d_buf,
        "g_inside_flag": inside_flag_buf,
    },
)

# %%
cursor = spy.BufferCursor(
    program.reflection.g_gaussian_2d.type_layout.element_type_layout,
    gaussian2d_buf,
)
cursor.element_count
cursor[0]

# %% [markdown]
# ## Cull Gaussians

# %%
ker_cull = device.create_compute_kernel(
    device.link_program([module], [module.entry_point("cull")])
)

# %%
# Cull the Gaussian points.
inside_offset_buf = prefix_sum(inside_flag_buf)
inside_offset_cursor = spy.BufferCursor(
    program.reflection.g_inside_offset.type_layout.element_type_layout,
    inside_offset_buf,
)
# Read the last element of the cull prefix buffer to get the number of culled points.
num_viewing = int(inside_offset_cursor[len(inside_offset_cursor) - 1].read())  # type: ignore
num_viewing

# %%
# Culled gaussians
culled_gaussian_buf = device.create_buffer(
    element_count=num_viewing,
    struct_type=program.reflection.g_gaussian_2d_culled,
    usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
)
culled_gaussian_buf

# %%
ker_cull.dispatch(
    thread_count=[len(gaussians), 1, 1],
    vars={
        "g_gaussian_2d": gaussian2d_buf,
        "g_inside_flag": inside_flag_buf,
        "g_inside_offset": inside_offset_buf,
        "g_gaussian_2d_culled": culled_gaussian_buf,
    },
)

# %%
cursor = spy.BufferCursor(
    program.reflection.g_gaussian_2d_culled.type_layout.element_type_layout,
    culled_gaussian_buf,
)
cursor[0]

# %% [markdown]
# # Tiling

# %%
ker_tile = device.create_compute_kernel(
    device.link_program([module], [module.entry_point("computeTile")])
)
ker_build_gs = device.create_compute_kernel(
    device.link_program([module], [module.entry_point("buildGaussianTable")])
)

# %%
num_tile_buf = device.create_buffer(
    element_count=num_viewing,
    struct_type=program.reflection.g_num_tiles,
    usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
)
num_tile_buf

# %%
ker_tile.dispatch(
    thread_count=[num_viewing, 1, 1],
    numViewing=num_viewing,
    vars={
        "g_gaussian_2d_culled": culled_gaussian_buf,
        "g_num_tiles": num_tile_buf,
    },
)

# %%
num_tile_arr = num_tile_buf.to_numpy().view(np.uint32)
num_table_entries = np.sum(num_tile_arr).item()
num_table_entries

# %%
gaussian_table_buf = device.create_buffer(
    element_count=num_table_entries,
    struct_type=program.reflection.g_gaussian_table,
    usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
)
gaussian_table_buf

# %%
num_tile_prefix_buf = prefix_sum(num_tile_buf)

# %%
ker_build_gs.dispatch(
    thread_count=[num_viewing, 1, 1],
    numViewing=num_viewing,
    vars={
        "g_gaussian_2d_culled": culled_gaussian_buf,
        "g_num_tiles_prefix": num_tile_prefix_buf,
        "g_gaussian_table": gaussian_table_buf,
    },
)

# %% [markdown]
# # Sort Tile

# %%
numpy_sort(gaussian_table_buf)

# %%
ker_tile_hist = device.create_compute_kernel(
    device.link_program([module], [module.entry_point("tileHistogram")])
)

# %%
hist_buf = device.create_buffer(
    element_count=2**8,
    struct_type=program.reflection.g_tile_hist_atomic,
    usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
)
# Compute the histogram of the Gaussian table.
ker_tile_hist.dispatch(
    thread_count=[num_table_entries, 1, 1],
    numEntries=num_table_entries,
    vars={
        "g_gaussian_table": gaussian_table_buf,
        "g_tile_hist_atomic": hist_buf,
    },
)

# %%
plt.imshow(hist_buf.to_numpy().view(np.uint32).reshape(16, 16))

# %% [markdown]
# # Rasterize

# %%
ker_duplicate = device.create_compute_kernel(
    device.link_program([module], [module.entry_point("duplicateGaussian")])
)
ker_rasterize = device.create_compute_kernel(
    device.link_program([module], [module.entry_point("rasterize")])
)

# %%
gaussian_2d_sorted_buf = device.create_buffer(
    element_count=num_table_entries,
    struct_type=program.reflection.g_gaussian_2d_sorted,
    usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
)
gaussian_2d_sorted_buf

# %%
ker_duplicate.dispatch(
    thread_count=[num_table_entries, 1, 1],
    vars={
        "g_gaussian_2d_culled": culled_gaussian_buf,
        "g_gaussian_table": gaussian_table_buf,
        "g_gaussian_2d_sorted": gaussian_2d_sorted_buf,
    },
)

# %%
render_target = device.create_texture(
    type=spy.TextureType.texture_2d,
    format=spy.Format.rgba32_float,
    width=camera.sensor_size.x,
    height=camera.sensor_size.y,
    usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
)

depth_target = device.create_texture(
    type=spy.TextureType.texture_2d,
    format=spy.Format.rgba32_float,
    width=camera.sensor_size.x,
    height=camera.sensor_size.y,
    usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
)

num_depth_buf = device.create_texture(
    type=spy.TextureType.texture_2d,
    format=spy.Format.r32_uint,
    width=camera.sensor_size.x,
    height=camera.sensor_size.y,
    usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
)

# %%
hist_arr = hist_buf.to_numpy().view(np.uint32)
hist_offs = np.zeros_like(hist_arr)
hist_offs[1:] = np.cumsum(hist_arr)[:-1]

# %%
tile_offst_buf = device.create_buffer(
    element_count=len(hist_offs),
    struct_type=program.reflection.g_tile_offs,
    usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
)
tile_offst_buf.copy_from_numpy(hist_offs)

# %%
ker_rasterize.dispatch(
    thread_count=[camera.sensor_size.x, camera.sensor_size.y, 1],
    vars={
        "g_camera": camera.to_slang(),
        "g_tile_hist": hist_buf,
        "g_tile_offs": tile_offst_buf,
        "g_gaussian_2d_sorted": gaussian_2d_sorted_buf,
        "g_render_target": render_target,
        "g_depth_target": depth_target,
        "g_num_rendered_gaussians": num_depth_buf,
    },
)

# %%
raw_image = jnp.array(render_target.to_numpy()[:, :, :3])
plt.imshow(raw_image)

# %% [markdown]
# # Image Loss

# %%
image_arr = jnp.array(image)
plt.imshow(image_arr)


# %%
def image_loss(src: jnp.ndarray, dst: jnp.ndarray):
    return jnp.mean((dst - src) ** 2)


# %%
image_loss(raw_image, image_arr)

# %%
# Image grad.
loss_grad = jax.value_and_grad(image_loss)

# %%
loss, render_target_grad = loss_grad(raw_image, image_arr)
plt.imshow(jnp.linalg.norm(render_target_grad, axis=-1))

# %%
render_target_grad.shape

# %%
rg_shape = render_target_grad.shape
render_target_grad = jnp.concatenate(
    (render_target_grad, jnp.zeros((rg_shape[0], rg_shape[1], 1))), axis=-1
)
render_target_grad.shape

# %% [markdown]
# # Backward Gradient into Slang

# %%
ker_bwd_rasterize = device.create_compute_kernel(
    device.link_program([module], [module.entry_point("bwdRasterize")])
)
ker_grad_descent = device.create_compute_kernel(
    device.link_program([module], [module.entry_point("gradDescentGaussian2D")])
)

# %%
grad_texture = device.create_texture(
    type=spy.TextureType.texture_2d,
    format=spy.Format.rgba32_float,
    width=camera.sensor_size.x,
    height=camera.sensor_size.y,
    usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
)

# %%
grad_texture.copy_from_numpy(render_target_grad)

# %%
a_gaussian_2d_sorted_grad_buf = device.create_buffer(
    element_count=num_table_entries,
    struct_type=program.reflection.d_a_gaussian_2d_sorted,
    usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
)
gaussian_2d_sorted_grad_buf = device.create_buffer(
    element_count=num_table_entries,
    struct_type=program.reflection.d_gaussian_2d_sorted,
    usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
)

# %%
num_depth_buf.to_numpy().view(np.uint32)
plt.imshow(num_depth_buf.to_numpy().view(np.uint32))

# %%
ker_bwd_rasterize.dispatch(
    thread_count=[camera.sensor_size.x, camera.sensor_size.y, 1],
    vars={
        "g_camera": camera.to_slang(),
        "g_tile_hist": hist_buf,
        "g_tile_offs": tile_offst_buf,
        "g_gaussian_2d_sorted": gaussian_2d_sorted_buf,
        "g_render_target": render_target,
        "d_render_target": grad_texture,
        "d_a_gaussian_2d_sorted": a_gaussian_2d_sorted_grad_buf,
        "g_num_rendered_gaussians": num_depth_buf,
    },
)

# %%
ker_grad_descent.dispatch(
    thread_count=[num_table_entries, 1, 1],
    lr=1e-2,
    vars={
        "d_a_gaussian_2d_sorted": a_gaussian_2d_sorted_grad_buf,
    },
)

# %% [markdown]
# ## A Very Simple Gradient Descent

# %%
num_epoch = 16
lr = 1e-4


for epoch in range(num_epoch):
    print(f"Epoch {epoch + 1}/{num_epoch}")

    ker_rasterize.dispatch(
        thread_count=[camera.sensor_size.x, camera.sensor_size.y, 1],
        vars={
            "g_camera": camera.to_slang(),
            "g_tile_hist": hist_buf,
            "g_tile_offs": tile_offst_buf,
            "g_gaussian_2d_sorted": gaussian_2d_sorted_buf,
            "g_render_target": render_target,
            "g_depth_target": depth_target,
            "g_num_rendered_gaussians": num_depth_buf,
        },
    )
    raw_image = jnp.array(render_target.to_numpy()[:, :, :3])
    loss, render_target_grad = loss_grad(raw_image, image_arr)

    print(f"Loss after gradient descent: {loss}")

    rg_shape = render_target_grad.shape
    render_target_grad = jnp.concatenate(
        (render_target_grad, jnp.zeros((rg_shape[0], rg_shape[1], 1))), axis=-1
    )

    a_gaussian_2d_sorted_grad_buf.copy_from_numpy(
        np.zeros((a_gaussian_2d_sorted_grad_buf.size,), dtype=np.uint8)
    )

    grad_texture.copy_from_numpy(render_target_grad)

    ker_bwd_rasterize.dispatch(
        thread_count=[camera.sensor_size.x, camera.sensor_size.y, 1],
        vars={
            "g_camera": camera.to_slang(),
            "g_tile_hist": hist_buf,
            "g_tile_offs": tile_offst_buf,
            "g_gaussian_2d_sorted": gaussian_2d_sorted_buf,
            "g_render_target": render_target,
            "d_render_target": grad_texture,
            "d_a_gaussian_2d_sorted": a_gaussian_2d_sorted_grad_buf,
            "g_num_rendered_gaussians": num_depth_buf,
        },
    )

    ker_grad_descent.dispatch(
        thread_count=[num_table_entries, 1, 1],
        lr=lr,
        vars={"d_a_gaussian_2d_sorted": a_gaussian_2d_sorted_grad_buf},
    )

    gd = a_gaussian_2d_sorted_grad_buf.to_numpy().view(np.float32)
    print(f"Max gradient: {np.max(gd)}", f"Min gradient: {np.min(gd)}")


# %%
import matplotlib.pyplot as plt

# histogram of gd
plt.figure(figsize=(10, 5))
plt.hist(gd, bins=100, range=(-0.1, 0.1), density=True)
plt.title("Histogram of Gaussian Gradient")
plt.xlabel("Gradient Value")
plt.ylabel("Density")
plt.xlim(-0.1, 0.1)
plt.grid()
plt.show()


# %%
a_gaussian_2d_sorted_grad_buf.size
