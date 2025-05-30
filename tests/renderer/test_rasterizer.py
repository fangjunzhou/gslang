import numpy as np
import pytest
import slangpy as spy
from pyglm import glm
from pytest_benchmark.fixture import BenchmarkFixture

from bvhgs import device
from bvhgs.camera import Camera
from bvhgs.gaussian import GaussianCloud
from bvhgs.prefix_sum import prefix_sum


@pytest.fixture(params=[2**i for i in range(10, 18)])
def benchmark_gaussian_count(request: pytest.FixtureRequest) -> int:
    """Fixture to provide a varying number of gaussians for benchmarking.

    :param request: The pytest request object.
    :return: The number of gaussian points to use.
    """
    return request.param


@pytest.fixture(params=[(800, 600), (1920, 1080)])
def benchmark_render_resolution(request: pytest.FixtureRequest) -> tuple:
    """Fixture to provide different render resolutions for benchmarking.

    :param request: The pytest request object.
    :return: A tuple of (width, height) for the render resolution.
    """
    return request.param


@pytest.fixture
def camera(benchmark_render_resolution: tuple) -> Camera:
    """Create a camera for rendering with specified resolution.

    :param benchmark_render_resolution: The render resolution (width, height).
    :return: A camera positioned to view the scene.
    """
    width, height = benchmark_render_resolution
    # Position camera to view the scene
    position = glm.vec3(10.0, 10.0, 10.0)

    # Look at the origin
    direction = -glm.normalize(position)

    # Create a quaternion for camera rotation
    z_axis = glm.vec3(0.0, 0.0, -1.0)
    rotation_axis = glm.cross(z_axis, direction)
    
    if glm.length(rotation_axis) < 1e-6:
        # If vectors are parallel, use an arbitrary orthogonal axis
        if abs(glm.dot(z_axis, direction) + 1.0) < 1e-6:
            # If they point in opposite directions
            rotation = glm.quat(0.0, 0.0, 1.0, 0.0)  # 180 degree rotation around Y
        else:
            # If they point in same direction
            rotation = glm.quat(1.0, 0.0, 0.0, 0.0)  # identity quaternion
    else:
        rotation_axis = glm.normalize(rotation_axis)
        angle = np.arccos(glm.dot(z_axis, direction))
        rotation = glm.angleAxis(angle, rotation_axis)

    return Camera(
        position=position,
        rotation=rotation,
        sensor_size=glm.uvec2(width, height),  # Use the parametrized resolution
        focal_length=580,
        near_plane=0.01,
        far_plane=100,
    )


@pytest.fixture
def setup_rasterize_data(benchmark_gaussian_count: int, camera: Camera):
    """Set up all the necessary buffers and data for the rasterize kernel.
    
    :param benchmark_gaussian_count: Number of gaussian points to use.
    :param camera: Camera to render from.
    :return: Dictionary containing all buffers and the kernel needed for rasterization.
    """
    # Load required modules
    module = device.load_module("renderer.slang")
    
    # Create a program with the rasterize kernel
    program = device.link_program([module], [])
    
    # Create rasterize kernel
    ker_rasterize = device.create_compute_kernel(
        device.link_program([module], [module.entry_point("rasterize")])
    )
    
    # Create a random gaussian cloud
    cloud = GaussianCloud()
    cloud.randomize(benchmark_gaussian_count, position_var=5.0, scale_offst=0.05)
    
    # Simulate the projection and culling step to get the required data for rasterization
    # Create gaussian 3D buffer
    gaussian_3d_buf = device.create_buffer(
        element_count=benchmark_gaussian_count,
        struct_type=program.reflection.g_gaussian_3d,
        usage=spy.BufferUsage.shader_resource,
    )
    
    # Store gaussian points in the buffer
    gaussian_3d_cursor = spy.BufferCursor(
        program.reflection.g_gaussian_3d.type_layout.element_type_layout,
        gaussian_3d_buf,
    )
    for i in range(benchmark_gaussian_count):
        gaussian_3d_cursor[i].write(cloud[i])
    gaussian_3d_cursor.apply()
    
    # Create gaussian 2D buffer for projection
    gaussian_2d_buf = device.create_buffer(
        element_count=benchmark_gaussian_count,
        struct_type=program.reflection.g_gaussian_2d,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )
    
    # Create inside flag buffer for culling
    inside_flag_buf = device.create_buffer(
        element_count=benchmark_gaussian_count,
        struct_type=program.reflection.g_inside_flag,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )
    
    # Project 3D gaussians to 2D
    ker_proj = device.create_compute_kernel(
        device.link_program([module], [module.entry_point("project")])
    )
    ker_proj.dispatch(
        thread_count=[benchmark_gaussian_count, 1, 1],
        vars={
            "g_camera": camera.to_slang(),
            "g_gaussian_3d": gaussian_3d_buf,
            "g_gaussian_2d": gaussian_2d_buf,
            "g_inside_flag": inside_flag_buf
        }
    )
    
    # Cull gaussians
    inside_offset_buf = prefix_sum(inside_flag_buf)
    inside_offset_cursor = spy.BufferCursor(
        program.reflection.g_inside_offset.type_layout.element_type_layout,
        inside_offset_buf,
    )
    # Get number of visible gaussians
    num_viewing = int(inside_offset_cursor[benchmark_gaussian_count - 1].read()) # pyright: ignore

    if num_viewing == 0:
        # Skip test if no gaussians are visible
        pytest.skip("No gaussians are visible after projection, skipping rasterization test.")
    
    # Create buffer for culled gaussians
    culled_gaussian_buf = device.create_buffer(
        element_count=num_viewing,
        struct_type=program.reflection.g_gaussian_2d_culled,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )
    
    # Run the cull kernel
    ker_cull = device.create_compute_kernel(
        device.link_program([module], [module.entry_point("cull")])
    )
    ker_cull.dispatch(
        thread_count=[benchmark_gaussian_count, 1, 1],
        vars={
            "g_gaussian_2d": gaussian_2d_buf,
            "g_inside_flag": inside_flag_buf,
            "g_inside_offset": inside_offset_buf,
            "g_gaussian_2d_culled": culled_gaussian_buf
        }
    )
    
    # Compute tiles
    ker_tile = device.create_compute_kernel(
        device.link_program([module], [module.entry_point("computeTile")])
    )
    
    # Create num_tile buffer
    num_tile_buf = device.create_buffer(
        element_count=num_viewing,
        struct_type=program.reflection.g_num_tiles,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )
    
    # Compute tiles
    ker_tile.dispatch(
        thread_count=[num_viewing, 1, 1],
        numViewing=num_viewing,
        vars={
            "g_gaussian_2d_culled": culled_gaussian_buf,
            "g_num_tiles": num_tile_buf
        }
    )
    
    # Calculate total number of tile entries
    num_tile_arr = num_tile_buf.to_numpy().view(np.uint32)
    num_table_entries = np.sum(num_tile_arr).item()
    
    # Create gaussian table buffer
    gaussian_table_buf = device.create_buffer(
        element_count=num_table_entries,
        struct_type=program.reflection.g_gaussian_table,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )
    
    # Compute prefix sum for tile offsets
    num_tile_prefix_buf = prefix_sum(num_tile_buf)
    
    # Build gaussian table
    ker_build_gs = device.create_compute_kernel(
        device.link_program([module], [module.entry_point("buildGaussianTable")])
    )
    ker_build_gs.dispatch(
        thread_count=[num_viewing, 1, 1],
        numViewing=num_viewing,
        vars={
            "g_gaussian_2d_culled": culled_gaussian_buf,
            "g_num_tiles_prefix": num_tile_prefix_buf,
            "g_gaussian_table": gaussian_table_buf,
        }
    )
    
    # Sort tile entries (using numpy sort for simplicity in this benchmark)
    from bvhgs.radix_sort import numpy_sort
    numpy_sort(gaussian_table_buf)
    
    # Generate tile histogram
    ker_tile_hist = device.create_compute_kernel(
        device.link_program([module], [module.entry_point("tileHistogram")])
    )
    
    # Create histogram buffer
    hist_buf = device.create_buffer(
        element_count=2**8,
        struct_type=program.reflection.g_tile_hist_atomic,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )
    
    # Compute histogram
    ker_tile_hist.dispatch(
        thread_count=[num_table_entries, 1, 1],
        numEntries=num_table_entries,
        vars={
            "g_gaussian_table": gaussian_table_buf,
            "g_tile_hist_atomic": hist_buf,
        },
    )
    
    # Convert histogram to numpy and calculate offsets
    hist_arr = hist_buf.to_numpy().view(np.uint32)
    hist_offs = np.zeros_like(hist_arr)
    hist_offs[1:] = np.cumsum(hist_arr)[:-1]
    
    # Create tile offset buffer
    tile_offst_buf = device.create_buffer(
        element_count=len(hist_offs),
        struct_type=program.reflection.g_tile_offs,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )
    tile_offst_buf.copy_from_numpy(hist_offs)
    
    # Create sorted gaussians buffer
    gaussian_2d_sorted_buf = device.create_buffer(
        element_count=num_table_entries,
        struct_type=program.reflection.g_gaussian_2d_sorted,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )
    
    # Duplicate gaussians to sorted buffer
    ker_duplicate = device.create_compute_kernel(
        device.link_program([module], [module.entry_point("duplicateGaussian")])
    )
    ker_duplicate.dispatch(
        thread_count=[num_table_entries, 1, 1],
        vars={
            "g_gaussian_table": gaussian_table_buf,
            "g_gaussian_2d_culled": culled_gaussian_buf,
            "g_gaussian_2d_sorted": gaussian_2d_sorted_buf
        }
    )
    
    # Create render target
    render_target = device.create_texture(
        type=spy.TextureType.texture_2d,
        format=spy.Format.rgba32_float,
        width=camera.sensor_size.x,
        height=camera.sensor_size.y,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
    )
    
    # Create depth target
    depth_target = device.create_texture(
        type=spy.TextureType.texture_2d,
        format=spy.Format.rgba32_float,
        width=camera.sensor_size.x,
        height=camera.sensor_size.y,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
    )
    
    # Create rendered gaussian counter buffer
    num_rendered_gaussians = device.create_texture(
        type=spy.TextureType.texture_2d,
        format=spy.Format.r32_uint,
        width=camera.sensor_size.x,
        height=camera.sensor_size.y,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
    )
    
    return {
        "kernel": ker_rasterize,
        "camera": camera,
        "render_target": render_target,
        "depth_target": depth_target,
        "num_rendered_gaussians": num_rendered_gaussians,
        "tile_hist": hist_buf,
        "tile_offs": tile_offst_buf,
        "gaussian_2d_sorted": gaussian_2d_sorted_buf,
        "num_table_entries": num_table_entries,
        "camera_slang": camera.to_slang()
    }


def run_rasterize_kernel(data):
    """Run the rasterize kernel with the provided data.
    
    :param data: The data setup from the setup_rasterize_data fixture.
    """
    # Run the rasterize kernel
    data["kernel"].dispatch(
        thread_count=[data["camera"].sensor_size.x, data["camera"].sensor_size.y, 1],
        vars={
            "g_camera": data["camera_slang"],
            "g_tile_hist": data["tile_hist"],
            "g_tile_offs": data["tile_offs"],
            "g_gaussian_2d_sorted": data["gaussian_2d_sorted"],
            "g_render_target": data["render_target"],
            "g_depth_target": data["depth_target"],
            "g_num_rendered_gaussians": data["num_rendered_gaussians"]
        }
    )


def test_rasterize_benchmark(benchmark: BenchmarkFixture, setup_rasterize_data):
    """Benchmark the rasterize kernel.

    :param benchmark: The benchmark fixture.
    :param setup_rasterize_data: The setup data for the rasterize kernel.
    """
    # Get the number of rounds from benchmark fixture (or default to 5)
    num_rounds = benchmark._min_rounds
    
    # Benchmark the rasterize kernel
    benchmark.pedantic(
        run_rasterize_kernel,  # Target function to benchmark
        args=(setup_rasterize_data,),  # Arguments to pass to the function
        rounds=num_rounds,  # Number of benchmark rounds
        iterations=1  # Number of iterations per round
    )
    
    # Optionally validate results - here we just check that the render target contains data
    rendered_image = setup_rasterize_data["render_target"].to_numpy()
    assert np.any(rendered_image > 0), "Rendered image should contain some pixel data"


@pytest.fixture
def setup_bwd_rasterize_data(setup_rasterize_data):
    """Set up all the necessary buffers and data for the bwdRasterize kernel.
    
    :param setup_rasterize_data: The data setup from the forward rasterize fixture.
    :return: Dictionary containing all buffers and the kernel needed for backward rasterization.
    """
    data = setup_rasterize_data.copy()  # Start with forward rasterize data

    # Run the rasterize kernel
    run_rasterize_kernel(data)
    
    # Load required modules
    module = device.load_module("renderer.slang")
    program = device.link_program([module], [])
    
    # Create bwdRasterize kernel
    ker_bwd_rasterize = device.create_compute_kernel(
        device.link_program([module], [module.entry_point("bwdRasterize")])
    )
    
    # Create the gradient texture (simulating loss gradient passed from optimization)
    grad_texture = device.create_texture(
        type=spy.TextureType.texture_2d,
        format=spy.Format.rgba32_float,
        width=data["camera"].sensor_size.x,
        height=data["camera"].sensor_size.y,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
    )
    
    # Fill gradient texture with simple values for benchmark
    camera = data["camera"]
    shape = (camera.sensor_size.y, camera.sensor_size.x, 4)
    grad_data = np.ones(shape, dtype=np.float32) * 0.1
    grad_texture.copy_from_numpy(grad_data)
    
    # Create buffer for gradient of gaussians
    a_gaussian_2d_sorted_grad_buf = device.create_buffer(
        element_count=data["num_table_entries"],
        struct_type=program.reflection.d_a_gaussian_2d_sorted,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )
    
    # Add new resources to the data dict
    data["bwd_kernel"] = ker_bwd_rasterize
    data["grad_texture"] = grad_texture
    data["a_gaussian_2d_sorted_grad_buf"] = a_gaussian_2d_sorted_grad_buf
    
    return data


def run_bwd_rasterize_kernel(data):
    """Run the bwdRasterize kernel with the provided data.
    
    :param data: The data setup from the setup_bwd_rasterize_data fixture.
    """
    # Run the backward rasterize kernel
    data["bwd_kernel"].dispatch(
        thread_count=[data["camera"].sensor_size.x, data["camera"].sensor_size.y, 1],
        vars={
            "g_camera": data["camera_slang"],
            "g_tile_hist": data["tile_hist"],
            "g_tile_offs": data["tile_offs"],
            "g_gaussian_2d_sorted": data["gaussian_2d_sorted"],
            "g_render_target": data["render_target"],
            "d_render_target": data["grad_texture"],
            "d_a_gaussian_2d_sorted": data["a_gaussian_2d_sorted_grad_buf"],
            "g_num_rendered_gaussians": data["num_rendered_gaussians"]
        }
    )


def test_bwd_rasterize_benchmark(benchmark: BenchmarkFixture, setup_bwd_rasterize_data):
    """Benchmark the bwdRasterize kernel.

    :param benchmark: The benchmark fixture.
    :param setup_bwd_rasterize_data: The setup data for the backward rasterize kernel.
    """
    # Get the number of rounds from benchmark fixture (or default to 5)
    num_rounds = getattr(benchmark, "_min_rounds", 5)
    
    # Benchmark the backward rasterize kernel
    benchmark.pedantic(
        run_bwd_rasterize_kernel,
        args=(setup_bwd_rasterize_data,),
        rounds=num_rounds,
        iterations=1
    )
    
    # Validate that gradients were written
    grad_buffer_data = setup_bwd_rasterize_data["a_gaussian_2d_sorted_grad_buf"].to_numpy().view(np.float32)
    assert np.any(grad_buffer_data != 0), "Backward pass should produce non-zero gradients"
