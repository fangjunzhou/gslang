import slangpy as spy
import numpy as np
import pytest
from bvhgs import device
from pytest_benchmark.fixture import BenchmarkFixture


def test_tile_computation():
    """Test the tile computation kernel with culled Gaussian2D data."""
    module = device.load_module("renderer.slang")
    program_tile = device.link_program(
        [module], [module.entry_point("computeTile")]
    )
    ker_tile = device.create_compute_kernel(program_tile)

    # Create Gaussian2D culled buffer
    num_gaussians = 10
    gaussian_2d_culled_buf = device.create_buffer(
        element_count=num_gaussians,
        struct_type=program_tile.reflection.g_gaussian_2d_culled,
        usage=spy.BufferUsage.shader_resource
        | spy.BufferUsage.unordered_access,
    )

    # Fill the buffer with mock data (only position and covariance are used)
    gaussian_cursor = spy.BufferCursor(
        program_tile.reflection.g_gaussian_2d_culled.type_layout.element_type_layout,
        gaussian_2d_culled_buf,
    )
    for i in range(num_gaussians):
        gaussian_cursor[i].write(
            {
                "position": spy.float3(i * 0.1, i * 0.1, 0.5),
                "covariance": spy.float2x2([0.01, 0, 0, 0.01]),
            }
        )
    gaussian_cursor.apply()

    # Create num_tiles buffer
    num_tiles_buf = device.create_buffer(
        element_count=num_gaussians,
        struct_type=program_tile.reflection.g_num_tiles,
        usage=spy.BufferUsage.shader_resource
        | spy.BufferUsage.unordered_access,
    )

    # Dispatch the kernel
    ker_tile.dispatch(
        thread_count=[num_gaussians, 1, 1],
        vars={
            "g_gaussian_2d_culled": gaussian_2d_culled_buf,
            "g_num_tiles": num_tiles_buf,
        },
    )

    # Validate the results
    num_tiles = num_tiles_buf.to_numpy().view(np.uint32)
    assert np.all(
        num_tiles > 0
    ), "All Gaussian2D entries should have at least one tile."

    print("Tile computation test passed.")


@pytest.fixture(params=[2**i for i in range(10, 20)])
def benchmark_buffer_size(request: pytest.FixtureRequest) -> int:
    """Fixture to provide a buffer size for benchmarking.

    :param request: The pytest request object.
    :return: The buffer size.
    """
    return request.param


def test_tile_benchmark(
    benchmark: BenchmarkFixture, benchmark_buffer_size: int
):
    """Benchmark the computeTile kernel with varying buffer sizes.

    :param benchmark: The benchmark fixture.
    :param benchmark_buffer_size: Size of the buffer for the test.
    """
    module = device.load_module("renderer.slang")
    program_tile = device.link_program(
        [module], [module.entry_point("computeTile")]
    )
    ker_tile = device.create_compute_kernel(program_tile)

    # Load the Gaussian2D initialization kernel
    gaussian_module = device.load_module("tests.slang")
    program_load = device.link_program(
        [gaussian_module], [gaussian_module.entry_point("loadGaussian2D")]
    )
    ker_load = device.create_compute_kernel(program_load)

    # Create buffers for Gaussian2D initialization using NDBuffer
    positions_buf = spy.NDBuffer(
        device, dtype=spy.float3, shape=(benchmark_buffer_size,)
    )
    covariances_buf = spy.NDBuffer(
        device, dtype=spy.float2x2, shape=(benchmark_buffer_size,)
    )
    # Create Gaussian2D buffer using device.create_buffer
    gaussian_2d_buf = device.create_buffer(
        element_count=benchmark_buffer_size,
        struct_type=program_load.reflection.loadGaussian2D.gaussians.type_layout.element_type_layout,
        usage=spy.BufferUsage.shader_resource
        | spy.BufferUsage.unordered_access,
    )

    # Fill positions and covariances with random data
    positions = np.random.uniform(
        -1.0, 1.0, size=(benchmark_buffer_size, 3)
    ).astype(np.float32)
    random_matrices = np.random.uniform(
        -0.1, 0.1, size=(benchmark_buffer_size, 2, 2)
    )
    covariances = (
        np.einsum(
            "bij,bjk->bik",
            random_matrices,
            np.transpose(random_matrices, axes=(0, 2, 1)),
        )
        + np.eye(2) * 1e-3
    )
    positions_buf.copy_from_numpy(positions)
    covariances_buf.copy_from_numpy(covariances.astype(np.float32))

    # Dispatch the loadGaussian2D kernel
    ker_load.dispatch(
        thread_count=[benchmark_buffer_size, 1, 1],
        positions=positions_buf.storage,
        covariances=covariances_buf.storage,
        gaussians=gaussian_2d_buf,
    )

    # Create num_tiles buffer
    num_tiles_buf = device.create_buffer(
        element_count=benchmark_buffer_size,
        struct_type=program_tile.reflection.g_num_tiles.type_layout.element_type_layout,
        usage=spy.BufferUsage.shader_resource
        | spy.BufferUsage.unordered_access,
    )

    # Benchmark the kernel dispatch
    benchmark(
        ker_tile.dispatch,
        thread_count=[benchmark_buffer_size, 1, 1],
        vars={
            "g_gaussian_2d_culled": gaussian_2d_buf,
            "g_num_tiles": num_tiles_buf,
        },
    )
