import numpy as np
import slangpy as spy
import pytest
from bvhgs import device
from bvhgs.prefix_sum import prefix_sum
from typing import cast
import pytest_benchmark

@pytest.fixture(
    params=[
        1,
        2,
        16,
        64,
        65,
        96,
        32 * 32 + 1,
        32 * 32,
        32 * 64 + 5,
        32 * 32 * 32 + 1,
        1024 * 1024 + 5,
    ]
)
def input_range(request):
    """Fixture to provide different input ranges for the test."""
    return request.param


def test_prefix_sum(input_range):

    input_data = np.random.randint(0, 2, size=input_range).tolist()
    mod = device.load_module("prefix-sum.slang")
    prog_scan = device.link_program([mod], [mod.entry_point("wave_scan")])
    src_buf = device.create_buffer(
        element_count=input_range,
        struct_type=prog_scan.reflection.wave_scan.src,
        usage=spy.BufferUsage.shader_resource,
    )
    src_cur = spy.BufferCursor(
        prog_scan.reflection.wave_scan.src.type_layout.element_type_layout,
        src_buf,
    )
    for i, v in enumerate(input_data):
        src_cur[i].write(int(v))
    src_cur.apply()

    out_buf = prefix_sum(src_buf)
    dst_layout = (
        prog_scan.reflection.wave_scan.dst.type_layout.element_type_layout
    )
    dst_cur = spy.BufferCursor(dst_layout, out_buf)
    actual = [cast(int, dst_cur[i].read()) for i in range(input_range)]

    expected = np.cumsum(input_data).tolist()
    assert (
        actual == expected
    ), f"prefix_sum failed: got {actual}, expected {expected}"


@pytest.fixture(params=[32, 64, 128, 1024, 2048, 4096, 8192])
def benchmark_buffer_size(request: pytest.FixtureRequest) -> int:
    """Fixture to provide a buffer size for benchmarking.

    :param request: The pytest request object.
    :return: The buffer size.
    """
    return request.param


def test_prefix_sum_benchmark(benchmark, benchmark_buffer_size: int):
    """Benchmark the prefix_sum function with a given buffer size."""
    input_data = np.random.randint(0, 2, size=benchmark_buffer_size).tolist()
    mod = device.load_module("prefix-sum.slang")
    prog_scan = device.link_program([mod], [mod.entry_point("wave_scan")])
    src_buf = device.create_buffer(
        element_count=benchmark_buffer_size,
        struct_type=prog_scan.reflection.wave_scan.src,
        usage=spy.BufferUsage.shader_resource,
    )
    src_cur = spy.BufferCursor(
        prog_scan.reflection.wave_scan.src.type_layout.element_type_layout,
        src_buf,
    )
    for i, v in enumerate(input_data):
        src_cur[i].write(int(v))
    src_cur.apply()

    benchmark(lambda: prefix_sum(src_buf))