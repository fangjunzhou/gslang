# tests/test_radix_sort.py

import logging
from typing import Dict, Tuple, cast
import numpy as np
import pytest
import slangpy as spy
from bvhgs import device
from bvhgs.radix_sort import radix_sort, numpy_sort, jax_sort
from pytest_benchmark.fixture import BenchmarkFixture


logger = logging.getLogger(__name__)


@pytest.fixture(params=[1, 16, 64, 255, 256, 257, 1024, 4096, 8192])
def buf_size(request):
    return request.param


@pytest.fixture(params=[4, 8, 16, 32, 40, 64])
def total_bits(request):
    return request.param


def test_radix_sort(buf_size, total_bits):
    keys = np.random.randint(0, 2**total_bits, size=buf_size, dtype=np.uint64)
    values = np.arange(buf_size, dtype=np.uint32)

    mod = device.load_module("radix-sort.slang")
    prog_bld = device.link_program([mod], [mod.entry_point("buildHist")])
    tuple_type = prog_bld.reflection.buildHist.state.src
    elem_layout = tuple_type.type_layout.element_type_layout

    src_buf = device.create_buffer(
        element_count=buf_size,
        struct_type=tuple_type,
        usage=spy.BufferUsage.shader_resource
        | spy.BufferUsage.unordered_access,
    )

    src_cur = spy.BufferCursor(elem_layout, src_buf)
    for i, (k, v) in enumerate(zip(keys, values)):
        src_cur[i].write({"key": int(k), "val": int(v)})
    src_cur.apply()

    logger.info(f"src_buf: {src_buf.to_numpy().view(np.uint64).reshape(-1, 2)}")

    sorted_buf = radix_sort(
        src_buf, bits_per_pass=8, total_bits=total_bits, entry_per_thread=256
    )

    logger.info(
        f"sorted_buf: {sorted_buf.to_numpy().view(np.uint64).reshape(-1, 2)}"
    )

    dst_cur = spy.BufferCursor(elem_layout, sorted_buf)
    out_keys, out_vals = [], []
    for i in range(buf_size):
        kv = cast(Dict[str, int], dst_cur[i].read())
        out_keys.append(kv["key"])
        out_vals.append(kv["val"])

    logger.info(f"out_keys: {sorted(out_keys)[:8]}")
    logger.info(f"keys: {sorted(keys)[:8]}")

    assert sorted(out_keys) == sorted(keys.tolist()), "Key mismatch"
    assert out_keys == sorted(out_keys), "Keys not sorted"


@pytest.fixture(params=[2**i for i in range(10, 20)])
def benchmark_buffer_size(request: pytest.FixtureRequest) -> int:
    """Fixture to provide a buffer size for benchmarking.

    :param request: The pytest request object.
    :return: The buffer size.
    """
    return request.param


@pytest.fixture(params=[2, 4, 8])
def benchmark_bits_per_pass(request):
    return request.param


@pytest.fixture(params=[32, 64, 128, 256])
def benchmark_entries_per_thread(request):
    return request.param


def test_radix_sort_benchmark(
    benchmark: BenchmarkFixture,
    benchmark_buffer_size: int,
    benchmark_bits_per_pass: int,
    benchmark_entries_per_thread: int,
):
    """Benchmark the radix sort function with varying buffer sizes.

    :param benchmark: The benchmark fixture.
    :param benchmark_buffer_size: Size of the buffer for the test.
    """
    # Create random key-value pairs
    keys = np.random.randint(
        0, 2**40, size=benchmark_buffer_size, dtype=np.uint64
    )
    values = np.arange(benchmark_buffer_size, dtype=np.uint32)

    # Load module and prepare buffer
    mod = device.load_module("radix-sort.slang")
    prog_bld = device.link_program([mod], [mod.entry_point("buildHist")])
    tuple_type = prog_bld.reflection.buildHist.state.src
    elem_layout = tuple_type.type_layout.element_type_layout

    src_buf = device.create_buffer(
        element_count=benchmark_buffer_size,
        struct_type=tuple_type,
        usage=spy.BufferUsage.shader_resource
        | spy.BufferUsage.unordered_access,
    )

    # Fill the buffer with data
    src_cur = spy.BufferCursor(elem_layout, src_buf)
    for i, (k, v) in enumerate(zip(keys, values)):
        src_cur[i].write({"key": int(k), "val": int(v)})
    src_cur.apply()

    # Benchmark the radix sort
    benchmark(
        radix_sort,
        src_buf,
        benchmark_bits_per_pass,
        40,
        benchmark_entries_per_thread,
    )


def test_numpy_sort_benchmark(
    benchmark: BenchmarkFixture, benchmark_buffer_size: int
):
    """Benchmark sorting using NumPy only, without GPU radix sort.

    This benchmark helps compare pure CPU-based sorting against GPU-based methods.

    :param benchmark: The benchmark fixture.
    :param benchmark_buffer_size: Size of the buffer for the test.
    """
    # Create random key-value pairs
    keys = np.random.randint(
        0, 2**40, size=benchmark_buffer_size, dtype=np.uint64
    )
    values = np.arange(benchmark_buffer_size, dtype=np.uint32)

    # Load module and prepare buffer
    mod = device.load_module("radix-sort.slang")
    prog_bld = device.link_program([mod], [mod.entry_point("buildHist")])
    tuple_type = prog_bld.reflection.buildHist.state.src
    elem_layout = tuple_type.type_layout.element_type_layout

    src_buf = device.create_buffer(
        element_count=benchmark_buffer_size,
        struct_type=tuple_type,
        usage=spy.BufferUsage.shader_resource
        | spy.BufferUsage.unordered_access,
    )

    # Fill the buffer with data
    src_cur = spy.BufferCursor(elem_layout, src_buf)
    for i, (k, v) in enumerate(zip(keys, values)):
        src_cur[i].write({"key": int(k), "val": int(v)})
    src_cur.apply()

    # Benchmark the NumPy-only sorting approach
    benchmark(
        numpy_sort,
        src_buf,
    )


def test_jax_sort_benchmark(
    benchmark: BenchmarkFixture, benchmark_buffer_size: int
):
    """Benchmark sorting using NumPy only, without GPU radix sort.

    This benchmark helps compare pure CPU-based sorting against GPU-based methods.

    :param benchmark: The benchmark fixture.
    :param benchmark_buffer_size: Size of the buffer for the test.
    """
    # Create random key-value pairs
    keys = np.random.randint(
        0, 2**40, size=benchmark_buffer_size, dtype=np.uint64
    )
    values = np.arange(benchmark_buffer_size, dtype=np.uint32)

    # Load module and prepare buffer
    mod = device.load_module("radix-sort.slang")
    prog_bld = device.link_program([mod], [mod.entry_point("buildHist")])
    tuple_type = prog_bld.reflection.buildHist.state.src
    elem_layout = tuple_type.type_layout.element_type_layout

    src_buf = device.create_buffer(
        element_count=benchmark_buffer_size,
        struct_type=tuple_type,
        usage=spy.BufferUsage.shader_resource
        | spy.BufferUsage.unordered_access,
    )

    # Fill the buffer with data
    src_cur = spy.BufferCursor(elem_layout, src_buf)
    for i, (k, v) in enumerate(zip(keys, values)):
        src_cur[i].write({"key": int(k), "val": int(v)})
    src_cur.apply()

    # Benchmark the NumPy-only sorting approach
    benchmark(jax_sort, src_buf)
