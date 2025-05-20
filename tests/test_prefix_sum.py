import numpy as np
import slangpy as spy
import pytest
from bvhgs import device

def test_prefix_sum():
    module = device.load_module("prefix-sum.slang")
    program = device.link_program(
        [module],
        [module.entry_point("wave_prefix_sum")]
    )
    kernel = device.create_compute_kernel(program)

    
    input_data = list(range(1, 65))
    n = len(input_data)

    
    src_buf = device.create_buffer(
        element_count=n,
        struct_type=program.reflection.src,
        usage=spy.BufferUsage.shader_resource,
    )
    src_cursor = spy.BufferCursor(
        program.reflection.src.type_layout.element_type_layout,
        src_buf
    )
    for i, v in enumerate(input_data):
        src_cursor[i].write(int(v))
    src_cursor.apply()

   
    dst_buf = device.create_buffer(
        element_count=n,
        struct_type=program.reflection.dst,
        usage=(
            spy.BufferUsage.shader_resource
            | spy.BufferUsage.unordered_access
        ),
    )

    # 5) Dispatch the compute shader
    kernel.dispatch(
        thread_count=[n, 1, 1],
        vars={
            "src": src_buf,
            "dst": dst_buf
        }
    )

    
    dst_cursor = spy.BufferCursor(
        program.reflection.dst.type_layout.element_type_layout,
        dst_buf
    )
    actual = [dst_cursor[i].read() for i in range(n)]

    # 7) Verify against NumPy’s cumsum
    expected = np.cumsum(input_data).tolist()
    assert actual == expected, f"prefix_sum failed: got {actual}, expected {expected}"
