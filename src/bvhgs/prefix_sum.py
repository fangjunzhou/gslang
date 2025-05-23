import numpy as np
import slangpy as spy
from bvhgs import device
from typing import cast


WAVE = 32


# parallel O(log_32(n)), space complexity O(n/32 + (n/32)^2 + n/32^3 + ... + 1) = O(n)
# dispatch: 2log_32(n)
def prefix_sum(src: spy.Buffer) -> spy.Buffer:
    """
    Hierarchical parallel scan on the GPU.
    Returns a Python list with the prefix sums.
    """

    n = src.size // src.struct_size

    # TODO: Load the module and link the program outside the function.
    mod = device.load_module("prefix-sum.slang")
    prog_scan = device.link_program([mod], [mod.entry_point("wave_scan")])
    prog_add = device.link_program([mod], [mod.entry_point("add_offset")])
    k_scan = device.create_compute_kernel(prog_scan)
    k_add = device.create_compute_kernel(prog_add)

    # dst0 will hold the final result
    dst0 = device.create_buffer(
        element_count=n,
        struct_type=prog_scan.reflection.wave_scan.dst,
        usage=spy.BufferUsage.shader_resource
        | spy.BufferUsage.unordered_access,
    )

    # upward sweep
    level_info = []  # (partial_buf, blocks, dst_buf, length)
    length = n
    cur_src = src
    cur_dst = dst0

    # example: src = [1,2,3,4,5,6,7,8,9,10], wave=4, n = 10

    while True:

        blocks = (length + WAVE - 1) // WAVE
        partial = device.create_buffer(
            element_count=blocks,
            struct_type=prog_scan.reflection.wave_scan.partial,
            usage=spy.BufferUsage.shader_resource
            | spy.BufferUsage.unordered_access,
        )

        k_scan.dispatch(
            thread_count=[blocks * WAVE, 1, 1],
            src=cur_src,
            dst=cur_dst,
            partial=partial,
            n=length,
        )
        level_info.append((partial, blocks, cur_dst, length))

        # example: after while, level info:[
        # (partial = [10, 26, 19], blocks = 3, dst = [1,3,6,10,5,11,18,26,9,19], length = 10),
        # (partial = [55], blocks = 1, dst = [10, 36, 55] (in place), length = 3),
        # ]

        if blocks <= 1:  # reached the top of the pyramid
            break

        cur_src = partial  # next level scans the partials in-place
        cur_dst = partial
        length = blocks

    for partial, blocks, dst_buf, length in reversed(
        level_info
    ):  # example: start from level 2

        # implicitly transform to exclusive scan
        # example: layer2: inclusive = [55], offsets = [0] (exclusive)
        # example: layer1: inclusive = [10, 36, 55], offsets = [0, 10, 36]
        # dispatch add_offset kernel
        # example: layer 2: [55] + 0 -> [55]
        # example: layer 1: [1, 3, 6, 10] + [0] -> [1, 3, 6, 10]; [5, 11, 18, 26] + [10] -> [15, 21, 28, 36]; [9, 19] + [36] -> [45, 55]
        k_add.dispatch(
            thread_count=[blocks * WAVE, 1, 1],
            dst=dst_buf,
            partial=partial,
            n=length,
        )

    return dst0
