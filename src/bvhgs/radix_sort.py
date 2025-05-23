from typing import Optional, Tuple
import numpy as np
import slangpy as spy
from bvhgs import device


def radix_sort(
    src_buf: spy.Buffer,
    bits_per_pass: int = 8,
    total_bits: Optional[int] = None,
) -> Tuple[spy.Buffer, spy.Buffer]:
    mod = device.load_module("radix-sort.slang")
    prog_clr = device.link_program([mod], [mod.entry_point("clearHist")])
    prog_bld = device.link_program([mod], [mod.entry_point("buildHist")])
    prog_sct = device.link_program([mod], [mod.entry_point("scatter")])

    k_clear = device.create_compute_kernel(prog_clr)
    k_build = device.create_compute_kernel(prog_bld)
    k_scatter = device.create_compute_kernel(prog_sct)

    if total_bits is None:
        total_bits = 8
    n = src_buf.size // src_buf.struct_size
    buckets = 1 << bits_per_pass
    mask = buckets - 1

    dst_buf = device.create_buffer(
        element_count=n,
        struct_type=prog_clr.reflection.clearHist.state.dst,
        usage=spy.BufferUsage.shader_resource
        | spy.BufferUsage.unordered_access,
    )

    hist_buf = device.create_buffer(
        element_count=buckets,
        struct_type=prog_bld.reflection.buildHist.state.hist,
        usage=spy.BufferUsage.shader_resource
        | spy.BufferUsage.unordered_access,
    )
    offs_buf = device.create_buffer(
        element_count=buckets,
        struct_type=prog_clr.reflection.clearHist.state.offs,
        usage=spy.BufferUsage.shader_resource
        | spy.BufferUsage.unordered_access,
    )

    for shift in range(0, total_bits, bits_per_pass):
        state = {
            "shift": shift,
            "mask": mask,
            "src": src_buf,
            "dst": dst_buf,
            "hist": hist_buf,
            "offs": offs_buf,
        }

        k_clear.dispatch(
            thread_count=[buckets, 1, 1],
            state=state,
        )

        k_build.dispatch(
            thread_count=[n, 1, 1],
            state=state,
        )

        hist_np = hist_buf.to_numpy()
        offs_np = np.empty_like(hist_np)
        offs_np[0] = 0
        offs_np[1:] = np.cumsum(hist_np[:-1])
        offs_buf.copy_from_numpy(offs_np)

        k_scatter.dispatch(
            thread_count=[n, 1, 1],
            state=state,
        )

        src_buf, dst_buf = dst_buf, src_buf

    return src_buf, hist_buf
