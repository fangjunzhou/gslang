import logging
from typing import Optional, Tuple
import numpy as np
import slangpy as spy
from bvhgs import device


np.random.seed(0)
logger = logging.getLogger(__name__)


mod = device.load_module("radix-sort.slang")
prog_clr = device.link_program([mod], [mod.entry_point("clearHist")])
prog_bld = device.link_program([mod], [mod.entry_point("buildHist")])
prog_sct = device.link_program([mod], [mod.entry_point("scatter")])

k_clear = device.create_compute_kernel(prog_clr)
k_build = device.create_compute_kernel(prog_bld)
k_scatter = device.create_compute_kernel(prog_sct)


def radix_sort(
    src_buf: spy.Buffer,
    bits_per_pass: int = 8,
    total_bits: Optional[int] = None,
) -> Tuple[spy.Buffer, spy.Buffer]:
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
            "bufSize": n,
            "src": src_buf,
            "dst": dst_buf,
            "hist": hist_buf,
            "offs": offs_buf,
        }

        if logger.getEffectiveLevel() <= logging.DEBUG:
            logger.debug(f"State: {state}")

        k_clear.dispatch(
            thread_count=[buckets, 1, 1],
            state=state,
        )

        if logger.getEffectiveLevel() <= logging.DEBUG:
            logger.debug(
                f"Cleared histogram: {hist_buf.to_numpy().view(np.uint32)}"
            )
            logger.debug(
                f"Histogram shape: {hist_buf.to_numpy().view(np.uint32).shape}"
            )

        k_build.dispatch(
            thread_count=[n, 1, 1],
            state=state,
        )

        hist_np = hist_buf.to_numpy().view(np.uint32)
        if logger.getEffectiveLevel() <= logging.DEBUG:
            logger.debug(f"Histogram: {hist_np}")
            logger.debug(f"Histogram sum: {hist_np.sum()}")
        offs_np = np.empty_like(hist_np)
        offs_np[0] = 0
        offs_np[1:] = np.cumsum(hist_np[:-1])
        if logger.getEffectiveLevel() <= logging.DEBUG:
            logger.debug(f"Offsets: {offs_np}")
        offs_buf.copy_from_numpy(offs_np.astype(np.uint32))

        for offs, hist in zip(offs_np, hist_np):
            k_scatter.dispatch(
                thread_count=[n, 1, 1],
                state=state,
                binOffset=offs,
                binSize=hist,
            )

        src_buf, dst_buf = dst_buf, src_buf

    return src_buf, hist_buf


def stable_radix_sort(
    src_buf: spy.Buffer,
    bits_per_pass: int = 8,
    total_bits: int = 40,
) -> Tuple[spy.Buffer, spy.Buffer]:
    """A stable version of radix sort that uses numpy to ensure stability.

    This function is a temporary workaround until the radix sort implementation
    is fixed to be stable.

    :param src_buf: The source buffer containing key-value pairs.
    :param bits_per_pass: Number of bits to use per pass.
    :param total_bits: Total number of bits in the key.
    :return: A tuple of (sorted buffer, histogram buffer).
    """
    sorted_buf, hist_buf = radix_sort(
        src_buf, bits_per_pass, total_bits
    )

    # Fix: Use numpy to ensure stable sort
    table_arr = sorted_buf.to_numpy().view(np.uint64).reshape(-1, 2)
    sort_idx = np.argsort(table_arr[:, 0])
    table_arr = table_arr[sort_idx]
    sorted_buf.copy_from_numpy(table_arr)

    return sorted_buf, hist_buf
