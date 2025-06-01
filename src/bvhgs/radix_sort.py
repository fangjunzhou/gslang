import logging
from typing import Optional, Tuple
import numpy as np
import slangpy as spy
import jax.numpy as jnp

from bvhgs import device


np.random.seed(0)
logger = logging.getLogger(__name__)


mod = device.load_module("radix-sort.slang")
prog_clr   = device.link_program([mod], [mod.entry_point("clearHist")])
prog_bld   = device.link_program([mod], [mod.entry_point("buildHist")])
prog_scan  = device.link_program([mod], [mod.entry_point("waveScan")])
prog_add   = device.link_program([mod], [mod.entry_point("addOffset")])
prog_sct   = device.link_program([mod], [mod.entry_point("scatter")])

k_clear  = device.create_compute_kernel(prog_clr)
k_build  = device.create_compute_kernel(prog_bld)
k_scan   = device.create_compute_kernel(prog_scan)
k_add    = device.create_compute_kernel(prog_add)
k_scatter = device.create_compute_kernel(prog_sct)

WAVE = 32
def prefix_sum_inplace(buf: spy.Buffer, length: int) -> None:
    dst0 = buf
    level_info = []
    cur_src = buf
    cur_dst = dst0
    cur_len = length
    while True:
        blocks = (cur_len + WAVE - 1) // WAVE
        partialBuf = device.create_buffer(
            element_count=blocks,
            struct_type=prog_scan.reflection.waveScan.partial,
            usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
        )
        k_scan.dispatch(
            thread_count=[blocks * WAVE, 1, 1],
            histOffs=cur_src,
            partial=partialBuf,
            n=cur_len
        )
        level_info.append((partialBuf, blocks, cur_dst, cur_len))
        if blocks <= 1:
            break
        cur_src = partialBuf
        cur_dst = partialBuf
        cur_len = blocks

    for partialBuf, blocks, dst_buf, length_in_level in reversed(level_info):
        k_add.dispatch(
            thread_count=[blocks * WAVE, 1, 1],
            histOffs=dst_buf,
            partial=partialBuf,
            n=length_in_level
        )

def radix_sort(
    src_buf: spy.Buffer,
    bits_per_pass: int = 8,
    total_bits: Optional[int] = None,
    entry_per_thread: int = 64,
) -> spy.Buffer:
    if total_bits is None:
        total_bits = bits_per_pass

    n = src_buf.size // src_buf.struct_size
    buckets = 1 << bits_per_pass
    mask = buckets - 1
    numThreads = (n + entry_per_thread - 1) // entry_per_thread
    numWaves = (numThreads + WAVE - 1) // WAVE

    dst_buf = device.create_buffer(
        element_count=n,
        struct_type=prog_clr.reflection.clearHist.state.dst,
        usage=spy.BufferUsage.shader_resource
        | spy.BufferUsage.unordered_access,
    )
    # Todo: change histogram buffer to per thread histogram
    hist_buf = device.create_buffer(
        element_count=numThreads * buckets,
        struct_type=prog_bld.reflection.buildHist.state.histOffs,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )
    # todo: add global offsets buffer
    global_offs_buf = device.create_buffer(
        element_count=buckets,
        struct_type=prog_clr.reflection.clearHist.state.globalOffs,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )

    partial_buf = device.create_buffer(
        element_count=buckets * numWaves,
        struct_type=prog_scan.reflection.waveScan.partial,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )

    for shift in range(0, total_bits, bits_per_pass):
        state = {
            "state.shift": shift,
            "state.bucket": buckets,
            "state.bufSize": n,
            "state.numThreads": numThreads,
            "state.entriesPerThread": entry_per_thread,
            "state.src": src_buf,
            "state.dst": dst_buf,
            "state.histOffs": hist_buf,
            "state.globalOffs": global_offs_buf,
        }



        if logger.getEffectiveLevel() <= logging.DEBUG:
            logger.debug(f"State: {state}")

        k_clear.dispatch(
            thread_count=[numThreads, 1, 1],
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
            thread_count=[numThreads, 1, 1],
            state=state,
        )

        k_scan.dispatch(
            thread_count=[numThreads * WAVE, 1, 1],
            state=state,
            histOffs=hist_buf,
            partial=partial_buf,
        )

        partial_np = partial_buf.to_numpy().view(np.uint32)
        for b in range(buckets):
            start = b * numWaves
            end   = start + numWaves
            seg = partial_np[start:end]
            new_seg = np.empty_like(seg)
            new_seg[0] = 0
            if numWaves > 1:
                new_seg[1:] = np.cumsum(seg[:-1])
            partial_np[start:end] = new_seg
        partial_buf.copy_from_numpy(partial_np.astype(np.uint32))

        k_add.dispatch(
            thread_count=[numThreads * WAVE, 1, 1],
            state=state,
            histOffs=hist_buf,
            partial=partial_buf,
        )
        k_scatter.dispatch(
            thread_count=[numThreads, 1, 1],
            state=state,
        )
        

        src_buf, dst_buf = dst_buf, src_buf

    return src_buf


def numpy_sort(
    buf: spy.Buffer,
):
    """Sorts a buffer using numpy for stability."""
    table_arr = buf.to_numpy().view(np.uint64).reshape(-1, 2)
    sort_idx = np.argsort(table_arr[:, 0])
    table_arr = table_arr[sort_idx]

    buf.copy_from_numpy(table_arr)


def jax_sort(buf: spy.Buffer):
    """Sorts a buffer using jax."""
    table_arr = buf.to_numpy().view(np.uint64).reshape(-1, 2)
    sort_idx = jnp.argsort(table_arr[:, 0])
    table_arr = table_arr[sort_idx]

    buf.copy_from_numpy(table_arr)

