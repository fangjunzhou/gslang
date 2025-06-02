import logging
from typing import Optional, Tuple
import numpy as np
import slangpy as spy
import jax.numpy as jnp

from bvhgs import device


np.random.seed(0)
logger = logging.getLogger(__name__)


mod = device.load_module("radix-sort.slang")
prog_clr = device.link_program([mod], [mod.entry_point("clearHist")])
prog_bld = device.link_program([mod], [mod.entry_point("buildHist")])
prog_scan = device.link_program([mod], [mod.entry_point("waveScan")])
prog_add = device.link_program([mod], [mod.entry_point("addOffset")])
prog_sct = device.link_program([mod], [mod.entry_point("scatter")])

k_clear = device.create_compute_kernel(prog_clr)
k_build = device.create_compute_kernel(prog_bld)
k_scan = device.create_compute_kernel(prog_scan)
k_add = device.create_compute_kernel(prog_add)
k_scatter = device.create_compute_kernel(prog_sct)

WAVE = 32
def prefix_sum_inplace(histBuf: spy.Buffer, offsBuf: spy.Buffer, state: dict, bucket: int) -> None:
    level_info = []
    cur_src = histBuf
    cur_dst = offsBuf
    cur_len = state["numThreads"]
    while True:
        blocks = (cur_len + WAVE - 1) // WAVE
        partialBuf = device.create_buffer(
            element_count=blocks * bucket,
            struct_type=prog_scan.reflection.waveScan.partial,
            usage=spy.BufferUsage.shader_resource
            | spy.BufferUsage.unordered_access,
        )
        k_scan.dispatch(
            thread_count=[blocks * WAVE, 1, 1],
            hist=cur_src,
            offs=cur_dst,
            partial=partialBuf,
            state=state,
            n=cur_len,
        )
        level_info.append((partialBuf, blocks, cur_dst, cur_len))
        if blocks <= 1:
            partial_buf_array = partialBuf.to_numpy().view(np.uint32)
            partial_buf_array = partial_buf_array.reshape(blocks, bucket)
            partial_buf_array = (
                np.cumsum(partial_buf_array, axis=0).astype(np.uint32)
                - partial_buf_array
            )
            partialBuf.copy_from_numpy(partial_buf_array)
            break
        cur_src = partialBuf
        cur_dst = partialBuf
        cur_len = blocks

    for partialBuf, blocks, dst_buf, length_in_level in reversed(level_info):
        k_add.dispatch(
            thread_count=[blocks * WAVE, 1, 1],
            state=state,
            offs=dst_buf,
            partial=partialBuf,
            n=length_in_level,
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
    numThreads = (n + entry_per_thread - 1) // entry_per_thread
    dst_buf = device.create_buffer(
        element_count=n,
        struct_type=prog_clr.reflection.clearHist.state.dst,
        usage=spy.BufferUsage.shader_resource
        | spy.BufferUsage.unordered_access,
    )
    # Todo: change histogram buffer to per thread histogram
    hist_buf = device.create_buffer(
        element_count=numThreads * buckets,
        struct_type=prog_bld.reflection.buildHist.state.hist,
        usage=spy.BufferUsage.shader_resource
        | spy.BufferUsage.unordered_access,
    )

    offs_buf = device.create_buffer(
        element_count=numThreads * buckets,
        struct_type=prog_bld.reflection.buildHist.state.offs,
        usage=spy.BufferUsage.shader_resource
        | spy.BufferUsage.unordered_access,
    )

    # todo: add global offsets buffer
    global_offs_buf = device.create_buffer(
        element_count=buckets,
        struct_type=prog_clr.reflection.clearHist.state.globalOffs,
        usage=spy.BufferUsage.shader_resource
        | spy.BufferUsage.unordered_access,
    )

    for shift in range(0, total_bits, bits_per_pass):
        state = {
            "shift": shift,
            "bucket": buckets,
            "bufSize": n,
            "numThreads": numThreads,
            "entriesPerThread": entry_per_thread,
            "src": src_buf,
            "dst": dst_buf,
            "hist": hist_buf,
            "offs": offs_buf,
            "globalOffs": global_offs_buf,
        }

        k_clear.dispatch(
            thread_count=[numThreads, 1, 1],
            state=state,
        )

        k_build.dispatch(
            thread_count=[numThreads, 1, 1],
            state=state,
        )

        prefix_sum_inplace(hist_buf, offs_buf, state, buckets)

        hist_cursor = spy.BufferCursor(
            prog_bld.reflection.buildHist.state.hist.type_layout.element_type_layout,
            hist_buf,
        )

        offs_cursor = spy.BufferCursor(
            prog_bld.reflection.buildHist.state.offs.type_layout.element_type_layout,
            offs_buf,
        )

        hist_length = len(hist_cursor)
        offs_length = len(offs_cursor)
        # get the last bucket's offset and histogram
        last_hist = []
        last_offs = []
        for i in range(hist_length - buckets, hist_length):
            last_hist.append(hist_cursor[i].read())
        for i in range(offs_length - buckets, offs_length):
            last_offs.append(offs_cursor[i].read())
        global_hist = [last_hist[i] + last_offs[i] for i in range(buckets)]
        global_offs = np.cumsum(global_hist, dtype=np.uint32) - global_hist
        global_offs_buf.copy_from_numpy(global_offs.astype(np.uint32))

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
