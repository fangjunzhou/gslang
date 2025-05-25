import numpy as np
import slangpy as spy
from bvhgs import device
from typing import Optional, Tuple

def radix_sort(src_buf: spy.Buffer,
               bits_per_pass: int = 4,
               total_bits: Optional[int] = None,
              ) -> Tuple[spy.Buffer, spy.Buffer]:

    mod     = device.load_module("radix-sort.slang")
    clr     = device.link_program([mod], [mod.entry_point("clearAll")])
    bld     = device.link_program([mod], [mod.entry_point("buildLocal")])
    scanG   = device.link_program([mod], [mod.entry_point("scanGroup")])
    scanGT  = device.link_program([mod], [mod.entry_point("scanGroupTotals")])
    addOffs = device.link_program([mod], [mod.entry_point("addGroupOffsets")])
    sct     = device.link_program([mod], [mod.entry_point("scatterBlock")])
    tile    = device.link_program([mod], [mod.entry_point("tileHist")])

    k_clear    = device.create_compute_kernel(clr)
    k_build    = device.create_compute_kernel(bld)
    k_scanG    = device.create_compute_kernel(scanG)
    k_scanGT   = device.create_compute_kernel(scanGT)
    k_addOffs  = device.create_compute_kernel(addOffs)
    k_scatter  = device.create_compute_kernel(sct)
    k_tile     = device.create_compute_kernel(tile)

    if total_bits is None:
        total_bits = bits_per_pass
    buckets      = 1 << bits_per_pass
    mask         = buckets - 1
    n_records    = src_buf.size // src_buf.struct_size
    block_size   = 64 
    n_blocks     = (n_records + block_size - 1) // block_size
    hist_elems   = n_blocks * buckets

    local_hist_buf = device.create_buffer(
    element_count=hist_elems,
    struct_type=clr.reflection.clearAll.state.localHist,
    usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
    )
    offs_buf = device.create_buffer(
        element_count=hist_elems,
        struct_type=clr.reflection.clearAll.state.offs,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
    )
    group_totals_buf = device.create_buffer(
        element_count=n_blocks,
        struct_type=scanG.reflection.scanGroup.state.groupTotals,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
    )
    group_offsets_buf = device.create_buffer(
        element_count=n_blocks,
        struct_type=scanGT.reflection.scanGroupTotals.state.groupOffsets,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
    )
    global_hist_buf = device.create_buffer(
        element_count=buckets,
        struct_type=tile.reflection.tileHist.state.globalHist,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
    )
    dst_buf = device.create_buffer(
        element_count=n_records,
        struct_type   = sct.reflection.scatterBlock.state.dst,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access
    )

    for shift in range(0, total_bits, bits_per_pass):
        state = {
            "shift":        shift,
            "mask":         mask,
            "bufSize":      n_records,
            "blockSize":    block_size,
            "histSize":     hist_elems,
            "numGroups":    n_blocks,
            "src":          src_buf,
            "dst":          dst_buf,
            "localHist":    local_hist_buf,
            "offs":         offs_buf,
            "groupTotals":  group_totals_buf,
            "groupOffsets": group_offsets_buf,
            "globalHist":   global_hist_buf,
        }

        k_clear.dispatch(thread_count=[(hist_elems + 63)//64, 1, 1], state=state)
        k_build.dispatch(thread_count=[n_blocks, 1, 1], state=state)
        k_scanG.dispatch(  thread_count=[n_blocks, 1, 1], state=state)
        k_scanGT.dispatch( thread_count=[1, 1, 1], state=state)
        k_addOffs.dispatch(thread_count=[n_blocks, 1, 1], state=state)
        k_scatter.dispatch(thread_count=[n_blocks, 1, 1], state=state)
        k_tile.dispatch(thread_count=[(buckets + 63)//64, 1, 1], state=state)

        src_buf, dst_buf = dst_buf, src_buf

    return src_buf, global_hist_buf
