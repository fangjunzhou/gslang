import slangpy as spy
from bvhgs import device
from bvhgs.prefix_sum import prefix_sum

def radix_sort(src_buf, mask: int = 0x3F):
    mod      = device.load_module("radix-sort.slang")
    prog_clr = device.link_program([mod], [mod.entry_point("clearHist")])
    prog_bld = device.link_program([mod], [mod.entry_point("buildHist")])
    prog_sct = device.link_program([mod], [mod.entry_point("scatter")])

    k_clear   = device.create_compute_kernel(prog_clr)
    k_build   = device.create_compute_kernel(prog_bld)
    k_scatter = device.create_compute_kernel(prog_sct)

    n = src_buf.size // src_buf.struct_size
    padded_n  = ((n + 63) // 64) * 64
    dst_buf = device.create_buffer(
        element_count=n,
        struct_type=prog_clr.reflection.clearHist.dst,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )

    bit_width = mask.bit_length()
    buckets   = mask + 1

    hist_buf = device.create_buffer(
        element_count=buckets,
        struct_type=prog_bld.reflection.buildHist.hist,
        usage=spy.BufferUsage.shader_resource
              | spy.BufferUsage.unordered_access,
    )
    offs_buf = device.create_buffer(
        element_count=buckets,
        struct_type=prog_clr.reflection.clearHist.offs,
        usage=spy.BufferUsage.shader_resource
              | spy.BufferUsage.unordered_access,
    )

    for shift in range(0, 64, bit_width):
        state = {
            "shift":       shift,
            "mask":        mask,
            "src":         src_buf,
            "dst":         dst_buf,
            "hist":        hist_buf,
            "offs":        offs_buf,
        }

        k_clear.dispatch(
            thread_count=[buckets, 1, 1],
            state=state,
        )

        k_build.dispatch(
            thread_count=[padded_n, 1, 1],
            state=state,
            numElements=n,
        )

        offs_new = prefix_sum(hist_buf)
        state["offs"] = offs_new

        k_scatter.dispatch(
            thread_count=[padded_n, 1, 1],
            state=state,
            numElements=n,
        )

        src_buf, dst_buf = dst_buf, src_buf 
        offs_buf = offs_new

    return src_buf
