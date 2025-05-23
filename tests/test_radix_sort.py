# tests/test_radix_sort.py

from typing import Dict, Tuple, cast
import numpy as np
import pytest
import slangpy as spy
from bvhgs import device
from bvhgs.radix_sort import radix_sort


@pytest.fixture(params=[1, 16, 64, 255, 256, 257])
def n(request):
    return request.param


def test_radix_sort(n):
    keys = np.random.randint(0, 256, size=n, dtype=np.uint64)
    values = np.arange(n, dtype=np.uint32)

    mod = device.load_module("radix-sort.slang")
    prog_bld = device.link_program([mod], [mod.entry_point("buildHist")])
    tuple_type = prog_bld.reflection.buildHist.state.src
    elem_layout = tuple_type.type_layout.element_type_layout

    src_buf = device.create_buffer(
        element_count=n,
        struct_type=tuple_type,
        usage=spy.BufferUsage.shader_resource
        | spy.BufferUsage.unordered_access,
    )

    src_cur = spy.BufferCursor(elem_layout, src_buf)
    for i, (k, v) in enumerate(zip(keys, values)):
        src_cur[i].write({"key": int(k), "val": int(v)})
    src_cur.apply()

    sorted_buf, hist_buf = radix_sort(src_buf)

    dst_cur = spy.BufferCursor(elem_layout, sorted_buf)
    out_keys, out_vals = [], []
    for i in range(n):
        kv = cast(Dict[str, int], dst_cur[i].read())
        out_keys.append(kv["key"])
        out_vals.append(kv["val"])

    assert sorted(out_keys) == sorted(keys), "Key mismatch"
    assert out_keys == sorted(out_keys), "Keys not sorted"

    hist_np = hist_buf.to_numpy()
    assert hist_np.sum() == n, "Histogram total count wrong"
    for bin_val in range(hist_np.shape[0]):
        expect = np.count_nonzero((keys & 0xFF) == bin_val)
        assert hist_np[bin_val] == expect, f"Hist[{bin_val}] wrong"
