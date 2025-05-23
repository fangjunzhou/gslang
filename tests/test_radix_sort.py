# tests/test_radix_sort.py

import logging
from typing import Dict, Tuple, cast
import numpy as np
import pytest
import slangpy as spy
from bvhgs import device
from bvhgs.radix_sort import radix_sort


logger = logging.getLogger(__name__)


@pytest.fixture(params=[1, 16, 64, 255, 256, 257])
def buf_size(request):
    return request.param


@pytest.fixture(params=[8, 16, 32, 40, 64])
def toal_bits(request):
    return request.param


def test_radix_sort(buf_size, toal_bits):
    keys = np.random.randint(0, 2**toal_bits, size=buf_size, dtype=np.uint64)
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

    logger.info(
        f"src_buf: {src_buf.to_numpy().view(np.uint64).reshape(-1, 2)[:8]}"
    )

    sorted_buf, hist_buf = radix_sort(
        src_buf, bits_per_pass=8, total_bits=toal_bits
    )

    logger.info(
        f"sorted_buf: {sorted_buf.to_numpy().view(np.uint64).reshape(-1, 2)[:8]}"
    )

    dst_cur = spy.BufferCursor(elem_layout, sorted_buf)
    out_keys, out_vals = [], []
    for i in range(buf_size):
        kv = cast(Dict[str, int], dst_cur[i].read())
        out_keys.append(kv["key"])
        out_vals.append(kv["val"])

    logger.info(f"out_keys: {out_keys[:8]}")

    assert sorted(out_keys) == sorted(keys.tolist()), "Key mismatch"
    assert out_keys == sorted(out_keys), "Keys not sorted"

    hist_np = hist_buf.to_numpy().view(np.uint32)
    assert hist_np.sum() == buf_size, "Histogram total count wrong"
    # for bin_val in range(hist_np.shape[0]):
    #     expect = np.count_nonzero((keys & 0xFF) == bin_val)
    #     assert hist_np[bin_val] == expect, f"Hist[{bin_val}] wrong"
