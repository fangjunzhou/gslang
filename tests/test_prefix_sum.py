import numpy as np
import slangpy as spy
import pytest
from bvhgs import device
from bvhgs.prefix_sum import prefix_sum
from typing import cast

@pytest.fixture(params=[1, 2, 16, 64, 65, 96, 32 * 32 + 1, 32 * 32, 32 * 64 + 5, 32 * 32 * 32 + 1, 1024 * 1024, 1024 * 2048 + 5])
def input_range(request):
    """Fixture to provide different input ranges for the test."""
    return request.param

def test_prefix_sum(input_range):

    input_data = np.random.randint(0, 2, size=input_range).tolist()
    
    actual = prefix_sum(input_data)
   
    expected = np.cumsum(input_data).tolist()
    assert actual == expected, f"prefix_sum failed: got {actual}, expected {expected}"
