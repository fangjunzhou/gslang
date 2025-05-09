import slangpy as spy
import numpy as np
from typing import Tuple
import pytest
import quaternion

import bvhgs
from bvhgs import math_module


def to_slangpy(q: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternion to xyzw quaternion.

    :param q: Input quaternion in wxyz format.
    :return: Converted quaternion in xyzw format.
    """
    return np.ascontiguousarray(q[:, [1, 2, 3, 0]])


def from_slangpy(q: np.ndarray) -> np.ndarray:
    """Convert xyzw quaternion to wxyz quaternion.

    :param q: Input quaternion in xyzw format.
    :return: Converted quaternion in wxyz format.
    """
    return np.ascontiguousarray(q[:, [3, 0, 1, 2]])


@pytest.fixture(params=[16])
def buffer_size(request: pytest.FixtureRequest) -> int:
    return request.param


def test_multiply_quaternion(buffer_size: int):
    """Test quaternion multiplication using slangpy.

    :param buffer_size: Size of the buffer for the test.
    """
    # Generate random unit quaternions in wxyz format.
    q1 = np.random.rand(buffer_size, 4).astype(np.float32)
    q1 /= np.linalg.norm(q1, axis=1, keepdims=True)
    q2 = np.random.rand(buffer_size, 4).astype(np.float32)
    q2 /= np.linalg.norm(q2, axis=1, keepdims=True)
    # Reference result using numpy-quaternion.
    q3_ref = quaternion.as_quat_array(q1) * quaternion.as_quat_array(q2)
    q3_ref = quaternion.as_float_array(q3_ref)
    # Convert to xyzw format.
    q1_spy = to_slangpy(q1)
    q2_spy = to_slangpy(q2)
    quat_mul = math_module.find_function("quat.mul")
    assert quat_mul is not None, "quat.mul function not found."
    q3_spy = quat_mul(q1_spy, q2_spy, _result="numpy")
    # Convert back to wxyz format.
    q3_spy = from_slangpy(q3_spy)
    # Check if the results are close.
    assert np.allclose(
        q3_spy, q3_ref, atol=1e-6
    ), "Quaternion multiplication failed."
