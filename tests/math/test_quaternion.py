import slangpy as spy
import numpy as np
from typing import Tuple
import pytest
import pytest_benchmark
import quaternion
import logging

import bvhgs
from bvhgs import math_module


logger = logging.getLogger(__name__)


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


def test_quaternion_multiply(buffer_size: int):
    """Test quaternion multiplication using slangpy.

    :param buffer_size: Size of the buffer for the test.
    """
    # Generate random unit quaternions in wxyz format.
    q1 = np.random.randn(buffer_size, 4).astype(np.float32)
    q1 /= np.linalg.norm(q1, axis=1, keepdims=True)
    q2 = np.random.randn(buffer_size, 4).astype(np.float32)
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
    q3 = from_slangpy(q3_spy)
    # Check if the results are close.
    assert np.allclose(
        q3, q3_ref, atol=1e-6
    ), "Quaternion multiplication failed."


def test_quaternion_conjugate(buffer_size: int):
    """Test quaternion conjugation using slangpy.

    :param buffer_size: Size of the buffer for the test.
    """
    # Generate random unit quaternions in wxyz format.
    q1 = np.random.randn(buffer_size, 4).astype(np.float32)
    q1 /= np.linalg.norm(q1, axis=1, keepdims=True)
    # Reference result using numpy-quaternion.
    q2_ref = quaternion.as_quat_array(q1).conjugate()
    q2_ref = quaternion.as_float_array(q2_ref)
    # Convert to xyzw format.
    q1_spy = to_slangpy(q1)
    quat_conjugate = math_module.find_function("quat.conj")
    assert quat_conjugate is not None, "quat.conj function not found."
    q2_spy = quat_conjugate(q1_spy, _result="numpy")
    # Convert back to wxyz format.
    q2 = from_slangpy(q2_spy)
    # Check if the results are close.
    assert np.allclose(q2, q2_ref, atol=1e-6), "Quaternion conjugation failed."


def test_quaternion_inverse(buffer_size: int):
    """Test quaternion inversion using slangpy.

    :param buffer_size: Size of the buffer for the test.
    """
    # Generate random unit quaternions in wxyz format.
    q1 = np.random.randn(buffer_size, 4).astype(np.float32)
    q1 /= np.linalg.norm(q1, axis=1, keepdims=True)
    # Convert to xyzw format.
    q1_spy = to_slangpy(q1)
    quat_inverse = math_module.find_function("quat.inv")
    assert quat_inverse is not None, "quat.inv function not found."
    q2_spy = quat_inverse(q1_spy, _result="numpy")
    # Convert back to wxyz format.
    q2 = from_slangpy(q2_spy)
    # Multiply q1 and q2 to get the identity quaternion.
    q3 = quaternion.as_quat_array(q1) * quaternion.as_quat_array(q2)
    q3 = quaternion.as_float_array(q3)
    # Check if the results are close to the identity quaternion.
    identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    assert np.allclose(q3, identity, atol=1e-6), "Quaternion inversion failed."


def test_quaternion_from_axis_angle(buffer_size: int):
    """Test quaternion from axis-angle conversion using slangpy.

    :param buffer_size: Size of the buffer for the test.
    """
    # Generate random axis-angle representations.
    axis = np.random.randn(buffer_size, 3).astype(np.float32)
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    angle = np.random.rand(buffer_size, 1).astype(np.float32) * 2 * np.pi
    axis_angle = axis * angle
    # Reference result using numpy-quaternion.
    q_ref = quaternion.from_rotation_vector(axis_angle)
    q_ref = quaternion.as_float_array(q_ref)
    # Convert axis-angle to xyzw quaternion.
    from_axis_angle = math_module.find_function("quat.fromAxisAngle")
    assert from_axis_angle is not None, "quat.fromAxisAngle function not found."
    q_spy = from_axis_angle(axis_angle, _result="numpy")
    # Convert back to wxyz format.
    q = from_slangpy(q_spy)
    # Check if the results are close.
    assert np.allclose(
        q, q_ref, atol=1e-6
    ), "Quaternion from axis-angle failed."


def test_quaternion_to_axis_angle(buffer_size: int):
    """Test quaternion to axis-angle conversion using slangpy.

    :param buffer_size: Size of the buffer for the test.
    """
    # Generate random unit quaternions in wxyz format.
    q1 = np.random.randn(buffer_size, 4).astype(np.float32)
    q1 /= np.linalg.norm(q1, axis=1, keepdims=True)
    # Reference result using numpy-quaternion.
    q2_ref = quaternion.as_quat_array(q1)
    axis_angle_ref = quaternion.as_rotation_vector(q2_ref)
    # Convert to xyzw format.
    q1_spy = to_slangpy(q1)
    to_axis_angle = math_module.find_function("quat.asAxisAngle")
    assert to_axis_angle is not None, "quat.asAxisAngle function not found."
    axis_angle = to_axis_angle(q1_spy, _result="numpy")
    # Check if the results are close.
    assert np.allclose(
        axis_angle, axis_angle_ref, atol=1e-6
    ), "Quaternion to axis-angle conversion failed."


def test_rotate_vector(buffer_size: int):
    """Test quaternion rotation of a vector using slangpy.

    :param buffer_size: Size of the buffer for the test.
    """
    # Generate random unit quaternions in wxyz format.
    q1 = np.random.randn(buffer_size, 4).astype(np.float32)
    q1 /= np.linalg.norm(q1, axis=1, keepdims=True)
    # Generate random vectors.
    v = np.random.randn(buffer_size, 3).astype(np.float32)
    # Reference result using numpy-quaternion.
    q1_ref = quaternion.as_quat_array(q1)
    v_quat = quaternion.as_quat_array(
        np.concatenate((np.zeros((buffer_size, 1)), v), axis=1)
    )
    v_ref = quaternion.as_float_array(q1_ref * v_quat * q1_ref.conjugate())[
        :, 1:
    ]
    # Convert to xyzw format.
    q1_spy = to_slangpy(q1)
    v_spy = np.ascontiguousarray(v)
    rotate_vector = math_module.find_function("quat.rotate")
    assert rotate_vector is not None, "quat.rotate function not found."
    v_rotated = rotate_vector(q1_spy, v_spy, _result="numpy")
    # Check if the results are close.
    assert np.allclose(
        v_rotated, v_ref, atol=1e-6
    ), "Quaternion rotation of vector failed."


def test_quaternion_as_rotation_matrix(buffer_size: int):
    """Test quaternion to rotation matrix conversion using slangpy.

    :param buffer_size: Size of the buffer for the test.
    """
    # Generate random unit quaternions in wxyz format.
    q1 = np.random.randn(buffer_size, 4).astype(np.float32)
    q1 /= np.linalg.norm(q1, axis=1, keepdims=True)
    # Reference result using numpy-quaternion.
    q1_ref = quaternion.as_quat_array(q1)
    rot_matrix_ref = quaternion.as_rotation_matrix(q1_ref)
    # Convert to xyzw format.
    q1_spy = to_slangpy(q1)
    as_rotation_matrix = math_module.find_function("quat.asRotMat")
    assert as_rotation_matrix is not None, "quat.asRotMat function not found."
    rot_matrix_buf = as_rotation_matrix(q1_spy)
    rot_matrix_spy = rot_matrix_buf.to_numpy()
    # Check if the results are close.
    logger.debug(rot_matrix_spy)
    logger.debug(rot_matrix_ref)
    assert np.allclose(
        rot_matrix_spy, rot_matrix_ref, atol=1e-6
    ), "Quaternion to rotation matrix conversion failed."


@pytest.fixture(params=[128, 256, 512, 1024, 2048])
def benchmark_buffer_size(request: pytest.FixtureRequest) -> int:
    """Fixture to provide a buffer size for benchmarking.

    :param request: The pytest request object.
    :return: The buffer size.
    """
    return request.param


def test_quaternion_as_rotation_matrix_benchmark(
    benchmark, benchmark_buffer_size: int
):
    """Benchmark quaternion to rotation matrix conversion using slangpy.

    :param benchmark: The benchmark fixture.
    :param benchmark_buffer_size: Size of the buffer for the test.
    """
    # Generate random unit quaternions in wxyz format.
    q1 = np.random.randn(benchmark_buffer_size, 4).astype(np.float32)
    q1 /= np.linalg.norm(q1, axis=1, keepdims=True)
    # Convert to xyzw format.
    q1_spy = to_slangpy(q1)
    as_rotation_matrix = math_module.find_function("quat.asRotMat")
    assert as_rotation_matrix is not None, "quat.asRotMat function not found."
    # Benchmark the conversion.
    benchmark(as_rotation_matrix, q1_spy)
