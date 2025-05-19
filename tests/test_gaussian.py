from typing import Tuple
import slangpy as spy
import pytest
import numpy as np

import bvhgs
from bvhgs import gaussian_module


@pytest.fixture(params=[(16,)])
def buffer_shape(request: pytest.FixtureRequest) -> Tuple[int]:
    return request.param


def test_gaussian3d_init_default(buffer_shape: Tuple[int]):
    """Test Gaussian3D default constructor.

    :param buffer_shape: Shape of the buffer.
    """
    gaussian_buf = spy.InstanceBuffer(
        struct=gaussian_module.Gaussian3D.as_struct(), shape=buffer_shape
    )

    gaussian_buf.construct()
    # Check default constructor.
    cursor = gaussian_buf.get_this().cursor()
    for i in range(buffer_shape[0]):
        gaussian = cursor[i].read()
        assert gaussian["position"] == spy.float3(0, 0, 0)
        assert gaussian["rotation"] == spy.float4(0, 0, 0, 1)
        assert gaussian["scale"] == spy.float3(1, 1, 1)
        assert gaussian["color"] == spy.float3(1, 0, 1)
        assert gaussian["opacity"] == 1


def test_gaussian3d_init_custom(buffer_shape: Tuple[int]):
    """Test Gaussian3D custom constructor.

    :param buffer_shape: Shape of the buffer.
    """
    gaussian_buf = spy.InstanceBuffer(
        struct=gaussian_module.Gaussian3D.as_struct(), shape=buffer_shape
    )
    default_sh = [spy.float3(0, 0, 0)] * 15

    gaussian_buf.construct(
        position=spy.float3(1, 2, 3),
        rotation=spy.float4(1, 0, 0, 0),
        scale=spy.float3(4, 5, 6),
        color=spy.float3(0.5, 0.5, 0.5),
        opacity=0.5,
        shCoeffs=default_sh,
    )

    # Check custom constructor.
    cursor = gaussian_buf.get_this().cursor()
    for i in range(buffer_shape[0]):
        gaussian = cursor[i].read()
        assert gaussian["position"] == spy.float3(1, 2, 3)
        assert gaussian["rotation"] == spy.float4(1, 0, 0, 0)
        assert gaussian["scale"] == spy.float3(4, 5, 6)
        assert gaussian["color"] == spy.float3(0.5, 0.5, 0.5)
        assert gaussian["opacity"] == 0.5


def test_gaussian3d_covariance(buffer_shape: Tuple[int]):
    """Test Gaussian3D covariance matrix calculation."""
    gaussian_buf = spy.InstanceBuffer(
        struct=gaussian_module.Gaussian3D.as_struct(), shape=buffer_shape
    )

    default_sh = [spy.float3(0, 0, 0)] * 15
    gaussian_buf.construct(
        position=spy.float3(0, 0, 0),
        rotation=spy.float4(0, 0, 0, 1),
        scale=spy.float3(0, 0, 0),
        color=spy.float3(0, 0, 0),
        opacity=0.0,
        shCoeffs=default_sh,
    )

    cov_buf = gaussian_buf.covariance()
    cursor = cov_buf.cursor()

    for i in range(buffer_shape[0]):
        covariance = cursor[i].read()

        assert covariance[0] == spy.float3(1, 0, 0)
        assert covariance[1] == spy.float3(0, 1, 0)
        assert covariance[2] == spy.float3(0, 0, 1)


def test_gaussian3d_covariance_matrix(buffer_shape: Tuple[int]):
    """Test Gaussian3D covariance matrix correctness."""
    gaussian_buf = spy.InstanceBuffer(
        struct=gaussian_module.Gaussian3D.as_struct(), shape=buffer_shape
    )

    scale_log = spy.float3(1, 2, 3)
    default_sh = [spy.float3(0, 0, 0)] * 15

    gaussian_buf.construct(
        position=spy.float3(0, 0, 0),
        rotation=spy.float4(0, 0, 0, 1),
        scale=scale_log,
        color=spy.float3(0, 0, 0),
        opacity=1.0,
        shCoeffs=default_sh,
    )

    cov_buf = gaussian_buf.covariance()
    cursor = cov_buf.cursor()

    for i in range(buffer_shape[0]):
        covariance = cursor[i].read()

        actual = np.array(
            [
                [covariance[0][0], covariance[0][1], covariance[0][2]],
                [covariance[1][0], covariance[1][1], covariance[1][2]],
                [covariance[2][0], covariance[2][1], covariance[2][2]],
            ]
        )

        decoded_scale = np.exp([scale_log.x, scale_log.y, scale_log.z])
        expected = np.diag(decoded_scale**2)

        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_gaussian3d_evalsh(buffer_shape: Tuple[int]):
    """Test Spherical harmonics."""
    gaussian_buf = spy.InstanceBuffer(
        struct=gaussian_module.Gaussian3D.as_struct(), shape=buffer_shape
    )

    default_sh = [spy.float3(0, 0, 0)] * 15
    gaussian_buf.construct(
        position=spy.float3(1, 2, 3),
        rotation=spy.float4(1, 0, 0, 0),
        scale=spy.float3(4, 5, 6),
        color=spy.float3(0.5, 0.5, 0.5),
        opacity=0.5,
        shCoeffs=default_sh,
    )

    red = spy.float3(1.0, 0.0, 0.0)
    cur = gaussian_buf.get_this().cursor()

    for i in range(buffer_shape[0]):
        for j in range(15):
            cur[i]["sh"][j] = red

    cur.apply()

    direction = spy.float3(0.577, 0.577, 0.577)
    result = gaussian_buf.sphericalHarmonics(direction, _result="numpy")

    assert result.shape == (buffer_shape[0], 3)
    assert np.all(
        result[:, 1] == pytest.approx(0.0)
    ), "Green channel should be zero"
    assert np.all(
        result[:, 2] == pytest.approx(0.0)
    ), "Blue channel should be zero"
    assert np.all(result[:, 0] > 0.0), "Red channel should be strictly positive"


def test_gaussian3d_eval(buffer_shape: Tuple[int]):
    """Test Evaluation."""
    gaussian_buf = spy.InstanceBuffer(
        struct=gaussian_module.Gaussian3D.as_struct(), shape=buffer_shape
    )

    red = spy.float3(1.0, 0.0, 0.0)
    sh_coeffs = [red for _ in range(15)]

    gaussian_buf.construct(
        position=spy.float3(0, 0, 0),
        rotation=spy.float4(0, 0, 0, 1),
        scale=spy.float3(0.0, 0.0, 0.0),
        color=spy.float3(0.0, 0.0, 0.0),
        opacity=0.0,
        shCoeffs=sh_coeffs,
    )

    pos = spy.float3(0, 0, 0)
    dir = spy.float3(0.577, 0.577, 0.577)

    result = gaussian_buf.eval(pos, dir, _result="numpy")

    assert result.shape == (buffer_shape[0], 4)

    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    expected_color = sigmoid(np.array([0.0, 0.0, 0.0]))
    expected_opacity = sigmoid(0.0)
    rho = np.exp(0.0)

    expected_base = expected_color * rho
    expected_alpha = expected_opacity * rho

    assert np.all(
        result[:, 3] == pytest.approx(expected_alpha)
    ), "Alpha should be sigmoid(opacity_logit) * rho"
    assert np.all(
        result[:, 1] == pytest.approx(expected_base[1])
    ), "Green should match sigmoid(g)"
    assert np.all(
        result[:, 2] == pytest.approx(expected_base[2])
    ), "Blue should match sigmoid(b)"
    assert np.all(
        result[:, 0] > expected_base[0]
    ), "Red should include SH contribution"
