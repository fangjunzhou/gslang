import os
from typing import Tuple
import slangpy as spy
import pytest
import numpy as np
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from bvhgs import device
from bvhgs import gaussian_module


np.random.seed(42)
logger = logging.getLogger(__name__)


@pytest.fixture(params=[(64,)])
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


def test_gaussian2d_init_default(buffer_shape: Tuple[int]):
    """Test Gaussian2D default constructor.

    :param buffer_shape: Shape of the buffer.
    """
    gaussian_buf = spy.InstanceBuffer(
        struct=gaussian_module.Gaussian2D.as_struct(), shape=buffer_shape
    )

    gaussian_buf.construct()
    # Check default constructor.
    cursor = gaussian_buf.get_this().cursor()
    for i in range(buffer_shape[0]):
        gaussian = cursor[i].read()
        assert gaussian["position"] == spy.float3(0, 0, 0)
        assert gaussian["covariance"] == spy.float2x2([1, 0, 0, 1])
        assert gaussian["color"] == spy.float3(1, 0, 1)
        assert gaussian["opacity"] == 1


def test_gaussian2d_bounding_box(buffer_shape: Tuple[int]):
    """Test Gaussian2D bounding box calculation."""
    position = np.random.uniform(
        low=0.1, high=0.9, size=(buffer_shape[0], 3)
    ).astype(np.float32)
    # Create SPD covariance matrices.
    covariance = np.random.randn(buffer_shape[0], 2, 2).astype(np.float32)
    for i in range(buffer_shape[0]):
        cov = covariance[i]
        cov = np.dot(cov, cov.T) * 0.01
        covariance[i] = cov
    color = np.random.rand(buffer_shape[0], 3).astype(np.float32)
    opacity = np.random.rand(buffer_shape[0]).astype(np.float32)

    position_buf = spy.NDBuffer(device, dtype="float3", shape=buffer_shape)
    position_buf.copy_from_numpy(position)
    covariance_buf = spy.NDBuffer(device, dtype="float2x2", shape=buffer_shape)
    covariance_buf.copy_from_numpy(covariance)
    color_buf = spy.NDBuffer(device, dtype="float3", shape=buffer_shape)
    color_buf.copy_from_numpy(color)
    opacity_buf = spy.NDBuffer(device, dtype="float", shape=buffer_shape)
    opacity_buf.copy_from_numpy(opacity)

    gaussian_buf = spy.InstanceList(
        struct=gaussian_module.Gaussian2D.as_struct(),
        data={
            "position": position_buf,
            "covariance": covariance_buf,
            "color": color_buf,
            "opacity": opacity_buf,
        },
    )

    # Calculate the bounding box.
    bbox_buf = gaussian_buf.boundingBox()
    cursor = bbox_buf.cursor()

    # Save the plot to debug plot.
    if not os.path.exists(".tests"):
        os.makedirs(".tests")

    # Scatter Gaussian positions.
    NUM_SAMPLES = 4096
    for i in range(buffer_shape[0]):
        # Draw the gaussian to a plot.
        fig, ax = plt.subplots()
        ax.set_title("Gaussian2D Bounding Box")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        mean = position[i]
        cov = covariance[i]
        opa = opacity[i]

        # Generate samples from the Gaussian distribution.
        samples = np.random.multivariate_normal(
            mean=mean[:2], cov=cov, size=NUM_SAMPLES
        )

        # Plot the samples.
        ax.scatter(
            samples[:, 0], samples[:, 1], color=color[i], alpha=opa / 16, s=1
        )

        # Draw the bounding box with the same color as the Gaussian.
        bbox = cursor[i].read()
        min = bbox["min"]
        max = bbox["max"]
        width = max.x - min.x
        height = max.y - min.y
        rect = mpatches.Rectangle(
            (min.x, min.y),
            width,
            height,
            linewidth=1,
            edgecolor=color[i],
            facecolor="none",
        )
        ax.add_patch(rect)

        plt.savefig(f".tests/gaussian2d_bbox_{i}.png")

        # Calculate the proportion of samples inside the bounding box.
        inside = np.logical_and(
            np.logical_and(samples[:, 0] >= min.x, samples[:, 0] <= max.x),
            np.logical_and(samples[:, 1] >= min.y, samples[:, 1] <= max.y),
        )
        proportion_inside = np.sum(inside) / NUM_SAMPLES
        logger.debug(
            f"Gaussian {i}: Proportion of samples inside bounding box: {proportion_inside:.2f}"
        )
        # Log the gaussian parameters.
        logger.debug(f"Gaussian {i}: Position: {mean}")
        logger.debug(f"Gaussian {i}: Covariance: {cov * 100}")
        assert (
            proportion_inside > 0.9
        ), f"Proportion of samples inside bounding box for Gaussian {i} is too low."
