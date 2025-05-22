import pytest
import slangpy as spy
import numpy as np
import logging

from bvhgs import device
from bvhgs import math_module


np.random.seed(0)
logger = logging.getLogger(__name__)


@pytest.fixture(params=[16])
def buffer_size(request: pytest.FixtureRequest) -> int:
    return request.param


def test_eigen2x2(buffer_size: int):
    """
    Test the eigen2x2 function.
    """
    # Create a random 2x2 matrix
    mat = np.random.rand(buffer_size, 2, 2).astype(np.float32)
    mat_buf = spy.NDBuffer(device, dtype=spy.float2x2, shape=mat.shape)
    mat_buf.copy_from_numpy(mat)

    # Compute the eigenvalues and eigenvectors using numpy
    eigenvalues, eigenvectors = np.linalg.eig(mat)

    # Call the eigen2x2 function from the math_module
    eigen_func = math_module.find_function("linalg.eigen")
    assert eigen_func is not None, "eigen function not found."
    res = eigen_func(mat_buf)
    assert res is not None, "eigen function returned None."

    cursor = res.cursor()
    for i in range(buffer_size):
        logger.debug(f"matrix {i}: {mat[i, :, :]}")

        eigen = cursor[i].read()
        eigenvalues_spy: spy.float2 = eigen["eigenvalues"]
        eigenvectors_spy: spy.float2x2 = eigen["eigenvectors"]

        logger.debug(f"Eigenvalues: {eigenvalues_spy}")
        logger.debug(f"Eigenvectors: {eigenvectors_spy}")

        eigenvectors_spy_np = eigenvectors_spy.to_numpy()

        eigenvalues_sorted = np.sort(eigenvalues[i, :])
        assert eigenvalues_spy.x == pytest.approx(
            eigenvalues_sorted[1], abs=1e-4
        )
        assert eigenvalues_spy.y == pytest.approx(
            eigenvalues_sorted[0], abs=1e-4
        )

        # Test eigenvectors.
        e0 = eigenvectors_spy_np[0, :].T
        mat_e0 = mat[i, :, :] @ e0
        logger.debug(f"mat_e0: {mat_e0}")
        logger.debug(f"eigenvalues_spy.x * e0: {eigenvalues_spy.x * e0}")
        assert np.allclose(eigenvalues_spy.x * e0, mat_e0, atol=1e-4)
        e1 = eigenvectors_spy_np[1, :].T
        mat_e1 = mat[i, :, :] @ e1
        logger.debug(f"mat_e1: {mat_e1}")
        logger.debug(f"eigenvalues_spy.y * e1: {eigenvalues_spy.y * e1}")
        assert np.allclose(eigenvalues_spy.y * e1, mat_e1, atol=1e-4)
