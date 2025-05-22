import pytest
import numpy as np
from bvhgs import math_module


np.random.seed(0)


def numpy_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@pytest.fixture
def sigmoid():
    func = math_module.find_function("utils.sigmoid")
    assert func is not None, "sigmoid function not found."
    return func


@pytest.fixture(params=[1, 16, 128])
def buffer_size(request):
    return request.param


def test_sigmoid_scalar_elementwise(buffer_size: int, sigmoid):
    x = np.random.randn(buffer_size).astype(np.float32)
    y = sigmoid(x, _result="numpy")
    y_ref = numpy_sigmoid(x)
    assert y.shape == x.shape
    assert np.allclose(y, y_ref, atol=1e-4)


@pytest.mark.parametrize("dims", [2, 3, 4])
def test_sigmoid_vector_dims(buffer_size, dims, sigmoid):
    x = np.random.randn(buffer_size, dims).astype(np.float32)
    y = sigmoid(x, _result="numpy")
    y_ref = numpy_sigmoid(x)
    assert y.shape == x.shape
    assert np.allclose(y, y_ref, atol=1e-4)


def test_sigmoid_values(sigmoid):
    """
    Test edge cases:
      - x = 0     → output 0.5
      - x = +10   → output close to 1.0
      - x = -10   → output close to 0.0
    """
    cases = {
        0.0: 0.5,
        10.0: 1.0,
        -10.0: 0.0,
    }
    for inp, expected in cases.items():
        x = np.array([inp], dtype=np.float32)
        y = sigmoid(x, _result="numpy")[0]
        assert pytest.approx(expected, abs=1e-4) == y


@pytest.fixture
def sigmoid3():
    fn = math_module.find_function("utils.sigmoid3")
    assert fn is not None, "sigmoid3 function not found."
    return fn


def test_sigmoid3(buffer_size, sigmoid3):
    """
    Test the float3 sigmoid overload:
    - Input is an (N×3) array of float32.
    - Output should have the same shape.
    - Each element should equal 1/(1+exp(-x)).
    """
    x = np.random.randn(buffer_size, 3).astype(np.float32)
    y = sigmoid3(x, _result="numpy")
    y_ref = 1.0 / (1.0 + np.exp(-x))

    assert y.shape == x.shape
    assert np.allclose(y, y_ref, atol=1e-4)
