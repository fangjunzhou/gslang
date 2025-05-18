from typing import Tuple
import slangpy as spy
import pytest

from bvhgs import gaussian_module
SH_COUNT = 15

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
        for j in range(SH_COUNT):
            assert gaussian["sh"][j] == spy.float3(0, 0, 0)


def test_gaussian3d_init_custom(buffer_shape: Tuple[int]):
    """Test Gaussian3D custom constructor.

    :param buffer_shape: Shape of the buffer.
    """
    gaussian_buf = spy.InstanceBuffer(
        struct=gaussian_module.Gaussian3D.as_struct(), shape=buffer_shape
    )
    
    sh = [spy.float3(i, -i, i * 2) for i in range(SH_COUNT)]
    gaussian_buf.construct(
        shCoeffs=sh,
        position=spy.float3(1, 2, 3),
        rotation=spy.float4(1, 0, 0, 0),
        scale=spy.float3(4, 5, 6),
        color=spy.float3(0.5, 0.5, 0.5),
        opacity=0.5,
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
        for j in range(SH_COUNT):
            assert gaussian["sh"][j] == sh[j]
