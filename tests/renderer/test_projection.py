import os
from typing import Tuple

import pytest
import numpy as np
import slangpy as spy

from bvhgs import device
from bvhgs import gaussian_module, camera_module

@pytest.fixture(params=[(1,), (4,)])
def buffer_shape(request) -> Tuple[int]:
    return request.param


def make_camera(sensor_size=(800, 600), focal_length=100.0):
    cam_buf = spy.InstanceBuffer(
        struct=camera_module.Camera.as_struct(),
        shape=(1,)
    )
    cam_buf.construct(
        rotation=spy.float4(0, 0, 0, 1),
        translation=spy.float3(0, 0, 0),
        sensorSize=spy.uint2(*sensor_size),
        focalLength=focal_length,
    )
    return cam_buf

def make_gaussian3d(buffer_shape, position, scale_log=(0.0,0.0,0.0)):
    buf = spy.InstanceBuffer(
        struct=gaussian_module.Gaussian3D.as_struct(),
        shape=buffer_shape
    )
    sh0 = [spy.float3(0,0,0)] * 15
    buf.construct(
        position=spy.float3(*position),
        rotation=spy.float4(0,0,0,1),
        scale=spy.float3(*scale_log),
        color=spy.float3(0,0,0),
        opacity=0.0,
        shCoeffs=sh0,
    )
    return buf

def quat(axis, radians):
    axis = np.asarray(axis, dtype=np.float32)
    axis /= np.linalg.norm(axis) + 1e-8
    s = float(np.sin(radians * 0.5, dtype=np.float32))
    c = float(np.cos(radians * 0.5, dtype=np.float32))
    ax, ay, az = map(float, axis * s)           
    return spy.float4(ax, ay, az, c)


def test_toGaussian2D_center_and_cov(buffer_shape):
    """
    For a Gaussian3D located at (0,0,Z) along the camera's optical axis:
        - It maps to the screen center uv=(0.5,0.5)
        - Its screen-space covariance matrix = S * I * S^T
        where S is the scale factor for the Gaussian3D.
    """

    W, H = 800, 600
    f = 100.0
    Z = 1000.0

    cam = make_camera(sensor_size=(W,H), focal_length=f)
    gauss3d = make_gaussian3d(buffer_shape, position=(0.0,0.0,Z))

    out_buffer = cam.toGaussian2D(gauss3d)
    out_cursor = out_buffer.cursor()
    out = [out_cursor[i].read() for i in range(buffer_shape[0])]


    exp_uv = [0.5, 0.5]
    exp_cov = np.diag([ (f/(W*Z))**2, (f/(H*Z))**2])
    
    for i in range(buffer_shape[0]):
        # uv
        pos = out[i]["position"]
        
        pos_uv = [pos[0], pos[1]]
        pos_z = pos[2]
        assert np.allclose(pos_uv, exp_uv, rtol=1e-5, atol=1e-6), \
            f"uv mismatch at {i}: got {pos_uv}, expected {exp_uv}"
       
        assert pos_z == pytest.approx(Z, rel=1e-6, abs=1e-6)

        cov = out[i]["covariance"].to_numpy()
        
        # covariance
        assert np.allclose(cov, exp_cov, rtol=1e-5, atol=1e-6), \
            f"covariance mismatch at {i}: got\n{cov}\nexpected\n{exp_cov}"
            
            
def test_toGaussian2D_offcenter(buffer_shape):
    """
    For a point not on the optical axis, we check the sign and range of uv-covariance:
        - uv should fall within the (0,1)^2 range
        - The covariance matrix should be positive semi-definite
    """
    W, H = 800, 600
    f = 50.0
    X, Y, Z = 200.0, 100.0, 500.0

    cam = make_camera(sensor_size=(W,H), focal_length=f)
    gauss3d = make_gaussian3d(buffer_shape, position=(X,Y,Z))

    out_buffer = cam.toGaussian2D(gauss3d)
    out_cursor = out_buffer.cursor()
    out = [out_cursor[i].read() for i in range(buffer_shape[0])]
    

    for i in range(buffer_shape[0]):
        # uv
        pos = out[i]["position"]
        
        pos_uv = [pos[0], pos[1]]

        assert 0.0 <= pos_uv[0] <= 1.0, f"uv mismatch at {i}: got {pos_uv}, expected in [0,1]"
        assert 0.0 <= pos_uv[1] <= 1.0, f"uv mismatch at {i}: got {pos_uv}, expected in [0,1]"
       

        cov = out[i]["covariance"].to_numpy()
        
        # covariance
        eigs = np.linalg.eigvalsh(cov)
        assert np.all(eigs >= -1e-6), f"cov not PSD at {i}, eigs = {eigs}"
        
