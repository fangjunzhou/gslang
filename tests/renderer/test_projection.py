import os
from typing import Tuple

import pytest
import numpy as np
import slangpy as spy

from bvhgs import device
from bvhgs import gaussian_module, camera_module
from bvhgs.camera import Camera
from pyglm import glm

@pytest.fixture(params=[(1,), (4,)])
def buffer_shape(request) -> Tuple[int]:
    return request.param


def make_camera(sensor_size=(800, 600), focal_length=100.0):
    cam_buf = spy.InstanceBuffer(
        struct=camera_module.Camera.as_struct(), shape=(1,)
    )
    cam_buf.construct(
        rotation=spy.float4(0, 0, 0, 1),
        translation=spy.float3(0, 0, 0),
        sensorSize=spy.uint2(*sensor_size),
        focalLength=focal_length,
        nearPlane=0.5,
        farPlane=1000.0,
    )
    return cam_buf


def make_gaussian3d(buffer_shape, position, scale_log=(0.0, 0.0, 0.0)):
    buf = spy.InstanceBuffer(
        struct=gaussian_module.Gaussian3D.as_struct(), shape=buffer_shape
    )
    sh0 = [spy.float3(0, 0, 0)] * 15
    buf.construct(
        position=spy.float3(*position),
        rotation=spy.float4(0, 0, 0, 1),
        scale=spy.float3(*scale_log),
        color=spy.float3(0, 0, 0),
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
    Z = 500.0

    cam = make_camera(sensor_size=(W, H), focal_length=f)
    gauss3d = make_gaussian3d(buffer_shape, position=(0.0, 0.0, Z))

    out_buffer = cam.toGaussian2D(gauss3d)
    out_cursor = out_buffer.cursor()
    out = [out_cursor[i].read() for i in range(buffer_shape[0])]

    exp_uv = [0.5, 0.5]
    exp_cov = np.diag([(f / (W * Z)) ** 2, (f / (H * Z)) ** 2])

    Z_norm = (Z - 0.5) / (1000.0 - 0.5)
    for i in range(buffer_shape[0]):
        # uv
        pos = out[i]["position"]

        pos_uv = [pos[0], pos[1]]
        pos_z = pos[2]
        assert np.allclose(
            pos_uv, exp_uv, rtol=1e-5, atol=1e-6
        ), f"uv mismatch at {i}: got {pos_uv}, expected {exp_uv}"

        assert pos_z == pytest.approx(Z_norm, rel=1e-6, abs=1e-6)

        cov = out[i]["covariance"].to_numpy()

        # covariance
        assert np.allclose(
            cov, exp_cov, rtol=1e-5, atol=1e-6
        ), f"covariance mismatch at {i}: got\n{cov}\nexpected\n{exp_cov}"


def test_toGaussian2D_offcenter(buffer_shape):
    """
    For a point not on the optical axis, we check the sign and range of uv-covariance:
        - uv should fall within the (0,1)^2 range
        - The covariance matrix should be positive semi-definite
    """
    W, H = 800, 600
    f = 50.0
    X, Y, Z = 200.0, 100.0, 500.0

    cam = make_camera(sensor_size=(W, H), focal_length=f)
    gauss3d = make_gaussian3d(buffer_shape, position=(X, Y, Z))

    out_buffer = cam.toGaussian2D(gauss3d)
    out_cursor = out_buffer.cursor()
    out = [out_cursor[i].read() for i in range(buffer_shape[0])]

    for i in range(buffer_shape[0]):
        # uv
        pos = out[i]["position"]

        pos_uv = [pos[0], pos[1]]

        assert (
            0.0 <= pos_uv[0] <= 1.0
        ), f"uv mismatch at {i}: got {pos_uv}, expected in [0,1]"
        assert (
            0.0 <= pos_uv[1] <= 1.0
        ), f"uv mismatch at {i}: got {pos_uv}, expected in [0,1]"

        cov = out[i]["covariance"].to_numpy()

        # covariance
        eigs = np.linalg.eigvalsh(cov)
        assert np.all(eigs >= -1e-6), f"cov not PSD at {i}, eigs = {eigs}"

def test_off_screen():
    mod = device.load_module("renderer.slang")
    program = device.link_program([mod], [])
        
    prog_project = device.link_program([mod], [mod.entry_point("project")])
    k_project = device.create_compute_kernel(prog_project)
    
    gaussian_buf = device.create_buffer(
        element_count=20,
        struct_type=program.reflection.g_gaussian_3d,
        usage=spy.BufferUsage.shader_resource,
    )
    gaussian_cursor = spy.BufferCursor(
        program.reflection.g_gaussian_3d.type_layout.element_type_layout,
        gaussian_buf,
    )

    for i in range(20):
        #generate random offscreen gaussians
        signx = 1 if np.random.rand() > 0.5 else -1
        signy = 1 if np.random.rand() > 0.5 else -1
      
        x = signx * (np.random.rand() * 400 + 600)
        y = signy * (np.random.rand() * 200 + 600)
        
        gaussian_cursor[i].write({
            "position": glm.vec3(x, y, 64),
            "rotation": glm.quat(0, 0, 0, 1),
            "scale": glm.vec3(0,0,0),
            "color": glm.vec3(1, 1, 1),
            "opacity": 1.0,
            "sh": [spy.float3(0, 0, 0) for _ in range(15)],
        })
    gaussian_cursor.apply()
    

    

    

    camera = Camera(
        rotation=glm.quat(1, 0, 0, 0),
        translation=glm.vec3(0, 0, 1),
        sensor_size=glm.uvec2(512, 512),
        focal_length=64
    )

    cull_flag_buf = device.create_buffer(
        element_count=20,
        struct_type=program.reflection.g_cull_flag,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )
    
    gaussian2d_buf = device.create_buffer(
        element_count=20,
        struct_type=program.reflection.g_gaussian_2d,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )
    
    k_project.dispatch(
        thread_count=[20, 1, 1],
        vars={
            "g_camera": camera.to_slang(),
            "g_gaussian_3d": gaussian_buf,
            "g_gaussian_2d": gaussian2d_buf,
            "g_cull_flag": cull_flag_buf
        }
    )
        
    cull_flag_cursor = spy.BufferCursor(
        program.reflection.g_cull_flag.type_layout.element_type_layout,
        cull_flag_buf,
    )
    
    gaussian2d_cursor = spy.BufferCursor(
        program.reflection.g_gaussian_2d.type_layout.element_type_layout,
        gaussian2d_buf,
    )
    #test bbox
    for i in range(cull_flag_cursor.element_count):
        flag = cull_flag_cursor[i].read()
        assert flag == 0, f"Gaussian {i} is on screen, position: {gaussian_cursor[i].read()}, transformed: {gaussian2d_cursor[i].read()}"

def test_near_far_culling():
    mod = device.load_module("renderer.slang")
    prog_project = device.link_program([mod], [mod.entry_point("project")])
    k_project = device.create_compute_kernel(prog_project)


    N = 3
    g3d_buf = device.create_buffer(
        element_count=N,
        struct_type=prog_project.reflection.g_gaussian_3d,
        usage=spy.BufferUsage.shader_resource,
    )
    g3d_cur = spy.BufferCursor(
        prog_project.reflection.g_gaussian_3d.type_layout.element_type_layout,
        g3d_buf,
    )

    depths = [2.0, 1200.0, 0.25]  # inside, beyond far, before near
    for i, z in enumerate(depths):
        g3d_cur[i].write({
            "position": glm.vec3(0, 0, z),
            "rotation": glm.quat(), 
            "scale": glm.vec3(0, 0, 0),
            "color": glm.vec3(1, 1, 1),
            "opacity": 1.0,
            "sh": [spy.float3(0, 0, 0)] * 15,
        })
    g3d_cur.apply()

    g2d_buf = device.create_buffer(
        element_count=N,
        struct_type=prog_project.reflection.g_gaussian_2d,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )
    flag_buf = device.create_buffer(
        element_count=N,
        struct_type=prog_project.reflection.g_cull_flag,
        usage=spy.BufferUsage.shader_resource | spy.BufferUsage.unordered_access,
    )


    cam = Camera(
        rotation = glm.quat(),
        translation = glm.vec3(0, 0, 0),
        sensor_size = glm.uvec2(512, 512),
        focal_length = 64,
        near_plane = 0.5,
        far_plane = 1000.0,
    )

    k_project.dispatch(
        thread_count=[N, 1, 1],
        vars={
            "g_camera": cam.to_slang(),
            "g_gaussian_3d": g3d_buf,
            "g_gaussian_2d": g2d_buf,
            "g_cull_flag": flag_buf,
        }
    )
    flag_cur = spy.BufferCursor(
        prog_project.reflection.g_cull_flag.type_layout.element_type_layout,
        flag_buf,
    )
    flags = [flag_cur[i].read() for i in range(N)]

    assert flags[0] == 1, "depth inside [near,far] should not be culled"
    assert flags[1] == 0, "depth beyond far plane should be culled"
    assert flags[2] == 0, "depth in front of near plane should be culled"