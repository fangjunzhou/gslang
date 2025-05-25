import numpy as np
import pytest
import slangpy as spy
from pyglm import glm
import quaternion
from pytest_benchmark.fixture import BenchmarkFixture

from bvhgs import device
from bvhgs.camera import Camera
from bvhgs.gaussian import GaussianCloud
from bvhgs.renderer import Renderer


@pytest.fixture(params=[2**i for i in range(10, 20)])
def benchmark_cloud_size(request: pytest.FixtureRequest) -> int:
    """Fixture to provide a gaussian cloud size for benchmarking.

    :param request: The pytest request object.
    :return: The number of gaussian points to generate.
    """
    return request.param


@pytest.fixture
def gaussian_cloud(benchmark_cloud_size: int) -> GaussianCloud:
    """Create a random gaussian cloud for rendering.

    :param benchmark_cloud_size: Size of the gaussian cloud.
    :return: A random gaussian cloud.
    """
    cloud = GaussianCloud()
    # Randomize gaussian points with reasonable parameters
    # position_var controls the spread of the points
    # scale_offset adds a constant to all scales to make them visible
    cloud.randomize(benchmark_cloud_size, position_var=5.0, scale_offst=0.05)
    return cloud


@pytest.fixture
def camera() -> Camera:
    """Create a camera for rendering.

    :return: A camera positioned to view the scene.
    """
    # Position camera to view the random scene
    # For random gaussians, a position slightly away from origin looking toward it works well
    position = glm.vec3(10.0, 10.0, 10.0)

    # Look at the origin
    direction = -glm.normalize(position)

    # Create a quaternion for camera rotation
    # First create a rotation that aligns -z with the direction
    z_axis = glm.vec3(0.0, 0.0, -1.0)
    rotation_axis = glm.cross(z_axis, direction)
    if glm.length(rotation_axis) < 1e-6:
        # If vectors are parallel, use an arbitrary orthogonal axis
        if abs(glm.dot(z_axis, direction) + 1.0) < 1e-6:
            # If they point in opposite directions
            rotation = glm.quat(
                0.0, 0.0, 1.0, 0.0
            )  # 180 degree rotation around Y
        else:
            # If they point in same direction (should not happen in this setup)
            rotation = glm.quat(1.0, 0.0, 0.0, 0.0)  # identity quaternion
    else:
        rotation_axis = glm.normalize(rotation_axis)
        angle = np.arccos(glm.dot(z_axis, direction))
        rotation = glm.angleAxis(angle, rotation_axis)

    return Camera(
        position=position,
        rotation=rotation,
        sensor_size=glm.uvec2(800, 600),  # Standard test resolution
        focal_length=580,
        near_plane=0.01,
        far_plane=100,
    )


def render_frame(renderer: Renderer) -> None:
    """Render a single frame.

    :param renderer: The renderer to use.
    """
    renderer.render()


def test_renderer_benchmark(
    benchmark: BenchmarkFixture, gaussian_cloud: GaussianCloud, camera: Camera
):
    """Benchmark the integrated rendering process.

    :param benchmark: The benchmark fixture.
    :param gaussian_cloud: Random gaussian cloud to render.
    :param camera: Camera to render from.
    """
    # Create renderer with the random gaussians and camera
    renderer = Renderer(gaussian_cloud, camera)

    # Benchmark the rendering process
    benchmark(render_frame, renderer)


def test_renderer_with_multiple_camera_positions(
    benchmark: BenchmarkFixture, gaussian_cloud: GaussianCloud
):
    """Benchmark rendering with camera movement.

    This test benchmarks rendering performance from different camera positions
    using pytest-benchmark's pedantic mode with a setup function to change
    the camera pose between rounds.

    :param benchmark: The benchmark fixture.
    :param gaussian_cloud: Random gaussian cloud to render.
    """
    # Initialize with a starting camera
    camera = Camera(
        position=glm.vec3(10.0, 10.0, 10.0),
        rotation=glm.quat(1.0, 0.0, 0.0, 0.0),  # Identity quaternion
        sensor_size=glm.uvec2(800, 600),
        focal_length=580,
        near_plane=0.01,
        far_plane=100,
    )

    renderer = Renderer(gaussian_cloud, camera)
    
    # Pre-compute camera positions for different angles
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    distance = 10.0
    camera_positions = []
    camera_rotations = []
    
    for angle in angles:
        # Calculate camera position to orbit around the scene
        camera_pos = glm.vec3(
            np.sin(angle) * distance, np.cos(angle) * distance, 5.0
        )
        
        # Look at origin
        direction = -glm.normalize(camera_pos)
        
        # Create a quaternion that rotates from forward (-z) to direction
        forward = glm.vec3(0.0, 0.0, -1.0)
        rotation_axis = glm.cross(forward, direction)
        
        if glm.length(rotation_axis) > 1e-6:
            rotation_axis = glm.normalize(rotation_axis)
            angle_between = np.arccos(glm.dot(forward, direction))
            rotation = glm.angleAxis(angle_between, rotation_axis)
        else:
            # Handle parallel vectors
            rotation = glm.quat(1.0, 0.0, 0.0, 0.0)
        
        camera_positions.append(camera_pos)
        camera_rotations.append(rotation)
    
    # Current position index
    position_index = 0
    
    def setup():
        """Setup function that changes camera position before each benchmark round."""
        nonlocal position_index
        
        # Update the camera with the next position in the orbit
        renderer.camera.position = camera_positions[position_index]
        renderer.camera.rotation = camera_rotations[position_index]
        
        # Move to next position for the next round
        position_index = (position_index + 1) % len(camera_positions)
    
    # Benchmark a single frame render using pedantic mode
    # This will run the setup function before each round, changing the camera position
    benchmark.pedantic(
        renderer.render,  # Target function to benchmark
        setup=setup,      # Setup function to run before each round
        rounds=8,         # Number of rounds (one for each camera position)
        iterations=1      # Number of iterations per round
    )
