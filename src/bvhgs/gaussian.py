import slangpy as spy
import pathlib
from typing import Any, Dict
import numpy as np
from pyntcloud import PyntCloud
import pandas as pd
import pycolmap
import scipy.spatial.transform as transform


class GaussianCloud:
    """A buffer for storing all the Gaussian point cloud in the scene.

    :param positions: (N, 3) array of positions of the Gaussian points.
    :param rotations: (N, 4) array of rotations of the Gaussian points.
    :param scales: (N, 3) array of scales of the Gaussian points.
    :param colors: (N, 3) array of colors of the Gaussian points.
    :param opacities: (N, 1) array of opacities of the Gaussian points.
    :param spherical_harmonics: (N, 15, 3) array of spherical harmonics of the Gaussian points.
    :param num_gaussians: number of Gaussian points N in the scene.
    """

    positions: np.ndarray
    rotations: np.ndarray
    scales: np.ndarray

    colors: np.ndarray
    opacities: np.ndarray
    spherical_harmonics: np.ndarray

    num_gaussians: int

    def __init__(self) -> None:
        """Initialize the GaussianCloud object."""
        self.positions = np.empty((0, 3), dtype=np.float32)
        self.rotations = np.empty((0, 4), dtype=np.float32)
        self.scales = np.empty((0, 3), dtype=np.float32)

        self.colors = np.empty((0, 3), dtype=np.float32)
        self.opacities = np.empty((0, 1), dtype=np.float32)
        self.spherical_harmonics = np.empty((0, 15, 3), dtype=np.float32)

        self.num_gaussians = 0

    def __len__(self) -> int:
        """Return the number of Gaussian points in the buffer."""
        return self.num_gaussians

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return the Gaussian point at the given index.

        :param index: index of the Gaussian point.
        :return: Gaussian point at the given index.
        """
        return {
            "position": self.positions[index],
            "rotation": self.rotations[index],
            "scale": self.scales[index],
            "color": self.colors[index],
            "opacity": self.opacities[index].item(),
            "sh": [col for col in self.spherical_harmonics[index]],
        }

    def randomize(
        self, size: int, position_var: float = 1.0, scale_offst: float = 0
    ):
        """Randomize the Gaussian point cloud.

        :param size: number of Gaussian points in the cloud.
        """
        # TODO: Add a seed for reproducibility. Add point distribution control.
        self.positions = (
            np.random.randn(size, 3).astype(np.float32) * position_var
        )
        self.rotations = np.random.rand(size, 4).astype(np.float32)
        # Normalize the rotations
        self.rotations /= np.linalg.norm(self.rotations, axis=1, keepdims=True)
        self.scales = np.random.randn(size, 3).astype(np.float32) + scale_offst

        self.colors = np.random.randn(size, 3).astype(np.float32)
        self.opacities = np.random.randn(size, 1).astype(np.float32)
        self.spherical_harmonics = np.random.randn(size, 15, 3).astype(
            np.float32
        )

        self.num_gaussians = size

    def load_from_ply(self, path: pathlib.Path):
        """Load a Gaussian point cloud from a PLY file.

        :param path: path to the PLY file.
        """
        if not path.exists():
            raise FileNotFoundError(f"File {path} does not exist.")

        cloud = PyntCloud.from_file(str(path.resolve()))
        pts: pd.DataFrame = cloud.points
        N = len(pts)
        if N == 0:
            raise ValueError(f"{path} contains no points")

        # position
        self.positions = pts[["x", "y", "z"]].to_numpy(np.float32)
        self.positions = np.ascontiguousarray(self.positions, dtype=np.float32)

        # orientation
        quat_cols = ["rot_1", "rot_2", "rot_3", "rot_0"]
        self.rotations = pts[quat_cols].to_numpy(np.float32)
        self.rotations /= (
            np.linalg.norm(self.rotations, axis=1, keepdims=True) + 1e-9
        )
        self.rotations = np.ascontiguousarray(self.rotations, dtype=np.float32)

        # scales
        self.scales = pts[["scale_0", "scale_1", "scale_2"]].to_numpy(
            np.float32
        )
        self.scales = np.ascontiguousarray(self.scales, dtype=np.float32)

        # colors
        # sigmoid is done by the shader
        self.colors = pts[["f_dc_0", "f_dc_1", "f_dc_2"]].to_numpy(np.float32)
        self.colors = np.ascontiguousarray(self.colors, dtype=np.float32)

        # opacities
        self.opacities = pts["opacity"].to_numpy(np.float32)
        self.opacities = np.ascontiguousarray(self.opacities, dtype=np.float32)

        # sh
        sh_cols = [f"f_rest_{i}" for i in range(45)]
        self.spherical_harmonics = (
            pts[sh_cols].to_numpy(np.float32).reshape(N, 15, 3)
        )
        self.spherical_harmonics = np.ascontiguousarray(
            self.spherical_harmonics, dtype=np.float32
        )

        self.num_gaussians = N

    def load_from_colmap(self, path: pathlib.Path, scale_factor: float = -3.0):
        """Load a Gaussian point cloud from a COLMAP sparse file.

        :param path: dir to the COLMAP sparse file.
        """
        if not path.exists():
            raise FileNotFoundError(f"Directory {path} does not exist.")

        maps = pycolmap.Reconstruction(str(path))
        num_points = len(maps.points3D)
        points = np.empty((num_points, 3))
        colors = np.empty((num_points, 3))

        # Extract point cloud.
        for i, point in enumerate(maps.points3D.values()):
            points[i] = point.xyz
            colors[i] = point.color
        # Filter points within 10 units.
        dists = np.linalg.norm(points, axis=1)
        points = points[dists <= 10, :]
        colors = colors[dists <= 10, :]
        # Scale colors to [0, 1]
        colors = colors / 256

        # inverse sigmoid
        colors = np.log(colors / (1 - colors + 1e-5) + 1e-5)

        self.positions = np.ascontiguousarray(points, dtype=np.float32)
        rotation = [0, 0, 0, 1]  # Identity quaternion
        self.rotations = np.tile(rotation, (len(points), 1)).astype(np.float32)
        self.scales = np.ones((len(points), 3), dtype=np.float32) * scale_factor
        self.colors = np.ascontiguousarray(colors, dtype=np.float32)
        # opacity=0.5
        self.opacities = np.full((len(points), 1), 0, dtype=np.float32)
        self.spherical_harmonics = np.zeros(
            (len(points), 15, 3), dtype=np.float32
        )
        self.num_gaussians = len(points)
