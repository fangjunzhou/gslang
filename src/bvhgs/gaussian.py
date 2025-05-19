import slangpy as spy
import pathlib
from typing import Any, Dict
import numpy as np


class GaussianBuffer:
    """A buffer for storing all the Gaussian points in the scene.

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

    def __init__(self, size: int) -> None:
        self.positions = np.random.rand(size, 3).astype(np.float32)
        self.rotations = np.random.rand(size, 4).astype(np.float32)
        self.scales = np.random.rand(size, 3).astype(np.float32)

        self.colors = np.random.rand(size, 3).astype(np.float32)
        self.opacities = np.random.rand(size, 1).astype(np.float32)
        self.spherical_harmonics = np.random.rand(size, 15, 3).astype(
            np.float32
        )

        self.num_gaussians = size

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

    def load_from_ply(self, path: pathlib.Path):
        """Load a Gaussian point cloud from a PLY file.

        :param path: path to the PLY file.
        """
        # TODO: Implement loading from PLY file
        pass

    def load_from_colmap(self, path: pathlib.Path):
        """Load a Gaussian point cloud from a COLMAP sparse file.

        :param path: path to the COLMAP sparse file.
        """
        # TODO: Implement loading from COLMAP file
        pass
