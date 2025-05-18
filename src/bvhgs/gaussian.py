import pathlib
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

    def __init__(self) -> None:
        self.positions = np.zeros((0, 3), dtype=np.float32)
        self.rotations = np.zeros((0, 4), dtype=np.float32)
        self.scales = np.zeros((0, 3), dtype=np.float32)

        self.colors = np.zeros((0, 3), dtype=np.float32)
        self.opacities = np.zeros((0, 1), dtype=np.float32)
        self.spherical_harmonics = np.zeros((0, 15, 3), dtype=np.float32)

        self.num_gaussians = 0

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
