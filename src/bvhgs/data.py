from pyglm import glm
from slangpy.core.utils import pathlib
import pycolmap
import quaternion
from PIL import Image

from bvhgs.camera import Camera


class SFMDataset:
    img_pairs: list[tuple[Camera, pathlib.Path]]

    def __init__(self) -> None:
        """Initialize the SFMDataset object."""
        self.img_pairs = []

    def __len__(self) -> int:
        """Return the number of image pairs in the dataset."""
        return len(self.img_pairs)

    def __getitem__(self, index: int) -> tuple[Camera, pathlib.Path]:
        """Return the image pair at the given index.

        :param index: Index of the image pair.
        :return: Tuple containing the Camera object and the image path.
        """
        if index < 0:
            # Handle negative indexing.
            index += len(self.img_pairs)
        if index >= len(self.img_pairs):
            raise IndexError("Index out of range.")
        return self.img_pairs[index]

    def load_from_colmap(
        self, colmap_path: pathlib.Path, image_dir: pathlib.Path
    ):
        """Load the dataset from a COLMAP database.

        :param colmap_path: Path to the COLMAP database.
        """
        if not colmap_path.exists():
            raise FileNotFoundError(
                f"COLMAP database {colmap_path} does not exist."
            )
        if not image_dir.exists():
            raise FileNotFoundError(f"Image dir {image_dir} does not exist.")

        maps = pycolmap.Reconstruction(str(colmap_path))
        for _, image in maps.images.items():
            assert type(image) == pycolmap.Image
            assert image.cam_from_world is not None
            colmap_cam = image.camera
            assert colmap_cam is not None
            # Get camera pose.
            pose = image.cam_from_world.inverse()
            rotation = glm.quat(pose.rotation.quat)
            position = glm.vec3(pose.translation)
            # Get camera intrinsics.
            sx = colmap_cam.width
            sy = colmap_cam.height
            f = colmap_cam.focal_length

            # Get image path.
            image_path = image_dir / image.name
            if not image_path.exists():
                raise FileNotFoundError(f"Image {image_path} does not exist.")
            # Load image.
            img = Image.open(image_path)
            # Scale camera sensor size according to image size.
            sensor_scale = glm.vec2(img.width / sx, img.height / sy)
            sensor_size = glm.uvec2(img.width, img.height)
            focal_length = f * sensor_scale.x
            # Create camera object.
            camera = Camera(
                rotation=rotation,
                position=position,
                sensor_size=sensor_size,
                focal_length=focal_length,
            )

            # Append camera and image path to the list.
            self.img_pairs.append((camera, image_path))

    def load_from_camera_json(
        self, camera_path: pathlib.Path, image_dir: pathlib.Path
    ):
        """Load the dataset from a camera JSON file and image path.

        :param camera_path: Path to the camera JSON file.
        :param image_dir: Path to the image directory.
        """
        import json
        import numpy as np

        # Check if paths exist
        if not camera_path.exists():
            raise FileNotFoundError(f"Camera file {camera_path} does not exist.")
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory {image_dir} does not exist.")

        # Load camera data from JSON
        with open(camera_path, "r") as f:
            cameras_data = json.load(f)

        # Process each camera entry
        for camera_data in cameras_data:
            # Skip incomplete entries
            if "img_name" not in camera_data or not all(key in camera_data for key in ["fx", "width", "height"]):
                print(f"Skipping incomplete camera data: {camera_data}")
                continue

            # Get image path
            img_name = camera_data["img_name"]
            image_path = image_dir / img_name
            if not image_path.exists():
                print(f"Image {image_path} does not exist, skipping.")
                continue

            # Load image to get dimensions
            img = Image.open(image_path)
            
            # Extract camera parameters
            focal_length = camera_data.get("fx", 0)
            width = camera_data.get("width", img.width)
            height = camera_data.get("height", img.height)
            
            # Create sensor size
            sensor_size = glm.uvec2(width, height)
            
            # Extract position (if available)
            position = glm.vec3(0.0)
            if "position" in camera_data:
                position = glm.vec3(camera_data["position"][0:3])
            
            # Extract rotation (if available)
            rotation = glm.quat(1.0, 0.0, 0.0, 0.0)  # Identity quaternion
            if "rotation" in camera_data:
                rotation_mat = np.array(camera_data["rotation"]).reshape(3, 3)
                rotation_quat = quaternion.from_rotation_matrix(rotation_mat)
                rotation = glm.quat(quaternion.as_float_array(rotation_quat))
            
            # Create camera object
            camera = Camera(
                rotation=rotation,
                position=position,
                sensor_size=sensor_size,
                focal_length=focal_length
            )
            
            # Append camera and image path to the list
            self.img_pairs.append((camera, image_path))
