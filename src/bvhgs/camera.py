from typing import Any, Dict
from pyglm import glm


class Camera:
    # -------------------- Camera Extrinsics  -------------------- #
    # World space to camera space rotation.
    rotation: glm.quat
    # Camera space to world space translation.
    translation: glm.vec3
    # -------------------- Camera Intrinsics  -------------------- #
    # Camera sensor size in pixels.
    sensor_size: glm.uvec2
    # Focal length in pixels.
    focal_length: float

    def __init__(
        self,
        rotation: glm.quat,
        translation: glm.vec3,
        sensor_size: glm.uvec2,
        focal_length: float,
    ) -> None:
        """Constructor for the Camera class.

        :param rotation: Camera rotation as a quaternion.
        :param translation: Camera translation as a 3D vector.
        :param sensor_size: Camera sensor size in pixels.
        :param focal_length: Focal length in pixels.
        """
        self.rotation = rotation
        self.translation = translation
        self.sensor_size = sensor_size
        self.focal_length = focal_length

    def to_slang(self) -> Dict[str, Any]:
        """Convert the camera parameters to a dictionary format for Slang.

        :return: Dictionary containing the camera parameters.
        """
        rot_xyzw = [
            self.rotation.x,
            self.rotation.y,
            self.rotation.z,
            self.rotation.w,
        ]
        return {
            "_rotation": rot_xyzw,
            "_translation": self.translation,
            "_sensorSize": self.sensor_size,
            "_focalLength": self.focal_length,
        }
