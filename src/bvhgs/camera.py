from typing import Any, Dict
from pyglm import glm


class Camera:
    # -------------------- Camera Extrinsics  -------------------- #
    # Camera pose.
    rotation: glm.quat
    position: glm.vec3
    # -------------------- Camera Intrinsics  -------------------- #
    # Camera sensor size in pixels.
    sensor_size: glm.uvec2
    # Focal length in pixels.
    focal_length: float
    # Near and far clipping planes.
    near_plane: float
    far_plane: float

    def __init__(
        self,
        rotation: glm.quat,
        position: glm.vec3,
        sensor_size: glm.uvec2,
        focal_length: float,
        near_plane: float = 0.1,
        far_plane: float = 1000.0,
    ) -> None:
        """Constructor for the Camera class.

        :param rotation: Camera rotation as a quaternion.
        :param translation: Camera translation as a 3D vector.
        :param sensor_size: Camera sensor size in pixels.
        :param focal_length: Focal length in pixels.
        """
        self.rotation = rotation
        self.position = position
        self.sensor_size = sensor_size
        self.focal_length = focal_length
        self.near_plane = near_plane
        self.far_plane = far_plane

    def to_slang(self) -> Dict[str, Any]:
        """Convert the camera parameters to a dictionary format for Slang.

        :return: Dictionary containing the camera parameters.
        """

        # Quaternion inverse.
        def inverse_quaternion(q: glm.quat) -> glm.quat:
            """Calculate the inverse of a quaternion."""
            return glm.conjugate(q) * (1 / glm.dot(q, q))

        # Camera space to world space translation.
        def rotate_vector(v: glm.vec3, q: glm.quat) -> glm.vec3:
            """Rotate a vector by a quaternion."""
            qv = glm.quat(0.0, v.x, v.y, v.z)
            rotated_vector: glm.quat = q * qv * glm.conjugate(q)
            return glm.vec3(
                rotated_vector.x, rotated_vector.y, rotated_vector.z
            )

        # World space to camera space rotation.
        rotation_view: glm.quat = inverse_quaternion(self.rotation)

        translation_view: glm.vec3 = -rotate_vector(
            self.position, rotation_view
        )
        rot_xyzw = [
            rotation_view.x,
            rotation_view.y,
            rotation_view.z,
            rotation_view.w,
        ]
        return {
            "_rotation": rot_xyzw,
            "_translation": translation_view,
            "_sensorSize": self.sensor_size,
            "_focalLength": self.focal_length,
            "_nearPlane": self.near_plane,
            "_farPlane": self.far_plane,
        }
