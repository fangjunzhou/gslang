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
