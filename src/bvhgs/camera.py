from pyglm import glm


class Camera:
    # -------------------- Camera Extrinsics  -------------------- #
    # World space to camera space rotation.
    rotation: glm.quat
    # Camera space to world space translation.
    translation: glm.vec3
    # -------------------- Camera Intrinsics  -------------------- #
    # Camera sensor size in pixels.
    sensor_size: glm.vec2
    # Focal length in pixels.
    focal_length: float
