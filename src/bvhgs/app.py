from typing import Tuple
import slangpy as spy
from pathlib import Path
import numpy as np
from pyglm import glm
import quaternion
import argparse

from bvhgs.gaussian import GaussianCloud
from bvhgs.camera import Camera
from bvhgs.renderer import Renderer


class App:
    def __init__(self, ply_path: Path):
        # Create window and device
        self.window = spy.Window(
            width=800, height=600, title="BVHGS Viewer", resizable=False
        )
        self.device = spy.Device(enable_debug_layers=True)
        self.surface = self.device.create_surface(self.window)
        self.surface.configure(
            width=self.window.width, height=self.window.height
        )

        # UI context
        self.ui = spy.ui.Context(self.device)

        # Load scene
        gaussians = GaussianCloud()
        gaussians.load_from_ply(ply_path)

        # Initialize camera and cursor at scene center
        self.cursor = glm.vec3(0, 0, 0)
        # Camera theta and phi angles.
        self.theta = 0
        self.phi = np.pi / 2
        # Camera distance from the center.
        self.distance = 5.0
        cam_pos, cam_rot = self.get_camera_pose()

        camera = Camera(
            position=cam_pos,
            rotation=cam_rot,
            sensor_size=glm.uvec2(self.window.width, self.window.height),
            focal_length=580,
            near_plane=0.01,
            far_plane=100,
        )

        # Renderer
        self.renderer = Renderer(gaussians, camera)

        # Interaction state
        self.last_mouse = spy.float2(0, 0)
        self.left_down = False
        self.right_down = False

        # Event handlers
        self.window.on_mouse_event = self.on_mouse_event
        self.window.on_resize = self.on_resize

    def get_camera_pose(self) -> Tuple[glm.vec3, glm.quat]:
        """Get the current camera position and rotation."""
        direction = glm.vec3(
            np.sin(self.phi) * np.cos(self.theta),
            np.sin(self.phi) * np.sin(self.theta),
            np.cos(self.phi),
        )
        position = self.cursor + direction * self.distance
        right = glm.cross(
            glm.vec3(np.cos(self.theta), np.sin(self.theta), 0),
            glm.vec3(0, 0, 1),
        )
        up = glm.cross(right, -direction)
        rotation = glm.quatLookAt(direction, up)
        return position, rotation

    def update_camera(self):
        """Update the camera based on the current angles and distance."""
        cam_pos, cam_rot = self.get_camera_pose()
        self.renderer.camera.position = cam_pos
        self.renderer.camera.rotation = cam_rot

    def on_mouse_event(self, event: spy.MouseEvent):
        if self.ui.handle_mouse_event(event):
            return
        if event.type == spy.MouseEventType.move:
            if self.last_mouse is None:
                self.last_mouse = event.pos
                return
            dx = event.pos.x - self.last_mouse.x
            dy = event.pos.y - self.last_mouse.y
            self.last_mouse = event.pos
            # rotation
            if self.left_down:
                sensitivity = 0.01
                self.theta -= dx * sensitivity
                self.phi -= dy * sensitivity
                self.phi = np.clip(self.phi, 0, np.pi)
                self.update_camera()
            # pan
            if self.right_down:
                pan_speed = 0.01
                cam = self.renderer.camera
                # axes in world
                cam_rot = quaternion.from_float_array(cam.rotation)
                right = quaternion.as_rotation_matrix(cam_rot).dot(
                    np.array([1, 0, 0], dtype=np.float32)
                )
                up = quaternion.as_rotation_matrix(cam_rot).dot(
                    np.array([0, 1, 0], dtype=np.float32)
                )
                shift = -right * dx * pan_speed - up * dy * pan_speed
                cam.position += glm.vec3(*shift)
                self.cursor += glm.vec3(*shift)
        elif event.type == spy.MouseEventType.scroll:
            if event.scroll.y != 0:
                zoom_speed = 0.1
                self.distance -= event.scroll.y * zoom_speed
                self.update_camera()
        elif event.type == spy.MouseEventType.button_down:
            if event.button == spy.MouseButton.left:
                self.left_down = True
            if event.button == spy.MouseButton.right:
                self.right_down = True
            self.last_mouse = None
        elif event.type == spy.MouseEventType.button_up:
            if event.button == spy.MouseButton.left:
                self.left_down = False
            if event.button == spy.MouseButton.right:
                self.right_down = False

    def on_resize(self, width: int, height: int):
        self.device.wait()
        self.surface.configure(width=width, height=height)
        # update camera sensor size
        # self.renderer.camera.sensor_size = glm.uvec2(width, height)

    def run(self):
        while not self.window.should_close():
            self.window.process_events()
            self.ui.process_events()

            surface_tex = self.surface.acquire_next_image()
            if not surface_tex:
                continue

            # render scene
            self.renderer.render()

            # blit to swapchain
            cmd = self.device.create_command_encoder()
            cmd.blit(surface_tex, self.renderer.render_target)

            self.ui.new_frame(self.window.width, self.window.height)
            self.ui.render(surface_tex, cmd)

            self.device.submit_command_buffer(cmd.finish())
            del surface_tex

            self.surface.present()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BVHGS Viewer")
    parser.add_argument(
        "ply_path",
        type=Path,
        help="Path to the PLY file containing Gaussian points",
    )
    args = parser.parse_args()
    if not args.ply_path.exists():
        raise FileNotFoundError(f"PLY file not found: {args.ply_path}")
    # Initialize and run the application
    App(ply_path=args.ply_path).run()
