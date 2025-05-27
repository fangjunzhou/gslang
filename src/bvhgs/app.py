from typing import Tuple
import slangpy as spy
from pathlib import Path
import numpy as np
from pyglm import glm
import quaternion
import argparse

from bvhgs import device
from bvhgs.gaussian import GaussianCloud
from bvhgs.camera import Camera
from bvhgs.renderer import Renderer


class App:
    def __init__(
        self,
        ply_path: Path,
        resolution: Tuple[int, int] = (800, 600),
        focal_length: float = 580,
    ):
        # Create window and device
        self.window = spy.Window(
            width=resolution[0],
            height=resolution[1],
            title="BVHGS Viewer",
            resizable=False,
        )
        self.device = device
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

        # Initialize UI-related variables
        self.scene_rotation = spy.float3(
            0, 0, 0
        )  # Initial scene rotation (Euler angles in degrees)
        self.scene_rotation_slider = None  # Will be set in setup_ui
        self.zoom_sensitivity = None  # Will be set in setup_ui

        # UI setup - do this before camera initialization
        self.setup_ui()

        # Now get camera pose after UI is setup
        cam_pos, cam_rot = self.get_camera_pose()

        camera = Camera(
            position=cam_pos,
            rotation=cam_rot,
            sensor_size=glm.uvec2(resolution[0], resolution[1]),
            focal_length=focal_length,
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

    def setup_ui(self):
        """Setup ImGui UI elements."""
        screen = self.ui.screen
        window = spy.ui.Window(
            screen, "BVHGS Controls", size=spy.float2(500, 250)
        )

        # Add a simple button
        def reset_camera():
            self.theta = 0
            self.phi = np.pi / 2
            self.distance = 5.0
            self.cursor = glm.vec3(0, 0, 0)
            self.scene_rotation = spy.float3(0, 0, 0)
            # Update slider with reset values if it exists
            if self.scene_rotation_slider is not None:
                self.scene_rotation_slider.value = self.scene_rotation
            # Update camera position and rotation
            if self.zoom_sensitivity is not None:
                self.zoom_sensitivity.value = 0.1
            if self.rotation_sensitivity is not None:
                self.rotation_sensitivity.value = 0.01
            if self.pan_sensitivity is not None:
                self.pan_sensitivity.value = 0.01
            self.update_camera()

        spy.ui.Button(window, "Reset Camera", callback=reset_camera)

        # Add a slider for zoom speed
        self.zoom_sensitivity = spy.ui.SliderFloat(
            window,
            "Zoom Sensitivity",
            value=0.1,  # Default value
            min=0.01,
            max=0.5,
        )

        # Add a slider for rotation speed
        self.rotation_sensitivity = spy.ui.SliderFloat(
            window,
            "Rotation Sensitivity",
            value=0.01,  # Default value
            min=0.001,
            max=0.1,
            flags=spy.ui.SliderFlags.logarithmic,
        )

        # Add a slider for pan speed
        self.pan_sensitivity = spy.ui.SliderFloat(
            window,
            "Pan Sensitivity",
            value=0.01,  # Default value
            min=0.001,
            max=0.1,
            flags=spy.ui.SliderFlags.logarithmic,
        )

        # Add a slider for global scene rotation (Euler angles in degrees)
        self.scene_rotation_slider = spy.ui.SliderFloat3(
            window,
            "Scene Rotation",
            value=self.scene_rotation,  # Use the pre-initialized value
            min=-180.0,
            max=180.0,
            callback=lambda _: self.update_camera(),
        )

        # Framerate display
        self.fps_avg = spy.ui.Text(window, f"FPS: 0")

    def get_scene_rotation_quat(self) -> glm.quat:
        # Make sure to use scene_rotation directly during initialization
        scene_rotation_values = self.scene_rotation
        if self.scene_rotation_slider is not None:
            scene_rotation_values = self.scene_rotation_slider.value

        # Convert Euler angles (in degrees) to radians
        x_rad = glm.radians(scene_rotation_values.x)
        y_rad = glm.radians(scene_rotation_values.y)
        z_rad = glm.radians(scene_rotation_values.z)

        # Create rotation quaternions for each axis
        q_x = glm.angleAxis(x_rad, glm.vec3(1.0, 0.0, 0.0))
        q_y = glm.angleAxis(y_rad, glm.vec3(0.0, 1.0, 0.0))
        q_z = glm.angleAxis(z_rad, glm.vec3(0.0, 0.0, 1.0))

        # Combine them (order matters: z * y * x)
        scene_rotation_quat = q_z * q_y * q_x

        return scene_rotation_quat

    def get_camera_pose(self) -> Tuple[glm.vec3, glm.quat]:
        """Get the current camera position and rotation."""
        # Calculate camera direction and position
        direction = glm.vec3(
            np.sin(self.phi) * np.cos(self.theta),
            np.sin(self.phi) * np.sin(self.theta),
            np.cos(self.phi),
        )
        position = self.cursor + direction * self.distance

        # Calculate camera orientation
        right = glm.cross(
            glm.vec3(np.cos(self.theta), np.sin(self.theta), 0),
            glm.vec3(0, 0, 1),
        )
        up = glm.cross(right, -direction)
        view_rotation = glm.quatLookAt(direction, up)

        # Apply scene rotation (convert slider values from degrees to radians)
        scene_rotation_quat = self.get_scene_rotation_quat()

        # Apply scene rotation to camera rotation
        final_rotation = scene_rotation_quat * view_rotation
        # Convert to glm.quat
        final_position = (
            scene_rotation_quat
            * glm.quat(0.0, position.x, position.y, position.z)
            * glm.conjugate(scene_rotation_quat)
        )
        final_position = glm.vec3(
            final_position.x, final_position.y, final_position.z
        )

        return final_position, final_rotation

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
                sensitivity = (
                    self.rotation_sensitivity.value
                    if self.rotation_sensitivity is not None
                    else 0.01
                )
                self.theta -= dx * sensitivity
                self.phi -= dy * sensitivity
                self.phi = np.clip(self.phi, 0, np.pi, dtype=float)
            # pan
            if self.right_down:
                pan_speed = (
                    self.pan_sensitivity.value
                    if self.pan_sensitivity is not None
                    else 0.01
                )
                # axes in world
                direction = glm.vec3(
                    np.sin(self.phi) * np.cos(self.theta),
                    np.sin(self.phi) * np.sin(self.theta),
                    np.cos(self.phi),
                )
                right = glm.cross(
                    glm.vec3(np.cos(self.theta), np.sin(self.theta), 0),
                    glm.vec3(0, 0, 1),
                )
                up = glm.cross(right, -direction)
                # Move cursor position based on mouse movement
                self.cursor += right * dx * pan_speed - up * dy * pan_speed
            # Update camera position based on cursor
            self.update_camera()

        elif event.type == spy.MouseEventType.scroll:
            if event.scroll.y != 0:
                # Base zoom speed from UI slider or default value
                base_zoom_speed = 0.1  # Default zoom speed
                if (
                    hasattr(self, "zoom_sensitivity")
                    and self.zoom_sensitivity is not None
                ):
                    base_zoom_speed = self.zoom_sensitivity.value
                # Scale zoom speed based on distance - faster when far, slower when close
                # The factor is proportional to the current distance
                distance_factor = self.distance * 0.05  # 5% of current distance
                zoom_speed = base_zoom_speed * (1.0 + distance_factor)

                # Apply zooming with distance-based speed
                new_distance = self.distance - event.scroll.y * zoom_speed
                # Enforce minimum distance to prevent the camera from going through the cursor
                min_distance = 0.1  # Minimum allowed distance
                self.distance = max(new_distance, min_distance)
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
        # Timer for FPS calculation
        timer = spy.Timer()
        fps_avg = 0.0

        # Store previous scene rotation to detect changes
        # Initialize with current scene rotation
        prev_scene_rotation = self.scene_rotation

        while not self.window.should_close():
            # Calculate frame time and FPS
            elapsed = timer.elapsed_s()
            timer.reset()
            fps_avg = 0.95 * fps_avg + 0.05 * (1.0 / max(elapsed, 0.001))
            # Display FPS in integer format
            self.fps_avg.text = f"FPS: {fps_avg:.0f}"

            # Process events
            self.window.process_events()
            self.ui.process_events()

            # Get render surface
            surface_tex = self.surface.acquire_next_image()
            if not surface_tex:
                continue

            # Render scene
            self.renderer.render()

            # Setup command encoder and blit rendered image to surface
            cmd = self.device.create_command_encoder()
            cmd.blit(surface_tex, self.renderer.render_target)

            # Setup and render UI
            self.ui.new_frame(self.window.width, self.window.height)
            self.ui.render(surface_tex, cmd)

            # Submit commands and present
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
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=(800, 600),
        help="Resolution of the window (width height)",
    )
    parser.add_argument(
        "--focal-length",
        type=float,
        default=580.0,
        help="Focal length for the camera",
    )
    args = parser.parse_args()
    if not args.ply_path.exists():
        raise FileNotFoundError(f"PLY file not found: {args.ply_path}")
    # Initialize and run the application
    App(
        ply_path=args.ply_path,
        resolution=tuple(args.resolution),
        focal_length=args.focal_length,
    ).run()
