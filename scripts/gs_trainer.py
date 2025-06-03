import argparse
from pathlib import Path
import multiprocessing as mp
from multiprocessing.connection import Connection
from dataclasses import dataclass
import numpy as np
from PIL import Image
from tqdm import tqdm
from enum import Enum
from pyglm import glm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import jax.numpy as jnp

from gslang.app import App
from gslang.camera import Camera
from gslang.gaussian import GaussianCloud
from gslang.data import SFMDataset
from gslang.renderer import Renderer


def convert_to_scalar(value):
    """Convert JAX arrays, numpy arrays, or tensors to Python scalars for TensorBoard."""
    if hasattr(value, "item"):
        # JAX array
        return float(value.item())
    elif hasattr(value, "item"):
        # NumPy array or torch tensor
        return float(value.item())
    elif hasattr(value, "__float__"):
        # Already a scalar
        return float(value)
    else:
        return value


@dataclass
class TrainingConfig:
    """Configuration for the gslang training process."""

    num_step: int = 7000
    # Learning rate and decay parameters.
    learning_rate: float = 1e-3
    position_lr_factor: float = 1.0
    pos_lr_decay_rate: float = 0.99
    rotation_lr_factor: float = 1.0
    scale_lr_factor: float = 1.0
    color_lr_factor: float = 1.0
    opacity_lr_factor: float = 1.0
    sh_lr_factor: float = 1.0
    decay_steps: int = 25
    # Adam optimizer parameters.
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 1e-4
    # Warmup parameters.
    warmup_levels: int = 2
    warmup_steps: int = 250
    # Densification parameters.
    densify_steps: int = 100
    densify_scale: float = 5e-3
    # Step to prune opacity below a threshold.
    opacity_prune_step: int = 100
    # Step to skip pruning opacity after a reset.
    opacity_prune_skip_step: int = 1000
    # Step to reset all opacity below a threshold.
    reset_opacity_steps: int = 7000
    gaussian_prune_threshold: float = -2
    gaussian_reset_opacity: float = -2.5
    # TensorBoard logging parameters.
    use_tensorboard: bool = False
    tensorboard_log_dir: str = "runs"
    log_step_interval: int = 10  # Log every N steps


@dataclass
class TrainerState:
    """State of the gslang trainer process."""

    step: int = 0
    total_steps: int = 0
    loss: float = 0.0
    lr: float = 1e-4
    num_gaussians: int = 0
    gaussian_arr: np.ndarray | None = None


def trainer_worker(
    path: Path,
    is_colmap: bool,
    images_path: Path,
    camera_path: Path | None,
    training_config: TrainingConfig,
    conn: Connection,
):
    """Main function to run the gslang trainer."""
    # logging.basicConfig(
    #     level=logging.INFO
    #

    # Initialize TensorBoard writer if enabled
    writer = None
    if training_config.use_tensorboard:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = (
            Path(training_config.tensorboard_log_dir)
            / f"gslang_training_{timestamp}"
        )
        writer = SummaryWriter(str(log_dir))
        print(f"TensorBoard logging enabled. Log directory: {log_dir}")
    elif training_config.use_tensorboard:
        print(
            "Warning: TensorBoard requested but not available. Install tensorboard: pip install tensorboard"
        )

    # Load scene
    gaussians = GaussianCloud()

    # Load SFM Dataset
    if is_colmap:
        gaussians.load_from_colmap(
            path,
            scale_factor=-4,
            opacity_factor=-2,
            add_random_gaussians=True,
            num_random_gaussians=10000,
            random_gaussian_scale=-1,
            random_gaussian_position_range=10,
        )
        sfm_dataset = SFMDataset()
        sfm_dataset.load_from_colmap(path, images_path)
    else:
        gaussians.load_from_ply(path)
        sfm_dataset = SFMDataset()
        assert (
            camera_path is not None
        ), "Camera path must be provided if not using COLMAP."
        sfm_dataset.load_from_camera_json(camera_path, images_path)

    # Init camera.
    camera = Camera()
    # Initialize renderer.
    renderer = Renderer(gaussians, camera)

    state = TrainerState(
        epoch=0,
        step=0,
        total_steps=len(sfm_dataset),
        loss=0.0,
        lr=training_config.learning_rate,
        num_gaussians=len(gaussians),
    )
    conn.send(state)

    # Training loop.
    curr_lr = training_config.learning_rate
    curr_pos_decay = 1
    curr_step = 0
    while curr_step < training_config.num_step:
        running_loss = 0.0
        # Shuffle indices for the dataset.
        indices = np.random.permutation(len(sfm_dataset))
        for step, idx in enumerate(indices):
            need_density = False
            need_opacity_prune = False
            need_reset_opacity = False
            camera, image_path = sfm_dataset[idx]
            # Load image.
            image = Image.open(image_path)
            # Warmup training.
            if (
                curr_step
                < training_config.warmup_steps * training_config.warmup_levels
            ):
                curr_level = curr_step // training_config.warmup_steps
                down_sample_factor = 2 ** (
                    training_config.warmup_levels - curr_level
                )
                image = image.resize(
                    (
                        image.width // down_sample_factor,
                        image.height // down_sample_factor,
                    )
                )
                camera = Camera(
                    position=camera.position,
                    rotation=camera.rotation,
                    sensor_size=glm.uvec2(image.width, image.height),
                    focal_length=camera.focal_length * down_sample_factor,
                )

            renderer.set_camera(camera)

            # Forward pass.
            renderer.zero_grad()
            loss = renderer.render(image)
            # Backward pass and optimization.
            renderer.backward(
                pos_lr=curr_lr
                * training_config.position_lr_factor
                * curr_pos_decay,
                rot_lr=curr_lr * training_config.rotation_lr_factor,
                scale_lr=curr_lr * training_config.scale_lr_factor,
                color_lr=curr_lr * training_config.color_lr_factor,
                opacity_lr=curr_lr * training_config.opacity_lr_factor,
                sh_lr=curr_lr * training_config.sh_lr_factor,
                beta1=training_config.beta1,
                beta2=training_config.beta2,
                weight_decay=training_config.weight_decay,
            )
            running_loss += loss
            if curr_step == 0:
                renderer.recalcuate_avg_2dgs_size()

            # Log to TensorBoard
            if writer and curr_step % training_config.log_step_interval == 0:
                # Convert JAX arrays to scalars for TensorBoard compatibility
                writer.add_scalar(
                    "Loss/Step", convert_to_scalar(loss), curr_step
                )
                writer.add_scalar(
                    "Learning_Rate/Position_Decay", curr_pos_decay, curr_step
                )
                writer.add_scalar(
                    "Gaussians/Count", renderer.num_gaussians, curr_step
                )

            # Optimizer step.
            curr_step += 1
            if (curr_step + 1) % training_config.decay_steps == 0:
                curr_pos_decay *= training_config.pos_lr_decay_rate
            renderer.optimizer_step()

            # Densification step.
            if (curr_step + 1) % training_config.densify_steps == 0:
                need_density = True

            if (curr_step + 1) % training_config.reset_opacity_steps == 0:
                need_reset_opacity = True
            # Do not reset opacity in the last few epochs.
            if (
                training_config.num_step - curr_step
                < training_config.reset_opacity_steps
            ):
                need_reset_opacity = False

            if (curr_step + 1) % training_config.opacity_prune_step == 0 and (
                curr_step + 1
            ) % training_config.reset_opacity_steps > training_config.opacity_prune_skip_step:
                need_opacity_prune = True

            if need_density or need_opacity_prune or need_reset_opacity:
                renderer.render(
                    image,
                    use_densify=need_density,
                    densify_scale=training_config.densify_scale,
                    use_opacity_prune=need_opacity_prune,
                    use_reset_opacity=need_reset_opacity,
                    gaussian_opacity_prune_threshold=training_config.gaussian_prune_threshold,
                    gaussian_reset_opacity=training_config.gaussian_reset_opacity,
                )

            # Send the current state to the parent process.
            state = TrainerState(
                step=step,
                total_steps=len(sfm_dataset),
                loss=running_loss / (idx + 1),
                lr=curr_lr,
            )
            conn.send(state)

        # Average loss for the epoch.
        avg_loss = running_loss / len(sfm_dataset)

        state = TrainerState(
            step=0,
            total_steps=len(sfm_dataset),
            loss=avg_loss,
            lr=curr_lr,
            num_gaussians=renderer.num_gaussians,
            gaussian_arr=renderer.gaussian_3d_buf.to_numpy(),
        )
        conn.send(state)

    # Close TensorBoard writer
    if writer:
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="gslang Trainer")
    parser.add_argument(
        "path",
        type=Path,
        help="Path to the file containing Gaussian points",
    )
    parser.add_argument(
        "--colmap",
        action="store_true",
        dest="is_colmap",
        help="If the path is a COLMAP database file",
    )
    parser.add_argument(
        "camera_path",
        type=Path,
        nargs="?",
        help="Path to the camera file (required if not using --colmap)",
    )

    parser.add_argument(
        "images",
        type=Path,
        help="Path to the directory containing images for training",
    )
    parser.add_argument(
        "save_path",
        type=Path,
        help="Path to the directory to save training outputs",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run the application in headless mode without GUI",
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Enable TensorBoard logging",
    )
    parser.add_argument(
        "--tensorboard-log-dir",
        type=str,
        default="runs",
        help="Directory for TensorBoard logs (default: runs)",
    )
    parser.add_argument(
        "--log-step-interval",
        type=int,
        default=10,
        help="Log metrics to TensorBoard every N steps (default: 10)",
    )
    args = parser.parse_args()

    camera_path = None
    if not args.is_colmap:
        if args.camera_path is None:
            parser.error("You must provide a camera path if not using COLMAP.")
        if not args.camera_path.exists():
            raise FileNotFoundError(
                f"Camera path not found: {args.camera_path}"
            )
        camera_path = args.camera_path

    is_colmap: bool = args.is_colmap

    if not args.path.exists():
        raise FileNotFoundError(f"Path not found: {args.path}")

    if not args.images.exists():
        raise FileNotFoundError(f"Images path not found: {args.images}")

    rendering_gaussians = GaussianCloud()
    rendering_gaussians.randomize(1)
    if not args.headless:
        app = App(rendering_gaussians)
        app_iter = app.run()
    else:
        headless_renderer = Renderer(rendering_gaussians, Camera())

    # Define training configuration
    training_config = TrainingConfig(
        use_tensorboard=args.tensorboard,
        tensorboard_log_dir=args.tensorboard_log_dir,
        log_step_interval=args.log_step_interval,
    )

    # Create a connection for inter-process communication
    parent_conn, child_conn = mp.Pipe()
    # Start the trainer worker in a separate process
    trainer_process = mp.Process(
        target=trainer_worker,
        args=(
            args.path,
            is_colmap,
            args.images,
            camera_path,
            training_config,
            child_conn,
        ),
    )

    # Start the trainer process
    print("Starting gslang trainer...")
    trainer_process.start()

    step_pbar = tqdm(
        total=0,
        desc="Training Steps",
        unit="step",
        leave=False,
    )
    step_pbar.total = training_config.num_step

    # Main loop to receive updates from the trainer process
    epoch = 0
    if not args.headless:
        for _ in app_iter:
            if trainer_process.is_alive() and parent_conn.poll():
                state: TrainerState = parent_conn.recv()
                # Update epoch progress bar
                step_pbar.update(1)
                step_pbar.set_description(f"Loss: {state.loss:.4f}")
                # Update rendering scene.
                if (
                    state.gaussian_arr is not None
                    and state.gaussian_arr.size > 0
                ):
                    app.renderer.sync_gaussians(
                        state.gaussian_arr, state.num_gaussians
                    )
                    app.renderer.to_ply(
                        args.save_path / f"epoch_{epoch:03d}.ply"
                    )
                    epoch += 1
    else:
        # If running in headless mode, just wait for the trainer to finish
        while trainer_process.is_alive():
            if parent_conn.poll():
                state: TrainerState = parent_conn.recv()
                step_pbar.update(1)
                step_pbar.set_description(f"Loss: {state.loss:.4f}, ")
                # Save the Gaussian cloud to a file.
                if (
                    state.gaussian_arr is not None
                    and state.gaussian_arr.size > 0
                ):
                    headless_renderer.sync_gaussians(
                        state.gaussian_arr, state.num_gaussians
                    )
                    app.renderer.to_ply(
                        args.save_path / f"epoch_{epoch:03d}.ply"
                    )
                    epoch += 1

    # Kill the trainer process if it's still running
    if trainer_process.is_alive():
        trainer_process.terminate()
        trainer_process.join()
