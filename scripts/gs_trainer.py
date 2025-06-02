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
import logging

from gslang.app import App
from gslang.camera import Camera
from gslang.gaussian import GaussianCloud
from gslang.data import SFMDataset
from gslang.renderer import Renderer


@dataclass
class TrainingConfig:
    """Configuration for the gslang training process."""

    num_epochs: int = 128
    # Learning rate and decay parameters.
    learning_rate: float = 1e-3
    position_lr_factor: float = 1.0
    pos_lr_decay_rate: float = 0.99
    rotation_lr_factor: float = 1.0
    scale_lr_factor: float = 1.0
    color_lr_factor: float = 1.0
    opacity_lr_factor: float = 2.0
    sh_lr_factor: float = 1.0
    decay_steps: int = 16
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
    opacity_prune_skip_step: int = 2000
    # Step to reset all opacity below a threshold.
    reset_opacity_steps: int = 10000
    gaussian_prune_threshold: float = -3
    gaussian_reset_opacity: float = -3.5


class TrainerStateType(Enum):
    """Enumeration for the type of trainer state."""

    STEP = "step"
    EPOCH = "epoch"


@dataclass
class TrainerState:
    """State of the gslang trainer process."""

    type: TrainerStateType = TrainerStateType.STEP
    epoch: int = 0
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
        type=TrainerStateType.EPOCH,
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
    optm_step = 0
    for epoch in range(training_config.num_epochs):
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
                optm_step
                < training_config.warmup_steps * training_config.warmup_levels
            ):
                curr_level = optm_step // training_config.warmup_steps
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
            if optm_step == 0:
                renderer.recalcuate_avg_2dgs_size()

            # Optimizer step.
            optm_step += 1
            if (optm_step + 1) % training_config.decay_steps == 0:
                curr_pos_decay *= training_config.pos_lr_decay_rate
            renderer.optimizer_step()

            # Densification step.
            if (optm_step + 1) % training_config.densify_steps == 0:
                need_density = True

            if (optm_step + 1) % training_config.reset_opacity_steps == 0:
                need_reset_opacity = True
            # Do not reset opacity in the last few epochs.
            if (
                training_config.num_epochs * len(sfm_dataset) - optm_step
                < training_config.reset_opacity_steps
            ):
                need_reset_opacity = False

            if (optm_step + 1) % training_config.opacity_prune_step == 0 and (
                optm_step + 1
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
                type=TrainerStateType.STEP,
                epoch=epoch,
                step=step,
                total_steps=len(sfm_dataset),
                loss=running_loss / (idx + 1),
                lr=curr_lr,
            )
            conn.send(state)

        # Average loss for the epoch.
        avg_loss = running_loss / len(sfm_dataset)
        state = TrainerState(
            type=TrainerStateType.EPOCH,
            epoch=epoch + 1,
            step=0,
            total_steps=len(sfm_dataset),
            loss=avg_loss,
            lr=curr_lr,
            num_gaussians=renderer.num_gaussians,
            gaussian_arr=renderer.gaussian_3d_buf.to_numpy(),
        )
        conn.send(state)


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
    training_config = TrainingConfig()

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

    epoch_pbar = tqdm(
        total=training_config.num_epochs,
        desc="Training Epochs",
        unit="epoch",
        leave=True,
    )
    step_pbar = tqdm(
        total=0,
        desc="Training Steps",
        unit="step",
        leave=False,
    )

    # Main loop to receive updates from the trainer process
    if not args.headless:
        for _ in app_iter:
            if trainer_process.is_alive() and parent_conn.poll():
                state: TrainerState = parent_conn.recv()
                if state.type == TrainerStateType.EPOCH:
                    # Update epoch progress bar
                    epoch_pbar.n = state.epoch
                    epoch_pbar.set_postfix(
                        loss=f"{state.loss:.4f}", lr=f"{state.lr:.6f}"
                    )
                    epoch_pbar.refresh()
                    step_pbar.reset()
                    step_pbar.total = state.total_steps
                    # Update rendering scene.
                    if (
                        state.gaussian_arr is not None
                        and state.gaussian_arr.size > 0
                    ):
                        app.renderer.sync_gaussians(
                            state.gaussian_arr, state.num_gaussians
                        )
                    if state.epoch > 0:
                        app.renderer.to_ply(
                            args.save_path / f"epoch_{state.epoch:03d}.ply"
                        )
                elif state.type == TrainerStateType.STEP:
                    # Update epoch progress bar
                    epoch_pbar.n = state.epoch
                    epoch_pbar.refresh()
                    # Update step progress bar
                    step_pbar.update(1)
                    step_pbar.total = state.total_steps
                    step_pbar.set_postfix(
                        loss=f"{state.loss:.4f}", lr=f"{state.lr:.6f}"
                    )
                    step_pbar.refresh()
    else:
        # If running in headless mode, just wait for the trainer to finish
        while trainer_process.is_alive():
            if parent_conn.poll():
                state: TrainerState = parent_conn.recv()
                if state.type == TrainerStateType.EPOCH:
                    epoch_pbar.n = state.epoch
                    epoch_pbar.set_postfix(
                        loss=f"{state.loss:.4f}", lr=f"{state.lr:.6f}"
                    )
                    epoch_pbar.refresh()
                    step_pbar.reset()
                    step_pbar.total = state.total_steps
                    # Save the Gaussian cloud to a file.
                    if (
                        state.gaussian_arr is not None
                        and state.gaussian_arr.size > 0
                    ):
                        headless_renderer.sync_gaussians(
                            state.gaussian_arr, state.num_gaussians
                        )
                        headless_renderer.to_ply(
                            args.save_path / f"epoch_{state.epoch:03d}.ply"
                        )

                elif state.type == TrainerStateType.STEP:
                    step_pbar.update(1)
                    step_pbar.total = state.total_steps
                    step_pbar.set_postfix(
                        loss=f"{state.loss:.4f}", lr=f"{state.lr:.6f}"
                    )
                    step_pbar.refresh()

    # Kill the trainer process if it's still running
    if trainer_process.is_alive():
        trainer_process.terminate()
        trainer_process.join()
