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

from bvhgs.app import App
from bvhgs.camera import Camera
from bvhgs.gaussian import GaussianCloud
from bvhgs.data import SFMDataset
from bvhgs.renderer import Renderer


@dataclass
class TrainingConfig:
    """Configuration for the BVHGS training process."""

    num_epochs: int = 64
    # Learning rate and decay parameters.
    learning_rate: float = 1e-3
    position_lr_factor: float = 1.0
    pos_lr_decay_rate: float = 0.99
    rotation_lr_factor: float = 1.0
    scale_lr_factor: float = 1.0
    color_lr_factor: float = 1.0
    opacity_lr_factor: float = 1.0
    sh_lr_factor: float = 1.0
    decay_steps: int = 16
    # Adam optimizer parameters.
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 1e-4
    # Warmup parameters.
    warmup_levels: int = 4
    warmup_steps: int = 250
    # Densification parameters.
    densify_steps: int = 100
    densify_scale: float = 1
    correction_steps: int = 100
    gaussian_removal_threshold: float = 0.1

class TrainerStateType(Enum):
    """Enumeration for the type of trainer state."""

    STEP = "step"
    EPOCH = "epoch"


@dataclass
class TrainerState:
    """State of the BVHGS trainer process."""

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
    """Main function to run the BVHGS trainer."""
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
            num_random_gaussians=50000,
            random_gaussian_position_range=10,
            random_gaussian_scale=-2,
        )
        # gaussians.randomize(
        #     size=100000,
        #     )
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
            need_correction = False
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
                renderer.recalcuate_avg_3dgs_size()
                
            # Optimizer step.
            optm_step += 1
            if (optm_step + 1) % training_config.decay_steps == 0:
                curr_pos_decay *= training_config.pos_lr_decay_rate
            renderer.optimizer_step()

            # Densification step.
            if (optm_step + 1) % training_config.densify_steps == 0:
                need_density = True

            if (optm_step + 1) % training_config.correction_steps == 0:
                need_correction = True
  
            if need_density or need_correction:
                renderer.render(
                    image,
                    use_densify=need_density,
                    densify_scale=training_config.densify_scale,
                    use_correction=need_correction,
                    gaussian_opacity_remove_threshold=training_config.gaussian_removal_threshold,
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
    parser = argparse.ArgumentParser(description="BVHGS Trainer")
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
    app = App(rendering_gaussians)
    app_iter = app.run()

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
    print("Starting BVHGS trainer...")
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

    # Kill the trainer process if it's still running
    if trainer_process.is_alive():
        trainer_process.terminate()
        trainer_process.join()
