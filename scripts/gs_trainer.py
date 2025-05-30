import argparse
from pathlib import Path
import multiprocessing as mp
from multiprocessing.connection import Connection
from dataclasses import dataclass
import numpy as np
from PIL import Image
from tqdm import tqdm
from enum import Enum

from bvhgs.app import App
from bvhgs.camera import Camera
from bvhgs.gaussian import GaussianCloud
from bvhgs.data import SFMDataset
from bvhgs.renderer import Renderer


@dataclass
class TrainingConfig:
    """Configuration for the BVHGS training process."""

    num_epochs: int = 64
    batch_size: int = 32
    # Learning rate and decay parameters.
    learning_rate: float = 2e-1
    gamma: float = 0.95
    decay_steps: int = 8
    # Adam optimizer parameters.
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 1e-4


class TrainerStateType(Enum):
    """Enumeration for the type of trainer state."""

    STEP = "step"
    EPOCH = "epoch"
    OPTM = "optimizer_step"


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
    colmap_path: Path,
    images_path: Path,
    training_config: TrainingConfig,
    conn: Connection,
):
    """Main function to run the BVHGS trainer."""
    # Load scene
    gaussians = GaussianCloud()
    # gaussians.load_from_colmap(colmap_path, scale_factor=-4, opacity_factor=-2)
    gaussians.randomize(
        100000,
        position_var=2.5,
        scale_var=0.25,
        scale_offst=-4,
        opacity_factor=-4,
    )

    # Load SFM Dataset
    sfm_dataset = SFMDataset()
    sfm_dataset.load_from_colmap(colmap_path, images_path)
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
    optm_step = 0
    for epoch in range(training_config.num_epochs):
        running_loss = 0.0
        # Shuffle indices for the dataset.
        indices = np.random.permutation(len(sfm_dataset))
        for step, idx in enumerate(indices):
            camera, image_path = sfm_dataset[idx]
            renderer.set_camera(camera)
            # Load image.
            image = Image.open(image_path)
            # Forward pass.
            loss = renderer.render(image)
            # Backward pass and optimization.
            renderer.backward(
                lr=curr_lr / training_config.batch_size,
                beta1=training_config.beta1,
                beta2=training_config.beta2,
                weight_decay=training_config.weight_decay,
            )
            running_loss += loss

            # Optimizer step.
            if (idx + 1) % training_config.batch_size == 0 or idx == len(
                sfm_dataset
            ) - 1:
                optm_step += 1
                if (optm_step + 1) % training_config.decay_steps == 0:
                    curr_lr *= training_config.gamma
                renderer.optimizer_step()
                renderer.zero_grad()
                state = TrainerState(
                    type=TrainerStateType.OPTM,
                    epoch=epoch,
                    step=step,
                    total_steps=len(sfm_dataset),
                    loss=running_loss / (idx + 1),
                    lr=curr_lr,
                    num_gaussians=len(gaussians),
                    gaussian_arr=renderer.gaussian_3d_buf.to_numpy(),
                )
                conn.send(state)

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
            num_gaussians=len(gaussians),
        )
        conn.send(state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BVHGS Trainer")
    parser.add_argument(
        "colmap",
        type=Path,
        help="Path to the COLMAP file containing Gaussian points",
    )
    parser.add_argument(
        "images",
        type=Path,
        help="Path to the directory containing images for training",
    )
    args = parser.parse_args()
    if not args.colmap.exists():
        raise FileNotFoundError(f"COLMAP path not found: {args.path}")
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
        args=(args.colmap, args.images, training_config, child_conn),
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
            elif state.type == TrainerStateType.OPTM:
                # Update rendering scene.
                if (
                    state.gaussian_arr is not None
                    and state.gaussian_arr.size > 0
                ):
                    app.renderer.sync_gaussians(
                        state.gaussian_arr, state.num_gaussians
                    )

    # Kill the trainer process if it's still running
    if trainer_process.is_alive():
        trainer_process.terminate()
        trainer_process.join()
