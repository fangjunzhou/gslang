import argparse
from pathlib import Path

from bvhgs.app import App
from bvhgs.gaussian import GaussianCloud


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BVHGS Viewer")
    parser.add_argument(
        "path",
        type=Path,
        help="Path to the PLY or COLMAP file containing Gaussian points",
    )
    parser.add_argument("--colmap", action="store_true", help="Is COLMAP file")
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
    if not args.path.exists():
        raise FileNotFoundError(f"Path not found: {args.path}")

    # Load scene
    gaussians = GaussianCloud()
    if not args.colmap:
        gaussians.load_from_ply(args.path)
    else:
        gaussians.load_from_colmap(args.path)

    # Initialize and run the application
    app = App(
        gaussians,
        resolution=tuple(args.resolution),
        focal_length=args.focal_length,
    )
    for _ in app.run():
        pass  # Keep the application running until closed
