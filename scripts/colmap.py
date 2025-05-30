from pathlib import Path
import pycolmap as pcm

def reconstruct(image_dir: str | Path, out_dir: str | Path):
    image_dir = Path(image_dir)
    out_dir   = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    db_path     = out_dir / "database.db"
    model_root  = out_dir / "sparse"
    model_root.mkdir(exist_ok=True)
    model_path  = model_root / "0"          # COLMAP’s default model‑id
    model_path.mkdir(exist_ok=True)

    # 1. SIFT features
    pcm.extract_features(db_path, image_dir)                   # GPU used if available
    # 2. Exhaustive matching (works for unordered sets)
    pcm.match_exhaustive(db_path)
    # 3. Incremental SfM
    maps = pcm.incremental_mapping(db_path, image_dir, model_root)
    # 4. Write the first (usually best) reconstruction
    maps[0].write(model_path)          # cameras.bin / images.bin / points3D.bin
    print("Reconstruction saved to", model_path)

if __name__ == "__main__":
    reconstruct("resources/dataset/tandt_db/tandt/truck/images", "resources/dataset/truck_colmap")
