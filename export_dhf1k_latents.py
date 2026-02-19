from pathlib import Path
import shutil

import fire
import torch

import run as run_unisal

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def _find_videos(video_root: Path):
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    videos = []
    for file in sorted(video_root.glob("*")):
        if file.is_file() and file.suffix.lower() in exts:
            videos.append(file)
    return videos


def export(
    video_root,
    train_id=None,
    source="DHF1K",
    model_domain=None,
    frame_step=1,
    keep_workdir=False,
):
    """
    Export per-frame UNISAL latents for DHF1K flat video layout.

    Expected layout:
      <DHF1K_ROOT>/video/<video_name>.(AVI|mp4|...)

    Saves:
      <DHF1K_ROOT>/saliency_unisal_latents/<video_name>/*.pt
    """
    video_root = Path(video_root).resolve()
    if not video_root.exists() or not video_root.is_dir():
        raise FileNotFoundError(f"video_root not found: {video_root}")

    if frame_step < 1:
        raise ValueError("frame_step must be >= 1")

    dhf1k_root = video_root.parent if video_root.name.lower() == "video" else video_root
    actual_video_root = dhf1k_root / "video" if dhf1k_root != video_root else video_root
    if not actual_video_root.exists() or not actual_video_root.is_dir():
        raise FileNotFoundError(f"video directory not found: {actual_video_root}")

    latent_root = dhf1k_root / "saliency_unisal_latents"
    work_root = dhf1k_root / "_saliency_unisal_workdir"
    latent_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    videos = _find_videos(actual_video_root)
    if not videos:
        raise RuntimeError(f"No videos found under: {actual_video_root}")

    print(f"Found {len(videos)} videos under {actual_video_root}")
    print(f"Saving latents to: {latent_root}")

    run_unisal.LATENT_SAVE_DIR = latent_root
    run_unisal.LATENT_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    run_unisal.LOG_LATENT_SHAPES = False

    first_latent_shape = None
    failed = []

    iterator = videos
    if tqdm is not None:
        iterator = tqdm(videos, desc="Videos", unit="video")

    for video_path in iterator:
        try:
            run_unisal.predict_custom_video(
                video_path=str(video_path),
                train_id=train_id,
                source=source,
                model_domain=model_domain,
                output_root=str(work_root),
                frame_step=frame_step,
            )

            if first_latent_shape is None:
                video_latent_dir = latent_root / video_path.stem
                pt_files = sorted(video_latent_dir.glob("*.pt"))
                if pt_files:
                    latent = torch.load(pt_files[0], map_location="cpu")
                    first_latent_shape = tuple(latent.shape)

            print(first_latent_shape)
        except Exception as exc:
            failed.append((str(video_path), str(exc)))
        finally:
            if not keep_workdir:
                this_workdir = work_root / video_path.stem
                if this_workdir.exists():
                    shutil.rmtree(this_workdir, ignore_errors=True)

    print(f"Completed. Success: {len(videos) - len(failed)}, Failed: {len(failed)}")
    if first_latent_shape is not None:
        print(f"Latent vector size (C,H,W): {first_latent_shape}")
    else:
        print("Latent vector size unavailable (no latent file found).")

    if failed:
        print("Failed videos:")
        for path, err in failed:
            print(f"- {path} :: {err}")


if __name__ == "__main__":
    fire.Fire({"export": export})
