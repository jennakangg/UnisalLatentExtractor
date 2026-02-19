from pathlib import Path
import shutil

import cv2
import fire
import torch

import run as run_unisal

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def _find_videos(data_root: Path):
    exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    videos = []
    for video_dir in data_root.glob("*/video"):
        if not video_dir.is_dir():
            continue
        for file in sorted(video_dir.glob("*")):
            if file.suffix.lower() in exts:
                videos.append(file)
    return videos


def _extract_frames(video_path: Path, images_dir: Path, frame_step: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = None
    if tqdm is not None and total_frames > 0:
        pbar = tqdm(total=total_frames, desc=f"Extracting {video_path.stem}", unit="frame")

    frame_idx = 0
    saved_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % frame_step == 0:
            saved_idx += 1
            out_file = images_dir / f"{saved_idx:06d}.jpg"
            cv2.imwrite(str(out_file), frame)
        frame_idx += 1
        if pbar is not None:
            pbar.update(1)

    cap.release()
    if pbar is not None:
        pbar.close()

    if saved_idx == 0:
        raise RuntimeError(f"No frames extracted from {video_path}")


def export(
    data_root,
    train_id=None,
    source="DHF1K",
    model_domain=None,
    frame_step=1,
    output_dirname="saliency_unisal_maps",
    keep_workdir=False,
):
    """
    Export saliency map images for DIEM-style folders.

    Expected layout:
      <DIEM_ROOT>/data/<video_name>/video/<video_name>.mp4

    Saves:
      <DIEM_ROOT>/<output_dirname>/<video_name>/*.jpg
    """
    data_root = Path(data_root).resolve()
    if not data_root.exists() or not data_root.is_dir():
        raise FileNotFoundError(f"data_root not found: {data_root}")
    if frame_step < 1:
        raise ValueError("frame_step must be >= 1")

    diem_root = data_root.parent
    output_root = diem_root / output_dirname
    work_root = diem_root / "_saliency_unisal_maps_workdir"
    output_root.mkdir(parents=True, exist_ok=True)
    work_root.mkdir(parents=True, exist_ok=True)

    videos = _find_videos(data_root)
    if not videos:
        raise RuntimeError(f"No videos found under: {data_root}")

    print(f"Found {len(videos)} videos under {data_root}")
    print(f"Saving saliency maps to: {output_root}")

    trainer = run_unisal.load_trainer(train_id)
    trainer.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trainer.model.to(trainer.device)
    print(f"Using device: {trainer.device}")

    failed = []
    load_weights = True

    iterator = videos
    if tqdm is not None:
        iterator = tqdm(videos, desc="Videos", unit="video")

    for video_path in iterator:
        video_name = video_path.stem
        work_dir = work_root / video_name
        images_dir = work_dir / "images"
        saliency_dir = work_dir / "saliency"
        final_dir = output_root / video_name

        try:
            images_dir.mkdir(parents=True, exist_ok=True)
            _extract_frames(video_path, images_dir, frame_step)

            trainer.generate_predictions_from_path(
                work_dir,
                True,
                source=source,
                model_domain=model_domain,
                load_weights=load_weights,
            )
            load_weights = False

            final_dir.mkdir(parents=True, exist_ok=True)
            for img_file in sorted(saliency_dir.glob("*")):
                if img_file.is_file():
                    shutil.copy2(img_file, final_dir / img_file.name)
        except Exception as exc:
            failed.append((str(video_path), str(exc)))
        finally:
            if not keep_workdir and work_dir.exists():
                shutil.rmtree(work_dir, ignore_errors=True)

    print(f"Completed. Success: {len(videos) - len(failed)}, Failed: {len(failed)}")
    if failed:
        print("Failed videos:")
        for path, err in failed:
            print(f"- {path} :: {err}")


if __name__ == "__main__":
    fire.Fire({"export": export})
