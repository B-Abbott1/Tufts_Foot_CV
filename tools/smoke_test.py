"""
End-to-end smoke test: synthetic MP4 → extract_frames → detect → track.

Run from repo root: python tools/smoke_test.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import ultralytics

from pipeline.detect import detect_for_play
from pipeline.extract_frames import extract_frames
from pipeline.track import track_from_parquet


def main() -> int:
    assets = Path(ultralytics.__file__).resolve().parent / "assets"
    sample = assets / "bus.jpg"
    if not sample.is_file():
        print("Missing ultralytics sample image:", sample, file=sys.stderr)
        return 1

    img = cv2.imread(str(sample))
    if img is None:
        print("Could not read sample image", file=sys.stderr)
        return 1

    game_id, play_id = "g98", "p001"
    raw_dir = ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    video_path = raw_dir / "smoke.mp4"

    frames_for_video = []
    for i in range(24):
        f = img.copy()
        cv2.putText(
            f,
            str(i),
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )
        frames_for_video.append(f)

    h, w = frames_for_video[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(video_path), fourcc, 30.0, (w, h))
    if not vw.isOpened():
        print("Could not open VideoWriter for", video_path, file=sys.stderr)
        return 1
    for f in frames_for_video:
        vw.write(f)
    vw.release()

    out_frames = ROOT / "data" / "frames" / game_id / play_id
    if out_frames.exists():
        shutil.rmtree(out_frames)
    n_ext = extract_frames(video_path, out_frames, target_fps=10.0)
    if n_ext < 1:
        print("Smoke test FAILED: extract_frames wrote no frames", file=sys.stderr)
        return 1

    det_path = ROOT / "data" / "detections" / f"{game_id}_{play_id}.parquet"
    detect_for_play(
        ROOT / "data" / "frames",
        game_id,
        play_id,
        "yolo11s.pt",
        det_path,
        conf=0.05,
        iou=0.55,
        imgsz=1280,
        classes=[0],
        tile_cols=2,
        tile_rows=1,
        tile_overlap=0.15,
        merge_iou=0.50,
        batch_size=4,
        device="cpu",
    )

    track_path = ROOT / "data" / "tracking" / f"{game_id}.parquet"
    track_from_parquet(det_path, track_path, format="parquet")

    import pandas as pd

    tdf = pd.read_parquet(track_path)
    if tdf.empty:
        print("Smoke test FAILED: no tracking rows", file=sys.stderr)
        return 1
    print("Smoke test OK: extracted", n_ext, "frames; ", len(tdf), "tracking rows")
    print(tdf.head())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
