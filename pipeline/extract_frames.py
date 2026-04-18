"""Extract frames from game film at a target frame rate (default 10 fps)."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from utils.video import frame_step_for_target_fps, get_video_properties, open_capture


def extract_frames(
    video_path: Path,
    out_dir: Path,
    *,
    target_fps: float = 10.0,
    image_ext: str = ".jpg",
) -> int:
    """
    Write sampled frames to ``out_dir`` as ``frame_{index:06d}{image_ext}``.
    Returns number of frames written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    props = get_video_properties(video_path)
    fps = props.fps if props.fps > 0 else 30.0
    step = frame_step_for_target_fps(fps, target_fps)

    cap = open_capture(video_path)
    written = 0
    out_index = 0
    try:
        src_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if src_index % step == 0:
                name = f"frame_{out_index:06d}{image_ext}"
                out_path = out_dir / name
                cv2.imwrite(str(out_path), frame)
                written += 1
                out_index += 1
            src_index += 1
    finally:
        cap.release()
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract frames from MP4 at ~target FPS.")
    parser.add_argument("video", type=Path, help="Path to input .mp4")
    parser.add_argument("--game-id", required=True, help="e.g. g01")
    parser.add_argument("--play-id", required=True, help="e.g. p004")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("data/frames"),
        help="Root folder; frames go to out-root/game_id/play_id/",
    )
    parser.add_argument("--target-fps", type=float, default=10.0, help="Sampling rate (default 10)")
    args = parser.parse_args()

    out_dir = args.out_root / args.game_id / args.play_id
    n = extract_frames(args.video, out_dir, target_fps=args.target_fps)
    print(f"Wrote {n} frames to {out_dir}")


if __name__ == "__main__":
    main()
