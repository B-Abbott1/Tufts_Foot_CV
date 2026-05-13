"""Extract frames from game film at a target frame rate (default 10 fps)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import cv2

from utils.video import frame_step_for_target_fps, get_video_properties, open_capture


def extract_frames(
    video_path: Path,
    out_dir: Path,
    *,
    target_fps: float = 10.0,
    image_ext: str = ".jpg",
    sample_stride: int = 1,
    max_frames: Optional[int] = None,
    extraction_mode: str = "fps",
    percentile_fracs: Optional[Sequence[float]] = None,
) -> int:
    """
    Write sampled frames to ``out_dir`` as ``frame_{index:06d}{image_ext}``.

    extraction_mode:
      - ``fps`` (default): sample by ``target_fps``, then keep every ``sample_stride``-th
        hit, stopping after ``max_frames`` writes if set.
      - ``percentiles``: ignore ``target_fps``; write one frame per fraction of clip
        length (default 0.1, 0.4, 0.7 of timeline). ``sample_stride`` / ``max_frames``
        still apply to the resulting sequence.

    Returns number of frames written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = extraction_mode.strip().lower()
    if mode == "percentiles":
        return _extract_frames_percentiles(
            video_path,
            out_dir,
            image_ext=image_ext,
            percentile_fracs=percentile_fracs,
            sample_stride=sample_stride,
            max_frames=max_frames,
        )
    if mode != "fps":
        raise ValueError(f"extraction_mode must be 'fps' or 'percentiles', got {extraction_mode!r}")

    props = get_video_properties(video_path)
    fps = props.fps if props.fps > 0 else 30.0
    step = frame_step_for_target_fps(fps, target_fps)
    stride = max(1, int(sample_stride))

    cap = open_capture(video_path)
    written = 0
    out_index = 0
    fps_hits = 0
    try:
        src_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if src_index % step == 0:
                if fps_hits % stride == 0:
                    name = f"frame_{out_index:06d}{image_ext}"
                    out_path = out_dir / name
                    cv2.imwrite(str(out_path), frame)
                    written += 1
                    out_index += 1
                    if max_frames is not None and written >= max_frames:
                        break
                fps_hits += 1
            src_index += 1
    finally:
        cap.release()
    return written


def _extract_frames_percentiles(
    video_path: Path,
    out_dir: Path,
    *,
    image_ext: str,
    percentile_fracs: Optional[Sequence[float]],
    sample_stride: int,
    max_frames: Optional[int],
) -> int:
    fracs = list(percentile_fracs) if percentile_fracs is not None else [0.1, 0.4, 0.7]
    fracs = [float(min(1.0, max(0.0, f))) for f in fracs]
    if not fracs:
        raise ValueError("percentile_fracs must be non-empty")

    props = get_video_properties(video_path)
    n = max(1, props.frame_count)
    indices = sorted({int(round(f * (n - 1))) for f in fracs})

    cap = open_capture(video_path)
    written = 0
    out_index = 0
    stride = max(1, int(sample_stride))
    try:
        for i, target_idx in enumerate(indices):
            if i % stride != 0:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            name = f"frame_{out_index:06d}{image_ext}"
            out_path = out_dir / name
            cv2.imwrite(str(out_path), frame)
            written += 1
            out_index += 1
            if max_frames is not None and written >= max_frames:
                break
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
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=1,
        help="Keep every Nth frame among those selected by target-fps (default 1 = all).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after writing this many frames (default: no limit).",
    )
    parser.add_argument(
        "--mode",
        choices=("fps", "percentiles"),
        default="fps",
        help="fps: target-fps sampling; percentiles: fixed fractions along clip (see --percentiles).",
    )
    parser.add_argument(
        "--percentiles",
        type=str,
        default=None,
        help="Comma-separated fractions in [0,1] for percentiles mode, e.g. 0.1,0.4,0.7",
    )
    args = parser.parse_args()

    pct = None
    if args.percentiles:
        pct = [float(x.strip()) for x in args.percentiles.split(",") if x.strip()]

    out_dir = args.out_root / args.game_id / args.play_id
    n = extract_frames(
        args.video,
        out_dir,
        target_fps=args.target_fps,
        sample_stride=args.sample_stride,
        max_frames=args.max_frames,
        extraction_mode=args.mode,
        percentile_fracs=pct,
    )
    print(f"Wrote {n} frames to {out_dir}")


if __name__ == "__main__":
    main()
