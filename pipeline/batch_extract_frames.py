"""Batch-extract frames from a plays manifest (one MP4 per play)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pipeline.extract_frames import extract_frames
from pipeline.play_manifest import load_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read data/plays/manifest.csv and extract frames per row into data/frames/{game_id}/{play_id}/."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/plays/manifest.csv"),
        help="CSV with columns game_id, play_id, source_video",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("data/frames"),
        help="Root folder for extracted frames",
    )
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=1,
        help="Among FPS-selected frames, keep every Nth (default 1).",
    )
    parser.add_argument(
        "--max-frames-per-play",
        type=int,
        default=None,
        help="Cap frames written per play.",
    )
    parser.add_argument(
        "--mode",
        choices=("fps", "percentiles"),
        default="fps",
        help="fps: target-fps sampling; percentiles: frames at --percentiles along clip.",
    )
    parser.add_argument(
        "--percentiles",
        type=str,
        default=None,
        help="Comma-separated fractions for percentiles mode, e.g. 0.1,0.4,0.7",
    )
    parser.add_argument("--image-ext", type=str, default=".jpg")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="If play output dir already has images, skip that row.",
    )
    args = parser.parse_args()

    pct = None
    if args.percentiles:
        pct = [float(x.strip()) for x in args.percentiles.split(",") if x.strip()]

    df = load_manifest(args.manifest)
    total_written = 0
    errors = 0
    for _, row in df.iterrows():
        game_id = str(row["game_id"])
        play_id = str(row["play_id"])
        video = Path(row["source_video"]).expanduser()
        out_dir = args.out_root / game_id / play_id
        if args.skip_existing and out_dir.is_dir():
            existing = list(out_dir.glob(f"*{args.image_ext}"))
            if existing:
                print(f"skip existing {game_id}/{play_id} ({len(existing)} frames)", file=sys.stderr)
                continue
        if not video.is_file():
            print(f"ERROR missing video: {video}", file=sys.stderr)
            errors += 1
            continue
        try:
            n = extract_frames(
                video,
                out_dir,
                target_fps=args.target_fps,
                image_ext=args.image_ext,
                sample_stride=args.sample_stride,
                max_frames=args.max_frames_per_play,
                extraction_mode=args.mode,
                percentile_fracs=pct,
            )
            total_written += n
            print(f"{game_id}/{play_id}: wrote {n} frames -> {out_dir}")
        except Exception as e:  # noqa: BLE001
            print(f"ERROR {game_id}/{play_id}: {e}", file=sys.stderr)
            errors += 1

    print(f"Total frames written: {total_written}")
    if errors:
        raise SystemExit(errors)


if __name__ == "__main__":
    main()
