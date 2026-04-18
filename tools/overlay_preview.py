"""Draw tracked boxes and player_id on one frame (sanity check)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Overlay tracking on a single frame.")
    parser.add_argument("--tracking", type=Path, required=True, help="Tracking Parquet or CSV")
    parser.add_argument("--frame", type=Path, required=True, help="BGR image path")
    parser.add_argument("--frame-id", type=int, required=True)
    parser.add_argument("--out", type=Path, default=Path("overlay_preview.jpg"))
    args = parser.parse_args()

    suf = args.tracking.suffix.lower()
    if suf == ".parquet":
        df = pd.read_parquet(args.tracking)
    else:
        df = pd.read_csv(args.tracking)

    sub = df[df["frame_id"] == args.frame_id]
    img = cv2.imread(str(args.frame))
    if img is None:
        print("Could not read frame:", args.frame, file=sys.stderr)
        return 1

    for _, row in sub.iterrows():
        x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
        pid = int(row["player_id"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            str(pid),
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    cv2.imwrite(str(args.out), img)
    print("Wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
