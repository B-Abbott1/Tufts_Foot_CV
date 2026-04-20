"""Draw detection boxes on one frame with x,y coordinate labels (debug)."""

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
    parser = argparse.ArgumentParser(
        description="Overlay raw detections on a frame with x,y labels (foot point and/or box center).",
    )
    parser.add_argument(
        "--detections",
        type=Path,
        required=True,
        help="Parquet from pipeline.detect (must include frame_id, x1..y2, confidence).",
    )
    parser.add_argument("--frame", type=Path, required=True, help="BGR image path (same frame as frame-id).")
    parser.add_argument("--frame-id", type=int, required=True)
    parser.add_argument("--out", type=Path, default=Path("overlay_detections_debug.jpg"))
    parser.add_argument(
        "--point",
        choices=("foot", "center", "both"),
        default="foot",
        help="foot: (cx, y2); center: (cx, cy); both: two lines of text.",
    )
    parser.add_argument(
        "--show-confidence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefix label with confidence (default: on). Use --no-show-confidence to omit.",
    )
    parser.add_argument("--font-scale", type=float, default=0.45, help="OpenCV font scale.")
    parser.add_argument("--thickness", type=int, default=1, help="Text and box line thickness.")
    args = parser.parse_args()

    suf = args.detections.suffix.lower()
    if suf != ".parquet":
        print("Only Parquet detections are supported.", file=sys.stderr)
        return 1

    df = pd.read_parquet(args.detections)
    sub = df[df["frame_id"] == args.frame_id]
    img = cv2.imread(str(args.frame))
    if img is None:
        print("Could not read frame:", args.frame, file=sys.stderr)
        return 1

    fs = float(args.font_scale)
    th = max(1, int(args.thickness))

    for _, row in sub.iterrows():
        x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
        conf = float(row.get("confidence", 0.0))
        cx = 0.5 * (float(x1) + float(x2))
        cy_c = 0.5 * (float(y1) + float(y2))
        cy_f = float(y2)

        if args.point == "foot":
            parts = [f"{int(round(cx))},{int(round(cy_f))}"]
        elif args.point == "center":
            parts = [f"{int(round(cx))},{int(round(cy_c))}"]
        else:
            parts = [f"ft {int(round(cx))},{int(round(cy_f))}", f"ctr {int(round(cx))},{int(round(cy_c))}"]

        if args.show_confidence:
            parts = [f"{conf:.2f}"] + parts

        label = " | ".join(parts)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), th)

        tx, ty = x1, max(12, y1 - 4)
        for line_i, line in enumerate(label.split(" | ")):
            y_line = ty - line_i * int(14 * fs + 8)
            y_line = max(12, y_line)
            cv2.putText(
                img,
                line,
                (tx, y_line),
                cv2.FONT_HERSHEY_SIMPLEX,
                fs,
                (0, 255, 255),
                th,
                lineType=cv2.LINE_AA,
            )

        # Dot at foot point for clarity when using foot or both
        if args.point in ("foot", "both"):
            cv2.circle(img, (int(round(cx)), int(round(cy_f))), 3, (255, 128, 0), -1, lineType=cv2.LINE_AA)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), img)
    print("Wrote", args.out, f"({len(sub)} boxes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
