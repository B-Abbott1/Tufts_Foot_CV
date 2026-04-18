"""Run YOLO inference on extracted frames; write detections Parquet for tracking."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.ops import nms
from ultralytics import YOLO

FRAME_RE = re.compile(r"frame_(\d+)", re.IGNORECASE)


def parse_frame_sort_key(path: Path) -> tuple:
    m = FRAME_RE.search(path.stem)
    n = int(m.group(1)) if m else 0
    return (n, path.name)


def list_frame_paths(frames_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = [p for p in frames_dir.iterdir() if p.suffix.lower() in exts]
    paths.sort(key=lambda p: parse_frame_sort_key(p))
    return paths


def frame_id_from_path(path: Path) -> int:
    m = FRAME_RE.search(path.stem)
    if not m:
        raise ValueError(f"Could not parse frame index from filename: {path.name}")
    return int(m.group(1))


def _tile_rects(w: int, h: int, cols: int, rows: int, overlap: float) -> List[Tuple[int, int, int, int]]:
    """Return (x0, y0, x1, y1) crop rectangles in full-image coordinates."""
    if cols < 1 or rows < 1:
        raise ValueError("tile cols/rows must be >= 1")
    if cols == 1 and rows == 1:
        return [(0, 0, w, h)]
    ox = int(w * overlap) if cols > 1 else 0
    oy = int(h * overlap) if rows > 1 else 0
    tw = max(32, (w + (cols - 1) * ox) // cols)
    th = max(32, (h + (rows - 1) * oy) // rows)
    xs = (
        [int(round(i * (w - tw) / (cols - 1))) for i in range(cols)]
        if cols > 1
        else [0]
    )
    ys = (
        [int(round(i * (h - th) / (rows - 1))) for i in range(rows)]
        if rows > 1
        else [0]
    )
    rects: List[Tuple[int, int, int, int]] = []
    for y0 in ys:
        y1 = min(h, y0 + th)
        y0 = max(0, y1 - th)
        for x0 in xs:
            x1 = min(w, x0 + tw)
            x0 = max(0, x1 - tw)
            rects.append((x0, y0, x1, y1))
    return rects


def _nms_xyxy(xyxy: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Return indices to keep after NMS (class-agnostic)."""
    if len(xyxy) == 0:
        return np.array([], dtype=np.int64)
    boxes = torch.from_numpy(xyxy.astype(np.float32))
    sc = torch.from_numpy(scores.astype(np.float32))
    keep = nms(boxes, sc, float(iou_threshold))
    return keep.cpu().numpy()


def _append_detection_rows(
    rows: list[dict],
    fid: int,
    w: int,
    h: int,
    xyxy: np.ndarray,
    confs: np.ndarray,
    clss: np.ndarray,
    cls_filter: Optional[set[int]],
) -> None:
    for i in range(len(xyxy)):
        c = int(clss[i])
        if cls_filter is not None and c not in cls_filter:
            continue
        x1, y1, x2, y2 = xyxy[i].tolist()
        x1 = float(max(0, min(w - 1, x1)))
        x2 = float(max(0, min(w, x2)))
        y1 = float(max(0, min(h - 1, y1)))
        y2 = float(max(0, min(h, y2)))
        if x2 <= x1 or y2 <= y1:
            continue
        rows.append(
            {
                "frame_id": fid,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "confidence": float(confs[i]),
                "class_id": c,
                "frame_width": int(w),
                "frame_height": int(h),
            }
        )


def run_detection_on_frames(
    frame_paths: Sequence[Path],
    weights: str | Path,
    *,
    conf: float = 0.05,
    iou: float = 0.55,
    imgsz: int = 1280,
    classes: Optional[Sequence[int]] = None,
    batch_size: int = 8,
    device: Optional[str] = None,
    tile_cols: int = 2,
    tile_rows: int = 1,
    tile_overlap: float = 0.15,
    merge_iou: float = 0.50,
) -> pd.DataFrame:
    model = YOLO(str(weights))
    if device:
        model.to(device)

    cls_filter = set(classes) if classes is not None else None
    rows: list[dict] = []
    use_tiling = not (tile_cols == 1 and tile_rows == 1)
    paths = list(frame_paths)

    if use_tiling:
        for img_path in paths:
            fid = frame_id_from_path(img_path)
            img = cv2.imread(str(img_path))
            if img is None:
                raise FileNotFoundError(f"Could not read image: {img_path}")
            h, w = img.shape[:2]
            rects = _tile_rects(w, h, tile_cols, tile_rows, tile_overlap)
            all_xy: list[np.ndarray] = []
            all_cf: list[np.ndarray] = []
            all_cl: list[np.ndarray] = []
            for x0, y0, x1, y1 in rects:
                crop = img[y0:y1, x0:x1]
                if crop.size == 0:
                    continue
                results = model.predict(
                    source=crop,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    verbose=False,
                    device=device,
                )
                r = results[0]
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                xyxy = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                clss = r.boxes.cls.cpu().numpy().astype(int)
                xyxy[:, [0, 2]] += x0
                xyxy[:, [1, 3]] += y0
                all_xy.append(xyxy)
                all_cf.append(confs)
                all_cl.append(clss)
            if not all_xy:
                continue
            xyxy_f = np.concatenate(all_xy, axis=0)
            confs_f = np.concatenate(all_cf, axis=0)
            clss_f = np.concatenate(all_cl, axis=0)
            if cls_filter is not None:
                keep_cls = np.isin(clss_f, list(cls_filter))
                xyxy_f = xyxy_f[keep_cls]
                confs_f = confs_f[keep_cls]
                clss_f = clss_f[keep_cls]
            if len(xyxy_f) == 0:
                continue
            keep = _nms_xyxy(xyxy_f, confs_f, merge_iou)
            xyxy_f = xyxy_f[keep]
            confs_f = confs_f[keep]
            clss_f = clss_f[keep]
            _append_detection_rows(rows, fid, w, h, xyxy_f, confs_f, clss_f, None)
        return pd.DataFrame(rows)

    # Ultralytics accepts a list of paths; process in chunks for memory
    for start in range(0, len(paths), batch_size):
        chunk = paths[start : start + batch_size]
        results = model.predict(
            source=[str(p) for p in chunk],
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            verbose=False,
            device=device,
        )
        for img_path, r in zip(chunk, results):
            fid = frame_id_from_path(img_path)
            h, w = r.orig_shape if r.orig_shape else (0, 0)
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.cpu().numpy().astype(int)
            _append_detection_rows(rows, fid, w, h, xyxy, confs, clss, cls_filter)

    return pd.DataFrame(rows)


def detect_for_play(
    frames_root: Path,
    game_id: str,
    play_id: str,
    weights: str | Path,
    out_path: Path,
    **kwargs,
) -> Path:
    frames_dir = frames_root / game_id / play_id
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
    paths = list_frame_paths(frames_dir)
    if not paths:
        raise FileNotFoundError(f"No images in {frames_dir}")

    df = run_detection_on_frames(paths, weights, **kwargs)
    df.insert(0, "game_id", game_id)
    df.insert(1, "play_id", play_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return out_path


def detect_all_plays_in_game(
    frames_root: Path,
    game_id: str,
    weights: str | Path,
    out_path: Path,
    **kwargs,
) -> Path:
    game_dir = frames_root / game_id
    if not game_dir.is_dir():
        raise FileNotFoundError(f"Game frames directory not found: {game_dir}")

    play_dirs = sorted([p for p in game_dir.iterdir() if p.is_dir()])
    if not play_dirs:
        raise FileNotFoundError(f"No play subfolders under {game_dir}")

    parts: list[pd.DataFrame] = []
    for play_dir in play_dirs:
        play_id = play_dir.name
        paths = list_frame_paths(play_dir)
        if not paths:
            continue
        df = run_detection_on_frames(paths, weights, **kwargs)
        df.insert(0, "game_id", game_id)
        df.insert(1, "play_id", play_id)
        parts.append(df)

    if not parts:
        raise RuntimeError(f"No frames found under any play in {game_dir}")

    out = pd.concat(parts, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO player detection on extracted frames.")
    parser.add_argument(
        "--weights",
        type=str,
        default="yolo11s.pt",
        help="YOLO weights path or hub name (e.g. yolo11s.pt or models/weights/your.pt)",
    )
    parser.add_argument("--frames-root", type=Path, default=Path("data/frames"))
    parser.add_argument("--game-id", required=True)
    parser.add_argument(
        "--play-id",
        default=None,
        help="Single play folder name; omit to process all plays under game-id",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output Parquet path (default data/detections/{game_id}.parquet or game_play.parquet)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.05,
        help="Confidence threshold (lower = more recall; default tuned for crowded sports film).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.55,
        help="NMS IoU inside YOLO predict (higher can help separate overlapping linemen).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Inference square size (~1900×930 sideline: 1280 default; slower on CPU).",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--no-tiling",
        action="store_true",
        help="Run one full-frame inference per image instead of overlapping crops.",
    )
    parser.add_argument(
        "--tile-cols",
        type=int,
        default=2,
        help="Horizontal tiles (sideline: 2 for near+far sideline). Ignored with --no-tiling.",
    )
    parser.add_argument(
        "--tile-rows",
        type=int,
        default=1,
        help="Vertical tiles (default 1). Use 2 for a 2×2 grid if needed.",
    )
    parser.add_argument(
        "--tile-overlap",
        type=float,
        default=0.15,
        help="Fraction of width/height used as overlap between adjacent tiles.",
    )
    parser.add_argument(
        "--merge-iou",
        type=float,
        default=0.50,
        help="IoU threshold for NMS when merging detections from overlapping tiles.",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="*",
        default=None,
        metavar="ID",
        help="YOLO class ids to keep. Default: 0 (COCO person). Use --all-classes for no filter.",
    )
    parser.add_argument(
        "--all-classes",
        action="store_true",
        help="Do not filter by class (all YOLO classes).",
    )
    parser.add_argument("--device", default=None, help="e.g. cuda:0 or cpu")
    args = parser.parse_args()

    if args.all_classes:
        classes_arg: Optional[Sequence[int]] = None
    elif args.classes is not None:
        # nargs=* can yield [] if user passed --classes with no ids; treat as default person
        classes_arg = args.classes if len(args.classes) > 0 else [0]
    else:
        classes_arg = [0]

    tile_cols = 1 if args.no_tiling else max(1, args.tile_cols)
    tile_rows = 1 if args.no_tiling else max(1, args.tile_rows)
    det_kw = dict(
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        classes=classes_arg,
        batch_size=args.batch_size,
        device=args.device,
        tile_cols=tile_cols,
        tile_rows=tile_rows,
        tile_overlap=args.tile_overlap,
        merge_iou=args.merge_iou,
    )

    if args.play_id:
        out = args.out or Path("data/detections") / f"{args.game_id}_{args.play_id}.parquet"
        path = detect_for_play(
            args.frames_root,
            args.game_id,
            args.play_id,
            args.weights,
            out,
            **det_kw,
        )
    else:
        out = args.out or Path("data/detections") / f"{args.game_id}.parquet"
        path = detect_all_plays_in_game(
            args.frames_root,
            args.game_id,
            args.weights,
            out,
            **det_kw,
        )
    print(f"Wrote detections to {path}")


if __name__ == "__main__":
    main()
