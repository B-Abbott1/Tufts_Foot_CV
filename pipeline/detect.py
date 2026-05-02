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


def _estimate_field_quad_from_image(
    image: np.ndarray,
    white_anchor_offset_px: float = 0.0,
    white_anchor_offset_frac: float = 0.10,
) -> Optional[np.ndarray]:
    """
    Estimate the 4-corner field quadrilateral from one broadcast sideline frame.
    Returns float32 points in order: top-left, top-right, bottom-right, bottom-left.
    """
    if image is None or image.size == 0:
        return None
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Broad turf mask (covers bright/dark turf under different lighting).
    mask1 = cv2.inRange(hsv, (25, 20, 20), (95, 255, 255))
    mask2 = cv2.inRange(hsv, (35, 10, 10), (110, 255, 220))
    mask = cv2.bitwise_or(mask1, mask2)
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Keep the largest connected turf-like region.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(np.argmax(areas))
    turf = (labels == largest_label).astype(np.uint8) * 255

    ys, xs = np.where(turf > 0)
    if len(xs) < 1000:
        return None
    # Build row support summaries for stable left/right extents.
    y_min, y_max = int(ys.min()), int(ys.max())
    row_left = np.full(h, np.nan, dtype=np.float32)
    row_right = np.full(h, np.nan, dtype=np.float32)
    for y in range(y_min, y_max + 1):
        row = np.where(turf[y] > 0)[0]
        if len(row) < max(20, w // 20):
            continue
        row_left[y] = float(row.min())
        row_right[y] = float(row.max())
    valid_rows = ~np.isnan(row_left)
    if valid_rows.sum() < max(40, h // 8):
        return None

    left_s = (
        pd.Series(row_left)
        .interpolate(limit_direction="both")
        .rolling(31, center=True, min_periods=1)
        .median()
        .to_numpy()
    )
    right_s = (
        pd.Series(row_right)
        .interpolate(limit_direction="both")
        .rolling(31, center=True, min_periods=1)
        .median()
        .to_numpy()
    )

    # White-line cue (sideline/yard-line paint) used to stabilize top boundary.
    # This is intentionally broad; we use it as a soft anchor, not a hard mask.
    white_mask = cv2.inRange(hsv, (0, 0, 150), (179, 70, 255))
    white_mask = cv2.medianBlur(white_mask, 5)
    # Favor long horizontal paint strokes (sideline-like) over general white clutter.
    hk = max(15, (w // 18) | 1)  # odd kernel width
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 3))
    white_horiz = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, h_kernel)

    # Column-wise boundary sampling from bottom-up:
    # find first turf pixel from bottom, then walk upward while allowing short
    # non-turf gaps (yard lines/logos) so we do not stop at midfield paint.
    x_step = max(4, w // 320)
    x_samples = np.arange(0, w, x_step, dtype=np.int32)
    top_pts: list[tuple[float, float]] = []
    gap_tol = max(4, h // 140)
    for x in x_samples.tolist():
        col = np.where(turf[:, x] > 0)[0]
        if len(col) == 0:
            continue
        yb = int(col[-1])
        y0 = yb
        gaps = 0
        while y0 > 0:
            if turf[y0 - 1, x] > 0:
                y0 -= 1
                continue
            if gaps < gap_tol and y0 > 1 and turf[y0 - 2, x] > 0:
                y0 -= 2
                gaps += 1
                continue
            break
        # Ignore implausible heights (sky/overlay noise, extreme bottom misses).
        if y0 < int(0.08 * h) or y0 > int(0.72 * h):
            continue
        top_pts.append((float(x), float(y0)))
    min_top_pts = max(10, w // 240)
    if len(top_pts) < min_top_pts:
        return None

    pts = np.array(top_pts, dtype=np.float32)
    x_all = pts[:, 0]
    y_all = pts[:, 1]

    # Robust line fit y = a*x + b with iterative MAD-based outlier rejection.
    inlier = np.ones(len(pts), dtype=bool)
    for _ in range(3):
        if inlier.sum() < max(8, len(pts) // 4):
            break
        x_fit = x_all[inlier]
        y_fit = y_all[inlier]
        a, b = np.polyfit(x_fit, y_fit, deg=1)
        pred = a * x_all + b
        resid = np.abs(y_all - pred)
        med = float(np.median(resid[inlier]))
        mad = float(np.median(np.abs(resid[inlier] - med)))
        scale = max(1.0, 1.4826 * mad)
        thr = max(8.0, 2.8 * scale)
        inlier = resid <= thr

    if inlier.sum() < max(8, len(pts) // 4):
        return None

    x_fit = x_all[inlier]
    y_fit = y_all[inlier]
    a, b = np.polyfit(x_fit, y_fit, deg=1)

    # Use stable row extents near the inferred top band for left/right anchors.
    y_mid = int(np.clip(np.median(a * x_fit + b), 0, h - 1))
    band = slice(max(0, y_mid - max(8, h // 80)), min(h, y_mid + max(8, h // 80) + 1))
    band_left = left_s[band]
    band_right = right_s[band]
    x_left = float(np.nanpercentile(band_left, 10)) if np.isfinite(band_left).any() else float(np.nanpercentile(left_s, 10))
    x_right = float(np.nanpercentile(band_right, 90)) if np.isfinite(band_right).any() else float(np.nanpercentile(right_s, 90))
    x_left = float(np.clip(x_left, 0.0, w - 1.0))
    x_right = float(np.clip(x_right, 0.0, w - 1.0))
    if x_right - x_left < 0.45 * w:
        # Fallback to full-mask horizontal span if band span is too narrow.
        x_left = float(np.clip(np.nanpercentile(left_s, 5), 0.0, w - 1.0))
        x_right = float(np.clip(np.nanpercentile(right_s, 95), 0.0, w - 1.0))
    if x_right - x_left < 0.45 * w:
        return None

    # Keep all corners on full image x-bounds to avoid cropping field width.
    x_left = 0.0
    x_right = float(w - 1)

    y_left = float(a * x_left + b)
    y_right = float(a * x_right + b)
    y_white_anchor_left: Optional[float] = None
    y_white_anchor_right: Optional[float] = None

    # White sideline first: derive top boundary from strong horizontal white paint.
    # In sideline footage this cue should always exist on at least one side.
    y_med = float(np.median(a * x_fit + b))
    lo = int(np.clip(0.02 * h, 0, h - 1))
    hi = int(np.clip(0.40 * h, 0, h - 1))
    if hi > lo:
        w_left = white_horiz[lo : hi + 1, : int(0.52 * w)]
        w_right = white_horiz[lo : hi + 1, int(0.48 * w) :]
        row_white_l = (w_left > 0).mean(axis=1) if w_left.size else np.array([])
        row_white_r = (w_right > 0).mean(axis=1) if w_right.size else np.array([])
        def _anchor_from_row_profile(row_white: np.ndarray) -> Optional[tuple[float, float]]:
            """
            Return (center_row_idx, thickness_rows) for strongest white horizontal band.
            Uses adaptive thresholding + contiguous run selection for robustness.
            """
            if row_white.size == 0:
                return None
            prof = (
                pd.Series(row_white)
                .rolling(5, center=True, min_periods=1)
                .mean()
                .to_numpy(dtype=np.float32)
            )
            pmax = float(np.max(prof))
            if pmax < 0.003:
                return None
            thr = max(0.003, 0.25 * pmax)
            hits = np.where(prof >= thr)[0]
            if hits.size == 0:
                return None

            # Build contiguous runs and pick the one with best score.
            runs: list[tuple[int, int]] = []
            s = int(hits[0])
            prev = int(hits[0])
            for idx in hits[1:]:
                idx_i = int(idx)
                if idx_i <= prev + 2:
                    prev = idx_i
                    continue
                runs.append((s, prev))
                s = idx_i
                prev = idx_i
            runs.append((s, prev))
            if not runs:
                return None

            # Keep thresholding/runs, then choose the topmost credible run:
            # - minimum run width
            # - minimum mean profile strength
            min_width = max(1, int(0.002 * h))
            pmax = float(np.max(prof))
            min_mean = max(0.004, 0.15 * pmax)
            width_valid_runs: list[tuple[int, int]] = []
            valid_runs: list[tuple[int, int]] = []
            for a_run, b_run in runs:
                width = b_run - a_run + 1
                if width < min_width:
                    continue
                width_valid_runs.append((a_run, b_run))
                mean_run = float(np.mean(prof[a_run : b_run + 1]))
                if mean_run < min_mean:
                    continue
                valid_runs.append((a_run, b_run))
            if not valid_runs:
                if not width_valid_runs:
                    return None
                valid_runs = width_valid_runs

            # Pick smallest row index => topmost credible sideline candidate.
            a_best, b_best = min(valid_runs, key=lambda ab: ab[0])
            thickness = float(max(1, b_best - a_best + 1))
            center = 0.5 * float(a_best + b_best)
            return center, thickness

        anchor_l = _anchor_from_row_profile(row_white_l)
        anchor_r = _anchor_from_row_profile(row_white_r)
        if anchor_l is not None:
            y_white_anchor_left = float(lo + anchor_l[0])
        if anchor_r is not None:
            y_white_anchor_right = float(lo + anchor_r[0])

        # Fallback pass: if horizontal-opened mask misses, search raw white mask.
        if y_white_anchor_left is None or y_white_anchor_right is None:
            w_left_raw = white_mask[lo : hi + 1, : int(0.52 * w)]
            w_right_raw = white_mask[lo : hi + 1, int(0.48 * w) :]
            row_white_l_raw = (w_left_raw > 0).mean(axis=1) if w_left_raw.size else np.array([])
            row_white_r_raw = (w_right_raw > 0).mean(axis=1) if w_right_raw.size else np.array([])
            if y_white_anchor_left is None:
                anchor_l_raw = _anchor_from_row_profile(row_white_l_raw)
                if anchor_l_raw is not None:
                    y_white_anchor_left = float(lo + anchor_l_raw[0])
            if y_white_anchor_right is None:
                anchor_r_raw = _anchor_from_row_profile(row_white_r_raw)
                if anchor_r_raw is not None:
                    y_white_anchor_right = float(lo + anchor_r_raw[0])

        # Force top-line to follow white sideline anchor(s) every time.
        if y_white_anchor_left is not None and y_white_anchor_right is not None:
            y_left = float(y_white_anchor_left)
            y_right = float(y_white_anchor_right)
        elif y_white_anchor_left is not None:
            y_left = float(y_white_anchor_left)
            y_right = float(y_white_anchor_left)
        elif y_white_anchor_right is not None:
            y_left = float(y_white_anchor_right)
            y_right = float(y_white_anchor_right)
        else:
            return None

    # Clamp top line into plausible vertical region.
    y_lo = 0.06 * h
    y_hi = 0.40 * h
    y_left = float(np.clip(y_left, y_lo, y_hi))
    y_right = float(np.clip(y_right, y_lo, y_hi))

    # Side-specific correction and safety:
    # set corners at the top of detected white-line boundary when available.
    anchor_offset_px = float(max(0.0, white_anchor_offset_px))
    anchor_offset_frac = float(max(0.0, white_anchor_offset_frac))
    if y_white_anchor_left is not None:
        # Estimate local white-band thickness on left side around anchor row.
        row_idx = int(np.clip(y_white_anchor_left, 0, h - 1))
        local = white_horiz[max(0, row_idx - max(6, h // 120)) : min(h, row_idx + max(6, h // 120) + 1), : int(0.52 * w)]
        thickness = float(max(1.0, np.mean((local > 0).sum(axis=0)) if local.size else 1.0))
        y_left = float(y_white_anchor_left + anchor_offset_px + anchor_offset_frac * thickness)
    if y_white_anchor_right is not None:
        row_idx = int(np.clip(y_white_anchor_right, 0, h - 1))
        local = white_horiz[max(0, row_idx - max(6, h // 120)) : min(h, row_idx + max(6, h // 120) + 1), int(0.48 * w) :]
        thickness = float(max(1.0, np.mean((local > 0).sum(axis=0)) if local.size else 1.0))
        y_right = float(y_white_anchor_right + anchor_offset_px + anchor_offset_frac * thickness)

    # Bottom anchors from smoothed field extents near frame bottom.
    bottom_y = int(np.nanmax(np.where(valid_rows)[0]))
    bottom_y = int(np.clip(bottom_y, int(0.80 * h), h - 1))
    x_bl = float(np.clip(left_s[bottom_y], 0.0, w - 1.0))
    x_br = float(np.clip(right_s[bottom_y], 0.0, w - 1.0))
    if not np.isfinite(x_bl):
        x_bl = x_left
    if not np.isfinite(x_br):
        x_br = x_right
    # Keep bottom corners on full image x-bounds as well.
    x_bl = 0.0
    x_br = float(w - 1)

    # Bottom edge BL–BR: use same slope as top sideline TL–TR so the quad is a
    # consistent trapezoid (not a skewed chevron from a flat bottom_y vs sloped top).
    # Anchor the bottom line at the turf bottom row mid-x, then lift if needed so
    # the bottom stays clearly below the top line (avoids shaving real players).
    dx_top = float(x_right - x_left)
    dy_top = float(y_right - y_left)
    if abs(dx_top) > 1e-3:
        m_edge = dy_top / dx_top
        x_mid_bt = 0.5 * (float(x_bl) + float(x_br))
        b_bot = float(bottom_y) - m_edge * x_mid_bt
        y_top_at_bl = float(y_left + m_edge * (float(x_bl) - x_left))
        y_top_at_br = float(y_left + m_edge * (float(x_br) - x_left))
        eps_bt = max(3.0, 0.004 * float(h))
        y_bl_cand = float(m_edge * float(x_bl) + b_bot)
        y_br_cand = float(m_edge * float(x_br) + b_bot)
        lift = max(0.0, y_top_at_bl + eps_bt - y_bl_cand, y_top_at_br + eps_bt - y_br_cand)
        b_bot += lift
        y_bl = float(np.clip(m_edge * float(x_bl) + b_bot, 0.0, float(h - 1)))
        y_br = float(np.clip(m_edge * float(x_br) + b_bot, 0.0, float(h - 1)))
    else:
        y_bl = float(bottom_y)
        y_br = float(bottom_y)

    tl = [x_left, y_left]
    tr = [x_right, y_right]
    br = [x_br, y_br]
    bl = [x_bl, y_bl]
    quad = np.array([tl, tr, br, bl], dtype=np.float32)

    # Reject degenerate / implausible quads.
    area = cv2.contourArea(quad)
    if area < float(h * w) * 0.18:
        return None
    if min(y_left, y_right) <= 0 or max(y_left, y_right) >= h - 1:
        return None
    return quad


def _build_field_homography(
    ref_image_path: Optional[str | Path],
    white_anchor_offset_px: float = 0.0,
    white_anchor_offset_frac: float = 0.10,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (H, H_inv, quad) that maps image pixels <-> normalized field [0,1]x[0,1]."""
    if not ref_image_path:
        return None, None, None
    img = cv2.imread(str(ref_image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read homography reference image: {ref_image_path}")
    quad = _estimate_field_quad_from_image(
        img,
        white_anchor_offset_px=white_anchor_offset_px,
        white_anchor_offset_frac=white_anchor_offset_frac,
    )
    if quad is None:
        raise RuntimeError("Could not estimate field quadrilateral from homography reference image.")
    dst = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    h_img_to_field = cv2.getPerspectiveTransform(quad, dst)
    h_field_to_img = cv2.getPerspectiveTransform(dst, quad)
    return h_img_to_field, h_field_to_img, quad


def _y_bottom_exclusion_at_cx(
    cx: np.ndarray,
    field_quad: np.ndarray,
    bottom_exclusion_px: float,
) -> np.ndarray:
    """
    Per-foot x, max allowed cy: parallel to quad bottom BL–BR, shifted **up** (smaller y)
    by bottom_exclusion_px so the cutoff sits above the field bottom toward the playing area.
    """
    bl = field_quad[3].astype(np.float64)
    br = field_quad[2].astype(np.float64)
    x1, y1 = bl[0], bl[1]
    x2, y2 = br[0], br[1]
    cx_d = np.asarray(cx, dtype=np.float64)
    if abs(x2 - x1) < 1e-3:
        y_line = np.full(cx_d.shape, max(y1, y2), dtype=np.float64)
    else:
        m = (y2 - y1) / (x2 - x1)
        y_line = m * (cx_d - x1) + y1
    # Pure parallel offset (no per-x clip): clipping would bend the line and break slope.
    y_excl = y_line - float(bottom_exclusion_px)
    return y_excl.astype(np.float32)


def _write_homography_debug_image(
    ref_image_path: str | Path,
    field_quad: np.ndarray,
    out_path: str | Path,
    bottom_exclusion_px: float = 0.0,
    top_inclusion_px: float = 0.0,
    show_white_mask: bool = False,
) -> None:
    """Write debug image overlay; optionally append white-line mask panel."""
    base = cv2.imread(str(ref_image_path))
    if base is None:
        raise FileNotFoundError(f"Could not read debug reference image: {ref_image_path}")
    h, w = base.shape[:2]
    img = base.copy()

    quad_i = field_quad.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [quad_i], isClosed=True, color=(0, 255, 255), thickness=3)

    tl = field_quad[0]
    tr = field_quad[1]
    bl = field_quad[3]
    br = field_quad[2]

    # Draw top field line (green), optional top inclusion line (cyan),
    # and bottom exclusion line (orange).
    top_p1 = (int(round(tl[0])), int(round(tl[1])))
    top_p2 = (int(round(tr[0])), int(round(tr[1])))
    cv2.line(img, top_p1, top_p2, (0, 255, 0), 3)
    incl = float(max(0.0, top_inclusion_px))
    if incl > 0.0:
        top_incl_p1 = (int(round(tl[0])), int(round(tl[1] - incl)))
        top_incl_p2 = (int(round(tr[0])), int(round(tr[1] - incl)))
        cv2.line(img, top_incl_p1, top_incl_p2, (255, 255, 0), 2)
    be = float(max(0.0, bottom_exclusion_px))
    if be > 0.0:
        bx1, by1 = float(bl[0]), float(bl[1]) - be
        bx2, by2 = float(br[0]), float(br[1]) - be
        mx = 0.5 * (bx1 + bx2)
        my = 0.5 * (by1 + by2)
        dx = bx2 - bx1
        dy = by2 - by1
        ln = float(np.hypot(dx, dy)) or 1.0
        scale = 2.0 * max(w, h) / ln
        p_ext1 = (int(round(mx - dx * scale)), int(round(my - dy * scale)))
        p_ext2 = (int(round(mx + dx * scale)), int(round(my + dy * scale)))
        ok_clip, p0, p1 = cv2.clipLine((0, 0, w, h), p_ext1, p_ext2)
        if not ok_clip:
            p0 = (int(round(bx1)), int(round(by1)))
            p1 = (int(round(bx2)), int(round(by2)))
        cv2.line(img, p0, p1, (0, 165, 255), 2)

    # Label corners for easier sanity checks.
    cv2.circle(img, (int(round(tl[0])), int(round(tl[1]))), 6, (255, 0, 0), -1)
    cv2.circle(img, (int(round(tr[0])), int(round(tr[1]))), 6, (255, 0, 0), -1)
    cv2.circle(img, (int(round(br[0])), int(round(br[1]))), 6, (255, 0, 0), -1)
    cv2.circle(img, (int(round(bl[0])), int(round(bl[1]))), 6, (255, 0, 0), -1)
    cv2.putText(img, "TL", (int(round(tl[0])) + 6, int(round(tl[1])) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(img, "TR", (int(round(tr[0])) + 6, int(round(tr[1])) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(img, "BR", (int(round(br[0])) + 6, int(round(br[1])) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    cv2.putText(img, "BL", (int(round(bl[0])) + 6, int(round(bl[1])) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.putText(
        img,
        "Green: top  Cyan: top incl.  Orange: bottom excl. (|| quad bottom)",
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    panel = img
    if show_white_mask:
        hsv = cv2.cvtColor(base, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, (0, 0, 150), (179, 70, 255))
        white_mask = cv2.medianBlur(white_mask, 5)
        hk = max(15, (w // 18) | 1)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 3))
        white_horiz = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, h_kernel)
        white_vis = cv2.cvtColor(white_horiz, cv2.COLOR_GRAY2BGR)
        cv2.putText(white_vis, "White-line mask (horizontal)", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.line(white_vis, top_p1, top_p2, (0, 255, 0), 2)
        if incl > 0.0:
            cv2.line(white_vis, top_incl_p1, top_incl_p2, (255, 255, 0), 2)
        if be > 0.0:
            cv2.line(white_vis, p0, p1, (0, 165, 255), 2)
        panel = cv2.hconcat([img, white_vis])
        cv2.putText(
            panel,
            "Overlay | White",
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), panel)
    if not ok:
        raise RuntimeError(f"Could not write homography debug image: {out_path}")


def _topline_gate(
    xyxy: np.ndarray,
    field_quad: Optional[np.ndarray],
    frame_h: Optional[int] = None,
    bottom_exclusion_px: float = 0.0,
    top_inclusion_px: float = 0.0,
) -> np.ndarray:
    """Keep detections below top line and with foot not below orange-style exclusion (parallel above BL–BR)."""
    if field_quad is None or len(xyxy) == 0:
        return np.ones(len(xyxy), dtype=bool)
    tl = field_quad[0]
    tr = field_quad[1]
    cx = 0.5 * (xyxy[:, 0] + xyxy[:, 2])
    dx = float(tr[0] - tl[0])
    if abs(dx) < 1e-6:
        y_top = np.full(len(xyxy), float(min(tl[1], tr[1])), dtype=np.float32)
    else:
        m = float((tr[1] - tl[1]) / dx)
        b = float(tl[1] - m * tl[0])
        y_top = (m * cx + b).astype(np.float32)
    cy = xyxy[:, 3]
    keep = cy >= (y_top - float(max(0.0, top_inclusion_px)))
    if frame_h is not None and frame_h > 0 and bottom_exclusion_px > 0.0:
        y_excl = _y_bottom_exclusion_at_cx(cx, field_quad, bottom_exclusion_px)
        # Do not let exclusion rise above top gate (avoids empty band if px is huge).
        y_excl = np.maximum(y_excl.astype(np.float32), y_top + 2.0)
        keep = keep & (cy <= y_excl)
    return keep


def _filter_by_field_homography(
    xyxy: np.ndarray,
    h_img_to_field: Optional[np.ndarray],
    margin_x: float,
    margin_y: float,
) -> np.ndarray:
    """Keep detections whose footpoints map inside the normalized field rectangle."""
    if h_img_to_field is None or len(xyxy) == 0:
        return np.ones(len(xyxy), dtype=bool)
    cx = 0.5 * (xyxy[:, 0] + xyxy[:, 2])
    cy = xyxy[:, 3]  # bottom-center (player footpoint proxy)
    pts = np.stack([cx, cy], axis=1).astype(np.float32).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(pts, h_img_to_field).reshape(-1, 2)
    x_ok = (mapped[:, 0] >= margin_x) & (mapped[:, 0] <= 1.0 - margin_x)
    y_ok = (mapped[:, 1] >= margin_y) & (mapped[:, 1] <= 1.0 - margin_y)
    finite = np.isfinite(mapped).all(axis=1)
    return x_ok & y_ok & finite


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
    homography_ref_image: Optional[str | Path] = None,
    field_margin_x: float = 0.01,
    field_margin_y: float = 0.02,
    top_inclusion_px: float = 0.0,
    bottom_exclusion_px: float = 25.0,
    min_box_h_frac: float = 0.0,
    homography_debug_image: Optional[str | Path] = None,
    homography_debug_show_white_mask: bool = False,
    white_anchor_offset_px: float = 3.0,
    white_anchor_offset_frac: float = 0.10,
) -> pd.DataFrame:
    model = YOLO(str(weights))
    if device:
        model.to(device)

    cls_filter = set(classes) if classes is not None else None
    h_img_to_field, _, field_quad = _build_field_homography(
        homography_ref_image,
        white_anchor_offset_px=white_anchor_offset_px,
        white_anchor_offset_frac=white_anchor_offset_frac,
    )
    if homography_debug_image is not None and homography_ref_image is not None and field_quad is not None:
        _write_homography_debug_image(
            homography_ref_image,
            field_quad,
            homography_debug_image,
            bottom_exclusion_px=bottom_exclusion_px,
            top_inclusion_px=top_inclusion_px,
            show_white_mask=homography_debug_show_white_mask,
        )
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
            keep_top = _topline_gate(
                xyxy_f,
                field_quad,
                frame_h=h,
                bottom_exclusion_px=bottom_exclusion_px,
                top_inclusion_px=top_inclusion_px,
            )
            xyxy_f = xyxy_f[keep_top]
            confs_f = confs_f[keep_top]
            clss_f = clss_f[keep_top]
            keep_field = _filter_by_field_homography(
                xyxy_f,
                h_img_to_field,
                margin_x=field_margin_x,
                margin_y=field_margin_y,
            )
            xyxy_f = xyxy_f[keep_field]
            confs_f = confs_f[keep_field]
            clss_f = clss_f[keep_field]
            if len(xyxy_f) and min_box_h_frac > 0.0:
                min_h = float(h) * float(min_box_h_frac)
                keep_h = (xyxy_f[:, 3] - xyxy_f[:, 1]) >= min_h
                xyxy_f = xyxy_f[keep_h]
                confs_f = confs_f[keep_h]
                clss_f = clss_f[keep_h]
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
            keep_top = _topline_gate(
                xyxy,
                field_quad,
                frame_h=h,
                bottom_exclusion_px=bottom_exclusion_px,
                top_inclusion_px=top_inclusion_px,
            )
            xyxy = xyxy[keep_top]
            confs = confs[keep_top]
            clss = clss[keep_top]
            keep_field = _filter_by_field_homography(
                xyxy,
                h_img_to_field,
                margin_x=field_margin_x,
                margin_y=field_margin_y,
            )
            xyxy = xyxy[keep_field]
            confs = confs[keep_field]
            clss = clss[keep_field]
            if len(xyxy) and min_box_h_frac > 0.0:
                min_h = float(h) * float(min_box_h_frac)
                keep_h = (xyxy[:, 3] - xyxy[:, 1]) >= min_h
                xyxy = xyxy[keep_h]
                confs = confs[keep_h]
                clss = clss[keep_h]
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
        default=0.30,
        help="Confidence threshold (lower = more recall; default tuned for crowded sports film).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.50,
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
        "--homography-ref-image",
        type=Path,
        default=None,
        help="Reference frame path used to estimate field homography and remove sideline detections.",
    )
    parser.add_argument(
        "--field-margin-x",
        type=float,
        default=0.01,
        help="Normalized horizontal in-field margin applied after homography (0..0.2).",
    )
    parser.add_argument(
        "--field-margin-y",
        type=float,
        default=0.03,
        help="Normalized vertical in-field margin applied after homography (0..0.2).",
    )
    parser.add_argument(
        "--top-inclusion-px",
        type=float,
        default=20.0,
        help="Allow detections this many pixels above the top line (recovers edge players).",
    )
    parser.add_argument(
        "--bottom-exclusion-px",
        type=float,
        default=75.0,
        help="Foot y must be <= quad bottom BL–BR at that x minus this offset (px; line parallel above field bottom).",
    )
    parser.add_argument(
        "--min-box-h-frac",
        type=float,
        default=0.0,
        help="Optional minimum box height as fraction of frame height (e.g. 0.02).",
    )
    parser.add_argument(
        "--homography-debug-image",
        type=Path,
        default=None,
        help="Optional output image path to visualize estimated field quad and top-line gate.",
    )
    parser.add_argument(
        "--homography-debug-show-white-mask",
        action="store_true",
        help="When set, append white-line mask panel next to the debug overlay.",
    )
    parser.add_argument(
        "--white-anchor-offset-px",
        type=float,
        default=0.0,
        help="Pixel offset added below detected white-line top anchor (helps avoid drawing above paint).",
    )
    parser.add_argument(
        "--white-anchor-offset-frac",
        type=float,
        default=0.10,
        help="Extra offset as a fraction of detected white-band thickness (more robust across zoom levels).",
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
        homography_ref_image=args.homography_ref_image,
        field_margin_x=max(0.0, min(0.2, args.field_margin_x)),
        field_margin_y=max(0.0, min(0.2, args.field_margin_y)),
        top_inclusion_px=max(0.0, args.top_inclusion_px),
        bottom_exclusion_px=max(0.0, args.bottom_exclusion_px),
        min_box_h_frac=max(0.0, min(0.25, args.min_box_h_frac)),
        homography_debug_image=args.homography_debug_image,
        homography_debug_show_white_mask=bool(args.homography_debug_show_white_mask),
        white_anchor_offset_px=max(0.0, args.white_anchor_offset_px),
        white_anchor_offset_frac=max(0.0, min(1.0, args.white_anchor_offset_frac)),
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
