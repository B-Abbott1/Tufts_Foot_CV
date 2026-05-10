"""Assign team/official labels to detection boxes using crop color features."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

FRAME_RE = re.compile(r"frame_0*(\d+)", re.IGNORECASE)
TEAM_COLUMNS = [
    "team",
    "team_confidence",
    "team_cluster",
    "official_score",
    "jersey_l",
    "jersey_a",
    "jersey_b",
    "jersey_s",
    "jersey_v",
]


def _frame_id_from_path(path: Path) -> Optional[int]:
    match = FRAME_RE.search(path.stem)
    if not match:
        return None
    return int(match.group(1))


def _index_frames(frames_root: Path, df: pd.DataFrame) -> dict[tuple[str, str, int], Path]:
    """Build lookup for (game_id, play_id, frame_id) to image path."""
    lookup: dict[tuple[str, str, int], Path] = {}
    game_plays = df[["game_id", "play_id"]].drop_duplicates()
    for _, row in game_plays.iterrows():
        game_id = str(row["game_id"])
        play_id = str(row["play_id"])
        play_dir = frames_root / game_id / play_id
        if not play_dir.is_dir():
            continue
        for path in sorted(play_dir.iterdir()):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            frame_id = _frame_id_from_path(path)
            if frame_id is None:
                continue
            lookup[(game_id, play_id, frame_id)] = path
    return lookup


def _torso_crop(image: np.ndarray, row: pd.Series) -> np.ndarray:
    h, w = image.shape[:2]
    x1 = int(np.clip(float(row["x1"]), 0, w - 1))
    x2 = int(np.clip(float(row["x2"]), x1 + 1, w))
    y1 = int(np.clip(float(row["y1"]), 0, h - 1))
    y2 = int(np.clip(float(row["y2"]), y1 + 1, h))
    box_w = x2 - x1
    box_h = y2 - y1
    # Focus on the jersey/torso region, avoiding helmets, legs, turf, and shadows.
    tx1 = int(np.clip(x1 + 0.18 * box_w, 0, w - 1))
    tx2 = int(np.clip(x1 + 0.82 * box_w, tx1 + 1, w))
    ty1 = int(np.clip(y1 + 0.18 * box_h, 0, h - 1))
    ty2 = int(np.clip(y1 + 0.66 * box_h, ty1 + 1, h))
    return image[ty1:ty2, tx1:tx2]


def _crop_features(crop: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Return jersey color features and an official-likeness score.

    Features are [Lab L, Lab a, Lab b, HSV saturation, HSV value], scaled to 0..1.
    """
    if crop.size == 0:
        return np.full(5, np.nan, dtype=np.float32), 0.0

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # Remove turf spill from the crop, but keep low-saturation uniforms like white jerseys.
    green = (hue >= 35) & (hue <= 95) & (sat > 35)
    usable = (val > 35) & ~green
    if int(np.count_nonzero(usable)) < max(6, crop.shape[0] * crop.shape[1] // 12):
        usable = val > 35

    if int(np.count_nonzero(usable)) == 0:
        return np.full(5, np.nan, dtype=np.float32), 0.0

    lab_px = lab[usable]
    hsv_px = hsv[usable]
    feature = np.array(
        [
            np.median(lab_px[:, 0]) / 255.0,
            np.median(lab_px[:, 1]) / 255.0,
            np.median(lab_px[:, 2]) / 255.0,
            np.median(hsv_px[:, 1]) / 255.0,
            np.median(hsv_px[:, 2]) / 255.0,
        ],
        dtype=np.float32,
    )

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    black_frac = float(np.mean(gray < 65))
    white_frac = float(np.mean(gray > 185))
    contrast = float(np.std(gray) / 80.0)
    col_mean = gray.mean(axis=0)
    if len(col_mean) > 2:
        transitions = float(np.mean(np.abs(np.diff(col_mean))) / 45.0)
    else:
        transitions = 0.0
    low_sat_frac = float(np.mean(sat < 55))
    bw_balance = 2.0 * min(black_frac, white_frac)
    official_score = np.clip(
        0.35 * bw_balance + 0.25 * min(1.0, contrast) + 0.25 * min(1.0, transitions) + 0.15 * low_sat_frac,
        0.0,
        1.0,
    )
    return feature, float(official_score)


def _two_cluster(features: np.ndarray, max_iter: int = 25) -> tuple[np.ndarray, np.ndarray]:
    """Small deterministic k-means for two clusters."""
    n = len(features)
    if n == 0:
        return np.array([], dtype=np.int32), np.empty((0, features.shape[1] if features.ndim == 2 else 0))
    if n == 1:
        return np.zeros(1, dtype=np.int32), features[[0]].copy()

    dists = np.linalg.norm(features[:, None, :] - features[None, :, :], axis=2)
    i, j = np.unravel_index(int(np.argmax(dists)), dists.shape)
    centroids = features[[i, j]].astype(np.float32)
    labels = np.zeros(n, dtype=np.int32)

    for _ in range(max_iter):
        dist_to_centroids = np.linalg.norm(features[:, None, :] - centroids[None, :, :], axis=2)
        next_labels = np.argmin(dist_to_centroids, axis=1).astype(np.int32)
        if np.array_equal(labels, next_labels):
            break
        labels = next_labels
        for k in (0, 1):
            members = features[labels == k]
            if len(members):
                centroids[k] = members.mean(axis=0)

    # Stable names: team_a is the brighter/lower-saturation cluster when possible.
    if centroids.shape[0] == 2:
        order_score = centroids[:, 0] - 0.25 * centroids[:, 3]
        if order_score[1] > order_score[0]:
            labels = 1 - labels
            centroids = centroids[[1, 0]]
    return labels, centroids


def _cluster_confidence(feature: np.ndarray, centroids: np.ndarray, label: int) -> float:
    if centroids.shape[0] < 2 or not np.isfinite(feature).all():
        return 0.5
    d0 = float(np.linalg.norm(feature - centroids[0]))
    d1 = float(np.linalg.norm(feature - centroids[1]))
    own = d0 if label == 0 else d1
    other = d1 if label == 0 else d0
    margin = other - own
    return float(np.clip(0.50 + margin * 1.75, 0.50, 0.99))


def assign_teams_df(
    detections: pd.DataFrame,
    frames_root: Path,
    *,
    official_threshold: float = 0.52,
) -> pd.DataFrame:
    """Return detections with team labels and crop color diagnostics."""
    df = detections.copy()
    for col in TEAM_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df["team"] = "unknown"
    df["team_confidence"] = 0.0
    df["team_cluster"] = "unknown"
    df["official_score"] = 0.0

    if df.empty:
        return df
    required = {"game_id", "play_id", "frame_id", "x1", "y1", "x2", "y2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Detections missing columns required for team separation: {missing}")

    frame_lookup = _index_frames(frames_root, df)
    image_cache: dict[Path, np.ndarray] = {}
    features = np.full((len(df), 5), np.nan, dtype=np.float32)
    official_scores = np.zeros(len(df), dtype=np.float32)

    for pos, (_, row) in enumerate(df.iterrows()):
        key = (str(row["game_id"]), str(row["play_id"]), int(row["frame_id"]))
        frame_path = frame_lookup.get(key)
        if frame_path is None:
            continue
        image = image_cache.get(frame_path)
        if image is None:
            image = cv2.imread(str(frame_path))
            if image is None:
                continue
            image_cache[frame_path] = image
        feature, official_score = _crop_features(_torso_crop(image, row))
        features[pos] = feature
        official_scores[pos] = official_score

    df.loc[:, "official_score"] = official_scores
    for feature_i, col in enumerate(["jersey_l", "jersey_a", "jersey_b", "jersey_s", "jersey_v"]):
        df.loc[:, col] = features[:, feature_i]

    official_mask = official_scores >= float(official_threshold)
    df.loc[official_mask, "team"] = "official"
    df.loc[official_mask, "team_cluster"] = "official"
    df.loc[official_mask, "team_confidence"] = np.clip(official_scores[official_mask], 0.50, 0.99)

    for (_, _), idx in df.groupby(["game_id", "play_id"], sort=True).groups.items():
        positions = np.array(list(idx), dtype=object)
        int_positions = df.index.get_indexer(positions)
        valid = (~official_mask[int_positions]) & np.isfinite(features[int_positions]).all(axis=1)
        valid_positions = int_positions[valid]
        if len(valid_positions) == 0:
            continue
        if len(valid_positions) < 3:
            df.iloc[valid_positions, df.columns.get_loc("team")] = "team_a"
            df.iloc[valid_positions, df.columns.get_loc("team_cluster")] = "team_a"
            df.iloc[valid_positions, df.columns.get_loc("team_confidence")] = 0.50
            continue

        labels, centroids = _two_cluster(features[valid_positions])
        for arr_i, df_pos in enumerate(valid_positions):
            label = int(labels[arr_i])
            team = "team_a" if label == 0 else "team_b"
            df.iat[df_pos, df.columns.get_loc("team")] = team
            df.iat[df_pos, df.columns.get_loc("team_cluster")] = team
            df.iat[df_pos, df.columns.get_loc("team_confidence")] = _cluster_confidence(
                features[df_pos],
                centroids,
                label,
            )
    return df


def assign_teams_parquet(
    detections_path: Path,
    out_path: Path,
    frames_root: Path,
    *,
    official_threshold: float = 0.52,
) -> Path:
    df = pd.read_parquet(detections_path)
    out_df = assign_teams_df(df, frames_root, official_threshold=official_threshold)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign team/official labels to detection Parquet rows.")
    parser.add_argument("--detections", type=Path, required=True, help="Input detection Parquet.")
    parser.add_argument("--out", type=Path, default=None, help="Output Parquet path.")
    parser.add_argument("--frames-root", type=Path, default=Path("data/frames"))
    parser.add_argument(
        "--official-threshold",
        type=float,
        default=0.52,
        help="Official score threshold; higher means fewer official labels.",
    )
    args = parser.parse_args()

    out = args.out or args.detections.with_name(f"{args.detections.stem}_teams.parquet")
    path = assign_teams_parquet(
        args.detections,
        out,
        args.frames_root,
        official_threshold=float(np.clip(args.official_threshold, 0.0, 1.0)),
    )
    print(f"Wrote team labels to {path}")


if __name__ == "__main__":
    main()
