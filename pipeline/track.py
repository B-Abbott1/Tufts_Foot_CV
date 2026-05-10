"""ByteTrack multi-object tracking on detection Parquet; write tracking CSV/Parquet."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import supervision as sv


def _box_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = np.maximum(0.0, a[:, 2] - a[:, 0]) * np.maximum(0.0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0.0, b[:, 2] - b[:, 0]) * np.maximum(0.0, b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter, dtype=np.float32), where=union > 0)


def _labels_for_tracked_boxes(tracked_xyxy: np.ndarray, source_rows: pd.DataFrame | None) -> tuple[list[str], list[float]]:
    if source_rows is None or source_rows.empty or "team" not in source_rows.columns:
        return ["unknown"] * len(tracked_xyxy), [0.0] * len(tracked_xyxy)

    source_xyxy = source_rows[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32)
    iou = _box_iou_matrix(tracked_xyxy.astype(np.float32), source_xyxy)
    best = iou.argmax(axis=1) if iou.size else np.zeros(len(tracked_xyxy), dtype=np.int64)
    best_iou = iou.max(axis=1) if iou.size else np.zeros(len(tracked_xyxy), dtype=np.float32)

    teams: list[str] = []
    confs: list[float] = []
    team_values = source_rows["team"].fillna("unknown").astype(str).to_numpy()
    if "team_confidence" in source_rows.columns:
        conf_values = source_rows["team_confidence"].fillna(0.0).to_numpy(dtype=np.float32)
    else:
        conf_values = np.zeros(len(source_rows), dtype=np.float32)

    for i in range(len(tracked_xyxy)):
        if best_iou[i] < 0.30:
            teams.append("unknown")
            confs.append(0.0)
            continue
        src_i = int(best[i])
        teams.append(str(team_values[src_i]))
        confs.append(float(conf_values[src_i]))
    return teams, confs


def _smooth_track_team_labels(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "team" not in df.columns:
        return df
    out = df.copy()
    if "team_confidence" not in out.columns:
        out["team_confidence"] = 0.0

    for (_, _), idx in out.groupby(["play_id", "player_id"], sort=False).groups.items():
        sub = out.loc[idx]
        scores: dict[str, float] = {}
        for _, row in sub.iterrows():
            team = str(row.get("team", "unknown"))
            if team == "unknown":
                continue
            weight = max(0.05, float(row.get("team_confidence", 0.0)))
            scores[team] = scores.get(team, 0.0) + weight
        if not scores:
            continue
        team, score = max(scores.items(), key=lambda kv: kv[1])
        total = sum(scores.values())
        smoothed_conf = float(np.clip(score / total if total > 0 else 0.0, 0.0, 0.99))
        out.loc[idx, "team"] = team
        out.loc[idx, "team_confidence"] = np.maximum(
            out.loc[idx, "team_confidence"].to_numpy(dtype=np.float32),
            smoothed_conf,
        )
    return out


def _frame_detections_from_rows(
    sub: pd.DataFrame,
) -> tuple[sv.Detections, tuple[int, int]]:
    if sub.empty:
        return sv.Detections.empty(), (0, 0)
    w = int(sub["frame_width"].iloc[0])
    h = int(sub["frame_height"].iloc[0])
    xyxy = sub[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32)
    confidence = sub["confidence"].to_numpy(dtype=np.float32)
    class_id = sub["class_id"].to_numpy(dtype=np.int32)
    dets = sv.Detections(xyxy=xyxy, confidence=confidence, class_id=class_id)
    return dets, (w, h)


def track_play(
    play_df: pd.DataFrame,
    game_id: str,
    play_id: str,
    tracker: sv.ByteTrack,
) -> List[dict]:
    """Run tracker on one play; returns list of per-detection records with tracker ids."""
    if play_df.empty:
        return []

    fmin = int(play_df["frame_id"].min())
    fmax = int(play_df["frame_id"].max())
    by_frame = {int(k): v for k, v in play_df.groupby("frame_id")}
    rows: List[dict] = []

    for fid in range(fmin, fmax + 1):
        sub = by_frame.get(fid)
        if sub is None:
            dets = sv.Detections.empty()
            wh = (0, 0)
        else:
            dets, wh = _frame_detections_from_rows(sub)

        tracked = tracker.update_with_detections(dets)
        if tracked.xyxy is None or len(tracked) == 0:
            continue

        xyxy = tracked.xyxy
        conf = tracked.confidence if tracked.confidence is not None else np.ones(len(xyxy))
        tids = tracked.tracker_id
        if tids is None:
            continue

        teams, team_confs = _labels_for_tracked_boxes(xyxy, sub)
        for i in range(len(xyxy)):
            tid = tids[i]
            if tid is None or (isinstance(tid, float) and np.isnan(tid)):
                continue
            x1, y1, x2, y2 = xyxy[i].tolist()
            rows.append(
                {
                    "frame_id": fid,
                    "play_id": play_id,
                    "game_id": game_id,
                    "player_id": int(tid),
                    "team": teams[i],
                    "team_confidence": float(team_confs[i]),
                    "x": np.nan,
                    "y": np.nan,
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "confidence": float(conf[i]) if i < len(conf) else 1.0,
                    "timestamp": np.nan,
                }
            )
    return rows


def track_from_detections_df(df: pd.DataFrame) -> pd.DataFrame:
    """Split by play_id, new ByteTrack per play, concatenate tracking rows."""
    all_rows: List[dict] = []
    game_id = str(df["game_id"].iloc[0])

    for play_id, play_df in df.groupby("play_id", sort=True):
        play_df = play_df.sort_values("frame_id")
        tracker = sv.ByteTrack()
        all_rows.extend(track_play(play_df, game_id, str(play_id), tracker))

    if not all_rows:
        return pd.DataFrame(
            columns=[
                "frame_id",
                "play_id",
                "game_id",
                "player_id",
                "team",
                "team_confidence",
                "x",
                "y",
                "x1",
                "y1",
                "x2",
                "y2",
                "confidence",
                "timestamp",
            ]
        )
    return _smooth_track_team_labels(pd.DataFrame(all_rows))


def track_from_parquet(
    detections_parquet: Path,
    out_path: Path,
    *,
    format: str = "parquet",
) -> Path:
    df = pd.read_parquet(detections_parquet)
    required = {"game_id", "play_id", "frame_id", "x1", "y1", "x2", "y2", "confidence", "class_id", "frame_width", "frame_height"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Detections Parquet missing columns: {missing}")

    out_df = track_from_detections_df(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = format.lower().strip()
    if fmt == "parquet":
        out_df.to_parquet(out_path, index=False)
    elif fmt in ("csv", "text"):
        out_df.to_csv(out_path, index=False)
    else:
        raise ValueError(f"Unknown format: {format}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="ByteTrack on detection Parquet.")
    parser.add_argument(
        "--detections",
        type=Path,
        required=True,
        help="Parquet from pipeline/detect.py (single game, one or more plays)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path (default data/tracking/{game_id}.parquet)",
    )
    parser.add_argument(
        "--format",
        choices=("parquet", "csv"),
        default="parquet",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.detections)
    if df.empty:
        raise SystemExit("Detections file is empty.")
    game_id = str(df["game_id"].iloc[0])
    out = args.out or Path("data/tracking") / f"{game_id}.parquet"
    if args.format == "csv" and args.out is None:
        out = Path("data/tracking") / f"{game_id}.csv"

    path = track_from_parquet(args.detections, out, format=args.format)
    print(f"Wrote tracking to {path}")


if __name__ == "__main__":
    main()
