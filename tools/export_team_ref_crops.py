"""Export torso crops from YOLO detection labels for team/ref classifier training."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from pipeline.play_manifest import load_game_splits

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _torso_xyxy_pixel(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> tuple[int, int, int, int]:
    """Same geometry as pipeline.team_separation._torso_crop (fractions of box)."""
    x1 = int(np.clip(x1, 0, w - 1))
    x2 = int(np.clip(x2, x1 + 1, w))
    y1 = int(np.clip(y1, 0, h - 1))
    y2 = int(np.clip(y2, y1 + 1, h))
    box_w = x2 - x1
    box_h = y2 - y1
    tx1 = int(np.clip(x1 + 0.18 * box_w, 0, w - 1))
    tx2 = int(np.clip(x1 + 0.82 * box_w, tx1 + 1, w))
    ty1 = int(np.clip(y1 + 0.18 * box_h, 0, h - 1))
    ty2 = int(np.clip(y1 + 0.66 * box_h, ty1 + 1, h))
    return tx1, ty1, tx2, ty2


def _yolo_norm_line_to_xyxy(
    parts: list[str], img_w: int, img_h: int
) -> tuple[int, float, float, float, float]:
    cls_id = int(float(parts[0]))
    xc, yc, bw, bh = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
    x1 = (xc - bw / 2.0) * img_w
    x2 = (xc + bw / 2.0) * img_w
    y1 = (yc - bh / 2.0) * img_h
    y2 = (yc + bh / 2.0) * img_h
    return cls_id, x1, y1, x2, y2


def _load_team_box_csv(path: Path) -> dict[tuple[str, str, str, int], str]:
    """Map (game_id, play_id, frame_stem, line_index) -> team label home|away|unknown."""
    if not path.is_file():
        raise FileNotFoundError(path)
    out: dict[tuple[str, str, str, int], str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"game_id", "play_id", "frame_stem", "line_index", "team"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"team CSV must have columns {required}, got {reader.fieldnames}")
        for row in reader:
            key = (
                str(row["game_id"]).strip(),
                str(row["play_id"]).strip(),
                str(row["frame_stem"]).strip(),
                int(row["line_index"]),
            )
            team = str(row["team"]).strip().lower()
            if team not in {"home", "away", "unknown"}:
                raise ValueError(f"Invalid team {team!r} in {path}; use home|away|unknown")
            out[key] = team
    return out


def export_crops(
    frames_root: Path,
    labels_root: Path,
    splits_path: Path,
    out_root: Path,
    *,
    player_yolo_class: int = 0,
    official_yolo_class: int = 1,
    team_box_csv: Path | None = None,
    manifest_csv: Path | None = None,
) -> Path:
    splits = load_game_splits(splits_path)
    team_lookup = _load_team_box_csv(team_box_csv) if team_box_csv else {}

    out_root = Path(out_root)
    for split in ("train", "val", "test"):
        for sub in ("home", "away", "official", "unknown"):
            (out_root / split / sub).mkdir(parents=True, exist_ok=True)

    manifest_path = manifest_csv or (out_root / "crop_manifest.csv")
    manifest_rows: list[dict[str, str | int]] = []

    frames_root = Path(frames_root)
    labels_root = Path(labels_root)
    n_crops = 0

    for game_dir in sorted(p for p in frames_root.iterdir() if p.is_dir()):
        game_id = game_dir.name
        split = splits.get(game_id)
        if split is None:
            print(f"warning: no split for game_id={game_id}, skip", file=sys.stderr)
            continue

        for play_dir in sorted(p for p in game_dir.iterdir() if p.is_dir()):
            play_id = play_dir.name
            for img_path in sorted(play_dir.iterdir()):
                if img_path.suffix.lower() not in IMAGE_EXTS:
                    continue
                stem = img_path.stem
                label_path = labels_root / game_id / play_id / f"{stem}.txt"
                if not label_path.is_file():
                    continue
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                h, w = image.shape[:2]
                lines = [
                    ln.strip()
                    for ln in label_path.read_text(encoding="utf-8").splitlines()
                    if ln.strip()
                ]
                for line_idx, line in enumerate(lines):
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cls_id, x1, y1, x2, y2 = _yolo_norm_line_to_xyxy(parts, w, h)
                    tx1, ty1, tx2, ty2 = _torso_xyxy_pixel(x1, y1, x2, y2, w, h)
                    crop = image[ty1:ty2, tx1:tx2]
                    if crop.size == 0:
                        continue

                    if cls_id == official_yolo_class:
                        folder = "official"
                    elif cls_id == player_yolo_class:
                        folder = team_lookup.get((game_id, play_id, stem, line_idx), "unknown")
                    else:
                        folder = "unknown"

                    base = f"{game_id}__{play_id}__{stem}__line{line_idx:03d}.jpg"
                    rel = f"{split}/{folder}/{base}"
                    out_path = out_root / split / folder / base
                    cv2.imwrite(str(out_path), crop)
                    n_crops += 1
                    manifest_rows.append(
                        {
                            "crop_relpath": rel.replace("\\", "/"),
                            "game_id": game_id,
                            "play_id": play_id,
                            "frame_stem": stem,
                            "line_index": line_idx,
                            "yolo_class_id": cls_id,
                            "team_folder": folder,
                        }
                    )

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        if manifest_rows:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
            writer.writeheader()
            writer.writerows(manifest_rows)

    print(f"Wrote {n_crops} crops under {out_root}")
    print(f"Manifest: {manifest_path}")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Torso crops from YOLO boxes: officials -> official/; players -> home|away from CSV or unknown/."
    )
    parser.add_argument("--frames-root", type=Path, default=Path("data/frames"))
    parser.add_argument(
        "--labels-root",
        type=Path,
        default=Path("data/labels"),
        help="YOLO txt per frame, parallel to frames layout",
    )
    parser.add_argument("--splits", type=Path, default=Path("data/plays/splits.csv"))
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("datasets/team_ref"),
        help="Output root with train|val|test/{home,away,official,unknown}/",
    )
    parser.add_argument("--player-yolo-class", type=int, default=0)
    parser.add_argument("--official-yolo-class", type=int, default=1)
    parser.add_argument(
        "--team-box-csv",
        type=Path,
        default=None,
        help="Optional CSV: game_id,play_id,frame_stem,line_index,team (home|away|unknown) for player boxes",
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=None,
        help="Write audit manifest here (default: out-root/crop_manifest.csv)",
    )
    args = parser.parse_args()

    export_crops(
        args.frames_root,
        args.labels_root,
        args.splits,
        args.out_root,
        player_yolo_class=args.player_yolo_class,
        official_yolo_class=args.official_yolo_class,
        team_box_csv=args.team_box_csv,
        manifest_csv=args.manifest_csv,
    )


if __name__ == "__main__":
    main()
