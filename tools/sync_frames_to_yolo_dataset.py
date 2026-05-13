"""Copy labeled frames from data/frames + parallel YOLO labels into Ultralytics dataset layout."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml

from pipeline.play_manifest import load_game_splits

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _unique_image_name(game_id: str, play_id: str, stem: str) -> str:
    return f"{game_id}__{play_id}__{stem}"


def sync_yolo_dataset(
    frames_root: Path,
    labels_root: Path,
    splits_path: Path,
    dataset_root: Path,
    *,
    symlink: bool = False,
    class_names: dict[int, str] | None = None,
) -> Path:
    """
    For each image under frames_root/{game}/{play}/, if labels_root/{game}/{play}/{same_stem}.txt exists,
    copy (or symlink) image and label into dataset_root/images/{split}/ and labels/{split}/.
    Image filenames are prefixed with game_id__play_id__ to avoid collisions.
    Writes dataset_root/data.yaml for Ultralytics.
    """
    splits = load_game_splits(splits_path)
    names = class_names if class_names is not None else {0: "player", 1: "official"}

    images_root = dataset_root / "images"
    labels_out_root = dataset_root / "labels"
    for sp in ("train", "val", "test"):
        (images_root / sp).mkdir(parents=True, exist_ok=True)
        (labels_out_root / sp).mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped_no_label = 0
    skipped_unknown_game = 0

    frames_root = Path(frames_root)
    labels_root = Path(labels_root)
    if not frames_root.is_dir():
        raise FileNotFoundError(f"frames_root not found: {frames_root}")

    for game_dir in sorted(p for p in frames_root.iterdir() if p.is_dir()):
        game_id = game_dir.name
        split = splits.get(game_id)
        if split is None:
            print(f"warning: no split for game_id={game_id}, skipping", file=sys.stderr)
            skipped_unknown_game += 1
            continue

        for play_dir in sorted(p for p in game_dir.iterdir() if p.is_dir()):
            play_id = play_dir.name
            for img_path in sorted(play_dir.iterdir()):
                if img_path.suffix.lower() not in IMAGE_EXTS:
                    continue
                stem = img_path.stem
                label_path = labels_root / game_id / play_id / f"{stem}.txt"
                if not label_path.is_file():
                    skipped_no_label += 1
                    continue
                base = _unique_image_name(game_id, play_id, stem)
                dst_img = images_root / split / f"{base}{img_path.suffix.lower()}"
                dst_lbl = labels_out_root / split / f"{base}.txt"
                if symlink:
                    if dst_img.exists() or dst_img.is_symlink():
                        dst_img.unlink()
                    if dst_lbl.exists() or dst_lbl.is_symlink():
                        dst_lbl.unlink()
                    dst_img.symlink_to(img_path.resolve())
                    dst_lbl.symlink_to(label_path.resolve())
                else:
                    shutil.copy2(img_path, dst_img)
                    shutil.copy2(label_path, dst_lbl)
                copied += 1

    data_yaml = dataset_root / "data.yaml"
    yaml_dict = {
        "path": str(dataset_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {int(k): v for k, v in sorted(names.items(), key=lambda kv: kv[0])},
    }
    with open(data_yaml, "w", encoding="utf-8") as f:
        yaml.dump(yaml_dict, f, default_flow_style=False, sort_keys=False)

    print(f"Copied or linked {copied} image+label pairs to {dataset_root}")
    print(f"Skipped (no label file): {skipped_no_label}")
    if skipped_unknown_game:
        print(f"Games skipped (missing from splits CSV): {skipped_unknown_game}", file=sys.stderr)
    return data_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync data/frames + YOLO labels into datasets/detection/.")
    parser.add_argument("--frames-root", type=Path, default=Path("data/frames"))
    parser.add_argument(
        "--labels-root",
        type=Path,
        default=Path("data/labels"),
        help="Parallel to frames: labels-root/{game_id}/{play_id}/{frame_stem}.txt",
    )
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path("data/plays/splits.csv"),
        help="CSV with game_id,split (train|val|test)",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/detection"),
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Symlink instead of copy (saves disk; paths must stay valid).",
    )
    args = parser.parse_args()

    path = sync_yolo_dataset(
        args.frames_root,
        args.labels_root,
        args.splits,
        args.dataset_root,
        symlink=args.symlink,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
