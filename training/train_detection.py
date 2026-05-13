"""Fine-tune YOLO detection; thin wrapper around Ultralytics."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import tempfile

import yaml
from ultralytics import YOLO


def _resolved_data_yaml_for_ultralytics(data_path: Path) -> Path:
    """Write a temp data.yaml with absolute dataset ``path`` for Ultralytics."""
    data_path = Path(data_path).resolve()
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    raw = yaml.safe_load(data_path.read_text(encoding="utf-8"))
    root = raw.get("path", "")
    root_p = Path(root)
    if not root_p.is_absolute():
        root_p = (data_path.parent / root_p).resolve()
        raw["path"] = str(root_p)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8") as f:
        yaml.dump(raw, f, default_flow_style=False, sort_keys=False)
        return Path(f.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train YOLO on datasets/detection (or custom data.yaml).")
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="data.yaml for Ultralytics (default: models/configs/detection_data.yaml if present, else datasets/detection/data.yaml)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolo11s.pt",
        help="Base checkpoint path or hub name",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", type=str, default=None, help="e.g. 0 or cpu")
    parser.add_argument(
        "--project",
        type=Path,
        default=Path("models/runs/detect"),
        help="Ultralytics project directory",
    )
    parser.add_argument("--name", type=str, default="train", help="Run name under project")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Allow overwriting existing run directory",
    )
    args = parser.parse_args()

    data_path = args.data
    if data_path is None:
        candidates = [
            Path("models/configs/detection_data.yaml"),
            Path("datasets/detection/data.yaml"),
        ]
        data_path = next((p for p in candidates if p.is_file()), candidates[0])
    data_path = Path(data_path)

    if not data_path.is_file():
        raise FileNotFoundError(
            f"data.yaml not found: {data_path}. Run tools/sync_frames_to_yolo_dataset.py first or pass --data."
        )

    project = Path(args.project)
    project.mkdir(parents=True, exist_ok=True)

    resolved_yaml = _resolved_data_yaml_for_ultralytics(data_path)
    try:
        model = YOLO(str(args.weights))
        model.train(
            data=str(resolved_yaml),
            epochs=int(args.epochs),
            imgsz=int(args.imgsz),
            batch=int(args.batch),
            device=args.device,
            project=str(project),
            name=args.name,
            exist_ok=bool(args.exist_ok),
        )
    finally:
        try:
            resolved_yaml.unlink(missing_ok=True)
        except OSError:
            pass

    best = project / args.name / "weights" / "best.pt"
    out_weights = Path("models/weights") / f"detection_{args.name}.pt"
    if best.is_file():
        out_weights.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, out_weights)
        print(f"Copied best weights to {out_weights.resolve()}")
    print(f"Ultralytics run directory: {(project / args.name).resolve()}")


if __name__ == "__main__":
    main()
