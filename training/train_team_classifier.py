"""Train a small team/ref image classifier (torchvision ImageFolder)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


def _build_model(backbone: str, num_classes: int) -> nn.Module:
    backbone = backbone.lower().strip()
    if backbone == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_f = m.classifier[0].in_features
        m.classifier = nn.Sequential(
            nn.Linear(in_f, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(256, num_classes),
        )
        return m
    if backbone == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, num_classes)
        return m
    raise ValueError(f"Unknown backbone {backbone!r}; use mobilenet_v3_small or resnet18")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train team/ref classifier on ImageFolder layout.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("datasets/team_ref"),
        help="Contains train/ and val/ subdirs with class folders",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone", type=str, default="mobilenet_v3_small")
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("models/weights/team_ref_run"),
        help="Directory for best.pt, label_map.json, metrics.json",
    )
    args = parser.parse_args()

    train_dir = args.data_root / "train"
    val_dir = args.data_root / "val"
    if not train_dir.is_dir():
        raise FileNotFoundError(f"Missing {train_dir}")
    if not val_dir.is_dir():
        raise FileNotFoundError(f"Missing {val_dir}")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    tf_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(args.imgsz, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.1, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tf_val = transforms.Compose(
        [
            transforms.Resize(int(args.imgsz * 1.1)),
            transforms.CenterCrop(args.imgsz),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    ds_train = datasets.ImageFolder(train_dir, transform=tf_train)
    ds_val = datasets.ImageFolder(val_dir, transform=tf_val)
    if ds_train.class_to_idx != ds_val.class_to_idx:
        raise ValueError(
            "train and val class folders differ; ensure same subdirectory names in train/ and val/."
        )

    num_classes = len(ds_train.classes)
    if num_classes < 2:
        raise ValueError("Need at least 2 classes in train/ for classification.")

    idx_to_name = {v: k for k, v in ds_train.class_to_idx.items()}

    train_loader = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = _build_model(args.backbone, num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    best_path = args.out_dir / "best.pt"

    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        n_seen = 0
        for x, y in tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs} train"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            running += float(loss.item()) * x.size(0)
            n_seen += x.size(0)
        train_loss = running / max(1, n_seen)

        model.eval()
        correct = 0
        total = 0
        val_loss_sum = 0.0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc="val"):
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss_sum += float(crit(logits, y).item()) * x.size(0)
                pred = logits.argmax(dim=1)
                correct += int((pred == y).sum().item())
                total += x.size(0)
        val_acc = correct / max(1, total)
        val_loss = val_loss_sum / max(1, total)
        print(f"epoch {epoch+1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "backbone": args.backbone,
                    "num_classes": num_classes,
                    "class_to_idx": ds_train.class_to_idx,
                    "val_acc": val_acc,
                    "epoch": epoch,
                },
                best_path,
            )

    label_map_path = args.out_dir / "label_map.json"
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump({"class_to_idx": ds_train.class_to_idx, "idx_to_class": idx_to_name}, f, indent=2)

    metrics_path = args.out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"best_val_acc": best_val_acc, "epochs": args.epochs}, f, indent=2)

    final_weights = Path("models/weights") / "team_ref_classifier.pt"
    if best_path.is_file():
        final_weights.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copy2(best_path, final_weights)
        print(f"Copied best checkpoint to {final_weights.resolve()}")

    print(f"Best val acc: {best_val_acc:.4f}")
    print(f"Artifacts: {best_path}, {label_map_path}, {metrics_path}")


if __name__ == "__main__":
    main()
