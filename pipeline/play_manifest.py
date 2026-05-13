"""Play manifest and game-level split CSV schema and loaders."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

MANIFEST_REQUIRED = ("game_id", "play_id", "source_video")
MANIFEST_OPTIONAL = ("notes",)
SPLITS_REQUIRED = ("game_id", "split")

SplitName = Literal["train", "val", "test"]


def load_manifest(path: Path) -> pd.DataFrame:
    """Load plays manifest CSV. Required columns: game_id, play_id, source_video."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    df = pd.read_csv(path)
    missing = set(MANIFEST_REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns {missing}: {path}")
    df = df.copy()
    df["game_id"] = df["game_id"].astype(str).str.strip()
    df["play_id"] = df["play_id"].astype(str).str.strip()
    df["source_video"] = df["source_video"].astype(str).str.strip()
    return df


def load_game_splits(path: Path) -> dict[str, SplitName]:
    """
    Load game_id -> split mapping. CSV columns: game_id, split
    split must be one of: train, val, test
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Splits file not found: {path}")
    df = pd.read_csv(path)
    missing = set(SPLITS_REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"Splits CSV missing columns {missing}: {path}")
    allowed = {"train", "val", "test"}
    out: dict[str, SplitName] = {}
    for _, row in df.iterrows():
        gid = str(row["game_id"]).strip()
        sp = str(row["split"]).strip().lower()
        if sp not in allowed:
            raise ValueError(f"Invalid split '{sp}' for game_id={gid}; use train|val|test")
        out[gid] = sp  # type: ignore[assignment]
    return out
