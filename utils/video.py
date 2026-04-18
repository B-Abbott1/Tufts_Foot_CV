"""Video I/O helpers: metadata and frame iteration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoProperties:
    """Basic metadata from an OpenCV VideoCapture."""

    path: Path
    fps: float
    width: int
    height: int
    frame_count: int


def open_capture(path: Path | str) -> cv2.VideoCapture:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Video not found: {p}")
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {p}")
    return cap


def get_video_properties(path: Path | str) -> VideoProperties:
    cap = open_capture(path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    p = Path(path)
    return VideoProperties(path=p, fps=fps, width=width, height=height, frame_count=n)


def iter_frames(
    path: Path | str,
    *,
    max_frames: Optional[int] = None,
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Yield (frame_index, BGR frame) for every frame in order.
    frame_index is the 0-based index in the file (CAP_PROP_POS_FRAMES).
    """
    cap = open_capture(path)
    try:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield idx, frame
            idx += 1
            if max_frames is not None and idx >= max_frames:
                break
    finally:
        cap.release()


def frame_step_for_target_fps(video_fps: float, target_fps: float) -> int:
    """How many source frames to skip between saves to approximate target_fps."""
    if video_fps <= 0 or target_fps <= 0:
        return 1
    step = max(1, int(round(video_fps / target_fps)))
    return step
