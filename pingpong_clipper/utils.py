from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple


def clamp_rect(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> Tuple[int, int, int, int]:
    """Clamp rectangle coordinates so masks never extend outside the frame."""
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return x1, y1, x2, y2


def parse_roi(text: str) -> Tuple[int, int, int, int]:
    """Parse ROI argument supplied as x,y,w,h."""
    pieces = [p.strip() for p in text.split(",")]
    if len(pieces) != 4:
        raise argparse.ArgumentTypeError("ROI must be x,y,w,h")
    try:
        x, y, w, h = map(int, pieces)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI values must be integers.") from exc
    if w <= 0 or h <= 0:
        raise argparse.ArgumentTypeError("ROI width and height must be > 0.")
    return x, y, w, h


def make_output_dir(video_path: str, explicit: str | None) -> Path:
    """Choose the export directory and create it if needed."""
    if explicit:
        output_dir = Path(explicit)
    else:
        stem = Path(video_path).stem
        output_dir = Path.cwd() / f"{stem}_clips"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm for console output."""
    total_ms = int(round(seconds * 1000))
    s, ms = divmod(total_ms, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
