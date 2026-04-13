from __future__ import annotations

import shutil
from typing import Iterable, Tuple

import cv2
import numpy as np


def require_ffmpeg() -> None:
    """Fail early if ffmpeg is unavailable, since export depends on it."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not installed or not on PATH.")


def open_video(video_path: str) -> cv2.VideoCapture:
    """Open a video file through OpenCV and validate success."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    return cap


def read_first_frame(video_path: str) -> np.ndarray:
    """Load the first frame for ROI selection."""
    cap = open_video(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError("Could not read first frame from video.")
    return frame


def iter_sampled_frames(video_path: str, sample_fps: float) -> Iterable[Tuple[int, float, np.ndarray, float]]:
    """Yield frames at a reduced rate for faster analysis."""
    cap = open_video(video_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if not native_fps or native_fps <= 0:
        native_fps = 30.0

    target_interval_frames = max(1, int(round(native_fps / sample_fps)))
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame_idx % target_interval_frames == 0:
                timestamp = frame_idx / native_fps
                yield frame_idx, timestamp, frame, native_fps
            frame_idx += 1
    finally:
        cap.release()
