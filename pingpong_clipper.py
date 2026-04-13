#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tkinter as tk
from dataclasses import asdict, dataclass
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class ClipSegment:
    """Represents one exported point clip."""

    index: int
    start_sec: float
    end_sec: float
    duration_sec: float
    source_start_sec: float
    source_end_sec: float
    path: str = ""


@dataclass
class ScoreSample:
    """Debug information for one analyzed timestamp."""

    timestamp_sec: float
    raw_score: float
    smoothed_score: float
    activity_score: float
    table_score: float
    table_blob_ratio: float
    suppressed_large_table_motion: bool


@dataclass
class AnalysisConfig:
    """Tuning values for motion analysis and clip generation."""

    sample_fps: float = 15.0
    blur_kernel: int = 7
    diff_threshold: int = 18
    smoothing_sec: float = 0.6
    min_active_sec: float = 1.0
    min_clip_sec: float = 1.2
    merge_gap_sec: float = 1.2
    pre_roll_sec: float = 2.0
    post_roll_sec: float = 2.0

    # Region expansion around the user-selected table rectangle.
    expand_x: float = 0.2
    expand_up: float = 0.4
    expand_down: float = 0.2
    upper_band_ratio: float = 0.45

    auto_threshold_percentile: float = 85.0
    threshold_floor: float = 0.0035
    hysteresis_drop_ratio: float = 0.72
    min_zero_run_sec: float = 0.5

    # New rule requested by the user:
    # if a single moving blob on the table is too large, it is probably not the ball.
    # Example failure case: a person walking in front of the camera or a large shadow.
    max_table_blob_ratio: float = 0.10

    # Print processing progress every N sampled frames.
    progress_every_samples: int = 75

    # Playback frame delay in milliseconds for the clip browser.
    playback_delay_ms: int = 20


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


def clamp_rect(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> Tuple[int, int, int, int]:
    """Clamp rectangle coordinates so masks never extend outside the frame."""
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return x1, y1, x2, y2


class EditableROISelector:
    """Interactive rectangle editor.

    This replaces cv2.selectROI because the user wanted the rectangle to feel more
    like an editable window instead of forcing a full reset every time.

    Controls:
    - drag inside rectangle: move it
    - drag one of the corner handles: resize it
    - press R: reset to a centered default box
    - press ENTER or SPACE: accept
    - press ESC or Q: cancel
    """

    HANDLE_RADIUS = 10

    def __init__(self, frame: np.ndarray, title: str = "Select table ROI") -> None:
        self.frame = frame
        self.title = title
        self.h, self.w = frame.shape[:2]

        # Start with a reasonable centered default rectangle.
        default_w = max(80, int(self.w * 0.38))
        default_h = max(60, int(self.h * 0.18))
        x = (self.w - default_w) // 2
        y = (self.h - default_h) // 2
        self.rect = [x, y, default_w, default_h]

        self.drag_mode: str | None = None
        self.drag_start = (0, 0)
        self.start_rect = tuple(self.rect)

    def _corner_points(self) -> dict[str, Tuple[int, int]]:
        x, y, w, h = self.rect
        return {
            "tl": (x, y),
            "tr": (x + w, y),
            "bl": (x, y + h),
            "br": (x + w, y + h),
        }

    def _hit_test(self, px: int, py: int) -> str | None:
        for name, (cx, cy) in self._corner_points().items():
            if (px - cx) ** 2 + (py - cy) ** 2 <= self.HANDLE_RADIUS ** 2:
                return name

        x, y, w, h = self.rect
        if x <= px <= x + w and y <= py <= y + h:
            return "move"
        return None

    def _normalize_rect(self) -> None:
        x, y, w, h = self.rect
        if w < 0:
            x = x + w
            w = -w
        if h < 0:
            y = y + h
            h = -h

        x1, y1, x2, y2 = clamp_rect(x, y, x + max(1, w), y + max(1, h), self.w, self.h)
        self.rect = [x1, y1, x2 - x1, y2 - y1]

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            hit = self._hit_test(x, y)
            if hit is not None:
                self.drag_mode = hit
                self.drag_start = (x, y)
                self.start_rect = tuple(self.rect)

        elif event == cv2.EVENT_MOUSEMOVE and self.drag_mode is not None:
            sx, sy = self.drag_start
            rx, ry, rw, rh = self.start_rect
            dx = x - sx
            dy = y - sy

            if self.drag_mode == "move":
                self.rect = [rx + dx, ry + dy, rw, rh]
            elif self.drag_mode == "tl":
                self.rect = [rx + dx, ry + dy, rw - dx, rh - dy]
            elif self.drag_mode == "tr":
                self.rect = [rx, ry + dy, rw + dx, rh - dy]
            elif self.drag_mode == "bl":
                self.rect = [rx + dx, ry, rw - dx, rh + dy]
            elif self.drag_mode == "br":
                self.rect = [rx, ry, rw + dx, rh + dy]

            self._normalize_rect()

        elif event == cv2.EVENT_LBUTTONUP:
            self.drag_mode = None

    def _draw(self) -> np.ndarray:
        display = self.frame.copy()
        x, y, w, h = self.rect

        cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
        overlay = display.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), -1)
        display = cv2.addWeighted(overlay, 0.12, display, 0.88, 0)

        for cx, cy in self._corner_points().values():
            cv2.circle(display, (cx, cy), self.HANDLE_RADIUS, (0, 255, 255), -1)
            cv2.circle(display, (cx, cy), self.HANDLE_RADIUS, (0, 0, 0), 1)

        lines = [
            "Drag corners to resize. Drag inside box to move.",
            "ENTER/SPACE = accept   R = reset   ESC/Q = cancel",
        ]
        for i, text in enumerate(lines):
            cv2.putText(
                display,
                text,
                (20, 35 + 28 * i),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            display,
            f"ROI: x={x} y={y} w={w} h={h}",
            (20, self.h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return display

    def reset(self) -> None:
        default_w = max(80, int(self.w * 0.38))
        default_h = max(60, int(self.h * 0.18))
        x = (self.w - default_w) // 2
        y = (self.h - default_h) // 2
        self.rect = [x, y, default_w, default_h]

    def run(self) -> Tuple[int, int, int, int]:
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.title, self._on_mouse)

        while True:
            cv2.imshow(self.title, self._draw())
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10, 32):  # enter / return / space
                break
            if key in (27, ord("q"), ord("Q")):
                cv2.destroyWindow(self.title)
                raise RuntimeError("ROI selection cancelled.")
            if key in (ord("r"), ord("R")):
                self.reset()

        cv2.destroyWindow(self.title)
        x, y, w, h = self.rect
        if w <= 0 or h <= 0:
            raise RuntimeError("No ROI selected.")
        return x, y, w, h


def select_roi_interactively(video_path: str) -> Tuple[int, int, int, int]:
    """Let the user edit a rectangle around the table."""
    frame = read_first_frame(video_path)
    return EditableROISelector(frame).run()


def build_masks(frame_shape: Sequence[int], roi: Tuple[int, int, int, int], cfg: AnalysisConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Build the table mask and the broader activity mask."""
    height, width = frame_shape[:2]
    x, y, w, h = roi

    table_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(table_mask, (x, y), (x + w, y + h), 255, -1)

    ex_left = int(w * cfg.expand_x)
    ex_right = int(w * cfg.expand_x)
    ex_up = int(h * cfg.expand_up)
    ex_down = int(h * cfg.expand_down)

    mx1, my1, mx2, my2 = clamp_rect(
        x - ex_left,
        y - ex_up,
        x + w + ex_right,
        y + h + ex_down,
        width,
        height,
    )
    activity_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(activity_mask, (mx1, my1), (mx2, my2), 255, -1)

    # Add a band above the table since rallies often live there too.
    band_height = int(h * cfg.upper_band_ratio)
    bx1, by1, bx2, by2 = clamp_rect(x, y - band_height, x + w, y, width, height)
    cv2.rectangle(activity_mask, (bx1, by1), (bx2, by2), 255, -1)

    return table_mask, activity_mask


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


def compute_table_blob_ratio(binary_table_motion: np.ndarray, table_area: int) -> float:
    """Return largest connected moving blob area as a fraction of table area.

    This is the user-requested suppression rule. A huge moving region on the
    table is unlikely to be the ball. It is more likely to be a person, bag,
    shadow, or some other false positive.
    """
    contours, _ = cv2.findContours(binary_table_motion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or table_area <= 0:
        return 0.0
    largest = max(cv2.contourArea(c) for c in contours)
    return float(largest / table_area)


def motion_scores(
    video_path: str,
    roi: Tuple[int, int, int, int],
    cfg: AnalysisConfig,
) -> Tuple[np.ndarray, np.ndarray, float, Tuple[np.ndarray, np.ndarray], List[ScoreSample]]:
    """Compute a motion score over time and collect debug rows."""
    scores: List[float] = []
    times: List[float] = []
    debug_rows: List[ScoreSample] = []

    prev_gray = None
    frame_shape = None
    table_mask = None
    activity_mask = None
    native_fps = 30.0

    print("Analyzing motion...")
    for sample_index, (_, timestamp, frame, native_fps) in enumerate(iter_sampled_frames(video_path, cfg.sample_fps), start=1):
        if frame_shape is None:
            frame_shape = frame.shape
            table_mask, activity_mask = build_masks(frame.shape, roi, cfg)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (cfg.blur_kernel, cfg.blur_kernel), 0)

        if prev_gray is None:
            prev_gray = gray
            scores.append(0.0)
            times.append(timestamp)
            debug_rows.append(
                ScoreSample(
                    timestamp_sec=timestamp,
                    raw_score=0.0,
                    smoothed_score=0.0,
                    activity_score=0.0,
                    table_score=0.0,
                    table_blob_ratio=0.0,
                    suppressed_large_table_motion=False,
                )
            )
            continue

        diff = cv2.absdiff(gray, prev_gray)
        _, thresh = cv2.threshold(diff, cfg.diff_threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.medianBlur(thresh, 5)

        activity_motion = cv2.bitwise_and(thresh, activity_mask)
        table_motion = cv2.bitwise_and(thresh, table_mask)

        activity_pixels = cv2.countNonZero(activity_motion)
        table_pixels = cv2.countNonZero(table_motion)

        activity_area = max(1, cv2.countNonZero(activity_mask))
        table_area = max(1, cv2.countNonZero(table_mask))

        activity_score = activity_pixels / activity_area
        table_score = table_pixels / table_area

        # Requested improvement:
        # if the largest blob on the table is too large, do not trust the table motion.
        table_blob_ratio = compute_table_blob_ratio(table_motion, table_area)
        suppressed = table_blob_ratio > cfg.max_table_blob_ratio
        if suppressed:
            table_score = 0.0

        score = 0.65 * activity_score + 0.35 * table_score

        scores.append(float(score))
        times.append(timestamp)
        debug_rows.append(
            ScoreSample(
                timestamp_sec=timestamp,
                raw_score=float(score),
                smoothed_score=0.0,
                activity_score=float(activity_score),
                table_score=float(table_score),
                table_blob_ratio=float(table_blob_ratio),
                suppressed_large_table_motion=suppressed,
            )
        )
        prev_gray = gray

        if sample_index % cfg.progress_every_samples == 0:
            print(f"  processed {sample_index} sampled frames... t={timestamp:.1f}s current_score={score:.5f}")

    if frame_shape is None or table_mask is None or activity_mask is None:
        raise RuntimeError("Could not analyze any frames from the video.")

    return np.array(times), np.array(scores), native_fps, (table_mask, activity_mask), debug_rows


def smooth_signal(scores: np.ndarray, cfg: AnalysisConfig) -> np.ndarray:
    """Smooth the raw motion signal using a simple moving average."""
    if scores.size == 0:
        return scores
    window = max(1, int(round(cfg.sample_fps * cfg.smoothing_sec)))
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(scores, kernel, mode="same")


def estimate_threshold(smoothed_scores: np.ndarray, cfg: AnalysisConfig) -> Tuple[float, float]:
    """Estimate start/stop thresholds for active rally detection."""
    if smoothed_scores.size == 0:
        return cfg.threshold_floor, cfg.threshold_floor * cfg.hysteresis_drop_ratio

    median = float(np.median(smoothed_scores))
    mad = float(np.median(np.abs(smoothed_scores - median)))
    robust = median + 4.5 * mad
    pct = float(np.percentile(smoothed_scores, cfg.auto_threshold_percentile))

    start_threshold = max(cfg.threshold_floor, min(max(robust, cfg.threshold_floor), max(pct, cfg.threshold_floor)))
    stop_threshold = max(cfg.threshold_floor * 0.7, start_threshold * cfg.hysteresis_drop_ratio)
    return start_threshold, stop_threshold


def detect_segments(
    times: np.ndarray,
    smoothed_scores: np.ndarray,
    cfg: AnalysisConfig,
) -> tuple[List[Tuple[float, float]], float, float]:
    """Convert the smoothed motion signal into active time ranges."""
    start_threshold, stop_threshold = estimate_threshold(smoothed_scores, cfg)
    active = False
    segments: List[Tuple[float, float]] = []
    start_time = 0.0
    below_run = 0
    min_below_run = max(1, int(round(cfg.sample_fps * cfg.min_zero_run_sec)))

    for i, score in enumerate(smoothed_scores):
        t = float(times[i])
        if not active:
            if score >= start_threshold:
                active = True
                start_time = t
                below_run = 0
        else:
            if score < stop_threshold:
                below_run += 1
            else:
                below_run = 0

            if below_run >= min_below_run:
                end_index = max(0, i - below_run + 1)
                end_time = float(times[end_index])
                if end_time - start_time >= cfg.min_active_sec:
                    segments.append((start_time, end_time))
                active = False
                below_run = 0

    if active:
        end_time = float(times[-1])
        if end_time - start_time >= cfg.min_active_sec:
            segments.append((start_time, end_time))

    return merge_and_pad_segments(segments, times, cfg), start_threshold, stop_threshold


def merge_and_pad_segments(
    segments: Sequence[Tuple[float, float]],
    times: np.ndarray,
    cfg: AnalysisConfig,
) -> List[Tuple[float, float]]:
    """Post-process raw active windows."""
    if not segments:
        return []

    video_end = float(times[-1]) if times.size else 0.0
    merged: List[List[float]] = []

    for start, end in segments:
        if not merged:
            merged.append([start, end])
            continue
        prev = merged[-1]
        if start - prev[1] <= cfg.merge_gap_sec:
            prev[1] = max(prev[1], end)
        else:
            merged.append([start, end])

    padded: List[Tuple[float, float]] = []
    for start, end in merged:
        start = max(0.0, start - cfg.pre_roll_sec)
        end = min(video_end, end + cfg.post_roll_sec)
        if end - start >= cfg.min_clip_sec:
            padded.append((start, end))

    if not padded:
        return []

    final_segments: List[List[float]] = []
    for start, end in padded:
        if not final_segments:
            final_segments.append([start, end])
        else:
            prev = final_segments[-1]
            if start - prev[1] <= 0.2:
                prev[1] = max(prev[1], end)
            else:
                final_segments.append([start, end])

    return [(float(s), float(e)) for s, e in final_segments]


def export_clip(video_path: str, output_path: Path, start_sec: float, end_sec: float) -> None:
    """Export one clip using ffmpeg."""
    duration = max(0.01, end_sec - start_sec)
    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-i",
        video_path,
        "-t",
        f"{duration:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-c:a",
        "aac",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for clip {output_path.name}:\n{result.stderr}")


def save_manifest(output_dir: Path, clips: Sequence[ClipSegment], roi: Tuple[int, int, int, int], cfg: AnalysisConfig) -> None:
    """Write metadata files so runs are easy to inspect and reproduce."""
    manifest_path = output_dir / "clips_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["clip_index", "start_sec", "end_sec", "duration_sec", "source_start_sec", "source_end_sec", "path"])
        for clip in clips:
            writer.writerow([
                clip.index,
                f"{clip.start_sec:.3f}",
                f"{clip.end_sec:.3f}",
                f"{clip.duration_sec:.3f}",
                f"{clip.source_start_sec:.3f}",
                f"{clip.source_end_sec:.3f}",
                clip.path,
            ])

    metadata = {
        "roi": {"x": roi[0], "y": roi[1], "w": roi[2], "h": roi[3]},
        "config": asdict(cfg),
        "clip_count": len(clips),
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def save_analysis_debug(
    output_dir: Path,
    times: np.ndarray,
    raw_scores: np.ndarray,
    smoothed_scores: np.ndarray,
    debug_rows: Sequence[ScoreSample],
    segments: Sequence[Tuple[float, float]],
    thresholds: Tuple[float, float],
) -> None:
    """Save the detector output so tuning is easier.

    This addresses the request to "output what it found while processing".
    """
    start_threshold, stop_threshold = thresholds

    with (output_dir / "analysis_scores.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp_sec",
            "raw_score",
            "smoothed_score",
            "activity_score",
            "table_score",
            "table_blob_ratio",
            "suppressed_large_table_motion",
            "start_threshold",
            "stop_threshold",
        ])
        for row, raw, smoothed in zip(debug_rows, raw_scores, smoothed_scores):
            writer.writerow([
                f"{row.timestamp_sec:.3f}",
                f"{raw:.7f}",
                f"{smoothed:.7f}",
                f"{row.activity_score:.7f}",
                f"{row.table_score:.7f}",
                f"{row.table_blob_ratio:.7f}",
                int(row.suppressed_large_table_motion),
                f"{start_threshold:.7f}",
                f"{stop_threshold:.7f}",
            ])

    with (output_dir / "detected_segments.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "start_threshold": start_threshold,
                "stop_threshold": stop_threshold,
                "segments": [
                    {"start_sec": round(start, 3), "end_sec": round(end, 3), "duration_sec": round(end - start, 3)}
                    for start, end in segments
                ],
            },
            f,
            indent=2,
        )


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


def play_video_file(path: str, delay_ms: int = 20) -> None:
    """Play a clip in an OpenCV window.

    This keeps the "view it in app" workflow simple without bringing in a heavy
    embedded video framework.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open clip: {path}")

    title = f"Clip Player - {Path(path).name}"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)

    paused = False
    try:
        while True:
            if not paused:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                cv2.imshow(title, frame)

            key = cv2.waitKey(delay_ms if not paused else 30) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (32, ord("p"), ord("P")):
                paused = not paused
            if key in (ord("r"), ord("R")):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    finally:
        cap.release()
        cv2.destroyWindow(title)


def launch_clip_browser(clips: Sequence[ClipSegment], output_dir: Path, cfg: AnalysisConfig) -> None:
    """Show a simple clip list UI after export.

    The user asked for available clips to be viewable in the app after clipping.
    This browser lists the clips and plays the selected one.
    """
    if not clips:
        return

    root = tk.Tk()
    root.title("Ping Pong Clipper - Clips")
    root.geometry("700x430")

    ttk.Label(root, text="Detected clips", font=("Segoe UI", 14, "bold")).pack(anchor="w", padx=12, pady=(12, 4))
    ttk.Label(
        root,
        text="Double-click a clip or press Play. Controls in player: Space/P pause, R restart, Q/Esc close.",
    ).pack(anchor="w", padx=12, pady=(0, 8))

    frame = ttk.Frame(root)
    frame.pack(fill="both", expand=True, padx=12, pady=8)

    columns = ("clip", "start", "end", "duration")
    tree = ttk.Treeview(frame, columns=columns, show="headings", height=14)
    for col, width in zip(columns, (160, 140, 140, 120)):
        tree.heading(col, text=col.title())
        tree.column(col, width=width, anchor="w")
    tree.pack(side="left", fill="both", expand=True)

    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    scrollbar.pack(side="right", fill="y")
    tree.configure(yscrollcommand=scrollbar.set)

    clip_by_iid: dict[str, ClipSegment] = {}
    for clip in clips:
        iid = str(clip.index)
        clip_by_iid[iid] = clip
        tree.insert(
            "",
            "end",
            iid=iid,
            values=(
                Path(clip.path).name,
                format_time(clip.start_sec),
                format_time(clip.end_sec),
                f"{clip.duration_sec:.2f}s",
            ),
        )

    selected_label = ttk.Label(root, text="")
    selected_label.pack(anchor="w", padx=12)

    def get_selected_clip() -> ClipSegment | None:
        selected = tree.selection()
        if not selected:
            messagebox.showinfo("No clip selected", "Select a clip first.")
            return None
        return clip_by_iid[selected[0]]

    def on_select(event: object | None = None) -> None:
        clip = get_selected_clip() if tree.selection() else None
        if clip:
            selected_label.config(text=f"Selected: {Path(clip.path).name}")
        else:
            selected_label.config(text="")

    def on_play() -> None:
        clip = get_selected_clip()
        if not clip:
            return
        try:
            play_video_file(clip.path, delay_ms=cfg.playback_delay_ms)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Playback error", str(exc))

    def on_open_folder() -> None:
        if sys.platform.startswith("win"):
            os.startfile(str(output_dir))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(output_dir)])
        else:
            subprocess.Popen(["xdg-open", str(output_dir)])

    tree.bind("<<TreeviewSelect>>", on_select)
    tree.bind("<Double-1>", lambda event: on_play())

    buttons = ttk.Frame(root)
    buttons.pack(fill="x", padx=12, pady=12)
    ttk.Button(buttons, text="Play selected clip", command=on_play).pack(side="left")
    ttk.Button(buttons, text="Open clips folder", command=on_open_folder).pack(side="left", padx=8)
    ttk.Button(buttons, text="Close", command=root.destroy).pack(side="right")

    first = tree.get_children()
    if first:
        tree.selection_set(first[0])
        on_select()

    root.mainloop()


def run(video_path: str, output_dir: Path, roi: Tuple[int, int, int, int], cfg: AnalysisConfig) -> List[ClipSegment]:
    """Run the full pipeline: analyze -> detect -> export -> save metadata."""
    require_ffmpeg()

    times, scores, _, _, debug_rows = motion_scores(video_path, roi, cfg)
    smoothed = smooth_signal(scores, cfg)
    for row, smoothed_score in zip(debug_rows, smoothed):
        row.smoothed_score = float(smoothed_score)

    segments, start_threshold, stop_threshold = detect_segments(times, smoothed, cfg)

    print(f"Detector thresholds: start={start_threshold:.6f} stop={stop_threshold:.6f}")
    if segments:
        print("Found segments before export:")
        for idx, (start, end) in enumerate(segments, start=1):
            print(f"  #{idx:03d}  {format_time(start)} -> {format_time(end)}  ({end - start:.2f}s)")
    else:
        print("No active segments found.")

    save_analysis_debug(output_dir, times, scores, smoothed, debug_rows, segments, (start_threshold, stop_threshold))

    clips: List[ClipSegment] = []
    for idx, (start, end) in enumerate(segments, start=1):
        filename = output_dir / f"point_{idx:03d}.mp4"
        print(f"Exporting {filename.name} ...")
        export_clip(video_path, filename, start, end)
        clips.append(
            ClipSegment(
                index=idx,
                start_sec=start,
                end_sec=end,
                duration_sec=end - start,
                source_start_sec=start,
                source_end_sec=end,
                path=str(filename),
            )
        )

    save_manifest(output_dir, clips, roi, cfg)
    return clips


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI arguments for the tool."""
    parser = argparse.ArgumentParser(
        description="Clip ping pong match videos into separate point clips by detecting rally motion near the table."
    )
    parser.add_argument("video", help="Path to the input video file.")
    parser.add_argument("--output-dir", help="Directory for exported clips. Defaults to <video_name>_clips.")
    parser.add_argument("--roi", type=parse_roi, help="Table ROI as x,y,w,h. If omitted, an interactive selector opens.")
    parser.add_argument("--sample-fps", type=float, default=15.0, help="Frames per second to sample during analysis.")
    parser.add_argument("--pre-roll", type=float, default=2.0, help="Seconds to include before detected ball-in-play.")
    parser.add_argument("--post-roll", type=float, default=2.0, help="Seconds to include after detected point end.")
    parser.add_argument("--merge-gap", type=float, default=1.2, help="Merge nearby active windows separated by less than this many seconds.")
    parser.add_argument("--min-clip", type=float, default=1.2, help="Discard clips shorter than this many seconds.")
    parser.add_argument("--threshold-percentile", type=float, default=85.0, help="Higher values make detection stricter. Default: 85.")
    parser.add_argument("--threshold-floor", type=float, default=0.0035, help="Absolute floor for motion threshold.")
    parser.add_argument(
        "--max-table-blob-ratio",
        type=float,
        default=0.10,
        help="Ignore table motion when the largest moving blob covers more than this fraction of the table.",
    )
    parser.add_argument("--no-browser", action="store_true", help="Do not open the clip browser after export.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    video_path = os.path.abspath(args.video)
    if not os.path.exists(video_path):
        parser.error(f"Video does not exist: {video_path}")

    cfg = AnalysisConfig(
        sample_fps=args.sample_fps,
        pre_roll_sec=args.pre_roll,
        post_roll_sec=args.post_roll,
        merge_gap_sec=args.merge_gap,
        min_clip_sec=args.min_clip,
        auto_threshold_percentile=args.threshold_percentile,
        threshold_floor=args.threshold_floor,
        max_table_blob_ratio=args.max_table_blob_ratio,
    )

    output_dir = make_output_dir(video_path, args.output_dir)
    roi = args.roi if args.roi else select_roi_interactively(video_path)

    print(f"Using ROI: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
    print(f"Writing clips to: {output_dir}")

    clips = run(video_path, output_dir, roi, cfg)
    if not clips:
        print("No clips were detected. Try lowering --threshold-floor or --threshold-percentile.")
        return 2

    print(f"Detected {len(clips)} clips:")
    for clip in clips:
        print(
            f"  point_{clip.index:03d}.mp4  {format_time(clip.start_sec)} -> {format_time(clip.end_sec)}  "
            f"({clip.duration_sec:.2f}s)"
        )

    print("Saved debug files:")
    print(f"  {output_dir / 'analysis_scores.csv'}")
    print(f"  {output_dir / 'detected_segments.json'}")
    print(f"  {output_dir / 'clips_manifest.csv'}")

    if not args.no_browser:
        launch_clip_browser(clips, output_dir, cfg)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
