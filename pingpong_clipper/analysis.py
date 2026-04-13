from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np

from .models import AnalysisConfig, ScoreSample
from .utils import clamp_rect
from .video_io import iter_sampled_frames


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

    band_height = int(h * cfg.upper_band_ratio)
    bx1, by1, bx2, by2 = clamp_rect(x, y - band_height, x + w, y, width, height)
    cv2.rectangle(activity_mask, (bx1, by1), (bx2, by2), 255, -1)

    return table_mask, activity_mask


def compute_table_blob_ratio(binary_table_motion: np.ndarray, table_area: int) -> float:
    """Return largest connected moving blob area as a fraction of table area."""
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
            debug_rows.append(ScoreSample(timestamp, 0.0, 0.0, 0.0, 0.0, 0.0, False))
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
