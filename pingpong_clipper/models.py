from __future__ import annotations

from dataclasses import dataclass


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

    expand_x: float = 0.2
    expand_up: float = 0.4
    expand_down: float = 0.2
    upper_band_ratio: float = 0.45

    auto_threshold_percentile: float = 85.0
    threshold_floor: float = 0.0035
    hysteresis_drop_ratio: float = 0.72
    min_zero_run_sec: float = 0.5

    max_table_blob_ratio: float = 0.10
    progress_every_samples: int = 75
    playback_delay_ms: int = 20
