from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np

from .models import AnalysisConfig, ClipSegment, ScoreSample


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
    raw_scores: np.ndarray,
    smoothed_scores: np.ndarray,
    debug_rows: Sequence[ScoreSample],
    segments: Sequence[Tuple[float, float]],
    thresholds: Tuple[float, float],
) -> None:
    """Save the detector output so tuning is easier."""
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
