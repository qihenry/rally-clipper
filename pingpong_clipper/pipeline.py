from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from .analysis import detect_segments, motion_scores, smooth_signal
from .export import export_clip, save_analysis_debug, save_manifest
from .models import AnalysisConfig, ClipSegment
from .utils import format_time
from .video_io import require_ffmpeg


def run_pipeline(video_path: str, output_dir: Path, roi: Tuple[int, int, int, int], cfg: AnalysisConfig) -> List[ClipSegment]:
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

    save_analysis_debug(output_dir, scores, smoothed, debug_rows, segments, (start_threshold, stop_threshold))

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
