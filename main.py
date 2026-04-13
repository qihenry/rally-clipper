#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

from pingpong_clipper.browser import launch_clip_browser
from pingpong_clipper.models import AnalysisConfig
from pingpong_clipper.pipeline import run_pipeline
from pingpong_clipper.roi import select_roi_interactively
from pingpong_clipper.utils import format_time, make_output_dir, parse_roi


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

    clips = run_pipeline(video_path, output_dir, roi, cfg)
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
