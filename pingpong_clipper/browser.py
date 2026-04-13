from __future__ import annotations

import os
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Sequence

import cv2

from .models import AnalysisConfig, ClipSegment
from .utils import format_time


def play_video_file(path: str, delay_ms: int = 20) -> None:
    """Play a clip in an OpenCV window."""
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
    """Show a simple clip list UI after export."""
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
        except Exception as exc:
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
