from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .utils import clamp_rect
from .video_io import read_first_frame


class EditableROISelector:
    """Interactive rectangle editor."""

    HANDLE_RADIUS = 10

    def __init__(self, frame: np.ndarray, title: str = "Select table ROI") -> None:
        self.frame = frame
        self.title = title
        self.h, self.w = frame.shape[:2]

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
            if key in (13, 10, 32):
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
