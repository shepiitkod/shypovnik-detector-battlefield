"""Military-style on-screen display (OSD) rendering for FPV targeting."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

# Status labels
STATUS_SEARCHING: str = "SEARCHING"
STATUS_LOCK_ON: str = "LOCK-ON"
STATUS_LOST_TRACKING: str = "LOST_TRACKING"

# Crosshair / reticle
CROSSHAIR_COLOR: tuple[int, int, int] = (0, 255, 0)
CROSSHAIR_THICKNESS: int = 1
CROSSHAIR_ARM_LENGTH: int = 30
CROSSHAIR_GAP: int = 8
CROSSHAIR_DOT_RADIUS: int = 3

# Detection boxes
SEARCH_COLOR: tuple[int, int, int] = (255, 100, 0)  # Blue (BGR)
SEARCH_THICKNESS: int = 2

# Locked target
LOCK_COLOR: tuple[int, int, int] = (0, 0, 255)  # Red (BGR)
LOCK_THICKNESS: int = 2
VECTOR_LINE_THICKNESS: int = 1
CORNER_BRACKET_LENGTH: int = 15

# Telemetry panel
PANEL_WIDTH: int = 320
PANEL_HEIGHT: int = 95
PANEL_ALPHA: float = 0.6
PANEL_BG_COLOR: tuple[int, int, int] = (20, 20, 20)
TEXT_COLOR: tuple[int, int, int] = (255, 255, 255)
STATUS_COLOR_SEARCHING: tuple[int, int, int] = (0, 255, 0)
STATUS_COLOR_LOCK_ON: tuple[int, int, int] = (0, 0, 255)
STATUS_COLOR_LOST: tuple[int, int, int] = (0, 165, 255)
LOST_LOCK_COLOR: tuple[int, int, int] = (0, 140, 255)
FONT: int = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE: float = 0.55
LINE_HEIGHT: int = 22
PANEL_PADDING: int = 10
LABEL_FONT_SCALE: float = 0.5
LABEL_THICKNESS: int = 1
LABEL_OFFSET_Y: int = 6


def draw_crosshair(frame: np.ndarray, center: tuple[int, int]) -> None:
    """
    Draw a military-style reticle crosshair at the frame center.

    Args:
        frame: BGR image to draw on.
        center: (cx, cy) crosshair position.
    """
    cx, cy = center

    cv2.line(
        frame,
        (cx - CROSSHAIR_ARM_LENGTH, cy),
        (cx - CROSSHAIR_GAP, cy),
        CROSSHAIR_COLOR,
        CROSSHAIR_THICKNESS,
    )
    cv2.line(
        frame,
        (cx + CROSSHAIR_GAP, cy),
        (cx + CROSSHAIR_ARM_LENGTH, cy),
        CROSSHAIR_COLOR,
        CROSSHAIR_THICKNESS,
    )
    cv2.line(
        frame,
        (cx, cy - CROSSHAIR_ARM_LENGTH),
        (cx, cy - CROSSHAIR_GAP),
        CROSSHAIR_COLOR,
        CROSSHAIR_THICKNESS,
    )
    cv2.line(
        frame,
        (cx, cy + CROSSHAIR_GAP),
        (cx, cy + CROSSHAIR_ARM_LENGTH),
        CROSSHAIR_COLOR,
        CROSSHAIR_THICKNESS,
    )
    cv2.circle(
        frame,
        (cx, cy),
        CROSSHAIR_DOT_RADIUS,
        CROSSHAIR_COLOR,
        CROSSHAIR_THICKNESS,
    )


def _draw_corner_brackets(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
) -> None:
    """Draw L-shaped corner brackets on a bounding box."""
    length = CORNER_BRACKET_LENGTH

    cv2.line(frame, (x1, y1), (x1 + length, y1), color, LOCK_THICKNESS)
    cv2.line(frame, (x1, y1), (x1, y1 + length), color, LOCK_THICKNESS)
    cv2.line(frame, (x2, y1), (x2 - length, y1), color, LOCK_THICKNESS)
    cv2.line(frame, (x2, y1), (x2, y1 + length), color, LOCK_THICKNESS)
    cv2.line(frame, (x1, y2), (x1 + length, y2), color, LOCK_THICKNESS)
    cv2.line(frame, (x1, y2), (x1, y2 - length), color, LOCK_THICKNESS)
    cv2.line(frame, (x2, y2), (x2 - length, y2), color, LOCK_THICKNESS)
    cv2.line(frame, (x2, y2), (x2, y2 - length), color, LOCK_THICKNESS)


def draw_detections(
    frame: np.ndarray,
    detections: list[dict[str, Any]],
    locked_target: dict[str, Any] | None = None,
) -> None:
    """
    Draw blue bounding boxes for all detections except the locked target.

    Args:
        frame: BGR image to draw on.
        detections: Full detection list from TargetDetector.
        locked_target: Currently locked target to skip (drawn separately).
    """
    for det in detections:
        if locked_target and _is_same_target(det, locked_target):
            continue

        x1, y1, x2, y2 = det["bbox_xyxy"]
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            SEARCH_COLOR,
            SEARCH_THICKNESS,
        )
        label = f"{det['class_name']} {det['confidence']:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, y1 - LABEL_OFFSET_Y),
            FONT,
            LABEL_FONT_SCALE,
            SEARCH_COLOR,
            LABEL_THICKNESS,
            cv2.LINE_AA,
        )


def _is_same_target(det: dict[str, Any], locked_target: dict[str, Any]) -> bool:
    """Return True when a detection corresponds to the locked target."""
    if det.get("class_name") != locked_target.get("class_name"):
        return False

    det_bbox = det["bbox_xyxy"]
    lock_bbox = locked_target["bbox_xyxy"]
    if det_bbox == lock_bbox:
        return True

    det_cx, det_cy = det["center"]
    lock_cx, lock_cy = locked_target["center"]
    return abs(det_cx - lock_cx) <= 5 and abs(det_cy - lock_cy) <= 5


def draw_locked_target(
    frame: np.ndarray,
    target: dict[str, Any],
    frame_center: tuple[int, int],
    status: str = STATUS_LOCK_ON,
) -> None:
    """
    Draw a solid red lock-on box, corner brackets, and tracking vector.

    Args:
        frame: BGR image to draw on.
        target: Locked target dict from track().
        frame_center: (cx, cy) screen center.
        status: Current tracking status for color selection.
    """
    x1, y1, x2, y2 = target["bbox_xyxy"]
    target_center = target["center"]
    color = LOST_LOCK_COLOR if status == STATUS_LOST_TRACKING else LOCK_COLOR

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        LOCK_THICKNESS,
    )
    cv2.line(
        frame,
        frame_center,
        target_center,
        color,
        VECTOR_LINE_THICKNESS,
    )
    _draw_corner_brackets(frame, x1, y1, x2, y2, color)


def draw_telemetry(
    frame: np.ndarray,
    status: str,
    class_name: str | None,
    d_x: float,
    d_y: float,
    fps: float,
) -> None:
    """
    Render a semi-transparent telemetry block in the top-left corner.

    Args:
        frame: BGR image to draw on.
        status: SEARCHING or LOCK-ON.
        class_name: Locked target class name, or None when searching.
        d_x: Horizontal deviation in pixels.
        d_y: Vertical deviation in pixels.
        fps: Current frames per second.
    """
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (0, 0),
        (PANEL_WIDTH, PANEL_HEIGHT),
        PANEL_BG_COLOR,
        -1,
    )
    cv2.addWeighted(overlay, PANEL_ALPHA, frame, 1.0 - PANEL_ALPHA, 0, frame)

    if status == STATUS_LOCK_ON and class_name:
        status_text = f"STATUS: {STATUS_LOCK_ON} [{class_name}]"
        status_color = STATUS_COLOR_LOCK_ON
    elif status == STATUS_LOST_TRACKING and class_name:
        status_text = f"STATUS: {STATUS_LOST_TRACKING} [{class_name}]"
        status_color = STATUS_COLOR_LOST
    else:
        status_text = f"STATUS: {STATUS_SEARCHING}"
        status_color = STATUS_COLOR_SEARCHING

    lines = [
        (status_text, status_color),
        (f"dX: {d_x:.0f}px  dY: {d_y:.0f}px", TEXT_COLOR),
        (f"FPS: {fps:.1f}", TEXT_COLOR),
    ]

    for i, (text, color) in enumerate(lines):
        y = PANEL_PADDING + (i + 1) * LINE_HEIGHT
        cv2.putText(
            frame,
            text,
            (PANEL_PADDING, y),
            FONT,
            FONT_SCALE,
            color,
            LABEL_THICKNESS,
            cv2.LINE_AA,
        )


def render_osd(
    frame: np.ndarray,
    detections: list[dict[str, Any]],
    locked_target: dict[str, Any] | None,
    frame_center: tuple[int, int],
    d_x: float,
    d_y: float,
    fps: float,
    status: str = STATUS_SEARCHING,
) -> np.ndarray:
    """
    Compose the full military OSD overlay onto a frame.

    Args:
        frame: BGR image to annotate.
        detections: All current detections.
        locked_target: Selected lock-on target, or None.
        frame_center: (cx, cy) screen center.
        d_x: Horizontal error vector.
        d_y: Vertical error vector.
        fps: Current FPS.
        status: Tracking status (SEARCHING, LOCK-ON, LOST_TRACKING).

    Returns:
        Annotated frame.
    """
    draw_detections(frame, detections, locked_target)

    if locked_target is not None:
        draw_locked_target(frame, locked_target, frame_center, status=status)
        class_name = locked_target["class_name"]
    else:
        class_name = None

    draw_crosshair(frame, frame_center)
    draw_telemetry(frame, status, class_name, d_x, d_y, fps)
    return frame
