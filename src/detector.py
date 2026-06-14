"""YOLOv8 target detection and stable tank-only tracking."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

from ultralytics import YOLO

logger = logging.getLogger(__name__)

CUSTOM_MODEL_PATH: str = "models/best.pt"
FALLBACK_MODEL_PATH: str = "yolov8n.pt"

# Inference — low floor so geometry filter can reject bad boxes
INFERENCE_CONF: float = 0.20

# Tank lock — must pass this after geometry validation
TANK_LOCK_CONF: float = 0.33

# Bbox geometry (percent of frame) — rejects fullscreen false positives
MIN_TANK_AREA_PCT: float = 1.5
MAX_TANK_AREA_PCT: float = 48.0
MIN_TANK_ASPECT: float = 0.8
MAX_TANK_ASPECT: float = 4.0
FULLSCREEN_WIDTH_PCT: float = 88.0
FULLSCREEN_HEIGHT_PCT: float = 65.0

# Lock persistence
STATUS_SEARCHING: str = "SEARCHING"
STATUS_LOCK_ON: str = "LOCK-ON"
STATUS_LOST_TRACKING: str = "LOST_TRACKING"
LOST_TARGET_TIMEOUT: int = 12
LOCK_IOU_THRESHOLD: float = 0.15
LOCK_VICINITY_MIN_PX: int = 80
LOCK_VICINITY_DIAGONAL_RATIO: float = 0.6
EMA_ALPHA: float = 0.3


class TargetDetector:
    """Detects and tracks tanks only with false-positive rejection."""

    def __init__(self, model_path: str | None = None) -> None:
        """Initialize YOLOv8 and resolve tank class IDs from model weights."""
        resolved_path = self._resolve_model_path(model_path)
        self.model_path = resolved_path
        self.model = YOLO(resolved_path)
        self.classes = {int(k): str(v) for k, v in self.model.names.items()}
        self.tank_class_ids = self._resolve_tank_class_ids()
        self._frame_size: tuple[int, int] = (0, 0)

        if not self.tank_class_ids:
            raise RuntimeError(
                f"No 'tank' class found in model. Classes: {self.classes}"
            )

        logger.info("Tank-only mode — classes: %s", self.classes)
        logger.info("Tank class IDs: %s", self.tank_class_ids)

        self._tracking_status: str = STATUS_SEARCHING
        self._locked_target: dict[str, Any] | None = None
        self._smoothed_center: tuple[float, float] | None = None
        self._lost_frame_count: int = 0

    def _resolve_model_path(self, model_path: str | None) -> str:
        """Resolve model weights path with fallback."""
        if model_path is not None:
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model not found: {model_path}")
            return model_path

        custom = Path(CUSTOM_MODEL_PATH)
        if custom.exists():
            logger.info("Loading custom model: %s", custom)
            return str(custom)

        logger.warning(
            "Custom model '%s' not found — falling back to '%s'",
            CUSTOM_MODEL_PATH,
            FALLBACK_MODEL_PATH,
        )
        return FALLBACK_MODEL_PATH

    def _resolve_tank_class_ids(self) -> list[int]:
        """Return class IDs whose label contains 'tank'."""
        return [
            class_id
            for class_id, name in self.classes.items()
            if "tank" in name.lower()
        ]

    def _update_frame_size(self, frame: Any) -> None:
        """Cache frame dimensions for bbox validation."""
        self._frame_size = (frame.shape[1], frame.shape[0])

    def _validate_tank_bbox(
        self,
        bbox_xyxy: tuple[int, int, int, int],
        confidence: float,
    ) -> tuple[bool, str]:
        """
        Reject fullscreen hallucinations and tiny noise boxes.

        Real tanks occupy a moderate portion of the frame. Training-domain
        false positives on YouTube footage often cover 80–95% of the image.
        """
        if confidence < TANK_LOCK_CONF:
            return False, "low_confidence"

        frame_w, frame_h = self._frame_size
        if frame_w == 0 or frame_h == 0:
            return False, "no_frame_size"

        x1, y1, x2, y2 = bbox_xyxy
        box_w = max(0, x2 - x1)
        box_h = max(0, y2 - y1)
        if box_w == 0 or box_h == 0:
            return False, "zero_size"

        frame_area = frame_w * frame_h
        area_pct = (box_w * box_h) / frame_area * 100.0
        width_pct = box_w / frame_w * 100.0
        height_pct = box_h / frame_h * 100.0
        aspect = box_w / box_h

        if area_pct < MIN_TANK_AREA_PCT:
            return False, "too_small"
        if area_pct > MAX_TANK_AREA_PCT:
            return False, "too_large"
        if aspect < MIN_TANK_ASPECT or aspect > MAX_TANK_ASPECT:
            return False, "bad_aspect"
        if (
            width_pct > FULLSCREEN_WIDTH_PCT
            and height_pct > FULLSCREEN_HEIGHT_PCT
        ):
            return False, "fullscreen_false_positive"

        return True, "ok"

    def detect(self, frame: Any) -> list[dict[str, Any]]:
        """
        Run inference and return validated tank detections only.

        Soldiers, humans, and fullscreen false positives are discarded.
        """
        if frame is None:
            return []

        self._update_frame_size(frame)

        results = self.model(
            frame,
            verbose=False,
            conf=INFERENCE_CONF,
            classes=self.tank_class_ids,
        )
        detections: list[dict[str, Any]] = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1_i, y1_i, x2_i, y2_i = int(x1), int(y1), int(x2), int(y2)
                bbox_xyxy = (x1_i, y1_i, x2_i, y2_i)

                valid, reason = self._validate_tank_bbox(bbox_xyxy, confidence)
                if not valid:
                    continue

                width = x2_i - x1_i
                height = y2_i - y1_i
                center_x = x1_i + width // 2
                center_y = y1_i + height // 2

                detections.append(
                    {
                        "class_id": class_id,
                        "class_name": self.classes[class_id],
                        "confidence": confidence,
                        "bbox_xyxy": bbox_xyxy,
                        "bbox_xywh": (x1_i, y1_i, width, height),
                        "center": (center_x, center_y),
                    }
                )

        return detections

    def track(
        self,
        detections: list[dict[str, Any]],
        frame_center: tuple[int, int],
    ) -> tuple[dict[str, Any] | None, str]:
        """Update tank-only lock with persistence and smoothing."""
        if self._locked_target is not None:
            matched = self._find_locked_match(detections)

            if matched is not None:
                self._lost_frame_count = 0
                self._tracking_status = STATUS_LOCK_ON
                self._locked_target = self._apply_smoothing(
                    self._build_target_dict(matched)
                )
                return self._locked_target, self._tracking_status

            self._lost_frame_count += 1
            if self._lost_frame_count <= LOST_TARGET_TIMEOUT:
                self._tracking_status = STATUS_LOST_TRACKING
                return self._locked_target, self._tracking_status

            self._reset_lock()

        new_target = self._acquire_new_target(detections, frame_center)
        if new_target is None:
            self._tracking_status = STATUS_SEARCHING
            return None, self._tracking_status

        self._tracking_status = STATUS_LOCK_ON
        self._lost_frame_count = 0
        self._locked_target = self._apply_smoothing(new_target)
        return self._locked_target, self._tracking_status

    def _acquire_new_target(
        self,
        detections: list[dict[str, Any]],
        frame_center: tuple[int, int],
    ) -> dict[str, Any] | None:
        """Pick the validated tank nearest to screen center."""
        if not detections:
            return None

        best = min(
            detections,
            key=lambda det: math.hypot(
                det["center"][0] - frame_center[0],
                det["center"][1] - frame_center[1],
            ),
        )
        return self._build_target_dict(best)

    def _find_locked_match(
        self,
        detections: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Match current frame to the locked tank by IoU / proximity."""
        if self._locked_target is None:
            return None

        locked_bbox = self._locked_target["bbox_xyxy"]
        locked_center = self._locked_target["center"]
        vicinity_limit = self._vicinity_threshold(locked_bbox)

        best_match: dict[str, Any] | None = None
        best_score = -1.0

        for det in detections:
            iou = self._bbox_iou(locked_bbox, det["bbox_xyxy"])
            center_dist = math.hypot(
                det["center"][0] - locked_center[0],
                det["center"][1] - locked_center[1],
            )

            if iou < LOCK_IOU_THRESHOLD and center_dist > vicinity_limit:
                continue

            score = iou + 1.0 / (1.0 + center_dist)
            if score > best_score:
                best_score = score
                best_match = det

        return best_match

    def _apply_smoothing(self, target: dict[str, Any]) -> dict[str, Any]:
        """Apply EMA smoothing to target center."""
        raw_cx, raw_cy = target["center"]

        if self._smoothed_center is None:
            smooth_cx, smooth_cy = float(raw_cx), float(raw_cy)
        else:
            prev_cx, prev_cy = self._smoothed_center
            smooth_cx = EMA_ALPHA * raw_cx + (1.0 - EMA_ALPHA) * prev_cx
            smooth_cy = EMA_ALPHA * raw_cy + (1.0 - EMA_ALPHA) * prev_cy

        self._smoothed_center = (smooth_cx, smooth_cy)
        smoothed_center = (int(round(smooth_cx)), int(round(smooth_cy)))

        smoothed = dict(target)
        smoothed["center"] = smoothed_center
        smoothed["raw_center"] = (raw_cx, raw_cy)
        smoothed["tracking_status"] = self._tracking_status
        return smoothed

    def _build_target_dict(self, detection: dict[str, Any]) -> dict[str, Any]:
        """Convert detection dict to lock-on target dict."""
        x, y, w, h = detection["bbox_xywh"]
        return {
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "class_name": detection["class_name"],
            "center": detection["center"],
            "confidence": detection["confidence"],
            "bbox_xyxy": detection["bbox_xyxy"],
            "tracking_status": self._tracking_status,
        }

    def _reset_lock(self) -> None:
        """Clear lock state."""
        self._tracking_status = STATUS_SEARCHING
        self._locked_target = None
        self._smoothed_center = None
        self._lost_frame_count = 0

    @staticmethod
    def _bbox_iou(
        box_a: tuple[int, int, int, int],
        box_b: tuple[int, int, int, int],
    ) -> float:
        """Compute IoU between two xyxy boxes."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union_area = area_a + area_b - inter_area

        if union_area <= 0:
            return 0.0
        return inter_area / union_area

    @staticmethod
    def _vicinity_threshold(bbox: tuple[int, int, int, int]) -> float:
        """Max center distance for lock re-acquisition."""
        x1, y1, x2, y2 = bbox
        diagonal = math.hypot(x2 - x1, y2 - y1)
        return max(LOCK_VICINITY_MIN_PX, diagonal * LOCK_VICINITY_DIAGONAL_RATIO)

    @staticmethod
    def compute_correction_vector(
        target_center: tuple[int, int],
        frame_center: tuple[int, int],
    ) -> tuple[float, float]:
        """Compute pixel offset from frame center to target."""
        d_x = target_center[0] - frame_center[0]
        d_y = target_center[1] - frame_center[1]
        return (d_x, d_y)

    def reset_tracking(self) -> None:
        """Public reset."""
        self._reset_lock()
