"""FPV drone target tracking and auto-lock — main entry point."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from src.detector import CUSTOM_MODEL_PATH, TargetDetector
from src.ui import render_osd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Paths
VIDEOS_DIR: Path = Path("videos")
MODELS_DIR: Path = Path("models")

# Runtime
WINDOW_NAME: str = "FPV Target Tracker"
QUIT_KEY: str = "q"
FPS_SMOOTHING_ALPHA: float = 0.1

SUPPORTED_VIDEO_EXTENSIONS: tuple[str, ...] = (
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
)


class FPVTargetingSystem:
    """Orchestrates video capture, detection, lock-on, and OSD rendering."""

    def __init__(
        self,
        video_source: str | int,
        model_path: str | None = None,
        export_path: str | None = None,
    ) -> None:
        """
        Initialize the targeting pipeline.

        Args:
            video_source: Webcam index or path to a video file.
            model_path: Optional YOLO weights override.
            export_path: If set, save annotated video instead of GUI preview.
        """
        self.video_source = video_source
        self.export_path = export_path
        self.detector = TargetDetector(model_path=model_path)
        self.capture = cv2.VideoCapture(video_source)
        self._prev_time: float = time.time()
        self._fps: float = 0.0
        self._writer: cv2.VideoWriter | None = None
        self._lock_frames: int = 0
        self._total_frames: int = 0

    def _update_fps(self) -> float:
        """Calculate smoothed FPS from frame-to-frame elapsed time."""
        now = time.time()
        elapsed = now - self._prev_time
        self._prev_time = now

        if elapsed > 0:
            instant_fps = 1.0 / elapsed
            self._fps = (
                FPS_SMOOTHING_ALPHA * instant_fps
                + (1.0 - FPS_SMOOTHING_ALPHA) * self._fps
            )
        return self._fps

    def _init_writer(self, frame: np.ndarray) -> None:
        """Create VideoWriter for export mode."""
        if not self.export_path:
            return
        height, width = frame.shape[:2]
        fps = self.capture.get(cv2.CAP_PROP_FPS) or 25.0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            self.export_path,
            fourcc,
            fps,
            (width, height),
        )
        logger.info("Exporting annotated video to: %s", self.export_path)

    def run(self) -> None:
        """Run the main detection and lock-on loop until the user quits."""
        if not self.capture.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.video_source}")

        if self.export_path:
            logger.info("Export mode — processing video (no GUI window).")
        else:
            logger.info("Press '%s' to quit.", QUIT_KEY)

        try:
            while True:
                ret, frame = self.capture.read()
                if not ret or frame is None:
                    logger.warning("Empty frame received — ending playback.")
                    break

                if self._writer is None and self.export_path:
                    self._init_writer(frame)

                height, width = frame.shape[:2]
                frame_center = (width // 2, height // 2)
                self._total_frames += 1

                detections = self.detector.detect(frame)
                locked_target, tracking_status = self.detector.track(
                    detections,
                    frame_center,
                )

                if tracking_status in ("LOCK-ON", "LOST_TRACKING"):
                    self._lock_frames += 1

                if locked_target is not None:
                    d_x, d_y = self.detector.compute_correction_vector(
                        locked_target["center"],
                        frame_center,
                    )
                else:
                    d_x, d_y = 0.0, 0.0

                fps = self._update_fps()
                frame = render_osd(
                    frame,
                    detections,
                    locked_target,
                    frame_center,
                    d_x,
                    d_y,
                    fps,
                    status=tracking_status,
                )

                if self._writer is not None:
                    self._writer.write(frame)
                else:
                    cv2.imshow(WINDOW_NAME, frame)
                    if cv2.waitKey(1) & 0xFF == ord(QUIT_KEY):
                        break
        finally:
            self.capture.release()
            if self._writer is not None:
                self._writer.release()
                logger.info(
                    "Export done: %s | frames=%d | lock_frames=%d",
                    self.export_path,
                    self._total_frames,
                    self._lock_frames,
                )
            cv2.destroyAllWindows()
            logger.info("Resources released.")


def find_video_in_folder(folder: Path) -> Path | None:
    """
    Return the first supported video file found in a directory.

    Args:
        folder: Directory to search.

    Returns:
        Path to a video file, or None if none found.
    """
    if not folder.is_dir():
        return None

    for ext in SUPPORTED_VIDEO_EXTENSIONS:
        matches = sorted(folder.glob(f"*{ext}"))
        if matches:
            return matches[0]
    return None


def resolve_video_source(source: str | None) -> str | int:
    """
    Resolve the video source from CLI arg, videos/ folder, or webcam.

    Args:
        source: Raw --source value, or None for auto-detection.

    Returns:
        Camera index (int) or video file path (str).
    """
    if source is None:
        video_file = find_video_in_folder(VIDEOS_DIR)
        if video_file is not None:
            logger.info("Using video from '%s': %s", VIDEOS_DIR, video_file)
            return str(video_file)
        logger.info("No video in '%s' — using webcam (0).", VIDEOS_DIR)
        return 0

    if source.isdigit():
        return int(source)

    video_path = Path(source)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {source}")
    return str(video_path)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="FPV drone target tracking with YOLOv8 auto-lock",
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Video file path or camera index (default: first file in videos/, else webcam)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"YOLO weights path (default: {CUSTOM_MODEL_PATH} with yolov8n.pt fallback)",
    )
    parser.add_argument(
        "--export",
        default=None,
        help="Save annotated output video to this path (e.g. videos/output.mp4)",
    )
    return parser.parse_args()


def main() -> int:
    """Application entry point."""
    args = parse_args()

    try:
        video_source = resolve_video_source(args.source)
        system = FPVTargetingSystem(
            video_source=video_source,
            model_path=args.model,
            export_path=args.export,
        )
        system.run()
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("%s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
