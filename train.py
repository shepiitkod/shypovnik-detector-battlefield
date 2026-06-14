#!/usr/bin/env python3
"""
=============================================================================
 FPV Military Target Detector — YOLOv8 Fine-Tuning Script
=============================================================================

 HOW TO FIND & DOWNLOAD OPEN-SOURCE DATASETS (Roboflow Universe)
 -----------------------------------------------------------------
 1. Go to https://universe.roboflow.com/
 2. Search with military / FPV-related keywords, for example:
      - "military vehicles"
      - "tanks dataset"
      - "artillery"
      - "drone thermal"
      - "military object detection"
      - "FPV drone"
      - "battlefield"
 3. Filter by:
      - License: CC BY, CC0, or MIT (verify attribution requirements)
      - Format: Object Detection
      - Task: Bounding boxes
 4. Open a dataset → click "Download Dataset"
 5. Select export format: **YOLOv8** (NOT YOLOv5 or COCO)
 6. Choose "show download code" or direct zip download
 7. Unzip the archive into this project:

      fpv-targeting-system/
      └── dataset/              <-- place Roboflow export here
          ├── data.yaml
          ├── train/
          │   ├── images/
          │   └── labels/
          ├── valid/
          │   ├── images/
          │   └── labels/
          └── test/               (optional)
              ├── images/
              └── labels/

 8. Edit data.yaml if needed — class names should map to:
      0: Infantry
      1: Tank
      2: Artillery
      3: Bunker

    If your Roboflow export uses different names/order, update data.yaml
    AND src/detector.py MILITARY_CLASSES to stay in sync.

 TIPS FOR GOOD RESULTS
 ---------------------
 - Combine multiple Roboflow datasets via Roboflow's merge feature
 - Aim for 500+ images per class if possible
 - Include thermal / low-light / motion-blur frames for FPV realism
 - Use 50 epochs for a quick local sanity check
 - Scale to 100–300 epochs for production-quality weights

 USAGE
 -----
   python train.py
   python train.py --epochs 100 --batch -1
   python train.py --data dataset/data.yaml --device cpu

 OUTPUT
 ------
 - Training run:  runs/detect/military_train/
 - Best weights:  models/best.pt  (copied automatically after training)
=============================================================================
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATASET_DIR: Path = PROJECT_ROOT / "dataset"
DATA_YAML: Path = DATASET_DIR / "data.yaml"
MODELS_DIR: Path = PROJECT_ROOT / "models"
BASE_MODEL: str = "yolov8n.pt"
OUTPUT_BEST: Path = MODELS_DIR / "best.pt"

# Training defaults (tuned for a typical local PC)
DEFAULT_EPOCHS: int = 50
# For final production training, increase to 100–300 epochs:
#   python train.py --epochs 200
DEFAULT_IMGSZ: int = 640
DEFAULT_BATCH: int = 16
# Use --batch -1 to let Ultralytics auto-select batch size based on VRAM
DEFAULT_WORKERS: int = 4
RUN_PROJECT: str = "runs/detect"
RUN_NAME: str = "military_train"


def resolve_device(requested: str | int | None) -> str | int:
    """
    Select the best available compute device without crashing.

    Args:
        requested: User override ('cpu', '0', 'cuda', etc.) or None for auto.

    Returns:
        Device string or GPU index for Ultralytics.
    """
    if requested is not None and str(requested).lower() != "auto":
        return requested

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        logger.info("CUDA available — using GPU 0 (%s)", gpu_name)
        return 0

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("CUDA not found — using Apple Metal (mps).")
        return "mps"

    logger.warning("No GPU detected — falling back to CPU (training will be slow).")
    return "cpu"


def validate_dataset(data_yaml: Path, dataset_dir: Path) -> None:
    """
    Verify that the dataset directory and data.yaml exist before training.

    Raises:
        SystemExit: When the dataset is missing or incomplete.
    """
    if not dataset_dir.is_dir():
        _print_dataset_instructions()
        logger.error("Dataset directory not found: %s", dataset_dir)
        sys.exit(1)

    if not data_yaml.is_file():
        _print_dataset_instructions()
        logger.error("data.yaml not found: %s", data_yaml)
        sys.exit(1)

    train_images = dataset_dir / "train" / "images"
    valid_images = dataset_dir / "valid" / "images"

    if not train_images.is_dir() or not any(train_images.iterdir()):
        logger.error(
            "Training images missing or empty. Expected: %s", train_images
        )
        _print_dataset_instructions()
        sys.exit(1)

    if not valid_images.is_dir() or not any(valid_images.iterdir()):
        logger.warning(
            "Validation images not found at %s — training may fail. "
            "Re-export from Roboflow with train/valid splits.",
            valid_images,
        )

    logger.info("Dataset validated: %s", data_yaml)


def _print_dataset_instructions() -> None:
    """Print setup instructions for the Roboflow dataset."""
    print(
        "\n"
        "╔══════════════════════════════════════════════════════════════════╗\n"
        "║  DATASET NOT FOUND — Setup Instructions                         ║\n"
        "╠══════════════════════════════════════════════════════════════════╣\n"
        "║  1. Visit https://universe.roboflow.com/                        ║\n"
        "║  2. Search: 'military vehicles', 'tanks', 'drone thermal'       ║\n"
        "║  3. Download in **YOLOv8** format                               ║\n"
        "║  4. Unzip into:                                                 ║\n"
        f"║     {DATASET_DIR}/{' ' * max(0, 47 - len(str(DATASET_DIR)))}║\n"
        "║  5. Ensure this file exists:                                   ║\n"
        f"║     {DATA_YAML}{' ' * max(0, 47 - len(str(DATA_YAML)))}║\n"
        "║                                                                 ║\n"
        "║  Expected structure:                                            ║\n"
        "║    dataset/data.yaml                                            ║\n"
        "║    dataset/train/images/  +  dataset/train/labels/            ║\n"
        "║    dataset/valid/images/  +  dataset/valid/labels/            ║\n"
        "╚══════════════════════════════════════════════════════════════════╝\n"
    )


def copy_best_weights(run_dir: Path, destination: Path) -> Path:
    """
    Copy trained best.pt weights into the models/ directory.

    Args:
        run_dir: Ultralytics run output directory.
        destination: Target path (models/best.pt).

    Returns:
        Path to the copied weights file.

    Raises:
        FileNotFoundError: When best.pt is not produced by the training run.
    """
    source = run_dir / "weights" / "best.pt"
    if not source.is_file():
        raise FileNotFoundError(
            f"Training finished but best.pt not found at: {source}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    logger.info("Best weights saved to: %s", destination)
    return destination


def print_validation_metrics(metrics: object) -> None:
    """
    Log final mAP and per-class precision/recall from model.val().

    Args:
        metrics: Metrics object returned by YOLO.val().
    """
    box = metrics.box
    logger.info("─── Validation Results ───")
    logger.info("  mAP50:      %.4f", box.map50)
    logger.info("  mAP50-95:   %.4f", box.map)
    logger.info("  Precision:  %.4f", box.mp)
    logger.info("  Recall:     %.4f", box.mr)

    if hasattr(box, "ap_class_index") and box.ap_class_index is not None:
        for i, class_idx in enumerate(box.ap_class_index):
            class_name = metrics.names.get(int(class_idx), str(class_idx))
            ap50 = box.ap50[i] if i < len(box.ap50) else 0.0
            logger.info("  AP50 [%s]: %.4f", class_name, ap50)


def train(
    data_yaml: Path,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str | int,
    workers: int,
    model_path: str = BASE_MODEL,
) -> Path:
    """
    Fine-tune YOLOv8 on the military dataset and validate the result.

    Args:
        data_yaml: Path to the dataset YAML config.
        epochs: Number of training epochs.
        imgsz: Input image size.
        batch: Batch size (-1 for auto-batch).
        device: Compute device.
        workers: DataLoader worker count.
        model_path: Path to pretrained weights for transfer learning.

    Returns:
        Path to best.pt copied into models/.
    """
    if not Path(model_path).exists() and model_path != BASE_MODEL:
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    logger.info("Loading base model: %s (transfer learning)", model_path)
    model = YOLO(model_path)

    logger.info(
        "Starting training — epochs=%d, imgsz=%d, batch=%d, device=%s, workers=%d",
        epochs,
        imgsz,
        batch,
        device,
        workers,
    )

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        project=RUN_PROJECT,
        name=RUN_NAME,
        exist_ok=True,
        pretrained=True,
        verbose=True,
    )

    run_dir = Path(results.save_dir)
    best_path = copy_best_weights(run_dir, OUTPUT_BEST)

    logger.info("Running post-training validation on best weights...")
    trained_model = YOLO(str(best_path))
    metrics = trained_model.val(data=str(data_yaml), imgsz=imgsz, device=device)
    print_validation_metrics(metrics)

    return best_path


def parse_args() -> argparse.Namespace:
    """Parse optional CLI overrides for training parameters."""
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8n on a military FPV dataset",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DATA_YAML,
        help=f"Path to data.yaml (default: {DATA_YAML})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Training epochs (default: {DEFAULT_EPOCHS}; use 100–300 for final runs)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=DEFAULT_IMGSZ,
        help=f"Image size (default: {DEFAULT_IMGSZ})",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=DEFAULT_BATCH,
        help=f"Batch size, -1 for auto (default: {DEFAULT_BATCH})",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: auto, cpu, 0, mps (default: auto)",
    )
    parser.add_argument(
        "--model",
        default=BASE_MODEL,
        help=f"Pretrained weights for transfer learning (default: {BASE_MODEL})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"DataLoader workers (default: {DEFAULT_WORKERS})",
    )
    return parser.parse_args()


def main() -> int:
    """Entry point."""
    args = parse_args()
    data_yaml: Path = args.data.resolve()
    dataset_dir = data_yaml.parent

    validate_dataset(data_yaml, dataset_dir)
    device = resolve_device(args.device if args.device != "auto" else None)

    try:
        best_path = train(
            data_yaml=data_yaml,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            workers=args.workers,
            model_path=args.model,
        )
        logger.info("Training complete. Deploy with: python main.py --model %s", best_path)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    except Exception as exc:
        logger.error("Training failed: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
