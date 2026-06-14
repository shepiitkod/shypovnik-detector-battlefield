# FPV Targeting System

Real-time object detection and lock-on targeting overlay for FPV drone video feeds, powered by YOLOv8.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-orange)

## Features

- **YOLOv8 detection** — Fast real-time inference with `yolov8n.pt`
- **Target filtering** — Tracks cars, trucks, and persons (COCO classes 2, 7, 0)
- **Nearest-target lock-on** — Automatically selects the detection closest to frame center
- **Correction vector** — Computes pixel offset (dX, dY) from crosshair to target
- **Military-style HUD** — Crosshair, corner brackets, telemetry panel, and status overlay
- **Flexible input** — Webcam or video file via `--source` argument

## Installation

```bash
cd fpv-targeting-system
pip install -r requirements.txt
```

> **Note:** The `yolov8n.pt` model weights are downloaded automatically by Ultralytics on the first run.

## Usage

### Webcam (default camera)

```bash
python main.py
```

### Video file

```bash
python main.py --source path/to/video.mp4
```

### Custom model

```bash
python main.py --source 0 --model yolov8s.pt
```

Press **q** to quit.

## Project Structure

```
fpv-targeting-system/
├── main.py           # FPVTargetingSystem — HUD, loop, CLI
├── detector.py       # TargetDetector — YOLOv8 inference & target selection
├── requirements.txt
└── README.md
```

## How It Works

1. Each frame is passed through YOLOv8 and filtered to target classes.
2. The detection whose center is nearest to the frame crosshair is selected as the lock-on target.
3. A correction vector `(dX, dY)` is computed as the pixel offset from crosshair to target center.
4. All detections, the locked target, crosshair, and telemetry are rendered onto the frame.

## Requirements

- Python 3.10+
- Webcam or video file with detectable objects (person, car, truck)
