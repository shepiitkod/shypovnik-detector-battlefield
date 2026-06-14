#!/usr/bin/env python3
"""Temporary script to merge the second Roboflow dataset into YOLO structure."""

from __future__ import annotations

import shutil
from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parent / "dataset"
TRAIN_IMAGES = DATASET_ROOT / "train" / "images"
TRAIN_LABELS = DATASET_ROOT / "train" / "labels"
VALID_IMAGES = DATASET_ROOT / "valid" / "images"
VALID_LABELS = DATASET_ROOT / "valid" / "labels"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TANK_CLASS_IDS = {0, 2}  # old dataset=0, new multi-class export tank=2
TARGET_CLASS_ID = 0

CLEANUP_FILES = [
    "data copy.yaml",
    "README.dataset copy.txt",
    "README.roboflow copy.txt",
    ".DS_Store",
]

CLEANUP_DIRS = [
    DATASET_ROOT / "train" / "labels.cache",
]


def is_tank_image(stem: str) -> bool:
    """Heuristic: tank samples use tank- or Tank_Images prefixes."""
    lowered = stem.lower()
    return lowered.startswith("tank") or lowered.startswith("tank_images")


def remap_label_file(src: Path, dst: Path) -> bool:
    """
    Keep only tank annotations and remap class IDs to 0.

    Returns True when at least one tank annotation is written.
    """
    if not src.exists():
        return False

    lines_out: list[str] = []
    for raw in src.read_text().splitlines():
        parts = raw.strip().split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        if class_id not in TANK_CLASS_IDS:
            continue
        parts[0] = str(TARGET_CLASS_ID)
        lines_out.append(" ".join(parts))

    if not lines_out:
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines_out) + "\n")
    return True


def discover_source_dirs() -> list[Path]:
    """Find folders that contain paired images/ and labels/ directories."""
    sources: list[Path] = []
    for path in sorted(DATASET_ROOT.rglob("images")):
        if path.parent == DATASET_ROOT:
            continue
        labels_dir = path.parent / "labels"
        if labels_dir.is_dir():
            sources.append(path.parent)
    return sources


def find_new_train_sources(all_sources: list[Path]) -> list[Path]:
    """
    Identify folders that contain new-export tank samples for training.

    New Roboflow exports use tank-/soldier- prefixes, while the original
    dataset uses Tank_Images_* names.
    """
    candidates: list[Path] = []
    canonical_train = (DATASET_ROOT / "train").resolve()
    canonical_valid = (DATASET_ROOT / "valid").resolve()

    for source in all_sources:
        resolved = source.resolve()
        if resolved == canonical_valid:
            continue

        images_dir = source / "images"
        has_new_style = any(
            p.stem.lower().startswith("tank-")
            for p in images_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if has_new_style:
            candidates.append(source)

    if canonical_train not in [s.resolve() for s in candidates]:
        has_new_in_train = any(
            p.stem.lower().startswith("tank-")
            for p in (canonical_train / "images").glob("*")
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if has_new_in_train:
            candidates.append(DATASET_ROOT / "train")

    return candidates


def merge_split(
    source_dir: Path,
    dest_images: Path,
    dest_labels: Path,
    *,
    skip_existing: bool = True,
) -> tuple[int, int]:
    """
    Merge tank image/label pairs from a source split into destination.

    Returns:
        (images_merged, images_skipped)
    """
    src_images = source_dir / "images"
    src_labels = source_dir / "labels"
    merged = 0
    skipped = 0

    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)

    for image_path in sorted(src_images.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if not is_tank_image(image_path.stem):
            continue

        label_src = src_labels / f"{image_path.stem}.txt"
        label_dst = dest_labels / f"{image_path.stem}.txt"
        image_dst = dest_images / image_path.name

        if not remap_label_file(label_src, label_dst):
            if label_dst.exists():
                label_dst.unlink()
            continue

        if image_dst.exists() and skip_existing:
            skipped += 1
            continue

        shutil.copy2(image_path, image_dst)
        merged += 1

    return merged, skipped


def rebuild_valid_from(source_dir: Path) -> tuple[int, int]:
    """Rebuild valid/ with tank-only, remapped labels."""
    staging_images = DATASET_ROOT / "_staging_valid_images"
    staging_labels = DATASET_ROOT / "_staging_valid_labels"

    if staging_images.exists():
        shutil.rmtree(staging_images)
    if staging_labels.exists():
        shutil.rmtree(staging_labels)

    merged, skipped = merge_split(
        source_dir,
        staging_images,
        staging_labels,
        skip_existing=False,
    )

    if VALID_IMAGES.exists():
        shutil.rmtree(VALID_IMAGES)
    if VALID_LABELS.exists():
        shutil.rmtree(VALID_LABELS)

    staging_images.rename(VALID_IMAGES)
    staging_labels.rename(VALID_LABELS)
    return merged, skipped


def cleanup_dataset_root() -> None:
    """Remove duplicate configs and cache files."""
    for name in CLEANUP_FILES:
        path = DATASET_ROOT / name
        if path.is_file():
            path.unlink()
            print(f"  removed file: {path.name}")

    for path in CLEANUP_DIRS:
        if path.is_file():
            path.unlink()
            print(f"  removed cache: {path.name}")


def write_data_yaml() -> None:
    """Write canonical single-class data.yaml."""
    content = (
        "train: train/images\n"
        "val: valid/images\n"
        "\n"
        "nc: 1\n"
        "names: ['Tanks']\n"
    )
    (DATASET_ROOT / "data.yaml").write_text(content)
    print("  updated data.yaml")


def count_pairs(images_dir: Path) -> int:
    """Count image files in a directory."""
    if not images_dir.is_dir():
        return 0
    return sum(
        1 for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )


def main() -> None:
    print("=== Dataset Merge ===\n")

    sources = discover_source_dirs()
    print(f"Found source splits: {[p.relative_to(DATASET_ROOT) for p in sources]}")

    new_train_sources = find_new_train_sources(sources)

    # Merge any new-export train folders (tank-* naming) into canonical train/.
    train_merged_total = 0
    train_skipped_total = 0
    if new_train_sources:
        for source in new_train_sources:
            merged, skipped = merge_split(source, TRAIN_IMAGES, TRAIN_LABELS)
            train_merged_total += merged
            train_skipped_total += skipped
            print(
                f"Merged train from {source.relative_to(DATASET_ROOT)}: "
                f"+{merged} new, {skipped} duplicates skipped"
            )
    else:
        print(
            "No separate new-train folder found (tank-* images). "
            "Keeping existing train/ images."
        )

    # Snapshot new valid export before rebuilding it.
    valid_source = DATASET_ROOT / "valid"
    if not valid_source.is_dir():
        raise FileNotFoundError("Expected new valid/ folder under dataset/")

    staging_valid = DATASET_ROOT / "_staging_valid_source"
    if staging_valid.exists():
        shutil.rmtree(staging_valid)
    shutil.copytree(valid_source, staging_valid)

    valid_merged, valid_skipped = rebuild_valid_from(staging_valid)
    shutil.rmtree(staging_valid)
    print(
        f"Rebuilt valid/: {valid_merged} tank images "
        f"({valid_skipped} without usable labels skipped)"
    )

    print("\n=== Cleanup ===")
    cleanup_dataset_root()
    write_data_yaml()

    print("\n=== Final counts ===")
    print(f"  train/images: {count_pairs(TRAIN_IMAGES)}")
    print(f"  train/labels: {count_pairs(TRAIN_LABELS)}")
    print(f"  valid/images: {count_pairs(VALID_IMAGES)}")
    print(f"  valid/labels: {count_pairs(VALID_LABELS)}")
    print("\nMerge complete.")


if __name__ == "__main__":
    main()
