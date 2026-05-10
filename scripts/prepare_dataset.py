"""
Convert Pascal VOC XML annotations to YOLO format and split into train/val sets.

Input:
    data/ANNOTATIONS/*.xml  — Pascal VOC bounding boxes
    data/IMAGES/*.jpg       — source images

Output:
    data/yolo_dataset/
    ├── images/train/       — 80% images per class
    ├── images/val/         — 20% images per class
    ├── labels/train/       — YOLO .txt labels for train
    ├── labels/val/         — YOLO .txt labels for val
    └── dataset.yaml        — config for ultralytics training

YOLO label format (one line per object):
    <class_id> <cx> <cy> <w> <h>
    All values normalized to [0, 1] relative to image size.
"""

import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).parent.parent

ANNOTATIONS_DIR = PROJECT_ROOT / "data/ANNOTATIONS"
IMAGES_DIR      = PROJECT_ROOT / "data/IMAGES"
OUTPUT_DIR      = PROJECT_ROOT / "data/yolo_dataset"

CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]

VAL_SPLIT = 0.2
RANDOM_SEED = 42


def parse_xml(xml_path: Path) -> tuple[int, int, list[tuple[int, str, float, float, float, float]]]:
    """Parse a Pascal VOC XML file and return image size and list of bounding boxes.

    Returns:
        (img_width, img_height, [(class_id, label, xmin, ymin, xmax, ymax), ...])
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)

    boxes = []
    for obj in root.findall("object"):
        label = obj.find("name").text.strip()
        if label not in CLASSES:
            print(f"  WARNING: unknown class '{label}' in {xml_path.name}, skipping")
            continue

        class_id = CLASSES.index(label)
        bb = obj.find("bndbox")
        xmin = float(bb.find("xmin").text)
        ymin = float(bb.find("ymin").text)
        xmax = float(bb.find("xmax").text)
        ymax = float(bb.find("ymax").text)
        boxes.append((class_id, label, xmin, ymin, xmax, ymax))

    return img_w, img_h, boxes


def to_yolo_format(xmin: float, ymin: float, xmax: float, ymax: float,
                   img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """Convert absolute pixel bbox to normalized YOLO format (cx, cy, w, h)."""
    cx = (xmin + xmax) / 2 / img_w
    cy = (ymin + ymax) / 2 / img_h
    w  = (xmax - xmin) / img_w
    h  = (ymax - ymin) / img_h
    return cx, cy, w, h


def write_label(label_path: Path, boxes: list, img_w: int, img_h: int) -> None:
    """Write YOLO-format label file for a single image."""
    lines = []
    for class_id, _, xmin, ymin, xmax, ymax in boxes:
        cx, cy, w, h = to_yolo_format(xmin, ymin, xmax, ymax, img_w, img_h)
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    label_path.write_text("\n".join(lines))


def stratified_split(xml_files: list[Path]) -> tuple[list[Path], list[Path]]:
    """Split files into train/val preserving class balance."""
    by_class: dict[str, list[Path]] = {}
    for xml_path in xml_files:
        # Class name is the prefix before the last underscore+number
        class_name = "_".join(xml_path.stem.split("_")[:-1])
        by_class.setdefault(class_name, []).append(xml_path)

    train, val = [], []
    rng = random.Random(RANDOM_SEED)

    for class_name, files in sorted(by_class.items()):
        shuffled = files[:]
        rng.shuffle(shuffled)
        split_idx = int(len(shuffled) * (1 - VAL_SPLIT))
        train.extend(shuffled[:split_idx])
        val.extend(shuffled[split_idx:])
        print(f"  {class_name}: {split_idx} train / {len(shuffled) - split_idx} val")

    return train, val


def process_split(xml_files: list[Path], split: str) -> int:
    """Convert and copy all files for a given split (train or val)."""
    images_out = OUTPUT_DIR / "images" / split
    labels_out = OUTPUT_DIR / "labels" / split
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    count = 0
    for xml_path in xml_files:
        stem = xml_path.stem
        img_src = IMAGES_DIR / f"{stem}.jpg"

        if not img_src.exists():
            print(f"  WARNING: image not found for {xml_path.name}, skipping")
            continue

        img_w, img_h, boxes = parse_xml(xml_path)
        if not boxes:
            print(f"  WARNING: no valid boxes in {xml_path.name}, skipping")
            continue

        shutil.copy2(img_src, images_out / img_src.name)
        write_label(labels_out / f"{stem}.txt", boxes, img_w, img_h)
        count += 1

    return count


def write_dataset_yaml() -> None:
    """Write dataset.yaml for ultralytics training."""
    yaml_path = OUTPUT_DIR / "dataset.yaml"
    config = {
        "path": str(OUTPUT_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(CLASSES),
        "names": CLASSES,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"\ndataset.yaml written to: {yaml_path}")


def main():
    print("=== Pascal VOC → YOLO dataset conversion ===\n")

    xml_files = sorted(ANNOTATIONS_DIR.glob("*.xml"))
    print(f"Found {len(xml_files)} annotation files\n")

    print("Splitting into train/val (stratified by class):")
    train_files, val_files = stratified_split(xml_files)

    print(f"\nProcessing train split ({len(train_files)} files)...")
    train_count = process_split(train_files, "train")

    print(f"Processing val split ({len(val_files)} files)...")
    val_count = process_split(val_files, "val")

    write_dataset_yaml()

    print(f"\n=== Done ===")
    print(f"Train: {train_count} images")
    print(f"Val:   {val_count} images")
    print(f"Output: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()