"""
Compare Faster R-CNN vs YOLOv8 fine-tuned on the validation set.

Both models are evaluated with the same pipeline using torchmetrics,
so the comparison is fair (same IoU thresholds, same ground truth).

Requirements:
    pip install torchmetrics

Usage:
    python scripts/evaluate_models.py
"""

import sys
from pathlib import Path

import torch
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image
from torchmetrics.detection import MeanAveragePrecision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

VAL_IMAGES_DIR = PROJECT_ROOT / "data/yolo_dataset/images/val"
VAL_LABELS_DIR = PROJECT_ROOT / "data/yolo_dataset/labels/val"

FASTER_RCNN_WEIGHTS = PROJECT_ROOT / "data/FastRCNNweights.pth"
YOLO_WEIGHTS        = PROJECT_ROOT / "data/yolov8n.pt"

# Shared class list (0-indexed, no background)
CLASSES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches",
]


# ── Ground truth ──────────────────────────────────────────────────────────────

def load_ground_truth(label_path: Path, img_w: int, img_h: int) -> dict:
    """Read a YOLO-format .txt label and convert back to absolute xyxy boxes."""
    boxes, labels = [], []
    for line in label_path.read_text().strip().splitlines():
        if not line:
            continue
        class_id, cx, cy, w, h = map(float, line.split())
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        boxes.append([x1, y1, x2, y2])
        labels.append(int(class_id))
    return {
        "boxes":  torch.tensor(boxes,  dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


# ── Faster R-CNN ──────────────────────────────────────────────────────────────

def build_faster_rcnn() -> torch.nn.Module:
    # +1 because Faster R-CNN reserves label 0 for background
    num_classes = len(CLASSES) + 1
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(str(FASTER_RCNN_WEIGHTS), map_location="cpu"))
    model.eval()
    return model


def predict_faster_rcnn(model: torch.nn.Module, img: Image.Image, threshold: float) -> dict:
    img_tensor = TF.to_tensor(img.convert("RGB"))
    with torch.no_grad():
        output = model([img_tensor])[0]

    keep = output["scores"] >= threshold
    # Shift labels: Faster R-CNN label 1 → class index 0 (remove background offset)
    return {
        "boxes":  output["boxes"][keep],
        "scores": output["scores"][keep],
        "labels": output["labels"][keep] - 1,
    }


# ── YOLOv8 fine-tuned ─────────────────────────────────────────────────────────

def predict_yolo(model: YOLO, img: Image.Image, threshold: float) -> dict:
    results = model(img, conf=threshold, verbose=False)[0]
    if results.boxes is None or len(results.boxes) == 0:
        return {
            "boxes":  torch.zeros((0, 4), dtype=torch.float32),
            "scores": torch.zeros(0,      dtype=torch.float32),
            "labels": torch.zeros(0,      dtype=torch.int64),
        }
    return {
        "boxes":  results.boxes.xyxy.cpu(),
        "scores": results.boxes.conf.cpu(),
        "labels": results.boxes.cls.cpu().to(torch.int64),
    }


# ── Evaluation loop ───────────────────────────────────────────────────────────

# Low threshold so torchmetrics receives all predictions and can sweep
# the full precision-recall curve internally. This is required for correct mAP.
EVAL_THRESHOLD = 0.001


def evaluate(name: str, predict_fn) -> dict:
    metric = MeanAveragePrecision(
        iou_type="bbox",
        class_metrics=True,
    )

    image_paths = sorted(VAL_IMAGES_DIR.glob("*.jpg"))
    total = len(image_paths)
    print(f"\nEvaluating {name} on {total} images...")

    for i, img_path in enumerate(image_paths, 1):
        if i % 10 == 0 or i == total:
            print(f"  {i}/{total} ({100 * i // total}%)", flush=True)
        img = Image.open(img_path)
        label_path = VAL_LABELS_DIR / img_path.with_suffix(".txt").name

        if not label_path.exists():
            continue

        gt   = load_ground_truth(label_path, img.width, img.height)
        pred = predict_fn(img)

        metric.update([pred], [gt])

    return metric.compute()


# ── Results table ─────────────────────────────────────────────────────────────

def print_results(name: str, results: dict) -> None:
    print(f"\n{'═' * 52}")
    print(f"  {name}")
    print(f"{'═' * 52}")
    print(f"  {'Metric':<20} {'Value':>8}")
    print(f"  {'-' * 30}")
    print(f"  {'mAP50':<20} {results['map_50'].item():>8.4f}")
    print(f"  {'mAP50-95':<20} {results['map'].item():>8.4f}")

    if "map_per_class" in results and results["map_per_class"].numel() == len(CLASSES):
        print(f"\n  {'Class':<22} {'mAP50':>8}")
        print(f"  {'-' * 32}")
        for cls_name, ap in zip(CLASSES, results["map_per_class"]):
            print(f"  {cls_name:<22} {ap.item():>8.4f}")


def print_comparison(rcnn_results: dict, yolo_results: dict) -> None:
    print(f"\n{'═' * 60}")
    print(f"  COMPARISON")
    print(f"{'═' * 60}")
    print(f"  {'Class':<22} {'Faster R-CNN':>12} {'YOLOv8-ft':>12}")
    print(f"  {'-' * 48}")

    rcnn_map50 = rcnn_results["map_50"].item()
    yolo_map50 = yolo_results["map_50"].item()
    print(f"  {'mAP50 (overall)':<22} {rcnn_map50:>12.4f} {yolo_map50:>12.4f}")

    rcnn_per = rcnn_results.get("map_per_class", [])
    yolo_per = yolo_results.get("map_per_class", [])
    if len(rcnn_per) == len(CLASSES) and len(yolo_per) == len(CLASSES):
        print(f"  {'-' * 48}")
        for cls_name, r_ap, y_ap in zip(CLASSES, rcnn_per, yolo_per):
            winner = "<" if r_ap < y_ap else ">"
            print(f"  {cls_name:<22} {r_ap.item():>12.4f} {y_ap.item():>12.4f}  {winner}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading YOLOv8 fine-tuned...")
    yolo_model = YOLO(str(YOLO_WEIGHTS))

    print("Loading Faster R-CNN...")
    rcnn_model = build_faster_rcnn()

    yolo_results = evaluate(
        "YOLOv8 fine-tuned",
        lambda img: predict_yolo(yolo_model, img, EVAL_THRESHOLD),
    )

    rcnn_results = evaluate(
        "Faster R-CNN",
        lambda img: predict_faster_rcnn(rcnn_model, img, EVAL_THRESHOLD),
    )



    print_results("YOLOv8 fine-tuned", yolo_results)
    print_results("Faster R-CNN", rcnn_results)

    print_comparison(rcnn_results, yolo_results)


if __name__ == "__main__":
    main()