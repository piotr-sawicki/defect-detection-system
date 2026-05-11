"""
Evaluate detection models on the validation set and save results to CSV.

YOLOv8n results are loaded from cache (already evaluated) by default.
Use --rerun-yolo-n to force re-evaluation.

Usage:
    python scripts/evaluate_models.py
    python scripts/evaluate_models.py --rcnn
    python scripts/evaluate_models.py --rerun-yolo-n
    python scripts/evaluate_models.py --out results/eval_results.csv
"""

import argparse
import csv
import sys
from datetime import datetime
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
YOLO_N_WEIGHTS      = PROJECT_ROOT / "data/yolov8n_ft.pt"
YOLO_S_WEIGHTS      = PROJECT_ROOT / "data/yolov8s_ft.pt"

CLASSES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches",
]

# Previously evaluated results — used instead of re-running inference
YOLO_N_CACHED = {
    "mAP50":    0.7261,
    "mAP50-95": 0.4167,
    "mAP50-95_crazing":        0.1519,
    "mAP50-95_inclusion":      0.4353,
    "mAP50-95_patches":        0.5974,
    "mAP50-95_pitted_surface": 0.5421,
    "mAP50-95_rolled-in_scale": 0.2290,
    "mAP50-95_scratches":      0.5446,
}


# ── Ground truth ──────────────────────────────────────────────────────────────

def load_ground_truth(label_path: Path, img_w: int, img_h: int) -> dict:
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


# ── Model builders ────────────────────────────────────────────────────────────

def build_faster_rcnn() -> torch.nn.Module:
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
    return {
        "boxes":  output["boxes"][keep],
        "scores": output["scores"][keep],
        "labels": output["labels"][keep] - 1,
    }


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

EVAL_THRESHOLD = 0.001


def evaluate(name: str, predict_fn) -> dict:
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
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

    raw = metric.compute()
    row = {
        "mAP50":    round(raw["map_50"].item(), 4),
        "mAP50-95": round(raw["map"].item(), 4),
    }
    for cls_name, ap in zip(CLASSES, raw.get("map_per_class", [])):
        row[f"mAP50-95_{cls_name}"] = round(ap.item(), 4)
    return row


# ── Printing ──────────────────────────────────────────────────────────────────

def print_results(name: str, row: dict) -> None:
    print(f"\n{'═' * 52}")
    print(f"  {name}")
    print(f"{'═' * 52}")
    print(f"  {'mAP50':<20} {row['mAP50']:>8.4f}")
    print(f"  {'mAP50-95':<20} {row['mAP50-95']:>8.4f}")
    per = {k: v for k, v in row.items() if k.startswith("mAP50-95_")}
    if per:
        print(f"\n  {'Class':<22} {'mAP50-95':>8}")
        print(f"  {'-' * 32}")
        for k, v in per.items():
            print(f"  {k.removeprefix('mAP50-95_'):<22} {v:>8.4f}")


def print_comparison(all_results: dict[str, dict]) -> None:
    names = list(all_results.keys())
    col = 12
    width = 22 + col * len(names) + 4
    print(f"\n{'═' * width}")
    print(f"  COMPARISON")
    print(f"{'═' * width}")
    print(f"  {'Class':<22}" + "".join(f"{n[:col]:>{col}}" for n in names))
    print(f"  {'-' * (width - 2)}")
    print(f"  {'mAP50 (overall)':<22}" +
          "".join(f"{r['mAP50']:>{col}.4f}" for r in all_results.values()))
    print(f"  {'mAP50-95 (overall)':<22}" +
          "".join(f"{r['mAP50-95']:>{col}.4f}" for r in all_results.values()))

    per_keys = [k for k in next(iter(all_results.values())) if k.startswith("mAP50-95_")]
    if per_keys:
        print(f"  {'-' * (width - 2)}")
        for k in per_keys:
            cls_name = k.removeprefix("mAP50-95_")
            print(f"  {cls_name:<22}" +
                  "".join(f"{r.get(k, float('nan')):>{col}.4f}" for r in all_results.values()))


# ── CSV ───────────────────────────────────────────────────────────────────────

def save_csv(all_results: dict[str, dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = [{"timestamp": ts, "model": name, **row} for name, row in all_results.items()]
    fieldnames = list(rows[0].keys())
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults appended to {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rcnn",         action="store_true", help="evaluate Faster R-CNN (~10 min)")
    parser.add_argument("--rerun-yolo-n", action="store_true", help="re-evaluate YOLOv8n instead of using cached values")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "results/eval_results.csv",
                        help="output CSV path (default: results/eval_results.csv)")
    args = parser.parse_args()

    all_results = {}

    # YOLOv8n — cached by default
    if args.rerun_yolo_n:
        print("Loading YOLOv8n fine-tuned...")
        yolo_n = YOLO(str(YOLO_N_WEIGHTS))
        all_results["YOLOv8n fine-tuned"] = evaluate(
            "YOLOv8n fine-tuned",
            lambda img: predict_yolo(yolo_n, img, EVAL_THRESHOLD),
        )
    else:
        print("YOLOv8n fine-tuned: using cached results (pass --rerun-yolo-n to re-evaluate)")
        all_results["YOLOv8n fine-tuned"] = YOLO_N_CACHED

    # YOLOv8s
    print("\nLoading YOLOv8s fine-tuned...")
    yolo_s = YOLO(str(YOLO_S_WEIGHTS))
    all_results["YOLOv8s fine-tuned"] = evaluate(
        "YOLOv8s fine-tuned",
        lambda img: predict_yolo(yolo_s, img, EVAL_THRESHOLD),
    )

    # Faster R-CNN (optional, slow)
    if args.rcnn:
        print("\nLoading Faster R-CNN...")
        rcnn = build_faster_rcnn()
        all_results["Faster R-CNN"] = evaluate(
            "Faster R-CNN",
            lambda img: predict_faster_rcnn(rcnn, img, EVAL_THRESHOLD),
        )

    for name, row in all_results.items():
        print_results(name, row)

    print_comparison(all_results)
    save_csv(all_results, args.out)


if __name__ == "__main__":
    main()
