"""
Diagnostic visualizations for YOLOv8s fine-tuned:
  1. Confusion matrix on validation set
  2. GT vs prediction side-by-side for one example per defect class

Usage:
    python scripts/plot_diagnostics.py
    python scripts/plot_diagnostics.py --out demo/
    python scripts/plot_diagnostics.py --conf 0.25 --iou 0.5
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

VAL_IMAGES_DIR = PROJECT_ROOT / "data/yolo_dataset/images/val"
VAL_LABELS_DIR = PROJECT_ROOT / "data/yolo_dataset/labels/val"
YOLO_S_WEIGHTS = PROJECT_ROOT / "data/yolov8s_ft.pt"

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

COLORS = {
    "crazing":          "#ef4444",
    "inclusion":        "#f97316",
    "patches":          "#eab308",
    "pitted_surface":   "#22c55e",
    "rolled-in_scale":  "#3b82f6",
    "scratches":        "#a855f7",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_gt(label_path: Path, img_w: int, img_h: int) -> list[dict]:
    boxes = []
    for line in label_path.read_text().strip().splitlines():
        if not line:
            continue
        cls_id, cx, cy, w, h = map(float, line.split())
        boxes.append({
            "label": CLASSES[int(cls_id)],
            "x1": (cx - w / 2) * img_w,
            "y1": (cy - h / 2) * img_h,
            "x2": (cx + w / 2) * img_w,
            "y2": (cy + h / 2) * img_h,
        })
    return boxes


def iou(a: dict, b: dict) -> float:
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    area_b = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def predict(model: YOLO, img: Image.Image, conf: float) -> list[dict]:
    res = model(img, conf=conf, verbose=False)[0]
    boxes = []
    if res.boxes is not None:
        for box in res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes.append({
                "label": res.names[int(box.cls[0])],
                "score": float(box.conf[0]),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })
    return boxes


# ── Chart 1: Confusion matrix ─────────────────────────────────────────────────

def build_confusion_matrix(model: YOLO, conf: float, iou_thresh: float) -> np.ndarray:
    n = len(CLASSES)
    # (n+1) x (n+1): indices 0..n-1 = defect classes, index n = background
    cm = np.zeros((n + 1, n + 1), dtype=int)

    image_paths = sorted(VAL_IMAGES_DIR.glob("*.jpg"))
    total = len(image_paths)
    print(f"Building confusion matrix on {total} images...")

    for i, img_path in enumerate(image_paths, 1):
        if i % 50 == 0 or i == total:
            print(f"  {i}/{total}", flush=True)

        label_path = VAL_LABELS_DIR / img_path.with_suffix(".txt").name
        if not label_path.exists():
            continue

        img = Image.open(img_path)
        gts   = load_gt(label_path, img.width, img.height)
        preds = sorted(predict(model, img, conf), key=lambda x: x["score"], reverse=True)
        matched_gt = set()

        for pred in preds:
            if pred["label"] not in CLASSES:
                continue
            best_iou, best_idx = 0.0, -1
            for j, gt in enumerate(gts):
                if j in matched_gt:
                    continue
                v = iou(pred, gt)
                if v > best_iou:
                    best_iou, best_idx = v, j

            pred_cls = CLASSES.index(pred["label"])
            if best_iou >= iou_thresh and best_idx >= 0:
                matched_gt.add(best_idx)
                gt_cls = CLASSES.index(gts[best_idx]["label"])
                cm[gt_cls][pred_cls] += 1          # TP / misclassification
            else:
                cm[n][pred_cls] += 1               # FP: background → pred_cls

        for j, gt in enumerate(gts):
            if j not in matched_gt:
                gt_cls = CLASSES.index(gt["label"])
                cm[gt_cls][n] += 1                 # FN: gt_cls → background

    return cm


def chart_confusion_matrix(model: YOLO, conf: float, iou_thresh: float, out_dir: Path):
    cm = build_confusion_matrix(model, conf, iou_thresh)
    n = len(CLASSES)
    labels = CLASSES + ["background"]

    # Normalize by row
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm / row_sums

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)

    ax.set_xticks(range(n + 1))
    ax.set_yticks(range(n + 1))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(
        f"Confusion matrix — YOLOv8s fine-tuned\n"
        f"(conf={conf}, IoU≥{iou_thresh}, normalized by row)\n"
        f"last row = FP (background→pred),  last col = FN (GT→missed)"
    )

    for i in range(n + 1):
        for j in range(n + 1):
            val = cm_norm[i, j]
            text_color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}\n({cm[i, j]})",
                    ha="center", va="center", fontsize=7.5, color=text_color)

    fig.tight_layout()
    out_path = out_dir / "chart_confusion_matrix.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Chart 2: GT vs prediction ─────────────────────────────────────────────────

def draw_boxes(ax, img: Image.Image, boxes: list[dict], show_score: bool):
    ax.imshow(img, cmap="gray" if img.mode == "L" else None)
    ax.axis("off")
    for box in boxes:
        color = COLORS.get(box["label"], "#888888")
        x, y = box["x1"], box["y1"]
        w, h = box["x2"] - box["x1"], box["y2"] - box["y1"]
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="square,pad=0",
            linewidth=1.5, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)
        label = box["label"]
        if show_score:
            label += f" {box['score']:.2f}"
        ax.text(x, y - 3, label, fontsize=7, color="white",
                bbox=dict(facecolor=color, alpha=0.85, pad=1, edgecolor="none"))


def chart_gt_vs_pred(model: YOLO, conf: float, out_dir: Path):
    # Pick one image per class from validation set
    class_examples: dict[str, Path] = {}
    for img_path in sorted(VAL_IMAGES_DIR.glob("*.jpg")):
        label_path = VAL_LABELS_DIR / img_path.with_suffix(".txt").name
        if not label_path.exists():
            continue
        for line in label_path.read_text().strip().splitlines():
            if not line:
                continue
            cls_id = int(line.split()[0])
            cls_name = CLASSES[cls_id]
            if cls_name not in class_examples:
                class_examples[cls_name] = img_path
        if len(class_examples) == len(CLASSES):
            break

    n = len(class_examples)
    fig, axes = plt.subplots(n, 2, figsize=(8, 3.2 * n))
    fig.suptitle("Ground truth  vs  YOLOv8s prediction", fontsize=13, fontweight="bold", y=1.01)

    for row, (cls_name, img_path) in enumerate(class_examples.items()):
        label_path = VAL_LABELS_DIR / img_path.with_suffix(".txt").name
        img = Image.open(img_path)

        gt_boxes   = load_gt(label_path, img.width, img.height)
        pred_boxes = predict(model, img, conf)

        ax_gt, ax_pred = axes[row]
        draw_boxes(ax_gt,   img, gt_boxes,   show_score=False)
        draw_boxes(ax_pred, img, pred_boxes, show_score=True)

        ax_gt.set_title(f"GT — {cls_name}", fontsize=9, loc="left")
        ax_pred.set_title("Prediction", fontsize=9, loc="left")

    fig.tight_layout()
    out_path = out_dir / "chart_gt_vs_pred.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Chart 3: Class examples grid ─────────────────────────────────────────────

def chart_class_examples(out_dir: Path):
    class_examples: dict[str, Path] = {}
    for img_path in sorted(VAL_IMAGES_DIR.glob("*.jpg")):
        label_path = VAL_LABELS_DIR / img_path.with_suffix(".txt").name
        if not label_path.exists():
            continue
        for line in label_path.read_text().strip().splitlines():
            if not line:
                continue
            cls_id = int(line.split()[0])
            cls_name = CLASSES[cls_id]
            if cls_name not in class_examples:
                class_examples[cls_name] = img_path
        if len(class_examples) == len(CLASSES):
            break

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    fig.suptitle("NEU-DET — defect classes with ground truth", fontsize=12, fontweight="bold")

    for ax, cls_name in zip(axes.flat, CLASSES):
        img_path   = class_examples[cls_name]
        label_path = VAL_LABELS_DIR / img_path.with_suffix(".txt").name
        img        = Image.open(img_path)
        boxes      = load_gt(label_path, img.width, img.height)
        draw_boxes(ax, img, boxes, show_score=False)
        ax.set_title(cls_name, fontsize=10, fontweight="bold",
                     color=COLORS.get(cls_name, "#222"))

    fig.tight_layout()
    out_path = out_dir / "chart_class_examples.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Chart 4: Precision-Recall curve ──────────────────────────────────────────

DATASET_YAML = str(PROJECT_ROOT / "data/yolo_dataset/dataset.yaml")


def chart_pr_curve(model: YOLO, out_dir: Path):
    import shutil
    print("Running model.val() to generate PR curves...")
    project_dir = PROJECT_ROOT / "runs" / "detect"
    run_name = "val_pr"
    model.val(
        data=DATASET_YAML,
        imgsz=640,
        conf=0.001,
        iou=0.5,
        plots=True,
        save_json=False,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        verbose=False,
    )

    copies = {
        "BoxPR_curve.png": "chart_pr_curve.png",
        "BoxF1_curve.png": "chart_f1_curve.png",
    }
    for src_name, dst_name in copies.items():
        src = project_dir / run_name / src_name
        if src.exists():
            dst = out_dir / dst_name
            shutil.copy(src, dst)
            print(f"Saved: {dst}")
        else:
            print(f"Not found: {src}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold (default: 0.25)")
    parser.add_argument("--iou",  type=float, default=0.5,  help="IoU match threshold for CM (default: 0.5)")
    parser.add_argument("--out",  type=Path,  default=Path("demo"))
    args = parser.parse_args()

    if not YOLO_S_WEIGHTS.exists():
        print(f"Weights not found: {YOLO_S_WEIGHTS}")
        return

    args.out.mkdir(parents=True, exist_ok=True)

    print("Loading YOLOv8s fine-tuned...")
    model = YOLO(str(YOLO_S_WEIGHTS))

    print("\n--- Class examples grid ---")
    chart_class_examples(args.out)

    print("\n--- Confusion matrix ---")
    chart_confusion_matrix(model, args.conf, args.iou, args.out)

    print("\n--- GT vs prediction ---")
    chart_gt_vs_pred(model, args.conf, args.out)

    print("\n--- PR curve ---")
    chart_pr_curve(model, args.out)

    print("\nDone.")


if __name__ == "__main__":
    main()
