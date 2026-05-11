"""
Task 1.2 — Measure real inference times for Faster R-CNN and YOLOv8n fine-tuned.

Image source priority:
  1. data/<class>/*.jpg  (full NEU-DET dataset, if downloaded)
  2. examples/images/    (fallback, 3 images per class)

Usage:
  python scripts/measure_inference_time.py
  python scripts/measure_inference_time.py --n 5 --warmup 3 --update-chart
"""

import argparse
import random
import sys
import time
from pathlib import Path

# run from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
EXAMPLES_DIR = Path("examples/images")
DATA_DIR = Path("data")


def collect_images(n: int) -> list[Path]:
    """Return up to n images per class, preferring full dataset over examples."""
    images = []
    for cls in CLASSES:
        # try full dataset first (e.g. data/crazing/crazing_1.jpg or data/images/crazing_1.jpg)
        candidates = (
            list((DATA_DIR / cls).glob("*.jpg"))
            + list((DATA_DIR / "images").glob(f"{cls}_*.jpg"))
            + list(DATA_DIR.glob(f"{cls}_*.jpg"))
        )
        if candidates:
            chosen = random.sample(candidates, min(n, len(candidates)))
        else:
            # fallback: examples/images
            chosen = sorted(EXAMPLES_DIR.glob(f"{cls}_*.jpg"))[:n]
        images.extend(chosen)
        src = "full dataset" if candidates else "examples"
        print(f"  {cls}: {len(chosen)} images ({src})")
    return images


def warmup(predictor, images: list[Path], runs: int):
    for img_path in images[:runs]:
        predictor.predict(str(img_path), threshold=0.5)


def measure(predictor, images: list[Path]) -> list[float]:
    times = []
    for img_path in images:
        t0 = time.perf_counter()
        predictor.predict(str(img_path), threshold=0.5)
        times.append((time.perf_counter() - t0) * 1000)
    return times


def print_stats(name: str, times: list[float]):
    arr = np.array(times)
    print(f"\n{name}")
    print(f"  n={len(arr)}  mean={arr.mean():.1f} ms  std={arr.std():.1f} ms"
          f"  min={arr.min():.1f}  max={arr.max():.1f}")
    return arr.mean(), arr.std()


def update_chart(results: dict):
    """Regenerate the speed-vs-accuracy chart with real measured times."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FASTER_RCNN_MAP50 = 0.9282
    YOLO_MAP50 = 0.7261
    COLORS = {"faster_rcnn": "#2563eb", "yolo": "#f97316"}

    fig, ax = plt.subplots(figsize=(6, 4.5))

    for key, (map50, color, label) in {
        "faster_rcnn": (FASTER_RCNN_MAP50, COLORS["faster_rcnn"], "Faster R-CNN"),
        "yolo_finetuned": (YOLO_MAP50, COLORS["yolo"], "YOLOv8n fine-tuned"),
    }.items():
        mean_ms, std_ms = results[key]
        ax.errorbar(mean_ms, map50, xerr=std_ms, fmt="o", ms=10,
                    color=color, label=label, capsize=5, zorder=3)
        ax.annotate(f"{label}\n{mean_ms:.0f} ± {std_ms:.0f} ms",
                    (mean_ms, map50), textcoords="offset points",
                    xytext=(10, -20), fontsize=9, color=color)

    ax.set_xlabel("Inference time per image (ms, mean ± std)")
    ax.set_ylabel("mAP50")
    ax.set_title("Speed vs Accuracy (measured)")
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend()
    fig.tight_layout()

    out = "demo/chart_speed_vs_accuracy.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nUpdated chart saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="images per class (default: 5)")
    parser.add_argument("--warmup", type=int, default=2, help="warmup runs per model (default: 2)")
    parser.add_argument("--update-chart", action="store_true", help="overwrite demo/chart_speed_vs_accuracy.png")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Collecting up to {args.n} images per class...")
    images = collect_images(args.n)
    print(f"Total images: {len(images)}")

    from app.predictors.faster_rcnn import FasterRCNNPredictor
    from app.predictors.yolo_finetuned import YOLOFineTunedPredictor

    results = {}

    # ── Faster R-CNN ────────────────────────────────────────────────────────
    print("\nLoading Faster R-CNN...")
    rcnn = FasterRCNNPredictor()
    if rcnn._model is None:
        print("  WARNING: weights not found, skipping.")
    else:
        print(f"  Warmup ({args.warmup} runs)...")
        warmup(rcnn, images, args.warmup)
        print("  Measuring...")
        times = measure(rcnn, images)
        mean, std = print_stats("Faster R-CNN", times)
        results["faster_rcnn"] = (mean, std)

    # ── YOLOv8n fine-tuned ──────────────────────────────────────────────────
    print("\nLoading YOLOv8n fine-tuned...")
    yolo = YOLOFineTunedPredictor()
    if yolo._model is None:
        print("  WARNING: weights not found, skipping.")
    else:
        print(f"  Warmup ({args.warmup} runs)...")
        warmup(yolo, images, args.warmup)
        print("  Measuring...")
        times = measure(yolo, images)
        mean, std = print_stats("YOLOv8n fine-tuned", times)
        results["yolo_finetuned"] = (mean, std)

    if args.update_chart and len(results) == 2:
        update_chart(results)
    elif args.update_chart:
        print("\nSkipping chart update — not all models available.")

    print("\nDone.")


if __name__ == "__main__":
    main()
