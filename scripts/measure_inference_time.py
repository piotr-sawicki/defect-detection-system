"""
Measure inference times for all models and save results to CSV.

Usage:
    python scripts/measure_inference_time.py
    python scripts/measure_inference_time.py --n 10 --warmup 5
    python scripts/measure_inference_time.py --out results/inference_times.csv
"""

import argparse
import csv
import random
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
EXAMPLES_DIR = Path("examples/images")
DATA_DIR = Path("data")


def collect_images(n: int) -> list[Path]:
    images = []
    for cls in CLASSES:
        candidates = (
            list((DATA_DIR / cls).glob("*.jpg"))
            + list((DATA_DIR / "images").glob(f"{cls}_*.jpg"))
            + list(DATA_DIR.glob(f"{cls}_*.jpg"))
        )
        if candidates:
            chosen = random.sample(candidates, min(n, len(candidates)))
        else:
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


def print_stats(name: str, times: list[float]) -> dict:
    arr = np.array(times)
    print(f"\n{name}")
    print(f"  n={len(arr)}  mean={arr.mean():.1f} ms  std={arr.std():.1f} ms"
          f"  min={arr.min():.1f}  max={arr.max():.1f}")
    return {
        "mean_ms": round(float(arr.mean()), 1),
        "std_ms":  round(float(arr.std()),  1),
        "min_ms":  round(float(arr.min()),  1),
        "max_ms":  round(float(arr.max()),  1),
        "n":       len(arr),
    }


def save_csv(results: dict[str, dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = [{"timestamp": ts, "model": name, **stats} for name, stats in results.items()]
    fieldnames = list(rows[0].keys())
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults appended to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",      type=int, default=5,  help="images per class (default: 5)")
    parser.add_argument("--warmup", type=int, default=2,  help="warmup runs per model (default: 2)")
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--out", type=Path,
                        default=Path("results/inference_times.csv"),
                        help="output CSV path (default: results/inference_times.csv)")
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Collecting up to {args.n} images per class...")
    images = collect_images(args.n)
    print(f"Total images: {len(images)}")

    from app.predictors.faster_rcnn import FasterRCNNPredictor
    from app.predictors.yolo_finetuned import YOLOFineTunedPredictor
    from app.predictors.yolo_s_finetuned import YOLOSFineTunedPredictor

    models_to_run = [
        ("YOLOv8s fine-tuned", YOLOSFineTunedPredictor),
        ("Faster R-CNN",       FasterRCNNPredictor),
        ("YOLOv8n fine-tuned", YOLOFineTunedPredictor),
    ]

    results = {}
    for name, cls in models_to_run:
        print(f"\nLoading {name}...")
        predictor = cls()
        if predictor._model is None:
            print("  WARNING: weights not found, skipping.")
            continue
        print(f"  Warmup ({args.warmup} runs)...")
        warmup(predictor, images, args.warmup)
        print("  Measuring...")
        times = measure(predictor, images)
        results[name] = print_stats(name, times)

    if results:
        save_csv(results, args.out)

    print("\nDone.")


if __name__ == "__main__":
    main()
