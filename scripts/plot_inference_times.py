"""
Generate speed vs accuracy chart from inference_times.csv and eval_results.csv.

Usage:
    python scripts/plot_inference_times.py
    python scripts/plot_inference_times.py --times results/inference_times.csv --eval results/eval_results.csv --out demo/
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {
    "Faster R-CNN":       "#2563eb",
    "YOLOv8n fine-tuned": "#f97316",
    "YOLOv8s fine-tuned": "#a855f7",
}
DEFAULT_COLOR = "#888888"


def load_latest(csv_path: Path, key_col: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows[row[key_col]] = {k: v for k, v in row.items() if k != key_col}
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--times", type=Path, default=Path("results/inference_times.csv"))
    parser.add_argument("--eval",  type=Path, default=Path("results/eval_results.csv"))
    parser.add_argument("--out",   type=Path, default=Path("demo"))
    args = parser.parse_args()

    for p in (args.times, args.eval):
        if not p.exists():
            print(f"File not found: {p}")
            print("Run measure_inference_time.py and evaluate_models.py first.")
            return

    times = load_latest(args.times, "model")
    evals = load_latest(args.eval,  "model")

    fig, ax = plt.subplots(figsize=(8, 5))

    for name, t in times.items():
        map50 = evals.get(name, {}).get("mAP50")
        if map50 is None:
            print(f"  No mAP50 found for '{name}', skipping.")
            continue
        map50    = float(map50)
        mean_ms  = float(t["mean_ms"])
        std_ms   = float(t["std_ms"])
        color    = COLORS.get(name, DEFAULT_COLOR)

        ax.errorbar(mean_ms, map50, xerr=std_ms, fmt="o", ms=10,
                    color=color, label=name, capsize=5, zorder=3)
        ax.annotate(f"{name}\n{mean_ms:.0f} ± {std_ms:.0f} ms",
                    (mean_ms, map50), textcoords="offset points",
                    xytext=(10, -20), fontsize=9, color=color)

    ax.set_xlabel("Inference time per image (ms, mean ± std)")
    ax.set_ylabel("mAP50")
    ax.set_title("Speed vs Accuracy")
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend()
    fig.tight_layout()

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / "chart_speed_vs_accuracy.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
