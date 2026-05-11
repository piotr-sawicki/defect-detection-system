"""
Generate mAP comparison charts from eval_results.csv.

Reads the most recent result per model from the CSV.

Usage:
    python scripts/plot_eval_results.py
    python scripts/plot_eval_results.py --csv results/eval_results.csv --out demo/
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

COLORS = [
    "#2563eb",  # blue  — Faster R-CNN
    "#f97316",  # orange — YOLOv8n
    "#a855f7",  # purple — YOLOv8s
    "#10b981",  # green  — extra models
    "#ef4444",
    "#eab308",
]


def load_latest(csv_path: Path) -> dict[str, dict]:
    """Return the most recent row per model name."""
    rows: dict[str, dict] = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows[row["model"]] = {k: float(v) for k, v in row.items()
                                  if k not in ("timestamp", "model")}
    return rows


def save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ── Chart 1: Overall mAP50 and mAP50-95 ──────────────────────────────────────

def chart_overall(data: dict[str, dict], out_dir: Path):
    model_names = list(data.keys())
    map50    = [data[m]["mAP50"]    for m in model_names]
    map50_95 = [data[m]["mAP50-95"] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(model_names) * 2), 4.5))
    bars1 = ax.bar(x - width / 2, map50,    width, label="mAP50",    zorder=3)
    bars2 = ax.bar(x + width / 2, map50_95, width, label="mAP50-95", zorder=3)

    for bars in (bars1, bars2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                    f"{bar.get_height():.3f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold")

    ax.set_ylabel("Score")
    ax.set_title("Overall mAP comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=10, ha="right")
    ax.set_ylim(0, 1.1)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.legend()
    fig.tight_layout()
    save(fig, out_dir, "chart_overall_map.png")


# ── Chart 2: Per-class mAP50-95 grouped bars ─────────────────────────────────

def chart_per_class(data: dict[str, dict], out_dir: Path):
    model_names = list(data.keys())
    n_models = len(model_names)
    x = np.arange(len(CLASSES))
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(11, 5))

    for i, (name, color) in enumerate(zip(model_names, COLORS)):
        vals = [data[name].get(f"mAP50-95_{cls}", 0.0) for cls in CLASSES]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=name, color=color, zorder=3)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{bar.get_height():.3f}", ha="center", va="bottom",
                    fontsize=7, color=color)

    ax.set_xlabel("Defect class")
    ax.set_ylabel("mAP50-95")
    ax.set_title("Per-class mAP50-95 comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=15, ha="right")
    ax.set_ylim(0, 0.9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.legend()
    fig.tight_layout()
    save(fig, out_dir, "chart_per_class_map.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("results/eval_results.csv"))
    parser.add_argument("--out", type=Path, default=Path("demo"))
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"CSV not found: {args.csv}")
        print("Run evaluate_models.py first.")
        return

    data = load_latest(args.csv)
    print(f"Loaded results for: {list(data.keys())}")

    chart_overall(data, args.out)
    chart_per_class(data, args.out)
    print("Done.")


if __name__ == "__main__":
    main()
