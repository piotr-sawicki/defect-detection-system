"""
Task 1.2 — Generate mAP comparison charts for the presentation.
Outputs PNG files to demo/ folder.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

FASTER_RCNN = {
    "mAP50": 0.9282,
    "mAP50_95": 0.5732,
    "per_class": [0.4412, 0.5749, 0.6861, 0.5896, 0.5753, 0.5720],
}

YOLO = {
    "mAP50": 0.7261,
    "mAP50_95": 0.4167,
    "per_class": [0.1519, 0.4353, 0.5974, 0.5421, 0.2290, 0.5446],
}

COLORS = {"faster_rcnn": "#2563eb", "yolo": "#f97316"}

OUTPUT_DIR = "demo"


def save(fig, name):
    path = f"{OUTPUT_DIR}/{name}"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ── Chart 1: Per-class mAP50-95 grouped bar chart ──────────────────────────
def chart_per_class():
    x = np.arange(len(CLASSES))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width / 2, FASTER_RCNN["per_class"], width,
                   label="Faster R-CNN", color=COLORS["faster_rcnn"], zorder=3)
    bars2 = ax.bar(x + width / 2, YOLO["per_class"], width,
                   label="YOLOv8n fine-tuned", color=COLORS["yolo"], zorder=3)

    ax.set_xlabel("Defect class")
    ax.set_ylabel("mAP50-95")
    ax.set_title("Per-class mAP50-95: Faster R-CNN vs YOLOv8n fine-tuned")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=15, ha="right")
    ax.set_ylim(0, 0.85)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.legend()

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8,
                color=COLORS["faster_rcnn"])
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8,
                color=COLORS["yolo"])

    fig.tight_layout()
    save(fig, "chart_per_class_map.png")


# ── Chart 2: Overall mAP50 and mAP50-95 side-by-side ──────────────────────
def chart_overall():
    metrics = ["mAP50", "mAP50-95"]
    rcnn_vals = [FASTER_RCNN["mAP50"], FASTER_RCNN["mAP50_95"]]
    yolo_vals = [YOLO["mAP50"], YOLO["mAP50_95"]]

    x = np.arange(len(metrics))
    width = 0.3

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars1 = ax.bar(x - width / 2, rcnn_vals, width,
                   label="Faster R-CNN", color=COLORS["faster_rcnn"], zorder=3)
    bars2 = ax.bar(x + width / 2, yolo_vals, width,
                   label="YOLOv8n fine-tuned", color=COLORS["yolo"], zorder=3)

    ax.set_ylabel("Score")
    ax.set_title("Overall mAP: Faster R-CNN vs YOLOv8n fine-tuned")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.legend()

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9,
                color=COLORS["faster_rcnn"], fontweight="bold")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9,
                color=COLORS["yolo"], fontweight="bold")

    fig.tight_layout()
    save(fig, "chart_overall_map.png")


# ── Chart 3: Speed vs accuracy scatter ────────────────────────────────────
def chart_speed_vs_accuracy():
    fig, ax = plt.subplots(figsize=(6, 4.5))

    ax.scatter([200], [FASTER_RCNN["mAP50"]], s=200, color=COLORS["faster_rcnn"],
               zorder=3, label="Faster R-CNN")
    ax.scatter([5], [YOLO["mAP50"]], s=200, color=COLORS["yolo"],
               zorder=3, label="YOLOv8n fine-tuned")

    ax.annotate("Faster R-CNN\n(ResNet50)", (200, FASTER_RCNN["mAP50"]),
                textcoords="offset points", xytext=(-60, -25), fontsize=9,
                color=COLORS["faster_rcnn"])
    ax.annotate("YOLOv8n\nfine-tuned", (5, YOLO["mAP50"]),
                textcoords="offset points", xytext=(10, -25), fontsize=9,
                color=COLORS["yolo"])

    ax.set_xlabel("Inference time per image (ms)")
    ax.set_ylabel("mAP50")
    ax.set_title("Speed vs Accuracy trade-off")
    ax.set_xlim(-10, 250)
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend()

    fig.tight_layout()
    save(fig, "chart_speed_vs_accuracy.png")


if __name__ == "__main__":
    chart_per_class()
    chart_overall()
    chart_speed_vs_accuracy()
    print("Done. Charts saved to demo/")
