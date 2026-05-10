"""
YOLOv8 fine-tuning script — designed to run on Kaggle GPU.

Usage (Kaggle notebook):
    !python train_yolo.py

Output:
    runs/detect/yolo_steel/weights/best.pt  — best checkpoint (use this!)
    runs/detect/yolo_steel/weights/last.pt  — last epoch checkpoint
"""

from ultralytics import YOLO

DATASET_YAML = "/kaggle/input/datasets/sawickipiotr/neu-dataset-for-yolov8/yolo_dataset/dataset.yaml"

# Starting point: yolov8n pre-trained on COCO.
# Fine-tuning reuses learned low-level features (edges, textures)
# and teaches the head to recognize steel defects specifically.
BASE_MODEL = "yolov8n.pt"

EPOCHS    = 100
IMG_SIZE  = 200   # matches our dataset (200x200 images)
BATCH     = 32    # Kaggle T4/P100 can handle 32 comfortably
PATIENCE  = 20    # early stopping: stop if no improvement for 20 epochs


def main():
    model = YOLO(BASE_MODEL)

    results = model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        patience=PATIENCE,
        device=0,               # GPU
        project="runs/detect",
        name="yolo_steel",
        exist_ok=True,          # overwrite previous run if re-running
        verbose=True,
    )

    print("\n=== Training complete ===")
    print(f"Best mAP50:    {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
    print(f"Best mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
    print("\nWeights saved to: runs/detect/yolo_steel/weights/best.pt")


if __name__ == "__main__":
    main()