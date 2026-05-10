# Model Evaluation Results

Evaluated on validation set (360 images, 6 classes).
Metric: **mAP50-95 per class**, **mAP50 overall** — computed with torchmetrics, threshold=0.001.

---

## YOLOv8 fine-tuned

| Metric | Value |
|---|---|
| mAP50 | **0.7261** |
| mAP50-95 | 0.4167 |

| Class | mAP50-95 |
|---|---|
| crazing | 0.1519 |
| inclusion | 0.4353 |
| patches | 0.5974 |
| pitted_surface | 0.5421 |
| rolled-in_scale | 0.2290 |
| scratches | 0.5446 |

---

## Faster R-CNN

| Metric | Value |
|---|---|
| mAP50 | **0.9282** |
| mAP50-95 | 0.5732 |

| Class | mAP50-95 |
|---|---|
| crazing | 0.4412 |
| inclusion | 0.5749 |
| patches | 0.6861 |
| pitted_surface | 0.5896 |
| rolled-in_scale | 0.5753 |
| scratches | 0.5720 |

---

## Comparison

| Class | Faster R-CNN | YOLOv8 fine-tuned | Winner |
|---|---|---|---|
| **mAP50 (overall)** | **0.9282** | 0.7261 | Faster R-CNN |
| crazing | **0.4412** | 0.1519 | Faster R-CNN |
| inclusion | **0.5749** | 0.4353 | Faster R-CNN |
| patches | **0.6861** | 0.5974 | Faster R-CNN |
| pitted_surface | **0.5896** | 0.5421 | Faster R-CNN |
| rolled-in_scale | **0.5753** | 0.2290 | Faster R-CNN |
| scratches | **0.5720** | 0.5446 | Faster R-CNN |
