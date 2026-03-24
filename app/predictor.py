import io
from pathlib import Path

import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

CLASSES = [
    "__background__", "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches",
]

WEIGHTS_PATH = Path("data/weights.pth")
SCORE_THRESHOLD = 0.5


def _build_model() -> torch.nn.Module:
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(CLASSES))
    return model


def _load_model() -> torch.nn.Module | None:
    if not WEIGHTS_PATH.exists():
        return None
    model = _build_model()
    model.load_state_dict(torch.load(str(WEIGHTS_PATH), map_location="cpu"))
    model.eval()
    return model


def _run_inference(model: torch.nn.Module, img: Image.Image) -> list[dict]:
    img_tensor = F.to_tensor(img.convert("RGB"))
    with torch.no_grad():
        outputs = model([img_tensor])[0]

    boxes = outputs["boxes"]
    labels = outputs["labels"]
    scores = outputs["scores"]

    keep = scores >= SCORE_THRESHOLD
    results = []
    for box, label, score in zip(boxes[keep], labels[keep], scores[keep]):
        x1, y1, x2, y2 = box.tolist()
        results.append({
            "label": CLASSES[label.item()],
            "score": round(score.item(), 4),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        })
    return results


def _boxes_to_response(boxes: list[dict]) -> dict:
    count = len(boxes)
    avg_score = round(sum(b["score"] for b in boxes) / count, 4) if count else 0.0
    confidence = max((b["score"] for b in boxes), default=0.0)
    return {
        "defect_detected": count > 0,
        "confidence": confidence,
        "boxes": boxes,
        "count": count,
        "avg_score": avg_score,
    }


class Predictor:
    def __init__(self):
        self._model = _load_model()

    def predict(self, image_path: str) -> dict:
        if self._model is None:
            return {"defect_detected": False, "confidence": 0.0, "boxes": [], "count": 0, "avg_score": 0.0}
        img = Image.open(image_path)
        boxes = _run_inference(self._model, img)
        return _boxes_to_response(boxes)

    def predict_bytes(self, image_bytes: bytes) -> dict:
        if self._model is None:
            return {"defect_detected": False, "confidence": 0.0, "boxes": [], "count": 0, "avg_score": 0.0}
        img = Image.open(io.BytesIO(image_bytes))
        boxes = _run_inference(self._model, img)
        return _boxes_to_response(boxes)


predictor = Predictor()