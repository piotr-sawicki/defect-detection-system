import io
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

from app.predictors.base import BasePredictor, boxes_to_response

WEIGHTS_PATH = Path("data/yolo8n_aug.pt")

_EMPTY_RESULT = {"defect_detected": False, "confidence": 0.0, "boxes": [], "count": 0, "avg_score": 0.0}


class YOLONAugPredictor(BasePredictor):
    def __init__(self):
        if not WEIGHTS_PATH.exists():
            print(f"WARNING: weights not found at {WEIGHTS_PATH}")
            self._model = None
        else:
            self._model = YOLO(str(WEIGHTS_PATH))

    def predict(self, image_path: str, threshold: float) -> dict:
        if self._model is None:
            return _EMPTY_RESULT
        results = self._model(image_path, conf=threshold, verbose=False)[0]
        return self._parse_results(results)

    def predict_bytes(self, image_bytes: bytes, threshold: float) -> dict:
        if self._model is None:
            return _EMPTY_RESULT
        img = Image.open(io.BytesIO(image_bytes))
        results = self._model(img, conf=threshold, verbose=False)[0]
        return self._parse_results(results)

    def _parse_results(self, results) -> dict:
        boxes = []
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            label = results.names[int(box.cls[0])]
            score = round(float(box.conf[0]), 4)
            boxes.append({"label": label, "score": score,
                          "x1": x1, "y1": y1, "x2": x2, "y2": y2})
        return boxes_to_response(boxes)
