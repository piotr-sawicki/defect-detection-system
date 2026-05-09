import io

from PIL import Image
from ultralytics import YOLO

from app.predictors.base import BasePredictor, boxes_to_response


class YOLOPredictor(BasePredictor):
    def __init__(self):
        # yolov8n = nano variant — fastest, smallest; pre-trained on COCO (80 classes)
        self._model = YOLO("yolov8n.pt")

    def predict(self, image_path: str, threshold: float) -> dict:
        results = self._model(image_path, conf=threshold, verbose=False)[0]
        return self._parse_results(results)

    def predict_bytes(self, image_bytes: bytes, threshold: float) -> dict:
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
