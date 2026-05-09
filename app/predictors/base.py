from abc import ABC, abstractmethod


def boxes_to_response(boxes: list[dict]) -> dict:
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


class BasePredictor(ABC):
    @abstractmethod
    def predict(self, image_path: str, threshold: float) -> dict:
        ...

    @abstractmethod
    def predict_bytes(self, image_bytes: bytes, threshold: float) -> dict:
        ...
