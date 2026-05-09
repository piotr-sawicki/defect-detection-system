from app.predictors.base import BasePredictor
from app.predictors.faster_rcnn import FasterRCNNPredictor
from app.predictors.yolo import YOLOPredictor

_instances: dict[str, BasePredictor] = {}


def get_predictor(model_name: str) -> BasePredictor:
    if model_name not in _instances:
        if model_name == "faster_rcnn":
            _instances[model_name] = FasterRCNNPredictor()
        elif model_name == "yolo":
            _instances[model_name] = YOLOPredictor()
        else:
            raise ValueError(f"Unknown model: {model_name!r}")
    return _instances[model_name]
