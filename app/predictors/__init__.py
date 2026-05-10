from app.predictors.base import BasePredictor
from app.predictors.faster_rcnn import FasterRCNNPredictor
from app.predictors.yolo import YOLOPredictor
from app.predictors.yolo_finetuned import YOLOFineTunedPredictor

_instances: dict[str, BasePredictor] = {}


def get_predictor(model_name: str) -> BasePredictor:
    if model_name not in _instances:
        if model_name == "faster_rcnn":
            _instances[model_name] = FasterRCNNPredictor()
        elif model_name == "yolo":
            _instances[model_name] = YOLOPredictor()
        elif model_name == "yolo_finetuned":
            _instances[model_name] = YOLOFineTunedPredictor()
        else:
            raise ValueError(f"Unknown model: {model_name!r}")
    return _instances[model_name]
