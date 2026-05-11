from app.predictors.base import BasePredictor
from app.predictors.faster_rcnn import FasterRCNNPredictor
from app.predictors.yolo import YOLOPredictor
from app.predictors.yolo_finetuned import YOLOFineTunedPredictor
from app.predictors.yolo_s_finetuned import YOLOSFineTunedPredictor

_instances: dict[str, BasePredictor] = {}

_REGISTRY = {
    "faster_rcnn": FasterRCNNPredictor,
    "yolo":        YOLOPredictor,
    "yolo_n_ft":   YOLOFineTunedPredictor,
    "yolo_s_ft":   YOLOSFineTunedPredictor,
}


def get_predictor(model_name: str) -> BasePredictor:
    if model_name not in _REGISTRY:
        raise ValueError(f"Unknown model: {model_name!r}. Available: {list(_REGISTRY)}")
    if model_name not in _instances:
        _instances[model_name] = _REGISTRY[model_name]()
    return _instances[model_name]
