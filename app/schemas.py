from pydantic import BaseModel


class Box(BaseModel):
    label: str
    score: float
    x1: float
    y1: float
    x2: float
    y2: float


class PredictResponse(BaseModel):
    image_id: str
    defect_detected: bool
    confidence: float
    boxes: list[Box]
    count: int
    avg_score: float