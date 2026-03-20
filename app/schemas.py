from pydantic import BaseModel


class PredictResponse(BaseModel):
    image_id: str
    defect_detected: bool
    confidence: float