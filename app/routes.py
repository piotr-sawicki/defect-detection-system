from fastapi import APIRouter
from app.schemas import PredictResponse
from app.predictor import predictor

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
def predict(image_id: str):
    result = predictor.predict(image_id)
    return PredictResponse(image_id=image_id, **result)