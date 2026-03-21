from pathlib import Path
from fastapi import APIRouter, HTTPException
from app.schemas import PredictResponse
from app.predictor import predictor

router = APIRouter()

EXAMPLES_DIR = Path("examples")


@router.get("/examples")
def list_examples():
    files = [f.name for f in EXAMPLES_DIR.iterdir() if f.is_file() and not f.name.startswith(".")]
    return {"examples": files}


@router.post("/predict/example/{filename}", response_model=PredictResponse)
def predict_example(filename: str):
    file_path = EXAMPLES_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    result = predictor.predict(str(file_path))
    return PredictResponse(image_id=filename, **result)
