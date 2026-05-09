from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File
from app.schemas import PredictResponse
from app.predictors import get_predictor

router = APIRouter()

EXAMPLES_DIR = Path("examples/images")


@router.get("/examples")
def list_examples():
    files = sorted(f.name for f in EXAMPLES_DIR.glob("*.jpg"))
    return {"examples": files}


@router.post("/predict/example/{filename}", response_model=PredictResponse)
def predict_example(filename: str, threshold: float = 0.5, model: str = "faster_rcnn"):
    file_path = EXAMPLES_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    predictor = get_predictor(model)
    result = predictor.predict(str(file_path), threshold)
    return PredictResponse(image_id=filename, **result)


@router.post("/predict/upload", response_model=PredictResponse)
async def predict_upload(file: UploadFile = File(...), threshold: float = 0.5, model: str = "faster_rcnn"):
    predictor = get_predictor(model)
    contents = await file.read()
    result = predictor.predict_bytes(contents, threshold)
    return PredictResponse(image_id=file.filename, **result)
