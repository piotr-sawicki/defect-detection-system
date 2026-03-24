from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File
from app.schemas import PredictResponse
from app.predictor import predictor

router = APIRouter()

EXAMPLES_DIR = Path("examples/images")


@router.get("/examples")
def list_examples():
    files = sorted(f.name for f in EXAMPLES_DIR.glob("*.jpg"))
    return {"examples": files}


@router.post("/predict/example/{filename}", response_model=PredictResponse)
def predict_example(filename: str):
    file_path = EXAMPLES_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    result = predictor.predict(str(file_path))
    return PredictResponse(image_id=filename, **result)


@router.post("/predict/upload", response_model=PredictResponse)
async def predict_upload(file: UploadFile = File(...)):
    contents = await file.read()
    result = predictor.predict_bytes(contents)
    return PredictResponse(image_id=file.filename, **result)