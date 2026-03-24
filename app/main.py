from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.routes import router

app = FastAPI(title="Defect Detection API")

app.include_router(router)

app.mount("/examples-images", StaticFiles(directory="examples/images"), name="examples-images")
app.mount("/static", StaticFiles(directory="frontend"), name="frontend")


@app.get("/")
def index():
    return FileResponse("frontend/index.html")
