from fastapi import FastAPI
from app.routes import router

app = FastAPI(title="Defect Detection API")

app.include_router(router)