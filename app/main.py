import os
from fastapi import FastAPI, HTTPException
from app.services import process_scenario
from app.database import save_child_progress
from app.models import EmotionAnalysisInput
from fastapi.middleware.cors import CORSMiddleware

os.environ["TOKENIZERS_PARALLELISM"] = "false"
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/process_scenario")
async def process_scenario_api(input: EmotionAnalysisInput, accuracy: float):
    try:
        response = process_scenario(input, accuracy)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/save_progress")
async def save_progress_api(child_id: str, progress: dict):
    save_child_progress(child_id, progress)
    return {"status": "success"}

@app.get("/")
async def root():
    return {"message": "Welcome to the Adaptive Learning API"}