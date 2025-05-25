import os
from fastapi import FastAPI, HTTPException
from app.services import process_scenario, predict_and_update_difficulty
from app.database import save_child, get_child, update_difficulty
from app.models import EmotionAnalysisInput, CreateChildInput, PredictDifficultyInput
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
async def process_scenario_api(input: EmotionAnalysisInput, accuracy: float, child_id: str):
    try:
        update_difficulty(child_id, input.current_difficulty)
        response = process_scenario(input, accuracy)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/create_child")
async def create_child_api(input: CreateChildInput):
    try:
        save_child(input.child_id, input.difficulty)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/get_child/{child_id}")
async def get_child_api(child_id: str):
    child = get_child(child_id)
    if not child:
        raise HTTPException(status_code=404, detail="Child not found")
    return child


@app.post("/predict_and_update_difficulty")
async def predict_and_update_difficulty_api(input: PredictDifficultyInput):
    try:
        new_difficulty = predict_and_update_difficulty(
            input.child_id, input.caretaker_input, input.accuracy)
        return {"child_id": input.child_id, "new_difficulty": new_difficulty}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {"message": "Welcome to the Adaptive Learning API"}
