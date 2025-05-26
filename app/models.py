from transformers import pipeline
import torch
import torch.nn as nn
from pydantic import BaseModel

# Emotion Detection Model
emotion_model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

def analyze_emotion(text: str):
    results = emotion_model(text)
    return results[0]['label'], results[0]['score']


# Difficulty Prediction Model
class DifficultyRegressor(nn.Module):
    def __init__(self):
        super(DifficultyRegressor, self).__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class EmotionAnalysisInput(BaseModel):
    caretaker_input: str
    current_difficulty: float

class CreateChildInput(BaseModel):
    child_id: str
    difficulty: float

class PredictDifficultyInput(BaseModel):
    child_id: str
    caretaker_input: str
    accuracy: float

class ScenarioInput(BaseModel):
    scenario: str
    difficulty: float = 1

# Load the PyTorch model
model_path = "ml/models/difficulty_model.pth"
difficulty_model = DifficultyRegressor()
difficulty_model.load_state_dict(torch.load(model_path))
difficulty_model.eval()

def predict_difficulty(accuracy, emotion_score, current_difficulty):
    input_tensor = torch.tensor([[accuracy, emotion_score, current_difficulty]], dtype=torch.float32)
    return round(difficulty_model(input_tensor).item())
