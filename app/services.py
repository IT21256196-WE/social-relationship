from app.models import analyze_emotion, predict_difficulty
from app.utils import generate_dynamic_content
from app.database import update_difficulty, get_child
import logging

logging.basicConfig(level=logging.INFO)


def process_scenario(input, accuracy):
    emotion, confidence = analyze_emotion(input.caretaker_input)
    new_difficulty = predict_difficulty(
        accuracy, confidence, input.current_difficulty)
    steps, story, mcqs = generate_dynamic_content(
        input.caretaker_input, new_difficulty)
    return {
        "steps": steps,
        "story": story,
        "mcqs": mcqs,
        "new_difficulty": new_difficulty,
        "emotion": emotion,
        "confidence": confidence
    }


def predict_and_update_difficulty(child_id, caretaker_input, accuracy):
    child = get_child(child_id)
    if not child:
        raise Exception("Child not found")
    current_difficulty = child["difficulty"]
    emotion, confidence = analyze_emotion(caretaker_input)
    new_difficulty = predict_difficulty(
        accuracy, confidence, current_difficulty)
    update_difficulty(child_id, new_difficulty)
    return new_difficulty
