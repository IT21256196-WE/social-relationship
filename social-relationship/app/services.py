from app.models import analyze_emotion, predict_difficulty
from app.utils import generate_dynamic_content
import logging

logging.basicConfig(level=logging.INFO)


def process_scenario(input, accuracy):
    try:
        logging.info("Analyzing emotion...")
        emotion, confidence = analyze_emotion(input.caretaker_input)
        logging.info(f"Emotion: {emotion}, Confidence: {confidence}")

        current_difficulty = input.current_difficulty
        logging.info(f"Current difficulty: {current_difficulty}")

        logging.info("Predicting new difficulty...")
        new_difficulty = predict_difficulty(
            accuracy, confidence, current_difficulty)
        logging.info(f"New difficulty: {new_difficulty}")

        logging.info("Generating dynamic content...")
        steps, story, mcqs = generate_dynamic_content(
            input.caretaker_input, new_difficulty)
        logging.info("Dynamic content generated successfully")

        return {
            "steps": steps,
            "story": story,
            "mcqs": mcqs,
            "new_difficulty": new_difficulty,
            "emotion": emotion,
            "confidence": confidence
        }
    except Exception as e:
        logging.error(f"Error in process_scenario: {e}")
        raise
