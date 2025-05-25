import random
import openai
from config import OPENAI_API_KEY

# OpenAI API setup
openai.api_key = OPENAI_API_KEY

# Generate dynamic content
def generate_dynamic_content(scenario, difficulty):
    # Generate steps using DALL-E
    image_urls = []
    for step in range(1,5):
        dalle_prompt = f"Think you are an visual intepreter to a kid to tell how a social scenario goes and that scenario have four steps. And remember when you generating this images there are difficulty level of such children. Create an image for the scenario '{scenario}' at difficulty level {difficulty}, step {step}."
        dalle_response = openai.images.generate(
            model="dall-e-3",
            prompt=dalle_prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        print(dalle_response)
        image_urls.append(dalle_response.data[0].url)

    # Generate story using GPT-4
    story_prompt = f"Write a simple story for a child on the topic '{scenario}' at difficulty level {difficulty}."
    story = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": story_prompt}
        ],
        max_tokens=150
    ).choices[0].message.content.strip()

    # Generate MCQs using GPT-4
    mcq_prompt = f"Create difficulty level of {difficulty} and multiple-choice questions (with 2 options) for a story about '{scenario}'."
    mcqs = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": mcq_prompt}
        ],
        max_tokens=200
    ).choices[0].message.content.strip()

    return image_urls, story, mcqs

# Load vectorizer for processing
def load_vectorizer():
    import pickle
    with open("ml/models/vectorizer.pkl", "rb") as f:
        return pickle.load(f)

# Load PyTorch model
def load_difficulty_model():
    import torch
    from app.models import DifficultyRegressor
    model = DifficultyRegressor()
    model.load_state_dict(torch.load("ml/models/difficulty_model.pth"))
    model.eval()
    return model
