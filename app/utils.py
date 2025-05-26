import json
import openai
from config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY


def generate_dynamic_content(scenario, difficulty):
    # Improved DALL-E prompt
    image_urls = []
    for step in range(1, 5):
        dalle_prompt = (
            f"You are an expert visual storyteller for children. "
            f"Create a highly detailed, child-friendly illustration for the scenario: '{scenario}'. "
            f"This scenario should be broken into four logical steps, each step representing a key moment in the scenario. "
            f"Generate the image for step {step} at difficulty level {difficulty}. "
            f"Consider the cognitive and emotional needs of children at this difficulty level. "
            f"Make the image engaging, clear, and supportive of learning. "
            f"Include context, characters, and actions relevant to the scenario and step."
        )
        dalle_response = openai.images.generate(
            model="dall-e-3",
            prompt=dalle_prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_urls.append(dalle_response.data[0].url)

    # Improved story prompt
    story_prompt = (
        f"You are a master children's storyteller and educator. "
        f"Write a vivid, engaging, and age-appropriate story for a child about the scenario '{scenario}'. "
        f"Adapt the language, complexity, and emotional tone to difficulty level {difficulty}. "
        f"Ensure the story is instructive, supportive, and helps the child understand the scenario step by step."
    )
    story = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": story_prompt}
        ],
        max_tokens=400
    ).choices[0].message.content.strip()

    # Improved MCQ prompt
    mcq_prompt = (
        f"You are an expert in child education. "
        f"Create 5 multiple-choice questions for a story about '{scenario}'. "
        f"Each question should be simple, direct, and suitable for children, with only four answer options. "
        f"Return the questions and answers as a JSON array of objects, using double quotes for all keys and string values. "
        f"Each object must contain: "
        f'"question" (string), "options" (a list of two strings), and "answer" (the correct option, as a single word from the options). '
        f"Do not include explanations, code blocks, or extra formatting. Only output the JSON array."
        f' Example: [{{"question": "What color is the sky?", "options": ["blue", "green", "yellow", "red"], "answer": "blue"}}]'
    )
    mcqs_response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": mcq_prompt}
        ],
        max_tokens=400
    ).choices[0].message.content.strip()

    try:
        mcqs = json.loads(mcqs_response)
    except Exception:
        mcqs = mcqs_response

    return image_urls, story, mcqs


def generate_asd_visual_steps(scenario, difficulty=1):
    image_urls = []
    for step in range(1, 5):
        dalle_prompt = (
            f"You are a world-class visual educator for children with autism spectrum disorder (ASD). "
            f"Your task is to create a clear, simple, and visually structured illustration for the scenario: '{scenario}'. "
            f"Break the scenario into four logical, easy-to-understand steps. "
            f"This is step {step}. "
            f"Use minimal distractions, clear backgrounds, and focus on the main action. "
            f"Characters should have clear facial expressions and body language. "
            f"Use high contrast and bright, friendly colors. "
            f"Make the image supportive for children with ASD at difficulty level {difficulty}."
        )
        dalle_response = openai.images.generate(
            model="dall-e-3",
            prompt=dalle_prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_urls.append(dalle_response.data[0].url)
    return image_urls


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
