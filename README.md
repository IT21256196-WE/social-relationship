# Social Relationship Project

This project aims to analyze children's emotions and predict the difficulty of social scenarios using machine learning models. It also generates dynamic content such as images, stories, and multiple-choice questions to help children understand social scenarios.

## Prerequisites

- Python 3.9 or higher
- Virtual environment (venv)
- MongoDB

## Setup

1. **Clone the repository:**

    ```sh
    git clone https://github.com/yourusername/social-relationship.git
    cd social-relationship
    ```

2. **Create and activate a virtual environment:**

    ```sh
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3. **Install the dependencies:**

    ```sh
    pip install -r requirements.txt
    ```

4. **Set up environment variables:**

    Create a `.env` file in the root directory and add the following variables:

    ```properties
    OPENAI_API_KEY=your_openai_api_key
    MONGO_URI=your_mongo_uri
    DB_NAME=child_learning
    PYTHONPATH=.
    TOKENIZERS_PARALLELISM=false
    ```

5. **Generate the TF-IDF vectorizer:**

    Run the `generate_vectorizer.py` script to create and save the TF-IDF vectorizer:

    ```sh
    python generate_vectorizer.py
    ```

6. **Train the difficulty prediction model:**

    Run the `train_difficulty.py` script to train and save the difficulty prediction model:

    ```sh
    python train_difficulty.py
    ```

## Running the Application

1. **Start the FastAPI server:**

    ```sh
    uvicorn app.main:app --reload
    ```

2. **Access the API:**

    The API will be available at `http://127.0.0.1:8000`.

## API Endpoints

- **POST /process_scenario**

    Analyze a social scenario and generate dynamic content.

    **Request:**

    ```json
    {
        "caretaker_input": "The child is happy and playing with friends.",
        "current_difficulty": 2.5
    }
    ```

    **Query Parameter:**

    - [accuracy](http://_vscodecontentref_/0): float

    **Response:**

    ```json
    {
        "steps": ["image_url1", "image_url2", "image_url3", "image_url4"],
        "story": "Generated story",
        "mcqs": [
            {"question": "Question 1", "options": ["Option A", "Option B"]},
            {"question": "Question 2", "options": ["Option A", "Option B"]}
        ],
        "new_difficulty": 3,
        "emotion": "happy",
        "confidence": 0.95
    }
    ```

- **POST /save_progress**

    Save the progress of a child.

    **Request:**

    ```json
    {
        "child_id": "child123",
        "progress": {
            "scenario": "How to cross the road safely",
            "difficulty": 3
        }
    }
    ```

    **Response:**

    ```json
    {
        "status": "success"
    }
    ```

## Project Structure

- [app](http://_vscodecontentref_/1): Contains the main application code.
- [models](http://_vscodecontentref_/2): Contains the machine learning models.
- [generate_vectorizer.py](http://_vscodecontentref_/3): Script to generate and save the TF-IDF vectorizer.
- [train_difficulty.py](http://_vscodecontentref_/4): Script to train and save the difficulty prediction model.
- [requirements.txt](http://_vscodecontentref_/5): List of dependencies.
- [.env](http://_vscodecontentref_/6): Environment variables.

## License

This project is licensed under the MIT License.