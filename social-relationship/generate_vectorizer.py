from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Sample data
corpus = [
    "How to cross the road safely",
    "How to behave in a classroom",
    "How to greet people politely",
]

# Train the TF-IDF vectorizer
vectorizer = TfidfVectorizer()
vectorizer.fit(corpus)

# Save the vectorizer
with open("ml/models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("TF-IDF Vectorizer saved!")
