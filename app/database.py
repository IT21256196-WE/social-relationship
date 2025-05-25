from pymongo import MongoClient
from config import MONGO_URI, DB_NAME

# Initialize MongoDB client
client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# Save progress function
def save_child_progress(child_id: str, progress: dict):
    collection = db["child_progress"]
    query = {"child_id": child_id}
    update = {"$set": {"progress": progress}}
    collection.update_one(query, update, upsert=True)

# Retrieve progress
def get_child_progress(child_id: str):
    collection = db["child_progress"]
    result = collection.find_one({"child_id": child_id})
    return result.get("progress", None) if result else None