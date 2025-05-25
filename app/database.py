from pymongo import MongoClient
from config import MONGO_URI, DB_NAME

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db["children"]


def save_child(child_id: str, difficulty: float):
    collection.update_one(
        {"child_id": child_id},
        {"$set": {"difficulty": difficulty}},
        upsert=True
    )


def get_child(child_id: str):
    return collection.find_one({"child_id": child_id}, {"_id": 0})


def update_difficulty(child_id: str, difficulty: float):
    collection.update_one(
        {"child_id": child_id},
        {"$set": {"difficulty": difficulty}}
    )
