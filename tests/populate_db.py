from pymongo import MongoClient
from uuid import uuid4

client = MongoClient("mongodb://localhost:27017/", username="admin", password="root")
db = client["chatdb"]
collection = db["chat_messages"]
session_id = str(uuid4())

chat_messages = [
    {
        "session_id": session_id,
        "role": "ai",
        "content": "Hello! How can I help you today?"
    },
    {
        "session_id": session_id,
        "role": "user",
        "content": "Hello, My name is John Doe"
    },
    {
        "session_id": session_id,
        "role": "ai",
        "content": "Hello, John Doe! How can I help you today?"
    }
]


collection.insert_many(chat_messages)