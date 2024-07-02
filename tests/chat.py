from pymongo import MongoClient
import requests
import json

# Database connection
def connect_to_mongo(uri, db_name, collection_name, username, password):
    client = MongoClient(uri, username=username, password=password)
    db = client[db_name]
    collection = db[collection_name]
    return collection

# Fetch messages from the database
def fetch_messages(collection, session_id):
    messages = []
    for message in collection.find({"session_id": session_id}):
        role = message["role"]
        content = message["content"]
        messages.append({"role": role, "content": content})
    return messages

# Send POST request to chat API
def send_chat_request(url, headers, data):
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response

# Main chat function
def chat_with_ai():
    # Database configuration
    uri = "mongodb://localhost:27017/"
    db_name = "chatdb"
    collection_name = "chat_messages"
    username = "admin"
    password = "root"
    session_id = "8bafb4b4-5306-4105-b237-94f7a99f48de"

    # Connect to the database
    collection = connect_to_mongo(uri, db_name, collection_name, username, password)

    # API configuration
    url = "http://localhost:80/api/chat"
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
    }

    # Fetch previous messages
    messages = fetch_messages(collection, session_id)

    while True:
        # Get user input
        prompt = input("You: ")
        if prompt.lower() == "/exit":
            break

        # Prepare data for the API request
        data = {
            "model": "llama3:instruct",
            "prompt": prompt,
            "messages": messages
        }

        # Send the request
        response = send_chat_request(url, headers, data)
        if response.status_code == 200:
            out = response.json()
            role = out["message"]["role"]
            content = out["message"]["content"]

            # Print the response
            print(f"{role}: {content}")

            # Update messages
            messages.append({"role": "user", "content": prompt})
            messages.append({"role": role, "content": content})

            # Store the messages in the database
            collection.insert_one({"session_id": session_id, "role": "user", "content": prompt})
            collection.insert_one({"session_id": session_id, "role": role, "content": content})
        else:
            print(f"Error: {response.status_code}")
            print(response.text)

if __name__ == "__main__":
    chat_with_ai()
