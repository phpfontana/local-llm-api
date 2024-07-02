# Local Llama3 Server

This is a production-ready local Llama3 server made with:

* Ollama
* Langchain
* FastAPI
* MongoDB

for a demo usage, go to tests/chat.py

## Generate Response
```
POST /api/generate
```

**Parameters**
* `model`: Specify the model to use.
* `prompt`: The prompt or question to generate a response for.

### Example
**Request**
```bash
curl -X 'POST' \
  'http://localhost/api/generate/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "llama3:instruct",
  "prompt": "What is the capital of France?"
}'
```

**Response**
```json
{
  "response": "The capital of France is Paris."
}
```

## Chat Completion

```
POST /api/chat
```

**Parameters**
* `model`: Specify the model to use.
* `prompt`: The initial prompt or question to start the chat.
* `messages`: An array of message objects containing previous interactions.

### Example
**Request**
```bash
curl -X 'POST' \
  'http://localhost/api/chat/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "llama3:instruct",
  "prompt": "What is your name?",
  "messages": [
    {
        "role": "ai",
        "content": "Hello! How can I help you today?"
    },
    {
        "role": "user",
        "content": "Hello, My name is John Doe"
    },
    {
        "role": "ai",
        "content": "Hello, John Doe! How can I help you today?"
    }
]
}'
```

**Response**
```json
{
  "message": {
    "role": "system",
    "content": "Nice to meet you, John! My name is LoLLa3, Local LLama3 Assistant. I'm a chat assistant designed to assist and provide information on a wide range of topics. I'm here to help answer any questions or concerns you may have, so feel free to ask me anything!"
  }
}
```

## Embeddings Generation

```
POST /api/embeddings
```

**Parameters**
* `model`: Specify the model to use for generating embeddings.
* `query`: The text to generate embeddings for.

### Example
**Request**
```bash
curl -X 'POST' \
  'http://localhost/api/embeddings/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "embedding-model",
  "query": "Hello world"
}'
```

**Response**
```json
{
  "embeddings": [0.123, 0.456, 0.789, ...]
}
```