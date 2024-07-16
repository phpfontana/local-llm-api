# Local LLM API

Welcome to the Local LLM API! This is a production-ready API server designed for running local large language models (LLMs) using:

- **LlamaCPP**
- **Langchain**
- **FastAPI**

## Features

- **Chat Completion**: Generate responses based on the conversation history.
- **Embeddings Generation**: Generate embeddings for text inputs.

## Endpoints

### Chat Completion

`POST /api/chat`

**Parameters**

- `model`: Specify the model to use.
- `messages`: An array of message objects.

**Example**

**Request**
```bash
curl -X 'POST' \
  'http://localhost/api/chat/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "model": "Meta-Llama-3-8B-Instruct-Q8_0.gguf",
  "messages": [
    {
      "role": "system",
      "content": "You are a chat assistant, always answer briefly and informatively, use the chat history to generate the responses."
    },
    {
      "role": "user",
      "content": "My name is John"
    },
    {
      "role": "assistant",
      "content": "Hello John, how can I help you today?"
    },
    {
      "role": "user",
      "content": "What is my name?"
    }
  ],
  "stream": false
}'
```

**Response**
```json
{
  "message": {
    "role": "assistant",
    "content": "Your name is John!"
  },
  "done": true
}
```

### Embeddings Generation

`POST /api/embeddings`

**Parameters**

- `model`: Specify the model to use for generating embeddings.
- `query`: The text to generate embeddings for.

**Example**

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

## Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/phpfontana/local-llm-api.git
   cd local-llm-api
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the server**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

4. **Access the API**
   Open your browser and go to `http://localhost:8000/docs` to see the interactive API documentation.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


---

Enjoy using the Local LLM API!