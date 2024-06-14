# trabalho-fp-2024-01
Trabalho Fábrica de Projetos 02 - 2024/01

## LLM RAG Server

Instantiate Milvus vectorstore, connect to milvus client. create a collection.

@Generate
- load ollama model
- stream detections


@load_documents
- load document
- split document based on markdown
- store with HF embeddings model

@retrieve_documents
- instantiate milvus langchain client
- as retriever + embedding
- query

https://python.langchain.com/v0.2/docs/integrations/vectorstores/milvus/ 

@rag
https://github.com/stephen37/Milvus_demo/blob/main/milvus_rag/rag_milvus_ollama.ipynb

/api/generate/

request body example
```
{
  "model": "llama3",
  "prompt": "Hello World",
  "stream": false,
  "options": {}
}
```

response
```
{
  "model": "llama3",
  "created_at": "2024-06-13 15:28:24",
  "response": "Hello World! Nice to meet you! What brings you here today?",
  "total duration": 5.958862066268921
}
```

