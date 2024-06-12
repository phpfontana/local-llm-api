# trabalho-fp-2024-01
Trabalho Fábrica de Projetos 02 - 2024/01

## LLM RAG Server

Instantiate Milvus vectorstore, connect to milvus client. create a collection.


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
