def main():
    # Load embeddings model
    embeddings_model = load_embeddings_model_hf()
    
    # Load db
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings_model)

    # Load retriever
    retriever = load_retriever(db)

    # retrieve documents
    docs = retriever.invoke(prompt)

    # Use retrieved documents as context
    context = "\n\n".join([doc.page_content for doc in docs])

    # Generate prompt
    prompt_template = ChatPromptTemplate.from_template(RAG_TEMPLATE)
    rag_prompt = prompt_template.format(context=context, question=prompt)

    # Load LLM
    llm = load_llm_ollama(base_url=f"http://{OLLAMA_HOST}:{OLLAMA_PORT}")

    # Invoke LLM
    response = llm.invoke(rag_prompt)

if __name__ == "__main__":
    main()