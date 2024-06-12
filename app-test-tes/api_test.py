from fastapi import FastAPI
from dotenv import load_dotenv
from app.utils import *

from langchain_milvus.vectorstores import Milvus


def main():

    # Load environment variables
    EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
    EMBEDDING_MODEL_KWARGS = {'device': 'cpu'}
    EMBEDDING_ENCODE_KWARGS = {'normalize_embeddings': False} 

    CONNECTION_ARGS = {
        'uri': './data/fp-2024-01.db',
        'collection': 'documents'
        }
    
    MILVUS_URI = './data/fp-2024-01.db'
    MILVUS_COLLECTIONS = {
        'document_collection': 'documents',
        'questions_collection' : 'questions'
    }

    # Initialize FastAPI app
    app = FastAPI()

    @app.get("/")
    async def root():
        return {"message": "Hello World"}


    @app.post("/load_documents")
    async def store_documents(path: str):
        """
        Load documents to database.
        """

        try:
            # Load embeddings model
            embeddings_model = get_embeddings_model(
                model_name=EMBEDDING_MODEL, model_kwargs=EMBEDDING_MODEL_KWARGS, encode_kwargs=EMBEDDING_ENCODE_KWARGS
                )

            # Load documents
            documents = load_documents(path)

            # Split documents
            splits = split_documents(
                documents=documents, chunk_size=1000, chunk_overlap=200
                )
            
            # Load documents to vectorstore
            vectorstore = Milvus.from_documents(
                documents=splits, 
                embedding=embeddings_model, 
                collection_name=MILVUS_COLLECTIONS.get('document_collection'),
                connection_args={'uri': MILVUS_URI}
            )

        except:
            pass
        
        return 
    
    @app.get("/retrieve_documents")
    async def retrieve_documents(query: str):
        """
        retrieves documents from query
        """

        try:
            # Load embeddings model
            embeddings_model = get_embeddings_model(
                    model_name=EMBEDDING_MODEL, model_kwargs=EMBEDDING_MODEL_KWARGS, encode_kwargs=EMBEDDING_ENCODE_KWARGS
                    )

            # Load vectordb
            vector_db = Milvus(
                embeddings_model,
                connection_args={'uri': CONNECTION_ARGS.get('uri')},
                collection_name=CONNECTION_ARGS.get('collection')
            )

            # Load docs
            docs = vector_db.similarity_search_with_score(query=query, k=5)
            # -> List[Tuple[Document, float]]
        except:
            pass

        return docs
    
        
if __name__ == "__main__":
    main()
    