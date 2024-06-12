from pymilvus import MilvusClient

def main():

    PATH = "./data/fp-2024-01.db"

    # Instantiate Milvus client
    milvus_client = MilvusClient(PATH)

    try:
        # Check if collection exists
        collections = milvus_client.list_collections()
        if "documents_collection" in collections:
            # Drop collection
            milvus_client.drop_collection(collection_name="documents_db")
        
        elif "questions_collection" in collections:
            # Drop collection
            milvus_client.drop_collection(collection_name="questions_db")

    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()

