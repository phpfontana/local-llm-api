from pymilvus import MilvusClient

def main():

    PATH = "./data/fp-2024-01.db"

    # Instantiate Milvus client
    milvus_client = MilvusClient(PATH)
    
if __name__ == "__main__":
    main()

