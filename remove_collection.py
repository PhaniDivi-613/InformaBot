from pymilvus import Collection, utility, connections

# Initialize Milvus client
MILVUS_HOST = 'localhost'
MILVUS_PORT = '19530'
COLLECTION_NAME = 'knowledge_collection'

# Connect to Milvus server
try:
    connections.connect(
        alias='default',
        host=MILVUS_HOST,
        port=MILVUS_PORT
    )
    print("Connected to Milvus successfully!")
except Exception as e:
    print(f"Connection failed: {e}")

# Check if the collection exists
if utility.has_collection(COLLECTION_NAME):
    # Drop the collection
    collection = Collection(name=COLLECTION_NAME)
    collection.drop()
    print(f"Collection '{COLLECTION_NAME}' has been dropped.")
else:
    print(f"Collection '{COLLECTION_NAME}' does not exist.")
