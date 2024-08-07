from pymilvus import Collection, utility, connections
import argparse
from dotenv import load_dotenv
import os

from ibm_watsonx_ai.client import APIClient

# Load environment variables from .env file
load_dotenv()

project_id = os.getenv('PROJECT_ID')
project_url = os.getenv('PROJECT_URL')
ic_api_key = os.getenv('IC_API_KEY')

# IBM Model credentials
credentials = {
    "apikey": ic_api_key,
    "url": project_url
}
client = APIClient(credentials)
client.set.default_project(project_id)

# Initialize Milvus client
MILVUS_HOST = 'localhost'
MILVUS_PORT = '19530'
COLLECTION_NAME = 'knowledge_collection'

def parse_arguments():
    parser = argparse.ArgumentParser(description="Script to check --opensource-milvus argument")
    parser.add_argument('--opensource-milvus', action='store_true', help='Flag to indicate if open source Milvus is used')
    return parser.parse_args()

if __name__ == "__main__":

    # Parse the command-line arguments
    args = parse_arguments()
    is_opensource_milvus = args.opensource_milvus
    if is_opensource_milvus:
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
    else:
        connections_list = client.connections.list()
        ibm_milvus_connection_id = connections_list[0]["ID"]
        milvus_credentials = client.connections.get_details(ibm_milvus_connection_id).get("entity").get("properties")
        # Conncet to IBM Milvus Engine
        try:
            connections.connect(alias="default",
                                host=milvus_credentials['host'],
                                port=milvus_credentials['port'],
                                user='ibmlhapikey',
                                password=milvus_credentials['password'],
                                secure=True
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
