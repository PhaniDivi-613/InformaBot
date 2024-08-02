import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from pymilvus import Collection, connections
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

project_id = os.getenv('PROJECT_ID')
project_url = os.getenv('PROJECT_URL')
ic_api_key = os.getenv('IC_API_KEY')

# Initialize Hugging Face models and tokenizers
EMBED_MODEL_NAME = 'nomic-ai/nomic-embed-text-v1'

# Embedding model
embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME, trust_remote_code=True)
embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME, trust_remote_code=True)

# Initialize Milvus client
MILVUS_HOST = 'localhost'
MILVUS_PORT = '19530'
COLLECTION_NAME = 'knowledge_collection'

# Connect to Milvus server
connections.connect(
    alias='default',
    host=MILVUS_HOST,
    port=MILVUS_PORT
)
print("Connected to Milvus successfully!")

# IBM Model Parameters
params = {
    GenParams.DECODING_METHOD: "greedy",
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 5000,
    GenParams.TEMPERATURE: 0,
}

# IBM Model credentials
credentials = {
    "apikey": ic_api_key,
    "url": project_url
}

model = Model(
    model_id='ibm/granite-13b-chat-v2',
    params=params, 
    credentials=credentials,
    project_id=project_id
)

def generate_query_embedding(query: str) -> np.ndarray:
    """Generates embedding for a query using Hugging Face model."""
    query = f"search_query {query}"  # Add prefix to query
    inputs = embed_tokenizer(query, return_tensors='pt', truncation=True, padding=True)
    with torch.no_grad():
        outputs = embed_model(**inputs)
    # Use mean pooling to get sentence embeddings
    last_hidden_states = outputs.last_hidden_state
    attention_mask = inputs['attention_mask']
    masked_embeddings = last_hidden_states * attention_mask.unsqueeze(-1)
    sentence_embeddings = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
    return sentence_embeddings.squeeze().numpy()

def search_milvus(query_embedding: np.ndarray):
    """Search for relevant documents in Milvus."""
    collection = Collection(name=COLLECTION_NAME)
    collection.load()  # Load the collection into memory
    print(f"Collection '{COLLECTION_NAME}' loaded.")
    
    # Perform search with limit set to 4
    search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
    results = collection.search([query_embedding], "embedding", search_params, limit=2)
    
    # Collect content from top 4 results
    contexts = []
    for result in results[0]:
        document = collection.query(expr=f"id in [{result.id}]", output_fields=["content"])
        content = document[0]["content"]
        contexts.append(content)
    
    # Combine contexts
    combined_context = "\n\n".join(contexts)
    return combined_context

def make_prompt(context, question_text):
    return (f"{context}\n\nPlease answer a question using this text. "
          + f"If the question is unanswerable, say \"unanswerable\"."
          + f"\n\nQuestion: {question_text}")

def query_model(context: str, question_text: str):
    """Generate a response using IBM Model."""
    # Create prompt
    prompt = make_prompt(context, question_text)
    
    # Prompt LLM
    response = model.generate_text(prompt)
    print(f"Question: {question_text}\n{response}")

if __name__ == "__main__":
    query = input("Enter your question: ")
    query_embedding = generate_query_embedding(query)
    combined_context = search_milvus(query_embedding)
    query_model(combined_context, query)
