import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, GPT2Tokenizer, GPT2LMHeadModel
from pymilvus import Collection, connections

# Initialize Hugging Face models and tokenizers
EMBED_MODEL_NAME = 'nomic-ai/nomic-embed-text-v1'
GEN_MODEL_NAME = 'gpt2'

# Embedding model
embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME, trust_remote_code=True)
embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME, trust_remote_code=True)

# Generation model
gen_tokenizer = GPT2Tokenizer.from_pretrained(GEN_MODEL_NAME)
gen_model = GPT2LMHeadModel.from_pretrained(GEN_MODEL_NAME)

# Add a padding token to the tokenizer
if gen_tokenizer.pad_token is None:
    gen_tokenizer.add_special_tokens({'pad_token': gen_tokenizer.eos_token})
    gen_model.resize_token_embeddings(len(gen_tokenizer))

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

# Set device to CPU
device = torch.device("cpu")
gen_model.to(device)

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

def search(query: str):
    """Search for relevant documents in Milvus and generate a response using GPT-2."""
    collection = Collection(name=COLLECTION_NAME)
    collection.load()  # Load the collection into memory
    print(f"Collection '{COLLECTION_NAME}' loaded.")
    
    query_embedding = generate_query_embedding(query)
    
    # Perform search with limit set to 4
    search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
    results = collection.search([query_embedding], "embedding", search_params, limit=4)
    
    # Collect content from top 4 results
    contexts = []
    for result in results[0]:
        print(f"ID: {result.id}, Distance: {result.distance}")
        document = collection.query(expr=f"id in [{result.id}]", output_fields=["content"])
        content = document[0]["content"]
        contexts.append(content)
    
    # Combine contexts
    combined_context = "\n\n".join(contexts)
    
    # Tokenize and handle context length
    inputs = gen_tokenizer(combined_context, return_tensors='pt', truncation=True, padding=True)
    max_length = gen_model.config.n_positions - 50  # Leave some space for generation tokens
    input_ids = inputs['input_ids']
    
    if input_ids.shape[1] > max_length:
        input_ids = input_ids[:, :max_length]
        attention_mask = inputs['attention_mask'][:, :max_length] if 'attention_mask' in inputs else None
    else:
        attention_mask = inputs['attention_mask'] if 'attention_mask' in inputs else None
    
    print(f"Combined Context (truncated): {combined_context[:500]}")  # Display part of the context for debugging

    # Use combined context to prompt the GPT-2 model
    outputs = gen_model.generate(
        input_ids,
        max_new_tokens=500,
        pad_token_id=gen_tokenizer.pad_token_id,
        attention_mask=attention_mask
    )
    generated_text = gen_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"Generated Response: {generated_text}")

if __name__ == "__main__":
    query = "how to upgrade redis"
    search(query)
