import os
import re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from pymilvus import Collection, CollectionSchema, FieldSchema, DataType, utility, connections
from typing import List
import pandas as pd

MODEL_NAME = 'nomic-ai/nomic-embed-text-v1'
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
except Exception as e:
    print(f"Error loading model: {e}")
    # Fallback or alternative actions
    raise

# Initialize Milvus client
MILVUS_HOST = 'localhost'
MILVUS_PORT = '19530'

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

def markdown_to_text_splitter_by_heading(file_path: str) -> List[str]:
    """Reads a Markdown file and extracts text sections based on headings."""
    with open(file_path, 'r') as file:
        content = file.read()

    # Regex to match the headings and their content
    regex = re.compile(r'(##+.*?)(?=##+|\Z)', re.DOTALL)
    matches = regex.findall(content)
    
    # Strip leading and trailing whitespaces from each chunk
    chunks = [match.strip() for match in matches]
    return chunks

def generate_embeddings(text_list: List[str], prefix: str) -> List[np.ndarray]:
    """Generates embeddings for a list of text paragraphs using Hugging Face model."""
    embeddings = []
    for text in text_list:
        text = f"{prefix} {text}"  # Add prefix to text
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        # Use mean pooling to get sentence embeddings
        last_hidden_states = outputs.last_hidden_state
        attention_mask = inputs['attention_mask']
        masked_embeddings = last_hidden_states * attention_mask.unsqueeze(-1)
        sentence_embeddings = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        embeddings.append(sentence_embeddings.squeeze().numpy())
    return embeddings

def create_collection(collection_name: str):
    """Creates a collection in Milvus."""
    if not utility.has_collection(collection_name):
        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True), # Primary key
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=20000),
            FieldSchema(name="source_title", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
        ]
        schema = CollectionSchema(fields, description="Knowledge base")
        # Create collection
        collection = Collection(name=collection_name, schema=schema)

        # Create index on the embedding field if it doesn't exist
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "IP",
            "params": {"nlist": 128}
        }
        collection.create_index("embedding", index_params)
        print("Index created.")

def insert_to_milvus(collection_name: str, texts: List[str], embeddings: List[np.ndarray], file_names: List[str]):
    """Inserts embeddings into Milvus."""
    collection = Collection(name=collection_name)

    # Ensure all lists are of the same length
    if not (len(texts) == len(embeddings) == len(file_names)):
        raise ValueError("Mismatch in lengths of texts, embeddings, and file_names")

    # Prepare data for insertion
    data = [
        texts,
        file_names,
        embeddings
    ]

    # Insert data
    collection.insert(data)

def populate_db(knowledge_source_folder: str, collection_name: str):
    """Populates Milvus database with text from Markdown files."""
    create_collection(collection_name)

    all_texts = []
    all_embeddings = []
    all_file_names = []

    # Process each Markdown file in the folder
    for filename in os.listdir(knowledge_source_folder):
        if filename.endswith('.md'):
            file_path = os.path.join(knowledge_source_folder, filename)
            texts = markdown_to_text_splitter_by_heading(file_path)
            
            embeddings = generate_embeddings(texts, 'search_document')  # Use prefix for documents
            all_texts.extend(texts)
            all_embeddings.extend(embeddings)
            all_file_names.extend([filename] * len(texts))  # Assign file_name to each text

    # Insert all texts and embeddings into Milvus
    insert_to_milvus(collection_name, all_texts, all_embeddings, all_file_names)

if __name__ == "__main__":
    KNOWLEDGE_SOURCE_FOLDER = 'knowledge-source'
    COLLECTION_NAME = 'knowledge_collection'
    
    populate_db(KNOWLEDGE_SOURCE_FOLDER, COLLECTION_NAME)
    print("Database population complete.")
