import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from pymilvus import Collection, connections
from dotenv import load_dotenv
import os
from flask import Flask, request, jsonify
from slack_sdk import WebClient
import requests
import subprocess
import json
import string
from slackeventsapi import SlackEventAdapter
import time
import re
import argparse

from ibm_watsonx_ai.client import APIClient

# Load environment variables from .env file
load_dotenv()

project_id = os.getenv('PROJECT_ID')
project_url = os.getenv('PROJECT_URL')
ic_api_key = os.getenv('IC_API_KEY')
slack_token = os.getenv("SLACK_TOKEN")
if not slack_token:
    raise ValueError("Slack API token not found.")
client = WebClient(token=slack_token)
signing_secret = os.getenv("SIGNING_SECRET")
if not signing_secret:
    raise ValueError("Slack signing secret not found.")


# Initialize Hugging Face models and tokenizers
EMBED_MODEL_NAME = 'nomic-ai/nomic-embed-text-v1'

# Embedding model
embed_tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_NAME, trust_remote_code=True)
embed_model = AutoModel.from_pretrained(EMBED_MODEL_NAME, trust_remote_code=True)

# Initialize Milvus client
MILVUS_HOST = 'localhost'
MILVUS_PORT = '19530'
COLLECTION_NAME = 'knowledge_collection'

app = Flask(__name__)
slack_events_adapter = SlackEventAdapter(signing_secret, "/events-endpoint", app)

BOT_ID = client.api_call("auth.test")['user_id']

BAD_WORDS = ['hmm', 'no', 'idiot']

# IBM Model credentials
credentials = {
    "apikey": ic_api_key,
    "url": project_url
}
client = APIClient(credentials)

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

def filter_question(text):
    # Define the pattern for a word starting and ending with '<'
    pattern = r'^<[^>]+> '
    # Use re.sub to remove the pattern from the start of the string
    modified_text = re.sub(pattern, '', text)
    return modified_text

def get_bearer_token(apikey):
    curl_command = f'curl -k -X POST --header "Content-Type: application/x-www-form-urlencoded" --header "Accept: application/json" --data-urlencode "grant_type=urn:ibm:params:oauth:grant-type:apikey" --data-urlencode "apikey={apikey}" "https://iam.cloud.ibm.com/identity/token"'
    try:
        token_response = subprocess.check_output(curl_command, shell=True).decode('utf-8')
        token_json = json.loads(token_response)
        access_token = token_json.get('access_token')
        return access_token
    except subprocess.CalledProcessError as e:
        print(f"Error generating token: {e}")
        return None

def text_generation(input_text):
    token = get_bearer_token(ic_api_key)
    if not token:
        return None

    url = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation?version=2023-05-29"
    body = {
        "input": input_text,
        "parameters": {
            "decoding_method": "greedy",
            "max_new_tokens": 5000,
            "repetition_penalty": 1,
            "temperature": 0
        },
        "model_id": "ibm/granite-13b-chat-v2",
        "project_id": os.environ.get('PROJECT_ID')
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}" 
    }

    try:
        response = requests.post(url, headers=headers, json=body)
        if response.status_code != 200:
            raise Exception("Non-200 response: " + str(response.text))
        return response.json()
    except Exception as e:
        print(f"Error: {e}")
        return None

@app.route('/generate_text', methods=['POST'])
def generate_text():
    input_text = request.form.get('text')
    if not input_text:
        return jsonify({"text": "Please provide input text."})

    response = text_generation(input_text)
    if response:
        generated_text = response.get('results', [{}])[0].get('generated_text', '')
        confidence = response.get('confidence', '')
        response_code = response.get('response_code', '')
        
        print(f"Generated Text: {generated_text}")

        return jsonify({"Response": generated_text})
    else:
        return jsonify({"text": "Error in text generation."})

def check_if_bad_words(message):
    msg = message.lower()
    msg = msg.translate(str.maketrans('', '', string.punctuation))

    return any(word in msg for word in BAD_WORDS)

processed_messages = set()

@slack_events_adapter.on('message')
def message(payload):
    print("Message is called")
    event = payload.get('event', {})
    channel_id = event.get('channel')
    user_id = event.get('user')
    text = event.get('text')
    message_ts = event.get('ts')

    if BOT_ID != user_id and message_ts not in processed_messages:
        processed_messages.add(message_ts)

        query = filter_question(text)
        query_embedding = generate_query_embedding(query)
        combined_context = search_milvus(query_embedding)
        prompt = make_prompt(combined_context, query)

        generated_response = text_generation(prompt)
        if generated_response:
            generated_text = generated_response.get('results', [{}])[0].get('generated_text', '')
            confidence = generated_response.get('confidence', '')
            response_code = generated_response.get('response_code', '')
            print(f"Generated Text: {generated_text}")
            client.chat_postMessage(
                channel=channel_id,
                text=f"{generated_text}"
            )
        else:
            client.chat_postMessage(channel=channel_id, text="Error in text generation.")

        time.sleep(1)

@app.route("/events-endpoint", methods=["POST"])
def events_endpoint():
    print("Received POST request to /events-endpoint")
    return "", 200

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

    print("[INFO] Server listening")
    app.run(port=8080)
