# InformaBot

## Overview

DocAssistant is an intelligent system designed to store and query runbooks or any other knowledge sources, retrieving relevant documents and prompting a chat model with the same query. While initially leveraging a Milvus vector database, the system is designed to be extensible to any database type, making it a versatile tool for efficient document retrieval and knowledge management.

## Setup Instructions

### 1. How to Set Up a Milvus Server

To set up a Milvus server, follow these steps:

1. **Allocate Additional Memory to Docker**:
   - Milvus requires a minimum of 8GB of available memory. Docker usually allocates only 2GB by default. Increase Docker memory through the Docker desktop settings under Resources.

2. **Download Docker Compose Configuration**:
   - Create a directory and download the Docker Compose configuration file for Milvus:
     ```bash
     mkdir milvus_compose
     cd milvus_compose
     wget https://github.com/milvus-io/milvus/releases/download/v2.2.8/milvus-standalone-docker-compose.yml -O docker-compose.yml
     ```

3. **Run Milvus Using Docker Compose**:
   - Start Milvus with Docker Compose:
     ```bash
     docker compose up -d
     ```

   - Verify that all containers are running:
     ```bash
     docker ps -a
     ```

   - Check the Milvus server logs to ensure it's up and running:
     ```bash
     docker logs milvus-standalone
     ```

   For detailed instructions, refer to the [Milvus Standalone Setup Guide](https://milvus.io/blog/how-to-get-started-with-milvus.md).

### 2. How to Populate the Database

1. Place all your runbooks and knowledge documents into the `knowledge-source` folder.

2. Run the `populate_db.py` script to populate the Milvus database with these documents:
   ```bash
   python populate_db.py
   ```
This script will index the documents and store them in your Milvus server.

### 2. How to Populate the Model

1. Ensure you have a .env file in your project directory with the following environment variables:
    ```bash
    PROJECT_ID=your_project_id
    PROJECT_URL=your_project_url
    IC_API_KEY=your_api_key
    ```

2. Run the query.py script to query the database and generate responses using the IBM model:
   ```bash
   python query.py
   ```
The script will prompt you to enter your question, then retrieve relevant documents from Milvus, and finally generate a response using the IBM model.
