import json
import os
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
import yaml

# load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f.read())

os.environ['OPENAI_API_KEY'] = config['OPENAI_API_KEY']
os.environ['OPENAI_API_BASE'] = config['OPENAI_API_BASE']

# Directory containing the JSON files with reviews
json_directory = 'hotel_databases'
# Directory to save the vector databases
vector_directory = 'vector_databases'
os.makedirs(vector_directory, exist_ok=True)

# Initialize the SentenceTransformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# List all files in the directory containing the JSON files
for file_name in os.listdir(json_directory):
    file_path = os.path.join(json_directory, file_name)
    # Load the reviews from the JSON file
    with open(file_path, 'r', encoding='utf-8') as file:
        reviews = json.load(file)
    
    # Check if there are reviews to process
    if reviews:
        # Encode the reviews using the SentenceTransformer
        embeddings = OpenAIEmbeddings()

        # You typically do not need to create Document objects unless the API specifically requires them
        # Since we just need to save embeddings to a FAISS index, we can proceed without this step
        docs = [Document(page_content=review) for review in reviews]
        # Create a FAISS vector store
        vector_store = FAISS.from_documents(docs, embeddings)  # This method creates an index directly from embeddings
        print('One database saved')

        # Define the file path for this hotel's vector database
        vector_path = os.path.join(vector_directory, f"{file_name.replace('.json', '.faiss')}")

        # Save the FAISS index
        vector_store.save_local(vector_path)


print('All vector databases have been saved.')