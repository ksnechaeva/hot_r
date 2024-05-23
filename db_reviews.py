import os
import json
from openai import OpenAI
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import yaml
from sentence_transformers import SentenceTransformer
from llama_index.core import StorageContext

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f.read())

os.environ['OPENAI_API_KEY'] = config['OPENAI_API_KEY']
os.environ['OPENAI_API_BASE'] = config['OPENAI_API_BASE']

file_path = 'data/hotels_info_eng.json'

save_directory = 'hotel_databases'
os.makedirs(save_directory, exist_ok=True)


with open(file_path, 'r', encoding='utf-8') as file:
    hotels_info = json.load(file)
for hotel_name, info in hotels_info.items():
    if 'reviews' in info and isinstance(info['reviews'], list):
        # Adjust the hotel name to create a safe file name
        safe_hotel_name = hotel_name.replace('/', '_')  # Replace '/' in hotel names to avoid path issues
        # Modify the file name according to the new specification
        adjusted_file_name = safe_hotel_name.split('*')[0] + '*'
        
        # Define the file path for this hotel's reviews
        file_path = os.path.join(save_directory, f"{adjusted_file_name}.json")
        
        # Write the reviews to a JSON file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(info['reviews'], f, ensure_ascii=False, indent=4)

print("All hotel databases have been saved.")
'''
model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode([doc.text for doc in docs])

storage_context = StorageContext.from_defaults()
vector = FAISS.from_documents(documents=docs, embeddings=embeddings, storage_context=storage_context)
vector.save_local("faiss_index_rev")
print('ready')
'''