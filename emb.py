import os
import json
from openai import OpenAI
os.environ['OPENAI_API_KEY'] = "sk-wIcqz35tV9EHW6aR9lMHQVYmMOq30iAI"
os.environ['OPENAI_API_BASE'] = "https://api.proxyapi.ru/openai/v1"
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

print('1')
file_path = 'hotels_info_eng.json'

with open(file_path, 'r', encoding='utf-8') as file:
    hotels_info = json.load(file)

embeddings = OpenAIEmbeddings()
docs = [
    Document(page_content=info['combined_text'], metadata={"name": hotel_name})
    for hotel_name, info in hotels_info.items()
    if 'combined_text' in info
]
vector = FAISS.from_documents(docs, embeddings)
vector.save_local("faiss_index")