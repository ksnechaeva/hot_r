#evaluation of obtained model

import os
import getpass
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.schema import StrOutputParser
from llama_index.core.base.response.schema import Response
from llama_index.core.evaluation import BaseEvaluator
from llama_index.core.evaluation.base import EvaluationResult
from llama_index.core.prompts.mixin import PromptDictType
from llama_index.core.evaluation import RelevancyEvaluator

os.environ['OPENAI_API_KEY'] = "sk-wIcqz35tV9EHW6aR9lMHQVYmMOq30iAI"
os.environ['OPENAI_API_BASE'] = "https://api.proxyapi.ru/openai/v1"
index_path = "/Users/a0000/hot_r-2/faiss_index" 
embeddings = OpenAIEmbeddings()
new_db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

class ResponseObject:
    def __init__(self, response):
        self.response = response

llm = ChatOpenAI()
evaluator = RelevancyEvaluator(llm=llm)

query = str(input("Enter your search query: "))

docs_and_scores = new_db.similarity_search_with_score(query)
search_results = []

for n, (doc, score) in enumerate(docs_and_scores, start=1):
    hotel_name = doc.metadata['name'].split()[0] 
    search_results.append(f"{n}) Hotel {hotel_name} with score {score:.2f}")

response = "\n".join(search_results) if search_results else "No hotels found matching your criteria."

eval_result = evaluator.evaluate_response(query=query, response=response)
print(str(eval_result))