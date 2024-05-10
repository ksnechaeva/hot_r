import os
import json
import yaml
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import PromptValue, BaseMessage
from llama_index.core import Document
import nest_asyncio
from llama_index.core import VectorStoreIndex
from llama_index.core.evaluation import RelevancyEvaluator, FaithfulnessEvaluator, DatasetGenerator
import nest_asyncio
from tqdm.asyncio import tqdm_asyncio
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser

nest_asyncio.apply()

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f.read())

os.environ['OPENAI_API_KEY'] = config['OPENAI_API_KEY']
os.environ['OPENAI_API_BASE'] = config['OPENAI_API_BASE']

with open('data/hotels.json', 'r') as f:
    hotels = json.load(f)

# Create a list of Document objects from the hotels data
documents = []
for hotel_name, hotel_desc in hotels.items():
    documents.append(Document(text=f"Hotel name: {hotel_name}\n Description:\n{hotel_desc}"))


llm = ChatOpenAI()
index = VectorStoreIndex.from_documents(documents)
evaluator = FaithfulnessEvaluator(llm=llm)
query_engine = index.as_query_engine()

def eval_with_llm():

    results = []
    template_q = """You are given hotel description.
    Create 2 questions for each hotel on which it will be possible to answer using the description.

    Hotel description:
    {desc}

    Question:"""

    prompt_q = PromptTemplate.from_template(template_q)
    query = prompt_q | llm | StrOutputParser()
    eval_questions = query.invoke({"query": query, "desc": documents})
    print(eval_questions)
    for i in range(len(eval_questions)):
        question = eval_questions[i]
        response = query_engine.query(question)
        response_str = response.response
        source_node_contents = [node.get_content() for node in response.source_nodes]

        for content in source_node_contents:
            eval_result = evaluator.evaluate(
                query=question,
                response=response_str,
                contexts=[content],
            )
            results.append(eval_result.score)
            print(f'Question {i} : {eval_questions[i]}: Score = {eval_result.score}')

    return sum(results)/len(results)


def eval_with_dsg():

    results = []
    data_generator = DatasetGenerator.from_documents(
        documents=documents,
        llm=llm,
        num_questions_per_chunk=2,
    )
    eval_questions = data_generator.generate_questions_from_nodes()

    for i in range(len(eval_questions)):
        question = eval_questions[i]
        response = query_engine.query(question)
        response_str = response.response
        source_node_contents = [node.get_content() for node in response.source_nodes]

        for content in source_node_contents:
            eval_result = evaluator.evaluate(
                query=question,
                response=response_str,
                contexts=[content],
            )
            results.append(eval_result.score)
            # print(f'Question {i} : {eval_questions[i]}: Score = {eval_result.score}')

    return sum(results)/len(results)

#score_1= eval_with_dsg()
score_2 = eval_with_llm()
print(score_2)
