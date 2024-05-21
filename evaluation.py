import os
import json
import yaml
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import PromptValue, BaseMessage
from llama_index.core import Document
import nest_asyncio
from llama_index.core import VectorStoreIndex
from llama_index.core.evaluation import RelevancyEvaluator, FaithfulnessEvaluator, DatasetGenerator
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
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


llm = ChatOpenAI(model='gpt-3.5-turbo-0125')
llm = ChatOpenAI(model='gpt-3.5-turbo-0125')
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()

def eval_with_pr(llm):
    evaluator = FaithfulnessEvaluator(llm=llm)
    results = []
    template_q = """You are given a hotel description.
    Create question for each hotel on which it will be possible to answer using the description.

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


def eval_with_dsg(llm):

    evaluator = FaithfulnessEvaluator(llm=llm)
    results = []

    template_q = """You are given a hotel description.
    Create question for each hotel on which it will be possible to answer using the description.

    Hotel description:
    {desc}

    Question:"""

    text_qa_template = """Read the following hotel description carefully:{desc}. 
    Generate 2 questions a potential guest might ask to learn more about the hotel. 
    For each question, provide an answer based on the information in the description."""
    
    question_gen_query = """You are an user trying to choose hotel for vacation. 
    Your task is to setup {num_questions_per_chunk} questions about each of the hotels 
    based solely on provided descriptions."""

    data_generator = RagDatasetGenerator.from_documents(
    documents=documents,
    num_questions_per_chunk=2,
    text_question_template=template_q,
    text_qa_template=text_qa_template
    question_gen_query=f"You are an user trying to choose hotel for vacation. Your task is to setup {num_questions_per_chunk} questions about each of the hotels based solely on provided descriptions.",
    show_progress=True,
)
    eval_questions = data_generator.generate_questions_from_nodes()
    rag_dataset = data_generator.generate_dataset_from_nodes()
    # прогоняем датасет через наш query_engine
    prediction_dataset = rag_dataset.amake_predictions_with(predictor=query_engine, batch_size=100, show_progress=True)
    '''for i in range(len(eval_questions)):
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
    '''

    return sum(results)/len(results)

llm = ChatOpenAI(model='gpt-3.5-turbo-0125')
llm_1 = ChatOpenAI(model='gpt-3.5-turbo-1106')
score_1= eval_with_dsg(llm)
score_2 = eval_with_pr(llm)

score_3= eval_with_dsg(llm_1)
score_4 = eval_with_pr(llm_1)
print(f"DSG 0125: {score_1}\n\n Prompt 0125: {score_2}\n\n DSG 1106: {score_3}\n\n Prompt 1106: {score_4}")
