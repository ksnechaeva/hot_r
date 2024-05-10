import json
import os
os.environ['OPENAI_API_KEY'] = "sk-wIcqz35tV9EHW6aR9lMHQVYmMOq30iAI"
os.environ['OPENAI_API_BASE'] = "https://api.proxyapi.ru/openai/v1"
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tqdm.autonotebook import tqdm
from langchain_community.callbacks import get_openai_callback
from llama_index.core import Document

with open("data/hotels_info_eng.json", "r") as f:
    hotels_info_eng = json.loads(f.read())
# теперь в переменной hotels_info_eng лежит словарь, где ключи - это названия отелей
for k, v in hotels_info_eng.items():
    k = k.replace("\n", "").strip() # удаляем переносы строки из названия
    print(f"Ключ: {k}")
    # v - это тоже словарь с полями, которые указаны в json
    v_fields = list(v.keys())
    print(f"Поля в v: {v_fields}")
    break
# в поле reviews содержатся комментарии к соответствующему отелю


llm = ChatOpenAI()
# темплейт для саммаризации отзывов
reviews_summary_template = """You are given a description and a list of reviews for a hotel {hotel_name}.
Please add the reviews summary with the important information that is not present in description.
Description: {desc}
List of reviews:
{reviews}
"""

reviews_summary_prompt = PromptTemplate.from_template(reviews_summary_template)
# chain для саммаризации
reviews_summary_chain = reviews_summary_prompt | llm | StrOutputParser()

with get_openai_callback() as cb: # специальная функция, которая подсчитывает кол-во токенов и расход
    hotels = {}
    for k, v in tqdm(hotels_info_eng.items(), total=len(hotels_info_eng)):
        # удаляю лишние токены (переносы строки в названии, пробелы перед и после текста) чтобы уменьшить кол-во токенов
        hotel_name = k.replace("\n", " ").replace("  ", "").strip()
        desc = v["description"].strip()
        reviews = "".join([f"{n+1}. {review.strip()}\n\n" for n, review in enumerate(v["reviews"][:10])])
        reviews_summary = reviews_summary_chain.invoke({"hotel_name": hotel_name, "desc": desc, "reviews": reviews})

        hotels[hotel_name] = desc + "\n" + reviews_summary
    print(cb) # печатаем, сколько токенов потратили (4 рубля)

with open('hotels.json', 'w') as f:
    json.dump(hotels, f)


