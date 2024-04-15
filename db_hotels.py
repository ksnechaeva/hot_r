import requests
from bs4 import BeautifulSoup
import os
import time
import json
from openai import OpenAI
os.environ['OPENAI_API_KEY'] = "sk-wIcqz35tV9EHW6aR9lMHQVYmMOq30iAI"
os.environ['OPENAI_API_BASE'] = "https://api.proxyapi.ru/openai/v1"
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

#this process should be run in env with httpx==0.27.0
top_100_url = "https://tophotels.ru/club/ratings/top100"
r = requests.get(top_100_url)
soup = BeautifulSoup(r.text, "html.parser")

top_10_list = soup.find_all("table")[0].find_all('tr')[1:16]

hotels = dict()

for item in top_10_list:
    hotels[item.a.get_text().strip()] = item.a.attrs['href']

hotels_info = {}
miss_desc = []
for hotel_name, href in hotels.items():
    try:
        hotels_info[hotel_name] = {"description": None, "reviews": []}


        desc_url = f'https://tophotels.ru{href}'
        r = requests.get(desc_url)
        print(f"Fetching description for {hotel_name}: {r}")
        soup = BeautifulSoup(r.text, "html.parser")
        divs = soup.find_all('div', {"class": "grey-item__white"})

        if divs:
            hotel_desc = divs[0].get_text().strip()
            hotels_info[hotel_name]["description"] = hotel_desc
        else:
            miss_desc.append(hotel_name)

        page_number = 1
        while page_number != 2:
            review_url = f'https://tophotels.ru{href}/reviews/list?page={page_number}'
            response = requests.get(review_url)
            soup = BeautifulSoup(response.text, "html.parser")

            current_page_reviews = soup.find_all("div", class_="review__txt-wrap")
            if not current_page_reviews:
                break

            for review in current_page_reviews:
                hotels_info[hotel_name]["reviews"].append(review.get_text().strip())

            page_number += 1
            time.sleep(1)

    except Exception as e:
        print(f"Error processing {hotel_name}: {e}")

for hotel_name, info in hotels_info.items():
    # Check if the description is None and replace it with an empty string if so
    description = info['description'] if info['description'] is not None else ""
    # Combine the description with all reviews
    combined_text = description + " " + " ".join(info['reviews'])
    hotels_info[hotel_name]['combined_text'] = combined_text

with open('hotels_info.json', 'w', encoding='utf-8') as f:
    json.dump(hotels_info, f, ensure_ascii=False, indent=4)

