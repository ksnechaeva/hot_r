from googletrans import Translator
import json

file_path = 'hotels_info.json'

with open(file_path, 'r', encoding='utf-8') as file:
    hotels_info = json.load(file)

def translate_hotels_info(hotels_info):
    translator = Translator()
    translated_hotels_info = {}

    for hotel_name, hotel_data in hotels_info.items():
        translated_hotel_data = {}
        try:
            description = hotel_data["description"]
            if description:  
                translated_description = translator.translate(description, dest='en').text
                translated_hotel_data["description"] = translated_description

            translated_reviews = []
            for review in hotel_data["reviews"]:
                if review:  
                    translated_review = translator.translate(review, dest='en').text
                    translated_reviews.append(translated_review)

            translated_hotel_data["reviews"] = translated_reviews

            combined_text = translated_description + " " + " ".join(translated_reviews)
            translated_hotel_data['combined_text'] = combined_text

        except Exception as e:
            print(f"Error translating {hotel_name}: {e}")

        translated_hotels_info[hotel_name] = translated_hotel_data
        print('1')

    return translated_hotels_info

translated_hotels_info = translate_hotels_info(hotels_info)

with open('hotels_info_eng.json', 'w', encoding='utf-8') as f:
    json.dump(translated_hotels_info, f, ensure_ascii=False, indent=4)