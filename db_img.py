import os
import requests
from bs4 import BeautifulSoup

def save_image(image_url, file_path):
    if image_url.startswith('//'):
        image_url = 'https:' + image_url  
    response = requests.get(image_url)
    response.raise_for_status()
    with open(file_path, 'wb') as file:
        file.write(response.content)


# URL of the page to scrape
top_100_url = "https://tophotels.ru/club/ratings/top100"

# Send request to the URL
response = requests.get(top_100_url)
response.raise_for_status()

# Parse the HTML content
soup = BeautifulSoup(response.text, "html.parser")

# Find the table rows containing top hotels
top_10_list = soup.find_all("table")[0].find_all('tr')[1:16]

# Dictionary to store hotel names and their respective URLs
hotels = {}

# Extract hotel names and URLs
for item in top_10_list:
    hotel_name = item.a.get_text().strip()
    hotel_url = item.a['href']
    hotels[hotel_name] = f'https://tophotels.ru{hotel_url}'

# Directory to store images
os.makedirs('hotel_images', exist_ok=True)

for hotel_name, url in hotels.items():
    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    img_tags = soup.find_all('img', class_="js-fsgr-change-tab bth__img")
    
    for img in img_tags:
        img_src = img.get('src')
        # Check the source and ensure it is a valid URL
        if img_src:
            file_name = f"{hotel_name.replace('/', '_').replace(' ', '_').split('*')[0][:-1]}.jpg"
            file_path = os.path.join('hotel_images', file_name)
            save_image(img_src, file_path)
            break  # Stop after saving the first valid image

print("Images have been saved.")