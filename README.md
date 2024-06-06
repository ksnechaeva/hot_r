# Repository description:

## **Data**

### Descriptions:
* data : contains 2 json files with hotels' reviews and descriptions ( english and russian version )
* faiss_index : contains embedded db with english version of hotels' descriptions
### Reviews:
* hotel_databases folder : contains separate json files with reviews for each hotel
* vector_databases folder : contains separate faiss indeces for reviews for each hotel 
### Images:
* hotel_images folder : contains images of all hotels
  
## **Data Collection Preprocessing**
* db_hotels.py : parse hotel reviews and descriptions from tophotels
* hotels_to_eng.py : translate db
* emb.py : file with constructing vector database for hotels' descriptions using OpenAIEmbeddings
* vec_db_reviews.py : file with constructing vector databases for hotels' reviews using OpenAIEmbeddings
* ~~upgraded_hotels_db.py~~ : draft with constructing database with descriptions and compressed, structured reviews

## **Bot**
* bot.py : code with establishing tg bot
* evaluation.ipynb : notebook for evalution performance with llama-index and selecting optimal parameters

## **To run**
   1. **Download all the data**
   2. **Run bot.py file**
