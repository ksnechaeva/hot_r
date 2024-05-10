Repository description:
* data : contains 2 json files with hotels' reviews and descriptions ( english and russian version )
* faiss_index : contains embedded db with english version of hotels' reviews

* preprocessing:

  - db_hotels.py : parse hotel reviews and descriptions from tophotels
  -  hotels_to_eng.py :translate db
  -  emb.py : file with creating faiss_index

* main_3 : draft 
* bot.py : code with establishing tg bot
* evaluation.py : script for evalution with llama-index

To run:
   run bot.py file 
