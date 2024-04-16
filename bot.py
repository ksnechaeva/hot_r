from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
import os
import getpass
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.schema import StrOutputParser
# Set your environment variables securely

os.environ['OPENAI_API_KEY'] = "sk-wIcqz35tV9EHW6aR9lMHQVYmMOq30iAI"
os.environ['OPENAI_API_BASE'] = "https://api.proxyapi.ru/openai/v1"
index_path = "/Users/a0000/Desktop/faiss_index"  # Adjust the path as needed
embeddings = OpenAIEmbeddings()
new_db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

llm = ChatOpenAI()
template = """Помоги пользователю подобрать отель под его запрос:

{context}

Question: {input}
"""
human_template = "{text}"
prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", human_template),
])

async def start(update: Update, context):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Hi, are you ready to choose plan your perfect vacation?! Please write your preferences below")

async def handle_query(update: Update, context):
    query = update.message.text
    try:
        # Fetch hotel recommendations based on the 'query'
        docs_and_scores = new_db.similarity_search_with_score(query)
        search_results = []

        for n, (doc, score) in enumerate(docs_and_scores, start=1):
            hotel_name = doc.metadata['name'].split()[0]  # Adjust this based on your actual data structure
            search_results.append(f"{n}) Hotel {hotel_name} with score {score:.2f}")

        # If no hotels found, summarize the results differently
        if search_results:
            response = "\n".join(search_results)
        else:
            response = "Sorry, I couldn't find any hotels matching your criteria."
            
    except Exception as e:
        response = f"Error during vector search: {e}"
        
    # Send the response back to the user
    await context.bot.send_message(chat_id=update.effective_chat.id, text=response)


if __name__ == '__main__':
    application = ApplicationBuilder().token('7177632813:AAEGgG6OG9kXE66GpSjXeYjJsb30BxirkR0').build()
    
    # Define handlers
    start_handler = CommandHandler('start', start)
    query_handler = MessageHandler(filters.TEXT & ~filters.COMMAND, handle_query)
    
    # Add handlers to the application
    application.add_handler(start_handler)
    application.add_handler(query_handler)
    
    # Run the bot
    application.run_polling()

