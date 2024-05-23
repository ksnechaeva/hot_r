import os
import yaml
import logging
from telegram.helpers import escape_markdown
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove, Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    filters,
    MessageHandler,
    ConversationHandler,
)
import json
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser

from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS

# load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f.read())

# set openai config
os.environ['OPENAI_API_KEY'] = config['OPENAI_API_KEY']
os.environ['OPENAI_API_BASE'] = config['OPENAI_API_BASE']

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# dialogue states
USER_QUERY, USER_CHOISE, QA_SESSION, QA_SESSION_COMP = range(4)

# lowest acceptable similarity for retriever
SIMILARITY_THRESH = 0.3

def create_justification_chain():
    llm = ChatOpenAI()
    template = """You are given user query and a hotel description.
    Using query and description in one sentence answer the question why this particualar hotel could suit the user.

    Query: {query}

    Hotel description:
    {desc}

    Answer:"""
    prompt = PromptTemplate.from_template(template)
    justification_chain = prompt | llm | StrOutputParser()
    return justification_chain

def create_qa_chain():
    llm = ChatOpenAI()
    template = """You are given user question and a hotel description.
    Answer the question using only the description or if you feel that
    provided in description information is insufficient answer with "I can not answer".
    Question: {question}

    Hotel name: {hotel_name}

    Hotel description: {hotel_desc}

    Answer:"""
    prompt = PromptTemplate.from_template(template)
    qa_chain = prompt | llm | StrOutputParser()
    return qa_chain

def create_qa_chain_rev():
    llm = ChatOpenAI()
    template = """You are given user question and a hotel reviews from other visitors.
    Answer the question using only the reviews or reply with "No info" if there is no sufficient info for the answer.
    Question: {question}

    Hotel name: {hotel_name}

    Hotel reviews: {hotel_rev}

    Answer:"""
    prompt = PromptTemplate.from_template(template)
    qa_chain = prompt | llm | StrOutputParser()
    return qa_chain

def create_comp_chain():
    llm = ChatOpenAI()
    template = """You are given descriptions of two hotels and user's main preferences {query}.
    Write structured, but brief (2 sentences for each hotel and 1 conclusion sentence) comparison of given 2 hotels using hotels' descriptions, 
    highlighting benefits and drawbacks mainly based on the user's query.
    Hotel1 name: {hotel_name1}
    Hotel2 name: {hotel_name2}
    Hotel1 description: {desc1}
    Hotel2 description: {desc2}

    Answer:"""
    prompt = PromptTemplate.from_template(template)
    qa_chain = prompt | llm | StrOutputParser()
    return qa_chain

def create_qa_comp_chain():
    llm = ChatOpenAI()
    template = """You are given user question and descriptions of two hotels.
    Answer the question using descriptions or it you feel that you can not answer type "I can't answer".
    Take into account that question may address either one of the hotels or both, so you need to choose appropriate option.
    Question: {question}

    Hotel1 name: {hotel_name1}
    Hotel2 name: {hotel_name2}
    Hotel1 description: {desc1}
    Hotel2 description: {desc2}

    Answer:"""
    prompt = PromptTemplate.from_template(template)
    qa_chain = prompt | llm | StrOutputParser()
    return qa_chain

def create_index():
    # load the data from data/hotels_info_eng.json to a dict
    with open("data/hotels_info_eng.json", "r") as f:
        hotels_info_eng = json.loads(f.read())

    # format hotel names and get description
    # hotels_info = {[HOTEL_NAME] : [HOTEL_DESCRIPTION]}
    hotels_info = {}
    for k, v in hotels_info_eng.items():
        if "*" in k:
            k = k[:k.find("*")+1]
        hotel_name = k.replace("\n", " ").replace("  ", " ").strip()
        hotel_desc = v['description']
        hotels_info[hotel_name] = hotel_desc

    # Load index from the disc if possible
    if "faiss_index" in os.listdir():
        index = FAISS.load_local("faiss_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    else:    
        # create index from hotels_info
        documents = []
        for hotel_name, desc in hotels_info.items():
            documents.append(Document(page_content=desc, metadata={"hotel_name": hotel_name}))
        index = FAISS.from_documents(documents, OpenAIEmbeddings())
        index.save_local("faiss_index")

    return index, hotels_info

def create_index_rev():
    # load the data from data/hotels_info_eng.json to a dict
    with open("data/hotels_rev.json", "r") as f:
        hotels = json.loads(f.read())

    # format hotel names and get description
    # hotels_info = {[HOTEL_NAME] : [HOTEL_DESCRIPTION]}
    hotels_rev = {}
    for k, v in hotels.items():
        hotel_r = hotels[k]
        if "*" in k:
            k = k[:k.find("*")+1]
        hotel_name = k.replace("\n", " ").replace("  ", " ").strip()
        hotels_rev[hotel_name] = hotel_r

    # Load index from the disc if possible
    if "faiss_index" in os.listdir():
        index_r = FAISS.load_local("faiss_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    else:    
        # create index from hotels_info
        documents = []
        for hotel_name, rev in hotels_rev.items():
            documents.append(Document(page_content=rev, metadata={"hotel_name": hotel_name}))
        index_r = FAISS.from_documents(documents, OpenAIEmbeddings())
        index_r.save_local("faiss_index_rev")

    return index_r, hotels_rev

index, hotels_info = create_index()
#print(hotels_info)

index_r, hotels_rev = create_index_rev()

justification_chain = create_justification_chain()
comparison_chain = create_comp_chain()
comparison_qa_chain = create_qa_comp_chain()

qa_chain_desc = create_qa_chain()
qa_chain_rev = create_qa_chain_rev()

# START message handlers 

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # initialize the dialog
    await update.message.reply_text("Hi! Which hotel would you like to book?")
    return USER_QUERY

async def user_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.message.text
    # save query for future comparison 
    context.user_data['query'] = query
    # log user activity
    user = update.message.from_user
    logger.info("Query from %s: %s", user.first_name, update.message.text)

    # find relevant hotels
    # argument k controls number of documents retireved
    docs = index.similarity_search_with_score(query, k=2)
    context.user_data['rel_hot'] = docs
    # create justification for every hotel found
    justification = {}
    for doc in docs:
        # filter out documents with low score
        if doc[1] < SIMILARITY_THRESH:
            continue
        hotel_name = doc[0].metadata["name"]
        desc = doc[0].page_content
        justification[hotel_name] = justification_chain.invoke({"query": query, "desc": desc})

    # handle the situation with empty justification dict
    if len(justification) == 0:
        await update.message.reply_text("I'm sorry, I can't find any relevant hotels. Please try again with different search criteria.")
        return USER_QUERY

    # prepare repy keyboard with list of hotel names
    reply_keyboard = [list(justification.keys()), ['I want the comparison of these hotels']]

    # format the reply according to markdownV2 style
    reply = escape_markdown("Here are some options I found. Please choose one of the following:\n\n", version=2)
    # compose a list of justifications
    for n, (k, v) in enumerate(justification.items()):
        hotel_name = escape_markdown(f"{n+1}. {k}", version=2)
        hotel_name_bold = f"*{hotel_name}*"
        hotel_desc = escape_markdown(v, version=2)
        list_item = f"{hotel_name_bold}\n{hotel_desc}\n\n"
        reply += list_item

    # reply to user and offer a keyboard
    await update.message.reply_markdown_v2(
        reply,
        reply_markup=ReplyKeyboardMarkup(
            reply_keyboard, one_time_keyboard=True, input_field_placeholder="Which hotel?"
        ),
    )
    return USER_CHOISE


async def user_choise(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # choise is supposed to be one of the hotels offered in the previous step
    choise = update.message.text

    if choise == "I want the comparison of these hotels":
        user = update.message.from_user
        logger.info("User %s choice: %s", user.first_name, choise)
        print('kjk')
        query = context.user_data.get('query', '')
        # log user activity
        #user = update.message.from_user
        #logger.info("Query from %s: %s", user.first_name, 'comparison session')

        docs = context.user_data.get('rel_hot')

        hotels = []
        desc = []
        for doc in docs:
            hotels.append(doc[0].metadata["name"])
            desc.append(doc[0].page_content)


        comp = comparison_chain.invoke({"query": query, "hotel_name1": hotels[0], "hotel_name2": hotels[1], "desc1": desc[0], "desc2": desc[1]})
        await update.message.reply_text(comp)
        
        reply_keyboard = [hotels]
        # format the reply according to markdownV2 style
        reply = escape_markdown("type your question or choose hotel\n\n", version=2)

        # reply to user and offer a keyboard
        await update.message.reply_markdown_v2(
            reply,
            reply_markup=ReplyKeyboardMarkup(
                reply_keyboard, one_time_keyboard=True, input_field_placeholder="Which hotel?"
            ),
        )

        choise = update.message.text

        if choise not in docs:
            context.user_data["choise1"] = hotels[0]
            context.user_data["choise2"] = hotels[1]
            return QA_SESSION_COMP
        else:
            context.user_data["choise"] = choise
            return USER_CHOISE
    
    #elif choise not in hotels_info:
        #await update.message.reply_text("Please pick one of the hotels or type /cancel")
        #return COMPARISON
    else:
        choice_key = choise.split('*')[0] + '*'
        if choice_key not in hotels_info:
            await update.message.reply_text("Please pick one of the hotels or type /cancel")
            return USER_CHOISE

        user = update.message.from_user
        logger.info("User %s choise: %s", user.first_name, choise)

        context.user_data["choise"] = choise

        await update.message.reply_photo(
            photo=f"hotel_images/{choise.replace('/', '_').replace(' ', '_').split('*')[0][:-1]}.jpg",
            caption=f"Great! Your choice is {choise}. Do you have any questions?",
            reply_markup=ReplyKeyboardRemove()
        )
        return QA_SESSION

#async def comparison(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    print('kjk')
    query = context.user_data.get('query', '')
    # log user activity
    #user = update.message.from_user
    #logger.info("Query from %s: %s", user.first_name, 'comparison session')

    docs = context.user_data.get('rel_hot')

    hotels = []
    desc = []
    for doc in docs:
        hotels.append(doc[0].metadata["name"])
        desc.append(doc[0].page_content)


    comp = comparison_chain.invoke({"query": query, "hotel_name1": hotels[0], "hotel_name2": hotels[1], "desc1": desc[0], "desc2": desc[1]})
    await update.message.reply_text(comp)
    
    reply_keyboard = [hotels]
    # format the reply according to markdownV2 style
    reply = escape_markdown("type your question or choose hotel\n\n", version=2)

    # reply to user and offer a keyboard
    await update.message.reply_markdown_v2(
        reply,
        reply_markup=ReplyKeyboardMarkup(
            reply_keyboard, one_time_keyboard=True, input_field_placeholder="Which hotel?"
        ),
    )

    choise = update.message.text

    if choise not in docs:
        context.user_data["choise1"] = hotels[0]
        context.user_data["choise2"] = hotels[1]
        return QA_SESSION_COMP
    else:
        context.user_data["choise"] = choise
        return USER_CHOISE

async def qa_session_comp(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    question = update.message.text
    # or context.user_data["choise"]
    # log user activity
    user = update.message.from_user
    logger.info("Question from %s: %s", user.first_name, question)
    # get chosen hotel from the dialogue context
    hotel_name1 = context.user_data["choise1"].split('*')[0] + '*'
    hotel_name2 = context.user_data["choise2"].split('*')[0] + '*'

    hotels = []
    hotels.append(hotel_name1)
    hotels.append(hotel_name2)
    # get a description for 2 given hotels
    hotel_desc1 = hotels_info[hotel_name1]
    #hotel_rev1 = hotels_rev[hotel_name1]
    hotel_desc2 = hotels_info[hotel_name2]
    #hotel_rev2 = hotels_rev[hotel_name2]
    # anwer the question if it is a question
    if question not in hotels:
        resp = comparison_qa_chain.invoke({"question": question, "hotel_name1": hotels[0], "hotel_name2":hotels[1], "desc1": hotel_desc1, "desc2":hotel_desc2})
        reply_keyboard = [hotels]
        #resp += "\n\nDo you have any other questions regarding comparison. If no choose hotel for future qa-session or type /cancel"
        # format the reply according to markdownV2 style
        reply = escape_markdown('choose one ot the hotels or type new question', version=2)
        await update.message.reply_text(resp)
        # reply to user and offer a keyboard
        await update.message.reply_markdown_v2(
            reply,
            reply_markup=ReplyKeyboardMarkup(
                reply_keyboard, one_time_keyboard=True, input_field_placeholder="Which hotel would you prefer?"
            ),
        )

    choise = update.message.text
    #print(choise)
    #print(hotels)
    if choise not in hotels:
        #print('ne tuda')
        context.user_data["choise1"] = hotels[0]
        context.user_data["choise2"] = hotels[1]
        return QA_SESSION_COMP
    else:

        user = update.message.from_user
        logger.info("User %s choise: %s", user.first_name, choise)

        context.user_data["choise"] = choise

        await update.message.reply_photo(
            photo=f"hotel_images/{choise.replace('/', '_').replace(' ', '_').split('*')[0][:-1]}.jpg",
            caption=f"Great! Your choice is {choise}. Do you have any questions?",
            reply_markup=ReplyKeyboardRemove()
        )
        return QA_SESSION

async def qa_session(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    question = update.message.text
    # log user activity
    user = update.message.from_user
    logger.info("Question from %s: %s", user.first_name, question)
    # get chosen hotel from the dialogue context
    hotel_name = context.user_data["choise"]
    # get a description for a given hotel
    hotel_desc = hotels_info[hotel_name]
    hotel_rev = hotels_rev[hotel_name]
    # anwer the question

    resp = qa_chain_desc.invoke({"question": question, "hotel_name": hotel_name, "hotel_desc": hotel_desc})
    if "I can not answer" in resp:
        # If the description doesn't suffice, try using reviews
        # resp = 'Switch to reviews'
        resp = qa_chain_rev.invoke({"question": question, "hotel_name": hotel_name, "hotel_rev": hotel_rev})

    resp += "\n\nDo you have any other questions? If no, type /cancel."

    await update.message.reply_text(resp)
    return QA_SESSION


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # cancel and end the conversation
    await update.message.reply_text(
        "Bye! I hope we can talk again some day.", reply_markup=ReplyKeyboardRemove()
    )
    return ConversationHandler.END

# END message handlers 

if __name__ == '__main__':
    application = ApplicationBuilder().token(config["BOT_TOKEN"]).build()

    # create a conversation handler.
    # converation handler makes the dialogue follow the script
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            USER_QUERY: [MessageHandler(filters.TEXT & (~filters.COMMAND), user_query)],
            USER_CHOISE: [MessageHandler(filters.TEXT & (~filters.COMMAND), user_choise)],
            #COMPARISON: [MessageHandler(filters.TEXT & (~filters.COMMAND), comparison)],
            QA_SESSION: [MessageHandler(filters.TEXT & (~filters.COMMAND), qa_session)],
            QA_SESSION_COMP: [MessageHandler(filters.TEXT & (~filters.COMMAND), qa_session_comp)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    application.add_handler(conv_handler)

    application.run_polling()