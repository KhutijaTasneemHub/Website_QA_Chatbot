Code Explanation - Line by Line Written by author – @KhutijaTasneemHub

This project is a chatbot that can read the content of any website you give it and then answer your questions about that website.
It’s like teaching your computer to "read a website" and then "talk back to you" about it.
Built using Python, Streamlit, LangChain, and OpenAI GPT model.

**Import the tools we need**
import streamlit as st
import requests
from bs4 import BeautifulSoup

Streamlit (st) - Makes a nice web app (so we don’t have to run things in the terminal).
Requests - Helps us "visit" a website and download its code (HTML).
BeautifulSoup - Like a cleaner: it goes into that messy HTML code and pulls out only the text we care about (paragraphs).

**Import AI-related tools**
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

**TextSplitter** - Breaks big website text into smaller pieces (like cutting a big pizza into slices).
**Embeddings (OpenAIEmbeddings)** - Turns text into "math numbers" (so AI can understand and compare).
**FAISS** - A special box that stores those math numbers and lets us quickly search similar ones.
**QA Chain + ChatOpenAI** - The actual AI brain (GPT model) that reads the chunks and gives smart answers.

**Load your API key safely**
_import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.stop()_

We keep our secret OpenAI key inside a hidden .env file.
This key lets us "talk" to OpenAI’s AI models.
If the key is missing - the app stops safely.

**Streamlit App Layout**
st.header("Website Content Chatbot")
url = st.text_input("Enter a website URL (example: https://www.python.org):")

Shows a title on the app.
Lets you type (paste) a website link.

**Fetch the Website Content**
resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
soup = BeautifulSoup(resp.text, "html.parser")
paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
website_text = " ".join(paragraphs)

requests.get(url) - goes to the website (like opening it in a browser).
BeautifulSoup - grabs only the readable text inside <p> tags (paragraphs).
Joins everything into one big text string = website_text.

**Split the Text into Chunks**
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n"],
    chunk_size=500,
    chunk_overlap=50,
    length_function=len
)
chunks = text_splitter.split_text(website_text)

A website might have thousands of words (too big for AI at once).
We cut it into small slices of 500 characters each, with a little overlap (50 chars) so AI doesn’t lose context.

**Store Website Text in Vector Database**
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vector_store = FAISS.from_texts(chunks, embeddings)

Each text slice is converted into embeddings (AI-readable math).
Stored in FAISS - so later we can search for the most relevant parts of the website.

**Ask Questions About the Website**
question = st.text_input("Ask a question about this website:")

The user types a question like:
“What is Python?” (if the website is python.org)

**Find Relevant Text & Get AI Answer**
similar_docs = vector_store.similarity_search(question, k=4)

llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    temperature=0,
    model_name="gpt-3.5-turbo"
)

chain = load_qa_chain(llm, chain_type="stuff")

answer = chain.run(input_documents=similar_docs, question=question)


Find relevant chunks (like searching inside the vector database).
ChatOpenAI is called - feeds those chunks + your question into GPT.
GPT thinks - then generates a clean answer for you.


**Show the Answer**
st.write("### Answer")
st.write(answer)

Finally, your chatbot prints the answer on the webpage!!!! 


**How to Run This Project:**
Clone this repo:
git clone https://github.com/<your-username>/Website_QA_Chatbot.git
cd Website_QA_Chatbot

Install requirements:
pip install -r requirements.txt

Create a .env file in the project folder:
OPENAI_API_KEY=your_openai_api_key_here


Run the app:
streamlit run app.py

Open the link shown (usually http://localhost:8501) in your browser.



