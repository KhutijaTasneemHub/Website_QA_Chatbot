import streamlit as st
import requests
from bs4 import BeautifulSoup

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

import os
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.stop()  # stop the app safely

st.header("Website Content Chatbot")

url = st.text_input("Enter a website URL (example: https://www.python.org):")

if not url:
    st.info("↑ Paste a URL above to begin.")

try:
    # Step 3: Fetch page HTML (pretend to be a normal browser, short timeout)
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    resp.raise_for_status()  # stop if page returned an error
    # st.success(f"Fetched page (status {resp.status_code})")
    # st.code(resp.text[:1000] + "...", language="html")  # show first 1000 chars of raw HTML

    # Step 4: Parse HTML and extract paragraph text
    soup = BeautifulSoup(resp.text, "html.parser")
    paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    website_text = " ".join(paragraphs)

    if not website_text.strip():
        # Fallback: get all text if no <p> found
        website_text = soup.get_text(" ", strip=True)

    # st.write("### Extracted text preview:")
    # st.text(website_text[:1000] + ("..." if len(website_text) > 1000 else ""))
    # st.write(f"Characters extracted: {len(website_text):,}")

except Exception as e:
    st.error(f"Could not fetch or parse page: {e}")


if url and website_text.strip():
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=500,  # each piece ~500 chars
        chunk_overlap=50,  # keep 50 char overlap for context
        length_function=len
    )
    chunks = text_splitter.split_text(website_text)
    # st.write(f"✅ Split into {len(chunks)} chunks.")

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vector_store = FAISS.from_texts(chunks, embeddings)
    # st.success("✅ Website content embedded and stored in a vector database.")

    question = st.text_input("Ask a question about this website:")
    if question:
        # 1) Find similar chunks
        similar_docs = vector_store.similarity_search(question, k=4)  # top 4 chunks

        # 2) LLM for answering
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            temperature=0,
            model_name="gpt-3.5-turbo"  # keep same as your other projects
        )

        # 3) QA chain
        chain = load_qa_chain(llm, chain_type="stuff")

        # 4) Get the answer
        answer = chain.run(input_documents=similar_docs, question=question)

        st.write("### Answer")
        st.write(answer)