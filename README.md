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







