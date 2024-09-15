# Integrate our code openAI API
import os
# from constants import openai_key
from langchain_openai import OpenAI
import streamlit as st 
from Langchain import config

os.environ["OPENAI_API_KEY"]=config.openai_secret
# streamlit framework

st.title('Langchain demo')
input_text=st.text_input("Search the topic u want")


llm=OpenAI(temperature=0.8, model='gpt2')


if input_text:
    st.write(llm(input_text))
