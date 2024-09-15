import os
from langchain.llms import OpenAI  # Correct import
import streamlit as st 
from Langchain import config

os.environ["OPENAI_API_KEY"]=config.openai_secret
# Streamlit framework setup
st.title('Langchain Demo')
input_text = st.text_input("Search the topic you want")

# Initialize the OpenAI model with the correct model name
llm = OpenAI(temperature=0.8, model='gpt-3.5-turbo')  # Change 'gpt2' to 'gpt-3.5-turbo'

# Generate text based on user input
if input_text:
    response = llm(input_text)
    st.write(response)
