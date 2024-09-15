# Integrate our code openAI API
import os
# from constants import openai_key
from langchain_openai import OpenAI
import streamlit as st 

os.environ["OPENAI_API_KEY"]='sk-a8qdUhFqeyYvVrFoQlbJHcj6AqqWZ_TdSeZc-pWuJmT3BlbkFJ198YZ3mR_qa7PYaFVKAdKiWow3b60jtkaatymNA0wA'

# streamlit framework

st.title('Langchain demo')
input_text=st.text_input("Search the topic u want")


llm=OpenAI(temperature=0.8, model='gpt2')


if input_text:
    st.write(llm(input_text))
