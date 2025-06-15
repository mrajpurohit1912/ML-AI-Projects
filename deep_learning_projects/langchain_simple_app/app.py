import os
from dotenv import load_dotenv
import streamlit as st

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.llms import OllamaLLM

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["HUGGINGFACE_API_KEY"] = os.getenv("HUGGINGFACE_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")


llm = OllamaLLM(model="gemma:2b")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","you are a weather expert so ans questions only realted to weather"),
        ("user","Question:{question}")
    ]
)
output_parser = StrOutputParser()
chain = prompt|llm|output_parser


st.title("LLM chatbot")

question = st.text_input("Ask Anything")


if question:

    st.write(chain.invoke(question))


