import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

#define model
model = OllamaLLM(model="gemma:2b")
#define promt for mode
prompt = ChatPromptTemplate.from_messages([
    ('system',"you are expert of everthing so answer the questions as per you knowlede"),
    ('user','{input}')
])
#parser to parse the response into text
output_parser = StrOutputParser()

#create basic chain
chain = prompt|model|output_parser

#create fastapi app
app = FastAPI()

#addign cors to interect with js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],

)

class QuestionRequest(BaseModel):
    question:str

@app.post("/qna")
def qna(req:QuestionRequest):
    print(req.question)
    return {'answer': chain.invoke(req.question)}



# if __name__ == "__main__":
#     question = "What was my previous question?"
#     ans = qna(question)
#     print(ans)

