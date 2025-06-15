import os
from dotenv import load_dotenv
from fastapi import FastAPI
from langserve import add_routes
from pydantic import BaseModel, Field # Import BaseModel and Field

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap

# Load env vars
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# --- Define the input schema using Pydantic ---
class TranslationInput(BaseModel):
    language: str = Field(..., description="The language to translate to (e.g., 'French', 'Spanish').")
    text: str = Field(..., description="The text to be translated.")

# Initialize components
model = ChatGroq(model="gemma-7b-it", api_key=groq_api_key)
parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages([
    ("system", "Translate the following into {language}"),
    ("user", "{text}")
])

# Use RunnableMap to explicitly define inputs for langserve
# The chain expects a dictionary matching TranslationInput fields
chain = RunnableMap({
    "language": lambda x: x["language"],
    "text": lambda x: x["text"]
}) | prompt | model | parser

# Setup FastAPI
app = FastAPI(
    title="LangChain Translation API",
    version="1.0",
    description="A simple API with LangServe"
)

# Register LangServe route, passing the input_type
add_routes(app, chain, path="/chain", input_type=TranslationInput)

# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)