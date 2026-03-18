from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt = PromptTemplate.from_template("It's always sunny in {city}!")

chain = prompt | llm

response = chain.invoke({"city": "San Francisco"})

print(response.content)