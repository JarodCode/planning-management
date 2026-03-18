import time
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()

# 1. Load a csv document (utf-8 for most caracters)
from langchain_community.document_loaders import CSVLoader
loader = CSVLoader(
    file_path="semaine_18_mars_2026.csv",
    encoding="utf-8"
)
docs = loader.load()

# 2. No need to split the document into smaller chunks because each row of the 
# csv is already a document itself
print(f"Loaded {len(docs)} rows from CSV")
print(f"Example row:\n{docs[0].page_content}\n")

# 3. Create embeddings for the document chunks with rate limiting
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Store the embeddings in a vector store (chroma in this case)
from langchain_chroma import Chroma
vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="planning_collection"
)

# 5. Create a retriever from the vector store
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "lambda_mult": 0.5}
)

# 6. Initialize LLM
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 7. Create prompt template for planning
from langchain_core.prompts import PromptTemplate
prompt = PromptTemplate(
    template="""
You are an AI assistant helping to manage and analyze planning schedules.
Use the following planning data to answer the question.
If the answer is not in the data, say you don't know.

Planning data:
{context}

Question: {question}
""",
    input_variables=["context", "question"]
)

# 8. OutputParser definition
from langchain_core.output_parsers import StrOutputParser
parser = StrOutputParser()

# 9. Create RAG chain
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

rag_chain = (
    {"context": retriever | format_docs, "question": lambda x: x} | prompt | llm | parser
)

# 10. Ask a question
query = "I have a job interview this week what are my availabilies, we are the 16 of march 2026"
response = rag_chain.invoke(query)
print(response)