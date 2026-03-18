import time
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()

# 1. Load a pdf document
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("C1_What_is_Cognition.pdf")
documents = loader.load()

# 2. Split the document into smaller chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100
)
docs = text_splitter.split_documents(documents)

# 3. Create embeddings for the document chunks with rate limiting
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 4. Store the embeddings in a vector store (chroma in this case)
from langchain_chroma import Chroma
vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="rag_collection"
)

# 5. Create a retriever from the vector store
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 2, "lambda_mult": 0.5}
)

# 6. Initialize LLM
from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# 7. Create prompt template
from langchain_core.prompts import PromptTemplate
prompt = PromptTemplate(
    template="""
You are an AI assistant. Use the following context to answer the question.
If the answer is not in the context, say you don't know.
Context: {context}
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
query = "What is definition of the 'cognition' in the document?"
response = rag_chain.invoke(query)
print(response)