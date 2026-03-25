from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ── Étapes 1 à 5 ──────────────────────────────────────────
loader = PyPDFLoader("C1_What_is_Cognition.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_store = Chroma.from_documents(
    documents=docs, embedding=embeddings, collection_name="rag_collection"
)

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 2, "lambda_mult": 0.5},
)

llm    = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
parser = StrOutputParser()

prompt = PromptTemplate(
    template="""
You are an AI assistant. Use the following context to answer the question.
If the answer is not in the context, say you don't know.

Context: {context}

Question: {question}
""",
    input_variables=["context", "question"],
)

# ──────────────────────────────────────────────
# Fonction utilitaire : formate les chunks en un seul texte
# ──────────────────────────────────────────────
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# ──────────────────────────────────────────────
# Assemblage de la chaîne RAG avec l'opérateur PIPE |
#
# LCEL (LangChain Expression Language) :
#   chaque composant expose .invoke() et peut être chaîné
#   avec | comme les pipes Unix
# ──────────────────────────────────────────────
rag_chain = (
    {
        "context":  retriever | format_docs,   # retriever → texte formaté
        "question": lambda x: x,               # question passée telle quelle
    }
    | prompt    # dict → PromptValue
    | llm       # PromptValue → AIMessage
    | parser    # AIMessage → str
)

# ──────────────────────────────────────────────
# Invocation de la chaîne
# ──────────────────────────────────────────────
query = "What is the definition of 'cognition' in the document?"
print(f"Question : {query}\n")
print("Interrogation de la chaîne RAG…\n")

response = rag_chain.invoke(query)

print("Réponse :")
print("─" * 50)
print(response)
print("─" * 50)

# ──────────────────────────────────────────────
# Bonus : poser plusieurs questions d'affilée
# ──────────────────────────────────────────────
questions = [
    "What are the main components of cognition?",
    "How does attention relate to cognition?",
    "Is there a mention of artificial intelligence in the document?",
]

print("\n\n📋 Questions supplémentaires :\n")
for q in questions:
    print(f"{q}")
    ans = rag_chain.invoke(q)
    print(f"{ans}\n")