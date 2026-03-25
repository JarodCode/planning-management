from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ── Reprise des étapes précédentes ───────────────────────
loader = PyPDFLoader("C1_What_is_Cognition.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ──────────────────────────────────────────────
# Création du Vector Store
# from_documents() fait 3 choses en une seule ligne :
#   1. Vectorise chaque chunk (embed_documents)
#   2. Stocke les vecteurs dans Chroma
#   3. Associe le texte original à chaque vecteur
# ──────────────────────────────────────────────

print(f"Vectorisation de {len(docs)} chunks…")
vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="rag_collection",
)
print("Vector Store créé !\n")

# ──────────────────────────────────────────────
# Recherche de similarité directe (sans LLM)
# -> montre ce que le vector store "trouve"
# ──────────────────────────────────────────────
query = "What is the definition of cognition?"
print(f"Requête : « {query} »\n")

results = vector_store.similarity_search_with_score(query, k=3)

for i, (doc, score) in enumerate(results):
    print(f"── Résultat {i+1} (score de distance : {score:.4f}) ──")
    print(doc.page_content[:250], "…")
    print(f"   Source : page {doc.metadata.get('page', '?')}")
    print()