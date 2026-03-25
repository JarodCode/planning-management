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
vector_store = Chroma.from_documents(
    documents=docs, embedding=embeddings, collection_name="rag_collection"
)

# ──────────────────────────────────────────────
# Création du retriever (mode MMR)
#   k            = nombre de chunks retournés
#   lambda_mult  = équilibre pertinence/diversité
#                  0.0 → max diversité | 1.0 → max pertinence
# ──────────────────────────────────────────────
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 2, "lambda_mult": 0.5},
)

# ──────────────────────────────────────────────
# Test du retriever
# ──────────────────────────────────────────────
query = "What is the definition of cognition?"
print(f"Question posée : « {query} »\n")

retrieved_docs = retriever.invoke(query)

print(f"{len(retrieved_docs)} chunks récupérés par le retriever :\n")
for i, doc in enumerate(retrieved_docs):
    print(f"── Chunk {i+1} ────────────────────────────────────")
    print(doc.page_content[:300], "…")
    print(f"   -> page {doc.metadata.get('page', '?')}")
    print()

# ──────────────────────────────────────────────
# Comparaison similarity vs MMR (bonus démo)
# ──────────────────────────────────────────────

print("="*55)
print("COMPARAISON : similarity vs MMR")
print("="*55)

retriever_sim = vector_store.as_retriever(
    search_type="similarity", search_kwargs={"k": 2}
)
retriever_mmr = vector_store.as_retriever(
    search_type="mmr", search_kwargs={"k": 2, "lambda_mult": 0.1}
)

sim_docs = retriever_sim.invoke(query)
mmr_docs = retriever_mmr.invoke(query)

print("\n[Similarity] chunks récupérés :")
for d in sim_docs:
    print(f"  • {d.page_content[:80]}…")

print("\n[MMR] chunks récupérés :")
for d in mmr_docs:
    print(f"  • {d.page_content[:80]}…")