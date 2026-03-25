from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = "C1_What_is_Cognition.pdf"
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

# ──────────────────────────────────────────────
# Paramètres importants :
#   chunk_size    = nombre max de caractères par chunk
#   chunk_overlap = chevauchement entre chunks consécutifs (évite de couper une idée en deux)
# ──────────────────────────────────────────────

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)

docs = text_splitter.split_documents(documents)

# ──────────────────────────────────────────────
# Exploration du résultat
# ──────────────────────────────────────────────
print(f"Pages chargées     : {len(documents)}")
print(f"Chunks créés       : {len(docs)}")
print()

print("── Chunk n°0 ──────────────────────────────────────")
print(repr(docs[0].page_content))
print(f"\n   → longueur : {len(docs[0].page_content)} caractères")
print(f"   → métadonnées : {docs[0].metadata}")
print()

print("── Chunk n°1 ──────────────────────────────────────")
print(repr(docs[1].page_content[:200]), "…")

# Démonstration du chevauchement (overlap)
print()
print("── Illustration du chevauchement (overlap) ────────")
end_chunk0   = docs[0].page_content[-100:]
start_chunk1 = docs[1].page_content[:100]
print(f"Fin du chunk 0   : …{repr(end_chunk0)}")
print(f"Début du chunk 1 : {repr(start_chunk1)}…")