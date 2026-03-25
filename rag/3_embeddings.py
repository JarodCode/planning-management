from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# ── Reprise des étapes précédentes ───────────────────────
loader = PyPDFLoader("C1_What_is_Cognition.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# ──────────────────────────────────────────────
# Initialisation du modèle d'embedding
# Le modèle est téléchargé automatiquement depuis HuggingFace
# ──────────────────────────────────────────────

print("Chargement du modèle d'embedding…")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("Modèle chargé !\n")

# ──────────────────────────────────────────────
# Démonstration : vectoriser une phrase
# ──────────────────────────────────────────────
sample_text = "What is cognition?"
vector = embeddings.embed_query(sample_text)

print(f"Texte              : « {sample_text} »")
print(f"Dimension du vecteur: {len(vector)}")
print(f"Premiers éléments  : {[round(v, 4) for v in vector[:6]]}…")
print()

# ──────────────────────────────────────────────
# Démonstration de la similarité sémantique
# ──────────────────────────────────────────────
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

phrases = [
    "What is cognition?",          # requête de référence
    "Cognition refers to mental processes like memory and perception.",  # proche
    "The weather is sunny today.",  # éloignée
]

query_vec = embeddings.embed_query(phrases[0])
print("── Similarité cosinus avec : « What is cognition? » ──")
for phrase in phrases[1:]:
    vec = embeddings.embed_query(phrase)
    sim = cosine_similarity(query_vec, vec)
    print(f"  {sim:.3f}  →  « {phrase} »")