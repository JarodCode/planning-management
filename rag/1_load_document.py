from langchain_community.document_loaders import PyPDFLoader

# ──────────────────────────────────────────────
# Chargement du PDF page par page
# PyPDFLoader crée UN Document par page
# ──────────────────────────────────────────────
PDF_PATH = "C1_What_is_Cognition.pdf"

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

# ──────────────────────────────────────────────
# Exploration du résultat
# ──────────────────────────────────────────────
print(f"Nombre de pages chargées : {len(documents)}")
print()

# Chaque Document a deux attributs clés :
first_doc = documents[0]
print("── Contenu (page_content) ─────────────────────────")
print(first_doc.page_content[:300], "…")
print()
print("── Métadonnées (metadata) ─────────────────────────")
print(first_doc.metadata)

