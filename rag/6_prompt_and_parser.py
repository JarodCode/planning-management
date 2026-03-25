from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# ──────────────────────────────────────────────
# 1. Prompt Template
# ──────────────────────────────────────────────

prompt = PromptTemplate(
    template="""
You are an AI assistant. Use the following context to answer the question.
If the answer is not in the context, say you don't know.

Context: {context}

Question: {question}
""",
    input_variables=["context", "question"],
)

# Visualiser le prompt formaté (sans LLM)
context_exemple = "Cognition refers to mental processes including memory, attention, and language."
question_exemple = "What is cognition?"

formatted = prompt.format(context=context_exemple, question=question_exemple)
print("── Prompt formaté (ce que le LLM va recevoir) ─────")
print(formatted)
print()

# ──────────────────────────────────────────────
# 2. LLM + Output Parser
# ──────────────────────────────────────────────
llm    = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
parser = StrOutputParser()

# Appel direct (sans retriever) pour montrer le rôle du parser
print("── Réponse brute du LLM (AIMessage) ───────────────")
raw_response = llm.invoke(formatted)
print(type(raw_response))   # <class 'langchain_core.messages.ai.AIMessage'>
print(raw_response)
print()

print("── Réponse après StrOutputParser (str) ────────────")
clean_response = parser.invoke(raw_response)
print(type(clean_response)) # <class 'str'>
print(clean_response)