from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.agents import create_agent
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Load existing FAISS store if it exists, otherwise create a new one
if os.path.exists("chat_history"):
    vector_store = FAISS.load_local("chat_history", embeddings, allow_dangerous_deserialization=True)
else:
    vector_store = FAISS.from_documents(
        [Document(page_content="init")],
        embeddings
    )

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

with SqliteSaver.from_conn_string("checkpoints.db") as memory:
    agent = create_agent(
        model=llm,
        tools=[get_weather],
        system_prompt="You are a helpful assistant",
        checkpointer=memory
    )

    config = {"configurable": {"thread_id": "conversation_1"}}

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            break

        response = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config
        )
        agent_response = response["messages"][-1].text

        # Store both messages in FAISS
        vector_store.add_documents([
            Document(page_content=user_input, metadata={"role": "user"}),
            Document(page_content=agent_response, metadata={"role": "assistant"})
        ])

        print("Agent:", agent_response)

    # Save FAISS to disk on exit
    vector_store.save_local("chat_history")
    print("Conversation saved!")