from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
    checkpointer=InMemorySaver()
)

config = {"configurable": {"thread_id": "conversation_1"}}

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break

    response = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config
    )
    last_message = response["messages"][-1].content
    if isinstance(last_message, list):
        print("Agent:", last_message[0]["text"])
    else:
        print("Agent:", last_message)