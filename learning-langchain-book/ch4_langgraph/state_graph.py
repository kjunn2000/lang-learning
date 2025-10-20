from typing import Annotated, TypedDict
from langchain_ollama import OllamaLLM
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.constants import START, END


class State(TypedDict):
    messages: Annotated[list, add_messages]


builder = StateGraph(State)
model = OllamaLLM(model="phi3")


def chatbot(state: State):
    answer = model.invoke(state["messages"])
    return {"messages": [answer]}


builder.add_node(chatbot, name="chatbot")
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile()

state = State(messages=[HumanMessage(content="Hello, how are you?")])

for chunk in graph.stream(state):
    print(chunk)

# Generate the PNG
png_bytes = graph.get_graph().draw_mermaid_png()

# Save to a file
with open("graph_output.png", "wb") as f:
    f.write(png_bytes)

print("Graph saved as graph_output.png")
