import ast
from typing import Annotated, TypedDict
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_ollama import ChatOllama
from langgraph.graph import START, add_messages, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.schema import HumanMessage


@tool(
    description="""
    Evaluate an expression node or a string containing only a Python
    expression.  The string or node provided may only consist of the following
    Python literal structures: strings, bytes, numbers, tuples, lists, dicts,
    sets, booleans, and None.
"""
)
def calculator(query: str):
    return ast.literal_eval(query)


search = DuckDuckGoSearchRun()
tools = [search, calculator]

model = ChatOllama(model="phi3").bind_tools(tools)


class State(TypedDict):
    messages: Annotated[list, add_messages]


def model_node(state: State) -> State:
    res = model.invoke(state["messages"])
    return {"messages": res}


builder = StateGraph(State)
builder.add_node("model", model_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "model")
builder.add_conditional_edges("model", tools_condition)
builder.add_edge("tools", "model")

graph = builder.compile()

input = {
    "messages": [
        HumanMessage(
            """
                How old was the 30th president of the United States when he died?
            """
        )
    ]
}

graph.invoke(input)
