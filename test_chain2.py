from langgraph.graph import StateGraph, START, END
from typing import Dict, Any, List

class State(dict):
    thinking_chain: List[Dict[str, Any]]

def node1(state: State):
    return {"thinking_chain": [{"hello": "world"}]}

graph = StateGraph(State)
graph.add_node("n1", node1)
graph.add_edge(START, "n1")
graph.add_edge("n1", END)
app = graph.compile()

res = app.invoke({"thinking_chain": []})
print("RES:", res)
