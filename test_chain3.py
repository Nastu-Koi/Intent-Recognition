from langgraph.graph import StateGraph, START, END
from typing import Dict, Any

class State(dict):
    plan: Dict[str, Any]
    results: Dict[str, Any]

def node1(state: State):
    return {"plan": {"rationale": "my rationale"}}

def node2(state: State):
    return {"results": {"data": "my data"}}

graph = StateGraph(State)
graph.add_node("n1", node1)
graph.add_node("n2", node2)
graph.add_edge(START, "n1")
graph.add_edge("n1", "n2")
graph.add_edge("n2", END)
app = graph.compile()

res = app.invoke({"plan": {}, "results": {}})
print("RES:", res)
