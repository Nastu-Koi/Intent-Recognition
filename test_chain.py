import asyncio
import logging
logging.basicConfig(level=logging.INFO)
from main import _get_graph
from langchain_core.messages import HumanMessage

async def main():
    graph = _get_graph()
    initial_state = {
        "messages": [HumanMessage(content="Hello")],
        "query": "Hello",
        "role": "admin",
        "available_agents": [],
        "plan": {},
        "results": {},
        "iter": 0,
        "feedback_history": [],
        "eval_action": "",
        "eval_thought": "",
        "final_text": "",
        "thinking_chain": [],
    }
    
    config = {"configurable": {"thread_id": "test_125"}, "recursion_limit": 5}
    result = await graph.ainvoke(initial_state, config=config)
    print("THINKING CHAIN LENGTH:", len(result.get("thinking_chain", [])))
    print("ITERATIONS:", result.get("iter"))
    print("PLAN:", result.get("plan"))

if __name__ == "__main__":
    asyncio.run(main())
