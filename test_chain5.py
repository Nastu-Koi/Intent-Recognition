import asyncio
from main import _get_graph
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError

async def main():
    graph = _get_graph()
    initial_state = {
        "messages": [HumanMessage(content="Upload this file")],
        "query": "Upload this file",
        "file_ctx": {
            "images": [{"file_path": "test.png", "file_name": "test.png", "file_type": "image"}]
        },
        "role": "admin",
        "available_agents": [
            {"agent_id": "dify_file_uploader", "name": "File Uploader", "description": "Upload files"}
        ],
        "plan": {},
        "results": {},
        "iter": 0,
        "feedback_history": [],
        "eval_action": "",
        "eval_thought": "",
        "final_text": "",
        "thinking_chain": [],
    }
    
    config = {"configurable": {"thread_id": "test_129"}, "recursion_limit": 5}
    try:
        result = await graph.ainvoke(initial_state, config=config)
    except GraphRecursionError:
        print("RECURSION ERROR")
        # get the current state
        state_config = graph.get_state(config)
        result = state_config.values
    
    print("THINKING CHAIN LENGTH:", len(result.get("thinking_chain", [])))
    for item in result.get("thinking_chain", []):
        print(f"Iter {item.get('iteration')}: Rationale = {item.get('plan_rationale')}")
        print(f"  Eval Action: {item.get('eval_action')}")

if __name__ == "__main__":
    asyncio.run(main())
