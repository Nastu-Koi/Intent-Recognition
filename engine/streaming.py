"""
流式输出工具模块 — Server-Sent Events (SSE) 支持。

职责:
  - 将 LangGraph 的状态更新转换为 SSE 事件流
  - 实时发送思维链的每个阶段信息
  - 支持多种事件类型 (planner, dispatcher, evaluator, final_reply)
"""

import json
from typing import AsyncGenerator, Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage
from engine.logging_config import get_logger

logger = get_logger(__name__)


class StreamEvent:
    """流式事件数据结构。"""
    
    def __init__(self, event_type: str, data: Dict[str, Any]):
        self.event_type = event_type
        self.data = data
    
    def to_sse_format(self) -> str:
        """转换为 SSE 格式。"""
        lines = [f"event: {self.event_type}"]
        lines.append(f"data: {json.dumps(self.data, ensure_ascii=False)}")
        lines.append("")
        lines.append("")
        return "\n".join(lines)


async def stream_orchestrator_graph(
    graph,
    initial_state: Dict[str, Any],
    config: Dict[str, Any],
) -> AsyncGenerator[str, None]:
    """
    使用 astream 流式执行 LangGraph，并转换为 SSE 格式。
    
    Args:
        graph: 编译后的 LangGraph
        initial_state: 初始状态
        config: LangGraph 配置（含 thread_id, recursion_limit）
    
    Yields:
        SSE 格式的流数据
    """
    
    try:
        # 发送开始事件
        yield StreamEvent(
            "start",
            {
                "message": "开始执行思维链",
                "timestamp": None,
            }
        ).to_sse_format()
        
        last_node = None
        iteration_count = 0
        
        # 使用 astream 获取每一步的状态更新
        async for event in graph.astream(initial_state, config=config):
            # event 是一个字典 {node_name: node_output}
            for node_name, node_output in event.items():
                
                # 检测迭代
                if node_name == "planner":
                    iteration_count = node_output.get("iter", 0)
                    
                    plan = node_output.get("plan", {})
                    plan_tasks = plan.get("tasks", [])
                    
                    # 发送 planner 事件
                    yield StreamEvent(
                        "planner",
                        {
                            "iteration": iteration_count,
                            "rationale": plan.get("rationale", ""),
                            "tasks_count": len(plan_tasks),
                            "tasks": [
                                {
                                    "target": task.get("target", task.target if hasattr(task, "target") else ""),
                                    "instruction": task.get("instruction", task.instruction if hasattr(task, "instruction") else ""),
                                }
                                for task in plan_tasks
                            ],
                        }
                    ).to_sse_format()
                    
                    last_node = "planner"
                
                elif node_name == "dispatcher":
                    results = node_output.get("results", {})
                    
                    # 发送 dispatcher 事件 - 显示每个 Agent 的结果
                    for agent_id, result_text in results.items():
                        if not agent_id.startswith("_"):
                            yield StreamEvent(
                                "agent_result",
                                {
                                    "agent_id": agent_id,
                                    "result_preview": result_text[:200] if isinstance(result_text, str) else str(result_text)[:200],
                                }
                            ).to_sse_format()
                    
                    yield StreamEvent(
                        "dispatcher",
                        {
                            "agents_count": len([k for k in results.keys() if not k.startswith("_")]),
                            "completed": True,
                        }
                    ).to_sse_format()
                    
                    last_node = "dispatcher"
                
                elif node_name == "evaluator":
                    eval_action = node_output.get("eval_action", "")
                    eval_thought = node_output.get("eval_thought", "")
                    current_iter = node_output.get("iter", 0)
                    
                    # 发送 evaluator 事件
                    yield StreamEvent(
                        "evaluator",
                        {
                            "action": eval_action,
                            "thought": eval_thought[:300] if eval_thought else "",
                            "iteration": current_iter,
                            "max_iterations": 5,
                        }
                    ).to_sse_format()
                    
                    last_node = "evaluator"
                
                elif node_name == "final_reply":
                    final_text = node_output.get("final_text", "")
                    thinking_chain = node_output.get("thinking_chain", [])
                    
                    # 发送 final_reply 事件
                    yield StreamEvent(
                        "final_reply",
                        {
                            "answer": final_text,
                            "total_iterations": node_output.get("iterations", 0),
                            "plan_rationale": node_output.get("plan_rationale", ""),
                            "eval_action": node_output.get("eval_action", ""),
                            "agent_results": {
                                k: str(v)[:200]  # 截断较长的结果
                                for k, v in node_output.get("agent_results", {}).items()
                                if not k.startswith("_")
                            },
                            "thinking_chain": [
                                {
                                    "iteration": item.get("iteration", 0),
                                    "plan_rationale": item.get("plan_rationale", ""),
                                    "eval_action": item.get("eval_action", ""),
                                    "eval_thought": item.get("eval_thought", ""),
                                }
                                for item in thinking_chain
                            ],
                        }
                    ).to_sse_format()
                    
                    last_node = "final_reply"
        
        # 发送完成事件
        yield StreamEvent(
            "done",
            {
                "message": "思维链执行完成",
                "status": "success",
            }
        ).to_sse_format()
        
        logger.info(f"[Stream] Graph execution completed. Nodes visited: {last_node}")
    
    except Exception as e:
        logger.error(f"[Stream] Error during graph execution: {e}", exc_info=True)
        yield StreamEvent(
            "error",
            {
                "message": str(e),
                "status": "error",
            }
        ).to_sse_format()


def format_sse_response(event_type: str, data: Dict[str, Any]) -> str:
    """
    格式化单个 SSE 响应。
    
    格式:
        event: <event_type>
        data: <json_data>
        
        (空行)
    """
    return StreamEvent(event_type, data).to_sse_format()
