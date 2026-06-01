"""
流式输出工具模块 — Server-Sent Events (SSE) 支持。

职责:
  - 将 LangGraph 的状态更新转换为 SSE 事件流
  - 实时发送思维链的每个阶段信息
  - 支持多种事件类型 (planner, dispatcher, evaluator, final_reply)
"""

import asyncio
import contextlib
import contextvars
import json
from typing import AsyncGenerator, Dict, Any, List
from langchain_core.messages import HumanMessage, AIMessage
from engine.logging_config import get_logger

logger = get_logger(__name__)

_progress_queue: contextvars.ContextVar[asyncio.Queue | None] = contextvars.ContextVar(
    "stream_progress_queue",
    default=None,
)


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


async def emit_stream_progress(event_type: str, data: Dict[str, Any]) -> None:
    """Emit an in-flight progress event from inside a graph node."""
    queue = _progress_queue.get()
    if queue is not None:
        await queue.put(StreamEvent(event_type, data).to_sse_format())


async def _yield_graph_event(
    event: Dict[str, Any],
    agent_name_map: Dict[str, str],
) -> AsyncGenerator[tuple[str, str | None], None]:
    """Convert one LangGraph update into SSE events."""
    for node_name, node_output in event.items():
        if node_name == "conversation_router":
            route = node_output.get("conversation_route", {}) if isinstance(node_output, dict) else {}
            yield StreamEvent(
                "conversation_router",
                {
                    "relation": route.get("relation", ""),
                    "related_type": route.get("related_type", "none"),
                    "confidence": route.get("confidence", 0),
                    "rationale": route.get("rationale", ""),
                    "context_note": route.get("context_note", ""),
                    "clarification_question": route.get("clarification_question", ""),
                }
            ).to_sse_format(), "conversation_router"

        elif node_name == "planner":
            iteration_count = node_output.get("iter", 0)

            plan = node_output.get("plan", {})
            plan_tasks = plan.get("tasks", [])

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
            ).to_sse_format(), "planner"

        elif node_name == "dispatcher":
            results = node_output.get("results", {})

            for agent_id, result_text in results.items():
                if not agent_id.startswith("_"):
                    agent_name = agent_name_map.get(agent_id, agent_id)
                    logger.info(f"[Streaming] Sending agent_result: agent_id={agent_id}, agent_name={agent_name}")
                    yield StreamEvent(
                        "agent_result",
                        {
                            "agent_id": agent_id,
                            "agent_name": agent_name,
                            "result_preview": result_text[:200] if isinstance(result_text, str) else str(result_text)[:200],
                        }
                    ).to_sse_format(), "dispatcher"

            yield StreamEvent(
                "dispatcher",
                {
                    "agents_count": len([k for k in results.keys() if not k.startswith("_")]),
                    "completed": True,
                }
            ).to_sse_format(), "dispatcher"

        elif node_name == "evaluator":
            eval_action = node_output.get("eval_action", "")
            eval_thought = node_output.get("eval_thought", "")
            current_iter = node_output.get("iter", 0)

            yield StreamEvent(
                "evaluator",
                {
                    "action": eval_action,
                    "thought": eval_thought[:300] if eval_thought else "",
                    "iteration": current_iter,
                    "max_iterations": 5,
                }
            ).to_sse_format(), "evaluator"

        elif node_name == "final_reply":
            final_text = node_output.get("final_text", "")
            thinking_chain = node_output.get("thinking_chain", [])

            yield StreamEvent(
                "final_reply",
                {
                    "answer": final_text,
                    "streamed": True,  # 标记: 内容已通过 final_reply_token 逐 token 推送
                    "total_iterations": node_output.get("iterations", 0),
                    "plan_rationale": node_output.get("plan_rationale", ""),
                    "eval_action": node_output.get("eval_action", ""),
                    "agent_results": {
                        k: str(v)
                        for k, v in node_output.get("agent_results", {}).items()
                        if not k.startswith("_")
                    },
                    "thinking_chain": [
                        {
                            "iteration": item.get("iteration", 0),
                            "plan_rationale": item.get("plan_rationale", ""),
                            "agent_results": {
                                k: str(v)
                                for k, v in (item.get("agent_results") or {}).items()
                                if not k.startswith("_")
                            },
                            "agent_names": {
                                k: str(v)
                                for k, v in (item.get("agent_names") or {}).items()
                                if not k.startswith("_")
                            },
                            "eval_action": item.get("eval_action", ""),
                            "eval_thought": item.get("eval_thought", ""),
                        }
                        for item in thinking_chain
                    ],
                }
            ).to_sse_format(), "final_reply"


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
        
        # 从 initial_state 获取可用 agents 映射 (agent_id -> agent info)
        available_agents = initial_state.get("available_agents", [])
        agent_name_map = {agent["agent_id"]: agent.get("name", agent["agent_id"]) for agent in available_agents}
        logger.info(f"[Streaming] agent_name_map created: {agent_name_map}")
        
        queue: asyncio.Queue = asyncio.Queue()
        token = _progress_queue.set(queue)
        graph_done = False
        graph_iter = graph.astream(initial_state, config=config).__aiter__()
        graph_task = asyncio.create_task(graph_iter.__anext__())
        queue_task = asyncio.create_task(queue.get())

        try:
            while not graph_done:
                done, _ = await asyncio.wait(
                    [graph_task, queue_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if queue_task in done:
                    yield queue_task.result()
                    queue_task = asyncio.create_task(queue.get())

                if graph_task in done:
                    try:
                        event = graph_task.result()
                    except StopAsyncIteration:
                        graph_done = True
                        continue

                    async for event_data, node_name in _yield_graph_event(event, agent_name_map):
                        yield event_data
                        if node_name:
                            last_node = node_name

                    graph_task = asyncio.create_task(graph_iter.__anext__())

            while not queue.empty():
                yield queue.get_nowait()
        finally:
            for task in (graph_task, queue_task):
                if task and not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
            _progress_queue.reset(token)
        
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
