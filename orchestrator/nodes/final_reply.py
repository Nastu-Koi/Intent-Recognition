"""
Final Reply Node — 最终回复生成节点。

职责:
  - 接收 Sub Agents 累积结果
  - 整理为用户友好的自然语言回答
  - 进行来源归因 (Source Attribution)
"""

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from orchestrator.state import OrchestratorState
from engine.llm_factory import get_llm_model
from engine.logging_config import get_logger
from engine.streaming import emit_stream_progress

logger = get_logger(__name__)

# 缓存 LLM 实例
_REPLY_LLM = None


def _get_reply_llm():
    """获取 Final Reply 专用的 LLM 实例 (较高温度，自然表达)。"""
    global _REPLY_LLM
    if _REPLY_LLM is None:
        _REPLY_LLM = get_llm_model()
    return _REPLY_LLM


def _messages_with_current_once(messages, query: str):
    """Return history containing the current human query at most once."""
    updated = [
        msg for msg in (messages or [])
        if not (
            isinstance(msg, HumanMessage)
            and "【Conversation Router】" in str(msg.content)
        )
    ]
    if not updated or not (
        isinstance(updated[-1], HumanMessage)
        and str(updated[-1].content).strip() == query.strip()
    ):
        updated.append(HumanMessage(content=query))
    return updated


def _display_query(state: OrchestratorState) -> str:
    """Return the user-visible query, not the router-mutated effective query."""
    route = state.get("conversation_route") or {}
    original = route.get("original_query")
    if original:
        return original

    for msg in reversed(state.get("messages", []) or []):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return state.get("query", "")


def _messages_for_history(state: OrchestratorState) -> list:
    """Persist only user-visible messages plus assistant replies."""
    return _messages_with_current_once(state.get("messages", []), _display_query(state))


async def final_reply_node(state: OrchestratorState) -> dict:
    """
    Final_Reply: 将 Sub Agents 结果综合成面向用户的自然语言回答。

    - 如果有 Sub Agents 结果，综合多源信息并标注来源
    - 如果无 Sub Agents 结果 (闲聊)，直接根据对话历史回答
    """
    llm = _get_reply_llm()

    query = state.get("query", "")
    route = state.get("conversation_route") or {}
    if route.get("relation") == "ambiguous":
        final_text = (
            state.get("final_text")
            or route.get("clarification_question")
            or "我还需要更多信息才能判断你想继续上一轮，还是开始一个新问题。可以再补充一点背景吗？"
        )
        # 保留完整的对话历史：包括当前用户问题 + 澄清问题
        updated_messages = _messages_for_history(state)
        updated_messages.append(AIMessage(content=final_text))
        
        return {
            "final_text": final_text,
            "messages": updated_messages,
            "iterations": state.get("iter", 0),
            "plan_rationale": "",
            "eval_action": "AMBIGUOUS",
            "eval_thought": route.get("rationale", ""),
            "agent_results": {},
            "thinking_chain": state.get("thinking_chain", []),
        }

    results = state.get("results", {})

    # ─── 组装 Agent 研报 ───
    if results:
        reports = "\n\n".join([
            f"--- {name} Report ---\n{content}"
            for name, content in results.items()
            if not name.startswith("_")  # 跳过内部标记字段
        ])
    else:
        reports = "No background reports generated."

    # 获取对话历史。not_related 时只看当前轮，避免旧会话污染新对话。
    if route.get("relation") == "not_related":
        messages = []
    else:
        messages = state.get("messages", [])

    # 获取可用 Agent 信息用于来源归因
    available_agents = state.get("available_agents", [])
    agent_names = {
        a["agent_id"]: a.get("name", a["agent_id"])
        for a in available_agents
    }
    source_guide = "\n".join([
        f"   - 来源为 `{aid}` ({aname}): 标注为来自该 Agent 的结果"
        for aid, aname in agent_names.items()
    ]) if agent_names else "   - 无特定来源指引"

    system_msg = SystemMessage(
        content=(
            "你是一个综合汇总与最终回答 Agent（Synthesizer & Final Responder）。\n"
            "你的任务是阅读用户的提问与对话历史，并审查所有累积的情报报告（Agent Reports）。\n"
            "### 核心规则 (严格遵守):\n"
            "1. **来源辨析 (Source Attribution)**: 你必须在回答中清晰地指明信息来源：\n"
            f"{source_guide}\n"
            "2. **时效性判定**: 参考对话历史（Messages）。如果用户当前的问题是针对特定领域的，"
            "请优先回答对应 Agent 的结果。\n"
            "3. **直接对话**: 如果『内部专家累积研报详情』为空，直接利用对话历史进行友好回复。\n"
            "4. **专业性**: 整合多轮迭代的情报，不要暴露内部任务调度的技术细节。\n\n"
            f"=== 内部专家累积研报详情 ===\n{reports}\n==============================\n"
            f"\n=== Conversation Router ===\n{route}\n===========================\n"
        )
    )

    llm_messages = _messages_with_current_once(messages, _display_query(state))

    try:
        # ─── 流式生成：逐 token 推送 SSE 事件 ───
        final_text = ""
        async for chunk in llm.astream([system_msg] + llm_messages):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            if token:
                final_text += token
                await emit_stream_progress("final_reply_token", {"token": token})

        logger.info(f"[Final_Reply] Streamed response of {len(final_text)} chars")

        # 保留完整的对话历史：之前的消息 + 当前用户输入 + AI回复
        # 注意：即使 not_related，也要保留 messages 以便后续对话能够获取上文进行对比
        updated_messages = _messages_for_history(state)
        updated_messages.append(AIMessage(content=final_text))

        logger.info(
            f"[Final_Reply] Returning {len(updated_messages)} messages: "
            f"{[(type(m).__name__, m.content[:50] if hasattr(m, 'content') else str(m)[:50]) for m in updated_messages]}"
        )

        thinking_chain = state.get("thinking_chain", [])
        result_dict = {
            "final_text": final_text,
            "messages": updated_messages,  # 保留完整历史而不是覆盖
            "thinking_chain": thinking_chain,
            "eval_action": state.get("eval_action", ""),
            "eval_thought": state.get("eval_thought", ""),
            "plan": state.get("plan", {}),
            "results": state.get("results", {}),
            "feedback_history": state.get("feedback_history", []),
            "conversation_route": state.get("conversation_route", {}),
            "iter": state.get("iter", 0),
            "iterations": state.get("iter", 0),
            "plan_rationale": state.get("plan", {}).get("rationale", ""),
            "agent_results": state.get("results", {}),
        }

        # 流式结束后，推送元信息事件（thinking_chain, iterations 等）
        await emit_stream_progress("final_reply_done", {
            "total_iterations": result_dict["iterations"],
            "plan_rationale": result_dict["plan_rationale"],
            "eval_action": result_dict["eval_action"],
            "agent_results": {
                k: str(v)[:200]
                for k, v in (result_dict["agent_results"] or {}).items()
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
        })

        return result_dict

    except Exception as e:
        logger.error(f"[Final_Reply Error]: {e}")
        messages = _messages_for_history(state)
        error_msg = f"生成回复时发生错误: {e}"
        updated_messages = list(messages) if messages else []
        updated_messages.append(AIMessage(content=error_msg))
        return {
            "final_text": error_msg,
            "messages": updated_messages,
            "thinking_chain": state.get("thinking_chain", []),
            "eval_action": state.get("eval_action", ""),
            "eval_thought": state.get("eval_thought", ""),
            "plan": state.get("plan", {}),
            "results": state.get("results", {}),
            "feedback_history": state.get("feedback_history", []),
            "conversation_route": state.get("conversation_route", {}),
            "iter": state.get("iter", 0),
        }
