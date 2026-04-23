"""
Final Reply Node — 最终回复生成节点。

职责:
  - 接收 Worker 累积结果
  - 整理为用户友好的自然语言回答
  - 进行来源归因 (Source Attribution)
"""

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from orchestrator.state import OrchestratorState
from engine.llm_factory import get_llm_model
from engine.logging_config import get_logger

logger = get_logger(__name__)

# 缓存 LLM 实例
_REPLY_LLM = None


def _get_reply_llm():
    """获取 Final Reply 专用的 LLM 实例 (较高温度，自然表达)。"""
    global _REPLY_LLM
    if _REPLY_LLM is None:
        _REPLY_LLM = get_llm_model()
    return _REPLY_LLM


async def final_reply_node(state: OrchestratorState) -> dict:
    """
    Final_Reply: 将 Worker 结果综合成面向用户的自然语言回答。

    - 如果有 Worker 结果，综合多源信息并标注来源
    - 如果无 Worker 结果 (闲聊)，直接根据对话历史回答
    """
    llm = _get_reply_llm()

    query = state.get("query", "")
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

    # 获取对话历史
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
        )
    )

    user_msg = HumanMessage(content=query)

    try:
        response = await llm.ainvoke([system_msg] + messages + [user_msg])
        final_text = response.content if hasattr(response, "content") else str(response)

        logger.info(f"[Final_Reply] Generated response of {len(final_text)} chars")

        return {
            "final_text": final_text,
            "messages": [AIMessage(content=final_text)],
            "iterations": len(state.get("thinking_chain", [])),
            "plan_rationale": state.get("plan_rationale", ""),
            "eval_action": state.get("eval_action", ""),
            "eval_thought": state.get("eval_thought", ""),
            "agent_results": state.get("results", {}),
            "thinking_chain": state.get("thinking_chain", []),
        }

    except Exception as e:
        logger.error(f"[Final_Reply Error]: {e}")
        return {"final_text": f"生成回复时发生错误: {e}"}
