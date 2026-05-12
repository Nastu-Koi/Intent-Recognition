"""
Conversation Router Node.

职责:
  - 判断当前用户输入与上一轮对话是否相关
  - 对 related 输入细分 supplement / correction / overturn
  - 生成 Context Mutation Layer 说明，供 Planner/Final Reply 使用
"""

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from engine.llm_factory import get_llm_model
from engine.logging_config import get_logger
from orchestrator.state import ConversationRouteResult, OrchestratorState

logger = get_logger(__name__)

_ROUTER_LLM = None


def _get_router_llm():
    """获取 Conversation Router 专用 LLM。"""
    global _ROUTER_LLM
    if _ROUTER_LLM is None:
        _ROUTER_LLM = get_llm_model()
    return _ROUTER_LLM


def _message_text(msg: Any) -> str:
    role = "user" if isinstance(msg, HumanMessage) else "assistant" if isinstance(msg, AIMessage) else "system"
    content = msg.content if hasattr(msg, "content") else str(msg)
    return f"{role}: {content}"


def _previous_dialogue(messages: list[Any], current_query: str) -> str:
    """提取当前输入之前的对话摘要文本。"""
    if not messages:
        return ""

    previous = list(messages)
    if previous and isinstance(previous[-1], HumanMessage):
        last_content = previous[-1].content if hasattr(previous[-1], "content") else str(previous[-1])
        if str(last_content).strip() == current_query.strip():
            previous = previous[:-1]

    if not previous:
        return ""

    return "\n".join(_message_text(msg) for msg in previous[-8:])


def _parse_router_json(text: str, current_query: str) -> ConversationRouteResult:
    clean = text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean)
        clean = re.sub(r"\s*```$", "", clean)

    try:
        raw = json.loads(clean)
    except json.JSONDecodeError:
        start = clean.find("{")
        end = clean.rfind("}")
        if start != -1 and end > start:
            raw = json.loads(clean[start:end + 1])
        else:
            raw = {
                "relation": "ambiguous",
                "related_type": "none",
                "confidence": 0.0,
                "rationale": clean,
                "clarification_question": "我还不太确定你这句话是接着上一轮说，还是要开始一个新问题。可以再补充一点背景吗？",
            }

    if not raw.get("standalone_query"):
        raw["standalone_query"] = current_query
    return ConversationRouteResult(**raw)


async def conversation_router_node(state: OrchestratorState) -> dict:
    """
    对当前输入进行会话关系打标，并生成上下文变异说明。

    输出 conversation_route:
      - relation: related / not_related / ambiguous
      - related_type: supplement / correction / overturn / none
      - context_note: 给后续模块的上下文说明
      - effective_query: 后续 Planner 应使用的问题
    """
    query = state.get("query", "")
    messages = state.get("messages", [])
    previous_dialogue = _previous_dialogue(messages, query)

    if not previous_dialogue:
        route = {
            "relation": "not_related",
            "related_type": "none",
            "confidence": 1.0,
            "rationale": "当前会话没有可参考的上一轮上下文，按新对话处理。",
            "context_note": "这是新对话，不依赖历史上下文。",
            "effective_query": query,
            "standalone_query": query,
            "clarification_question": "",
        }
        logger.info("[ConversationRouter] no previous dialogue; route=not_related")
        return {"conversation_route": route, "query": query}

    llm = _get_router_llm()
    try:
        structured_llm = llm.with_structured_output(ConversationRouteResult, method="function_calling")
    except (AttributeError, NotImplementedError):
        structured_llm = None

    system_msg = SystemMessage(
        content=(
            "你是 Conversation Router，负责判断用户新输入和上一轮会话的关系。\n\n"
            "你只能输出以下三类 relation：\n"
            "1. related: 新输入是在延续上一轮对话。\n"
            "   - supplement: 补充新条件、细节、约束或背景。\n"
            "   - correction: 修正上一轮中的事实、参数、对象或表达。\n"
            "   - overturn: 推翻上一轮目标或关键前提，需要按新意图重做。\n"
            "2. not_related: 新输入开启新话题，应作为新对话处理，不要继承上一轮任务上下文。\n"
            "3. ambiguous: 无法可靠判断用户含义，或缺少必要指代对象，应请求更多信息。\n\n"
            "判断规则：\n"
            "- 代词/省略表达（例如“这个”“刚才那个”“换成”“再加上”“不是...而是...”）通常是 related。\n"
            "- 明确的新业务、新对象、新问题，且无需上一轮才能理解，通常是 not_related。\n"
            "- 如果既可能相关也可能无关，或者用户输入太短无法执行，标为 ambiguous。\n"
            "- related 时必须生成 context_note，说明如何变异上一轮上下文。\n"
            "- ambiguous 时必须生成 clarification_question，用中文向用户索取更多信息。\n"
        )
    )
    user_msg = HumanMessage(
        content=(
            f"### 上一轮对话上下文\n{previous_dialogue}\n\n"
            f"### 用户新输入\n{query}\n\n"
            "请输出 JSON："
            "{\"relation\":\"related|not_related|ambiguous\","
            "\"related_type\":\"supplement|correction|overturn|none\","
            "\"confidence\":0.0,"
            "\"rationale\":\"...\","
            "\"context_note\":\"...\","
            "\"standalone_query\":\"...\","
            "\"clarification_question\":\"...\"}"
        )
    )

    try:
        if structured_llm is not None:
            result: ConversationRouteResult = await structured_llm.ainvoke([system_msg, user_msg])
        else:
            response = await llm.ainvoke([system_msg, user_msg])
            text = response.content if hasattr(response, "content") else str(response)
            result = _parse_router_json(text, query)
    except Exception as e:
        logger.error(f"[ConversationRouter] failed: {e}", exc_info=True)
        result = ConversationRouteResult(
            relation="ambiguous",
            related_type="none",
            confidence=0.0,
            rationale=f"Router 异常: {e}",
            standalone_query=query,
            clarification_question="我还不太确定你这句话要接着上一轮处理，还是开始一个新问题。可以补充说明一下吗？",
        )

    if result.relation != "related":
        result.related_type = "none"

    effective_query = result.standalone_query.strip() or query
    if result.relation == "related":
        effective_query = (
            f"{effective_query}\n\n"
            f"【Conversation Router】当前输入与上一轮相关，类型为 {result.related_type}。"
            f"上下文变更说明：{result.context_note or result.rationale}"
        )
    elif result.relation == "ambiguous":
        effective_query = query

    route = result.model_dump()
    route["effective_query"] = effective_query

    logger.info(
        "[ConversationRouter] relation=%s related_type=%s confidence=%.2f",
        result.relation,
        result.related_type,
        result.confidence,
    )

    update = {
        "conversation_route": route,
        "query": effective_query,
    }
    if result.relation == "ambiguous":
        clarification = result.clarification_question or "可以再补充一点信息吗？"
        update.update(
            {
                "final_text": clarification,
                "eval_action": "AMBIGUOUS",
                "eval_thought": result.rationale,
            }
        )

    return update
