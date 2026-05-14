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


def _is_internal_router_message(msg: Any) -> bool:
    content = msg.content if hasattr(msg, "content") else str(msg)
    return isinstance(msg, HumanMessage) and "【Conversation Router】" in str(content)


def _previous_dialogue(messages: list[Any], current_query: str) -> str:
    """提取当前输入之前的对话摘要文本。"""
    if not messages:
        logger.debug("[_previous_dialogue] messages list is empty")
        return ""

    logger.debug(f"[_previous_dialogue] input messages count: {len(messages)}")
    previous = [msg for msg in messages if not _is_internal_router_message(msg)]
    
    if previous and isinstance(previous[-1], HumanMessage):
        last_content = previous[-1].content if hasattr(previous[-1], "content") else str(previous[-1])
        if str(last_content).strip() == current_query.strip():
            logger.debug(f"[_previous_dialogue] filtering out current query from messages")
            previous = previous[:-1]

    logger.debug(f"[_previous_dialogue] after filtering: {len(previous)} messages remain")
    if not previous:
        logger.debug("[_previous_dialogue] no messages remain after filtering -> empty string")
        return ""

    result = "\n".join(_message_text(msg) for msg in previous[-8:])
    logger.debug(f"[_previous_dialogue] returning {len(previous[-8:])} messages")
    return result


def _previous_execution_context(state: OrchestratorState) -> str:
    """提取上一轮规划与 Agent 执行摘要，补足 messages 中缺失的路由信号。"""
    parts: list[str] = []

    plan = state.get("plan") or {}
    tasks = plan.get("tasks") or []
    if tasks:
        task_lines = []
        for task in tasks[-5:]:
            if hasattr(task, "model_dump"):
                task = task.model_dump()
            if not isinstance(task, dict):
                continue
            target = task.get("target", "")
            instruction = str(task.get("instruction", "")).replace("\n", " ")
            task_lines.append(f"- target={target}, instruction={instruction[:300]}")
        if task_lines:
            parts.append("上一轮 Planner 任务:\n" + "\n".join(task_lines))

    results = state.get("results") or {}
    if results:
        result_lines = []
        for agent_id, content in list(results.items())[-5:]:
            if str(agent_id).startswith("_"):
                continue
            preview = str(content).replace("\n", " ")[:500]
            result_lines.append(f"- agent={agent_id}, result={preview}")
        if result_lines:
            parts.append("上一轮 Agent 执行结果:\n" + "\n".join(result_lines))

    thinking_chain = state.get("thinking_chain") or []
    if thinking_chain:
        last = thinking_chain[-1] or {}
        if isinstance(last, dict):
            agent_results = last.get("agent_results") or {}
            if agent_results:
                agents = [
                    str(agent_id)
                    for agent_id in agent_results.keys()
                    if not str(agent_id).startswith("_")
                ]
                if agents:
                    parts.append("上一轮已调用 Agent: " + ", ".join(agents))

    return "\n\n".join(parts)


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
    
    logger.info(f"[ConversationRouter] START: query={query[:80]} | total_messages={len(messages)}")
    for i, msg in enumerate(messages):
        msg_type = type(msg).__name__
        msg_content = msg.content if hasattr(msg, "content") else str(msg)
        logger.info(f"  messages[{i}] {msg_type}: {msg_content[:100]}")
    
    previous_dialogue = _previous_dialogue(messages, query)
    execution_context = _previous_execution_context(state)
    previous_context = "\n\n".join(
        part for part in [previous_dialogue, execution_context] if part
    )
    logger.info(
        "[ConversationRouter] previous_dialogue length=%s execution_context length=%s total_context_length=%s",
        len(previous_dialogue),
        len(execution_context),
        len(previous_context),
    )
    if previous_dialogue:
        logger.info(f"[ConversationRouter] previous_dialogue:\n{previous_dialogue[:200]}")
    if execution_context:
        logger.info(f"[ConversationRouter] execution_context:\n{execution_context[:200]}")

    if not previous_context:
        logger.warning("[ConversationRouter] No previous context! Treating as not_related (新对话)")
        route = {
            "relation": "not_related",
            "related_type": "none",
            "confidence": 1.0,
            "rationale": "当前会话没有可参考的上一轮上下文，按新对话处理。",
            "context_note": "这是新对话，不依赖历史上下文。",
            "effective_query": query,
            "original_query": query,
            "standalone_query": query,
            "clarification_question": "",
        }
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
            "   - supplement: 补充新条件、细节、约束或背景。例如：还要加上发票、项目编码是XXX\n"
            "   - correction: 修正、纠正、调整前一轮的执行方式、工具选择、目标或参数。\n"
            "     例如：别用那个工具、用这个、改成调用XXX助手、不要这样做、应该、我改主意了\n"
            "   - overturn: 推翻上一轮的核心目标或关键前提，需要按新意图重新开始。\n"
            "     例如：算了、我不问这个了、我想问、刚才那个不重要、关键是\n"
            "2. not_related: 新输入开启新话题，应作为新对话处理，不要继承上一轮任务上下文。\n"
            "3. ambiguous: 无法可靠判断用户含义，或缺少必要指代对象，应请求更多信息。\n\n"
            "判断规则（优先级从高到低）：\n"
            "1. 如果新输入在修正、调整、改变上一轮的执行方式/工具选择/流程 → correction（related）\n"
            "   特别注意：如果用户说“用某某助手回答/处理”“别用某某助手/Agent”“换成/改用某某助手”，"
            "这是在纠正上一轮 Agent 选择或执行方式，通常应判为 related + correction。\n"
            "   此时 standalone_query 应保留用户本轮要求，并结合上一轮原始问题形成可执行问题；"
            "context_note 应明确说明需要继承上一轮问题，只调整 Agent/执行方式。\n"
            "2. 如果新输入包含代词或省略表达（例如 这个、刚才那个、换成、再加上、不是...而是）→ related\n"
            "3. 如果新输入是完全新的业务、新对象、新问题，且无需上一轮才能理解 → not_related\n"
            "4. 其他情况 → ambiguous（请求用户澄清）\n\n"
            "输出要求：\n"
            "- related 时必须生成 context_note，说明如何变异上一轮上下文\n"
            "- ambiguous 时必须生成 clarification_question，用中文向用户索取更多信息\n"
        )
    )
    user_msg = HumanMessage(
        content=(
            f"### 上一轮对话与执行上下文\n{previous_context}\n\n"
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
        raise RuntimeError(f"Conversation Router LLM 调用失败: {e}") from e

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
    route["original_query"] = query

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
