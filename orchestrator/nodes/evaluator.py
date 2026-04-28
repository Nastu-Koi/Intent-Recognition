"""
Evaluator Node — 动态评估节点 (系统核心)。

职责:
  - 使用 LLM 评估 Worker 结果
  - 内嵌查重熔断与动态宽容度等防死锁逻辑
  - 5 轮硬性熔断上限
"""

from langchain_core.messages import SystemMessage, HumanMessage
from orchestrator.state import OrchestratorState, EvalResult
from engine.llm_factory import get_llm_model
from engine.logging_config import get_logger

logger = get_logger(__name__)

# 最大迭代轮次 (5轮上限，超过直接强制进入 final_reply)
MAX_ITER = 5

# 缓存 LLM 实例
_EVALUATOR_LLM = None


def _get_evaluator_llm():
    """获取 Evaluator 专用的 LLM 实例。"""
    global _EVALUATOR_LLM
    if _EVALUATOR_LLM is None:
        _EVALUATOR_LLM = get_llm_model()
    return _EVALUATOR_LLM


async def evaluator_node(state: OrchestratorState) -> dict:
    """
    Evaluator: 评估 Worker 执行结果，决定是否需要重新规划。

    防死锁机制:
    1. 查重熔断: 检测新反馈是否与 feedback_history 中的历史反馈高度重复
    2. 动态宽容度: 接近 MAX_ITER 时放宽标准至 70% 完成即可放行
    3. 硬性熔断: iter >= MAX_ITER 时 graph 层面强制走 Final_Reply (不经过本节点判断)
    """
    llm = _get_evaluator_llm()

    # 尝试结构化输出
    try:
        structured_llm = llm.with_structured_output(EvalResult, method="function_calling")
    except (AttributeError, NotImplementedError):
        structured_llm = None

    query = state.get("query", "")
    results = state.get("results", {})
    current_iter = state.get("iter", 1)
    feedback_history = state.get("feedback_history", [])

    # ─── 构建 Worker 结果展示 ───
    if results:
        results_block = "\n".join(
            [f"  ### {k}\n  {str(v)}" for k, v in results.items()]
        )
    else:
        results_block = "（无 Worker 返回结果）"

    # ─── 构建反馈历史 ───
    if feedback_history:
        history_block = "\n".join(
            [f"  - 第{i+1}次: {fb}" for i, fb in enumerate(feedback_history)]
        )
    else:
        history_block = "（本轮为首次评估，无历史反馈）"

    # ─── 构建文件上下文 ───
    file_ctx = state.get("file_ctx") or {}
    file_summary = []
    if "images" in file_ctx and file_ctx["images"]:
        names = [f.get("file_name", "unknown") for f in file_ctx["images"]]
        file_summary.append(f"{len(names)} 张图片 [{', '.join(names)}]")
    if "documents" in file_ctx and file_ctx["documents"]:
        names = [f.get("file_name", "unknown") for f in file_ctx["documents"]]
        file_summary.append(f"{len(names)} 份文档 [{', '.join(names)}]")
    file_str = "; ".join(file_summary) if file_summary else "无可用文件"

    # ─── 动态宽容度 ───
    tolerance_note = ""
    if current_iter >= MAX_ITER - 1:
        tolerance_note = (
            "\n⚠️ **注意**：当前已接近最大迭代次数限制。"
            "请适当放宽评估标准 — 只要核心任务已完成（≥70%），即可选择 PARTIAL_ACCEPT 放行。\n"
        )

    # 包含完整的对话历史
    messages = state.get("messages", [])

    system_msg = SystemMessage(
        content=(
            "你是一个多智能体系统的质量评估师，专职评估 Worker 的执行结果。\n"
            "你**绝不**进行任务规划或调度 — 那是 Planner 的职责。\n\n"

            "### 评估上下文:\n"
            "1. **对话历史 (Messages)**: 用户的原始意图与背景。\n"
            f"2. **当前环境文件**: {file_str}\n"
            f"3. **历史评估反馈**: \n{history_block}\n"
            f"4. **Worker 累积执行结果**: \n{results_block}\n\n"
            f"{tolerance_note}"

            "### 评估规则 (严格遵守):\n"
            "1. **整体目标达成判定**: 审查『Worker 累积执行结果』。由于系统采用多轮迭代规划，之前的结果都在这个集合中。"
            "只要全量结果集已经满足了用户问题的各个方面，就应判定为 PASS。\n"
            "2. **闲聊与直接对话**: 如果 Planner 判定这是一个通用问题且**未调度任何任务**（累积执行结果为空），"
            "请结合对话历史判断 Responder 是否可以直接回答。如果是，则判定为 PASS。\n"
            "3. **查重熔断**: 如果你准备提出的修改意见与「历史评估反馈」高度重复，说明 Worker 已达能力瓶颈，"
            "此时**必须**选择 PARTIAL_ACCEPT。\n"
            "4. **决策标准**:\n"
            "   - PASS: 累积结果已完整解答了用户提问的所有维度，或者是无需 Worker 的直接对话。\n"
            "   - PARTIAL_ACCEPT: 核心任务已完成，或由于能力限制无法通过简单修正获得更好结果。\n"
            "   - NEEDS_REVISION: 存在实质性错误或遗漏，且修改建议与历史不重复。\n\n"
            "5. **feedback 字段**: 仅在 NEEDS_REVISION 时填写具体的修改建议。PASS 和 PARTIAL_ACCEPT 时留空。\n"
        )
    )

    try:
        if structured_llm is not None:
            eval_result: EvalResult = await structured_llm.ainvoke(
                [system_msg] + messages + [
                    HumanMessage(content="请根据对话历史与已累积的执行结果，评估整体任务是否已完成。")
                ]
            )
        else:
            # 降级：原始 LLM + JSON 解析
            import json
            import re

            response = await llm.ainvoke(
                [system_msg] + messages + [
                    HumanMessage(
                        content=(
                            "请根据对话历史与已累积的执行结果，评估整体任务是否已完成。\n"
                            "输出 JSON: {\"thought\": \"...\", \"action\": \"PASS|PARTIAL_ACCEPT|NEEDS_REVISION\", \"feedback\": \"...\"}"
                        )
                    )
                ]
            )
            text = response.content if hasattr(response, "content") else str(response)
            text = text.strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*", "", text)
                text = re.sub(r"\s*```$", "", text)
            try:
                raw = json.loads(text)
            except json.JSONDecodeError:
                start = text.find("{")
                end = text.rfind("}")
                if start != -1 and end > start:
                    raw = json.loads(text[start:end + 1])
                else:
                    raw = {"thought": text, "action": "PASS", "feedback": ""}

            eval_result = EvalResult(
                thought=raw.get("thought", ""),
                action=raw.get("action", "PASS"),
                feedback=raw.get("feedback", "")
            )

        logger.info(
            f"[Evaluator] Action={eval_result.action} | "
            f"Thought={eval_result.thought[:100]}... | "
            f"Iter={current_iter}/{MAX_ITER}"
        )

        update = {
            "eval_action": eval_result.action,
            "eval_thought": eval_result.thought,
        }

        # 累积思维链历史 (自行管理追加)
        thinking_chain = list(state.get("thinking_chain") or [])
        current_thinking = {
            "iteration": current_iter,
            "plan_rationale": state.get("plan", {}).get("rationale", ""),
            "eval_action": eval_result.action,
            "eval_thought": eval_result.thought,
            "agent_results": results.copy() if isinstance(results, dict) else results,
        }
        thinking_chain.append(current_thinking)
        update["thinking_chain"] = thinking_chain

        # 仅在 NEEDS_REVISION 时追加反馈到历史 (自行管理追加)
        feedback_history = list(state.get("feedback_history") or [])
        if eval_result.action == "NEEDS_REVISION" and eval_result.feedback:
            feedback_history.append(eval_result.feedback)
        update["feedback_history"] = feedback_history

        return update

    except Exception as e:
        logger.error(f"[Evaluator Error]: {e}")
        # 降级：直接放行，防止评估失败阻塞管线
        return {
            "eval_action": "PASS",
            "eval_thought": f"评估异常，降级放行: {e}",
            "feedback_history": [],
        }
