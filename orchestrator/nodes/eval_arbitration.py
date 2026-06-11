"""
Evaluator Arbitration Node — LangGraph interrupt/resume bridge for low-confidence evaluation decisions.
"""

from langchain_core.messages import HumanMessage
from langgraph.types import interrupt

from orchestrator.state import OrchestratorState
from engine.logging_config import get_logger

logger = get_logger(__name__)


async def eval_arbitration_node(state: OrchestratorState) -> dict:
    """Pause after Evaluator when its decision needs human arbitration."""
    arbitration = state.get("eval_arbitration") or {}
    if not arbitration.get("needs_human_arbitration"):
        return {}

    resume_value = interrupt(
        {
            "type": "eval_arbitration",
            "eval_arbitration": arbitration,
            "eval_action": state.get("eval_action", ""),
            "eval_thought": state.get("eval_thought", ""),
            "session_action_required": True,
        }
    )
    if not isinstance(resume_value, dict):
        resume_value = {"action": "override_with_feedback", "message": str(resume_value or "")}

    action = str(resume_value.get("action") or "").strip().lower()
    message = str(resume_value.get("message") or "").strip()
    logger.info("[EvalArbitration] resumed with action=%s message=%s", action, message[:120])

    next_arbitration = {**arbitration, "needs_human_arbitration": False, "human_action": action}
    if message:
        next_arbitration["human_feedback"] = message

    if action == "accept_evaluation":
        feedback_history = list(state.get("feedback_history") or [])
        feedback = str(arbitration.get("feedback") or "").strip()
        if state.get("eval_action") == "NEEDS_REVISION" and feedback and feedback not in feedback_history:
            feedback_history.append(feedback)
        return {
            "eval_arbitration": next_arbitration,
            "feedback_history": feedback_history,
        }

    if action == "force_final":
        return {
            "eval_action": "PARTIAL_ACCEPT",
            "eval_thought": f"用户仲裁要求直接出答案。原评估：{state.get('eval_thought', '')}",
            "eval_arbitration": next_arbitration,
        }

    if action == "override_with_feedback":
        feedback = message or "用户要求根据人工仲裁意见继续修正。"
        feedback_history = list(state.get("feedback_history") or [])
        feedback_history.append(feedback)
        return {
            "eval_action": "NEEDS_REVISION",
            "eval_thought": f"用户人工仲裁要求修正：{feedback}",
            "feedback_history": feedback_history,
            "messages": [HumanMessage(content=feedback)],
            "eval_arbitration": next_arbitration,
        }

    return {
        "eval_action": "PARTIAL_ACCEPT",
        "eval_thought": "用户仲裁动作无法识别，降级为直接出答案。",
        "eval_arbitration": next_arbitration,
    }
