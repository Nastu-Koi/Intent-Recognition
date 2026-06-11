"""
Human Gate Node — LangGraph interrupt/resume bridge for planner HITL decisions.
"""

from copy import deepcopy
from langchain_core.messages import HumanMessage
from langgraph.types import interrupt
from orchestrator.state import OrchestratorState
from engine.logging_config import get_logger

logger = get_logger(__name__)


async def human_gate_node(state: OrchestratorState) -> dict:
    """Pause graph execution until the user approves, cancels, or supplements."""
    plan = state.get("plan") or {}
    human_gate = plan.get("human_gate") or {}
    if not human_gate.get("needs_human_input"):
        return {}

    resume_value = interrupt(
        {
            "type": "human_gate",
            "human_gate": human_gate,
            "rationale": plan.get("rationale", ""),
            "session_action_required": True,
        }
    )
    if not isinstance(resume_value, dict):
        resume_value = {"action": "supplement", "message": str(resume_value or "")}

    action = str(resume_value.get("action") or "supplement").strip().lower()
    message = str(resume_value.get("message") or "").strip()
    logger.info("[HumanGate] resumed with action=%s message=%s", action, message[:120])

    next_plan = deepcopy(plan)
    next_gate = deepcopy(human_gate)
    next_gate["needs_human_input"] = False
    next_plan["human_gate"] = next_gate

    if action in {"deny", "cancel", "no"}:
        return {
            "plan": next_plan,
            "human_gate_response": {"action": "deny", "message": message},
            "eval_action": "HUMAN_CANCELLED",
            "eval_thought": message or "用户取消了 Planner 拟定计划。",
            "final_text": "已取消这次计划，不会继续执行后续 Agent。",
        }

    normalized_action = "approve" if action in {"approve", "yes"} else "supplement"
    user_note = message or (
        "用户已确认 Planner 拟定计划，可以继续执行。"
        if normalized_action == "approve"
        else "用户补充了信息，请基于补充内容重新规划。"
    )
    update = {
        "plan": next_plan,
        "human_gate_response": {
            "action": normalized_action,
            "message": user_note,
            "gate_type": human_gate.get("gate_type", "none"),
            "previous_gate": human_gate,
        },
        "eval_action": "",
        "eval_thought": "",
        "final_text": "",
    }
    if normalized_action == "supplement":
        update.update({
            "messages": [HumanMessage(content=user_note)],
            "query": user_note,
        })
    return update
