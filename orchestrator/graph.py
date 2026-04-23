"""
LangGraph 图定义与编译。

流程:
  START → planner →(有tasks)→ workers → evaluator →(PASS/PARTIAL)→ final_reply → END
                  →(无tasks)→ final_reply → END
                                            →(NEEDS_REVISION, iter<5)→ planner (循环)
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from orchestrator.state import OrchestratorState
from orchestrator.nodes.planner import planner_node
from orchestrator.nodes.workers import workers_node
from orchestrator.nodes.evaluator import evaluator_node, MAX_ITER
from orchestrator.nodes.final_reply import final_reply_node
from engine.logging_config import get_logger

logger = get_logger(__name__)


def route_after_planner(state: OrchestratorState) -> str:
    """
    Planner 之后的条件路由。

    逻辑:
    1. 存在任务 -> workers
    2. 无任务 (闲聊/直接回复) -> final_reply
    """
    plan_data = state.get("plan") or {}
    tasks = plan_data.get("tasks", [])
    if tasks:
        return "workers"

    logger.info("[Router] Planner 未生成任务，直接转入 final_reply")
    return "final_reply"


def route_after_eval(state: OrchestratorState) -> str:
    """
    Evaluator 之后的条件路由。

    路由逻辑:
    1. PASS / PARTIAL_ACCEPT → final_reply
    2. NEEDS_REVISION 且 iter < MAX_ITER → planner (重新规划)
    3. iter >= MAX_ITER (硬性熔断) → final_reply
    """
    action = state.get("eval_action", "PASS")
    current_iter = state.get("iter", 1)

    # 硬性熔断: 达到最大迭代次数 (5轮)，无论评估结果如何都强制输出
    if current_iter >= MAX_ITER:
        logger.info(
            f"[Router] 硬性熔断: iter={current_iter} >= MAX_ITER={MAX_ITER}, 强制走 final_reply"
        )
        return "final_reply"

    if action in ("PASS", "PARTIAL_ACCEPT"):
        logger.info(f"[Router] Evaluator 放行: action={action}")
        return "final_reply"

    if action == "NEEDS_REVISION":
        logger.info(
            f"[Router] 需要修正: action={action}, iter={current_iter}/{MAX_ITER}"
        )
        return "planner"

    # 未知 action，安全降级
    logger.warning(f"[Router] 未知 action='{action}'，降级走 final_reply")
    return "final_reply"


def build_graph() -> StateGraph:
    """构建并编译 Planner-Evaluator 工作流图。"""
    workflow = StateGraph(OrchestratorState)

    # ─── 注册节点 ───
    workflow.add_node("planner", planner_node)
    workflow.add_node("workers", workers_node)
    workflow.add_node("evaluator", evaluator_node)
    workflow.add_node("final_reply", final_reply_node)

    # ─── 编排边 ───
    # 1. 入口 → Planner
    workflow.add_edge(START, "planner")

    # 2. Planner →(条件路由)→ Workers 或 Final_Reply
    workflow.add_conditional_edges(
        "planner",
        route_after_planner,
        ["workers", "final_reply"]
    )

    # 3. Workers → Evaluator
    workflow.add_edge("workers", "evaluator")

    # 4. Evaluator →(条件路由)→ final_reply 或 planner (反馈循环)
    workflow.add_conditional_edges(
        "evaluator",
        route_after_eval,
        ["final_reply", "planner"]
    )

    # 5. Final_Reply → END
    workflow.add_edge("final_reply", END)

    # ─── 编译 (含 MemorySaver 支持多轮对话记忆) ───
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)
