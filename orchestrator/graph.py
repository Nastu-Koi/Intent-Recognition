"""
LangGraph 图定义与编译。

流程:
  START → conversation_router → context mutation → planner →(有tasks)→ dispatcher → sub agents→ evaluator →(PASS/PARTIAL)→ final_reply → END
                  →(无tasks)→ final_reply → END
                                            →(NEEDS_REVISION, iter<5)→ planner (循环)
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from orchestrator.state import OrchestratorState
from orchestrator.nodes.conversation_router import conversation_router_node
from orchestrator.nodes.planner import planner_node
from orchestrator.nodes.human_gate import human_gate_node
from orchestrator.nodes.dispatcher import dispatcher_node
from orchestrator.nodes.evaluator import evaluator_node, MAX_ITER
from orchestrator.nodes.eval_arbitration import eval_arbitration_node
from orchestrator.nodes.final_reply import final_reply_node
from engine.logging_config import get_logger

logger = get_logger(__name__)


def route_after_conversation_router(state: OrchestratorState) -> str:
    """Conversation Router 之后的条件路由。"""
    route = state.get("conversation_route") or {}
    relation = route.get("relation")
    if relation == "ambiguous":
        logger.info("[Router] Conversation ambiguous, asking user for more info")
        return "final_reply"
    return "planner"


def route_after_planner(state: OrchestratorState) -> str:
    """
    Planner 之后的条件路由。

    逻辑:
    1. Human gate 需要用户参与 -> human_gate interrupt
    2. 存在任务 -> dispatcher
    3. 无任务 (闲聊/直接回复) -> final_reply
    """
    plan_data = state.get("plan") or {}
    human_gate = plan_data.get("human_gate") or {}
    if human_gate.get("needs_human_input"):
        logger.info("[Router] Planner human gate triggered, interrupting before execution")
        return "human_gate"

    tasks = plan_data.get("tasks", [])
    if tasks:
        return "dispatcher"

    logger.info("[Router] Planner 未生成任务，直接转入 final_reply")
    return "final_reply"


def route_after_human_gate(state: OrchestratorState) -> str:
    """Route after user resumes the human gate."""
    if state.get("eval_action") == "HUMAN_CANCELLED":
        return "final_reply"
    return "planner"


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

    arbitration = state.get("eval_arbitration") or {}
    if arbitration.get("needs_human_arbitration"):
        logger.info("[Router] Evaluator 低置信度，进入人工仲裁")
        return "eval_arbitration"

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


def route_after_eval_arbitration(state: OrchestratorState) -> str:
    """Route after human arbitration of an evaluator decision."""
    action = state.get("eval_action", "PASS")
    current_iter = state.get("iter", 1)
    if action == "NEEDS_REVISION" and current_iter < MAX_ITER:
        logger.info("[Router] 用户仲裁要求重新规划")
        return "planner"
    logger.info("[Router] 用户仲裁后直接出答案")
    return "final_reply"


def build_graph(checkpointer=None) -> StateGraph:
    """构建并编译 Planner-Evaluator 工作流图。"""
    workflow = StateGraph(OrchestratorState)

    # ─── 注册节点 ───
    workflow.add_node("conversation_router", conversation_router_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("human_gate", human_gate_node)
    workflow.add_node("dispatcher", dispatcher_node)
    workflow.add_node("evaluator", evaluator_node)
    workflow.add_node("eval_arbitration", eval_arbitration_node)
    workflow.add_node("final_reply", final_reply_node)

    # ─── 编排边 ───
    # 1. 入口 → Conversation Router
    workflow.add_edge(START, "conversation_router")

    # 2. Conversation Router → Planner 或 Final_Reply (ambiguous)
    workflow.add_conditional_edges(
        "conversation_router",
        route_after_conversation_router,
        ["planner", "final_reply"]
    )

    # 3. Planner →(条件路由)→ Dispatcher 或 Final_Reply
    workflow.add_conditional_edges(
        "planner",
        route_after_planner,
        ["human_gate", "dispatcher", "final_reply"]
    )

    # 4. Human_Gate → Planner 或 Final_Reply
    workflow.add_conditional_edges(
        "human_gate",
        route_after_human_gate,
        ["planner", "final_reply"]
    )

    # 5. Dispatcher → Evaluator
    workflow.add_edge("dispatcher", "evaluator")

    # 6. Evaluator →(条件路由)→ final_reply、planner 或人工仲裁
    workflow.add_conditional_edges(
        "evaluator",
        route_after_eval,
        ["eval_arbitration", "final_reply", "planner"]
    )

    # 7. Eval_Arbitration → final_reply 或 planner
    workflow.add_conditional_edges(
        "eval_arbitration",
        route_after_eval_arbitration,
        ["final_reply", "planner"]
    )

    # 8. Final_Reply → END
    workflow.add_edge("final_reply", END)

    # ─── 编译 ───
    return workflow.compile(checkpointer=checkpointer)
