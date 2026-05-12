"""
LangGraph State — Orchestrator 全局状态定义。

State 设计原则:
- query / file_ctx / role / available_agents: 不变的输入上下文 (由 FastAPI 层注入)
- plan / results: 每轮可覆盖的动态数据
- feedback_history: 使用 operator.add 实现增量追加，用于查重熔断
- iter: 递增计数器
"""

import operator
from typing import Annotated, Optional, Dict, Any, List, Literal
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState


def _merge_file_ctx(
    old: Optional[Dict[str, Any]], new: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    Merge reducer for file_ctx.

    If the new value is None (text-only turn), keep the existing file_ctx
    from the checkpoint so files from earlier turns are not lost.
    When both exist, accumulate file lists so multi-turn uploads accumulate.
    """
    if new is None:
        return old
    if old is None:
        return new
    merged: Dict[str, Any] = {}
    for key in ("images", "documents"):
        old_list = old.get(key) or []
        new_list = new.get(key) or []
        combined = old_list + new_list
        if combined:
            merged[key] = combined
    return merged if merged else None


# ──────────────────────────────────────────────
# Pydantic Models (结构化输出约束)
# ──────────────────────────────────────────────

class TaskItem(BaseModel):
    """单条任务指令，由 Planner 生成。"""
    target: str = Field(
        description="执行目标 agent_id，必须是 available_agents 中的合法 agent_id。"
    )
    instruction: str = Field(
        description="分派给该 Agent 的具体执行指令。"
    )


class PlanOutput(BaseModel):
    """Planner 节点的结构化输出。"""
    rationale: str = Field(
        description="规划思路：分析用户需求与当前上下文，解释本轮任务分发逻辑。"
    )
    tasks: List[TaskItem] = Field(
        description="本轮需要并发执行的任务列表。空列表表示无需调度 Agent（闲聊/直接回复）。"
    )


class EvalResult(BaseModel):
    """Evaluator 节点的结构化输出 — 系统核心防死锁模型。"""
    thought: str = Field(
        description="思考过程：分析 Dispatcher 返回结果的质量、完整性，以及是否与历史反馈重复。"
    )
    action: Literal["PASS", "PARTIAL_ACCEPT", "NEEDS_REVISION"] = Field(
        description=(
            "决策枚举。PASS=完全通过；PARTIAL_ACCEPT=部分接受但足够输出；"
            "NEEDS_REVISION=需要修改（将回流 Planner 重新规划）。"
        )
    )
    feedback: str = Field(
        default="",
        description="具体修改意见。仅当 action=NEEDS_REVISION 时有意义，将追加到 feedback_history。"
    )


class ConversationRouteResult(BaseModel):
    """Conversation Router 的结构化判断结果。"""
    relation: Literal["related", "not_related", "ambiguous"] = Field(
        description="当前用户输入与上一轮对话的关系。"
    )
    related_type: Literal["supplement", "correction", "overturn", "none"] = Field(
        default="none",
        description="仅 relation=related 时使用：补充、纠正、推翻上一轮上下文。"
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="判断置信度，0 到 1。"
    )
    rationale: str = Field(
        default="",
        description="简要解释判断依据。"
    )
    context_note: str = Field(
        default="",
        description="给后续 Planner/Responder 的上下文变异说明。"
    )
    standalone_query: str = Field(
        default="",
        description="结合上下文后改写出的当前轮完整问题；not_related 时为原始新问题。"
    )
    clarification_question: str = Field(
        default="",
        description="ambiguous 时返回给用户的澄清问题。"
    )


# ──────────────────────────────────────────────
# LangGraph State (全局状态字典)
# ──────────────────────────────────────────────

class OrchestratorState(MessagesState):
    """
    Planner-Evaluator 架构的全局状态。
    继承自 MessagesState，内置 messages 字段及其 add_messages reducer。

    不可变输入上下文 (由 FastAPI 层注入):
    - query: 用户当前输入文本
    - file_ctx: 文件上下文 {images: [...], documents: [...]}
    - role: 用户角色 (RBAC)
    - available_agents: 经 RBAC 过滤后的可用 Agent 描述列表

    动态数据 (每轮覆盖/累加):
    - plan: Planner 生成的任务计划 (Dict)
    - results: Dispatcher 并发结果 (自动合并 via operator.ior)
    - iter: 当前迭代轮次 (自动累加 via operator.add)
    - feedback_history: Evaluator 的增量反馈记录 (自动追加 via operator.add)
    - eval_action: Evaluator 的最新 action
    - eval_thought: Evaluator 的最新思考过程
    - final_text: Final_Reply 的输出文本
    """

    # 不可变输入上下文
    query: str                                                  # 当前轮次用户输入
    file_ctx: Annotated[Optional[Dict[str, Any]], _merge_file_ctx]  # 文件上下文 (合并 reducer，多轮不丢失)
    role: str                                                   # 用户角色
    available_agents: List[Dict[str, Any]]                      # 可用 Agent 描述列表

    # 动态数据
    plan: Dict[str, Any]                                        # Planner 输出的任务计划
    results: Dict[str, Any]                                     # Dispatcher 并发结果 (明确管理)
    _agent_outputs: Dict[str, Any]                              # SubAgent 结构化输出
    iter: int                                                   # 当前迭代轮次
    feedback_history: List[str]                                 # 增量反馈历史
    eval_action: str                                            # Evaluator 的最新 action
    eval_thought: str                                           # Evaluator 的最新思考过程
    final_text: str                                             # Final_Reply 的输出
    thinking_chain: List[Dict[str, Any]]                        # 完整思维链历史
    conversation_route: Dict[str, Any]                           # Conversation Router 打标结果
