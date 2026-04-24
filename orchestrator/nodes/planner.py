"""
Planner Node — 动态感知 Agent Cards 的规划节点。

职责:
  - 分析用户需求 + 文件上下文 + 历史反馈
  - 根据 available_agents（经 RBAC 过滤）生成任务分发计划
  - 绝不进行任何结果评估或质检
"""

import json
from langchain_core.messages import SystemMessage, HumanMessage
from orchestrator.state import OrchestratorState, PlanOutput
from engine.llm_factory import get_llm_model
from engine.logging_config import get_logger

logger = get_logger(__name__)

# 缓存 LLM 实例
_PLANNER_LLM = None


def _get_planner_llm():
    """获取 Planner 专用的 LLM 实例 (低温度，精准规划)。"""
    global _PLANNER_LLM
    if _PLANNER_LLM is None:
        _PLANNER_LLM = get_llm_model()
    return _PLANNER_LLM


def _build_agent_catalog(available_agents: list[dict]) -> str:
    """将可用 Agent 列表格式化为 Planner 提示词。"""
    if not available_agents:
        return "（当前无可用 Agent，请直接回复用户问题）"

    lines = []
    for agent in available_agents:
        lines.append(
            f"- `{agent['agent_id']}` ({agent.get('name', '')}): "
            f"{agent.get('description', '')}\n"
            f"  技能: {', '.join(agent.get('skills', []))}\n"
            f"  关键词: {', '.join(agent.get('keywords', []))}\n"
            f"  意图模式: {', '.join(agent.get('intent_patterns', []))}\n"
            f"  业务范围: {', '.join(agent.get('scope', []))}\n"
            f"  示例问题: {', '.join(agent.get('examples', []))}"
        )
    return "\n".join(lines)


async def planner_node(state: OrchestratorState) -> dict:
    """
    Planner: 根据用户查询和上下文生成任务分发计划。

    - 首轮: 纯粹分析用户需求
    - 迭代轮: 参考 feedback_history 进行修正规划
    """
    llm = _get_planner_llm()

    # 尝试使用结构化输出 (需要 ChatOpenAI 兼容的模型)
    try:
        structured_llm = llm.with_structured_output(PlanOutput, method="function_calling")
    except (AttributeError, NotImplementedError):
        # 降级：用原始 LLM + JSON 解析
        structured_llm = None

    # ─── 动态构建 Agent 能力清单 ───
    available_agents = state.get("available_agents", [])
    agent_catalog = _build_agent_catalog(available_agents)
    valid_agent_ids = [a["agent_id"] for a in available_agents]

    # ─── 构建文件上下文描述 ───
    file_ctx = state.get("file_ctx") or {}
    file_summary = []
    if "images" in file_ctx and file_ctx["images"]:
        names = [
            f"{f.get('file_name', 'unknown')} ({f.get('file_type', 'image')})"
            for f in file_ctx["images"]
        ]
        file_summary.append(f"{len(names)} 张图片 [{', '.join(names)}]")
    if "documents" in file_ctx and file_ctx["documents"]:
        names = [
            f"{f.get('file_name', 'unknown')} ({f.get('file_type', 'document')})"
            for f in file_ctx["documents"]
        ]
        file_summary.append(f"{len(names)} 份文档 [{', '.join(names)}]")
    file_str = "; ".join(file_summary) if file_summary else "无文件"

    # ─── 构建反馈历史上下文 ───
    current_iter = state.get("iter", 0)
    feedback_history = state.get("feedback_history", [])

    if current_iter > 0 and feedback_history:
        feedback_block = "\n".join(
            [f"  第{i+1}次反馈: {fb}" for i, fb in enumerate(feedback_history)]
        )
        iteration_ctx = (
            f"\n### ⚠️ 迭代修正模式 (第 {current_iter + 1} 轮)\n"
            f"Evaluator 对前一轮执行结果不满意，以下是历史修改意见，请在本次规划中针对性调整：\n"
            f"{feedback_block}\n"
            f"请勿重复之前已成功的任务，仅针对反馈意见生成修正任务。\n"
        )
    else:
        iteration_ctx = "\n### 首次规划\n这是第一轮任务分发，请全面分析用户需求。\n"

    # ─── 构建前一轮结果上下文 ───
    prev_results = state.get("results", {})
    if prev_results:
        results_block = "\n".join(
            [f"  - {k}: {str(v)[:4000]}" for k, v in prev_results.items()]
        )
        results_ctx = f"\n### 上一轮 Worker 执行结果:\n{results_block}\n"
    else:
        results_ctx = ""

    system_msg = SystemMessage(
        content=(
            "你是一个高度智能的多智能体系统规划专家（Strategic Planner）。\n"
            "你的任务是根据对话历史和当前环境，逻辑严密地拆解并分发任务给可用的 Agents。\n\n"

            "### 核心原则 (必须遵守):\n"
            "1. **思维链分析 (Rationale)**: 在分发任务前，必须在 rationale 中分析：当前已知什么？还缺什么？哪些任务存在先后依赖关系？\n"
            "2. **有序性与数据依赖**: 如果任务 B 需要参考任务 A 的输出结果，则**严禁**在同一轮内同时调度 A 和 B。应在本轮只调度 A，等待下一轮拿到结果后再调度 B。\n"
            "3. **分步执行**: 宁愿多花几个轮次稳扎稳打，也不要尝试在单轮内并行调度具有因果逻辑的任务。\n"
            "4. **多文件处理与类型匹配**（关键）:\n"
            "   - **第1轮**: 如果检测到多个文件，必须先调度 `dify_file_uploader` 一次性上传所有文件，获取 file_id 映射。\n"
            "   - **后续轮次**: 根据文件类型分别调度对应 Agent：\n"
            "     * 图片文件 (image/*.png/*.jpg etc) → 调度 `dify_vision` 进行图片识别和内容分析\n"
            "     * 文档文件 (document/*.pdf etc) → 调度 `dify_doc_summary` 进行文档摘要和内容总结\n"
            "   - **严格限制**: 切勿将图片传给 `dify_doc_summary`，也不要将 PDF 传给 `dify_vision`。\n"
            "   - **示例**: 若同时上传了 PNG 和 PDF，第1轮上传，第2轮可并发调度 dify_vision(处理PNG) 和 dify_doc_summary(处理PDF)。\n"
            "5. **直接回复 (General Chat)**: 对于不涉及任何 Agent 能力范围的通用问题，**不要调度任何 Agent**。这类问题将由 Final Responder 直接根据对话历史回答。\n"
            "6. **背景注入**: 如果指派的任务依赖之前的结果，必须在指令中包含【背景参考】。\n"
            f"7. **合法 target**: target 字段只能使用以下 agent_id: {valid_agent_ids}\n\n"

            f"【当前可用文件】: {file_str}\n"
            f"{iteration_ctx}"
            f"{results_ctx}"

            f"\n### 可用 Agent 能力清单:\n{agent_catalog}\n\n"

            "### 输出格式:\n"
            "你必须输出 JSON，包含 rationale (规划思路) 和 tasks (任务列表)。\n"
            "每个 task 包含 target (agent_id) 和 instruction (执行指令)。\n"
            "如果是通用问题无需调度 Agent，则 tasks 为空数组。\n"
        )
    )

    # 包含完整的对话历史
    messages = state.get("messages", [])

    try:
        if structured_llm is not None:
            plan: PlanOutput = await structured_llm.ainvoke([system_msg] + messages)
            plan_data = {
                "rationale": plan.rationale,
                "tasks": [t.model_dump() for t in plan.tasks]
            }
        else:
            # 降级：原始 LLM + JSON 手动解析
            response = await llm.ainvoke([system_msg] + messages)
            response_text = response.content if hasattr(response, "content") else str(response)
            logger.info(f"[Planner] Raw LLM response: {response_text[:500]}")

            # 提取 JSON
            import re
            text = response_text.strip()
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
                    raw = {"rationale": text, "tasks": []}

            plan_data = {
                "rationale": raw.get("rationale", ""),
                "tasks": raw.get("tasks", [])
            }

        # 校验 target 合法性
        valid_tasks = []
        for task in plan_data.get("tasks", []):
            if task.get("target") in valid_agent_ids:
                valid_tasks.append(task)
            else:
                logger.warning(f"[Planner] Skipped invalid target: {task.get('target')}")
        plan_data["tasks"] = valid_tasks

        logger.info(
            f"[Planner] Rationale: {plan_data['rationale'][:200]}... | "
            f"Tasks: {[t['target'] for t in plan_data['tasks']]}"
        )

        return {
            "plan": plan_data,
            "iter": 1
        }

    except Exception as e:
        logger.error(f"[Planner Error]: {e}")
        # 降级：生成空计划
        fallback = {"rationale": f"规划异常: {e}", "tasks": []}
        return {
            "plan": fallback,
            "iter": 1
        }
