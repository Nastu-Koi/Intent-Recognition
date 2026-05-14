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

# 可选：动态加载 Agent Cards
try:
    from engine.agent_cards import load_cards_async, get_agent_card_manager
    from orchestrator.nodes.planner_modules.agent_router import get_agent_metadata
    AGENT_CARDS_AVAILABLE = True
except ImportError:
    AGENT_CARDS_AVAILABLE = False

# 可选：任务验证
try:
    from orchestrator.nodes.planner_modules.task_builder import filter_valid_tasks
    TASK_VALIDATION_AVAILABLE = True
except ImportError:
    TASK_VALIDATION_AVAILABLE = False

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
    
    集成的功能：
    - 优先动态加载 Agent Cards（最新信息）
    - 降级到 state 中的 available_agents（RBAC 过滤）
    """
    llm = _get_planner_llm()

    # 尝试使用结构化输出 (需要 ChatOpenAI 兼容的模型)
    try:
        structured_llm = llm.with_structured_output(PlanOutput, method="function_calling")
    except (AttributeError, NotImplementedError):
        # 降级：用原始 LLM + JSON 解析
        structured_llm = None

    # ─── 加载可用 Agents（优先动态加载，然后 fallback 到 state） ───
    available_agents = []
    
    if AGENT_CARDS_AVAILABLE:
        try:
            available_cards = await load_cards_async(force_refresh=True)
            for card in available_cards:
                meta = get_agent_metadata(card)
                
                # 提取完整的 skills 和 intent_patterns
                skills = card.capabilities.skills if hasattr(card, 'capabilities') and hasattr(card.capabilities, 'skills') else []
                intent_patterns = meta.get('intent_patterns', [])
                
                # 构建统一格式的 agent 信息
                available_agents.append({
                    'agent_id': meta['agent_id'],
                    'name': meta['name'],
                    'description': meta['description'],
                    'skills': skills,
                    'keywords': meta.get('keywords', []),
                    'intent_patterns': intent_patterns,
                    'scope': meta.get('scope', []),
                    'examples': meta.get('examples', []),
                })
            logger.info(f"[Planner] 动态加载了 {len(available_agents)} 个 Agent Cards")
        except Exception as e:
            logger.warning(f"[Planner] 动态加载 Agent Cards 失败，回退到 state 配置: {e}")
            available_agents = state.get("available_agents", [])
    else:
        # Agent Cards 模块不可用，使用 state 中的配置
        available_agents = state.get("available_agents", [])
        logger.debug("[Planner] 使用 state 中的 available_agents")
    
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

    # Conversation Router 上下文变异结果。
    # not_related 必须作为真正的新话题处理，不能让上一轮 results/feedback/iter 污染 Planner。
    route = state.get("conversation_route") or {}
    relation = route.get("relation")
    context_note = route.get("context_note") or route.get("rationale") or ""
    effective_query = state.get("query", "")

    # ─── 构建反馈历史上下文 ───
    if relation == "not_related":
        current_iter = 0
        feedback_history = []
    else:
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
    prev_results = {} if relation == "not_related" else state.get("results", {})
    if prev_results:
        results_block = "\n".join(
            [f"  - {k}: {str(v)[:4000]}" for k, v in prev_results.items()]
        )
        results_ctx = f"\n### 上一轮 Dispatcher 执行结果:\n{results_block}\n"
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
            "4. **文件处理**: 如果用户上传了图片或文档，直接将文件处理任务分配给具有相应工具能力的 Agent（如 `general_chat`），该 Agent 内部会自动完成文件上传和处理，无需额外的上传步骤。\n"
            "5. **直接回复 (General Chat)**: 对于通用问题、日常对话、图片识别、文档总结等任务，调度 `general_chat` Agent。对于完全不涉及任何 Agent 能力范围的极简问题，可以不调度任何 Agent，由 Final Responder 直接回答。\n"
            "6. **背景注入**: 如果指派的任务依赖之前的结果，必须在指令中包含【背景参考】。\n"
            f"7. **合法 target**: target 字段只能使用以下 agent_id: {valid_agent_ids}\n"
            "8. **上下文纠正优先**: 如果 Conversation Router 标记 related_type=correction，"
            "且用户是在纠正上一轮的 Agent/工具选择（例如“用某助手回答”“不要用某助手”），"
            "这属于执行方式约束，优先级高于普通关键词匹配。只要被指定的 Agent 是合法 target，"
            "就必须为该 Agent 生成任务；如果你认为该 Agent 可能不擅长，也应让该 Agent 基于其能力边界作答，"
            "不要因为关键词不匹配而返回空 tasks。\n\n"

            f"【当前可用文件】: {file_str}\n"
            f"{iteration_ctx}"
            f"{results_ctx}"

            f"\n### 可用 Agent 能力清单:\n{agent_catalog}\n\n"

            "\n### 🎯 LLM 任务分配指南（关键！）:\n"
            "**分配原理**：不要基于你的常识或通用理解来分配任务。相反，应该严格基于上面列出的 Agent 的描述（description）、技能（skills）和意图模式（intent_patterns）。\n\n"
            "**关键词精确匹配优先规则（⭐ 最重要）**：\n"
            "1. **逐字检查**：先从用户查询中提取关键词，然后在每个 Agent 的 keywords 和 intent_patterns 中查找精确匹配。\n"
            "2. **部分词匹配**：如果查询包含 Agent 关键词的子串或变体（如\"报销指南在哪\"包含\"报销指南\"），视为命中。\n"
            "3. **优先级排序**：\n"
            "   - 🔴 **完全匹配** intent_patterns 中的词 → 极高优先级（95+）→ 立即分配\n"
            "   - 🟡 **匹配** keywords 中的词 → 高优先级（75+）\n"
            "   - 🟢 **部分相关**（description 提及）→ 中优先级（40-75）\n"
            "   - ⚪ **无关** → 0 分，不考虑\n"
            "4. **不要忽视 intent_patterns**：这是该 Agent 专门设计处理的意图，必须优先匹配。\n\n"
            "**分配步骤**：\n"
            "1. **理解每个 Agent 的职责**：仔细阅读每个 Agent 的 description、skills、keywords 和 intent_patterns。\n"
            "2. **分析用户查询**：识别查询中的关键词和核心需求。\n"
            "3. **关键词匹配**：对每个关键词，检查是否在某个 Agent 的 keywords/intent_patterns 中出现。\n"
            "4. **选择最高分 Agent**：为每个需求分配匹配分数最高的 Agent。如果多个 Agent 分数相同，选择优先级较高的。\n"
            "5. **拆分多需求**：如果用户有多个不同领域的需求，必须拆分为多个任务。\n\n"
            "**关键词匹配示例**：\n"
            "- 用户说\"报销指南在哪\" → 查询包含\"报销指南\" → Agent.intent_patterns 包含\"报销指南\" → 分配给对应 Agent\n"
            "- 用户说\"报销系统怎么用\" → 查询包含\"报销\"和\"系统\" → Agent.keywords 包含\"报销系统\" → 分配给对应 Agent\n"
            "- 用户说\"用户手册\" → 查询包含\"用户手册\" → Agent.intent_patterns 包含\"用户手册\" → 分配给对应 Agent\n"
            "- 用户说\"你好\" → 不匹配任何专有 Agent → 分配给 general_chat 或不分配\n\n"
            
            "### 📋 Instruction 生成指南\n"
            "当生成 instruction 时，**必须清晰表达用户的真实意图和背景**：\n"
            "- ❌ 错误示例：\"用户问报销指南\"\n"
            "- ✅ 正确示例：\"用户想了解报销系统的用户手册/操作指南，具体想查看报销流程、发票要求和项目编码选择方法\"\n"
            "- 指令应该足够具体，让 Agent 知道用户最终想要什么信息\n"
            "- 如果用户的表述不清楚，instruction 中应该做「语义转译」，明确转译为标准业务术语\n"
            "  - \"报销指南在哪\" → \"用户想查询报销系统的用户手册和操作流程说明\"\n"
            "  - \"怎么填报销\" → \"用户需要了解报销单的填报步骤和注意事项\"\n\n"

            "### 输出格式:\n"
            "你必须输出 JSON，包含 rationale (规划思路) 和 tasks (任务列表)。\n"
            "每个 task 包含 target (agent_id) 和 instruction (执行指令)。\n"
            "如果是通用问题无需调度 Agent，则 tasks 为空数组。\n"
        )
    )

    logger.info(f"[Planner] relation={relation} effective_query={effective_query[:80]}")
    if relation == "related":
        logger.info(f"[Planner] context_note: {context_note[:100]}")
    
    if relation == "related":
        system_msg.content += (
            "\n\n### Conversation Router / Context Mutation Layer\n"
            f"- relation: related\n"
            f"- related_type: {route.get('related_type', 'none')}\n"
            f"- effective_query: {effective_query}\n"
            f"- context_note: {context_note}\n"
            "请将该说明视为对上一轮上下文的补充/纠正/推翻，并据此规划当前轮任务。\n"
            "特别是当 related_type=correction 表示用户要求改用某个可用 Agent 时，"
            "必须继承上一轮原始问题，并为用户指定的 Agent 生成任务；instruction 中要同时包含上一轮问题和本轮改用 Agent 的要求。\n"
        )
    elif relation == "not_related":
        system_msg.content += (
            "\n\n### Conversation Router / Context Mutation Layer\n"
            "- relation: not_related\n"
            "当前输入开启新对话。规划时只使用当前用户输入和当前可用文件，不要继承上一轮任务目标或 Agent 结果。\n"
        )

    # related 保留完整历史；not_related 只保留当前轮，避免旧上下文污染新对话
    all_messages = state.get("messages", [])
    if relation == "not_related":
        messages = [HumanMessage(content=state.get("query", ""))]
    else:
        messages = all_messages

    logger.debug(f"[Planner] messages_count={len(messages)} relation={relation}")

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

        # 校验任务格式和 target 合法性
        tasks = plan_data.get("tasks", [])
        
        # 第 1 步：格式验证（如果 filter_valid_tasks 可用）
        if TASK_VALIDATION_AVAILABLE:
            tasks = filter_valid_tasks(tasks)
            logger.debug(f"[Planner] 格式验证后任务数: {len(tasks)}")
        
        # 第 2 步：target 合法性校验
        valid_tasks = []
        for task in tasks:
            target = task.get("target")
            if target in valid_agent_ids:
                valid_tasks.append(task)
            else:
                logger.warning(f"[Planner] Skipped invalid target: {target}")
        
        # 第 3 步：保留空任务作为“直接回复”信号。
        # Planner 提示词已经要求图片/文档/通用问答在需要时显式调度 general_chat。
        # 因此这里不能再无条件补一个 general_chat，否则普通追问或路由异常会被误触发工具调用。
        if not valid_tasks:
            if relation == "related" and route.get("related_type") == "correction":
                logger.warning(
                    "[Planner] correction 轮次未生成任务，可能违反用户对 Agent/工具选择的纠正要求 | query=%s | context_note=%s",
                    state.get("query", "")[:120],
                    context_note[:200],
                )
            else:
                logger.info("[Planner] 未生成可执行任务，保留空计划并交由 Final Reply 直接回答")
        
        plan_data["tasks"] = valid_tasks

        logger.info(
            f"[Planner] Rationale: {plan_data['rationale'][:200]}... | "
            f"Tasks: {[t['target'] for t in plan_data['tasks']]}"
        )

        update = {
            "plan": plan_data,
            "iter": current_iter + 1,
        }
        if relation == "not_related":
            update.update(
                {
                    "results": {},
                    "_agent_outputs": {},
                    "feedback_history": [],
                    "eval_action": "",
                    "eval_thought": "",
                    "thinking_chain": [],
                }
            )
        return update

    except Exception as e:
        logger.error(f"[Planner Error]: {e}")
        # 降级：生成空计划
        fallback = {"rationale": f"规划异常: {e}", "tasks": []}
        return {
            "plan": fallback,
            "iter": 1
        }
