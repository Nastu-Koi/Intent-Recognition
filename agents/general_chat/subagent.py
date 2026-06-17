"""
General Chat SubAgent — 通用对话 + 工具调用。

使用 LLM tool-calling 机制，在需要时自动调用图像识别和文档总结工具，
无需 Planner 手动编排文件上传依赖。

LLM 工具调用流程:
  1. 用户请求 + 文件上下文 → LLM 决定是否需要工具
  2. 需要工具 → 自动调用 image_recognition / document_summary
  3. 工具返回结果 → LLM 综合生成最终回复
  4. 不需要工具 → LLM 直接回复（通用对话）
"""

import os
from typing import Dict, Any, List

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage

from engine.subagent import SubAgent
from engine.llm_factory import get_llm_model
from engine.logging_config import get_logger
from engine.streaming import emit_stream_progress
from agents.general_chat.tools import GENERAL_CHAT_TOOLS

logger = get_logger(__name__)

AGENT_ID = "general_chat"

# 最大工具调用轮次（防止无限循环）
MAX_TOOL_ROUNDS = 5

# 缓存 LLM 实例
_CHAT_LLM = None


def _get_chat_llm():
    """获取 General Chat 专用的 LLM 实例。"""
    global _CHAT_LLM
    if _CHAT_LLM is None:
        _CHAT_LLM = get_llm_model()
    return _CHAT_LLM


class GeneralChatAgent(SubAgent):
    """
    通用对话 Agent — 支持工具调用的 LLM SubAgent。

    能力:
    - 日常对话与通用问答（直接 LLM 回复）
    - 图片识别（自动上传 + Dify Vision API）
    - 文档总结（自动上传 + Dify Doc Summary API）

    工具调用完全由 LLM 自主决策，Agent 内部实现 ReAct 循环。
    """

    def __init__(self):
        super().__init__(agent_id=AGENT_ID)

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        同步执行入口 (仅用于非 async 上下文的兜底)。
        
        正常 A2A 调用路径使用 aexecute → _execute_async，保留 contextvars。
        """
        import asyncio
        return asyncio.run(self._execute_async(input_data))

    async def aexecute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        原生异步执行 — 直接调用 _execute_async，保留 contextvar 传播。
        
        这样 emit_stream_progress 的 _progress_queue contextvar 能正确传播，
        前端可以实时看到 agent_reasoning / agent_tool_call 等流式事件。
        """
        return await self._execute_async(input_data)

    async def _execute_async(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """异步执行通用对话与工具调用。"""
        query = input_data.get("query", "")
        context = input_data.get("context", {})
        
        # 保存 conversation_id 供工具使用
        self._conversation_id = context.get("conversation_id", "")

        # 收集可用文件路径
        file_paths = self._collect_file_paths(context)

        # 构建文件上下文描述（注入到 system prompt）
        file_context_str = self._build_file_context_str(file_paths)
        skill_context = context.get("skill_context") or {}

        # 构建 system message
        system_msg = SystemMessage(content=self._build_system_prompt(file_context_str, skill_context))

        # 初始化 LLM 并绑定工具
        llm = _get_chat_llm()
        llm_with_tools = llm.bind_tools(GENERAL_CHAT_TOOLS)

        # 构建初始消息列表
        messages: List = [system_msg, HumanMessage(content=query)]

        logger.info(
            f"[GeneralChat] query={query[:100]}... | "
            f"files={len(file_paths)} | file_paths={file_paths}"
        )

        # 发送流式事件：开始处理
        await emit_stream_progress(
            "agent_reasoning",
            {
                "agent_id": self.agent_id,
                "agent_name": "通用对话助手",
                "status": "started",
                "message": "开始分析问题，判断是否需要调用工具...",
            },
        )

        # ReAct 循环: LLM 调用 → 解析 tool_calls → 执行工具 → 反馈 → 再调用
        for round_idx in range(MAX_TOOL_ROUNDS):
            response = await llm_with_tools.ainvoke(messages)
            messages.append(response)

            # 检查是否有工具调用
            tool_calls = getattr(response, "tool_calls", None) or []

            if not tool_calls:
                # LLM 未调用任何工具 → 直接返回文本回复
                final_text = response.content if hasattr(response, "content") else str(response)
                logger.info(
                    f"[GeneralChat] 直接回复 (round={round_idx + 1}), "
                    f"长度={len(final_text)}"
                )

                # 发送流式事件：生成最终回复
                await emit_stream_progress(
                    "agent_reasoning",
                    {
                        "agent_id": self.agent_id,
                        "agent_name": "通用对话助手",
                        "status": "completed",
                        "message": "已生成回复",
                        "result_preview": final_text[:300],
                    },
                )

                return {
                    "status": "success",
                    "result": final_text,
                    "agent": self.agent_id,
                }

            # 发送流式事件：开始执行工具调用
            await emit_stream_progress(
                "agent_reasoning",
                {
                    "agent_id": self.agent_id,
                    "agent_name": "通用对话助手",
                    "status": "tool_calling",
                    "message": f"需要调用 {len(tool_calls)} 个工具来完成任务",
                    "tools": [tc["name"] for tc in tool_calls],
                },
            )

            # 执行所有工具调用
            for tc_idx, tc in enumerate(tool_calls, 1):
                tool_name = tc["name"]
                tool_args = tc["args"]
                tool_call_id = tc["id"]

                logger.info(
                    f"[GeneralChat] Tool call: {tool_name}({tool_args}) "
                    f"[round={round_idx + 1}, tool={tc_idx}/{len(tool_calls)}]"
                )

                # 发送流式事件：工具调用开始
                await emit_stream_progress(
                    "agent_tool_call",
                    {
                        "agent_id": self.agent_id,
                        "agent_name": "通用对话助手",
                        "tool_name": tool_name,
                        "status": "started",
                        "message": f"正在调用工具: {tool_name}",
                    },
                )

                # 查找并执行对应的工具
                tool_result = self._invoke_tool(tool_name, tool_args)

                logger.info(
                    f"[GeneralChat] Tool result: {tool_name} → {str(tool_result)[:100]}..."
                )

                # 发送流式事件：工具调用完成
                await emit_stream_progress(
                    "agent_tool_call",
                    {
                        "agent_id": self.agent_id,
                        "agent_name": "通用对话助手",
                        "tool_name": tool_name,
                        "status": "completed",
                        "message": f"工具执行完成: {tool_name}",
                        "result_preview": str(tool_result)[:200],
                    },
                )

                # 将工具结果作为 ToolMessage 反馈给 LLM
                messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call_id,
                    )
                )

        # 达到最大轮次仍未完成，取最后一次 LLM 回复
        logger.warning(f"[GeneralChat] 达到最大工具调用轮次 ({MAX_TOOL_ROUNDS})")
        
        # 发送流式事件：达到最大轮次
        await emit_stream_progress(
            "agent_reasoning",
            {
                "agent_id": self.agent_id,
                "agent_name": "通用对话助手",
                "status": "max_iterations",
                "message": f"达到最大工具调用轮次 ({MAX_TOOL_ROUNDS})，返回当前结果",
            },
        )

        last_response = await llm_with_tools.ainvoke(messages)
        final_text = last_response.content if hasattr(last_response, "content") else str(last_response)

        # 发送流式事件：完成
        await emit_stream_progress(
            "agent_reasoning",
            {
                "agent_id": self.agent_id,
                "agent_name": "通用对话助手",
                "status": "completed",
                "message": "任务完成",
                "result_preview": final_text[:300],
            },
        )

        return {
            "status": "success",
            "result": final_text,
            "agent": self.agent_id,
        }

    def _collect_file_paths(self, context: Dict[str, Any]) -> List[str]:
        """从上下文中收集所有可用的文件路径。"""
        file_paths = []

        # 1. 直接从 context.file_paths 获取
        if context.get("file_paths"):
            file_paths.extend(context["file_paths"])

        # 2. 从 context.file_path 获取（单个文件）
        if context.get("file_path"):
            fp = context["file_path"]
            if fp not in file_paths:
                file_paths.append(fp)

        # 3. 从 A2A metadata.file_ctx 获取
        file_ctx = context.get("file_ctx") or context.get("metadata", {}).get("file_ctx") or {}
        for category in ("images", "documents"):
            for f in file_ctx.get(category, []):
                if isinstance(f, dict) and f.get("file_path"):
                    fp = f["file_path"]
                    if fp not in file_paths:
                        file_paths.append(fp)

        return file_paths

    def _build_file_context_str(self, file_paths: List[str]) -> str:
        """构建文件上下文描述字符串。"""
        if not file_paths:
            return "当前没有可用的文件。"

        lines = ["以下是用户上传的文件（你可以通过工具来处理这些文件）:"]
        for fp in file_paths:
            ext = os.path.splitext(fp)[1].lower()
            if ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"):
                file_type = "图片"
            elif ext in (".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".txt", ".md", ".csv"):
                file_type = "文档"
            else:
                file_type = "文件"
            lines.append(f"  - [{file_type}] {fp}")

        return "\n".join(lines)

    def _build_system_prompt(self, file_context_str: str, skill_context: Dict[str, Any] | None = None) -> str:
        """构建 General Chat 的 system prompt。"""
        skill_context = skill_context or {}
        if skill_context:
            skill_prompt = (
                "\n### 用户选中的 Skill 指令（必须遵守）\n"
                f"Skill: {skill_context.get('name', '')}\n"
                f"Description: {skill_context.get('description', '')}\n"
                "以下是该 skill 的完整 SKILL.md 指令：\n"
                f"{skill_context.get('instruction', '')}\n"
                "### Skill 指令结束\n\n"
            )
        else:
            skill_prompt = ""

        return (
            "你是一个智能通用对话助手。你可以:\n"
            "1. 回答各种通用问题和进行日常对话\n"
            "2. 使用 `image_recognition` 工具识别和分析图片（支持 OCR、发票识别、场景分析）\n"
            "3. 使用 `document_summary` 工具总结和分析文档\n"
            "4. 使用 `pdf_add_watermark` 工具给 PDF 添加文本水印并生成新文件\n"
            "5. 使用 `docx_create` 工具创建 Word/DOCX 文档并生成可下载文件\n"
            "6. 使用 `pptx_create` 工具创建 PowerPoint/PPTX 演示文稿并生成可下载文件\n\n"
            f"{skill_prompt}"

            "### 工具使用规则:\n"
            "- 当用户提供了图片文件且需要识别/分析时，调用 `image_recognition` 工具\n"
            "- 当用户提供了文档文件且需要总结/分析时，调用 `document_summary` 工具\n"
            "- 当用户要求给 PDF 添加水印时，必须调用 `pdf_add_watermark` 工具，不要只提供操作建议\n"
            "- 当用户要求创建、生成、输出 Word/DOCX 文档时，必须先整理正文内容，再调用 `docx_create` 工具，不要声称自己无法生成 Word 文件\n"
            "- 当用户要求创建、生成、输出 PPT/PPTX/PowerPoint/演示文稿时，必须先整理每页标题和要点，再调用 `pptx_create` 工具，不要声称自己无法生成 PPT 文件\n"
            "- 工具的 `file_path` 参数必须使用下面「可用文件」中列出的完整路径\n"
            "- 如果是普通聊天或无需文件处理的问题，直接回复即可，不需要调用任何工具\n\n"

            f"### 可用文件:\n{file_context_str}\n"
        )

    def _invoke_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """根据工具名称执行对应的工具。"""
        tool_map = {t.name: t for t in GENERAL_CHAT_TOOLS}
        tool_fn = tool_map.get(tool_name)

        if tool_fn is None:
            return f"未知工具: {tool_name}"

        try:
            # 注入 conversation_id 到工具参数中（如果工具支持）
            enhanced_args = dict(tool_args)
            if hasattr(self, '_conversation_id') and self._conversation_id:
                enhanced_args['conversation_id'] = self._conversation_id
            
            return tool_fn.invoke(enhanced_args)
        except Exception as e:
            logger.error(f"[GeneralChat] 工具 {tool_name} 执行失败: {e}")
            return f"工具执行失败: {e}"
